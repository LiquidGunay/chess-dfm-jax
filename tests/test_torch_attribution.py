from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from research.interpretability.attribution import (
    gradcam_square_map,
    input_gradient_attribution,
    integrated_gradients,
    legal_piece_removal_variants,
    pairwise_occlusion_interaction,
    sarfa_scores,
    smoothgrad,
)


def _linear_forward(inputs: torch.Tensor) -> torch.Tensor:
    score = 2.0 * inputs[:, 0].sum(dim=(1, 2)) - inputs[:, 1].sum(dim=(1, 2))
    return torch.stack((score, torch.zeros_like(score), -score), dim=1)


def test_gradient_and_integrated_gradients_are_exact_for_linear_target() -> None:
    inputs = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    inputs[:, 0, 0, 0] = 3.0
    inputs[:, 1, 1, 1] = 2.0
    targets = torch.zeros(2, dtype=torch.long)
    gradient = input_gradient_attribution(
        _linear_forward,
        inputs,
        targets,
        kind="target_logit",
        multiply_by_input=True,
    )
    assert gradient.attribution[:, 0, 0, 0].tolist() == [6.0, 6.0]
    assert gradient.attribution[:, 1, 1, 1].tolist() == [-2.0, -2.0]

    baselines = torch.stack((torch.zeros_like(inputs), torch.ones_like(inputs) * 0.25))
    integrated = integrated_gradients(
        _linear_forward,
        inputs,
        targets,
        baselines,
        kind="target_logit",
        steps=8,
    )
    torch.testing.assert_close(
        integrated.completeness_residual,
        torch.zeros_like(integrated.completeness_residual),
        rtol=0.0,
        atol=1e-5,
    )
    assert integrated.per_baseline_attribution.shape == (2, 2, 3, 8, 8)

    gauss = integrated_gradients(
        _linear_forward,
        inputs,
        targets,
        baselines,
        kind="target_logit",
        steps=8,
        quadrature="gauss_legendre",
    )
    torch.testing.assert_close(
        gauss.completeness_residual,
        torch.zeros_like(gauss.completeness_residual),
        rtol=0.0,
        atol=1e-5,
    )
    torch.testing.assert_close(gauss.attribution, integrated.attribution)
    assert gauss.method == "integrated_gradients_gauss_legendre"

    with pytest.raises(ValueError, match="quadrature"):
        integrated_gradients(
            _linear_forward,
            inputs,
            targets,
            baselines,
            kind="target_logit",
            steps=8,
            quadrature="unknown",  # type: ignore[arg-type]
        )


def test_smoothgrad_is_seeded_and_gradcam_has_square_shape() -> None:
    inputs = torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(3))
    targets = torch.zeros(1, dtype=torch.long)
    first = smoothgrad(
        _linear_forward,
        inputs,
        targets,
        kind="target_logit",
        samples=4,
        seed=9,
    )
    second = smoothgrad(
        _linear_forward,
        inputs,
        targets,
        kind="target_logit",
        samples=4,
        seed=9,
    )
    torch.testing.assert_close(first.attribution, second.attribution, rtol=0.0, atol=0.0)

    activations = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    with pytest.raises(ValueError, match="64 square"):
        gradcam_square_map(activations, torch.ones_like(activations))
    activations = torch.arange(256, dtype=torch.float32).reshape(1, 64, 4)
    result = gradcam_square_map(activations, torch.ones_like(activations))
    assert result.shape == (1, 8, 8)
    assert bool((result >= 0).all())


def test_sarfa_matches_released_equation_and_clamps_negative_specificity() -> None:
    original = torch.tensor([[2.0, 1.0, 0.0]])
    perturbed = torch.tensor([[[0.0, 1.0, 0.0], [3.0, 1.0, 0.0], [2.0, 0.0, 1.0]]])
    result = sarfa_scores(
        original,
        perturbed,
        torch.tensor([[0, 1, 2]]),
        torch.tensor([3]),
        torch.tensor([0]),
        eligible=torch.tensor([[True, True, True]]),
        target_legal=torch.tensor([[True, True, False]]),
    )
    assert 0.0 < float(result["saliency"][0, 0]) < 1.0
    assert float(result["saliency"][0, 1]) == 0.0
    assert float(result["saliency"][0, 2]) == 1.0
    expected_relevance = 1.0 / (1.0 + float(result["other_action_kl_original_to_perturbed"][0, 0]))
    assert float(result["relevance"][0, 0]) == pytest.approx(expected_relevance)


def test_piece_removal_reencodes_legal_nonking_squares_and_history() -> None:
    board = chess.Board()
    target = chess.Move.from_uci("e2e4")
    variants = legal_piece_removal_variants(board, [board], target_move=target)
    assert variants.planes.shape == (64, 112, 8, 8)
    assert variants.eligible.sum() == 30
    assert not variants.eligible[chess.E1]
    assert variants.eligible[chess.E2]
    assert not variants.target_legal[chess.E2]
    assert variants.planes.dtype == np.float32


def test_pairwise_occlusion_uses_inclusion_exclusion() -> None:
    original = torch.tensor([10.0])
    singles = torch.tensor([[7.0, 8.0]])
    pairs = torch.tensor([[[4.0, 3.0], [3.0, 6.0]]])
    interaction = pairwise_occlusion_interaction(original, singles, pairs)
    assert interaction.tolist() == [[[0.0, -2.0], [-2.0, 0.0]]]
