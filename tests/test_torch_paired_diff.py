from __future__ import annotations

import pytest
import torch

from research.interpretability.paired_diff import (
    DeltaCovarianceAccumulator,
    counterfactual_delta_consistency,
    fit_delta_subspace,
    fit_orthogonal_alignment,
    fit_ridge_mediator,
    mediator_metrics,
    steer_along_direction,
)


def test_procrustes_recovers_rotation_and_delta_subspace_is_low_rank() -> None:
    generator = torch.Generator().manual_seed(4)
    raw = torch.randn((20, 64, 4), generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn((4, 4), generator=generator))
    offset = torch.tensor([1.0, -2.0, 0.5, 3.0])
    hero = raw @ rotation + offset
    alignment = fit_orthogonal_alignment(raw, hero)
    aligned = alignment.transform_raw(raw)
    torch.testing.assert_close(aligned, hero, rtol=1e-4, atol=1e-4)
    assert alignment.training_residual_rms < 1e-5

    direction = torch.tensor([1.0, 2.0, -1.0, 0.5])
    coefficient = torch.randn((20, 64, 1), generator=generator)
    shifted = raw + coefficient * direction
    subspace = fit_delta_subspace(raw, shifted, component_count=2)
    assert float(subspace.explained_variance_ratio[0]) > 0.999
    reconstructed = subspace.reconstruct_delta(shifted - raw)
    torch.testing.assert_close(reconstructed, shifted - raw, rtol=1e-4, atol=1e-4)


def test_delta_accumulator_reports_square_concentration_and_behavior_coupling() -> None:
    raw = torch.zeros((5, 64, 3))
    hero = raw.clone()
    hero[:, 7, 0] = torch.arange(1, 6, dtype=torch.float32)
    behavior = torch.arange(1, 6, dtype=torch.float32)
    accumulator = DeltaCovarianceAccumulator(3)
    accumulator.update(raw, hero, behavior_delta=behavior)
    summary, subspace = accumulator.finalize(top_component_count=2)
    assert summary["highest_energy_square"] == 7
    assert summary["effective_rank_participation"] == pytest.approx(1.0)
    assert summary["behavior_coupling"]["pearson_position_delta_rms"] == pytest.approx(1.0)
    assert subspace.components.shape == (2, 3)


def test_steering_ridge_mediator_and_counterfactual_consistency() -> None:
    activation = torch.zeros((2, 4, 3))
    steered = steer_along_direction(
        activation,
        torch.tensor([3.0, 0.0, 0.0]),
        2.0,
        square_mask=torch.tensor([[True, False, False, False], [False, True, False, False]]),
    )
    assert steered[0, 0, 0] == 2.0
    assert steered[0, 1, 0] == 0.0

    generator = torch.Generator().manual_seed(9)
    features = torch.randn((100, 4), generator=generator)
    true_weight = torch.randn((4, 3), generator=generator)
    outputs = features @ true_weight + 0.25
    mediator = fit_ridge_mediator(features, outputs, l2=1e-6)
    prediction = mediator.predict(features)
    metrics = mediator_metrics(outputs, prediction)
    assert metrics["global_r2"] > 0.999

    base = torch.zeros((2, 3))
    change = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    consistency = counterfactual_delta_consistency(base, change, base, 2 * change)
    torch.testing.assert_close(
        consistency["raw_hero_response_cosine"],
        torch.ones(2),
    )
    assert bool((consistency["response_delta_rms"] > 0).all())
