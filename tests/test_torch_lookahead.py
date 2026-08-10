from __future__ import annotations

import numpy as np
import torch

from research.interpretability.lookahead import (
    BilinearSquareProbe,
    FittedBilinearSquareProbe,
    evaluate_future_move_pair,
    fit_bilinear_square_probe,
    frequency_baseline_metrics,
    normalized_probe_gradient,
    select_future_ply,
    square_prediction_metrics,
    steer_with_probe_gradient,
)


def _manual_probe(
    width: int,
    *,
    candidate_feature: int = 0,
    condition_feature: int = 1,
) -> FittedBilinearSquareProbe:
    module = BilinearSquareProbe(width, rank=1, seed=0)
    with torch.no_grad():
        module.candidate_weight.zero_()
        module.condition_weight.zero_()
        module.candidate_weight[0, candidate_feature] = 1.0
        module.condition_weight[0, condition_feature] = 1.0
        module.bias.zero_()
    return FittedBilinearSquareProbe(module=module.eval(), metadata={})


def test_bilinear_probe_scores_candidate_by_conditioned_feature() -> None:
    probe = _manual_probe(2)
    states = torch.zeros((2, 64, 2))
    states[0, 9, 0] = 3.0
    states[0, 2, 1] = 2.0
    states[1, 7, 0] = 4.0
    states[1, 5, 1] = 1.5
    logits = probe.logits(states, torch.tensor([2, 5]))
    assert logits.argmax(dim=1).tolist() == [9, 7]
    metrics = square_prediction_metrics(logits, [9, 7])
    assert metrics["accuracy"] == 1.0
    assert metrics["count"] == 2


def test_future_selector_uses_explicit_root_frame_and_valid_mask() -> None:
    states = torch.zeros((3, 64, 4))
    origins = np.array([[1, 2, -1], [3, -1, -1], [4, 5, 6]])
    destinations = np.array([[7, 8, -1], [9, -1, -1], [10, 11, 12]])
    valid = origins >= 0
    selected, source, target, condition = select_future_ply(
        states,
        origins,
        destinations,
        valid,
        ply=1,
    )
    assert selected.shape == (2, 64, 4)
    assert source.tolist() == [2, 5]
    assert target.tolist() == [8, 11]
    assert condition.tolist() == [7, 10]


def test_published_two_stage_evaluation_and_probe_gradient() -> None:
    destination_probe = _manual_probe(4)
    source_probe = _manual_probe(4, candidate_feature=2, condition_feature=3)
    states = torch.zeros((2, 64, 4))
    states[0, 9, 0] = 3.0
    states[0, 2, 1] = 2.0
    states[0, 4, 2] = 5.0
    states[0, 9, 3] = 2.0
    states[1, 7, 0] = 4.0
    states[1, 5, 1] = 1.5
    states[1, 3, 2] = 6.0
    states[1, 7, 3] = 1.0
    report = evaluate_future_move_pair(
        destination_probe,
        source_probe,
        states,
        source_labels=[4, 3],
        destination_labels=[9, 7],
        first_destination_squares=[2, 5],
    )
    assert report["joint_move_accuracy"] == 1.0
    gradient, objective = normalized_probe_gradient(
        destination_probe,
        states,
        [2, 5],
        [9, 7],
    )
    torch.testing.assert_close(gradient.flatten(1).norm(dim=1), torch.ones(2))
    assert torch.isfinite(objective).all()
    steered = steer_with_probe_gradient(states, gradient, dose=0.5)
    assert not torch.equal(steered, states)


def test_probe_fit_is_deterministic_and_frequency_control_is_finite() -> None:
    generator = torch.Generator().manual_seed(4)
    states = torch.randn((32, 64, 3), generator=generator)
    condition = torch.arange(32) % 64
    labels = states[:, :, 0].argmax(dim=1)
    kwargs = dict(rank=2, epochs=2, learning_rate=0.02, batch_size=8, seed=12)
    first = fit_bilinear_square_probe(states, labels, condition, **kwargs)
    second = fit_bilinear_square_probe(states, labels, condition, **kwargs)
    torch.testing.assert_close(first.logits(states, condition), second.logits(states, condition))
    assert first.metadata["replication"]["architecture_match"] is False
    baseline = frequency_baseline_metrics(labels[:20], labels[20:])
    assert np.isfinite(list(baseline.values())).all()
