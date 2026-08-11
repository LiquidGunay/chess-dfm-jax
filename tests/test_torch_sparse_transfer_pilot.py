from __future__ import annotations

import torch

from research.interpretability.sparse.transfer_pilot import (
    policy_replacement_metrics,
    source_compatibility_gate,
    summarize_policy_metrics,
)


def test_policy_replacement_metrics_are_zero_for_identity() -> None:
    logits = torch.tensor([[1.0, 3.0, -1.0, 2.0]])
    legal = torch.tensor([[0, 1, 3, 65535]])
    count = torch.tensor([3])
    target = torch.tensor([1])
    metrics = policy_replacement_metrics(logits, logits.clone(), legal, count, target)
    torch.testing.assert_close(metrics["js_divergence"], torch.zeros(1), atol=1e-7, rtol=0)
    torch.testing.assert_close(metrics["total_variation"], torch.zeros(1))
    assert metrics["top1_agreement"].tolist() == [True]
    summary = summarize_policy_metrics(metrics)
    gate = source_compatibility_gate(
        {"normalized_mse": 0.1, "cosine": 0.95},
        summary,
    )
    assert gate["passed"]


def test_source_gate_fails_closed_on_reconstruction() -> None:
    policy = {
        "js_divergence": {"mean": 0.0},
        "top1_agreement": {"mean": 1.0},
    }
    gate = source_compatibility_gate(
        {"normalized_mse": 0.251, "cosine": 0.99},
        policy,
    )
    assert not gate["passed"]
    assert not gate["criteria"]["normalized_mse_le_0.25"]
