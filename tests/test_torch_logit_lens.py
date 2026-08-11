from __future__ import annotations

import torch

from research.interpretability.logit_lens import (
    cross_head_lenses,
    first_emergence_layer,
    intermediate_policy_logits,
    logit_lens_metrics,
    validate_final_lens,
)
from research.train_torch import BT4EncoderCapture


class _Head:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, states: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        value = states.to(compute_dtype).sum(dim=(1, 2)) * self.scale
        return torch.stack((value, -value, torch.zeros_like(value)), dim=1)


def _capture() -> BT4EncoderCapture:
    states = torch.stack(
        (
            torch.zeros((2, 64, 4)),
            torch.ones((2, 64, 4)),
            torch.ones((2, 64, 4)) * 2,
        )
    )
    return BT4EncoderCapture(
        layer_indices=(0, 4, 9),
        hook_attn_in=states,
        hook_attn_out=states,
        resid_mid_after_ln=states,
        hook_mlp_out=states,
        resid_post_after_ln=states,
    )


def test_intermediate_and_cross_head_lenses_retain_depth_and_arm_semantics() -> None:
    capture = _capture()
    logits = intermediate_policy_logits(
        capture,
        _Head(1.0),
        compute_dtype=torch.float32,
        layer_chunk_size=2,
    )
    assert logits.shape == (3, 2, 3)
    lenses = cross_head_lenses(
        capture,
        capture,
        _Head(1.0),
        _Head(2.0),
        compute_dtype=torch.float32,
    )
    assert set(lenses) == {"RR", "HR", "RH", "HH"}
    torch.testing.assert_close(lenses["RH"], 2.0 * lenses["RR"])


def test_lens_metrics_and_emergence_use_actual_layer_indices() -> None:
    logits = torch.tensor(
        [
            [[0.0, 2.0, 1.0]],
            [[1.5, 1.0, 0.0]],
            [[3.0, 1.0, 0.0]],
        ]
    )
    metrics = logit_lens_metrics(
        logits,
        torch.tensor([[0, 1, 2]]),
        torch.tensor([3]),
        torch.tensor([0]),
    )
    assert metrics["target_rank"][:, 0].tolist() == [3, 1, 1]
    emergence = first_emergence_layer(
        metrics["target_probability"],
        (0, 4, 9),
        threshold=0.5,
    )
    assert emergence.tolist() == [4]
    rank_emergence = first_emergence_layer(
        metrics["target_rank"].float(),
        (0, 4, 9),
        threshold=1,
        comparison="le",
    )
    assert rank_emergence.tolist() == [4]


def test_final_lens_gate_detects_exact_and_drifted_readouts() -> None:
    logits = intermediate_policy_logits(
        _capture(),
        _Head(1.0),
        compute_dtype=torch.float32,
    )
    assert validate_final_lens(logits, logits[-1])["passed"]
    drifted = logits[-1].clone()
    drifted[0, 0] += 1e-3
    report = validate_final_lens(logits, drifted)
    assert not report["passed"]
    assert report["maximum_absolute_error"] > 0
