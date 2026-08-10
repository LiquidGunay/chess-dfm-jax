from __future__ import annotations

from typing import Any

import pytest
import torch

from research.interpretability.models import ComparisonPolicyOutput
from research.interpretability.patching import (
    legal_log_probabilities,
    make_patch_tensor,
    patch_comparison_boundary,
    policy_effect_metrics,
    summarize_patch_metrics,
    target_log_odds,
)
from research.train_torch import BT4EncoderCapture


def _legal() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[0, 1, 2, 65535], [0, 2, 65535, 65535]]),
        torch.tensor([3, 2]),
        torch.tensor([0, 2]),
    )


def test_legal_policy_is_padding_safe_and_target_log_odds_is_exact() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 99.0, 2.0]])
    legal, counts, targets = _legal()
    logp, mask = legal_log_probabilities(logits, legal, counts, targets=targets)
    assert mask.tolist() == [[True, True, True, False], [True, True, False, False]]
    assert bool(torch.isneginf(logp[~mask]).all())
    odds = target_log_odds(logits, legal, counts, targets)
    expected = torch.stack(
        (
            logits[0, 0] - torch.logsumexp(logits[0, 1:3], dim=0),
            logits[1, 2] - logits[1, 0],
        )
    )
    torch.testing.assert_close(odds, expected)

    with pytest.raises(ValueError, match="duplicate"):
        legal_log_probabilities(
            logits[:1],
            torch.tensor([[0, 0]]),
            torch.tensor([2]),
        )
    with pytest.raises(ValueError, match="exactly once"):
        legal_log_probabilities(
            logits[:1],
            torch.tensor([[0, 1]]),
            torch.tensor([2]),
            targets=torch.tensor([2]),
        )


def test_policy_effect_metrics_recovers_projection_and_flags_overshoot() -> None:
    source = torch.tensor([[0.0, 0.0, 0.0]])
    destination = torch.tensor([[2.0, 0.0, 0.0]])
    legal = torch.tensor([[0, 1, 2]])
    counts = torch.tensor([3])
    targets = torch.tensor([0])

    at_destination = policy_effect_metrics(
        source,
        destination,
        destination,
        legal,
        counts,
        targets,
    )
    torch.testing.assert_close(
        at_destination["distribution_delta_projection"],
        torch.ones(1),
    )
    torch.testing.assert_close(at_destination["distribution_restoration"], torch.ones(1))
    torch.testing.assert_close(
        at_destination["target_log_odds_mediation"],
        torch.ones(1),
    )
    assert not bool(at_destination["distribution_overshoot"][0])

    overshot = policy_effect_metrics(
        source,
        destination,
        torch.tensor([[4.0, 0.0, 0.0]]),
        legal,
        counts,
        targets,
    )
    assert bool(overshot["distribution_overshoot"][0])
    assert bool(overshot["target_overshoot"][0])

    unresolved = policy_effect_metrics(
        source,
        source,
        source,
        legal,
        counts,
        targets,
    )
    assert not bool(unresolved["distribution_effect_resolved"][0])
    assert bool(torch.isnan(unresolved["distribution_delta_projection"][0]))


def test_patch_tensor_supports_square_feature_and_dose_selection() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    destination = torch.zeros_like(source)
    patched = make_patch_tensor(
        source,
        destination,
        square_indices=(1, 3),
        feature_indices=(0, 5),
        scale=0.5,
    )
    expected = torch.zeros_like(source)
    expected[:, [1, 3], 0] = 0.5 * source[:, [1, 3], 0]
    expected[:, [1, 3], 5] = 0.5 * source[:, [1, 3], 5]
    torch.testing.assert_close(patched, expected)
    with pytest.raises(ValueError, match="duplicates"):
        make_patch_tensor(source, destination, square_indices=(1, 1))


class _FakeModels:
    def policy_logits_with_captures(
        self,
        planes: torch.Tensor,
        *,
        arm: str,
        compute_dtype: torch.dtype,
        capture_layers: tuple[int, ...],
        **overrides: Any,
    ) -> ComparisonPolicyOutput:
        del compute_dtype
        base = planes.float() + (0.0 if arm[0] == "R" else 1.0)
        boundary_names = {
            "attention_input_overrides": "hook_attn_in",
            "attention_output_overrides": "hook_attn_out",
            "resid_mid_overrides": "resid_mid_after_ln",
            "mlp_output_overrides": "hook_mlp_out",
            "resid_post_overrides": "resid_post_after_ln",
        }
        captured = {name: base for name in boundary_names.values()}
        for argument, values in overrides.items():
            if values is not None:
                captured[boundary_names[argument]] = values[capture_layers[0]]
        state = captured["resid_post_after_ln"]
        total = state.sum(dim=(1, 2))
        logits = torch.stack((total, -total, torch.zeros_like(total)), dim=1)
        stacked = {name: value.unsqueeze(0) for name, value in captured.items()}
        capture = BT4EncoderCapture(layer_indices=capture_layers, **stacked)
        return ComparisonPolicyOutput(logits, state, capture)


def test_comparison_patch_holds_head_fixed_and_reports_full_restoration() -> None:
    planes = torch.zeros((1, 2, 2))
    result = patch_comparison_boundary(
        _FakeModels(),
        planes,
        source_arm="RR",
        destination_arm="HR",
        layer=0,
        boundary="resid_post_after_ln",
        compute_dtype=torch.float32,
        legal_indices=torch.tensor([[0, 1, 2]]),
        legal_counts=torch.tensor([3]),
        targets=torch.tensor([0]),
    )
    torch.testing.assert_close(result.patched_logits, result.source_logits)
    assert float(result.metrics["distribution_restoration"][0]) == pytest.approx(1.0)
    summary = summarize_patch_metrics(result.metrics)
    assert summary["distribution_effect_resolved"]["true_count"] == 1

    with pytest.raises(ValueError, match="fixed policy head"):
        patch_comparison_boundary(
            _FakeModels(),
            planes,
            source_arm="RR",
            destination_arm="HH",
            layer=0,
            boundary="resid_post_after_ln",
            compute_dtype=torch.float32,
            legal_indices=torch.tensor([[0, 1, 2]]),
            legal_counts=torch.tensor([3]),
            targets=torch.tensor([0]),
        )
