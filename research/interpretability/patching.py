"""Bidirectional activation patching with full-policy causal effect metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from research.interpretability.models import ArmId, ComparisonPolicyOutput
from research.train_torch import BT4EncoderCapture


Boundary = Literal[
    "hook_attn_in",
    "hook_attn_out",
    "resid_mid_after_ln",
    "hook_mlp_out",
    "resid_post_after_ln",
]
BOUNDARIES: tuple[Boundary, ...] = (
    "hook_attn_in",
    "hook_attn_out",
    "resid_mid_after_ln",
    "hook_mlp_out",
    "resid_post_after_ln",
)
_OVERRIDE_ARGUMENT: Mapping[Boundary, str] = {
    "hook_attn_in": "attention_input_overrides",
    "hook_attn_out": "attention_output_overrides",
    "resid_mid_after_ln": "resid_mid_overrides",
    "hook_mlp_out": "mlp_output_overrides",
    "resid_post_after_ln": "resid_post_overrides",
}


class _ComparisonModels(Protocol):
    def policy_logits_with_captures(
        self,
        planes: Tensor,
        *,
        arm: ArmId,
        compute_dtype: torch.dtype,
        capture_layers: Sequence[int] | None = None,
        **overrides: Mapping[int, Tensor] | None,
    ) -> ComparisonPolicyOutput: ...


@dataclass(frozen=True)
class BoundaryPatchResult:
    """One source-to-destination intervention at a named encoder boundary."""

    source_arm: ArmId
    destination_arm: ArmId
    layer: int
    boundary: Boundary
    source_logits: Tensor
    destination_logits: Tensor
    patched_logits: Tensor
    patched_activation: Tensor
    metrics: dict[str, Tensor]


def _validate_logits(logits: Tensor, *, name: str) -> tuple[int, int]:
    if logits.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, action], got {tuple(logits.shape)}")
    if not torch.is_floating_point(logits):
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return int(logits.shape[0]), int(logits.shape[1])


def legal_log_probabilities(
    logits: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
    *,
    targets: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return padded legal-action log probabilities and a validity mask.

    The padding values in legal_indices are never gathered, so the public
    corpus sentinel 65535 is safe. Every legal row is checked for duplicates;
    optional targets must occur exactly once in their corresponding row.
    """

    batch, action_count = _validate_logits(logits, name="logits")
    if legal_indices.ndim != 2 or legal_indices.shape[0] != batch:
        raise ValueError("legal_indices must have shape [batch, max_legal]")
    if legal_counts.shape != (batch,):
        raise ValueError("legal_counts must have shape [batch]")
    if legal_indices.device != logits.device or legal_counts.device != logits.device:
        raise ValueError("logits, legal_indices, and legal_counts must share a device")
    counts = legal_counts.to(dtype=torch.long)
    maximum = int(legal_indices.shape[1])
    if bool(((counts < 2) | (counts > maximum)).any()):
        raise ValueError("Each row must contain between 2 and max_legal actions")
    mask = torch.arange(maximum, device=logits.device).unsqueeze(0) < counts.unsqueeze(1)
    indices = legal_indices.to(dtype=torch.long)
    invalid_live = mask & ((indices < 0) | (indices >= action_count))
    if bool(invalid_live.any()):
        raise ValueError("A live legal action index is outside the logit vocabulary")
    safe = torch.where(mask, indices, torch.zeros_like(indices))
    gathered = logits.gather(1, safe)
    gathered = gathered.masked_fill(~mask, -torch.inf)
    sorted_indices = safe.masked_fill(~mask, action_count).sort(dim=1).values
    duplicate = (
        (sorted_indices[:, 1:] == sorted_indices[:, :-1]) & (sorted_indices[:, 1:] != action_count)
    ).any(dim=1)
    if bool(duplicate.any()):
        raise ValueError("A legal action row contains duplicate indices")
    if targets is not None:
        if targets.shape != (batch,) or targets.device != logits.device:
            raise ValueError("targets must have shape [batch] on the logits device")
        matches = mask & (indices == targets.to(dtype=torch.long).unsqueeze(1))
        if bool((matches.sum(dim=1) != 1).any()):
            raise ValueError("Each target must occur exactly once in its legal action row")
    return torch.log_softmax(gathered, dim=1), mask


def target_log_odds(
    logits: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
    targets: Tensor,
) -> Tensor:
    """Target logit minus log-sum-exp of every other legal action."""

    log_probabilities, mask = legal_log_probabilities(
        logits,
        legal_indices,
        legal_counts,
        targets=targets,
    )
    matches = mask & (
        legal_indices.to(dtype=torch.long) == targets.to(dtype=torch.long).unsqueeze(1)
    )
    target_log_probability = torch.where(
        matches,
        log_probabilities,
        torch.zeros_like(log_probabilities),
    ).sum(dim=1)
    other = log_probabilities.masked_fill(matches, -torch.inf)
    return target_log_probability - torch.logsumexp(other, dim=1)


def _js_divergence(
    first_log_probability: Tensor,
    second_log_probability: Tensor,
    mask: Tensor,
) -> Tensor:
    first = first_log_probability.exp().masked_fill(~mask, 0.0)
    second = second_log_probability.exp().masked_fill(~mask, 0.0)
    midpoint = 0.5 * (first + second)
    midpoint_log = torch.where(mask, midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log(), 0)
    first_kl = torch.where(
        mask,
        first * (first_log_probability - midpoint_log),
        torch.zeros_like(first),
    ).sum(dim=1)
    second_kl = torch.where(
        mask,
        second * (second_log_probability - midpoint_log),
        torch.zeros_like(second),
    ).sum(dim=1)
    return (0.5 * (first_kl + second_kl)).clamp_min(0.0)


def policy_effect_metrics(
    baseline_logits: Tensor,
    reference_logits: Tensor,
    patched_logits: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
    targets: Tensor,
    *,
    effect_epsilon: float = 1e-8,
) -> dict[str, Tensor]:
    """Measure how a patch moves a baseline policy toward a reference policy.

    Distribution projection is the primary normalized mediation statistic:
    zero is baseline-like, one reaches the reference along its full legal
    policy delta, values above one overshoot, and negative values reverse.
    Target-log-odds mediation is reported separately and becomes NaN when the
    full target effect is unresolved.
    """

    shape = _validate_logits(baseline_logits, name="baseline_logits")
    if _validate_logits(reference_logits, name="reference_logits") != shape:
        raise ValueError("reference_logits shape differs from baseline_logits")
    if _validate_logits(patched_logits, name="patched_logits") != shape:
        raise ValueError("patched_logits shape differs from baseline_logits")
    if effect_epsilon <= 0.0 or not math.isfinite(effect_epsilon):
        raise ValueError("effect_epsilon must be finite and positive")

    baseline_logp, mask = legal_log_probabilities(
        baseline_logits,
        legal_indices,
        legal_counts,
        targets=targets,
    )
    reference_logp, reference_mask = legal_log_probabilities(
        reference_logits,
        legal_indices,
        legal_counts,
        targets=targets,
    )
    patched_logp, patched_mask = legal_log_probabilities(
        patched_logits,
        legal_indices,
        legal_counts,
        targets=targets,
    )
    if not torch.equal(mask, reference_mask) or not torch.equal(mask, patched_mask):
        raise AssertionError("Legal masks unexpectedly differ")
    baseline_probability = baseline_logp.exp().masked_fill(~mask, 0.0)
    reference_probability = reference_logp.exp().masked_fill(~mask, 0.0)
    patched_probability = patched_logp.exp().masked_fill(~mask, 0.0)

    full_delta = reference_probability - baseline_probability
    patch_delta = patched_probability - baseline_probability
    denominator = full_delta.square().sum(dim=1)
    resolved_distribution = denominator > effect_epsilon**2
    projection = (patch_delta * full_delta).sum(dim=1) / denominator.clamp_min(effect_epsilon**2)
    projection = torch.where(
        resolved_distribution,
        projection,
        torch.full_like(projection, torch.nan),
    )
    baseline_to_reference_l2 = denominator.sqrt()
    patched_to_reference_l2 = (
        (patched_probability - reference_probability).square().sum(dim=1).sqrt()
    )
    restoration = 1.0 - patched_to_reference_l2 / baseline_to_reference_l2.clamp_min(effect_epsilon)
    restoration = torch.where(
        resolved_distribution,
        restoration,
        torch.full_like(restoration, torch.nan),
    )

    baseline_odds = target_log_odds(
        baseline_logits,
        legal_indices,
        legal_counts,
        targets,
    )
    reference_odds = target_log_odds(
        reference_logits,
        legal_indices,
        legal_counts,
        targets,
    )
    patched_odds = target_log_odds(
        patched_logits,
        legal_indices,
        legal_counts,
        targets,
    )
    full_target_effect = reference_odds - baseline_odds
    patch_target_effect = patched_odds - baseline_odds
    resolved_target = full_target_effect.abs() > effect_epsilon
    target_mediation = patch_target_effect / torch.where(
        resolved_target,
        full_target_effect,
        torch.ones_like(full_target_effect),
    )
    target_mediation = torch.where(
        resolved_target,
        target_mediation,
        torch.full_like(target_mediation, torch.nan),
    )

    return {
        "baseline_to_reference_js": _js_divergence(
            baseline_logp,
            reference_logp,
            mask,
        ),
        "baseline_to_patched_js": _js_divergence(baseline_logp, patched_logp, mask),
        "patched_to_reference_js": _js_divergence(
            patched_logp,
            reference_logp,
            mask,
        ),
        "baseline_to_reference_probability_l2": baseline_to_reference_l2,
        "patched_to_reference_probability_l2": patched_to_reference_l2,
        "distribution_delta_projection": projection,
        "distribution_restoration": restoration,
        "distribution_effect_resolved": resolved_distribution,
        "distribution_overshoot": resolved_distribution & (projection > 1.0),
        "distribution_sign_reversal": resolved_distribution & (projection < 0.0),
        "baseline_target_log_odds": baseline_odds,
        "reference_target_log_odds": reference_odds,
        "patched_target_log_odds": patched_odds,
        "full_target_log_odds_effect": full_target_effect,
        "patch_target_log_odds_effect": patch_target_effect,
        "target_log_odds_mediation": target_mediation,
        "target_effect_resolved": resolved_target,
        "target_overshoot": resolved_target & (target_mediation > 1.0),
        "target_sign_reversal": resolved_target & (target_mediation < 0.0),
    }


def make_patch_tensor(
    source: Tensor,
    destination: Tensor,
    *,
    square_indices: Sequence[int] | None = None,
    feature_indices: Sequence[int] | None = None,
    scale: float = 1.0,
) -> Tensor:
    """Interpolate selected source cells into a destination [B, square, feature]."""

    if source.shape != destination.shape:
        raise ValueError("source and destination activation shapes differ")
    if source.ndim != 3:
        raise ValueError("Activation patches require [batch, square, feature] tensors")
    if source.dtype != destination.dtype or source.device != destination.device:
        raise ValueError("source and destination activations must share dtype and device")
    if not math.isfinite(scale):
        raise ValueError("Patch scale must be finite")
    square_count, feature_count = int(source.shape[1]), int(source.shape[2])

    def validated(values: Sequence[int] | None, upper: int, name: str) -> list[int]:
        selected = list(range(upper)) if values is None else list(values)
        if not selected:
            raise ValueError(f"{name} must not be empty")
        if any(type(value) is not int for value in selected):
            raise TypeError(f"{name} must contain integers")
        if len(set(selected)) != len(selected):
            raise ValueError(f"{name} must not contain duplicates")
        if min(selected) < 0 or max(selected) >= upper:
            raise ValueError(f"{name} must lie within [0, {upper - 1}]")
        return selected

    squares = validated(square_indices, square_count, "square_indices")
    features = validated(feature_indices, feature_count, "feature_indices")
    result = destination.clone()
    square_tensor = torch.tensor(squares, dtype=torch.long, device=source.device)
    feature_tensor = torch.tensor(features, dtype=torch.long, device=source.device)
    destination_selection = destination[:, square_tensor[:, None], feature_tensor[None, :]]
    source_selection = source[:, square_tensor[:, None], feature_tensor[None, :]]
    result[:, square_tensor[:, None], feature_tensor[None, :]] = destination_selection + scale * (
        source_selection - destination_selection
    )
    return result


def _captured_boundary(
    capture: BT4EncoderCapture | None,
    boundary: Boundary,
) -> Tensor:
    if capture is None:
        raise ValueError("Patch run did not return an encoder capture")
    value = getattr(capture, boundary)
    if value.shape[0] != 1:
        raise ValueError("Boundary patch expects exactly one captured layer")
    return value[0]


def patch_comparison_boundary(
    models: _ComparisonModels,
    destination_planes: Tensor,
    *,
    source_arm: ArmId,
    destination_arm: ArmId,
    layer: int,
    boundary: Boundary,
    compute_dtype: torch.dtype,
    legal_indices: Tensor,
    legal_counts: Tensor,
    targets: Tensor,
    source_planes: Tensor | None = None,
    square_indices: Sequence[int] | None = None,
    feature_indices: Sequence[int] | None = None,
    scale: float = 1.0,
) -> BoundaryPatchResult:
    """Patch one model/position into another under a fixed policy head.

    For raw-versus-Hero model diffing, the two arms must share their second
    letter so that the policy head is held fixed. source_planes may differ for
    clean/corrupt activation patching, but the supplied legal set always
    defines the destination policy comparison.
    """

    if source_arm[1] != destination_arm[1]:
        raise ValueError("Causal encoder patching requires a fixed policy head")
    if boundary not in BOUNDARIES:
        raise ValueError(f"Unknown boundary {boundary!r}")
    if type(layer) is not int or layer < 0:
        raise ValueError("layer must be a non-negative integer")
    source_input = destination_planes if source_planes is None else source_planes
    if source_input.shape != destination_planes.shape:
        raise ValueError("source_planes and destination_planes shapes differ")

    with torch.inference_mode():
        source_output = models.policy_logits_with_captures(
            source_input,
            arm=source_arm,
            compute_dtype=compute_dtype,
            capture_layers=(layer,),
        )
        destination_output = models.policy_logits_with_captures(
            destination_planes,
            arm=destination_arm,
            compute_dtype=compute_dtype,
            capture_layers=(layer,),
        )
        source_activation = _captured_boundary(source_output.captures, boundary)
        destination_activation = _captured_boundary(destination_output.captures, boundary)
        patch = make_patch_tensor(
            source_activation,
            destination_activation,
            square_indices=square_indices,
            feature_indices=feature_indices,
            scale=scale,
        )
        overrides: dict[str, Mapping[int, Tensor] | None] = {
            _OVERRIDE_ARGUMENT[boundary]: {layer: patch}
        }
        patched_output = models.policy_logits_with_captures(
            destination_planes,
            arm=destination_arm,
            compute_dtype=compute_dtype,
            capture_layers=(layer,),
            **overrides,
        )
        metrics = policy_effect_metrics(
            destination_output.logits,
            source_output.logits,
            patched_output.logits,
            legal_indices,
            legal_counts,
            targets,
        )
    return BoundaryPatchResult(
        source_arm=source_arm,
        destination_arm=destination_arm,
        layer=layer,
        boundary=boundary,
        source_logits=source_output.logits.detach(),
        destination_logits=destination_output.logits.detach(),
        patched_logits=patched_output.logits.detach(),
        patched_activation=patch.detach(),
        metrics={name: value.detach() for name, value in metrics.items()},
    )


def summarize_patch_metrics(metrics: Mapping[str, Tensor]) -> dict[str, Any]:
    """Convert per-position patch tensors into finite-aware scalar summaries."""

    result: dict[str, Any] = {}
    for name, value in sorted(metrics.items()):
        array = value.detach().cpu()
        if array.ndim != 1:
            raise ValueError(f"Patch metric {name} must be one-dimensional")
        if array.dtype == torch.bool:
            result[name] = {
                "count": int(array.numel()),
                "true_count": int(array.sum()),
                "rate": float(array.float().mean()),
            }
            continue
        finite = torch.isfinite(array)
        retained = array[finite].double()
        result[name] = {
            "count": int(array.numel()),
            "finite_count": int(finite.sum()),
            "mean": float(retained.mean()) if retained.numel() else None,
            "median": float(retained.median()) if retained.numel() else None,
            "minimum": float(retained.min()) if retained.numel() else None,
            "maximum": float(retained.max()) if retained.numel() else None,
        }
    return result
