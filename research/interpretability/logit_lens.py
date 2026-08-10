"""Post-layer policy lenses for BT4's nonlinear attention-policy readout."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor

from research.interpretability.patching import (
    legal_log_probabilities,
    target_log_odds,
)
from research.train_torch import BT4EncoderCapture


LensBoundary = Literal[
    "hook_attn_in",
    "resid_mid_after_ln",
    "resid_post_after_ln",
]
LENS_BOUNDARIES: tuple[LensBoundary, ...] = (
    "hook_attn_in",
    "resid_mid_after_ln",
    "resid_post_after_ln",
)


def intermediate_policy_logits(
    capture: BT4EncoderCapture,
    head: Callable[[Tensor, torch.dtype], Tensor],
    *,
    compute_dtype: torch.dtype,
    boundary: LensBoundary = "resid_post_after_ln",
    layer_chunk_size: int = 4,
) -> Tensor:
    """Apply a policy head to each captured residual state.

    Output is [layer, batch, action]. The operation is an empirical readout,
    not an assumption that intermediate states live on the final residual
    distribution.
    """

    if boundary not in LENS_BOUNDARIES:
        raise ValueError(f"Unknown lens boundary {boundary!r}")
    if type(layer_chunk_size) is not int or layer_chunk_size <= 0:
        raise ValueError("layer_chunk_size must be positive")
    states = getattr(capture, boundary)
    if states.ndim != 4 or states.shape[0] != len(capture.layer_indices) or states.shape[2] != 64:
        raise ValueError("Lens states must have shape [layer, batch, 64, feature]")
    if not bool(torch.isfinite(states).all()):
        raise ValueError("Lens states contain nonfinite values")
    layer_count, batch = int(states.shape[0]), int(states.shape[1])
    chunks: list[Tensor] = []
    for start in range(0, layer_count, layer_chunk_size):
        value = states[start : start + layer_chunk_size]
        flat = value.reshape(-1, value.shape[2], value.shape[3])
        logits = head(flat, compute_dtype)
        if logits.ndim != 2 or logits.shape[0] != flat.shape[0]:
            raise ValueError("Policy head returned an invalid lens shape")
        chunks.append(logits.reshape(value.shape[0], batch, logits.shape[1]))
    result = torch.cat(chunks, dim=0)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Policy lens returned nonfinite logits")
    return result


def cross_head_lenses(
    raw_capture: BT4EncoderCapture,
    hero_capture: BT4EncoderCapture,
    raw_head: Callable[[Tensor, torch.dtype], Tensor],
    hero_head: Callable[[Tensor, torch.dtype], Tensor],
    *,
    compute_dtype: torch.dtype,
    boundary: LensBoundary = "resid_post_after_ln",
) -> dict[str, Tensor]:
    """RR/HR/RH/HH depth trajectories under both fixed policy heads."""

    if raw_capture.layer_indices != hero_capture.layer_indices:
        raise ValueError("Raw and Hero lens depths differ")
    return {
        "RR": intermediate_policy_logits(
            raw_capture,
            raw_head,
            compute_dtype=compute_dtype,
            boundary=boundary,
        ),
        "HR": intermediate_policy_logits(
            hero_capture,
            raw_head,
            compute_dtype=compute_dtype,
            boundary=boundary,
        ),
        "RH": intermediate_policy_logits(
            raw_capture,
            hero_head,
            compute_dtype=compute_dtype,
            boundary=boundary,
        ),
        "HH": intermediate_policy_logits(
            hero_capture,
            hero_head,
            compute_dtype=compute_dtype,
            boundary=boundary,
        ),
    }


def logit_lens_metrics(
    logits: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
    targets: Tensor,
) -> dict[str, Tensor]:
    """Return per-layer, per-position emergence and calibration primitives."""

    if logits.ndim != 3:
        raise ValueError("Lens logits must have shape [layer, batch, action]")
    layer_count, batch, action_count = logits.shape
    if legal_indices.ndim != 2 or legal_indices.shape[0] != batch:
        raise ValueError("legal_indices must align with lens batch")
    if legal_counts.shape != (batch,) or targets.shape != (batch,):
        raise ValueError("legal_counts and targets must align with lens batch")
    repeated_legal = (
        legal_indices.unsqueeze(0)
        .expand(layer_count, -1, -1)
        .reshape(
            layer_count * batch,
            -1,
        )
    )
    repeated_counts = legal_counts.unsqueeze(0).expand(layer_count, -1).reshape(-1)
    repeated_targets = targets.unsqueeze(0).expand(layer_count, -1).reshape(-1)
    flat_logits = logits.reshape(layer_count * batch, action_count)
    legal_logp, mask = legal_log_probabilities(
        flat_logits,
        repeated_legal,
        repeated_counts,
        targets=repeated_targets,
    )
    target_matches = mask & (
        repeated_legal.to(dtype=torch.long) == repeated_targets.to(dtype=torch.long).unsqueeze(1)
    )
    target_log_probability = torch.where(target_matches, legal_logp, 0.0).sum(dim=1)
    legal_probability = legal_logp.exp().masked_fill(~mask, 0.0)
    target_probability = target_log_probability.exp()
    target_legal_logit = torch.where(
        target_matches,
        flat_logits.gather(
            1,
            torch.where(mask, repeated_legal.to(dtype=torch.long), 0),
        ),
        0.0,
    ).sum(dim=1)
    gathered = flat_logits.gather(
        1,
        torch.where(mask, repeated_legal.to(dtype=torch.long), 0),
    ).masked_fill(~mask, -torch.inf)
    rank = 1 + ((gathered > target_legal_logit.unsqueeze(1)) & mask).sum(dim=1)
    entropy = -(legal_probability * legal_logp.masked_fill(~mask, 0.0)).sum(dim=1)
    full_probability = torch.softmax(flat_logits, dim=1)
    legal_mass = (
        full_probability.gather(
            1,
            torch.where(mask, repeated_legal.to(dtype=torch.long), 0),
        )
        .masked_fill(~mask, 0.0)
        .sum(dim=1)
    )
    odds = target_log_odds(
        flat_logits,
        repeated_legal,
        repeated_counts,
        repeated_targets,
    )

    shape = (layer_count, batch)
    return {
        "target_log_probability": target_log_probability.reshape(shape),
        "target_probability": target_probability.reshape(shape),
        "target_nll": (-target_log_probability).reshape(shape),
        "target_rank": rank.reshape(shape),
        "top1_accuracy": (rank == 1).reshape(shape),
        "top3_accuracy": (rank <= 3).reshape(shape),
        "entropy": entropy.reshape(shape),
        "legal_mass_before_masking": legal_mass.reshape(shape),
        "target_log_odds": odds.reshape(shape),
    }


def first_emergence_layer(
    metric: Tensor,
    layer_indices: Sequence[int],
    *,
    threshold: float,
    comparison: Literal["ge", "le"] = "ge",
) -> Tensor:
    """First actual layer satisfying a per-position threshold, or -1 if absent."""

    if metric.ndim != 2 or metric.shape[0] != len(layer_indices):
        raise ValueError("metric must have shape [layer, position]")
    layers = torch.as_tensor(layer_indices, dtype=torch.long, device=metric.device)
    if tuple(sorted(set(int(value) for value in layers.tolist()))) != tuple(layers.tolist()):
        raise ValueError("layer_indices must be unique and increasing")
    selected = metric >= threshold if comparison == "ge" else metric <= threshold
    if comparison not in {"ge", "le"}:
        raise ValueError("comparison must be 'ge' or 'le'")
    sentinel = torch.full_like(layers, -1)
    candidates = torch.where(selected, layers[:, None], sentinel[:, None])
    maximum = layers[-1] + 1
    sortable = torch.where(candidates >= 0, candidates, maximum)
    first = sortable.min(dim=0).values
    return torch.where(first == maximum, -1, first)


def validate_final_lens(
    lens_logits: Tensor,
    native_logits: Tensor,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Gate logit-lens use on equality of its final state and native readout."""

    if lens_logits.ndim != 3 or native_logits.ndim != 2:
        raise ValueError("Expected lens [layer,batch,action] and native [batch,action]")
    if lens_logits.shape[1:] != native_logits.shape:
        raise ValueError("Final lens and native logit shapes differ")
    delta = lens_logits[-1] - native_logits
    passed = bool(torch.allclose(lens_logits[-1], native_logits, rtol=rtol, atol=atol))
    return {
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_error": float(delta.abs().max()),
        "rms_error": float(delta.float().square().mean().sqrt()),
    }


def summarize_lens_metrics(
    metrics: Mapping[str, Tensor],
    layer_indices: Sequence[int],
) -> dict[str, Any]:
    """Finite-aware depth summaries without retaining full logits."""

    result: dict[str, Any] = {"layer_indices": list(layer_indices), "metrics": {}}
    for name, value in sorted(metrics.items()):
        if value.ndim != 2 or value.shape[0] != len(layer_indices):
            raise ValueError(f"Lens metric {name} has invalid layer dimension")
        if value.dtype == torch.bool:
            means = value.float().mean(dim=1)
        else:
            finite = torch.isfinite(value)
            if not bool(finite.all()):
                raise ValueError(f"Lens metric {name} contains nonfinite values")
            means = value.double().mean(dim=1)
        result["metrics"][name] = {
            str(layer): float(means[offset]) for offset, layer in enumerate(layer_indices)
        }
    return result
