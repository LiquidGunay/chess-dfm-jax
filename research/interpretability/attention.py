"""Exact BT4 attention decomposition and component-level interventions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from research.train_torch import BT4EncoderLayer


AttentionComponent = Literal[
    "query",
    "key",
    "value",
    "smolgen_bias",
    "pattern",
    "head_value_output",
]


@dataclass(frozen=True)
class BT4AttentionInternals:
    query: Tensor
    key: Tensor
    value: Tensor
    qk_logits: Tensor
    smolgen_bias: Tensor
    pattern: Tensor
    head_value_output: Tensor
    projected_head_contributions: Tensor
    projected_output: Tensor
    contribution_reconstruction_max_abs_error: float


def _override(name: str, native: Tensor, replacement: Tensor | None) -> Tensor:
    if replacement is None:
        return native
    if not isinstance(replacement, Tensor):
        raise TypeError(f"{name} override must be a Tensor")
    if replacement.shape != native.shape:
        raise ValueError(
            f"{name} override must have shape {tuple(native.shape)}, got {replacement.shape}"
        )
    if replacement.dtype != native.dtype or replacement.device != native.device:
        raise ValueError(f"{name} override must share native dtype and device")
    if not bool(torch.isfinite(replacement).all()):
        raise ValueError(f"{name} override contains nonfinite values")
    return replacement


def attention_internals(
    layer: BT4EncoderLayer,
    states: Tensor,
    *,
    compute_dtype: torch.dtype,
    query_override: Tensor | None = None,
    key_override: Tensor | None = None,
    value_override: Tensor | None = None,
    smolgen_bias_override: Tensor | None = None,
    pattern_override: Tensor | None = None,
    head_value_output_override: Tensor | None = None,
) -> BT4AttentionInternals:
    """Decompose one BT4 attention branch using its eager source equations."""

    if states.ndim != 3 or states.shape[1:] != (64, 1024):
        raise ValueError("BT4 attention states must have shape [batch, 64, 1024]")
    if not bool(torch.isfinite(states).all()):
        raise ValueError("BT4 attention states contain nonfinite values")
    batch = int(states.shape[0])
    query = states @ layer.wq.to(compute_dtype) + layer.wq_b.to(compute_dtype)
    key = states @ layer.wk.to(compute_dtype) + layer.wk_b.to(compute_dtype)
    value = states @ layer.wv.to(compute_dtype) + layer.wv_b.to(compute_dtype)
    query = query.reshape(batch, 64, 32, 32).transpose(1, 2)
    key = key.reshape(batch, 64, 32, 32).transpose(1, 2)
    value = value.reshape(batch, 64, 32, 32).transpose(1, 2)
    query = _override("query", query, query_override)
    key = _override("key", key, key_override)
    value = _override("value", value, value_override)
    qk_logits = (query @ key.transpose(-2, -1)) / math.sqrt(32.0)
    smolgen_bias = _override(
        "smolgen_bias",
        layer.smolgen(states, compute_dtype),
        smolgen_bias_override,
    )
    pattern = torch.softmax(qk_logits + smolgen_bias, dim=-1)
    if pattern_override is not None:
        pattern = _override("pattern", pattern, pattern_override)
        if bool((pattern < 0).any()) or not bool(
            torch.allclose(
                pattern.sum(dim=-1),
                torch.ones_like(pattern[..., 0]),
                rtol=1e-4,
                atol=1e-5,
            )
        ):
            raise ValueError("pattern override must be non-negative and row-normalized")
    head_value_output = pattern @ value
    head_value_output = _override(
        "head_value_output",
        head_value_output,
        head_value_output_override,
    )
    concatenated = head_value_output.transpose(1, 2).reshape(batch * 64, 1024)
    projected_output = layer.wo(concatenated, compute_dtype).reshape(batch, 64, 1024)

    weight_by_head = layer.wo.w.to(compute_dtype).reshape(32, 32, 1024)
    contributions = torch.einsum("bhsd,hdo->bhso", head_value_output, weight_by_head)
    reconstructed = contributions.sum(dim=1)
    if layer.wo.b is not None:
        reconstructed = reconstructed + layer.wo.b.to(compute_dtype)
    error = float((reconstructed - projected_output).abs().max())
    return BT4AttentionInternals(
        query=query,
        key=key,
        value=value,
        qk_logits=qk_logits,
        smolgen_bias=smolgen_bias,
        pattern=pattern,
        head_value_output=head_value_output,
        projected_head_contributions=contributions,
        projected_output=projected_output,
        contribution_reconstruction_max_abs_error=error,
    )


def replace_attention_heads(
    source: BT4AttentionInternals,
    destination: BT4AttentionInternals,
    head_indices: Sequence[int],
    *,
    scale: float = 1.0,
) -> Tensor:
    """Replace selected post-output-projection head contributions."""

    if source.projected_head_contributions.shape != destination.projected_head_contributions.shape:
        raise ValueError("Source and destination head contribution shapes differ")
    heads = list(head_indices)
    if not heads or any(type(head) is not int for head in heads):
        raise ValueError("head_indices must contain integers")
    if len(set(heads)) != len(heads) or min(heads) < 0 or max(heads) >= 32:
        raise ValueError("head_indices must be unique values in [0, 31]")
    if not math.isfinite(scale):
        raise ValueError("Head patch scale must be finite")
    difference = (
        source.projected_head_contributions[:, heads]
        - destination.projected_head_contributions[:, heads]
    ).sum(dim=1)
    return destination.projected_output + scale * difference


def replace_attention_component_heads(
    source: BT4AttentionInternals,
    destination: BT4AttentionInternals,
    component: AttentionComponent,
    head_indices: Sequence[int],
) -> Tensor:
    """Build a full component tensor with selected donor heads inserted."""

    source_value = getattr(source, component)
    destination_value = getattr(destination, component)
    if source_value.shape != destination_value.shape or source_value.ndim < 2:
        raise ValueError("Source and destination attention components differ")
    heads = list(head_indices)
    if not heads or len(set(heads)) != len(heads) or min(heads) < 0 or max(heads) >= 32:
        raise ValueError("head_indices must be unique values in [0, 31]")
    result = destination_value.clone()
    result[:, heads] = source_value[:, heads]
    return result


def attention_move_alignment(
    pattern: Tensor,
    origin_tokens: Tensor,
    destination_tokens: Tensor,
) -> dict[str, Tensor]:
    """Per-head attention mass and top-1 hits along target move endpoints."""

    if pattern.ndim != 4 or pattern.shape[1:] != (32, 64, 64):
        raise ValueError("Attention pattern must have shape [batch, 32, 64, 64]")
    batch = int(pattern.shape[0])
    if origin_tokens.shape != (batch,) or destination_tokens.shape != (batch,):
        raise ValueError("Move endpoints must have one token per position")
    origin = origin_tokens.to(device=pattern.device, dtype=torch.long)
    destination = destination_tokens.to(device=pattern.device, dtype=torch.long)
    if bool(((origin < 0) | (origin >= 64) | (destination < 0) | (destination >= 64)).any()):
        raise ValueError("Move endpoints must lie in [0, 63]")
    rows = torch.arange(batch, device=pattern.device)
    heads = torch.arange(32, device=pattern.device)
    forward = pattern[rows[:, None], heads[None, :], origin[:, None], destination[:, None]]
    reverse = pattern[rows[:, None], heads[None, :], destination[:, None], origin[:, None]]
    forward_argmax = pattern[rows[:, None], heads[None, :], origin[:, None]].argmax(dim=-1)
    reverse_argmax = pattern[rows[:, None], heads[None, :], destination[:, None]].argmax(dim=-1)
    entropy = (
        -(pattern * pattern.clamp_min(torch.finfo(pattern.dtype).tiny).log())
        .sum(dim=-1)
        .mean(dim=-1)
    )
    return {
        "origin_to_destination_mass": forward,
        "destination_to_origin_mass": reverse,
        "symmetric_endpoint_mass": 0.5 * (forward + reverse),
        "origin_to_destination_top1": forward_argmax == destination[:, None],
        "destination_to_origin_top1": reverse_argmax == origin[:, None],
        "mean_query_entropy": entropy,
    }


def future_move_attention_alignment(
    pattern: Tensor,
    future_origins: Tensor,
    future_destinations: Tensor,
    future_valid: Tensor,
) -> dict[str, Tensor]:
    """Extend endpoint alignment to each available future solution ply."""

    if (
        future_origins.shape != future_destinations.shape
        or future_valid.shape != future_origins.shape
    ):
        raise ValueError("Future move endpoint arrays must be aligned")
    if future_origins.ndim != 2 or future_origins.shape[0] != pattern.shape[0]:
        raise ValueError("Future move arrays must have shape [batch, ply]")
    masses: list[Tensor] = []
    hits: list[Tensor] = []
    for ply in range(future_origins.shape[1]):
        valid = future_valid[:, ply].bool()
        safe_origin = torch.where(valid, future_origins[:, ply], 0)
        safe_destination = torch.where(valid, future_destinations[:, ply], 0)
        report = attention_move_alignment(pattern, safe_origin, safe_destination)
        masses.append(report["symmetric_endpoint_mass"].masked_fill(~valid[:, None], torch.nan))
        hits.append(
            report["origin_to_destination_top1"]
            .float()
            .masked_fill(
                ~valid[:, None],
                torch.nan,
            )
        )
    return {
        "symmetric_endpoint_mass": torch.stack(masses, dim=1),
        "origin_to_destination_top1": torch.stack(hits, dim=1),
    }


def qk_smolgen_balance(internals: BT4AttentionInternals) -> dict[str, Tensor]:
    """Compare QK and SmolGen logit scales per position and head."""

    qk_rms = internals.qk_logits.float().square().mean(dim=(-2, -1)).sqrt()
    smolgen_rms = internals.smolgen_bias.float().square().mean(dim=(-2, -1)).sqrt()
    combined_rms = (
        (internals.qk_logits + internals.smolgen_bias).float().square().mean(dim=(-2, -1)).sqrt()
    )
    return {
        "qk_rms": qk_rms,
        "smolgen_rms": smolgen_rms,
        "combined_logit_rms": combined_rms,
        "smolgen_to_qk_rms_ratio": smolgen_rms / qk_rms.clamp_min(1e-12),
    }
