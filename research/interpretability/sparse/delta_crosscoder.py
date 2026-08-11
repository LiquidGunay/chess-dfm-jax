"""Joint sparse decomposition of paired Raw and Hero BT4 activations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from research.interpretability.sparse.transcoder import (
    positive_topk,
    reconstruction_metrics,
)


@dataclass(frozen=True)
class DeltaCrosscoderOutput:
    shared_features: Tensor
    raw_specific_features: Tensor
    hero_specific_features: Tensor
    shared_raw_component: Tensor
    shared_hero_component: Tensor
    raw_specific_component: Tensor
    hero_specific_component: Tensor
    raw_reconstruction: Tensor
    hero_reconstruction: Tensor


class PairedDeltaCrosscoder(nn.Module):
    """Joint shared/model-specific dictionary over aligned activation pairs.

    Shared features are inferred from both models and have one decoder effect
    per model.  Explicit Raw-only and Hero-only feature banks capture residual
    structure.  Joint feature identity avoids post-hoc matching of independent
    sparse dictionaries.
    """

    def __init__(
        self,
        d_model: int,
        *,
        shared_features: int,
        specific_features: int,
        shared_k: int,
        specific_k: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        values = (d_model, shared_features, specific_features, shared_k, specific_k)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Delta-Crosscoder dimensions and sparsities must be positive")
        if shared_k > shared_features or specific_k > specific_features:
            raise ValueError("Sparse k cannot exceed its feature-bank width")
        self.d_model = d_model
        self.shared_feature_count = shared_features
        self.specific_feature_count = specific_features
        self.shared_k = shared_k
        self.specific_k = specific_k
        factory = {"device": device, "dtype": dtype}
        self.E_shared_raw = nn.Parameter(torch.empty((d_model, shared_features), **factory))
        self.E_shared_hero = nn.Parameter(torch.empty((d_model, shared_features), **factory))
        self.b_shared = nn.Parameter(torch.empty((shared_features,), **factory))
        self.D_shared_raw = nn.Parameter(torch.empty((shared_features, d_model), **factory))
        self.D_shared_hero = nn.Parameter(torch.empty((shared_features, d_model), **factory))

        self.E_raw_specific = nn.Parameter(torch.empty((d_model, specific_features), **factory))
        self.b_raw_specific = nn.Parameter(torch.empty((specific_features,), **factory))
        self.D_raw_specific = nn.Parameter(torch.empty((specific_features, d_model), **factory))
        self.E_hero_specific = nn.Parameter(torch.empty((d_model, specific_features), **factory))
        self.b_hero_specific = nn.Parameter(torch.empty((specific_features,), **factory))
        self.D_hero_specific = nn.Parameter(torch.empty((specific_features, d_model), **factory))
        self.b_raw = nn.Parameter(torch.empty((d_model,), **factory))
        self.b_hero = nn.Parameter(torch.empty((d_model,), **factory))

    def _validate_pair(self, raw: Tensor, hero: Tensor) -> None:
        if raw.shape != hero.shape or raw.shape[-1] != self.d_model:
            raise ValueError("Raw/Hero activation pairs must align on d_model")
        if not torch.is_floating_point(raw) or raw.dtype != hero.dtype or raw.device != hero.device:
            raise ValueError("Raw/Hero activations must share floating dtype and device")

    def forward(self, raw: Tensor, hero: Tensor) -> DeltaCrosscoderOutput:
        self._validate_pair(raw, hero)
        shared_pre = (raw @ self.E_shared_raw + hero @ self.E_shared_hero) / math.sqrt(
            2.0
        ) + self.b_shared
        shared = positive_topk(shared_pre, self.shared_k, mode="exact")
        raw_pre = raw @ self.E_raw_specific + self.b_raw_specific
        hero_pre = hero @ self.E_hero_specific + self.b_hero_specific
        raw_specific = positive_topk(raw_pre, self.specific_k, mode="exact")
        hero_specific = positive_topk(hero_pre, self.specific_k, mode="exact")
        shared_raw_component = shared @ self.D_shared_raw
        shared_hero_component = shared @ self.D_shared_hero
        raw_specific_component = raw_specific @ self.D_raw_specific
        hero_specific_component = hero_specific @ self.D_hero_specific
        return DeltaCrosscoderOutput(
            shared_features=shared,
            raw_specific_features=raw_specific,
            hero_specific_features=hero_specific,
            shared_raw_component=shared_raw_component,
            shared_hero_component=shared_hero_component,
            raw_specific_component=raw_specific_component,
            hero_specific_component=hero_specific_component,
            raw_reconstruction=shared_raw_component + raw_specific_component + self.b_raw,
            hero_reconstruction=shared_hero_component + hero_specific_component + self.b_hero,
        )


def initialize_delta_crosscoder(model: PairedDeltaCrosscoder, *, seed: int) -> None:
    if next(model.parameters()).device.type == "meta":
        raise ValueError("Cannot initialize a meta-device Delta-Crosscoder")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def unit_rows(shape: torch.Size) -> Tensor:
        value = torch.randn(shape, generator=generator)
        return value / value.norm(dim=1, keepdim=True).clamp_min(1e-12)

    with torch.no_grad():
        shared_seed = unit_rows(model.D_shared_raw.shape)
        raw_specific_seed = unit_rows(model.D_raw_specific.shape)
        hero_specific_seed = unit_rows(model.D_hero_specific.shape)
        model.D_shared_raw.copy_(shared_seed.to(model.D_shared_raw))
        model.D_shared_hero.copy_(shared_seed.to(model.D_shared_hero))
        model.D_raw_specific.copy_(raw_specific_seed.to(model.D_raw_specific))
        model.D_hero_specific.copy_(hero_specific_seed.to(model.D_hero_specific))
        model.E_shared_raw.copy_((shared_seed.T / math.sqrt(2.0)).to(model.E_shared_raw))
        model.E_shared_hero.copy_((shared_seed.T / math.sqrt(2.0)).to(model.E_shared_hero))
        model.E_raw_specific.copy_(raw_specific_seed.T.to(model.E_raw_specific))
        model.E_hero_specific.copy_(hero_specific_seed.T.to(model.E_hero_specific))
        for bias in (
            model.b_shared,
            model.b_raw_specific,
            model.b_hero_specific,
            model.b_raw,
            model.b_hero,
        ):
            bias.zero_()


def delta_crosscoder_metrics(
    raw: Tensor,
    hero: Tensor,
    output: DeltaCrosscoderOutput,
) -> dict[str, Any]:
    if raw.shape != hero.shape:
        raise ValueError("Raw/Hero metrics require aligned activations")
    delta = hero - raw
    reconstructed_delta = output.hero_reconstruction - output.raw_reconstruction
    shared_cosine = F.cosine_similarity(
        output.shared_raw_component.double().reshape(-1, raw.shape[-1]),
        output.shared_hero_component.double().reshape(-1, raw.shape[-1]),
        dim=1,
    )
    return {
        "raw_reconstruction": reconstruction_metrics(raw, output.raw_reconstruction),
        "hero_reconstruction": reconstruction_metrics(hero, output.hero_reconstruction),
        "delta_reconstruction": reconstruction_metrics(delta, reconstructed_delta),
        "sparsity": {
            "shared_mean_l0": float((output.shared_features > 0).sum(dim=-1).double().mean()),
            "raw_specific_mean_l0": float(
                (output.raw_specific_features > 0).sum(dim=-1).double().mean()
            ),
            "hero_specific_mean_l0": float(
                (output.hero_specific_features > 0).sum(dim=-1).double().mean()
            ),
            "shared_dead_fraction": float(
                (~(output.shared_features > 0).reshape(-1, output.shared_features.shape[-1]).any(0))
                .double()
                .mean()
            ),
        },
        "shared_effect": {
            "mean_raw_hero_component_cosine": float(shared_cosine.mean()),
            "rms_effect_delta": float(
                (output.shared_hero_component - output.shared_raw_component)
                .double()
                .square()
                .mean()
                .sqrt()
            ),
        },
    }


def shared_feature_taxonomy(
    model: PairedDeltaCrosscoder,
    *,
    ratio_threshold: float = 2.0,
    minimum_norm: float = 1e-8,
) -> dict[str, Any]:
    """Classify joint shared features by cross-model decoder effect norms."""

    if ratio_threshold <= 1 or minimum_norm <= 0:
        raise ValueError("Taxonomy thresholds must be positive and ratio > 1")
    raw_norm = model.D_shared_raw.detach().double().norm(dim=1)
    hero_norm = model.D_shared_hero.detach().double().norm(dim=1)
    live_raw = raw_norm >= minimum_norm
    live_hero = hero_norm >= minimum_norm
    ratio = hero_norm / raw_norm.clamp_min(minimum_norm)
    shared = live_raw & live_hero & (ratio >= 1 / ratio_threshold) & (ratio <= ratio_threshold)
    raw_dominant = live_raw & ((~live_hero) | (ratio < 1 / ratio_threshold))
    hero_dominant = live_hero & ((~live_raw) | (ratio > ratio_threshold))
    dead = ~live_raw & ~live_hero
    cosine = F.cosine_similarity(
        model.D_shared_raw.detach().double(),
        model.D_shared_hero.detach().double(),
        dim=1,
        eps=minimum_norm,
    )
    return {
        "shared_count": int(shared.sum()),
        "raw_dominant_count": int(raw_dominant.sum()),
        "hero_dominant_count": int(hero_dominant.sum()),
        "dead_count": int(dead.sum()),
        "unclassified_count": int(
            model.shared_feature_count - (shared | raw_dominant | hero_dominant | dead).sum()
        ),
        "mean_decoder_cosine": float(cosine.mean()),
        "median_hero_to_raw_norm_ratio": float(ratio.median()),
        "ratio_threshold": ratio_threshold,
    }


def remove_sparse_features(
    output: DeltaCrosscoderOutput,
    model: PairedDeltaCrosscoder,
    *,
    shared_indices: Tensor | None = None,
    raw_specific_indices: Tensor | None = None,
    hero_specific_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Decode counterfactuals with selected feature effects removed."""

    def removed(features: Tensor, decoder: Tensor, indices: Tensor | None) -> Tensor:
        if indices is None:
            return torch.zeros_like(features[..., :1] @ decoder[:1])
        selected = indices.to(device=features.device, dtype=torch.long)
        if selected.ndim != 1 or selected.numel() == 0:
            raise ValueError("Feature removal indices must be a non-empty vector")
        if bool(((selected < 0) | (selected >= features.shape[-1])).any()):
            raise ValueError("Feature removal index outside its bank")
        return features.index_select(-1, selected) @ decoder.index_select(0, selected)

    shared_raw_removed = removed(
        output.shared_features,
        model.D_shared_raw,
        shared_indices,
    )
    shared_hero_removed = removed(
        output.shared_features,
        model.D_shared_hero,
        shared_indices,
    )
    raw_removed = removed(
        output.raw_specific_features,
        model.D_raw_specific,
        raw_specific_indices,
    )
    hero_removed = removed(
        output.hero_specific_features,
        model.D_hero_specific,
        hero_specific_indices,
    )
    return (
        output.raw_reconstruction - shared_raw_removed - raw_removed,
        output.hero_reconstruction - shared_hero_removed - hero_removed,
    )


@dataclass(frozen=True)
class FittedDeltaCrosscoder:
    module: PairedDeltaCrosscoder
    metadata: dict[str, Any]


def fit_delta_crosscoder(
    raw: Tensor,
    hero: Tensor,
    *,
    shared_features: int,
    specific_features: int,
    shared_k: int,
    specific_k: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    delta_loss_coefficient: float = 1.0,
    shared_alignment_coefficient: float = 1e-3,
    device: torch.device | str = "cpu",
) -> FittedDeltaCrosscoder:
    """Fit a bounded paired sparse model on aligned development activations."""

    raw_value = raw.detach().float().reshape(-1, raw.shape[-1]).cpu()
    hero_value = hero.detach().float().reshape(-1, hero.shape[-1]).cpu()
    if raw_value.shape != hero_value.shape or raw_value.shape[0] == 0:
        raise ValueError("Delta-Crosscoder training pairs must be aligned and non-empty")
    if not bool(torch.isfinite(raw_value).all() and torch.isfinite(hero_value).all()):
        raise ValueError("Delta-Crosscoder data contain nonfinite values")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("Invalid Delta-Crosscoder training configuration")
    if delta_loss_coefficient < 0 or shared_alignment_coefficient < 0:
        raise ValueError("Delta-Crosscoder loss coefficients must be non-negative")
    active_device = torch.device(device)
    model = PairedDeltaCrosscoder(
        raw_value.shape[1],
        shared_features=shared_features,
        specific_features=specific_features,
        shared_k=shared_k,
        specific_k=specific_k,
        device=active_device,
    )
    initialize_delta_crosscoder(model, seed=seed)
    with torch.no_grad():
        model.b_raw.copy_(raw_value.mean(0).to(model.b_raw))
        model.b_hero.copy_(hero_value.mean(0).to(model.b_hero))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for _epoch in range(epochs):
        order = torch.randperm(raw_value.shape[0], generator=generator)
        total = 0.0
        for start in range(0, raw_value.shape[0], batch_size):
            rows = order[start : start + batch_size]
            batch_raw = raw_value[rows].to(active_device)
            batch_hero = hero_value[rows].to(active_device)
            output = model(batch_raw, batch_hero)
            reconstruction_loss = 0.5 * (
                F.mse_loss(output.raw_reconstruction, batch_raw)
                + F.mse_loss(output.hero_reconstruction, batch_hero)
            )
            delta_loss = F.mse_loss(
                output.hero_reconstruction - output.raw_reconstruction,
                batch_hero - batch_raw,
            )
            shared_alignment = F.mse_loss(model.D_shared_raw, model.D_shared_hero)
            loss = (
                reconstruction_loss
                + delta_loss_coefficient * delta_loss
                + shared_alignment_coefficient * shared_alignment
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * rows.numel()
        losses.append(total / raw_value.shape[0])
    model = model.cpu().eval()
    return FittedDeltaCrosscoder(
        module=model,
        metadata={
            "schema_version": "bt4-delta-crosscoder-pilot-v1",
            "training_pairs": int(raw_value.shape[0]),
            "d_model": int(raw_value.shape[1]),
            "shared_features": shared_features,
            "specific_features_per_model": specific_features,
            "shared_k": shared_k,
            "specific_k": specific_k,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
            "delta_loss_coefficient": delta_loss_coefficient,
            "shared_alignment_coefficient": shared_alignment_coefficient,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "scope": "bounded development pilot; heldout causal evaluation required",
        },
    )
