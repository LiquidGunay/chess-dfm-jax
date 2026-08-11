"""Paired raw-to-Hero representation deltas, subspaces, and mediators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _flatten_rows(value: Tensor, *, name: str) -> tuple[Tensor, tuple[int, ...]]:
    if value.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape [row, feature] or [position, square, feature]")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")
    prefix = tuple(value.shape[:-1])
    return value.reshape(-1, value.shape[-1]), prefix


@dataclass(frozen=True)
class OrthogonalAlignment:
    raw_mean: Tensor
    hero_mean: Tensor
    rotation: Tensor
    training_row_count: int
    training_residual_rms: float

    def transform_raw(self, raw: Tensor) -> Tensor:
        if raw.shape[-1] != self.rotation.shape[0]:
            raise ValueError("Raw activation width differs from alignment")
        mean = self.raw_mean.to(device=raw.device, dtype=raw.dtype)
        target_mean = self.hero_mean.to(device=raw.device, dtype=raw.dtype)
        rotation = self.rotation.to(device=raw.device, dtype=raw.dtype)
        return (raw - mean) @ rotation + target_mean


def fit_orthogonal_alignment(
    raw: Tensor,
    hero: Tensor,
    *,
    max_rows: int | None = None,
    seed: int = 20260808,
) -> OrthogonalAlignment:
    """Fit a train-only orthogonal Procrustes map from raw to Hero states."""

    raw_rows, _ = _flatten_rows(raw, name="raw")
    hero_rows, _ = _flatten_rows(hero, name="hero")
    if raw_rows.shape != hero_rows.shape or raw_rows.shape[0] < 2:
        raise ValueError("Raw and Hero alignment rows must match and contain at least two rows")
    if max_rows is not None:
        if type(max_rows) is not int or max_rows < 2:
            raise ValueError("max_rows must be at least two")
        if max_rows < raw_rows.shape[0]:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            selected = torch.randperm(raw_rows.shape[0], generator=generator)[:max_rows]
            selected = selected.sort().values.to(raw_rows.device)
            raw_rows = raw_rows.index_select(0, selected)
            hero_rows = hero_rows.index_select(0, selected)
    raw64 = raw_rows.detach().cpu().double()
    hero64 = hero_rows.detach().cpu().double()
    raw_mean = raw64.mean(dim=0)
    hero_mean = hero64.mean(dim=0)
    cross = (raw64 - raw_mean).T @ (hero64 - hero_mean)
    left, _singular, right_h = torch.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_h
    aligned = (raw64 - raw_mean) @ rotation + hero_mean
    residual_rms = float((aligned - hero64).square().mean().sqrt())
    return OrthogonalAlignment(
        raw_mean=raw_mean.float(),
        hero_mean=hero_mean.float(),
        rotation=rotation.float(),
        training_row_count=int(raw64.shape[0]),
        training_residual_rms=residual_rms,
    )


def _fix_component_signs(components: Tensor) -> Tensor:
    result = components.clone()
    for index in range(result.shape[0]):
        pivot = int(result[index].abs().argmax())
        if result[index, pivot] < 0:
            result[index].neg_()
    return result


@dataclass(frozen=True)
class DeltaSubspace:
    mean: Tensor
    components: Tensor
    eigenvalues: Tensor
    total_variance: float
    training_row_count: int
    aligned: bool

    def project_delta(self, delta: Tensor) -> Tensor:
        if delta.shape[-1] != self.mean.numel():
            raise ValueError("Delta width differs from fitted subspace")
        mean = self.mean.to(device=delta.device, dtype=delta.dtype)
        components = self.components.to(device=delta.device, dtype=delta.dtype)
        return (delta - mean) @ components.T

    def reconstruct_delta(self, delta: Tensor) -> Tensor:
        scores = self.project_delta(delta)
        mean = self.mean.to(device=delta.device, dtype=delta.dtype)
        components = self.components.to(device=delta.device, dtype=delta.dtype)
        return scores @ components + mean

    @property
    def explained_variance_ratio(self) -> Tensor:
        if self.total_variance <= 0.0:
            return torch.zeros_like(self.eigenvalues)
        return self.eigenvalues / self.total_variance


def fit_delta_subspace(
    raw: Tensor,
    hero: Tensor,
    *,
    component_count: int,
    alignment: OrthogonalAlignment | None = None,
) -> DeltaSubspace:
    """Fit principal directions of Hero minus aligned-raw on development rows."""

    raw_rows, _ = _flatten_rows(raw, name="raw")
    hero_rows, _ = _flatten_rows(hero, name="hero")
    if raw_rows.shape != hero_rows.shape or raw_rows.shape[0] < 2:
        raise ValueError("Raw and Hero delta rows must be aligned")
    width = int(raw_rows.shape[1])
    if type(component_count) is not int or not 1 <= component_count <= width:
        raise ValueError("component_count must lie within feature width")
    aligned_raw = raw_rows if alignment is None else alignment.transform_raw(raw_rows)
    delta = (hero_rows - aligned_raw).detach().cpu().double()
    mean = delta.mean(dim=0)
    centered = delta - mean
    covariance = centered.T @ centered / max(1, delta.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    components = eigenvectors[:, order[:component_count]].T
    components = _fix_component_signs(components)
    return DeltaSubspace(
        mean=mean.float(),
        components=components.float(),
        eigenvalues=eigenvalues[:component_count].float(),
        total_variance=float(eigenvalues.sum()),
        training_row_count=int(delta.shape[0]),
        aligned=alignment is not None,
    )


class DeltaCovarianceAccumulator:
    """Bounded exact covariance and concentration statistics for paired deltas."""

    def __init__(
        self,
        feature_width: int,
        *,
        alignment: OrthogonalAlignment | None = None,
    ) -> None:
        if feature_width <= 0:
            raise ValueError("feature_width must be positive")
        self.feature_width = feature_width
        self.alignment = alignment
        self.row_count = 0
        self.position_count = 0
        self.delta_sum = torch.zeros(feature_width, dtype=torch.float64)
        self.delta_gram = torch.zeros((feature_width, feature_width), dtype=torch.float64)
        self.square_energy_sum = torch.zeros(64, dtype=torch.float64)
        self._position_rms: list[Tensor] = []
        self._behavior: list[Tensor] = []

    def update(
        self,
        raw: Tensor,
        hero: Tensor,
        *,
        behavior_delta: Tensor | None = None,
    ) -> None:
        if raw.shape != hero.shape or raw.ndim != 3 or raw.shape[1:] != (64, self.feature_width):
            raise ValueError("Delta accumulator expects paired [batch, 64, feature] tensors")
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(hero).all()):
            raise ValueError("Delta accumulator received nonfinite activations")
        aligned_raw = raw if self.alignment is None else self.alignment.transform_raw(raw)
        delta = (hero - aligned_raw).detach()
        flat = delta.reshape(-1, self.feature_width).cpu().double()
        self.delta_sum += flat.sum(dim=0)
        self.delta_gram += flat.T @ flat
        energy = delta.float().square().sum(dim=2).detach().cpu().double()
        self.square_energy_sum += energy.sum(dim=0)
        position_rms = delta.float().square().mean(dim=(1, 2)).sqrt().detach().cpu().double()
        self._position_rms.append(position_rms)
        if behavior_delta is not None:
            if behavior_delta.shape != (raw.shape[0],):
                raise ValueError("behavior_delta must have one value per position")
            if not bool(torch.isfinite(behavior_delta).all()):
                raise ValueError("behavior_delta contains nonfinite values")
            self._behavior.append(behavior_delta.detach().cpu().double())
        elif self._behavior:
            raise ValueError("behavior_delta must be supplied consistently across updates")
        self.row_count += flat.shape[0]
        self.position_count += raw.shape[0]

    def finalize(self, *, top_component_count: int = 16) -> tuple[dict[str, Any], DeltaSubspace]:
        if self.row_count < 2 or self.position_count == 0:
            raise ValueError("Cannot finalize an empty delta accumulator")
        if not 1 <= top_component_count <= self.feature_width:
            raise ValueError("top_component_count outside feature width")
        mean = self.delta_sum / self.row_count
        covariance = (
            self.delta_gram - torch.outer(self.delta_sum, self.delta_sum) / self.row_count
        ) / (self.row_count - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        components = _fix_component_signs(eigenvectors[:, order[:top_component_count]].T)
        total = float(eigenvalues.sum())
        probability = eigenvalues / eigenvalues.sum().clamp_min(torch.finfo(torch.float64).tiny)
        nonzero = probability > 0
        effective_rank = float(
            torch.exp(-(probability[nonzero] * probability[nonzero].log()).sum())
        )
        participation = float(
            eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-30)
        )
        square_share = self.square_energy_sum / self.square_energy_sum.sum().clamp_min(1e-30)
        position_rms = torch.cat(self._position_rms)
        behavior_report: dict[str, float | int] | None = None
        if self._behavior:
            behavior = torch.cat(self._behavior)
            if behavior.shape != position_rms.shape:
                raise AssertionError("Behavior/delta position alignment failed")
            centered_x = position_rms - position_rms.mean()
            centered_y = behavior - behavior.mean()
            denominator = centered_x.norm() * centered_y.norm()
            correlation = (
                float(torch.dot(centered_x, centered_y) / denominator)
                if float(denominator) > 0
                else 0.0
            )
            behavior_report = {
                "pearson_position_delta_rms": correlation,
                "position_count": int(position_rms.numel()),
            }
        cumulative = eigenvalues.cumsum(dim=0) / eigenvalues.sum().clamp_min(1e-30)
        summary = {
            "schema_version": "bt4-paired-delta-atlas-v1",
            "row_count": self.row_count,
            "position_count": self.position_count,
            "feature_width": self.feature_width,
            "procrustes_aligned": self.alignment is not None,
            "delta_mean_rms": float(mean.square().mean().sqrt()),
            "total_centered_variance": total,
            "effective_rank_entropy": effective_rank,
            "effective_rank_participation": participation,
            "top_eigenvalues": eigenvalues[:top_component_count].tolist(),
            "top_cumulative_explained_variance": cumulative[:top_component_count].tolist(),
            "square_energy_share": square_share.tolist(),
            "highest_energy_square": int(square_share.argmax()),
            "top_8_square_energy_share": float(torch.topk(square_share, 8).values.sum()),
            "position_delta_rms": {
                "mean": float(position_rms.mean()),
                "median": float(position_rms.median()),
                "maximum": float(position_rms.max()),
            },
            "behavior_coupling": behavior_report,
        }
        subspace = DeltaSubspace(
            mean=mean.float(),
            components=components.float(),
            eigenvalues=eigenvalues[:top_component_count].float(),
            total_variance=total,
            training_row_count=self.row_count,
            aligned=self.alignment is not None,
        )
        return summary, subspace


def steer_along_direction(
    activation: Tensor,
    direction: Tensor,
    coefficient: float | Tensor,
    *,
    square_mask: Tensor | None = None,
) -> Tensor:
    """Add a unit-normalized feature direction with optional square specificity."""

    if activation.ndim != 3 or direction.ndim != 1 or activation.shape[-1] != direction.numel():
        raise ValueError("Expected activation [B, square, feature] and direction [feature]")
    if not bool(torch.isfinite(activation).all()) or not bool(torch.isfinite(direction).all()):
        raise ValueError("Steering inputs must be finite")
    unit = direction.to(device=activation.device, dtype=activation.dtype)
    norm = unit.norm()
    if float(norm) <= 0.0:
        raise ValueError("Steering direction must be nonzero")
    unit = unit / norm
    coefficient_tensor = torch.as_tensor(
        coefficient,
        device=activation.device,
        dtype=activation.dtype,
    )
    if not bool(torch.isfinite(coefficient_tensor).all()):
        raise ValueError("Steering coefficient must be finite")
    while coefficient_tensor.ndim < 3:
        coefficient_tensor = coefficient_tensor.unsqueeze(-1)
    update = coefficient_tensor * unit.reshape(1, 1, -1)
    if square_mask is not None:
        if square_mask.shape != activation.shape[:2]:
            raise ValueError("square_mask must have shape [batch, square]")
        update = update * square_mask.to(
            device=activation.device, dtype=activation.dtype
        ).unsqueeze(-1)
    return activation + update


@dataclass(frozen=True)
class RidgeMediator:
    feature_mean: Tensor
    feature_scale: Tensor
    output_mean: Tensor
    weight: Tensor
    l2: float

    def predict(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("Mediator features have an invalid shape")
        normalized = (
            features.to(dtype=self.weight.dtype, device=self.weight.device)
            - self.feature_mean.to(self.weight.device)
        ) / self.feature_scale.to(self.weight.device)
        return normalized @ self.weight + self.output_mean.to(self.weight.device)


def fit_ridge_mediator(features: Tensor, outputs: Tensor, *, l2: float) -> RidgeMediator:
    """Fit a train-only multi-output ridge map from delta features to behavior."""

    if features.ndim != 2 or outputs.ndim != 2 or features.shape[0] != outputs.shape[0]:
        raise ValueError("Mediator features and outputs must be aligned matrices")
    if (
        features.shape[0] < 2
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(outputs).all())
    ):
        raise ValueError("Mediator inputs must be finite with at least two rows")
    if l2 <= 0.0 or not math.isfinite(l2):
        raise ValueError("Mediator l2 must be finite and positive")
    x = features.detach().cpu().double()
    y = outputs.detach().cpu().double()
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, 1.0)
    x = (x - mean) / scale
    output_mean = y.mean(dim=0)
    y = y - output_mean
    identity = torch.eye(x.shape[1], dtype=torch.float64)
    weight = torch.linalg.solve(x.T @ x + l2 * identity, x.T @ y)
    return RidgeMediator(
        feature_mean=mean.float(),
        feature_scale=scale.float(),
        output_mean=output_mean.float(),
        weight=weight.float(),
        l2=l2,
    )


def mediator_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Mediator target and prediction must be aligned matrices")
    target64 = target.detach().cpu().double()
    prediction64 = prediction.detach().cpu().double()
    residual = target64 - prediction64
    centered = target64 - target64.mean(dim=0, keepdim=True)
    total = centered.square().sum()
    row_denominator = target64.norm(dim=1) * prediction64.norm(dim=1)
    row_cosine = torch.where(
        row_denominator > 0,
        (target64 * prediction64).sum(dim=1) / row_denominator.clamp_min(1e-30),
        0.0,
    )
    return {
        "global_r2": 1.0 - float(residual.square().sum() / total) if float(total) > 0 else 0.0,
        "rmse": float(residual.square().mean().sqrt()),
        "mean_row_cosine": float(row_cosine.mean()),
    }


def counterfactual_delta_consistency(
    raw_base: Tensor,
    raw_counterfactual: Tensor,
    hero_base: Tensor,
    hero_counterfactual: Tensor,
) -> dict[str, Tensor]:
    """Compare raw and Hero finite-difference responses to controlled board changes."""

    if not (
        raw_base.shape == raw_counterfactual.shape == hero_base.shape == hero_counterfactual.shape
    ):
        raise ValueError("Counterfactual activation tensors must share a shape")
    if raw_base.ndim < 2:
        raise ValueError("Counterfactual activations need batch and feature dimensions")
    raw_change = (raw_counterfactual - raw_base).flatten(1)
    hero_change = (hero_counterfactual - hero_base).flatten(1)
    response_delta = hero_change - raw_change
    denominator = raw_change.norm(dim=1) * hero_change.norm(dim=1)
    cosine = torch.where(
        denominator > 0,
        (raw_change * hero_change).sum(dim=1) / denominator.clamp_min(1e-30),
        0.0,
    )
    return {
        "raw_change_rms": raw_change.square().mean(dim=1).sqrt(),
        "hero_change_rms": hero_change.square().mean(dim=1).sqrt(),
        "raw_hero_response_cosine": cosine,
        "response_delta_rms": response_delta.square().mean(dim=1).sqrt(),
    }
