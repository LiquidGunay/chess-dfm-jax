"""Mergeable, bounded-memory statistics for paired representation studies."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


_MAX_BOOTSTRAP_SAMPLE_INDICES = 1_000_000


def _as_rows(value: np.ndarray, width: int | None = None) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"Expected a rank-2 array, got shape {rows.shape}")
    if width is not None and rows.shape[1] != width:
        raise ValueError(f"Expected width {width}, got {rows.shape[1]}")
    if not np.isfinite(rows).all():
        raise ValueError("Statistics input contains nonfinite values")
    return rows


@dataclass
class StreamingPairedMoments:
    """Exact elementwise moments for two aligned tensors.

    Only scalar sufficient statistics are retained, so update tensors may be
    arbitrarily large and are released by the caller after every batch.
    """

    count: int = 0
    raw_sum: float = 0.0
    hero_sum: float = 0.0
    raw_squared_sum: float = 0.0
    hero_squared_sum: float = 0.0
    cross_sum: float = 0.0
    delta_squared_sum: float = 0.0
    absolute_delta_sum: float = 0.0
    unchanged_count: int = 0

    def update(self, raw: np.ndarray, hero: np.ndarray) -> None:
        raw_array = np.asarray(raw, dtype=np.float64)
        hero_array = np.asarray(hero, dtype=np.float64)
        if raw_array.shape != hero_array.shape:
            raise ValueError(f"Paired shapes differ: {raw_array.shape} != {hero_array.shape}")
        if not np.isfinite(raw_array).all() or not np.isfinite(hero_array).all():
            raise ValueError("Paired moments input contains nonfinite values")
        delta = hero_array - raw_array
        self.count += raw_array.size
        self.raw_sum += float(raw_array.sum(dtype=np.float64))
        self.hero_sum += float(hero_array.sum(dtype=np.float64))
        self.raw_squared_sum += float(np.square(raw_array).sum(dtype=np.float64))
        self.hero_squared_sum += float(np.square(hero_array).sum(dtype=np.float64))
        self.cross_sum += float((raw_array * hero_array).sum(dtype=np.float64))
        self.delta_squared_sum += float(np.square(delta).sum(dtype=np.float64))
        self.absolute_delta_sum += float(np.abs(delta).sum(dtype=np.float64))
        self.unchanged_count += int(np.count_nonzero(raw_array == hero_array))

    def merge(self, other: StreamingPairedMoments) -> None:
        for name in (
            "count",
            "raw_sum",
            "hero_sum",
            "raw_squared_sum",
            "hero_squared_sum",
            "cross_sum",
            "delta_squared_sum",
            "absolute_delta_sum",
            "unchanged_count",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def finalize(self) -> dict[str, float | int]:
        if self.count <= 0:
            raise ValueError("Cannot finalize empty paired moments")
        count = float(self.count)
        raw_mean = self.raw_sum / count
        hero_mean = self.hero_sum / count
        raw_variance = max(0.0, self.raw_squared_sum / count - raw_mean**2)
        hero_variance = max(0.0, self.hero_squared_sum / count - hero_mean**2)
        centered_cross = self.cross_sum - self.raw_sum * self.hero_sum / count
        centered_raw = max(
            0.0,
            self.raw_squared_sum - self.raw_sum * self.raw_sum / count,
        )
        centered_hero = max(
            0.0,
            self.hero_squared_sum - self.hero_sum * self.hero_sum / count,
        )
        raw_norm = math.sqrt(max(0.0, self.raw_squared_sum))
        hero_norm = math.sqrt(max(0.0, self.hero_squared_sum))
        delta_norm = math.sqrt(max(0.0, self.delta_squared_sum))
        cosine_denominator = raw_norm * hero_norm
        correlation_denominator = math.sqrt(centered_raw * centered_hero)
        symmetric_scale = math.sqrt(max(0.0, 0.5 * (self.raw_squared_sum + self.hero_squared_sum)))
        return {
            "element_count": self.count,
            "raw_mean": raw_mean,
            "hero_mean": hero_mean,
            "raw_variance": raw_variance,
            "hero_variance": hero_variance,
            "raw_rms": math.sqrt(self.raw_squared_sum / count),
            "hero_rms": math.sqrt(self.hero_squared_sum / count),
            "delta_mean": (self.hero_sum - self.raw_sum) / count,
            "delta_rms": math.sqrt(self.delta_squared_sum / count),
            "delta_mae": self.absolute_delta_sum / count,
            "relative_l2_to_raw": delta_norm / raw_norm if raw_norm else 0.0,
            "symmetric_relative_l2": (delta_norm / symmetric_scale if symmetric_scale else 0.0),
            "cosine_similarity": (
                self.cross_sum / cosine_denominator if cosine_denominator else 0.0
            ),
            "centered_correlation": (
                centered_cross / correlation_denominator if correlation_denominator else 0.0
            ),
            "unchanged_fraction": self.unchanged_count / count,
        }


def paired_position_metrics(raw: np.ndarray, hero: np.ndarray) -> dict[str, np.ndarray]:
    """Return one exact paired effect row per leading (position) dimension."""

    raw_array = np.asarray(raw, dtype=np.float64)
    hero_array = np.asarray(hero, dtype=np.float64)
    if raw_array.shape != hero_array.shape or raw_array.ndim < 2:
        raise ValueError("Position metrics require equal arrays of rank at least two")
    if not np.isfinite(raw_array).all() or not np.isfinite(hero_array).all():
        raise ValueError("Position metrics input contains nonfinite values")
    raw_rows = raw_array.reshape(raw_array.shape[0], -1)
    hero_rows = hero_array.reshape(hero_array.shape[0], -1)
    delta = hero_rows - raw_rows
    raw_squared = np.einsum("ij,ij->i", raw_rows, raw_rows)
    hero_squared = np.einsum("ij,ij->i", hero_rows, hero_rows)
    delta_squared = np.einsum("ij,ij->i", delta, delta)
    cross = np.einsum("ij,ij->i", raw_rows, hero_rows)
    cosine_denominator = np.sqrt(raw_squared * hero_squared)
    symmetric_denominator = np.sqrt(0.5 * (raw_squared + hero_squared))
    return {
        "relative_l2_to_raw": np.divide(
            np.sqrt(delta_squared),
            np.sqrt(raw_squared),
            out=np.zeros_like(delta_squared),
            where=raw_squared > 0.0,
        ),
        "symmetric_relative_l2": np.divide(
            np.sqrt(delta_squared),
            symmetric_denominator,
            out=np.zeros_like(delta_squared),
            where=symmetric_denominator > 0.0,
        ),
        "cosine_similarity": np.divide(
            cross,
            cosine_denominator,
            out=np.zeros_like(cross),
            where=cosine_denominator > 0.0,
        ),
        "delta_rms": np.sqrt(delta_squared / raw_rows.shape[1]),
    }


def rademacher_projection(
    input_width: int,
    output_width: int,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Create a frozen JL projection and its byte-level identity."""

    if input_width <= 0 or output_width <= 0:
        raise ValueError("Projection widths must be positive")
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(input_width, output_width), dtype=np.int8)
    projection = (signs.astype(np.float32) * 2.0 - 1.0) / math.sqrt(output_width)
    projection = np.ascontiguousarray(projection)
    return projection, hashlib.sha256(projection.tobytes()).hexdigest()


def _effective_rank(scatter: np.ndarray) -> dict[str, float | int]:
    symmetric = 0.5 * (scatter + scatter.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    positive = eigenvalues[eigenvalues > max(1e-12, total * 1e-12)]
    if total <= 0.0 or positive.size == 0:
        return {
            "numerical_rank": 0,
            "participation_rank": 0.0,
            "entropy_effective_rank": 0.0,
        }
    probabilities = positive / total
    return {
        "numerical_rank": int(positive.size),
        "participation_rank": float(1.0 / np.square(probabilities).sum()),
        "entropy_effective_rank": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
    }


def _pca_basis(
    scatter: np.ndarray,
    *,
    variance_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (scatter + scatter.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return np.zeros((scatter.shape[0], 0)), np.zeros((0,))
    positive = eigenvalues > max(1e-12, total * 1e-12)
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    cumulative = np.cumsum(eigenvalues) / total
    rank = min(
        eigenvalues.size,
        int(np.searchsorted(cumulative, variance_fraction, side="left")) + 1,
    )
    return eigenvectors[:, :rank], eigenvalues[:rank]


def representation_similarity(
    raw_scatter: np.ndarray,
    hero_scatter: np.ndarray,
    cross_scatter: np.ndarray,
    *,
    variance_fraction: float = 0.99,
) -> dict[str, Any]:
    """Similarity metrics from centered projected covariance statistics."""

    raw_scatter = np.asarray(raw_scatter, dtype=np.float64)
    hero_scatter = np.asarray(hero_scatter, dtype=np.float64)
    cross_scatter = np.asarray(cross_scatter, dtype=np.float64)
    if (
        raw_scatter.ndim != 2
        or raw_scatter.shape[0] != raw_scatter.shape[1]
        or hero_scatter.shape != raw_scatter.shape
        or cross_scatter.shape != raw_scatter.shape
    ):
        raise ValueError("Representation scatters must be equal square matrices")
    raw_frobenius = float(np.linalg.norm(raw_scatter, ord="fro"))
    hero_frobenius = float(np.linalg.norm(hero_scatter, ord="fro"))
    denominator = raw_frobenius * hero_frobenius
    linear_cka = float(np.square(cross_scatter).sum()) / denominator if denominator else 0.0

    raw_basis, raw_values = _pca_basis(
        raw_scatter,
        variance_fraction=variance_fraction,
    )
    hero_basis, hero_values = _pca_basis(
        hero_scatter,
        variance_fraction=variance_fraction,
    )
    cca_count = min(raw_values.size, hero_values.size)
    if cca_count:
        raw_whitener = raw_basis / np.sqrt(raw_values)[None, :]
        hero_whitener = hero_basis / np.sqrt(hero_values)[None, :]
        whitened_cross = raw_whitener.T @ cross_scatter @ hero_whitener
        cca_left, cca_values, _cca_right = np.linalg.svd(
            whitened_cross,
            full_matrices=False,
        )
        cca_values = np.clip(cca_values, 0.0, 1.0)
        raw_coefficients = raw_whitener @ cca_left
        pwcca_weights = np.abs(raw_scatter @ raw_coefficients).sum(axis=0)
        weight_total = float(pwcca_weights.sum())
        pwcca = float(np.dot(pwcca_weights, cca_values) / weight_total) if weight_total else 0.0
        subspace_cosines = np.linalg.svd(
            raw_basis.T @ hero_basis,
            compute_uv=False,
        )
        principal_angles = np.degrees(np.arccos(np.clip(subspace_cosines, 0.0, 1.0)))
    else:
        cca_values = np.zeros((0,))
        pwcca = 0.0
        principal_angles = np.zeros((0,))

    procrustes_squared = max(
        0.0,
        float(np.trace(raw_scatter) + np.trace(hero_scatter))
        - 2.0 * float(np.linalg.svd(cross_scatter, compute_uv=False).sum()),
    )
    hero_energy = max(0.0, float(np.trace(hero_scatter)))
    delta_scatter = raw_scatter + hero_scatter - cross_scatter - cross_scatter.T
    return {
        "linear_cka": linear_cka,
        "svcca_mean": float(cca_values.mean()) if cca_values.size else 0.0,
        "pwcca": pwcca,
        "cca_correlations": cca_values.tolist(),
        "raw_pca_rank": int(raw_values.size),
        "hero_pca_rank": int(hero_values.size),
        "pca_variance_fraction": variance_fraction,
        "orthogonal_procrustes_relative_residual": (
            math.sqrt(procrustes_squared / hero_energy) if hero_energy else 0.0
        ),
        "principal_angles_degrees": principal_angles[:10].tolist(),
        "raw_effective_rank": _effective_rank(raw_scatter),
        "hero_effective_rank": _effective_rank(hero_scatter),
        "delta_effective_rank": _effective_rank(delta_scatter),
    }


class ProjectedPairAccumulator:
    """Mergeable FP64 covariance sufficient statistics in a frozen sketch."""

    def __init__(self, width: int):
        if width <= 0:
            raise ValueError("Sketch width must be positive")
        self.width = width
        self.count = 0
        self.raw_sum = np.zeros(width, dtype=np.float64)
        self.hero_sum = np.zeros(width, dtype=np.float64)
        self.raw_gram = np.zeros((width, width), dtype=np.float64)
        self.hero_gram = np.zeros((width, width), dtype=np.float64)
        self.cross_gram = np.zeros((width, width), dtype=np.float64)

    def update(self, raw: np.ndarray, hero: np.ndarray) -> None:
        raw_rows = _as_rows(raw, self.width)
        hero_rows = _as_rows(hero, self.width)
        if raw_rows.shape != hero_rows.shape:
            raise ValueError("Projected pair row counts differ")
        self.count += raw_rows.shape[0]
        self.raw_sum += raw_rows.sum(axis=0, dtype=np.float64)
        self.hero_sum += hero_rows.sum(axis=0, dtype=np.float64)
        self.raw_gram += raw_rows.T @ raw_rows
        self.hero_gram += hero_rows.T @ hero_rows
        self.cross_gram += raw_rows.T @ hero_rows

    def merge(self, other: ProjectedPairAccumulator) -> None:
        if self.width != other.width:
            raise ValueError("Cannot merge different sketch widths")
        self.count += other.count
        self.raw_sum += other.raw_sum
        self.hero_sum += other.hero_sum
        self.raw_gram += other.raw_gram
        self.hero_gram += other.hero_gram
        self.cross_gram += other.cross_gram

    def centered_scatters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise ValueError("Cannot center an empty projected accumulator")
        scale = 1.0 / self.count
        return (
            self.raw_gram - np.outer(self.raw_sum, self.raw_sum) * scale,
            self.hero_gram - np.outer(self.hero_sum, self.hero_sum) * scale,
            self.cross_gram - np.outer(self.raw_sum, self.hero_sum) * scale,
        )

    def finalize(self, *, variance_fraction: float = 0.99) -> dict[str, Any]:
        raw, hero, cross = self.centered_scatters()
        return {
            "observation_count": self.count,
            **representation_similarity(
                raw,
                hero,
                cross,
                variance_fraction=variance_fraction,
            ),
        }


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Paired cluster bootstrap CI, keeping all rows from sampled games."""

    observations = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups)
    if observations.ndim != 1 or group_array.shape != observations.shape:
        raise ValueError("Bootstrap values/groups must be aligned vectors")
    if observations.size == 0 or not np.isfinite(observations).all():
        raise ValueError("Bootstrap values must be non-empty and finite")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration")
    unique_groups, inverse = np.unique(group_array.astype(str), return_inverse=True)
    group_sums = np.bincount(inverse, weights=observations)
    group_counts = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(seed)
    replicate_means = np.empty(replicates, dtype=np.float64)
    replicates_per_chunk = max(
        1,
        min(
            replicates,
            _MAX_BOOTSTRAP_SAMPLE_INDICES // unique_groups.size,
        ),
    )
    for start in range(0, replicates, replicates_per_chunk):
        end = min(replicates, start + replicates_per_chunk)
        sampled = rng.integers(
            0,
            unique_groups.size,
            size=(end - start, unique_groups.size),
        )
        replicate_means[start:end] = group_sums[sampled].sum(axis=1) / group_counts[sampled].sum(
            axis=1
        )
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicate_means, [tail, 1.0 - tail])
    return {
        "estimate": float(observations.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "replicates": replicates,
        "cluster_count": int(unique_groups.size),
        "observation_count": int(observations.size),
        "seed": seed,
        "method": "paired percentile cluster bootstrap over game_id",
    }
