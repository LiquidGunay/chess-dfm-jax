"""Small dependency-light estimators used in the interactive lessons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class IntervalEstimate:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    group_count: int
    weighting: str

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0


def stable_softmax(logits: Array, axis: int = -1) -> Array:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def masked_softmax(logits: Array, legal: Array, axis: int = -1) -> Array:
    values = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(legal, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {values.shape} != {mask.shape}")
    if np.any(~np.any(mask, axis=axis)):
        raise ValueError("Every row must contain at least one legal action")
    return stable_softmax(np.where(mask, values, -np.inf), axis=axis)


def total_variation(first: Array, second: Array, axis: int = -1) -> Array:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Probability arrays must have identical shapes")
    return 0.5 * np.sum(np.abs(first - second), axis=axis)


def js_divergence(first: Array, second: Array, axis: int = -1) -> Array:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Probability arrays must have identical shapes")
    midpoint = 0.5 * (first + second)

    def kl(left: Array, right: Array) -> Array:
        positive = left > 0.0
        safe_left = np.where(positive, left, 1.0)
        terms = np.where(
            positive,
            left * np.log(safe_left / np.clip(right, 1e-300, None)),
            0.0,
        )
        return np.sum(terms, axis=axis)

    return 0.5 * (kl(first, midpoint) + kl(second, midpoint))


def paired_cluster_bootstrap(
    first: Array,
    second: Array,
    groups: Iterable[object],
    *,
    weighting: Literal["observation", "equal_group"],
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> IntervalEstimate:
    """Bootstrap a paired mean while resampling whole groups.

    ``weighting="observation"`` targets the mean over rows, so larger groups
    receive more weight. ``weighting="equal_group"`` first averages within
    each group and then gives every group equal weight. Requiring this choice
    at every call site prevents a clustered standard error from silently being
    mistaken for an equal-group estimand.
    """

    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    group_values = np.asarray(list(groups), dtype=object).reshape(-1)
    if first.shape != second.shape or first.shape != group_values.shape:
        raise ValueError("Paired values and groups must have the same length")
    if first.size == 0 or not np.all(np.isfinite(first - second)):
        raise ValueError("Paired values must be non-empty and finite")
    if not 0.0 < confidence < 1.0 or resamples < 1:
        raise ValueError("Invalid confidence or resample count")

    unique = list(dict.fromkeys(group_values.tolist()))
    index_by_group = [np.flatnonzero(group_values == group) for group in unique]
    delta = first - second
    group_means = np.asarray(
        [delta[indices].mean() for indices in index_by_group], dtype=np.float64
    )
    if len(unique) < 2:
        raise ValueError(
            "A population interval requires at least two independent groups"
        )
    if weighting == "observation":
        estimate = float(delta.mean())
    elif weighting == "equal_group":
        estimate = float(group_means.mean())
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}")
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        chosen = rng.integers(0, len(unique), size=len(unique))
        if weighting == "equal_group":
            sampled[draw] = group_means[chosen].mean()
        else:
            indices = np.concatenate([index_by_group[index] for index in chosen])
            sampled[draw] = delta[indices].mean()
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sampled, [alpha, 1.0 - alpha])
    return IntervalEstimate(
        estimate,
        float(lower),
        float(upper),
        confidence,
        resamples,
        len(unique),
        weighting,
    )


def _center_rows(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Representations must have shape [observations, features]")
    return values - values.mean(axis=0, keepdims=True)


def linear_cka(first: Array, second: Array) -> float:
    first = _center_rows(first)
    second = _center_rows(second)
    if first.shape[0] != second.shape[0]:
        raise ValueError("Representations must have paired observations")
    cross = first.T @ second
    numerator = float(np.sum(cross * cross))
    denominator = float(
        np.linalg.norm(first.T @ first, ord="fro")
        * np.linalg.norm(second.T @ second, ord="fro")
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def procrustes_similarity(first: Array, second: Array) -> float:
    first = _center_rows(first)
    second = _center_rows(second)
    if first.shape[0] != second.shape[0]:
        raise ValueError("Representations must have paired observations")
    singular_values = np.linalg.svd(first.T @ second, compute_uv=False)
    denominator = np.linalg.norm(first, ord="fro") * np.linalg.norm(second, ord="fro")
    return 0.0 if denominator == 0.0 else float(singular_values.sum() / denominator)


def mean_row_cosine(first: Array, second: Array) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Representations must have identical rank-2 shapes")
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(cosine.mean())


def symmetric_relative_l2(first: Array, second: Array) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    numerator = np.linalg.norm(first - second)
    denominator = 0.5 * (np.linalg.norm(first) + np.linalg.norm(second))
    return 0.0 if denominator == 0.0 else float(numerator / denominator)


def fit_ridge_binary(features: Array, labels: Array, *, ridge: float) -> Array:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Features/labels have incompatible shapes")
    if ridge < 0.0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Ridge must be nonnegative and labels binary")
    design = np.column_stack([features, np.ones(features.shape[0])])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    target = labels.astype(np.float64) * 2.0 - 1.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def ridge_binary_scores(features: Array, weights: Array) -> Array:
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or weights.size != features.shape[1] + 1:
        raise ValueError("Feature/weight ABI mismatch")
    return features @ weights[:-1] + weights[-1]


def binary_accuracy(scores: Array, labels: Array) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("Score/label shape mismatch")
    return float(np.mean((scores >= 0.0) == labels))


def support_jaccard(first: Array, second: Array) -> float:
    first = np.asarray(first, dtype=bool).reshape(-1)
    second = np.asarray(second, dtype=bool).reshape(-1)
    if first.shape != second.shape:
        raise ValueError("Support masks must have identical shapes")
    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def first_illegal_mask(legal_by_horizon: Array) -> tuple[Array, Array]:
    """Return valid-prefix mask and zero-based first-illegal index (-1 if none)."""

    legal = np.asarray(legal_by_horizon, dtype=bool)
    if legal.ndim != 2:
        raise ValueError("Legality must have shape [examples, horizons]")
    valid_prefix = np.logical_and.accumulate(legal, axis=1)
    has_illegal = np.any(~legal, axis=1)
    first_illegal = np.where(has_illegal, np.argmax(~legal, axis=1), -1)
    return valid_prefix, first_illegal.astype(np.int64)


def benjamini_hochberg(p_values: Array) -> Array:
    values = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result
