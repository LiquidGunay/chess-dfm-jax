"""Dependency-light probes, controls, metrics, and group-aware uncertainty."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ProbeTask = Literal["binary", "multiclass", "regression"]
Pooling = Literal["mean", "max", "mean_max"]


def _as_float_tensor(value: np.ndarray | Tensor) -> Tensor:
    result = torch.as_tensor(value)
    if result.ndim != 2:
        raise ValueError("Probe features must have shape [example, feature]")
    result = result.detach().to(dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Probe features contain nonfinite values")
    return result


def validate_group_disjoint_splits(group_ids: Sequence[object], split_codes: Sequence[int]) -> None:
    """Reject any group represented in more than one data split."""

    groups = np.asarray(group_ids).astype(str)
    splits = np.asarray(split_codes)
    if groups.ndim != 1 or splits.shape != groups.shape or groups.size == 0:
        raise ValueError("group_ids and split_codes must be aligned non-empty vectors")
    seen: dict[str, int] = {}
    for group, split in zip(groups, splits, strict=True):
        code = int(split)
        previous = seen.setdefault(group, code)
        if previous != code:
            raise ValueError(f"Group {group!r} crosses splits {previous} and {code}")


def group_block_permutation(
    labels: np.ndarray,
    group_ids: Sequence[object],
    *,
    seed: int,
) -> np.ndarray:
    """Permute complete equal-sized group blocks while preserving label marginals."""

    values = np.asarray(labels)
    groups = np.asarray(group_ids).astype(str)
    if values.shape[0] != groups.shape[0] or values.shape[0] == 0:
        raise ValueError("labels and group_ids must have an aligned first dimension")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        indices[group].append(index)
    by_size: dict[int, list[str]] = defaultdict(list)
    for group, rows in indices.items():
        by_size[len(rows)].append(group)
    result = np.empty_like(values)
    generator = np.random.default_rng(seed)
    for size, target_groups in sorted(by_size.items()):
        source_groups = list(target_groups)
        generator.shuffle(source_groups)
        for target_group, source_group in zip(target_groups, source_groups, strict=True):
            target_rows = indices[target_group]
            source_rows = indices[source_group]
            if len(target_rows) != size or len(source_rows) != size:
                raise AssertionError("Group permutation size invariant failed")
            result[target_rows] = values[source_rows]
    return result


def pool_board_activations(activations: np.ndarray | Tensor, *, mode: Pooling) -> Tensor:
    """Convert [position, 64, feature] activations into board-level features."""

    value = torch.as_tensor(activations).detach().to(dtype=torch.float32, device="cpu")
    if value.ndim != 3 or value.shape[1] != 64:
        raise ValueError("Board activations must have shape [position, 64, feature]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Board activations contain nonfinite values")
    if mode == "mean":
        return value.mean(dim=1)
    if mode == "max":
        return value.amax(dim=1)
    if mode == "mean_max":
        return torch.cat((value.mean(dim=1), value.amax(dim=1)), dim=1)
    raise ValueError(f"Unknown pooling mode {mode!r}")


def move_pair_features(
    activations: np.ndarray | Tensor,
    origin_tokens: Sequence[int],
    destination_tokens: Sequence[int],
) -> Tensor:
    """Concatenate source and destination token states for each target move."""

    value = torch.as_tensor(activations).detach().to(dtype=torch.float32, device="cpu")
    origins = torch.as_tensor(origin_tokens, dtype=torch.long)
    destinations = torch.as_tensor(destination_tokens, dtype=torch.long)
    if value.ndim != 3 or value.shape[1] != 64:
        raise ValueError("Move features require [position, 64, feature] activations")
    if origins.shape != (value.shape[0],) or destinations.shape != origins.shape:
        raise ValueError("Move token arrays must align with positions")
    if bool(((origins < 0) | (origins >= 64) | (destinations < 0) | (destinations >= 64)).any()):
        raise ValueError("Move token index outside [0, 63]")
    rows = torch.arange(value.shape[0])
    return torch.cat((value[rows, origins], value[rows, destinations]), dim=1)


def flatten_token_probe_examples(
    activations: np.ndarray | Tensor,
    labels: np.ndarray | Tensor,
    group_ids: Sequence[object],
    *,
    valid: np.ndarray | Tensor | None = None,
    max_positions: int | None = None,
    seed: int = 20260808,
) -> tuple[Tensor, Tensor, np.ndarray]:
    """Flatten square labels without forgetting their position-level groups."""

    value = torch.as_tensor(activations).detach().to(dtype=torch.float32, device="cpu")
    target = torch.as_tensor(labels).detach().to(device="cpu")
    groups = np.asarray(group_ids).astype(str)
    if value.ndim != 3 or value.shape[1] != 64:
        raise ValueError("Token activations must have shape [position, 64, feature]")
    if target.shape != value.shape[:2] or groups.shape != (value.shape[0],):
        raise ValueError("Token labels and groups must align with activations")
    position_indices = np.arange(value.shape[0])
    if max_positions is not None:
        if type(max_positions) is not int or max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if max_positions < position_indices.size:
            generator = np.random.default_rng(seed)
            position_indices = np.sort(
                generator.choice(position_indices, size=max_positions, replace=False)
            )
    value = value[position_indices]
    target = target[position_indices]
    groups = groups[position_indices]
    if valid is None:
        mask = torch.ones(target.shape, dtype=torch.bool)
    else:
        full_valid = torch.as_tensor(valid, dtype=torch.bool)
        if full_valid.shape != torch.as_tensor(labels).shape:
            raise ValueError("valid must align with token labels")
        mask = full_valid[position_indices]
    repeated_groups = np.repeat(groups, 64)[mask.reshape(-1).numpy()]
    return (
        value.reshape(-1, value.shape[-1])[mask.reshape(-1)],
        target.reshape(-1)[mask.reshape(-1)],
        repeated_groups,
    )


class _ProbeNetwork(nn.Module):
    def __init__(self, input_width: int, output_width: int, hidden_width: int, seed: int):
        super().__init__()
        if hidden_width:
            self.layers = nn.Sequential(
                nn.Linear(input_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, output_width),
            )
        else:
            self.layers = nn.Linear(input_width, output_width)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, 0.01, generator=generator)
                    module.bias.zero_()

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


@dataclass
class FittedProbe:
    task: ProbeTask
    module: nn.Module
    feature_mean: Tensor
    feature_scale: Tensor
    classes: Tensor | None
    target_mean: float
    target_scale: float
    metadata: dict[str, Any]

    def _logits(self, features: np.ndarray | Tensor, *, batch_size: int = 16_384) -> Tensor:
        value = _as_float_tensor(features)
        if value.shape[1] != self.feature_mean.numel():
            raise ValueError("Prediction feature width differs from fitted probe")
        chunks: list[Tensor] = []
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, value.shape[0], batch_size):
                batch = (value[start : start + batch_size] - self.feature_mean) / self.feature_scale
                chunks.append(self.module(batch).cpu())
        return torch.cat(chunks)

    def predict(self, features: np.ndarray | Tensor) -> Tensor:
        logits = self._logits(features)
        if self.task == "binary":
            return torch.sigmoid(logits.squeeze(1))
        if self.task == "multiclass":
            return torch.softmax(logits, dim=1)
        return logits.squeeze(1) * self.target_scale + self.target_mean


def _sample_weights(
    labels: Tensor,
    *,
    task: ProbeTask,
    class_balance: bool,
    group_ids: Sequence[object] | None,
) -> Tensor:
    count = labels.shape[0]
    weights = torch.ones(count, dtype=torch.float32)
    if class_balance and task in {"binary", "multiclass"}:
        classes, counts = torch.unique(labels, return_counts=True)
        if classes.numel() < 2:
            raise ValueError("Classification probe requires at least two classes")
        for value, frequency in zip(classes, counts, strict=True):
            weights[labels == value] *= count / (classes.numel() * int(frequency))
    if group_ids is not None:
        groups = np.asarray(group_ids).astype(str)
        if groups.shape != (count,):
            raise ValueError("group_ids must align with probe examples")
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        group_weight = torch.from_numpy(1.0 / counts[inverse]).to(torch.float32)
        group_weight *= count / group_weight.sum()
        weights *= group_weight
    return weights / weights.mean()


def fit_probe(
    features: np.ndarray | Tensor,
    labels: np.ndarray | Tensor,
    *,
    task: ProbeTask,
    l2: float,
    hidden_width: int = 0,
    epochs: int = 100,
    learning_rate: float = 0.03,
    batch_size: int = 4096,
    seed: int = 20260808,
    class_balance: bool = True,
    group_ids: Sequence[object] | None = None,
    device: torch.device | str = "cpu",
) -> FittedProbe:
    """Fit a linear or one-hidden-layer probe with training-only normalization."""

    value = _as_float_tensor(features)
    target = torch.as_tensor(labels).detach().to(device="cpu")
    if target.ndim != 1 or target.shape[0] != value.shape[0] or target.numel() == 0:
        raise ValueError("labels must be a non-empty vector aligned with features")
    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    if type(hidden_width) is not int or hidden_width < 0:
        raise ValueError("hidden_width must be non-negative")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")

    feature_mean = value.double().mean(dim=0).float()
    feature_scale = value.double().std(dim=0, unbiased=False).float()
    feature_scale = torch.where(feature_scale > 1e-6, feature_scale, 1.0)
    standardized = (value - feature_mean) / feature_scale
    classes: Tensor | None = None
    target_mean = 0.0
    target_scale = 1.0
    if task == "binary":
        unique = torch.unique(target)
        if unique.tolist() != [0, 1]:
            raise ValueError("Binary labels must contain both 0 and 1")
        encoded_target = target.to(torch.float32)
        output_width = 1
    elif task == "multiclass":
        classes = torch.unique(target).sort().values
        if classes.numel() < 2:
            raise ValueError("Multiclass labels require at least two classes")
        encoded_target = torch.searchsorted(classes, target).to(torch.long)
        output_width = int(classes.numel())
    elif task == "regression":
        encoded_target = target.to(torch.float32)
        if not bool(torch.isfinite(encoded_target).all()):
            raise ValueError("Regression labels contain nonfinite values")
        target_mean = float(encoded_target.double().mean())
        scale_tensor = encoded_target.double().std(unbiased=False)
        target_scale = float(scale_tensor) if float(scale_tensor) > 1e-6 else 1.0
        encoded_target = (encoded_target - target_mean) / target_scale
        output_width = 1
    else:
        raise ValueError(f"Unknown probe task {task!r}")

    weights = _sample_weights(
        encoded_target,
        task=task,
        class_balance=class_balance,
        group_ids=group_ids,
    )
    active_device = torch.device(device)
    module = _ProbeNetwork(value.shape[1], output_width, hidden_width, seed).to(active_device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loss_history: list[float] = []
    for _epoch in range(epochs):
        permutation = torch.randperm(value.shape[0], generator=generator)
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, value.shape[0], batch_size):
            rows = permutation[start : start + batch_size]
            batch_x = standardized[rows].to(active_device)
            batch_y = encoded_target[rows].to(active_device)
            batch_weight = weights[rows].to(active_device)
            prediction = module(batch_x)
            if task == "binary":
                element_loss = F.binary_cross_entropy_with_logits(
                    prediction.squeeze(1),
                    batch_y,
                    reduction="none",
                )
            elif task == "multiclass":
                element_loss = F.cross_entropy(prediction, batch_y, reduction="none")
            else:
                element_loss = (prediction.squeeze(1) - batch_y).square()
            data_loss = (element_loss * batch_weight).sum() / batch_weight.sum()
            penalty = (
                0.5
                * l2
                * sum(
                    parameter.square().sum()
                    for name, parameter in module.named_parameters()
                    if name.endswith("weight")
                )
            )
            loss = data_loss + penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(data_loss.detach()) * float(batch_weight.sum())
            epoch_weight += float(batch_weight.sum())
        loss_history.append(epoch_loss / epoch_weight)
    module = module.cpu().eval()
    return FittedProbe(
        task=task,
        module=module,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        classes=classes,
        target_mean=target_mean,
        target_scale=target_scale,
        metadata={
            "schema_version": "bt4-concept-probe-v1",
            "architecture": "linear" if hidden_width == 0 else "one_hidden_layer_gelu",
            "input_width": int(value.shape[1]),
            "hidden_width": hidden_width,
            "output_width": output_width,
            "training_examples": int(value.shape[0]),
            "l2": l2,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
            "class_balance": class_balance,
            "group_balanced_loss": group_ids is not None,
            "initial_data_loss": loss_history[0],
            "final_data_loss": loss_history[-1],
            "normalization_fit": "training split only",
        },
    )


def _ece(probability: np.ndarray, correct: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean() - probability[selected].mean())
            )
    return result


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    """Binary classification metrics with tied-score-correct AUROC and AP."""

    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    if target.ndim != 1 or score.shape != target.shape or target.size == 0:
        raise ValueError("Binary labels and scores must be aligned non-empty vectors")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("Binary metric labels must contain both classes")
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("Binary probabilities must be finite values in [0, 1]")
    order = np.argsort(-score, kind="stable")
    ordered_score = score[order]
    ordered_target = target[order]
    true_positive = np.cumsum(ordered_target)
    false_positive = np.cumsum(1 - ordered_target)
    threshold_end = np.r_[np.flatnonzero(ordered_score[:-1] != ordered_score[1:]), target.size - 1]
    tp = true_positive[threshold_end].astype(np.float64)
    fp = false_positive[threshold_end].astype(np.float64)
    positives = float(target.sum())
    negatives = float(target.size - target.sum())
    recall = np.r_[0.0, tp / positives]
    false_positive_rate = np.r_[0.0, fp / negatives]
    precision = tp / np.maximum(tp + fp, 1.0)
    average_precision = float(np.sum(np.diff(recall) * precision))
    prediction = score >= 0.5
    true_positive_rate = float(prediction[target == 1].mean())
    true_negative_rate = float((~prediction[target == 0]).mean())
    confidence = np.maximum(score, 1.0 - score)
    correct = prediction == target
    clipped = np.clip(score, 1e-12, 1.0 - 1e-12)
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": 0.5 * (true_positive_rate + true_negative_rate),
        "auroc": float(np.trapezoid(recall, false_positive_rate)),
        "average_precision": average_precision,
        "brier": float(np.mean(np.square(score - target))),
        "log_loss": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
        "ece_10_equal_width_bins": _ece(confidence, correct, 10),
        "positive_rate": float(target.mean()),
        "predicted_positive_rate": float(prediction.mean()),
    }


def multiclass_metrics(labels: Sequence[int], probabilities: np.ndarray) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    if score.ndim != 2 or target.shape != (score.shape[0],) or target.size == 0:
        raise ValueError("Multiclass labels and probabilities are not aligned")
    if not np.isfinite(score).all() or not np.allclose(score.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Multiclass probabilities must be finite and row-normalized")
    if target.min() < 0 or target.max() >= score.shape[1]:
        raise ValueError("Multiclass target outside probability columns")
    prediction = score.argmax(axis=1)
    f1_values: list[float] = []
    for class_index in range(score.shape[1]):
        tp = np.count_nonzero((prediction == class_index) & (target == class_index))
        fp = np.count_nonzero((prediction == class_index) & (target != class_index))
        fn = np.count_nonzero((prediction != class_index) & (target == class_index))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    confidence = score.max(axis=1)
    correct = prediction == target
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "log_loss": float(
            -np.log(np.clip(score[np.arange(target.size), target], 1e-12, 1.0)).mean()
        ),
        "ece_10_equal_width_bins": _ece(confidence, correct, 10),
    }


def regression_metrics(labels: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.ndim != 1 or prediction.shape != target.shape or target.size < 2:
        raise ValueError("Regression labels and predictions must be aligned vectors")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression values must be finite")
    residual = prediction - target
    total = np.square(target - target.mean()).sum()
    centered_prediction = prediction - prediction.mean()
    denominator = math.sqrt(
        float(np.square(target - target.mean()).sum() * np.square(centered_prediction).sum())
    )
    return {
        "r2": 1.0 - float(np.square(residual).sum() / total) if total > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "pearson": (
            float(np.dot(target - target.mean(), centered_prediction) / denominator)
            if denominator > 0
            else 0.0
        ),
    }


def evaluate_probe(
    probe: FittedProbe,
    features: np.ndarray | Tensor,
    labels: np.ndarray | Tensor,
) -> dict[str, float]:
    target = torch.as_tensor(labels).cpu()
    prediction = probe.predict(features)
    if probe.task == "binary":
        return binary_metrics(target.numpy(), prediction.numpy())
    if probe.task == "multiclass":
        if probe.classes is None:
            raise AssertionError("Multiclass probe lost its class map")
        encoded = torch.searchsorted(probe.classes, target)
        return multiclass_metrics(encoded.numpy(), prediction.numpy())
    return regression_metrics(target.numpy(), prediction.numpy())


@dataclass(frozen=True)
class ProbeGridResult:
    best_probe: FittedProbe
    best_index: int
    selection_metric: str
    candidates: tuple[dict[str, Any], ...]


def fit_probe_grid(
    train_features: np.ndarray | Tensor,
    train_labels: np.ndarray | Tensor,
    development_features: np.ndarray | Tensor,
    development_labels: np.ndarray | Tensor,
    *,
    task: ProbeTask,
    l2_values: Sequence[float],
    hidden_widths: Sequence[int] = (0,),
    selection_metric: str | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> ProbeGridResult:
    """Select probe capacity and regularization on development data only."""

    metric = (
        selection_metric
        or {
            "binary": "auroc",
            "multiclass": "accuracy",
            "regression": "r2",
        }[task]
    )
    if not l2_values or not hidden_widths:
        raise ValueError("Probe grid must contain l2 values and hidden widths")
    candidates: list[dict[str, Any]] = []
    probes: list[FittedProbe] = []
    options = {} if fit_kwargs is None else dict(fit_kwargs)
    for hidden_width in hidden_widths:
        for l2 in l2_values:
            probe = fit_probe(
                train_features,
                train_labels,
                task=task,
                l2=float(l2),
                hidden_width=int(hidden_width),
                **options,
            )
            metrics = evaluate_probe(probe, development_features, development_labels)
            if metric not in metrics:
                raise ValueError(f"Selection metric {metric!r} is unavailable")
            probes.append(probe)
            candidates.append(
                {
                    "hidden_width": int(hidden_width),
                    "l2": float(l2),
                    "development_metrics": metrics,
                }
            )
    best_index = max(
        range(len(candidates)),
        key=lambda index: (candidates[index]["development_metrics"][metric], -index),
    )
    return ProbeGridResult(
        best_probe=probes[best_index],
        best_index=best_index,
        selection_metric=metric,
        candidates=tuple(candidates),
    )


def paired_group_bootstrap_metric(
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    hero_predictions: np.ndarray,
    group_ids: Sequence[object],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    seed: int,
    replicates: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Paired group bootstrap for an arbitrary scalar probe metric."""

    target = np.asarray(labels)
    raw = np.asarray(raw_predictions)
    hero = np.asarray(hero_predictions)
    groups = np.asarray(group_ids).astype(str)
    if target.shape[0] == 0 or raw.shape[0] != target.shape[0] or hero.shape != raw.shape:
        raise ValueError("Paired predictions and labels are not aligned")
    if groups.shape != (target.shape[0],):
        raise ValueError("group_ids must align with predictions")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration")
    unique, inverse = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inverse == index) for index in range(unique.size)]
    generator = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = generator.integers(0, unique.size, size=unique.size)
        selected = np.concatenate([rows[index] for index in sampled])
        deltas[replicate] = metric(target[selected], hero[selected]) - metric(
            target[selected],
            raw[selected],
        )
    alpha = 0.5 * (1.0 - confidence)
    return {
        "raw": float(metric(target, raw)),
        "hero": float(metric(target, hero)),
        "delta_hero_minus_raw": float(metric(target, hero) - metric(target, raw)),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
        "confidence": confidence,
        "replicates": replicates,
        "group_count": int(unique.size),
    }


def random_direction_scores(
    features: np.ndarray | Tensor,
    *,
    direction_count: int,
    seed: int,
) -> Tensor:
    """Matched normalized random linear directions for probe controls."""

    value = _as_float_tensor(features)
    if direction_count <= 0:
        raise ValueError("direction_count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(
        (value.shape[1], direction_count),
        generator=generator,
        dtype=torch.float32,
    )
    directions /= directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    centered = value - value.mean(dim=0, keepdim=True)
    return centered @ directions
