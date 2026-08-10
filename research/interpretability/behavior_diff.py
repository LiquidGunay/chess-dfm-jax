"""Legal-policy metrics for the raw/Hero encoder-head comparison lattice."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch import Tensor

from research.interpretability.accumulators import cluster_bootstrap_mean_ci
from research.interpretability.models import ARM_IDS, ArmId


PAIR_IDS: tuple[tuple[ArmId, ArmId], ...] = (
    ("RR", "HR"),
    ("RH", "HH"),
    ("RR", "RH"),
    ("HR", "HH"),
    ("RR", "HH"),
    ("HR", "RH"),
)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size <= 1:
        return 1.0
    first_rank = _rankdata_average(first)
    second_rank = _rankdata_average(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = float(np.linalg.norm(first_rank) * np.linalg.norm(second_rank))
    if denominator == 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.dot(first_rank, second_rank) / denominator)


def _topk_set(probabilities: np.ndarray, count: int) -> set[int]:
    k = min(count, probabilities.size)
    return set(np.argsort(-probabilities, kind="stable")[:k].tolist())


def _jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _arm_row(
    all_logits: np.ndarray,
    legal_indices: np.ndarray,
    target: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    legal_logits = all_logits[legal_indices].astype(np.float64, copy=False)
    legal_log_normalizer = _logsumexp(legal_logits)
    legal_log_probabilities = legal_logits - legal_log_normalizer
    probabilities = np.exp(legal_log_probabilities)
    target_locations = np.flatnonzero(legal_indices == target)
    if target_locations.size != 1:
        raise ValueError(f"Target action {target} does not appear exactly once in legal set")
    target_location = int(target_locations[0])
    descending = np.argsort(-legal_logits, kind="stable")
    target_rank = int(np.flatnonzero(descending == target_location)[0]) + 1
    top1_location = int(descending[0])
    target_probability = float(probabilities[target_location])
    confidence = float(probabilities[top1_location])
    one_hot = np.zeros_like(probabilities)
    one_hot[target_location] = 1.0
    legal_mass = math.exp(legal_log_normalizer - _logsumexp(all_logits))
    result: dict[str, float | int] = {
        "target_nll": -float(legal_log_probabilities[target_location]),
        "target_probability": target_probability,
        "target_rank": target_rank,
        "top1_action": int(legal_indices[top1_location]),
        "top1_accuracy": int(top1_location == target_location),
        "top3_accuracy": int(target_location in _topk_set(probabilities, 3)),
        "top5_accuracy": int(target_location in _topk_set(probabilities, 5)),
        "confidence": confidence,
        "entropy": -float(np.dot(probabilities, legal_log_probabilities)),
        "multiclass_brier": float(np.square(probabilities - one_hot).sum()),
        "legal_mass_before_masking": legal_mass,
    }
    return result, probabilities, legal_log_probabilities


def evaluate_lattice_batch(
    logits: Mapping[ArmId, Tensor | np.ndarray],
    legal_idx: Tensor | np.ndarray,
    legal_count: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
) -> list[dict[str, Any]]:
    """Return one fully paired behavior record for every input position."""

    if tuple(logits) != ARM_IDS or set(logits) != set(ARM_IDS):
        raise ValueError(f"Lattice logits must be ordered as {ARM_IDS}")

    def to_numpy(value: Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, Tensor):
            return value.detach().float().cpu().numpy()
        return np.asarray(value)

    arrays = {arm: to_numpy(logits[arm]).astype(np.float64) for arm in ARM_IDS}
    legal_array = to_numpy(legal_idx).astype(np.int64)
    count_array = to_numpy(legal_count).astype(np.int64)
    target_array = to_numpy(targets).astype(np.int64)
    batch = next(iter(arrays.values())).shape[0]
    if any(value.ndim != 2 or value.shape[0] != batch for value in arrays.values()):
        raise ValueError("Every arm must have rank-2 logits with a shared batch")
    vocab_size = next(iter(arrays.values())).shape[1]
    if any(value.shape != (batch, vocab_size) for value in arrays.values()):
        raise ValueError("Every arm must have equal logits shape")
    if legal_array.ndim != 2 or legal_array.shape[0] != batch:
        raise ValueError("legal_idx must have shape [B, Lmax]")
    if count_array.shape != (batch,) or target_array.shape != (batch,):
        raise ValueError("legal_count and targets must have shape [B]")

    records: list[dict[str, Any]] = []
    for row in range(batch):
        count = int(count_array[row])
        if not 1 <= count <= legal_array.shape[1]:
            raise ValueError(f"Invalid legal count {count} at row {row}")
        legal = legal_array[row, :count]
        if len(set(legal.tolist())) != count or np.any((legal < 0) | (legal >= vocab_size)):
            raise ValueError(f"Invalid legal action set at row {row}")
        arm_metrics: dict[str, Any] = {}
        probabilities: dict[ArmId, np.ndarray] = {}
        log_probabilities: dict[ArmId, np.ndarray] = {}
        for arm in ARM_IDS:
            if not np.isfinite(arrays[arm][row]).all():
                raise ValueError(f"Nonfinite {arm} logits at row {row}")
            metrics, probability, log_probability = _arm_row(
                arrays[arm][row],
                legal,
                int(target_array[row]),
            )
            arm_metrics[arm] = metrics
            probabilities[arm] = probability
            log_probabilities[arm] = log_probability

        pair_metrics: dict[str, dict[str, float | int]] = {}
        for first, second in PAIR_IDS:
            p = probabilities[first]
            q = probabilities[second]
            log_p = log_probabilities[first]
            log_q = log_probabilities[second]
            midpoint = 0.5 * (p + q)
            log_midpoint = np.log(midpoint)
            pair_metrics[f"{first}__{second}"] = {
                "kl_first_to_second": float(np.dot(p, log_p - log_q)),
                "kl_second_to_first": float(np.dot(q, log_q - log_p)),
                "js_divergence": 0.5
                * float(np.dot(p, log_p - log_midpoint) + np.dot(q, log_q - log_midpoint)),
                "total_variation": 0.5 * float(np.abs(p - q).sum()),
                "top1_agreement": int(np.argmax(p) == np.argmax(q)),
                "top3_jaccard": _jaccard(_topk_set(p, 3), _topk_set(q, 3)),
                "top5_jaccard": _jaccard(_topk_set(p, 5), _topk_set(q, 5)),
                "legal_rank_spearman": _spearman(p, q),
            }

        probability_interaction = (
            probabilities["HH"] - probabilities["HR"] - probabilities["RH"] + probabilities["RR"]
        )
        log_probability_interaction = (
            log_probabilities["HH"]
            - log_probabilities["HR"]
            - log_probabilities["RH"]
            + log_probabilities["RR"]
        )
        records.append(
            {
                "legal_count": count,
                "target_action": int(target_array[row]),
                "arms": arm_metrics,
                "pairs": pair_metrics,
                "decomposition": {
                    "probability_interaction_rms": float(
                        np.sqrt(np.mean(np.square(probability_interaction)))
                    ),
                    "probability_interaction_l1": float(np.abs(probability_interaction).sum()),
                    "log_probability_interaction_rms": float(
                        np.sqrt(np.mean(np.square(log_probability_interaction)))
                    ),
                    "semantics": "HH - HR - RH + RR over the legal-action vector",
                },
            }
        )
    return records


def _bootstrap_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (base_seed ^ int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _ece(records: Sequence[dict[str, Any]], arm: ArmId, bins: int = 10) -> float:
    confidence = np.asarray([record["arms"][arm]["confidence"] for record in records])
    correct = np.asarray([record["arms"][arm]["top1_accuracy"] for record in records])
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean() - confidence[selected].mean())
            )
    return ece


def summarize_behavior_records(
    records: Sequence[dict[str, Any]],
    game_ids: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Aggregate paired position records with game-cluster intervals."""

    if not records or len(records) != len(game_ids):
        raise ValueError("Behavior records and game IDs must be non-empty and aligned")

    def summarize(path: tuple[str, ...], label: str) -> dict[str, Any]:
        values = np.asarray(
            [
                float(__import__("functools").reduce(dict.__getitem__, path, record))
                for record in records
            ]
        )
        return cluster_bootstrap_mean_ci(
            values,
            game_ids,
            seed=_bootstrap_seed(bootstrap_seed, label),
            replicates=bootstrap_replicates,
        )

    arm_metrics = (
        "target_nll",
        "target_rank",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "entropy",
        "multiclass_brier",
        "legal_mass_before_masking",
    )
    pair_metrics = (
        "kl_first_to_second",
        "kl_second_to_first",
        "js_divergence",
        "total_variation",
        "top1_agreement",
        "top3_jaccard",
        "top5_jaccard",
        "legal_rank_spearman",
    )
    arms = {
        arm: {
            **{
                metric: summarize(("arms", arm, metric), f"arms.{arm}.{metric}")
                for metric in arm_metrics
            },
            "ece_10_equal_width_bins": _ece(records, arm),
        }
        for arm in ARM_IDS
    }
    pairs = {
        f"{first}__{second}": {
            metric: summarize(
                ("pairs", f"{first}__{second}", metric),
                f"pairs.{first}__{second}.{metric}",
            )
            for metric in pair_metrics
        }
        for first, second in PAIR_IDS
    }
    decomposition = {
        metric: summarize(("decomposition", metric), f"decomposition.{metric}")
        for metric in (
            "probability_interaction_rms",
            "probability_interaction_l1",
            "log_probability_interaction_rms",
        )
    }
    return {
        "position_count": len(records),
        "game_count": int(np.unique(game_ids.astype(str)).size),
        "arm_semantics": "first letter encoder; second letter native policy head",
        "pair_semantics": {
            "RR__HR": "encoder effect under the raw head",
            "RH__HH": "encoder effect under the Hero head",
            "RR__RH": "head effect on the raw encoder",
            "HR__HH": "head effect on the Hero encoder",
            "RR__HH": "total raw-to-Hero effect",
            "HR__RH": "mixed-arm contrast",
        },
        "arms": arms,
        "pairs": pairs,
        "decomposition": decomposition,
    }
