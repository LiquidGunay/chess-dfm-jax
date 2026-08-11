"""Fail-closed semantic adapter for compact per-position LoRSA summaries.

The LoRSA transfer artifact intentionally does not retain dense
``[position, square, feature]`` activations.  Its compact example rows contain
only the strongest features in each position and one peak square per reported
feature.  Consequently this module never feeds those rows into the streaming
token-level semantic accumulator and never labels them as token semantics.

What *is* identifiable is narrower: whether a feature that is among a
position's reported strongest features tends to peak on a particular square
label.  Feature/concept pairs are selected using Raw positions in a fit split,
then evaluated without reselection on group-disjoint Raw and Hero positions.
The result is descriptive peak-location association, not feature causality or
semantic equivalence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COMPACT_CAPABILITY_SCHEMA = "bt4-lorsa-compact-feature-capability-v1"
COMPACT_PEAK_REPORT_SCHEMA = "bt4-lorsa-compact-peak-semantic-heldout-v1"


@dataclass(frozen=True)
class CompactPeakConceptSpec:
    """One categorical square label that can annotate a retained peak token."""

    name: str
    array: str
    class_names: tuple[str, ...]

    def validate(self) -> None:
        if not self.name or self.name.startswith("_"):
            raise ValueError("Compact concepts require public non-empty names")
        if not self.array:
            raise ValueError("Compact concepts require a corpus array")
        if len(self.class_names) < 2 or any(not value for value in self.class_names):
            raise ValueError("Compact concepts require at least two named classes")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Compact concept class names must be unique")


@dataclass
class _PeakStatistics:
    feature_support: np.ndarray
    class_tokens: dict[str, np.ndarray]
    feature_class_peaks: dict[str, np.ndarray]
    reported_occurrences: int
    positive_occurrences: int


def load_compact_examples(path: Path) -> list[dict[str, Any]]:
    """Load a compact JSONL example ledger without accepting non-object rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Compact example line {line_number} is not an object")
            rows.append(value)
    return rows


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _validate_feature_summary(
    summary: Any,
    *,
    n_features: int,
    location: str,
) -> tuple[int, int, set[int]]:
    if summary is None:
        return 0, 0, set()
    if not isinstance(summary, Mapping):
        raise ValueError(f"{location} feature summary must be an object or null")
    reported = summary.get("reported_feature_count")
    if type(reported) is not int or reported < 0 or reported > n_features:
        raise ValueError(f"{location} reported feature count is invalid")
    mean_l0 = _finite_nonnegative(summary.get("mean_token_l0"), label=f"{location} mean L0")
    if mean_l0 > n_features:
        raise ValueError(f"{location} mean L0 exceeds feature width")
    rows = summary.get("top_features")
    if not isinstance(rows, list) or len(rows) != reported:
        raise ValueError(f"{location} top-feature rows do not match reported count")

    seen: set[int] = set()
    positive = 0
    previous_strength = math.inf
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{location} top feature {index} is not an object")
        feature = row.get("feature")
        if type(feature) is not int or not 0 <= feature < n_features:
            raise ValueError(f"{location} feature index is out of bounds")
        if feature in seen:
            raise ValueError(f"{location} repeats a feature index")
        seen.add(feature)
        strength = _finite_nonnegative(
            row.get("summed_activation"), label=f"{location} summed activation"
        )
        maximum = _finite_nonnegative(
            row.get("maximum_activation"), label=f"{location} maximum activation"
        )
        active_tokens = row.get("active_token_count")
        peak_token = row.get("peak_token")
        if type(active_tokens) is not int or not 0 <= active_tokens <= 64:
            raise ValueError(f"{location} active token count is invalid")
        if type(peak_token) is not int or not 0 <= peak_token < 64:
            raise ValueError(f"{location} peak token is invalid")
        if strength + 1e-12 < maximum:
            raise ValueError(f"{location} maximum activation exceeds summed activation")
        if strength > previous_strength + 1e-12:
            raise ValueError(f"{location} top features are not strength-sorted")
        if (active_tokens == 0) != (maximum == 0.0):
            raise ValueError(f"{location} zero-support and maximum activation disagree")
        previous_strength = strength
        positive += int(active_tokens > 0 and maximum > 0)
    return reported, positive, seen


def validate_compact_feature_capability(
    examples: Sequence[Mapping[str, Any]],
    *,
    n_features: int,
) -> dict[str, Any]:
    """Validate compact rows and declare exactly which semantic claims survive."""

    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    position_ids: set[str] = set()
    group_ids: set[str] = set()
    model_counts = {"raw": 0, "hero": 0}
    positive_counts = {"raw": 0, "hero": 0}
    observed_features = {"raw": set(), "hero": set()}
    missing_models: set[str] = set()
    null_summary_counts = {"raw": 0, "hero": 0}

    for position_index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("Compact examples must be objects")
        position_id = str(example.get("position_id", ""))
        group_id = str(example.get("group_id", ""))
        if not position_id or not group_id:
            raise ValueError("Compact examples require position_id and group_id")
        if position_id in position_ids:
            raise ValueError("Compact position IDs must be unique")
        position_ids.add(position_id)
        group_ids.add(group_id)
        for model in ("raw", "hero"):
            model_row = example.get(model)
            if not isinstance(model_row, Mapping) or "features" not in model_row:
                missing_models.add(model)
                continue
            if model_row["features"] is None:
                null_summary_counts[model] += 1
            reported, positive, seen = _validate_feature_summary(
                model_row["features"],
                n_features=n_features,
                location=f"position {position_index} {model}",
            )
            model_counts[model] += reported
            positive_counts[model] += positive
            observed_features[model].update(seen)

    blocking_reasons: list[str] = []
    if not examples:
        blocking_reasons.append("no compact position examples")
    if missing_models:
        blocking_reasons.append(
            "missing compact summaries for model(s): " + ", ".join(sorted(missing_models))
        )
    for model in ("raw", "hero"):
        if null_summary_counts[model]:
            blocking_reasons.append(
                f"{model} has null feature summaries for "
                f"{null_summary_counts[model]} position(s)"
            )
    for model in ("raw", "hero"):
        if positive_counts[model] == 0:
            blocking_reasons.append(f"{model} has no positive reported feature observations")

    return {
        "schema_version": COMPACT_CAPABILITY_SCHEMA,
        "structurally_valid": True,
        "passed_for_descriptive_peak_association": not blocking_reasons,
        "passed_for_token_level_semantics": False,
        "passed_for_feature_causality": False,
        "position_count": len(examples),
        "group_count": len(group_ids),
        "feature_count": n_features,
        "reported_feature_occurrences": model_counts,
        "positive_feature_occurrences": positive_counts,
        "null_feature_summary_count": null_summary_counts,
        "observed_feature_count": {
            model: len(indices) for model, indices in observed_features.items()
        },
        "observed_feature_fraction": {
            model: len(indices) / n_features for model, indices in observed_features.items()
        },
        "blocking_reasons": blocking_reasons,
        "token_semantics_blockers": [
            "unreported features are censored separately in every position",
            "per-token activations are absent",
            "active-token identities are absent except for one peak token",
            "inactive feature/token observations cannot be reconstructed",
        ],
        "permitted_estimand": (
            "class association of the retained peak square conditional on a feature "
            "appearing in the per-position strongest-feature summary"
        ),
    }

def _ordered_group_split(
    examples: Sequence[Mapping[str, Any]],
    *,
    fit_fraction: float,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]] | None:
    if not 0 < fit_fraction < 1 or not math.isfinite(fit_fraction):
        raise ValueError("fit_fraction must lie strictly between zero and one")
    groups = sorted({str(row["group_id"]) for row in examples})
    if len(groups) < 2:
        return None

    def rank(group: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()
        return digest, group

    ordered = sorted(groups, key=rank)
    fit_group_count = min(max(int(math.floor(len(ordered) * fit_fraction)), 1), len(ordered) - 1)
    fit_groups = set(ordered[:fit_group_count])
    evaluation_groups = set(ordered[fit_group_count:])
    fit = [row for row in examples if str(row["group_id"]) in fit_groups]
    evaluation = [row for row in examples if str(row["group_id"]) in evaluation_groups]
    if not fit or not evaluation or fit_groups & evaluation_groups:
        raise AssertionError("Deterministic group split construction failed")

    def identity(values: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()

    return fit, evaluation, {
        "method": "SHA-256(seed, group_id) ordering followed by a fixed group cut",
        "seed": seed,
        "fit_fraction_requested": fit_fraction,
        "fit_group_count": len(fit_groups),
        "evaluation_group_count": len(evaluation_groups),
        "fit_position_count": len(fit),
        "evaluation_position_count": len(evaluation),
        "fit_group_identity_sha256": identity(list(fit_groups)),
        "evaluation_group_identity_sha256": identity(list(evaluation_groups)),
        "fit_position_identity_sha256": identity([str(row["position_id"]) for row in fit]),
        "evaluation_position_identity_sha256": identity(
            [str(row["position_id"]) for row in evaluation]
        ),
        "group_disjoint": True,
        "selection_split": "fit",
        "evaluation_split": "evaluation",
        "selection_source_model": "raw",
    }


def _align_labels(
    examples: Sequence[Mapping[str, Any]],
    puzzles: Mapping[str, np.ndarray],
    concepts: Sequence[CompactPeakConceptSpec],
) -> dict[str, dict[str, np.ndarray]]:
    required = {"position_id", "split_u8", *(concept.array for concept in concepts)}
    missing = sorted(required - set(puzzles))
    if missing:
        raise ValueError("Corpus is missing compact semantic arrays: " + ", ".join(missing))
    position_ids = np.asarray(puzzles["position_id"])
    split = np.asarray(puzzles["split_u8"])
    if position_ids.ndim != 1 or split.shape != position_ids.shape:
        raise ValueError("Corpus position IDs and split codes must be aligned vectors")
    index_by_id: dict[str, int] = {}
    for index, value in enumerate(position_ids):
        position_id = str(value)
        if position_id in index_by_id:
            raise ValueError("Corpus position IDs must be unique")
        index_by_id[position_id] = index

    arrays: dict[str, np.ndarray] = {}
    for concept in concepts:
        concept.validate()
        value = np.asarray(puzzles[concept.array])
        if value.shape != (position_ids.size, 64):
            raise ValueError(f"Corpus concept {concept.name!r} must have shape [position, 64]")
        if value.dtype.kind not in "iub":
            raise ValueError(f"Corpus concept {concept.name!r} must be integer-valued")
        arrays[concept.name] = value.astype(np.int64, copy=False)

    aligned: dict[str, dict[str, np.ndarray]] = {}
    for example in examples:
        position_id = str(example["position_id"])
        if position_id not in index_by_id:
            raise ValueError(f"Compact position {position_id!r} is absent from the corpus")
        index = index_by_id[position_id]
        if int(split[index]) != 1:
            raise ValueError("Compact semantic adapter accepts development positions only")
        labels: dict[str, np.ndarray] = {}
        for concept in concepts:
            value = arrays[concept.name][index]
            if bool(((value < 0) | (value >= len(concept.class_names))).any()):
                raise ValueError(f"Corpus labels for {concept.name!r} violate class bounds")
            labels[concept.name] = value
        aligned[position_id] = labels
    return aligned


def _accumulate_peak_statistics(
    examples: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, np.ndarray]],
    concepts: Sequence[CompactPeakConceptSpec],
    *,
    n_features: int,
    model: str,
) -> _PeakStatistics:
    support = np.zeros(n_features, dtype=np.int64)
    class_tokens = {
        concept.name: np.zeros(len(concept.class_names), dtype=np.int64)
        for concept in concepts
    }
    feature_class = {
        concept.name: np.zeros((n_features, len(concept.class_names)), dtype=np.int64)
        for concept in concepts
    }
    reported = 0
    positive = 0
    for example in examples:
        position_id = str(example["position_id"])
        for concept in concepts:
            class_tokens[concept.name] += np.bincount(
                labels[position_id][concept.name], minlength=len(concept.class_names)
            )
        summary = example[model]["features"]
        if summary is None:
            continue
        for row in summary["top_features"]:
            reported += 1
            if int(row["active_token_count"]) == 0 or float(row["maximum_activation"]) == 0:
                continue
            positive += 1
            feature = int(row["feature"])
            peak_token = int(row["peak_token"])
            support[feature] += 1
            for concept in concepts:
                class_index = int(labels[position_id][concept.name][peak_token])
                feature_class[concept.name][feature, class_index] += 1
    return _PeakStatistics(support, class_tokens, feature_class, reported, positive)


def _effect_tensors(
    statistics: _PeakStatistics,
    concept: CompactPeakConceptSpec,
    *,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = len(concept.class_names)
    baseline_count = statistics.class_tokens[concept.name].astype(np.float64)
    baseline = (baseline_count + smoothing) / (baseline_count.sum() + smoothing * width)
    # Shrink each feature's categorical peak distribution toward the observed
    # square-label baseline, not toward a uniform class distribution. Uniform
    # pseudo-counts can make an unobserved rare class look spuriously enriched.
    conditional = (
        statistics.feature_class_peaks[concept.name].astype(np.float64)
        + smoothing * baseline[None, :]
    ) / (statistics.feature_support[:, None].astype(np.float64) + smoothing)
    return np.log2(conditional / baseline[None, :]), baseline


def _direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _select_raw_fit_pairs(
    statistics: _PeakStatistics,
    concepts: Sequence[CompactPeakConceptSpec],
    *,
    top_per_class: int,
    minimum_fit_support: int,
    minimum_fit_joint_support: int,
    smoothing: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    eligible_features = np.flatnonzero(statistics.feature_support >= minimum_fit_support)
    for concept in concepts:
        lift, baseline = _effect_tensors(statistics, concept, smoothing=smoothing)
        for class_index, class_name in enumerate(concept.class_names):
            if statistics.class_tokens[concept.name][class_index] == 0:
                continue
            candidates: dict[str, list[tuple[float, int, int, float]]] = {
                "positive": [],
                "negative": [],
            }
            for feature_value in eligible_features:
                feature = int(feature_value)
                value = float(lift[feature, class_index])
                direction = _direction(value)
                if direction == "zero":
                    continue
                joint = int(
                    statistics.feature_class_peaks[concept.name][feature, class_index]
                )
                expected_joint = float(
                    statistics.feature_support[feature] * baseline[class_index]
                )
                # Positive evidence needs repeated observed peaks. Negative
                # evidence needs enough expected peaks to make absence meaningful.
                if direction == "positive" and joint < minimum_fit_joint_support:
                    continue
                if direction == "negative" and expected_joint < minimum_fit_joint_support:
                    continue
                candidates[direction].append(
                    (abs(value), feature, joint, expected_joint)
                )
            for direction in ("positive", "negative"):
                candidates[direction].sort(key=lambda item: (-item[0], item[1]))
                for _, feature, joint, expected_joint in candidates[direction][
                    :top_per_class
                ]:
                    value = float(lift[feature, class_index])
                    selected.append(
                        {
                            "concept": concept.name,
                            "class_index": class_index,
                            "class_name": class_name,
                            "feature_index": feature,
                            "selection_direction": direction,
                            "raw_fit_log2_lift": value,
                            "raw_fit_direction": direction,
                            "raw_fit_feature_support": int(
                                statistics.feature_support[feature]
                            ),
                            "raw_fit_joint_peak_support": joint,
                            "raw_fit_expected_joint_peak_support": expected_joint,
                            "raw_fit_square_baseline_probability": float(
                                baseline[class_index]
                            ),
                        }
                    )
    return selected


def _evaluate_frozen_pairs(
    selection: Sequence[Mapping[str, Any]],
    raw: _PeakStatistics,
    hero: _PeakStatistics,
    concepts: Sequence[CompactPeakConceptSpec],
    *,
    minimum_evaluation_support: int,
    smoothing: float,
) -> list[dict[str, Any]]:
    concept_by_name = {concept.name: concept for concept in concepts}
    raw_effects = {
        concept.name: _effect_tensors(raw, concept, smoothing=smoothing)[0]
        for concept in concepts
    }
    hero_effects = {
        concept.name: _effect_tensors(hero, concept, smoothing=smoothing)[0]
        for concept in concepts
    }
    rows: list[dict[str, Any]] = []
    for frozen in selection:
        concept_name = str(frozen["concept"])
        if concept_name not in concept_by_name:
            raise AssertionError("Frozen selection lost its concept contract")
        feature = int(frozen["feature_index"])
        class_index = int(frozen["class_index"])
        raw_support = int(raw.feature_support[feature])
        hero_support = int(hero.feature_support[feature])
        raw_lift = float(raw_effects[concept_name][feature, class_index])
        hero_lift = float(hero_effects[concept_name][feature, class_index])
        raw_gate = raw_support >= minimum_evaluation_support
        hero_gate = hero_support >= minimum_evaluation_support
        paired_gate = raw_gate and hero_gate
        raw_direction = _direction(raw_lift)
        hero_direction = _direction(hero_lift)
        fit_direction = str(frozen["raw_fit_direction"])
        rows.append(
            {
                **dict(frozen),
                "raw_evaluation_feature_support": raw_support,
                "hero_evaluation_feature_support": hero_support,
                "raw_evaluation_joint_peak_support": int(
                    raw.feature_class_peaks[concept_name][feature, class_index]
                ),
                "hero_evaluation_joint_peak_support": int(
                    hero.feature_class_peaks[concept_name][feature, class_index]
                ),
                "raw_evaluation_log2_lift": raw_lift,
                "hero_evaluation_log2_lift": hero_lift,
                "hero_minus_raw_log2_lift": hero_lift - raw_lift,
                "raw_evaluation_direction": raw_direction,
                "hero_evaluation_direction": hero_direction,
                "raw_support_gate": raw_gate,
                "hero_support_gate": hero_gate,
                "paired_support_gate": paired_gate,
                "raw_direction_replicated_from_fit": raw_gate
                and raw_direction == fit_direction,
                "hero_direction_transferred_from_raw_fit": hero_gate
                and hero_direction == fit_direction,
                "raw_hero_direction_agreement": paired_gate
                and raw_direction == hero_direction,
            }
        )
    return rows


def _coverage(statistics: _PeakStatistics, *, n_features: int) -> dict[str, Any]:
    live = int((statistics.feature_support > 0).sum())
    return {
        "reported_feature_occurrences": statistics.reported_occurrences,
        "positive_feature_occurrences": statistics.positive_occurrences,
        "observed_feature_count": live,
        "observed_feature_fraction": live / n_features,
        "maximum_feature_support": int(statistics.feature_support.max(initial=0)),
    }


def _blocked_report(
    capability: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": COMPACT_PEAK_REPORT_SCHEMA,
        "status": "blocked_insufficient_evidence",
        "capability_gate": dict(capability),
        "blocking_reason": reason,
        "claim_gate": {
            "descriptive_peak_location_association": False,
            "token_level_feature_semantics": False,
            "feature_causality": False,
            "raw_hero_semantic_equivalence": False,
        },
        "notebook_rows": [],
        "negative_and_null_results": {
            "selected_pair_count": 0,
            "reason": reason,
        },
    }


def build_compact_peak_feature_comparison(
    examples: Sequence[Mapping[str, Any]],
    puzzles: Mapping[str, np.ndarray],
    *,
    n_features: int,
    concepts: Sequence[CompactPeakConceptSpec],
    fit_fraction: float = 0.5,
    seed: int = 20260808,
    top_per_class: int = 5,
    minimum_fit_support: int = 5,
    minimum_fit_joint_support: int = 2,
    minimum_evaluation_support: int = 3,
    smoothing: float = 0.5,
) -> dict[str, Any]:
    """Build a Raw-selected, group-heldout compact peak-location comparison."""

    if not concepts:
        raise ValueError("At least one compact peak concept is required")
    for concept in concepts:
        concept.validate()
    if len({concept.name for concept in concepts}) != len(concepts):
        raise ValueError("Compact peak concept names must be unique")
    if type(top_per_class) is not int or top_per_class <= 0:
        raise ValueError("top_per_class must be a positive integer")
    if type(minimum_fit_support) is not int or minimum_fit_support <= 0:
        raise ValueError("minimum_fit_support must be a positive integer")
    if type(minimum_fit_joint_support) is not int or minimum_fit_joint_support <= 0:
        raise ValueError("minimum_fit_joint_support must be a positive integer")
    if type(minimum_evaluation_support) is not int or minimum_evaluation_support <= 0:
        raise ValueError("minimum_evaluation_support must be a positive integer")
    if smoothing <= 0 or not math.isfinite(smoothing):
        raise ValueError("smoothing must be finite and positive")

    capability = validate_compact_feature_capability(examples, n_features=n_features)
    if not capability["passed_for_descriptive_peak_association"]:
        return _blocked_report(
            capability,
            reason="; ".join(capability["blocking_reasons"]),
        )
    split = _ordered_group_split(examples, fit_fraction=fit_fraction, seed=seed)
    if split is None:
        return _blocked_report(
            capability,
            reason="fewer than two groups; group-disjoint fit/evaluation is impossible",
        )
    fit_examples, evaluation_examples, split_contract = split
    labels = _align_labels(examples, puzzles, concepts)
    raw_fit = _accumulate_peak_statistics(
        fit_examples, labels, concepts, n_features=n_features, model="raw"
    )
    raw_evaluation = _accumulate_peak_statistics(
        evaluation_examples, labels, concepts, n_features=n_features, model="raw"
    )
    hero_evaluation = _accumulate_peak_statistics(
        evaluation_examples, labels, concepts, n_features=n_features, model="hero"
    )
    selection = _select_raw_fit_pairs(
        raw_fit,
        concepts,
        top_per_class=top_per_class,
        minimum_fit_support=minimum_fit_support,
        minimum_fit_joint_support=minimum_fit_joint_support,
        smoothing=smoothing,
    )
    rows = _evaluate_frozen_pairs(
        selection,
        raw_evaluation,
        hero_evaluation,
        concepts,
        minimum_evaluation_support=minimum_evaluation_support,
        smoothing=smoothing,
    )
    paired = [row for row in rows if row["paired_support_gate"]]
    raw_supported = [row for row in rows if row["raw_support_gate"]]
    hero_supported = [row for row in rows if row["hero_support_gate"]]
    raw_replicated = sum(row["raw_direction_replicated_from_fit"] for row in raw_supported)
    hero_transferred = sum(
        row["hero_direction_transferred_from_raw_fit"] for row in hero_supported
    )
    cross_agreement = sum(row["raw_hero_direction_agreement"] for row in paired)
    status = "complete_descriptive"
    if not selection:
        status = "complete_no_selected_pairs"
    elif not paired:
        status = "complete_no_paired_supported_pairs"

    notebook_rows = [
        {
            "concept": row["concept"],
            "class": row["class_name"],
            "feature": row["feature_index"],
            "selection_direction": row["selection_direction"],
            "raw_fit_lift": row["raw_fit_log2_lift"],
            "raw_eval_lift": row["raw_evaluation_log2_lift"],
            "hero_eval_lift": row["hero_evaluation_log2_lift"],
            "hero_minus_raw": row["hero_minus_raw_log2_lift"],
            "raw_fit_joint_support": row["raw_fit_joint_peak_support"],
            "raw_fit_expected_joint_support": row[
                "raw_fit_expected_joint_peak_support"
            ],
            "raw_support": row["raw_evaluation_feature_support"],
            "hero_support": row["hero_evaluation_feature_support"],
            "paired_support_gate": row["paired_support_gate"],
            "raw_direction_replicated": row["raw_direction_replicated_from_fit"],
            "hero_direction_transferred": row[
                "hero_direction_transferred_from_raw_fit"
            ],
            "raw_hero_direction_agreement": row["raw_hero_direction_agreement"],
        }
        for row in rows
    ]
    return {
        "schema_version": COMPACT_PEAK_REPORT_SCHEMA,
        "status": status,
        "capability_gate": capability,
        "claim_gate": {
            "descriptive_peak_location_association": status == "complete_descriptive",
            "token_level_feature_semantics": False,
            "feature_causality": False,
            "raw_hero_semantic_equivalence": False,
        },
        "split_contract": split_contract,
        "selection_contract": {
            "source_model": "raw",
            "source_split": "fit",
            "evaluation_models": ["raw", "hero"],
            "evaluation_split": "evaluation",
            "top_per_class": top_per_class,
            "top_per_class_interpretation": "separate slots for positive and negative",
            "minimum_fit_support": minimum_fit_support,
            "minimum_fit_joint_support": minimum_fit_joint_support,
            "minimum_evaluation_support": minimum_evaluation_support,
            "smoothing": smoothing,
            "smoothing_prior": "observed square-label baseline",
            "reselection_on_evaluation": False,
        },
        "concepts": [
            {
                "name": concept.name,
                "array": concept.array,
                "class_names": list(concept.class_names),
            }
            for concept in concepts
        ],
        "coverage": {
            "raw_fit": _coverage(raw_fit, n_features=n_features),
            "raw_evaluation": _coverage(raw_evaluation, n_features=n_features),
            "hero_evaluation": _coverage(hero_evaluation, n_features=n_features),
        },
        "summary": {
            "selected_pair_count": len(rows),
            "raw_supported_pair_count": len(raw_supported),
            "hero_supported_pair_count": len(hero_supported),
            "paired_supported_pair_count": len(paired),
            "raw_direction_replication_count": raw_replicated,
            "raw_direction_replication_rate_supported": (
                raw_replicated / len(raw_supported) if raw_supported else None
            ),
            "hero_direction_transfer_count": hero_transferred,
            "hero_direction_transfer_rate_supported": (
                hero_transferred / len(hero_supported) if hero_supported else None
            ),
            "raw_hero_direction_agreement_count": cross_agreement,
            "raw_hero_direction_agreement_rate_paired": (
                cross_agreement / len(paired) if paired else None
            ),
        },
        "negative_and_null_results": {
            "fit_positive_selected_pairs": sum(
                row["selection_direction"] == "positive" for row in rows
            ),
            "fit_negative_selected_pairs": sum(
                row["selection_direction"] == "negative" for row in rows
            ),
            "unsupported_in_one_or_both_models": len(rows) - len(paired),
            "raw_fit_direction_not_replicated": len(raw_supported) - raw_replicated,
            "raw_fit_direction_not_transferred_to_hero": (
                len(hero_supported) - hero_transferred
            ),
            "raw_hero_direction_disagreement": len(paired) - cross_agreement,
            "inferential_intervals": "not estimated",
            "multiple_comparison_correction": "not applicable; descriptive adapter only",
        },
        "pairs": rows,
        "notebook_rows": notebook_rows,
        "interpretation_contract": [
            "Nested development analysis; the frozen test split remains unopened.",
            "Feature/concept pairs are selected on Raw fit groups only and never reselected on evaluation.",
            "The estimand is peak-square association among per-position reported top features.",
            "Unreported features and non-peak active tokens are censored and cannot be analyzed.",
            "No uncertainty interval or causal feature claim is available from compact summaries.",
            "Raw/Hero direction agreement is not semantic equivalence.",
        ],
    }
