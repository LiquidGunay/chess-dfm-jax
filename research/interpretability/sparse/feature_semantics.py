"""Streaming, split-aware semantic audits for sparse chess features.

The accumulator accepts dense top-k feature tensors because that is the native
output ABI of the transcoder and LoRSA modules, but it stores only sufficient
statistics.  It therefore avoids retaining ``[positions, 64, d_sae]`` dumps.

Feature/concept pairs must be selected on one split and evaluated on another.
The reported lifts are descriptive effect sizes, not significance tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseConceptSpec:
    """One categorical square-token concept.

    Labels use integers in ``[0, class_count)``.  ``-1`` is reserved for an
    explicitly missing label and is excluded from that concept's statistics.
    """

    name: str
    class_names: tuple[str, ...]

    def validate(self) -> None:
        if not self.name or self.name.startswith("_"):
            raise ValueError("Sparse concept names must be public, non-empty identifiers")
        if len(self.class_names) < 2 or any(not value for value in self.class_names):
            raise ValueError("Sparse concepts require at least two named classes")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Sparse concept class names must be unique")


@dataclass
class _ConceptCounts:
    valid_tokens: int
    class_tokens: Tensor
    feature_class_active: Tensor
    feature_class_activation: Tensor


class SparseFeatureConceptAccumulator:
    """Accumulate sparse feature/concept sufficient statistics on CPU."""

    def __init__(self, n_features: int, concepts: Sequence[SparseConceptSpec]):
        if type(n_features) is not int or n_features <= 0:
            raise ValueError("n_features must be a positive integer")
        if not concepts:
            raise ValueError("At least one sparse concept is required")
        for concept in concepts:
            concept.validate()
        if len({concept.name for concept in concepts}) != len(concepts):
            raise ValueError("Sparse concept names must be unique")
        self.n_features = n_features
        self.concepts = tuple(concepts)
        self.total_tokens = 0
        self.total_nonzero = 0
        self.feature_active = torch.zeros(n_features, dtype=torch.int64)
        self.feature_activation = torch.zeros(n_features, dtype=torch.float64)
        self._counts: dict[str, _ConceptCounts] = {}
        for concept in concepts:
            width = len(concept.class_names)
            self._counts[concept.name] = _ConceptCounts(
                valid_tokens=0,
                class_tokens=torch.zeros(width, dtype=torch.int64),
                feature_class_active=torch.zeros((n_features, width), dtype=torch.int64),
                feature_class_activation=torch.zeros(
                    (n_features, width), dtype=torch.float64
                ),
            )

    @torch.no_grad()
    def update(self, features: Tensor, labels: Mapping[str, Tensor]) -> None:
        """Add one batch of ``[batch, square, feature]`` activations and labels."""

        if features.ndim != 3 or features.shape[-1] != self.n_features:
            raise ValueError("Sparse features must have shape [batch, token, n_features]")
        if not torch.is_floating_point(features):
            raise ValueError("Sparse features must use a floating dtype")
        if not bool(torch.isfinite(features).all()) or bool((features < 0).any()):
            raise ValueError("Sparse features must be finite and non-negative")
        expected = {concept.name for concept in self.concepts}
        if set(labels) != expected:
            raise ValueError("Sparse semantic label keys differ from the concept contract")

        token_shape = features.shape[:-1]
        flat = features.detach().reshape(-1, self.n_features)
        active_indices = torch.nonzero(flat > 0, as_tuple=False)
        if active_indices.numel():
            token_indices = active_indices[:, 0]
            feature_indices = active_indices[:, 1]
            active_values = flat[token_indices, feature_indices]
            token_indices_cpu = token_indices.to(device="cpu", dtype=torch.int64)
            feature_indices_cpu = feature_indices.to(device="cpu", dtype=torch.int64)
            active_values_cpu = active_values.to(device="cpu", dtype=torch.float64)
        else:
            token_indices_cpu = torch.empty(0, dtype=torch.int64)
            feature_indices_cpu = torch.empty(0, dtype=torch.int64)
            active_values_cpu = torch.empty(0, dtype=torch.float64)

        token_count = flat.shape[0]
        self.total_tokens += token_count
        self.total_nonzero += feature_indices_cpu.numel()
        self.feature_active += torch.bincount(
            feature_indices_cpu, minlength=self.n_features
        )
        self.feature_activation += torch.bincount(
            feature_indices_cpu,
            weights=active_values_cpu,
            minlength=self.n_features,
        )

        for concept in self.concepts:
            label = labels[concept.name]
            if tuple(label.shape) != tuple(token_shape):
                raise ValueError(f"Label {concept.name!r} does not match feature tokens")
            if label.dtype == torch.bool or torch.is_floating_point(label):
                raise ValueError(f"Label {concept.name!r} must use an integer dtype")
            label_cpu = label.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
            width = len(concept.class_names)
            if bool(((label_cpu < -1) | (label_cpu >= width)).any()):
                raise ValueError(f"Label {concept.name!r} lies outside its class contract")
            valid = label_cpu >= 0
            counts = self._counts[concept.name]
            counts.valid_tokens += int(valid.sum())
            counts.class_tokens += torch.bincount(
                label_cpu[valid], minlength=width
            )

            active_valid = valid[token_indices_cpu]
            if not bool(active_valid.any()):
                continue
            selected_feature = feature_indices_cpu[active_valid]
            selected_label = label_cpu[token_indices_cpu[active_valid]]
            selected_value = active_values_cpu[active_valid]
            combined = selected_feature * width + selected_label
            counts.feature_class_active += torch.bincount(
                combined, minlength=self.n_features * width
            ).reshape(self.n_features, width)
            counts.feature_class_activation += torch.bincount(
                combined,
                weights=selected_value,
                minlength=self.n_features * width,
            ).reshape(self.n_features, width)

    def activity_metrics(self) -> dict[str, Any]:
        if self.total_tokens == 0:
            raise ValueError("Cannot summarize an empty sparse accumulator")
        live = self.feature_active > 0
        return {
            "token_count": self.total_tokens,
            "feature_count": self.n_features,
            "nonzero_count": self.total_nonzero,
            "mean_l0": self.total_nonzero / self.total_tokens,
            "live_feature_count": int(live.sum()),
            "dead_feature_fraction": float((~live).double().mean()),
            "mean_activation_given_active": float(
                self.feature_activation.sum() / max(self.total_nonzero, 1)
            ),
        }

    def _effect_tensors(
        self,
        concept_name: str,
        *,
        smoothing: float,
    ) -> dict[str, Tensor | int]:
        if smoothing <= 0 or not math.isfinite(smoothing):
            raise ValueError("smoothing must be finite and positive")
        if concept_name not in self._counts:
            raise ValueError(f"Unknown sparse concept {concept_name!r}")
        counts = self._counts[concept_name]
        if counts.valid_tokens == 0:
            raise ValueError(f"Sparse concept {concept_name!r} has no valid tokens")
        width = counts.class_tokens.numel()
        active_support = counts.feature_class_active.sum(dim=1)
        activation_support = counts.feature_class_activation.sum(dim=1)
        baseline = (counts.class_tokens.double() + smoothing) / (
            counts.valid_tokens + smoothing * width
        )
        conditional = (counts.feature_class_active.double() + smoothing) / (
            active_support.double().unsqueeze(1) + smoothing * width
        )
        activation_conditional = (
            counts.feature_class_activation + smoothing
        ) / (activation_support.unsqueeze(1) + smoothing * width)
        return {
            "valid_tokens": counts.valid_tokens,
            "class_tokens": counts.class_tokens,
            "active_support": active_support,
            "joint_support": counts.feature_class_active,
            "activation_support": activation_support,
            "joint_activation": counts.feature_class_activation,
            "baseline": baseline,
            "conditional": conditional,
            "activation_conditional": activation_conditional,
            "log2_lift": torch.log2(conditional / baseline.unsqueeze(0)),
            "activation_log2_lift": torch.log2(
                activation_conditional / baseline.unsqueeze(0)
            ),
        }


def select_sparse_feature_concepts(
    accumulator: SparseFeatureConceptAccumulator,
    *,
    top_per_class: int = 10,
    minimum_feature_support: int = 20,
    smoothing: float = 0.5,
) -> list[dict[str, Any]]:
    """Select the largest absolute feature/concept lifts on a fit split."""

    if type(top_per_class) is not int or top_per_class <= 0:
        raise ValueError("top_per_class must be a positive integer")
    if type(minimum_feature_support) is not int or minimum_feature_support <= 0:
        raise ValueError("minimum_feature_support must be a positive integer")
    selected: list[dict[str, Any]] = []
    for concept in accumulator.concepts:
        effects = accumulator._effect_tensors(concept.name, smoothing=smoothing)
        lift = effects["log2_lift"]
        activation_lift = effects["activation_log2_lift"]
        support = effects["active_support"]
        joint = effects["joint_support"]
        assert isinstance(lift, Tensor)
        assert isinstance(activation_lift, Tensor)
        assert isinstance(support, Tensor)
        assert isinstance(joint, Tensor)
        eligible = support >= minimum_feature_support
        for class_index, class_name in enumerate(concept.class_names):
            candidates = [
                (float(abs(lift[index, class_index])), index)
                for index in torch.nonzero(eligible, as_tuple=False).flatten().tolist()
            ]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _, feature_index in candidates[:top_per_class]:
                value = float(lift[feature_index, class_index])
                selected.append(
                    {
                        "concept": concept.name,
                        "class_index": class_index,
                        "class_name": class_name,
                        "feature_index": feature_index,
                        "fit_log2_lift": value,
                        "fit_activation_log2_lift": float(
                            activation_lift[feature_index, class_index]
                        ),
                        "fit_direction": "positive" if value >= 0 else "negative",
                        "fit_feature_support": int(support[feature_index]),
                        "fit_joint_support": int(joint[feature_index, class_index]),
                    }
                )
    return selected


def evaluate_sparse_feature_concepts(
    accumulator: SparseFeatureConceptAccumulator,
    selection: Sequence[Mapping[str, Any]],
    *,
    minimum_evaluation_support: int = 10,
    smoothing: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a frozen fit-split selection on a disjoint split."""

    if type(minimum_evaluation_support) is not int or minimum_evaluation_support <= 0:
        raise ValueError("minimum_evaluation_support must be a positive integer")
    concept_names = {concept.name for concept in accumulator.concepts}
    effect_cache = {
        name: accumulator._effect_tensors(name, smoothing=smoothing)
        for name in concept_names
    }
    rows: list[dict[str, Any]] = []
    supported = 0
    replicated = 0
    for frozen in selection:
        concept_name = str(frozen["concept"])
        if concept_name not in effect_cache:
            raise ValueError(f"Frozen selection uses unknown concept {concept_name!r}")
        feature_index = int(frozen["feature_index"])
        class_index = int(frozen["class_index"])
        effects = effect_cache[concept_name]
        support = effects["active_support"]
        joint = effects["joint_support"]
        lift = effects["log2_lift"]
        activation_lift = effects["activation_log2_lift"]
        assert isinstance(support, Tensor)
        assert isinstance(joint, Tensor)
        assert isinstance(lift, Tensor)
        assert isinstance(activation_lift, Tensor)
        if not 0 <= feature_index < accumulator.n_features:
            raise ValueError("Frozen sparse feature index is out of bounds")
        if not 0 <= class_index < lift.shape[1]:
            raise ValueError("Frozen sparse concept class index is out of bounds")
        evaluation_support = int(support[feature_index])
        is_supported = evaluation_support >= minimum_evaluation_support
        evaluation_lift = float(lift[feature_index, class_index])
        fit_lift = float(frozen["fit_log2_lift"])
        direction_replicated = is_supported and (
            (evaluation_lift >= 0) == (fit_lift >= 0)
        )
        supported += int(is_supported)
        replicated += int(direction_replicated)
        rows.append(
            {
                **dict(frozen),
                "evaluation_feature_support": evaluation_support,
                "evaluation_joint_support": int(joint[feature_index, class_index]),
                "evaluation_log2_lift": evaluation_lift,
                "evaluation_activation_log2_lift": float(
                    activation_lift[feature_index, class_index]
                ),
                "evaluation_support_gate": is_supported,
                "direction_replicated": direction_replicated,
            }
        )
    return {
        "schema_version": "bt4-sparse-feature-semantic-heldout-v1",
        "selection_split_is_distinct_required": True,
        "selected_pair_count": len(rows),
        "supported_pair_count": supported,
        "direction_replicated_count": replicated,
        "direction_replication_rate_supported": replicated / supported if supported else None,
        "minimum_evaluation_support": minimum_evaluation_support,
        "smoothing": smoothing,
        "activity": accumulator.activity_metrics(),
        "pairs": rows,
        "interpretation_contract": (
            "Descriptive heldout association only; feature causality requires ablation, "
            "insertion, specificity, dose-response, and matched random controls."
        ),
    }
