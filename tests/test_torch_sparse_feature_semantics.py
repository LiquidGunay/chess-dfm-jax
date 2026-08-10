from __future__ import annotations

import pytest
import torch

from research.interpretability.sparse.feature_semantics import (
    SparseConceptSpec,
    SparseFeatureConceptAccumulator,
    evaluate_sparse_feature_concepts,
    select_sparse_feature_concepts,
)


def _concepts() -> tuple[SparseConceptSpec, ...]:
    return (SparseConceptSpec("occupied", ("empty", "occupied")),)


def _batch() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features = torch.zeros((2, 4, 3), dtype=torch.float32)
    labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.int64)
    features[..., 0] = labels.float()
    features[..., 1] = (1 - labels).float() * 2
    features[0, 0, 2] = 0.5
    return features, {"occupied": labels}


def test_sparse_semantics_select_then_replicate_on_heldout() -> None:
    fit = SparseFeatureConceptAccumulator(3, _concepts())
    evaluation = SparseFeatureConceptAccumulator(3, _concepts())
    features, labels = _batch()
    fit.update(features, labels)
    evaluation.update(features.flip(0), {"occupied": labels["occupied"].flip(0)})

    selected = select_sparse_feature_concepts(
        fit,
        top_per_class=2,
        minimum_feature_support=2,
    )
    result = evaluate_sparse_feature_concepts(
        evaluation,
        selected,
        minimum_evaluation_support=2,
    )

    assert result["selected_pair_count"] == 4
    assert result["supported_pair_count"] == 4
    assert result["direction_replicated_count"] == 4
    assert result["direction_replication_rate_supported"] == 1.0
    by_pair = {
        (row["feature_index"], row["class_index"]): row for row in result["pairs"]
    }
    assert by_pair[(0, 1)]["evaluation_log2_lift"] > 0
    assert by_pair[(0, 0)]["evaluation_log2_lift"] < 0
    assert by_pair[(1, 0)]["evaluation_log2_lift"] > 0
    assert result["activity"]["dead_feature_fraction"] == pytest.approx(0.0)


def test_sparse_semantics_is_chunk_invariant_and_supports_missing_labels() -> None:
    features, labels = _batch()
    labels = {"occupied": labels["occupied"].clone()}
    labels["occupied"][0, 0] = -1
    whole = SparseFeatureConceptAccumulator(3, _concepts())
    chunked = SparseFeatureConceptAccumulator(3, _concepts())
    whole.update(features, labels)
    for index in range(2):
        chunked.update(features[index : index + 1], {"occupied": labels["occupied"][index : index + 1]})

    assert whole.activity_metrics() == chunked.activity_metrics()
    assert select_sparse_feature_concepts(
        whole, top_per_class=2, minimum_feature_support=2
    ) == select_sparse_feature_concepts(
        chunked, top_per_class=2, minimum_feature_support=2
    )


@pytest.mark.parametrize(
    "features,labels,error",
    [
        (torch.full((1, 2, 3), -1.0), torch.zeros((1, 2), dtype=torch.long), "non-negative"),
        (torch.zeros((1, 2, 2)), torch.zeros((1, 2), dtype=torch.long), "shape"),
        (torch.zeros((1, 2, 3)), torch.full((1, 2), 2, dtype=torch.long), "outside"),
        (torch.zeros((1, 2, 3)), torch.zeros((1, 3), dtype=torch.long), "does not match"),
    ],
)
def test_sparse_semantics_rejects_invalid_batches(
    features: torch.Tensor,
    labels: torch.Tensor,
    error: str,
) -> None:
    accumulator = SparseFeatureConceptAccumulator(3, _concepts())
    with pytest.raises(ValueError, match=error):
        accumulator.update(features, {"occupied": labels})
