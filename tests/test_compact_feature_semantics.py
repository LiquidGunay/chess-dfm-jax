from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.interpretability.sparse import compact_feature_semantics as SEMANTICS
from research.interpretability.sparse.compact_feature_semantics import (
    CompactPeakConceptSpec as Concept,
)


def _feature_row(feature: int, *, peak_token: int, strength: float) -> dict[str, object]:
    return {
        "feature": feature,
        "summed_activation": strength,
        "maximum_activation": 1.0,
        "active_token_count": 2,
        "peak_token": peak_token,
    }


def _summary(*, feature_one_peak: int) -> dict[str, object]:
    return {
        "reported_feature_count": 2,
        "mean_token_l0": 1.5,
        "top_features": [
            _feature_row(0, peak_token=40, strength=3.0),
            _feature_row(1, peak_token=feature_one_peak, strength=2.0),
        ],
    }


def _examples(*, groups: int = 8, hero_feature_one_peak: int = 40):
    return [
        {
            "position_id": f"p{index}",
            "puzzle_id": f"z{index}",
            "group_id": f"g{index % groups}",
            "target_action": index,
            "raw": {"features": _summary(feature_one_peak=0)},
            "hero": {"features": _summary(feature_one_peak=hero_feature_one_peak)},
        }
        for index in range(8)
    ]


def _puzzles(*, split_code: int = 1) -> dict[str, np.ndarray]:
    labels = np.zeros((8, 64), dtype=np.uint8)
    labels[:, 32:] = 1
    return {
        "position_id": np.asarray([f"p{index}" for index in range(8)]),
        "split_u8": np.full(8, split_code, dtype=np.uint8),
        "legal_destination_u8": labels,
    }


def _build(examples=None, puzzles=None, **overrides):
    options = {
        "n_features": 4,
        "concepts": [
            Concept(
                name="legal_destination",
                array="legal_destination_u8",
                class_names=("not legal", "legal"),
            )
        ],
        "fit_fraction": 0.5,
        "seed": 9,
        "top_per_class": 2,
        "minimum_fit_support": 2,
        "minimum_evaluation_support": 2,
        "smoothing": 0.5,
    }
    options.update(overrides)
    return SEMANTICS.build_compact_peak_feature_comparison(
        _examples() if examples is None else examples,
        _puzzles() if puzzles is None else puzzles,
        **options,
    )


def _frozen_selection(report: dict[str, object]) -> list[tuple[object, ...]]:
    return sorted(
        (
            row["concept"],
            row["class_index"],
            row["feature_index"],
            row["raw_fit_log2_lift"],
            row["raw_fit_direction"],
            row["raw_fit_feature_support"],
        )
        for row in report["pairs"]
    )


def test_raw_fit_selection_is_frozen_for_group_disjoint_raw_hero_evaluation():
    report = _build()

    assert report["status"] == "complete_descriptive"
    assert report["schema_version"] == "bt4-lorsa-compact-peak-semantic-heldout-v1"
    assert report["split_contract"]["group_disjoint"] is True
    assert report["split_contract"]["fit_group_count"] == 4
    assert report["split_contract"]["evaluation_group_count"] == 4
    assert report["split_contract"]["fit_position_count"] == 4
    assert report["split_contract"]["evaluation_position_count"] == 4
    assert report["selection_contract"]["source_model"] == "raw"
    assert report["selection_contract"]["source_split"] == "fit"
    assert report["selection_contract"]["reselection_on_evaluation"] is False

    assert report["claim_gate"] == {
        "descriptive_peak_location_association": True,
        "token_level_feature_semantics": False,
        "feature_causality": False,
        "raw_hero_semantic_equivalence": False,
    }
    assert report["summary"] == {
        "selected_pair_count": 4,
        "raw_supported_pair_count": 4,
        "hero_supported_pair_count": 4,
        "paired_supported_pair_count": 4,
        "raw_direction_replication_count": 4,
        "raw_direction_replication_rate_supported": 1.0,
        "hero_direction_transfer_count": 2,
        "hero_direction_transfer_rate_supported": 0.5,
        "raw_hero_direction_agreement_count": 2,
        "raw_hero_direction_agreement_rate_paired": 0.5,
    }
    assert report["negative_and_null_results"]["raw_fit_direction_not_replicated"] == 0
    assert report["negative_and_null_results"]["raw_fit_direction_not_transferred_to_hero"] == 2
    assert report["negative_and_null_results"]["raw_hero_direction_disagreement"] == 2
    assert len(report["notebook_rows"]) == 4
    assert {row["feature"] for row in report["notebook_rows"]} == {0, 1}

    # Changing Hero alone changes held-out transfer outcomes, but cannot change
    # the Raw-fit pair selection or its frozen fit statistics.
    matching_hero = _build(examples=_examples(hero_feature_one_peak=0))
    assert _frozen_selection(matching_hero) == _frozen_selection(report)
    assert matching_hero["summary"]["hero_direction_transfer_count"] == 4
    assert matching_hero["summary"]["raw_hero_direction_agreement_count"] == 4


def test_null_compact_summaries_fail_closed_even_when_only_one_row_is_missing():
    examples = _examples()
    examples[0]["hero"]["features"] = None
    report = _build(examples=examples)

    assert report["status"] == "blocked_insufficient_evidence"
    assert report["capability_gate"]["null_feature_summary_count"]["hero"] == 1
    assert "hero has null feature summaries" in report["blocking_reason"]
    assert not any(report["claim_gate"].values())
    assert report["notebook_rows"] == []


def test_all_absent_summaries_fail_closed_with_explicit_negative_result():
    examples = _examples()
    for example in examples:
        example["raw"]["features"] = None
        example["hero"]["features"] = None
    report = _build(examples=examples)

    assert report["status"] == "blocked_insufficient_evidence"
    assert report["negative_and_null_results"]["selected_pair_count"] == 0
    assert "no positive reported feature observations" in report["blocking_reason"]
    assert report["capability_gate"]["passed_for_token_level_semantics"] is False


def test_single_group_fails_closed_before_any_selection():
    report = _build(examples=_examples(groups=1))

    assert report["status"] == "blocked_insufficient_evidence"
    assert "group-disjoint fit/evaluation is impossible" in report["blocking_reason"]
    assert report["notebook_rows"] == []


def test_no_fit_supported_features_is_an_explicit_null_result():
    report = _build(minimum_fit_support=100)

    assert report["status"] == "complete_no_selected_pairs"
    assert report["summary"]["selected_pair_count"] == 0
    assert report["claim_gate"]["descriptive_peak_location_association"] is False
    assert report["pairs"] == []
    assert report["notebook_rows"] == []


@pytest.mark.parametrize("failure", ["duplicate", "peak_out_of_bounds"])
def test_malformed_compact_feature_rows_are_rejected(failure: str):
    examples = _examples()
    rows = examples[0]["raw"]["features"]["top_features"]
    if failure == "duplicate":
        rows[1]["feature"] = rows[0]["feature"]
    else:
        rows[0]["peak_token"] = 64

    with pytest.raises(ValueError):
        _build(examples=examples)


def test_non_development_corpus_rows_are_rejected():
    with pytest.raises(ValueError, match="development positions only"):
        _build(puzzles=_puzzles(split_code=2))


def test_loader_rejects_non_object_jsonl_rows(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not an object"):
        SEMANTICS.load_compact_examples(path)


def test_baseline_shrinkage_does_not_invent_rare_class_enrichment():
    concept = Concept(
        name="rare_square",
        array="rare_square_u8",
        class_names=("common", "rare"),
    )
    statistics = SEMANTICS._PeakStatistics(
        feature_support=np.asarray([4], dtype=np.int64),
        class_tokens={"rare_square": np.asarray([252, 4], dtype=np.int64)},
        feature_class_peaks={
            "rare_square": np.asarray([[4, 0]], dtype=np.int64)
        },
        reported_occurrences=4,
        positive_occurrences=4,
    )

    lift, baseline = SEMANTICS._effect_tensors(
        statistics,
        concept,
        smoothing=0.5,
    )

    assert baseline[1] < 0.02
    assert lift[0, 0] > 0
    assert lift[0, 1] < 0
