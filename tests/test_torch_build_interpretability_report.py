from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from research.interpretability.build_interpretability_report import (
    _artifact,
    _build_cost_audit,
    _probe_datasets,
)


def _layer_row() -> dict[str, float | int]:
    row: dict[str, float | int] = {"layer": 0}
    for prefix, raw_primary, hero_primary in (
        ("piece_code", 0.9, 0.8),
        ("legal_destination", 0.7, 0.8),
        ("opponent_attack_count", 0.4, 0.5),
    ):
        row[f"{prefix}_raw_primary"] = raw_primary
        row[f"{prefix}_hero_primary"] = hero_primary
        row[f"{prefix}_raw"] = raw_primary
        row[f"{prefix}_hero"] = hero_primary
        row[f"{prefix}_delta"] = hero_primary - raw_primary
        row[f"{prefix}_ci_low"] = 0.01
        row[f"{prefix}_ci_high"] = 0.2
    for prefix in (
        "lookahead_destination_accuracy",
        "lookahead_source_given_predicted_destination_accuracy",
        "lookahead_joint_move_accuracy",
    ):
        row[f"{prefix}_raw"] = 0.4
        row[f"{prefix}_hero"] = 0.5
        row[f"{prefix}_delta"] = 0.1
        row[f"{prefix}_ci_low"] = -0.01
        row[f"{prefix}_ci_high"] = 0.2
    return row


def test_probe_datasets_keep_display_and_paired_metric_semantics() -> None:
    datasets = _probe_datasets({"layer_metrics": [_layer_row()]})
    legal = [row for row in datasets["concept_metrics"] if row["concept"] == "Legal destination"]
    assert {row["metric"] for row in legal} == {"AUROC"}
    assert {row["score"] for row in legal} == {0.7, 0.8}

    legal_delta = next(
        row for row in datasets["concept_deltas"] if row["concept"] == "Legal destination"
    )
    assert legal_delta["paired_metric"] == "accuracy"
    assert legal_delta["direction"] == "Hero higher"

    attack_delta = next(
        row for row in datasets["concept_deltas"] if row["concept"] == "Opponent attack count"
    )
    assert attack_delta["paired_metric"] == "negative MAE"
    assert len(datasets["lookahead_metrics"]) == 6


def test_cost_audit_separates_cpu_transfer_from_gpu_network() -> None:
    audit = _build_cost_audit(
        metered_cost=Decimal("0.46430113"),
        billed_cost=Decimal("0"),
        pre_powered_run_cost=Decimal("0.02072433"),
        monthly_budget=Decimal("42.50"),
        observed_at="2026-08-08T07:58:34Z",
    )
    assert audit["powered_pipeline_incremental_metered_cost_dollars"] == 0.4435768
    assert audit["pipeline"]["cpu_input_stage"]["network_download_bytes"] == 1_053_370_877
    assert audit["pipeline"]["t4_all_layer_run"]["network_download_bytes"] == 0
    assert audit["breakdown"]["volumes_dollars"] == 0.0


def test_portable_artifact_has_title_relative_sources_and_runnable_sql() -> None:
    concept_metrics = []
    for concept, metric in (
        ("Legal destination", "AUROC"),
        ("Opponent attack count", "R²"),
        ("Piece code", "Accuracy"),
    ):
        for model in ("Raw BT4", "Hero"):
            concept_metrics.append(
                {"layer": 0, "model": model, "concept": concept, "metric": metric, "score": 0.5}
            )
    lookahead_metrics = [
        {"layer": 0, "model": model, "target": target, "accuracy": 0.5}
        for target in ("Third-ply destination", "Third-ply joint move")
        for model in ("Raw BT4", "Hero")
    ]
    validation = {
        "run": {"split_counts": {"evaluation": 2}, "layers": [0]},
        "recomputation": {
            "metric_groups_recomputed_from_saved_predictions": 1,
            "maximum_absolute_metric_difference": 0.0,
        },
        "controls": {
            "primary_beats_permutation_and_frequency_or_mean_control_count": 1,
            "primary_control_comparison_count": 1,
        },
    }
    synthesis = {
        "source_files": {},
        "validation": validation,
        "datasets": {
            "concept_metrics": concept_metrics,
            "lookahead_metrics": lookahead_metrics,
            "concept_deltas": [],
            "lookahead_deltas": [],
            "model_diff": {
                "activation_depth": [
                    {
                        "layer": 0,
                        "corresponding_layer_cka": 0.5,
                        "metric_label": "Corresponding-layer CKA",
                    }
                ]
            },
            "literature": {
                "patching": [
                    {
                        "layer": 0,
                        "direction": "Hero → Raw",
                        "distribution_delta_projection": 0.5,
                    }
                ]
            },
            "evidence_matrix": [],
            "transcoder_table": [],
        },
    }
    cost = {
        "metered_cost_dollars": 0.4,
        "billed_cost_after_credits_dollars": 0.0,
        "metered_fraction_of_budget": 0.01,
        "powered_pipeline_incremental_metered_cost_dollars": 0.3,
    }
    artifact = _artifact(
        generated_at="2026-08-08T07:58:34Z",
        synthesis=synthesis,
        cost_audit=cost,
        report_dir=Path("research/analysis/report"),
    )
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# ")
    assert all(not source["path"].startswith("/") for source in artifact["manifest"]["sources"])
    assert all(source["query"]["sql"].startswith("SELECT ") for source in artifact["sources"])
