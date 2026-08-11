"""Build the audited Raw-BT4 versus Hero interpretability report payload.

The module deliberately performs no network calls.  Provider billing values are
passed explicitly, while scientific values are read from immutable, checksummed
run directories and the independent probe validator output.  It writes a
portable-report ``artifact.json`` plus compact supporting JSON files; the HTML
is rendered separately by the repository-independent packaged report builder.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from research.interpretability.artifacts import (
    sha256_file,
    verify_checksums,
    write_json_atomic,
)

SCHEMA_VERSION = "bt4-raw-hero-interpretability-synthesis-v1"
DEFAULT_REPORT_DIR = Path("research/analysis/raw_hero_interpretability_report_v1")
PROBE_RUN_DIR = Path("research/analysis/modal_probe_all_layers_dev_v1")
DIFF_RUN_DIR = Path("research/analysis/raw_hero_pilot_v1")
LITERATURE_RUN_DIR = Path("research/analysis/raw_hero_literature_pilot_v2")
TRANSCODER_RUN_DIR = Path("research/analysis/published_tc_l14_transfer_pilot_v1")
REPORT_BUILDER_PATH = Path("research/interpretability/build_interpretability_report.py")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite")
    return number


def _interval_excludes_zero(row: dict[str, Any], prefix: str) -> bool:
    return float(row[f"{prefix}_ci_low"]) > 0.0 or float(row[f"{prefix}_ci_high"]) < 0.0


def _direction(delta: float, significant: bool) -> str:
    if not significant:
        return "unresolved"
    return "Hero higher" if delta > 0 else "Raw higher"


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_inputs(report_dir: Path) -> dict[str, dict[str, str]]:
    ledgers: dict[str, dict[str, str]] = {}
    for label, directory in (
        ("probe", PROBE_RUN_DIR),
        ("model_diff", DIFF_RUN_DIR),
        ("literature", LITERATURE_RUN_DIR),
        ("transcoder", TRANSCODER_RUN_DIR),
    ):
        ledgers[label] = verify_checksums(directory)
    validation = _load_json(report_dir / "validation.json")
    integrity = validation["integrity"]
    required_true = (
        "corpus_integrity_match",
        "deterministic_split_hashes_match",
        "evaluation_ids_match",
        "evaluation_rows_all_development",
        "pickle_free_weight_metadata",
        "all_causal_metrics_finite",
    )
    failed = [name for name in required_true if integrity.get(name) is not True]
    if failed:
        raise ValueError(f"Independent validation gates failed: {failed}")
    if integrity.get("test_rows_evaluated") is not False:
        raise ValueError("Frozen test split was unexpectedly opened")
    recomputation = validation["recomputation"]
    if float(recomputation["maximum_absolute_metric_difference"]) > 1e-10:
        raise ValueError("Saved metric recomputation mismatch exceeds tolerance")
    if recomputation.get("paired_intervals_contain_point_estimates") is not True:
        raise ValueError("A paired interval does not contain its point estimate")
    return ledgers


def _probe_datasets(validation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    concept_metrics: list[dict[str, Any]] = []
    lookahead_metrics: list[dict[str, Any]] = []
    concept_deltas: list[dict[str, Any]] = []
    lookahead_deltas: list[dict[str, Any]] = []

    concept_specs = (
        ("Piece code", "piece_code", "Accuracy"),
        ("Legal destination", "legal_destination", "AUROC"),
        ("Opponent attack count", "opponent_attack_count", "R²"),
    )
    lookahead_specs = (
        ("Third-ply destination", "lookahead_destination_accuracy"),
        (
            "Third-ply source | predicted destination",
            "lookahead_source_given_predicted_destination_accuracy",
        ),
        ("Third-ply joint move", "lookahead_joint_move_accuracy"),
    )

    for row in validation["layer_metrics"]:
        layer = int(row["layer"])
        for label, prefix, metric in concept_specs:
            for model in ("raw", "hero"):
                primary_key = f"{prefix}_{model}_primary"
                concept_metrics.append(
                    {
                        "layer": layer,
                        "model": "Raw BT4" if model == "raw" else "Hero",
                        "concept": label,
                        "metric": metric,
                        "score": float(row[primary_key]),
                    }
                )
            delta = float(row[f"{prefix}_delta"])
            significant = _interval_excludes_zero(row, prefix)
            concept_deltas.append(
                {
                    "layer": layer,
                    "concept": label,
                    "paired_metric": (
                        "accuracy"
                        if prefix in {"piece_code", "legal_destination"}
                        else "negative MAE"
                    ),
                    "hero_minus_raw": delta,
                    "ci_low": float(row[f"{prefix}_ci_low"]),
                    "ci_high": float(row[f"{prefix}_ci_high"]),
                    "interval_excludes_zero": significant,
                    "direction": _direction(delta, significant),
                }
            )

        for label, prefix in lookahead_specs:
            for model in ("raw", "hero"):
                lookahead_metrics.append(
                    {
                        "layer": layer,
                        "model": "Raw BT4" if model == "raw" else "Hero",
                        "target": label,
                        "accuracy": float(row[f"{prefix}_{model}"]),
                    }
                )
            delta = float(row[f"{prefix}_delta"])
            significant = _interval_excludes_zero(row, prefix)
            lookahead_deltas.append(
                {
                    "layer": layer,
                    "target": label,
                    "hero_minus_raw": delta,
                    "ci_low": float(row[f"{prefix}_ci_low"]),
                    "ci_high": float(row[f"{prefix}_ci_high"]),
                    "interval_excludes_zero": significant,
                    "direction": _direction(delta, significant),
                }
            )

    return {
        "concept_metrics": concept_metrics,
        "lookahead_metrics": lookahead_metrics,
        "concept_deltas": concept_deltas,
        "lookahead_deltas": lookahead_deltas,
    }


def _metric_extrema(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model in ("Raw BT4", "Hero"):
        selected = [row for row in rows if row["model"] == model]
        best = max(selected, key=lambda row: float(row[field]))
        result[model] = {"layer": int(best["layer"]), "value": float(best[field])}
    return result


def _build_model_diff(diff: dict[str, Any]) -> dict[str, Any]:
    arms = diff["behavior"]["arms"]
    pairs = diff["behavior"]["pairs"]
    resid = diff["activation"]["hooks"]["resid_post_after_ln"]["layers"]
    diagonal = diff["activation"]["layer_correspondence"]["best_hero_layer_for_each_raw_layer"]
    cka_rows = [
        {
            "layer": int(row["raw_layer"]),
            "corresponding_layer_cka": float(
                resid[str(row["raw_layer"])]["projected_multivariate"]["linear_cka"]
            ),
            "symmetric_relative_l2": float(
                resid[str(row["raw_layer"])]["exact_elementwise"]["symmetric_relative_l2"]
            ),
            "best_hero_layer": int(row["hero_layer"]),
            "best_layer_cka": float(row["linear_cka"]),
        }
        for row in diagonal
    ]
    return {
        "arms": [
            {
                "arm": arm,
                "target_nll": float(arms[arm]["target_nll"]["estimate"]),
                "top1_accuracy": float(arms[arm]["top1_accuracy"]["estimate"]),
            }
            for arm in ("RR", "RH", "HR", "HH")
        ],
        "swaps": [
            {
                "comparison": label,
                "js_divergence": float(pairs[key]["js_divergence"]["estimate"]),
                "top1_agreement": float(pairs[key]["top1_agreement"]["estimate"]),
            }
            for label, key in (
                ("Raw encoder → Hero encoder, raw head", "RR__HR"),
                ("Raw head → Hero head, raw encoder", "RR__RH"),
                ("Raw encoder → Hero encoder, Hero head", "RH__HH"),
                ("Raw head → Hero head, Hero encoder", "HR__HH"),
            )
        ],
        "activation_depth": cka_rows,
    }


def _build_literature(literature: dict[str, Any]) -> dict[str, Any]:
    patch = literature["causal_patching"]
    patching = []
    for direction, key in (
        ("Hero → Raw", "hero_to_raw_raw_head"),
        ("Raw → Hero", "raw_to_hero_raw_head"),
    ):
        for layer in (0, 7, 14):
            patching.append(
                {
                    "direction": direction,
                    "layer": layer,
                    "distribution_delta_projection": float(
                        patch[key][str(layer)]["distribution_delta_projection"]["mean"]
                    ),
                }
            )

    delta_rows = []
    for layer in (0, 7, 14):
        row = literature["paired_delta"][str(layer)]
        delta_rows.append(
            {
                "layer": layer,
                "mean_position_delta_rms": float(row["position_delta_rms"]["mean"]),
                "entropy_effective_rank": float(row["effective_rank_entropy"]),
                "top16_variance_share": float(row["top_cumulative_explained_variance"][15]),
            }
        )

    lens_rows = []
    for arm in ("RR", "RH", "HR", "HH"):
        for layer in (0, 7, 14):
            lens_rows.append(
                {
                    "arm": arm,
                    "layer": layer,
                    "target_probability": float(
                        literature["logit_lens"][arm]["metrics"]["target_probability"][str(layer)]
                    ),
                }
            )

    attribution = literature["attribution"]
    sarfa = attribution["raw_hero_map_agreement"]
    attention = literature["attention"]["raw_hero"]
    fixed_head_js = literature["behavior"]["raw_vs_hero_fixed_raw_head_policy_js"]
    return {
        "fixed_raw_head_policy_js": fixed_head_js,
        "patching": patching,
        "paired_delta": delta_rows,
        "logit_lens": lens_rows,
        "attention_pattern_js": [
            {
                "layer": layer,
                "pattern_js": float(attention[str(layer)]["pattern_js"]["global_mean"]),
            }
            for layer in (0, 7, 14)
        ],
        "attribution": {
            "sarfa_cosine": sarfa["sarfa_piece_removal.cosine"],
            "sarfa_pearson": sarfa["sarfa_piece_removal.pearson"],
            "sarfa_top8_jaccard": sarfa["sarfa_piece_removal.top_k_jaccard"],
            "integrated_gradients_relative_completeness_error": attribution[
                "integrated_gradients_relative_completeness_error"
            ],
        },
    }


def _build_evidence_matrix(
    validation: dict[str, Any],
    model_diff: dict[str, Any],
    literature: dict[str, Any],
    transcoder: dict[str, Any],
) -> list[dict[str, Any]]:
    del model_diff
    significant_policy = validation["causal_policy_intervals_excluding_zero"]
    fixed_js = literature["fixed_raw_head_policy_js"]
    sarfa = literature["attribution"]
    return [
        {
            "method": "2×2 encoder/head model diff",
            "scope": "128 development positions",
            "finding": "Encoder swaps changed policy strongly; head swaps were almost inert.",
            "evidence_rank": "strong exploratory localization",
            "main_limit": "One placeholder game cluster; bounded CKA sketch.",
        },
        {
            "method": "Bidirectional residual patching",
            "scope": "8 development positions; layers 0/7/14",
            "finding": "Final-layer patches reproduced essentially all fixed-head distribution delta.",
            "evidence_rank": "causal pilot",
            "main_limit": "Tiny position sample; whole-residual intervention.",
        },
        {
            "method": "Chess-concept probes",
            "scope": "15 layers; 512 development evaluations",
            "finding": "120/120 primary scores beat permutation and simple controls.",
            "evidence_rank": "validated decodability",
            "main_limit": "One probe seed; uncorrected layer scan; test unopened.",
        },
        {
            "method": "Third-ply lookahead probes",
            "scope": "15 layers; 512 public puzzle continuations",
            "finding": "Both models peak late; no systematic Hero advantage.",
            "evidence_rank": "validated exploratory comparison",
            "main_limit": "Puzzle continuation is not a fresh engine-PV benchmark.",
        },
        {
            "method": "Probe-direction causal steering",
            "scope": "16 development positions per model/layer",
            "finding": (
                f"Only {len(significant_policy)}/30 policy-derivative intervals excluded zero."
            ),
            "evidence_rank": "weak causal-use evidence",
            "main_limit": "Small n; learned direction; random controls failed 4/30 zero checks.",
        },
        {
            "method": "SARFA / integrated gradients",
            "scope": "2 SARFA examples; 4 IG arm-examples",
            "finding": (
                f"SARFA maps aligned (cosine {sarfa['sarfa_cosine']['mean']:.3f}); "
                f"IG completeness error was {sarfa['integrated_gradients_relative_completeness_error']['mean']:.2f}."
            ),
            "evidence_rank": "method diagnostic",
            "main_limit": "Attribution sample is too small; IG failed its own completeness check.",
        },
        {
            "method": "Published L14 transcoder transfer",
            "scope": "16 development positions",
            "finding": (
                "Source compatibility passed; Hero normalized MSE was "
                f"{transcoder['transfer_degradation']['normalized_mse_ratio_hero_over_raw']:.3f}× Raw."
            ),
            "evidence_rank": "compatibility pilot",
            "main_limit": "High dead-feature fraction; not a semantic feature audit.",
        },
        {
            "method": "Fixed-head policy comparison",
            "scope": "8 development positions",
            "finding": (
                f"Raw/Hero fixed-head JS was {fixed_js['mean']:.4f} "
                f"(95% interval {fixed_js['lower']:.4f}–{fixed_js['upper']:.4f})."
            ),
            "evidence_rank": "behavioral pilot",
            "main_limit": "Small, deliberately selected prefix.",
        },
    ]


def _build_cost_audit(
    *,
    metered_cost: Decimal,
    billed_cost: Decimal,
    pre_powered_run_cost: Decimal,
    monthly_budget: Decimal,
    observed_at: str,
) -> dict[str, Any]:
    incremental = metered_cost - pre_powered_run_cost
    return {
        "schema_version": "bt4-modal-cost-audit-v1",
        "observed_at": observed_at,
        "provider_query": "modal billing summary --for 'this month' --json",
        "workspace_monthly_budget_dollars": float(monthly_budget),
        "local_launch_stop_dollars": 34.0,
        "metered_cost_dollars": float(metered_cost),
        "billed_cost_after_credits_dollars": float(billed_cost),
        "metered_fraction_of_budget": float(metered_cost / monthly_budget),
        "pre_powered_run_metered_cost_dollars": float(pre_powered_run_cost),
        "powered_pipeline_incremental_metered_cost_dollars": float(incremental),
        "breakdown": {
            "ephemeral_apps_dollars": float(metered_cost),
            "volumes_dollars": 0.0,
            "credits_dollars": float(-metered_cost + billed_cost),
        },
        "pipeline": {
            "cpu_input_stage": {
                "network_download_bytes": 1_053_370_877,
                "full_sha256_files": 7,
                "declared_upper_bound_dollars": 0.073668,
            },
            "t4_all_layer_run": {
                "network_download_bytes": 0,
                "gpu_elapsed_seconds": 1629.4245435869998,
                "declared_upper_bound_dollars": 2.184408,
                "peak_allocated_bytes": 1_406_192_640,
            },
            "cpu_result_publish": {
                "result_bytes": 74_753_320,
                "declared_upper_bound_dollars": 0.021048,
            },
        },
        "reliability_audit": [
            {
                "event": "Invalid ephemeral-disk declaration rejected locally",
                "gpu_work": False,
                "resolution": "Removed unsupported override and retained provider default.",
            },
            {
                "event": "Two wrong logical raw-model paths",
                "gpu_work": True,
                "resolution": "Corrected to models/raw/BT4_exported.pb.gz; each attempt ended in under one second of model work.",
            },
            {
                "event": "Scientific smoke completed, provenance step lacked git binary",
                "gpu_work": True,
                "resolution": "Made git provenance explicitly optional in the lean remote image; 21.7-second attempt.",
            },
            {
                "event": "Successful smoke and powered all-layer run",
                "gpu_work": True,
                "resolution": "Both checksum-verified; CPU publisher uploaded immutable Railway bundles.",
            },
        ],
        "interpretation": (
            "Railway downloads and full input hashing ran on CPU-only Modal functions. "
            "The GPU job began after a verified Volume cache hit, reported zero network "
            "download bytes, and the result upload ran on a separate CPU-only function."
        ),
    }


def _chart(
    *,
    chart_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    y_field: str,
    y_label: str,
    color_field: str,
) -> dict[str, Any]:
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "type": "line",
        "dataset": dataset,
        "sourceId": "synthesis",
        "valueFormat": "number",
        "layout": "full",
        "encodings": {
            "x": {"field": "layer", "type": "ordinal", "label": "Layer"},
            "y": {"field": y_field, "type": "quantitative", "label": y_label},
            "color": {"field": color_field, "type": "nominal", "label": "Series"},
            "tooltip": [
                {"field": "layer", "type": "ordinal", "label": "Layer"},
                {"field": color_field, "type": "nominal", "label": "Series"},
                {"field": y_field, "type": "quantitative", "label": y_label},
            ],
        },
    }


def _artifact(
    *,
    generated_at: str,
    synthesis: dict[str, Any],
    cost_audit: dict[str, Any],
    report_dir: Path,
) -> dict[str, Any]:
    data = synthesis["datasets"]
    legal_rows = [row for row in data["concept_metrics"] if row["concept"] == "Legal destination"]
    attack_rows = [
        row for row in data["concept_metrics"] if row["concept"] == "Opponent attack count"
    ]
    piece_rows = [row for row in data["concept_metrics"] if row["concept"] == "Piece code"]
    destination_rows = [
        row for row in data["lookahead_metrics"] if row["target"] == "Third-ply destination"
    ]
    joint_rows = [
        row for row in data["lookahead_metrics"] if row["target"] == "Third-ply joint move"
    ]
    validation = synthesis["validation"]
    controls = validation["controls"]
    summary_row = {
        "evaluation_positions": validation["run"]["split_counts"]["evaluation"],
        "layers": len(validation["run"]["layers"]),
        "recomputed_metric_groups": validation["recomputation"][
            "metric_groups_recomputed_from_saved_predictions"
        ],
        "max_abs_recompute_error": validation["recomputation"][
            "maximum_absolute_metric_difference"
        ],
        "controls_passed": controls[
            "primary_beats_permutation_and_frequency_or_mean_control_count"
        ],
        "controls_total": controls["primary_control_comparison_count"],
    }
    cost_row = {
        "metered_cost_dollars": cost_audit["metered_cost_dollars"],
        "billed_cost_dollars": cost_audit["billed_cost_after_credits_dollars"],
        "budget_fraction": cost_audit["metered_fraction_of_budget"],
        "pipeline_incremental_cost_dollars": cost_audit[
            "powered_pipeline_incremental_metered_cost_dollars"
        ],
    }

    datasets = {
        "probe_summary": [summary_row],
        "cost_summary": [cost_row],
        "legal_metrics": legal_rows,
        "attack_metrics": attack_rows,
        "piece_metrics": piece_rows,
        "destination_metrics": destination_rows,
        "joint_metrics": joint_rows,
        "concept_deltas": data["concept_deltas"],
        "lookahead_deltas": data["lookahead_deltas"],
        "activation_depth": data["model_diff"]["activation_depth"],
        "patching": data["literature"]["patching"],
        "evidence_matrix": data["evidence_matrix"],
        "transcoder": data["transcoder_table"],
    }

    source_paths = {
        "synthesis": (report_dir / "synthesis.json").as_posix(),
        "validation": (report_dir / "validation.json").as_posix(),
        "cost": (report_dir / "cost_audit.json").as_posix(),
        "probe": (PROBE_RUN_DIR / "metrics.json").as_posix(),
        "model_diff": (DIFF_RUN_DIR / "metrics.json").as_posix(),
        "literature": (LITERATURE_RUN_DIR / "metrics.json").as_posix(),
        "transcoder": (TRANSCODER_RUN_DIR / "metrics.json").as_posix(),
    }
    source_labels = {
        "synthesis": "Audited synthesis and chart data",
        "validation": "Independent all-layer result validation",
        "cost": "Modal billing and execution cost audit",
        "probe": "All-layer probe/lookahead metrics",
        "model_diff": "Raw/Hero 2×2 model-diff metrics",
        "literature": "Literature-method replication metrics",
        "transcoder": "Published transcoder transfer metrics",
    }
    manifest_sources = [
        {"id": key, "label": source_labels[key], "path": source_paths[key]} for key in source_paths
    ]
    source_sql = {
        key: f"SELECT * FROM read_json_auto('{path}');" for key, path in source_paths.items()
    }
    sources = [
        {
            "id": key,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": source_labels[key],
                "sql": source_sql[key],
                "path": source_paths[key],
                **(
                    {"file_sha256": synthesis["source_files"][key]["sha256"]}
                    if key in synthesis["source_files"]
                    else {}
                ),
            },
        }
        for key in source_paths
    ]

    charts = [
        _chart(
            chart_id="legal_chart",
            title="Legal-destination information increases in most Hero layers",
            subtitle="Absolute linear-probe AUROC; paired intervals use threshold accuracy, not AUROC.",
            dataset="legal_metrics",
            y_field="score",
            y_label="AUROC",
            color_field="model",
        ),
        _chart(
            chart_id="attack_chart",
            title="Opponent-attack decoding changes are mixed across depth",
            subtitle="Absolute probe R²; paired inference is based on negative MAE.",
            dataset="attack_metrics",
            y_field="score",
            y_label="R²",
            color_field="model",
        ),
        _chart(
            chart_id="piece_chart",
            title="Piece identity is nearly perfect early and degrades late",
            subtitle="Absolute square-level piece-code accuracy.",
            dataset="piece_metrics",
            y_field="score",
            y_label="Accuracy",
            color_field="model",
        ),
        _chart(
            chart_id="destination_chart",
            title="Third-ply destination information peaks late in both models",
            subtitle="Public puzzle-continuation destination accuracy; test split remains unopened.",
            dataset="destination_metrics",
            y_field="accuracy",
            y_label="Accuracy",
            color_field="model",
        ),
        _chart(
            chart_id="joint_chart",
            title="Third-ply joint-move information is strongest near the output",
            subtitle="Hero is higher at layer 14, but not systematically across layers.",
            dataset="joint_metrics",
            y_field="accuracy",
            y_label="Accuracy",
            color_field="model",
        ),
        _chart(
            chart_id="cka_chart",
            title="Corresponding Raw/Hero representations diverge with depth",
            subtitle="Projected linear CKA on the 128-position model-diff pilot.",
            dataset="activation_depth",
            y_field="corresponding_layer_cka",
            y_label="Linear CKA",
            color_field="metric_label",
        ),
        _chart(
            chart_id="patch_chart",
            title="Whole-residual patches mediate the fixed-head policy delta by layer 14",
            subtitle="Distribution-delta projection; 1 means full alignment with the reference change.",
            dataset="patching",
            y_field="distribution_delta_projection",
            y_label="Delta projection",
            color_field="direction",
        ),
    ]
    for row in datasets["activation_depth"]:
        row["metric_label"] = "Corresponding-layer CKA"

    concept_table = {
        "id": "concept_delta_table",
        "title": "Layerwise concept differences",
        "subtitle": "Hero minus Raw; exploratory paired group-bootstrap 95% intervals.",
        "dataset": "concept_deltas",
        "sourceId": "validation",
        "defaultSort": {"field": "layer", "direction": "asc"},
        "columns": [
            {"field": "layer", "label": "Layer", "format": "number"},
            {"field": "concept", "label": "Concept", "type": "text"},
            {"field": "paired_metric", "label": "Paired metric", "type": "text"},
            {"field": "hero_minus_raw", "label": "Hero − Raw", "format": "number", "signed": True},
            {"field": "ci_low", "label": "95% low", "format": "number", "signed": True},
            {"field": "ci_high", "label": "95% high", "format": "number", "signed": True},
            {"field": "direction", "label": "Resolved direction", "type": "text"},
        ],
    }
    lookahead_table = {
        "id": "lookahead_delta_table",
        "title": "Layerwise lookahead differences",
        "subtitle": "Hero minus Raw accuracy; exploratory paired group-bootstrap 95% intervals.",
        "dataset": "lookahead_deltas",
        "sourceId": "validation",
        "defaultSort": {"field": "layer", "direction": "asc"},
        "columns": [
            {"field": "layer", "label": "Layer", "format": "number"},
            {"field": "target", "label": "Target", "type": "text"},
            {"field": "hero_minus_raw", "label": "Hero − Raw", "format": "number", "signed": True},
            {"field": "ci_low", "label": "95% low", "format": "number", "signed": True},
            {"field": "ci_high", "label": "95% high", "format": "number", "signed": True},
            {"field": "direction", "label": "Resolved direction", "type": "text"},
        ],
    }
    evidence_table = {
        "id": "evidence_table",
        "title": "Evidence ladder",
        "subtitle": "What each method establishes—and what it does not.",
        "dataset": "evidence_matrix",
        "sourceId": "synthesis",
        "defaultSort": {"field": "method", "direction": "asc"},
        "columns": [
            {"field": "method", "label": "Method", "type": "text"},
            {"field": "scope", "label": "Scope", "type": "text"},
            {"field": "finding", "label": "Finding", "type": "text"},
            {"field": "evidence_rank", "label": "Status", "type": "text"},
            {"field": "main_limit", "label": "Main limitation", "type": "text"},
        ],
    }
    transcoder_table = {
        "id": "transcoder_table",
        "title": "Published L14 transcoder transfer",
        "subtitle": "Same published artifact applied to Raw and Hero on 16 development positions.",
        "dataset": "transcoder",
        "sourceId": "transcoder",
        "defaultSort": {"field": "model", "direction": "asc"},
        "columns": [
            {"field": "model", "label": "Model", "type": "text"},
            {"field": "cosine", "label": "Reconstruction cosine", "format": "number"},
            {"field": "normalized_mse", "label": "Normalized MSE", "format": "number"},
            {"field": "policy_js", "label": "Policy JS", "format": "number"},
            {"field": "top1_agreement", "label": "Top-1 agreement", "format": "percent"},
        ],
    }

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Raw BT4 vs Hero: exploratory interpretability audit",
        },
        {
            "id": "status",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "**Decision:** share with caveats. The end-to-end tooling is working and independently "
                "recomputes every saved probe metric, but this is a nested-development study—not a "
                "confirmatory result. The frozen test partition remains unopened."
            ),
        },
        {
            "id": "metrics",
            "type": "metric-strip",
            "cardIds": ["run_card", "validation_card", "cost_card"],
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "synthesis",
            "body": (
                "## Technical summary\n\n"
                "The 2×2 encoder/head lattice localizes nearly all resolved policy change to the Hero "
                "encoder. Hero makes legal-destination information more accessible through most middle "
                "and late layers, while attack-count changes are mixed and late piece identity is often "
                "slightly worse. Both models contain late third-ply information; Hero does not show a "
                "systematic layerwise lookahead advantage. Probe directions reliably move their trained "
                "objectives, yet only one of 30 current-policy derivative intervals excludes zero. The "
                "right conclusion is representational redistribution with weak evidence of policy use—not "
                "evidence that Hero learned an internal search algorithm."
            ),
        },
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## Scope, data, and metric definitions\n\n"
                "The powered run fits probes on 1,024 training roots, selects hyperparameters on 256 "
                "training roots, and evaluates on 512 development roots across all 15 residual-post "
                "layers. The target is the third ply (zero-indexed future ply 2) from public puzzle "
                "continuations. Piece code uses accuracy; legal destination displays AUROC but paired "
                "inference uses threshold accuracy; opponent attack count displays R² but paired inference "
                "uses negative MAE. Every interval is a paired 2,000-replicate group bootstrap."
            ),
        },
        {
            "id": "model_diff_heading",
            "type": "markdown",
            "sourceId": "model_diff",
            "body": (
                "## Finding 1 — the encoder mediates the resolved behavioral change\n\n"
                "On the 128-position development pilot, RR/RH target NLL was 2.8776/2.8816 and both "
                "reached 25.0% top-1 accuracy; HR/HH reached 1.4288/1.4284 and 53.1%. Encoder swaps "
                "produced about 0.256 JS divergence with only 36.7–37.5% top-1 agreement, whereas head "
                "swaps produced 5.84e-6–1.40e-5 JS and 98.4–100% agreement. Corresponding-layer CKA "
                "falls from 0.819 to 0.236 while symmetric relative L2 rises from 0.531 to 1.144."
            ),
        },
        {"id": "cka", "type": "chart", "chartId": "cka_chart", "layout": "full"},
        {
            "id": "model_diff_limit",
            "type": "markdown",
            "sourceId": "model_diff",
            "body": (
                "This localizer is decisive enough to route experiments toward the encoder, but its "
                "uncertainty intervals are not confirmatory: all rows share a placeholder game ID and the "
                "64-wide CKA sketch is not convergence-certified."
            ),
        },
        {
            "id": "concept_heading",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## Finding 2 — Hero redistributes chess concepts rather than uniformly improving them\n\n"
                "Hero is higher on legal-destination paired accuracy at 13 of 15 layers, Raw is higher at "
                "layer 0, and layer 3 is unresolved. Attack-count differences resolve in Hero's direction "
                "at nine layers and Raw's at three, with three unresolved. Piece-code differences resolve "
                "only at layers 10, 12, 13, and 14; Hero is lower at three of those four, including a "
                "−0.0061 final-layer difference (95% interval −0.0119 to −0.0005)."
            ),
        },
        {"id": "legal", "type": "chart", "chartId": "legal_chart", "layout": "full"},
        {"id": "attack", "type": "chart", "chartId": "attack_chart", "layout": "full"},
        {"id": "piece", "type": "chart", "chartId": "piece_chart", "layout": "full"},
        {
            "id": "concept_table_block",
            "type": "table",
            "tableId": "concept_delta_table",
            "layout": "full",
        },
        {
            "id": "lookahead_heading",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## Finding 3 — both models encode future moves late; Hero has no systematic advantage\n\n"
                "Raw destination accuracy peaks at 0.713 on layer 13; Hero peaks at 0.703 on layer 12. "
                "Joint-move accuracy peaks at layer 13 for both (Raw 0.670; Hero 0.662). Only layer 6 "
                "resolves for destination accuracy, favoring Raw by 0.0234. At layer 14, Hero resolves a "
                "+0.0547 joint-move difference (95% interval +0.0137 to +0.0977). This final-layer "
                "difference is a localized redistribution, not a depth-wide gain."
            ),
        },
        {"id": "destination", "type": "chart", "chartId": "destination_chart", "layout": "full"},
        {"id": "joint", "type": "chart", "chartId": "joint_chart", "layout": "full"},
        {
            "id": "lookahead_table_block",
            "type": "table",
            "tableId": "lookahead_delta_table",
            "layout": "full",
        },
        {
            "id": "causal_heading",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## Causal-use audit\n\n"
                "All 30 learned lookahead directions have positive lower bounds for movement of their "
                "probe objective. That proves the interventions work in probe space. It does not prove "
                "policy use: only Raw layer 2 has a current-policy target-log-odds derivative interval "
                "excluding zero (mean 0.00252; 95% interval 0.000307–0.00558). Random-direction probe "
                "intervals contain zero in 26 of 30 comparisons, leaving four control failures that must "
                "be resolved before stronger claims."
            ),
        },
        {
            "id": "literature_heading",
            "type": "markdown",
            "sourceId": "literature",
            "body": (
                "## Literature-method replications and negative diagnostics\n\n"
                "Logit-lens target probability stays near 0.06 through layer 7 and rises to 0.867–0.872 "
                "at layer 14. Bidirectional residual patching progresses with depth and reaches essentially "
                "1.0 distribution-delta projection at layer 14. Raw/Hero attention-pattern JS is small "
                "(0.00176, 0.000504, 0.00606 at layers 0, 7, 14). Exact SARFA piece-removal maps align "
                "closely on two examples (cosine 0.953), but 16-step integrated gradients fails its "
                "completeness diagnostic with mean relative error 4.50; IG maps are therefore excluded "
                "from substantive interpretation."
            ),
        },
        {"id": "patching_chart", "type": "chart", "chartId": "patch_chart", "layout": "full"},
        {
            "id": "sparse_heading",
            "type": "markdown",
            "sourceId": "transcoder",
            "body": (
                "## Sparse-feature compatibility\n\n"
                "The published layer-14 transcoder passes its source-compatibility gate for both models. "
                "Hero reconstruction is marginally better (cosine 0.949 vs 0.948; normalized MSE 0.0992 "
                "vs 0.1013), and replacing the MLP output preserves top-1 policy choice on all 16 rows. "
                "This supports using the published artifact as a transfer baseline; it does not establish "
                "that its individual features retain the same chess semantics."
            ),
        },
        {
            "id": "transcoder_table_block",
            "type": "table",
            "tableId": "transcoder_table",
            "layout": "full",
        },
        {
            "id": "evidence_table_block",
            "type": "table",
            "tableId": "evidence_table",
            "layout": "full",
        },
        {
            "id": "cost_heading",
            "type": "markdown",
            "sourceId": "cost",
            "body": (
                "## Compute, cost, and reliability\n\n"
                "Inputs were downloaded and fully SHA-256 verified by a CPU-only Modal function into an "
                "immutable Volume. The T4 job started only after a cache/path preflight and reported zero "
                "network-download bytes; a separate CPU function published the result. Current settled "
                "usage is $0.46430113 metered and $0 billed after credits. The powered pipeline added "
                "$0.44357680, about 1.04% of the $42.50 workspace budget; total metered usage is 1.09%. "
                "No Volume charge has accrued."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## Limitations and robustness\n\n"
                "- Exploratory nested-development analysis; the frozen test partition remains unopened.\n"
                "- One training seed for learned probes; seed variance is unknown.\n"
                "- All-layer interval scans are uncorrected for multiple comparisons.\n"
                "- Linear decodability does not establish causal use or search.\n"
                "- Policy steering uses only 16 development positions per model and layer.\n"
                "- Third-ply labels are public puzzle continuations, not a fresh standardized engine-PV benchmark."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "synthesis",
            "body": (
                "## Recommended next experiments\n\n"
                "1. Freeze the present metrics and run a three-seed confirmatory probe/lookahead analysis "
                "on the unopened test partition with family-wise or FDR correction.\n"
                "2. Build a licensed, versioned Stockfish-PV benchmark with matched quiet/tactical roots, "
                "then repeat future-move decoding and causal steering. Public searchless-chess positions "
                "can supply roots; engine version, nodes, multipv, and label confidence must be stored.\n"
                "3. Replace whole-residual patching with head/MLP/path patching at layers 7–14 to localize "
                "the encoder-mediated delta.\n"
                "4. Audit transferred transcoder features semantically, then train a small paired delta "
                "crosscoder or LoRSA model only at the validated high-change layers.\n"
                "5. Spend in gates: roughly $2–$4 for confirmatory probes, $5–$10 for engine-labelled causal "
                "work, and reserve the remaining budget for a sparse model only after the causal gate."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "Does Hero's late legal-destination gain survive true game-disjoint test data? Is the final "
                "joint-move gain tied to particular tactical motifs or merely readout geometry? Which "
                "attention heads or MLP features carry the bidirectional patching effect? Do sparse "
                "features preserve semantics across the model pair, or does Hero rotate them into a new "
                "basis? These are now answerable with the retained tooling."
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Raw BT4 vs Hero: exploratory interpretability audit",
            "description": "A source-linked technical audit of model diffing, probes, causal tests, attribution, sparse transfer, and compute cost.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "run_card",
                    "description": "Powered development evaluation scope.",
                    "dataset": "probe_summary",
                    "sourceId": "validation",
                    "metrics": [
                        {
                            "label": "Evaluation roots",
                            "field": "evaluation_positions",
                            "format": "number",
                        },
                        {"label": "Layers", "field": "layers", "format": "number"},
                    ],
                },
                {
                    "id": "validation_card",
                    "description": "Independent recomputation and controls.",
                    "dataset": "probe_summary",
                    "sourceId": "validation",
                    "metrics": [
                        {
                            "label": "Metrics recomputed",
                            "field": "recomputed_metric_groups",
                            "format": "number",
                        },
                        {
                            "label": "Controls beaten",
                            "field": "controls_passed",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "cost_card",
                    "description": "Current Modal workspace usage after credits.",
                    "dataset": "cost_summary",
                    "sourceId": "cost",
                    "metrics": [
                        {
                            "label": "Modal metered",
                            "field": "metered_cost_dollars",
                            "format": "currency",
                        },
                        {"label": "Budget used", "field": "budget_fraction", "format": "percent"},
                    ],
                },
            ],
            "charts": charts,
            "tables": [concept_table, lookahead_table, transcoder_table, evidence_table],
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://raw-hero-interpretability-report-v1",
            "controls": {"edit": False, "refresh": False},
        },
    }


def build_report_payloads(
    *,
    report_dir: Path,
    metered_cost: Decimal,
    billed_cost: Decimal,
    pre_powered_run_cost: Decimal,
    monthly_budget: Decimal,
    generated_at: str,
) -> dict[str, Path]:
    ledgers = _verify_inputs(report_dir)
    validation_path = report_dir / "validation.json"
    validation = _load_json(validation_path)
    probe_metrics_path = PROBE_RUN_DIR / "metrics.json"
    diff_metrics_path = DIFF_RUN_DIR / "metrics.json"
    literature_metrics_path = LITERATURE_RUN_DIR / "metrics.json"
    transcoder_metrics_path = TRANSCODER_RUN_DIR / "metrics.json"
    _load_json(probe_metrics_path)
    diff = _load_json(diff_metrics_path)
    literature_raw = _load_json(literature_metrics_path)
    transcoder = _load_json(transcoder_metrics_path)

    expected_run = validation["run"]
    if expected_run["metrics_sha256"] != sha256_file(probe_metrics_path):
        raise ValueError("Validator metrics identity does not match powered run metrics")
    if expected_run["run_id"] != _load_json(PROBE_RUN_DIR / "manifest.json")["run_id"]:
        raise ValueError("Validator run identity does not match powered run manifest")

    generated_at_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if generated_at_dt.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    observed_at = generated_at_dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
    cost_audit = _build_cost_audit(
        metered_cost=metered_cost,
        billed_cost=billed_cost,
        pre_powered_run_cost=pre_powered_run_cost,
        monthly_budget=monthly_budget,
        observed_at=observed_at,
    )
    probe_data = _probe_datasets(validation)
    model_diff = _build_model_diff(diff)
    literature = _build_literature(literature_raw)
    evidence_matrix = _build_evidence_matrix(validation, model_diff, literature, transcoder)
    transcoder_table = [
        {
            "model": "Raw BT4",
            "cosine": float(transcoder["reconstruction"]["raw"]["cosine"]),
            "normalized_mse": float(transcoder["reconstruction"]["raw"]["normalized_mse"]),
            "policy_js": float(transcoder["policy_replacement"]["raw"]["js_divergence"]["mean"]),
            "top1_agreement": float(
                transcoder["policy_replacement"]["raw"]["top1_agreement"]["mean"]
            ),
        },
        {
            "model": "Hero",
            "cosine": float(transcoder["reconstruction"]["hero"]["cosine"]),
            "normalized_mse": float(transcoder["reconstruction"]["hero"]["normalized_mse"]),
            "policy_js": float(transcoder["policy_replacement"]["hero"]["js_divergence"]["mean"]),
            "top1_agreement": float(
                transcoder["policy_replacement"]["hero"]["top1_agreement"]["mean"]
            ),
        },
    ]
    source_files = {
        "report_builder": _source_record(REPORT_BUILDER_PATH),
        "validation": _source_record(validation_path),
        "probe": _source_record(probe_metrics_path),
        "model_diff": _source_record(diff_metrics_path),
        "literature": _source_record(literature_metrics_path),
        "transcoder": _source_record(transcoder_metrics_path),
    }
    synthesis = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": observed_at,
        "scientific_status": validation["overall_assessment"],
        "source_files": source_files,
        "verified_checksum_ledgers": {
            label: {"files": len(records)} for label, records in ledgers.items()
        },
        "validation": {
            "run": validation["run"],
            "integrity": validation["integrity"],
            "recomputation": validation["recomputation"],
            "controls": validation["controls"],
            "paired_interval_excludes_zero_layers": validation[
                "paired_interval_excludes_zero_layers"
            ],
            "causal_policy_intervals_excluding_zero": validation[
                "causal_policy_intervals_excluding_zero"
            ],
            "required_caveats": validation["required_caveats"],
        },
        "recommended_next_spend_dollars": {
            "confirmatory_probe": {"low": 2.0, "high": 4.0},
            "engine_labeled_causal": {"low": 5.0, "high": 10.0},
            "sparse_training": {"gate": "only after causal localization"},
        },
        "datasets": {
            **probe_data,
            "model_diff": model_diff,
            "literature": literature,
            "transcoder_table": transcoder_table,
            "evidence_matrix": evidence_matrix,
        },
        "extrema": {
            "legal_destination": _metric_extrema(
                [
                    row
                    for row in probe_data["concept_metrics"]
                    if row["concept"] == "Legal destination"
                ],
                "score",
            ),
            "opponent_attack_count": _metric_extrema(
                [
                    row
                    for row in probe_data["concept_metrics"]
                    if row["concept"] == "Opponent attack count"
                ],
                "score",
            ),
            "piece_code": _metric_extrema(
                [row for row in probe_data["concept_metrics"] if row["concept"] == "Piece code"],
                "score",
            ),
            "lookahead_destination": _metric_extrema(
                [
                    row
                    for row in probe_data["lookahead_metrics"]
                    if row["target"] == "Third-ply destination"
                ],
                "accuracy",
            ),
            "lookahead_joint": _metric_extrema(
                [
                    row
                    for row in probe_data["lookahead_metrics"]
                    if row["target"] == "Third-ply joint move"
                ],
                "accuracy",
            ),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    cost_path = report_dir / "cost_audit.json"
    synthesis_path = report_dir / "synthesis.json"
    chart_map_path = report_dir / "chart_map.json"
    artifact_path = report_dir / "artifact.json"
    write_json_atomic(cost_path, cost_audit)
    write_json_atomic(synthesis_path, synthesis)
    chart_map = {
        "schema_version": "bt4-interpretability-chart-map-v1",
        "charts": [
            {
                "chart_id": "cka_chart",
                "question": "Where do corresponding Raw and Hero representations diverge?",
                "dataset": "model_diff.activation_depth",
                "source": diff_metrics_path.as_posix(),
                "caveat": "128-position development pilot; projected CKA sketch.",
            },
            {
                "chart_id": "legal_chart",
                "question": "Where is legal-destination information linearly accessible?",
                "dataset": "concept_metrics filtered to Legal destination",
                "source": validation_path.as_posix(),
                "caveat": "Displays AUROC; paired intervals use threshold accuracy.",
            },
            {
                "chart_id": "attack_chart",
                "question": "Where is opponent attack count linearly decodable?",
                "dataset": "concept_metrics filtered to Opponent attack count",
                "source": validation_path.as_posix(),
                "caveat": "Displays R²; paired intervals use negative MAE.",
            },
            {
                "chart_id": "piece_chart",
                "question": "How is explicit piece identity retained across depth?",
                "dataset": "concept_metrics filtered to Piece code",
                "source": validation_path.as_posix(),
                "caveat": "Square-level accuracy; exploratory layer scan.",
            },
            {
                "chart_id": "destination_chart",
                "question": "Where is the public-continuation third-ply destination encoded?",
                "dataset": "lookahead_metrics filtered to destination",
                "source": validation_path.as_posix(),
                "caveat": "Public puzzle continuation, not fresh engine-PV labels.",
            },
            {
                "chart_id": "joint_chart",
                "question": "Where is the full third-ply move encoded?",
                "dataset": "lookahead_metrics filtered to joint move",
                "source": validation_path.as_posix(),
                "caveat": "One learned-probe seed; test split unopened.",
            },
            {
                "chart_id": "patch_chart",
                "question": "How much of the fixed-head policy delta is mediated by each residual boundary?",
                "dataset": "literature.patching",
                "source": literature_metrics_path.as_posix(),
                "caveat": "Eight development positions; whole-residual intervention.",
            },
        ],
    }
    write_json_atomic(chart_map_path, chart_map)
    artifact = _artifact(
        generated_at=observed_at,
        synthesis=synthesis,
        cost_audit=cost_audit,
        report_dir=report_dir,
    )
    write_json_atomic(artifact_path, artifact)
    return {
        "artifact": artifact_path,
        "chart_map": chart_map_path,
        "cost_audit": cost_path,
        "synthesis": synthesis_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--modal-metered-cost", type=Decimal, required=True)
    parser.add_argument("--modal-billed-cost", type=Decimal, required=True)
    parser.add_argument("--pre-powered-run-cost", type=Decimal, required=True)
    parser.add_argument("--monthly-budget", type=Decimal, default=Decimal("42.50"))
    parser.add_argument(
        "--generated-at",
        default=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        help="Timezone-aware ISO-8601 timestamp used for the report and billing observation.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    outputs = build_report_payloads(
        report_dir=args.report_dir,
        metered_cost=args.modal_metered_cost,
        billed_cost=args.modal_billed_cost,
        pre_powered_run_cost=args.pre_powered_run_cost,
        monthly_budget=args.monthly_budget,
        generated_at=args.generated_at,
    )
    print(json.dumps({key: path.as_posix() for key, path in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
