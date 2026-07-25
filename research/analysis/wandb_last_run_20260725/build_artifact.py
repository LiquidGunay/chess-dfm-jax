#!/usr/bin/env python3
"""Build the canonical Data Analytics artifact for the lineage report."""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
OUTPUT = HERE / "artifact.json"
TITLE = "May 2026 BT4/JEPA-DFM Training Reconstruction"

RUN_URLS = {
    "A": (
        "https://wandb.ai/gunays-independent/chess_dfm_jax-joint/runs/"
        "joint-h8-v5p4-b256-rawproj-officialsig1024-normloss-"
        "dclip0p5-sig0p1-tf20k-free-2full-20260503e"
    ),
    "B": (
        "https://wandb.ai/gunays-independent/chess_dfm_jax-joint/runs/"
        "jph8-b512-ln-sig005-lr3e4-init221432-8ep-v5e4-20260506a"
    ),
    "C": (
        "https://wandb.ai/gunays-independent/chess_dfm_jax-joint/runs/"
        "jph8-b512-ln-sig005-lr3e4-init221432-8ep-v5e64-euw4-20260506a"
    ),
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = DERIVED / name
    if not rows:
        path.write_text("")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return None if number is None else int(number)


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_definitions(generated_at: str) -> list[dict[str, Any]]:
    run_ids = [
        "wandb://gunays-independent/chess_dfm_jax-joint/"
        "joint-h8-v5p4-b256-rawproj-officialsig1024-normloss-"
        "dclip0p5-sig0p1-tf20k-free-2full-20260503e",
        "wandb://gunays-independent/chess_dfm_jax-joint/"
        "jph8-b512-ln-sig005-lr3e4-init221432-8ep-v5e4-20260506a",
        "wandb://gunays-independent/chess_dfm_jax-joint/"
        "jph8-b512-ln-sig005-lr3e4-init221432-8ep-v5e64-euw4-20260506a",
    ]
    sources = [
        {
            "id": "lineage_analysis",
            "label": "Reproducible W&B lineage analysis",
            "path": (
                "research/analysis/wandb_last_run_20260725/"
                "wandb_last_run_reconstruction.ipynb"
            ),
            "href": RUN_URLS["C"],
            "query": {
                "engine": "Python over bounded W&B API export",
                "id": "wandb-lineage-reconstruction-20260725",
                "url": RUN_URLS["C"],
                "description": (
                    "Joins three continuation runs by checkpoint ancestry and cumulative "
                    "examples, audits validation protocols, and fits post-change "
                    "saturation scenarios."
                ),
                "language": "python",
                "executed_at": generated_at,
                "tables_used": run_ids
                + [
                    "data/trajectory_v3/manifest.json",
                    "checkpoints/source/step0265000/metrics.jsonl",
                ],
                "filters": [
                    "Only the three identified May 2026 joint-training runs",
                    "Run-B steps above 10,000 excluded from the final-model lineage",
                    "Projection fits use run-C validation from step 180,000 onward",
                    "Stable-phase projections stop at step 255,000 before deterioration",
                ],
                "metric_definitions": [
                    (
                        "Cumulative epoch = inherited training examples divided by "
                        "28,343,296 training examples."
                    ),
                    (
                        "Validation DFM CE = the unweighted DFM cross-entropy component "
                        "reported by the training evaluator; it excludes JEPA and "
                        "legality terms."
                    ),
                    (
                        "Conditional epoch-100 range = spread across floor-plus-power "
                        "and floor-plus-exponential fits using post-SIGReg-change windows "
                        "ending at step 255,000; it is not a confidence interval."
                    ),
                ],
            },
        },
        {
            "id": "wandb_final_run",
            "label": "Final reused W&B run",
            "href": RUN_URLS["C"],
            "query": {
                "engine": "Weights & Biases",
                "id": (
                    "jph8-b512-ln-sig005-lr3e4-init221432-8ep-"
                    "v5e64-euw4-20260506a"
                ),
                "url": RUN_URLS["C"],
                "description": (
                    "Per-update and validation metrics for the reused final run ID."
                ),
                "executed_at": generated_at,
                "tables_used": [run_ids[2]],
                "filters": [
                    "Steps 1 through 267,158",
                    "Fixed allowlist of optimization, validation, and topology metrics",
                ],
                "metric_definitions": [
                    "Target SIGReg is the raw jepa_sigreg_loss before its coefficient.",
                    (
                        "Global batch = data_parallel_devices multiplied by "
                        "data_parallel_per_device_batch_size."
                    ),
                ],
            },
        },
        {
            "id": "training_code_audit",
            "label": "Training code and data-manifest audit",
            "path": "scripts/train_joint_latent_sasa.py",
            "query": {
                "engine": "Repository source audit",
                "id": "joint-trainer-audit-20260725",
                "description": (
                    "Reviews checkpoint restore semantics, process-local validation, "
                    "batch construction, and checkpoint cadence."
                ),
                "language": "python",
                "executed_at": generated_at,
                "tables_used": [
                    "scripts/train_joint_latent_sasa.py",
                    "data/trajectory_v3/manifest.json",
                ],
                "filters": [
                    "Checkpoint initialization and resume paths",
                    "Validation loader and evaluate_validation_batches",
                    "W&B logging and save cadence",
                ],
                "metric_definitions": [
                    (
                        "Model-only continuation loads optimizer=None and starts a fresh "
                        "optimizer at step zero."
                    ),
                    (
                        "Validation examples = val_batches multiplied by "
                        "process_batch_size on process 0."
                    ),
                ],
            },
        },
    ]
    derived_sources = [
        (
            "headline_source",
            "Headline lineage metrics",
            "report_headline.csv",
            "Loads the exact headline values shown in the metric strip.",
        ),
        (
            "validation_trajectory_source",
            "Observed validation and projection curves",
            "report_validation_trajectory.csv",
            "Loads observed validation points and the two reviewed projection curves.",
        ),
        (
            "lineage_source",
            "Checkpoint-lineage segment table",
            "report_lineage_segments.csv",
            "Loads the reconstructed inherited-exposure table in lineage order.",
        ),
        (
            "projection_source",
            "Projection sensitivity table",
            "report_projection_fits.csv",
            "Loads fit windows, errors, holdout diagnostics, and epoch-100 scenarios.",
        ),
        (
            "sigreg_source",
            "Late-run target SIGReg series",
            "report_sigreg_tail.csv",
            "Loads the sampled raw target-SIGReg series near the W&B endpoint.",
        ),
        (
            "protocol_source",
            "Validation-protocol audit table",
            "report_validation_protocols.csv",
            "Loads the process-local validation population and sample size by segment.",
        ),
        (
            "quality_source",
            "Tracking and data-quality findings",
            "report_quality_issues.csv",
            "Loads the severity-ranked audit findings and remediations.",
        ),
    ]
    for source_id, label, filename, description in derived_sources:
        relative_path = (
            "research/analysis/wandb_last_run_20260725/derived/" + filename
        )
        sources.append(
            {
                "id": source_id,
                "label": label,
                "path": relative_path,
                "query": {
                    "engine": "DuckDB",
                    "id": f"{source_id}-20260725",
                    "sql": (
                        "SELECT *\n"
                        f"FROM read_csv_auto('{relative_path}', header = true);"
                    ),
                    "description": description,
                    "language": "sql",
                    "executed_at": generated_at,
                    "tables_used": [relative_path],
                    "filters": ["No additional row filters; the CSV is a reviewed bounded output"],
                    "metric_definitions": [
                        (
                            "Rows and fields are produced by "
                            "research/analysis/wandb_last_run_20260725/"
                            "analyze_lineage.py and build_artifact.py."
                        )
                    ],
                },
            }
        )
    return sources


def build() -> dict[str, Any]:
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    summary = json.loads((DERIVED / "summary.json").read_text())
    quality = json.loads((DERIVED / "data_quality.json").read_text())
    validation = read_csv("validation_points.csv")
    projection_curves = read_csv("projection_curves.csv")
    projection_fits = read_csv("projection_fits.csv")
    lineage = read_csv("lineage_segments.csv")
    tail = read_csv("final_run_tail_training.csv")

    headline = [
        {
            "retained_epochs": summary["retained_lineage_epochs"],
            "wandb_endpoint_epochs": summary["wandb_endpoint_lineage_epochs"],
            "best_val_dfm_ce": summary["best_comparable_validation"]["val_dfm_ce_loss"],
            "retained_val_dfm_ce": summary["retained_checkpoint_validation"][
                "val_dfm_ce_loss"
            ],
            "retained_minus_best": (
                summary["retained_checkpoint_validation"]["val_dfm_ce_loss"]
                - summary["best_comparable_validation"]["val_dfm_ce_loss"]
            ),
            "projection_midpoint": (
                summary["conditional_epoch_100_projection"]["range_min"]
                + summary["conditional_epoch_100_projection"]["range_max"]
            )
            / 2,
            "projection_low": summary["conditional_epoch_100_projection"]["range_min"],
            "projection_high": summary["conditional_epoch_100_projection"]["range_max"],
        }
    ]

    validation_trajectory: list[dict[str, Any]] = []
    run_names = {
        "initial_b256": "Run A observed",
        "continuation_b512": "Run B observed",
        "final_reused_run": "Run C observed",
    }
    for row in validation:
        series = run_names[row["source_run"]]
        if row["lineage_status"].startswith("side branch"):
            series = "Run B side branch"
        validation_trajectory.append(
            {
                "epoch": as_float(row["cumulative_epochs"]),
                "val_dfm_ce_loss": as_float(row["val_dfm_ce_loss"]),
                "series": series,
                "kind": "observed",
                "source_step": as_int(row["source_step"]),
                "validation_examples": as_int(row["validation_examples"]),
                "validation_protocol": row["validation_protocol"],
            }
        )
    projection_names = {
        "exponential to floor": "Exponential-to-floor projection",
        "power law to floor": "Power-law-to-floor projection",
    }
    for row in projection_curves:
        validation_trajectory.append(
            {
                "epoch": as_float(row["epoch"]),
                "val_dfm_ce_loss": as_float(row["val_dfm_ce_loss"]),
                "series": projection_names[row["series"]],
                "kind": "conditional projection",
                "source_step": None,
                "validation_examples": None,
                "validation_protocol": (
                    "Fit to post-change run-C validation through step 255,000"
                ),
            }
        )

    lineage_rows: list[dict[str, Any]] = []
    for order, row in enumerate(lineage, start=1):
        lineage_rows.append(
            {
                "order": order,
                "segment": row["segment"],
                "inherited_updates": as_int(row["inherited_updates"]),
                "global_batch": as_int(row["global_batch"]),
                "examples": as_int(row["examples_in_final_lineage"]),
                "epochs": as_float(row["epochs_in_final_lineage"]),
                "optimizer_start": row["optimizer_start"],
                "sigreg_coefficient": row["jepa_sigreg_coefficient"],
                "notes": row["notes"],
            }
        )

    protocol_rows = [
        {
            "order": 1,
            "segment": "Run A",
            "validation_pool": "Full 1,524-shard pool",
            "process_batch": 256,
            "val_batches": 32,
            "validation_examples": 8_192,
            "comparable_fit_use": "No — different population/size",
        },
        {
            "order": 2,
            "segment": "Run B",
            "validation_pool": "Full 1,524-shard pool",
            "process_batch": 512,
            "val_batches": 32,
            "validation_examples": 16_384,
            "comparable_fit_use": "No — different population",
        },
        {
            "order": 3,
            "segment": "Run C, steps 5k/10k",
            "validation_pool": "Process-0 96-shard slice",
            "process_batch": 32,
            "val_batches": 32,
            "validation_examples": 1_024,
            "comparable_fit_use": "No — undersampled protocol",
        },
        {
            "order": 4,
            "segment": "Run C, step 15k onward",
            "validation_pool": "Process-0 96-shard slice",
            "process_batch": 512,
            "val_batches": 32,
            "validation_examples": 16_384,
            "comparable_fit_use": "Yes, within this segment only",
        },
    ]

    fit_rows: list[dict[str, Any]] = []
    for row in projection_fits:
        fit_rows.append(
            {
                "scenario": row["scenario"],
                "model": row["model"].replace("_", " "),
                "start_step": as_int(row["start_step"]),
                "cutoff_step": as_int(row["cutoff_step"]),
                "fit_points": as_int(row["fit_points"]),
                "rmse": as_float(row["rmse"]),
                "holdout_mae": as_float(row.get("holdout_mae")),
                "epoch_100_projection": as_float(
                    row["projected_val_dfm_ce_at_epoch_100"]
                ),
            }
        )

    sigreg_tail: list[dict[str, Any]] = []
    for row in tail:
        step = as_int(row["step"])
        sigreg = as_float(row["jepa_sigreg_loss"])
        if step is not None and sigreg is not None and (
            step % 1_000 == 0 or step == 267_158
        ):
            sigreg_tail.append({"step": step, "target_sigreg": sigreg})

    issue_rows: list[dict[str, Any]] = []
    severity_rank = {"high": 1, "medium": 2, "low": 3}
    for issue in quality["issues"]:
        issue_rows.append(
            {
                "severity_rank": severity_rank[issue["severity"]],
                "severity": issue["severity"].title(),
                "confidence": issue["confidence"].title(),
                "issue": issue["issue"],
                "impact": issue["impact"],
                "remediation": issue["remediation"],
            }
        )

    write_csv("report_headline.csv", headline)
    write_csv("report_validation_trajectory.csv", validation_trajectory)
    write_csv("report_lineage_segments.csv", lineage_rows)
    write_csv("report_validation_protocols.csv", protocol_rows)
    write_csv("report_projection_fits.csv", fit_rows)
    write_csv("report_sigreg_tail.csv", sigreg_tail)
    write_csv("report_quality_issues.csv", issue_rows)

    sources = source_definitions(generated_at)
    cards = [
        {
            "id": "epochs_card",
            "description": (
                "Training-set-equivalent exposure inherited by the retained step-265k "
                "checkpoint."
            ),
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {"label": "Retained lineage epochs", "field": "retained_epochs", "format": "number"},
                {"label": "W&B endpoint epochs", "field": "wandb_endpoint_epochs", "format": "number"},
            ],
        },
        {
            "id": "best_val_card",
            "description": (
                "Best comparable run-C validation DFM CE versus the retained checkpoint."
            ),
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {"label": "Best validation DFM CE", "field": "best_val_dfm_ce", "format": "number"},
                {
                    "label": "Retained minus best",
                    "field": "retained_minus_best",
                    "format": "number",
                    "signed": True,
                },
            ],
        },
        {
            "id": "projection_card",
            "description": (
                "Midpoint and model-spread range under a stable post-change saturation "
                "counterfactual."
            ),
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Conditional CE at epoch 100",
                    "field": "projection_midpoint",
                    "format": "number",
                },
                {"label": "Low model estimate", "field": "projection_low", "format": "number"},
                {"label": "High model estimate", "field": "projection_high", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "validation_projection_chart",
            "title": "Validation DFM cross-entropy over cumulative epochs",
            "subtitle": (
                "The stable-phase fits approach 4.41–4.45 at epoch 100, but the "
                "observed run deteriorates after step 255k."
            ),
            "type": "line",
            "dataset": "validation_trajectory",
            "sourceId": "validation_trajectory_source",
            "encodings": {
                "x": {
                    "field": "epoch",
                    "type": "quantitative",
                    "label": "Cumulative epochs",
                },
                "y": {
                    "field": "val_dfm_ce_loss",
                    "type": "quantitative",
                    "label": "Validation DFM CE",
                    "format": "number",
                },
                "color": {"field": "series", "type": "nominal", "label": "Series"},
                "tooltip": [
                    {"field": "source_step", "type": "quantitative", "label": "Source step"},
                    {
                        "field": "validation_examples",
                        "type": "quantitative",
                        "label": "Validation examples",
                    },
                    {
                        "field": "validation_protocol",
                        "type": "text",
                        "label": "Protocol",
                    },
                ],
            },
            "xAxisTitle": "Cumulative full-dataset epochs",
            "yAxisTitle": "Validation DFM cross-entropy",
            "valueFormat": "number",
        },
        {
            "id": "sigreg_tail_chart",
            "title": "Training target SIGReg near the W&B endpoint",
            "subtitle": (
                "The raw target SIGReg remains near 40 before spiking above 180 at "
                "the crashed endpoint."
            ),
            "type": "line",
            "dataset": "sigreg_tail",
            "sourceId": "sigreg_source",
            "encodings": {
                "x": {"field": "step", "type": "quantitative", "label": "Run-C step"},
                "y": {
                    "field": "target_sigreg",
                    "type": "quantitative",
                    "label": "Raw target SIGReg",
                    "format": "number",
                },
            },
            "xAxisTitle": "Run-C step",
            "yAxisTitle": "Raw target SIGReg",
            "valueFormat": "number",
        },
    ]

    tables = [
        {
            "id": "lineage_table",
            "title": "Checkpoint lineage and inherited exposure",
            "subtitle": (
                "The retained model inherits only the first 10k updates of run B."
            ),
            "dataset": "lineage_segments",
            "sourceId": "lineage_source",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "#", "type": "number"},
                {"field": "segment", "label": "Segment", "type": "text"},
                {"field": "global_batch", "label": "Global batch", "format": "number"},
                {"field": "examples", "label": "Inherited examples", "format": "compact"},
                {"field": "epochs", "label": "Inherited epochs", "format": "number"},
                {"field": "optimizer_start", "label": "Optimizer state", "type": "text"},
                {"field": "sigreg_coefficient", "label": "JEPA SIGReg coeff.", "type": "text"},
                {"field": "notes", "label": "Notes", "type": "text"},
            ],
        },
        {
            "id": "projection_table",
            "title": "Projection sensitivity",
            "subtitle": (
                "Scenario spread across model family, fit window, and inclusion of "
                "the deteriorating tail."
            ),
            "dataset": "projection_fits",
            "sourceId": "projection_source",
            "defaultSort": {"field": "epoch_100_projection", "direction": "asc"},
            "columns": [
                {"field": "scenario", "label": "Scenario", "type": "text"},
                {"field": "model", "label": "Curve", "type": "text"},
                {"field": "start_step", "label": "Fit start", "format": "number"},
                {"field": "cutoff_step", "label": "Fit cutoff", "format": "number"},
                {"field": "fit_points", "label": "Points", "format": "number"},
                {"field": "rmse", "label": "In-sample RMSE", "format": "number"},
                {"field": "holdout_mae", "label": "Holdout MAE", "format": "number"},
                {
                    "field": "epoch_100_projection",
                    "label": "Projected CE @ 100 epochs",
                    "format": "number",
                },
            ],
        },
        {
            "id": "protocol_table",
            "title": "Validation protocol by training segment",
            "subtitle": (
                "Only run C from step 15k onward uses one internally stable protocol."
            ),
            "dataset": "validation_protocols",
            "sourceId": "protocol_source",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "#", "type": "number"},
                {"field": "segment", "label": "Segment", "type": "text"},
                {"field": "validation_pool", "label": "Visible pool", "type": "text"},
                {"field": "process_batch", "label": "Process batch", "format": "number"},
                {"field": "val_batches", "label": "Val batches", "format": "number"},
                {
                    "field": "validation_examples",
                    "label": "Evaluated examples",
                    "format": "number",
                },
                {"field": "comparable_fit_use", "label": "Fit use", "type": "text"},
            ],
        },
        {
            "id": "quality_table",
            "title": "Material tracking and data-quality issues",
            "subtitle": "Issues are ordered by decision impact.",
            "dataset": "quality_issues",
            "sourceId": "quality_source",
            "defaultSort": {"field": "severity_rank", "direction": "asc"},
            "columns": [
                {"field": "severity_rank", "label": "Rank", "type": "number"},
                {"field": "severity", "label": "Severity", "type": "text"},
                {"field": "confidence", "label": "Confidence", "type": "text"},
                {"field": "issue", "label": "Issue", "type": "text"},
                {"field": "impact", "label": "Impact", "type": "text"},
                {"field": "remediation", "label": "Clean-run fix", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "lineage_analysis",
            "body": (
                "## Technical summary\n\n"
                "The retained step-265k checkpoint is best understood as a **76.06-epoch "
                "model**, not an eight-epoch run. It inherited 2.00 epochs from the "
                "initial run, 0.181 epochs from the first 10k steps of the intermediate "
                "continuation, and 73.88 epochs from the final reused run. Its best "
                "comparable validation DFM CE was **4.503 at step 255k (73.17 epochs)**; "
                "the retained checkpoint had worsened to **4.541**. Stable-phase "
                "saturation fits imply **4.41–4.45 at 100 epochs**, conditional on "
                "removing the late instability. The literal crashed run does not support "
                "that continuation assumption."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": ["epochs_card", "best_val_card", "projection_card"],
        },
        {
            "id": "lineage_finding",
            "type": "markdown",
            "sourceId": "lineage_analysis",
            "body": (
                "## The checkpoint lineage contains three optimizer phases\n\n"
                "Run A started fresh. Run B loaded only model weights from run A, so "
                "its optimizer and step counter restarted. Run C likewise loaded only "
                "the step-10k model from run B before later full-state resumes. Run B's "
                "steps 10,001–15,176 are a side branch: they consumed compute but are "
                "not ancestors of the retained model. The large-batch run-C restart "
                "most likely resumed from step 10k, giving the 76.06-epoch estimate; "
                "treating W&B's displayed geometry literally gives 75.87 epochs."
            ),
        },
        {"id": "lineage_detail", "type": "table", "tableId": "lineage_table"},
        {
            "id": "trajectory_finding",
            "type": "markdown",
            "sourceId": "lineage_analysis",
            "body": (
                "## Loss improved for most of the run, then reversed\n\n"
                "Validation DFM CE fell smoothly through step 255k. The step-260k and "
                "step-265k validations then worsened, and the run-C training signal "
                "destabilized further by step 267,158. The retained checkpoint is "
                "therefore about 10k updates and 2.89 epochs past the best observed "
                "validation point."
            ),
        },
        {
            "id": "validation_projection",
            "type": "chart",
            "chartId": "validation_projection_chart",
        },
        {
            "id": "projection_method",
            "type": "markdown",
            "sourceId": "lineage_analysis",
            "body": (
                "## The 100-epoch estimate is a counterfactual range\n\n"
                "Two saturating hypotheses—a floor-plus-exponential and a "
                "floor-plus-power law—were fit to post-SIGReg-change validation points "
                "ending at step 255k. Across two start windows they project **4.414 to "
                "4.451** at 100 cumulative epochs. Fits that include the deteriorating "
                "step-260k/265k tail project **4.466 to 4.499**. A prefix fit through "
                "step 230k predicts steps 235k–255k with 0.004–0.005 MAE, but that "
                "backtest cannot anticipate the later instability."
            ),
        },
        {"id": "projection_sensitivity", "type": "table", "tableId": "projection_table"},
        {
            "id": "instability_finding",
            "type": "markdown",
            "sourceId": "wandb_final_run",
            "body": (
                "## The W&B endpoint is not a usable projection anchor\n\n"
                "Raw target SIGReg was roughly 40 near the best-validation region and "
                "reached **188.7 at step 267,158**. Training DFM CE also rose to 3.765 "
                "at the endpoint. Because no validation was logged after step 265k and "
                "the run state is crashed, forecasting the unchanged optimizer to epoch "
                "100 would hide the central observed fact: this trajectory was already "
                "unstable."
            ),
        },
        {"id": "sigreg_tail", "type": "chart", "chartId": "sigreg_tail_chart"},
        {
            "id": "scope_methodology",
            "type": "markdown",
            "sourceId": "lineage_analysis",
            "body": (
                "## Scope, data, and methodology\n\n"
                "The analysis uses 503,762 exported per-update W&B rows across the "
                "three identified runs, 102 validation snapshots, the 28,343,296-example "
                "training manifest, the retained checkpoint's local metrics, and the "
                "trainer's restore/evaluation code. Runs are joined by parent checkpoint "
                "rather than displayed W&B step. Cumulative examples use each inherited "
                "segment's global batch. The nine local validation snapshots at steps "
                "225k–265k match W&B exactly on all five checked metrics."
            ),
        },
        {"id": "validation_protocols", "type": "table", "tableId": "protocol_table"},
        {
            "id": "quality_findings",
            "type": "markdown",
            "sourceId": "lineage_analysis",
            "body": (
                "## Tracking inconsistencies are material, but reconstructable\n\n"
                "The final W&B ID was reused across a 512→8,192 global-batch change and "
                "a JEPA SIGReg coefficient change from 0.05 to 0.01 at logged step "
                "176,187. Checkpoint rollback also caused duplicate retraining that W&B "
                "rejected as out-of-order steps. These issues make the displayed config, "
                "step, and weighted loss insufficient for lineage analysis without the "
                "raw components and code audit."
            ),
        },
        {"id": "quality_detail", "type": "table", "tableId": "quality_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "lineage_analysis",
            "body": (
                "## Limitations and robustness\n\n"
                "The 76.06-epoch count assumes the 8,192-batch restart rolled back to "
                "the periodic step-10k checkpoint; wall time, checkpoint cadence, and "
                "the first accepted large-batch log support this, but the overwritten "
                "run log prevents direct proof. The literal W&B-geometry alternative is "
                "75.87 epochs, only 0.20 epochs lower. Cross-run validation values are "
                "not directly comparable because the evaluated population changed. "
                "Curve-model spread is sensitivity analysis, not statistical coverage, "
                "and DFM CE has not yet been calibrated to Elo."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "training_code_audit",
            "body": (
                "## Recommended clean-run prerequisites\n\n"
                "1. Freeze a topology-invariant validation index and aggregate "
                "example-weighted metrics across processes.\n"
                "2. Give every immutable recipe segment its own run ID; record "
                "parent checkpoint, optimizer restore status, objective version, global "
                "batch, examples seen, and retry count.\n"
                "3. Select and retain checkpoints by comparable validation DFM CE plus "
                "collapse diagnostics, not final step.\n"
                "4. After the loss/systems experiments settle, run fixed-checkpoint Elo "
                "evaluations throughout one clean trajectory to estimate the empirical "
                "loss↔Elo relationship."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Did the late instability originate in the target projector, optimizer "
                "state, batch transition, or loss scaling?\n"
                "- Does step-255k outperform step-265k in Elo, matching the DFM-CE reversal?\n"
                "- Which validation horizon is most predictive of searchless Elo?\n"
                "- Once evaluation is fixed, does a 30-minute ranking preserve the ordering "
                "of longer clean runs?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "Reconstructed checkpoint lineage, data-quality audit, and bounded "
                "100-epoch loss scenarios for the May 2026 joint BT4/JEPA-DFM training."
            ),
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "validation_trajectory": validation_trajectory,
                "lineage_segments": lineage_rows,
                "validation_protocols": protocol_rows,
                "projection_fits": fit_rows,
                "sigreg_tail": sigreg_tail,
                "quality_issues": issue_rows,
            },
        },
        "sources": sources,
        "package_info": {
            "origin": "chess-dfm-jax/research/analysis/wandb_last_run_20260725",
            "analysisVersion": 1,
        },
    }
    return clean(artifact)


def main() -> None:
    artifact = build()
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    payload_bytes = len(json.dumps(artifact["snapshot"], separators=(",", ":")))
    rows = {
        key: len(value) for key, value in artifact["snapshot"]["datasets"].items()
    }
    print(f"wrote {OUTPUT}")
    print(f"snapshot_bytes={payload_bytes} dataset_rows={rows}")


if __name__ == "__main__":
    main()
