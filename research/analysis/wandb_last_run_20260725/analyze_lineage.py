#!/usr/bin/env python3
"""Reconstruct the May 2026 joint-training lineage and fit bounded projections.

This module is intentionally offline: run fetch_wandb_lineage.py once, then this
script reads only the bounded export plus the repository's data manifest and
retained checkpoint metrics.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", "/mountpoint/.exp/chess-dfm-jax/.local/cache/matplotlib"
)
os.environ.setdefault("TMPDIR", "/mountpoint/.exp/chess-dfm-jax/.local/tmp")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RAW_DIR = HERE / "raw"
DERIVED_DIR = HERE / "derived"
FIGURES_DIR = HERE / "figures"
DATASET_MANIFEST = REPO_ROOT / "data" / "trajectory_v3" / "manifest.json"
LOCAL_CHECKPOINT_METRICS = (
    REPO_ROOT / "checkpoints" / "source" / "step0265000" / "metrics.jsonl"
)

RUN_LABELS = ("initial_b256", "continuation_b512", "final_reused_run")
ANCESTOR_B_STEP = 10_000
RETAINED_C_STEP = 265_000
WANDB_C_ENDPOINT = 267_158
INFERRED_C_LARGE_BATCH_START = 10_001

VAL_KEYS = [
    "val_loss",
    "val_dfm_ce_loss",
    *[f"val_dfm_ce_loss_by_horizon_h{i}" for i in range(1, 9)],
    "val_accuracy",
    "val_first_legal_mass",
    "val_first_legality_loss",
    "val_jepa_positive_loss",
    "val_jepa_raw_mse",
    "val_jepa_norm_loss",
    "val_jepa_sigreg_loss",
    "val_jepa_pred_sigreg_loss",
]


@dataclass
class HistoryAudit:
    label: str
    row_count: int
    unique_steps: int
    min_step: int
    max_step: int
    missing_steps: list[int]
    duplicate_step_count: int
    validation_points: int
    first_timestamp: float
    last_timestamp: float
    last_runtime: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def history_rows(label: str) -> Iterable[dict[str, Any]]:
    path = RAW_DIR / label / "history.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def close_to(value: float, target: float, tolerance: float = 2e-3) -> bool:
    return math.isfinite(value) and abs(value - target) <= tolerance


def inferred_sigreg_coefficient(row: dict[str, Any]) -> float | None:
    values = [
        row.get("loss"),
        row.get("dfm_ce_loss"),
        row.get("first_legality_loss"),
        row.get("jepa_positive_loss"),
        row.get("jepa_sigreg_loss"),
    ]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    loss, dfm_ce, first_legality, jepa_positive, sigreg = map(float, values)
    if not math.isfinite(sigreg) or abs(sigreg) < 1e-9:
        return None
    return (loss - dfm_ce - 2.0 * first_legality - jepa_positive) / sigreg


def audit_history(label: str) -> tuple[HistoryAudit, list[dict[str, Any]]]:
    validation: list[dict[str, Any]] = []
    seen: set[int] = set()
    row_count = 0
    duplicate_count = 0
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None

    for row in history_rows(label):
        row_count += 1
        first = row if first is None else first
        last = row
        step = int(row["step"])
        if step in seen:
            duplicate_count += 1
        seen.add(step)
        if row.get("val_dfm_ce_loss") is not None:
            validation.append(
                {
                    "source_run": label,
                    "source_step": step,
                    "timestamp": float(row["_timestamp"]),
                    **{key: row.get(key) for key in VAL_KEYS},
                }
            )

    if first is None or last is None:
        raise ValueError(f"No history rows found for {label}")
    min_step = min(seen)
    max_step = max(seen)
    missing = sorted(set(range(min_step, max_step + 1)) - seen)
    return (
        HistoryAudit(
            label=label,
            row_count=row_count,
            unique_steps=len(seen),
            min_step=min_step,
            max_step=max_step,
            missing_steps=missing,
            duplicate_step_count=duplicate_count,
            validation_points=len(validation),
            first_timestamp=float(first["_timestamp"]),
            last_timestamp=float(last["_timestamp"]),
            last_runtime=float(last["_runtime"]),
        ),
        validation,
    )


def inspect_final_run() -> dict[str, Any]:
    geometry_segments: list[dict[str, Any]] = []
    interruptions: list[dict[str, Any]] = []
    coefficient_changes: list[dict[str, Any]] = []
    tail_training: list[dict[str, Any]] = []

    previous: dict[str, Any] | None = None
    segment_start: dict[str, Any] | None = None
    segment_geometry: tuple[int, int, int, int, int] | None = None
    last_stable_coefficient: float | None = None

    def geometry(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            int(row.get("data_parallel_devices") or 0),
            int(row.get("data_parallel_local_devices") or 0),
            int(row.get("data_parallel_per_device_batch_size") or 0),
            int(row.get("process_batch_size") or 0),
            int(row.get("process_count") or 0),
        )

    def append_segment(end_row: dict[str, Any]) -> None:
        if segment_start is None or segment_geometry is None:
            return
        global_devices, local_devices, per_device, process_batch, process_count = (
            segment_geometry
        )
        geometry_segments.append(
            {
                "start_step": int(segment_start["step"]),
                "end_step": int(end_row["step"]),
                "updates_spanned": int(end_row["step"]) - int(segment_start["step"]) + 1,
                "global_devices": global_devices,
                "local_devices": local_devices,
                "per_device_batch": per_device,
                "global_batch": global_devices * per_device,
                "process_batch": process_batch,
                "process_count": process_count,
                "start_utc": utc_iso(float(segment_start["_timestamp"])),
                "end_utc": utc_iso(float(end_row["_timestamp"])),
            }
        )

    for row in history_rows("final_reused_run"):
        current_geometry = geometry(row)
        if segment_start is None:
            segment_start = row
            segment_geometry = current_geometry
        elif current_geometry != segment_geometry:
            assert previous is not None
            append_segment(previous)
            segment_start = row
            segment_geometry = current_geometry

        if previous is not None:
            wall_gap = float(row["_timestamp"]) - float(previous["_timestamp"])
            runtime_gap = float(row["_runtime"]) - float(previous["_runtime"])
            step_gap = int(row["step"]) - int(previous["step"])
            geometry_changed = current_geometry != geometry(previous)
            if wall_gap > 60 or step_gap != 1 or geometry_changed:
                prior_step = int(previous["step"])
                periodic_checkpoint = (prior_step // 5_000) * 5_000
                replay_steps = max(0, prior_step - periodic_checkpoint)
                expected_replay_s = replay_steps * 1.45
                runtime_signature = (
                    replay_steps > 0
                    and runtime_gap >= 0.5 * expected_replay_s
                    and runtime_gap <= 2.0 * expected_replay_s + 1_500
                )
                wall_signature = (
                    replay_steps > 0
                    and wall_gap >= 0.75 * expected_replay_s
                    and wall_gap <= 3.0 * expected_replay_s + 1_500
                )
                inferred_retry = bool(runtime_signature or (geometry_changed and wall_signature))
                confidence = (
                    "confirmed"
                    if int(row["step"]) == 176_187
                    else "high"
                    if runtime_signature
                    else "medium"
                    if inferred_retry
                    else "low"
                )
                interruptions.append(
                    {
                        "previous_step": prior_step,
                        "next_logged_step": int(row["step"]),
                        "step_delta": step_gap,
                        "wall_gap_seconds": wall_gap,
                        "wandb_runtime_delta_seconds": runtime_gap,
                        "geometry_changed": geometry_changed,
                        "periodic_checkpoint_hypothesis": periodic_checkpoint,
                        "inferred_replayed_updates": replay_steps if inferred_retry else 0,
                        "retry_included_in_compute_estimate": inferred_retry,
                        "confidence": confidence,
                        "next_utc": utc_iso(float(row["_timestamp"])),
                    }
                )

        coefficient = inferred_sigreg_coefficient(row)
        stable_coefficient = None
        if coefficient is not None:
            if close_to(coefficient, 0.05):
                stable_coefficient = 0.05
            elif close_to(coefficient, 0.01):
                stable_coefficient = 0.01
        if stable_coefficient is not None and stable_coefficient != last_stable_coefficient:
            coefficient_changes.append(
                {
                    "step": int(row["step"]),
                    "inferred_jepa_sigreg_coefficient": stable_coefficient,
                    "timestamp_utc": utc_iso(float(row["_timestamp"])),
                }
            )
            last_stable_coefficient = stable_coefficient

        step = int(row["step"])
        if step >= 240_000 and (
            step % 1_000 == 0
            or row.get("val_dfm_ce_loss") is not None
            or step == WANDB_C_ENDPOINT
        ):
            tail_training.append(
                {
                    "step": step,
                    "timestamp_utc": utc_iso(float(row["_timestamp"])),
                    "dfm_ce_loss": row.get("dfm_ce_loss"),
                    "jepa_positive_loss": row.get("jepa_positive_loss"),
                    "jepa_sigreg_loss": row.get("jepa_sigreg_loss"),
                    "jepa_norm_loss": row.get("jepa_norm_loss"),
                    "accuracy": row.get("accuracy"),
                    "val_dfm_ce_loss": row.get("val_dfm_ce_loss"),
                    "val_jepa_positive_loss": row.get("val_jepa_positive_loss"),
                    "val_jepa_sigreg_loss": row.get("val_jepa_sigreg_loss"),
                }
            )
        previous = row

    if previous is not None:
        append_segment(previous)
    return {
        "geometry_segments": geometry_segments,
        "interruptions": interruptions,
        "coefficient_changes": coefficient_changes,
        "tail_training": tail_training,
    }


def load_dataset_size() -> int:
    manifest = read_json(DATASET_MANIFEST)
    if isinstance(manifest.get("train"), dict):
        for key in ("examples", "example_count", "num_examples"):
            if key in manifest["train"]:
                return int(manifest["train"][key])
    for key in ("train_examples", "example_count", "num_examples"):
        if key in manifest:
            return int(manifest[key])
    # The checked-in manifest currently declares 27,679 full 1,024-row shards.
    shard_count = int(manifest.get("train_shard_count", 27_679))
    shard_size = int(manifest.get("shard_size", 1_024))
    return shard_count * shard_size


def c_lineage_examples(step: int) -> int:
    small_updates = min(step, ANCESTOR_B_STEP)
    large_updates = max(0, step - ANCESTOR_B_STEP)
    return small_updates * 512 + large_updates * 8_192


def cumulative_lineage_examples(run_label: str, step: int) -> int:
    initial_examples = 221_432 * 256
    if run_label == "initial_b256":
        return step * 256
    if run_label == "continuation_b512":
        return initial_examples + step * 512
    if run_label == "final_reused_run":
        inherited = initial_examples + ANCESTOR_B_STEP * 512
        return inherited + c_lineage_examples(step)
    raise KeyError(run_label)


def validation_protocol(run_label: str, step: int) -> dict[str, Any]:
    if run_label == "initial_b256":
        process_batch, shard_count = 256, 1_524
        protocol = "A: full validation pool, 32×256"
    elif run_label == "continuation_b512":
        process_batch, shard_count = 512, 1_524
        protocol = "B: full validation pool, 32×512"
    elif step <= 10_000:
        process_batch, shard_count = 32, 96
        protocol = "C1: process-0 shard, 32×32"
    else:
        process_batch, shard_count = 512, 96
        protocol = "C2: process-0 shard, 32×512"
    return {
        "validation_protocol": protocol,
        "validation_examples": process_batch * 32,
        "validation_visible_shards": shard_count,
    }


def prepare_validation(
    validation_by_run: dict[str, list[dict[str, Any]]], dataset_examples: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in RUN_LABELS:
        for raw in validation_by_run[label]:
            step = int(raw["source_step"])
            if label == "continuation_b512" and step > ANCESTOR_B_STEP:
                lineage_status = "side branch; not inherited by final model"
            else:
                lineage_status = "final-model lineage"
            cumulative_examples = cumulative_lineage_examples(label, step)
            protocol = validation_protocol(label, step)
            rows.append(
                {
                    **raw,
                    "timestamp_utc": utc_iso(float(raw["timestamp"])),
                    "cumulative_examples": cumulative_examples,
                    "cumulative_epochs": cumulative_examples / dataset_examples,
                    "lineage_status": lineage_status,
                    **protocol,
                }
            )
    return rows


def exponential_floor(x: np.ndarray, floor: float, amplitude: float, rate: float) -> np.ndarray:
    return floor + amplitude * np.exp(-rate * (x - 50.0))


def power_floor(x: np.ndarray, floor: float, amplitude: float, exponent: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -exponent)


FitSpec = tuple[
    Callable[..., np.ndarray],
    list[float],
    tuple[list[float], list[float]],
]


FIT_SPECS: dict[str, FitSpec] = {
    "exponential_to_floor": (
        exponential_floor,
        [4.3, 0.6, 0.05],
        ([0.0, 0.0, 1e-5], [4.8, 10.0, 2.0]),
    ),
    "power_law_to_floor": (
        power_floor,
        [4.0, 10_000.0, 2.0],
        ([0.0, 0.0, 0.01], [4.8, 100_000_000.0, 10.0]),
    ),
}


def fit_one(
    points: list[dict[str, Any]],
    model_name: str,
    start_step: int,
    cutoff_step: int,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = [
        row
        for row in points
        if row["source_run"] == "final_reused_run"
        and start_step <= int(row["source_step"]) <= cutoff_step
    ]
    x = np.asarray([row["cumulative_epochs"] for row in selected], dtype=float)
    y = np.asarray([row["val_dfm_ce_loss"] for row in selected], dtype=float)
    fn, initial, bounds = FIT_SPECS[model_name]
    parameters, _ = curve_fit(
        fn, x, y, p0=initial, bounds=bounds, maxfev=200_000
    )
    fitted = fn(x, *parameters)
    prediction = float(fn(np.asarray([100.0]), *parameters)[0])
    return (
        {
            "model": model_name,
            "start_step": start_step,
            "cutoff_step": cutoff_step,
            "fit_points": len(selected),
            "start_epoch": float(x.min()),
            "cutoff_epoch": float(x.max()),
            "rmse": float(np.sqrt(np.mean(np.square(fitted - y)))),
            "projected_val_dfm_ce_at_epoch_100": prediction,
            "parameters_json": json.dumps(parameters.tolist()),
        },
        parameters,
    )


def fit_projections(
    validation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fits: list[dict[str, Any]] = []
    parameters: dict[tuple[str, int, int], np.ndarray] = {}
    for cutoff in (255_000, 265_000):
        for start in (180_000, 200_000):
            for model_name in FIT_SPECS:
                fit, fitted_parameters = fit_one(
                    validation_rows, model_name, start, cutoff
                )
                fit["scenario"] = (
                    "stable-phase, before deterioration"
                    if cutoff == 255_000
                    else "tail-inclusive"
                )
                fits.append(fit)
                parameters[(model_name, start, cutoff)] = fitted_parameters

    # Holdout: fit only through step 230k and score the unseen 235k..255k points.
    for model_name in FIT_SPECS:
        fit, fitted_parameters = fit_one(
            validation_rows, model_name, 180_000, 230_000
        )
        fn = FIT_SPECS[model_name][0]
        test = [
            row
            for row in validation_rows
            if row["source_run"] == "final_reused_run"
            and 235_000 <= int(row["source_step"]) <= 255_000
        ]
        x_test = np.asarray([row["cumulative_epochs"] for row in test], dtype=float)
        y_test = np.asarray([row["val_dfm_ce_loss"] for row in test], dtype=float)
        y_pred = fn(x_test, *fitted_parameters)
        fit.update(
            {
                "scenario": "holdout backtest",
                "holdout_start_step": 235_000,
                "holdout_end_step": 255_000,
                "holdout_points": len(test),
                "holdout_mae": float(np.mean(np.abs(y_pred - y_test))),
                "holdout_bias": float(np.mean(y_pred - y_test)),
            }
        )
        fits.append(fit)

    # One preferred line per hypothesis, fit from the coefficient change through
    # the last point before deterioration.
    projection_rows: list[dict[str, Any]] = []
    epochs = np.linspace(50.0, 100.0, 201)
    for model_name in FIT_SPECS:
        fn = FIT_SPECS[model_name][0]
        fitted_parameters = parameters[(model_name, 180_000, 255_000)]
        for epoch, value in zip(epochs, fn(epochs, *fitted_parameters), strict=True):
            projection_rows.append(
                {
                    "epoch": float(epoch),
                    "series": model_name.replace("_", " "),
                    "val_dfm_ce_loss": float(value),
                    "kind": "conditional projection",
                }
            )
    return fits, projection_rows


def local_checkpoint_crosscheck() -> dict[str, Any]:
    wandb_values: dict[int, dict[str, Any]] = {}
    for row in history_rows("final_reused_run"):
        if row.get("val_dfm_ce_loss") is not None:
            wandb_values[int(row["step"])] = row
    local_values: dict[int, dict[str, Any]] = {}
    with LOCAL_CHECKPOINT_METRICS.open() as handle:
        local_row_count = 0
        for line in handle:
            local_row_count += 1
            row = json.loads(line)
            if row.get("val_dfm_ce_loss") is not None:
                local_values[int(row["step"])] = row
    common_steps = sorted(set(wandb_values) & set(local_values))
    keys = [
        "val_dfm_ce_loss",
        "val_loss",
        "val_accuracy",
        "val_jepa_positive_loss",
        "val_jepa_sigreg_loss",
    ]
    max_differences = {
        key: max(
            abs(float(wandb_values[step][key]) - float(local_values[step][key]))
            for step in common_steps
        )
        for key in keys
    }
    return {
        "local_history_rows": local_row_count,
        "local_validation_points": len(local_values),
        "wandb_validation_points": len(wandb_values),
        "common_validation_steps": common_steps,
        "max_absolute_differences": max_differences,
        "all_checked_values_exact": all(value == 0.0 for value in max_differences.values()),
    }


def build_lineage_rows(dataset_examples: int) -> list[dict[str, Any]]:
    initial = 221_432 * 256
    b_inherited = ANCESTOR_B_STEP * 512
    c_retained = c_lineage_examples(RETAINED_C_STEP)
    c_endpoint = c_lineage_examples(WANDB_C_ENDPOINT)
    return [
        {
            "segment": "A — fresh joint training",
            "source_run": "initial_b256",
            "inherited_updates": 221_432,
            "global_batch": 256,
            "examples_in_final_lineage": initial,
            "epochs_in_final_lineage": initial / dataset_examples,
            "optimizer_start": "fresh",
            "main_learning_rate": 6e-4,
            "jepa_sigreg_coefficient": 0.1,
            "notes": "20k teacher-forcing steps; raw BT4 initialization",
        },
        {
            "segment": "B — inherited prefix only",
            "source_run": "continuation_b512",
            "inherited_updates": ANCESTOR_B_STEP,
            "global_batch": 512,
            "examples_in_final_lineage": b_inherited,
            "epochs_in_final_lineage": b_inherited / dataset_examples,
            "optimizer_start": "fresh (model-only checkpoint load)",
            "main_learning_rate": 3e-4,
            "jepa_sigreg_coefficient": 0.05,
            "notes": (
                "Steps 10,001–15,176 formed a 0.0935-epoch side branch and "
                "were not inherited"
            ),
        },
        {
            "segment": "C1 — final run prefix",
            "source_run": "final_reused_run",
            "inherited_updates": 10_000,
            "global_batch": 512,
            "examples_in_final_lineage": 10_000 * 512,
            "epochs_in_final_lineage": (10_000 * 512) / dataset_examples,
            "optimizer_start": "fresh (model-only checkpoint load)",
            "main_learning_rate": 3e-4,
            "jepa_sigreg_coefficient": 0.05,
            "notes": "The later 8,192-batch resume appears to restart from step 10,000",
        },
        {
            "segment": "C2 — large-batch continuation to retained checkpoint",
            "source_run": "final_reused_run",
            "inherited_updates": RETAINED_C_STEP - 10_000,
            "global_batch": 8_192,
            "examples_in_final_lineage": (RETAINED_C_STEP - 10_000) * 8_192,
            "epochs_in_final_lineage": (
                (RETAINED_C_STEP - 10_000) * 8_192 / dataset_examples
            ),
            "optimizer_start": "resumed within run",
            "main_learning_rate": 3e-4,
            "jepa_sigreg_coefficient": "0.05 through step 175k; 0.01 thereafter",
            "notes": "Validation best at step 255k; retained checkpoint is step 265k",
        },
        {
            "segment": "Retained final model — total",
            "source_run": "three-run lineage",
            "inherited_updates": 221_432 + ANCESTOR_B_STEP + RETAINED_C_STEP,
            "global_batch": "",
            "examples_in_final_lineage": initial + b_inherited + c_retained,
            "epochs_in_final_lineage": (
                initial + b_inherited + c_retained
            )
            / dataset_examples,
            "optimizer_start": "three optimizer phases",
            "main_learning_rate": "6e-4, then 3e-4",
            "jepa_sigreg_coefficient": "0.1 → 0.05 → 0.01",
            "notes": "Exact under the high-confidence step-10k resume reconstruction",
        },
        {
            "segment": "W&B endpoint — not retained locally",
            "source_run": "final_reused_run",
            "inherited_updates": 221_432 + ANCESTOR_B_STEP + WANDB_C_ENDPOINT,
            "global_batch": "",
            "examples_in_final_lineage": initial + b_inherited + c_endpoint,
            "epochs_in_final_lineage": (
                initial + b_inherited + c_endpoint
            )
            / dataset_examples,
            "optimizer_start": "",
            "main_learning_rate": "",
            "jepa_sigreg_coefficient": 0.01,
            "notes": "Training signal was unstable; run state is crashed",
        },
    ]


def build_data_quality(
    audits: list[HistoryAudit],
    final_inspection: dict[str, Any],
    crosscheck: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile": {
            "runs": len(audits),
            "history_rows": sum(audit.row_count for audit in audits),
            "validation_points": sum(audit.validation_points for audit in audits),
            "duplicate_steps": sum(audit.duplicate_step_count for audit in audits),
            "missing_steps_by_run": {
                audit.label: audit.missing_steps for audit in audits
            },
            "local_checkpoint_crosscheck": crosscheck,
        },
        "issues": [
            {
                "severity": "high",
                "confidence": "high",
                "issue": "One W&B run ID was reused across batch and loss-recipe changes.",
                "impact": (
                    "The final config describes only the latest resume, while early "
                    "history used global batch 512 and JEPA SIGReg coefficient 0.05."
                ),
                "remediation": (
                    "Use one immutable run ID per recipe segment and an explicit "
                    "parent_run_id/parent_checkpoint_step."
                ),
            },
            {
                "severity": "high",
                "confidence": "high",
                "issue": "Validation population changed with process topology.",
                "impact": (
                    "A/B evaluated the full validation pool; C evaluated process 0's "
                    "96-shard subset. C also used only 1,024 examples at steps 5k/10k "
                    "before switching to 16,384."
                ),
                "remediation": (
                    "Create a topology-invariant frozen validation index and aggregate "
                    "example-weighted metrics across all processes."
                ),
            },
            {
                "severity": "high",
                "confidence": "confirmed",
                "issue": "Checkpoint rollback caused duplicate training that W&B rejected.",
                "impact": (
                    "At the step-175k restart, retrained steps were omitted from W&B "
                    "until the logged step exceeded 176,186; several earlier gaps show "
                    "the same signature."
                ),
                "remediation": (
                    "Log monotonic optimizer_updates_total and examples_seen_total, "
                    "plus checkpoint lineage and retry counters."
                ),
            },
            {
                "severity": "medium",
                "confidence": "high",
                "issue": "The displayed composite loss is not longitudinally comparable.",
                "impact": (
                    "Teacher forcing and JEPA SIGReg weights changed (0.1→0.05→0.01), "
                    "including a coefficient change at step 176,187 inside run C."
                ),
                "remediation": (
                    "Version the objective and log raw components plus a fixed "
                    "reference-objective score."
                ),
            },
            {
                "severity": "medium",
                "confidence": "high",
                "issue": "The retained checkpoint is not the W&B endpoint.",
                "impact": (
                    "W&B reached step 267,158, while the locally evaluable checkpoint "
                    "is step 265,000. The endpoint was already unstable."
                ),
                "remediation": (
                    "Log checkpoint artifact IDs and mark the canonical evaluation "
                    "checkpoint explicitly."
                ),
            },
        ],
        "final_run_geometry_segments": final_inspection["geometry_segments"],
        "final_run_coefficient_changes": final_inspection["coefficient_changes"],
        "final_run_interruptions": final_inspection["interruptions"],
    }


def create_figures(
    validation_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    tail_training: list[dict[str, Any]],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = {
        "initial_b256": "#6b7280",
        "continuation_b512": "#d97706",
        "final_reused_run": "#2563eb",
    }
    labels = {
        "initial_b256": "Run A",
        "continuation_b512": "Run B",
        "final_reused_run": "Run C",
    }
    for label in RUN_LABELS:
        selected = [
            row
            for row in validation_rows
            if row["source_run"] == label
            and row["lineage_status"] == "final-model lineage"
        ]
        ax.plot(
            [row["cumulative_epochs"] for row in selected],
            [row["val_dfm_ce_loss"] for row in selected],
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color=colors[label],
            label=labels[label],
        )
    side = [
        row
        for row in validation_rows
        if row["lineage_status"].startswith("side branch")
    ]
    ax.scatter(
        [row["cumulative_epochs"] for row in side],
        [row["val_dfm_ce_loss"] for row in side],
        marker="x",
        s=55,
        color="#dc2626",
        label="Run B side branch",
        zorder=5,
    )
    projection_colors = {
        "exponential to floor": "#059669",
        "power law to floor": "#7c3aed",
    }
    for name in projection_colors:
        selected = [row for row in projection_rows if row["series"] == name]
        ax.plot(
            [row["epoch"] for row in selected],
            [row["val_dfm_ce_loss"] for row in selected],
            linestyle="--",
            linewidth=2,
            color=projection_colors[name],
            label=f"{name.title()} (conditional)",
        )
    ax.axvline(50.39, color="#9333ea", linestyle=":", linewidth=1.5)
    ax.text(50.9, 5.9, "SIGReg 0.05→0.01", color="#7e22ce", fontsize=9)
    ax.axvspan(74.6, 76.1, color="#fee2e2", alpha=0.75)
    ax.text(75.0, 5.75, "deterioration", rotation=90, color="#b91c1c", fontsize=9)
    ax.set_xlim(0, 101)
    ax.set_xlabel("Cumulative full-dataset epochs in the final-model lineage")
    ax.set_ylabel("Validation DFM cross-entropy")
    ax.set_title("Reconstructed validation trajectory and conditional 100-epoch fits")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_lineage_projection.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    val_tail = [
        row
        for row in validation_rows
        if row["source_run"] == "final_reused_run" and row["source_step"] >= 225_000
    ]
    axes[0].plot(
        [row["source_step"] for row in val_tail],
        [row["val_dfm_ce_loss"] for row in val_tail],
        marker="o",
        color="#2563eb",
    )
    axes[0].axvline(255_000, color="#059669", linestyle="--")
    axes[0].set_ylabel("Validation DFM CE")
    axes[0].set_title("The retained checkpoint is after the best validation point")
    training = [
        row
        for row in tail_training
        if row["jepa_sigreg_loss"] is not None and row["step"] % 1_000 == 0
    ]
    axes[1].plot(
        [row["step"] for row in training],
        [row["jepa_sigreg_loss"] for row in training],
        color="#dc2626",
        linewidth=1.6,
    )
    axes[1].set_ylabel("Training target SIGReg")
    axes[1].set_xlabel("Run-C step")
    axes[1].set_title("Target SIGReg spikes as the W&B endpoint destabilizes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tail_instability.png", dpi=170)
    plt.close(fig)


def analyze() -> dict[str, Any]:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dataset_examples = load_dataset_size()

    audits: list[HistoryAudit] = []
    validation_by_run: dict[str, list[dict[str, Any]]] = {}
    for label in RUN_LABELS:
        audit, validation = audit_history(label)
        audits.append(audit)
        validation_by_run[label] = validation

    final_inspection = inspect_final_run()
    validation_rows = prepare_validation(validation_by_run, dataset_examples)
    fits, projection_rows = fit_projections(validation_rows)
    lineage_rows = build_lineage_rows(dataset_examples)
    crosscheck = local_checkpoint_crosscheck()
    data_quality = build_data_quality(audits, final_inspection, crosscheck)

    preferred = [
        row
        for row in fits
        if row["scenario"] == "stable-phase, before deterioration"
    ]
    tail_inclusive = [row for row in fits if row["scenario"] == "tail-inclusive"]
    best_val_row = min(
        (
            row
            for row in validation_rows
            if row["source_run"] == "final_reused_run" and row["source_step"] >= 15_000
        ),
        key=lambda row: float(row["val_dfm_ce_loss"]),
    )
    inferred_replays = sum(
        int(row["inferred_replayed_updates"])
        for row in final_inspection["interruptions"]
        if row["retry_included_in_compute_estimate"]
    )
    retained_examples = int(
        next(
            row["examples_in_final_lineage"]
            for row in lineage_rows
            if row["segment"] == "Retained final model — total"
        )
    )
    summary = {
        "dataset_examples": dataset_examples,
        "retained_checkpoint_step": RETAINED_C_STEP,
        "retained_lineage_examples": retained_examples,
        "retained_lineage_epochs": retained_examples / dataset_examples,
        "literal_wandb_geometry_epochs": (
            221_432 * 256
            + ANCESTOR_B_STEP * 512
            + 10_722 * 512
            + (RETAINED_C_STEP - 10_722) * 8_192
        )
        / dataset_examples,
        "wandb_endpoint_step": WANDB_C_ENDPOINT,
        "wandb_endpoint_lineage_epochs": (
            221_432 * 256
            + ANCESTOR_B_STEP * 512
            + c_lineage_examples(WANDB_C_ENDPOINT)
        )
        / dataset_examples,
        "intermediate_side_branch_epochs": (
            (15_176 - ANCESTOR_B_STEP) * 512 / dataset_examples
        ),
        "inferred_hidden_replayed_updates": inferred_replays,
        "inferred_hidden_replay_epochs": inferred_replays * 8_192 / dataset_examples,
        "best_comparable_validation": {
            "step": int(best_val_row["source_step"]),
            "cumulative_epochs": float(best_val_row["cumulative_epochs"]),
            "val_dfm_ce_loss": float(best_val_row["val_dfm_ce_loss"]),
        },
        "retained_checkpoint_validation": next(
            {
                "step": int(row["source_step"]),
                "cumulative_epochs": float(row["cumulative_epochs"]),
                "val_dfm_ce_loss": float(row["val_dfm_ce_loss"]),
            }
            for row in validation_rows
            if row["source_run"] == "final_reused_run"
            and row["source_step"] == RETAINED_C_STEP
        ),
        "conditional_epoch_100_projection": {
            "preferred_hypothesis": (
                "Post-SIGReg-change saturation, fit through the last point before "
                "the observed deterioration"
            ),
            "range_min": min(
                float(row["projected_val_dfm_ce_at_epoch_100"]) for row in preferred
            ),
            "range_max": max(
                float(row["projected_val_dfm_ce_at_epoch_100"]) for row in preferred
            ),
            "tail_inclusive_range_min": min(
                float(row["projected_val_dfm_ce_at_epoch_100"])
                for row in tail_inclusive
            ),
            "tail_inclusive_range_max": max(
                float(row["projected_val_dfm_ce_at_epoch_100"])
                for row in tail_inclusive
            ),
            "warning": (
                "Conditional only: the actual run deteriorated after step 255k and "
                "was unstable by the W&B endpoint, so uninterrupted continuation to "
                "100 epochs is not supported by the observed run."
            ),
        },
    }

    write_csv(DERIVED_DIR / "history_audit.csv", [asdict(row) for row in audits])
    write_csv(DERIVED_DIR / "lineage_segments.csv", lineage_rows)
    write_csv(DERIVED_DIR / "validation_points.csv", validation_rows)
    write_csv(DERIVED_DIR / "projection_fits.csv", fits)
    write_csv(DERIVED_DIR / "projection_curves.csv", projection_rows)
    write_csv(
        DERIVED_DIR / "final_run_geometry_segments.csv",
        final_inspection["geometry_segments"],
    )
    write_csv(
        DERIVED_DIR / "final_run_interruptions.csv",
        final_inspection["interruptions"],
    )
    write_csv(
        DERIVED_DIR / "final_run_tail_training.csv",
        final_inspection["tail_training"],
    )
    json_dump(DERIVED_DIR / "data_quality.json", data_quality)
    json_dump(DERIVED_DIR / "summary.json", summary)
    create_figures(
        validation_rows, projection_rows, final_inspection["tail_training"]
    )
    return {
        "summary": summary,
        "audits": [asdict(row) for row in audits],
        "lineage_rows": lineage_rows,
        "projection_fits": fits,
        "data_quality": data_quality,
    }


def main() -> None:
    results = analyze()
    summary = results["summary"]
    projection = summary["conditional_epoch_100_projection"]
    print(
        f"retained_lineage_epochs={summary['retained_lineage_epochs']:.6f}\n"
        f"best_val_dfm_ce={summary['best_comparable_validation']['val_dfm_ce_loss']:.6f} "
        f"at_step={summary['best_comparable_validation']['step']}\n"
        f"retained_val_dfm_ce="
        f"{summary['retained_checkpoint_validation']['val_dfm_ce_loss']:.6f}\n"
        f"epoch100_conditional_range={projection['range_min']:.4f}.."
        f"{projection['range_max']:.4f}\n"
        f"epoch100_tail_inclusive_range={projection['tail_inclusive_range_min']:.4f}.."
        f"{projection['tail_inclusive_range_max']:.4f}"
    )


if __name__ == "__main__":
    main()
