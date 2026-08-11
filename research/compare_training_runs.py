"""Freeze a control curve and compare matched Torch training runs.

The Phase-1 optimizer search uses this module in two deliberately separate
steps.  ``freeze-control`` records every decision boundary from Hero 1 before
candidate results are inspected.  ``compare`` then fails closed unless the
candidate used the same source model, ordered training examples, and frozen
validation pool.

Only compact JSON artifacts are required; checkpoints stay on remote storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "chess-dfm-matched-training-comparison-v1"
OBJECTIVE_ABLATION_SCHEMA_VERSION = "chess-dfm-objective-ablation-comparison-v1"
POLICY_DISTILLATION_SCHEMA_VERSION = (
    "chess-dfm-policy-distillation-comparison-v1"
)
FIXED_POLICY_DISTILLATION_SCHEMA_VERSION = (
    "chess-dfm-fixed-policy-distillation-comparison-v1"
)
CONTROL_SCHEMA_VERSION = "chess-dfm-training-control-contract-v1"
SMOOTHING_WINDOWS = (32, 128)
THRESHOLD_ANCHORS = (300, 400, 500, 554)
SLOPE_START_UPDATE = 101
BOOTSTRAP_BLOCK_UPDATES = 32
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_809
MOVEMENT_GROUPS = ("all", "trunk", "embedding", "layers", "policy_head")
MAX_TRUNK_RELATIVE_DELTA_L2 = 0.005
MAX_GLOBAL_RELATIVE_DELTA_L2 = 0.01

OBJECTIVE_METRICS = (
    "loss",
    "dfm_ce_loss",
    "weighted_legality_loss",
    "jepa_positive_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "weighted_root_legal_conditional_ce",
    "wdl_weighted_loss",
)
HEALTH_METRICS = (
    "accuracy",
    "first_legal_mass",
    "root_legal_conditional_ce",
    "wdl_loss",
    "wdl_accuracy",
    "gradient_global_norm",
    "gradient_clip_scale",
    "learning_rate",
    "bt4_learning_rate",
)
CURVE_METRICS = OBJECTIVE_METRICS + HEALTH_METRICS
BOOTSTRAP_METRICS = (
    "loss",
    "dfm_ce_loss",
    "root_legal_conditional_ce",
)

# These are screening non-inferiority margins, not statistical confidence
# bounds.  Absolute floors keep near-zero metrics from receiving a meaningless
# zero tolerance; relative margins prevent scale-heavy losses from receiving
# an overly strict absolute threshold.
VALIDATION_GUARDRAILS: dict[str, tuple[str, float, float]] = {
    "dfm_ce_loss": ("lower", 0.01, 0.02),
    "root_legal_conditional_ce": ("lower", 0.02, 0.02),
    "jepa_positive_loss": ("lower", 0.05, 0.01),
    "wdl_loss": ("lower", 0.02, 0.01),
    "accuracy": ("higher", 0.0, 0.01),
    "first_legal_mass": ("higher", 0.0, 0.01),
    "wdl_accuracy": ("higher", 0.0, 0.01),
}
VALIDATION_DIAGNOSTICS: dict[str, str] = {
    "root_legal_top1_accuracy": "higher",
    "wdl_brier_score": "lower",
    "wdl_ece_15": "lower",
    "wdl_expected_value_mse": "lower",
}

POLICY_DISTILLATION_DIAGNOSTICS = (
    "base_policy_root_legal_conditional_ce",
    "base_policy_dfm_root_legal_kl",
    "weighted_base_policy_dfm_root_legal_kl",
    "base_policy_root_legal_top1_accuracy",
    "base_policy_dfm_root_legal_top1_agreement",
)
FIXED_POLICY_TEACHER_DIAGNOSTICS = (
    "policy_teacher_active",
    "policy_teacher_root_legal_conditional_ce",
    "policy_teacher_dfm_root_legal_kl",
    "weighted_policy_teacher_dfm_root_legal_kl",
    "policy_teacher_root_legal_top1_accuracy",
    "policy_teacher_dfm_root_legal_top1_agreement",
)

_MODEL_AND_LOSS_CONFIG_KEYS = (
    "action_codec",
    "dfm_ce_coeff",
    "dfm_heads",
    "dfm_layers",
    "dfm_mlp_dim",
    "future_trainable_tail_layers",
    "horizon",
    "jepa_layers",
    "jepa_mlp_dim",
    "jepa_positive_coeff",
    "legality_coeff",
    "pred_sigreg_coeff",
    "projector_heads",
    "projector_layers",
    "projector_mlp_dim",
    "root_legal_ce_coeff",
    "sigreg_example_count",
    "sigreg_proj_dim",
    "target_sample_count",
    "target_sigreg_coeff",
    "token_dim",
    "use_bt4_policy_residual",
    "use_qk_norm",
    "use_xsa",
    "wdl_coeff",
    "z_dim",
)
_DATA_IDENTITY_KEYS = (
    "batch_schedule",
    "batch_size",
    "batches_per_shard",
    "file_manifest_sha256",
    "samples_per_shard",
    "seed",
    "seed_effective",
    "shard_count",
    "shuffle_files",
    "steps_per_epoch",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _metrics_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read metrics {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"No training records in {path}")

    previous_update: int | None = None
    previous_examples: int | None = None
    for index, record in enumerate(records):
        try:
            update = int(record["update"])
            examples = int(record["examples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed update/examples in record {index}") from exc
        if update < 1 or (previous_update is not None and update != previous_update + 1):
            raise ValueError("Training updates must be positive, unique, and contiguous")
        if examples < 1 or (previous_examples is not None and examples <= previous_examples):
            raise ValueError("Training example counts must be positive and increasing")
        if bool(record.get("optimizer_skipped_nonfinite", False)):
            raise ValueError(f"Run contains a skipped non-finite update at {update}")
        for metric in CURVE_METRICS:
            if metric not in record:
                raise ValueError(f"Training record {update} is missing {metric!r}")
            _finite_float(record[metric], label=f"update {update} {metric}")
        previous_update = update
        previous_examples = examples
    return records


def _stable_subset(value: Any, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: value[key] for key in keys if key in value}


def _pool_identity(run_config: Mapping[str, Any]) -> dict[str, Any]:
    validation = run_config.get("live_validation")
    if not isinstance(validation, Mapping):
        return {}
    pool = validation.get("fast_pool")
    if not isinstance(pool, Mapping):
        return {}
    data = pool.get("data")
    pool_definition = pool.get("pool_definition")
    return {
        "pool_name": pool.get("pool_name"),
        "indices_sha256": pool.get("indices_sha256"),
        "data": _stable_subset(
            data,
            (
                "file_manifest_sha256",
                "indices_sha256",
                "pool_name",
                "position_count",
                "samples_per_shard",
                "shard_count",
            ),
        ),
        "pool_definition": dict(pool_definition) if isinstance(pool_definition, Mapping) else {},
    }


def _comparison_identity(run_config: Mapping[str, Any]) -> dict[str, Any]:
    source = run_config.get("source")
    source = source if isinstance(source, Mapping) else {}
    raw_asset = source.get("raw_asset")
    raw_asset = raw_asset if isinstance(raw_asset, Mapping) else {}
    args = run_config.get("args")
    args = args if isinstance(args, Mapping) else {}
    return {
        "source": {
            "combined_sha256": source.get("combined_sha256"),
            "raw_asset_sha256": raw_asset.get("sha256"),
            "raw_asset_size_bytes": raw_asset.get("size_bytes"),
        },
        "model_and_loss": _stable_subset(
            run_config.get("config"),
            _MODEL_AND_LOSS_CONFIG_KEYS,
        ),
        "data": _stable_subset(run_config.get("data"), _DATA_IDENTITY_KEYS),
        "ordered_prefix": {
            "batch_size": args.get("batch_size"),
            "data_start": args.get("data_start"),
            "seed": args.get("seed"),
        },
        "validation_pool": _pool_identity(run_config),
    }


def _run_artifacts(run_dir: Path) -> dict[str, Any]:
    required = ("metrics.jsonl", "run_config.json", "report.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Run {run_dir} is missing compact artifacts: {missing}")
    run_config = _json_file(run_dir / "run_config.json")
    report = _json_file(run_dir / "report.json")
    return {
        "path": str(run_dir),
        "run_config": run_config,
        "report": report,
        "records": _metrics_records(run_dir),
        "sha256": {name: _sha256_file(run_dir / name) for name in required},
    }


def _moving_average_by_update(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    window: int,
) -> dict[int, float]:
    if window < 1:
        raise ValueError("Moving-average window must be positive")
    result: dict[int, float] = {}
    total = 0.0
    values: list[float] = []
    for record in records:
        value = _finite_float(record[metric], label=metric)
        values.append(value)
        total += value
        if len(values) > window:
            total -= values[-window - 1]
        if len(values) >= window:
            result[int(record["update"])] = total / window
    return result


def _slope_per_million_examples(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    start_update: int,
) -> float:
    selected = [record for record in records if int(record["update"]) >= start_update]
    if len(selected) < 2:
        raise ValueError(f"Need at least two records at or after update {start_update}")
    xs = [float(record["examples"]) / 1_000_000.0 for record in selected]
    ys = [_finite_float(record[metric], label=metric) for record in selected]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0.0:
        raise ValueError("Cannot calculate a slope from identical example counts")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def _curve_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terminal = records[-1]
    result: dict[str, Any] = {
        "record_count": len(records),
        "update_range": [int(records[0]["update"]), int(terminal["update"])],
        "example_range": [int(records[0]["examples"]), int(terminal["examples"])],
        "terminal": {metric: float(terminal[metric]) for metric in CURVE_METRICS},
        "moving_average_endpoint": {},
        "ols_slope_per_million_examples": {},
    }
    for window in SMOOTHING_WINDOWS:
        endpoints: dict[str, float | None] = {}
        for metric in CURVE_METRICS:
            moving = _moving_average_by_update(records, metric, window)
            endpoints[metric] = moving.get(int(terminal["update"]))
        result["moving_average_endpoint"][str(window)] = endpoints
    effective_start = max(SLOPE_START_UPDATE, int(records[0]["update"]))
    if int(records[-1]["update"]) > effective_start:
        result["ols_slope_per_million_examples"] = {
            metric: _slope_per_million_examples(
                records,
                metric,
                start_update=effective_start,
            )
            for metric in OBJECTIVE_METRICS
        }
    result["slope_start_update"] = effective_start
    return result


def _latest_validation(report: Mapping[str, Any]) -> dict[str, Any]:
    live = report.get("live_validation")
    if not isinstance(live, Mapping):
        raise ValueError("Training report has no live_validation object")
    records = live.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Training report has no frozen validation record")
    record = max(records, key=lambda value: int(value.get("update", -1)))
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Frozen validation record has no metrics object")
    parsed = {
        metric: _finite_float(metrics.get(metric), label=f"validation {metric}")
        for metric in VALIDATION_GUARDRAILS
    }
    return {
        "update": int(record["update"]),
        "examples": int(record["examples"]),
        "evaluation_examples": int(record["evaluation_examples"]),
        "metrics": parsed,
    }


def _latest_validation_diagnostics(report: Mapping[str, Any]) -> dict[str, float]:
    live = report.get("live_validation")
    records = live.get("records") if isinstance(live, Mapping) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Training report has no frozen validation record")
    record = max(records, key=lambda value: int(value.get("update", -1)))
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Frozen validation record has no metrics object")
    return {
        metric: _finite_float(metrics[metric], label=f"validation diagnostic {metric}")
        for metric in VALIDATION_DIAGNOSTICS
        if metric in metrics
    }


def _latest_required_validation_metrics(
    report: Mapping[str, Any],
    metric_names: Sequence[str],
    *,
    label: str,
) -> dict[str, float]:
    live = report.get("live_validation")
    records = live.get("records") if isinstance(live, Mapping) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Training report has no frozen validation record")
    record = max(records, key=lambda value: int(value.get("update", -1)))
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Frozen validation record has no metrics object")
    missing = [metric for metric in metric_names if metric not in metrics]
    if missing:
        raise ValueError(f"{label} validation is missing required metrics: {missing}")
    return {
        metric: _finite_float(
            metrics[metric],
            label=f"{label} validation diagnostic {metric}",
        )
        for metric in metric_names
    }


def _checkpoint_state_sha256(report: Mapping[str, Any], *, label: str) -> str:
    checkpoint = report.get("checkpoint")
    state = checkpoint.get("state") if isinstance(checkpoint, Mapping) else None
    value = state.get("sha256") if isinstance(state, Mapping) else None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} report has no valid final checkpoint SHA-256")
    return value


def _movement_report(path: Path, *, expected_checkpoint_sha256: str) -> dict[str, Any]:
    report = _json_file(path)
    if report.get("schema_version") != "bt4-raw-hero-parameter-diff-v1":
        raise ValueError(f"Unsupported parameter-movement schema in {path}")
    models = report.get("models")
    groups = report.get("group_metrics")
    if not isinstance(models, Mapping) or not isinstance(groups, Mapping):
        raise ValueError(f"Malformed parameter-movement report {path}")
    if models.get("hero_state_sha256") != expected_checkpoint_sha256:
        raise ValueError(f"Parameter movement does not match the final checkpoint in {path}")
    parsed_groups: dict[str, Any] = {}
    for group in MOVEMENT_GROUPS:
        metrics = groups.get(group)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Parameter movement is missing group {group!r}")
        relative = _finite_float(
            metrics.get("relative_delta_l2"),
            label=f"{group} relative_delta_l2",
        )
        unchanged = _finite_float(
            metrics.get("unchanged_fraction"),
            label=f"{group} unchanged_fraction",
        )
        cosine = _finite_float(metrics.get("cosine"), label=f"{group} cosine")
        if relative < 0.0 or not 0.0 <= unchanged <= 1.0 or not -1.0 <= cosine <= 1.0:
            raise ValueError(f"Out-of-range parameter movement for group {group!r}")
        parsed_groups[group] = {
            "relative_delta_l2": relative,
            "changed_fraction": 1.0 - unchanged,
            "cosine": cosine,
        }
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "raw_asset_sha256": models.get("raw_asset_sha256"),
        "encoder_layout_sha256": models.get("encoder_layout_sha256"),
        "checkpoint_state_sha256": expected_checkpoint_sha256,
        "groups": parsed_groups,
    }


def _compare_movement_reports(
    control_path: Path,
    candidate_path: Path,
    *,
    control_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
) -> dict[str, Any]:
    control = _movement_report(
        control_path,
        expected_checkpoint_sha256=control_checkpoint_sha256,
    )
    candidate = _movement_report(
        candidate_path,
        expected_checkpoint_sha256=candidate_checkpoint_sha256,
    )
    if (
        candidate["raw_asset_sha256"] != control["raw_asset_sha256"]
        or candidate["encoder_layout_sha256"] != control["encoder_layout_sha256"]
    ):
        raise ValueError("Control and candidate parameter-movement identities differ")
    groups: dict[str, Any] = {}
    for group in MOVEMENT_GROUPS:
        control_group = control["groups"][group]
        candidate_group = candidate["groups"][group]
        groups[group] = {
            "control": control_group,
            "candidate": candidate_group,
            "relative_delta_l2_ratio": (
                candidate_group["relative_delta_l2"]
                / max(control_group["relative_delta_l2"], 1e-30)
            ),
            "changed_fraction_ratio": (
                candidate_group["changed_fraction"]
                / max(control_group["changed_fraction"], 1e-30)
            ),
        }
    trunk = groups["trunk"]
    healthier = (
        trunk["candidate"]["relative_delta_l2"]
        > trunk["control"]["relative_delta_l2"]
        and trunk["candidate"]["changed_fraction"]
        > trunk["control"]["changed_fraction"]
    )
    guardrail_pass = (
        groups["trunk"]["candidate"]["relative_delta_l2"]
        <= MAX_TRUNK_RELATIVE_DELTA_L2
        and groups["all"]["candidate"]["relative_delta_l2"]
        <= MAX_GLOBAL_RELATIVE_DELTA_L2
    )
    return {
        "control": {key: value for key, value in control.items() if key != "groups"},
        "candidate": {
            key: value for key, value in candidate.items() if key != "groups"
        },
        "groups": groups,
        "screen": {
            "healthier_visible_encoder_movement": healthier,
            "guardrail_pass": guardrail_pass,
            "max_trunk_relative_delta_l2": MAX_TRUNK_RELATIVE_DELTA_L2,
            "max_global_relative_delta_l2": MAX_GLOBAL_RELATIVE_DELTA_L2,
            "interpretation": (
                "requires more visible trunk movement than Hero 1 without a runaway; "
                "does not assert that more movement implies better chess"
            ),
        },
    }


def _validation_bounds(metrics: Mapping[str, float]) -> dict[str, Any]:
    bounds: dict[str, Any] = {}
    for metric, (direction, relative_margin, absolute_margin) in VALIDATION_GUARDRAILS.items():
        control_value = _finite_float(metrics[metric], label=f"control validation {metric}")
        tolerance = max(abs(control_value) * relative_margin, absolute_margin)
        bounds[metric] = {
            "direction": direction,
            "control": control_value,
            "tolerance": tolerance,
            "bound": control_value + tolerance if direction == "lower" else control_value - tolerance,
        }
    return bounds


def _threshold_contract(
    records: Sequence[Mapping[str, Any]],
    anchors: Sequence[int],
) -> list[dict[str, Any]]:
    moving = _moving_average_by_update(records, "loss", SMOOTHING_WINDOWS[0])
    available = {int(record["update"]): int(record["examples"]) for record in records}
    thresholds: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor not in moving or anchor not in available:
            raise ValueError(f"Control does not contain a full moving-average window at update {anchor}")
        thresholds.append(
            {
                "anchor_update": anchor,
                "anchor_examples": available[anchor],
                "metric": "loss",
                "window_updates": SMOOTHING_WINDOWS[0],
                "direction": "lower",
                "threshold": moving[anchor],
            }
        )
    return thresholds


def _first_threshold_crossing(
    records: Sequence[Mapping[str, Any]],
    threshold: Mapping[str, Any],
) -> dict[str, Any] | None:
    metric = str(threshold["metric"])
    window = int(threshold["window_updates"])
    target = _finite_float(threshold["threshold"], label="threshold")
    moving = _moving_average_by_update(records, metric, window)
    examples = {int(record["update"]): int(record["examples"]) for record in records}
    direction = str(threshold["direction"])
    for update in sorted(moving):
        value = moving[update]
        crossed = value <= target if direction == "lower" else value >= target
        if crossed:
            return {"update": update, "examples": examples[update], "value": value}
    return None


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_block_bootstrap(
    deltas: Sequence[float],
    *,
    block_updates: int = BOOTSTRAP_BLOCK_UPDATES,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if len(deltas) < block_updates:
        raise ValueError("Paired block bootstrap requires at least one complete block")
    if replicates < 100:
        raise ValueError("Paired block bootstrap requires at least 100 replicates")
    block_count = math.ceil(len(deltas) / block_updates)
    randomizer = random.Random(seed)
    means: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for _ in range(block_count):
            start = randomizer.randrange(len(deltas))
            sample.extend(deltas[(start + offset) % len(deltas)] for offset in range(block_updates))
        means.append(statistics.fmean(sample[: len(deltas)]))
    means.sort()
    return {
        "mean_candidate_minus_control": statistics.fmean(deltas),
        "confidence_level": 0.95,
        "ci": [_quantile(means, 0.025), _quantile(means, 0.975)],
        "block_updates": block_updates,
        "replicates": replicates,
        "seed": seed,
        "interpretation": "negative favors candidate",
    }


def freeze_control(
    run_dir: Path,
    *,
    expected_update: int = THRESHOLD_ANCHORS[-1],
    anchors: Sequence[int] = THRESHOLD_ANCHORS,
) -> dict[str, Any]:
    run = _run_artifacts(run_dir)
    records = run["records"]
    if int(records[0]["update"]) != 1 or int(records[-1]["update"]) != expected_update:
        raise ValueError(f"Control must contain the complete update 1..{expected_update} prefix")
    report = run["report"]
    if int(report.get("updates", -1)) != expected_update:
        raise ValueError("Control report update does not match its metrics curve")
    validation = _latest_validation(report)
    if validation["update"] != expected_update:
        raise ValueError("Control frozen validation is not at the comparison boundary")
    thresholds = _threshold_contract(records, anchors)
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "control": {
            "path": str(run_dir),
            "artifact_sha256": run["sha256"],
            "comparison_identity": _comparison_identity(run["run_config"]),
            "curve": _curve_summary(records),
            "validation": validation,
            "threshold_crossings": [
                _first_threshold_crossing(records, threshold) for threshold in thresholds
            ],
        },
        "analysis_contract": {
            "matched_prefix_required": True,
            "expected_update": expected_update,
            "smoothing_windows": list(SMOOTHING_WINDOWS),
            "slope_start_update": SLOPE_START_UPDATE,
            "objective_metrics": list(OBJECTIVE_METRICS),
            "health_metrics": list(HEALTH_METRICS),
            "paired_bootstrap": {
                "metrics": list(BOOTSTRAP_METRICS),
                "start_update": SLOPE_START_UPDATE,
                "block_updates": BOOTSTRAP_BLOCK_UPDATES,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
            "loss_thresholds": thresholds,
            "validation_guardrails": _validation_bounds(validation["metrics"]),
            "decision_rule": {
                "eligible_for_more_compute": (
                    "identities and prefix match; all updates are finite; validation guardrails pass; "
                    "paired loss evidence is not slower"
                ),
                "curve_status": (
                    "faster when paired loss 95% block-bootstrap CI is below zero; slower when above "
                    "zero; otherwise inconclusive"
                ),
                "limitation": (
                    "eligibility allocates compute only; encoder movement and chess behavior remain "
                    "required for final promotion"
                ),
            },
        },
    }


def _matched_records(
    control: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    expected_update: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    control_by_update = {int(record["update"]): record for record in control}
    candidate_by_update = {int(record["update"]): record for record in candidate}
    expected = list(range(1, expected_update + 1))
    if sorted(control_by_update) != expected or sorted(candidate_by_update) != expected:
        raise ValueError(f"Both Phase-1 runs must contain the complete matched 1..{expected_update} prefix")
    control_rows = [control_by_update[update] for update in expected]
    candidate_rows = [candidate_by_update[update] for update in expected]
    for control_row, candidate_row in zip(control_rows, candidate_rows, strict=True):
        if int(control_row["examples"]) != int(candidate_row["examples"]):
            raise ValueError(f"Example-count mismatch at update {control_row['update']}")
    return control_rows, candidate_rows


def compare_candidate(
    control_contract_path: Path,
    control_dir: Path,
    candidate_dir: Path,
    *,
    control_movement_path: Path | None = None,
    candidate_movement_path: Path | None = None,
) -> dict[str, Any]:
    contract = _json_file(control_contract_path)
    if contract.get("schema_version") != CONTROL_SCHEMA_VERSION:
        raise ValueError("Unsupported control contract schema")
    expected_update = int(contract["analysis_contract"]["expected_update"])
    control = _run_artifacts(control_dir)
    candidate = _run_artifacts(candidate_dir)

    expected_hashes = contract["control"]["artifact_sha256"]
    if control["sha256"] != expected_hashes:
        raise ValueError("Control artifacts changed after the decision contract was frozen")
    control_identity = _comparison_identity(control["run_config"])
    candidate_identity = _comparison_identity(candidate["run_config"])
    if control_identity != contract["control"]["comparison_identity"]:
        raise ValueError("Control identity changed after the decision contract was frozen")
    if candidate_identity != control_identity:
        raise ValueError("Candidate source/data/validation identity does not match the control")

    control_records, candidate_records = _matched_records(
        control["records"],
        candidate["records"],
        expected_update=expected_update,
    )
    candidate_report = candidate["report"]
    if int(candidate_report.get("updates", -1)) != expected_update:
        raise ValueError("Candidate report update does not match the comparison boundary")
    candidate_validation = _latest_validation(candidate_report)
    if candidate_validation["update"] != expected_update:
        raise ValueError("Candidate frozen validation is not at the comparison boundary")

    control_diagnostics = _latest_validation_diagnostics(control["report"])
    candidate_diagnostics = _latest_validation_diagnostics(candidate_report)
    validation_diagnostics = {
        metric: {
            "direction": VALIDATION_DIAGNOSTICS[metric],
            "control": control_diagnostics[metric],
            "candidate": candidate_diagnostics[metric],
            "delta": candidate_diagnostics[metric] - control_diagnostics[metric],
        }
        for metric in VALIDATION_DIAGNOSTICS
        if metric in control_diagnostics and metric in candidate_diagnostics
    }

    bounds = contract["analysis_contract"]["validation_guardrails"]
    validation_checks: dict[str, Any] = {}
    for metric, bound in bounds.items():
        value = candidate_validation["metrics"][metric]
        passed = value <= float(bound["bound"]) if bound["direction"] == "lower" else value >= float(bound["bound"])
        validation_checks[metric] = {
            **bound,
            "candidate": value,
            "delta": value - float(bound["control"]),
            "passed": passed,
        }
    validation_passed = all(check["passed"] for check in validation_checks.values())

    bootstrap_start = int(contract["analysis_contract"]["paired_bootstrap"]["start_update"])
    paired: dict[str, Any] = {}
    for metric in contract["analysis_contract"]["paired_bootstrap"]["metrics"]:
        deltas = [
            float(candidate_row[metric]) - float(control_row[metric])
            for control_row, candidate_row in zip(control_records, candidate_records, strict=True)
            if int(control_row["update"]) >= bootstrap_start
        ]
        paired[metric] = _paired_block_bootstrap(
            deltas,
            block_updates=int(contract["analysis_contract"]["paired_bootstrap"]["block_updates"]),
            replicates=int(contract["analysis_contract"]["paired_bootstrap"]["replicates"]),
            seed=int(contract["analysis_contract"]["paired_bootstrap"]["seed"]),
        )

    loss_ci = paired["loss"]["ci"]
    curve_status = "faster" if loss_ci[1] < 0.0 else "slower" if loss_ci[0] > 0.0 else "inconclusive"
    threshold_results = []
    for threshold, control_crossing in zip(
        contract["analysis_contract"]["loss_thresholds"],
        contract["control"]["threshold_crossings"],
        strict=True,
    ):
        candidate_crossing = _first_threshold_crossing(candidate_records, threshold)
        threshold_results.append(
            {
                "threshold": threshold,
                "control_crossing": control_crossing,
                "candidate_crossing": candidate_crossing,
                "candidate_minus_control_examples": (
                    candidate_crossing["examples"] - control_crossing["examples"]
                    if candidate_crossing is not None and control_crossing is not None
                    else None
                ),
            }
        )

    if (control_movement_path is None) != (candidate_movement_path is None):
        raise ValueError("Control and candidate movement reports must be supplied together")
    movement = None
    if control_movement_path is not None and candidate_movement_path is not None:
        movement = _compare_movement_reports(
            control_movement_path,
            candidate_movement_path,
            control_checkpoint_sha256=_checkpoint_state_sha256(
                control["report"],
                label="control",
            ),
            candidate_checkpoint_sha256=_checkpoint_state_sha256(
                candidate_report,
                label="candidate",
            ),
        )
    movement_passed = movement is None or (
        movement["screen"]["healthier_visible_encoder_movement"]
        and movement["screen"]["guardrail_pass"]
    )
    eligible = validation_passed and curve_status != "slower" and movement_passed
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "control_contract": {
            "path": str(control_contract_path),
            "sha256": _sha256_file(control_contract_path),
        },
        "control": {
            "path": str(control_dir),
            "artifact_sha256": control["sha256"],
            "curve": _curve_summary(control_records),
        },
        "candidate": {
            "path": str(candidate_dir),
            "artifact_sha256": candidate["sha256"],
            "curve": _curve_summary(candidate_records),
        },
        "matched_contract": {
            "comparison_identity_sha256": hashlib.sha256(_canonical_json(control_identity)).hexdigest(),
            "updates": [1, expected_update],
            "examples": [
                int(control_records[0]["examples"]),
                int(control_records[-1]["examples"]),
            ],
        },
        "paired_curve_differences": paired,
        "loss_threshold_crossings": threshold_results,
        "frozen_validation": {
            "candidate": candidate_validation,
            "checks": validation_checks,
            "passed": validation_passed,
            "descriptive_diagnostics": {
                "eligibility_effect": False,
                "declared_after_control_freeze": True,
                "comparisons": validation_diagnostics,
            },
        },
        "parameter_movement": movement,
        "allocation_decision": {
            "curve_status": curve_status,
            "eligible_for_more_compute": eligible,
            "requires_encoder_movement_check": movement is None,
            "requires_chess_behavior_before_final_promotion": True,
            "reason": (
                "matched finite run passes curve, validation, and supplied movement screens"
                if eligible
                else (
                    "candidate failed a curve, frozen-validation, or supplied movement screen"
                )
            ),
        },
    }


def _canonical_nonlegality_loss(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    """Reweight comparable objectives while intentionally excluding legality."""

    components = (
        ("dfm_ce_coeff", "dfm_ce_loss"),
        ("root_legal_ce_coeff", "root_legal_conditional_ce"),
        ("jepa_positive_coeff", "jepa_positive_loss"),
        ("target_sigreg_coeff", "jepa_sigreg_loss"),
        ("pred_sigreg_coeff", "jepa_pred_sigreg_loss"),
        ("wdl_coeff", "wdl_loss"),
    )
    total = 0.0
    for coefficient_name, metric_name in components:
        coefficient = _finite_float(
            config.get(coefficient_name),
            label=f"config {coefficient_name}",
        )
        if coefficient < 0.0:
            raise ValueError(f"config {coefficient_name} must be non-negative")
        metric = _finite_float(
            record.get(metric_name),
            label=f"canonical component {metric_name}",
        )
        total += coefficient * metric
    return total


def _identity_without_legality(
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    copied = json.loads(json.dumps(identity))
    model_and_loss = copied.get("model_and_loss")
    if not isinstance(model_and_loss, dict):
        raise ValueError("Comparison identity has no model_and_loss object")
    legality_coeff = _finite_float(
        model_and_loss.pop("legality_coeff", None),
        label="identity legality_coeff",
    )
    return copied, legality_coeff


def compare_legality_ablation(
    control_contract_path: Path,
    control_dir: Path,
    candidate_dir: Path,
    *,
    expected_control_legality_coeff: float = 2.0,
    expected_candidate_legality_coeff: float = 0.0,
    root_top1_floor: float = 0.445078125,
    control_movement_path: Path | None = None,
    candidate_movement_path: Path | None = None,
) -> dict[str, Any]:
    """Compare a preregistered legality ablation without comparing unlike losses."""

    contract = _json_file(control_contract_path)
    if contract.get("schema_version") != CONTROL_SCHEMA_VERSION:
        raise ValueError("Unsupported control contract schema")
    expected_update = int(contract["analysis_contract"]["expected_update"])
    control = _run_artifacts(control_dir)
    candidate = _run_artifacts(candidate_dir)

    if control["sha256"] != contract["control"]["artifact_sha256"]:
        raise ValueError("Control artifacts changed after the decision contract was frozen")
    control_identity = _comparison_identity(control["run_config"])
    candidate_identity = _comparison_identity(candidate["run_config"])
    if control_identity != contract["control"]["comparison_identity"]:
        raise ValueError("Control identity changed after the decision contract was frozen")
    control_matched, control_legality = _identity_without_legality(control_identity)
    candidate_matched, candidate_legality = _identity_without_legality(candidate_identity)
    if control_legality != expected_control_legality_coeff:
        raise ValueError(
            "Unexpected control legality coefficient: "
            f"{control_legality} != {expected_control_legality_coeff}"
        )
    if candidate_legality != expected_candidate_legality_coeff:
        raise ValueError(
            "Unexpected candidate legality coefficient: "
            f"{candidate_legality} != {expected_candidate_legality_coeff}"
        )
    if candidate_matched != control_matched:
        raise ValueError(
            "Candidate source/data/validation/model identity differs beyond legality_coeff"
        )

    control_records, candidate_records = _matched_records(
        control["records"],
        candidate["records"],
        expected_update=expected_update,
    )
    candidate_report = candidate["report"]
    if int(candidate_report.get("updates", -1)) != expected_update:
        raise ValueError("Candidate report update does not match the comparison boundary")
    candidate_validation = _latest_validation(candidate_report)
    if candidate_validation["update"] != expected_update:
        raise ValueError("Candidate frozen validation is not at the comparison boundary")

    control_config = control["run_config"].get("config")
    candidate_config = candidate["run_config"].get("config")
    if not isinstance(control_config, Mapping) or not isinstance(
        candidate_config,
        Mapping,
    ):
        raise ValueError("Both runs require serialized model configs")
    for coefficient_name in (
        "dfm_ce_coeff",
        "root_legal_ce_coeff",
        "jepa_positive_coeff",
        "target_sigreg_coeff",
        "pred_sigreg_coeff",
        "wdl_coeff",
    ):
        control_value = _finite_float(
            control_config.get(coefficient_name),
            label=f"control {coefficient_name}",
        )
        candidate_value = _finite_float(
            candidate_config.get(coefficient_name),
            label=f"candidate {coefficient_name}",
        )
        if candidate_value != control_value:
            raise ValueError(
                f"Canonical objective coefficient drift for {coefficient_name}"
            )

    bootstrap = contract["analysis_contract"]["paired_bootstrap"]
    bootstrap_start = int(bootstrap["start_update"])
    selected_pairs = [
        (control_row, candidate_row)
        for control_row, candidate_row in zip(
            control_records,
            candidate_records,
            strict=True,
        )
        if int(control_row["update"]) >= bootstrap_start
    ]
    canonical_deltas = [
        _canonical_nonlegality_loss(candidate_row, candidate_config)
        - _canonical_nonlegality_loss(control_row, control_config)
        for control_row, candidate_row in selected_pairs
    ]
    paired = {
        "canonical_nonlegality_loss": _paired_block_bootstrap(
            canonical_deltas,
            block_updates=int(bootstrap["block_updates"]),
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        )
    }
    for metric in (
        "dfm_ce_loss",
        "root_legal_conditional_ce",
        "jepa_positive_loss",
        "wdl_loss",
    ):
        paired[metric] = _paired_block_bootstrap(
            [
                float(candidate_row[metric]) - float(control_row[metric])
                for control_row, candidate_row in selected_pairs
            ],
            block_updates=int(bootstrap["block_updates"]),
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        )

    bounds = contract["analysis_contract"]["validation_guardrails"]
    validation_checks: dict[str, Any] = {}
    for metric, bound in bounds.items():
        value = candidate_validation["metrics"][metric]
        passed = (
            value <= float(bound["bound"])
            if bound["direction"] == "lower"
            else value >= float(bound["bound"])
        )
        validation_checks[metric] = {
            **bound,
            "candidate": value,
            "delta": value - float(bound["control"]),
            "passed": passed,
            "promotion_effect": metric != "first_legal_mass",
        }
    predictive_validation_passed = all(
        check["passed"]
        for check in validation_checks.values()
        if check["promotion_effect"]
    )
    control_diagnostics = _latest_validation_diagnostics(control["report"])
    candidate_diagnostics = _latest_validation_diagnostics(candidate_report)
    root_top1 = candidate_diagnostics.get("root_legal_top1_accuracy")
    if root_top1 is None:
        raise ValueError("Candidate frozen validation lacks root legal top-1")
    behavior_safe = root_top1 >= root_top1_floor
    canonical_ci = paired["canonical_nonlegality_loss"]["ci"]
    canonical_success = canonical_ci[1] < 0.0

    if (control_movement_path is None) != (candidate_movement_path is None):
        raise ValueError("Control and candidate movement reports must be supplied together")
    movement = None
    if control_movement_path is not None and candidate_movement_path is not None:
        movement = _compare_movement_reports(
            control_movement_path,
            candidate_movement_path,
            control_checkpoint_sha256=_checkpoint_state_sha256(
                control["report"],
                label="control",
            ),
            candidate_checkpoint_sha256=_checkpoint_state_sha256(
                candidate_report,
                label="candidate",
            ),
        )

    return {
        "schema_version": OBJECTIVE_ABLATION_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "control_contract": {
            "path": str(control_contract_path),
            "sha256": _sha256_file(control_contract_path),
        },
        "objective_difference": {
            "only_allowed_config_key": "legality_coeff",
            "control_legality_coeff": control_legality,
            "candidate_legality_coeff": candidate_legality,
            "total_training_loss_comparable": False,
            "first_legal_mass_promotion_effect": False,
        },
        "matched_contract": {
            "identity_without_legality_sha256": hashlib.sha256(
                _canonical_json(control_matched)
            ).hexdigest(),
            "updates": [1, expected_update],
            "bootstrap_start_update": bootstrap_start,
            "examples": [
                int(control_records[0]["examples"]),
                int(control_records[-1]["examples"]),
            ],
        },
        "paired_curve_differences": paired,
        "frozen_validation": {
            "candidate": candidate_validation,
            "checks": validation_checks,
            "predictive_checks_passed": predictive_validation_passed,
            "diagnostics": {
                metric: {
                    "direction": VALIDATION_DIAGNOSTICS[metric],
                    "control": control_diagnostics[metric],
                    "candidate": candidate_diagnostics[metric],
                    "delta": candidate_diagnostics[metric] - control_diagnostics[metric],
                }
                for metric in VALIDATION_DIAGNOSTICS
                if metric in control_diagnostics and metric in candidate_diagnostics
            },
        },
        "behavior_gate": {
            "root_legal_top1_accuracy": root_top1,
            "noninferiority_floor": root_top1_floor,
            "passed": behavior_safe,
        },
        "parameter_movement": movement,
        "decision": {
            "canonical_nonlegality_loss_improved": canonical_success,
            "predictive_validation_passed": predictive_validation_passed,
            "behavior_safe": behavior_safe,
            "eligible_for_movement_and_arena": (
                canonical_success and predictive_validation_passed and behavior_safe
            ),
            "phase2_authorized": False,
            "reason": (
                "canonical predictive learning and frozen behavior gates passed; "
                "movement and arena remain required"
                if canonical_success and predictive_validation_passed and behavior_safe
                else "candidate failed canonical predictive learning or frozen behavior"
            ),
        },
    }


def _serialized_policy_distill_coeff(run_dir: Path, *, label: str) -> float:
    run_config = _json_file(run_dir / "run_config.json")
    config = run_config.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} run requires a serialized model config")
    value = _finite_float(
        config.get("policy_distill_coeff", 0.0),
        label=f"{label} policy_distill_coeff",
    )
    if value < 0.0:
        raise ValueError(f"{label} policy_distill_coeff must be non-negative")
    return value


def compare_policy_distillation(
    control_contract_path: Path,
    control_dir: Path,
    candidate_dir: Path,
    *,
    expected_control_policy_distill_coeff: float = 0.0,
    expected_candidate_policy_distill_coeff: float = 1.0,
    root_top1_floor: float = 0.445078125,
    control_movement_path: Path | None = None,
    candidate_movement_path: Path | None = None,
) -> dict[str, Any]:
    """Compare stopped online-policy distillation on a matched legality-zero arm."""

    control_coeff = _serialized_policy_distill_coeff(control_dir, label="control")
    candidate_coeff = _serialized_policy_distill_coeff(candidate_dir, label="candidate")
    if control_coeff != expected_control_policy_distill_coeff:
        raise ValueError(
            "Unexpected control policy-distillation coefficient: "
            f"{control_coeff} != {expected_control_policy_distill_coeff}"
        )
    if candidate_coeff != expected_candidate_policy_distill_coeff:
        raise ValueError(
            "Unexpected candidate policy-distillation coefficient: "
            f"{candidate_coeff} != {expected_candidate_policy_distill_coeff}"
        )

    result = compare_legality_ablation(
        control_contract_path,
        control_dir,
        candidate_dir,
        expected_control_legality_coeff=0.0,
        expected_candidate_legality_coeff=0.0,
        root_top1_floor=root_top1_floor,
        control_movement_path=control_movement_path,
        candidate_movement_path=candidate_movement_path,
    )
    result["schema_version"] = POLICY_DISTILLATION_SCHEMA_VERSION
    result["objective_difference"] = {
        "only_allowed_config_key": "policy_distill_coeff",
        "control_policy_distill_coeff": control_coeff,
        "candidate_policy_distill_coeff": candidate_coeff,
        "shared_legality_coeff": 0.0,
        "total_training_loss_comparable": False,
        "canonical_predictive_loss_excludes": [
            "policy_distillation",
            "root_illegal_mass",
        ],
        "first_legal_mass_promotion_effect": False,
    }
    matched = result["matched_contract"]
    matched["matched_scientific_identity_sha256"] = matched.pop(
        "identity_without_legality_sha256"
    )
    paired = result["paired_curve_differences"]
    paired["canonical_predictive_loss"] = paired.pop(
        "canonical_nonlegality_loss"
    )
    candidate_report = _json_file(candidate_dir / "report.json")
    result["distillation_diagnostics"] = _latest_required_validation_metrics(
        candidate_report,
        POLICY_DISTILLATION_DIAGNOSTICS,
        label="candidate policy distillation",
    )
    decision = result["decision"]
    decision["canonical_predictive_loss_improved"] = decision.pop(
        "canonical_nonlegality_loss_improved"
    )
    decision["reason"] = (
        "canonical predictive learning and frozen behavior gates passed; "
        "movement and arena remain required"
        if decision["eligible_for_movement_and_arena"]
        else "candidate failed canonical predictive learning or frozen behavior"
    )
    return result


def _serialized_policy_teacher(run_dir: Path, *, label: str) -> dict[str, Any]:
    run_config = _json_file(run_dir / "run_config.json")
    config = run_config.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} run requires a serialized model config")
    mode = str(config.get("policy_distill_teacher_mode", "online"))
    state_sha256 = str(config.get("policy_distill_teacher_state_sha256", ""))
    if mode not in {"online", "checkpoint"}:
        raise ValueError(f"{label} policy teacher mode is invalid: {mode!r}")
    if mode == "online" and state_sha256:
        raise ValueError(f"{label} online policy teacher cannot bind a state SHA")
    teacher_record = run_config.get("policy_distill_teacher")
    if teacher_record is not None and not isinstance(teacher_record, Mapping):
        raise ValueError(f"{label} policy teacher record must be an object")
    if mode == "checkpoint":
        if (
            len(state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in state_sha256)
        ):
            raise ValueError(f"{label} checkpoint policy teacher requires a lowercase SHA-256")
        if not isinstance(teacher_record, Mapping):
            raise ValueError(f"{label} checkpoint policy teacher lacks provenance")
        if (
            teacher_record.get("mode") != "checkpoint"
            or teacher_record.get("checkpoint_state_sha256") != state_sha256
            or int(teacher_record.get("trainable_parameter_count", -1)) != 0
        ):
            raise ValueError(f"{label} checkpoint policy teacher provenance drift")
    elif isinstance(teacher_record, Mapping) and teacher_record.get("mode") != "online":
        raise ValueError(f"{label} online policy teacher provenance drift")
    return {
        "mode": mode,
        "checkpoint_state_sha256": state_sha256 or None,
        "record": dict(teacher_record) if isinstance(teacher_record, Mapping) else None,
    }


def compare_fixed_policy_distillation(
    control_contract_path: Path,
    control_dir: Path,
    candidate_dir: Path,
    *,
    expected_candidate_teacher_state_sha256: str,
    expected_control_policy_distill_coeff: float = 0.0,
    expected_candidate_policy_distill_coeff: float = 1.0,
    root_top1_floor: float = 0.455078125,
    control_movement_path: Path | None = None,
    candidate_movement_path: Path | None = None,
) -> dict[str, Any]:
    """Compare an immutable checkpoint teacher with a matched online/control arm."""

    if (
        len(expected_candidate_teacher_state_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_candidate_teacher_state_sha256
        )
    ):
        raise ValueError("Expected fixed policy teacher state must be a lowercase SHA-256")
    control_teacher = _serialized_policy_teacher(control_dir, label="control")
    candidate_teacher = _serialized_policy_teacher(candidate_dir, label="candidate")
    if control_teacher["mode"] != "online":
        raise ValueError("Fixed-teacher comparison requires an online/control reference")
    if candidate_teacher["mode"] != "checkpoint":
        raise ValueError("Fixed-teacher candidate must use checkpoint teacher mode")
    if (
        candidate_teacher["checkpoint_state_sha256"]
        != expected_candidate_teacher_state_sha256
    ):
        raise ValueError("Fixed-teacher candidate checkpoint state drift")

    result = compare_policy_distillation(
        control_contract_path,
        control_dir,
        candidate_dir,
        expected_control_policy_distill_coeff=(
            expected_control_policy_distill_coeff
        ),
        expected_candidate_policy_distill_coeff=(
            expected_candidate_policy_distill_coeff
        ),
        root_top1_floor=root_top1_floor,
        control_movement_path=control_movement_path,
        candidate_movement_path=candidate_movement_path,
    )
    result["schema_version"] = FIXED_POLICY_DISTILLATION_SCHEMA_VERSION
    objective = result["objective_difference"]
    changed_config_keys = [
        "policy_distill_teacher_mode",
        "policy_distill_teacher_state_sha256",
    ]
    if (
        objective["control_policy_distill_coeff"]
        != objective["candidate_policy_distill_coeff"]
    ):
        changed_config_keys.insert(0, "policy_distill_coeff")
    objective.pop("only_allowed_config_key", None)
    objective["allowed_changed_config_keys"] = changed_config_keys
    objective["control_policy_distill_teacher_mode"] = control_teacher["mode"]
    objective["candidate_policy_distill_teacher_mode"] = candidate_teacher["mode"]
    objective["candidate_policy_distill_teacher_state_sha256"] = (
        expected_candidate_teacher_state_sha256
    )
    result["teacher_contrast"] = {
        "control": control_teacher,
        "candidate": candidate_teacher,
    }
    candidate_report = _json_file(candidate_dir / "report.json")
    result["fixed_teacher_diagnostics"] = _latest_required_validation_metrics(
        candidate_report,
        FIXED_POLICY_TEACHER_DIAGNOSTICS,
        label="candidate fixed policy teacher",
    )
    canonical_ci = result["paired_curve_differences"][
        "canonical_predictive_loss"
    ]["ci"]
    curve_status = (
        "faster"
        if canonical_ci[1] < 0.0
        else "slower"
        if canonical_ci[0] > 0.0
        else "inconclusive"
    )
    decision = result["decision"]
    frozen_safe = bool(
        result["frozen_validation"]["predictive_checks_passed"]
        and result["behavior_gate"]["passed"]
    )
    decision["canonical_predictive_curve_status"] = curve_status
    decision["eligible_for_movement_and_arena"] = frozen_safe
    decision["phase2_authorized"] = False
    decision["reason"] = (
        "frozen predictive and DFM-root behavior gates passed; movement and "
        "the preregistered arena remain required"
        if frozen_safe
        else "candidate failed frozen predictive or DFM-root behavior gates"
    )
    return result


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-control")
    freeze.add_argument("run_dir", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--expected-update", type=int, default=THRESHOLD_ANCHORS[-1])
    freeze.add_argument("--anchors", type=int, nargs="+", default=list(THRESHOLD_ANCHORS))

    compare = subparsers.add_parser("compare")
    compare.add_argument("--control-contract", type=Path, required=True)
    compare.add_argument("--control-dir", type=Path, required=True)
    compare.add_argument("--candidate-dir", type=Path, required=True)
    compare.add_argument("--control-movement", type=Path)
    compare.add_argument("--candidate-movement", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    ablation = subparsers.add_parser("compare-legality-ablation")
    ablation.add_argument("--control-contract", type=Path, required=True)
    ablation.add_argument("--control-dir", type=Path, required=True)
    ablation.add_argument("--candidate-dir", type=Path, required=True)
    ablation.add_argument("--control-movement", type=Path)
    ablation.add_argument("--candidate-movement", type=Path)
    ablation.add_argument(
        "--root-top1-floor",
        type=float,
        default=0.445078125,
    )
    ablation.add_argument("--output", type=Path, required=True)

    distillation = subparsers.add_parser("compare-policy-distillation")
    distillation.add_argument("--control-contract", type=Path, required=True)
    distillation.add_argument("--control-dir", type=Path, required=True)
    distillation.add_argument("--candidate-dir", type=Path, required=True)
    distillation.add_argument("--control-movement", type=Path)
    distillation.add_argument("--candidate-movement", type=Path)
    distillation.add_argument(
        "--root-top1-floor",
        type=float,
        default=0.445078125,
    )
    distillation.add_argument("--output", type=Path, required=True)

    fixed = subparsers.add_parser("compare-fixed-policy-distillation")
    fixed.add_argument("--control-contract", type=Path, required=True)
    fixed.add_argument("--control-dir", type=Path, required=True)
    fixed.add_argument("--candidate-dir", type=Path, required=True)
    fixed.add_argument("--control-movement", type=Path)
    fixed.add_argument("--candidate-movement", type=Path)
    fixed.add_argument(
        "--expected-candidate-teacher-state-sha256",
        required=True,
    )
    fixed.add_argument(
        "--expected-control-policy-distill-coeff",
        type=float,
        default=0.0,
    )
    fixed.add_argument(
        "--root-top1-floor",
        type=float,
        default=0.455078125,
    )
    fixed.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-control":
        result = freeze_control(
            args.run_dir,
            expected_update=args.expected_update,
            anchors=args.anchors,
        )
    elif args.command == "compare":
        result = compare_candidate(
            args.control_contract,
            args.control_dir,
            args.candidate_dir,
            control_movement_path=args.control_movement,
            candidate_movement_path=args.candidate_movement,
        )
    elif args.command == "compare-legality-ablation":
        result = compare_legality_ablation(
            args.control_contract,
            args.control_dir,
            args.candidate_dir,
            root_top1_floor=args.root_top1_floor,
            control_movement_path=args.control_movement,
            candidate_movement_path=args.candidate_movement,
        )
    elif args.command == "compare-policy-distillation":
        result = compare_policy_distillation(
            args.control_contract,
            args.control_dir,
            args.candidate_dir,
            root_top1_floor=args.root_top1_floor,
            control_movement_path=args.control_movement,
            candidate_movement_path=args.candidate_movement,
        )
    else:
        result = compare_fixed_policy_distillation(
            args.control_contract,
            args.control_dir,
            args.candidate_dir,
            expected_candidate_teacher_state_sha256=(
                args.expected_candidate_teacher_state_sha256
            ),
            expected_control_policy_distill_coeff=(
                args.expected_control_policy_distill_coeff
            ),
            root_top1_floor=args.root_top1_floor,
            control_movement_path=args.control_movement,
            candidate_movement_path=args.candidate_movement,
        )
    _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
