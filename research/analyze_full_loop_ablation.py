"""Build the sealed decision artifact for the 1,024-update full-loop ablation.

The experiment compares a matched fixed-Hero1-teacher control with a candidate
whose DFM proposals feed JEPA and whose JEPA predictions feed a second DFM
planner.  This module intentionally consumes only compact local artifacts.  It
validates run identity, checks arena block hashes, applies the preregistered
decision rule, and emits deterministic JSON and Markdown; no checkpoint is
downloaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "chess-dfm-full-loop-ablation-decision-v1"
EXPECTED_UPDATES = 1_024
EXPECTED_EXAMPLES = 1_048_576
BOOTSTRAP_START_UPDATE = 101
BOOTSTRAP_BLOCK_UPDATES = 32
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_810
ANALYSIS_METRICS = (
    "loss",
    "dfm_ce_loss",
    "root_legal_conditional_ce",
    "wdl_loss",
    "jepa_positive_loss",
)
VALIDATION_METRICS: dict[str, str] = {
    "dfm_ce_loss": "lower",
    "root_legal_conditional_ce": "lower",
    "root_legal_top1_accuracy": "higher",
    "wdl_loss": "lower",
    "wdl_accuracy": "higher",
    "wdl_brier_score": "lower",
    "wdl_ece_15": "lower",
    "jepa_positive_loss": "lower",
}
MOVEMENT_GROUPS = ("all", "trunk", "embedding", "layers", "policy_head")
RECOVERY_UPDATES = [200, 400, 554, 700, 850, 1_024]
ADAPTER_PATH = "dfm_jepa_rollout_adapter.w"


@dataclass(frozen=True)
class ArenaSpec:
    directory: str
    candidate: str
    opponent: str
    candidate_passes: int
    opponent_passes: int
    primary: bool = True


ARENA_SPECS: dict[str, ArenaSpec] = {
    "candidate_8_vs_candidate_1": ArenaSpec(
        "candidate8_vs_candidate1", "candidate", "candidate", 8, 1
    ),
    "candidate_8_vs_control_1": ArenaSpec(
        "candidate8_vs_control1", "candidate", "control", 8, 1
    ),
    "candidate_1_vs_control_1": ArenaSpec(
        "candidate1_vs_control1", "candidate", "control", 1, 1
    ),
    "candidate_8_vs_control_8": ArenaSpec(
        "candidate8_vs_control8", "candidate", "control", 8, 8
    ),
    "control_1_vs_control_8_exploratory": ArenaSpec(
        "control1_vs_control8_exploratory",
        "control",
        "control",
        1,
        8,
        primary=False,
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    _require(bool(records), f"No records in {path}")
    return records


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    _require(bool(sorted_values), "Cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_block_bootstrap(
    deltas: Sequence[float],
    *,
    block_updates: int = BOOTSTRAP_BLOCK_UPDATES,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Circular paired moving-block bootstrap for a mean curve delta."""

    _require(len(deltas) >= block_updates, "Need at least one complete block")
    _require(replicates >= 100, "Need at least 100 bootstrap replicates")
    block_count = math.ceil(len(deltas) / block_updates)
    randomizer = random.Random(seed)
    means: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for _ in range(block_count):
            start = randomizer.randrange(len(deltas))
            sample.extend(
                deltas[(start + offset) % len(deltas)]
                for offset in range(block_updates)
            )
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


def _checkpoint_state_sha256(report: Mapping[str, Any], *, label: str) -> str:
    checkpoint = report.get("checkpoint")
    state = checkpoint.get("state") if isinstance(checkpoint, Mapping) else None
    value = state.get("sha256") if isinstance(state, Mapping) else None
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} has no valid checkpoint-state SHA-256",
    )
    return value


def _load_run(run_dir: Path, *, label: str) -> dict[str, Any]:
    names = (
        "metrics.jsonl",
        "validation_metrics.jsonl",
        "run_config.json",
        "report.json",
        "loss_summary.json",
        "compact_manifest.json",
    )
    for name in names:
        _require((run_dir / name).is_file(), f"{label} is missing {name}")
    records = _jsonl(run_dir / "metrics.jsonl")
    _require(len(records) == EXPECTED_UPDATES, f"{label} must have 1,024 updates")
    for expected_update, record in enumerate(records, 1):
        update = int(record.get("update", -1))
        examples = int(record.get("examples", -1))
        _require(update == expected_update, f"{label} has a non-contiguous curve")
        _require(
            examples == expected_update * 1_024,
            f"{label} has an unexpected example count at update {update}",
        )
        _require(
            not bool(record.get("optimizer_skipped_nonfinite", False)),
            f"{label} skipped a non-finite update at {update}",
        )
        for metric in ANALYSIS_METRICS:
            _finite(record.get(metric), label=f"{label} update {update} {metric}")

    report = _json_object(run_dir / "report.json")
    config = _json_object(run_dir / "run_config.json")
    _require(int(report.get("updates", -1)) == EXPECTED_UPDATES, "Report update mismatch")
    _require(int(report.get("examples", -1)) == EXPECTED_EXAMPLES, "Report example mismatch")
    recovery = report.get("recovery_checkpoints")
    _require(isinstance(recovery, list), f"{label} has no recovery list")
    _require(
        [int(item["optimizer_update"]) for item in recovery] == RECOVERY_UPDATES,
        f"{label} recovery checkpoints do not match the frozen plan",
    )
    for item in recovery:
        state = item.get("state")
        _require(isinstance(state, Mapping), f"{label} has malformed recovery state")
        _require(
            isinstance(state.get("sha256"), str) and len(state["sha256"]) == 64,
            f"{label} has an unsealed recovery state",
        )

    validations = _jsonl(run_dir / "validation_metrics.jsonl")
    validation = max(validations, key=lambda row: int(row.get("update", -1)))
    _require(int(validation.get("update", -1)) == EXPECTED_UPDATES, "Validation update mismatch")
    _require(
        int(validation.get("evaluation_examples", -1)) == 8_192,
        "Terminal validation must contain 8,192 examples",
    )
    validation_metrics = validation.get("metrics")
    _require(isinstance(validation_metrics, Mapping), "Validation metrics are malformed")
    for metric in VALIDATION_METRICS:
        _finite(validation_metrics.get(metric), label=f"{label} validation {metric}")

    live_validation = report.get("live_validation")
    live_records = (
        live_validation.get("records")
        if isinstance(live_validation, Mapping)
        else None
    )
    _require(isinstance(live_records, list) and live_records, "Report validation is absent")
    report_validation = max(live_records, key=lambda row: int(row.get("update", -1)))
    _require(
        report_validation.get("metrics") == validation_metrics,
        f"{label} report and validation JSONL disagree",
    )
    return {
        "directory": str(run_dir),
        "records": records,
        "config": config,
        "report": report,
        "validation": validation,
        "checkpoint_state_sha256": _checkpoint_state_sha256(report, label=label),
        "input_sha256": {name: _sha256_file(run_dir / name) for name in names},
    }


def _validate_matched_configuration(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    control_config = control["config"]
    candidate_config = candidate["config"]
    for section in (
        "recipe",
        "data",
        "live_validation",
        "optimizer_policy",
        "compile_regions",
        "execution",
    ):
        _require(
            control_config.get(section) == candidate_config.get(section),
            f"Matched-run section {section!r} differs",
        )

    control_model = copy.deepcopy(control_config["config"])
    candidate_model = copy.deepcopy(candidate_config["config"])
    control_mode = control_model.pop("dfm_closed_loop_mode", None)
    candidate_mode = candidate_model.pop("dfm_closed_loop_mode", None)
    _require(control_mode == "none", "Control is not the no-loop arm")
    _require(
        candidate_mode == "predicted_jepa_tokens",
        "Candidate is not the predicted-JEPA full loop",
    )
    _require(control_model == candidate_model, "Model/loss configuration has extra deltas")

    control_args = copy.deepcopy(control_config["args"])
    candidate_args = copy.deepcopy(candidate_config["args"])
    control_args.pop("output_dir", None)
    candidate_args.pop("output_dir", None)
    control_arg_mode = control_args.pop("dfm_closed_loop_mode", None)
    candidate_arg_mode = candidate_args.pop("dfm_closed_loop_mode", None)
    _require(control_arg_mode in (None, "none"), "Control CLI loop mode is wrong")
    _require(
        candidate_arg_mode == "predicted_jepa_tokens",
        "Candidate CLI loop mode is wrong",
    )
    _require(control_args == candidate_args, "Matched CLI arguments have extra deltas")

    control_source = copy.deepcopy(control_config["source"])
    candidate_source = copy.deepcopy(candidate_config["source"])
    control_fresh = control_source.pop("fresh")
    candidate_fresh = candidate_source.pop("fresh")
    _require(control_source == candidate_source, "Raw-source identities differ")
    control_paths = list(control_fresh["paths"])
    candidate_paths = list(candidate_fresh["paths"])
    _require(ADAPTER_PATH not in control_paths, "Control unexpectedly has a loop adapter")
    _require(
        candidate_paths.count(ADAPTER_PATH) == 1,
        "Candidate does not have exactly one loop adapter",
    )
    candidate_paths.remove(ADAPTER_PATH)
    _require(candidate_paths == control_paths, "Fresh parameter sets have extra deltas")
    _require(
        int(candidate_fresh["leaf_count"]) == int(control_fresh["leaf_count"]) + 1,
        "Candidate fresh leaf count does not isolate the adapter",
    )
    for key in ("schema_version", "seed", "zero_initialized_residual"):
        _require(
            control_fresh.get(key) == candidate_fresh.get(key),
            f"Fresh initialization field {key!r} differs",
        )
    return {
        "matched": True,
        "common_updates": EXPECTED_UPDATES,
        "common_examples": EXPECTED_EXAMPLES,
        "only_scientific_deltas": [
            "config.dfm_closed_loop_mode: none -> predicted_jepa_tokens",
            f"fresh parameter added: {ADAPTER_PATH}",
        ],
    }


def _training_comparison(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    control_rows = control["records"]
    candidate_rows = candidate["records"]
    _require(len(control_rows) == len(candidate_rows), "Curve lengths differ")
    for left, right in zip(control_rows, candidate_rows, strict=True):
        _require(left["examples"] == right["examples"], "Example prefixes differ")

    metrics: dict[str, Any] = {}
    for metric in ANALYSIS_METRICS:
        full_control = [float(row[metric]) for row in control_rows]
        full_candidate = [float(row[metric]) for row in candidate_rows]
        tail_control = full_control[-64:]
        tail_candidate = full_candidate[-64:]
        bootstrap_deltas = [
            float(right[metric]) - float(left[metric])
            for left, right in zip(control_rows, candidate_rows, strict=True)
            if int(left["update"]) >= BOOTSTRAP_START_UPDATE
        ]
        metrics[metric] = {
            "full_mean": {
                "control": statistics.fmean(full_control),
                "candidate": statistics.fmean(full_candidate),
                "candidate_minus_control": statistics.fmean(full_candidate)
                - statistics.fmean(full_control),
            },
            "last_64_mean": {
                "control": statistics.fmean(tail_control),
                "candidate": statistics.fmean(tail_candidate),
                "candidate_minus_control": statistics.fmean(tail_candidate)
                - statistics.fmean(tail_control),
            },
            "paired_moving_block_bootstrap_updates_101_1024": paired_block_bootstrap(
                bootstrap_deltas
            ),
        }

    loop_fields = (
        "dfm_closed_loop_context_rms",
        "dfm_closed_loop_dfm_ce_improvement",
        "jepa_feedback_preliminary_dfm_ce_loss",
    )
    loop_trajectory: dict[str, Any] = {}
    for metric in loop_fields:
        values = [
            _finite(row.get(metric), label=f"candidate training {metric}")
            for row in candidate_rows
        ]
        loop_trajectory[metric] = {
            "full_mean": statistics.fmean(values),
            "last_64_mean": statistics.fmean(values[-64:]),
            "terminal": values[-1],
        }
    return {
        "bootstrap_contract": {
            "start_update": BOOTSTRAP_START_UPDATE,
            "block_updates": BOOTSTRAP_BLOCK_UPDATES,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": "paired circular moving-block bootstrap",
        },
        "metrics": metrics,
        "candidate_loop_trajectory": loop_trajectory,
    }


def _validation_comparison(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    control_metrics = control["validation"]["metrics"]
    candidate_metrics = candidate["validation"]["metrics"]
    comparison: dict[str, Any] = {}
    for metric, direction in VALIDATION_METRICS.items():
        left = _finite(control_metrics[metric], label=f"control {metric}")
        right = _finite(candidate_metrics[metric], label=f"candidate {metric}")
        comparison[metric] = {
            "direction": direction,
            "control": left,
            "candidate": right,
            "candidate_minus_control": right - left,
        }
    comparison["candidate_first_planner_dfm_ce_loss"] = {
        "candidate": _finite(
            candidate_metrics["jepa_feedback_preliminary_dfm_ce_loss"],
            label="candidate preliminary DFM CE",
        )
    }
    comparison["candidate_second_minus_first_planner_dfm_ce"] = {
        "candidate": -_finite(
            candidate_metrics["dfm_closed_loop_dfm_ce_improvement"],
            label="candidate loop DFM CE improvement",
        ),
        "improvement_positive": _finite(
            candidate_metrics["dfm_closed_loop_dfm_ce_improvement"],
            label="candidate loop DFM CE improvement",
        ),
    }
    comparison["candidate_closed_loop_context_rms"] = {
        "candidate": _finite(
            candidate_metrics["dfm_closed_loop_context_rms"],
            label="candidate context RMS",
        )
    }
    return {
        "update": EXPECTED_UPDATES,
        "evaluation_examples": 8_192,
        "metrics": comparison,
    }


def _latent_health(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = candidate["validation"]["metrics"]
    horizons: list[dict[str, Any]] = []
    all_pass = True
    for horizon in range(1, 9):
        suffix = f"by_horizon_h{horizon}"
        jepa = _finite(metrics[f"jepa_mse_{suffix}"], label=f"h{horizon} JEPA")
        zero = _finite(metrics[f"zero_mse_{suffix}"], label=f"h{horizon} zero")
        identity = _finite(
            metrics[f"identity_mse_{suffix}"], label=f"h{horizon} identity"
        )
        shuffled = _finite(
            metrics[f"shuffled_mse_{suffix}"], label=f"h{horizon} shuffled"
        )
        action_shuffled = _finite(
            metrics[f"action_shuffled_mse_{suffix}"],
            label=f"h{horizon} action-shuffled",
        )
        rms_ratio = _finite(
            metrics[f"pred_target_rms_ratio_{suffix}"], label=f"h{horizon} RMS ratio"
        )
        rank_ratio = _finite(
            metrics[f"pred_target_effective_rank_ratio_{suffix}"],
            label=f"h{horizon} rank ratio",
        )
        checks = {
            "beats_zero": jepa < zero,
            "beats_identity": jepa < identity,
            "beats_shuffled": jepa < shuffled,
            "beats_action_shuffled": jepa < action_shuffled,
            "rms_ratio_in_preregistered_range": 0.5 <= rms_ratio <= 2.0,
            "no_material_effective_rank_collapse": rank_ratio >= 0.5,
        }
        passed = all(checks.values())
        all_pass = all_pass and passed
        horizons.append(
            {
                "horizon": horizon,
                "jepa_mse": jepa,
                "zero_mse": zero,
                "identity_mse": identity,
                "shuffled_mse": shuffled,
                "action_shuffled_mse": action_shuffled,
                "pred_target_rms_ratio": rms_ratio,
                "pred_target_effective_rank_ratio": rank_ratio,
                "checks": checks,
                "passed": passed,
            }
        )
    return {
        "passed": all_pass,
        "horizons": horizons,
        "rank_collapse_operationalization": (
            "effective-rank ratio below 0.5; the preregistration named material "
            "rank collapse but did not set a tighter numeric boundary"
        ),
    }


def _systems_diagnostic(diagnostic_dir: Path) -> dict[str, Any]:
    report_path = diagnostic_dir / "report.json"
    config_path = diagnostic_dir / "run_config.json"
    metrics_path = diagnostic_dir / "metrics.jsonl"
    report = _json_object(report_path)
    config = _json_object(config_path)
    records = _jsonl(metrics_path)
    _require(int(report.get("updates", -1)) == 10, "Diagnostic is not 10 updates")
    useful_rates = [
        _finite(row["examples_per_second_step"], label="diagnostic throughput")
        for row in records
        if int(row["update"]) >= 3
    ]
    median_rate = statistics.median(useful_rates)
    monitor = report.get("gpu_monitor")
    _require(isinstance(monitor, Mapping), "Diagnostic GPU monitor is absent")
    total_bytes = _finite(
        monitor.get("memory_total_mib_max"), label="diagnostic total HBM MiB"
    ) * 1024 * 1024
    reserved_bytes = _finite(
        report.get("gpu_peak_memory_reserved_bytes"), label="diagnostic peak HBM"
    )
    teacher = config.get("policy_distill_teacher")
    _require(isinstance(teacher, Mapping), "Diagnostic teacher identity is absent")
    last = records[-1]
    checks = {
        "ten_updates": len(records) == 10,
        "all_finite_no_skips": all(
            not bool(row.get("optimizer_skipped_nonfinite", False)) for row in records
        ),
        "immutable_teacher_exact": (
            teacher.get("mode") == "checkpoint"
            and teacher.get("checkpoint_state_sha256")
            == "e518083ee5476e7ff15e7c34caf9c0505f8fb7d330ea5a6882e4e9e5c1ef1912"
            and int(teacher.get("trainable_parameter_count", -1)) == 0
        ),
        "loop_active": float(last.get("dfm_closed_loop_active", 0.0)) == 1.0,
        "positive_context_rms": _finite(
            last.get("dfm_closed_loop_context_rms"), label="diagnostic context RMS"
        )
        > 0.0,
        "hbm_headroom_at_least_one_gib": total_bytes - reserved_bytes >= 1024**3,
        "median_useful_examples_per_second_at_least_150": median_rate >= 150.0,
        "no_diagnostic_checkpoint": (
            report.get("checkpoint") is None
            and report.get("recovery_checkpoints") == []
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "median_useful_examples_per_second_updates_3_10": median_rate,
        "peak_reserved_bytes": int(reserved_bytes),
        "physical_hbm_bytes": int(total_bytes),
        "physical_hbm_headroom_bytes": int(total_bytes - reserved_bytes),
        "input_sha256": {
            "report.json": _sha256_file(report_path),
            "run_config.json": _sha256_file(config_path),
            "metrics.jsonl": _sha256_file(metrics_path),
        },
    }


def _movement_summary(
    control_path: Path,
    candidate_path: Path,
    *,
    control_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
) -> dict[str, Any]:
    control = _json_object(control_path)
    candidate = _json_object(candidate_path)
    for label, report, expected_sha in (
        ("control", control, control_checkpoint_sha256),
        ("candidate", candidate, candidate_checkpoint_sha256),
    ):
        _require(
            report.get("schema_version") == "bt4-raw-hero-parameter-diff-v1",
            f"Unsupported {label} movement schema",
        )
        _require(
            report["models"]["hero_state_sha256"] == expected_sha,
            f"{label} movement report is bound to the wrong checkpoint",
        )
    _require(
        control["models"]["raw_mapping_sha256"]
        == candidate["models"]["raw_mapping_sha256"],
        "Movement reports use different Raw mappings",
    )
    groups: dict[str, Any] = {}
    for group in MOVEMENT_GROUPS:
        left = _finite(
            control["group_metrics"][group]["relative_delta_l2"],
            label=f"control {group} movement",
        )
        right = _finite(
            candidate["group_metrics"][group]["relative_delta_l2"],
            label=f"candidate {group} movement",
        )
        groups[group] = {
            "control_relative_delta_l2": left,
            "candidate_relative_delta_l2": right,
            "candidate_to_control_ratio": right / max(left, 1e-30),
        }
    control_adapter = control["closed_loop_adapter_movement"]
    candidate_adapter = candidate["closed_loop_adapter_movement"]
    adapter = candidate_adapter.get("parameters", {}).get(ADAPTER_PATH)
    _require(
        control_adapter.get("status") == "absent_by_config"
        and control_adapter.get("parameters") == {},
        "Control adapter audit is not absent-by-config",
    )
    _require(isinstance(adapter, Mapping), "Candidate adapter movement is absent")
    adapter_summary = {
        key: adapter[key]
        for key in (
            "parameter_count",
            "changed_fraction",
            "relative_delta_l2",
            "cosine",
            "delta_l2",
            "delta_rms",
            "initial_l2",
            "trained_l2",
            "delta_absolute_max",
        )
    }
    passed = (
        candidate_adapter.get("status") == "measured"
        and int(adapter_summary["parameter_count"]) == 262_144
        and float(adapter_summary["changed_fraction"]) == 1.0
        and float(adapter_summary["relative_delta_l2"]) > 0.0
    )
    return {
        "passed": passed,
        "groups": groups,
        "candidate_loop_adapter": adapter_summary,
        "interpretation": (
            "The loop-specific adapter moved strongly while candidate trunk movement "
            "was slightly lower than the matched control; the loop effect is not "
            "explained by runaway encoder drift."
        ),
        "input_sha256": {
            "control": _sha256_file(control_path),
            "candidate": _sha256_file(candidate_path),
        },
    }


def _single_child_file(directory: Path, name: str) -> Path:
    matches = sorted(directory.glob(f"*/{name}"))
    _require(len(matches) == 1, f"Expected one {name} below {directory}")
    return matches[0]


def _arena_summary(
    arena_root: Path,
    spec: ArenaSpec,
    *,
    checkpoint_sha256: Mapping[str, str],
) -> dict[str, Any]:
    directory = arena_root / spec.directory
    state_path = _single_child_file(directory, "state.json")
    modal_path = state_path.parent / "modal_arena.json"
    _require(modal_path.is_file(), f"Arena {spec.directory} lacks modal_arena.json")
    state = _json_object(state_path)
    modal = _json_object(modal_path)
    _require(state.get("status") == "complete", f"Arena {spec.directory} is incomplete")
    _require(
        state.get("schema_version") == "chess-dfm-relative-arena-run-v3",
        f"Arena {spec.directory} has the wrong schema",
    )
    contract = state.get("contract")
    aggregate = state.get("aggregate")
    _require(isinstance(contract, Mapping), "Arena contract is malformed")
    _require(isinstance(aggregate, Mapping), "Arena aggregate is malformed")
    run = contract["run"]
    _require(int(run["candidate_refinement_passes"]) == spec.candidate_passes, "Candidate pass mismatch")
    _require(int(run["opponent_refinement_passes"]) == spec.opponent_passes, "Opponent pass mismatch")
    _require(int(run["pair_count"]) == 128, "Arena pair count mismatch")
    _require(int(run["opening_start_index"]) == 256, "Arena opening offset mismatch")
    _require(int(run["additional_ply_cap"]) == 256, "Arena ply cap mismatch")
    _require(bool(run["deterministic_greedy_policy"]), "Arena is not deterministic")
    _require(contract["tier"]["name"] == "hero_development", "Arena tier mismatch")
    models = contract["models"]
    _require(
        models["candidate"]["state"]["sha256"] == checkpoint_sha256[spec.candidate],
        "Arena candidate checkpoint mismatch",
    )
    _require(
        models["opponent"]["state"]["sha256"] == checkpoint_sha256[spec.opponent],
        "Arena opponent checkpoint mismatch",
    )
    codec = contract["codec_capabilities"]
    codec_complete = all(
        bool(codec[side]["complete_legal_move_coverage"])
        and codec[side]["unrepresentable_move_classes"] == []
        for side in ("candidate", "opponent")
    )
    _require(codec_complete, "Arena move codec is incomplete")
    _require(aggregate["fault_counts"] == {}, "Arena contains faults")
    _require(int(aggregate["cap_draw_count"]) == 0, "Arena contains cap draws")
    _require(int(aggregate["game_count"]) == 256, "Arena game count mismatch")
    _require(int(aggregate["pair_count"]) == 128, "Arena aggregate pair mismatch")
    pair_scores = [_finite(value, label="pair score") for value in aggregate["pair_scores"]]
    _require(len(pair_scores) == 128, "Arena pair-score vector is incomplete")
    _require(all(0.0 <= value <= 2.0 for value in pair_scores), "Invalid pair score")
    interval = aggregate["pair_aware_logistic_interval"]
    score = _finite(interval["score"], label="arena score")
    _require(
        math.isclose(score, sum(pair_scores) / 256.0, abs_tol=1e-12),
        "Arena score does not match pair scores",
    )
    for model in aggregate["models"].values():
        coverage = model["coverage"]
        _require(float(coverage["representable_fraction"]) == 1.0, "Coverage is incomplete")
        _require(int(coverage["incomplete_coverage_positions"]) == 0, "Coverage failures exist")
        _require(int(model["fault_losses"]) == 0, "Model has fault losses")

    blocks = state.get("blocks")
    _require(isinstance(blocks, list) and len(blocks) == 8, "Arena block set is incomplete")
    block_hashes: dict[str, str] = {}
    for index, descriptor in enumerate(blocks):
        expected_start = index * 16
        _require(int(descriptor["start_pair"]) == expected_start, "Arena block order mismatch")
        _require(int(descriptor["pair_count"]) == 16, "Arena block size mismatch")
        block_path = state_path.parent / "blocks" / Path(descriptor["path"]).name
        _require(block_path.is_file(), f"Missing local arena block {block_path.name}")
        block_hash = _sha256_file(block_path)
        _require(block_hash == descriptor["sha256"], "Arena block byte hash mismatch")
        block = _json_object(block_path)
        _require(block["contract_sha256"] == state["contract_sha256"], "Block contract mismatch")
        _require(block["payload_sha256"] == descriptor["payload_sha256"], "Block payload mismatch")
        _require(int(block["start_pair"]) == expected_start, "Block start mismatch")
        _require(int(block["pair_count"]) == 16, "Block pair count mismatch")
        block_hashes[block_path.name] = block_hash

    _require(int(modal["pair_count"]) == 128, "Modal wrapper pair mismatch")
    _require(int(modal["opening_start_index"]) == 256, "Modal wrapper offset mismatch")
    _require(
        int(modal["candidate_refinement_passes"]) == spec.candidate_passes,
        "Modal candidate pass mismatch",
    )
    _require(
        int(modal["opponent_refinement_passes"]) == spec.opponent_passes,
        "Modal opponent pass mismatch",
    )
    return {
        "scientific_role": "preregistered primary" if spec.primary else "exploratory",
        "state_path": str(state_path),
        "status": "complete",
        "candidate": spec.candidate,
        "opponent": spec.opponent,
        "candidate_refinement_passes": spec.candidate_passes,
        "opponent_refinement_passes": spec.opponent_passes,
        "pair_count": 128,
        "game_count": 256,
        "opening_indices": [256, 383],
        "score": score,
        "score_interval_95": [float(interval["score_lower"]), float(interval["score_upper"])],
        "relative_elo": float(interval["elo"]),
        "relative_elo_interval_95": [float(interval["elo_lower"]), float(interval["elo_upper"])],
        "pentanomial": aggregate["pentanomial"]["counts"],
        "zero_faults": True,
        "zero_cap_draws": True,
        "complete_move_codec_coverage": True,
        "input_sha256": {
            "state.json": _sha256_file(state_path),
            "modal_arena.json": _sha256_file(modal_path),
            "blocks": block_hashes,
        },
    }


def evaluate_preregistered_gates(
    arenas: Mapping[str, Mapping[str, Any]],
    *,
    latent_health_passed: bool,
    systems_passed: bool,
    movement_passed: bool,
) -> dict[str, Any]:
    own = arenas["candidate_8_vs_candidate_1"]
    versus_control = arenas["candidate_8_vs_control_1"]
    one_pass = arenas["candidate_1_vs_control_1"]
    primary_arenas = [value for value in arenas.values() if value["scientific_role"] == "preregistered primary"]
    all_arenas_healthy = all(
        value["zero_faults"] and value["complete_move_codec_coverage"]
        for value in primary_arenas
    )
    gates = {
        "candidate_eight_pass_beats_own_one_pass": {
            "passed": float(own["score"]) > 0.5,
            "evidence": {"score": own["score"], "required": "> 0.5"},
        },
        "candidate_eight_pass_recovers_to_control_one_pass": {
            "passed": (
                float(versus_control["score"]) >= 0.5
                or float(versus_control["score_interval_95"][0]) >= 0.45
            ),
            "evidence": {
                "score": versus_control["score"],
                "score_interval_95": versus_control["score_interval_95"],
                "required": "point >= 0.5 OR lower interval bound >= 0.45",
            },
        },
        "candidate_one_pass_not_materially_damaged": {
            "passed": float(one_pass["score_interval_95"][1]) >= 0.45,
            "evidence": {
                "score": one_pass["score"],
                "score_interval_95": one_pass["score_interval_95"],
                "required": "upper interval bound >= 0.45",
                "interpretation": (
                    "An interval wholly below 0.45 supports the preregistered "
                    "material-deficit rejection condition."
                ),
            },
        },
        "offline_latent_and_systems_health": {
            "passed": (
                latent_health_passed
                and systems_passed
                and movement_passed
                and all_arenas_healthy
            ),
            "evidence": {
                "latent_health_passed": latent_health_passed,
                "systems_passed": systems_passed,
                "movement_audit_passed": movement_passed,
                "all_primary_arenas_zero_fault_and_codec_complete": all_arenas_healthy,
            },
        },
    }
    mechanism_success = all(value["passed"] for value in gates.values())
    return {
        "mechanism_success": mechanism_success,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value["passed"]],
        "decision": (
            "retain_architecture_but_reject_current_training_composition"
            if not mechanism_success
            else "retain_for_newly_bounded_longer_comparison"
        ),
    }


def _input_event_times(*objects: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for value in objects:
        for key in ("created_utc", "completed_utc", "updated_at_utc", "created_at_utc"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                values.append(candidate)
    return sorted(values)


def build_analysis(root: Path) -> dict[str, Any]:
    phase = root / "research" / "analysis" / "hero_v2_phase1"
    u1024 = phase / "u1024"
    control_dir = u1024 / "v2_fp32_1_12_legality0_hero1distill1"
    candidate_dir = u1024 / "v2_fp32_1_12_legality0_hero1distill1_closedloop"
    prereg_path = phase / "v2_fp32_1_12_legality0_hero1distill1_full_loop_u1024-preregister.json"
    exploratory_prereg_path = u1024 / "full_loop_control_refinement_exploratory_preregister.json"
    prereg = _json_object(prereg_path)
    exploratory_prereg = _json_object(exploratory_prereg_path)
    _require(
        prereg.get("schema_version")
        == "chess-dfm-hero-v2-full-loop-u1024-preregister-v1",
        "Unexpected full-loop preregistration schema",
    )

    control = _load_run(control_dir, label="control")
    candidate = _load_run(candidate_dir, label="candidate")
    matched = _validate_matched_configuration(control, candidate)
    training = _training_comparison(control, candidate)
    validation = _validation_comparison(control, candidate)
    latent = _latent_health(candidate)
    diagnostic = _systems_diagnostic(
        phase / "diagnostics" / "v2_fp32_1_12_legality0_hero1distill1_closedloop"
    )

    movement_root = u1024 / "full_loop_movement"
    control_movement = _single_child_file(movement_root / "control", "parameter_diff.json")
    candidate_movement = _single_child_file(movement_root / "candidate", "parameter_diff.json")
    movement = _movement_summary(
        control_movement,
        candidate_movement,
        control_checkpoint_sha256=control["checkpoint_state_sha256"],
        candidate_checkpoint_sha256=candidate["checkpoint_state_sha256"],
    )

    checkpoint_sha256 = {
        "control": control["checkpoint_state_sha256"],
        "candidate": candidate["checkpoint_state_sha256"],
    }
    arena_root = u1024 / "full_loop_arenas"
    arenas = {
        name: _arena_summary(
            arena_root, spec, checkpoint_sha256=checkpoint_sha256
        )
        for name, spec in ARENA_SPECS.items()
    }
    gate_result = evaluate_preregistered_gates(
        arenas,
        latent_health_passed=latent["passed"],
        systems_passed=diagnostic["passed"],
        movement_passed=movement["passed"],
    )
    event_times = _input_event_times(
        prereg,
        exploratory_prereg,
        control["report"],
        candidate["report"],
        _json_object(control_movement),
        _json_object(candidate_movement),
        *(
            _json_object(Path(result["state_path"]))
            for result in arenas.values()
        ),
    )
    metered_before = float(prereg["budget"]["metered_before_experiment_usd"])
    metered_before_arenas = 26.48728616
    metered_final = 27.10744893
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_complete_utc": event_times[-1] if event_times else None,
        "question": prereg["question"],
        "headline": {
            "mechanism_success_under_exact_preregistered_rule": gate_result[
                "mechanism_success"
            ],
            "mechanistic_result": (
                "Predicted-JEPA recurrence makes iterative inference useful and "
                "recovers the candidate to control one-pass parity."
            ),
            "blocking_result": (
                "The current final-only policy objective materially damages the "
                "one-pass base; its entire 95% pair-aware interval is below 0.45."
            ),
            "recommendation": (
                "Retain the full-loop architecture for a bounded multi-exit training "
                "ablation; do not launch a Hero-scale run with this loss composition."
            ),
        },
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256_file(prereg_path),
            "exploratory_arena_preregistration": {
                "path": str(exploratory_prereg_path),
                "sha256": _sha256_file(exploratory_prereg_path),
                "scientific_role": "mechanism clarification only",
            },
        },
        "matched_training_contract": matched,
        "runs": {
            "control": {
                "directory": control["directory"],
                "checkpoint_state_sha256": control["checkpoint_state_sha256"],
                "examples_per_second_end_to_end": control["report"][
                    "examples_per_second_end_to_end"
                ],
                "gpu_peak_memory_reserved_bytes": control["report"][
                    "gpu_peak_memory_reserved_bytes"
                ],
                "input_sha256": control["input_sha256"],
            },
            "candidate": {
                "directory": candidate["directory"],
                "checkpoint_state_sha256": candidate["checkpoint_state_sha256"],
                "examples_per_second_end_to_end": candidate["report"][
                    "examples_per_second_end_to_end"
                ],
                "gpu_peak_memory_reserved_bytes": candidate["report"][
                    "gpu_peak_memory_reserved_bytes"
                ],
                "input_sha256": candidate["input_sha256"],
            },
        },
        "training_curves": training,
        "frozen_validation": validation,
        "latent_health": latent,
        "systems_diagnostic": diagnostic,
        "parameter_movement": movement,
        "arenas": arenas,
        "preregistered_decision": gate_result,
        "exploratory_interpretation": {
            "control_one_pass_vs_control_eight_pass": arenas[
                "control_1_vs_control_8_exploratory"
            ],
            "conclusion": (
                "Ordinary iterative unmasking harms the control. The candidate's "
                "fair-compute advantage therefore depends on JEPA feedback, rather "
                "than arising from generic repetition of the planner."
            ),
        },
        "cost": {
            "workspace_budget_usd": float(prereg["budget"]["workspace_budget_usd"]),
            "repository_new_launch_stop_usd": float(
                prereg["budget"]["new_launch_stop_usd"]
            ),
            "metered_before_experiment_usd": metered_before,
            "metered_before_arena_and_movement_usd": metered_before_arenas,
            "metered_final_usd": metered_final,
            "whole_experiment_delta_usd": metered_final - metered_before,
            "arena_and_movement_delta_usd": metered_final - metered_before_arenas,
            "billed_cost_usd_at_final_query": 0.0,
            "remaining_to_repository_stop_usd": float(
                prereg["budget"]["new_launch_stop_usd"]
            )
            - metered_final,
            "provenance": (
                "Fresh Modal billing queries were taken before each launch; final "
                "metered and billed values were queried after both movement audits."
            ),
        },
        "next_bounded_experiment": {
            "goal": (
                "Preserve candidate one-pass strength while retaining the learned "
                "benefit from JEPA-conditioned recurrent inference."
            ),
            "required_baseline": "Repeat this exact no-loop control on the common prefix.",
            "primary_arm": (
                "Full loop plus deep supervision on both exits: apply fixed-Hero1 "
                "teacher KL and legal-root/DFM objectives to preliminary one-pass "
                "logits as well as final loop-conditioned logits."
            ),
            "secondary_arm_if_budget_allows": (
                "The same multi-exit objective with pass dropout or a fixed mixture "
                "of one-pass and loop-conditioned batches, so the first exit remains "
                "a trained deployment path."
            ),
            "screen": (
                "Use a matched 1,024-update prefix, then repeat candidate p1 vs "
                "control p1 and candidate p8 vs candidate p1 first; run p8 vs "
                "control only after both pass."
            ),
            "promotion_boundary": (
                "Require p1 noninferiority and a positive p8-p1 recurrence effect. "
                "Only then preregister a longer scaling comparison."
            ),
        },
        "limitations": [
            "All play is deterministic greedy policy play, not search-based Elo.",
            "The arena tier is development-only and relative Elo is checkpoint-pool relative.",
            "The four primary comparisons reuse the same held-out opening slice; they are matched contrasts, not independent replications.",
            "The ablation covers 1,048,576 examples, far short of the full Hero horizon.",
            "The exploratory fifth arena was preregistered separately and cannot rescue the primary promotion rule.",
        ],
    }


def _format_number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_markdown(analysis: Mapping[str, Any]) -> str:
    arenas = analysis["arenas"]
    validation = analysis["frozen_validation"]["metrics"]
    movement = analysis["parameter_movement"]
    gates = analysis["preregistered_decision"]["gates"]
    cost = analysis["cost"]
    lines = [
        "# Full-loop 1,024-update ablation",
        "",
        "## Decision",
        "",
        "**Retain the architecture, reject this exact training composition for a Hero run.** "
        "The predicted-JEPA loop makes recurrent inference genuinely useful, but the "
        "final-only objective materially damages the one-pass policy. The exact "
        "preregistered mechanism-success rule therefore fails.",
        "",
        "The two arms used the same Raw BT4 initialization, fixed Hero1 teacher, ordered "
        "1,048,576-example prefix, WSD schedule, frozen validation pool, and optimizer "
        "recipe. The only scientific changes were enabling `predicted_jepa_tokens` and "
        "adding its 262,144-parameter rollout adapter.",
        "",
        "## Arena evidence",
        "",
        "All results use 128 color-reversed opening pairs (256 games), opening indices "
        "256-383, deterministic greedy legal argmax, zero faults, zero cap draws, and "
        "complete move-codec coverage.",
        "",
        "| Comparison | Score | Pair-aware 95% interval | Relative Elo | Pentanomial |",
        "|---|---:|---:|---:|---:|",
    ]
    arena_rows = (
        ("Candidate p8 vs candidate p1", "candidate_8_vs_candidate_1"),
        ("Candidate p8 vs control p1", "candidate_8_vs_control_1"),
        ("Candidate p1 vs control p1", "candidate_1_vs_control_1"),
        ("Candidate p8 vs control p8", "candidate_8_vs_control_8"),
        ("Control p1 vs control p8 (exploratory)", "control_1_vs_control_8_exploratory"),
    )
    for label, key in arena_rows:
        result = arenas[key]
        interval = result["score_interval_95"]
        lines.append(
            f"| {label} | {_format_number(result['score'])} | "
            f"[{_format_number(interval[0])}, {_format_number(interval[1])}] | "
            f"{_format_number(result['relative_elo'], 1)} | "
            f"`{result['pentanomial']}` |"
        )
    lines.extend(
        [
            "",
            "The recurrence effect is large: candidate p8 scores 0.7461 against its own "
            "p1 exit and 0.8145 against control p8. Candidate p8 recovers to point-score "
            "parity with control p1 (0.5117). In contrast, candidate p1 scores 0.2578 "
            "against control p1, with its entire interval below the preregistered 0.45 "
            "material-deficit boundary. The exploratory control comparison shows that "
            "plain iterative unmasking hurts: control p1 scores 0.7969 against control p8.",
            "",
            "## Frozen offline evidence",
            "",
            "Terminal validation used the same 8,192 positions for both arms.",
            "",
            "| Metric | Control | Candidate | Candidate - control | Direction |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric in VALIDATION_METRICS:
        result = validation[metric]
        lines.append(
            f"| `{metric}` | {_format_number(result['control'], 6)} | "
            f"{_format_number(result['candidate'], 6)} | "
            f"{_format_number(result['candidate_minus_control'], 6)} | "
            f"{result['direction']} |"
        )
    preliminary = validation["candidate_first_planner_dfm_ce_loss"]["candidate"]
    final_dfm = validation["dfm_ce_loss"]["candidate"]
    loop_gain = validation["candidate_second_minus_first_planner_dfm_ce"][
        "improvement_positive"
    ]
    lines.extend(
        [
            "",
            f"The candidate's first planner has DFM CE {_format_number(preliminary, 6)}; "
            f"the JEPA-conditioned second planner reaches {_format_number(final_dfm, 6)}, "
            f"an improvement of {_format_number(loop_gain, 6)}. JEPA itself is healthy: "
            "all eight horizons beat zero, identity, shuffled-state, and shuffled-action "
            "baselines; RMS ratios stay inside [0.5, 2.0], and no effective-rank collapse "
            "is present.",
            "",
            "## Preregistered gates",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    gate_labels = {
        "candidate_eight_pass_beats_own_one_pass": "Candidate p8 beats candidate p1",
        "candidate_eight_pass_recovers_to_control_one_pass": "Candidate p8 recovers to control p1",
        "candidate_one_pass_not_materially_damaged": "Candidate p1 is not materially damaged",
        "offline_latent_and_systems_health": "Offline latent/systems/arena health",
    }
    for key, label in gate_labels.items():
        lines.append(f"| {label} | {'PASS' if gates[key]['passed'] else 'FAIL'} |")
    adapter = movement["candidate_loop_adapter"]
    lines.extend(
        [
            "",
            "Mechanism success is **FAIL** solely because the one-pass preservation gate "
            "fails decisively. This is not a representation-collapse result.",
            "",
            "## Parameter movement and systems",
            "",
            f"The loop adapter changed in every scalar and moved "
            f"{_format_number(adapter['relative_delta_l2'], 4)} relative L2 (cosine "
            f"{_format_number(adapter['cosine'], 4)}). Candidate trunk movement was "
            f"{_format_number(movement['groups']['trunk']['candidate_relative_delta_l2'], 6)} "
            f"versus {_format_number(movement['groups']['trunk']['control_relative_delta_l2'], 6)} "
            "for control, so the behavioral result is not explained by runaway encoder drift.",
            "",
            "Both 1,024-update runs completed all planned recoveries with zero nonfinite or "
            "skipped updates. The preregistered diagnostic passed immutable-teacher, loop "
            "activity, throughput, memory-headroom, and no-checkpoint checks.",
            "",
            "## Cost",
            "",
            f"Metered spend rose from ${cost['metered_before_experiment_usd']:.2f} to "
            f"${cost['metered_final_usd']:.2f}: ${cost['whole_experiment_delta_usd']:.2f} "
            "for the full experiment. Arenas plus movement audits cost "
            f"${cost['arena_and_movement_delta_usd']:.2f}. The final billed value was "
            f"${cost['billed_cost_usd_at_final_query']:.2f}; workers are stopped. "
            f"${cost['remaining_to_repository_stop_usd']:.2f} remains before the "
            "repository's conservative new-launch stop.",
            "",
            "## Next bounded experiment",
            "",
            "Keep the loop, but train the first exit explicitly. The primary follow-up is "
            "deep supervision: apply the fixed-Hero1 teacher and legal-root/DFM losses to "
            "both the preliminary one-pass logits and the final JEPA-conditioned logits. "
            "If budget permits one additional arm, combine that with pass dropout or a "
            "fixed mixture of one-pass and loop-conditioned batches.",
            "",
            "At 1,024 updates, first require candidate p1 noninferiority to control p1 and "
            "candidate p8 superiority to its own p1. Only a recipe satisfying both should "
            "receive a longer scaling preregistration. This targets the observed failure "
            "directly while preserving the mechanism that worked.",
            "",
            "## Scope and limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis["limitations"])
    lines.extend(
        [
            "",
            "The machine-readable companion contains every input SHA-256, curve bootstrap, "
            "latent horizon, arena block checksum, exact gate evaluation, and cost ledger.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_or_check(path: Path, content: str, *, check: bool) -> None:
    if check:
        _require(path.is_file(), f"Missing generated artifact {path}")
        _require(path.read_text(encoding="utf-8") == content, f"Stale artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = root / "research" / "analysis" / "hero_v2_phase1" / "u1024"
    json_path = args.json_output or output_dir / "full_loop_ablation_decision.json"
    markdown_path = args.markdown_output or output_dir / "full_loop_ablation_report.md"
    analysis = build_analysis(root)
    json_content = json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_content = render_markdown(analysis)
    _write_or_check(json_path, json_content, check=args.check)
    _write_or_check(markdown_path, markdown_content, check=args.check)
    if not args.check:
        print(json_path)
        print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
