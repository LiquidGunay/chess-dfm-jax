"""Validate and rank preregistered Proposal-A LR or weight-decay screens."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


TRAIN_FILE_MANIFEST_SHA256 = "7cbd73ab0580c9fa532e7f453c90b0e0b669ce774ba0abce8387436f63c10f2d"
VALIDATION_FILE_MANIFEST_SHA256 = "4b9091d5710bbac31e4d107995c5adb9e0e05616f43892d7d576e8ce6fbeab7b"
EXPECTED_UPDATES = 252
EXPECTED_VALIDATION_EXAMPLES = 1_152
EXPECTED_LEARNING_RATES = (5e-5, 1e-4, 2e-4, 4e-4)
EXPECTED_COMPILE_SOURCE_TREE_SHA256 = (
    "8e2b3f748bf57b2927cad029c612cf98d779c5cd3672d38d055352b338b03202"
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite metric {key}: {value}")
    return value


def load_run(directory: Path) -> dict[str, Any]:
    directory = directory.resolve(strict=True)
    run_config = _read_json(directory / "run_config.json")
    modal_run = _read_json(directory / "modal_run.json")
    report = _read_json(directory / "report.json")
    training = _read_jsonl(directory / "metrics.jsonl")
    validation = _read_jsonl(directory / "validation_metrics.jsonl")
    config = run_config["config"]
    policy = run_config["optimizer_policy"]
    args = run_config["args"]
    data = run_config["data"]
    final_validation = validation[-1]
    pool = final_validation["pool"]
    metrics = final_validation["metrics"]
    steady_step_rates = [_finite_metric(row, "examples_per_second_step") for row in training[1:]]

    checks = {
        "profile": modal_run.get("profile") == "proposal_screen",
        "updates": int(report.get("updates", -1)) == EXPECTED_UPDATES,
        "contiguous_metrics": [int(row["update"]) for row in training]
        == list(range(1, EXPECTED_UPDATES + 1)),
        "finite_training": all(
            not bool(row.get("optimizer_skipped_nonfinite", False))
            and all(
                math.isfinite(float(row[key]))
                for key in ("loss", "learning_rate", "bt4_learning_rate")
            )
            for row in training
        ),
        "data_format": args.get("data_format") == "lc0_sequential",
        "train_inventory": data.get("file_manifest_sha256") == TRAIN_FILE_MANIFEST_SHA256,
        "train_steps": int(data.get("steps_per_epoch", -1)) == EXPECTED_UPDATES,
        "validation_inventory": pool["data"].get("file_manifest_sha256")
        == VALIDATION_FILE_MANIFEST_SHA256,
        "validation_examples": int(final_validation.get("evaluation_examples", -1))
        == EXPECTED_VALIDATION_EXAMPLES,
        "validation_update": int(final_validation.get("update", -1)) == EXPECTED_UPDATES,
        "compile_source": modal_run.get("compile_source_tree_sha256")
        == EXPECTED_COMPILE_SOURCE_TREE_SHA256,
        "gpu": modal_run.get("gpu") == "L40S",
        "equal_peak_rates": float(config["learning_rate"]) == float(config["bt4_learning_rate"]),
        "full_epoch_clock": int(config["lr_total_examples"]) == 7_960_576,
        "fp32_master": policy.get("precision") == "fp32_master",
        "wsd": policy.get("schedule_kind") == "warmup_stable_linear_decay",
        "no_teacher_term": float(config["policy_distill_coeff"]) == 0.0,
        "open_loop": config.get("dfm_closed_loop_mode") == "none"
        and config.get("jepa_feedback_mode") == "none",
        "current_value": config.get("wdl_include_current_state") is True,
        "no_illegal_mass_aux": float(config["legality_coeff"]) == 0.0,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Proposal A run contract failed in {directory}: {failed}")
    return {
        "directory": str(directory),
        "result_label": modal_run["result_label"],
        "arm": modal_run["arm"],
        "provenance": {
            key: modal_run[key]
            for key in (
                "git_commit",
                "source_tree_sha256",
                "compile_source_tree_sha256",
                "gpu",
            )
        },
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "weight_decay_mode": policy["weight_decay_mode"],
        "validation": {
            key: _finite_metric(metrics, key)
            for key in (
                "loss",
                "dfm_ce_loss",
                "root_legal_top1_accuracy",
                "root_legal_conditional_ce",
                "jepa_positive_loss",
                "wdl_loss",
                "wdl_current_loss",
                "wdl_current_accuracy",
                "wdl_expected_value_mse",
            )
        },
        "train_seconds": float(report["train_seconds"]),
        "examples_per_second": float(report["examples_per_second_end_to_end"]),
        "timing": {
            "first_step_seconds": _finite_metric(training[0], "step_seconds"),
            "mean_steady_examples_per_second_step": sum(steady_step_rates) / len(steady_step_rates),
            "validation_seconds": _finite_metric(
                final_validation,
                "evaluation_seconds",
            ),
        },
        "checks": checks,
    }


def _lr_analysis(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda row: row["learning_rate"])
    observed_rates = tuple(row["learning_rate"] for row in ordered)
    if observed_rates != EXPECTED_LEARNING_RATES:
        raise ValueError(
            "LR analysis requires the complete preregistered rate grid: "
            f"expected {EXPECTED_LEARNING_RATES}, found {observed_rates}"
        )
    if any(row["weight_decay"] != 0.0 for row in ordered):
        raise ValueError("LR analysis requires zero weight decay in every arm")
    best_top1 = max(row["validation"]["root_legal_top1_accuracy"] for row in ordered)
    best_wdl = min(row["validation"]["wdl_current_loss"] for row in ordered)
    for row in ordered:
        row["promotion_eligible"] = (
            row["validation"]["root_legal_top1_accuracy"] >= best_top1 - 0.02
            and row["validation"]["wdl_current_loss"] <= best_wdl + 0.05
        )
    eligible = [row for row in ordered if row["promotion_eligible"]]
    if not eligible:
        raise ValueError("No LR arm satisfies the preregistered guardrails")
    winner = min(
        eligible,
        key=lambda row: (row["validation"]["loss"], row["learning_rate"]),
    )
    control = ordered[0]
    substantial = (
        control["validation"]["loss"] - winner["validation"]["loss"] >= 0.10
        and control["validation"]["wdl_current_loss"] - winner["validation"]["wdl_current_loss"]
        >= 0.05
        and winner["validation"]["root_legal_top1_accuracy"]
        >= control["validation"]["root_legal_top1_accuracy"] - 0.02
    )
    return {
        "kind": "learning_rate",
        "runs": ordered,
        "winner_arm": winner["arm"],
        "winner_learning_rate": winner["learning_rate"],
        "substantial_vs_lowest_rate": substantial,
    }


def _decay_analysis(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 3 or len({row["learning_rate"] for row in runs}) != 1:
        raise ValueError("Decay analysis requires exactly three runs at one rate")
    controls = [row for row in runs if row["weight_decay"] == 0.0]
    if len(controls) != 1:
        raise ValueError("Decay analysis requires exactly one zero-decay control")
    control = controls[0]
    decayed = [row for row in runs if row is not control]
    if (
        control["weight_decay_mode"] != "decoupled"
        or len({row["weight_decay"] for row in decayed}) != 1
        or decayed[0]["weight_decay"] <= 0.0
        or {row["weight_decay_mode"] for row in decayed} != {"decoupled", "cautious"}
    ):
        raise ValueError(
            "Decay analysis requires zero, matched decoupled, and matched cautious arms"
        )
    eligible = [control]
    for row in runs:
        if row is control:
            row["promotion_eligible"] = True
            continue
        row["promotion_eligible"] = row["validation"]["root_legal_top1_accuracy"] >= control[
            "validation"
        ]["root_legal_top1_accuracy"] - 0.01 and (
            control["validation"]["loss"] - row["validation"]["loss"] >= 0.03
            or control["validation"]["wdl_current_loss"] - row["validation"]["wdl_current_loss"]
            >= 0.02
        )
        if row["promotion_eligible"]:
            eligible.append(row)
    winner = min(
        eligible,
        key=lambda row: (
            row["validation"]["loss"],
            row["weight_decay"] != 0.0,
        ),
    )
    return {
        "kind": "weight_decay",
        "runs": sorted(
            runs,
            key=lambda row: (row["weight_decay"], row["weight_decay_mode"]),
        ),
        "winner_arm": winner["arm"],
        "winner_weight_decay": winner["weight_decay"],
        "winner_weight_decay_mode": winner["weight_decay_mode"],
    }


def analyze(kind: str, directories: Sequence[Path]) -> dict[str, Any]:
    runs = [load_run(path) for path in directories]
    body = _lr_analysis(runs) if kind == "lr" else _decay_analysis(runs)
    return {
        "schema_version": "proposal-a-tuning-analysis-v1",
        **body,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("lr", "decay"))
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(args.kind, args.directories)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
