"""Reproducible post-hoc audit of the Hero encoder update budget.

The audit joins the immutable Hero run configuration, validation trajectory,
loss summary, optimizer partition, and Raw->Hero parameter diff.  It diagnoses
whether the encoder was capable of moving under the chosen learning-rate and
precision contract; it does not pretend to replace a controlled LR sweep.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from research.interpretability.artifacts import (
    content_identity,
    sha256_file,
    write_json_atomic,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_CONFIG = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/run_config.json"
DEFAULT_LOSS_SUMMARY = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/loss_summary.json"
DEFAULT_VALIDATION = (
    _REPO_ROOT / "research/runs/torch_hero_epoch_v1/hero_validation_metrics.jsonl"
)
DEFAULT_PARAMETER_DIFF = (
    _REPO_ROOT / "research/analysis/raw_hero_parameter_diff_20260807.json"
)
DEFAULT_OUTPUT = _REPO_ROOT / "research/analysis/hero_encoder_lr_audit_20260808.json"
AUDIT_SCHEMA = "torch-hero-encoder-lr-audit-v1"


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Validation row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("Validation trajectory is empty")
    return rows


def _schedule_ratio(
    *, position: int, warmup_examples: int, total_examples: int, minimum_ratio: float
) -> float:
    position = min(position, total_examples)
    if position <= warmup_examples:
        return position / warmup_examples
    progress = (position - warmup_examples) / (total_examples - warmup_examples)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def schedule_budget(config: Mapping[str, Any], *, examples_per_update: int) -> dict[str, Any]:
    required = {
        "learning_rate",
        "bt4_learning_rate",
        "lr_warmup_examples",
        "lr_total_examples",
        "lr_min_ratio",
    }
    if not required.issubset(config) or examples_per_update <= 0:
        raise ValueError("Learning-rate schedule contract is incomplete")
    total = int(config["lr_total_examples"])
    updates = math.ceil(total / examples_per_update)
    ratios = np.asarray(
        [
            _schedule_ratio(
                position=(update + 1) * examples_per_update,
                warmup_examples=int(config["lr_warmup_examples"]),
                total_examples=total,
                minimum_ratio=float(config["lr_min_ratio"]),
            )
            for update in range(updates)
        ],
        dtype=np.float64,
    )
    main_peak = float(config["learning_rate"])
    encoder_peak = float(config["bt4_learning_rate"])
    checkpoints = {}
    for fraction in (0.02, 0.1, 0.5, 0.9, 1.0):
        index = min(updates - 1, max(0, math.ceil(updates * fraction) - 1))
        checkpoints[str(fraction)] = {
            "update": index + 1,
            "ratio": float(ratios[index]),
            "main_learning_rate": float(main_peak * ratios[index]),
            "encoder_learning_rate": float(encoder_peak * ratios[index]),
        }
    return {
        "updates": updates,
        "examples_per_update": examples_per_update,
        "main_peak_learning_rate": main_peak,
        "encoder_peak_learning_rate": encoder_peak,
        "main_to_encoder_peak_ratio": main_peak / encoder_peak,
        "mean_schedule_ratio": float(ratios.mean()),
        "integrated_main_learning_rate": float((main_peak * ratios).sum()),
        "integrated_encoder_learning_rate": float((encoder_peak * ratios).sum()),
        "checkpoints": checkpoints,
    }


def optimizer_movement(
    partition: Mapping[str, Any], parameter_diff: Mapping[str, Any]
) -> dict[str, Any]:
    partition_rows = partition.get("leaves")
    diff_rows = parameter_diff.get("leaves")
    if not isinstance(partition_rows, list) or not isinstance(diff_rows, list):
        raise ValueError("Optimizer partition or parameter diff has no leaf rows")
    diff_by_name = {str(row["name"]): row for row in diff_rows}
    grouped: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "leaf_count": 0,
            "parameter_count": 0,
            "unchanged_count": 0,
            "delta_energy": 0.0,
        }
    )
    inventory = Counter()
    matched = 0
    for row in partition_rows:
        key = (
            str(row["learning_rate_kind"]),
            str(row["optimizer"]),
            str(row["dtype"]),
        )
        inventory[key] += 1
        if row["learning_rate_kind"] != "bt4":
            continue
        name = str(row["path"]).removeprefix("encoder.")
        if name not in diff_by_name:
            raise ValueError(f"Optimizer leaf {name} is absent from parameter diff")
        diff = diff_by_name[name]
        group = grouped[str(row["optimizer"])]
        group["leaf_count"] = int(group["leaf_count"]) + 1
        group["parameter_count"] = int(group["parameter_count"]) + int(
            diff["parameter_count"]
        )
        group["unchanged_count"] = int(group["unchanged_count"]) + int(
            diff["unchanged_count"]
        )
        group["delta_energy"] = float(group["delta_energy"]) + float(
            diff["delta_l2"]
        ) ** 2
        matched += 1
    global_energy = float(parameter_diff["group_metrics"]["all"]["delta_l2"]) ** 2
    summaries: dict[str, Any] = {}
    for optimizer, raw in sorted(grouped.items()):
        parameter_count = int(raw["parameter_count"])
        unchanged_count = int(raw["unchanged_count"])
        energy = float(raw["delta_energy"])
        summaries[optimizer] = {
            "leaf_count": int(raw["leaf_count"]),
            "parameter_count": parameter_count,
            "unchanged_count": unchanged_count,
            "unchanged_fraction": unchanged_count / parameter_count,
            "changed_fraction": 1.0 - unchanged_count / parameter_count,
            "delta_l2": math.sqrt(energy),
            "global_delta_energy_fraction": energy / global_energy,
        }
    return {
        "partition_summary": {
            key: value for key, value in partition.items() if key != "leaves"
        },
        "leaf_inventory": [
            {
                "learning_rate_kind": key[0],
                "optimizer": key[1],
                "dtype": key[2],
                "leaf_count": count,
            }
            for key, count in sorted(inventory.items())
        ],
        "matched_encoder_leaf_count": matched,
        "by_optimizer": summaries,
    }


def validation_trajectory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "loss",
        "root_legal_conditional_ce",
        "root_legal_top1_accuracy",
        "first_legal_mass",
        "dfm_ce_loss",
        "jepa_positive_loss",
        "wdl_loss",
    )
    result = []
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping) or any(key not in metrics for key in keys):
            raise ValueError("Validation milestone is missing required metrics")
        result.append(
            {
                "target_percentage": int(row["target_percentage"]),
                "update": int(row["update"]),
                "examples": int(row["examples"]),
                **{key: float(metrics[key]) for key in keys},
            }
        )
    if [row["target_percentage"] for row in result] != list(range(10, 101, 10)):
        raise ValueError("Hero validation milestones are not the exact 10%-100% sequence")
    first, last = result[0], result[-1]
    return {
        "milestones": result,
        "ten_to_hundred_percent_delta": {
            key: last[key] - first[key] for key in keys
        },
        "caveat": "the retained trajectory begins at 10%; it is not a Raw step-zero baseline",
    }


def build_audit(
    *,
    run_config_path: Path,
    loss_summary_path: Path,
    validation_path: Path,
    parameter_diff_path: Path,
) -> dict[str, Any]:
    run_config = _load_object(run_config_path, label="Hero run config")
    loss_summary = _load_object(loss_summary_path, label="Hero loss summary")
    parameter_diff = _load_object(parameter_diff_path, label="Parameter diff")
    validation = _load_jsonl(validation_path)
    config = run_config.get("config", {})
    resume = run_config.get("resume_contract", {})
    partition = resume.get("optimizer_partition", {}) if isinstance(resume, Mapping) else {}
    if not isinstance(config, Mapping) or not isinstance(partition, Mapping):
        raise ValueError("Hero run lacks config or optimizer partition")
    arguments = run_config.get("args", {})
    data_contract = run_config.get("data", {})
    if not isinstance(arguments, Mapping) or not isinstance(data_contract, Mapping):
        raise ValueError("Hero run lacks argument or data contracts")
    examples_per_update = int(arguments["batch_size"])
    if int(data_contract["batch_size"]) != examples_per_update:
        raise ValueError("Hero argument and data batch sizes differ")
    schedule = schedule_budget(config, examples_per_update=examples_per_update)
    movement = optimizer_movement(partition, parameter_diff)
    groups = parameter_diff["group_metrics"]
    components = {
        name: {
            key: groups[name][key]
            for key in (
                "parameter_count",
                "relative_delta_l2",
                "delta_l2",
                "unchanged_fraction",
            )
        }
        for name in (
            "all",
            "embedding",
            "attention",
            "mlp",
            "smolgen",
            "layer_norm",
            "policy_head",
            "layer_14",
            "layer_14.attention",
            "layer_14.mlp",
            "layer_14.smolgen",
        )
    }
    main_source = _REPO_ROOT / "research/train_torch.py"
    optimizer_source = main_source.read_text(encoding="utf-8")
    precision_criteria = {
        "all_encoder_leaves_bfloat16": all(
            row["dtype"] == "torch.bfloat16"
            for row in partition["leaves"]
            if row["learning_rate_kind"] == "bt4"
        ),
        "optimizer_first_moment_uses_parameter_dtype": (
            "first_moment=torch.zeros_like(parameter)" in optimizer_source
        ),
        "optimizer_second_moment_uses_parameter_dtype": (
            "torch.zeros_like(parameter)" in optimizer_source
            and "second_moment=" in optimizer_source
        ),
        "optimizer_updates_parameter_in_place": (
            "parameter.add_(update, alpha=-self._learning_rate(leaf.learning_rate_kind))"
            in optimizer_source
        ),
        "fp32_master_weight_contract_recorded": False,
    }
    muon = movement["by_optimizer"]["muon"]
    all_metrics = groups["all"]
    evidence = {
        "learning_rate_ratio_ge_30": schedule["main_to_encoder_peak_ratio"] >= 30.0,
        "all_encoder_leaves_low_precision": precision_criteria[
            "all_encoder_leaves_bfloat16"
        ],
        "muon_parameters_unchanged_fraction_ge_0p99": muon["unchanged_fraction"]
        >= 0.99,
        "muon_global_delta_energy_fraction_le_0p01": muon[
            "global_delta_energy_fraction"
        ]
        <= 0.01,
        "all_encoder_parameters_unchanged_fraction_ge_0p95": all_metrics[
            "unchanged_fraction"
        ]
        >= 0.95,
    }
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "post-hoc Hero-v1 encoder update-budget diagnosis",
        "sources": {
            "run_config": {
                "path": str(run_config_path.resolve()),
                "sha256": sha256_file(run_config_path),
            },
            "loss_summary": {
                "path": str(loss_summary_path.resolve()),
                "sha256": sha256_file(loss_summary_path),
            },
            "validation": {
                "path": str(validation_path.resolve()),
                "sha256": sha256_file(validation_path),
            },
            "parameter_diff": {
                "path": str(parameter_diff_path.resolve()),
                "sha256": sha256_file(parameter_diff_path),
            },
            "optimizer_source": {
                "path": str(main_source.resolve()),
                "sha256": sha256_file(main_source),
            },
        },
        "schedule": schedule,
        "precision_contract": {
            "criteria": precision_criteria,
            "interpretation": (
                "BF16 parameters and same-dtype optimizer moments were updated in place; "
                "there is no recorded FP32 master-weight accumulation contract"
            ),
        },
        "optimizer_movement": movement,
        "component_movement": components,
        "loss_summary": loss_summary,
        "validation": validation_trajectory(validation),
        "evidence_gate": {"criteria": evidence, "passed": all(evidence.values())},
        "verdict": {
            "encoder_update_budget_was_severely_constrained": all(evidence.values()),
            "lower_lr_alone_proven_causal": False,
            "precision_and_lr_are_confounded": True,
            "hero_training_invalid": False,
            "recommended_next_test": (
                "controlled short runs with FP32 master weights/moments crossed with encoder "
                "peak LR ratios 1/30, 1/10, and 1/3; promote only after policy-retention and "
                "multi-objective validation gates"
            ),
        },
    }
    audit["content_identity"] = content_identity(audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--loss-summary", type=Path, default=DEFAULT_LOSS_SUMMARY)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--parameter-diff", type=Path, default=DEFAULT_PARAMETER_DIFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output.resolve()}")
    audit = build_audit(
        run_config_path=args.run_config,
        loss_summary_path=args.loss_summary,
        validation_path=args.validation,
        parameter_diff_path=args.parameter_diff,
    )
    write_json_atomic(args.output, audit)
    print(json.dumps({"output": str(args.output.resolve()), **audit["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
