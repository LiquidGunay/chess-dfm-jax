#!/usr/bin/env python3
"""Build a compact profiler-driven Torch systems-sprint report."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/analysis/torch_kernel_sprint_20260727.json"
RUNS = {
    "eager_reference": ROOT
    / "research/runs/torch_kernel_sprint_b1024_baseline_v1",
    "fused_backward": ROOT
    / "research/runs/torch_kernel_sprint_eager_fused_backward_b1024_v1",
    "fused_backward_delayed_prefetch": ROOT
    / "research/runs/torch_kernel_sprint_fused_backward_after_forward_b1024_v1",
    "selected": ROOT
    / "research/runs/torch_kernel_sprint_fast_canonical_after_forward_b1024_v1",
}
INPUTS = {
    "annotated_profile": ROOT
    / "research/runs/torch_kernel_sprint_b1024_profile_v1/report.json",
    "rejected_native_layernorm": ROOT
    / "artifacts/profiles/hero_native_fp32_layernorm_runtime_parity_b512.json",
    "accepted_fused_backward": ROOT
    / "artifacts/profiles/"
    "hero_eager_fused_backward_layernorm_runtime_parity_b512.json",
    "canonicalization_parity": ROOT
    / "artifacts/profiles/hero_canonicalization_parity_v1.json",
}
SCIENTIFIC_METRICS = (
    "loss",
    "unclipped_loss",
    "dfm_ce_loss",
    "accuracy",
    "first_legal_mass",
    "first_legality_loss",
    "weighted_legality_loss",
    "root_legal_conditional_ce",
    "weighted_root_legal_conditional_ce",
    "jepa_positive_loss",
    "jepa_raw_mse",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "jepa_sigreg_valid_count",
    "jepa_pred_sigreg_valid_count",
    "jepa_target_mean_horizon",
    "z_state_norm",
    "z_pred_norm",
    "z_target_norm",
    "wdl_loss",
    "wdl_weighted_loss",
    "wdl_accuracy",
    "wdl_expected_value_mse",
    "wdl_valid_count",
    "gradient_global_norm",
    "gradient_clip_scale",
    "learning_rate",
    "bt4_learning_rate",
    "mask_prob",
    "optimizer_skipped_nonfinite",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _steady_summary(run_dir: Path) -> dict[str, Any]:
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    steady = [row for row in rows if 2 <= int(row["update"]) <= 19]
    if len(steady) != 18:
        raise ValueError(f"Expected 18 steady rows in {run_dir}, got {len(steady)}")
    run_config = _load_json(run_dir / "run_config.json")
    batch_size = int(run_config["args"]["batch_size"])
    mean_step_seconds = statistics.mean(
        float(row["step_seconds"]) for row in steady
    )
    report = _load_json(run_dir / "report.json")
    return {
        "git_commit": run_config["git_commit"],
        "steady_update_range": [2, 19],
        "steady_update_count": len(steady),
        "batch_size": batch_size,
        "mean_step_seconds": mean_step_seconds,
        "examples_per_second": batch_size / mean_step_seconds,
        "mean_forward_cuda_seconds": statistics.mean(
            float(row["forward_cuda_seconds"]) for row in steady
        ),
        "mean_backward_cuda_seconds": statistics.mean(
            float(row["backward_cuda_seconds"]) for row in steady
        ),
        "mean_optimizer_cuda_seconds": statistics.mean(
            float(row["optimizer_cuda_seconds"]) for row in steady
        ),
        "mean_data_prepare_seconds": statistics.mean(
            float(row["data_prepare_seconds"]) for row in steady
        ),
        "mean_data_wait_seconds": statistics.mean(
            float(row["data_wait_seconds"]) for row in steady
        ),
        "peak_allocated_bytes": int(
            report["gpu_peak_memory_allocated_bytes"]
        ),
        "peak_reserved_bytes": int(report["gpu_peak_memory_reserved_bytes"]),
        "terminal_dfm_ce_loss": float(rows[-1]["dfm_ce_loss"]),
        "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
        "report_sha256": _sha256(run_dir / "report.json"),
        "_rows": rows,
    }


def _strip_private(value: dict[str, Any]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if not key.startswith("_")}


def main() -> int:
    summaries = {
        name: _steady_summary(path) for name, path in RUNS.items()
    }
    previous_rows = summaries["fused_backward_delayed_prefetch"]["_rows"]
    selected_rows = summaries["selected"]["_rows"]
    scientific_mismatches: list[dict[str, Any]] = []
    for previous, selected in zip(previous_rows, selected_rows, strict=True):
        if previous["update"] != selected["update"]:
            raise ValueError("Compared runs have different update schedules")
        for metric in SCIENTIFIC_METRICS:
            if previous[metric] != selected[metric]:
                scientific_mismatches.append(
                    {
                        "update": int(previous["update"]),
                        "metric": metric,
                        "previous": previous[metric],
                        "selected": selected[metric],
                    }
                )

    eager = summaries["eager_reference"]
    previous = summaries["fused_backward_delayed_prefetch"]
    selected = summaries["selected"]
    annotated = _load_json(INPUTS["annotated_profile"])["profiler"]
    rejected = _load_json(INPUTS["rejected_native_layernorm"])
    accepted = _load_json(INPUTS["accepted_fused_backward"])
    canonical = _load_json(INPUTS["canonicalization_parity"])
    selected_throughput = float(selected["examples_per_second"])
    report = {
        "schema_version": "torch-hero-kernel-sprint-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "selection": {
            "bt4_norm_impl": "eager-fused-backward",
            "prefetch_launch": "after-forward",
            "trajectory_canonicalization": "fixed-index-map-with-promotion-recovery",
            "physical_batch_size": 1024,
            "sigreg_example_count": 64,
        },
        "profiler": {
            "aggregate_flops_per_profiled_step": int(
                annotated["aggregate_flops"]
            ),
            "trace": annotated["trace"],
            "bt4_current_cuda_span_ms": 719.875815,
            "bt4_future_cuda_span_ms": 717.101029,
            "state_projector_cuda_span_ms": 144.581404,
            "dfm_noisy_cuda_span_ms": 20.971602,
            "dfm_clean_cuda_span_ms": 20.969234,
            "jepa_rollout_cuda_span_ms": 22.319443,
            "target_sigreg_cuda_span_ms": 1.274753,
            "prediction_sigreg_cuda_span_ms": 3.329154,
            "interpretation": (
                "The two eager BT4 encodes dominate forward time; SIGReg and "
                "cross-entropy are too small to meet the 10% adoption gate."
            ),
        },
        "runtime_parity": {
            "native_fp32_layernorm_gate_pass": bool(rejected["gate_pass"]),
            "native_fp32_legal_action_agreement": float(
                rejected["root_logits"]["legal_top1_action_agreement"]
            ),
            "native_fp32_raw_bt4_gradient_cosine": float(
                rejected["gradient_groups"]["raw_bt4"]["cosine"]
            ),
            "fused_backward_gate_pass": bool(accepted["gate_pass"]),
            "fused_backward_loss_relative_difference": float(
                accepted["loss_relative_difference"]
            ),
            "fused_backward_global_gradient_cosine": float(
                accepted["gradient_groups"]["all"]["cosine"]
            ),
            "fused_backward_raw_bt4_gradient_cosine": float(
                accepted["gradient_groups"]["raw_bt4"]["cosine"]
            ),
        },
        "canonicalization_parity": {
            "gate_pass": bool(canonical["gate_pass"]),
            "audited_shard_count": int(canonical["audited_shard_count"]),
            "audited_example_count": int(canonical["audited_example_count"]),
            "audited_valid_action_count": int(
                canonical["audited_valid_action_count"]
            ),
            "recovered_promotion_slot_count": int(
                canonical["recovered_promotion_slot_count"]
            ),
            "conversion_speedup": float(canonical["conversion_speedup"]),
            "selection_sha256": canonical["selection_sha256"],
        },
        "runs": {
            name: _strip_private(summary)
            for name, summary in summaries.items()
        },
        "result": {
            "scientific_metrics_exact_update_by_update": not scientific_mismatches,
            "scientific_mismatches": scientific_mismatches,
            "throughput_gain_vs_eager_reference": (
                selected_throughput / float(eager["examples_per_second"]) - 1.0
            ),
            "throughput_gain_vs_previous_best": (
                selected_throughput / float(previous["examples_per_second"]) - 1.0
            ),
            "projected_epoch_training_seconds": (
                28_343_296 / selected_throughput
            ),
            "projected_epoch_training_hours": (
                28_343_296 / selected_throughput / 3600.0
            ),
            "adoption_gate": {
                "minimum_end_to_end_gain": 0.10,
                "numerical_gate_pass": (
                    bool(accepted["gate_pass"])
                    and bool(canonical["gate_pass"])
                    and not scientific_mismatches
                ),
                "systems_gate_pass": (
                    selected_throughput
                    / float(eager["examples_per_second"])
                    - 1.0
                    >= 0.10
                ),
            },
        },
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for name, path in INPUTS.items()
        },
    }
    report["result"]["adopt"] = all(
        report["result"]["adoption_gate"].values()
    )
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "adopt": report["result"]["adopt"],
                "selected_examples_per_second": selected_throughput,
                "gain_vs_eager": report["result"][
                    "throughput_gain_vs_eager_reference"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["result"]["adopt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
