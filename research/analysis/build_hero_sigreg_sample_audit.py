#!/usr/bin/env python3
"""Build the compact decision record for the hero SIGReg sample audit."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/analysis/hero_sigreg_sample_audit_20260727.json"
INPUTS = {
    "fresh_counts": ROOT
    / "artifacts/profiles/hero_sigreg_sample_audit_fresh_b1024_v1.json",
    "terminal_counts": ROOT
    / "artifacts/profiles/hero_sigreg_sample_audit_terminal_b1024_v1.json",
    "fresh_pair_128_256": ROOT
    / "artifacts/profiles/"
    "hero_sigreg_sample_audit_fresh_pair128_256_b1024_v1.json",
    "terminal_pair_128_256": ROOT
    / "artifacts/profiles/"
    "hero_sigreg_sample_audit_terminal_pair128_256_b1024_v1.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _count_summary(audit: dict[str, Any], count: int) -> dict[str, Any]:
    key = str(count)
    scalar = audit["scalar_dispersion"][key]
    record = audit["representative_records"][key]
    gradient = record["gradient_vs_baseline"]["groups"]
    return {
        "weighted_shared_sigreg_mean": float(scalar["weighted_shared"]["mean"]),
        "weighted_shared_sigreg_sample_std": float(
            scalar["weighted_shared"]["sample_standard_deviation"]
        ),
        "mean_statistic_cuda_ms": (
            1_000.0 * float(scalar["mean_statistic_cuda_seconds"])
        ),
        "representative_forward_cuda_ms": (
            1_000.0 * float(record["forward_cuda_seconds"])
        ),
        "representative_backward_cuda_ms": (
            1_000.0 * float(record["backward_cuda_seconds"])
        ),
        "peak_allocated_bytes": int(record["peak_allocated_bytes"]),
        "shared_sigreg_gradient_norm": float(
            record["gradient_groups"]["all"]["norm"]
        ),
        "gradient_vs_64": {
            group: {
                "cosine": values["cosine"],
                "norm_ratio": values["candidate_to_reference_norm_ratio"],
            }
            for group, values in gradient.items()
        },
    }


def _state_summary(audit: dict[str, Any]) -> dict[str, Any]:
    counts = {
        str(count): _count_summary(audit, count)
        for count in (64, 128, 256)
    }
    std_64 = counts["64"]["weighted_shared_sigreg_sample_std"]
    std_128 = counts["128"]["weighted_shared_sigreg_sample_std"]
    std_256 = counts["256"]["weighted_shared_sigreg_sample_std"]
    return {
        "state": audit["state"],
        "gate_pass": bool(audit["gate_pass"]),
        "batch_size": int(audit["batch_size"]),
        "replicates": int(audit["replicates"]),
        "shared_coefficient": float(audit["shared_sigreg_coefficient"]),
        "counts": counts,
        "dispersion_reduction_vs_64": {
            "128": 1.0 - std_128 / std_64,
            "256": 1.0 - std_256 / std_64,
        },
        "weighted_mean_ratio_256_vs_64": (
            counts["256"]["weighted_shared_sigreg_mean"]
            / counts["64"]["weighted_shared_sigreg_mean"]
        ),
        "statistic_cuda_ms_delta_256_vs_64": (
            counts["256"]["mean_statistic_cuda_ms"]
            - counts["64"]["mean_statistic_cuda_ms"]
        ),
        "peak_allocated_bytes_delta_256_vs_64": (
            counts["256"]["peak_allocated_bytes"]
            - counts["64"]["peak_allocated_bytes"]
        ),
    }


def _pair_summary(audit: dict[str, Any]) -> dict[str, Any]:
    record_128 = audit["representative_records"]["128"]
    record_256 = audit["representative_records"]["256"]
    gradient = record_256["gradient_vs_baseline"]["groups"]
    return {
        "state": audit["state"],
        "gradient_256_vs_128": {
            group: {
                "cosine": values["cosine"],
                "norm_ratio": values["candidate_to_reference_norm_ratio"],
            }
            for group, values in gradient.items()
        },
        "forward_backward_wall_ms": {
            "128": 1_000.0
            * float(record_128["forward_backward_wall_seconds"]),
            "256": 1_000.0
            * float(record_256["forward_backward_wall_seconds"]),
        },
        "forward_backward_wall_ms_delta_256_vs_128": 1_000.0
        * (
            float(record_256["forward_backward_wall_seconds"])
            - float(record_128["forward_backward_wall_seconds"])
        ),
    }


def main() -> int:
    audits = {name: _load_json(path) for name, path in INPUTS.items()}
    if not all(bool(audit["gate_pass"]) for audit in audits.values()):
        raise RuntimeError("At least one SIGReg audit input failed its gate")

    fresh = _state_summary(audits["fresh_counts"])
    terminal = _state_summary(audits["terminal_counts"])
    report = {
        "schema_version": "torch-hero-sigreg-sample-decision-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "fixed_contract": {
            "physical_batch_size": 1024,
            "sample_counts": [64, 128, 256],
            "target_sigreg_coefficient": 2.0,
            "prediction_sigreg_coefficient": 2.0,
            "coefficient_rescaled": False,
            "nested_deterministic_subsets": True,
            "optimizer_updates_performed": 0,
        },
        "states": {
            "fresh_initialization": fresh,
            "hero_epoch_terminal": terminal,
        },
        "direct_128_to_256": {
            "fresh_initialization": _pair_summary(
                audits["fresh_pair_128_256"]
            ),
            "hero_epoch_terminal": _pair_summary(
                audits["terminal_pair_128_256"]
            ),
        },
        "decision": {
            "baseline_sample_count": 64,
            "candidate_sample_count": 256,
            "shared_coefficient": 2.0,
            "automatic_coefficient_rescale": False,
            "adopt_without_training_comparison": False,
            "next_gate": (
                "Run seed/data-matched fresh-initialization 64 and 256 "
                "sample experiments with every other model, loss, schedule, "
                "runtime, validation, and Arena setting frozen."
            ),
            "rationale": [
                (
                    "Count 256 reduces weighted-statistic subset dispersion "
                    f"by {fresh['dispersion_reduction_vs_64']['256']:.1%} "
                    "at initialization and "
                    f"{terminal['dispersion_reduction_vs_64']['256']:.1%} "
                    "at the hero terminal state."
                ),
                (
                    "Count 256 adds "
                    f"{fresh['statistic_cuda_ms_delta_256_vs_64']:.3f} ms "
                    "to the isolated fresh-state statistic and does not "
                    "increase representative peak allocated HBM."
                ),
                (
                    "The estimator is normalized by selected-example count; "
                    "its fresh-state weighted mean and shared gradient norm "
                    "remain close to count 64, so the coefficient stays 2.0."
                ),
                (
                    "The terminal gradient direction changes materially, so "
                    "larger samples are an objective/noise ablation whose "
                    "quality must be measured, not a semantics-preserving "
                    "systems change."
                ),
            ],
            "full_batch_1024_status": (
                "Deferred until 256 has training evidence. Full-batch "
                "statistics may remove useful stochastic regularization and "
                "require a bounded/chunked implementation audit."
            ),
        },
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for name, path in INPUTS.items()
        },
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "fresh_dispersion_reduction_256_vs_64": fresh[
                    "dispersion_reduction_vs_64"
                ]["256"],
                "terminal_dispersion_reduction_256_vs_64": terminal[
                    "dispersion_reduction_vs_64"
                ]["256"],
                "candidate_sample_count": 256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
