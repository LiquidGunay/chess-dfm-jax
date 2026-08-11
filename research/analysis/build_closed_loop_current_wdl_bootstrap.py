#!/usr/bin/env python3
"""Build the sealed current-WDL + closed-loop experiment summary."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "research" / "analysis"
OUTPUT_JSON = ANALYSIS_DIR / "closed_loop_current_wdl_bootstrap_20260729.json"
OUTPUT_CSV = ANALYSIS_DIR / "closed_loop_current_wdl_bootstrap_20260729.csv"

TRAIN_DIR = ROOT / "research" / "runs" / "posthero_closedloop_currentwdl_u1024_v1"
EVAL_DIR = (
    ROOT
    / "research"
    / "runs"
    / "posthero_closedloop_currentwdl_u1024_v1_fast_eval_v1"
)
SUITE_JSON = ANALYSIS_DIR / "posthero_architecture_suite_20260728.json"
ARENA_DIR = ROOT / "artifacts" / "arena"

ARENAS = {
    "combo_p1_vs_hero_p1": (
        ARENA_DIR / "posthero-combined-first128-p1-vs-hero-p1-v1" / "state.json"
    ),
    "combo_p8_vs_combo_p1": (
        ARENA_DIR / "posthero-combined-first128-p8-vs-p1-v1" / "state.json"
    ),
    "combo_p8_vs_hero_p1": (
        ARENA_DIR / "posthero-combined-first128-p8-vs-hero-p1-v1" / "state.json"
    ),
    "all_heads_policy_only_vs_hero_policy_only": (
        ARENA_DIR
        / "posthero-allheads-policy-only-vs-hero-policy-only-first128-v1"
        / "state.json"
    ),
    "closed_p1_vs_hero_p1": (
        ARENA_DIR
        / "posthero-suite-first128-p1-closed-loop-vs-hero-v1"
        / "state.json"
    ),
    "closed_p8_vs_closed_p1": (
        ARENA_DIR
        / "posthero-suite-first128-closed-loop-p8-vs-p1-v1"
        / "state.json"
    ),
    "closed_p8_vs_hero_p1": (
        ARENA_DIR
        / "posthero-suite-first128-closed-loop-p8-vs-hero-p1-v1"
        / "state.json"
    ),
    "current_wdl_p1_vs_hero_p1": (
        ARENA_DIR
        / "posthero-suite-first128-p1-current-wdl-vs-hero-v1"
        / "state.json"
    ),
}

EXPECTED_COMBO_CHECKPOINT = (
    "845a4a6a08eb71f16a5e493e25be99fba9443b45c3f88818898bc7d9e85ef08e"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def source(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def arena_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload["status"] != "complete":
        raise ValueError(f"Incomplete Arena state: {path}")
    aggregate = payload["aggregate"]
    contract = payload["contract"]
    if (
        aggregate["pair_count"] != 128
        or aggregate["game_count"] != 256
        or aggregate["cap_draw_count"] != 0
        or aggregate["fault_counts"]
        or aggregate["termination_counts"] != {"normal": 256}
        or contract["run"]["block_pairs"] != 16
        or contract["run"]["additional_ply_cap"] != 256
        or contract["run"]["policy_batch_size_cap"] != 16
    ):
        raise ValueError(f"Arena contract/gate drift: {path}")
    candidate_key = next(
        key for key in aggregate["models"] if key.startswith("candidate")
    )
    opponent_key = next(
        key for key in aggregate["models"] if key.startswith("incumbent")
    )
    candidate = aggregate["models"][candidate_key]
    opponent = aggregate["models"][opponent_key]
    interval = aggregate["pair_aware_logistic_interval"]
    return {
        "score": aggregate["pentanomial"]["score"],
        "points": aggregate["pentanomial"]["points"],
        "games": aggregate["game_count"],
        "wins": candidate["wins"],
        "draws": candidate["draws"],
        "losses": candidate["losses"],
        "pentanomial": aggregate["pentanomial"]["counts"],
        "direct_logistic_elo": interval["elo"],
        "direct_hoeffding_elo_95": [
            interval["elo_lower"],
            interval["elo_upper"],
        ],
        "normalized_elo": aggregate["normalized_elo_diagnostics"][
            "normalized_elo"
        ],
        "candidate_refinement_passes": contract["run"][
            "candidate_refinement_passes"
        ],
        "opponent_refinement_passes": contract["run"][
            "opponent_refinement_passes"
        ],
        "candidate_policy_mode": contract["run"][
            "candidate_torch_hero_policy_mode"
        ],
        "opponent_policy_mode": contract["run"][
            "opponent_torch_hero_policy_mode"
        ],
        "candidate_mean_physical_call_ms": (
            1000.0 * candidate["coverage"]["mean_call_seconds"]
        ),
        "opponent_mean_physical_call_ms": (
            1000.0 * opponent["coverage"]["mean_call_seconds"]
        ),
        "candidate_mean_position_ms": (
            1000.0
            * candidate["coverage"]["call_seconds"]
            / candidate["coverage"]["positions"]
        ),
        "opponent_mean_position_ms": (
            1000.0
            * opponent["coverage"]["call_seconds"]
            / opponent["coverage"]["positions"]
        ),
        "gameplay_wall_seconds": aggregate["gameplay_wall_seconds"],
        "cap_draw_count": aggregate["cap_draw_count"],
        "fault_counts": aggregate["fault_counts"],
        "incomplete_coverage_positions": (
            candidate["coverage"]["incomplete_coverage_positions"]
            + opponent["coverage"]["incomplete_coverage_positions"]
        ),
        "pair_scores": aggregate["pair_scores"],
        "source": source(path),
    }


def paired_bootstrap_delta(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    seed: int,
    samples: int = 50_000,
) -> dict[str, Any]:
    left_scores = np.asarray(left["pair_scores"], dtype=np.float64) / 2.0
    right_scores = np.asarray(right["pair_scores"], dtype=np.float64) / 2.0
    if left_scores.shape != right_scores.shape or left_scores.shape != (128,):
        raise ValueError("Matched Arena pair vectors must both have length 128")
    delta = left_scores - right_scores
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    draws = delta[indices].mean(axis=1)
    return {
        "delta_score": float(delta.mean()),
        "matched_opening_bootstrap_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "samples": samples,
        "seed": seed,
        "unit": "one opening-pair index, synchronously matched across runs",
    }


def compact_validation(metrics: dict[str, Any]) -> dict[str, Any]:
    pred_rank = [
        metrics[f"pred_effective_rank_by_horizon_h{horizon}"]
        for horizon in range(1, 9)
    ]
    target_rank = [
        metrics[f"target_effective_rank_by_horizon_h{horizon}"]
        for horizon in range(1, 9)
    ]
    rms_ratio = [
        metrics[f"pred_target_rms_ratio_by_horizon_h{horizon}"]
        for horizon in range(1, 9)
    ]
    return {
        "examples": 8192,
        "dfm_ce_loss": metrics["dfm_ce_loss"],
        "dfm_ce_loss_by_horizon": [
            metrics[f"dfm_ce_loss_by_horizon_h{horizon}"]
            for horizon in range(1, 9)
        ],
        "accuracy": metrics["accuracy"],
        "first_legal_mass": metrics["first_legal_mass"],
        "root_legal_conditional_ce": metrics["root_legal_conditional_ce"],
        "root_legal_top1_accuracy": metrics["root_legal_top1_accuracy"],
        "dfm_closed_loop_dfm_ce_improvement": metrics[
            "dfm_closed_loop_dfm_ce_improvement"
        ],
        "jepa_positive_loss": metrics["jepa_positive_loss"],
        "jepa_sigreg_loss": metrics["jepa_sigreg_loss"],
        "jepa_pred_sigreg_loss": metrics["jepa_pred_sigreg_loss"],
        "wdl_loss": metrics["wdl_loss"],
        "wdl_accuracy": metrics["wdl_accuracy"],
        "wdl_current_loss": metrics["wdl_current_loss"],
        "wdl_current_accuracy": metrics["wdl_current_accuracy"],
        "wdl_expected_value_mse": metrics["wdl_expected_value_mse"],
        "wdl_ece_15": metrics["wdl_ece_15"],
        "z_pred_norm": metrics["z_pred_norm"],
        "z_target_norm": metrics["z_target_norm"],
        "pred_effective_rank_by_horizon": pred_rank,
        "target_effective_rank_by_horizon": target_rank,
        "pred_target_effective_rank_ratio_by_horizon": [
            left / right for left, right in zip(pred_rank, target_rank, strict=True)
        ],
        "pred_target_rms_ratio_by_horizon": rms_ratio,
        "pred_target_rms_ratio_range": [min(rms_ratio), max(rms_ratio)],
    }


def main() -> None:
    training_report_path = TRAIN_DIR / "report.json"
    loss_summary_path = TRAIN_DIR / "loss_summary.json"
    checkpoint_manifest_path = TRAIN_DIR / "checkpoint" / "manifest.json"
    evaluation_report_path = EVAL_DIR / "report.json"
    training_report = load_json(training_report_path)
    loss_summary = load_json(loss_summary_path)
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    evaluation_report = load_json(evaluation_report_path)
    suite = load_json(SUITE_JSON)

    checkpoint_sha = checkpoint_manifest["state"]["sha256"]
    if (
        training_report["updates"] != 1024
        or training_report["examples"] != 1_048_576
        or checkpoint_manifest["optimizer_update"] != 1024
        or checkpoint_sha != EXPECTED_COMBO_CHECKPOINT
        or not evaluation_report["gate_pass"]
        or evaluation_report["evaluation_examples"] != 8192
    ):
        raise ValueError("Combined training/evaluation contract drift")

    arenas = {name: arena_summary(path) for name, path in ARENAS.items()}
    combo_training = {
        "updates": training_report["updates"],
        "examples": training_report["examples"],
        "train_wall_seconds": training_report["train_wall_seconds"],
        "examples_per_second_end_to_end": training_report[
            "examples_per_second_end_to_end"
        ],
        "peak_hbm_bytes": training_report["gpu_peak_memory_allocated_bytes"],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": checkpoint_manifest["state"]["size_bytes"],
        "trailing_64": loss_summary["last_window"],
        "source_files": {
            "report": source(training_report_path),
            "loss_summary": source(loss_summary_path),
            "checkpoint_manifest": source(checkpoint_manifest_path),
        },
    }
    combo_validation = compact_validation(evaluation_report["metrics"])
    combo_validation["evaluation_seconds"] = evaluation_report["evaluation_seconds"]
    combo_validation["source"] = source(evaluation_report_path)

    matched_deltas = {
        "combo_minus_closed_loop_refinement": paired_bootstrap_delta(
            arenas["combo_p8_vs_combo_p1"],
            arenas["closed_p8_vs_closed_p1"],
            seed=20260729,
        ),
        "combo_minus_closed_loop_one_pass_vs_hero": paired_bootstrap_delta(
            arenas["combo_p1_vs_hero_p1"],
            arenas["closed_p1_vs_hero_p1"],
            seed=20260730,
        ),
        "combo_minus_current_wdl_one_pass_vs_hero": paired_bootstrap_delta(
            arenas["combo_p1_vs_hero_p1"],
            arenas["current_wdl_p1_vs_hero_p1"],
            seed=20260731,
        ),
        "combo_minus_closed_loop_eight_pass_vs_hero": paired_bootstrap_delta(
            arenas["combo_p8_vs_hero_p1"],
            arenas["closed_p8_vs_hero_p1"],
            seed=20260732,
        ),
    }

    payload = {
        "schema_version": "closed-loop-current-wdl-bootstrap-analysis-v1",
        "generated_utc": "2026-07-29T06:30:00+00:00",
        "question": (
            "Can current-state WDL retain a stronger one-pass policy while "
            "preserving the isolated predicted-JEPA closed-loop refinement gain?"
        ),
        "answer": (
            "No at this budget. One-pass strength partially recovers, but "
            "eight passes score below one pass and below Hero."
        ),
        "combo": {
            "training": combo_training,
            "validation": combo_validation,
        },
        "controls": {
            name: suite["models"][name]
            for name in ("hero", "current_wdl", "closed_loop", "all_heads")
        },
        "arena": arenas,
        "matched_deltas": matched_deltas,
        "interpretation": {
            "one_pass_bootstrap": (
                "Combo p1 improves 2.34 percentage points over isolated "
                "closed-loop p1 versus Hero, but the matched interval includes zero."
            ),
            "refinement_interference": (
                "Combo p8-vs-p1 is 6.05 percentage points below isolated "
                "closed-loop p8-vs-p1; the matched opening bootstrap excludes zero."
            ),
            "all_heads_h1_diagnosis": (
                "All-heads policy-only scores 24.80% against Hero policy-only, "
                "so shared multi-horizon optimization damaged the learned H1 "
                "BT4 path despite identical H1 architecture at initialization."
            ),
            "promotion": "Rejected as a refinement architecture; checkpoint retained.",
        },
        "sources": {
            "posthero_suite": source(SUITE_JSON),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    rows = []
    control_matches = {
        match["candidate"]: match
        for match in suite["primary_round_robin"]["direct_matches"]
        if match["opponent"] == "hero"
    }
    control_refinement = {
        row["candidate"]: row for row in suite["refinement_diagnostics"]
    }
    for name in ("hero", "current_wdl", "closed_loop", "all_heads"):
        model = suite["models"][name]
        rows.append(
            {
                "model": name,
                "train_dfm_ce_trailing64": model["training"]["last_64"][
                    "dfm_ce_loss"
                ],
                "val_dfm_ce": model["validation"]["dfm_ce_loss"],
                "val_accuracy": model["validation"]["accuracy"],
                "val_legal_mass": model["validation"]["first_legal_mass"],
                "p1_score_vs_hero": (
                    0.5 if name == "hero" else control_matches[name]["score"]
                ),
                "p8_score_vs_own_p1": (
                    "" if name == "hero" else control_refinement[name]["score"]
                ),
            }
        )
    rows.append(
        {
            "model": "closed_loop_current_wdl",
            "train_dfm_ce_trailing64": combo_training["trailing_64"]["dfm_ce_loss"],
            "val_dfm_ce": combo_validation["dfm_ce_loss"],
            "val_accuracy": combo_validation["accuracy"],
            "val_legal_mass": combo_validation["first_legal_mass"],
            "p1_score_vs_hero": arenas["combo_p1_vs_hero_p1"]["score"],
            "p8_score_vs_own_p1": arenas["combo_p8_vs_combo_p1"]["score"],
        }
    )
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "output_json": relative(OUTPUT_JSON),
                "output_csv": relative(OUTPUT_CSV),
                "combo_val_dfm_ce": combo_validation["dfm_ce_loss"],
                "combo_p1_vs_hero": arenas["combo_p1_vs_hero_p1"]["score"],
                "combo_p8_vs_p1": arenas["combo_p8_vs_combo_p1"]["score"],
                "combo_p8_vs_hero": arenas["combo_p8_vs_hero_p1"]["score"],
                "all_heads_policy_only_vs_hero_policy_only": arenas[
                    "all_heads_policy_only_vs_hero_policy_only"
                ]["score"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
