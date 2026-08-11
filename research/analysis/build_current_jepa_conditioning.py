#!/usr/bin/env python3
"""Build the current-JEPA-state DFM-conditioning experiment record."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats

from research.analysis.build_sigreg_sample_count_experiment import (
    ROOT,
    checkpoint_record,
    load_json,
    load_jsonl,
    relative_source,
    rolling_mean,
    sha256_file,
    validation_record,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_JSON = (
    ROOT / "research/analysis/current_jepa_conditioning_20260728.json"
)
OUTPUT_PNG = (
    ROOT / "research/analysis/current_jepa_conditioning_20260728.png"
)
RUNS = {
    "control": "torch_autoresearch_sigreg64_u1024_v1",
    "candidate": "torch_autoresearch_jepa_condition_u1024_v1",
}
EVALS = {
    "control": "torch_autoresearch_sigreg64_u1024_v1_fast_eval_v1",
    "candidate": (
        "torch_autoresearch_jepa_condition_u1024_v1_fast_eval_v1"
    ),
}
ARENA = (
    "autoresearch-jepa-condition-u1024-vs-sigreg64-u1024-"
    "128pairs-p1-v1"
)
TRAIN_METRICS = (
    "loss",
    "dfm_ce_loss",
    "accuracy",
    "first_legal_mass",
    "root_legal_conditional_ce",
    "jepa_positive_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "wdl_loss",
    "wdl_weighted_loss",
    "z_pred_norm",
    "z_state_norm",
    "z_target_norm",
)
CONDITIONING_METRICS = (
    "dfm_jepa_conditioning_rms",
    "dfm_jepa_conditioning_state_rms",
    "dfm_jepa_conditioning_ratio",
    "dfm_jepa_conditioning_weight_rms",
)


def dictionary_diff(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "control": control.get(key),
            "candidate": candidate.get(key),
        }
        for key in sorted(set(control) | set(candidate))
        if control.get(key) != candidate.get(key)
    }


def metric_delta(
    control: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for key in sorted(set(control) & set(candidate)):
        control_value = float(control[key])
        candidate_value = float(candidate[key])
        result[key] = {
            "control": control_value,
            "candidate": candidate_value,
            "candidate_minus_control": candidate_value - control_value,
            "candidate_over_control": (
                None
                if control_value == 0.0
                else candidate_value / control_value
            ),
        }
    return result


def summarize_rows(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in metrics
    }


def training_record(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    conditioning_active: bool,
) -> dict[str, Any]:
    if len(rows) != int(report["updates"]) or len(rows) != 1024:
        raise ValueError("Training trace is not the matched 1,024 updates")
    record = {
        "updates": int(report["updates"]),
        "examples": int(report["examples"]),
        "train_seconds": float(report["train_seconds"]),
        "examples_per_second_end_to_end": float(
            report["examples_per_second_end_to_end"]
        ),
        "gpu_peak_memory_allocated_bytes": int(
            report["gpu_peak_memory_allocated_bytes"]
        ),
        "gpu_peak_memory_reserved_bytes": int(
            report["gpu_peak_memory_reserved_bytes"]
        ),
        "optimizer_skipped_nonfinite_count": int(
            sum(bool(row["optimizer_skipped_nonfinite"]) for row in rows)
        ),
        "first_64_update_mean": summarize_rows(
            rows[:64],
            TRAIN_METRICS,
        ),
        "last_64_update_mean": summarize_rows(
            rows[-64:],
            TRAIN_METRICS,
        ),
        "terminal": {
            metric: float(rows[-1][metric]) for metric in TRAIN_METRICS
        },
        "conditioning_active": conditioning_active,
    }
    if conditioning_active:
        record["conditioning_last_64_update_mean"] = summarize_rows(
            rows[-64:],
            CONDITIONING_METRICS,
        )
        record["conditioning_terminal"] = {
            metric: float(rows[-1][metric])
            for metric in CONDITIONING_METRICS
        }
    return record


def conditioning_validation_record(
    report: dict[str, Any],
) -> dict[str, Any]:
    record = validation_record(report)
    record["conditioning_active"] = bool(
        report["config"].get(
            "dfm_condition_on_current_jepa_state",
            False,
        )
    )
    record["conditioning_diagnostics"] = {
        metric: float(report["metrics"].get(metric, 0.0))
        for metric in CONDITIONING_METRICS
    }
    return record


def rank_retention(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for latent in ("pred", "target"):
        result[latent] = {}
        for horizon in ("h1", "h8"):
            result[latent][horizon] = {}
            for metric in (
                "effective_rank",
                "stable_rank",
                "feature_std_mean",
                "feature_std_p05",
            ):
                control_value = float(control[latent][horizon][metric])
                candidate_value = float(candidate[latent][horizon][metric])
                result[latent][horizon][metric] = {
                    "control": control_value,
                    "candidate": candidate_value,
                    "candidate_over_control": (
                        candidate_value / control_value
                    ),
                }
    return result


def direct_arena_record(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("status") != "complete":
        raise ValueError("Direct incumbent Arena did not complete")
    aggregate = state["aggregate"]
    candidate_id = next(
        model_id
        for model_id in aggregate["models"]
        if model_id.startswith("candidate-torch-hero-")
    )
    incumbent_id = next(
        model_id
        for model_id in aggregate["models"]
        if model_id.startswith("incumbent-torch-hero-")
    )
    candidate = aggregate["models"][candidate_id]
    incumbent = aggregate["models"][incumbent_id]
    normalized_scores = (
        np.asarray(aggregate["pair_scores"], dtype=np.float64) / 2.0
    )
    if len(normalized_scores) != 128:
        raise ValueError("Direct incumbent Arena did not use 128 pairs")
    score = float(normalized_scores.mean())
    standard_deviation = float(normalized_scores.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(len(normalized_scores))
    critical = float(stats.t.ppf(0.975, len(normalized_scores) - 1))
    t_statistic = (score - 0.5) / standard_error
    interval = aggregate["pair_aware_logistic_interval"]
    return {
        "contract_sha256": state["contract_sha256"],
        "payload_sha256": state["payload_sha256"],
        "candidate_model_id": candidate_id,
        "incumbent_model_id": incumbent_id,
        "pair_count": int(aggregate["pair_count"]),
        "game_count": int(aggregate["game_count"]),
        "score": score,
        "points": float(aggregate["pentanomial"]["points"]),
        "pentanomial": list(aggregate["pentanomial"]["counts"]),
        "candidate_wins": int(candidate["wins"]),
        "candidate_draws": int(candidate["draws"]),
        "candidate_losses": int(candidate["losses"]),
        "candidate_mean_call_ms": (
            1000.0 * float(candidate["coverage"]["mean_call_seconds"])
        ),
        "incumbent_mean_call_ms": (
            1000.0 * float(incumbent["coverage"]["mean_call_seconds"])
        ),
        "latency_ratio": (
            float(candidate["coverage"]["mean_call_seconds"])
            / float(incumbent["coverage"]["mean_call_seconds"])
        ),
        "candidate_incomplete_coverage_positions": int(
            candidate["coverage"]["incomplete_coverage_positions"]
        ),
        "incumbent_incomplete_coverage_positions": int(
            incumbent["coverage"]["incomplete_coverage_positions"]
        ),
        "fault_counts": dict(aggregate["fault_counts"]),
        "cap_draw_count": int(aggregate["cap_draw_count"]),
        "termination_counts": dict(aggregate["termination_counts"]),
        "gameplay_seconds": float(aggregate["gameplay_wall_seconds"]),
        "descriptive_logistic_elo": float(interval["elo"]),
        "hoeffding_score_interval": [
            float(interval["score_lower"]),
            float(interval["score_upper"]),
        ],
        "hoeffding_logistic_elo_interval": [
            float(interval["elo_lower"]),
            float(interval["elo_upper"]),
        ],
        "descriptive_paired_t": {
            "standard_deviation": standard_deviation,
            "standard_error": standard_error,
            "t_statistic_vs_half": t_statistic,
            "two_sided_p_value": float(
                2.0
                * stats.t.sf(
                    abs(t_statistic),
                    len(normalized_scores) - 1,
                )
            ),
            "score_95_lower": score - critical * standard_error,
            "score_95_upper": score + critical * standard_error,
            "candidate_better_pairs": int(
                np.count_nonzero(normalized_scores > 0.5)
            ),
            "equal_pairs": int(
                np.count_nonzero(normalized_scores == 0.5)
            ),
            "candidate_worse_pairs": int(
                np.count_nonzero(normalized_scores < 0.5)
            ),
            "score_counts": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(
                        float(value) for value in normalized_scores
                    ).items()
                )
            },
            "method_note": (
                "Descriptive paired t interval over the frozen development "
                "pairs; it is not a promotion test."
            ),
        },
        "run_contract": state["contract"]["run"],
        "tier_contract": state["contract"]["tier"],
    }


def build_plot(
    rows: dict[str, list[dict[str, Any]]],
    validation: dict[str, dict[str, Any]],
    arena: dict[str, Any],
) -> None:
    colors = {"control": "#31688e", "candidate": "#d1495b"}
    labels = {"control": "Retained Hero", "candidate": "JEPA→DFM bridge"}
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7.5),
        constrained_layout=True,
    )
    for name, metrics in rows.items():
        updates, values = rolling_mean(
            np.asarray(
                [float(row["dfm_ce_loss"]) for row in metrics],
                dtype=np.float64,
            ),
            64,
        )
        axes[0, 0].plot(
            updates,
            values,
            color=colors[name],
            label=labels[name],
        )
        _, values = rolling_mean(
            np.asarray(
                [float(row["jepa_positive_loss"]) for row in metrics],
                dtype=np.float64,
            ),
            64,
        )
        axes[0, 1].plot(
            updates,
            values,
            color=colors[name],
            label=labels[name],
        )
    _, conditioning = rolling_mean(
        np.asarray(
            [
                float(row["dfm_jepa_conditioning_ratio"])
                for row in rows["candidate"]
            ],
            dtype=np.float64,
        ),
        64,
    )
    axes[1, 0].plot(
        updates,
        conditioning,
        color=colors["candidate"],
        label="conditioning / DFM-state RMS",
    )
    for axis, title, ylabel in (
        (axes[0, 0], "Training DFM CE", "cross-entropy"),
        (axes[0, 1], "Training JEPA MSE", "MSE"),
        (axes[1, 0], "Learned conditioning scale", "RMS ratio"),
    ):
        axis.axvline(
            566866 / 1024,
            color="0.5",
            linestyle=":",
            linewidth=1,
        )
        axis.set(
            title=f"{title} (trailing 64-update mean)",
            xlabel="optimizer update",
            ylabel=ylabel,
        )
        axis.grid(alpha=0.25)
        axis.legend()

    score = float(arena["score"])
    descriptive = arena["descriptive_paired_t"]
    axes[1, 1].errorbar(
        [0],
        [score],
        yerr=[
            [score - descriptive["score_95_lower"]],
            [descriptive["score_95_upper"] - score],
        ],
        fmt="o",
        color=colors["candidate"],
        capsize=5,
    )
    axes[1, 1].axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axes[1, 1].annotate(
        (
            f"score {score:.4f}\n"
            f"candidate val CE "
            f"{validation['candidate']['metrics']['dfm_ce_loss']:.4f}\n"
            f"control val CE "
            f"{validation['control']['metrics']['dfm_ce_loss']:.4f}"
        ),
        (0, score),
        xytext=(8, 8),
        textcoords="offset points",
    )
    axes[1, 1].set(
        title="Direct score vs retained Hero",
        ylabel="paired game score",
        xticks=[0],
        xticklabels=["JEPA→DFM bridge"],
        xlim=(-0.5, 0.7),
        ylim=(0.43, 0.58),
    )
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Current-JEPA-state conditioning of DFM",
        fontsize=14,
    )
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> None:
    paths: dict[str, dict[str, Path]] = {}
    for name in RUNS:
        run_root = ROOT / "research/runs" / RUNS[name]
        eval_root = ROOT / "research/runs" / EVALS[name]
        paths[name] = {
            "run_config": run_root / "run_config.json",
            "training_report": run_root / "report.json",
            "metrics": run_root / "metrics.jsonl",
            "loss_summary": run_root / "loss_summary.json",
            "checkpoint_manifest": run_root / "checkpoint/manifest.json",
            "checkpoint_state": run_root / "checkpoint/model.safetensors",
            "validation": eval_root / "report.json",
        }
    arena_path = ROOT / "artifacts/arena" / ARENA / "state.json"
    required = [
        path
        for run_paths in paths.values()
        for label, path in run_paths.items()
        if label != "checkpoint_state"
    ] + [arena_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing conditioning inputs: {missing}")

    configs = {
        name: load_json(run_paths["run_config"])
        for name, run_paths in paths.items()
    }
    normalized_configs = {}
    for name, config in configs.items():
        normalized = dict(config["config"])
        normalized.setdefault("jepa_feedback_mode", "none")
        normalized.setdefault(
            "dfm_condition_on_current_jepa_state",
            False,
        )
        normalized_configs[name] = normalized
    scientific_diff = dictionary_diff(
        normalized_configs["control"],
        normalized_configs["candidate"],
    )
    if set(scientific_diff) != {
        "dfm_condition_on_current_jepa_state"
    }:
        raise ValueError(
            f"Unexpected scientific config differences: {scientific_diff}"
        )
    if configs["control"]["data"] != configs["candidate"]["data"]:
        raise ValueError("Conditioning runs have different data contracts")

    reports = {
        name: load_json(run_paths["training_report"])
        for name, run_paths in paths.items()
    }
    rows = {
        name: load_jsonl(run_paths["metrics"])
        for name, run_paths in paths.items()
    }
    if (
        len(rows["control"]) != len(rows["candidate"])
        or len(rows["control"]) != 1024
    ):
        raise ValueError("Training traces do not have matched lengths")
    exact_initial_metrics = all(
        rows["control"][0][metric] == rows["candidate"][0][metric]
        for metric in TRAIN_METRICS
    )
    if not exact_initial_metrics:
        raise ValueError("Zero bridge did not preserve initial metrics")

    training = {
        name: training_record(
            reports[name],
            rows[name],
            conditioning_active=(name == "candidate"),
        )
        for name in RUNS
    }
    checkpoints = {
        name: checkpoint_record(
            reports[name],
            paths[name]["checkpoint_manifest"],
            paths[name]["checkpoint_state"],
        )
        for name in RUNS
    }
    validation_reports = {
        name: load_json(run_paths["validation"])
        for name, run_paths in paths.items()
    }
    validation = {
        name: conditioning_validation_record(validation_reports[name])
        for name in RUNS
    }
    if validation["control"]["conditioning_active"]:
        raise ValueError("Control validation unexpectedly used conditioning")
    if not validation["candidate"]["conditioning_active"]:
        raise ValueError("Candidate validation omitted conditioning")
    for key in ("pool_manifest_sha256", "pool_indices_sha256", "examples"):
        if validation["control"][key] != validation["candidate"][key]:
            raise ValueError(f"Frozen validation mismatch: {key}")

    validation_delta = metric_delta(
        validation["control"]["metrics"],
        validation["candidate"]["metrics"],
    )
    retention = rank_retention(
        validation["control"]["rank_diagnostics"],
        validation["candidate"]["rank_diagnostics"],
    )
    arena_state = load_json(arena_path)
    arena = direct_arena_record(arena_state)
    if (
        arena_state["contract"]["models"]["candidate"]["state"]["sha256"]
        != checkpoints["candidate"]["expected_sha256"]
    ):
        raise ValueError("Arena candidate checkpoint hash mismatch")
    if (
        arena_state["contract"]["models"]["opponent"]["state"]["sha256"]
        != checkpoints["control"]["expected_sha256"]
    ):
        raise ValueError("Arena incumbent checkpoint hash mismatch")
    expected_run_contract = {
        "additional_ply_cap": 256,
        "block_pairs": 16,
        "collect_diagnostics": False,
        "deterministic_greedy_policy": True,
        "jepa_used_at_inference": True,
        "opening_start_index": 0,
        "pair_count": 128,
        "policy_batch_size_cap": 16,
        "policy_timeout_seconds": 30.0,
        "refinement_passes": 1,
        "seed": 0,
    }
    if arena["run_contract"] != expected_run_contract:
        raise ValueError("Direct Arena run contract drift")

    gates = {
        "arena_score_strictly_above_half": arena["score"] > 0.5,
        "validation_dfm_ce_within_0p005": (
            validation_delta["dfm_ce_loss"]["candidate_minus_control"]
            <= 0.005
        ),
        "root_legal_ce_within_0p01": (
            validation_delta["root_legal_conditional_ce"][
                "candidate_minus_control"
            ]
            <= 0.01
        ),
        "accuracy_loss_below_0p1pp": (
            validation_delta["accuracy"]["candidate_minus_control"]
            > -0.001
        ),
        "legal_mass_loss_below_0p2pp": (
            validation_delta["first_legal_mass"][
                "candidate_minus_control"
            ]
            > -0.002
        ),
        "h8_rank_retention_at_least_95pct": all(
            retention[latent]["h8"][metric]["candidate_over_control"]
            >= 0.95
            for latent in ("pred", "target")
            for metric in ("effective_rank", "stable_rank")
        ),
        "h8_feature_tail_retention_at_least_90pct": all(
            retention[latent]["h8"]["feature_std_p05"][
                "candidate_over_control"
            ]
            >= 0.90
            for latent in ("pred", "target")
        ),
        "conditioning_is_finite_and_nonzero": all(
            math.isfinite(
                validation["candidate"]["conditioning_diagnostics"][metric]
            )
            and validation["candidate"]["conditioning_diagnostics"][metric]
            > 0.0
            for metric in (
                "dfm_jepa_conditioning_rms",
                "dfm_jepa_conditioning_weight_rms",
            )
        ),
        "training_throughput_at_least_241p84": (
            training["candidate"]["examples_per_second_end_to_end"]
            >= 241.84
        ),
        "arena_call_time_within_1p25x": arena["latency_ratio"] <= 1.25,
        "zero_nonfinite_skips": all(
            training[name]["optimizer_skipped_nonfinite_count"] == 0
            for name in RUNS
        ),
        "zero_arena_faults": not arena["fault_counts"],
        "zero_cap_draws": arena["cap_draw_count"] == 0,
        "normal_arena_terminations": (
            arena["termination_counts"] == {"normal": 256}
        ),
        "complete_legal_action_coverage": (
            arena["candidate_incomplete_coverage_positions"] == 0
            and arena["incumbent_incomplete_coverage_positions"] == 0
        ),
    }
    passes = all(gates.values())
    build_plot(rows, validation, arena)

    sources = {
        name: {
            label: relative_source(path)
            for label, path in run_paths.items()
            if label != "checkpoint_state"
        }
        for name, run_paths in paths.items()
    }
    sources["arena"] = relative_source(arena_path)
    result = {
        "schema_version": "current-jepa-conditioning-v1",
        "question": (
            "Does a zero-initialized current-JEPA-state residual improve "
            "the one-pass DFM policy?"
        ),
        "comparison": {
            "control": "retained_hero",
            "candidate": "current_jepa_state_conditioning",
            "only_scientific_config_difference": scientific_diff,
            "same_data_contract": True,
            "same_initialization_seed": True,
            "exact_initial_shared_metrics": exact_initial_metrics,
            "added_parameters": 262144,
            "training_horizon": 8,
            "arena_refinement_passes": 1,
        },
        "training": training,
        "training_last_64_update_delta": metric_delta(
            training["control"]["last_64_update_mean"],
            training["candidate"]["last_64_update_mean"],
        ),
        "systems_delta": {
            "throughput_relative_change": (
                training["candidate"]["examples_per_second_end_to_end"]
                / training["control"]["examples_per_second_end_to_end"]
                - 1.0
            ),
            "peak_hbm_allocated_delta_bytes": (
                training["candidate"]["gpu_peak_memory_allocated_bytes"]
                - training["control"]["gpu_peak_memory_allocated_bytes"]
            ),
            "arena_call_time_ratio": arena["latency_ratio"],
        },
        "checkpoints": checkpoints,
        "frozen_fast_validation": validation,
        "frozen_fast_validation_delta": validation_delta,
        "rank_retention": retention,
        "direct_incumbent_arena": arena,
        "decision": {
            "status": (
                "advance_to_fresh_repeat_and_10pct"
                if passes
                else "reject_current_jepa_broadcast_conditioning"
            ),
            "gates": gates,
            "all_preregistered_gates_pass": passes,
            "active_incumbent_checkpoint_sha256": checkpoints["control"][
                "expected_sha256"
            ],
            "causal_scope": (
                "One deterministic matched 3.70%-epoch screen. The negative "
                "result closes only the zero-initialized shared broadcast "
                "bridge; it does not reject other JEPA/DFM coupling designs."
            ),
        },
        "plot": {
            "path": str(OUTPUT_PNG.relative_to(ROOT)),
            "sha256": sha256_file(OUTPUT_PNG),
            "training_smoothing": "trailing 64-update arithmetic mean",
            "arena_interval": "descriptive paired t; not promotion eligible",
        },
        "sources": sources,
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": str(OUTPUT_JSON),
                "output_png": str(OUTPUT_PNG),
                "validation_dfm_ce_delta": validation_delta[
                    "dfm_ce_loss"
                ]["candidate_minus_control"],
                "arena_score": arena["score"],
                "arena_paired_t_p_value": arena[
                    "descriptive_paired_t"
                ]["two_sided_p_value"],
                "decision": result["decision"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
