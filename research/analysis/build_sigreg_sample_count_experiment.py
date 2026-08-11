#!/usr/bin/env python3
"""Build the immutable count-64 versus count-256 experiment record."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = (
    ROOT
    / "research/analysis/sigreg_sample_count_256_experiment_20260727.json"
)
OUTPUT_PNG = (
    ROOT
    / "research/analysis/sigreg_sample_count_256_experiment_20260727.png"
)
RUNS = {
    "control_64": "torch_autoresearch_sigreg64_u1024_v1",
    "candidate_256": "torch_autoresearch_sigreg256_u1024_v1",
}
EVALS = {
    "control_64": "torch_autoresearch_sigreg64_u1024_v1_fast_eval_v1",
    "candidate_256": "torch_autoresearch_sigreg256_u1024_v1_fast_eval_v1",
}
ARENAS = {
    "control_64": "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-v1",
    "candidate_256": "autoresearch-sigreg256-u1024-vs-raw-bt4-128pairs-v1",
}
VALIDATION_METRICS = (
    "dfm_ce_loss",
    "accuracy",
    "first_legal_mass",
    "root_legal_conditional_ce",
    "root_legal_top1_accuracy",
    "jepa_positive_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "loss",
    "wdl_loss",
    "wdl_accuracy",
    "wdl_ece_15",
    "wdl_expected_value_mse",
    "z_pred_norm",
    "z_state_norm",
    "z_target_norm",
)
TRAIN_METRICS = (
    "dfm_ce_loss",
    "accuracy",
    "first_legal_mass",
    "jepa_positive_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "loss",
    "wdl_loss",
    "z_pred_norm",
    "z_state_norm",
    "z_target_norm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def relative_source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def dictionary_diff(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(control) | set(candidate))
    return {
        key: {
            "control_64": control.get(key),
            "candidate_256": candidate.get(key),
        }
        for key in keys
        if control.get(key) != candidate.get(key)
    }


def checkpoint_record(
    report: dict[str, Any],
    manifest_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    expected = manifest["state"]
    if report["checkpoint"]["state"] != expected:
        raise ValueError(f"Checkpoint report/manifest mismatch: {manifest_path}")
    state_present = state_path.is_file()
    if state_present:
        observed_size = state_path.stat().st_size
        observed_sha256 = sha256_file(state_path)
        if observed_size != int(expected["size_bytes"]):
            raise ValueError(f"Checkpoint size mismatch: {state_path}")
        if observed_sha256 != expected["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {state_path}")
    else:
        observed_size = None
        observed_sha256 = None
    return {
        "expected_sha256": expected["sha256"],
        "expected_size_bytes": int(expected["size_bytes"]),
        "leaf_count": int(expected["leaf_count"]),
        "model_present": state_present,
        "observed_sha256": observed_sha256,
        "observed_size_bytes": observed_size,
        "manifest": relative_source(manifest_path),
    }


def training_record(
    report: dict[str, Any],
    loss_summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "updates": int(report["updates"]),
        "examples": int(report["examples"]),
        "train_seconds": float(report["train_seconds"]),
        "train_wall_seconds": float(report["train_wall_seconds"]),
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
            sum(
                bool(row["optimizer_skipped_nonfinite"])
                for row in metric_rows
            )
        ),
        "terminal": {
            key: float(loss_summary["terminal"][key])
            for key in TRAIN_METRICS
            if key in loss_summary["terminal"]
        },
        "last_64_update_mean": {
            key: float(loss_summary["last_window"][key])
            for key in TRAIN_METRICS
            if key in loss_summary["last_window"]
        },
    }


def validation_record(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("gate_pass"):
        raise ValueError("Frozen fast validation gate did not pass")
    metrics = report["metrics"]
    horizons = (1, 8)
    rank = {}
    for latent in ("pred", "target"):
        rank[latent] = {
            f"h{horizon}": {
                "effective_rank": float(
                    metrics[
                        f"{latent}_effective_rank_by_horizon_h{horizon}"
                    ]
                ),
                "stable_rank": float(
                    metrics[f"{latent}_stable_rank_by_horizon_h{horizon}"]
                ),
                "feature_std_mean": float(
                    metrics[
                        f"{latent}_feature_std_mean_by_horizon_h{horizon}"
                    ]
                ),
                "feature_std_p05": float(
                    metrics[
                        f"{latent}_feature_std_p05_by_horizon_h{horizon}"
                    ]
                ),
            }
            for horizon in horizons
        }
    rank["pred_target_cosine"] = {
        f"h{horizon}": float(
            metrics[f"pred_target_cosine_by_horizon_h{horizon}"]
        )
        for horizon in horizons
    }
    return {
        "examples": int(report["evaluation_examples"]),
        "seconds": float(report["evaluation_seconds"]),
        "sigreg_example_count": int(report["config"]["sigreg_example_count"]),
        "pool_manifest_sha256": report["pool"]["manifest_sha256"],
        "pool_indices_sha256": report["pool"]["indices_sha256"],
        "metrics": {
            key: float(metrics[key])
            for key in VALIDATION_METRICS
            if key in metrics
        },
        "rank_diagnostics": rank,
    }


def arena_record(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("status") != "complete":
        raise ValueError("Arena did not complete")
    aggregate = state["aggregate"]
    candidate_id = next(
        model_id
        for model_id in aggregate["models"]
        if model_id.startswith("candidate-")
    )
    reference_id = next(
        model_id
        for model_id in aggregate["models"]
        if model_id.startswith("raw-bt4-")
    )
    candidate = aggregate["models"][candidate_id]
    interval = aggregate["pair_aware_logistic_interval"]
    return {
        "contract_sha256": state["contract_sha256"],
        "payload_sha256": state["payload_sha256"],
        "pair_count": int(aggregate["pair_count"]),
        "game_count": int(aggregate["game_count"]),
        "score": float(aggregate["pentanomial"]["score"]),
        "points": float(aggregate["pentanomial"]["points"]),
        "pentanomial": list(aggregate["pentanomial"]["counts"]),
        "logistic_elo": float(interval["elo"]),
        "score_lower": float(interval["score_lower"]),
        "score_upper": float(interval["score_upper"]),
        "elo_lower": float(interval["elo_lower"]),
        "elo_upper": float(interval["elo_upper"]),
        "candidate_model_id": candidate_id,
        "reference_model_id": reference_id,
        "wins": int(candidate["wins"]),
        "draws": int(candidate["draws"]),
        "losses": int(candidate["losses"]),
        "mean_physical_call_ms": (
            1000.0 * float(candidate["coverage"]["mean_call_seconds"])
        ),
        "gameplay_seconds": float(aggregate["gameplay_wall_seconds"]),
        "fault_counts": dict(aggregate["fault_counts"]),
        "cap_draw_count": int(aggregate["cap_draw_count"]),
        "termination_counts": dict(aggregate["termination_counts"]),
        "run_contract": state["contract"]["run"],
        "tier_contract": state["contract"]["tier"],
        "_pair_scores": (
            np.asarray(aggregate["pair_scores"], dtype=np.float64) / 2.0
        ),
    }


def paired_arena_delta(
    control: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    if control.shape != candidate.shape or control.ndim != 1:
        raise ValueError("Arena pair-score arrays are not aligned")
    delta = candidate - control
    count = len(delta)
    mean = float(delta.mean())
    standard_deviation = float(delta.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(count)
    critical = float(stats.t.ppf(0.975, count - 1))
    t_statistic = mean / standard_error if standard_error else math.nan
    p_value = (
        float(2.0 * stats.t.sf(abs(t_statistic), count - 1))
        if standard_error
        else math.nan
    )
    counts = Counter(float(value) for value in delta)
    return {
        "pair_count": count,
        "candidate_minus_control_score": mean,
        "paired_standard_deviation": standard_deviation,
        "paired_standard_error": standard_error,
        "paired_t_95_lower": mean - critical * standard_error,
        "paired_t_95_upper": mean + critical * standard_error,
        "paired_t_statistic": t_statistic,
        "paired_t_two_sided_p_value": p_value,
        "candidate_better_pairs": int(np.count_nonzero(delta > 0)),
        "equal_pairs": int(np.count_nonzero(delta == 0)),
        "candidate_worse_pairs": int(np.count_nonzero(delta < 0)),
        "delta_counts": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        "method_note": (
            "Descriptive paired t interval over the same frozen opening-pair "
            "scores. The preregistered non-worse Arena gate, not this "
            "post-hoc p-value, determines the screening decision."
        ),
    }


def metric_delta(
    control: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    result = {}
    for key in sorted(set(control) & set(candidate)):
        control_value = float(control[key])
        candidate_value = float(candidate[key])
        result[key] = {
            "control_64": control_value,
            "candidate_256": candidate_value,
            "candidate_minus_control": candidate_value - control_value,
            "relative_change": (
                None
                if control_value == 0.0
                else candidate_value / control_value - 1.0
            ),
        }
    return result


def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        raise ValueError("Training trace is shorter than rolling window")
    weights = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, weights, mode="valid")
    updates = np.arange(window, len(values) + 1)
    return updates, smoothed


def build_plot(
    metrics: dict[str, list[dict[str, Any]]],
    validations: dict[str, dict[str, Any]],
    arenas: dict[str, dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7.5),
        constrained_layout=True,
    )
    colors = {"control_64": "#31688e", "candidate_256": "#b73779"}
    labels = {"control_64": "count 64", "candidate_256": "count 256"}
    for name, rows in metrics.items():
        for axis, metric, title in (
            (axes[0, 0], "dfm_ce_loss", "Training DFM CE"),
            (axes[0, 1], "jepa_positive_loss", "Training JEPA MSE"),
            (axes[1, 0], "loss", "Training weighted total"),
        ):
            values = np.asarray(
                [float(row[metric]) for row in rows],
                dtype=np.float64,
            )
            updates, smoothed = rolling_mean(values, 64)
            axis.plot(
                updates,
                smoothed,
                color=colors[name],
                label=labels[name],
            )
            axis.set(
                title=f"{title} (trailing 64-update mean)",
                xlabel="optimizer update",
                ylabel=metric,
            )
            axis.axvline(
                566866 / 1024,
                color="0.5",
                linestyle=":",
                linewidth=1,
            )
    for axis in axes.flat[:3]:
        axis.legend()
        axis.grid(alpha=0.25)

    names = list(RUNS)
    x = np.arange(len(names))
    scores = [arenas[name]["score"] for name in names]
    lower = [
        arenas[name]["score"] - arenas[name]["score_lower"] for name in names
    ]
    upper = [
        arenas[name]["score_upper"] - arenas[name]["score"] for name in names
    ]
    axes[1, 1].bar(
        x,
        scores,
        color=[colors[name] for name in names],
        width=0.55,
    )
    axes[1, 1].errorbar(
        x,
        scores,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        color="black",
        capsize=4,
    )
    axes[1, 1].axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    for index, name in enumerate(names):
        axes[1, 1].annotate(
            (
                f"{scores[index]:.4f}\n"
                f"val CE {validations[name]['metrics']['dfm_ce_loss']:.4f}"
            ),
            (index, scores[index]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )
    axes[1, 1].set(
        title="Paired Arena score vs raw BT4",
        ylabel="score",
        xticks=x,
        xticklabels=[labels[name] for name in names],
        ylim=(0.5, 1.02),
    )
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "SIGReg finite-sample estimator: count 64 versus 256",
        fontsize=14,
    )
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> None:
    paths: dict[str, dict[str, Path]] = {}
    for name in RUNS:
        run_root = ROOT / "research/runs" / RUNS[name]
        eval_root = ROOT / "research/runs" / EVALS[name]
        arena_root = ROOT / "artifacts/arena" / ARENAS[name]
        paths[name] = {
            "run_config": run_root / "run_config.json",
            "training_report": run_root / "report.json",
            "loss_summary": run_root / "loss_summary.json",
            "metrics": run_root / "metrics.jsonl",
            "checkpoint_manifest": run_root / "checkpoint/manifest.json",
            "checkpoint_state": run_root / "checkpoint/model.safetensors",
            "validation": eval_root / "report.json",
            "arena": arena_root / "state.json",
        }
    required = [
        path
        for run_paths in paths.values()
        for label, path in run_paths.items()
        if label != "checkpoint_state"
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing experiment inputs: {missing}")

    configs = {
        name: load_json(run_paths["run_config"])
        for name, run_paths in paths.items()
    }
    reports = {
        name: load_json(run_paths["training_report"])
        for name, run_paths in paths.items()
    }
    loss_summaries = {
        name: load_json(run_paths["loss_summary"])
        for name, run_paths in paths.items()
    }
    metric_rows = {
        name: load_jsonl(run_paths["metrics"])
        for name, run_paths in paths.items()
    }
    validation_reports = {
        name: load_json(run_paths["validation"])
        for name, run_paths in paths.items()
    }
    arena_states = {
        name: load_json(run_paths["arena"])
        for name, run_paths in paths.items()
    }

    config_diff = dictionary_diff(
        configs["control_64"]["config"],
        configs["candidate_256"]["config"],
    )
    if set(config_diff) != {"sigreg_example_count"}:
        raise ValueError(f"Unexpected scientific config differences: {config_diff}")
    args_control = dict(configs["control_64"]["args"])
    args_candidate = dict(configs["candidate_256"]["args"])
    args_control.pop("output_dir")
    args_candidate.pop("output_dir")
    args_diff = dictionary_diff(args_control, args_candidate)
    if set(args_diff) != {"sigreg_example_count"}:
        raise ValueError(f"Unexpected command differences: {args_diff}")
    if configs["control_64"]["data"] != configs["candidate_256"]["data"]:
        raise ValueError("Matched runs did not record the same data contract")
    if len(metric_rows["control_64"]) != len(metric_rows["candidate_256"]):
        raise ValueError("Matched runs have different metric-row counts")

    training = {
        name: training_record(
            reports[name],
            loss_summaries[name],
            metric_rows[name],
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
    validations = {
        name: validation_record(validation_reports[name]) for name in RUNS
    }
    if validations["control_64"]["sigreg_example_count"] != 64:
        raise ValueError("Control validation did not use the fixed count-64 estimator")
    if validations["candidate_256"]["sigreg_example_count"] != 64:
        raise ValueError("Candidate validation did not use the fixed count-64 estimator")
    for key in ("pool_manifest_sha256", "pool_indices_sha256", "examples"):
        if validations["control_64"][key] != validations["candidate_256"][key]:
            raise ValueError(f"Frozen validation mismatch: {key}")

    arenas = {
        name: arena_record(arena_states[name]) for name in RUNS
    }
    if arenas["control_64"]["run_contract"] != arenas["candidate_256"]["run_contract"]:
        raise ValueError("Arena run contracts are not matched")
    if arenas["control_64"]["tier_contract"] != arenas["candidate_256"]["tier_contract"]:
        raise ValueError("Arena opening tiers are not matched")
    if (
        arenas["control_64"]["reference_model_id"]
        != arenas["candidate_256"]["reference_model_id"]
    ):
        raise ValueError("Arena reference models are not matched")

    arena_delta = paired_arena_delta(
        arenas["control_64"]["_pair_scores"],
        arenas["candidate_256"]["_pair_scores"],
    )
    arena_delta["descriptive_logistic_elo_delta"] = (
        arenas["candidate_256"]["logistic_elo"]
        - arenas["control_64"]["logistic_elo"]
    )
    public_arenas = {
        name: {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        for name, record in arenas.items()
    }

    sources = {
        name: {
            label: relative_source(path)
            for label, path in run_paths.items()
            if label != "checkpoint_state"
        }
        for name, run_paths in paths.items()
    }
    validation_delta = metric_delta(
        validations["control_64"]["metrics"],
        validations["candidate_256"]["metrics"],
    )
    training_last_window_delta = metric_delta(
        training["control_64"]["last_64_update_mean"],
        training["candidate_256"]["last_64_update_mean"],
    )
    throughput_delta = (
        training["candidate_256"]["examples_per_second_end_to_end"]
        / training["control_64"]["examples_per_second_end_to_end"]
        - 1.0
    )
    peak_hbm_delta = (
        training["candidate_256"]["gpu_peak_memory_allocated_bytes"]
        - training["control_64"]["gpu_peak_memory_allocated_bytes"]
    )
    build_plot(metric_rows, validations, public_arenas)

    result = {
        "schema_version": "sigreg-sample-count-experiment-v1",
        "question": (
            "Does count 256 improve fresh BT4+DFM+JEPA learning relative "
            "to the matched count-64 finite-sample SIGReg estimator?"
        ),
        "comparison": {
            "control": "control_64",
            "candidate": "candidate_256",
            "only_scientific_config_difference": config_diff,
            "only_normalized_command_difference": args_diff,
            "same_data_contract": True,
            "same_initialization_seed": (
                configs["control_64"]["config"]["init_seed"]
                == configs["candidate_256"]["config"]["init_seed"]
                == 0
            ),
            "same_git_commit": (
                configs["control_64"]["git_commit"]
                == configs["candidate_256"]["git_commit"]
            ),
            "fixed_validation_sigreg_example_count": 64,
        },
        "training": training,
        "training_last_64_update_delta": training_last_window_delta,
        "systems_delta": {
            "candidate_throughput_relative_change": throughput_delta,
            "candidate_peak_hbm_allocated_delta_bytes": peak_hbm_delta,
        },
        "checkpoints": checkpoints,
        "frozen_fast_validation": validations,
        "frozen_fast_validation_delta": validation_delta,
        "paired_arena": public_arenas,
        "paired_arena_delta": arena_delta,
        "decision": {
            "status": "reject_candidate_256",
            "retain_sigreg_example_count": 64,
            "retain_sigreg_coefficients": {
                "target": 2.0,
                "prediction": 2.0,
            },
            "advance_candidate_to_10_percent_epoch": False,
            "candidate_passes_validation_ce_direction": (
                validation_delta["dfm_ce_loss"]["candidate_minus_control"] < 0
            ),
            "candidate_passes_no_rank_regression": False,
            "candidate_passes_arena_non_worse": (
                arena_delta["candidate_minus_control_score"] >= 0
            ),
            "reason": (
                "Count 256 slightly improves fixed validation DFM CE and "
                "materially improves JEPA MSE, but modestly reduces several "
                "rank diagnostics and regresses the same-opening Arena score "
                "by 0.095703. It therefore fails the preregistered non-worse "
                "Arena and no-rank-regression gates."
            ),
            "causal_scope": (
                "One matched short-run screen. The result rejects count 256 "
                "for this recipe; it does not establish that estimator noise "
                "is generally beneficial."
            ),
        },
        "plot": {
            "path": str(OUTPUT_PNG.relative_to(ROOT)),
            "sha256": sha256_file(OUTPUT_PNG),
            "training_smoothing": "trailing 64-update arithmetic mean",
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
                "validation_dfm_ce_delta": validation_delta["dfm_ce_loss"][
                    "candidate_minus_control"
                ],
                "validation_jepa_mse_delta": validation_delta[
                    "jepa_positive_loss"
                ]["candidate_minus_control"],
                "arena_score_delta": arena_delta[
                    "candidate_minus_control_score"
                ],
                "arena_paired_t_95": [
                    arena_delta["paired_t_95_lower"],
                    arena_delta["paired_t_95_upper"],
                ],
                "decision": result["decision"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
