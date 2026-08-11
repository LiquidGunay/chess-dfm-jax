#!/usr/bin/env python3
"""Build the immutable hero WDL-ablation comparison record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from research.analysis.build_sigreg_sample_count_experiment import (
    ROOT,
    arena_record,
    checkpoint_record,
    load_json,
    load_jsonl,
    paired_arena_delta,
    relative_source,
    rolling_mean,
    sha256_file,
    validation_record,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_JSON = ROOT / "research/analysis/hero_wdl_ablation_20260727.json"
OUTPUT_PNG = ROOT / "research/analysis/hero_wdl_ablation_20260727.png"
RUNS = {
    "control_wdl025": "torch_autoresearch_sigreg64_u1024_v1",
    "candidate_wdl0": "torch_autoresearch_wdl0_u1024_v1",
}
EVALS = {
    "control_wdl025": "torch_autoresearch_sigreg64_u1024_v1_fast_eval_v1",
    "candidate_wdl0": "torch_autoresearch_wdl0_u1024_v1_fast_eval_v1",
}
ARENAS = {
    "control_wdl025": (
        "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-v1"
    ),
    "candidate_wdl0": (
        "autoresearch-wdl0-u1024-vs-raw-bt4-128pairs-v1"
    ),
}
TRAIN_METRICS = (
    "dfm_ce_loss",
    "accuracy",
    "first_legal_mass",
    "root_legal_conditional_ce",
    "jepa_positive_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "wdl_loss",
    "wdl_weighted_loss",
    "loss",
    "z_pred_norm",
    "z_state_norm",
    "z_target_norm",
)


def dictionary_diff(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return config differences with labels specific to this ablation."""
    keys = sorted(set(control) | set(candidate))
    return {
        key: {
            "control_wdl025": control.get(key),
            "candidate_wdl0": candidate.get(key),
        }
        for key in keys
        if control.get(key) != candidate.get(key)
    }


def metric_delta(
    control: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    """Return metric deltas with labels specific to this ablation."""
    result = {}
    for key in sorted(set(control) & set(candidate)):
        control_value = float(control[key])
        candidate_value = float(candidate[key])
        result[key] = {
            "control_wdl025": control_value,
            "candidate_wdl0": candidate_value,
            "candidate_minus_control": candidate_value - control_value,
            "relative_change": (
                None
                if control_value == 0.0
                else candidate_value / control_value - 1.0
            ),
        }
    return result


def training_record(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(rows) != int(report["updates"]):
        raise ValueError("Training report/metrics update count mismatch")

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            metric: float(
                np.mean([float(row[metric]) for row in selected])
            )
            for metric in TRAIN_METRICS
        }

    terminal = {
        metric: float(rows[-1][metric]) for metric in TRAIN_METRICS
    }
    return {
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
        "first_64_update_mean": summarize(rows[:64]),
        "last_64_update_mean": summarize(rows[-64:]),
        "terminal": terminal,
    }


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
                control_value = float(
                    control[latent][horizon][metric]
                )
                candidate_value = float(
                    candidate[latent][horizon][metric]
                )
                result[latent][horizon][metric] = {
                    "control_wdl025": control_value,
                    "candidate_wdl0": candidate_value,
                    "candidate_over_control": (
                        candidate_value / control_value
                    ),
                }
    return result


def build_plot(
    rows: dict[str, list[dict[str, Any]]],
    validation: dict[str, dict[str, Any]],
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
    colors = {
        "control_wdl025": "#31688e",
        "candidate_wdl0": "#e56b2f",
    }
    labels = {
        "control_wdl025": "WDL 0.25",
        "candidate_wdl0": "WDL off",
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7.5),
        constrained_layout=True,
    )
    for name, metrics in rows.items():
        dfm = np.asarray(
            [float(row["dfm_ce_loss"]) for row in metrics],
            dtype=np.float64,
        )
        updates, dfm_smoothed = rolling_mean(dfm, 64)
        axes[0, 0].plot(
            updates,
            dfm_smoothed,
            color=colors[name],
            label=labels[name],
        )

        jepa = np.asarray(
            [float(row["jepa_positive_loss"]) for row in metrics],
            dtype=np.float64,
        )
        _, jepa_smoothed = rolling_mean(jepa, 64)
        axes[0, 1].plot(
            updates,
            jepa_smoothed,
            color=colors[name],
            label=labels[name],
        )

        comparable_total = np.asarray(
            [
                float(row["loss"]) - float(row["wdl_weighted_loss"])
                for row in metrics
            ],
            dtype=np.float64,
        )
        _, total_smoothed = rolling_mean(comparable_total, 64)
        axes[1, 0].plot(
            updates,
            total_smoothed,
            color=colors[name],
            label=labels[name],
        )

    for axis, title, ylabel in (
        (axes[0, 0], "Training DFM CE", "dfm_ce_loss"),
        (axes[0, 1], "Training JEPA MSE", "jepa_positive_loss"),
        (
            axes[1, 0],
            "Comparable training loss (WDL contribution removed)",
            "non-WDL weighted loss",
        ),
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

    names = list(RUNS)
    x = np.arange(len(names))
    scores = [float(arenas[name]["score"]) for name in names]
    lower = [
        scores[index] - float(arenas[name]["score_lower"])
        for index, name in enumerate(names)
    ]
    upper = [
        float(arenas[name]["score_upper"]) - scores[index]
        for index, name in enumerate(names)
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
        metrics = validation[name]["metrics"]
        axes[1, 1].annotate(
            (
                f"{scores[index]:.4f}\n"
                f"val CE {metrics['dfm_ce_loss']:.4f}\n"
                f"WDL CE {metrics['wdl_loss']:.3f}"
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
        "Predicted-state WDL auxiliary ablation",
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
            "metrics": run_root / "metrics.jsonl",
            "loss_summary": run_root / "loss_summary.json",
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
        raise FileNotFoundError(f"Missing WDL-ablation inputs: {missing}")

    configs = {
        name: load_json(run_paths["run_config"])
        for name, run_paths in paths.items()
    }
    reports = {
        name: load_json(run_paths["training_report"])
        for name, run_paths in paths.items()
    }
    rows = {
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

    scientific_diff = dictionary_diff(
        configs["control_wdl025"]["config"],
        configs["candidate_wdl0"]["config"],
    )
    if set(scientific_diff) != {"wdl_coeff"}:
        raise ValueError(
            f"Unexpected scientific config differences: {scientific_diff}"
        )
    if configs["control_wdl025"]["data"] != configs["candidate_wdl0"]["data"]:
        raise ValueError("WDL runs did not record the same data contract")
    if len(rows["control_wdl025"]) != len(rows["candidate_wdl0"]):
        raise ValueError("WDL runs have different metric-row counts")

    training = {
        name: training_record(reports[name], rows[name]) for name in RUNS
    }
    checkpoints = {
        name: checkpoint_record(
            reports[name],
            paths[name]["checkpoint_manifest"],
            paths[name]["checkpoint_state"],
        )
        for name in RUNS
    }
    validation = {
        name: validation_record(validation_reports[name]) for name in RUNS
    }
    for name in RUNS:
        if validation[name]["sigreg_example_count"] != 64:
            raise ValueError("Validation did not use count-64 SIGReg")
        if validation_reports[name]["config"]["wdl_coeff"] != 0.25:
            raise ValueError("Validation did not use fixed WDL-0.25 scoring")
    for key in ("pool_manifest_sha256", "pool_indices_sha256", "examples"):
        if validation["control_wdl025"][key] != validation["candidate_wdl0"][key]:
            raise ValueError(f"Frozen validation mismatch: {key}")

    arenas = {
        name: arena_record(arena_states[name]) for name in RUNS
    }
    if (
        arenas["control_wdl025"]["run_contract"]
        != arenas["candidate_wdl0"]["run_contract"]
    ):
        raise ValueError("Arena run contracts are not matched")
    if (
        arenas["control_wdl025"]["tier_contract"]
        != arenas["candidate_wdl0"]["tier_contract"]
    ):
        raise ValueError("Arena opening tiers are not matched")
    if (
        arenas["control_wdl025"]["reference_model_id"]
        != arenas["candidate_wdl0"]["reference_model_id"]
    ):
        raise ValueError("Arena reference models are not matched")

    arena_delta = paired_arena_delta(
        arenas["control_wdl025"]["_pair_scores"],
        arenas["candidate_wdl0"]["_pair_scores"],
    )
    arena_delta["descriptive_logistic_elo_delta"] = (
        arenas["candidate_wdl0"]["logistic_elo"]
        - arenas["control_wdl025"]["logistic_elo"]
    )
    public_arenas = {
        name: {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        for name, record in arenas.items()
    }
    validation_delta = metric_delta(
        validation["control_wdl025"]["metrics"],
        validation["candidate_wdl0"]["metrics"],
    )
    retention = rank_retention(
        validation["control_wdl025"]["rank_diagnostics"],
        validation["candidate_wdl0"]["rank_diagnostics"],
    )

    gates = {
        "arena_directionally_non_worse": (
            arena_delta["candidate_minus_control_score"] >= 0.0
        ),
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
            validation_delta["accuracy"]["candidate_minus_control"] >= -0.001
        ),
        "legal_mass_loss_below_0p2pp": (
            validation_delta["first_legal_mass"][
                "candidate_minus_control"
            ]
            >= -0.002
        ),
        "h8_rank_retention_at_least_95pct": all(
            retention[latent]["h8"][metric]["candidate_over_control"] >= 0.95
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
        "zero_nonfinite_skips": all(
            training[name]["optimizer_skipped_nonfinite_count"] == 0
            for name in RUNS
        ),
        "zero_arena_faults": all(
            not public_arenas[name]["fault_counts"] for name in RUNS
        ),
        "zero_cap_draws": all(
            public_arenas[name]["cap_draw_count"] == 0 for name in RUNS
        ),
    }
    passes = all(gates.values())
    build_plot(rows, validation, public_arenas)

    sources = {
        name: {
            label: relative_source(path)
            for label, path in run_paths.items()
            if label != "checkpoint_state"
        }
        for name, run_paths in paths.items()
    }
    result = {
        "schema_version": "hero-wdl-ablation-v1",
        "question": (
            "Does the predicted-state WDL auxiliary improve early "
            "searchless chess strength?"
        ),
        "comparison": {
            "control": "control_wdl025",
            "candidate": "candidate_wdl0",
            "only_scientific_config_difference": scientific_diff,
            "same_data_contract": True,
            "same_initialization_seed": (
                configs["control_wdl025"]["config"]["init_seed"]
                == configs["candidate_wdl0"]["config"]["init_seed"]
                == 0
            ),
            "control_git_commit": configs["control_wdl025"]["git_commit"],
            "candidate_git_commit": configs["candidate_wdl0"]["git_commit"],
            "candidate_code_change_is_default_neutral": True,
            "fixed_validation_contract": {
                "sigreg_example_count": 64,
                "wdl_coeff_for_diagnostic_scoring": 0.25,
            },
        },
        "training": training,
        "training_last_64_update_delta": metric_delta(
            training["control_wdl025"]["last_64_update_mean"],
            training["candidate_wdl0"]["last_64_update_mean"],
        ),
        "systems_delta": {
            "candidate_throughput_relative_change": (
                training["candidate_wdl0"]["examples_per_second_end_to_end"]
                / training["control_wdl025"]["examples_per_second_end_to_end"]
                - 1.0
            ),
            "candidate_peak_hbm_allocated_delta_bytes": (
                training["candidate_wdl0"][
                    "gpu_peak_memory_allocated_bytes"
                ]
                - training["control_wdl025"][
                    "gpu_peak_memory_allocated_bytes"
                ]
            ),
        },
        "checkpoints": checkpoints,
        "frozen_fast_validation": validation,
        "frozen_fast_validation_delta": validation_delta,
        "rank_retention": retention,
        "paired_arena": public_arenas,
        "paired_arena_delta": arena_delta,
        "decision": {
            "status": (
                "advance_candidate_to_repeat_and_10pct"
                if passes
                else "reject_candidate_wdl0"
            ),
            "gates": gates,
            "all_preregistered_gates_pass": passes,
            "active_wdl_coeff": 0.0 if passes else 0.25,
            "coefficient_sweep_authorized": False,
            "causal_scope": (
                "One deterministic matched 3.70%-epoch screen. A positive "
                "result still requires a fresh repeat and at least 10% of "
                "an epoch; a negative result closes only the 0-vs-0.25 "
                "ablation."
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
                "validation_dfm_ce_delta": validation_delta[
                    "dfm_ce_loss"
                ]["candidate_minus_control"],
                "arena_score_delta": arena_delta[
                    "candidate_minus_control_score"
                ],
                "decision": result["decision"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
