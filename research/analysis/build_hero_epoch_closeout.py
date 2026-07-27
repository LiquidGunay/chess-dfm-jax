#!/usr/bin/env python3
"""Build the immutable one-epoch hero loss/Elo closeout artifact."""

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
OUTPUT_JSON = ROOT / "research/analysis/hero_epoch_v1_closeout_20260727.json"
OUTPUT_PNG = ROOT / "research/analysis/hero_epoch_v1_closeout_20260727.png"
INPUTS = {
    "training_report": ROOT / "research/runs/torch_hero_epoch_v1/report.json",
    "training_config": ROOT
    / "research/runs/torch_hero_epoch_v1/run_config.json",
    "fast_validation": ROOT
    / "research/runs/torch_hero_epoch_v1/hero_validation_metrics.jsonl",
    "initial_primary": ROOT
    / "research/runs/torch_hero_init_primary_eval_v1/report.json",
    "halfway_primary": ROOT
    / "research/runs/torch_hero_epoch_v1_halfway_primary_eval_v1/report.json",
    "terminal_primary": ROOT
    / "research/runs/torch_hero_epoch_v1_terminal_primary_eval_v1/report.json",
    "terminal_blind": ROOT
    / "research/runs/torch_hero_epoch_v1_terminal_blind_eval_v1/report.json",
    "halfway_arena": ROOT
    / "artifacts/arena/hero-epoch-v1-halfway-vs-raw-bt4-1024pairs/state.json",
    "terminal_arena": ROOT
    / "artifacts/arena/hero-epoch-v1-terminal-vs-raw-bt4-1024pairs/state.json",
    "halfway_jax_roundtrip": ROOT
    / "artifacts/pytorch/hero_epoch_v1_halfway_jax_roundtrip.json",
    "terminal_jax_roundtrip": ROOT
    / "artifacts/pytorch/hero_epoch_v1_terminal_jax_roundtrip.json",
}
VALIDATION_KEYS = (
    "loss",
    "dfm_ce_loss",
    "dfm_ce_loss_by_horizon_h1",
    "accuracy",
    "first_legal_mass",
    "root_legal_top1_accuracy",
    "jepa_positive_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "wdl_loss",
    "z_pred_norm",
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


def validation_point(
    path: Path,
    *,
    epoch: float,
    pool: str,
) -> dict[str, Any]:
    report = load_json(path)
    if not report.get("gate_pass"):
        raise ValueError(f"Validation gate did not pass: {path}")
    metrics = report["metrics"]
    rank_ratio_keys = [
        f"pred_target_effective_rank_ratio_by_horizon_h{index}"
        for index in range(1, 9)
    ]
    rms_ratio_keys = [
        f"pred_target_centered_rms_ratio_by_horizon_h{index}"
        for index in range(1, 9)
    ]
    rank_ratios = (
        [float(metrics[key]) for key in rank_ratio_keys]
        if all(key in metrics for key in rank_ratio_keys)
        else None
    )
    rms_ratios = (
        [float(metrics[key]) for key in rms_ratio_keys]
        if all(key in metrics for key in rms_ratio_keys)
        else None
    )
    return {
        "epoch": epoch,
        "pool": pool,
        "state": report["state"],
        "examples": int(report["evaluation_examples"]),
        "evaluation_seconds": float(report["evaluation_seconds"]),
        "report_sha256": sha256_file(path),
        "metrics": {
            key: float(metrics[key])
            for key in VALIDATION_KEYS
            if key in metrics
        },
        "latent_alignment": {
            "pred_target_effective_rank_ratio_min": (
                None if rank_ratios is None else min(rank_ratios)
            ),
            "pred_target_effective_rank_ratio_max": (
                None if rank_ratios is None else max(rank_ratios)
            ),
            "pred_target_centered_rms_ratio_min": (
                None if rms_ratios is None else min(rms_ratios)
            ),
            "pred_target_centered_rms_ratio_max": (
                None if rms_ratios is None else max(rms_ratios)
            ),
        },
    }


def arena_point(path: Path, *, epoch: float) -> dict[str, Any]:
    state = load_json(path)
    if state.get("status") != "complete":
        raise ValueError(f"Arena is not complete: {path}")
    aggregate = state["aggregate"]
    models = aggregate["models"]
    candidate_id = next(
        model_id
        for model_id in models
        if model_id.startswith("candidate-torch-hero-")
    )
    reference_id = next(
        model_id for model_id in models if model_id.startswith("raw-bt4-")
    )
    candidate = models[candidate_id]
    reference = models[reference_id]
    candidate_coverage = candidate["coverage"]
    reference_coverage = reference["coverage"]
    interval = aggregate["pair_aware_logistic_interval"]
    normalized = aggregate["normalized_elo_diagnostics"]
    return {
        "epoch": epoch,
        "state_sha256": state["payload_sha256"],
        "contract_sha256": state["contract_sha256"],
        "pair_count": int(aggregate["pair_count"]),
        "game_count": int(aggregate["game_count"]),
        "score": float(aggregate["pentanomial"]["score"]),
        "score_lower": float(interval["score_lower"]),
        "score_upper": float(interval["score_upper"]),
        "pentanomial": list(aggregate["pentanomial"]["counts"]),
        "logistic_elo": float(interval["elo"]),
        "logistic_elo_lower": float(interval["elo_lower"]),
        "logistic_elo_upper": float(interval["elo_upper"]),
        "normalized_elo": float(normalized["normalized_elo"]),
        "candidate": {
            "model_id": candidate_id,
            "wins": int(candidate["wins"]),
            "draws": int(candidate["draws"]),
            "losses": int(candidate["losses"]),
            "positions": int(candidate_coverage["positions"]),
            "calls": int(candidate_coverage["calls"]),
            "call_seconds": float(candidate_coverage["call_seconds"]),
            "amortized_ms_per_position": (
                1000.0
                * float(candidate_coverage["call_seconds"])
                / int(candidate_coverage["positions"])
            ),
            "mean_physical_call_ms": (
                1000.0 * float(candidate_coverage["mean_call_seconds"])
            ),
        },
        "reference": {
            "model_id": reference_id,
            "wins": int(reference["wins"]),
            "draws": int(reference["draws"]),
            "losses": int(reference["losses"]),
            "positions": int(reference_coverage["positions"]),
            "calls": int(reference_coverage["calls"]),
            "call_seconds": float(reference_coverage["call_seconds"]),
            "amortized_ms_per_position": (
                1000.0
                * float(reference_coverage["call_seconds"])
                / int(reference_coverage["positions"])
            ),
            "mean_physical_call_ms": (
                1000.0 * float(reference_coverage["mean_call_seconds"])
            ),
        },
        "gameplay_seconds": float(aggregate["gameplay_wall_seconds"]),
        "fault_counts": dict(aggregate["fault_counts"]),
        "cap_draw_count": int(aggregate["cap_draw_count"]),
        "termination_counts": dict(aggregate["termination_counts"]),
        "_pair_scores": np.asarray(
            aggregate["pair_scores"],
            dtype=np.float64,
        )
        / 2.0,
    }


def paired_delta(
    halfway: np.ndarray,
    terminal: np.ndarray,
) -> dict[str, Any]:
    if halfway.shape != terminal.shape or halfway.ndim != 1:
        raise ValueError("Arena pair-score arrays are not aligned")
    delta = terminal - halfway
    count = len(delta)
    mean = float(delta.mean())
    standard_deviation = float(delta.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(count)
    critical = float(stats.t.ppf(0.975, count - 1))
    counts = Counter(float(value) for value in delta)
    return {
        "pair_count": count,
        "terminal_minus_halfway_score": mean,
        "paired_standard_deviation": standard_deviation,
        "paired_standard_error": standard_error,
        "paired_t_95_lower": mean - critical * standard_error,
        "paired_t_95_upper": mean + critical * standard_error,
        "terminal_better_pairs": int(np.count_nonzero(delta > 0)),
        "equal_pairs": int(np.count_nonzero(delta == 0)),
        "terminal_worse_pairs": int(np.count_nonzero(delta < 0)),
        "delta_counts": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        "method_note": (
            "Descriptive paired t interval over the frozen opening-pair "
            "scores; it is not a promotion decision."
        ),
    }


def strip_private(value: dict[str, Any]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if not key.startswith("_")}


def build_plot(
    fast_points: list[dict[str, float]],
    validations: list[dict[str, Any]],
    arenas: list[dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    epochs = [point["epoch"] for point in fast_points]
    axes[0, 0].plot(
        epochs,
        [point["dfm_ce_loss"] for point in fast_points],
        marker="o",
        label="DFM CE",
    )
    axes[0, 0].plot(
        epochs,
        [point["loss"] for point in fast_points],
        marker="o",
        label="weighted total",
    )
    axes[0, 0].set(
        title="Fast validation loss during the epoch",
        xlabel="training epoch",
        ylabel="loss",
    )
    axes[0, 0].legend()

    primary = [point for point in validations if point["pool"] == "primary"]
    primary_epochs = [point["epoch"] for point in primary]
    axes[0, 1].plot(
        primary_epochs,
        [point["metrics"]["dfm_ce_loss"] for point in primary],
        marker="o",
        label="primary DFM CE",
    )
    axes[0, 1].plot(
        primary_epochs,
        [
            point["metrics"]["dfm_ce_loss_by_horizon_h1"]
            for point in primary
        ],
        marker="o",
        label="primary H1 CE",
    )
    blind = next(point for point in validations if point["pool"] == "blind")
    axes[0, 1].scatter(
        [blind["epoch"]],
        [blind["metrics"]["dfm_ce_loss"]],
        marker="*",
        s=90,
        label="terminal blind DFM CE",
        zorder=3,
    )
    axes[0, 1].set(
        title="Frozen held-out action loss",
        xlabel="training epoch",
        ylabel="cross-entropy",
    )
    axes[0, 1].legend()

    arena_by_epoch = {point["epoch"]: point for point in arenas}
    primary_arena = [
        point for point in primary if point["epoch"] in arena_by_epoch
    ]
    x = [point["metrics"]["dfm_ce_loss"] for point in primary_arena]
    y = [arena_by_epoch[point["epoch"]]["logistic_elo"] for point in primary_arena]
    axes[1, 0].plot(x, y, marker="o")
    for point, x_value, y_value in zip(primary_arena, x, y, strict=True):
        axes[1, 0].annotate(
            f"{point['epoch']:.1f} epoch",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axes[1, 0].invert_xaxis()
    axes[1, 0].set(
        title="Held-out DFM CE and relative Elo",
        xlabel="primary DFM CE (lower is better)",
        ylabel="descriptive logistic Elo vs raw BT4",
    )

    arena_epochs = [point["epoch"] for point in arenas]
    scores = [point["score"] for point in arenas]
    lower = [point["score"] - point["score_lower"] for point in arenas]
    upper = [point["score_upper"] - point["score"] for point in arenas]
    axes[1, 1].errorbar(
        arena_epochs,
        scores,
        yerr=np.asarray([lower, upper]),
        marker="o",
        capsize=4,
    )
    axes[1, 1].axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    for point in arenas:
        axes[1, 1].annotate(
            (
                f"{point['logistic_elo']:.1f} Elo\n"
                f"{point['candidate']['amortized_ms_per_position']:.2f} ms/pos"
            ),
            (point["epoch"], point["score"]),
            xytext=(5, -28),
            textcoords="offset points",
        )
    axes[1, 1].set(
        title="Frozen 1,024-pair Arena score",
        xlabel="training epoch",
        ylabel="score vs raw BT4",
        ylim=(0.75, 0.97),
    )

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "BT4 + DFM + JEPA one-epoch closeout",
        fontsize=14,
    )
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing closeout inputs: {missing}")

    training = load_json(INPUTS["training_report"])
    run_config = load_json(INPUTS["training_config"])
    fast_rows = [
        json.loads(line)
        for line in INPUTS["fast_validation"]
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    fast_points = [
        {
            "epoch": float(row["observed_fraction"]),
            "loss": float(row["metrics"]["loss"]),
            "dfm_ce_loss": float(row["metrics"]["dfm_ce_loss"]),
            "dfm_ce_loss_by_horizon_h1": float(
                row["metrics"]["dfm_ce_loss_by_horizon_h1"]
            ),
            "first_legal_mass": float(row["metrics"]["first_legal_mass"]),
            "root_legal_top1_accuracy": float(
                row["metrics"]["root_legal_top1_accuracy"]
            ),
        }
        for row in fast_rows
    ]
    validations = [
        validation_point(
            INPUTS["initial_primary"],
            epoch=0.0,
            pool="primary",
        ),
        validation_point(
            INPUTS["halfway_primary"],
            epoch=0.5,
            pool="primary",
        ),
        validation_point(
            INPUTS["terminal_primary"],
            epoch=1.0,
            pool="primary",
        ),
        validation_point(
            INPUTS["terminal_blind"],
            epoch=1.0,
            pool="blind",
        ),
    ]
    halfway_arena = arena_point(INPUTS["halfway_arena"], epoch=0.5)
    terminal_arena = arena_point(INPUTS["terminal_arena"], epoch=1.0)
    arena_delta = paired_delta(
        halfway_arena["_pair_scores"],
        terminal_arena["_pair_scores"],
    )
    arenas = [
        strip_private(halfway_arena),
        strip_private(terminal_arena),
    ]
    roundtrips = {}
    for label in ("halfway_jax_roundtrip", "terminal_jax_roundtrip"):
        audit = load_json(INPUTS[label])
        if not audit.get("gate_pass"):
            raise ValueError(f"JAX round-trip gate failed: {label}")
        roundtrips[label.removesuffix("_jax_roundtrip")] = audit

    sources = {
        label: {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in INPUTS.items()
    }
    result = {
        "schema_version": "torch-hero-epoch-closeout-v1",
        "run_id": "torch_hero_epoch_v1",
        "training": {
            "git_commit": run_config["git_commit"],
            "updates": int(training["updates"]),
            "examples": int(training["examples"]),
            "train_seconds": float(training["train_seconds"]),
            "train_wall_seconds": float(training["train_wall_seconds"]),
            "examples_per_second_end_to_end": float(
                training["examples_per_second_end_to_end"]
            ),
            "peak_hbm_allocated_bytes": int(
                training["gpu_peak_memory_allocated_bytes"]
            ),
            "peak_hbm_reserved_bytes": int(
                training["gpu_peak_memory_reserved_bytes"]
            ),
            "terminal_checkpoint": training["checkpoint"],
        },
        "loss_contract": {
            key: run_config["config"][key]
            for key in (
                "dfm_ce_coeff",
                "root_legal_ce_coeff",
                "legality_coeff",
                "jepa_positive_coeff",
                "target_sigreg_coeff",
                "pred_sigreg_coeff",
                "wdl_coeff",
                "sigreg_example_count",
            )
        },
        "fast_validation_note": (
            "Repeated 8,192-position development pool; useful for curve shape, "
            "not a terminal generalization estimate."
        ),
        "fast_validation_points": fast_points,
        "frozen_validation_points": validations,
        "arena_points": arenas,
        "paired_terminal_minus_halfway": arena_delta,
        "jax_roundtrip": roundtrips,
        "interpretation": {
            "primary_dfm_ce_improves_halfway_to_terminal": True,
            "arena_score_improves_halfway_to_terminal": True,
            "terminal_blind_reproduces_primary": True,
            "pred_only_latent_collapse_detected": False,
            "loss_elo_evidence_scope": (
                "Positive within one run at two strength checkpoints; "
                "insufficient to establish a cross-recipe calibration."
            ),
            "elo_scope": (
                "Relative only to the frozen raw-BT4 implementation and "
                "opening pool; not human or published-BT4 absolute Elo."
            ),
        },
        "plot": str(OUTPUT_PNG.relative_to(ROOT)),
        "sources": sources,
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_plot(fast_points, validations, arenas)
    print(
        json.dumps(
            {
                "output_json": str(OUTPUT_JSON),
                "output_png": str(OUTPUT_PNG),
                "terminal_primary_dfm_ce": validations[2]["metrics"][
                    "dfm_ce_loss"
                ],
                "terminal_blind_dfm_ce": validations[3]["metrics"][
                    "dfm_ce_loss"
                ],
                "halfway_arena_score": arenas[0]["score"],
                "terminal_arena_score": arenas[1]["score"],
                "paired_score_delta": arena_delta[
                    "terminal_minus_halfway_score"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
