#!/usr/bin/env python3
"""Validate and summarize the post-Hero architecture experiment suite."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from research.evaluate_arena import load_run_state
from research.prepare import sha256_file


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = (
    ROOT / "research/analysis/posthero_architecture_suite_20260728.json"
)
OUTPUT_CSV = (
    ROOT / "research/analysis/posthero_architecture_suite_20260728.csv"
)
TRAINING_PLOT = (
    ROOT
    / "research/analysis/posthero_architecture_training_curves_20260728.png"
)
ARENA_PLOT = (
    ROOT / "research/analysis/posthero_architecture_arena_20260728.png"
)
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_728
ELO_LOGIT_SCALE = math.log(10.0) / 400.0
ANCHOR = "hero"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    run_name: str
    eval_name: str
    checkpoint_sha256: str
    color: str


MODELS = (
    ModelSpec(
        key="hero",
        label="Hero-1024",
        run_name="torch_autoresearch_sigreg64_u1024_v1",
        eval_name="torch_autoresearch_sigreg64_u1024_v1_fast_eval_v1",
        checkpoint_sha256=(
            "05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9"
        ),
        color="#4c566a",
    ),
    ModelSpec(
        key="all_heads",
        label="All-horizon heads",
        run_name="posthero_all_horizon_heads_u1024_v1",
        eval_name="posthero_all_horizon_heads_u1024_v1_fast_eval_v1",
        checkpoint_sha256=(
            "a8e4f6124bb94d3b9671de4a06655f74a678ab91d127c657fa33bc1159cbdc1e"
        ),
        color="#bf616a",
    ),
    ModelSpec(
        key="jepa_fusion",
        label="Normalized JEPA fusion",
        run_name="posthero_jepa_fusion_u1024_v1",
        eval_name="posthero_jepa_fusion_u1024_v1_fast_eval_v1",
        checkpoint_sha256=(
            "eddce082ff10866c382afc66ea2327b13eae60602eee8d0ab93851b0d1e39d51"
        ),
        color="#5e81ac",
    ),
    ModelSpec(
        key="current_wdl",
        label="Current-state WDL",
        run_name="posthero_current_wdl_u1024_v2",
        eval_name="posthero_current_wdl_u1024_v2_fast_eval_v1",
        checkpoint_sha256=(
            "16640d2e02609245c862a074951f68adedec871a3872d86bdf49b1d5fafb7d12"
        ),
        color="#a3be8c",
    ),
    ModelSpec(
        key="closed_loop",
        label="Predicted-JEPA closed loop",
        run_name="posthero_closed_loop_u1024_v1",
        eval_name="posthero_closed_loop_u1024_v1_fast_eval_v1",
        checkpoint_sha256=(
            "90a3429d6d17b6df4563b4babf589fb200dc8f59254316ab857f954bff717ea2"
        ),
        color="#b48ead",
    ),
    ModelSpec(
        key="policy_prelogit",
        label="Policy-prelogit",
        run_name="posthero_policy_prelogit_u1024_v1",
        eval_name="posthero_policy_prelogit_u1024_v1_fast_eval_v1",
        checkpoint_sha256=(
            "51e822ccdb5ade6bbe8f7e1c92d851ea3fe45dce89b01d200e2ab4918b2a0040"
        ),
        color="#d08770",
    ),
)
MODEL_BY_KEY = {model.key: model for model in MODELS}
PLAYERS = tuple(model.key for model in MODELS)


@dataclasses.dataclass(frozen=True)
class EdgeSpec:
    candidate: str
    opponent: str
    arena_name: str


PRIMARY_EDGES = (
    EdgeSpec(
        "current_wdl",
        "hero",
        "posthero-suite-first128-p1-current-wdl-vs-hero-v1",
    ),
    EdgeSpec(
        "jepa_fusion",
        "hero",
        "posthero-suite-first128-p1-jepa-fusion-vs-hero-v1",
    ),
    EdgeSpec(
        "closed_loop",
        "hero",
        "posthero-suite-first128-p1-closed-loop-vs-hero-v1",
    ),
    EdgeSpec(
        "policy_prelogit",
        "hero",
        "posthero-suite-first128-p1-policy-prelogit-vs-hero-v1",
    ),
    EdgeSpec(
        "all_heads",
        "hero",
        "posthero-suite-first128-p1-all-heads-vs-hero-v1",
    ),
    EdgeSpec(
        "all_heads",
        "jepa_fusion",
        "posthero-suite-first128-p1-all-heads-vs-jepa-fusion-v1",
    ),
    EdgeSpec(
        "all_heads",
        "current_wdl",
        "posthero-suite-first128-p1-all-heads-vs-current-wdl-v1",
    ),
    EdgeSpec(
        "all_heads",
        "closed_loop",
        "posthero-suite-first128-p1-all-heads-vs-closed-loop-v1",
    ),
    EdgeSpec(
        "all_heads",
        "policy_prelogit",
        "posthero-suite-first128-p1-all-heads-vs-policy-prelogit-v1",
    ),
    EdgeSpec(
        "jepa_fusion",
        "current_wdl",
        "posthero-suite-first128-p1-jepa-fusion-vs-current-wdl-v1",
    ),
    EdgeSpec(
        "jepa_fusion",
        "closed_loop",
        "posthero-suite-first128-p1-jepa-fusion-vs-closed-loop-v1",
    ),
    EdgeSpec(
        "jepa_fusion",
        "policy_prelogit",
        "posthero-suite-first128-p1-jepa-fusion-vs-policy-prelogit-v1",
    ),
    EdgeSpec(
        "current_wdl",
        "closed_loop",
        "posthero-suite-first128-p1-current-wdl-vs-closed-loop-v1",
    ),
    EdgeSpec(
        "current_wdl",
        "policy_prelogit",
        "posthero-suite-first128-p1-current-wdl-vs-policy-prelogit-v1",
    ),
    EdgeSpec(
        "closed_loop",
        "policy_prelogit",
        "posthero-suite-first128-p1-closed-loop-vs-policy-prelogit-v1",
    ),
)

REFINEMENT_EDGES = (
    EdgeSpec(
        "all_heads",
        "all_heads",
        "posthero-suite-first128-all-heads-p8-vs-p1-v1",
    ),
    EdgeSpec(
        "jepa_fusion",
        "jepa_fusion",
        "posthero-suite-first128-jepa-fusion-p8-vs-p1-v1",
    ),
    EdgeSpec(
        "current_wdl",
        "current_wdl",
        "posthero-suite-first128-current-wdl-p8-vs-p1-v1",
    ),
    EdgeSpec(
        "closed_loop",
        "closed_loop",
        "posthero-suite-first128-closed-loop-p8-vs-p1-v1",
    ),
    EdgeSpec(
        "policy_prelogit",
        "policy_prelogit",
        "posthero-suite-first128-policy-prelogit-p8-vs-p1-v1",
    ),
)

FOLLOWUP_EDGE = EdgeSpec(
    "closed_loop",
    "hero",
    "posthero-suite-first128-closed-loop-p8-vs-hero-p1-v1",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected object in {path}")
    return value


def _checkpoint_state_sha(descriptor: dict[str, Any]) -> str:
    _require(descriptor["kind"] == "torch_hero", "Non-Hero Arena descriptor")
    _require(
        descriptor["arena_policy_mode"] == "dfm",
        "Arena did not use the DFM policy",
    )
    return str(descriptor["state"]["sha256"])


def _load_arena_edge(
    spec: EdgeSpec,
    *,
    candidate_passes: int,
    opponent_passes: int,
) -> dict[str, Any]:
    arena_dir = ROOT / "artifacts/arena" / spec.arena_name
    state_path = arena_dir / "state.json"
    raw_state = _read_json(state_path)
    state = load_run_state(
        arena_dir,
        expected_contract=raw_state["contract"],
    )
    _require(state == raw_state, f"Reloaded state drift: {spec.arena_name}")
    _require(state["status"] == "complete", f"Incomplete Arena: {spec.arena_name}")

    contract = state["contract"]
    run = contract["run"]
    aggregate = state["aggregate"]
    candidate = contract["models"]["candidate"]
    opponent = contract["models"]["opponent"]
    _require(
        _checkpoint_state_sha(candidate)
        == MODEL_BY_KEY[spec.candidate].checkpoint_sha256,
        f"Candidate checkpoint drift: {spec.arena_name}",
    )
    _require(
        _checkpoint_state_sha(opponent)
        == MODEL_BY_KEY[spec.opponent].checkpoint_sha256,
        f"Opponent checkpoint drift: {spec.arena_name}",
    )
    expected_run = {
        "pair_count": 128,
        "block_pairs": 16,
        "additional_ply_cap": 256,
        "candidate_refinement_passes": candidate_passes,
        "opponent_refinement_passes": opponent_passes,
        "policy_batch_size_cap": 16,
        "seed": 0,
        "collect_diagnostics": False,
        "deterministic_greedy_policy": True,
    }
    for name, expected in expected_run.items():
        _require(
            run[name] == expected,
            f"Run contract drift for {spec.arena_name}: {name}",
        )
    _require(
        contract["tier"]["name"] == "hero_development",
        f"Opening tier drift: {spec.arena_name}",
    )
    _require(aggregate["pair_count"] == 128, "Arena pair-count drift")
    _require(aggregate["game_count"] == 256, "Arena game-count drift")
    _require(aggregate["fault_counts"] == {}, "Arena policy fault")
    _require(aggregate["cap_draw_count"] == 0, "Arena cap draw")
    _require(
        aggregate["termination_counts"] == {"normal": 256},
        "Abnormal Arena termination",
    )

    pair_scores = np.asarray(aggregate["pair_scores"], dtype=np.float64)
    _require(pair_scores.shape == (128,), "Pair-score shape drift")
    _require(
        bool(np.all(np.isin(pair_scores, (0.0, 0.5, 1.0, 1.5, 2.0)))),
        "Invalid pentanomial score",
    )
    candidate_stats = aggregate["models"][candidate["model_id"]]
    opponent_stats = aggregate["models"][opponent["model_id"]]
    for stats in (candidate_stats, opponent_stats):
        coverage = stats["coverage"]
        _require(
            coverage["incomplete_coverage_positions"] == 0
            and coverage["representable_fraction"] == 1.0,
            "Incomplete legal-action coverage",
        )
        _require(
            math.isfinite(float(coverage["mean_call_seconds"]))
            and float(coverage["mean_call_seconds"]) > 0.0,
            "Invalid policy latency",
        )

    score = float(pair_scores.sum() / (2.0 * pair_scores.size))
    interval = aggregate["pair_aware_logistic_interval"]
    _require(float(interval["score"]) == score, "Stored Arena score drift")
    return {
        "spec": spec,
        "state": state,
        "pair_scores": pair_scores,
        "public": {
            "candidate": spec.candidate,
            "opponent": spec.opponent,
            "candidate_passes": candidate_passes,
            "opponent_passes": opponent_passes,
            "score": score,
            "points": float(pair_scores.sum()),
            "games": 256,
            "wins": int(candidate_stats["wins"]),
            "draws": int(candidate_stats["draws"]),
            "losses": int(candidate_stats["losses"]),
            "pentanomial": aggregate["pentanomial"]["counts"],
            "direct_logistic_elo": float(interval["elo"]),
            "direct_hoeffding_elo_95": [
                float(interval["elo_lower"]),
                float(interval["elo_upper"]),
            ],
            "candidate_mean_physical_call_ms": (
                1_000.0
                * float(candidate_stats["coverage"]["mean_call_seconds"])
            ),
            "candidate_mean_position_ms": (
                1_000.0
                * float(candidate_stats["coverage"]["call_seconds"])
                / float(candidate_stats["coverage"]["positions"])
            ),
            "candidate_positions_per_second": (
                float(candidate_stats["coverage"]["positions"])
                / float(candidate_stats["coverage"]["call_seconds"])
            ),
            "opponent_mean_physical_call_ms": (
                1_000.0
                * float(opponent_stats["coverage"]["mean_call_seconds"])
            ),
            "opponent_mean_position_ms": (
                1_000.0
                * float(opponent_stats["coverage"]["call_seconds"])
                / float(opponent_stats["coverage"]["positions"])
            ),
            "opponent_positions_per_second": (
                float(opponent_stats["coverage"]["positions"])
                / float(opponent_stats["coverage"]["call_seconds"])
            ),
            "gameplay_wall_seconds": float(
                aggregate["gameplay_wall_seconds"]
            ),
            "jepa_used_at_inference": bool(run["jepa_used_at_inference"]),
            "source_state": _relative(state_path),
            "source_state_sha256": sha256_file(state_path),
        },
    }


def _design_vector(candidate: str, opponent: str) -> np.ndarray:
    non_anchor = PLAYERS[1:]
    vector = np.zeros(len(non_anchor), dtype=np.float64)
    if candidate != ANCHOR:
        vector[non_anchor.index(candidate)] += 1.0
    if opponent != ANCHOR:
        vector[non_anchor.index(opponent)] -= 1.0
    return vector


def _fit_bradley_terry(
    edges: list[dict[str, Any]],
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    ratings = np.zeros(len(PLAYERS) - 1, dtype=np.float64)
    for _ in range(100):
        gradient = np.zeros_like(ratings)
        information = np.zeros((ratings.size, ratings.size), dtype=np.float64)
        for edge in edges:
            scores = edge["pair_scores"]
            if sample_indices is not None:
                scores = scores[sample_indices]
            trials = float(2 * scores.size)
            points = float(scores.sum())
            spec = edge["spec"]
            design = _design_vector(spec.candidate, spec.opponent)
            logit = float(
                np.clip(ELO_LOGIT_SCALE * (design @ ratings), -50.0, 50.0)
            )
            probability = 1.0 / (1.0 + math.exp(-logit))
            gradient += (
                ELO_LOGIT_SCALE
                * (points - trials * probability)
                * design
            )
            information += (
                ELO_LOGIT_SCALE**2
                * trials
                * probability
                * (1.0 - probability)
                * np.outer(design, design)
            )
        step = np.linalg.solve(information, gradient)
        ratings += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    else:
        raise RuntimeError("Bradley-Terry fit did not converge")
    _require(bool(np.all(np.isfinite(ratings))), "Non-finite Elo fit")
    return np.concatenate((np.zeros(1, dtype=np.float64), ratings))


def _score_from_elo(delta: float) -> float:
    return 1.0 / (1.0 + math.exp(-ELO_LOGIT_SCALE * delta))


def _percentile_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return [float(lower), float(upper)]


def _training_record(model: ModelSpec) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    run_dir = ROOT / "research/runs" / model.run_name
    checkpoint_path = run_dir / "checkpoint/model.safetensors"
    _require(
        sha256_file(checkpoint_path) == model.checkpoint_sha256,
        f"Checkpoint digest drift for {model.key}",
    )
    loss_summary = _read_json(run_dir / "loss_summary.json")
    report = _read_json(run_dir / "report.json")
    manifest = _read_json(run_dir / "checkpoint/manifest.json")
    lines = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    _require(len(lines) == 1024, f"Training update count drift for {model.key}")
    updates = np.asarray([line["update"] for line in lines], dtype=np.int64)
    _require(
        bool(np.array_equal(updates, np.arange(1, 1025))),
        f"Training update order drift for {model.key}",
    )
    curves = {
        "updates": updates,
        "dfm_ce_loss": np.asarray(
            [line["dfm_ce_loss"] for line in lines],
            dtype=np.float64,
        ),
        "loss": np.asarray(
            [line["loss"] for line in lines],
            dtype=np.float64,
        ),
    }
    last = loss_summary["last_window"]

    def last_window_metric(name: str, *, default: float = 0.0) -> float:
        if name in last:
            return float(last[name])
        values = [
            float(line.get(name, default))
            for line in lines[-64:]
        ]
        return float(np.mean(values))

    return (
        {
            "run_dir": _relative(run_dir),
            "checkpoint_sha256": model.checkpoint_sha256,
            "checkpoint_size_bytes": int(
                manifest["state"]["size_bytes"]
            ),
            "updates": int(report["updates"]),
            "examples": int(report["segment_examples"]),
            "train_wall_seconds": float(report["train_wall_seconds"]),
            "last_64": {
                "dfm_ce_loss": last_window_metric("dfm_ce_loss"),
                "loss": last_window_metric("loss"),
                "accuracy": last_window_metric("accuracy"),
                "first_legal_mass": last_window_metric("first_legal_mass"),
                "jepa_positive_loss": last_window_metric(
                    "jepa_positive_loss"
                ),
                "jepa_pred_sigreg_loss": last_window_metric(
                    "jepa_pred_sigreg_loss"
                ),
                "jepa_sigreg_loss": last_window_metric("jepa_sigreg_loss"),
                "wdl_loss": last_window_metric("wdl_loss"),
                "wdl_accuracy": last_window_metric("wdl_accuracy"),
                "z_pred_norm": last_window_metric("z_pred_norm"),
                "z_target_norm": last_window_metric("z_target_norm"),
                "dfm_closed_loop_dfm_ce_improvement": last_window_metric(
                    "dfm_closed_loop_dfm_ce_improvement"
                ),
            },
            "sources": {
                "loss_summary": _relative(run_dir / "loss_summary.json"),
                "loss_summary_sha256": sha256_file(
                    run_dir / "loss_summary.json"
                ),
                "report": _relative(run_dir / "report.json"),
                "report_sha256": sha256_file(run_dir / "report.json"),
                "metrics": _relative(run_dir / "metrics.jsonl"),
                "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
            },
        },
        curves,
    )


def _validation_record(model: ModelSpec) -> dict[str, Any]:
    eval_dir = ROOT / "research/runs" / model.eval_name
    report_path = eval_dir / "report.json"
    report = _read_json(report_path)
    _require(report["gate_pass"] is True, f"Validation gate failed: {model.key}")
    _require(
        report["evaluation_examples"] == 8192,
        f"Validation count drift: {model.key}",
    )
    _require(
        report["checkpoint"]["state"]["sha256"] == model.checkpoint_sha256,
        f"Validation checkpoint drift: {model.key}",
    )
    metrics = report["metrics"]
    return {
        "eval_dir": _relative(eval_dir),
        "examples": int(report["evaluation_examples"]),
        "evaluation_seconds": float(report["evaluation_seconds"]),
        "dfm_ce_loss": float(metrics["dfm_ce_loss"]),
        "accuracy": float(metrics["accuracy"]),
        "first_legal_mass": float(metrics["first_legal_mass"]),
        "jepa_positive_loss": float(metrics["jepa_positive_loss"]),
        "jepa_pred_sigreg_loss": float(metrics["jepa_pred_sigreg_loss"]),
        "jepa_sigreg_loss": float(metrics["jepa_sigreg_loss"]),
        "wdl_loss": float(metrics["wdl_loss"]),
        "wdl_accuracy": float(metrics["wdl_accuracy"]),
        "z_pred_norm": float(metrics["z_pred_norm"]),
        "z_target_norm": float(metrics["z_target_norm"]),
        "dfm_ce_loss_by_horizon": [
            float(metrics[f"dfm_ce_loss_by_horizon_h{horizon}"])
            for horizon in range(1, 9)
        ],
        "jepa_mse_by_horizon": [
            float(metrics[f"jepa_mse_by_horizon_h{horizon}"])
            for horizon in range(1, 9)
        ],
        "source_report": _relative(report_path),
        "source_report_sha256": sha256_file(report_path),
    }


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(
        values,
        np.ones(window, dtype=np.float64) / float(window),
        mode="valid",
    )


def _write_training_plot(curves: dict[str, dict[str, np.ndarray]]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(ROOT / ".local/cache/matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for model in MODELS:
        model_curves = curves[model.key]
        x = model_curves["updates"][63:]
        axes[0].plot(
            x,
            _rolling_mean(model_curves["dfm_ce_loss"], 64),
            label=model.label,
            color=model.color,
            linewidth=2.0,
        )
        axes[1].plot(
            x,
            _rolling_mean(model_curves["loss"], 64),
            label=model.label,
            color=model.color,
            linewidth=2.0,
        )
    axes[0].set_title("DFM action CE (64-update rolling mean)")
    axes[0].set_ylabel("cross-entropy")
    axes[1].set_title("Total training loss (64-update rolling mean)")
    axes[1].set_ylabel("loss")
    for axis in axes:
        axis.set_xlabel("optimizer update")
        axis.grid(alpha=0.25)
        axis.set_xlim(64, 1024)
    axes[1].legend(loc="upper right", fontsize=8)
    figure.suptitle(
        "Post-Hero architecture suite — matched 1,024-update runs",
        fontsize=14,
    )
    figure.savefig(TRAINING_PLOT, dpi=180)
    plt.close(figure)


def _write_arena_plot(
    *,
    ratings: np.ndarray,
    bootstrap: np.ndarray,
    refinement: list[dict[str, Any]],
    followup: dict[str, Any],
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(ROOT / ".local/cache/matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ordered = [
        "current_wdl",
        "jepa_fusion",
        "policy_prelogit",
        "closed_loop",
        "all_heads",
    ]
    y = np.arange(len(ordered))
    centers = np.asarray([ratings[PLAYERS.index(key)] for key in ordered])
    intervals = np.asarray(
        [
            _percentile_interval(bootstrap[:, PLAYERS.index(key)])
            for key in ordered
        ]
    )
    colors = [MODEL_BY_KEY[key].color for key in ordered]
    for index, color in enumerate(colors):
        axes[0].errorbar(
            centers[index],
            y[index],
            xerr=np.asarray(
                [
                    [
                        centers[index] - intervals[index, 0],
                        intervals[index, 1] - centers[index],
                    ]
                ]
            ).T,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=7,
            elinewidth=2,
            capsize=4,
            zorder=3,
        )
    axes[0].axvline(0.0, color="#4c566a", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, [MODEL_BY_KEY[key].label for key in ordered])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("pool-relative Elo vs Hero-1024 (Hero = 0)")
    axes[0].set_title("One-pass connected Bradley–Terry fit")
    axes[0].grid(axis="x", alpha=0.25)

    refinement_by_key = {
        edge["candidate"]: edge
        for edge in refinement
    }
    ref_keys = [
        "all_heads",
        "jepa_fusion",
        "current_wdl",
        "closed_loop",
        "policy_prelogit",
    ]
    scores = [
        100.0 * refinement_by_key[key]["score"]
        for key in ref_keys
    ]
    bars = axes[1].bar(
        np.arange(len(ref_keys)),
        scores,
        color=[MODEL_BY_KEY[key].color for key in ref_keys],
    )
    axes[1].axhline(50.0, color="#2e3440", linestyle="--", linewidth=1)
    axes[1].set_xticks(
        np.arange(len(ref_keys)),
        ["Heads", "Fusion", "WDL", "Closed", "Prelogit"],
        rotation=20,
        ha="right",
    )
    axes[1].set_ylabel("8-pass score vs own 1-pass policy (%)")
    axes[1].set_ylim(25.0, 60.0)
    axes[1].set_title(
        "Refinement diagnostic\n"
        f"Closed-loop 8-pass vs Hero 1-pass: "
        f"{100.0 * followup['score']:.1f}%"
    )
    axes[1].grid(axis="y", alpha=0.25)
    for bar, score in zip(bars, scores, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.7,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.suptitle(
        "Post-Hero architecture suite — frozen first-128-pair Arena",
        fontsize=14,
    )
    figure.savefig(ARENA_PLOT, dpi=180)
    plt.close(figure)


def _write_csv(
    models: dict[str, dict[str, Any]],
    ratings: np.ndarray,
    bootstrap: np.ndarray,
    direct_vs_hero: dict[str, float],
    refinement_scores: dict[str, float],
) -> None:
    fieldnames = [
        "model",
        "label",
        "training_last64_dfm_ce",
        "training_last64_total_loss",
        "training_last64_accuracy",
        "validation_dfm_ce",
        "validation_accuracy",
        "validation_first_legal_mass",
        "arena_bt_elo_vs_hero",
        "arena_bt_elo_lower_95",
        "arena_bt_elo_upper_95",
        "direct_score_vs_hero",
        "eight_pass_score_vs_one_pass",
        "train_wall_minutes",
        "checkpoint_sha256",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for model in MODELS:
            index = PLAYERS.index(model.key)
            interval = _percentile_interval(bootstrap[:, index])
            record = models[model.key]
            writer.writerow(
                {
                    "model": model.key,
                    "label": model.label,
                    "training_last64_dfm_ce": (
                        record["training"]["last_64"]["dfm_ce_loss"]
                    ),
                    "training_last64_total_loss": (
                        record["training"]["last_64"]["loss"]
                    ),
                    "training_last64_accuracy": (
                        record["training"]["last_64"]["accuracy"]
                    ),
                    "validation_dfm_ce": record["validation"]["dfm_ce_loss"],
                    "validation_accuracy": record["validation"]["accuracy"],
                    "validation_first_legal_mass": (
                        record["validation"]["first_legal_mass"]
                    ),
                    "arena_bt_elo_vs_hero": float(ratings[index]),
                    "arena_bt_elo_lower_95": interval[0],
                    "arena_bt_elo_upper_95": interval[1],
                    "direct_score_vs_hero": direct_vs_hero.get(model.key, 0.5),
                    "eight_pass_score_vs_one_pass": refinement_scores.get(
                        model.key,
                        "",
                    ),
                    "train_wall_minutes": (
                        record["training"]["train_wall_seconds"] / 60.0
                    ),
                    "checkpoint_sha256": model.checkpoint_sha256,
                }
            )


def main() -> int:
    model_records: dict[str, dict[str, Any]] = {}
    curves: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        training, model_curves = _training_record(model)
        model_records[model.key] = {
            "label": model.label,
            "training": training,
            "validation": _validation_record(model),
        }
        curves[model.key] = model_curves

    primary = [
        _load_arena_edge(edge, candidate_passes=1, opponent_passes=1)
        for edge in PRIMARY_EDGES
    ]
    refinement = [
        _load_arena_edge(edge, candidate_passes=8, opponent_passes=1)
        for edge in REFINEMENT_EDGES
    ]
    followup = _load_arena_edge(
        FOLLOWUP_EDGE,
        candidate_passes=8,
        opponent_passes=1,
    )
    all_arena_edges = [*primary, *refinement, followup]
    opening_orders = {
        edge["state"]["contract"]["tier"]["ordered_fens_sha256"]
        for edge in all_arena_edges
    }
    evaluator_hashes = {
        edge["state"]["contract"]["code"]["files"][
            "research/evaluate_arena.py"
        ]
        for edge in all_arena_edges
    }
    evaluator_commits = {
        edge["state"]["contract"]["code"]["git_commit"]
        for edge in all_arena_edges
    }
    _require(len(opening_orders) == 1, "Arena opening order drift")
    _require(len(evaluator_hashes) == 1, "Arena evaluator file drift")
    _require(len(evaluator_commits) == 1, "Arena evaluator commit drift")

    ratings = _fit_bradley_terry(primary)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(
        (BOOTSTRAP_SAMPLES, len(PLAYERS)),
        dtype=np.float64,
    )
    for sample in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, 128, size=128)
        bootstrap[sample] = _fit_bradley_terry(primary, indices)

    rating_records: dict[str, Any] = {}
    for index, player in enumerate(PLAYERS):
        rating_records[player] = {
            "elo_vs_hero": float(ratings[index]),
            "matched_opening_bootstrap_95": _percentile_interval(
                bootstrap[:, index]
            ),
        }

    public_primary: list[dict[str, Any]] = []
    direct_vs_hero: dict[str, float] = {}
    for edge in primary:
        spec = edge["spec"]
        candidate_index = PLAYERS.index(spec.candidate)
        opponent_index = PLAYERS.index(spec.opponent)
        delta = float(ratings[candidate_index] - ratings[opponent_index])
        public = dict(edge["public"])
        predicted_score = _score_from_elo(delta)
        public["joint_model_predicted_score"] = predicted_score
        public["joint_model_score_residual"] = (
            public["score"] - predicted_score
        )
        public_primary.append(public)
        if spec.opponent == "hero":
            direct_vs_hero[spec.candidate] = public["score"]

    public_refinement = [edge["public"] for edge in refinement]
    refinement_scores = {
        edge["candidate"]: float(edge["score"])
        for edge in public_refinement
    }
    public_followup = followup["public"]

    _write_training_plot(curves)
    _write_arena_plot(
        ratings=ratings,
        bootstrap=bootstrap,
        refinement=public_refinement,
        followup=public_followup,
    )
    _write_csv(
        model_records,
        ratings,
        bootstrap,
        direct_vs_hero,
        refinement_scores,
    )

    record = {
        "schema_version": "posthero-architecture-suite-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Which of five isolated passthrough/JEPA/DFM changes improves "
            "a fresh 1,024-update Hero recipe, and does any make recurrent "
            "inference useful?"
        ),
        "frozen_contract": {
            "training_examples_per_candidate": 1_048_576,
            "training_updates_per_candidate": 1024,
            "physical_batch_size": 1024,
            "sigreg_example_count": 64,
            "initialization": "raw BT4 plus seed-0 fresh JEPA/DFM modules",
            "validation_pool": "fast",
            "validation_examples": 8192,
            "arena_tier": "hero_development",
            "ordered_fens_sha256": next(iter(opening_orders)),
            "opening_pairs_per_edge": 128,
            "games_per_edge": 256,
            "primary_edges": len(primary),
            "primary_games": 256 * len(primary),
            "refinement_edges": len(refinement),
            "refinement_games": 256 * len(refinement),
            "focused_followup_games": 256,
            "additional_ply_cap": 256,
            "physical_inference_batch_cap": 16,
            "selection": "deterministic greedy legal argmax",
            "arena_evaluator_sha256": next(iter(evaluator_hashes)),
            "arena_recorded_git_commit": next(iter(evaluator_commits)),
        },
        "models": model_records,
        "primary_round_robin": {
            "direct_matches": public_primary,
            "joint_bradley_terry": {
                "anchor": "Hero-1024 = 0 Elo",
                "ratings": rating_records,
                "bootstrap": {
                    "samples": BOOTSTRAP_SAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "unit": (
                        "one opening-pair index resampled synchronously "
                        "across all 15 edges"
                    ),
                    "interval": "percentile 95%",
                },
                "max_absolute_score_residual": max(
                    abs(edge["joint_model_score_residual"])
                    for edge in public_primary
                ),
            },
        },
        "refinement_diagnostics": public_refinement,
        "focused_followup": public_followup,
        "artifacts": {
            "ledger_csv": _relative(OUTPUT_CSV),
            "training_plot": _relative(TRAINING_PLOT),
            "arena_plot": _relative(ARENA_PLOT),
        },
        "interpretation": {
            "one_pass": (
                "Current-state WDL is the strongest new one-pass arm and "
                "is approximately level with the retained Hero; normalized "
                "JEPA fusion and policy-prelogit are also close. Cloned "
                "all-horizon policy heads are decisively harmful."
            ),
            "refinement": (
                "Predicted-JEPA closed loop is the only arm whose eight-pass "
                "policy beats its own one-pass policy. Its focused eight-pass "
                "match is approximately level with Hero-1024 one-pass."
            ),
            "scope": (
                "All Elo values are frozen-pool checkpoint-relative estimates, "
                "not human, Lichess, or engine ratings."
            ),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": _relative(OUTPUT_JSON),
                "ratings": rating_records,
                "refinement_scores": refinement_scores,
                "closed_loop_p8_vs_hero_p1": public_followup["score"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
