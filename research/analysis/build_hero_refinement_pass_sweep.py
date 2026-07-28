#!/usr/bin/env python3
"""Build the immutable hero refinement-pass sweep record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from research.analysis.build_sigreg_sample_count_experiment import (
    ROOT,
    arena_record,
    load_json,
    paired_arena_delta,
    relative_source,
    sha256_file,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_JSON = (
    ROOT / "research/analysis/hero_refinement_pass_sweep_20260728.json"
)
OUTPUT_PNG = (
    ROOT / "research/analysis/hero_refinement_pass_sweep_20260728.png"
)
CHECKPOINT_PATH = (
    ROOT
    / "research/runs/torch_autoresearch_sigreg64_u1024_v1"
    / "checkpoint/model.safetensors"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9"
)
SCREEN_ARENAS = {
    1: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p1-v1",
    2: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p2-v1",
    4: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p4-v1",
    8: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-v1",
    16: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p16-v1",
}
CONFIRMATION_ARENAS = {
    1: "autoresearch-sigreg64-u1024-vs-raw-bt4-256pairs-p1-confirm-v1",
    8: "autoresearch-sigreg64-u1024-vs-raw-bt4-256pairs-p8-confirm-v1",
}


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def arena_path(name: str) -> Path:
    return ROOT / "artifacts/arena" / name / "state.json"


def load_checked_arena(
    name: str,
    *,
    expected_passes: int,
    expected_pairs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = arena_path(name)
    state = load_json(path)
    record = arena_record(state)
    contract = state["contract"]
    run = contract["run"]
    candidate = contract["models"]["candidate"]
    if int(run["refinement_passes"]) != expected_passes:
        raise ValueError(f"Wrong pass count in {name}")
    if int(run["pair_count"]) != expected_pairs:
        raise ValueError(f"Wrong pair count in {name}")
    if bool(run["jepa_used_at_inference"]):
        raise ValueError(f"JEPA feedback unexpectedly active in {name}")
    if candidate["state"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Wrong checkpoint in {name}")
    if candidate["model_config"].get("jepa_feedback_mode", "none") != "none":
        raise ValueError(f"Feedback mode unexpectedly active in {name}")
    if record["fault_counts"] or record["cap_draw_count"] != 0:
        raise ValueError(f"Fault or cap draw in {name}")
    if record["termination_counts"] != {"normal": record["game_count"]}:
        raise ValueError(f"Abnormal termination in {name}")
    if not np.isfinite(record["mean_physical_call_ms"]):
        raise ValueError(f"Non-finite timing in {name}")
    return state, record


def normalized_run_contract(
    record: dict[str, Any],
) -> dict[str, Any]:
    contract = dict(record["run_contract"])
    contract.pop("refinement_passes")
    contract.pop("pair_count")
    return contract


def assert_matched_contracts(
    states: dict[int, dict[str, Any]],
    records: dict[int, dict[str, Any]],
) -> None:
    first_pass = next(iter(records))
    expected_run = normalized_run_contract(records[first_pass])
    expected_tier = records[first_pass]["tier_contract"]
    expected_reference = records[first_pass]["reference_model_id"]
    expected_pool = states[first_pass]["contract"]["tier"]["ordered_fens_sha256"]
    for passes, record in records.items():
        if normalized_run_contract(record) != expected_run:
            raise ValueError(f"Run contract drift at {passes} passes")
        if record["tier_contract"] != expected_tier:
            raise ValueError(f"Tier contract drift at {passes} passes")
        if record["reference_model_id"] != expected_reference:
            raise ValueError(f"Reference drift at {passes} passes")
        ordered_fens = states[passes]["contract"]["tier"][
            "ordered_fens_sha256"
        ]
        if ordered_fens != expected_pool:
            raise ValueError(f"Opening order drift at {passes} passes")


def paired_against_eight(
    records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    baseline = records[8]["_pair_scores"]
    return {
        f"pass_{passes}": paired_arena_delta(
            baseline,
            records[passes]["_pair_scores"],
        )
        for passes in sorted(records)
        if passes != 8
    }


def build_plot(
    screen: dict[int, dict[str, Any]],
    confirmation: dict[int, dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.2),
        constrained_layout=True,
    )
    passes = np.asarray(sorted(screen), dtype=np.int64)
    scores = np.asarray(
        [screen[int(count)]["score"] for count in passes],
        dtype=np.float64,
    )
    call_ms = np.asarray(
        [
            screen[int(count)]["mean_physical_call_ms"]
            for count in passes
        ],
        dtype=np.float64,
    )
    color = "#31688e"
    highlight = "#d1495b"

    axes[0].plot(passes, scores, marker="o", color=color)
    axes[0].scatter([1], [screen[1]["score"]], color=highlight, zorder=3)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(passes, [str(value) for value in passes])
    axes[0].set(
        title="Development strength by refinement count",
        xlabel="DFM planner calls",
        ylabel="score vs raw BT4",
    )

    axes[1].plot(passes, call_ms, marker="o", color=color)
    axes[1].scatter(
        [1],
        [screen[1]["mean_physical_call_ms"]],
        color=highlight,
        zorder=3,
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(passes, [str(value) for value in passes])
    axes[1].set(
        title="Measured candidate policy-call latency",
        xlabel="DFM planner calls",
        ylabel="mean physical call (ms)",
    )

    axes[2].plot(call_ms, scores, color="0.65", linewidth=1)
    axes[2].scatter(call_ms, scores, color=color, s=45)
    for count, x_value, y_value in zip(
        passes,
        call_ms,
        scores,
        strict=True,
    ):
        axes[2].annotate(
            f"{count} pass{'es' if count != 1 else ''}",
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[2].scatter(
        [
            confirmation[1]["mean_physical_call_ms"],
            confirmation[8]["mean_physical_call_ms"],
        ],
        [confirmation[1]["score"], confirmation[8]["score"]],
        marker="*",
        s=120,
        color=[highlight, "#443983"],
        label="256-pair confirmation",
        zorder=4,
    )
    axes[2].set(
        title="Strength/latency frontier",
        xlabel="mean physical call (ms)",
        ylabel="score vs raw BT4",
    )
    axes[2].legend(loc="lower left")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Hero DFM refinement passes: extra iterations hurt root strength",
        fontsize=14,
    )
    figure.savefig(OUTPUT_PNG, dpi=180)
    plt.close(figure)


def main() -> int:
    if sha256_file(CHECKPOINT_PATH) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Retained checkpoint checksum mismatch")

    screen_states: dict[int, dict[str, Any]] = {}
    screen_records: dict[int, dict[str, Any]] = {}
    for passes, name in SCREEN_ARENAS.items():
        state, record = load_checked_arena(
            name,
            expected_passes=passes,
            expected_pairs=128,
        )
        screen_states[passes] = state
        screen_records[passes] = record
    assert_matched_contracts(screen_states, screen_records)

    confirmation_states: dict[int, dict[str, Any]] = {}
    confirmation_records: dict[int, dict[str, Any]] = {}
    for passes, name in CONFIRMATION_ARENAS.items():
        state, record = load_checked_arena(
            name,
            expected_passes=passes,
            expected_pairs=256,
        )
        confirmation_states[passes] = state
        confirmation_records[passes] = record
    assert_matched_contracts(
        confirmation_states,
        confirmation_records,
    )
    if (
        normalized_run_contract(screen_records[1])
        != normalized_run_contract(confirmation_records[1])
    ):
        raise ValueError("Screen/confirmation run contracts differ")
    if (
        screen_records[1]["tier_contract"]
        != confirmation_records[1]["tier_contract"]
    ):
        raise ValueError("Screen/confirmation tier contracts differ")

    first_half_exact = {
        f"pass_{passes}": bool(
            np.array_equal(
                confirmation_records[passes]["_pair_scores"][:128],
                screen_records[passes]["_pair_scores"],
            )
        )
        for passes in (1, 8)
    }
    if not all(first_half_exact.values()):
        raise ValueError("Confirmation did not reproduce the first half")

    combined_delta = paired_arena_delta(
        confirmation_records[8]["_pair_scores"],
        confirmation_records[1]["_pair_scores"],
    )
    tail_delta = paired_arena_delta(
        confirmation_records[8]["_pair_scores"][128:],
        confirmation_records[1]["_pair_scores"][128:],
    )
    latency_ratio = (
        confirmation_records[1]["mean_physical_call_ms"]
        / confirmation_records[8]["mean_physical_call_ms"]
    )
    gates = {
        "first_128_pair_scores_reproduce_exactly": all(
            first_half_exact.values()
        ),
        "new_128_pair_delta_positive": (
            tail_delta["candidate_minus_control_score"] > 0.0
        ),
        "combined_256_interval_lower_positive": (
            combined_delta["paired_t_95_lower"] > 0.0
        ),
        "one_pass_latency_at_most_0p8x_eight": latency_ratio <= 0.8,
        "zero_faults_and_cap_draws": all(
            not record["fault_counts"] and record["cap_draw_count"] == 0
            for record in confirmation_records.values()
        ),
    }
    confirmed = all(gates.values())

    build_plot(screen_records, confirmation_records)
    record = {
        "schema_version": "hero-refinement-pass-sweep-v1",
        "question": (
            "How does DFM refinement count trade searchless strength for "
            "latency on the retained hero checkpoint?"
        ),
        "model": {
            "checkpoint": relative_source(CHECKPOINT_PATH),
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "training_horizon": 8,
            "jepa_feedback_mode": "none",
        },
        "development_screen": {
            f"pass_{passes}": {
                **public_record(screen_records[passes]),
                "source": relative_source(
                    arena_path(SCREEN_ARENAS[passes])
                ),
            }
            for passes in sorted(screen_records)
        },
        "development_paired_against_eight": paired_against_eight(
            screen_records
        ),
        "confirmation": {
            "first_half_exact_reproduction": first_half_exact,
            "pass_1": {
                **public_record(confirmation_records[1]),
                "source": relative_source(
                    arena_path(CONFIRMATION_ARENAS[1])
                ),
            },
            "pass_8": {
                **public_record(confirmation_records[8]),
                "source": relative_source(
                    arena_path(CONFIRMATION_ARENAS[8])
                ),
            },
            "new_pairs_128_to_255": tail_delta,
            "combined_first_256": combined_delta,
            "latency_ratio_pass_1_over_pass_8": latency_ratio,
            "latency_reduction_fraction": 1.0 - latency_ratio,
        },
        "decision": {
            "status": (
                "confirm_one_pass_for_checkpoint"
                if confirmed
                else "retain_eight_pass_anchor"
            ),
            "all_preregistered_confirmation_gates_pass": confirmed,
            "gates": gates,
            "confirmed_fast_strength_passes": 1 if confirmed else 8,
            "continuity_passes": 8,
            "training_horizon_unchanged": 8,
            "scope": (
                "Inference-count result for checkpoint SHA "
                f"{EXPECTED_CHECKPOINT_SHA256}; future architectures must "
                "still report their own pass-count sensitivity."
            ),
        },
        "plot": {
            "path": str(OUTPUT_PNG.relative_to(ROOT)),
            "sha256": sha256_file(OUTPUT_PNG),
        },
        "guarded_pre_game_stops": {
            "count": 2,
            "model_or_game_state_written": False,
            "cause": (
                "Two duplicated legacy pass-equals-horizon guards were "
                "generalized only for feedback-off inference and retained "
                "as exact-eight guards for final-pass JEPA feedback."
            ),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": record["decision"]["status"],
                "tail_delta": tail_delta["candidate_minus_control_score"],
                "combined_delta": combined_delta[
                    "candidate_minus_control_score"
                ],
                "latency_ratio": latency_ratio,
                "output_json": str(OUTPUT_JSON),
                "output_png": str(OUTPUT_PNG),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
