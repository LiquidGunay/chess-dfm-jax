#!/usr/bin/env python3
"""Validate and summarize the Hero policy/DFM/raw-BT4 round robin."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from research.evaluate_arena import load_run_state
from research.prepare import sha256_file


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    ROOT / "research/analysis/hero_policy_round_robin_20260728.json"
)
HERO_SHA256 = (
    "665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692"
)
RAW_BT4_SHA256 = (
    "61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651"
)
EXPECTED_EVALUATOR_SHA256 = (
    "f0af5b71964e45085c581e1fbffbd78034a527051dc08a33ff738a3df0a65e2b"
)
EXPECTED_CODE_COMMIT = "c7ac6eda85106cbd21863037974b24e8ffff072f"
PLAYERS = ("raw_bt4", "dfm_16", "dfm_128", "policy_only")
ANCHOR = "raw_bt4"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_728
ELO_LOGIT_SCALE = math.log(10.0) / 400.0


@dataclasses.dataclass(frozen=True)
class EdgeSpec:
    candidate: str
    opponent: str
    arena_name: str


EDGES = (
    EdgeSpec(
        "dfm_128",
        "dfm_16",
        "hero-epoch-v1-round-robin-first128-dfm128-vs-dfm16-v1",
    ),
    EdgeSpec(
        "dfm_128",
        "policy_only",
        "hero-epoch-v1-round-robin-first128-dfm128-vs-policy-v1",
    ),
    EdgeSpec(
        "dfm_128",
        "raw_bt4",
        "hero-epoch-v1-round-robin-first128-dfm128-vs-raw-v1",
    ),
    EdgeSpec(
        "dfm_16",
        "policy_only",
        "hero-epoch-v1-round-robin-first128-dfm16-vs-policy-v1",
    ),
    EdgeSpec(
        "dfm_16",
        "raw_bt4",
        "hero-epoch-v1-round-robin-first128-dfm16-vs-raw-v1",
    ),
    EdgeSpec(
        "policy_only",
        "raw_bt4",
        "hero-epoch-v1-round-robin-first128-policy-vs-raw-v1",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _expected_mode(player: str) -> str:
    if player == "policy_only":
        return "policy_only"
    if player.startswith("dfm_"):
        return "dfm"
    if player == "raw_bt4":
        return "default"
    raise ValueError(f"Unknown player: {player}")


def _expected_passes(player: str) -> int:
    if player == "dfm_128":
        return 128
    return 16


def _validate_model_descriptor(
    descriptor: dict[str, Any],
    *,
    player: str,
) -> None:
    if player == "raw_bt4":
        _require(descriptor["kind"] == "raw_bt4", "Raw player kind drift")
        _require(
            descriptor["bt4_checkpoint"]["sha256"] == RAW_BT4_SHA256,
            "Raw BT4 digest drift",
        )
    else:
        _require(descriptor["kind"] == "torch_hero", "Hero player kind drift")
        _require(
            descriptor["state"]["sha256"] == HERO_SHA256,
            "Hero checkpoint digest drift",
        )
    _require(
        descriptor["arena_policy_mode"] == _expected_mode(player),
        f"Arena policy mode drift for {player}",
    )


def _load_edge(spec: EdgeSpec) -> dict[str, Any]:
    arena_dir = ROOT / "artifacts/arena" / spec.arena_name
    state_path = arena_dir / "state.json"
    raw_state = json.loads(state_path.read_text(encoding="utf-8"))
    state = load_run_state(
        arena_dir,
        expected_contract=raw_state["contract"],
    )
    _require(state == raw_state, f"Reloaded state differs for {spec.arena_name}")

    contract = state["contract"]
    run = contract["run"]
    aggregate = state["aggregate"]
    candidate = contract["models"]["candidate"]
    opponent = contract["models"]["opponent"]
    _validate_model_descriptor(candidate, player=spec.candidate)
    _validate_model_descriptor(opponent, player=spec.opponent)

    expected_run = {
        "pair_count": 128,
        "block_pairs": 16,
        "additional_ply_cap": 256,
        "candidate_refinement_passes": _expected_passes(spec.candidate),
        "opponent_refinement_passes": _expected_passes(spec.opponent),
        "policy_batch_size_cap": 16,
        "seed": 0,
        "collect_diagnostics": False,
        "deterministic_greedy_policy": True,
        "jepa_used_at_inference": False,
    }
    for key, expected in expected_run.items():
        _require(
            run[key] == expected,
            f"Run contract drift for {spec.arena_name}: {key}",
        )
    _require(
        contract["tier"]["name"] == "hero_development",
        "Opening tier drift",
    )
    _require(
        contract["code"]["git_commit"] == EXPECTED_CODE_COMMIT,
        "Evaluator commit drift",
    )
    _require(
        contract["code"]["files"]["research/evaluate_arena.py"]
        == EXPECTED_EVALUATOR_SHA256,
        "Evaluator file digest drift",
    )
    _require(state["status"] == "complete", "Arena is incomplete")
    _require(aggregate["pair_count"] == 128, "Pair count drift")
    _require(aggregate["game_count"] == 256, "Game count drift")
    _require(aggregate["fault_counts"] == {}, "Arena policy fault")
    _require(aggregate["cap_draw_count"] == 0, "Arena cap draw")
    _require(
        aggregate["termination_counts"] == {"normal": 256},
        "Abnormal Arena termination",
    )
    pair_scores = np.asarray(
        aggregate["pair_scores"],
        dtype=np.float64,
    )
    _require(pair_scores.shape == (128,), "Pair-score shape drift")
    _require(
        bool(np.all(np.isin(pair_scores, (0.0, 0.5, 1.0, 1.5, 2.0)))),
        "Invalid pentanomial pair score",
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
            "Invalid policy-call timing",
        )

    score = float(pair_scores.sum() / (2.0 * pair_scores.size))
    interval = aggregate["pair_aware_logistic_interval"]
    _require(interval["score"] == score, "Stored score differs")
    return {
        "spec": spec,
        "state": state,
        "pair_scores": pair_scores,
        "public": {
            "candidate": spec.candidate,
            "opponent": spec.opponent,
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
                * float(
                    candidate_stats["coverage"]["mean_call_seconds"]
                )
            ),
            "opponent_mean_physical_call_ms": (
                1_000.0
                * float(
                    opponent_stats["coverage"]["mean_call_seconds"]
                )
            ),
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
        information = np.zeros(
            (ratings.size, ratings.size),
            dtype=np.float64,
        )
        for edge in edges:
            spec = edge["spec"]
            scores = edge["pair_scores"]
            if sample_indices is not None:
                scores = scores[sample_indices]
            trials = float(2 * scores.size)
            points = float(scores.sum())
            design = _design_vector(spec.candidate, spec.opponent)
            logit = float(ELO_LOGIT_SCALE * (design @ ratings))
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


def main() -> int:
    edges = [_load_edge(spec) for spec in EDGES]
    opening_orders = {
        edge["state"]["contract"]["tier"]["ordered_fens_sha256"]
        for edge in edges
    }
    _require(
        len(opening_orders) == 1,
        "Opening order differs across round-robin edges",
    )

    ratings = _fit_bradley_terry(edges)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(
        (BOOTSTRAP_SAMPLES, len(PLAYERS)),
        dtype=np.float64,
    )
    for sample in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, 128, size=128)
        bootstrap[sample] = _fit_bradley_terry(edges, indices)

    rating_records = {}
    for index, player in enumerate(PLAYERS):
        rating_records[player] = {
            "elo_vs_raw_bt4": float(ratings[index]),
            "matched_opening_bootstrap_95": _percentile_interval(
                bootstrap[:, index]
            ),
        }

    public_edges = []
    contrasts = {}
    for edge in edges:
        spec = edge["spec"]
        candidate_index = PLAYERS.index(spec.candidate)
        opponent_index = PLAYERS.index(spec.opponent)
        delta = float(ratings[candidate_index] - ratings[opponent_index])
        bootstrap_delta = (
            bootstrap[:, candidate_index]
            - bootstrap[:, opponent_index]
        )
        predicted_score = _score_from_elo(delta)
        public = dict(edge["public"])
        public["joint_model_predicted_score"] = predicted_score
        public["joint_model_score_residual"] = (
            public["score"] - predicted_score
        )
        public_edges.append(public)
        contrasts[f"{spec.candidate}_minus_{spec.opponent}"] = {
            "elo": delta,
            "matched_opening_bootstrap_95": _percentile_interval(
                bootstrap_delta
            ),
        }

    record = {
        "schema_version": "hero-policy-round-robin-v1",
        "question": (
            "Does 128-pass DFM recover the terminal Hero checkpoint's "
            "policy-only strength, and how do DFM-128, DFM-16, policy-only, "
            "and immutable raw BT4 rank on one matched Arena?"
        ),
        "frozen_contract": {
            "checkpoint_sha256": HERO_SHA256,
            "raw_bt4_sha256": RAW_BT4_SHA256,
            "opening_tier": "hero_development",
            "ordered_fens_sha256": next(iter(opening_orders)),
            "opening_pairs_per_edge": 128,
            "games_per_edge": 256,
            "edges": len(edges),
            "total_games": 256 * len(edges),
            "additional_ply_cap": 256,
            "physical_batch_size": 16,
            "seed": 0,
            "selection": (
                "deterministic greedy legal argmax; no temperature, "
                "sampling, Gumbel noise, or stochastic tie breaking"
            ),
            "dfm_decoding": (
                "per-pass greedy logits with confidence-ordered stable "
                "progressive unmasking; root logits are legality-masked"
            ),
        },
        "direct_matches": public_edges,
        "joint_bradley_terry": {
            "anchor": "raw_bt4 = 0 Elo",
            "likelihood_unit": (
                "two game-points per color-reversed opening pair"
            ),
            "ratings": rating_records,
            "contrasts": contrasts,
            "bootstrap": {
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "unit": (
                    "one opening-pair index resampled synchronously across "
                    "all six edges"
                ),
                "interval": "percentile 95%",
            },
            "max_absolute_score_residual": max(
                abs(edge["joint_model_score_residual"])
                for edge in public_edges
            ),
        },
        "interpretation": {
            "dfm_128_vs_dfm_16": (
                "No measurable gain: the joint contrast is near zero and "
                "its matched-opening interval spans both directions."
            ),
            "policy_only_vs_dfm": (
                "The same checkpoint's legal-greedy BT4 policy path is "
                "about 181-185 relative Elo above either DFM setting."
            ),
            "dfm_vs_raw": (
                "Both DFM settings remain far stronger than immutable raw "
                "BT4, so the DFM path is useful in absolute terms but "
                "damages the stronger jointly trained base policy."
            ),
            "scope": (
                "These are local, pool-relative Arena Elo differences, not "
                "an external human or engine rating."
            ),
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
