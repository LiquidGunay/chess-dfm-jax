#!/usr/bin/env python3
"""Build the immutable post-Hero pass-scaling and initialization audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from research.analysis.build_sigreg_sample_count_experiment import (
    ROOT,
    arena_record,
    load_json,
    load_jsonl,
    paired_arena_delta,
    relative_source,
    sha256_file,
)


OUTPUT_JSON = (
    ROOT / "research/analysis/posthero_pass_bootstrap_audit_20260728.json"
)
CONTROL_CHECKPOINT = (
    ROOT
    / "research/runs/torch_autoresearch_sigreg64_u1024_v1"
    / "checkpoint/model.safetensors"
)
CANDIDATE_CHECKPOINT = (
    ROOT
    / "research/runs/torch_autoresearch_jepa_condition_u1024_replay_v1"
    / "checkpoint/model.safetensors"
)
HERO_CHECKPOINT = (
    ROOT
    / "research/runs/torch_hero_epoch_v1"
    / "checkpoint/model.safetensors"
)
EXPECTED_HASHES = {
    "control_1024": (
        "05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9"
    ),
    "candidate_1024": (
        "eb20c698fea1eabd4ebf71c13a1c195204736183f7b173079bd34f2a6e9a995f"
    ),
    "hero_one_epoch": (
        "665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692"
    ),
}
PASSES = (1, 2, 4, 8, 16)
CONTROL_ARENAS = {
    1: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p1-v1",
    2: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p2-v1",
    4: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p4-v1",
    8: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-v1",
    16: "autoresearch-sigreg64-u1024-vs-raw-bt4-128pairs-p16-v1",
}
CANDIDATE_ARENAS = {
    count: (
        "autoresearch-jepa-condition-u1024-replay-vs-raw-bt4-"
        f"128pairs-p{count}-v1"
    )
    for count in PASSES
}
HERO_ARENAS = {
    1: "hero-epoch-v1-terminal-vs-raw-bt4-128pairs-p1-v1",
    2: "hero-epoch-v1-terminal-vs-raw-bt4-128pairs-p2-v1",
    4: "hero-epoch-v1-terminal-vs-raw-bt4-128pairs-p4-v1",
    8: "hero-epoch-v1-terminal-vs-raw-bt4-1024pairs",
    16: "hero-epoch-v1-terminal-vs-raw-bt4-128pairs-p16-v1",
}
EVALUATIONS = {
    "initialization": "torch_hero_init_primary_eval_v1",
    "control_1024": "torch_autoresearch_sigreg64_u1024_v1_fast_eval_v1",
    "candidate_1024": (
        "torch_autoresearch_jepa_condition_u1024_v1_fast_eval_v1"
    ),
    "hero_one_epoch": "torch_hero_epoch_v1_terminal_blind_eval_v1",
}
TIMING_OR_RESOURCE_MARKERS = (
    "seconds",
    "memory",
    "examples_per_second",
    "checkpoint_write",
    "prefetch",
    "data_prepare",
    "data_wait",
    "data_cursor",
    "segment_",
)


def public_arena(record: dict[str, Any]) -> dict[str, Any]:
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
    expected_checkpoint_sha256: str,
    expected_jepa_used_at_inference: bool,
    allow_more_pairs: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = arena_path(name)
    state = load_json(path)
    record = arena_record(state)
    run = state["contract"]["run"]
    candidate = state["contract"]["models"]["candidate"]
    expected_pairs = 1024 if allow_more_pairs else 128
    if int(run["pair_count"]) != expected_pairs:
        raise ValueError(f"Unexpected pair count in {name}")
    if int(run["refinement_passes"]) != expected_passes:
        raise ValueError(f"Unexpected pass count in {name}")
    if candidate["state"]["sha256"] != expected_checkpoint_sha256:
        raise ValueError(f"Unexpected checkpoint in {name}")
    if (
        bool(run["jepa_used_at_inference"])
        != expected_jepa_used_at_inference
    ):
        raise ValueError(f"Unexpected JEPA inference use in {name}")
    if (
        int(run["additional_ply_cap"]) != 256
        or int(run["policy_batch_size_cap"]) != 16
        or int(run["seed"]) != 0
    ):
        raise ValueError(f"Frozen Arena contract drift in {name}")
    if record["fault_counts"] or record["cap_draw_count"] != 0:
        raise ValueError(f"Fault or cap draw in {name}")
    if record["termination_counts"] != {"normal": record["game_count"]}:
        raise ValueError(f"Abnormal termination in {name}")
    return state, record


def first_128_slice(record: dict[str, Any]) -> dict[str, Any]:
    scores = np.asarray(record["_pair_scores"][:128], dtype=np.float64)
    if scores.shape != (128,):
        raise ValueError("Arena does not contain the required first 128 pairs")
    return {
        "pair_count": 128,
        "score": float(scores.mean()),
        "points": float(scores.sum() * 2.0),
        "full_run_mean_physical_call_ms": float(
            record["mean_physical_call_ms"]
        ),
        "full_run_pair_count": int(record["pair_count"]),
        "_pair_scores": scores,
    }


def normalized_contract(state: dict[str, Any]) -> dict[str, Any]:
    run = dict(state["contract"]["run"])
    run.pop("refinement_passes")
    run.pop("pair_count")
    run.pop("jepa_used_at_inference")
    return {
        "run": run,
        "tier": state["contract"]["tier"],
        "opponent": state["contract"]["models"]["opponent"],
    }


def assert_contracts_match(
    states: dict[str, dict[int, dict[str, Any]]],
) -> None:
    reference = normalized_contract(states["control_1024"][1])
    for series, by_pass in states.items():
        for passes, state in by_pass.items():
            if normalized_contract(state) != reference:
                raise ValueError(
                    f"Pass-sweep contract drift for {series} at {passes}"
                )


def scientific_replay_comparison() -> dict[str, Any]:
    original_path = (
        ROOT
        / "research/runs/torch_autoresearch_jepa_condition_u1024_v1"
        / "metrics.jsonl"
    )
    replay_path = (
        ROOT
        / "research/runs/torch_autoresearch_jepa_condition_u1024_replay_v1"
        / "metrics.jsonl"
    )
    original = load_jsonl(original_path)
    replay = load_jsonl(replay_path)
    if len(original) != 1024 or len(replay) != 1024:
        raise ValueError("Candidate replay does not contain 1,024 updates")
    scientific_keys = sorted(
        key
        for key in original[0]
        if not any(marker in key for marker in TIMING_OR_RESOURCE_MARKERS)
    )
    mismatches = [
        {
            "update": index,
            "metric": key,
            "original": left.get(key),
            "replay": right.get(key),
        }
        for index, (left, right) in enumerate(
            zip(original, replay, strict=True),
            start=1,
        )
        for key in scientific_keys
        if left.get(key) != right.get(key)
    ]
    return {
        "original_updates": len(original),
        "replay_updates": len(replay),
        "scientific_scalar_keys": scientific_keys,
        "scientific_scalar_comparisons": (
            len(original) * len(scientific_keys)
        ),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
        "checkpoint_sha256": sha256_file(CANDIDATE_CHECKPOINT),
        "checkpoint_matches_sealed_original": (
            sha256_file(CANDIDATE_CHECKPOINT)
            == EXPECTED_HASHES["candidate_1024"]
        ),
        "sources": {
            "original_metrics": relative_source(original_path),
            "replay_metrics": relative_source(replay_path),
        },
    }


def horizon_record(
    report: dict[str, Any],
) -> list[dict[str, float | int | None]]:
    metrics = report["metrics"]
    rows: list[dict[str, float | int | None]] = []
    for horizon in range(1, 9):
        mse = float(metrics[f"jepa_mse_by_horizon_h{horizon}"])
        identity = float(
            metrics[f"identity_mse_by_horizon_h{horizon}"]
        )
        rows.append(
            {
                "horizon": horizon,
                "jepa_mse": mse,
                "identity_mse": identity,
                "mse_over_identity": mse / identity,
                "pred_target_cosine": float(
                    metrics[
                        f"pred_target_cosine_by_horizon_h{horizon}"
                    ]
                ),
                "dfm_ce": float(
                    metrics[f"dfm_ce_loss_by_horizon_h{horizon}"]
                ),
                "pred_effective_rank": float(
                    metrics[
                        f"pred_effective_rank_by_horizon_h{horizon}"
                    ]
                ),
                "pred_stable_rank": (
                    None
                    if (
                        value := metrics.get(
                            f"pred_stable_rank_by_horizon_h{horizon}"
                        )
                    )
                    is None
                    else float(value)
                ),
            }
        )
    return rows


def main() -> int:
    observed_hashes = {
        "control_1024": sha256_file(CONTROL_CHECKPOINT),
        "candidate_1024": sha256_file(CANDIDATE_CHECKPOINT),
        "hero_one_epoch": sha256_file(HERO_CHECKPOINT),
    }
    if observed_hashes != EXPECTED_HASHES:
        raise ValueError("Pass-audit checkpoint hash mismatch")

    states: dict[str, dict[int, dict[str, Any]]] = {
        "control_1024": {},
        "candidate_1024": {},
        "hero_one_epoch": {},
    }
    records: dict[str, dict[int, dict[str, Any]]] = {
        "control_1024": {},
        "candidate_1024": {},
        "hero_one_epoch": {},
    }
    sweep_specs = (
        (
            "control_1024",
            CONTROL_ARENAS,
            EXPECTED_HASHES["control_1024"],
            False,
        ),
        (
            "candidate_1024",
            CANDIDATE_ARENAS,
            EXPECTED_HASHES["candidate_1024"],
            True,
        ),
        (
            "hero_one_epoch",
            HERO_ARENAS,
            EXPECTED_HASHES["hero_one_epoch"],
            False,
        ),
    )
    for series, arenas, checkpoint_hash, jepa_used in sweep_specs:
        for passes, name in arenas.items():
            state, record = load_checked_arena(
                name,
                expected_passes=passes,
                expected_checkpoint_sha256=checkpoint_hash,
                expected_jepa_used_at_inference=jepa_used,
                allow_more_pairs=(series == "hero_one_epoch" and passes == 8),
            )
            states[series][passes] = state
            records[series][passes] = record
    assert_contracts_match(states)

    hero_eight_slice = first_128_slice(records["hero_one_epoch"][8])
    pass_curves: dict[str, list[dict[str, Any]]] = {}
    for series in records:
        pass_curves[series] = []
        for passes in PASSES:
            record = records[series][passes]
            if series == "hero_one_epoch" and passes == 8:
                public = {
                    key: value
                    for key, value in hero_eight_slice.items()
                    if not key.startswith("_")
                }
            else:
                public = public_arena(record)
            pass_curves[series].append(
                {
                    "passes": passes,
                    **public,
                }
            )

    candidate_vs_control = {
        f"pass_{passes}": paired_arena_delta(
            records["control_1024"][passes]["_pair_scores"],
            records["candidate_1024"][passes]["_pair_scores"],
        )
        for passes in PASSES
    }
    hero_pair_scores = {
        passes: (
            hero_eight_slice["_pair_scores"]
            if passes == 8
            else records["hero_one_epoch"][passes]["_pair_scores"]
        )
        for passes in PASSES
    }
    within_series_vs_one_pass = {
        series: {
            f"pass_{passes}": paired_arena_delta(
                (
                    records[series][1]["_pair_scores"]
                    if series != "hero_one_epoch"
                    else hero_pair_scores[1]
                ),
                (
                    records[series][passes]["_pair_scores"]
                    if series != "hero_one_epoch"
                    else hero_pair_scores[passes]
                ),
            )
            for passes in PASSES
            if passes != 1
        }
        for series in records
    }

    horizon_audit: dict[str, Any] = {}
    evaluation_sources: dict[str, Any] = {}
    for stage, run_name in EVALUATIONS.items():
        path = ROOT / "research/runs" / run_name / "report.json"
        report = load_json(path)
        horizon_audit[stage] = horizon_record(report)
        evaluation_sources[stage] = relative_source(path)

    arena_sources = {
        series: {
            f"pass_{passes}": relative_source(arena_path(name))
            for passes, name in arenas.items()
        }
        for series, arenas in (
            ("control_1024", CONTROL_ARENAS),
            ("candidate_1024", CANDIDATE_ARENAS),
            ("hero_one_epoch", HERO_ARENAS),
        )
    }
    replay = scientific_replay_comparison()
    result = {
        "schema_version": "posthero-pass-bootstrap-audit-v1",
        "question": (
            "Which model the 0.509765625 candidate faced, how inference-pass "
            "scaling changes with training, and which target-safe "
            "initialization bootstrap should be tested next."
        ),
        "model_identities": {
            "control_1024": {
                "updates": 1024,
                "examples": 1048576,
                "checkpoint_sha256": observed_hashes["control_1024"],
            },
            "candidate_1024": {
                "updates": 1024,
                "examples": 1048576,
                "checkpoint_sha256": observed_hashes["candidate_1024"],
                "sole_architecture_change": (
                    "zero-initialized broadcast residual from current JEPA "
                    "state into every DFM state token"
                ),
            },
            "hero_one_epoch": {
                "updates": 27679,
                "examples": 28343296,
                "checkpoint_sha256": observed_hashes["hero_one_epoch"],
            },
        },
        "direct_candidate_screen": {
            "opponent": "matched control_1024, not hero_one_epoch",
            "score": 0.509765625,
            "candidate_wins_draws_losses": [66, 129, 61],
            "descriptive_paired_t_95": [0.467119, 0.552412],
            "interpretation": (
                "Chess-inconclusive near tie. The candidate failed "
                "preregistered offline policy/rank promotion gates; that is "
                "not evidence that the architecture is weaker at chess."
            ),
        },
        "deterministic_candidate_replay": replay,
        "pass_curves": pass_curves,
        "candidate_vs_control_paired_by_pass": candidate_vs_control,
        "within_series_vs_one_pass": within_series_vs_one_pass,
        "horizon_audit": horizon_audit,
        "initialization": {
            "dfm_h1": (
                "Learned output projection starts at zero and current-board "
                "BT4 policy logits are permanently added to H1."
            ),
            "dfm_h2_h8": (
                "Learned output projection starts at zero with no future "
                "BT4 residual, so H2-H8 begin uniform."
            ),
            "jepa": (
                "Conditioning modulation starts at zero and transition "
                "down-projection uses tiny scale, so recurrent predictions "
                "begin approximately as the current-state identity."
            ),
        },
        "bootstrap_recommendation": {
            "first_candidate": (
                "training-only future-policy distillation from the already "
                "encoded sampled future BT4 tokens"
            ),
            "alignment": (
                "future_planes[s] is the board after action s+1; for "
                "s in 0..6, its stopped-gradient BT4 legal policy supervises "
                "DFM slot s+1 (H2-H8). The sampled H8 future board has no "
                "in-chunk next-action slot and is ineligible."
            ),
            "no_inference_leakage": True,
            "extra_encoder_calls": 0,
            "extra_training_work": (
                "one frozen policy-head forward on the selected future "
                "tokens plus legal KL; profile before training"
            ),
            "schedule": (
                "calibrate the initial weighted contribution, use it only as "
                "a warm-start auxiliary, and decay it to zero so LC0-MCTS "
                "actions remain the terminal objective"
            ),
            "separate_second_candidate": (
                "teacher-forced adjacent JEPA transition "
                "T(z_target[h-1], action[h]) -> z_target[h], followed by "
                "scheduled transition to free recurrent rollout"
            ),
            "do_not_combine_first": True,
        },
        "decision": {
            "candidate_status": (
                "not promoted; current evidence is scientifically "
                "inconclusive, with favorable reused-pool pass curves but "
                "failed offline gates"
            ),
            "delete_candidate_checkpoint": False,
            "close_all_state_action_coupling": False,
            "next_training_priority": (
                "confirm whether future-policy distillation improves "
                "long-horizon CE/pass scaling without degrading one-pass Elo"
            ),
        },
        "sources": {
            "arenas": arena_sources,
            "evaluations": evaluation_sources,
            "audit_plan": relative_source(
                ROOT
                / "research/experiment_posthero_pass_and_bootstrap_audit.md"
            ),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT_JSON.relative_to(ROOT)),
                "candidate_curve": [
                    row["score"] for row in pass_curves["candidate_1024"]
                ],
                "control_curve": [
                    row["score"] for row in pass_curves["control_1024"]
                ],
                "hero_curve": [
                    row["score"] for row in pass_curves["hero_one_epoch"]
                ],
                "candidate_minus_control": {
                    key: value["candidate_minus_control_score"]
                    for key, value in candidate_vs_control.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
