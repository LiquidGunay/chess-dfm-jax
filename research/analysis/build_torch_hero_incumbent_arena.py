#!/usr/bin/env python3
"""Validate and seal the native Torch hero-incumbent Arena smoke."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from research.evaluate_arena import load_run_state
from research.prepare import sha256_file


ROOT = Path(__file__).resolve().parents[2]
ARENA_DIR = (
    ROOT / "artifacts/arena/torch-hero-selfmatch-p1-16pairs-v1"
)
OUTPUT_PATH = (
    ROOT
    / "research/analysis/torch_hero_incumbent_arena_20260728.json"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9"
)
EXPECTED_CODE_COMMIT = "00b7e82237deb2923dee85a8bc6b2dd9d13acbc7"
EXPECTED_COMPILE_REGIONS = [
    "state_projector_blocks",
    "dfm_blocks",
    "jepa_transition",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    state_path = ARENA_DIR / "state.json"
    raw_state = json.loads(state_path.read_text(encoding="utf-8"))
    state = load_run_state(
        ARENA_DIR,
        expected_contract=raw_state["contract"],
    )
    _require(state == raw_state, "Reloaded Arena state differs")

    contract = state["contract"]
    aggregate = state["aggregate"]
    run = contract["run"]
    candidate = contract["models"]["candidate"]
    opponent = contract["models"]["opponent"]
    candidate_id = candidate["model_id"]
    opponent_id = opponent["model_id"]
    candidate_stats = aggregate["models"][candidate_id]
    opponent_stats = aggregate["models"][opponent_id]
    session = state["sessions"][0]

    output_files = sorted(
        _relative(path)
        for path in ARENA_DIR.rglob("*")
        if path.is_file()
    )
    gates = {
        "state_complete": state["status"] == "complete",
        "code_commit_exact": (
            contract["code"]["git_commit"] == EXPECTED_CODE_COMMIT
        ),
        "run_contract_exact": (
            run["pair_count"] == 16
            and run["block_pairs"] == 16
            and run["additional_ply_cap"] == 256
            and run["refinement_passes"] == 1
            and run["policy_batch_size_cap"] == 16
            and run["seed"] == 0
            and run["collect_diagnostics"] is False
            and run["deterministic_greedy_policy"] is True
            and run["jepa_used_at_inference"] is False
        ),
        "independent_role_ids": (
            candidate_id
            == "candidate-torch-hero-"
            + EXPECTED_CHECKPOINT_SHA256[:12]
            and opponent_id
            == "incumbent-torch-hero-"
            + EXPECTED_CHECKPOINT_SHA256[:12]
            and candidate_id != opponent_id
        ),
        "descriptor_contracts_exact": (
            candidate["kind"] == "torch_hero"
            and opponent["kind"] == "torch_hero"
            and candidate["action_codec_id"] == "lc0_canonical_1858"
            and opponent["action_codec_id"] == "lc0_canonical_1858"
            and candidate["state"]["sha256"]
            == EXPECTED_CHECKPOINT_SHA256
            and opponent["state"]["sha256"]
            == EXPECTED_CHECKPOINT_SHA256
            and candidate["state"] == opponent["state"]
            and candidate["model_config"] == opponent["model_config"]
        ),
        "loads_independently_verified": (
            len(state["sessions"]) == 1
            and session["framework"] == "torch"
            and session["torch_devices"] == ["NVIDIA A10G"]
            and session["candidate_load"]["checkpoint_state_sha256"]
            == EXPECTED_CHECKPOINT_SHA256
            and session["opponent_load"]["checkpoint_state_sha256"]
            == EXPECTED_CHECKPOINT_SHA256
            and session["candidate_load"]["compile_regions"]
            == EXPECTED_COMPILE_REGIONS
            and session["opponent_load"]["compile_regions"]
            == EXPECTED_COMPILE_REGIONS
            and session["candidate_load"]["raw_initialization_sha256"]
            == session["opponent_load"]["raw_initialization_sha256"]
        ),
        "all_pair_scores_exactly_centered": (
            aggregate["pair_scores"] == [1.0] * 16
            and aggregate["pentanomial"]["counts"] == [0, 0, 16, 0, 0]
            and aggregate["pentanomial"]["score"] == 0.5
            and aggregate["pentanomial"]["points"] == 16.0
        ),
        "all_games_normal": (
            aggregate["pair_count"] == 16
            and aggregate["game_count"] == 32
            and aggregate["termination_counts"] == {"normal": 32}
            and aggregate["fault_counts"] == {}
            and aggregate["cap_draw_count"] == 0
        ),
        "role_stats_symmetric": (
            candidate_stats["games"] == opponent_stats["games"] == 32
            and candidate_stats["points"]
            == opponent_stats["points"]
            == 16.0
            and candidate_stats["wins"]
            == opponent_stats["wins"]
            == 7
            and candidate_stats["draws"]
            == opponent_stats["draws"]
            == 18
            and candidate_stats["losses"]
            == opponent_stats["losses"]
            == 7
            and candidate_stats["positions_evaluated"]
            == opponent_stats["positions_evaluated"]
            and candidate_stats["policy_calls"]
            == opponent_stats["policy_calls"]
        ),
        "complete_action_coverage": all(
            stats["coverage"]["incomplete_coverage_positions"] == 0
            and stats["coverage"]["representable_fraction"] == 1.0
            and stats["coverage"]["legal_moves"]
            == stats["coverage"]["representable_legal_actions"]
            for stats in (candidate_stats, opponent_stats)
        ),
        "finite_policy_timing": all(
            _finite_positive(stats["coverage"]["mean_call_seconds"])
            and _finite_positive(stats["coverage"]["max_call_seconds"])
            for stats in (candidate_stats, opponent_stats)
        ),
        "no_model_state_written": (
            output_files
            == [
                (
                    "artifacts/arena/torch-hero-selfmatch-p1-16pairs-v1/"
                    "blocks/block-00000000-pairs-0016.json"
                ),
                (
                    "artifacts/arena/torch-hero-selfmatch-p1-16pairs-v1/"
                    "state.json"
                ),
            ]
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"Torch hero incumbent Arena gates failed: {failed}")

    record = {
        "schema_version": "torch-hero-incumbent-arena-v1",
        "question": (
            "Can two independently loaded native Torch hero policies play "
            "a role-symmetric one-pass Arena for centered autoresearch?"
        ),
        "source": {
            "state": _relative(state_path),
            "state_sha256": sha256_file(state_path),
            "block": {
                **state["blocks"][0],
                "path": _relative(
                    Path(state["blocks"][0]["path"])
                ),
            },
            "code_commit": contract["code"]["git_commit"],
        },
        "model": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "candidate_id": candidate_id,
            "opponent_id": opponent_id,
            "research_update": candidate["research_update"],
            "jepa_feedback_mode": candidate["model_config"].get(
                "jepa_feedback_mode",
                "none",
            ),
        },
        "arena": {
            "pair_count": aggregate["pair_count"],
            "game_count": aggregate["game_count"],
            "score": aggregate["pentanomial"]["score"],
            "points": aggregate["pentanomial"]["points"],
            "pentanomial_counts": aggregate["pentanomial"]["counts"],
            "wins_draws_losses": [
                candidate_stats["wins"],
                candidate_stats["draws"],
                candidate_stats["losses"],
            ],
            "termination_counts": aggregate["termination_counts"],
            "fault_counts": aggregate["fault_counts"],
            "cap_draw_count": aggregate["cap_draw_count"],
            "candidate_mean_call_ms": (
                1000.0
                * candidate_stats["coverage"]["mean_call_seconds"]
            ),
            "opponent_mean_call_ms": (
                1000.0
                * opponent_stats["coverage"]["mean_call_seconds"]
            ),
            "candidate_positions": candidate_stats[
                "positions_evaluated"
            ],
            "opponent_positions": opponent_stats[
                "positions_evaluated"
            ],
        },
        "output_files": output_files,
        "gates": gates,
        "decision": {
            "status": "open_one_pass_candidate_vs_incumbent_arena",
            "all_preregistered_gates_pass": True,
            "raw_bt4_role": "periodic_absolute_anchor",
            "eight_pass_role": "continuity_diagnostic",
            "model_checkpoint_written": False,
        },
    }
    OUTPUT_PATH.write_text(
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
                "score": record["arena"]["score"],
                "candidate_mean_call_ms": record["arena"][
                    "candidate_mean_call_ms"
                ],
                "opponent_mean_call_ms": record["arena"][
                    "opponent_mean_call_ms"
                ],
                "output": str(OUTPUT_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
