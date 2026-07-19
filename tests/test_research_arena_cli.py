from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_dfm_jax.policy import (
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    legal_action_mask,
)
from research.arena import build_opening_pool
from research.evaluate_arena import (
    ARENA_RUN_SCHEMA,
    FROZEN_TIERS,
    _resolved_run_options,
    load_run_state,
    run_blocks,
)
from research.play_arena import (
    OPENING_HISTORY_CONVENTION,
    OPENING_HISTORY_SCHEMA,
    histories_for_pairs,
    load_opening_history_sidecar,
)
from research.prepare import REPO_ROOT


@pytest.fixture
def workspace_tmp() -> Path:
    root = REPO_ROOT / ".local" / "tmp" / "arena-cli-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _history(*moves: str) -> list[str]:
    board = chess.Board()
    result = [board.fen(en_passant="legal")]
    for move in moves:
        board.push_uci(move)
        result.append(board.fen(en_passant="legal"))
    return result


def _fixture_assets(root: Path):
    histories = [
        _history("e2e4", "e7e5"),
        _history("d2d4", "d7d5"),
    ]
    val_dir = root / "val"
    val_dir.mkdir()
    shard = val_dir / "chunk_000000.npz"
    np.savez(
        shard,
        schema_version=np.asarray("trajectory-v3"),
        fen_t=np.asarray([values[-1] for values in histories]),
        ply=np.asarray([2, 2], dtype=np.int32),
        source_uri=np.asarray("fixture://arena"),
    )
    pool = build_opening_pool(
        [shard],
        seed=7,
        count=2,
        opening_ply=2,
    )
    by_fen = {values[-1]: values for values in histories}
    sidecar = {
        "schema_version": OPENING_HISTORY_SCHEMA,
        "history_convention": OPENING_HISTORY_CONVENTION,
        "pool_sha256": pool["pool_sha256"],
        "entries": [
            {
                "opening_index": index,
                "fen": opening["fen"],
                "history_fens": by_fen[opening["fen"]],
            }
            for index, opening in enumerate(pool["openings"])
        ],
    }
    sidecar["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(sidecar)).hexdigest()
    sidecar_path = root / "histories.json"
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True),
        encoding="utf-8",
    )
    loaded = load_opening_history_sidecar(
        sidecar_path,
        opening_pool=pool,
        expected_manifest_sha256=sidecar["manifest_sha256"],
    )
    return pool, sidecar_path, sidecar, loaded


class _Selection:
    def __init__(self, action_indices):
        self.action_indices = np.asarray(action_indices, dtype=np.int32)


class _FirstLegalPolicy:
    action_codec_id = ACTION_CODEC_LEGACY_ABSOLUTE_1858

    def __init__(self, model_id: str):
        self.model_id = model_id

    def select_actions(self, boards, histories):
        del histories
        return _Selection(
            [
                int(
                    np.flatnonzero(
                        legal_action_mask(
                            board,
                            codec_id=self.action_codec_id,
                        )
                    )[0]
                )
                for board in boards
            ]
        )


def _contract(*, promotion: bool = False):
    return {
        "relative_elo_scope": "checkpoint_pool_relative_only",
        "absolute_elo_claim": False,
        "tier": {
            "name": "promotion" if promotion else "correctness",
            "promotion_eligible": promotion,
        },
        "models": {
            "candidate": {"model_id": "candidate"},
            "opponent": {"model_id": "opponent"},
        },
        "run": {
            "pair_count": 1 if promotion else 2,
            "block_pairs": 1,
            "additional_ply_cap": 1,
            "policy_timeout_seconds": 30.0,
            "policy_batch_size_cap": 2,
        },
    }


def test_history_sidecar_is_digest_checked_aligned_and_replayed(
    workspace_tmp: Path,
):
    pool, sidecar_path, sidecar, loaded = _fixture_assets(workspace_tmp)
    assert loaded.pool_sha256 == pool["pool_sha256"]
    assert loaded.manifest_sha256 == sidecar["manifest_sha256"]
    assert len(loaded.histories_by_opening_index) == 2

    from research.arena import make_color_reversed_pairs

    second = make_color_reversed_pairs(
        [pool["openings"][1]["fen"]],
        model_a="candidate",
        model_b="opponent",
        start_index=1,
    )
    bound = histories_for_pairs(second, loaded)
    assert tuple(bound.values())[0].fens[-1] == pool["openings"][1]["fen"]

    tampered = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tampered["entries"][0]["history_fens"][1] = tampered["entries"][1]["history_fens"][1]
    sidecar_path.write_text(
        json.dumps(tampered, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_opening_history_sidecar(
            sidecar_path,
            opening_pool=pool,
            expected_manifest_sha256=sidecar["manifest_sha256"],
        )


def test_resumable_blocks_persist_relative_stats_and_verify_immutable_files(
    workspace_tmp: Path,
):
    pool, _path, _sidecar, loaded = _fixture_assets(workspace_tmp)
    output_dir = workspace_tmp / "run"
    contract = _contract()
    candidate = _FirstLegalPolicy("candidate")
    opponent = _FirstLegalPolicy("opponent")

    state = run_blocks(
        output_dir=output_dir,
        contract=contract,
        opening_pool=pool,
        loaded_histories=loaded,
        candidate_policy=candidate,
        opponent_policy=opponent,
        resume=False,
        session_setup={"kind": "fixture"},
    )

    assert state["schema_version"] == ARENA_RUN_SCHEMA
    assert state["status"] == "complete"
    assert state["absolute_elo_claim"] is False
    assert state["candidate_promoted"] is False
    assert state["aggregate"]["pair_count"] == 2
    assert state["aggregate"]["game_count"] == 4
    assert state["aggregate"]["cap_draw_rate"] == 1.0
    assert len(state["blocks"]) == 2
    assert state["normalized_promotion_state"] is None
    assert (
        load_run_state(
            output_dir,
            expected_contract=contract,
        )
        == state
    )

    resumed = run_blocks(
        output_dir=output_dir,
        contract=contract,
        opening_pool=pool,
        loaded_histories=loaded,
        candidate_policy=candidate,
        opponent_policy=opponent,
        resume=True,
        session_setup={"kind": "unused"},
    )
    assert resumed == state

    block_path = Path(state["blocks"][0]["path"])
    block_path.write_text(
        block_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_run_state(
            output_dir,
            expected_contract=contract,
        )


def test_promotion_state_advances_at_one_complete_pair_boundary(
    workspace_tmp: Path,
):
    pool, _path, _sidecar, loaded = _fixture_assets(workspace_tmp)
    contract = _contract(promotion=True)
    state = run_blocks(
        output_dir=workspace_tmp / "promotion-run",
        contract=contract,
        opening_pool=pool,
        loaded_histories=loaded,
        candidate_policy=_FirstLegalPolicy("candidate"),
        opponent_policy=_FirstLegalPolicy("opponent"),
        resume=False,
        session_setup={"kind": "fixture"},
    )

    promotion = state["normalized_promotion_state"]
    assert promotion["pair_count"] == 1
    assert promotion["decision"] == "continue"
    assert promotion["update_unit"] == "completed_color_reversed_pair"
    assert state["candidate_promoted"] is False


def test_pinned_promotion_tier_fails_before_running_terminal_history():
    args = type(
        "Args",
        (),
        {
            "pair_count": None,
            "block_pairs": None,
            "additional_ply_cap": 128,
        },
    )()
    with pytest.raises(ValueError, match="entry 1245"):
        _resolved_run_options(args, FROZEN_TIERS["promotion"])
