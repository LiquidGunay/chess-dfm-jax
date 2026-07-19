from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import chess
import numpy as np
import pytest

from research.arena import build_opening_pool
from research.prepare import REPO_ROOT


_SCRIPT = (
    REPO_ROOT
    / "research"
    / "prepare"
    / "support"
    / "regenerate_promotion_assets.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "regenerate_promotion_assets",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
promotion_assets = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = promotion_assets
_SPEC.loader.exec_module(promotion_assets)


@pytest.fixture
def workspace_tmp() -> Path:
    root = REPO_ROOT / ".local" / "tmp" / "promotion-assets-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def _trajectory(*moves: str) -> tuple[list[str], list[str]]:
    board = chess.Board()
    fens = [board.fen(en_passant="legal")]
    actions: list[str] = []
    for move_uci in moves:
        actions.append(move_uci)
        board.push_uci(move_uci)
        fens.append(board.fen(en_passant="legal"))
    actions.append("0000")
    return fens, actions


def _write_shard(
    path: Path,
    *,
    plies: list[int],
    fens: list[str],
    actions: list[str],
    source_uri: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    action_rows = np.full((len(plies), 1), "0000", dtype="<U5")
    action_rows[:, 0] = np.asarray(actions)
    np.savez(
        path,
        schema_version=np.asarray("trajectory-v3"),
        ply=np.asarray(plies, dtype=np.int32),
        fen_t=np.asarray(fens),
        actions_uci=action_rows,
        source_uri=np.asarray(source_uri),
    )


def test_reader_reconstructs_across_provenance_contiguous_shards(
    workspace_tmp: Path,
):
    moves = (
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
        "a7a6",
        "b5a4",
        "g8f6",
        "e1g1",
        "f8e7",
        "f1e1",
        "b7b5",
    )
    fens, actions = _trajectory(*moves)
    split = workspace_tmp / "test"
    _write_shard(
        split / "chunk_000000.npz",
        plies=list(range(8)),
        fens=fens[:8],
        actions=actions[:8],
        source_uri="fixture://test/chunk_000123.npz",
    )
    _write_shard(
        split / "chunk_000001.npz",
        plies=list(range(8, 13)),
        fens=fens[8:],
        actions=actions[8:],
        source_uri="fixture://test/chunk_000124.npz",
    )
    reader = promotion_assets.TrajectoryHistoryReader(split)
    opening = {
        "fen": fens[-1],
        "ply": 12,
        "source_shard": "test/chunk_000001.npz",
        "source_row": 4,
    }

    assert reader.reconstruct(opening) == tuple(fens)


def test_reader_rejects_exact_threefold_root_that_fen_alone_hides(
    workspace_tmp: Path,
):
    moves = (
        "d2d4",
        "g8f6",
        "c2c3",
        "e7e6",
        "c1f4",
        "f6h5",
        "f4d2",
        "h5f6",
        "d2f4",
        "f6h5",
        "f4d2",
        "h5f6",
    )
    fens, actions = _trajectory(*moves)
    split = workspace_tmp / "test"
    _write_shard(
        split / "chunk_000000.npz",
        plies=list(range(13)),
        fens=fens,
        actions=actions,
        source_uri="fixture://test/chunk_000000.npz",
    )
    reader = promotion_assets.TrajectoryHistoryReader(split)
    opening = {
        "fen": fens[-1],
        "ply": 12,
        "source_shard": "test/chunk_000000.npz",
        "source_row": 12,
    }

    with pytest.raises(
        ValueError,
        match="history_root_terminal_threefold_repetition",
    ):
        reader.reconstruct(opening)
    assert not chess.Board(fens[-1]).is_game_over(claim_draw=True)


def test_sidecar_uses_runtime_canonical_manifest_and_pool_alignment(
    workspace_tmp: Path,
):
    first = _trajectory(
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
        "a7a6",
        "b5a4",
        "g8f6",
        "e1g1",
        "f8e7",
        "f1e1",
        "b7b5",
    )
    second = _trajectory(
        "d2d4",
        "d7d5",
        "c2c4",
        "e7e6",
        "b1c3",
        "g8f6",
        "c1g5",
        "f8e7",
        "e2e3",
        "e8g8",
        "g1f3",
        "h7h6",
    )
    split = workspace_tmp / "test"
    plies = list(range(13)) + list(range(13))
    fens = first[0] + second[0]
    actions = first[1] + second[1]
    shard = split / "chunk_000000.npz"
    _write_shard(
        shard,
        plies=plies,
        fens=fens,
        actions=actions,
        source_uri="fixture://test/chunk_000000.npz",
    )
    pool = build_opening_pool(
        [shard],
        seed=7,
        count=2,
        opening_ply=12,
        heldout_split="test",
    )
    reader = promotion_assets.TrajectoryHistoryReader(split)
    histories, rejections = promotion_assets.audit_opening_pool(
        pool,
        reader=reader,
    )
    assert rejections == []

    sidecar = promotion_assets.make_history_sidecar(pool, histories)
    digest_payload = dict(sidecar)
    observed = digest_payload.pop("manifest_sha256")
    assert observed == promotion_assets.json_sha256(digest_payload)
    assert sidecar["pool_sha256"] == pool["pool_sha256"]
    assert [entry["fen"] for entry in sidecar["entries"]] == [
        opening["fen"] for opening in pool["openings"]
    ]
    assert json.loads(json.dumps(sidecar, allow_nan=False)) == sidecar


def test_regeneration_requires_named_split_directories(workspace_tmp: Path):
    wrong_split = workspace_tmp / "validation"
    wrong_split.mkdir()

    with pytest.raises(ValueError, match="'val' split directory"):
        promotion_assets._require_split_directory(
            wrong_split,
            expected_split="val",
        )


@pytest.mark.parametrize(
    "names",
    [
        ("../pool.json", "histories.json", "contract.json"),
        ("pool.txt", "histories.json", "contract.json"),
        ("pool.json", "pool.json", "contract.json"),
    ],
)
def test_regeneration_rejects_unsafe_or_colliding_output_names(names):
    with pytest.raises(
        ValueError,
        match="distinct .json basenames|must be distinct",
    ):
        promotion_assets._validate_output_names(*names)
