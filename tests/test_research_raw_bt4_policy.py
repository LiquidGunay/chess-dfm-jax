from __future__ import annotations

import os
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import research.raw_bt4_policy as raw_policy
from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.encoding import encode_board as real_encode_board
from chess_dfm_jax.nnx_bt4 import make_bt4_model
from chess_dfm_jax.policy import (
    ACTION_CODEC_LC0_CANONICAL_1858,
    encode_action,
    legal_action_mask,
)
from research.arena import (
    CANONICAL_ACTION_CODEC_CAPABILITY,
    action_codec_capability,
)
from research.evaluate_arena import parse_args
from research.local_policy import LocalPolicyError
from research.raw_bt4_policy import LocalBT4Policy


class _FakeBT4:
    dtype = jnp.dtype(jnp.bfloat16)


def _history(*moves: str) -> tuple[chess.Board, ...]:
    board = chess.Board()
    result = [board.copy(stack=False)]
    for move in moves:
        board.push_uci(move)
        result.append(board.copy(stack=False))
    return tuple(result)


def test_raw_bt4_policy_uses_current_only_planes_and_canonical_black_codec(
    monkeypatch: pytest.MonkeyPatch,
):
    white_history = _history()
    black_history = _history("e2e4")
    boards = (white_history[-1], black_history[-1])
    desired_moves = (
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
    )
    desired_indices = np.asarray(
        [
            encode_action(
                move,
                codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
                board=board,
            )
            for board, move in zip(boards, desired_moves, strict=True)
        ],
        dtype=np.int32,
    )
    encoded_histories: list[tuple[object, ...]] = []
    observed: dict[str, np.ndarray] = {}

    def recording_encode(board, history, **kwargs):
        encoded_histories.append(tuple(history))
        return real_encode_board(board, history, **kwargs)

    def fake_infer(model, planes, legal_mask):
        del model
        observed["planes"] = np.asarray(planes)
        observed["legal_mask"] = np.asarray(legal_mask)
        actions = np.resize(desired_indices, planes.shape[0]).astype(np.int32)
        return jnp.asarray(actions), jnp.ones((planes.shape[0],), dtype=jnp.bool_)

    monkeypatch.setattr(raw_policy, "encode_board", recording_encode)
    monkeypatch.setattr(raw_policy, "infer_raw_bt4_actions", fake_infer)
    result = LocalBT4Policy(
        model=_FakeBT4(),
        model_id="raw-bt4",
        inference_batch_size=4,
    ).select_actions(
        boards,
        (white_history, black_history),
    )

    assert result.action_indices.tolist() == desired_indices.tolist()
    assert result.moves == desired_moves
    assert result.diagnostics is None
    assert encoded_histories == [(), ()]
    assert observed["planes"].shape == (4, 112, 8, 8)
    assert observed["legal_mask"].shape == (4, 1858)
    np.testing.assert_array_equal(observed["planes"][2], observed["planes"][0])
    np.testing.assert_array_equal(observed["planes"][3], observed["planes"][0])
    assert int(observed["legal_mask"][0].sum()) == boards[0].legal_moves.count()
    assert int(observed["legal_mask"][1].sum()) == boards[1].legal_moves.count()


def test_raw_bt4_policy_fails_closed_on_nonfinite_policy_logits(
    monkeypatch: pytest.MonkeyPatch,
):
    history = _history()
    board = history[-1]
    action = encode_action(
        chess.Move.from_uci("e2e4"),
        codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
        board=board,
    )

    def fake_infer(model, planes, legal_mask):
        del model, legal_mask
        return (
            jnp.full((planes.shape[0],), action, dtype=jnp.int32),
            jnp.zeros((planes.shape[0],), dtype=jnp.bool_),
        )

    monkeypatch.setattr(raw_policy, "infer_raw_bt4_actions", fake_infer)
    with pytest.raises(LocalPolicyError, match="nonfinite"):
        LocalBT4Policy(
            model=_FakeBT4(),
            model_id="raw-bt4",
        ).select_actions((board,), (history,))


def test_raw_bt4_policy_pins_bfloat16_and_complete_codec_capability():
    class _Float32BT4:
        dtype = jnp.dtype(jnp.float32)

    with pytest.raises(LocalPolicyError, match="bfloat16"):
        LocalBT4Policy(model=_Float32BT4(), model_id="raw-bt4")

    capability = action_codec_capability(ACTION_CODEC_LC0_CANONICAL_1858)
    assert capability == CANONICAL_ACTION_CODEC_CAPABILITY
    assert capability["complete_legal_move_coverage"] is True
    capability["complete_legal_move_coverage"] = False
    assert CANONICAL_ACTION_CODEC_CAPABILITY["complete_legal_move_coverage"] is True


def test_arena_cli_accepts_raw_bt4_as_an_exclusive_opponent():
    parsed = parse_args(
        [
            "--candidate",
            "candidate",
            "--opponent-raw-bt4",
            "--output-dir",
            "output",
        ]
    )
    assert parsed.opponent_raw_bt4 is True
    assert parsed.opponent_checkpoint is None
    assert parsed.opponent_source is False

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--candidate",
                "candidate",
                "--opponent-raw-bt4",
                "--opponent-source",
                "--output-dir",
                "output",
            ]
        )


@pytest.mark.skipif(
    os.environ.get("CHESS_DFM_RUN_RAW_BT4_INTEGRATION") != "1",
    reason="requires the local raw BT4 asset and accelerator",
)
def test_real_raw_bt4_adapter_matches_direct_canonical_policy_argmax():
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models" / "source" / "extracted"
    params = load_mapped_bt4_params(models_dir=str(models_dir))
    model = make_bt4_model(params, dtype=jnp.bfloat16)
    histories = (
        _history(),
        _history("e2e4"),
    )
    boards = tuple(history[-1] for history in histories)
    adapter = LocalBT4Policy(
        model=model,
        model_id="raw-bt4-integration",
        inference_batch_size=2,
    )
    result = adapter.select_actions(boards, histories)

    planes = np.stack(
        [
            real_encode_board(
                board,
                [],
                input_format="INPUT_CLASSICAL_112_PLANE",
            )
            for board in boards
        ]
    )
    masks = np.stack(
        [
            legal_action_mask(
                board,
                codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
            )
            for board in boards
        ]
    )
    direct_actions, finite = raw_policy.infer_raw_bt4_actions(
        model,
        planes,
        masks,
    )
    direct_actions, finite = jax.device_get(
        jax.block_until_ready((direct_actions, finite))
    )
    np.testing.assert_array_equal(result.action_indices, direct_actions)
    assert np.all(finite)
    assert all(board.is_legal(move) for board, move in zip(boards, result.moves, strict=True))
    assert int(masks[0].sum()) == boards[0].legal_moves.count()
    assert int(masks[1].sum()) == boards[1].legal_moves.count()
