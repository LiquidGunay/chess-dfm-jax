import chess
import numpy as np
import pytest

from chess_dfm_jax.encoding import encode_board
from research.hero_history_data import (
    IncrementalHistoryEncoder, initial_board, unique_missing_move,
    source_predecessor, reconstruct_shard, write_npz_once, training_order,
)
from tests.test_hero_history_recovery import payload


@pytest.mark.parametrize("position", [None, 0, 123, 518, 959])
def test_incremental_matches_reference(position):
    board = chess.Board() if position is None else chess.Board.from_chess960_pos(position)
    encoder, history = IncrementalHistoryEncoder(), []
    rng = np.random.default_rng(0)
    for _ in range(70):
        history.append(board.copy(stack=False))
        np.testing.assert_array_equal(encoder.append(board), encode_board(board, history))
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(legal[int(rng.integers(len(legal)))])


def test_repetition_uses_whole_prefix():
    board, history, encoder = chess.Board(), [], IncrementalHistoryEncoder()
    for uci in [None] + "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split():
        if uci:
            board.push_uci(uci)
        history.append(board.copy(stack=False))
        np.testing.assert_array_equal(encoder.append(board), encode_board(board, history))


def test_roots_require_legitimate_standard_or_chess960_start():
    assert not initial_board(chess.STARTING_FEN).chess960
    assert initial_board(chess.Board.from_chess960_pos(123).fen()).chess960
    board = chess.Board(); board.push_uci("e2e4")
    with pytest.raises(ValueError, match="Unverified setup"):
        initial_board(board.fen())


def test_unique_promotion_recovery():
    board = chess.Board("8/4P3/R4Pkp/4r3/6p1/8/p4K2/8 b - - 3 55")
    target = board.copy(); target.push_uci("a2a1q")
    assert unique_missing_move(board, target.fen()).uci() == "a2a1q"
    with pytest.raises(ValueError, match="0 legal"):
        unique_missing_move(board, chess.STARTING_FEN)


@pytest.mark.parametrize("left,right", [(9999,10000),(1,3)])
def test_worker_or_shard_boundary_cannot_be_joined(left,right):
    with pytest.raises(ValueError, match="discontinuity"):
        source_predecessor({"source_uri": f"gs://b/train/chunk_{left:06d}.npz"},
                           {"source_uri": f"gs://b/train/chunk_{right:06d}.npz"})


def test_reconstruction_preserves_labels_and_selected_rows(tmp_path):
    data = payload()
    data.update(plane_codec=np.array("uint8"), source_uri=np.array("gs://b/train/chunk_000000.npz"),
                wdl_targets=np.tile([1.,0.,0.], (8,3,1)))
    path = tmp_path / "chunk_000000.npz"
    write_npz_once(path, data)
    rebuilt, record = reconstruct_shard(path, rows=[1,4,7])
    np.testing.assert_array_equal(rebuilt["wdl_targets"], data["wdl_targets"][[1,4,7]])
    np.testing.assert_array_equal(rebuilt["actions_uci"], data["actions_uci"][[1,4,7]])
    assert np.any(rebuilt["planes_t_u8"][:,13:104])
    assert record["examples"] == 3
    assert not record["recovered_moves"]


def test_exact_short_hero_schedule():
    assert len(training_order()) == 554
    assert training_order()[:3] == [24436,21647,17541]
