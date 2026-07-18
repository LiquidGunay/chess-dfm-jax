from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_dfm_jax.policy import (
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
    ActionCodecError,
    decode_action,
    decode_lc0_canonical_1858,
    encode_action,
    encode_lc0_canonical_1858,
    legal_action_mask,
    legal_mask_lc0_canonical_1858,
    legal_move_mask,
    move_to_policy_index,
    policy_index_to_move,
)


BLACK_TO_MOVE_START = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
)
WHITE_CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
BLACK_CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
WHITE_EN_PASSANT = "7k/8/8/3pP3/8/8/8/7K w - d6 0 1"
BLACK_EN_PASSANT = "7k/8/8/8/3Pp3/8/8/7K b - d3 0 1"
WHITE_PROMOTION = "7k/P7/8/8/8/8/8/7K w - - 0 1"
BLACK_PROMOTION = "7k/8/8/8/8/8/p7/7K b - - 0 1"
WHITE_E_PROMOTION = "7k/4P3/8/8/8/8/8/7K w - - 0 1"
BLACK_E_PROMOTION = "7k/8/8/8/8/8/4p3/7K b - - 0 1"
WHITE_E_CAPTURE_PROMOTION = "3r3k/4P3/8/8/8/8/8/7K w - - 0 1"
BLACK_E_CAPTURE_PROMOTION = "7k/8/8/8/8/8/4p3/3R3K b - - 0 1"


def test_legacy_absolute_codec_keeps_existing_indices_and_boardless_decode():
    assert ACTION_CODEC_LEGACY_ABSOLUTE_1858 == "legacy_absolute_1858"
    assert move_to_policy_index("e2e4", "lc0_1858") == 322
    assert move_to_policy_index("e7e5", "lc0_1858") == 1498
    assert move_to_policy_index("g1f3", "lc0_1858") == 159
    assert move_to_policy_index("g8f6", "lc0_1858") == 1755
    assert policy_index_to_move(1498, "lc0_1858") == chess.Move.from_uci("e7e5")

    black_board = chess.Board(BLACK_TO_MOVE_START)
    mask = legal_move_mask(black_board, "lc0_1858")
    assert mask[1498]
    assert not mask[322]
    assert encode_action(
        "e7e5",
        codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ) == 1498
    assert decode_action(
        1498,
        codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ) == chess.Move.from_uci("e7e5")


def test_lc0_canonical_codec_mirrors_black_to_white_action_indices():
    assert ACTION_CODEC_LC0_CANONICAL_1858 == "lc0_canonical_1858"
    white_board = chess.Board()
    black_board = chess.Board(BLACK_TO_MOVE_START)

    golden_pairs = (
        ("e2e4", white_board, "e7e5", black_board, 322),
        ("g1f3", white_board, "g8f6", black_board, 159),
    )
    for white_uci, white, black_uci, black, expected_index in golden_pairs:
        assert encode_lc0_canonical_1858(white, white_uci) == expected_index
        assert encode_lc0_canonical_1858(black, black_uci) == expected_index
        assert decode_lc0_canonical_1858(white, expected_index).uci() == white_uci
        assert decode_lc0_canonical_1858(black, expected_index).uci() == black_uci

    canonical_mask = legal_mask_lc0_canonical_1858(black_board)
    assert canonical_mask.shape == (ACTION_VOCAB_SIZE,)
    assert canonical_mask.dtype == np.bool_
    assert canonical_mask[322]
    assert canonical_mask[159]
    assert not canonical_mask[1498]
    assert int(canonical_mask.sum()) == black_board.legal_moves.count() == 20


@pytest.mark.parametrize(
    ("white_fen", "white_uci", "black_fen", "black_uci", "expected_index"),
    (
        (WHITE_CASTLING, "e1g1", BLACK_CASTLING, "e8g8", 102),
        (WHITE_EN_PASSANT, "e5d6", BLACK_EN_PASSANT, "e4d3", 1041),
    ),
)
def test_lc0_canonical_special_move_golden_color_pairs(
    white_fen: str,
    white_uci: str,
    black_fen: str,
    black_uci: str,
    expected_index: int,
):
    white_board = chess.Board(white_fen)
    black_board = chess.Board(black_fen)
    assert encode_lc0_canonical_1858(white_board, white_uci) == expected_index
    assert encode_lc0_canonical_1858(black_board, black_uci) == expected_index
    assert decode_lc0_canonical_1858(white_board, expected_index).uci() == white_uci
    assert decode_lc0_canonical_1858(black_board, expected_index).uci() == black_uci


@pytest.mark.parametrize(
    ("suffix", "expected_index"),
    (("n", 1401), ("q", 1792), ("r", 1793), ("b", 1794)),
)
def test_lc0_canonical_promotion_slots_are_color_symmetric(
    suffix: str,
    expected_index: int,
):
    white_board = chess.Board(WHITE_PROMOTION)
    black_board = chess.Board(BLACK_PROMOTION)
    white_move = f"a7a8{suffix}"
    black_move = f"a2a1{suffix}"

    assert encode_lc0_canonical_1858(white_board, white_move) == expected_index
    assert encode_lc0_canonical_1858(black_board, black_move) == expected_index
    assert decode_lc0_canonical_1858(white_board, expected_index).uci() == white_move
    assert decode_lc0_canonical_1858(black_board, expected_index).uci() == black_move


@pytest.mark.parametrize(
    ("white_uci", "black_uci", "expected_index"),
    (
        ("e7e8n", "e2e1n", 1515),
        ("e7e8q", "e2e1q", 1828),
        ("e7e8r", "e2e1r", 1829),
        ("e7e8b", "e2e1b", 1830),
    ),
)
def test_lc0_canonical_e_file_straight_promotion_oracle(
    white_uci: str,
    black_uci: str,
    expected_index: int,
):
    white_board = chess.Board(WHITE_E_PROMOTION)
    black_board = chess.Board(BLACK_E_PROMOTION)
    assert encode_lc0_canonical_1858(white_board, white_uci) == expected_index
    assert encode_lc0_canonical_1858(black_board, black_uci) == expected_index
    assert decode_lc0_canonical_1858(white_board, expected_index).uci() == white_uci
    assert decode_lc0_canonical_1858(black_board, expected_index).uci() == black_uci


@pytest.mark.parametrize(
    ("white_uci", "black_uci", "expected_index"),
    (
        ("e7d8n", "e2d1n", 1514),
        ("e7d8q", "e2d1q", 1825),
        ("e7d8r", "e2d1r", 1826),
        ("e7d8b", "e2d1b", 1827),
    ),
)
def test_lc0_canonical_e_file_capture_promotion_oracle(
    white_uci: str,
    black_uci: str,
    expected_index: int,
):
    white_board = chess.Board(WHITE_E_CAPTURE_PROMOTION)
    black_board = chess.Board(BLACK_E_CAPTURE_PROMOTION)
    assert encode_lc0_canonical_1858(white_board, white_uci) == expected_index
    assert encode_lc0_canonical_1858(black_board, black_uci) == expected_index
    assert decode_lc0_canonical_1858(white_board, expected_index).uci() == white_uci
    assert decode_lc0_canonical_1858(black_board, expected_index).uci() == black_uci


@pytest.mark.parametrize("fen", (BLACK_TO_MOVE_START, WHITE_PROMOTION, BLACK_PROMOTION))
def test_lc0_canonical_legal_mask_round_trips_every_legal_move(fen: str):
    board = chess.Board(fen)
    mask = legal_action_mask(
        board,
        codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
    )
    legal_moves = set(board.legal_moves)
    decoded_moves = {
        decode_action(
            int(index),
            board=board,
            codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
        )
        for index in np.flatnonzero(mask)
    }

    assert int(mask.sum()) == len(legal_moves)
    assert decoded_moves == legal_moves
    for move in legal_moves:
        index = encode_action(
            move,
            board=board,
            codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
        )
        assert mask[index]
        assert decode_lc0_canonical_1858(board, index) == move


def test_lc0_canonical_codec_is_invariant_under_board_mirror():
    board = chess.Board(WHITE_CASTLING)
    mirrored_board = board.mirror()
    np.testing.assert_array_equal(
        legal_mask_lc0_canonical_1858(board),
        legal_mask_lc0_canonical_1858(mirrored_board),
    )
    for move in board.legal_moves:
        mirrored_move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
        assert mirrored_board.is_legal(mirrored_move)
        assert encode_lc0_canonical_1858(
            board,
            move,
        ) == encode_lc0_canonical_1858(mirrored_board, mirrored_move)


def test_lc0_canonical_codec_fails_closed_without_unique_legal_move():
    black_board = chess.Board(BLACK_TO_MOVE_START)
    with pytest.raises(ActionCodecError, match="illegal move"):
        encode_lc0_canonical_1858(black_board, "e7e4")
    with pytest.raises(ActionCodecError, match="0 legal decodes"):
        decode_lc0_canonical_1858(black_board, 1498)
    with pytest.raises(ActionCodecError, match="out of range"):
        decode_lc0_canonical_1858(black_board, ACTION_VOCAB_SIZE)
    with pytest.raises(ActionCodecError, match="requires the current board"):
        encode_action("e7e5", codec_id=ACTION_CODEC_LC0_CANONICAL_1858)
    with pytest.raises(ActionCodecError, match="defined only for"):
        encode_lc0_canonical_1858(
            black_board,
            "e7e5",
            input_format="INPUT_112_WITH_CANONICALIZATION",
        )
