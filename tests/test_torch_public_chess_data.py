from __future__ import annotations

import struct
from pathlib import Path

import chess
import numpy as np
import pytest

from research.interpretability.chess_concepts import (
    PIECE_CODE_NAMES,
    absolute_square,
    canonical_move_squares,
    canonical_square,
    move_labels,
    position_labels,
    reconstruct_puzzle,
    solution_token_labels,
    square_labels,
)
from research.interpretability.chessbench import (
    ActionValueRecord,
    BagFile,
    BehavioralCloningRecord,
    StateValueRecord,
    _game_id,
    decode_action_value_record,
    decode_behavioral_cloning_record,
    decode_state_value_record,
)


def _varint(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(result)


def _nested(value: str) -> bytes:
    payload = value.encode("utf-8")
    return _varint(len(payload)) + payload


def _write_bag(path: Path, records: list[bytes]) -> None:
    data = b"".join(records)
    endpoints = []
    offset = 0
    for record in records:
        offset += len(record)
        endpoints.append(offset)
    path.write_bytes(data + b"".join(struct.pack("<q", value) for value in endpoints))


def test_dependency_free_beam_record_decoders_match_known_layout() -> None:
    fen = "5rk1/2r4p/3Q2pn/8/3P4/8/PPq5/K2R3R b - - 0 29"
    action_payload = _nested(fen) + _nested("c2c1") + struct.pack(">d", 0.0723136059066038)
    assert action_payload.hex() == (
        "2d35726b312f327234702f335132706e2f382f3350342f382f505071352f"
        "4b325233522062202d202d203020323904633263313fb28324fc6cb8dc"
    )
    assert decode_action_value_record(action_payload) == ActionValueRecord(
        fen=fen,
        move="c2c1",
        win_probability=0.0723136059066038,
    )

    state_payload = _nested(fen) + struct.pack(">d", 0.75)
    assert decode_state_value_record(state_payload) == StateValueRecord(fen, 0.75)
    cloning_payload = _nested(fen) + b"c2c1"
    assert decode_behavioral_cloning_record(cloning_payload) == BehavioralCloningRecord(
        fen,
        "c2c1",
    )

    with pytest.raises(ValueError, match="exactly one"):
        decode_action_value_record(action_payload + b"x")
    with pytest.raises(ValueError, match="empty move"):
        decode_behavioral_cloning_record(_nested(fen))


def test_bag_file_validates_footer_and_supports_random_access(tmp_path: Path) -> None:
    records = [b"first", b"", b"third"]
    path = tmp_path / "tiny.bag"
    _write_bag(path, records)
    with BagFile(path) as bag:
        assert len(bag) == 3
        assert list(bag) == records
        assert bag[0] == b"first"
        assert bag[-1] == b"third"
        with pytest.raises(IndexError):
            _ = bag[3]

    invalid = tmp_path / "invalid.bag"
    invalid.write_bytes(b"not-a-valid-bag")
    with pytest.raises(ValueError, match="footer"):
        BagFile(invalid)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://lichess.org/yyznGmXs/black#34", "yyznGmXs"),
        ("https://lichess.org/abcd1234/white#19", "abcd1234"),
        ("https://lichess.org/abcd1234#42", "abcd1234"),
        ("", "puzzle:00abc"),
    ],
)
def test_lichess_game_id_ignores_color_and_ply_fragments(url: str, expected: str) -> None:
    assert _game_id(url, "00abc") == expected


def test_square_labels_follow_black_side_to_move_token_orientation() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    labels = square_labels(board)
    black_king_token = chess.E1
    white_pawn_token = chess.E7

    assert canonical_square(board, chess.E8) == black_king_token
    assert canonical_square(board, chess.E2) == white_pawn_token
    assert absolute_square(board, white_pawn_token) == chess.E2
    assert PIECE_CODE_NAMES[labels["piece_code_i8"][black_king_token]] == "our_king"
    assert PIECE_CODE_NAMES[labels["piece_code_i8"][white_pawn_token]] == "their_pawn"
    assert labels["occupied_ours_u8"][black_king_token] == 1
    assert labels["occupied_theirs_u8"][white_pawn_token] == 1


def test_exact_position_and_move_labels_are_rule_derived() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    move = chess.Move.from_uci("e2e4")
    scalar = position_labels(board)
    action = move_labels(board, move)

    assert scalar["material_balance"] == 1
    assert scalar["our_passed_pawns"] == 1
    assert scalar["legal_move_count"] == board.legal_moves.count()
    assert action["origin_token"] == chess.E2
    assert action["destination_token"] == chess.E4
    assert action["moving_piece_type"] == chess.PAWN
    assert action["is_capture"] == 0
    assert canonical_move_squares(board, move) == (chess.E2, chess.E4)


def test_puzzle_reconstruction_preserves_available_history_and_future_frames() -> None:
    puzzle = reconstruct_puzzle(
        puzzle_id="unit",
        pgn="1. e4 e5",
        fen_before_opponent=("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        moves="g1f3 b8c6 f1b5",
    )
    assert puzzle.board.fen() == ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    assert len(puzzle.history) == 4
    assert [move.uci() for move in puzzle.solution_moves] == ["b8c6", "f1b5"]

    future = solution_token_labels(puzzle.board, puzzle.solution_moves, max_plies=4)
    np.testing.assert_array_equal(future["future_valid_u8"], [1, 1, 0, 0])
    # Black's b8->c6 is vertically mirrored; White's next move uses absolute squares.
    assert future["future_origin_token_i16"][0] == chess.B1
    assert future["future_destination_token_i16"][0] == chess.C3
    assert future["future_origin_token_i16"][1] == chess.F1
    assert future["future_destination_token_i16"][1] == chess.B5
    # The fixed root is Black-to-move, so every future square remains mirrored.
    assert future["future_origin_token_root_i16"][0] == chess.B1
    assert future["future_destination_token_root_i16"][0] == chess.C3
    assert future["future_origin_token_root_i16"][1] == chess.F8
    assert future["future_destination_token_root_i16"][1] == chess.B4
    np.testing.assert_array_equal(
        future["future_origin_token_native_i16"],
        future["future_origin_token_i16"],
    )

    with pytest.raises(ValueError, match="PGN/FEN mismatch"):
        reconstruct_puzzle(
            puzzle_id="bad",
            pgn="1. e4",
            fen_before_opponent=chess.STARTING_FEN,
            moves="e7e5 g1f3",
        )
