"""Exact, orientation-aware chess labels for BT4 representation analysis.

The functions in this module deliberately separate facts computable from the
board rules from engine- or dataset-inferred concepts.  They never call an
engine.  Square arrays follow the BT4 token order: a1..h8 when White is to
move, and the vertically mirrored side-to-move frame when Black is to move.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import chess
import chess.pgn
import numpy as np


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
PIECE_CODE_NAMES = (
    "empty",
    "our_pawn",
    "our_knight",
    "our_bishop",
    "our_rook",
    "our_queen",
    "our_king",
    "their_pawn",
    "their_knight",
    "their_bishop",
    "their_rook",
    "their_queen",
    "their_king",
)


def canonical_square(board: chess.Board, square: chess.Square) -> chess.Square:
    """Map an absolute square into the side-to-move BT4 token frame."""

    return square if board.turn == chess.WHITE else chess.square_mirror(square)


def absolute_square(board: chess.Board, token_square: chess.Square) -> chess.Square:
    """Map one BT4 token-frame square back to absolute board coordinates."""

    return token_square if board.turn == chess.WHITE else chess.square_mirror(token_square)


def canonical_move_squares(
    board: chess.Board,
    move: chess.Move,
) -> tuple[chess.Square, chess.Square]:
    if not board.is_legal(move):
        raise ValueError(f"Move {move.uci()} is not legal in {board.fen()}")
    return canonical_square(board, move.from_square), canonical_square(board, move.to_square)


def _piece_code(board: chess.Board, square: chess.Square) -> int:
    piece = board.piece_at(square)
    if piece is None:
        return 0
    relative_offset = 0 if piece.color == board.turn else 6
    return relative_offset + piece.piece_type


def _attack_count(board: chess.Board, color: chess.Color, square: chess.Square) -> int:
    return len(board.attackers(color, square))


def _king_zone(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    king = board.king(color)
    if king is None:
        return chess.SquareSet()
    return chess.SquareSet(chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king])


def square_labels(board: chess.Board) -> dict[str, np.ndarray]:
    """Return exact per-token board, attack, pin, and legal-move labels."""

    if not board.is_valid():
        raise ValueError(f"Cannot label invalid board: {board.fen()}")
    ours = board.turn
    theirs = not ours
    labels: dict[str, np.ndarray] = {
        "piece_code_i8": np.zeros(64, dtype=np.int8),
        "attack_count_ours_u8": np.zeros(64, dtype=np.uint8),
        "attack_count_theirs_u8": np.zeros(64, dtype=np.uint8),
        "occupied_ours_u8": np.zeros(64, dtype=np.uint8),
        "occupied_theirs_u8": np.zeros(64, dtype=np.uint8),
        "pinned_ours_u8": np.zeros(64, dtype=np.uint8),
        "pinned_theirs_u8": np.zeros(64, dtype=np.uint8),
        "attacked_undefended_ours_u8": np.zeros(64, dtype=np.uint8),
        "attacked_undefended_theirs_u8": np.zeros(64, dtype=np.uint8),
        "legal_origin_u8": np.zeros(64, dtype=np.uint8),
        "legal_destination_u8": np.zeros(64, dtype=np.uint8),
        "capture_destination_u8": np.zeros(64, dtype=np.uint8),
        "checking_destination_u8": np.zeros(64, dtype=np.uint8),
        "our_king_zone_u8": np.zeros(64, dtype=np.uint8),
        "their_king_zone_u8": np.zeros(64, dtype=np.uint8),
    }
    our_zone = _king_zone(board, ours)
    their_zone = _king_zone(board, theirs)
    for absolute in chess.SQUARES:
        token = canonical_square(board, absolute)
        piece = board.piece_at(absolute)
        labels["piece_code_i8"][token] = _piece_code(board, absolute)
        our_attackers = _attack_count(board, ours, absolute)
        their_attackers = _attack_count(board, theirs, absolute)
        labels["attack_count_ours_u8"][token] = our_attackers
        labels["attack_count_theirs_u8"][token] = their_attackers
        labels["our_king_zone_u8"][token] = int(absolute in our_zone)
        labels["their_king_zone_u8"][token] = int(absolute in their_zone)
        if piece is None:
            continue
        side_name = "ours" if piece.color == ours else "theirs"
        labels[f"occupied_{side_name}_u8"][token] = 1
        labels[f"pinned_{side_name}_u8"][token] = int(board.is_pinned(piece.color, absolute))
        friendly = our_attackers if piece.color == ours else their_attackers
        enemy = their_attackers if piece.color == ours else our_attackers
        labels[f"attacked_undefended_{side_name}_u8"][token] = int(enemy > 0 and friendly == 0)

    for move in board.legal_moves:
        origin, destination = canonical_move_squares(board, move)
        labels["legal_origin_u8"][origin] = 1
        labels["legal_destination_u8"][destination] = 1
        if board.is_capture(move):
            labels["capture_destination_u8"][destination] = 1
        if board.gives_check(move):
            labels["checking_destination_u8"][destination] = 1
    return labels


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        PIECE_VALUES[piece_type] * len(board.pieces(piece_type, color))
        for piece_type in PIECE_VALUES
    )


def _non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        PIECE_VALUES[piece_type] * len(board.pieces(piece_type, color))
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _passed_pawn_count(board: chess.Board, color: chess.Color) -> int:
    enemy_pawns = board.pieces(chess.PAWN, not color)
    count = 0
    for square in board.pieces(chess.PAWN, color):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        blocking_files = range(max(0, file_index - 1), min(7, file_index + 1) + 1)
        is_passed = True
        for enemy in enemy_pawns:
            enemy_file = chess.square_file(enemy)
            enemy_rank = chess.square_rank(enemy)
            ahead = enemy_rank > rank_index if color == chess.WHITE else enemy_rank < rank_index
            if enemy_file in blocking_files and ahead:
                is_passed = False
                break
        count += int(is_passed)
    return count


def _king_zone_pressure(
    board: chess.Board,
    *,
    attacker: chess.Color,
    king_color: chess.Color,
) -> int:
    return sum(_attack_count(board, attacker, square) for square in _king_zone(board, king_color))


def position_labels(board: chess.Board) -> dict[str, int | float]:
    """Return exact scalar concepts in the side-to-move perspective."""

    if not board.is_valid():
        raise ValueError(f"Cannot label invalid board: {board.fen()}")
    ours = board.turn
    theirs = not ours
    legal_moves = list(board.legal_moves)
    our_material = _material(board, ours)
    their_material = _material(board, theirs)
    our_non_pawn = _non_pawn_material(board, ours)
    their_non_pawn = _non_pawn_material(board, theirs)
    occupied = board.occupied_co[ours] | board.occupied_co[theirs]
    attacked_undefended = 0
    pinned_ours = 0
    pinned_theirs = 0
    for square in chess.scan_forward(occupied):
        piece = board.piece_at(square)
        if piece is None:  # pragma: no cover - occupied invariant
            continue
        friendly = _attack_count(board, piece.color, square)
        enemy = _attack_count(board, not piece.color, square)
        attacked_undefended += int(enemy > 0 and friendly == 0)
        if board.is_pinned(piece.color, square):
            if piece.color == ours:
                pinned_ours += 1
            else:
                pinned_theirs += 1

    center = (chess.D4, chess.E4, chess.D5, chess.E5)
    return {
        "side_to_move_is_white": int(ours == chess.WHITE),
        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
        "is_check": int(board.is_check()),
        "is_checkmate": int(board.is_checkmate()),
        "is_stalemate": int(board.is_stalemate()),
        "legal_move_count": len(legal_moves),
        "capture_move_count": sum(board.is_capture(move) for move in legal_moves),
        "checking_move_count": sum(board.gives_check(move) for move in legal_moves),
        "promotion_move_count": sum(move.promotion is not None for move in legal_moves),
        "our_material": our_material,
        "their_material": their_material,
        "material_balance": our_material - their_material,
        "total_material": our_material + their_material,
        "our_non_pawn_material": our_non_pawn,
        "their_non_pawn_material": their_non_pawn,
        "total_non_pawn_material": our_non_pawn + their_non_pawn,
        "our_passed_pawns": _passed_pawn_count(board, ours),
        "their_passed_pawns": _passed_pawn_count(board, theirs),
        "our_pinned_pieces": pinned_ours,
        "their_pinned_pieces": pinned_theirs,
        "attacked_undefended_pieces": attacked_undefended,
        "our_center_attack_count": sum(_attack_count(board, ours, square) for square in center),
        "their_center_attack_count": sum(_attack_count(board, theirs, square) for square in center),
        "pressure_on_our_king": _king_zone_pressure(
            board,
            attacker=theirs,
            king_color=ours,
        ),
        "pressure_on_their_king": _king_zone_pressure(
            board,
            attacker=ours,
            king_color=theirs,
        ),
    }


def move_labels(board: chess.Board, move: chess.Move) -> dict[str, int]:
    """Return exact properties of one legal move without engine inference."""

    if not board.is_legal(move):
        raise ValueError(f"Move {move.uci()} is not legal in {board.fen()}")
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured_piece = chess.Piece(chess.PAWN, not board.turn)
    origin, destination = canonical_move_squares(board, move)
    next_board = board.copy(stack=False)
    next_board.push(move)
    return {
        "origin_token": origin,
        "destination_token": destination,
        "moving_piece_type": 0 if moving_piece is None else moving_piece.piece_type,
        "captured_piece_type": 0 if captured_piece is None else captured_piece.piece_type,
        "is_capture": int(board.is_capture(move)),
        "gives_check": int(board.gives_check(move)),
        "gives_checkmate": int(next_board.is_checkmate()),
        "is_castling": int(board.is_castling(move)),
        "is_en_passant": int(board.is_en_passant(move)),
        "is_promotion": int(move.promotion is not None),
        "promotion_piece_type": 0 if move.promotion is None else move.promotion,
    }


@dataclass(frozen=True)
class ReconstructedPuzzle:
    """A Searchless/Lichess puzzle root with its exact available history."""

    puzzle_id: str
    board: chess.Board
    history: tuple[chess.Board, ...]
    solution_moves: tuple[chess.Move, ...]


def reconstruct_puzzle(
    *,
    puzzle_id: str,
    pgn: str,
    fen_before_opponent: str,
    moves: str | Sequence[str],
) -> ReconstructedPuzzle:
    """Rebuild a puzzle root and all preceding boards from its PGN.

    Lichess stores the FEN before the opponent's setup move.  The first UCI
    move is applied before presenting the position; the second move is the
    first solution move.  The returned history ends at that presented root.
    """

    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError(f"Puzzle {puzzle_id} has invalid PGN")
    board = game.board()
    history: list[chess.Board] = [board.copy(stack=False)]
    for move in game.mainline_moves():
        if not board.is_legal(move):
            raise ValueError(f"Puzzle {puzzle_id} PGN contains illegal move {move.uci()}")
        board.push(move)
        history.append(board.copy(stack=False))
    if board.fen() != fen_before_opponent:
        raise ValueError(
            f"Puzzle {puzzle_id} PGN/FEN mismatch: {board.fen()} != {fen_before_opponent}"
        )
    move_tokens = moves.split() if isinstance(moves, str) else list(moves)
    if len(move_tokens) < 2:
        raise ValueError(f"Puzzle {puzzle_id} needs a setup and solution move")
    parsed: list[chess.Move] = []
    for token in move_tokens:
        try:
            move = chess.Move.from_uci(token)
        except ValueError as exc:
            raise ValueError(f"Puzzle {puzzle_id} has invalid UCI move {token!r}") from exc
        if not board.is_legal(move):
            raise ValueError(f"Puzzle {puzzle_id} line move {token} is illegal in {board.fen()}")
        parsed.append(move)
        board.push(move)
    root_board = chess.Board(fen_before_opponent)
    setup_move = parsed[0]
    if not root_board.is_legal(setup_move):
        raise ValueError(f"Puzzle {puzzle_id} setup move is not legal")
    root_board.push(setup_move)
    history.append(root_board.copy(stack=False))

    replay = root_board.copy(stack=False)
    solution: list[chess.Move] = []
    for move in parsed[1:]:
        if not replay.is_legal(move):
            raise ValueError(
                f"Puzzle {puzzle_id} solution move {move.uci()} is illegal in {replay.fen()}"
            )
        solution.append(move)
        replay.push(move)
    return ReconstructedPuzzle(
        puzzle_id=puzzle_id,
        board=root_board,
        history=tuple(history),
        solution_moves=tuple(solution),
    )


def solution_token_labels(
    root: chess.Board,
    moves: Iterable[chess.Move],
    *,
    max_plies: int = 7,
) -> dict[str, np.ndarray]:
    """Encode a future line in both native-ply and fixed-root token frames.

    The native frame is useful when pairing a move with activations computed
    after replaying to that ply.  A lookahead probe over *root* activations
    instead needs every future square expressed in the root board's fixed
    orientation.  Keeping both arrays explicit prevents the alternating frame
    bug that otherwise affects every odd future ply.
    """

    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    native_origins = np.full(max_plies, -1, dtype=np.int16)
    native_destinations = np.full(max_plies, -1, dtype=np.int16)
    root_origins = np.full(max_plies, -1, dtype=np.int16)
    root_destinations = np.full(max_plies, -1, dtype=np.int16)
    valid = np.zeros(max_plies, dtype=np.uint8)
    board = root.copy(stack=False)
    for index, move in enumerate(moves):
        if index >= max_plies:
            break
        if not board.is_legal(move):
            raise ValueError(f"Future move {move.uci()} is illegal at ply {index}")
        native_origin, native_destination = canonical_move_squares(board, move)
        native_origins[index] = native_origin
        native_destinations[index] = native_destination
        root_origins[index] = canonical_square(root, move.from_square)
        root_destinations[index] = canonical_square(root, move.to_square)
        valid[index] = 1
        board.push(move)
    return {
        # Backward-compatible names are deliberately documented as native.
        "future_origin_token_i16": native_origins,
        "future_destination_token_i16": native_destinations,
        "future_origin_token_native_i16": native_origins.copy(),
        "future_destination_token_native_i16": native_destinations.copy(),
        "future_origin_token_root_i16": root_origins,
        "future_destination_token_root_i16": root_destinations,
        "future_valid_u8": valid,
    }


def concept_schema() -> dict[str, Any]:
    """Describe label semantics for manifests and downstream audits."""

    return {
        "schema_version": "bt4-exact-chess-concepts-v1",
        "engine_calls": False,
        "orientation": (
            "BT4 side-to-move token frame; black-to-move squares are vertically mirrored"
        ),
        "piece_code_names": list(PIECE_CODE_NAMES),
        "material_values": {
            chess.piece_name(piece_type): value for piece_type, value in PIECE_VALUES.items()
        },
        "attacked_undefended_definition": (
            "occupied square has at least one enemy attacker and zero friendly attackers"
        ),
        "passed_pawn_definition": (
            "no enemy pawn ahead on the same or adjacent file; rule-derived, not engine-derived"
        ),
    }
