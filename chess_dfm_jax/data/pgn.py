"""PGN-derived sequence utilities for JEPA analysis and probing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    import chess
    import chess.pgn
except ImportError:  # pragma: no cover
    chess = None

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import move_to_policy_index


@dataclass
class PgnPositionRecord:
    board: "chess.Board"
    move: "chess.Move | None"
    next_board: "chess.Board | None"


@dataclass
class TwoPlyTarget:
    next_key: str
    future_fen: str
    future_action_idx: int
    future_action_uci: str


@dataclass
class TwoPlySequenceSample:
    current_board: "chess.Board"
    next_board: "chess.Board"
    future_board: "chess.Board"
    action_uci: str
    future_action_uci: str
    action_idx: int
    future_action_idx: int
    input_format: str = "INPUT_CLASSICAL_112_PLANE"


def board_position_key(board: "chess.Board") -> str:
    ep_square = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    turn = "w" if board.turn == chess.WHITE else "b"
    return f"{board.board_fen()} {turn} {board.castling_xfen()} {ep_square}"


def iter_pgn_position_records(
    pgn_paths: Sequence[str | Path],
    *,
    max_records: int | None = None,
) -> Iterator[PgnPositionRecord]:
    if chess is None:  # pragma: no cover
        raise ImportError("python-chess is required for PGN sequence parsing.")

    yielded = 0
    for path in pgn_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                board = game.board()
                moves = list(game.mainline_moves())
                move = moves[0] if moves else None
                next_board = None
                if move is not None and move in board.legal_moves:
                    next_board = board.copy(stack=False)
                    next_board.push(move)
                yield PgnPositionRecord(board=board, move=move, next_board=next_board)
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return


def iter_two_ply_sequence_samples(
    pgn_paths: Sequence[str | Path],
    *,
    max_records: int | None = None,
) -> Iterator[TwoPlySequenceSample]:
    it = iter_pgn_position_records(pgn_paths, max_records=max_records)
    try:
        first = next(it)
        second = next(it)
    except StopIteration:
        return

    for third in it:
        if first.move is not None and second.move is not None:
            try:
                action_idx = move_to_policy_index(first.move, "lc0_1858")
                future_action_idx = move_to_policy_index(second.move, "lc0_1858")
            except Exception:
                pass
            else:
                yield TwoPlySequenceSample(
                    current_board=first.board,
                    next_board=second.board,
                    future_board=third.board,
                    action_uci=first.move.uci(),
                    future_action_uci=second.move.uci(),
                    action_idx=action_idx,
                    future_action_idx=future_action_idx,
                )
        first, second = second, third


def build_two_ply_lookup(
    pgn_paths: Sequence[str | Path],
    *,
    max_records: int | None = None,
) -> dict[tuple[str, str], TwoPlyTarget]:
    lookup: dict[tuple[str, str], TwoPlyTarget] = {}
    for sample in iter_two_ply_sequence_samples(pgn_paths, max_records=max_records):
        key = (board_position_key(sample.current_board), sample.action_uci)
        if key in lookup:
            continue
        lookup[key] = TwoPlyTarget(
            next_key=board_position_key(sample.next_board),
            future_fen=sample.future_board.fen(),
            future_action_idx=sample.future_action_idx,
            future_action_uci=sample.future_action_uci,
        )
    return lookup


def stack_two_ply_sequence_samples(
    samples: list[TwoPlySequenceSample],
    *,
    input_format: str = "INPUT_CLASSICAL_112_PLANE",
) -> dict[str, object]:
    current_planes = []
    next_planes = []
    future_planes = []
    current_boards = []
    next_boards = []
    future_boards = []
    action_idx = []
    future_action_idx = []
    action_uci = []
    future_action_uci = []

    for sample in samples:
        current_planes.append(encode_board(sample.current_board, [], input_format=input_format).astype(np.float32))
        next_planes.append(encode_board(sample.next_board, [], input_format=input_format).astype(np.float32))
        future_planes.append(encode_board(sample.future_board, [], input_format=input_format).astype(np.float32))
        current_boards.append(sample.current_board)
        next_boards.append(sample.next_board)
        future_boards.append(sample.future_board)
        action_idx.append(sample.action_idx)
        future_action_idx.append(sample.future_action_idx)
        action_uci.append(sample.action_uci)
        future_action_uci.append(sample.future_action_uci)

    batch_size = len(samples)
    return {
        "current_planes": np.stack(current_planes, axis=0).astype(np.float32),
        "next_planes": np.stack(next_planes, axis=0).astype(np.float32),
        "future_planes": np.stack(future_planes, axis=0).astype(np.float32),
        "action_idx": np.asarray(action_idx, dtype=np.int32),
        "future_action_idx": np.asarray(future_action_idx, dtype=np.int32),
        "valid": np.ones((batch_size,), dtype=np.float32),
        "future_valid": np.ones((batch_size,), dtype=np.float32),
        "sequence_probe_valid": np.ones((batch_size,), dtype=np.float32),
        "boards": current_boards,
        "next_boards": next_boards,
        "future_boards": future_boards,
        "action_uci": action_uci,
        "future_action_uci": future_action_uci,
        "input_format": [input_format] * batch_size,
    }


class PgnTwoPlyDataLoader:
    def __init__(
        self,
        pgn_paths: Sequence[str | Path],
        *,
        batch_size: int,
        input_format: str = "INPUT_CLASSICAL_112_PLANE",
        max_records: int | None = None,
    ):
        self.pgn_paths = [str(path) for path in pgn_paths]
        self.batch_size = int(batch_size)
        self.input_format = input_format
        self.max_records = max_records

    def __iter__(self):
        batch: list[TwoPlySequenceSample] = []
        for sample in iter_two_ply_sequence_samples(self.pgn_paths, max_records=self.max_records):
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield stack_two_ply_sequence_samples(batch, input_format=self.input_format)
                batch = []
        if batch:
            yield stack_two_ply_sequence_samples(batch, input_format=self.input_format)


__all__ = [
    "PgnPositionRecord",
    "PgnTwoPlyDataLoader",
    "TwoPlySequenceSample",
    "TwoPlyTarget",
    "board_position_key",
    "build_two_ply_lookup",
    "iter_pgn_position_records",
    "iter_two_ply_sequence_samples",
    "stack_two_ply_sequence_samples",
]
