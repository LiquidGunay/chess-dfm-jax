"""Strict public-data bridge for ChessBench and Lichess interpretability labels.

This module intentionally avoids the upstream JAX/Beam dependency stack.  It
implements only the documented Bag container and the three Apache Beam coder
layouts used by Searchless Chess, then validates every decoded chess object
against python-chess before retaining it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import mmap
import os
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import chess
import numpy as np
import zstandard

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import (
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_VOCAB_SIZE,
    LC0_CANONICAL_1858_INPUT_FORMAT,
    encode_action,
    legal_action_mask,
    move_to_policy_index,
)
from research.interpretability.artifacts import (
    canonical_json_bytes,
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
)
from research.interpretability.chess_concepts import (
    concept_schema,
    move_labels,
    position_labels,
    reconstruct_puzzle,
    solution_token_labels,
    square_labels,
)
from research.interpretability.pilot_corpus import write_deterministic_npz


_REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CORPUS_SCHEMA = "bt4-public-interpretability-corpus-v2"
SEARCHLESS_COMMIT = "90ae0e6b121673fc3079aaeffa047580bb600c0a"
SEARCHLESS_BASE_URL = "https://storage.googleapis.com/searchless_chess/data"
LICHESS_DUMP_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICHESS_DUMP_DATE = "2026-08-02"
MAX_CHESS_LEGAL_MOVES = 218
LEGAL_PAD = np.iinfo(np.uint16).max
DEFAULT_EXTERNAL_DIR = _REPO_ROOT / "data/external/chessbench"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "research/eval/interpretability_public_v2"

SOURCE_IDENTITIES: dict[str, dict[str, Any]] = {
    "searchless_puzzles": {
        "relative_path": "puzzles.csv",
        "url": f"{SEARCHLESS_BASE_URL}/puzzles.csv",
        "size_bytes": 4_705_735,
        "sha256": "099d54e681a07b9e82b4462aaf5436a6e833edc6f56b92ea7c2d2413f7efbe48",
        "md5": "25c0ed4842bd5ff5e507f00ca29b4d53",
    },
    "action_value_test": {
        "relative_path": "test/action_value_data.bag",
        "url": f"{SEARCHLESS_BASE_URL}/test/action_value_data.bag",
        "size_bytes": 148_297_469,
        "sha256": "5f73aac8f60e31734cdbf276ba3fca8d5ba5cb6171ba600e31af4f36327986b0",
        "md5": "bfe6a98f096b21ed836d2f4da157f704",
    },
    "state_value_test": {
        "relative_path": "test/state_value_data.bag",
        "url": f"{SEARCHLESS_BASE_URL}/test/state_value_data.bag",
        "size_bytes": 4_619_413,
        "sha256": "2b132a207868cfb49eb1033b962a7688ab3987ce6d137142d2faf1a8cf312480",
        "md5": "4f79a1e0183b7d6b83b37dea365cb38a",
    },
    "behavioral_cloning_test": {
        "relative_path": "test/behavioral_cloning_data.bag",
        "url": f"{SEARCHLESS_BASE_URL}/test/behavioral_cloning_data.bag",
        "size_bytes": 4_351_421,
        "sha256": "fc1d27fc98baae3a7f15fee85439b499f4a6f1815dffde26d047a77f02cc50b1",
        "md5": "5b311d2a8096f2bea3aa81c5278f36d1",
    },
    "lichess_puzzle_dump": {
        "relative_path": "lichess_db_puzzle_20260802.csv.zst",
        "url": LICHESS_DUMP_URL,
        "size_bytes": 304_384_407,
        "sha256": "a0ea9129c6b6434dfb34a9ac4ec660c9cfff22b2de465e01854f018fc847f073",
        "license": "CC0",
    },
}


def _read_unsigned_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload):
            raise ValueError("Truncated Beam varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift >= 64:
            raise ValueError("Beam varint exceeds 64 bits")


def _read_nested_utf8(payload: bytes, offset: int, *, label: str) -> tuple[str, int]:
    length, offset = _read_unsigned_varint(payload, offset)
    end = offset + length
    if end > len(payload):
        raise ValueError(f"Truncated nested {label}")
    try:
        value = payload[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 in nested {label}") from exc
    return value, end


def _decode_float64(payload: bytes, offset: int, *, label: str) -> float:
    if len(payload) - offset != 8:
        raise ValueError(f"{label} must end in exactly one Beam FloatCoder value")
    value = struct.unpack(">d", payload[offset:])[0]
    if not math.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite float")
    return value


@dataclass(frozen=True)
class ActionValueRecord:
    fen: str
    move: str
    win_probability: float


@dataclass(frozen=True)
class StateValueRecord:
    fen: str
    win_probability: float


@dataclass(frozen=True)
class BehavioralCloningRecord:
    fen: str
    move: str


def decode_action_value_record(payload: bytes) -> ActionValueRecord:
    fen, offset = _read_nested_utf8(payload, 0, label="FEN")
    move, offset = _read_nested_utf8(payload, offset, label="move")
    value = _decode_float64(payload, offset, label="action-value record")
    return ActionValueRecord(fen, move, value)


def decode_state_value_record(payload: bytes) -> StateValueRecord:
    fen, offset = _read_nested_utf8(payload, 0, label="FEN")
    value = _decode_float64(payload, offset, label="state-value record")
    return StateValueRecord(fen, value)


def decode_behavioral_cloning_record(payload: bytes) -> BehavioralCloningRecord:
    fen, offset = _read_nested_utf8(payload, 0, label="FEN")
    if offset == len(payload):
        raise ValueError("Behavioral-cloning record has an empty move")
    try:
        move = payload[offset:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid UTF-8 in behavioral-cloning move") from exc
    return BehavioralCloningRecord(fen, move)


class BagFile:
    """Read-only, mmap-backed reader for the Searchless Chess Bag container."""

    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        self._handle: BinaryIO = self.path.open("rb")
        try:
            self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._handle.close()
            raise
        size = len(self._mmap)
        if size < 16:
            self.close()
            raise ValueError(f"Bag file is too small: {self.path}")
        self._index_start = struct.unpack("<Q", self._mmap[-8:])[0]
        index_size = size - self._index_start
        if not 0 < self._index_start < size or index_size % 8:
            self.close()
            raise ValueError(f"Invalid Bag footer in {self.path}")
        self._record_count = index_size // 8
        limits = np.frombuffer(
            self._mmap,
            dtype="<i8",
            count=self._record_count,
            offset=self._index_start,
        )
        valid = (
            limits.size > 0
            and int(limits[-1]) == self._index_start
            and int(limits[0]) >= 0
            and bool(np.all(np.diff(limits) >= 0))
        )
        del limits
        if not valid:
            self.close()
            raise ValueError(f"Bag record limits are invalid in {self.path}")

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int) -> bytes:
        if index < 0:
            index += self._record_count
        if not 0 <= index < self._record_count:
            raise IndexError("Bag record index out of range")
        limit_offset = self._index_start + 8 * index
        end = struct.unpack("<q", self._mmap[limit_offset : limit_offset + 8])[0]
        if index:
            start = struct.unpack("<q", self._mmap[limit_offset - 8 : limit_offset])[0]
        else:
            start = 0
        if not 0 <= start <= end <= self._index_start:
            raise ValueError(f"Invalid Bag record range at index {index}")
        return bytes(self._mmap[start:end])

    def __iter__(self) -> Iterator[bytes]:
        for index in range(self._record_count):
            yield self[index]

    def close(self) -> None:
        mapping = getattr(self, "_mmap", None)
        if mapping is not None:
            mapping.close()
            self._mmap = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def __enter__(self) -> BagFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


@dataclass(frozen=True)
class ActionValuePosition:
    fen: str
    actions: tuple[tuple[str, float], ...]


def _validate_source(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    size = resolved.stat().st_size
    if size != int(identity["size_bytes"]):
        raise ValueError(f"Source size drift for {resolved}: {size}")
    digest = sha256_file(resolved)
    if digest != identity["sha256"]:
        raise ValueError(f"Source SHA-256 drift for {resolved}: {digest}")
    return {
        **dict(identity),
        "path": resolved.relative_to(_REPO_ROOT).as_posix(),
        "verified": True,
    }


def _selection_digest(namespace: str, identifier: str) -> int:
    digest = hashlib.sha256(f"{namespace}\0{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _absolute_move_slot(move: str) -> int:
    """Map absolute UCI to the compact shipped move list.

    LC0 represents a knight promotion in the ordinary from/to slot. No legal
    position can simultaneously contain the corresponding non-promotion pawn
    move, so stripping only the n suffix is collision-free within a FEN.
    """

    normalized = move[:-1] if len(move) == 5 and move.endswith("n") else move
    try:
        return move_to_policy_index(normalized, "lc0_1858")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Action-value record has an unmappable move {move!r}") from exc


def _expected_absolute_move_mask(fen: str) -> int:
    board = chess.Board(fen)
    mask = 0
    count = 0
    for move in board.legal_moves:
        bit = 1 << _absolute_move_slot(move.uci())
        if mask & bit:
            raise ValueError(f"Absolute move-slot collision in {fen}")
        mask |= bit
        count += 1
    if mask.bit_count() != count:  # pragma: no cover - defended above
        raise AssertionError("Legal move mask lost an action")
    return mask


def select_action_value_positions(
    path: Path,
    *,
    count: int,
) -> tuple[list[ActionValuePosition], dict[str, int]]:
    """Select the lowest deterministic hashes with bounded resident memory."""

    if count <= 0:
        raise ValueError("Action-value selection count must be positive")
    observed_masks: dict[str, int] = {}
    invalid_fens: set[str] = set()
    record_count = 0
    with BagFile(path) as bag:
        for payload in bag:
            record = decode_action_value_record(payload)
            record_count += 1
            if not 0.0 <= record.win_probability <= 1.0:
                invalid_fens.add(record.fen)
                continue
            try:
                bit = 1 << _absolute_move_slot(record.move)
            except ValueError:
                invalid_fens.add(record.fen)
                continue
            previous = observed_masks.get(record.fen, 0)
            if previous & bit:
                invalid_fens.add(record.fen)
            observed_masks[record.fen] = previous | bit

    valid_fens: list[str] = []
    for fen, observed in observed_masks.items():
        if fen in invalid_fens:
            continue
        try:
            expected = _expected_absolute_move_mask(fen)
        except (ValueError, TypeError):
            invalid_fens.add(fen)
            continue
        if observed == expected:
            valid_fens.append(fen)
        else:
            invalid_fens.add(fen)
    if len(valid_fens) < count:
        raise ValueError(f"Only {len(valid_fens)} valid action-value positions for count {count}")
    valid_fens.sort(
        key=lambda fen: (
            _selection_digest("chessbench-action-value-test-v1", fen),
            fen,
        )
    )
    selected_fens = valid_fens[:count]
    selected_set = set(selected_fens)
    actions: dict[str, list[tuple[str, float]]] = {fen: [] for fen in selected_fens}
    with BagFile(path) as bag:
        for payload in bag:
            record = decode_action_value_record(payload)
            if record.fen in selected_set:
                actions[record.fen].append((record.move, record.win_probability))
    selected = [ActionValuePosition(fen=fen, actions=tuple(actions[fen])) for fen in selected_fens]
    for position in selected:
        observed = 0
        for move, _value in position.actions:
            observed |= 1 << _absolute_move_slot(move)
        if observed != _expected_absolute_move_mask(position.fen):
            raise AssertionError("Selected action-value position changed between passes")
    return selected, {
        "total_records": record_count,
        "total_positions": len(observed_masks),
        "valid_positions": len(valid_fens),
        "rejected": len(invalid_fens),
    }


def _read_searchless_puzzles(path: Path) -> list[dict[str, str]]:
    expected = ("PuzzleId", "Rating", "PGN", "Solution", "FEN", "Moves")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Searchless puzzle columns drift: {reader.fieldnames}")
        rows = [dict(row) for row in reader]
    if len(rows) != 10_000:
        raise ValueError(f"Expected 10,000 Searchless puzzles, found {len(rows)}")
    identifiers = [row["PuzzleId"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Searchless puzzle IDs are not unique")
    return rows


def _join_lichess_metadata(
    dump_path: Path,
    searchless_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    wanted = {row["PuzzleId"]: row for row in searchless_rows}
    result: dict[str, dict[str, str]] = {}
    scanned = 0
    mismatched = 0
    with dump_path.open("rb") as compressed:
        stream = zstandard.ZstdDecompressor().stream_reader(compressed)
        with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            required = {
                "PuzzleId",
                "FEN",
                "Moves",
                "Rating",
                "Themes",
                "GameUrl",
                "OpeningTags",
            }
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("Lichess puzzle dump is missing required columns")
            for row in reader:
                scanned += 1
                puzzle_id = row["PuzzleId"]
                source = wanted.get(puzzle_id)
                if source is None:
                    continue
                if row["FEN"] != source["FEN"] or row["Moves"] != source["Moves"]:
                    mismatched += 1
                    continue
                result[puzzle_id] = {key: row[key] for key in required}
    return result, {
        "decompressed_rows_scanned": scanned,
        "matched": len(result),
        "missing": len(wanted) - len(result) - mismatched,
        "fen_or_line_mismatch": mismatched,
    }


def _game_id(game_url: str, puzzle_id: str) -> str:
    if game_url:
        path = game_url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        parts = path.split("/")
        stem = parts[-1]
        if stem in {"white", "black"} and len(parts) >= 2:
            stem = parts[-2]
        if stem:
            return stem
    return f"puzzle:{puzzle_id}"


def _split_code(group_id: str) -> int:
    bucket = _selection_digest("bt4-public-puzzle-split-v1", group_id) % 10
    return 0 if bucket < 6 else 1 if bucket < 8 else 2


def _planes_u8(board: chess.Board, history: Sequence[chess.Board]) -> np.ndarray:
    planes = encode_board(
        board,
        history,
        planes_layout="nchw",
        input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
    )
    rounded = np.rint(planes)
    if not np.array_equal(planes, rounded) or np.any((rounded < 0) | (rounded > 255)):
        raise ValueError("Classical LC0 planes are not losslessly representable as uint8")
    return rounded.astype(np.uint8)


def _legal_indices(board: chess.Board) -> np.ndarray:
    mask = legal_action_mask(
        board,
        codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
        input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
    )
    return np.flatnonzero(mask).astype(np.uint16)


def _theme_vocabulary(metadata: Mapping[str, Mapping[str, str]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {theme for row in metadata.values() for theme in row.get("Themes", "").split() if theme}
        )
    )


def _materialize_puzzle_arrays(
    rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = len(rows)
    theme_vocabulary = _theme_vocabulary(metadata)
    theme_index = {theme: index for index, theme in enumerate(theme_vocabulary)}
    arrays: dict[str, np.ndarray] = {
        "current_planes_u8": np.zeros((count, 112, 8, 8), dtype=np.uint8),
        "legal_idx_u16": np.full((count, MAX_CHESS_LEGAL_MOVES), LEGAL_PAD, dtype=np.uint16),
        "legal_count_u16": np.zeros(count, dtype=np.uint16),
        "target_action_u16": np.zeros(count, dtype=np.uint16),
        "future_action_u16": np.full((count, 7), LEGAL_PAD, dtype=np.uint16),
        "theme_multi_hot_u8": np.zeros((count, len(theme_vocabulary)), dtype=np.uint8),
        "split_u8": np.zeros(count, dtype=np.uint8),
        "rating_i16": np.zeros(count, dtype=np.int16),
        "lichess_rating_i16": np.full(count, -1, dtype=np.int16),
        "history_position_count_u16": np.zeros(count, dtype=np.uint16),
    }
    text_fields: dict[str, list[str]] = {
        "puzzle_id": [],
        "position_id": [],
        "group_id": [],
        "root_fen": [],
        "solution_uci": [],
        "themes": [],
        "opening_tags": [],
        "game_url": [],
        "plane_sha256": [],
    }
    square_arrays: dict[str, np.ndarray] = {}
    scalar_arrays: dict[str, np.ndarray] = {}
    move_arrays: dict[str, np.ndarray] = {}
    future_arrays: dict[str, np.ndarray] = {}
    theme_counts = np.zeros(len(theme_vocabulary), dtype=np.int64)

    for index, row in enumerate(rows):
        puzzle_id = row["PuzzleId"]
        reconstructed = reconstruct_puzzle(
            puzzle_id=puzzle_id,
            pgn=row["PGN"],
            fen_before_opponent=row["FEN"],
            moves=row["Moves"],
        )
        board = reconstructed.board
        planes = _planes_u8(board, reconstructed.history)
        legal = _legal_indices(board)
        target_move = reconstructed.solution_moves[0]
        target_action = encode_action(
            target_move,
            codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
            board=board,
            input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
        )
        if target_action not in legal:
            raise ValueError(f"Puzzle {puzzle_id} target is missing from legal actions")
        arrays["current_planes_u8"][index] = planes
        arrays["legal_idx_u16"][index, : legal.size] = legal
        arrays["legal_count_u16"][index] = legal.size
        arrays["target_action_u16"][index] = target_action
        arrays["rating_i16"][index] = int(row["Rating"])
        arrays["history_position_count_u16"][index] = len(reconstructed.history)

        replay = board.copy(stack=False)
        for ply, move in enumerate(reconstructed.solution_moves[:7]):
            arrays["future_action_u16"][index, ply] = encode_action(
                move,
                codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
                board=replay,
                input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
            )
            replay.push(move)

        exact_square = square_labels(board)
        exact_scalar = position_labels(board)
        exact_move = move_labels(board, target_move)
        exact_future = solution_token_labels(board, reconstructed.solution_moves)
        if not square_arrays:
            square_arrays = {
                name: np.zeros((count, *value.shape), dtype=value.dtype)
                for name, value in exact_square.items()
            }
            scalar_arrays = {
                f"concept_{name}_i16": np.zeros(count, dtype=np.int16) for name in exact_scalar
            }
            move_arrays = {
                f"target_{name}_i16": np.zeros(count, dtype=np.int16) for name in exact_move
            }
            future_arrays = {
                name: np.zeros((count, *value.shape), dtype=value.dtype)
                for name, value in exact_future.items()
            }
        for name, value in exact_square.items():
            square_arrays[name][index] = value
        for name, value in exact_scalar.items():
            scalar_arrays[f"concept_{name}_i16"][index] = int(value)
        for name, value in exact_move.items():
            move_arrays[f"target_{name}_i16"][index] = int(value)
        for name, value in exact_future.items():
            future_arrays[name][index] = value

        joined = metadata.get(puzzle_id, {})
        themes = joined.get("Themes", "").split()
        for theme in themes:
            theme_column = theme_index[theme]
            arrays["theme_multi_hot_u8"][index, theme_column] = 1
            theme_counts[theme_column] += 1
        group_id = _game_id(joined.get("GameUrl", ""), puzzle_id)
        arrays["split_u8"][index] = _split_code(group_id)
        if joined.get("Rating"):
            arrays["lichess_rating_i16"][index] = int(joined["Rating"])
        plane_sha256 = hashlib.sha256(np.ascontiguousarray(planes).tobytes()).hexdigest()
        position_id = hashlib.sha256(
            (
                f"{SOURCE_IDENTITIES['searchless_puzzles']['sha256']}\0{puzzle_id}\0"
                f"{board.fen()}\0{plane_sha256}\0{target_action}"
            ).encode("utf-8")
        ).hexdigest()
        values = {
            "puzzle_id": puzzle_id,
            "position_id": position_id,
            "group_id": group_id,
            "root_fen": board.fen(),
            "solution_uci": " ".join(move.uci() for move in reconstructed.solution_moves),
            "themes": " ".join(themes),
            "opening_tags": joined.get("OpeningTags", ""),
            "game_url": joined.get("GameUrl", ""),
            "plane_sha256": plane_sha256,
        }
        for name, value in values.items():
            text_fields[name].append(value)

    arrays.update(square_arrays)
    arrays.update(scalar_arrays)
    arrays.update(move_arrays)
    arrays.update(future_arrays)
    arrays.update({name: np.asarray(values) for name, values in text_fields.items()})
    split_counts = {
        name: int(np.count_nonzero(arrays["split_u8"] == code))
        for code, name in enumerate(("train", "development", "test"))
    }
    return arrays, {
        "count": count,
        "theme_vocabulary": list(theme_vocabulary),
        "theme_counts": {
            theme: int(theme_counts[index]) for index, theme in enumerate(theme_vocabulary)
        },
        "split_codes": {"train": 0, "development": 1, "test": 2},
        "split_counts": split_counts,
        "group_disjoint": True,
    }


def _materialize_engine_arrays(
    positions: Sequence[ActionValuePosition],
) -> dict[str, np.ndarray]:
    count = len(positions)
    arrays: dict[str, np.ndarray] = {
        "engine_current_planes_u8": np.zeros((count, 112, 8, 8), dtype=np.uint8),
        "engine_legal_idx_u16": np.full((count, MAX_CHESS_LEGAL_MOVES), LEGAL_PAD, dtype=np.uint16),
        "engine_win_probability_f64": np.full(
            (count, MAX_CHESS_LEGAL_MOVES), np.nan, dtype=np.float64
        ),
        "engine_legal_count_u16": np.zeros(count, dtype=np.uint16),
        "engine_target_action_u16": np.zeros(count, dtype=np.uint16),
    }
    fens: list[str] = []
    identifiers: list[str] = []
    best_moves: list[str] = []
    for index, position in enumerate(positions):
        board = chess.Board(position.fen)
        planes = _planes_u8(board, [board])
        encoded: list[tuple[int, str, float]] = []
        for move_text, value in position.actions:
            move = chess.Move.from_uci(move_text)
            action = encode_action(
                move,
                codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
                board=board,
                input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
            )
            encoded.append((action, move_text, value))
        encoded.sort(key=lambda value: value[0])
        if len({action for action, _move, _score in encoded}) != len(encoded):
            raise ValueError(f"Canonical action collision in {position.fen}")
        legal = np.asarray([value[0] for value in encoded], dtype=np.uint16)
        scores = np.asarray([value[2] for value in encoded], dtype=np.float64)
        best = min(
            encoded,
            key=lambda value: (-value[2], value[1]),
        )
        arrays["engine_current_planes_u8"][index] = planes
        arrays["engine_legal_idx_u16"][index, : legal.size] = legal
        arrays["engine_win_probability_f64"][index, : legal.size] = scores
        arrays["engine_legal_count_u16"][index] = legal.size
        arrays["engine_target_action_u16"][index] = best[0]
        plane_sha256 = hashlib.sha256(np.ascontiguousarray(planes).tobytes()).hexdigest()
        identifier = hashlib.sha256(
            (
                f"{SOURCE_IDENTITIES['action_value_test']['sha256']}\0{position.fen}\0"
                f"{plane_sha256}"
            ).encode("utf-8")
        ).hexdigest()
        fens.append(position.fen)
        identifiers.append(identifier)
        best_moves.append(best[1])
    arrays["engine_fen"] = np.asarray(fens)
    arrays["engine_position_id"] = np.asarray(identifiers)
    arrays["engine_best_move_uci"] = np.asarray(best_moves)
    return arrays


def _array_inventory(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def build_public_corpus(
    *,
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    engine_position_count: int = 2_048,
) -> dict[str, Any]:
    """Build and strict-load the immutable public-data corpus."""

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable corpus: {target}")
    if target in {Path("/"), Path.home().resolve(), _REPO_ROOT.resolve()}:
        raise ValueError(f"Unsafe corpus output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(f"Corpus staging path already exists: {staging}")
    staging.mkdir()
    try:
        sources = {
            name: _validate_source(external_dir / record["relative_path"], record)
            for name, record in SOURCE_IDENTITIES.items()
        }
        puzzle_rows = _read_searchless_puzzles(
            external_dir / SOURCE_IDENTITIES["searchless_puzzles"]["relative_path"]
        )
        metadata, join_report = _join_lichess_metadata(
            external_dir / SOURCE_IDENTITIES["lichess_puzzle_dump"]["relative_path"],
            puzzle_rows,
        )
        puzzle_arrays, puzzle_report = _materialize_puzzle_arrays(puzzle_rows, metadata)
        engine_positions, engine_report = select_action_value_positions(
            external_dir / SOURCE_IDENTITIES["action_value_test"]["relative_path"],
            count=engine_position_count,
        )
        engine_arrays = _materialize_engine_arrays(engine_positions)
        puzzle_path = staging / "puzzles.npz"
        engine_path = staging / "engine_positions.npz"
        write_deterministic_npz(puzzle_path, puzzle_arrays)
        write_deterministic_npz(engine_path, engine_arrays)
        base_manifest: dict[str, Any] = {
            "schema_version": PUBLIC_CORPUS_SCHEMA,
            "immutable_after_creation": True,
            "sources": sources,
            "upstream": {
                "searchless_chess_commit": SEARCHLESS_COMMIT,
                "searchless_code_license": "Apache-2.0",
                "searchless_dataset_license": "CC0 and CC-BY as documented upstream",
                "lichess_dump_date": LICHESS_DUMP_DATE,
                "lichess_license": "CC0",
            },
            "puzzles": {
                **puzzle_report,
                "metadata_join": join_report,
                "history_contract": (
                    "Full PGN reconstructed through the FEN, then the first Lichess setup "
                    "move applied; all available boards feed the classical 112-plane encoder."
                ),
                "data": {
                    "path": puzzle_path.name,
                    "size_bytes": puzzle_path.stat().st_size,
                    "sha256": sha256_file(puzzle_path),
                },
                "arrays": _array_inventory(puzzle_arrays),
            },
            "engine_positions": {
                **engine_report,
                "selected_count": len(engine_positions),
                "selection": (
                    "lowest SHA256(namespace + NUL + FEN) among positions whose recorded "
                    "moves exactly equal python-chess legal moves"
                ),
                "label": ("Stockfish 16 per-legal-action win probability released with ChessBench"),
                "history_contract": (
                    "FEN-only: current board is exact and seven unavailable history slots are "
                    "zero. Use for engine agreement and matched model comparisons, not "
                    "history-sensitive claims."
                ),
                "data": {
                    "path": engine_path.name,
                    "size_bytes": engine_path.stat().st_size,
                    "sha256": sha256_file(engine_path),
                },
                "arrays": _array_inventory(engine_arrays),
            },
            "concept_labels": concept_schema(),
            "action_codec": ACTION_CODEC_LC0_CANONICAL_1858,
            "input_format": LC0_CANONICAL_1858_INPUT_FORMAT,
            "action_vocab_size": ACTION_VOCAB_SIZE,
            "scientific_use": {
                "development": (
                    "train/development puzzle splits may select layers, probes, and thresholds"
                ),
                "confirmatory": (
                    "test puzzle split and engine-position metrics are opened only after "
                    "development choices are frozen"
                ),
                "forbidden": (
                    "Treating FEN-only engine positions as in-distribution history-complete data"
                ),
                "future_square_frames": {
                    "root": (
                        "Use future_*_token_root_i16 for lookahead probes over root activations"
                    ),
                    "native": (
                        "Use future_*_token_native_i16 only with activations replayed to that ply"
                    ),
                },
            },
        }
        manifest = {
            **base_manifest,
            "manifest_integrity": {
                "algorithm": "sha256-canonical-json-without-manifest_integrity",
                "sha256": content_identity(base_manifest),
            },
        }
        write_json_atomic(staging / "manifest.json", manifest)
        write_checksums(staging)
        os.replace(staging, target)
        _arrays, verified = load_public_corpus(target / "manifest.json")
        return verified
    finally:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()


def load_public_corpus(
    manifest_path: Path = DEFAULT_OUTPUT_DIR / "manifest.json",
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    manifest_path = manifest_path.resolve(strict=True)
    run_dir = manifest_path.parent
    verify_checksums(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PUBLIC_CORPUS_SCHEMA:
        raise ValueError("Unsupported public interpretability corpus schema")
    if manifest.get("immutable_after_creation") is not True:
        raise ValueError("Public corpus is not immutable")
    integrity = manifest.get("manifest_integrity", {})
    base = dict(manifest)
    base.pop("manifest_integrity", None)
    if integrity.get("sha256") != hashlib.sha256(canonical_json_bytes(base)).hexdigest():
        raise ValueError("Public corpus manifest integrity drift")

    result: dict[str, dict[str, np.ndarray]] = {}
    for family, key in (("puzzles", "puzzles"), ("engine_positions", "engine_positions")):
        record = manifest[key]
        data_record = record["data"]
        path = (run_dir / data_record["path"]).resolve(strict=True)
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(f"{family} data path escapes corpus") from exc
        if path.stat().st_size != int(data_record["size_bytes"]):
            raise ValueError(f"{family} data size drift")
        if sha256_file(path) != data_record["sha256"]:
            raise ValueError(f"{family} data SHA-256 drift")
        inventory = record["arrays"]
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(inventory):
                raise ValueError(f"{family} NPZ inventory drift")
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        for name, value in arrays.items():
            expected = inventory[name]
            if value.dtype.hasobject:
                raise ValueError(f"{family}.{name} has forbidden object dtype")
            if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                raise ValueError(f"{family}.{name} ABI drift")
        result[family] = arrays
    return result, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--engine-position-count", type=int, default=2_048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_public_corpus(
        external_dir=args.external_dir,
        output_dir=args.output_dir,
        engine_position_count=args.engine_position_count,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "manifest_integrity_sha256": manifest["manifest_integrity"]["sha256"],
                "puzzles": manifest["puzzles"]["count"],
                "engine_positions": manifest["engine_positions"]["selected_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
