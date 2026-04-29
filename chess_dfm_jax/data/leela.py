"""Leela self-play chunk utilities for standalone training experiments."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import gzip
import os
import random
from typing import Callable, Iterator, Sequence

import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

from chess_dfm_jax import encoding as encode_mod
from chess_dfm_jax import policy as policy_mod
from chess_dfm_jax.data.trajectory import trajectory_shard_from_npz, trajectory_shard_to_batch


V3_RECORD_SIZE = 8276
V4_RECORD_SIZE = 8292
V5_RECORD_SIZE = 8308
V6_RECORD_SIZE = 8356

INPUT_FORMAT_NAMES = {
    0: "INPUT_CLASSICAL_112_PLANE",
    1: "INPUT_CLASSICAL_112_PLANE",
    2: "INPUT_112_WITH_CASTLING_PLANE",
    3: "INPUT_112_WITH_CANONICALIZATION",
    4: "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES",
    5: "INPUT_112_WITH_CANONICALIZATION_V2",
    132: "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON",
    133: "INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON",
}


@dataclass
class TrainingRecord:
    version: int
    input_format: int
    planes: np.ndarray
    castling: tuple[int, int, int, int]
    side_to_move: int
    rule50: int
    invariance_info: int
    played_idx: int | None
    best_idx: int | None
    q_value: float | None = None
    wdl: tuple[float, float, float] | None = None


@dataclass
class ChunkSample:
    planes: np.ndarray
    played_idx: int | None
    best_idx: int | None
    input_format: str
    side_to_move: int
    rule50: int
    invariance_info: int
    board: "chess.Board | None" = None
    played_move: "chess.Move | None" = None
    best_move: "chess.Move | None" = None


class LeelaChunkDataLoader:
    """Batched iterator with optional file shuffling and background prefetch."""

    def __init__(
        self,
        chunk_paths: Sequence[str],
        *,
        batch_size: int,
        shuffle_files: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 0,
        decode_mode: str = "raw",
        include_board: bool = False,
        include_moves: bool = False,
        drop_last: bool = False,
        prefetch_batches: int = 0,
        horizon: int = 1,
        include_metadata: bool = False,
        chunk_paths_provider: Callable[[], Sequence[str]] | None = None,
        action_source: str = "best",
    ):
        self.chunk_paths = [str(path) for path in chunk_paths]
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.shuffle_buffer = shuffle_buffer
        self.rng = random.Random(seed)
        self.decode_mode = decode_mode
        self.include_board = include_board
        self.include_moves = include_moves
        self.drop_last = drop_last
        self.prefetch_batches = prefetch_batches
        self.horizon = horizon
        self.include_metadata = include_metadata
        self.chunk_paths_provider = chunk_paths_provider
        if action_source not in {"best", "played"}:
            raise ValueError(f"Unsupported action_source: {action_source}")
        self.action_source = action_source

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        paths = (
            [str(path) for path in self.chunk_paths_provider()]
            if self.chunk_paths_provider is not None
            else list(self.chunk_paths)
        )
        if self.shuffle_files:
            self.rng.shuffle(paths)

        current_batch = collections.defaultdict(list)

        for path in paths:
            if path.endswith(".npz"):
                try:
                    with np.load(path, allow_pickle=False) as data:
                        shard = trajectory_shard_from_npz(data)
                    batch = trajectory_shard_to_batch(shard, include_metadata=self.include_metadata)
                    for i in range(shard.batch_size):
                        for key, value in batch.items():
                            current_batch[key].append(value[i])
                        if len(current_batch["current_planes"]) >= self.batch_size:
                            yield self._finalize_batch(current_batch)
                            current_batch = collections.defaultdict(list)
                except Exception as e:
                    print(f"Failed to read npz {path}: {e}")
                continue

            for record in iter_records(path):
                sample = self._unroll_jepa_sample(record)
                if sample is None:
                    continue
                
                for k, v in sample.items():
                    current_batch[k].append(v)
                
                if len(current_batch["current_planes"]) >= self.batch_size:
                    yield self._finalize_batch(current_batch)
                    current_batch = collections.defaultdict(list)
        
        if not self.drop_last and len(current_batch["current_planes"]) > 0:
            yield self._finalize_batch(current_batch)

    def _unroll_jepa_sample(self, record: TrainingRecord) -> dict[str, np.ndarray] | None:
        try:
            board = record_to_board(record)
        except Exception:
            return None

        fmt = INPUT_FORMAT_NAMES.get(record.input_format, "INPUT_CLASSICAL_112_PLANE")
        current_planes = encode_mod.encode_board(board, history=[], input_format=fmt)

        if self.action_source == "played":
            move_idx = record.played_idx if record.played_idx is not None else record.best_idx
        else:
            move_idx = record.best_idx if record.best_idx is not None else record.played_idx
        if move_idx is None:
            return None

        if self.horizon != 1:
            raise ValueError(
                "Raw LC0 chunks only support horizon=1 exact rollouts. "
                "Preprocess to trajectory-v2 .npz shards for multi-step training."
            )

        temp_board = board.copy(stack=False)
        try:
            move = policy_mod.policy_index_to_move(move_idx, "lc0_1858")
            if move in temp_board.legal_moves:
                legal_mask = policy_mod.legal_move_mask(temp_board, "lc0_1858").astype(np.float32)
                temp_board.push(move)
                next_planes = encode_mod.encode_board(
                    temp_board, history=[], input_format=fmt
                ).astype(np.float32)
            else:
                return None
        except Exception:
            return None

        future_value = (
            np.asarray(-record.q_value, dtype=np.float32)
            if record.q_value is not None
            else np.zeros((), dtype=np.float32)
        )
        if record.wdl is not None:
            future_wdl = np.asarray([record.wdl[2], record.wdl[1], record.wdl[0]], dtype=np.float32)
        else:
            future_wdl = np.zeros((3,), dtype=np.float32)

        return {
            "current_planes": current_planes,
            "action_indices": np.asarray([move_idx], dtype=np.int32),
            "action_idx": np.asarray(move_idx, dtype=np.int32),
            "future_planes": next_planes[None, ...],
            "future_valid": np.ones((1,), dtype=np.float32),
            "terminal_target_index": np.asarray(0, dtype=np.int32),
            "next_planes": next_planes,
            "valid": np.array(1.0, dtype=np.float32),
            "value_targets": future_value[None],
            "value_target": future_value,
            "wdl_targets": future_wdl[None, :],
            "wdl_target": future_wdl,
            "legal_masks": legal_mask[None, :],
            "legal_mask": legal_mask,
        }

    def _finalize_batch(self, batch_dict: dict[str, list]) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in batch_dict.items()}


def discover_chunk_files(chunk_dir: str | None) -> list[str]:
    if not chunk_dir:
        return []
    paths = []
    for root, _dirs, files in os.walk(chunk_dir):
        for name in sorted(files):
            if name.endswith((".gz", ".zst", ".npz")):
                paths.append(os.path.join(root, name))
    return paths


def _open_chunk(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".zst"):
        if zstd is None:
            raise ImportError("zstandard is required to read .zst LC0 chunks.")
        return zstd.open(path, "rb")
    return open(path, "rb")


def iter_records(path: str) -> Iterator[TrainingRecord]:
    with _open_chunk(path) as f:
        while True:
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                break
            version = int.from_bytes(version_bytes, "little", signed=False)
            if version == 3:
                record_size = V3_RECORD_SIZE
            elif version == 4:
                record_size = V4_RECORD_SIZE
            elif version == 5:
                record_size = V5_RECORD_SIZE
            elif version == 6:
                record_size = V6_RECORD_SIZE
            else:
                break
            rest = f.read(record_size - 4)
            if len(rest) < record_size - 4:
                break
            record = version_bytes + rest

            offset = 4
            input_format = 0
            if version >= 5:
                input_format = int.from_bytes(record[offset : offset + 4], "little")
                offset += 4

            probs_offset = offset
            planes_offset = probs_offset + 1858 * 4
            planes = np.frombuffer(record, dtype="<u8", count=104, offset=planes_offset).copy()

            castling_offset = planes_offset + 104 * 8
            castling = tuple(int(b) for b in record[castling_offset : castling_offset + 4])

            stm_offset = castling_offset + 4
            side_to_move = int(record[stm_offset])
            rule50 = int(record[stm_offset + 1])
            invariance_info = int(record[stm_offset + 2]) if version >= 5 else 0

            played_idx = None
            best_idx = None
            q_value = None
            wdl = None
            if version >= 6:
                floats_offset = stm_offset + 4
                
                import struct
                floats_bytes = record[floats_offset:floats_offset + 16]
                if len(floats_bytes) == 16:
                    q, w, d, loss_prob = struct.unpack("<4f", floats_bytes)
                    q_value = q
                    wdl = (w, d, loss_prob)

                visits_offset = floats_offset + 15 * 4
                played_idx = int.from_bytes(record[visits_offset + 4 : visits_offset + 6], "little")
                best_idx = int.from_bytes(record[visits_offset + 6 : visits_offset + 8], "little")

            yield TrainingRecord(
                version=version,
                input_format=input_format,
                planes=planes,
                castling=castling,
                side_to_move=side_to_move,
                rule50=rule50,
                invariance_info=invariance_info,
                played_idx=played_idx,
                best_idx=best_idx,
                q_value=q_value,
                wdl=wdl,
            )


def record_to_board(record: TrainingRecord) -> "chess.Board":
    if chess is None:  # pragma: no cover
        raise ImportError("python-chess is required for chunk conversion.")

    planes = record.planes.copy()
    transform = record.invariance_info & 0x07
    if transform:
        for idx in range(12):
            planes[idx] = encode_mod._transform_mask(int(planes[idx]), transform)

    ours = [int(x) for x in planes[0:6]]
    theirs = [int(x) for x in planes[6:12]]
    
    if record.side_to_move == 1:
        ours = [encode_mod._reverse_bytes_in_bytes(bb) for bb in ours]
        theirs = [encode_mod._reverse_bytes_in_bytes(bb) for bb in theirs]

    board = chess.Board(None)
    if record.side_to_move == 0:
        white_planes, black_planes = ours, theirs
    else:
        white_planes, black_planes = theirs, ours

    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for planes_list, color in [(white_planes, chess.WHITE), (black_planes, chess.BLACK)]:
        for plane, ptype in zip(planes_list, piece_types):
            bb = plane
            while bb:
                lsb = bb & -bb
                sq = lsb.bit_length() - 1
                board.set_piece_at(sq, chess.Piece(ptype, color))
                bb &= bb - 1

    board.turn = chess.WHITE if record.side_to_move == 0 else chess.BLACK
    us_ooo, us_oo, them_ooo, them_oo = record.castling
    rights = 0
    if record.side_to_move == 0:
        if us_ooo:
            rights |= chess.BB_A1
        if us_oo:
            rights |= chess.BB_H1
        if them_ooo:
            rights |= chess.BB_A8
        if them_oo:
            rights |= chess.BB_H8
    else:
        if us_ooo:
            rights |= chess.BB_A8
        if us_oo:
            rights |= chess.BB_H8
        if them_ooo:
            rights |= chess.BB_A1
        if them_oo:
            rights |= chess.BB_H1
    board.castling_rights = rights
    board.halfmove_clock = int(record.rule50)
    return board


__all__ = ["LeelaChunkDataLoader", "discover_chunk_files", "record_to_board", "iter_records"]
