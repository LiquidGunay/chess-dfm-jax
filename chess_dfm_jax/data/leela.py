"""Leela self-play chunk utilities for standalone training experiments."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
import os
import queue
import random
import threading
from typing import Iterable, Iterator, Sequence, Any

import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

from chess_dfm_jax import encoding as encode_mod
from chess_dfm_jax import policy as policy_mod


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

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        # For JEPA scaling, we wrap the raw records into unrolled sequences
        paths = list(self.chunk_paths)
        if self.shuffle_files:
            self.rng.shuffle(paths)

        current_batch = collections.defaultdict(list)

        for path in paths:
            if path.endswith(".npz"):
                try:
                    data = np.load(path)
                    planes_t = data["planes_t"]
                    actions = data["actions"]
                    planes_target = data["planes_target"]
                    value_target = data.get("value_target", np.zeros((len(planes_t),), dtype=np.float32))
                    wdl_target = data.get("wdl_target", np.zeros((len(planes_t), 3), dtype=np.float32))
                    legal_mask = data.get("legal_mask", np.ones((len(planes_t), 1858), dtype=np.float32))
                    
                    for i in range(len(planes_t)):
                        current_batch["current_planes"].append(planes_t[i])
                        current_batch["action_indices"].append(actions[i])
                        current_batch["next_planes"].append(planes_target[i])
                        current_batch["valid"].append(np.array(1.0, dtype=np.float32))
                        current_batch["value_target"].append(value_target[i])
                        current_batch["wdl_target"].append(wdl_target[i])
                        current_batch["legal_mask"].append(legal_mask[i])
                        
                        if len(current_batch["current_planes"]) >= self.batch_size:
                            yield self._finalize_batch(current_batch)
                            current_batch = collections.defaultdict(list)
                except Exception as e:
                    print(f"Failed to read npz {path}: {e}")
                continue

            for record in iter_records(path):

                # Try to unroll a JEPA sequence from this record
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
        # 1. Reconstruct current board
        try:
            board = record_to_board(record)
        except Exception:
            return None
        
        # 2. Extract initial planes (current state)
        # record.planes are the dense 8x8x13 planes
        # We need them in float32 NCHW format [112, 8, 8]
        # But for scaling sweep, we can just use the raw record planes 
        # provided they are encoded correctly for the horizon.
        # Actually, record.planes is 104x u64, which are the 13*8 board history bits.
        # We only care about the latest board [0:13].
        
        fmt = INPUT_FORMAT_NAMES.get(record.input_format, "INPUT_CLASSICAL_112_PLANE")
        current_planes = encode_mod.encode_board(board, history=[], input_format=fmt)
        
        # 3. Unroll K steps using engine best moves if available, or just played moves
        action_indices = []
        temp_board = board.copy()
        
        # We only have the move for the *current* state in the chunk record.
        # To get more moves, we'd need sequential records or an engine.
        # Since we want to use REAL data, and chunks are usually game segments,
        # we'll assume for this scaling sweep that we're testing the model's 
        # ability to unroll. If we only have 1 move, we fill the rest with zeros 
        # and set valid=0 for those samples? No, let's just use 1-step for now 
        # if we can't find sequential data, but allow the architecture to be multi-step.
        
        move_idx = record.best_idx if record.best_idx is not None else record.played_idx
        if move_idx is None:
            return None
            
        # For this phase, we unroll the same move if needed or just use 1-step logic
        # while keeping the [B, K] shape.
        actions = np.zeros((self.horizon,), dtype=np.int32)
        actions[0] = move_idx
        
        # Next state
        try:
            move = policy_mod.policy_index_to_move(move_idx, "lc0_1858")
            if move in temp_board.legal_moves:
                temp_board.push(move)
                next_planes = encode_mod.encode_board(temp_board, history=[], input_format=fmt)
            else:
                return None
        except:
            return None
            
        return {
            "current_planes": current_planes,
            "action_indices": actions, # [K]
            "next_planes": next_planes,
            "valid": np.array(1.0, dtype=np.float32),
            "value_target": np.array(record.q_value, dtype=np.float32) if record.q_value is not None else np.zeros((), dtype=np.float32),
            "wdl_target": np.array(record.wdl, dtype=np.float32) if record.wdl is not None else np.zeros((3,), dtype=np.float32),
        }

    def _finalize_batch(self, batch_dict: dict[str, list]) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in batch_dict.items()}


def discover_chunk_files(chunk_dir: str) -> list[str]:
    paths = []
    for root, _dirs, files in os.walk(chunk_dir):
        for name in sorted(files):
            if name.endswith((".gz", ".zst", ".npz")):
                paths.append(os.path.join(root, name))
    return paths


def _open_chunk(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
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
                    q, w, d, l = struct.unpack("<4f", floats_bytes)
                    q_value = q
                    wdl = (w, d, l)

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
        if us_ooo: rights |= chess.BB_A1
        if us_oo: rights |= chess.BB_H1
        if them_ooo: rights |= chess.BB_A8
        if them_oo: rights |= chess.BB_H8
    else:
        if us_ooo: rights |= chess.BB_A8
        if us_oo: rights |= chess.BB_H8
        if them_ooo: rights |= chess.BB_A1
        if them_oo: rights |= chess.BB_H1
    board.castling_rights = rights
    board.halfmove_clock = int(record.rule50)
    return board

import collections
__all__ = ["LeelaChunkDataLoader", "discover_chunk_files", "record_to_board", "iter_records"]
