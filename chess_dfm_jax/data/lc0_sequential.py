"""Compact, memory-mappable sequential shards for official LC0 V6 games."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import uuid
from collections import OrderedDict
from typing import Sequence

import numpy as np

from chess_dfm_jax.data.leela import TrainingRecord


LC0_SEQUENTIAL_V1 = "lc0-sequential-v1"
ACTION_VOCAB_SIZE = 1858
ACTION_PAD = np.iinfo(np.uint16).max
MAX_CHESS_LEGAL_MOVES = 218

# Official LC0 training records encode standard castling as king-to-rook.
# Hero's persisted canonical codec uses the ordinary king destination.
_LC0_QUEENSIDE_CASTLE = 97   # e1a1
_HERO_QUEENSIDE_CASTLE = 99  # e1c1
_LC0_KINGSIDE_CASTLE = 103   # e1h1
_HERO_KINGSIDE_CASTLE = 102  # e1g1

VALUE_FIELDS = (
    "root_q",
    "best_q",
    "root_d",
    "best_d",
    "root_m",
    "best_m",
    "plies_left",
    "result_q",
    "result_d",
    "played_q",
    "played_d",
    "played_m",
    "orig_q",
    "orig_d",
    "orig_m",
)

METADATA_FIELDS = (
    "castling_us_ooo",
    "castling_us_oo",
    "castling_them_ooo",
    "castling_them_oo",
    "side_to_move_or_enpassant",
    "rule50",
    "invariance_info",
    "dummy",
)

_STANDARD_INITIAL_CURRENT_PLANES = np.asarray(
    (
        65280,
        66,
        36,
        129,
        16,
        8,
        71776119061217280,
        4755801206503243776,
        2594073385365405696,
        9295429630892703744,
        1152921504606846976,
        576460752303423488,
    ),
    dtype=np.uint64,
)


@dataclass(frozen=True)
class SequentialGame:
    """One game converted from dense V6 records to compact arrays."""

    name: str
    planes_u64: np.ndarray
    metadata_u8: np.ndarray
    input_format_u16: np.ndarray
    played_actions_u16: np.ndarray
    best_actions_u16: np.ndarray
    values_f32: np.ndarray
    visits_u32: np.ndarray
    policy_kld_f32: np.ndarray
    reserved_u32: np.ndarray
    policy_offsets_i64: np.ndarray
    policy_indices_u16: np.ndarray
    policy_values_f16: np.ndarray

    @property
    def position_count(self) -> int:
        return int(self.planes_u64.shape[0])


def _require_v6_classical(record: TrainingRecord) -> None:
    if record.version != 6:
        raise ValueError(f"Sequential conversion requires V6 records, got {record.version}")
    if record.input_format != 1:
        raise ValueError(
            "Sequential conversion currently requires INPUT_CLASSICAL_112_PLANE "
            f"(1), got {record.input_format}"
        )


def has_standard_initial_position(records: Sequence[TrainingRecord]) -> bool:
    """Return whether a complete member starts from ordinary standard chess."""

    if not records:
        return False
    first = records[0]
    return bool(
        first.version == 6
        and first.input_format == 1
        and first.side_to_move == 0
        and first.castling == (1, 1, 1, 1)
        and np.array_equal(
            first.planes[:12],
            _STANDARD_INITIAL_CURRENT_PLANES,
        )
    )


def translate_lc0_training_action(
    record: TrainingRecord,
    action: int | None,
) -> int | None:
    """Translate official king-to-rook castling into Hero's persisted codec."""

    if action is None:
        return None
    action = int(action)
    us_ooo, us_oo, _, _ = record.castling
    if us_ooo and action == _LC0_QUEENSIDE_CASTLE:
        return _HERO_QUEENSIDE_CASTLE
    if us_oo and action == _LC0_KINGSIDE_CASTLE:
        return _HERO_KINGSIDE_CASTLE
    return action


def _sparse_policy(record: TrainingRecord) -> tuple[np.ndarray, np.ndarray]:
    probabilities = record.probabilities
    if probabilities is None:
        raise ValueError("Sequential conversion requires include_probabilities=True")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (ACTION_VOCAB_SIZE,):
        raise ValueError(f"Policy shape drift: {probabilities.shape}")
    valid = np.isfinite(probabilities) & (probabilities >= 0.0)
    indices = np.flatnonzero(valid).astype(np.uint16)
    values = probabilities[indices.astype(np.int32)]
    if indices.size == 0 or indices.size > MAX_CHESS_LEGAL_MOVES:
        raise ValueError(f"Invalid legal policy size: {indices.size}")
    total = float(values.sum(dtype=np.float64))
    if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise ValueError(f"LC0 policy mass drift: {total}")

    translated = indices.astype(np.int32)
    us_ooo, us_oo, _, _ = record.castling
    if us_ooo:
        translated[translated == _LC0_QUEENSIDE_CASTLE] = _HERO_QUEENSIDE_CASTLE
    if us_oo:
        translated[translated == _LC0_KINGSIDE_CASTLE] = _HERO_KINGSIDE_CASTLE
    order = np.argsort(translated, kind="stable")
    translated = translated[order]
    values = values[order]
    if np.unique(translated).size != translated.size:
        duplicate_values, duplicate_counts = np.unique(
            translated,
            return_counts=True,
        )
        duplicates = duplicate_values[duplicate_counts > 1].tolist()
        raise ValueError(
            "Policy codec translation produced duplicate indices: "
            f"raw={indices.tolist()}, translated={translated.tolist()}, "
            f"duplicates={duplicates}"
        )
    return translated.astype(np.uint16), values.astype(np.float16)


def records_to_sequential_game(
    records: Sequence[TrainingRecord],
    *,
    name: str,
) -> SequentialGame:
    """Convert one complete gzip member while retaining all V6 value fields."""

    if len(records) < 2:
        raise ValueError(f"Game {name!r} has fewer than two positions")
    for record in records:
        _require_v6_classical(record)

    position_count = len(records)
    planes = np.stack([record.planes for record in records]).astype("<u8", copy=False)
    metadata = np.asarray(
        [
            (
                *record.castling,
                record.side_to_move_or_enpassant,
                record.rule50,
                record.invariance_info,
                record.dummy,
            )
            for record in records
        ],
        dtype=np.uint8,
    )
    input_formats = np.asarray(
        [record.input_format for record in records],
        dtype=np.uint16,
    )

    def encoded_action(record: TrainingRecord, action: int | None) -> np.uint16:
        translated = translate_lc0_training_action(record, action)
        return np.uint16(ACTION_PAD if translated is None else translated)

    played_actions = np.asarray(
        [encoded_action(record, record.played_idx) for record in records],
        dtype=np.uint16,
    )
    best_actions = np.asarray(
        [encoded_action(record, record.best_idx) for record in records],
        dtype=np.uint16,
    )
    values = np.asarray(
        [
            [
                np.nan if getattr(record, field) is None else getattr(record, field)
                for field in VALUE_FIELDS
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    visits = np.asarray(
        [0 if record.visits is None else record.visits for record in records],
        dtype=np.uint32,
    )
    policy_kld = np.asarray(
        [np.nan if record.policy_kld is None else record.policy_kld for record in records],
        dtype=np.float32,
    )
    reserved = np.asarray(
        [0 if record.reserved is None else record.reserved for record in records],
        dtype=np.uint32,
    )

    policy_offsets = np.zeros((position_count + 1,), dtype=np.int64)
    policy_indices: list[np.ndarray] = []
    policy_values: list[np.ndarray] = []
    for position, record in enumerate(records):
        try:
            indices, probabilities = _sparse_policy(record)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sparse policy in game {name!r}, position {position}"
            ) from exc
        policy_indices.append(indices)
        policy_values.append(probabilities)
        policy_offsets[position + 1] = policy_offsets[position] + indices.size

    return SequentialGame(
        name=name,
        planes_u64=planes,
        metadata_u8=metadata,
        input_format_u16=input_formats,
        played_actions_u16=played_actions,
        best_actions_u16=best_actions,
        values_f32=values,
        visits_u32=visits,
        policy_kld_f32=policy_kld,
        reserved_u32=reserved,
        policy_offsets_i64=policy_offsets,
        policy_indices_u16=np.concatenate(policy_indices),
        policy_values_f16=np.concatenate(policy_values),
    )


def _concatenate_policy_offsets(games: Sequence[SequentialGame]) -> np.ndarray:
    total_positions = sum(game.position_count for game in games)
    result = np.zeros((total_positions + 1,), dtype=np.int64)
    position_cursor = 0
    policy_cursor = 0
    for game in games:
        count = game.position_count
        result[position_cursor + 1 : position_cursor + count + 1] = (
            game.policy_offsets_i64[1:] + policy_cursor
        )
        position_cursor += count
        policy_cursor += int(game.policy_offsets_i64[-1])
    return result


def save_sequential_shard(
    games: Sequence[SequentialGame],
    path: str | os.PathLike[str],
    *,
    split: str,
    shard_index: int,
) -> dict[str, object]:
    """Atomically write uncompressed NPY arrays that can be memory-mapped."""

    if not games:
        raise ValueError("Cannot save an empty sequential shard")
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()

    game_lengths = np.asarray([game.position_count for game in games], dtype=np.int64)
    game_offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(game_lengths, dtype=np.int64))
    )
    arrays = {
        "planes_u64": np.concatenate([game.planes_u64 for game in games], axis=0),
        "metadata_u8": np.concatenate([game.metadata_u8 for game in games], axis=0),
        "input_format_u16": np.concatenate(
            [game.input_format_u16 for game in games], axis=0
        ),
        "played_actions_u16": np.concatenate(
            [game.played_actions_u16 for game in games], axis=0
        ),
        "best_actions_u16": np.concatenate(
            [game.best_actions_u16 for game in games], axis=0
        ),
        "values_f32": np.concatenate([game.values_f32 for game in games], axis=0),
        "visits_u32": np.concatenate([game.visits_u32 for game in games], axis=0),
        "policy_kld_f32": np.concatenate(
            [game.policy_kld_f32 for game in games], axis=0
        ),
        "reserved_u32": np.concatenate(
            [game.reserved_u32 for game in games], axis=0
        ),
        "policy_offsets_i64": _concatenate_policy_offsets(games),
        "policy_indices_u16": np.concatenate(
            [game.policy_indices_u16 for game in games], axis=0
        ),
        "policy_values_f16": np.concatenate(
            [game.policy_values_f16 for game in games], axis=0
        ),
        "game_offsets_i64": game_offsets,
    }
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        game_names = [game.name for game in games]
        (temporary / "game_names.json").write_text(
            json.dumps(game_names, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "schema_version": LC0_SEQUENTIAL_V1,
            "split": split,
            "shard_index": int(shard_index),
            "game_count": len(games),
            "position_count": int(game_offsets[-1]),
            "trainable_start_count": int(game_offsets[-1] - len(games)),
            "policy_entry_count": int(arrays["policy_indices_u16"].shape[0]),
            "value_fields": list(VALUE_FIELDS),
            "metadata_fields": list(METADATA_FIELDS),
            "action_codec": "lc0-canonical-1858-v1",
            "source_action_translation": {
                "e1a1": "e1c1",
                "e1h1": "e1g1",
            },
            "storage": "uncompressed-npy-mmap",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        # Leave the uniquely named temporary directory for forensic recovery.
        raise
    return manifest


def decode_classical_planes(
    planes_u64: np.ndarray,
    metadata_u8: np.ndarray,
    input_format_u16: np.ndarray,
) -> np.ndarray:
    """Vectorize packed LC0 bitboards into float32 model inputs."""

    packed = np.ascontiguousarray(np.asarray(planes_u64, dtype="<u8"))
    metadata = np.asarray(metadata_u8, dtype=np.uint8)
    input_formats = np.asarray(input_format_u16, dtype=np.uint16)
    if packed.ndim != 2 or packed.shape[1] != 104:
        raise ValueError(f"planes_u64 must have shape [batch, 104], got {packed.shape}")
    batch_size = packed.shape[0]
    if metadata.shape != (batch_size, len(METADATA_FIELDS)):
        raise ValueError(f"metadata_u8 shape drift: {metadata.shape}")
    if input_formats.shape != (batch_size,) or np.any(input_formats != 1):
        raise ValueError("Only classical input-format 1 is supported")

    byte_view = packed.view(np.uint8).reshape(batch_size, 104, 8)
    history = np.unpackbits(byte_view, axis=-1, bitorder="big").reshape(
        batch_size, 104, 8, 8
    )
    result = np.zeros((batch_size, 112, 8, 8), dtype=np.float32)
    result[:, :104] = history
    result[:, 104:108] = metadata[:, :4, None, None]
    result[:, 108] = metadata[:, 4, None, None]
    result[:, 109] = metadata[:, 5, None, None]
    result[:, 111] = 1.0
    return result


def _wdl_from_qd_array(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    win = 0.5 * (1.0 - d + q)
    loss = 0.5 * (1.0 - d - q)
    result = np.stack((win, d, loss), axis=-1).astype(np.float32)
    if np.any(result < -1e-5) or np.any(result > 1.0 + 1e-5):
        raise ValueError("LC0 Q/D values do not define a valid WDL distribution")
    return np.clip(result, 0.0, 1.0)


class SequentialShard:
    """Memory-mapped shard with on-demand H-step window materialization."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != LC0_SEQUENTIAL_V1:
            raise ValueError(f"Unsupported sequential schema: {manifest}")
        self.manifest = manifest
        array_names = (
            "planes_u64",
            "metadata_u8",
            "input_format_u16",
            "played_actions_u16",
            "best_actions_u16",
            "values_f32",
            "visits_u32",
            "policy_kld_f32",
            "reserved_u32",
            "policy_offsets_i64",
            "policy_indices_u16",
            "policy_values_f16",
            "game_offsets_i64",
        )
        for name in array_names:
            setattr(
                self,
                name,
                np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False),
            )
        if self.planes_u64.shape[0] != int(manifest["position_count"]):
            raise ValueError("Sequential shard position count drift")

    @property
    def position_count(self) -> int:
        return int(self.planes_u64.shape[0])

    @property
    def trainable_start_count(self) -> int:
        return int(self.manifest["trainable_start_count"])

    def materialize(
        self,
        starts: np.ndarray,
        *,
        horizon: int,
        legal_capacity: int = MAX_CHESS_LEGAL_MOVES,
    ) -> dict[str, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int64)
        if starts.ndim != 1 or starts.size == 0:
            raise ValueError("starts must be a non-empty rank-1 array")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if legal_capacity < 1 or legal_capacity > MAX_CHESS_LEGAL_MOVES:
            raise ValueError(f"Invalid legal capacity: {legal_capacity}")
        if np.any(starts < 0) or np.any(starts >= self.position_count):
            raise IndexError("Sequential start is out of range")

        game_indices = np.searchsorted(
            np.asarray(self.game_offsets_i64[1:]),
            starts,
            side="right",
        )
        game_ends = np.asarray(self.game_offsets_i64[game_indices + 1], dtype=np.int64)
        valid_counts = np.minimum(horizon, game_ends - starts - 1).clip(min=0)
        steps = np.arange(horizon, dtype=np.int64)[None, :]
        valid_mask = steps < valid_counts[:, None]
        action_rows = np.minimum(starts[:, None] + steps, game_ends[:, None] - 1)
        future_rows = np.minimum(action_rows + 1, game_ends[:, None] - 1)

        current_planes = decode_classical_planes(
            self.planes_u64[starts],
            self.metadata_u8[starts],
            self.input_format_u16[starts],
        )
        flat_future = future_rows.reshape(-1)
        future_planes = decode_classical_planes(
            self.planes_u64[flat_future],
            self.metadata_u8[flat_future],
            self.input_format_u16[flat_future],
        ).reshape(starts.size, horizon, 112, 8, 8)

        actions = np.asarray(self.played_actions_u16[action_rows], dtype=np.int32)
        actions = np.where(valid_mask, actions, 0).astype(np.int32)
        if np.any(actions[valid_mask] == ACTION_PAD):
            raise ValueError("A valid sequential transition has no played action")

        legal_indices = np.full(
            (starts.size, horizon, legal_capacity),
            ACTION_PAD,
            dtype=np.uint16,
        )
        legal_counts = np.zeros((starts.size, horizon), dtype=np.uint16)
        for row in range(starts.size):
            for step in range(horizon):
                if not valid_mask[row, step]:
                    continue
                position = int(action_rows[row, step])
                begin = int(self.policy_offsets_i64[position])
                end = int(self.policy_offsets_i64[position + 1])
                count = end - begin
                if count > legal_capacity:
                    raise ValueError(
                        f"Legal set needs {count} slots, capacity is {legal_capacity}"
                    )
                legal_indices[row, step, :count] = self.policy_indices_u16[begin:end]
                legal_counts[row, step] = count

        future_values = np.asarray(
            self.values_f32[future_rows],
            dtype=np.float32,
        )
        current_values = np.asarray(
            self.values_f32[starts],
            dtype=np.float32,
        )
        root_q_index = VALUE_FIELDS.index("root_q")
        root_d_index = VALUE_FIELDS.index("root_d")
        current_value_target = current_values[..., root_q_index]
        current_wdl_target = _wdl_from_qd_array(
            current_value_target,
            current_values[..., root_d_index],
        ).astype(np.float32)
        value_targets = future_values[..., root_q_index]
        wdl_targets = _wdl_from_qd_array(
            value_targets,
            future_values[..., root_d_index],
        )
        future_valid = valid_mask.astype(np.float32)
        value_targets = np.where(valid_mask, value_targets, 0.0).astype(np.float32)
        wdl_targets = np.where(
            valid_mask[..., None],
            wdl_targets,
            0.0,
        ).astype(np.float32)
        valid = (valid_counts > 0).astype(np.float32)
        terminal = np.maximum(valid_counts - 1, 0).astype(np.int32)
        return {
            "current_planes": current_planes,
            "future_planes": future_planes,
            "next_planes": future_planes[np.arange(starts.size), terminal],
            "action_indices": actions,
            "action_idx": actions[:, 0],
            "future_valid": future_valid,
            "valid": valid,
            "terminal_target_index": terminal,
            "legal_idx": legal_indices.astype(np.int32),
            "legal_count": legal_counts.astype(np.int32),
            "legal_masks_valid": future_valid,
            "current_value_target": current_value_target.astype(np.float32),
            "current_wdl_target": current_wdl_target,
            "value_targets": value_targets,
            "value_target": value_targets[np.arange(starts.size), terminal],
            "wdl_targets": wdl_targets,
            "wdl_target": wdl_targets[np.arange(starts.size), terminal],
        }


class SequentialBatches:
    """Stateless, restart-safe batches over variable-sized sequential shards."""

    def __init__(
        self,
        dataset_root: str | os.PathLike[str],
        *,
        split: str,
        batch_size: int,
        horizon: int,
        seed: int,
        shuffle_batches: bool,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.shuffle_batches = bool(shuffle_batches)
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")
        if self.batch_size < 1 or self.horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        self.paths = sorted(
            self.dataset_root.glob(f"chunks/chunk-*/{split}")
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No {split!r} sequential shards under {self.dataset_root}"
            )

        self.manifests = [
            json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            for path in self.paths
        ]
        for path, manifest in zip(self.paths, self.manifests, strict=True):
            if manifest.get("schema_version") != LC0_SEQUENTIAL_V1:
                raise ValueError(f"Schema drift in {path}: {manifest}")
            if manifest.get("split") != split:
                raise ValueError(f"Split drift in {path}: {manifest.get('split')}")
        self.batches_per_shard = tuple(
            int(manifest["trainable_start_count"]) // self.batch_size
            for manifest in self.manifests
        )
        self.slots = tuple(
            (shard_index, batch_in_shard)
            for shard_index, count in enumerate(self.batches_per_shard)
            for batch_in_shard in range(count)
        )
        if not self.slots:
            raise ValueError("No full sequential batches fit in the selected split")
        self.steps_per_epoch = len(self.slots)
        self._cache: OrderedDict[int, SequentialShard] = OrderedDict()

    def _slot_for_step(self, step: int) -> tuple[int, int]:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        epoch, step_in_epoch = divmod(int(step), self.steps_per_epoch)
        if not self.shuffle_batches:
            return self.slots[step_in_epoch]
        order = list(range(self.steps_per_epoch))
        random.Random(self.seed + epoch).shuffle(order)
        return self.slots[order[step_in_epoch]]

    def _load_shard(self, shard_index: int) -> SequentialShard:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        shard = SequentialShard(self.paths[shard_index])
        self._cache[shard_index] = shard
        self._cache.move_to_end(shard_index)
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return shard

    @staticmethod
    def _physical_starts(
        shard: SequentialShard,
        ordinals: np.ndarray,
    ) -> np.ndarray:
        game_offsets = np.asarray(shard.game_offsets_i64, dtype=np.int64)
        game_lengths = np.diff(game_offsets)
        trainable_lengths = game_lengths - 1
        if np.any(trainable_lengths < 1):
            raise ValueError("Sequential shard contains a game shorter than two positions")
        trainable_offsets = np.concatenate(
            (
                np.zeros((1,), dtype=np.int64),
                np.cumsum(trainable_lengths, dtype=np.int64),
            )
        )
        game_indices = np.searchsorted(
            trainable_offsets[1:],
            ordinals,
            side="right",
        )
        within_game = ordinals - trainable_offsets[game_indices]
        return game_offsets[game_indices] + within_game

    def batch_at(self, step: int) -> dict[str, np.ndarray]:
        shard_index, batch_in_shard = self._slot_for_step(step)
        shard = self._load_shard(shard_index)
        start = batch_in_shard * self.batch_size
        ordinals = np.arange(start, start + self.batch_size, dtype=np.int64)
        physical_starts = self._physical_starts(shard, ordinals)
        batch = shard.materialize(
            physical_starts,
            horizon=self.horizon,
        )
        for key, value in batch.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != self.batch_size:
                raise ValueError(
                    f"Unexpected sequential batch leaf {key!r}: {array.shape}"
                )
        return batch

    def provenance(self) -> dict[str, object]:
        entries = [
            (
                f"{path.relative_to(self.dataset_root)}\t"
                f"{manifest['position_count']}\t"
                f"{manifest['trainable_start_count']}"
            )
            for path, manifest in zip(self.paths, self.manifests, strict=True)
        ]
        return {
            "schema_version": LC0_SEQUENTIAL_V1,
            "dataset_root": str(self.dataset_root),
            "split": self.split,
            "shard_count": len(self.paths),
            "batch_size": self.batch_size,
            "horizon": self.horizon,
            "steps_per_epoch": self.steps_per_epoch,
            "seed": self.seed,
            "shuffle_batches": self.shuffle_batches,
            "file_manifest_sha256": hashlib.sha256(
                "\n".join(entries).encode("utf-8")
            ).hexdigest(),
        }


__all__ = [
    "ACTION_PAD",
    "LC0_SEQUENTIAL_V1",
    "MAX_CHESS_LEGAL_MOVES",
    "METADATA_FIELDS",
    "SequentialGame",
    "SequentialBatches",
    "SequentialShard",
    "VALUE_FIELDS",
    "decode_classical_planes",
    "has_standard_initial_position",
    "records_to_sequential_game",
    "save_sequential_shard",
    "translate_lc0_training_action",
]
