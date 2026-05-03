"""Optional Grain-backed loaders for trajectory NPZ shards."""

from __future__ import annotations

import bisect
import collections
from collections.abc import Iterable, Sequence
from pathlib import Path
import random
import threading
from typing import Any

import numpy as np

from chess_dfm_jax.data.trajectory import (
    trajectory_action_batch_from_npz,
    trajectory_joint_batch_from_npz,
    trajectory_latent_batch_from_npz,
    trajectory_shard_from_npz,
    trajectory_shard_to_batch,
)
from chess_dfm_jax.data.trajectory_v3 import TRAJECTORY_V3, trajectory_v3_to_batch


class GrainUnavailableError(ImportError):
    """Raised when the optional Grain dependency is requested but unavailable."""


def _import_grain() -> Any:
    try:
        import grain.python as grain  # type: ignore[import-not-found]

        return grain
    except ImportError as exc:
        try:
            import grain  # type: ignore[import-not-found]

            return grain
        except ImportError:
            raise GrainUnavailableError(
                "Grain data loading was requested, but the optional `grain` package is not "
                "installed. Install grain or use --data-loader=leela."
            ) from exc


def grain_available() -> bool:
    try:
        _import_grain()
        return True
    except GrainUnavailableError:
        return False


def _row_count(path: str) -> int:
    with np.load(path, allow_pickle=False) as data:
        if "schema_version" in data and str(np.asarray(data["schema_version"]).item()) == TRAJECTORY_V3:
            return int(data["actions_u16"].shape[0])
        return int(data["actions"].shape[0])


def _decode_npz_batch(
    path: str,
    *,
    batch_view: str,
    horizon: int,
    legal_lmax: int,
    include_metadata: bool,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema_version"]).item()) if "schema_version" in data else ""
        if schema == TRAJECTORY_V3:
            return trajectory_v3_to_batch(
                data,
                view=batch_view,
                horizon=horizon,
                include_metadata=include_metadata,
            )
        if batch_view == "dfm_action":
            return trajectory_action_batch_from_npz(
                data,
                horizon=horizon,
                include_metadata=include_metadata,
            )
        if batch_view == "jepa_latent":
            return trajectory_latent_batch_from_npz(
                data,
                horizon=horizon,
                include_metadata=include_metadata,
            )
        if batch_view == "joint_latent_sasa":
            return trajectory_joint_batch_from_npz(
                data,
                horizon=horizon,
                legal_lmax=legal_lmax,
                include_metadata=include_metadata,
            )
        shard = trajectory_shard_from_npz(data)
        return trajectory_shard_to_batch(shard, include_metadata=include_metadata)


class _TrajectoryShardSource:
    """Random-access row source over local trajectory NPZ shards.

    Grain owns sampling and batching. This source keeps a tiny decoded-shard LRU
    because batches typically access adjacent rows from the same shard.
    """

    def __init__(
        self,
        chunk_paths: Sequence[str],
        *,
        batch_view: str,
        horizon: int,
        legal_lmax: int,
        include_metadata: bool,
        shard_cache_size: int,
    ):
        self.chunk_paths = [str(Path(path)) for path in chunk_paths if str(path).endswith(".npz")]
        if not self.chunk_paths:
            raise ValueError("Grain trajectory loading requires at least one local .npz shard.")
        self.batch_view = batch_view
        self.horizon = horizon
        self.legal_lmax = legal_lmax
        self.include_metadata = include_metadata
        self.shard_cache_size = max(1, int(shard_cache_size))
        counts = [_row_count(path) for path in self.chunk_paths]
        self._offsets = np.cumsum([0, *counts]).astype(np.int64)
        self._cache: collections.OrderedDict[int, dict[str, np.ndarray]] = collections.OrderedDict()
        self._cache_lock = threading.Lock()

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = collections.OrderedDict()
        state.pop("_cache_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()

    def _load_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        with self._cache_lock:
            cached = self._cache.get(shard_idx)
            if cached is not None:
                self._cache.move_to_end(shard_idx)
                return cached
        batch = _decode_npz_batch(
            self.chunk_paths[shard_idx],
            batch_view=self.batch_view,
            horizon=self.horizon,
            legal_lmax=self.legal_lmax,
            include_metadata=self.include_metadata,
        )
        with self._cache_lock:
            self._cache[shard_idx] = batch
            while len(self._cache) > self.shard_cache_size:
                self._cache.popitem(last=False)
        return batch

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self._offsets, index) - 1
        row_idx = index - int(self._offsets[shard_idx])
        shard = self._load_shard(shard_idx)
        return {key: np.asarray(value[row_idx]) for key, value in shard.items()}


class _TrajectoryBatchSource:
    """Random-access source where each item is already a contiguous train batch.

    Compressed NPZ shards are expensive to decode one row at a time. The fast path
    therefore samples file order, decodes a shard once, and slices multiple
    contiguous batches out of it, matching `LeelaChunkDataLoader`'s data-access
    pattern while still using Grain for iteration/prefetch plumbing.
    """

    def __init__(
        self,
        batch_specs: Sequence[tuple[str, int, int]],
        *,
        batch_view: str,
        horizon: int,
        legal_lmax: int,
        include_metadata: bool,
        shard_cache_size: int,
    ):
        self.batch_specs = [(str(path), int(start), int(end)) for path, start, end in batch_specs]
        if not self.batch_specs:
            raise ValueError("Grain trajectory loading requires at least one complete batch.")
        self.batch_view = batch_view
        self.horizon = horizon
        self.legal_lmax = legal_lmax
        self.include_metadata = include_metadata
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._cache: collections.OrderedDict[str, dict[str, np.ndarray]] = collections.OrderedDict()
        self._cache_lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.batch_specs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = collections.OrderedDict()
        state.pop("_cache_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()

    def _load_path(self, path: str) -> dict[str, np.ndarray]:
        with self._cache_lock:
            cached = self._cache.get(path)
            if cached is not None:
                self._cache.move_to_end(path)
                return cached
        batch = _decode_npz_batch(
            path,
            batch_view=self.batch_view,
            horizon=self.horizon,
            legal_lmax=self.legal_lmax,
            include_metadata=self.include_metadata,
        )
        with self._cache_lock:
            self._cache[path] = batch
            while len(self._cache) > self.shard_cache_size:
                self._cache.popitem(last=False)
        return batch

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        path, start, end = self.batch_specs[int(index)]
        shard = self._load_path(path)
        return {key: np.asarray(value[start:end]) for key, value in shard.items()}


def _make_source_class(grain: Any):
    base = getattr(grain, "RandomAccessDataSource", object)
    if base is object:
        return _TrajectoryShardSource

    class TrajectoryShardSource(_TrajectoryShardSource, base):  # type: ignore[misc, valid-type]
        pass

    return TrajectoryShardSource


def _make_batch_source_class(grain: Any):
    base = getattr(grain, "RandomAccessDataSource", object)
    if base is object:
        return _TrajectoryBatchSource

    class TrajectoryBatchSource(_TrajectoryBatchSource, base):  # type: ignore[misc, valid-type]
        pass

    return TrajectoryBatchSource


def _make_sampler(grain: Any, *, num_records: int, shuffle: bool, seed: int):
    if not hasattr(grain, "IndexSampler"):
        raise GrainUnavailableError("Installed Grain package does not expose IndexSampler.")
    shard_options = grain.NoSharding() if hasattr(grain, "NoSharding") else None
    kwargs = {
        "num_records": int(num_records),
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "num_epochs": None,
    }
    if shard_options is not None:
        kwargs["shard_options"] = shard_options
    try:
        return grain.IndexSampler(**kwargs)
    except TypeError:
        kwargs.pop("shard_options", None)
        return grain.IndexSampler(**kwargs)


def create_grain_trajectory_loader(
    chunk_paths: Sequence[str],
    *,
    batch_size: int,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    worker_count: int = 0,
    prefetch_batches: int = 0,
    horizon: int = 1,
    include_metadata: bool = False,
    batch_view: str = "joint_latent_sasa",
    legal_lmax: int = 128,
    shard_cache_size: int = 2,
) -> Iterable[dict[str, np.ndarray]]:
    """Create a Grain DataLoader for local trajectory-v2/v3 NPZ shards.

    The existing `LeelaChunkDataLoader` remains the default. This path is a
    pragmatic first Grain integration for already-local trajectory shards.
    """
    grain = _import_grain()
    source_cls = _make_source_class(grain)
    source = source_cls(
        chunk_paths,
        batch_view=batch_view,
        horizon=horizon,
        legal_lmax=legal_lmax,
        include_metadata=include_metadata,
        shard_cache_size=shard_cache_size,
    )
    sampler = _make_sampler(grain, num_records=len(source), shuffle=shuffle, seed=seed)
    if not hasattr(grain, "Batch"):
        raise GrainUnavailableError("Installed Grain package does not expose Batch.")
    operations = [grain.Batch(batch_size=int(batch_size), drop_remainder=bool(drop_last))]

    loader_kwargs = {
        "data_source": source,
        "sampler": sampler,
        "operations": operations,
    }
    if int(worker_count) > 0:
        loader_kwargs["worker_count"] = int(worker_count)
    if int(prefetch_batches) > 0 and hasattr(grain, "ReadOptions"):
        try:
            loader_kwargs["read_options"] = grain.ReadOptions(
                prefetch_buffer_size=int(prefetch_batches)
            )
        except TypeError:
            pass
    if not hasattr(grain, "DataLoader"):
        raise GrainUnavailableError("Installed Grain package does not expose DataLoader.")
    return grain.DataLoader(**loader_kwargs)


class _EpochShuffledGrainBatchLoader:
    """Finite-epoch Grain loader that reshuffles shard order on each `iter()`."""

    def __init__(
        self,
        chunk_paths: Sequence[str],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
        worker_count: int,
        prefetch_batches: int,
        horizon: int,
        include_metadata: bool,
        batch_view: str,
        legal_lmax: int,
        shard_cache_size: int,
    ):
        self.chunk_paths = [str(Path(path)) for path in chunk_paths if str(path).endswith(".npz")]
        if not self.chunk_paths:
            raise ValueError("Grain trajectory loading requires at least one local .npz shard.")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.worker_count = int(worker_count)
        self.prefetch_batches = int(prefetch_batches)
        self.horizon = int(horizon)
        self.include_metadata = bool(include_metadata)
        self.batch_view = batch_view
        self.legal_lmax = int(legal_lmax)
        self.shard_cache_size = int(shard_cache_size)
        self._epoch = 0

    def __iter__(self):
        grain = _import_grain()
        paths = list(self.chunk_paths)
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(paths)
        self._epoch += 1

        specs: list[tuple[str, int, int]] = []
        for path in paths:
            count = _row_count(path)
            stop = count if not self.drop_last else (count // self.batch_size) * self.batch_size
            for start in range(0, stop, self.batch_size):
                end = min(start + self.batch_size, count)
                if end - start == self.batch_size or not self.drop_last:
                    specs.append((path, start, end))

        source_cls = _make_batch_source_class(grain)
        source = source_cls(
            specs,
            batch_view=self.batch_view,
            horizon=self.horizon,
            legal_lmax=self.legal_lmax,
            include_metadata=self.include_metadata,
            shard_cache_size=self.shard_cache_size,
        )
        sampler = _make_sampler(grain, num_records=len(source), shuffle=False, seed=self.seed + self._epoch)
        loader_kwargs = {
            "data_source": source,
            "sampler": sampler,
            "operations": (),
        }
        if self.worker_count > 0:
            loader_kwargs["worker_count"] = self.worker_count
        if self.prefetch_batches > 0 and hasattr(grain, "ReadOptions"):
            try:
                loader_kwargs["read_options"] = grain.ReadOptions(
                    prefetch_buffer_size=max(self.prefetch_batches, 16)
                )
            except TypeError:
                pass
        return iter(grain.DataLoader(**loader_kwargs))


def create_grain_trajectory_batch_loader(
    chunk_paths: Sequence[str],
    *,
    batch_size: int,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    worker_count: int = 0,
    prefetch_batches: int = 0,
    horizon: int = 1,
    include_metadata: bool = False,
    batch_view: str = "joint_latent_sasa",
    legal_lmax: int = 128,
    shard_cache_size: int = 2,
) -> Iterable[dict[str, np.ndarray]]:
    """Create the fast Grain path for local trajectory NPZ shards.

    This returns one already-batched item per Grain sample. It is the preferred
    mode for compressed shard files because it preserves shard-local sequential
    reads and avoids repeated per-row NPZ decoding.
    """
    _import_grain()
    return _EpochShuffledGrainBatchLoader(
        chunk_paths,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
        worker_count=worker_count,
        prefetch_batches=prefetch_batches,
        horizon=horizon,
        include_metadata=include_metadata,
        batch_view=batch_view,
        legal_lmax=legal_lmax,
        shard_cache_size=shard_cache_size,
    )


__all__ = [
    "GrainUnavailableError",
    "create_grain_trajectory_batch_loader",
    "create_grain_trajectory_loader",
    "grain_available",
]
