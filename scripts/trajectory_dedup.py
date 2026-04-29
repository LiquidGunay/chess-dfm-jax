#!/usr/bin/env python3
"""Compute duplicate statistics and write deduplicated trajectory-v2 shards."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Iterable, Mapping

import numpy as np


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetPrefix:
    name: str
    uri: str


def run_gcloud(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def parse_dataset_prefix(value: str) -> DatasetPrefix:
    if "=" in value:
        name, uri = value.split("=", 1)
        return DatasetPrefix(name=name.strip(), uri=uri.strip().rstrip("/"))
    uri = value.strip().rstrip("/")
    return DatasetPrefix(name=Path(uri).name or "dataset", uri=uri)


def list_npz(prefix: str, split: str) -> list[str]:
    split_prefix = f"{prefix.rstrip('/')}/{split}"
    if split_prefix.startswith("gs://"):
        result = run_gcloud(["gcloud", "storage", "ls", f"{split_prefix}/*.npz"])
        if result.returncode != 0:
            return []
        return sorted(line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".npz"))
    return sorted(str(path) for path in Path(split_prefix).glob("*.npz"))


def copy_to_local(uri: str, work_dir: Path) -> Path:
    if not uri.startswith("gs://"):
        return Path(uri)
    local = work_dir / uri.rsplit("/", 1)[-1]
    result = run_gcloud(["gcloud", "storage", "cp", uri, str(local), "--quiet"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to copy {uri}: {result.stderr}")
    return local


def upload_file(local: Path, uri: str) -> None:
    result = run_gcloud(["gcloud", "storage", "cp", str(local), uri, "--quiet"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to upload {local} to {uri}: {result.stderr}")


ArrayMap = Mapping[str, np.ndarray]


def sample_position_bytes(data: ArrayMap, idx: int) -> bytes:
    if "fen_t" in data:
        return str(data["fen_t"][idx]).encode("utf-8")
    return hashlib.blake2b(np.ascontiguousarray(data["planes_t"][idx]).view(np.uint8), digest_size=16).digest()


def sample_actions_bytes(data: ArrayMap, idx: int) -> bytes:
    if "actions_uci" in data:
        return "|".join(str(item) for item in data["actions_uci"][idx]).encode("utf-8")
    return np.ascontiguousarray(data["actions"][idx]).view(np.uint8).tobytes()


def sample_key(data: ArrayMap, idx: int, mode: str) -> bytes:
    position = sample_position_bytes(data, idx)
    if mode == "position":
        payload = b"p\0" + position
    elif mode == "position_first_action":
        first = np.asarray(data["actions"][idx, 0], dtype=np.int32).tobytes()
        payload = b"pf\0" + position + b"\0" + first
    elif mode == "position_actions":
        payload = b"pa\0" + position + b"\0" + sample_actions_bytes(data, idx)
    else:
        raise ValueError(f"Unknown dedup key mode: {mode}")
    return hashlib.blake2b(payload, digest_size=16).digest()


def position_hash(data: ArrayMap, idx: int) -> bytes:
    return hashlib.blake2b(b"p\0" + sample_position_bytes(data, idx), digest_size=16).digest()


def first_action_id(data: ArrayMap, idx: int) -> int:
    return int(np.asarray(data["actions"][idx, 0], dtype=np.int32))


def stats_for_split(
    inputs: list[DatasetPrefix],
    split: str,
    *,
    key_mode: str,
    sample_top_k: int,
    work_dir: Path,
    progress_every: int,
) -> dict:
    seen_keys: set[bytes] = set()
    seen_positions: set[bytes] = set()
    position_first_action: dict[bytes, int] = {}
    conflict_positions: set[bytes] = set()
    duplicate_counts: Counter[bytes] = Counter()
    totals_by_dataset: dict[str, int] = {}
    shard_count = 0
    total = 0

    for dataset in inputs:
        dataset_total = 0
        for uri in list_npz(dataset.uri, split):
            shard_count += 1
            if progress_every > 0 and shard_count % progress_every == 0:
                print(f"stats_progress split={split} shards={shard_count} samples={total}", flush=True)
            local = copy_to_local(uri, work_dir)
            with np.load(local, allow_pickle=False) as data:
                key_data = {
                    key: data[key]
                    for key in ("fen_t", "actions_uci", "actions", "planes_t")
                    if key in data.files
                }
                batch_size = int(key_data["actions"].shape[0])
                for idx in range(batch_size):
                    total += 1
                    dataset_total += 1
                    key = sample_key(key_data, idx, key_mode)
                    duplicate_counts[key] += 1
                    seen_keys.add(key)

                    pos = position_hash(key_data, idx)
                    seen_positions.add(pos)
                    action = first_action_id(key_data, idx)
                    previous = position_first_action.setdefault(pos, action)
                    if previous != action:
                        conflict_positions.add(pos)
            if uri.startswith("gs://"):
                local.unlink(missing_ok=True)
        totals_by_dataset[dataset.name] = dataset_total

    top_duplicates = [
        {"hash": key.hex(), "count": count}
        for key, count in duplicate_counts.most_common(sample_top_k)
        if count > 1
    ]
    unique = len(seen_keys)
    return {
        "split": split,
        "key_mode": key_mode,
        "shards": shard_count,
        "samples": total,
        "unique_keys": unique,
        "duplicate_samples": total - unique,
        "duplicate_fraction": (total - unique) / total if total else 0.0,
        "unique_positions": len(seen_positions),
        "position_duplicate_samples": total - len(seen_positions),
        "position_conflict_count": len(conflict_positions),
        "samples_by_dataset": totals_by_dataset,
        "top_duplicate_keys": top_duplicates,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_stats(args: argparse.Namespace) -> int:
    inputs = [parse_dataset_prefix(value) for value in args.input_prefix]
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="trajectory-dedup-stats-") as tmp:
        work_dir = Path(tmp)
        splits = args.splits.split(",") if args.splits else SPLITS
        split_stats = [
            stats_for_split(
                inputs,
                split,
                key_mode=args.key,
                sample_top_k=args.top_k,
                work_dir=work_dir,
                progress_every=args.progress_every,
            )
            for split in splits
        ]

    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
        "inputs": [dataset.__dict__ for dataset in inputs],
        "splits": split_stats,
        "total_samples": sum(item["samples"] for item in split_stats),
        "total_unique_keys": sum(item["unique_keys"] for item in split_stats),
        "total_duplicate_samples": sum(item["duplicate_samples"] for item in split_stats),
        "key_mode": args.key,
    }
    output_path = Path(args.output_json)
    write_json(output_path, payload)
    if args.output_gcs:
        upload_file(output_path, args.output_gcs)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def append_block(
    buffers: dict[str, list[np.ndarray]],
    scalars: dict[str, np.ndarray],
    data: ArrayMap,
    indices: np.ndarray,
) -> None:
    batch_size = int(data["planes_t"].shape[0])
    for key in data.keys():
        value = data[key]
        if value.shape == ():
            scalars.setdefault(key, np.asarray(value))
        elif value.shape[0] == batch_size:
            buffers.setdefault(key, []).append(np.asarray(value[indices]))


def flush_buffers(
    *,
    buffers: dict[str, list[np.ndarray]],
    scalars: dict[str, np.ndarray],
    local_dir: Path,
    output_prefix: str,
    split: str,
    chunk_index: int,
    keep_local: bool,
    executor: ThreadPoolExecutor | None,
    pending: list[Future],
    max_pending: int,
) -> int:
    if not buffers:
        return chunk_index
    arrays = {key: np.concatenate(values, axis=0) for key, values in buffers.items()}
    arrays.update(scalars)
    if executor is None:
        write_chunk(
            arrays=arrays,
            local_dir=local_dir,
            output_prefix=output_prefix,
            split=split,
            chunk_index=chunk_index,
            keep_local=keep_local,
        )
    else:
        pending.append(
            executor.submit(
                write_chunk,
                arrays=arrays,
                local_dir=local_dir,
                output_prefix=output_prefix,
                split=split,
                chunk_index=chunk_index,
                keep_local=keep_local,
            )
        )
        drain_pending(pending, max_pending=max_pending)
    buffers.clear()
    return chunk_index + 1


def write_chunk(
    *,
    arrays: dict[str, np.ndarray],
    local_dir: Path,
    output_prefix: str,
    split: str,
    chunk_index: int,
    keep_local: bool,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    out_path = local_dir / f"chunk_{chunk_index:06d}.npz"
    np.savez_compressed(out_path, **arrays)
    if output_prefix:
        upload_file(out_path, f"{output_prefix.rstrip('/')}/{split}/chunk_{chunk_index:06d}.npz")
    if output_prefix and not keep_local:
        out_path.unlink(missing_ok=True)


def drain_pending(pending: list[Future], *, max_pending: int | None = None) -> None:
    if not pending:
        return
    if max_pending is None:
        done = set(pending)
    elif len(pending) < max_pending:
        return
    else:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in list(done):
        future.result()
        pending.remove(future)


def iter_split_shards(inputs: Iterable[DatasetPrefix], split: str) -> Iterable[tuple[DatasetPrefix, str]]:
    for dataset in inputs:
        for uri in list_npz(dataset.uri, split):
            yield dataset, uri


def command_write(args: argparse.Namespace) -> int:
    inputs = [parse_dataset_prefix(value) for value in args.input_prefix]
    out_dir = Path(args.local_out_dir)
    if out_dir.exists() and args.clean_local_out:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    split_payloads = []
    executor = (
        ThreadPoolExecutor(max_workers=args.flush_workers)
        if args.flush_workers > 1
        else None
    )
    pending: list[Future] = []
    with tempfile.TemporaryDirectory(prefix="trajectory-dedup-write-") as tmp:
        work_dir = Path(tmp)
        for split in args.splits.split(",") if args.splits else SPLITS:
            seen: set[bytes] = set()
            buffers: dict[str, list[np.ndarray]] = {}
            scalars: dict[str, np.ndarray] = {}
            buffered = 0
            chunk_index = 0
            written = 0
            read = 0
            for _dataset, uri in iter_split_shards(inputs, split):
                local = copy_to_local(uri, work_dir)
                with np.load(local, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
                    batch_size = int(arrays["planes_t"].shape[0])
                    keep_indices = []
                    for idx in range(batch_size):
                        read += 1
                        key = sample_key(arrays, idx, args.key)
                        if key in seen:
                            continue
                        seen.add(key)
                        keep_indices.append(idx)
                        written += 1
                    keep = np.asarray(keep_indices, dtype=np.int64)
                    offset = 0
                    while offset < keep.shape[0]:
                        space = args.samples_per_chunk - buffered
                        part = keep[offset : offset + space]
                        append_block(buffers, scalars, arrays, part)
                        buffered += int(part.shape[0])
                        offset += int(part.shape[0])
                        if buffered >= args.samples_per_chunk:
                            chunk_index = flush_buffers(
                                buffers=buffers,
                                scalars=scalars,
                                local_dir=out_dir / split,
                                output_prefix=args.output_prefix,
                                split=split,
                                chunk_index=chunk_index,
                                keep_local=args.keep_local,
                                executor=executor,
                                pending=pending,
                                max_pending=args.max_pending_flushes,
                            )
                            buffered = 0
                if uri.startswith("gs://"):
                    local.unlink(missing_ok=True)
            chunk_index = flush_buffers(
                buffers=buffers,
                scalars=scalars,
                local_dir=out_dir / split,
                output_prefix=args.output_prefix,
                split=split,
                chunk_index=chunk_index,
                keep_local=args.keep_local,
                executor=executor,
                pending=pending,
                max_pending=args.max_pending_flushes,
            )
            drain_pending(pending, max_pending=None)
            buffered = 0
            split_payloads.append(
                {
                    "split": split,
                    "samples_read": read,
                    "samples_written": written,
                    "samples_removed": read - written,
                    "chunks_written": chunk_index,
                }
            )
            print(json.dumps(split_payloads[-1], sort_keys=True))
    drain_pending(pending, max_pending=None)
    if executor is not None:
        executor.shutdown(wait=True)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
        "inputs": [dataset.__dict__ for dataset in inputs],
        "dedup_key": args.key,
        "samples_per_chunk": args.samples_per_chunk,
        "splits": split_payloads,
        "total_samples_read": sum(item["samples_read"] for item in split_payloads),
        "total_samples_written": sum(item["samples_written"] for item in split_payloads),
        "schema_version": "trajectory-v2-dedup",
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    if args.output_prefix:
        upload_file(manifest_path, f"{args.output_prefix.rstrip('/')}/manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser("stats")
    stats.add_argument("--input-prefix", action="append", required=True, help="Name and URI as name=prefix, repeated.")
    stats.add_argument("--splits", default="train,val,test")
    stats.add_argument("--key", choices=["position", "position_first_action", "position_actions"], default="position_actions")
    stats.add_argument("--top-k", type=int, default=20)
    stats.add_argument("--progress-every", type=int, default=100)
    stats.add_argument("--output-json", required=True)
    stats.add_argument("--output-gcs", default="")
    stats.set_defaults(func=command_stats)

    write = subparsers.add_parser("write-dedup")
    write.add_argument("--input-prefix", action="append", required=True, help="Name and URI as name=prefix, repeated.")
    write.add_argument("--splits", default="train,val,test")
    write.add_argument("--key", choices=["position", "position_first_action", "position_actions"], default="position_actions")
    write.add_argument("--samples-per-chunk", type=int, default=1024)
    write.add_argument("--local-out-dir", default="/tmp/trajectory_v2_dedup")
    write.add_argument("--output-prefix", default="")
    write.add_argument("--keep-local", action="store_true")
    write.add_argument("--clean-local-out", action="store_true")
    write.add_argument("--flush-workers", type=int, default=1, help="Parallel chunk compression/upload workers.")
    write.add_argument("--max-pending-flushes", type=int, default=8)
    write.set_defaults(func=command_write)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
