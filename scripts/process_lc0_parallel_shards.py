#!/usr/bin/env python3
"""Run LC0 PGN trajectory conversion in parallel over disjoint archive ranges."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.data.trajectory import TRAJECTORY_V2  # noqa: E402
from scripts.process_lc0_pgns_to_gcs import (  # noqa: E402
    DEFAULT_INDEX_URL,
    _write_json_gcs,
    discover_archive_urls,
    index_url_with_subdir,
)


def chunked(values: list[str], parts: int) -> list[list[str]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    size = math.ceil(len(values) / parts)
    return [values[i * size : (i + 1) * size] for i in range(parts) if values[i * size : (i + 1) * size]]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--subdir", default="test80")
    parser.add_argument("--filename-pattern", default=r".*\.tar\.bz2$")
    parser.add_argument("--skip-archives", type=int, default=0)
    parser.add_argument("--max-archives", type=int, default=640)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--launch-stagger-s", type=float, default=0.0)
    parser.add_argument("--max-chunks-per-worker", type=int, default=1024)
    parser.add_argument("--chunk-index-stride", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--upload-gcs", required=True)
    parser.add_argument("--local-out-dir", default="/tmp/lc0_parallel_chunks")
    parser.add_argument("--cache-dir", default="/tmp/lc0_pgn_cache")
    parser.add_argument("--cache-gcs", default="")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--require-standard-variant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-startpos", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    index_url = index_url_with_subdir(args.index_url, args.subdir)
    all_urls = discover_archive_urls(index_url, args.filename_pattern, 0)
    selected_urls = all_urls[args.skip_archives :]
    if args.max_archives > 0:
        selected_urls = selected_urls[: args.max_archives]
    if not selected_urls:
        raise SystemExit("No LC0 archives selected.")

    root_out = Path(args.local_out_dir)
    root_out.mkdir(parents=True, exist_ok=True)
    groups = chunked(selected_urls, min(args.workers, len(selected_urls)))
    processes: list[tuple[int, subprocess.Popen, Path, Path]] = []

    for worker_idx, urls in enumerate(groups):
        worker_dir = root_out / f"worker_{worker_idx:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        url_file = worker_dir / "urls.txt"
        log_file = worker_dir / "worker.log"
        url_file.write_text("\n".join(urls) + "\n", encoding="utf-8")
        start_chunk_index = worker_idx * args.chunk_index_stride
        command = [
            sys.executable,
            "scripts/process_lc0_pgns_to_gcs.py",
            "--url-file",
            str(url_file),
            "--horizon",
            str(args.horizon),
            "--batch-size",
            str(args.batch_size),
            "--max-chunks",
            str(args.max_chunks_per_worker),
            "--start-chunk-index",
            str(start_chunk_index),
            "--upload-gcs",
            args.upload_gcs,
            "--local-out-dir",
            str(worker_dir / "chunks"),
            "--cache-dir",
            args.cache_dir,
            "--val-fraction",
            str(args.val_fraction),
            "--test-fraction",
            str(args.test_fraction),
        ]
        if args.cache_gcs:
            command.extend(["--cache-gcs", args.cache_gcs])
        if args.keep_local:
            command.append("--keep-local")
        command.append("--require-standard-variant" if args.require_standard_variant else "--no-require-standard-variant")
        command.append("--require-startpos" if args.require_startpos else "--no-require-startpos")
        print(
            json.dumps(
                {
                    "event": "launch_worker",
                    "worker": worker_idx,
                    "archives": len(urls),
                    "start_chunk_index": start_chunk_index,
                    "log_file": str(log_file),
                },
                sort_keys=True,
            )
        )
        sys.stdout.flush()
        log_handle = log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((worker_idx, process, worker_dir, log_file))
        if args.launch_stagger_s > 0 and worker_idx != len(groups) - 1:
            time.sleep(args.launch_stagger_s)

    failures = []
    worker_manifests = []
    for worker_idx, process, worker_dir, log_file in processes:
        returncode = process.wait()
        print(json.dumps({"event": "worker_done", "worker": worker_idx, "returncode": returncode}, sort_keys=True))
        if returncode != 0:
            failures.append((worker_idx, returncode, log_file))
            continue
        manifest_path = worker_dir / "chunks" / "manifest.json"
        if not manifest_path.exists():
            failures.append((worker_idx, "missing_manifest", log_file))
            continue
        worker_manifests.append(load_json(manifest_path))

    if failures:
        print(json.dumps({"event": "failures", "failures": [(idx, code, str(log)) for idx, code, log in failures]}))
        raise SystemExit(1)

    splits = ("train", "val", "test")
    aggregate = {
        "schema_version": TRAJECTORY_V2,
        "source": "lc0_pgn_parallel",
        "index_url": index_url,
        "subdir": args.subdir,
        "filename_pattern": args.filename_pattern,
        "horizon": args.horizon,
        "samples_per_chunk": args.batch_size,
        "skip_archives": args.skip_archives,
        "max_archives": args.max_archives,
        "url_count": len(selected_urls),
        "worker_count": len(worker_manifests),
        "max_chunks_per_worker": args.max_chunks_per_worker,
        "chunk_index_stride": args.chunk_index_stride,
        "chunks_written": {split: sum(int(m["chunks_written"].get(split, 0)) for m in worker_manifests) for split in splits},
        "samples_written": {split: sum(int(m["samples_written"].get(split, 0)) for m in worker_manifests) for split in splits},
        "samples_seen_before_chunking": {
            split: sum(int(m["samples_seen_before_chunking"].get(split, 0)) for m in worker_manifests) for split in splits
        },
        "games_seen": sum(int(m.get("games_seen", 0)) for m in worker_manifests),
        "games_kept": sum(int(m.get("games_kept", 0)) for m in worker_manifests),
        "games_by_split": {split: sum(int(m["games_by_split"].get(split, 0)) for m in worker_manifests) for split in splits},
        "splits": {"train": 1.0 - args.val_fraction - args.test_fraction, "val": args.val_fraction, "test": args.test_fraction},
        "filters": {
            "require_standard_variant": args.require_standard_variant,
            "require_startpos": args.require_startpos,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
        "worker_manifests": worker_manifests,
    }
    manifest_path = root_out / "manifest.json"
    manifest_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_json_gcs(aggregate, f"{args.upload_gcs.rstrip('/')}/manifest.json")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
