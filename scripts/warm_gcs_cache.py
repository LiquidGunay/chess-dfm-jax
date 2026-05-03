#!/usr/bin/env python3
"""Warm a local GCS shard cache and verify manifest-visible shard counts."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from chess_dfm_jax.data.gcs_cache import GCSShardCache


def warm_split(prefix: str, cache_dir: Path, *, workers: int, timeout_s: int) -> dict[str, int]:
    cache = GCSShardCache(
        prefix,
        cache_dir,
        download_workers=workers,
        shuffle_downloads=False,
    )
    cache.wait_for_all_visible(timeout_s=timeout_s)
    return cache.stats()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--val-prefix", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=14_400)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = Path(args.cache_dir)
    if args.clean:
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    for split, prefix in (("train", args.train_prefix), ("val", args.val_prefix)):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"warming {split}: {prefix} -> {split_dir}", flush=True)
        stats = warm_split(prefix, split_dir, workers=args.workers, timeout_s=args.timeout_s)
        print(f"done {split}: {stats}", flush=True)
        if stats["remote_seen"] <= 0:
            raise SystemExit(f"{split} saw no remote shards")
        if stats["cached"] != stats["remote_seen"]:
            raise SystemExit(
                f"{split} cache count mismatch: cached={stats['cached']} remote_seen={stats['remote_seen']}"
            )

    print("cache_warm_complete", flush=True)


if __name__ == "__main__":
    main()
