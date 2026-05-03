#!/usr/bin/env python3
"""Wait for a local shard cache to complete, then launch an experiment queue."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file())


def _warm_process_running() -> bool:
    proc = subprocess.run(
        ["pgrep", "-af", "[w]arm_gcs_cache"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _log(message: str) -> None:
    timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    print(f"{timestamp} {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-val", type=int, required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--status-dir", required=True)
    parser.add_argument("--status-uri", default="")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=float, default=14400.0)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    train_dir = cache_dir / "train"
    val_dir = cache_dir / "val"
    deadline = time.monotonic() + args.timeout_s

    Path(args.status_dir).mkdir(parents=True, exist_ok=True)

    while True:
        train_count = _count_files(train_dir)
        val_count = _count_files(val_dir)
        _log(
            "cache_counts "
            f"train={train_count}/{args.expected_train} "
            f"val={val_count}/{args.expected_val}"
        )
        if train_count == args.expected_train and val_count == args.expected_val:
            break
        if time.monotonic() >= deadline:
            _log("cache_wait_timeout")
            return 1
        if not _warm_process_running():
            _log("cache_warmer_not_running_and_cache_incomplete")
            return 1
        time.sleep(args.poll_s)

    command = [
        args.python,
        "-u",
        "scripts/run_experiment_queue.py",
        "--queue",
        args.queue,
        "--workdir",
        args.workdir,
        "--status-dir",
        args.status_dir,
    ]
    if args.status_uri:
        command.extend(["--status-uri", args.status_uri])
    if args.env_file:
        command.extend(["--env-file", args.env_file])
    if args.keep_going:
        command.append("--keep-going")

    _log("launching_queue " + " ".join(command))
    return subprocess.call(command, cwd=args.workdir, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
