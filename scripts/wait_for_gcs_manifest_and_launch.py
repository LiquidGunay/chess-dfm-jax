#!/usr/bin/env python3
"""Wait for a GCS manifest, then launch a command."""

from __future__ import annotations

import argparse
import subprocess
import time


def gcloud_ok(args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    return proc.returncode == 0, proc.stdout.strip()


def count_npz(prefix: str, split: str) -> int:
    ok, stdout = gcloud_ok(["gcloud", "storage", "ls", f"{prefix.rstrip('/')}/{split}/*.npz"])
    if not ok:
        return 0
    return len([line for line in stdout.splitlines() if line.endswith(".npz")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--poll-s", type=int, default=600)
    parser.add_argument("--command", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        ok, _stdout = gcloud_ok(["gcloud", "storage", "cat", args.manifest_uri])
        if ok:
            print(f"manifest_ready={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
            return subprocess.run(args.command, shell=True, check=False).returncode
        train_count = count_npz(args.dataset_prefix, "train")
        val_count = count_npz(args.dataset_prefix, "val")
        print(
            " ".join(
                [
                    f"waiting_for_manifest={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                    f"train_chunks={train_count}",
                    f"val_chunks={val_count}",
                ]
            ),
            flush=True,
        )
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
