#!/usr/bin/env python3
"""Convert dense trajectory-v2 shards to compact trajectory-v3 shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.data.trajectory_v3 import (  # noqa: E402
    TRAJECTORY_V3,
    trajectory_v3_from_v2_npz,
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")


def split_prefixes(raw: str) -> list[str]:
    return [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]


def list_npz(prefix: str, split: str) -> list[str]:
    if is_gcs(prefix):
        result = run(["gcloud", "storage", "ls", f"{prefix.rstrip('/')}/{split}/*.npz"], check=False)
        if result.returncode != 0:
            return []
        return sorted(line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".npz"))
    root = Path(prefix) / split
    return sorted(str(path) for path in root.glob("*.npz"))


def copy_from_source(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_gcs(source):
        run(["gcloud", "storage", "cp", source, str(destination)])
    else:
        shutil.copy2(source, destination)


def copy_to_destination(local_path: Path, destination: str) -> None:
    if is_gcs(destination):
        run(["gcloud", "storage", "cp", str(local_path), destination])
    else:
        out = Path(destination)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, out)


def write_json(payload: dict[str, Any], destination: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if is_gcs(destination):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            handle.write(text)
            temp_name = handle.name
        try:
            run(["gcloud", "storage", "cp", temp_name, destination])
        finally:
            Path(temp_name).unlink(missing_ok=True)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def output_uri(output_prefix: str, split: str, filename: str) -> str:
    return f"{output_prefix.rstrip('/')}/{split}/{filename}" if is_gcs(output_prefix) else str(Path(output_prefix) / split / filename)


def source_hash(source: str) -> str:
    parent = source.rsplit("/", 1)[0]
    return hashlib.sha1(parent.encode("utf-8")).hexdigest()[:8]


def convert_one(
    *,
    source: str,
    destination: str,
    work_dir: Path,
    legal_lmax: int,
    strict_binary_planes: bool,
    plane_codec: str,
    include_metadata: bool,
    keep_local: bool,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    local_in = work_dir / "in" / f"{source_hash(source)}_{Path(source).name}"
    local_out = work_dir / "out" / Path(destination).name
    copy_from_source(source, local_in)
    with np.load(local_in, allow_pickle=False) as data:
        payload = trajectory_v3_from_v2_npz(
            data,
            legal_lmax=legal_lmax,
            strict_binary_planes=strict_binary_planes,
            plane_codec=plane_codec,
            include_metadata=include_metadata,
        )
    payload["source_uri"] = np.asarray(source)
    local_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(local_out, **payload)
    copy_to_destination(local_out, destination)
    source_size = local_in.stat().st_size
    output_size = local_out.stat().st_size
    if not keep_local:
        local_in.unlink(missing_ok=True)
        local_out.unlink(missing_ok=True)
    return {
        "source": source,
        "destination": destination,
        "source_size_bytes": source_size,
        "output_size_bytes": output_size,
        "samples": int(payload["batch_size"]),
        "horizon": int(payload["horizon"]),
    }


def build_work_items(args: argparse.Namespace) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        split_sources: list[str] = []
        for prefix in split_prefixes(args.input_prefix):
            split_sources.extend(list_npz(prefix, split))
        if args.limit_per_split > 0:
            split_sources = split_sources[: args.limit_per_split]
        for idx, source in enumerate(split_sources):
            filename = f"chunk_{idx:06d}.npz"
            items.append({"split": split, "source": source, "destination": output_uri(args.output_prefix, split, filename)})
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", required=True, help="Comma-separated local or gs:// trajectory-v2 roots.")
    parser.add_argument("--output-prefix", required=True, help="Local or gs:// trajectory-v3 root.")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--legal-lmax", type=int, default=128)
    parser.add_argument("--local-work-dir", default="/tmp/trajectory_v3_convert")
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--status-every", type=int, default=25)
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--include-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plane-codec", choices=["uint8", "packbits"], default="uint8")
    parser.add_argument("--strict-binary-planes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    work_items = build_work_items(args)
    if not work_items:
        raise SystemExit("No input shards found.")
    work_root = Path(args.local_work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    splits = sorted({item["split"] for item in work_items})
    manifest: dict[str, Any] = {
        "schema_version": TRAJECTORY_V3,
        "source_schema_version": "trajectory-v2",
        "input_prefix": split_prefixes(args.input_prefix),
        "output_prefix": args.output_prefix.rstrip("/"),
        "splits": splits,
        "legal_lmax": args.legal_lmax,
        "include_metadata": args.include_metadata,
        "plane_codec": args.plane_codec,
        "strict_binary_planes": args.strict_binary_planes,
        "created_utc": utc_now(),
        "items_total": len(work_items),
        "items_completed": 0,
        "items_failed": 0,
        "samples_written": {split: 0 for split in splits},
        "chunks_written": {split: 0 for split in splits},
        "source_size_bytes": 0,
        "output_size_bytes": 0,
    }
    status_uri = f"{args.output_prefix.rstrip('/')}/status.json"
    write_json({**manifest, "state": "running"}, status_uri)

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for item in work_items:
            shard_work_dir = work_root / item["split"] / Path(item["destination"]).stem
            future = executor.submit(
                convert_one,
                source=item["source"],
                destination=item["destination"],
                work_dir=shard_work_dir,
                legal_lmax=args.legal_lmax,
                strict_binary_planes=args.strict_binary_planes,
                plane_codec=args.plane_codec,
                include_metadata=args.include_metadata,
                keep_local=args.keep_local,
            )
            futures[future] = item

        for future in as_completed(futures):
            item = futures[future]
            split = item["split"]
            try:
                result = future.result()
            except Exception as exc:
                failure = {"split": split, "source": item["source"], "error": repr(exc)}
                failures.append(failure)
                manifest["items_failed"] += 1
                print(json.dumps({"event": "failed", **failure}), flush=True)
            else:
                manifest["items_completed"] += 1
                manifest["chunks_written"][split] += 1
                manifest["samples_written"][split] += int(result["samples"])
                manifest["source_size_bytes"] += int(result["source_size_bytes"])
                manifest["output_size_bytes"] += int(result["output_size_bytes"])
                print(json.dumps({"event": "converted", "split": split, **result}), flush=True)

            completed = int(manifest["items_completed"]) + int(manifest["items_failed"])
            if completed % max(1, args.status_every) == 0 or completed == len(work_items):
                elapsed_s = time.time() - started
                write_json(
                    {
                        **manifest,
                        "state": "running",
                        "updated_utc": utc_now(),
                        "elapsed_s": elapsed_s,
                        "failures": failures[-10:],
                    },
                    status_uri,
                )

    manifest["finished_utc"] = utc_now()
    manifest["elapsed_s"] = time.time() - started
    manifest["failures"] = failures
    manifest["state"] = "completed" if not failures else "failed"
    write_json(manifest, f"{args.output_prefix.rstrip('/')}/manifest.json")
    write_json(manifest, status_uri)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
