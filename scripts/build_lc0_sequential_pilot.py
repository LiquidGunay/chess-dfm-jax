"""Build the immutable, evenly spaced Proposal-A LC0 tuning pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


PILOT_SCHEMA = "chess-dfm-modal-lc0-sequential-pilot-v1"
PILOT_DATASET_LABEL = "lc0-sequential-test80-20240401-0117-pilot-v1"
PILOT_CHUNK_INDICES = (0, 32, 64, 96, 128, 160, 192, 224, 253)
FULL_BATCH_1024_EPOCH_EXAMPLES = 7_960_576


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".stage.json"
    ]
    return rows, hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def build_pilot(
    *,
    source_root: Path,
    output_root: Path,
    hardlink: bool = True,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Pilot output already exists: {output_root}")
    source_manifest_path = source_root / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != "lc0-sequential-v1":
        raise ValueError("Source LC0 sequential schema drift")

    partial = output_root.with_name(f".{output_root.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Pilot partial output already exists: {partial}")
    sequential_root = partial / "lc0_sequential"
    sequential_root.mkdir(parents=True)
    shutil.copyfile(
        source_manifest_path,
        sequential_root / "source_dataset_manifest.json",
    )
    copy_function = os.link if hardlink else shutil.copy2
    split_totals = {
        split: {"positions": 0, "trainable_starts": 0, "full_batches": 0}
        for split in ("train", "validation", "test")
    }
    try:
        for index in PILOT_CHUNK_INDICES:
            name = f"chunk-{index:05d}"
            source_chunk = source_root / "chunks" / name
            if not source_chunk.is_dir():
                raise FileNotFoundError(f"Missing selected source chunk: {source_chunk}")
            target_chunk = sequential_root / "chunks" / name
            shutil.copytree(
                source_chunk,
                target_chunk,
                copy_function=copy_function,
            )
            for split, totals in split_totals.items():
                manifest_path = target_chunk / split / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("schema_version") != "lc0-sequential-v1"
                    or manifest.get("split") != split
                ):
                    raise ValueError(f"Selected shard manifest drift: {manifest_path}")
                starts = int(manifest["trainable_start_count"])
                totals["positions"] += int(manifest["position_count"])
                totals["trainable_starts"] += starts
                totals["full_batches"] += starts // (64 if split != "train" else 1024)

        rows, inventory_sha256 = _inventory(partial)
        marker = {
            "schema_version": PILOT_SCHEMA,
            "dataset_label": PILOT_DATASET_LABEL,
            "source_dataset_schema": source_manifest["schema_version"],
            "source_dataset_manifest_sha256": _sha256_file(source_manifest_path),
            "source_archive_sha256": source_manifest["source_sha256"],
            "selected_chunk_indices": list(PILOT_CHUNK_INDICES),
            "selection": "deterministic evenly spaced archive chunks including endpoints",
            "full_batch_1024_epoch_examples": FULL_BATCH_1024_EPOCH_EXAMPLES,
            "split_totals": split_totals,
            "inventory_file_count": len(rows),
            "inventory_size_bytes": sum(int(row["size_bytes"]) for row in rows),
            "inventory_sha256": inventory_sha256,
            "hardlinked_locally": hardlink,
        }
        (partial / ".stage.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output_root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy bytes instead of making local hardlinks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    marker = build_pilot(
        source_root=args.source_root,
        output_root=args.output_root,
        hardlink=not args.copy,
    )
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
