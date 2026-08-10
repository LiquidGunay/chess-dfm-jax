#!/usr/bin/env python3
"""Stream an official LC0 tar into resumable, memory-mappable game shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import uuid
import gzip


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.data.lc0_sequential import (  # noqa: E402
    LC0_SEQUENTIAL_V1,
    SequentialGame,
    has_standard_initial_position,
    records_to_sequential_game,
    save_sequential_shard,
)
from chess_dfm_jax.data.leela import iter_records  # noqa: E402


SPLITS = ("train", "validation", "test")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _split_for_game(
    name: str,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    digest = hashlib.blake2b(
        f"{seed}\0{name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)
    if unit < test_fraction:
        return "test"
    if unit < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _source_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_progress(source: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "lc0-sequential-conversion-progress-v1",
        "source": source,
        "next_game_index": 0,
        "next_chunk_index": 0,
        "processed_games": 0,
        "committed_games": 0,
        "skipped_nonstandard_games": 0,
        "committed_positions": 0,
        "split_games": {split: 0 for split in SPLITS},
        "split_positions": {split: 0 for split in SPLITS},
        "started_unix": time.time(),
        "updated_unix": time.time(),
        "completed": False,
    }


def _load_or_initialize_progress(
    output: Path,
    *,
    source: dict[str, object],
    resume: bool,
) -> dict[str, object]:
    progress_path = output / "progress.json"
    if output.exists() and not resume:
        raise FileExistsError(
            f"Output already exists; pass --resume after inspection: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "chunks").mkdir(exist_ok=True)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("source") != source:
            raise ValueError("Resume source identity does not match the original archive")
    else:
        progress = _initial_progress(source)
        _atomic_json(progress_path, progress)

    # A chunk rename can complete immediately before progress.json is replaced.
    # Roll such committed chunks forward by reading their immutable receipt.
    while True:
        chunk_index = int(progress["next_chunk_index"])
        receipt_path = (
            output
            / "chunks"
            / f"chunk-{chunk_index:05d}"
            / "chunk_manifest.json"
        )
        if not receipt_path.exists():
            break
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if int(receipt["start_game_index"]) != int(progress["next_game_index"]):
            raise ValueError(f"Committed chunk receipt drift at {receipt_path}")
        progress = _advance_progress(progress, receipt)
        _atomic_json(progress_path, progress)
    return progress


def _advance_progress(
    progress: dict[str, object],
    receipt: dict[str, object],
) -> dict[str, object]:
    result = json.loads(json.dumps(progress))
    result["next_game_index"] = int(receipt["end_game_index"])
    result["next_chunk_index"] = int(receipt["chunk_index"]) + 1
    result["processed_games"] = int(result.get("processed_games", 0)) + int(
        receipt["source_game_count"]
    )
    result["committed_games"] = int(result["committed_games"]) + int(
        receipt["game_count"]
    )
    result["skipped_nonstandard_games"] = int(
        result.get("skipped_nonstandard_games", 0)
    ) + int(receipt["skipped_nonstandard_count"])
    result["committed_positions"] = int(result["committed_positions"]) + int(
        receipt["position_count"]
    )
    for split in SPLITS:
        split_receipt = receipt["splits"].get(split)
        if split_receipt is None:
            continue
        result["split_games"][split] = int(result["split_games"][split]) + int(
            split_receipt["game_count"]
        )
        result["split_positions"][split] = int(
            result["split_positions"][split]
        ) + int(split_receipt["position_count"])
    result["updated_unix"] = time.time()
    return result


def _write_chunk(
    output: Path,
    buffers: dict[str, list[SequentialGame]],
    *,
    chunk_index: int,
    start_game_index: int,
    end_game_index: int,
    skipped_nonstandard_count: int,
) -> dict[str, object]:
    chunks_root = output / "chunks"
    target = chunks_root / f"chunk-{chunk_index:05d}"
    if target.exists():
        raise FileExistsError(target)
    temporary = chunks_root / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    split_receipts: dict[str, object] = {}
    game_count = 0
    position_count = 0
    try:
        for split in SPLITS:
            games = buffers[split]
            if not games:
                continue
            receipt = save_sequential_shard(
                games,
                temporary / split,
                split=split,
                shard_index=chunk_index,
            )
            split_receipts[split] = receipt
            game_count += len(games)
            position_count += sum(game.position_count for game in games)
        receipt: dict[str, object] = {
            "schema_version": "lc0-sequential-chunk-receipt-v1",
            "chunk_index": chunk_index,
            "start_game_index": start_game_index,
            "end_game_index": end_game_index,
            "source_game_count": end_game_index - start_game_index,
            "game_count": game_count,
            "skipped_nonstandard_count": skipped_nonstandard_count,
            "position_count": position_count,
            "splits": split_receipts,
        }
        (temporary / "chunk_manifest.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        raise
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions-per-chunk", type=int, default=32768)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=20260810)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path = args.input_tar.resolve()
    output = args.output_dir.resolve()
    if args.positions_per_chunk < 2:
        raise ValueError("--positions-per-chunk must be at least 2")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if not 0.0 <= args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1)")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1")
    if args.max_games is not None and args.max_games < 1:
        raise ValueError("--max-games must be positive")

    source = _source_identity(source_path)
    progress = _load_or_initialize_progress(
        output,
        source=source,
        resume=args.resume,
    )
    if bool(progress["completed"]):
        print(json.dumps({"status": "already-complete", **progress}), flush=True)
        return 0

    with tarfile.open(source_path, mode="r:") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".gz")
        ]
        selected_count = len(members)
        if args.max_games is not None:
            selected_count = min(selected_count, args.max_games)
        next_game = int(progress["next_game_index"])
        if next_game > selected_count:
            raise ValueError("Resume cursor exceeds the selected archive game count")

        while next_game < selected_count:
            chunk_start = next_game
            buffers: dict[str, list[SequentialGame]] = {
                split: [] for split in SPLITS
            }
            buffered_positions = 0
            skipped_nonstandard = 0
            while (
                next_game < selected_count
                and buffered_positions < args.positions_per_chunk
            ):
                member = members[next_game]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not extract archive member {member.name}")
                with extracted, gzip.GzipFile(fileobj=extracted, mode="rb") as decompressed:
                    records = tuple(
                        iter_records(
                            decompressed,
                            include_probabilities=True,
                        )
                    )
                next_game += 1
                if not has_standard_initial_position(records):
                    skipped_nonstandard += 1
                    continue
                game = records_to_sequential_game(records, name=member.name)
                split = _split_for_game(
                    member.name,
                    seed=args.split_seed,
                    validation_fraction=args.validation_fraction,
                    test_fraction=args.test_fraction,
                )
                buffers[split].append(game)
                buffered_positions += game.position_count

            chunk_index = int(progress["next_chunk_index"])
            receipt = _write_chunk(
                output,
                buffers,
                chunk_index=chunk_index,
                start_game_index=chunk_start,
                end_game_index=next_game,
                skipped_nonstandard_count=skipped_nonstandard,
            )
            progress = _advance_progress(progress, receipt)
            _atomic_json(output / "progress.json", progress)
            elapsed = time.time() - float(progress["started_unix"])
            print(
                json.dumps(
                    {
                        "status": "chunk-committed",
                        "chunk_index": chunk_index,
                        "games": progress["committed_games"],
                        "processed_games": progress["processed_games"],
                        "skipped_nonstandard_games": progress[
                            "skipped_nonstandard_games"
                        ],
                        "positions": progress["committed_positions"],
                        "selected_games": selected_count,
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    progress["completed"] = True
    progress["updated_unix"] = time.time()
    progress["archive_game_count"] = len(members)
    progress["selected_game_count"] = selected_count
    progress["schema_version"] = "lc0-sequential-conversion-progress-v1"
    _atomic_json(output / "progress.json", progress)
    dataset_manifest: dict[str, object] = {
        "schema_version": LC0_SEQUENTIAL_V1,
        "source": source,
        "source_sha256": _sha256_file(source_path),
        "archive_game_count": len(members),
        "selected_game_count": selected_count,
        "positions_per_chunk": args.positions_per_chunk,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "split_policy": "blake2b-64-over-seed-null-member-name",
        "variant_filter": "standard-initial-position-only",
        "completed_unix": time.time(),
        "totals": {
            "games": progress["committed_games"],
            "processed_games": progress["processed_games"],
            "skipped_nonstandard_games": progress["skipped_nonstandard_games"],
            "positions": progress["committed_positions"],
            "split_games": progress["split_games"],
            "split_positions": progress["split_positions"],
        },
    }
    _atomic_json(output / "dataset_manifest.json", dataset_manifest)
    print(json.dumps({"status": "complete", **dataset_manifest}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
