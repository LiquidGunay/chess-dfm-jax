#!/usr/bin/env python3
"""Freeze disjoint validation/test indices and pin the existing arena pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path("/mountpoint/.exp")
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "trajectory_v3"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "eval" / "hero_epoch_v1"
OPENING_POOL = (
    REPO_ROOT
    / "research"
    / "assets"
    / "arena"
    / "promotion-ply12-n2048-v3.json"
)
OPENING_HISTORIES = (
    REPO_ROOT
    / "research"
    / "assets"
    / "arena"
    / "promotion-ply12-n2048-histories-v3.json"
)
OPENING_CONTRACT = (
    REPO_ROOT
    / "research"
    / "assets"
    / "arena"
    / "promotion-ply12-n2048-v3-contract.json"
)


def require_workspace(path: Path, *, exists: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=exists)
    try:
        resolved.relative_to(WORKSPACE_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"Path escapes {WORKSPACE_ROOT}: {resolved}") from exc
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_inventory(split_dir: Path) -> dict[str, Any]:
    paths = sorted(split_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shards under {split_dir}")
    entries = [f"{path.name}\t{path.stat().st_size}" for path in paths]
    with np.load(paths[0], allow_pickle=False) as first:
        samples_per_shard = int(np.asarray(first["actions_u16"]).shape[0])
    for path in (paths[-1],):
        with np.load(path, allow_pickle=False) as payload:
            observed = int(np.asarray(payload["actions_u16"]).shape[0])
        if observed != samples_per_shard:
            raise ValueError(
                f"Shard-size drift: {path} has {observed}, expected {samples_per_shard}"
            )
    return {
        "path": str(split_dir),
        "shard_count": len(paths),
        "samples_per_shard": samples_per_shard,
        "sample_count": len(paths) * samples_per_shard,
        "filename_size_manifest_sha256": hashlib.sha256(
            "\n".join(entries).encode("utf-8")
        ).hexdigest(),
    }


def arena_source_global_indices(
    *,
    opening_pool_path: Path,
    test_dir: Path,
    samples_per_shard: int,
) -> np.ndarray:
    pool = json.loads(opening_pool_path.read_text(encoding="utf-8"))
    shard_ordinals = {
        path.name: ordinal
        for ordinal, path in enumerate(sorted(test_dir.glob("*.npz")))
    }
    indices: list[int] = []
    for opening in pool["openings"]:
        source_shard = Path(str(opening["source_shard"]))
        if source_shard.parent.name != "test":
            raise ValueError(f"Arena opening is not held out: {source_shard}")
        try:
            shard_ordinal = shard_ordinals[source_shard.name]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Arena source shard is absent locally: {source_shard}"
            ) from exc
        row = int(opening["source_row"])
        if not 0 <= row < samples_per_shard:
            raise ValueError(f"Arena source row is out of range: {source_shard}:{row}")
        indices.append(shard_ordinal * samples_per_shard + row)
    result = np.asarray(indices, dtype=np.uint32)
    if np.unique(result).size != result.size:
        raise ValueError("Arena opening source positions are not unique")
    return result


def write_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def freeze(args: argparse.Namespace) -> int:
    data_root = require_workspace(args.data_root, exists=True)
    output_dir = require_workspace(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Frozen evaluation directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    val = split_inventory(data_root / "val")
    test = split_inventory(data_root / "test")
    if args.fast_count + args.primary_count > val["sample_count"]:
        raise ValueError("Fast and primary validation pools exceed the val split")
    if args.blind_count > test["sample_count"]:
        raise ValueError("Blind pool exceeds the test split")

    validation_rng = np.random.Generator(np.random.PCG64(args.validation_seed))
    validation_indices = validation_rng.choice(
        val["sample_count"],
        size=args.fast_count + args.primary_count,
        replace=False,
        shuffle=True,
    ).astype(np.uint32)
    fast_indices = validation_indices[: args.fast_count]
    primary_indices = validation_indices[args.fast_count :]
    arena_files = {
        "opening_pool": require_workspace(OPENING_POOL, exists=True),
        "opening_histories": require_workspace(OPENING_HISTORIES, exists=True),
        "opening_contract": require_workspace(OPENING_CONTRACT, exists=True),
    }
    arena_indices = arena_source_global_indices(
        opening_pool_path=arena_files["opening_pool"],
        test_dir=data_root / "test",
        samples_per_shard=test["samples_per_shard"],
    )
    blind_candidates = np.arange(test["sample_count"], dtype=np.uint32)
    blind_candidates = blind_candidates[
        ~np.isin(blind_candidates, arena_indices, assume_unique=True)
    ]
    blind_rng = np.random.Generator(np.random.PCG64(args.blind_seed))
    blind_indices = blind_rng.choice(
        blind_candidates,
        size=args.blind_count,
        replace=False,
        shuffle=True,
    ).astype(np.uint32)
    if np.intersect1d(fast_indices, primary_indices).size:
        raise RuntimeError("Fast and primary validation pools overlap")
    if np.intersect1d(blind_indices, arena_indices).size:
        raise RuntimeError("Blind test and paired-arena source positions overlap")

    indices_path = output_dir / "position_indices.npz"
    np.savez(
        indices_path,
        fast_val_global_index=fast_indices,
        primary_val_global_index=primary_indices,
        blind_test_global_index=blind_indices,
    )
    arena = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in arena_files.items()
    }
    manifest = {
        "schema_version": "chess-dfm-hero-eval-manifest-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "immutable_after_creation": True,
        "action_codec": "lc0_canonical_1858",
        "horizon": 8,
        "dataset": {
            "archive_sha256": (
                "d8feffa259580563c8097fa4a604ca415469dd80da66dabda170ac9ce929b968"
            ),
            "manifest": str(data_root / "manifest.json"),
            "val": val,
            "test": test,
            "split_disjointness": {
                "status": "game-disjoint-by-construction",
                "construction": (
                    "scripts/process_lc0_pgns_to_gcs.py hashes "
                    "'archive URL | PGN header id | stream ordinal' once per game "
                    "and assigns the complete sliced game to train, val, or test"
                ),
                "local_reaudit_limitation": (
                    "trajectory metadata stores only the PGN-header component of "
                    "the augmented split id; archive URL and stream ordinal were "
                    "not retained, so local shards cannot independently re-join "
                    "all positions by the original split key"
                ),
            },
        },
        "position_indices": {
            "path": str(indices_path),
            "size_bytes": indices_path.stat().st_size,
            "sha256": sha256_file(indices_path),
            "global_index_definition": (
                "sorted shard filename ordinal * samples_per_shard + row ordinal"
            ),
            "fast_validation": {
                "split": "val",
                "count": args.fast_count,
                "seed": args.validation_seed,
                "array": "fast_val_global_index",
            },
            "primary_validation": {
                "split": "val",
                "count": args.primary_count,
                "seed": args.validation_seed,
                "array": "primary_val_global_index",
                "disjoint_from_fast": True,
            },
            "blind_test": {
                "split": "test",
                "count": args.blind_count,
                "seed": args.blind_seed,
                "array": "blind_test_global_index",
                "disjoint_from_paired_arena_sources": True,
                "selection_use": "terminal hero evaluation only",
            },
        },
        "paired_arena": {
            **arena,
            "opening_count": 2048,
            "opening_ply": 12,
            "pairing": "each opening is played with colors reversed",
            "source_positions_excluded_from_blind_test": int(arena_indices.size),
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "indices_sha256": manifest["position_indices"]["sha256"],
                "fast_count": args.fast_count,
                "primary_count": args.primary_count,
                "blind_count": args.blind_count,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fast-count", type=int, default=8192)
    parser.add_argument("--primary-count", type=int, default=65536)
    parser.add_argument("--blind-count", type=int, default=65536)
    parser.add_argument("--validation-seed", type=int, default=2026072501)
    parser.add_argument("--blind-seed", type=int, default=2026072502)
    return parser


if __name__ == "__main__":
    raise SystemExit(freeze(build_parser().parse_args()))
