#!/usr/bin/env python3
"""Compare fast trajectory canonicalization with its board-enumerating oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_dfm_jax.data.trajectory_v3 import (  # noqa: E402
    trajectory_v3_to_batch,
)
from research.train_torch import (  # noqa: E402
    _canonicalize_trajectory_batch_reference,
    canonicalize_trajectory_batch,
)

DEFAULT_OUTPUT = (
    ROOT / "artifacts/profiles/hero_canonicalization_parity_v1.json"
)
DEFAULT_SHARD_INDICES = (0, 1, 137, 1024, 4096, 8192, 16000, 27678)


def _require_workspace(path: Path, *, exists: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=exists)
    resolved.relative_to(Path("/mountpoint/.exp"))
    return resolved


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mismatch_keys(
    fast: dict[str, Any],
    reference: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for key in sorted(set(fast) | set(reference)):
        if key not in fast or key not in reference:
            mismatches.append(key)
            continue
        if not np.array_equal(np.asarray(fast[key]), np.asarray(reference[key])):
            mismatches.append(key)
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/trajectory_v3/train",
    )
    parser.add_argument(
        "--shard-indices",
        type=int,
        nargs="+",
        default=DEFAULT_SHARD_INDICES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = _require_workspace(args.data_root, exists=True)
    output = _require_workspace(args.output)
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    paths = sorted(data_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No trajectory shards under {data_root}")
    if len(set(args.shard_indices)) != len(args.shard_indices):
        raise ValueError("--shard-indices must be unique")
    if any(index < 0 or index >= len(paths) for index in args.shard_indices):
        raise IndexError(
            f"Shard index must be in [0, {len(paths)}), got "
            f"{args.shard_indices}"
        )

    records: list[dict[str, Any]] = []
    for index in args.shard_indices:
        path = paths[index]
        with np.load(path, allow_pickle=False) as payload:
            batch = trajectory_v3_to_batch(
                payload,
                view="joint_latent_sasa",
                horizon=8,
                include_metadata=True,
            )
        started = time.perf_counter()
        fast = canonicalize_trajectory_batch(batch)
        fast_seconds = time.perf_counter() - started
        started = time.perf_counter()
        reference = _canonicalize_trajectory_batch_reference(batch)
        reference_seconds = time.perf_counter() - started
        records.append(
            {
                "shard_index": index,
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "example_count": int(batch["future_valid"].shape[0]),
                "valid_action_count": int(
                    np.count_nonzero(batch["future_valid"] > 0.0)
                ),
                "recovered_promotion_slot_count": int(
                    np.asarray(fast["legal_count"]).sum()
                    - np.asarray(batch["legal_count"]).sum()
                ),
                "fast_seconds": fast_seconds,
                "reference_seconds": reference_seconds,
                "mismatch_keys": _mismatch_keys(fast, reference),
            }
        )

    fast_seconds = sum(record["fast_seconds"] for record in records)
    reference_seconds = sum(
        record["reference_seconds"] for record in records
    )
    selection = "\n".join(record["path"] for record in records)
    report = {
        "schema_version": "canonical-trajectory-parity-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "data_root": str(data_root),
        "dataset_shard_count": len(paths),
        "selection_sha256": hashlib.sha256(
            selection.encode("utf-8")
        ).hexdigest(),
        "audited_shard_count": len(records),
        "audited_example_count": sum(
            record["example_count"] for record in records
        ),
        "audited_valid_action_count": sum(
            record["valid_action_count"] for record in records
        ),
        "recovered_promotion_slot_count": sum(
            record["recovered_promotion_slot_count"] for record in records
        ),
        "fast_seconds": fast_seconds,
        "reference_seconds": reference_seconds,
        "conversion_speedup": reference_seconds / fast_seconds,
        "exact_array_equality": all(
            not record["mismatch_keys"] for record in records
        ),
        "records": records,
    }
    report["gate_pass"] = report["exact_array_equality"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "gate_pass": report["gate_pass"],
                "audited_example_count": report["audited_example_count"],
                "conversion_speedup": report["conversion_speedup"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
