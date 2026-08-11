#!/usr/bin/env python3
"""Audit a completed LC0 sequential dataset before training."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.data.lc0_sequential import (  # noqa: E402
    ACTION_PAD,
    LC0_SEQUENTIAL_V1,
    METADATA_FIELDS,
    SequentialShard,
    VALUE_FIELDS,
)


SPLITS = ("train", "validation", "test")


class CorrelationAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_absolute_error = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        finite = np.isfinite(x64) & np.isfinite(y64)
        x64 = x64[finite]
        y64 = y64[finite]
        self.count += int(x64.size)
        self.sum_x += float(x64.sum())
        self.sum_y += float(y64.sum())
        self.sum_x2 += float(np.square(x64).sum())
        self.sum_y2 += float(np.square(y64).sum())
        self.sum_xy += float((x64 * y64).sum())
        self.sum_absolute_error += float(np.abs(x64 - y64).sum())

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "correlation": None, "mae": None}
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        denominator_x = self.count * self.sum_x2 - self.sum_x * self.sum_x
        denominator_y = self.count * self.sum_y2 - self.sum_y * self.sum_y
        denominator = math.sqrt(max(0.0, denominator_x * denominator_y))
        return {
            "count": self.count,
            "correlation": numerator / denominator if denominator > 0.0 else None,
            "mae": self.sum_absolute_error / self.count,
        }


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.dataset_root.resolve()
    root_manifest = json.loads(
        (root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if root_manifest.get("schema_version") != LC0_SEQUENTIAL_V1:
        raise ValueError(f"Dataset manifest schema drift: {root_manifest}")

    started = time.perf_counter()
    split_totals = {
        split: {"games": 0, "positions": 0, "trainable_starts": 0, "shards": 0}
        for split in SPLITS
    }
    all_game_names: set[str] = set()
    duplicate_game_names: list[str] = []
    played_missing_from_legal = 0
    best_missing_from_legal = 0
    padded_played_actions = 0
    required_value_nonfinite = 0
    invalid_wdl = 0
    policy_mass_max_error = 0.0
    legal_count_min = np.iinfo(np.int64).max
    legal_count_max = 0
    played_successor_q = CorrelationAccumulator()
    played_successor_d = CorrelationAccumulator()
    root_q_index = VALUE_FIELDS.index("root_q")
    root_d_index = VALUE_FIELDS.index("root_d")
    played_q_index = VALUE_FIELDS.index("played_q")
    played_d_index = VALUE_FIELDS.index("played_d")

    for split in SPLITS:
        paths = sorted(root.glob(f"chunks/chunk-*/{split}"))
        for path in paths:
            shard = SequentialShard(path)
            positions = shard.position_count
            game_offsets = np.asarray(shard.game_offsets_i64, dtype=np.int64)
            if (
                game_offsets.ndim != 1
                or game_offsets[0] != 0
                or game_offsets[-1] != positions
                or np.any(np.diff(game_offsets) < 2)
            ):
                raise ValueError(f"Invalid game offsets in {path}")
            game_names = json.loads(
                (path / "game_names.json").read_text(encoding="utf-8")
            )
            if len(game_names) != game_offsets.size - 1:
                raise ValueError(f"Game-name count drift in {path}")
            for name in game_names:
                if name in all_game_names:
                    duplicate_game_names.append(name)
                all_game_names.add(name)

            if shard.planes_u64.shape != (positions, 104):
                raise ValueError(f"Plane shape drift in {path}")
            if shard.metadata_u8.shape != (positions, len(METADATA_FIELDS)):
                raise ValueError(f"Metadata shape drift in {path}")
            if shard.values_f32.shape != (positions, len(VALUE_FIELDS)):
                raise ValueError(f"Value shape drift in {path}")
            if not bool(np.all(shard.input_format_u16 == 1)):
                raise ValueError(f"Non-classical input format in {path}")

            policy_offsets = np.asarray(shard.policy_offsets_i64, dtype=np.int64)
            if (
                policy_offsets.shape != (positions + 1,)
                or policy_offsets[0] != 0
                or policy_offsets[-1] != shard.policy_indices_u16.shape[0]
                or np.any(np.diff(policy_offsets) <= 0)
            ):
                raise ValueError(f"Policy offsets drift in {path}")
            counts = np.diff(policy_offsets)
            legal_count_min = min(legal_count_min, int(counts.min()))
            legal_count_max = max(legal_count_max, int(counts.max()))
            policy_values = np.asarray(shard.policy_values_f16, dtype=np.float32)
            masses = np.add.reduceat(policy_values, policy_offsets[:-1])
            policy_mass_max_error = max(
                policy_mass_max_error,
                float(np.abs(masses - 1.0).max()),
            )

            repeated_played = np.repeat(
                np.asarray(shard.played_actions_u16, dtype=np.uint16),
                counts,
            )
            repeated_best = np.repeat(
                np.asarray(shard.best_actions_u16, dtype=np.uint16),
                counts,
            )
            policy_indices = np.asarray(shard.policy_indices_u16, dtype=np.uint16)
            played_present = np.logical_or.reduceat(
                repeated_played == policy_indices,
                policy_offsets[:-1],
            )
            best_present = np.logical_or.reduceat(
                repeated_best == policy_indices,
                policy_offsets[:-1],
            )
            played_missing_from_legal += int(np.count_nonzero(~played_present))
            best_missing_from_legal += int(np.count_nonzero(~best_present))
            padded_played_actions += int(
                np.count_nonzero(shard.played_actions_u16 == ACTION_PAD)
            )

            values = np.asarray(shard.values_f32, dtype=np.float32)
            required_value_nonfinite += int(
                np.count_nonzero(~np.isfinite(values[:, :12]))
            )
            root_q = values[:, root_q_index]
            root_d = values[:, root_d_index]
            wins = 0.5 * (1.0 - root_d + root_q)
            losses = 0.5 * (1.0 - root_d - root_q)
            invalid_wdl += int(
                np.count_nonzero(
                    (wins < -1e-5)
                    | (root_d < -1e-5)
                    | (losses < -1e-5)
                    | (wins > 1.0 + 1e-5)
                    | (root_d > 1.0 + 1e-5)
                    | (losses > 1.0 + 1e-5)
                )
            )
            for begin, end in zip(game_offsets[:-1], game_offsets[1:], strict=True):
                played_successor_q.add(
                    values[begin : end - 1, played_q_index],
                    -values[begin + 1 : end, root_q_index],
                )
                played_successor_d.add(
                    values[begin : end - 1, played_d_index],
                    values[begin + 1 : end, root_d_index],
                )

            split_totals[split]["games"] += len(game_names)
            split_totals[split]["positions"] += positions
            split_totals[split]["trainable_starts"] += int(
                shard.manifest["trainable_start_count"]
            )
            split_totals[split]["shards"] += 1
            del shard
            gc.collect()

    expected = root_manifest["totals"]
    totals_match_manifest = all(
        split_totals[split]["games"] == int(expected["split_games"][split])
        and split_totals[split]["positions"]
        == int(expected["split_positions"][split])
        for split in SPLITS
    )
    checks = {
        "root_totals_reproduced": totals_match_manifest,
        "game_splits_disjoint": not duplicate_game_names,
        "all_played_actions_present_in_legal_policy": played_missing_from_legal == 0,
        "all_best_actions_present_in_legal_policy": best_missing_from_legal == 0,
        "no_padded_played_actions": padded_played_actions == 0,
        "required_value_fields_finite": required_value_nonfinite == 0,
        "root_qd_defines_valid_wdl": invalid_wdl == 0,
        "float16_policy_mass_error_below_1e-3": policy_mass_max_error <= 1e-3,
        "all_games_accounted_for": (
            len(all_game_names)
            == int(expected["games"])
            == sum(split_totals[split]["games"] for split in SPLITS)
        ),
        "source_sha256_recorded": (
            isinstance(root_manifest.get("source_sha256"), str)
            and len(root_manifest["source_sha256"]) == 64
        ),
    }
    report: dict[str, object] = {
        "schema_version": "lc0-sequential-audit-v1",
        "dataset_schema_version": LC0_SEQUENTIAL_V1,
        "dataset_root": str(root),
        "source_sha256": root_manifest.get("source_sha256"),
        "wall_seconds": time.perf_counter() - started,
        "storage_bytes": _directory_bytes(root),
        "split_totals": split_totals,
        "unique_game_count": len(all_game_names),
        "duplicate_game_names": duplicate_game_names[:100],
        "legal_count": {
            "minimum": int(legal_count_min),
            "maximum": int(legal_count_max),
        },
        "policy_mass_max_absolute_error": policy_mass_max_error,
        "played_missing_from_legal": played_missing_from_legal,
        "best_missing_from_legal": best_missing_from_legal,
        "padded_played_actions": padded_played_actions,
        "required_value_nonfinite": required_value_nonfinite,
        "invalid_wdl": invalid_wdl,
        "played_q_vs_negative_successor_root_q": played_successor_q.result(),
        "played_d_vs_successor_root_d": played_successor_d.result(),
        "checks": checks,
        "gate_pass": all(checks.values()),
    }
    if args.output is not None:
        _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["gate_pass"]:
        raise RuntimeError("LC0 sequential dataset audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
