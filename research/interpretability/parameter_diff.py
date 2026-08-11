"""Exact aligned-parameter diff for raw BT4 and the one-epoch Hero encoder."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from research.interpretability.models import (
    BT4ComparisonModels,
    load_bt4_comparison_models,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


class TensorStats(NamedTuple):
    count: int
    raw_sum_squares: float
    hero_sum_squares: float
    delta_sum_squares: float
    raw_hero_dot: float
    delta_sum: float
    delta_absolute_sum: float
    delta_absolute_max: float
    unchanged_count: int


@dataclasses.dataclass
class NormAccumulator:
    count: int = 0
    raw_sum_squares: float = 0.0
    hero_sum_squares: float = 0.0
    delta_sum_squares: float = 0.0
    raw_hero_dot: float = 0.0
    delta_sum: float = 0.0
    delta_absolute_sum: float = 0.0
    delta_absolute_max: float = 0.0
    unchanged_count: int = 0

    def add(self, stats: TensorStats) -> None:
        self.count += stats.count
        self.raw_sum_squares += stats.raw_sum_squares
        self.hero_sum_squares += stats.hero_sum_squares
        self.delta_sum_squares += stats.delta_sum_squares
        self.raw_hero_dot += stats.raw_hero_dot
        self.delta_sum += stats.delta_sum
        self.delta_absolute_sum += stats.delta_absolute_sum
        self.delta_absolute_max = max(
            self.delta_absolute_max,
            stats.delta_absolute_max,
        )
        self.unchanged_count += stats.unchanged_count

    def metrics(self) -> dict[str, float | int | None]:
        if self.count <= 0:
            raise ValueError("Cannot summarize an empty parameter group")
        raw_norm = math.sqrt(max(0.0, self.raw_sum_squares))
        hero_norm = math.sqrt(max(0.0, self.hero_sum_squares))
        delta_norm = math.sqrt(max(0.0, self.delta_sum_squares))
        denominator = raw_norm * hero_norm
        return {
            "parameter_count": self.count,
            "raw_l2": raw_norm,
            "hero_l2": hero_norm,
            "delta_l2": delta_norm,
            "relative_delta_l2": (delta_norm / raw_norm if raw_norm > 0.0 else None),
            "cosine": self.raw_hero_dot / denominator if denominator > 0.0 else None,
            "delta_mean": self.delta_sum / self.count,
            "delta_rms": delta_norm / math.sqrt(self.count),
            "delta_absolute_mean": self.delta_absolute_sum / self.count,
            "delta_absolute_max": self.delta_absolute_max,
            "unchanged_count": self.unchanged_count,
            "unchanged_fraction": self.unchanged_count / self.count,
        }


def _sum_float64(value: Tensor) -> float:
    return float(torch.sum(value, dtype=torch.float64).item())


def _tensor_stats(raw: Tensor, hero: Tensor) -> TensorStats:
    if raw.shape != hero.shape or raw.dtype != hero.dtype:
        raise ValueError(
            f"Parameter ABI mismatch: {raw.shape}/{raw.dtype} != {hero.shape}/{hero.dtype}"
        )
    raw_float = raw.detach().cpu().float()
    hero_float = hero.detach().cpu().float()
    delta = hero_float - raw_float
    absolute_delta = delta.abs()
    result = TensorStats(
        count=raw.numel(),
        raw_sum_squares=_sum_float64(raw_float.square()),
        hero_sum_squares=_sum_float64(hero_float.square()),
        delta_sum_squares=_sum_float64(delta.square()),
        raw_hero_dot=_sum_float64(raw_float * hero_float),
        delta_sum=_sum_float64(delta),
        delta_absolute_sum=_sum_float64(absolute_delta),
        delta_absolute_max=float(absolute_delta.max().item()),
        unchanged_count=int(torch.count_nonzero(raw.detach().cpu() == hero.detach().cpu())),
    )
    del raw_float, hero_float, delta, absolute_delta
    return result


def _parameter_groups(name: str) -> tuple[str, ...]:
    groups = ["all"]
    if name.startswith("embedding."):
        groups.extend(("trunk", "embedding"))
    elif name.startswith("layers."):
        parts = name.split(".")
        layer_index = int(parts[1])
        layer = f"layer_{layer_index:02d}"
        groups.extend(("trunk", "layers", layer))
        component = parts[2]
        if component in {"wq", "wq_b", "wk", "wk_b", "wv", "wv_b", "wo"}:
            branch = "attention"
        elif component == "smolgen":
            branch = "smolgen"
        elif component in {"ffn1", "ffn2"}:
            branch = "mlp"
        elif component in {"ln_attn", "ln_ffn"}:
            branch = "layer_norm"
        else:  # pragma: no cover - guarded by the frozen encoder ABI
            branch = "other"
        groups.extend((branch, f"{layer}.{branch}"))
    elif name.startswith("policy_head."):
        groups.append("policy_head")
    elif name.startswith("future_policy_heads."):
        groups.append("future_policy_heads")
    else:  # pragma: no cover - guarded by the frozen encoder ABI
        groups.append("unclassified")
    return tuple(groups)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def parameter_diff(models: BT4ComparisonModels) -> dict[str, Any]:
    """Compute exact scalar diff statistics without retaining tensor deltas."""

    raw_named = dict(models.raw_encoder.named_parameters())
    hero_named = dict(models.hero_encoder.named_parameters())
    if raw_named.keys() != hero_named.keys():
        raise ValueError("Raw/Hero parameter names differ")

    accumulators: dict[str, NormAccumulator] = {}
    leaf_rows: list[dict[str, Any]] = []
    leaf_delta_squares: dict[str, float] = {}
    for name in raw_named:
        raw = raw_named[name]
        hero = hero_named[name]
        stats = _tensor_stats(raw, hero)
        leaf_accumulator = NormAccumulator()
        leaf_accumulator.add(stats)
        leaf_metrics = leaf_accumulator.metrics()
        leaf_rows.append(
            {
                "name": name,
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
                **leaf_metrics,
            }
        )
        leaf_delta_squares[name] = stats.delta_sum_squares
        for group in _parameter_groups(name):
            accumulators.setdefault(group, NormAccumulator()).add(stats)

    group_metrics = {
        name: accumulator.metrics() for name, accumulator in sorted(accumulators.items())
    }
    global_delta_squares = accumulators["all"].delta_sum_squares
    for row in leaf_rows:
        row["delta_energy_fraction"] = (
            leaf_delta_squares[row["name"]] / global_delta_squares
            if global_delta_squares > 0.0
            else 0.0
        )
    largest_delta_contributors = sorted(
        (
            {
                "name": row["name"],
                "delta_l2": row["delta_l2"],
                "relative_delta_l2": row["relative_delta_l2"],
                "delta_energy_fraction": row["delta_energy_fraction"],
            }
            for row in leaf_rows
        ),
        key=lambda row: row["delta_energy_fraction"],
        reverse=True,
    )[:30]
    largest_relative_changes = sorted(
        (
            {
                "name": row["name"],
                "parameter_count": row["parameter_count"],
                "relative_delta_l2": row["relative_delta_l2"],
                "delta_l2": row["delta_l2"],
            }
            for row in leaf_rows
            if row["relative_delta_l2"] is not None
        ),
        key=lambda row: row["relative_delta_l2"],
        reverse=True,
    )[:30]
    return {
        "schema_version": "bt4-raw-hero-parameter-diff-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": "cpu",
        },
        "models": models.descriptor(),
        "scope": "embedding, 15-layer BT4 trunk, and native policy head",
        "delta_definition": "hero_parameter - raw_parameter after strict path alignment",
        "group_metrics": group_metrics,
        "leaves": leaf_rows,
        "ranked_views": {
            "largest_delta_energy_contributors": largest_delta_contributors,
            "largest_relative_changes": largest_relative_changes,
        },
        "deferred": {
            "matrix_delta_spectra": "next parameter-diff slice",
            "parameter_update_low_rank_tests": "next parameter-diff slice",
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Output must remain inside {_REPO_ROOT}: {resolved}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-bt4-path",
        type=Path,
        default=_REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz",
    )
    parser.add_argument(
        "--hero-checkpoint-dir",
        type=Path,
        default=_REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = load_bt4_comparison_models(
        raw_bt4_path=args.raw_bt4_path,
        hero_checkpoint_dir=args.hero_checkpoint_dir,
        device=torch.device("cpu"),
    )
    report = parameter_diff(models)
    _write_json_atomic(args.output, report)
    summary = report["group_metrics"]["all"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256_file(args.output),
                "leaf_count": len(report["leaves"]),
                "relative_delta_l2": summary["relative_delta_l2"],
                "cosine": summary["cosine"],
                "unchanged_fraction": summary["unchanged_fraction"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
