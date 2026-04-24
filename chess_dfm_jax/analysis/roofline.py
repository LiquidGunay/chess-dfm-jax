"""Small JAX roofline helpers for forward or training step profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


@dataclass
class RooflinePoint:
    name: str
    seconds_per_step: float
    steps_per_second: float
    flops_per_step: float | None
    bytes_per_step: float | None
    arithmetic_intensity: float | None
    achieved_tflops: float | None
    achieved_gbps: float | None
    cost_analysis: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cost_analysis"] = dict(self.cost_analysis)
        return out


def _normalize_cost_analysis(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, list):
        merged: dict[str, float] = {}
        for entry in raw:
            for key, value in entry.items():
                merged[key] = merged.get(key, 0.0) + float(value)
        return merged
    return {key: float(value) for key, value in raw.items()}


def _block_until_ready(value: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def compile_and_analyze(step_fn, *args, **kwargs):
    lowered = step_fn.lower(*args, **kwargs) if hasattr(step_fn, "lower") else jax.jit(step_fn).lower(*args, **kwargs)
    compiled = lowered.compile()
    analysis = _normalize_cost_analysis(compiled.cost_analysis()) if hasattr(compiled, "cost_analysis") else {}
    return compiled, analysis


def benchmark(compiled, *args, warmup: int = 2, repeat: int = 10, **kwargs) -> float:
    for _ in range(warmup):
        _block_until_ready(compiled(*args, **kwargs))
    start = time.perf_counter()
    for _ in range(repeat):
        _block_until_ready(compiled(*args, **kwargs))
    elapsed = time.perf_counter() - start
    return elapsed / max(repeat, 1)


def measure_roofline_point(
    name: str,
    step_fn,
    *args,
    warmup: int = 2,
    repeat: int = 10,
    **kwargs,
) -> RooflinePoint:
    compiled, analysis = compile_and_analyze(step_fn, *args, **kwargs)
    seconds = benchmark(compiled, *args, warmup=warmup, repeat=repeat, **kwargs)
    flops = analysis.get("flops")
    bytes_accessed = analysis.get("bytes accessed")
    intensity = None
    if flops is not None and bytes_accessed not in (None, 0.0):
        intensity = flops / bytes_accessed
    achieved_tflops = None if flops is None else flops / seconds / 1e12
    achieved_gbps = None if bytes_accessed is None else bytes_accessed / seconds / 1e9
    return RooflinePoint(
        name=name,
        seconds_per_step=seconds,
        steps_per_second=1.0 / seconds,
        flops_per_step=flops,
        bytes_per_step=bytes_accessed,
        arithmetic_intensity=intensity,
        achieved_tflops=achieved_tflops,
        achieved_gbps=achieved_gbps,
        cost_analysis=analysis,
    )


def roofline_bound(arithmetic_intensity: float, *, peak_tflops: float, peak_gbps: float) -> float:
    return min(peak_tflops, arithmetic_intensity * peak_gbps / 1e3)


def write_points_csv(points: list[RooflinePoint], out_path: str | Path) -> None:
    rows = []
    for point in points:
        row = point.as_dict()
        cost_analysis = row.pop("cost_analysis")
        for key, value in cost_analysis.items():
            row[f"cost_{key}"] = value
        rows.append(row)

    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def format_point(point: RooflinePoint) -> str:
    lines = [
        f"name: {point.name}",
        f"seconds/step: {point.seconds_per_step:.6f}",
        f"steps/sec: {point.steps_per_second:.3f}",
    ]
    if point.flops_per_step is not None:
        lines.append(f"flops/step: {point.flops_per_step:.3e}")
    if point.bytes_per_step is not None:
        lines.append(f"bytes/step: {point.bytes_per_step:.3e}")
    if point.arithmetic_intensity is not None:
        lines.append(f"arithmetic_intensity: {point.arithmetic_intensity:.3f} flop/byte")
    if point.achieved_tflops is not None:
        lines.append(f"achieved_tflops: {point.achieved_tflops:.3f}")
    if point.achieved_gbps is not None:
        lines.append(f"achieved_gbps: {point.achieved_gbps:.3f}")
    return "\n".join(lines)


def trace_compiled(logdir: str | Path, compiled, *args, repeat: int = 5, **kwargs) -> None:
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(logdir))
    try:
        for _ in range(repeat):
            _block_until_ready(compiled(*args, **kwargs))
    finally:
        jax.profiler.stop_trace()
