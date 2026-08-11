#!/usr/bin/env python3
"""Fit bounded learning-curve sensitivities to frozen hero validations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "research/runs/torch_hero_epoch_v1/hero_validation_metrics.jsonl"
OUTPUT = ROOT / "research/analysis/hero_epoch_curve_20260727.json"
WARMUP_FRACTION = 0.02
MIN_RATIO = 1e-3


def lr_mass(epoch: float) -> float:
    """Integral of LR/peak over epochs under the literal frozen schedule."""

    if epoch <= WARMUP_FRACTION:
        return epoch * epoch / (2.0 * WARMUP_FRACTION)
    warmup_mass = WARMUP_FRACTION / 2.0
    capped = min(epoch, 1.0)
    progress = (capped - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
    cosine_mass = (1.0 - WARMUP_FRACTION) * (
        (MIN_RATIO + (1.0 - MIN_RATIO) / 2.0) * progress
        + (1.0 - MIN_RATIO) * math.sin(math.pi * progress) / (2.0 * math.pi)
    )
    tail_mass = max(epoch - 1.0, 0.0) * MIN_RATIO
    return warmup_mass + cosine_mass + tail_mass


def fit_model(name: str, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    floor_upper = float(y.min() - 1e-6)
    if name == "power_floor":
        # L(x) = floor + amplitude * x^-alpha
        function = lambda p, values: p[0] + p[1] * values ** (-p[2])
        lower = np.array([0.0, 0.0, 0.01])
        upper = np.array([floor_upper, 100.0, 5.0])
        seeds = [
            np.array([floor, amplitude, alpha])
            for floor in (0.0, floor_upper * 0.5, floor_upper * 0.9)
            for amplitude in (0.05, 0.5, 2.0, 10.0)
            for alpha in (0.1, 0.3, 0.7, 1.5)
        ]
        parameter_names = ["floor", "amplitude", "alpha"]
    elif name == "exponential_floor":
        # L(x) = floor + amplitude * exp(-rate*x)
        function = lambda p, values: p[0] + p[1] * np.exp(-p[2] * values)
        lower = np.array([0.0, 0.0, 0.001])
        upper = np.array([floor_upper, 100.0, 100.0])
        seeds = [
            np.array([floor, amplitude, rate])
            for floor in (0.0, floor_upper * 0.5, floor_upper * 0.9)
            for amplitude in (0.1, 1.0, 5.0, 10.0)
            for rate in (0.2, 1.0, 3.0, 10.0)
        ]
        parameter_names = ["floor", "amplitude", "rate"]
    elif name == "log_linear":
        # Deliberately floor-free sensitivity model.
        design = np.column_stack((np.ones_like(x), np.log(x)))
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        function = lambda p, values: p[0] + p[1] * np.log(values)
        fitted = function(params, x)
        return {
            "name": name,
            "parameters": {"intercept": float(params[0]), "log_slope": float(params[1])},
            "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
            "predictions": {
                "epoch_1_literal": float(function(params, np.array([lr_mass(1.0)]))[0]),
                "epoch_100_literal": float(
                    function(params, np.array([lr_mass(100.0)]))[0]
                ),
                "epoch_100_stretched_schedule": float(
                    function(params, np.array([100.0 * lr_mass(1.0)]))[0]
                ),
            },
            "last_point_anchored_predictions": {
                "epoch_1_literal": float(
                    y[-1]
                    + function(params, np.array([lr_mass(1.0)]))[0]
                    - function(params, np.array([x[-1]]))[0]
                ),
                "epoch_100_literal": float(
                    y[-1]
                    + function(params, np.array([lr_mass(100.0)]))[0]
                    - function(params, np.array([x[-1]]))[0]
                ),
                "epoch_100_stretched_schedule": float(
                    y[-1]
                    + function(params, np.array([100.0 * lr_mass(1.0)]))[0]
                    - function(params, np.array([x[-1]]))[0]
                ),
            },
        }
    else:
        raise ValueError(name)

    best = None
    for seed in seeds:
        result = least_squares(
            lambda p: function(p, x) - y,
            np.clip(seed, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            max_nfev=100_000,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    params = best.x
    fitted = function(params, x)
    return {
        "name": name,
        "parameters": {
            key: float(value) for key, value in zip(parameter_names, params, strict=True)
        },
        "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
        "predictions": {
            "epoch_1_literal": float(function(params, np.array([lr_mass(1.0)]))[0]),
            "epoch_100_literal": float(
                function(params, np.array([lr_mass(100.0)]))[0]
            ),
            "epoch_100_stretched_schedule": float(
                function(params, np.array([100.0 * lr_mass(1.0)]))[0]
            ),
        },
        "last_point_anchored_predictions": {
            "epoch_1_literal": float(
                y[-1]
                + function(params, np.array([lr_mass(1.0)]))[0]
                - function(params, np.array([x[-1]]))[0]
            ),
            "epoch_100_literal": float(
                y[-1]
                + function(params, np.array([lr_mass(100.0)]))[0]
                - function(params, np.array([x[-1]]))[0]
            ),
            "epoch_100_stretched_schedule": float(
                y[-1]
                + function(params, np.array([100.0 * lr_mass(1.0)]))[0]
                - function(params, np.array([x[-1]]))[0]
            ),
        },
    }


def main() -> None:
    rows = [
        json.loads(line)
        for line in METRICS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    points = []
    for row in rows:
        metrics = row["metrics"]
        epoch = float(row["observed_fraction"])
        h2_h8 = np.mean(
            [metrics[f"dfm_ce_loss_by_horizon_h{horizon}"] for horizon in range(2, 9)]
        )
        points.append(
            {
                "epoch": epoch,
                "lr_mass": lr_mass(epoch),
                "dfm_ce": float(metrics["dfm_ce_loss"]),
                "dfm_ce_h2_h8": float(h2_h8),
                "total_loss": float(metrics["loss"]),
            }
        )

    x = np.array([point["lr_mass"] for point in points])
    result = {
        "source": str(METRICS.relative_to(ROOT)),
        "point_count": len(points),
        "points": points,
        "schedule": {
            "warmup_fraction": WARMUP_FRACTION,
            "minimum_lr_ratio": MIN_RATIO,
            "lr_mass_epoch_1": lr_mass(1.0),
            "lr_mass_epoch_100_literal": lr_mass(100.0),
            "lr_mass_epoch_100_stretched_schedule": 100.0 * lr_mass(1.0),
        },
        "fits": {},
    }
    for metric in ("dfm_ce", "dfm_ce_h2_h8", "total_loss"):
        y = np.array([point[metric] for point in points])
        result["fits"][metric] = [
            fit_model(model, x, y)
            for model in ("power_floor", "exponential_floor", "log_linear")
        ]

    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
