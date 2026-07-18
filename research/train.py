#!/usr/bin/env python3
"""Single-GPU compatibility trainer for the clean research path.

This first milestone intentionally imports the legacy model and loss so it can
serve as a local-GPU training/parity harness. It is *not* yet eligible for
autoresearch: the final version will define the experimental model and objective
directly in this file while continuing to use immutable support from
``research.prepare``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.training.checkpoints import load_training_checkpoint  # noqa: E402
from chess_dfm_jax.training.joint_latent_sasa import (  # noqa: E402
    create_joint_components,
    eval_joint_stage1_step,
    train_joint_stage1_step,
    train_joint_stage1_step_donated,
)
from research.legacy_baseline import flatten_metrics, resolve_config  # noqa: E402
from research.prepare import (  # noqa: E402
    REPO_ROOT,
    FixedTrajectoryBatches,
    require_within_workspace,
    validate_environment,
    write_json,
)


AUTORESEARCH_READY = False
ARCHITECTURE_SOURCE = "legacy_joint_latent_sasa_import"

DEFAULT_RUN_ROOT = REPO_ROOT / "checkpoints" / "source" / "step0265000"
DEFAULT_CHECKPOINT_DIR = DEFAULT_RUN_ROOT / "checkpoints"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "source" / "extracted"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "trajectory_v3"


class SigRegResult(NamedTuple):
    normalized: jax.Array
    official: jax.Array
    valid_count: jax.Array
    discrepancy: jax.Array


def normalized_le_jepa_sigreg(
    z: jax.Array,
    *,
    proj_dim: int,
    rng: jax.Array,
    reference_count: float = 1.0,
    sample_weight: jax.Array | None = None,
    t_max: float = 3.0,
    n_points: int = 17,
) -> SigRegResult:
    """Batch-count-invariant LeJEPA ECF discrepancy.

    ``official`` preserves the published Epps-Pulley statistic
    ``valid_count * discrepancy`` for diagnostics. ``normalized`` replaces the
    variable valid count with a fixed reference count, so duplicating or
    zero-padding a batch does not change the optimized scalar.

    This function operates on one physical batch. Epps-Pulley is non-additive,
    so callers must not average independently evaluated microbatch results.
    """

    if proj_dim < 1:
        raise ValueError(f"proj_dim must be positive, got {proj_dim}")
    if reference_count <= 0:
        raise ValueError(f"reference_count must be positive, got {reference_count}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    z = jnp.asarray(z, dtype=jnp.float32)
    sample_count, dim = z.shape
    directions = jax.random.normal(rng, (dim, proj_dim), dtype=jnp.float32)
    directions = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=0, keepdims=True),
        1e-12,
    )
    if sample_weight is None:
        sample_weight = jnp.ones((sample_count,), dtype=jnp.float32)
    sample_weight = jnp.maximum(jnp.asarray(sample_weight, dtype=jnp.float32), 0.0)

    projected = z @ directions
    t = jnp.linspace(0.0, t_max, n_points, dtype=jnp.float32)
    dt = jnp.asarray(t_max / (n_points - 1), dtype=jnp.float32)
    quadrature = jnp.full((n_points,), 2.0 * dt, dtype=jnp.float32)
    quadrature = quadrature.at[0].set(dt)
    quadrature = quadrature.at[-1].set(dt)
    normal_ecf = jnp.exp(-0.5 * jnp.square(t))
    quadrature = quadrature * normal_ecf

    xt = projected[:, :, None] * t[None, None, :]
    weight = sample_weight[:, None, None]
    valid_count = jnp.sum(sample_weight)
    denom = jnp.maximum(valid_count, 1.0)
    cos_mean = jnp.sum(jnp.cos(xt) * weight, axis=0) / denom
    sin_mean = jnp.sum(jnp.sin(xt) * weight, axis=0) / denom
    error = jnp.square(cos_mean - normal_ecf[None, :]) + jnp.square(sin_mean)
    discrepancy = jnp.mean(error @ quadrature)
    discrepancy = jnp.where(valid_count > 0.0, discrepancy, 0.0)
    official = discrepancy * valid_count
    normalized = discrepancy * jnp.asarray(reference_count, dtype=jnp.float32)
    return SigRegResult(normalized, official, valid_count, discrepancy)


def legal_mass_fp32(
    probs: jax.Array,
    legal_idx: jax.Array,
    legal_count: jax.Array,
) -> jax.Array:
    """Sum compact legal probabilities in FP32 and enforce probability bounds."""

    probs = jnp.asarray(probs, dtype=jnp.float32)
    safe_idx = jnp.clip(jnp.asarray(legal_idx, dtype=jnp.int32), 0, probs.shape[-1] - 1)
    legal_probs = jnp.take_along_axis(probs, safe_idx, axis=-1)
    slots = jnp.arange(safe_idx.shape[-1], dtype=jnp.int32)
    valid = slots < jnp.asarray(legal_count, dtype=jnp.int32)[..., None]
    mass = jnp.sum(jnp.where(valid, legal_probs, 0.0), axis=-1)
    return jnp.clip(mass, 0.0, 1.0)


def latent_collapse_diagnostics(
    pred_z: jax.Array,
    target_z: jax.Array,
    current_z: jax.Array,
    valid: jax.Array,
    *,
    rng: jax.Array,
) -> dict[str, jax.Array]:
    """Compute real per-horizon JEPA baselines and collapse measurements."""

    pred = jnp.asarray(pred_z, dtype=jnp.float32)
    target = jnp.asarray(target_z, dtype=jnp.float32)
    current = jnp.asarray(current_z, dtype=jnp.float32)
    weight = jnp.maximum(jnp.asarray(valid, dtype=jnp.float32), 0.0)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} != {target.shape}")
    if current.shape != (pred.shape[0], pred.shape[-1]):
        raise ValueError(
            f"current_z must have shape {(pred.shape[0], pred.shape[-1])}, got {current.shape}"
        )
    if weight.shape != pred.shape[:2]:
        raise ValueError(f"valid must have shape {pred.shape[:2]}, got {weight.shape}")

    denom = jnp.maximum(jnp.sum(weight, axis=0), 1.0)
    horizon_weight = weight[..., None]

    pred_mean = jnp.sum(pred * horizon_weight, axis=0) / denom[:, None]
    target_mean = jnp.sum(target * horizon_weight, axis=0) / denom[:, None]
    pred_variance = (
        jnp.sum(jnp.square(pred - pred_mean[None, :, :]) * horizon_weight, axis=0)
        / denom[:, None]
    )
    target_variance = (
        jnp.sum(jnp.square(target - target_mean[None, :, :]) * horizon_weight, axis=0)
        / denom[:, None]
    )
    pred_feature_std = jnp.sqrt(jnp.maximum(pred_variance, 0.0) + 1e-12)
    target_feature_std = jnp.sqrt(jnp.maximum(target_variance, 0.0) + 1e-12)

    pred_hbd = jnp.transpose(pred - pred_mean[None, :, :], (1, 0, 2))
    sqrt_weight_hb = jnp.sqrt(jnp.transpose(weight, (1, 0)))
    centered = pred_hbd * sqrt_weight_hb[..., None]
    covariance_denom = jnp.maximum(denom - 1.0, 1.0)
    gram = jnp.einsum("hbd,hcd->hbc", centered, centered)
    gram = gram / covariance_denom[:, None, None]
    eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(gram), 0.0)
    eigenvalue_sum = jnp.sum(eigenvalues, axis=-1, keepdims=True)
    spectrum = eigenvalues / jnp.maximum(eigenvalue_sum, 1e-12)
    entropy = -jnp.sum(jnp.where(spectrum > 0.0, spectrum * jnp.log(spectrum), 0.0), axis=-1)
    effective_rank = jnp.where(eigenvalue_sum[:, 0] > 1e-12, jnp.exp(entropy), 0.0)

    sample_mse = jnp.mean(jnp.square(pred - target), axis=-1)
    identity_mse = jnp.mean(jnp.square(current[:, None, :] - target), axis=-1)
    permutation = jax.random.permutation(rng, pred.shape[0])
    shuffled_target = target[permutation]
    shuffled_mse = jnp.mean(jnp.square(pred - shuffled_target), axis=-1)

    pred_norm = jnp.linalg.norm(pred, axis=-1)
    target_norm = jnp.linalg.norm(target, axis=-1)
    cosine = jnp.sum(pred * target, axis=-1) / jnp.maximum(pred_norm * target_norm, 1e-12)

    def horizon_mean(values: jax.Array) -> jax.Array:
        return jnp.sum(values * weight, axis=0) / denom

    return {
        "jepa_mse_by_horizon": horizon_mean(sample_mse),
        "identity_mse_by_horizon": horizon_mean(identity_mse),
        "shuffled_mse_by_horizon": horizon_mean(shuffled_mse),
        "pred_target_cosine_by_horizon": horizon_mean(cosine),
        "pred_rms_by_horizon": jnp.sqrt(
            jnp.sum(jnp.mean(jnp.square(pred), axis=-1) * weight, axis=0) / denom
        ),
        "target_rms_by_horizon": jnp.sqrt(
            jnp.sum(jnp.mean(jnp.square(target), axis=-1) * weight, axis=0) / denom
        ),
        "pred_feature_std_mean_by_horizon": jnp.mean(pred_feature_std, axis=-1),
        "pred_feature_std_p05_by_horizon": jnp.quantile(pred_feature_std, 0.05, axis=-1),
        "target_feature_std_mean_by_horizon": jnp.mean(target_feature_std, axis=-1),
        "pred_effective_rank_by_horizon": effective_rank,
        "valid_count_by_horizon": denom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--init",
        choices=("exact", "model-only"),
        default="exact",
        help="Restore legacy optimizer state exactly or start a fresh optimizer from restored weights.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Updates to run. Set 0 with --train-seconds for a time-budgeted run.",
    )
    parser.add_argument(
        "--train-seconds",
        type=float,
        default=0.0,
        help="Steady-state training budget after the first compile/update.",
    )
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=10_000)
    parser.add_argument("--val-deterministic-t", type=float, default=0.0)
    parser.add_argument("--donate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def gpu_memory_stats() -> dict[str, int]:
    stats = jax.devices()[0].memory_stats() or {}
    return {
        key: int(value)
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        if (value := stats.get(key)) is not None
    }


def evaluate(
    model,
    batches: FixedTrajectoryBatches,
    *,
    count: int,
    seed: int,
    deterministic_t: float,
) -> tuple[dict[str, float], float]:
    if count < 1:
        return {}, 0.0
    totals: dict[str, float] = {}
    started = time.perf_counter()
    for index in range(count):
        batch = batches.batch_at(index)
        batch["deterministic_t"] = np.asarray(deterministic_t, dtype=np.float32)
        rng = jax.random.fold_in(jax.random.PRNGKey(seed), index)
        loss, aux = eval_joint_stage1_step(model, batch, rng)
        jax.block_until_ready((loss, aux))
        metrics = {"loss": float(loss), **flatten_metrics(aux)}
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / count for key, value in totals.items()}, time.perf_counter() - started


def should_continue(*, updates: int, steps: int, deadline: float | None) -> bool:
    if steps > 0 and updates >= steps:
        return False
    if deadline is not None and updates > 0 and time.perf_counter() >= deadline:
        return False
    return steps > 0 or deadline is not None


def main() -> int:
    args = parse_args()
    validate_environment()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"research/train.py requires GPU, found {jax.default_backend()!r}")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.steps == 0 and args.train_seconds <= 0:
        raise ValueError("Set --steps > 0 or --train-seconds > 0")
    if args.train_seconds < 0:
        raise ValueError("--train-seconds must be non-negative")
    if not 0.0 <= args.val_deterministic_t <= 1.0:
        raise ValueError("--val-deterministic-t must be in [0, 1]")

    run_root = require_within_workspace(args.run_root)
    checkpoint_dir = require_within_workspace(args.checkpoint_dir)
    models_dir = require_within_workspace(args.models_dir)
    data_root = require_within_workspace(args.data_root)
    commit = git_commit()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"compat-step265k-b{args.batch_size}-{timestamp}"
    output_dir = require_within_workspace(
        args.output_dir or REPO_ROOT / "research" / "runs" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"

    config, metadata = resolve_config(run_root)
    train_batches = FixedTrajectoryBatches(
        data_root / "train",
        batch_size=args.batch_size,
        horizon=config.horizon,
        seed=args.seed,
        shuffle_files=True,
    )
    val_batches = FixedTrajectoryBatches(
        data_root / "val",
        batch_size=args.batch_size,
        horizon=config.horizon,
        seed=args.val_seed,
        shuffle_files=False,
    )

    model_params = load_mapped_bt4_params(models_dir=models_dir)
    model, optimizer = create_joint_components(model_params, config, seed=args.seed)
    restore_started = time.perf_counter()
    payload = load_training_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer if args.init == "exact" else None,
        step=int(metadata["latest_step"]),
        strict=True,
    )
    restore_seconds = time.perf_counter() - restore_started
    checkpoint_step = int(payload["step"])
    del payload

    initial_val, initial_val_seconds = evaluate(
        model,
        val_batches,
        count=args.eval_batches,
        seed=args.val_seed,
        deterministic_t=args.val_deterministic_t,
    )

    run_config = {
        "autoresearch_ready": AUTORESEARCH_READY,
        "architecture_source": ARCHITECTURE_SOURCE,
        "git_commit": commit,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "args": vars(args) | {"output_dir": str(output_dir)},
        "model_config": dataclasses.asdict(config),
        "checkpoint_step": checkpoint_step,
        "train_data": train_batches.provenance(),
        "val_data": val_batches.provenance(),
    }
    run_config["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_config["args"].items()
    }
    write_json(output_dir / "run_config.json", run_config)

    updates = 0
    examples = 0
    compile_update_seconds: float | None = None
    steady_update_seconds: list[float] = []
    final_train_metrics: dict[str, float] = {}
    deadline: float | None = float("inf") if args.train_seconds > 0 else None
    training_wall_started = time.perf_counter()

    with metrics_path.open("w", encoding="utf-8") as metrics_log:
        while should_continue(
            updates=updates,
            steps=args.steps,
            deadline=deadline,
        ):
            data_step = updates
            fetch_started = time.perf_counter()
            batch = train_batches.batch_at(data_step)
            fetch_seconds = time.perf_counter() - fetch_started
            step_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), data_step)
            update_started = time.perf_counter()
            train_fn = train_joint_stage1_step_donated if args.donate else train_joint_stage1_step
            loss, aux = train_fn(model, optimizer, batch, step_rng)
            jax.block_until_ready((loss, aux))
            update_seconds = time.perf_counter() - update_started

            updates += 1
            examples += args.batch_size
            if compile_update_seconds is None:
                compile_update_seconds = update_seconds
                if args.train_seconds > 0:
                    deadline = time.perf_counter() + args.train_seconds
            else:
                steady_update_seconds.append(update_seconds)

            final_train_metrics = {"loss": float(loss), **flatten_metrics(aux)}
            record = {
                "update": updates,
                "optimizer_step": int(optimizer.step[...]),
                "data_step": data_step,
                "fetch_seconds": fetch_seconds,
                "update_seconds": update_seconds,
                "examples_per_second": args.batch_size / max(update_seconds, 1e-12),
                **final_train_metrics,
            }
            metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_log.flush()

    training_wall_seconds = time.perf_counter() - training_wall_started
    final_val, final_val_seconds = evaluate(
        model,
        val_batches,
        count=args.eval_batches,
        seed=args.val_seed,
        deterministic_t=args.val_deterministic_t,
    )

    report = {
        **run_config,
        "restore_seconds": restore_seconds,
        "initial_validation_seconds": initial_val_seconds,
        "final_validation_seconds": final_val_seconds,
        "training_wall_seconds": training_wall_seconds,
        "compile_and_first_update_seconds": compile_update_seconds,
        "steady_update_seconds_mean": (
            float(np.mean(steady_update_seconds)) if steady_update_seconds else None
        ),
        "updates": updates,
        "examples": examples,
        "initial_validation": initial_val,
        "final_train": final_train_metrics,
        "final_validation": final_val,
        "gpu_memory": gpu_memory_stats(),
    }
    write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "autoresearch_ready": AUTORESEARCH_READY,
                "updates": updates,
                "examples": examples,
                "initial_val_dfm_ce": initial_val.get("dfm_ce_loss"),
                "final_val_dfm_ce": final_val.get("dfm_ce_loss"),
                "compile_and_first_update_seconds": compile_update_seconds,
                "steady_update_seconds_mean": report["steady_update_seconds_mean"],
                "gpu_memory": report["gpu_memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
