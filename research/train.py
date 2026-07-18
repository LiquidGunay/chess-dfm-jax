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
import csv
import dataclasses
import gc
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.training.checkpoints import load_training_checkpoint  # noqa: E402
from chess_dfm_jax.training.joint_latent_sasa import (  # noqa: E402
    create_joint_components,
    eval_joint_stage1_step,
    joint_stage1_loss_fn,
    train_joint_stage1_step,
    train_joint_stage1_step_donated,
)
from chess_dfm_jax.nnx_bt4 import TrainableParam  # noqa: E402
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

GRADIENT_COMPONENT_NAMES = (
    "dfm_ce",
    "jepa_positive",
    "target_sigreg",
    "pred_sigreg",
    "fp32_legality",
)
GRADIENT_GROUP_NAMES = ("backbone", "dfm", "jepa", "other", "all")


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
    zero_mse = jnp.mean(jnp.square(target), axis=-1)
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
        "zero_mse_by_horizon": horizon_mean(zero_mse),
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


@nnx.jit
def diagnose_joint_latents(
    model,
    batch: dict[str, jax.Array],
    rng: jax.Array,
) -> dict[str, jax.Array]:
    """Authoritative free-rollout collapse and action-dependence diagnostics."""

    actions = batch["action_indices"][:, : model.config.horizon]
    future_planes = batch["future_planes"][:, : model.config.horizon]
    all_tokens, z_all = model.encode_current_and_future_tokens_and_vectors(
        batch["current_planes"],
        future_planes,
    )
    z_dfm = model.dfm_latents(all_tokens[:, 0])
    clean_t = jnp.ones((actions.shape[0],), dtype=jnp.float32)
    _, clean_hidden = model.planner_from_latents(
        z_dfm,
        actions,
        clean_t,
        return_hidden=True,
    )
    pred_z = model.jepa_rollout_from_latents(
        z_all[:, 0],
        actions,
        clean_hidden["action_tokens"],
        z0_normalized=True,
    )
    valid = (
        jnp.asarray(batch["future_valid"], dtype=jnp.float32)[:, : model.config.horizon]
        * jnp.asarray(batch["valid"], dtype=jnp.float32)[:, None]
    )
    rng_baseline, rng_action = jax.random.split(rng)
    metrics = latent_collapse_diagnostics(
        pred_z,
        z_all[:, 1:],
        z_all[:, 0],
        valid,
        rng=rng_baseline,
    )

    permutation = jax.random.permutation(rng_action, actions.shape[0])
    action_shuffled_pred = model.jepa_rollout_from_latents(
        z_all[:, 0],
        actions[permutation],
        clean_hidden["action_tokens"][permutation],
        z0_normalized=True,
    )
    action_shuffled_mse = jnp.mean(
        jnp.square(jnp.asarray(action_shuffled_pred, dtype=jnp.float32) - z_all[:, 1:]),
        axis=-1,
    )
    denom = jnp.maximum(jnp.sum(valid, axis=0), 1.0)
    metrics["action_shuffled_mse_by_horizon"] = (
        jnp.sum(action_shuffled_mse * valid, axis=0) / denom
    )
    return metrics


def _clip_loss_preserve_gradient(
    loss: jax.Array,
    clip_value: float,
) -> tuple[jax.Array, jax.Array]:
    loss = jnp.asarray(loss, dtype=jnp.float32)
    if clip_value <= 0.0:
        return loss, jnp.asarray(1.0, dtype=jnp.float32)
    clip = jnp.asarray(clip_value, dtype=jnp.float32)
    scale = jnp.minimum(1.0, clip / jax.lax.stop_gradient(jnp.maximum(loss, 1e-6)))
    return loss * scale, scale


def normalized_stage1_loss_fn(
    model,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compatibility loss with corrected count scaling and legal bounds.

    The legacy forward graph remains the oracle in this milestone. Its official
    EP statistics are converted to fixed-reference discrepancies algebraically,
    preserving their exact gradients while removing valid-count scaling.
    """

    _, legacy_aux = joint_stage1_loss_fn(
        model,
        batch,
        rng,
        compute_fp32_legality=True,
    )
    target_official = jnp.asarray(legacy_aux["jepa_sigreg_loss"], dtype=jnp.float32)
    pred_official = jnp.asarray(legacy_aux["jepa_pred_sigreg_loss"], dtype=jnp.float32)
    target_count = jnp.asarray(legacy_aux["jepa_sigreg_valid_count"], dtype=jnp.float32)
    pred_count = jnp.asarray(legacy_aux["jepa_pred_sigreg_valid_count"], dtype=jnp.float32)

    target_normalized = target_official * (
        jnp.asarray(target_reference_count, dtype=jnp.float32) / jnp.maximum(target_count, 1.0)
    )
    pred_normalized = pred_official * (
        jnp.asarray(pred_reference_count, dtype=jnp.float32) / jnp.maximum(pred_count, 1.0)
    )

    corrected_first_legality = jnp.asarray(
        legacy_aux["first_legality_loss_fp32"],
        dtype=jnp.float32,
    )
    corrected_legal_mass = 1.0 - corrected_first_legality
    corrected_weighted_legality = model.config.first_legality_coeff * corrected_first_legality

    unclipped = jnp.asarray(legacy_aux["unclipped_loss"], dtype=jnp.float32)
    unclipped = (
        unclipped
        - jnp.asarray(legacy_aux["weighted_legality_loss"], dtype=jnp.float32)
        - model.config.jepa_sigreg_coeff * target_official
        - model.config.jepa_pred_sigreg_coeff * pred_official
        + corrected_weighted_legality
        + model.config.jepa_sigreg_coeff * target_normalized
        + model.config.jepa_pred_sigreg_coeff * pred_normalized
    )
    loss, clip_scale = _clip_loss_preserve_gradient(unclipped, model.config.loss_clip_value)

    aux = dict(legacy_aux)
    aux.update(
        {
            "loss": loss,
            "unclipped_loss": unclipped,
            "loss_clip_scale": clip_scale,
            "first_legal_mass": corrected_legal_mass,
            "first_legality_loss": corrected_first_legality,
            "legality_loss": corrected_first_legality,
            "weighted_legality_loss": corrected_weighted_legality,
            "jepa_sigreg_official_loss": target_official,
            "jepa_pred_sigreg_official_loss": pred_official,
            "jepa_sigreg_loss": target_normalized,
            "jepa_pred_sigreg_loss": pred_normalized,
            "jepa_sigreg_reference_count": jnp.asarray(
                target_reference_count, dtype=jnp.float32
            ),
            "jepa_pred_sigreg_reference_count": jnp.asarray(
                pred_reference_count, dtype=jnp.float32
            ),
        }
    )
    return loss, aux


def gradient_component_vector(
    model,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    sigreg_reference_count: float,
) -> jax.Array:
    """Return unweighted scalars whose gradients define objective calibration."""

    _, aux = normalized_stage1_loss_fn(
        model,
        batch,
        rng,
        sigreg_reference_count,
        sigreg_reference_count,
    )
    return jnp.stack(
        [
            jnp.asarray(aux["dfm_ce_loss"], dtype=jnp.float32),
            jnp.asarray(aux["jepa_positive_loss"], dtype=jnp.float32),
            jnp.asarray(aux["jepa_sigreg_loss"], dtype=jnp.float32),
            jnp.asarray(aux["jepa_pred_sigreg_loss"], dtype=jnp.float32),
            jnp.asarray(aux["first_legality_loss"], dtype=jnp.float32),
        ]
    )


_normalized_loss_and_grad = nnx.value_and_grad(
    normalized_stage1_loss_fn,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)


def _normalized_train_step_impl(
    model,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    with jax.named_scope("research_normalized_loss_and_grad"):
        (loss, aux), grads = _normalized_loss_and_grad(
            model,
            batch,
            rng,
            target_reference_count,
            pred_reference_count,
        )
    with jax.named_scope("research_normalized_optimizer_update"):
        optimizer.update(model, grads)
    return loss, aux


@nnx.jit
def eval_normalized_stage1_step(
    model,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    return normalized_stage1_loss_fn(
        model,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
    )


@nnx.jit
def train_normalized_stage1_step(
    model,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    return _normalized_train_step_impl(
        model,
        optimizer,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
    )


@nnx.jit(donate_argnums=(0, 1))
def train_normalized_stage1_step_donated(
    model,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    return _normalized_train_step_impl(
        model,
        optimizer,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
    )


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
    parser.add_argument(
        "--objective",
        choices=("legacy", "normalized"),
        default="legacy",
        help="Exact legacy loss or fixed-reference SIGReg plus bounded legality.",
    )
    parser.add_argument("--target-sigreg-coeff", type=float)
    parser.add_argument("--pred-sigreg-coeff", type=float)
    parser.add_argument("--sigreg-reference-count", type=float, default=1.0)
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
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument(
        "--collapse-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=10_000)
    parser.add_argument("--val-deterministic-t", type=float, default=0.0)
    parser.add_argument("--donate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile-ahead",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lower/compile the training step explicitly and record compiler cost analysis.",
    )
    parser.add_argument(
        "--gpu-monitor-interval-ms",
        type=int,
        default=0,
        help="Sample nvidia-smi during steady training; 100 ms is useful for profile runs.",
    )
    parser.add_argument(
        "--gradient-audit",
        action="store_true",
        help="Measure unweighted component gradients instead of training.",
    )
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
    objective: str,
    sigreg_reference_count: float,
    collapse_diagnostics: bool,
) -> tuple[dict[str, float], float]:
    if count < 1:
        return {}, 0.0
    totals: dict[str, float] = {}
    started = time.perf_counter()
    for index in range(count):
        batch = batches.batch_at(index)
        batch["deterministic_t"] = np.asarray(deterministic_t, dtype=np.float32)
        rng = jax.random.fold_in(jax.random.PRNGKey(seed), index)
        if objective == "normalized":
            loss, aux = eval_normalized_stage1_step(
                model,
                batch,
                rng,
                sigreg_reference_count,
                sigreg_reference_count,
            )
        else:
            loss, aux = eval_joint_stage1_step(model, batch, rng)
        jax.block_until_ready((loss, aux))
        metrics = {"loss": float(loss), **flatten_metrics(aux)}
        if collapse_diagnostics:
            diagnostic_rng = jax.random.fold_in(rng, 0xC011A95E)
            diagnostics = diagnose_joint_latents(model, batch, diagnostic_rng)
            jax.block_until_ready(diagnostics)
            metrics.update(flatten_metrics(diagnostics))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / count for key, value in totals.items()}, time.perf_counter() - started


def should_continue(*, updates: int, steps: int, deadline: float | None) -> bool:
    if steps > 0 and updates >= steps:
        return False
    if deadline is not None and updates > 0 and time.perf_counter() >= deadline:
        return False
    return steps > 0 or deadline is not None


def normalize_cost_analysis(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, list):
        merged: dict[str, float] = {}
        for entry in raw:
            for key, value in entry.items():
                merged[key] = merged.get(key, 0.0) + float(value)
        return merged
    return {key: float(value) for key, value in raw.items()}


def compiler_cost_summary(raw: dict[str, float]) -> dict[str, float]:
    keys = ("flops", "transcendentals", "bytes accessed", "optimal_seconds")
    return {key: raw[key] for key in keys if key in raw}


def normalize_memory_analysis(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if dataclasses.is_dataclass(raw):
        values = dataclasses.asdict(raw)
    elif hasattr(raw, "_asdict"):
        values = raw._asdict()
    else:
        values = {
            key: getattr(raw, key)
            for key in dir(raw)
            if key.endswith("_in_bytes") and not key.startswith("_")
        }
    return {
        key: int(value)
        for key, value in values.items()
        if value is not None and key.endswith("_in_bytes")
    }


def start_gpu_monitor(
    output_dir: Path,
    *,
    interval_ms: int,
) -> tuple[subprocess.Popen[str], IO[str], IO[str]]:
    samples_handle = (output_dir / "gpu_samples.csv").open("w", encoding="utf-8")
    stderr_handle = (output_dir / "gpu_monitor.stderr.log").open("w", encoding="utf-8")
    fields = (
        "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,clocks.sm,clocks.mem"
    )
    process = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ],
        stdout=samples_handle,
        stderr=stderr_handle,
        text=True,
    )
    return process, samples_handle, stderr_handle


def stop_gpu_monitor(
    process: subprocess.Popen[str],
    samples_handle: IO[str],
    stderr_handle: IO[str],
) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    samples_handle.close()
    stderr_handle.close()


def summarize_gpu_samples(path: Path) -> dict[str, float | int]:
    columns = {
        "gpu_utilization_percent": 1,
        "memory_utilization_percent": 2,
        "memory_used_mib": 3,
        "memory_total_mib": 4,
        "power_watts": 5,
        "sm_clock_mhz": 6,
        "memory_clock_mhz": 7,
    }
    values = {name: [] for name in columns}
    if not path.is_file():
        return {"sample_count": 0}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 8:
                continue
            try:
                for name, index in columns.items():
                    values[name].append(float(row[index].strip()))
            except ValueError:
                continue
    sample_count = len(values["gpu_utilization_percent"])
    summary: dict[str, float | int] = {"sample_count": sample_count}
    if sample_count == 0:
        return summary
    for name, series in values.items():
        array = np.asarray(series, dtype=np.float64)
        summary[f"{name}_mean"] = float(np.mean(array))
        summary[f"{name}_p50"] = float(np.quantile(array, 0.50))
        summary[f"{name}_p95"] = float(np.quantile(array, 0.95))
        summary[f"{name}_max"] = float(np.max(array))
    return summary


def training_function(*, objective: str, donate: bool):
    if objective == "normalized":
        return (
            train_normalized_stage1_step_donated
            if donate
            else train_normalized_stage1_step
        )
    return train_joint_stage1_step_donated if donate else train_joint_stage1_step


def training_call_args(
    *,
    objective: str,
    model,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    sigreg_reference_count: float,
) -> tuple[Any, ...]:
    common = (model, optimizer, batch, rng)
    if objective == "normalized":
        return common + (sigreg_reference_count, sigreg_reference_count)
    return common


def compiler_performance(
    cost_analysis: dict[str, float],
    seconds_per_update: float | None,
) -> dict[str, float | None]:
    """Convert XLA's static cost estimate into explicitly labelled rates."""

    flops = cost_analysis.get("flops")
    bytes_accessed = cost_analysis.get("bytes accessed")
    if seconds_per_update is None or seconds_per_update <= 0.0:
        return {
            "compiler_estimated_flops_per_update": flops,
            "compiler_estimated_bytes_per_update": bytes_accessed,
            "compiler_estimated_arithmetic_intensity": None,
            "compiler_estimated_achieved_tflops": None,
            "compiler_estimated_achieved_gbps": None,
        }
    return {
        "compiler_estimated_flops_per_update": flops,
        "compiler_estimated_bytes_per_update": bytes_accessed,
        "compiler_estimated_arithmetic_intensity": (
            None
            if flops is None or bytes_accessed in (None, 0.0)
            else flops / bytes_accessed
        ),
        "compiler_estimated_achieved_tflops": (
            None if flops is None else flops / seconds_per_update / 1e12
        ),
        "compiler_estimated_achieved_gbps": (
            None if bytes_accessed is None else bytes_accessed / seconds_per_update / 1e9
        ),
    }


def state_path_parts(path: tuple[Any, ...]) -> tuple[str, ...]:
    parts = []
    for item in path:
        if hasattr(item, "key"):
            value = item.key
        elif hasattr(item, "name"):
            value = item.name
        elif hasattr(item, "idx"):
            value = item.idx
        else:
            value = item
        parts.append(str(value))
    return tuple(parts)


def gradient_group_for_path(path: tuple[Any, ...]) -> str:
    parts = state_path_parts(path)
    root = parts[0] if parts else ""
    if root == "encoder":
        return "backbone"
    if root in {
        "action_embed",
        "dfm_blocks",
        "dfm_out_norm",
        "dfm_state_projector",
        "out_bias",
        "out_proj",
        "pos_embed",
        "time_bias",
        "time_embed1",
        "time_embed2",
    }:
        return "dfm"
    if root in {
        "jepa_action_embed",
        "jepa_hidden_adapter",
        "jepa_state_norm",
        "jepa_transition",
        "state_projector",
    }:
        return "jepa"
    return "other"


def reconstruct_polarized_grams(
    diagonal_q: np.ndarray,
    pair_q: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Reconstruct per-group component-gradient Grams in host FP64."""

    diagonal_q = np.asarray(diagonal_q, dtype=np.float64)
    pair_q = np.asarray(pair_q, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    component_count, group_count = diagonal_q.shape
    expected_pairs = component_count * (component_count - 1) // 2
    if pair_q.shape != (expected_pairs, 2, group_count):
        raise ValueError(
            f"pair_q must have shape {(expected_pairs, 2, group_count)}, "
            f"got {pair_q.shape}"
        )
    if scales.shape != (component_count,):
        raise ValueError(
            f"scales must have shape {(component_count,)}, got {scales.shape}"
        )

    explicit_grams = np.zeros(
        (group_count, component_count, component_count),
        dtype=np.float64,
    )
    for component_index in range(component_count):
        explicit_grams[
            :,
            component_index,
            component_index,
        ] = diagonal_q[component_index]
    pair_index = 0
    for left in range(component_count):
        for right in range(left + 1, component_count):
            cross = (pair_q[pair_index, 0] - pair_q[pair_index, 1]) / (
                4.0 * scales[left] * scales[right]
            )
            explicit_grams[:, left, right] = cross
            explicit_grams[:, right, left] = cross
            pair_index += 1
    all_gram = np.sum(explicit_grams, axis=0, keepdims=True)
    return np.concatenate([explicit_grams, all_gram], axis=0)


def run_gradient_audit(
    model,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    *,
    sigreg_reference_count: float,
) -> dict[str, Any]:
    """Measure exact component-gradient Gram matrices from one fixed batch."""

    graphdef, trainable_state, nondiff_state = nnx.split(model, TrainableParam, ...)
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(trainable_state)
    leaf_groups = [gradient_group_for_path(path) for path, _ in path_leaves]
    parameter_counts = {name: 0 for name in GRADIENT_GROUP_NAMES}
    for (_, leaf), group in zip(path_leaves, leaf_groups, strict=True):
        count = int(np.prod(leaf.shape, dtype=np.int64))
        parameter_counts[group] += count
        parameter_counts["all"] += count

    group_name_to_index = {
        name: index for index, name in enumerate(GRADIENT_GROUP_NAMES[:-1])
    }

    def audit_kernel(
        params,
        nondiff,
        audit_batch,
        audit_rng,
    ):
        def components_for_params(candidate_params):
            candidate_model = nnx.merge(graphdef, candidate_params, nondiff)
            return gradient_component_vector(
                candidate_model,
                audit_batch,
                audit_rng,
                sigreg_reference_count,
            )

        components, pullback = jax.vjp(components_for_params, params)
        component_count = len(GRADIENT_COMPONENT_NAMES)

        def squared_group_norms(cotangent):
            gradient = pullback(cotangent)[0]
            leaves = jax.tree_util.tree_leaves(gradient)
            group_squares = [
                jnp.asarray(0.0, dtype=jnp.float32)
                for _ in GRADIENT_GROUP_NAMES[:-1]
            ]
            for leaf, group in zip(leaves, leaf_groups, strict=True):
                group_index = group_name_to_index[group]
                array = jnp.asarray(leaf, dtype=jnp.float32)
                group_squares[group_index] = (
                    group_squares[group_index] + jnp.sum(jnp.square(array))
                )
            return jnp.stack(group_squares)

        basis = jnp.eye(component_count, dtype=jnp.float32)
        diagonal_q = jax.lax.map(squared_group_norms, basis)
        total_diagonal_q = jnp.sum(diagonal_q, axis=1)
        scales = jnp.where(
            total_diagonal_q > 0.0,
            jax.lax.rsqrt(jnp.maximum(total_diagonal_q, 1e-30)),
            1.0,
        )
        pair_weights = jnp.stack(
            [
                direction
                for left in range(component_count)
                for right in range(left + 1, component_count)
                for direction in (
                    scales[left] * basis[left] + scales[right] * basis[right],
                    scales[left] * basis[left] - scales[right] * basis[right],
                )
            ]
        )
        pair_q = jax.lax.map(squared_group_norms, pair_weights).reshape(
            (-1, 2, len(GRADIENT_GROUP_NAMES) - 1)
        )
        return components, diagonal_q, pair_q, scales

    compile_started = time.perf_counter()
    compiled_audit = (
        jax.jit(audit_kernel)
        .lower(
            trainable_state,
            nondiff_state,
            batch,
            rng,
        )
        .compile()
    )
    compile_seconds = time.perf_counter() - compile_started

    compiler_cost = (
        compiler_cost_summary(normalize_cost_analysis(compiled_audit.cost_analysis()))
        if hasattr(compiled_audit, "cost_analysis")
        else {}
    )
    compiler_memory = (
        normalize_memory_analysis(compiled_audit.memory_analysis())
        if hasattr(compiled_audit, "memory_analysis")
        else {}
    )

    execute_started = time.perf_counter()
    component_values_raw, diagonal_q_raw, pair_q_raw, scales_raw = compiled_audit(
        trainable_state,
        nondiff_state,
        batch,
        rng,
    )
    jax.block_until_ready(
        (component_values_raw, diagonal_q_raw, pair_q_raw, scales_raw)
    )
    execute_seconds = time.perf_counter() - execute_started
    component_values = np.asarray(component_values_raw, dtype=np.float64)
    diagonal_q = np.asarray(diagonal_q_raw, dtype=np.float64)
    pair_q = np.asarray(pair_q_raw, dtype=np.float64)
    scales = np.asarray(scales_raw, dtype=np.float64)

    component_count = len(GRADIENT_COMPONENT_NAMES)
    grams = reconstruct_polarized_grams(diagonal_q, pair_q, scales)

    gradient_norms: dict[str, dict[str, float]] = {}
    gradient_cosines: dict[str, dict[str, float | None]] = {}
    gram_diagnostics: dict[str, dict[str, float]] = {}
    for group_index, group_name in enumerate(GRADIENT_GROUP_NAMES):
        gram = grams[group_index]
        diagonal = np.maximum(np.diag(gram), 0.0)
        norms = np.sqrt(diagonal)
        norm_floor = float(np.max(norms)) * 1e-8
        gradient_norms[group_name] = {
            component: float(norm)
            for component, norm in zip(GRADIENT_COMPONENT_NAMES, norms, strict=True)
        }
        cosines: dict[str, float | None] = {}
        for left in range(component_count):
            for right in range(left + 1, component_count):
                denom = norms[left] * norms[right]
                key = (
                    f"{GRADIENT_COMPONENT_NAMES[left]}"
                    f"__{GRADIENT_COMPONENT_NAMES[right]}"
                )
                cosines[key] = (
                    None
                    if norms[left] <= norm_floor or norms[right] <= norm_floor
                    else float(gram[left, right] / denom)
                )
        gradient_cosines[group_name] = cosines
        eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
        gram_diagnostics[group_name] = {
            "minimum_eigenvalue": float(np.min(eigenvalues)),
            "maximum_eigenvalue": float(np.max(eigenvalues)),
            "maximum_asymmetry": float(np.max(np.abs(gram - gram.T))),
        }

    primary_coefficients = np.asarray(
        [1.0, 1.0, 0.0, 0.0, model.config.first_legality_coeff],
        dtype=np.float64,
    )
    fractions = (0.01, 0.03, 0.10, 0.30)
    suggested_coefficients: dict[str, dict[str, dict[str, float]]] = {}
    for component_index, component_name in (
        (2, "target_sigreg"),
        (3, "pred_sigreg"),
    ):
        per_group: dict[str, dict[str, float]] = {}
        for group_index, group_name in enumerate(GRADIENT_GROUP_NAMES):
            gram = grams[group_index]
            primary_norm = float(
                np.sqrt(
                    max(
                        float(primary_coefficients @ gram @ primary_coefficients),
                        0.0,
                    )
                )
            )
            auxiliary_norm = float(
                np.sqrt(max(float(gram[component_index, component_index]), 0.0))
            )
            if primary_norm == 0.0 or auxiliary_norm == 0.0:
                continue
            per_group[group_name] = {
                f"{fraction:.2f}": fraction * primary_norm / auxiliary_norm
                for fraction in fractions
            }
        suggested_coefficients[component_name] = per_group

    return {
        "component_names": list(GRADIENT_COMPONENT_NAMES),
        "component_values": {
            name: float(value)
            for name, value in zip(
                GRADIENT_COMPONENT_NAMES,
                component_values,
                strict=True,
            )
        },
        "parameter_counts": parameter_counts,
        "gradient_norms": gradient_norms,
        "gradient_cosines": gradient_cosines,
        "gradient_gram_matrices": {
            group_name: grams[group_index].tolist()
            for group_index, group_name in enumerate(GRADIENT_GROUP_NAMES)
        },
        "gradient_gram_diagnostics": gram_diagnostics,
        "primary_gradient_coefficients": {
            name: float(value)
            for name, value in zip(
                GRADIENT_COMPONENT_NAMES,
                primary_coefficients,
                strict=True,
            )
        },
        "suggested_sigreg_coefficients_by_gradient_fraction": suggested_coefficients,
        "sigreg_reference_count": sigreg_reference_count,
        "vjp_compile_seconds": compile_seconds,
        "vjp_execute_seconds": execute_seconds,
        "polarization_scales": {
            name: float(scale)
            for name, scale in zip(
                GRADIENT_COMPONENT_NAMES,
                scales,
                strict=True,
            )
        },
        "polarization_diagonal_q": diagonal_q.tolist(),
        "polarization_pair_q": pair_q.tolist(),
        "compiler_cost_analysis": compiler_cost,
        "compiler_memory_analysis": compiler_memory,
        "gpu_memory": gpu_memory_stats(),
    }


def main() -> int:
    args = parse_args()
    validate_environment()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"research/train.py requires GPU, found {jax.default_backend()!r}")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if (
        not args.eval_only
        and not args.gradient_audit
        and args.steps == 0
        and args.train_seconds <= 0
    ):
        raise ValueError("Set --steps > 0 or --train-seconds > 0")
    if args.train_seconds < 0:
        raise ValueError("--train-seconds must be non-negative")
    if args.sigreg_reference_count <= 0:
        raise ValueError("--sigreg-reference-count must be positive")
    if args.gpu_monitor_interval_ms < 0:
        raise ValueError("--gpu-monitor-interval-ms must be non-negative")
    if 0 < args.gpu_monitor_interval_ms < 50:
        raise ValueError("--gpu-monitor-interval-ms must be 0 or at least 50")
    if args.gradient_audit and args.eval_only:
        raise ValueError("--gradient-audit and --eval-only are mutually exclusive")
    if args.gradient_audit and args.objective != "normalized":
        raise ValueError("--gradient-audit requires --objective normalized")
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
    config = dataclasses.replace(
        config,
        jepa_sigreg_coeff=(
            config.jepa_sigreg_coeff
            if args.target_sigreg_coeff is None
            else args.target_sigreg_coeff
        ),
        jepa_pred_sigreg_coeff=(
            config.jepa_pred_sigreg_coeff
            if args.pred_sigreg_coeff is None
            else args.pred_sigreg_coeff
        ),
    )
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
        optimizer=(
            optimizer
            if args.init == "exact" and not args.gradient_audit
            else None
        ),
        step=int(metadata["latest_step"]),
        strict=True,
    )
    restore_seconds = time.perf_counter() - restore_started
    checkpoint_step = int(payload["step"])
    del payload

    if args.gradient_audit:
        if config.jepa_sigreg_coeff == 0.0 or config.jepa_pred_sigreg_coeff == 0.0:
            raise ValueError(
                "--gradient-audit requires nonzero --target-sigreg-coeff and "
                "--pred-sigreg-coeff so both statistics are present in the graph"
            )
        audit_batch = train_batches.batch_at(0)
        audit_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 0)
        del optimizer
        gc.collect()
        audit_started = time.perf_counter()
        audit = run_gradient_audit(
            model,
            audit_batch,
            audit_rng,
            sigreg_reference_count=args.sigreg_reference_count,
        )
        audit["wall_seconds"] = time.perf_counter() - audit_started
        audit_config = {
            "autoresearch_ready": AUTORESEARCH_READY,
            "architecture_source": ARCHITECTURE_SOURCE,
            "git_commit": commit,
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in (
                    vars(args) | {"output_dir": str(output_dir)}
                ).items()
            },
            "model_config": dataclasses.asdict(config),
            "checkpoint_step": checkpoint_step,
            "restore_seconds": restore_seconds,
            "train_data": train_batches.provenance(),
            "audit": audit,
        }
        write_json(output_dir / "run_config.json", audit_config)
        write_json(output_dir / "gradient_audit.json", audit_config)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "output_dir": str(output_dir),
                    "component_values": audit["component_values"],
                    "gradient_norms": audit["gradient_norms"],
                    "suggested_sigreg_coefficients": audit[
                        "suggested_sigreg_coefficients_by_gradient_fraction"
                    ],
                    "wall_seconds": audit["wall_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    initial_val, initial_val_seconds = evaluate(
        model,
        val_batches,
        count=args.eval_batches,
        seed=args.val_seed,
        deterministic_t=args.val_deterministic_t,
        objective=args.objective,
        sigreg_reference_count=args.sigreg_reference_count,
        collapse_diagnostics=args.collapse_diagnostics,
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

    train_fn = training_function(objective=args.objective, donate=args.donate)
    executable = train_fn
    explicit_compile_seconds: float | None = None
    compiler_cost_analysis_raw: dict[str, float] = {}
    compiler_memory_analysis: dict[str, int] = {}
    if args.compile_ahead and not args.eval_only:
        compile_batch = train_batches.batch_at(0)
        compile_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 0)
        compile_args = training_call_args(
            objective=args.objective,
            model=model,
            optimizer=optimizer,
            batch=compile_batch,
            rng=compile_rng,
            sigreg_reference_count=args.sigreg_reference_count,
        )
        compile_started = time.perf_counter()
        executable = train_fn.lower(*compile_args).compile()
        explicit_compile_seconds = time.perf_counter() - compile_started
        if hasattr(executable, "cost_analysis"):
            compiler_cost_analysis_raw = normalize_cost_analysis(executable.cost_analysis())
            write_json(
                output_dir / "compiler_cost_analysis.json",
                compiler_cost_analysis_raw,
            )
        if hasattr(executable, "memory_analysis"):
            compiler_memory_analysis = normalize_memory_analysis(
                executable.memory_analysis()
            )

    updates = 0
    examples = 0
    first_update_seconds: float | None = None
    steady_update_seconds: list[float] = []
    final_train_metrics: dict[str, float] = {}
    deadline: float | None = float("inf") if args.train_seconds > 0 else None
    training_wall_started = time.perf_counter()
    gpu_monitor: tuple[subprocess.Popen[str], IO[str], IO[str]] | None = None

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_log:
            while not args.eval_only and should_continue(
                updates=updates, steps=args.steps, deadline=deadline
            ):
                if (
                    args.gpu_monitor_interval_ms > 0
                    and updates == 1
                    and gpu_monitor is None
                ):
                    gpu_monitor = start_gpu_monitor(
                        output_dir,
                        interval_ms=args.gpu_monitor_interval_ms,
                    )

                data_step = updates
                fetch_started = time.perf_counter()
                batch = train_batches.batch_at(data_step)
                fetch_seconds = time.perf_counter() - fetch_started
                step_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), data_step)
                update_started = time.perf_counter()
                call_args = training_call_args(
                    objective=args.objective,
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    rng=step_rng,
                    sigreg_reference_count=args.sigreg_reference_count,
                )
                loss, aux = executable(*call_args)
                jax.block_until_ready((loss, aux))
                update_seconds = time.perf_counter() - update_started

                updates += 1
                examples += args.batch_size
                if first_update_seconds is None:
                    first_update_seconds = update_seconds
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
    finally:
        if gpu_monitor is not None:
            stop_gpu_monitor(*gpu_monitor)

    training_wall_seconds = time.perf_counter() - training_wall_started
    final_val, final_val_seconds = evaluate(
        model,
        val_batches,
        count=args.eval_batches,
        seed=args.val_seed,
        deterministic_t=args.val_deterministic_t,
        objective=args.objective,
        sigreg_reference_count=args.sigreg_reference_count,
        collapse_diagnostics=args.collapse_diagnostics,
    )
    steady_update_seconds_mean = (
        float(np.mean(steady_update_seconds)) if steady_update_seconds else None
    )
    steady_update_seconds_p50 = (
        float(np.quantile(steady_update_seconds, 0.50))
        if steady_update_seconds
        else None
    )
    steady_update_seconds_p95 = (
        float(np.quantile(steady_update_seconds, 0.95))
        if steady_update_seconds
        else None
    )
    performance_seconds = steady_update_seconds_mean or first_update_seconds
    compiler_cost_analysis = compiler_cost_summary(compiler_cost_analysis_raw)
    performance = compiler_performance(compiler_cost_analysis, performance_seconds)
    throughput = {
        "steady_examples_per_second": (
            None
            if steady_update_seconds_mean is None
            else args.batch_size / steady_update_seconds_mean
        ),
        "steady_encoded_boards_per_second": (
            None
            if steady_update_seconds_mean is None
            else args.batch_size * (config.horizon + 1) / steady_update_seconds_mean
        ),
    }
    gpu_samples_path = output_dir / "gpu_samples.csv"
    gpu_monitor_summary = summarize_gpu_samples(gpu_samples_path)

    report = {
        **run_config,
        "restore_seconds": restore_seconds,
        "initial_validation_seconds": initial_val_seconds,
        "final_validation_seconds": final_val_seconds,
        "training_wall_seconds": training_wall_seconds,
        "explicit_compile_seconds": explicit_compile_seconds,
        "first_update_seconds": first_update_seconds,
        "compile_and_first_update_seconds": (
            None
            if first_update_seconds is None
            else first_update_seconds + (explicit_compile_seconds or 0.0)
        ),
        "steady_update_seconds_mean": steady_update_seconds_mean,
        "steady_update_seconds_p50": steady_update_seconds_p50,
        "steady_update_seconds_p95": steady_update_seconds_p95,
        "compiler_cost_analysis": compiler_cost_analysis,
        "compiler_memory_analysis": compiler_memory_analysis,
        "gpu_monitor": gpu_monitor_summary,
        "gpu_samples_path": str(gpu_samples_path) if gpu_samples_path.is_file() else None,
        **performance,
        **throughput,
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
                "explicit_compile_seconds": explicit_compile_seconds,
                "first_update_seconds": first_update_seconds,
                "steady_update_seconds_mean": report["steady_update_seconds_mean"],
                "steady_examples_per_second": report["steady_examples_per_second"],
                "compiler_estimated_achieved_tflops": report[
                    "compiler_estimated_achieved_tflops"
                ],
                "gpu_memory": report["gpu_memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
