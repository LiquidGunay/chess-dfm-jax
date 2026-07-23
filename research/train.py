#!/usr/bin/env python3
"""Single-GPU compatibility trainer for the clean research path.

The checkpoint-compatible model and stage-1 objective live directly in this
file so architecture and loss experiments have one production edit surface.
The trainer is not yet eligible for unattended autoresearch while the remaining
support boundaries and research protocol are being stabilized.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.nnx_bt4 import (  # noqa: E402
    BT4Model,
    TrainableEmbedding,
    TrainableParam,
    TrainableRMSNorm,
    TrainableTransformerStack,
    make_bt4_model,
    rounded_swiglu_dim,
)
from chess_dfm_jax.policy import (  # noqa: E402
    legacy_to_lc0_canonical_1858_index_map,
)
from research.import_legacy import import_legacy_checkpoint  # noqa: E402
from research.prepare import (  # noqa: E402
    REPO_ROOT,
    FixedTrajectoryBatches,
    load_asset_manifest,
    require_within_workspace,
    sha256_file,
    validate_environment,
    write_json,
)


AUTORESEARCH_READY = True
ARCHITECTURE_SOURCE = "research_train_local_model_and_loss"

# AUTORESEARCH EDIT SURFACE: change model/objective knobs here. Checkpoint
# metadata is loaded first, then these values, then explicit CLI overrides.
EXPERIMENT_OVERRIDES: dict[str, Any] = {
    "lr_warmup_steps": 0,
    "lr_decay_start_steps": 400,
    "lr_decay_steps": 800,
    "lr_min_ratio": 0.1,
    "jepa_target_sample_count": 1,
    "jepa_target_sampling_unit": "example_balanced",
    "bt4_encode_chunk_size": 0,
    # "dfm_active_layers": 3,
    # "jepa_projector_active_layers": 1,
    # "jepa_sampled_target_anchors": True,
    # Normalized autoresearch runs fix these at 0.0/1.0 on the CLI. They
    # remain explicit there because the default legacy parity path requires
    # the historical 1.0/0.0 pair.
    # "jepa_norm_loss_coeff": 0.0,
    # "jepa_pred_sigreg_coeff": 1.0,
    # "jepa_sigreg_estimator": "u_stat",
    # "jepa_sigreg_example_count": 64,
    # "jepa_target_stop_gradient": True,
    # "jepa_target_semantics": "ema",
    # "jepa_target_ema_decay": 0.99,
    # "jepa_target_variance_hinge_coeff": 1.0,
    # "jepa_target_variance_hinge_gamma": 0.9,
    # "jepa_state_fixed_unit_rms": True,
    "bt4_future_target_stop_gradient": True,
    "bt4_future_target_trainable_tail_layers": 1,
}

DEFAULT_RUN_ROOT = REPO_ROOT / "checkpoints" / "source" / "step0265000"
DEFAULT_CHECKPOINT_DIR = DEFAULT_RUN_ROOT / "checkpoints"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "source" / "extracted"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "trajectory_v3"

RESEARCH_CHECKPOINT_FORMAT = "chess-dfm-research-checkpoint-v1"
RESEARCH_CHECKPOINT_PATTERN = re.compile(r"^update(\d{8,})$")

GRADIENT_COMPONENT_NAMES = (
    "dfm_ce",
    "jepa_positive",
    "target_sigreg",
    "pred_sigreg",
    "fp32_legality",
)
TARGET_VARIANCE_HINGE_COMPONENT = "target_variance_hinge"
WDL_COMPONENT = "wdl"
BT4_POLICY_DISTILL_COMPONENT = "bt4_policy_distill"
ROOT_LEGAL_CONDITIONAL_COMPONENT = "root_legal_conditional_ce"
GRADIENT_GROUP_NAMES = ("backbone", "dfm", "jepa", "other", "all")
TARGET_VARIANCE_HINGE_EPSILON = 1e-4
JEPA_STATE_RMS_EPSILON = 1e-6
JEPA_FEEDBACK_MAX_STATE_RMS_RATIO = 0.5
CLASSICAL_SIDE_TO_MOVE_PLANE = 108
BT4_POLICY_DISTILL_TEMPERATURE = 1.0
_LEGACY_TO_CANONICAL_WHITE = (
    legacy_to_lc0_canonical_1858_index_map(black_to_move=False)
)
_LEGACY_TO_CANONICAL_BLACK = (
    legacy_to_lc0_canonical_1858_index_map(black_to_move=True)
)

UNEVALUATED_LEGACY_AUX_METRICS = frozenset(
    {
        "horizon_legality_loss",
        "horizon_legal_mass",
        "horizon_legality_evaluated",
        "jepa_cosine_loss",
        "jepa_normalized_mse",
        "mean_token_cosine",
        "mean_token_cosine_by_horizon",
        "pred_token_norm",
        "target_token_norm",
        "identity_jepa_loss",
        "identity_jepa_cosine_loss",
        "identity_mean_token_cosine",
        "jepa_shuffled_loss",
        "jepa_shuffled_cosine_loss",
        "jepa_shuffled_mean_token_cosine",
        "jepa_action_contrast_loss",
        "jepa_true_minus_shuffled",
        "jepa_loss_minus_identity",
    }
)


@dataclasses.dataclass
class JointLatentSASAConfig:
    token_dim: int = 256
    z_dim: int = 2048
    projector_layers: int = 2
    jepa_projector_active_layers: int = 0
    projector_num_heads: int = 8
    projector_mlp_dim: int = 0
    jepa_condition_dim: int = 0
    dfm_layers: int = 4
    dfm_active_layers: int = 0
    jepa_layers: int = 4
    jepa_rollout_mode: str = "recurrent"
    jepa_feedback_mode: str = "none"
    jepa_num_heads: int = 0
    jepa_mlp_dim: int = 0
    num_heads: int = 4
    mlp_dim: int = 1024
    learning_rate: float = 6e-4
    bt4_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    encoder_dtype: str = "bfloat16"
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    action_vocab_size: int = 1858
    horizon: int = 2
    loss_horizon: int = 0
    dfm_ce_coeff: float = 1.0
    bt4_policy_distill_coeff: float = 0.0
    root_legal_conditional_ce_coeff: float = 0.0
    dfm_first_action_loss_share: float = 0.0
    dfm_force_first_action_mask: bool = False
    dfm_training_time_power: float = 1.0
    first_legality_coeff: float = 7.64
    horizon_legality_coeff: float = 0.0
    legality_on_masked_only: bool = True
    jepa_positive_coeff: float = 1.0
    jepa_norm_loss_coeff: float = 1.0
    jepa_loss_type: str = "raw_mse"
    jepa_target_mode: str = "projected_bt4"
    jepa_target_stop_gradient: bool = False
    jepa_target_semantics: str = "online"
    jepa_target_ema_decay: float = 0.99
    jepa_target_variance_hinge_coeff: float = 0.0
    jepa_target_variance_hinge_gamma: float = 0.9
    jepa_target_sample_count: int = 0
    jepa_target_sampling_unit: str = "batch_shared"
    jepa_sampled_target_anchors: bool = False
    jepa_gamma: float = 1.0
    jepa_sigreg_coeff: float = 0.1
    jepa_pred_sigreg_coeff: float = 0.0
    jepa_sigreg_kind: str = "le_jepa"
    jepa_sigreg_estimator: str = "v_stat"
    jepa_sigreg_example_count: int = 0
    jepa_sigreg_proj_dim: int = 1024
    value_coeff: float = 0.0
    wdl_coeff: float = 0.0
    loss_clip_value: float = 0.0
    jepa_action_contrast_coeff: float = 0.0
    jepa_action_contrast_margin: float = 0.05
    contrastive_coeff: float = 0.0
    contrastive_temperature: float = 0.1
    candidate_count: int = 1
    use_qk_gain: bool = False
    use_qk_norm: bool = True
    use_xsa: bool = True
    use_muon: bool = True
    grad_clip_norm: float = 1.0
    lr_warmup_steps: int = 1000
    lr_decay_start_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 1.0
    skip_nonfinite_updates: bool = True
    unfreeze_bt4_encoder: bool = True
    bt4_freeze_backbone: bool = False
    bt4_future_target_stop_gradient: bool = False
    bt4_future_target_trainable_tail_layers: int = 0
    bt4_encode_chunk_size: int = 0
    jepa_state_rmsnorm: bool = False
    jepa_state_fixed_unit_rms: bool = False
    jepa_state_rms_scale_max: float = 2.0
    jepa_teacher_forcing_steps: int = 0
    jepa_delta_rms_clip: float = 0.5
    remat_blocks: bool = True
    scan_layers: bool = False


_INERT_OR_DEPRECATED_CONFIG_FIELDS = (
    "jepa_num_heads",
    "horizon_legality_coeff",
    "jepa_action_contrast_coeff",
    "jepa_action_contrast_margin",
    "contrastive_coeff",
    "contrastive_temperature",
    "candidate_count",
    "jepa_teacher_forcing_steps",
)


def _parse_compute_dtype(dtype_str: str) -> jnp.dtype:
    dtype = dtype_str.lower()
    mapping = {
        "float16": jnp.float16,
        "fp16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "bf16": jnp.bfloat16,
        "float32": jnp.float32,
        "fp32": jnp.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported compute dtype: {dtype_str}")
    return mapping[dtype]


def _weighted_horizon_mean(
    sample_values: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum(sample_values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def dfm_objective_horizon_weights(
    active_horizons: jnp.ndarray,
    *,
    first_action_loss_share: float,
) -> jnp.ndarray:
    """Return unit-sum DFM objective weights over active horizons."""

    active = jnp.asarray(active_horizons, dtype=jnp.float32)
    uniform = active / jnp.maximum(jnp.sum(active), 1.0)
    if first_action_loss_share == 0.0:
        return uniform

    remaining = active.at[0].set(0.0)
    tail = remaining / jnp.maximum(jnp.sum(remaining), 1.0)
    weights = tail * jnp.asarray(
        1.0 - first_action_loss_share,
        dtype=jnp.float32,
    )
    weights = weights.at[0].set(
        active[0]
        * jnp.asarray(first_action_loss_share, dtype=jnp.float32)
    )
    return weights / jnp.maximum(jnp.sum(weights), 1.0)


def balanced_example_target_horizons(
    rng: jax.Array,
    *,
    batch_size: int,
    horizon: int,
) -> jax.Array:
    """Assign one nearly equally represented target horizon per example."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, found {batch_size}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, found {horizon}")
    tiled_horizons = jnp.arange(batch_size, dtype=jnp.int32) % horizon
    return tiled_horizons[jax.random.permutation(rng, batch_size)]


class TargetVarianceHingeResult(NamedTuple):
    loss: jax.Array
    valid_count_by_horizon: jax.Array
    eligible_by_horizon: jax.Array
    feature_std_mean_by_horizon: jax.Array
    feature_std_p05_by_horizon: jax.Array
    feature_std_median_by_horizon: jax.Array
    active_fraction_by_horizon: jax.Array
    hinge_by_horizon: jax.Array


class LeJepaSigRegStatistics(NamedTuple):
    official: jax.Array
    v_stat_discrepancy: jax.Array
    u_stat_discrepancy: jax.Array
    valid_count: jax.Array


def per_horizon_target_variance_hinge(
    target_z: jax.Array,
    valid: jax.Array,
    *,
    gamma: float,
) -> TargetVarianceHingeResult:
    """Penalize low online-target feature variance independently per horizon."""

    target = jnp.asarray(target_z, dtype=jnp.float32)
    weight = jnp.maximum(jnp.asarray(valid, dtype=jnp.float32), 0.0)
    if target.ndim != 3:
        raise ValueError(
            "target_z must have shape [batch, horizon, feature], "
            f"got {target.shape}"
        )
    if weight.shape != target.shape[:2]:
        raise ValueError(
            f"valid must have shape {target.shape[:2]}, got {weight.shape}"
        )
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and positive, got {gamma}")

    active_sample = weight > 0.0
    safe_target = jnp.where(active_sample[..., None], target, 0.0)
    count = jnp.sum(weight, axis=0)
    denominator = jnp.maximum(count, 1.0)
    mean = (
        jnp.sum(safe_target * weight[..., None], axis=0)
        / denominator[:, None]
    )
    centered = jnp.where(
        active_sample[..., None],
        safe_target - mean[None, :, :],
        0.0,
    )
    variance = (
        jnp.sum(jnp.square(centered) * weight[..., None], axis=0)
        / denominator[:, None]
    )
    feature_std = jnp.sqrt(
        jnp.maximum(variance, 0.0)
        + jnp.asarray(TARGET_VARIANCE_HINGE_EPSILON, dtype=jnp.float32)
    )
    gamma_array = jnp.asarray(gamma, dtype=jnp.float32)
    per_feature_hinge = jax.nn.relu(gamma_array - feature_std)
    hinge_by_horizon = jnp.mean(per_feature_hinge, axis=-1)
    eligible = (count >= 2.0).astype(jnp.float32)
    loss = (
        jnp.sum(hinge_by_horizon * eligible)
        / jnp.maximum(jnp.sum(eligible), 1.0)
    )
    return TargetVarianceHingeResult(
        loss=loss,
        valid_count_by_horizon=count,
        eligible_by_horizon=eligible,
        feature_std_mean_by_horizon=jnp.mean(feature_std, axis=-1),
        feature_std_p05_by_horizon=jnp.quantile(
            feature_std,
            0.05,
            axis=-1,
        ),
        feature_std_median_by_horizon=jnp.quantile(
            feature_std,
            0.50,
            axis=-1,
        ),
        active_fraction_by_horizon=jnp.mean(
            (feature_std < gamma_array).astype(jnp.float32),
            axis=-1,
        ),
        hinge_by_horizon=hinge_by_horizon,
    )


def _sigreg_moments_loss(
    z: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Cheap SigReg surrogate without per-dimension sorting."""
    z = jnp.asarray(z, dtype=jnp.float32)
    if sample_weight is None:
        sample_weight = jnp.ones((z.shape[0],), dtype=jnp.float32)
    sample_weight = jnp.maximum(
        jnp.asarray(sample_weight, dtype=jnp.float32),
        0.0,
    )
    denom = jnp.maximum(jnp.sum(sample_weight), 1.0)
    weight = sample_weight[:, None]
    mean = jnp.sum(z * weight, axis=0) / denom
    centered = z - mean
    variance = jnp.sum(jnp.square(centered) * weight, axis=0) / denom
    return jnp.mean(jnp.square(mean)) + jnp.mean(
        jnp.square(variance - 1.0)
    )


def _sync_rng_for_pmap(
    rng: jnp.ndarray,
    axis_name: str | None,
) -> jnp.ndarray:
    """Use one projection seed across replicas, while still changing it each step."""
    rng = jnp.asarray(rng, dtype=jnp.uint32)
    if axis_name is None:
        return rng
    return jax.lax.pmin(rng, axis_name=axis_name)


def _le_jepa_sigreg_statistics(
    z: jnp.ndarray,
    *,
    proj_dim: int,
    rng: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
    axis_name: str | None = None,
    t_max: float = 3.0,
    n_points: int = 17,
    compute_u_stat: bool = False,
) -> LeJepaSigRegStatistics:
    """LeJEPA Epps-Pulley statistics with random slices.

    This follows the official structure:
      1. Project latents onto random unit directions.
      2. Match each projected empirical characteristic function to N(0, 1).
      3. Average the Epps-Pulley statistic over slices.

    The optional U-statistic removes the diagonal self-pair term from the
    squared empirical characteristic function. It is unbiased across
    independently sampled batches, but unlike the V-statistic it is not
    invariant to duplicating the same observations.
    """
    z = jnp.asarray(z, dtype=jnp.float32)
    sample_count, dim = z.shape
    if sample_count == 0:
        zero = jnp.zeros((), dtype=jnp.float32)
        return LeJepaSigRegStatistics(zero, zero, zero, zero)

    rng = _sync_rng_for_pmap(rng, axis_name)
    directions = jax.random.normal(
        rng,
        (dim, proj_dim),
        dtype=jnp.float32,
    )
    directions = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=0, keepdims=True),
        1e-12,
    )
    if sample_weight is None:
        sample_weight = jnp.ones(
            (sample_count,),
            dtype=jnp.float32,
        )
    sample_weight = jnp.maximum(
        jnp.asarray(sample_weight, dtype=jnp.float32),
        0.0,
    )

    projected = z @ directions

    t = jnp.linspace(0.0, t_max, n_points, dtype=jnp.float32)
    dt = jnp.asarray(
        t_max / max(n_points - 1, 1),
        dtype=jnp.float32,
    )
    weights = jnp.full((n_points,), 2.0 * dt, dtype=jnp.float32)
    weights = weights.at[0].set(dt)
    weights = weights.at[-1].set(dt)
    phi = jnp.exp(-0.5 * jnp.square(t))
    weights = weights * phi

    xt = projected[:, :, None] * t[None, None, :]
    weight = sample_weight[:, None, None]
    cos_sum = jnp.sum(jnp.cos(xt) * weight, axis=0)
    sin_sum = jnp.sum(jnp.sin(xt) * weight, axis=0)
    global_sample_count = jnp.sum(sample_weight)
    if axis_name is not None:
        cos_sum = jax.lax.psum(cos_sum, axis_name=axis_name)
        sin_sum = jax.lax.psum(sin_sum, axis_name=axis_name)
        global_sample_count = jax.lax.psum(
            global_sample_count,
            axis_name=axis_name,
        )
    denom = jnp.maximum(global_sample_count, 1.0)
    cos_mean = cos_sum / denom
    sin_mean = sin_sum / denom

    err = jnp.square(cos_mean - phi[None, :]) + jnp.square(sin_mean)
    per_slice = (err @ weights) * global_sample_count
    official = jnp.where(
        global_sample_count > 0.0,
        jnp.mean(per_slice),
        jnp.zeros((), dtype=jnp.float32),
    )
    v_stat_discrepancy = official / jnp.maximum(
        global_sample_count,
        1.0,
    )

    if compute_u_stat:
        squared_weight_sum = jnp.sum(jnp.square(sample_weight))
        if axis_name is not None:
            squared_weight_sum = jax.lax.psum(
                squared_weight_sum,
                axis_name=axis_name,
            )
        pair_denominator = (
            jnp.square(global_sample_count) - squared_weight_sum
        )
        off_diagonal_ecf_squared = (
            jnp.square(cos_sum)
            + jnp.square(sin_sum)
            - squared_weight_sum
        ) / jnp.maximum(pair_denominator, 1.0)
        u_error = (
            off_diagonal_ecf_squared
            - 2.0 * phi[None, :] * cos_mean
            + jnp.square(phi)[None, :]
        )
        u_stat_discrepancy = jnp.where(
            pair_denominator > 0.0,
            jnp.mean(u_error @ weights),
            jnp.zeros((), dtype=jnp.float32),
        )
    else:
        u_stat_discrepancy = jnp.zeros((), dtype=jnp.float32)

    return LeJepaSigRegStatistics(
        official,
        v_stat_discrepancy,
        u_stat_discrepancy,
        global_sample_count,
    )


def _official_le_jepa_sigreg_loss(
    z: jnp.ndarray,
    *,
    proj_dim: int,
    rng: jnp.ndarray,
    sample_weight: jnp.ndarray | None = None,
    axis_name: str | None = None,
    t_max: float = 3.0,
    n_points: int = 17,
) -> jnp.ndarray:
    """Published valid-count-scaled LeJEPA Epps-Pulley statistic."""

    return _le_jepa_sigreg_statistics(
        z,
        proj_dim=proj_dim,
        rng=rng,
        sample_weight=sample_weight,
        axis_name=axis_name,
        t_max=t_max,
        n_points=n_points,
        compute_u_stat=False,
    ).official


def _quantile_sigreg_loss(
    z: jnp.ndarray,
    d_proj: int = 128,
    rng: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if rng is None:
        rng = jax.random.PRNGKey(0)
    batch_size, dim = z.shape
    if batch_size == 0:
        return jnp.zeros(())
    W = jax.random.normal(rng, (dim, d_proj))
    W = W / jnp.maximum(
        jnp.linalg.norm(W, axis=0, keepdims=True),
        1e-12,
    )
    z_proj = jnp.matmul(z, W)
    z_proj_sorted = jnp.sort(z_proj, axis=0)
    p = (jnp.arange(batch_size, dtype=jnp.float32) + 0.5) / batch_size
    from jax.scipy.special import ndtri

    target_quantiles = ndtri(p)
    target_quantiles = jnp.expand_dims(target_quantiles, axis=-1)
    return jnp.mean(jnp.square(z_proj_sorted - target_quantiles))


def _clip_loss_preserve_gradient(
    loss: jnp.ndarray,
    clip_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scale an oversized scalar loss without making its gradient exactly zero."""
    loss = jnp.asarray(loss, dtype=jnp.float32)
    if clip_value <= 0.0:
        return loss, jnp.asarray(1.0, dtype=jnp.float32)
    clip = jnp.asarray(clip_value, dtype=jnp.float32)
    stopped_loss = jax.lax.stop_gradient(jnp.maximum(loss, 1e-6))
    scale = jnp.minimum(1.0, clip / stopped_loss)
    return loss * scale, scale


def mask_actions(
    actions: jnp.ndarray,
    mask_prob: jnp.ndarray,
    mask_token_id: int,
    rng: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply masking for Categorical Diffusion.
    actions: [B, K]
    mask_prob: [B]
    mask_token_id: int
    rng: PRNGKey
    """
    batch, K = actions.shape
    r = jax.random.uniform(rng, shape=(batch, K))
    mask = r < mask_prob[:, None]
    noisy_actions = jnp.where(mask, mask_token_id, actions)
    return noisy_actions, mask


def force_first_action_mask(
    noisy_actions: jnp.ndarray,
    is_masked: jnp.ndarray,
    *,
    mask_token_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Force only the played action to the DFM mask token."""

    return (
        noisy_actions.at[:, 0].set(mask_token_id),
        is_masked.at[:, 0].set(True),
    )


def transform_dfm_training_time(
    uniform_time: jnp.ndarray,
    *,
    power: float,
) -> jnp.ndarray:
    """Transform a uniform DFM time draw, preserving exact identity at one."""

    if power == 1.0:
        return uniform_time
    return jnp.power(
        uniform_time,
        jnp.asarray(power, dtype=uniform_time.dtype),
    )


def legal_mass_from_indices(
    probs: jnp.ndarray,
    legal_idx: jnp.ndarray,
    legal_count: jnp.ndarray,
) -> jnp.ndarray:
    safe_idx = jnp.clip(
        jnp.asarray(legal_idx, dtype=jnp.int32),
        0,
        probs.shape[-1] - 1,
    )
    legal_probs = jnp.take_along_axis(probs, safe_idx, axis=-1)
    slot_valid = jnp.arange(
        safe_idx.shape[-1],
        dtype=jnp.int32,
    ) < jnp.asarray(legal_count, dtype=jnp.int32)[..., None]
    return jnp.sum(jnp.where(slot_valid, legal_probs, 0.0), axis=-1)


class RootProposal(NamedTuple):
    action: jax.Array
    confidence: jax.Array
    feedback_gate: jax.Array


class Bt4PolicyDistillationResult(NamedTuple):
    loss: jax.Array
    eligible_count: jax.Array
    eligible_fraction: jax.Array
    side_plane_valid_fraction: jax.Array
    mapping_complete_fraction: jax.Array
    legal_mapping_coverage: jax.Array
    teacher_finite_fraction: jax.Array
    teacher_entropy: jax.Array
    teacher_played_nll: jax.Array
    teacher_played_accuracy: jax.Array
    teacher_student_top1_agreement: jax.Array


class RootLegalConditionalCeResult(NamedTuple):
    loss: jax.Array
    candidate_count: jax.Array
    eligible_count: jax.Array
    eligible_fraction: jax.Array
    legal_metadata_valid_fraction: jax.Array
    legal_count_valid_fraction: jax.Array
    played_in_legal_fraction: jax.Array
    finite_fraction: jax.Array
    mean_legal_count: jax.Array
    top1_accuracy: jax.Array


class JepaFeedbackResult(NamedTuple):
    latents: jax.Array
    delta_rms: jax.Array
    raw_feedback_rms: jax.Array
    applied_feedback_rms: jax.Array
    state_rms: jax.Array
    cap_fraction: jax.Array


def root_legal_mask_from_indices(
    legal_idx: jax.Array,
    legal_count: jax.Array,
    *,
    action_vocab_size: int,
) -> jax.Array:
    """Build one dense root mask without allowing invalid padding to overwrite."""

    indices = jnp.asarray(legal_idx, dtype=jnp.int32)
    counts = jnp.asarray(legal_count, dtype=jnp.int32)
    if indices.ndim != 2:
        raise ValueError(
            "root legal_idx must have shape [batch, slots], found "
            f"{indices.shape}"
        )
    if counts.shape != (indices.shape[0],):
        raise ValueError(
            "root legal_count must have shape "
            f"{(indices.shape[0],)}, found {counts.shape}"
        )
    safe_indices = jnp.clip(indices, 0, action_vocab_size - 1)
    valid_slots = (
        jnp.arange(indices.shape[1], dtype=jnp.int32)[None, :]
        < counts[:, None]
    )
    batch_indices = jnp.broadcast_to(
        jnp.arange(indices.shape[0], dtype=jnp.int32)[:, None],
        safe_indices.shape,
    )
    legal_counts = jnp.zeros(
        (indices.shape[0], action_vocab_size),
        dtype=jnp.int32,
    ).at[batch_indices, safe_indices].add(valid_slots.astype(jnp.int32))
    return legal_counts > 0


def root_legal_conditional_ce_from_logits(
    root_logits: jax.Array,
    played_actions: jax.Array,
    legal_idx: jax.Array,
    legal_count: jax.Array,
    sample_valid: jax.Array,
    root_masked: jax.Array,
    legal_metadata_valid: jax.Array | None = None,
) -> RootLegalConditionalCeResult:
    """Compute played-action CE only over each stored root legal set."""

    logits = jnp.asarray(root_logits, dtype=jnp.float32)
    if logits.ndim != 2:
        raise ValueError(
            "root_logits must have shape [batch, actions], found "
            f"{logits.shape}"
        )
    batch_size, action_vocab_size = logits.shape
    indices = jnp.asarray(legal_idx, dtype=jnp.int32)
    if indices.ndim != 2 or indices.shape[0] != batch_size:
        raise ValueError(
            "legal_idx must have shape [batch, slots], found "
            f"{indices.shape}"
        )
    actions = jnp.asarray(played_actions, dtype=jnp.int32)
    counts = jnp.asarray(legal_count, dtype=jnp.int32)
    valid = jnp.asarray(sample_valid, dtype=jnp.bool_)
    masked = jnp.asarray(root_masked, dtype=jnp.bool_)
    for name, value in (
        ("played_actions", actions),
        ("legal_count", counts),
        ("sample_valid", valid),
        ("root_masked", masked),
    ):
        if value.shape != (batch_size,):
            raise ValueError(
                f"{name} must have shape {(batch_size,)}, found "
                f"{value.shape}"
            )
    if legal_metadata_valid is None:
        metadata_valid = jnp.ones((batch_size,), dtype=jnp.bool_)
    else:
        metadata_valid = jnp.asarray(
            legal_metadata_valid,
            dtype=jnp.bool_,
        )
        if metadata_valid.shape != (batch_size,):
            raise ValueError(
                "legal_metadata_valid must have shape "
                f"{(batch_size,)}, found {metadata_valid.shape}"
            )

    slot_count = indices.shape[1]
    count_valid = (counts > 0) & (counts <= slot_count)
    slot_valid = (
        jnp.arange(slot_count, dtype=jnp.int32)[None, :]
        < counts[:, None]
    )
    index_valid = (indices >= 0) & (indices < action_vocab_size)
    legal_support = slot_valid & index_valid
    support_nonempty = jnp.any(legal_support, axis=-1)
    all_legal_indices_valid = jnp.all(
        (~slot_valid) | index_valid,
        axis=-1,
    )

    safe_indices = jnp.clip(indices, 0, action_vocab_size - 1)
    gathered_logits = jnp.take_along_axis(
        logits,
        safe_indices,
        axis=-1,
    )
    gathered_finite = jnp.isfinite(gathered_logits)
    legal_logits_finite = jnp.all(
        (~legal_support) | gathered_finite,
        axis=-1,
    )
    finite_gathered_logits = jnp.nan_to_num(
        gathered_logits,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_support = legal_support | (~support_nonempty[:, None])
    legal_log_normalizer = jax.nn.logsumexp(
        jnp.where(
            safe_support,
            finite_gathered_logits,
            -jnp.inf,
        ),
        axis=-1,
    )

    action_in_range = (actions >= 0) & (actions < action_vocab_size)
    safe_actions = jnp.clip(actions, 0, action_vocab_size - 1)
    played_logits = jnp.take_along_axis(
        logits,
        safe_actions[:, None],
        axis=-1,
    )[:, 0]
    played_finite = jnp.isfinite(played_logits)
    finite_played_logits = jnp.nan_to_num(
        played_logits,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    played_in_legal = action_in_range & jnp.any(
        legal_support & (indices == actions[:, None]),
        axis=-1,
    )
    finite = legal_logits_finite & played_finite

    candidate = valid & masked
    eligible = (
        candidate
        & metadata_valid
        & count_valid
        & support_nonempty
        & all_legal_indices_valid
        & played_in_legal
        & finite
    )
    candidate_weight = candidate.astype(jnp.float32)
    eligible_weight = eligible.astype(jnp.float32)
    candidate_count = jnp.sum(candidate_weight)
    eligible_count = jnp.sum(eligible_weight)

    loss_by_sample = legal_log_normalizer - finite_played_logits
    loss = jnp.sum(
        jnp.where(eligible, loss_by_sample, 0.0)
    ) / jnp.maximum(eligible_count, 1.0)

    top1_slot = jnp.argmax(
        jnp.where(
            safe_support,
            finite_gathered_logits,
            -jnp.inf,
        ),
        axis=-1,
    )
    top1_action = jnp.take_along_axis(
        safe_indices,
        top1_slot[:, None],
        axis=-1,
    )[:, 0]

    def candidate_fraction(values: jax.Array) -> jax.Array:
        return jnp.sum(
            jnp.asarray(values, dtype=jnp.float32) * candidate_weight
        ) / jnp.maximum(candidate_count, 1.0)

    return RootLegalConditionalCeResult(
        loss=loss,
        candidate_count=candidate_count,
        eligible_count=eligible_count,
        eligible_fraction=eligible_count
        / jnp.maximum(candidate_count, 1.0),
        legal_metadata_valid_fraction=candidate_fraction(metadata_valid),
        legal_count_valid_fraction=candidate_fraction(
            count_valid
            & support_nonempty
            & all_legal_indices_valid
        ),
        played_in_legal_fraction=candidate_fraction(played_in_legal),
        finite_fraction=candidate_fraction(finite),
        mean_legal_count=(
            jnp.sum(
                jnp.asarray(counts, dtype=jnp.float32)
                * eligible_weight
            )
            / jnp.maximum(eligible_count, 1.0)
        ),
        top1_accuracy=(
            jnp.sum(
                (top1_action == actions).astype(jnp.float32)
                * eligible_weight
            )
            / jnp.maximum(eligible_count, 1.0)
        ),
    )


def bt4_policy_distillation_from_logits(
    teacher_canonical_logits: jax.Array,
    student_legacy_logits: jax.Array,
    current_planes: jax.Array,
    played_actions: jax.Array,
    legal_idx: jax.Array,
    legal_count: jax.Array,
    sample_valid: jax.Array,
    root_masked: jax.Array,
    legal_metadata_valid: jax.Array | None = None,
) -> Bt4PolicyDistillationResult:
    """Match legal root rankings across canonical and legacy policy codecs."""

    teacher = jnp.asarray(teacher_canonical_logits, dtype=jnp.float32)
    student = jnp.asarray(student_legacy_logits, dtype=jnp.float32)
    if (
        teacher.ndim != 2
        or teacher.shape[-1] != _LEGACY_TO_CANONICAL_WHITE.shape[0]
    ):
        raise ValueError(
            "teacher_canonical_logits must have shape "
            f"[batch, {_LEGACY_TO_CANONICAL_WHITE.shape[0]}], found "
            f"{teacher.shape}"
        )
    if student.shape != teacher.shape:
        raise ValueError(
            "student_legacy_logits must match teacher logits, found "
            f"{student.shape} != {teacher.shape}"
        )
    batch_size, action_vocab_size = teacher.shape
    planes = jnp.asarray(current_planes)
    if (
        planes.ndim != 4
        or planes.shape[0] != batch_size
        or planes.shape[1] <= CLASSICAL_SIDE_TO_MOVE_PLANE
    ):
        raise ValueError(
            "current_planes must have shape [batch, channels, height, width] "
            f"with channels>{CLASSICAL_SIDE_TO_MOVE_PLANE}, found "
            f"{planes.shape}"
        )
    actions = jnp.asarray(played_actions, dtype=jnp.int32)
    counts = jnp.asarray(legal_count, dtype=jnp.int32)
    valid = jnp.asarray(sample_valid, dtype=jnp.bool_)
    masked = jnp.asarray(root_masked, dtype=jnp.bool_)
    for name, value in (
        ("played_actions", actions),
        ("legal_count", counts),
        ("sample_valid", valid),
        ("root_masked", masked),
    ):
        if value.shape != (batch_size,):
            raise ValueError(
                f"{name} must have shape {(batch_size,)}, found "
                f"{value.shape}"
            )
    if legal_metadata_valid is None:
        metadata_valid = jnp.ones((batch_size,), dtype=jnp.bool_)
    else:
        metadata_valid = jnp.asarray(
            legal_metadata_valid,
            dtype=jnp.bool_,
        )
        if metadata_valid.shape != (batch_size,):
            raise ValueError(
                "legal_metadata_valid must have shape "
                f"{(batch_size,)}, found {metadata_valid.shape}"
            )

    root_legal_mask = root_legal_mask_from_indices(
        legal_idx,
        counts,
        action_vocab_size=action_vocab_size,
    )
    side_plane = planes[:, CLASSICAL_SIDE_TO_MOVE_PLANE, :, :]
    side_is_white = jnp.all(side_plane == 0, axis=(1, 2))
    side_is_black = jnp.all(side_plane == 1, axis=(1, 2))
    side_valid = side_is_white | side_is_black
    black_to_move = jax.lax.stop_gradient(side_is_black)

    white_map = jnp.asarray(
        _LEGACY_TO_CANONICAL_WHITE,
        dtype=jnp.int32,
    )
    black_map = jnp.asarray(
        _LEGACY_TO_CANONICAL_BLACK,
        dtype=jnp.int32,
    )
    canonical_index = jnp.where(
        black_to_move[:, None],
        black_map[None, :],
        white_map[None, :],
    )
    mapping_valid = canonical_index >= 0
    safe_canonical_index = jnp.clip(
        canonical_index,
        0,
        action_vocab_size - 1,
    )

    teacher = jax.lax.stop_gradient(teacher)
    teacher_finite = jnp.all(jnp.isfinite(teacher), axis=-1)
    finite_teacher = jnp.nan_to_num(
        teacher,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    teacher_legacy = jnp.take_along_axis(
        finite_teacher,
        safe_canonical_index,
        axis=-1,
    )
    teacher_support = root_legal_mask & mapping_valid
    legal_nonempty = jnp.any(root_legal_mask, axis=-1)
    teacher_support_nonempty = jnp.any(teacher_support, axis=-1)
    mapping_complete = jnp.all(
        (~root_legal_mask) | mapping_valid,
        axis=-1,
    )

    safe_teacher_support = teacher_support | (
        ~teacher_support_nonempty[:, None]
    )
    safe_student_support = root_legal_mask | (~legal_nonempty[:, None])
    teacher_log_probs = jax.nn.log_softmax(
        jnp.where(
            safe_teacher_support,
            teacher_legacy / BT4_POLICY_DISTILL_TEMPERATURE,
            -jnp.inf,
        ),
        axis=-1,
    )
    teacher_probs = jax.lax.stop_gradient(jnp.exp(teacher_log_probs))
    teacher_log_probs = jax.lax.stop_gradient(teacher_log_probs)
    student_log_probs = jax.nn.log_softmax(
        jnp.where(
            safe_student_support,
            student / BT4_POLICY_DISTILL_TEMPERATURE,
            -jnp.inf,
        ),
        axis=-1,
    )
    kl_by_sample = jnp.sum(
        jnp.where(
            teacher_support,
            teacher_probs * (teacher_log_probs - student_log_probs),
            0.0,
        ),
        axis=-1,
    )
    eligible = (
        valid
        & masked
        & metadata_valid
        & side_valid
        & legal_nonempty
        & teacher_support_nonempty
        & mapping_complete
        & teacher_finite
    )
    eligible_weight = eligible.astype(jnp.float32)
    eligible_count = jnp.sum(eligible_weight)
    loss = jnp.sum(kl_by_sample * eligible_weight) / jnp.maximum(
        eligible_count,
        1.0,
    )

    teacher_entropy_by_sample = -jnp.sum(
        jnp.where(
            teacher_support,
            teacher_probs * teacher_log_probs,
            0.0,
        ),
        axis=-1,
    )
    teacher_top1 = jnp.argmax(teacher_log_probs, axis=-1)
    student_top1 = jnp.argmax(student_log_probs, axis=-1)
    safe_actions = jnp.clip(actions, 0, action_vocab_size - 1)
    played_in_support = jnp.take_along_axis(
        teacher_support,
        safe_actions[:, None],
        axis=-1,
    )[:, 0] & (actions >= 0) & (actions < action_vocab_size)
    label_eligible_weight = (
        eligible & played_in_support
    ).astype(jnp.float32)
    label_eligible_count = jnp.sum(label_eligible_weight)
    teacher_played_log_prob = jnp.take_along_axis(
        teacher_log_probs,
        safe_actions[:, None],
        axis=-1,
    )[:, 0]

    valid_weight = valid.astype(jnp.float32)
    valid_count = jnp.sum(valid_weight)
    legal_slot_count = jnp.sum(
        root_legal_mask.astype(jnp.float32) * valid_weight[:, None]
    )
    mapped_legal_slot_count = jnp.sum(
        teacher_support.astype(jnp.float32) * valid_weight[:, None]
    )

    def eligible_mean(values: jax.Array) -> jax.Array:
        return jnp.sum(
            jnp.asarray(values, dtype=jnp.float32) * eligible_weight
        ) / jnp.maximum(eligible_count, 1.0)

    return Bt4PolicyDistillationResult(
        loss=loss,
        eligible_count=eligible_count,
        eligible_fraction=eligible_count / jnp.maximum(valid_count, 1.0),
        side_plane_valid_fraction=(
            jnp.sum(side_valid.astype(jnp.float32) * valid_weight)
            / jnp.maximum(valid_count, 1.0)
        ),
        mapping_complete_fraction=(
            jnp.sum(mapping_complete.astype(jnp.float32) * valid_weight)
            / jnp.maximum(valid_count, 1.0)
        ),
        legal_mapping_coverage=(
            mapped_legal_slot_count
            / jnp.maximum(legal_slot_count, 1.0)
        ),
        teacher_finite_fraction=(
            jnp.sum(teacher_finite.astype(jnp.float32) * valid_weight)
            / jnp.maximum(valid_count, 1.0)
        ),
        teacher_entropy=eligible_mean(teacher_entropy_by_sample),
        teacher_played_nll=(
            jnp.sum(
                jnp.where(
                    label_eligible_weight > 0.0,
                    -teacher_played_log_prob,
                    0.0,
                )
            )
            / jnp.maximum(label_eligible_count, 1.0)
        ),
        teacher_played_accuracy=(
            jnp.sum(
                (teacher_top1 == actions).astype(jnp.float32)
                * label_eligible_weight
            )
            / jnp.maximum(label_eligible_count, 1.0)
        ),
        teacher_student_top1_agreement=eligible_mean(
            teacher_top1 == student_top1
        ),
    )


def proposal_from_root_logits(
    root_logits: jax.Array,
    current_root_action: jax.Array,
    root_legal_mask: jax.Array,
    root_legal_valid: jax.Array,
    *,
    mask_token_id: int,
) -> RootProposal:
    """Select a target-independent root proposal and stopped confidence gate."""

    logits = jnp.asarray(root_logits, dtype=jnp.float32)
    current = jnp.asarray(current_root_action, dtype=jnp.int32)
    legal_mask = jnp.asarray(root_legal_mask, dtype=jnp.bool_)
    legal_valid = jnp.asarray(root_legal_valid, dtype=jnp.bool_)
    expected_mask_shape = (logits.shape[0], logits.shape[-1])
    if legal_mask.shape != expected_mask_shape:
        raise ValueError(
            "root_legal_mask must have shape "
            f"{expected_mask_shape}, found {legal_mask.shape}"
        )
    if current.shape != (logits.shape[0],):
        raise ValueError(
            "current_root_action must have shape "
            f"{(logits.shape[0],)}, found {current.shape}"
        )
    if legal_valid.shape != (logits.shape[0],):
        raise ValueError(
            "root_legal_valid must have shape "
            f"{(logits.shape[0],)}, found {legal_valid.shape}"
        )

    # Invalid training metadata uses an all-action numerical fallback only to
    # keep softmax finite. Its feedback gate is exactly zero below.
    safe_mask = legal_mask | ~legal_valid[:, None]
    legal_log_probs = jax.nn.log_softmax(
        jnp.where(safe_mask, logits, -jnp.inf),
        axis=-1,
    )
    prediction = jnp.argmax(legal_log_probs, axis=-1).astype(jnp.int32)
    confidence = jax.lax.stop_gradient(
        jnp.exp(jnp.max(legal_log_probs, axis=-1))
    )
    root_is_masked = current == mask_token_id
    action = jnp.where(root_is_masked, prediction, current)
    gate = jnp.where(root_is_masked, confidence, 1.0)
    gate = jnp.where(legal_valid, gate, 0.0)
    return RootProposal(
        action=jax.lax.stop_gradient(action),
        confidence=confidence,
        feedback_gate=jax.lax.stop_gradient(gate),
    )


def _jepa_positive_target_vectors(
    z_jepa: jax.Array,
    target_z: jax.Array,
    pred_z: jax.Array,
    *,
    target_mode: str,
    stop_gradient: bool,
) -> jax.Array:
    """Select the positive JEPA target and optionally detach only that path."""
    if target_mode == "current_repeat":
        with jax.named_scope("joint_jepa_current_repeat_target"):
            target_vectors = jnp.broadcast_to(
                z_jepa[:, None, :],
                pred_z.shape,
            )
    elif target_mode == "projected_bt4":
        target_vectors = target_z
    else:
        raise ValueError(f"Unsupported jepa_target_mode: {target_mode!r}.")
    if stop_gradient:
        return jax.lax.stop_gradient(target_vectors)
    return target_vectors


class LinearAdapter(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
    ):
        output_dim = input_dim if output_dim is None else output_dim
        self.w = TrainableParam(
            jax.random.normal(rngs.params(), (input_dim, output_dim), dtype=param_dtype)
            / np.sqrt(max(input_dim, 1))
        )
        self.b = TrainableParam(jnp.zeros((output_dim,), dtype=param_dtype))
        self.compute_dtype = jnp.dtype(compute_dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (
            x @ jnp.asarray(self.w[...], dtype=self.compute_dtype)
            + jnp.asarray(self.b[...], dtype=self.compute_dtype)
        )


def _transformer_stack_prefix_params(
    blocks: TrainableTransformerStack,
    *,
    active_layers: int,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, ...]:
    """Materialize one checkpoint-compatible transformer prefix."""

    qk_gain = (
        jnp.asarray(
            blocks.qk_gain[...],
            dtype=compute_dtype,
        )[:active_layers]
        if blocks.use_qk_gain
        else jnp.ones((active_layers,), dtype=compute_dtype)
    )
    return (
        jnp.asarray(blocks.attn_norm_scale[...], dtype=compute_dtype)[
            :active_layers
        ],
        jnp.asarray(blocks.mlp_norm_scale[...], dtype=compute_dtype)[
            :active_layers
        ],
        jnp.asarray(blocks.w_qkv[...], dtype=compute_dtype)[:active_layers],
        jnp.asarray(blocks.b_qkv[...], dtype=compute_dtype)[:active_layers],
        jnp.asarray(blocks.w_o[...], dtype=compute_dtype)[:active_layers],
        jnp.asarray(blocks.b_o[...], dtype=compute_dtype)[:active_layers],
        jnp.asarray(blocks.w_gate_up[...], dtype=compute_dtype)[
            :active_layers
        ],
        jnp.asarray(blocks.b_gate_up[...], dtype=compute_dtype)[
            :active_layers
        ],
        jnp.asarray(blocks.w_down[...], dtype=compute_dtype)[:active_layers],
        jnp.asarray(blocks.b_down[...], dtype=compute_dtype)[:active_layers],
        qk_gain,
    )


def _apply_transformer_stack_prefix(
    blocks: TrainableTransformerStack,
    seq: jnp.ndarray,
    *,
    active_layers: int,
    compute_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Run a stored transformer prefix while preserving the full fast path."""

    if active_layers in (0, blocks.num_layers):
        return blocks(seq)

    params = _transformer_stack_prefix_params(
        blocks,
        active_layers=active_layers,
        compute_dtype=compute_dtype,
    )

    def apply_layer(
        carry: jnp.ndarray,
        layer_params: tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        return blocks._layer(carry, layer_params)

    layer_fn = (
        jax.checkpoint(apply_layer, prevent_cse=False)
        if blocks.remat_blocks
        else apply_layer
    )
    if blocks.scan_layers:

        def body(carry, layer_params):
            return layer_fn(carry, layer_params), None

        seq, _ = jax.lax.scan(
            body,
            jnp.asarray(seq, dtype=compute_dtype),
            params,
        )
        return seq

    for layer_index in range(active_layers):
        layer_params = tuple(param[layer_index] for param in params)
        seq = layer_fn(seq, layer_params)
    return seq


class StateVectorProjector(nnx.Module):
    """Project BT4 square tokens to one global JEPA state vector."""

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        *,
        num_layers: int,
        active_layers: int,
        num_heads: int,
        mlp_dim: int,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        use_qk_gain: bool = False,
        use_qk_norm: bool = False,
        use_xsa: bool = False,
        scan_layers: bool = False,
        remat_blocks: bool = True,
    ):
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        self.compute_dtype = jnp.dtype(compute_dtype)
        if (
            isinstance(active_layers, bool)
            or not isinstance(active_layers, (int, np.integer))
            or not 0 <= active_layers <= num_layers
        ):
            raise ValueError(
                "active projector layers must be an integer in "
                f"[0, {num_layers}], found {active_layers!r}"
            )
        self.active_layers = int(active_layers)
        mlp_dim = int(mlp_dim) if mlp_dim > 0 else int(4 * z_dim)
        self.in_proj = TrainableParam(
            jax.random.normal(rngs.params(), (input_dim, z_dim), dtype=param_dtype)
            / np.sqrt(max(input_dim, 1))
        )
        self.in_bias = TrainableParam(jnp.zeros((z_dim,), dtype=param_dtype))
        self.cls = TrainableParam(jnp.zeros((1, z_dim), dtype=param_dtype))
        self.pos_embed = TrainableParam(
            jax.random.normal(rngs.params(), (65, z_dim), dtype=param_dtype)
            * (0.02 / np.sqrt(max(z_dim, 1)))
        )
        self.blocks = TrainableTransformerStack(
            num_layers=num_layers,
            width=z_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            use_qk_gain=use_qk_gain,
            use_qk_norm=use_qk_norm,
            use_xsa=use_xsa,
            scan_layers=scan_layers,
            remat_blocks=remat_blocks,
        )

    @property
    def effective_active_layers(self) -> int:
        return (
            self.blocks.num_layers
            if self.active_layers == 0
            else self.active_layers
        )

    def _active_block_params(self) -> tuple[jnp.ndarray, ...]:
        return _transformer_stack_prefix_params(
            self.blocks,
            active_layers=self.effective_active_layers,
            compute_dtype=self.compute_dtype,
        )

    def _apply_active_blocks(self, seq: jnp.ndarray) -> jnp.ndarray:
        return _apply_transformer_stack_prefix(
            self.blocks,
            seq,
            active_layers=self.active_layers,
            compute_dtype=self.compute_dtype,
        )

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        tokens = jnp.asarray(tokens, dtype=self.compute_dtype)
        batch = tokens.shape[0]
        square_tokens = (
            tokens.reshape((batch * 64, self.input_dim))
            @ jnp.asarray(self.in_proj[...], dtype=self.compute_dtype)
            + jnp.asarray(self.in_bias[...], dtype=self.compute_dtype)
        ).reshape((batch, 64, self.z_dim))
        cls = jnp.broadcast_to(
            jnp.asarray(self.cls[...], dtype=self.compute_dtype),
            (batch, 1, self.z_dim),
        )
        seq = jnp.concatenate([cls, square_tokens], axis=1)
        seq = seq + jnp.asarray(self.pos_embed[...], dtype=self.compute_dtype)[None, :, :]
        seq = self._apply_active_blocks(seq)
        return jnp.asarray(seq[:, 0, :], dtype=self.compute_dtype)


class VectorValueWDLHead(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
    ):
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.w1 = TrainableParam(
            jax.random.normal(rngs.params(), (input_dim, hidden_dim), dtype=param_dtype)
            / np.sqrt(max(input_dim, 1))
        )
        self.b1 = TrainableParam(jnp.zeros((hidden_dim,), dtype=param_dtype))
        self.value_w = TrainableParam(
            jax.random.normal(rngs.params(), (hidden_dim, 1), dtype=param_dtype)
            / np.sqrt(max(hidden_dim, 1))
        )
        self.value_b = TrainableParam(jnp.zeros((1,), dtype=param_dtype))
        self.wdl_w = TrainableParam(
            jax.random.normal(rngs.params(), (hidden_dim, 3), dtype=param_dtype)
            / np.sqrt(max(hidden_dim, 1))
        )
        self.wdl_b = TrainableParam(jnp.zeros((3,), dtype=param_dtype))

    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        z = jnp.asarray(z, dtype=self.compute_dtype)
        hidden = (
            z @ jnp.asarray(self.w1[...], dtype=self.compute_dtype)
            + jnp.asarray(self.b1[...], dtype=self.compute_dtype)
        )
        hidden = jax.nn.silu(hidden)
        value = (
            hidden @ jnp.asarray(self.value_w[...], dtype=self.compute_dtype)
            + jnp.asarray(self.value_b[...], dtype=self.compute_dtype)
        )
        wdl = (
            hidden @ jnp.asarray(self.wdl_w[...], dtype=self.compute_dtype)
            + jnp.asarray(self.wdl_b[...], dtype=self.compute_dtype)
        )
        return value.squeeze(-1), wdl


class ConditionedVectorTransition(nnx.Module):
    """AdaLN-style recurrent JEPA transition over projected state vectors."""

    def __init__(
        self,
        z_dim: int,
        condition_dim: int,
        *,
        num_layers: int,
        mlp_dim: int,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        init_scale: float = 1e-3,
        rms_eps: float = 1e-6,
        delta_rms_clip: float = 0.0,
    ):
        self.z_dim = int(z_dim)
        self.condition_dim = int(condition_dim)
        self.num_layers = int(num_layers)
        self.swiglu_dim = rounded_swiglu_dim(mlp_dim)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.rms_eps = float(rms_eps)
        self.delta_rms_clip = float(delta_rms_clip)

        def normal(shape, scale):
            return jax.random.normal(rngs.params(), shape, dtype=param_dtype) * scale

        self.norm_scale = TrainableParam(jnp.ones((num_layers, z_dim), dtype=param_dtype))
        self.cond_w = TrainableParam(
            jnp.zeros((num_layers, condition_dim, 2 * z_dim), dtype=param_dtype)
        )
        self.cond_b = TrainableParam(
            jnp.zeros((num_layers, 2 * z_dim), dtype=param_dtype)
        )
        self.w_gate_up = TrainableParam(
            normal(
                (num_layers, z_dim, 2 * self.swiglu_dim),
                1.0 / np.sqrt(max(z_dim, 1)),
            )
        )
        self.b_gate_up = TrainableParam(
            jnp.zeros((num_layers, 2 * self.swiglu_dim), dtype=param_dtype)
        )
        self.w_down = TrainableParam(
            normal(
                (num_layers, self.swiglu_dim, z_dim),
                init_scale / np.sqrt(max(self.swiglu_dim, 1)),
            )
        )
        self.b_down = TrainableParam(
            jnp.zeros((num_layers, z_dim), dtype=param_dtype)
        )

    def _rms_norm(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        stats_x = jnp.asarray(x, dtype=jnp.float32)
        inv_rms = jax.lax.rsqrt(
            jnp.mean(jnp.square(stats_x), axis=-1, keepdims=True) + self.rms_eps
        )
        out = stats_x * inv_rms * jnp.asarray(scale, dtype=jnp.float32)
        return jnp.asarray(out, dtype=self.compute_dtype)

    def _layer(
        self,
        z: jnp.ndarray,
        condition: jnp.ndarray,
        params: tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        norm_scale, cond_w, cond_b, w_gate_up, b_gate_up, w_down, b_down = params
        z = jnp.asarray(z, dtype=self.compute_dtype)
        condition = jnp.asarray(condition, dtype=self.compute_dtype)
        shift_scale = condition @ cond_w + cond_b
        shift, scale = jnp.split(shift_scale, 2, axis=-1)
        hidden = self._rms_norm(z, norm_scale)
        hidden = hidden * (1.0 + scale) + shift
        gate_up = hidden @ w_gate_up + b_gate_up
        gate, up = jnp.split(gate_up, 2, axis=-1)
        delta = (jax.nn.silu(gate) * up) @ w_down + b_down
        if self.delta_rms_clip > 0.0:
            delta_f32 = jnp.asarray(delta, dtype=jnp.float32)
            delta_rms = jnp.sqrt(
                jnp.mean(jnp.square(delta_f32), axis=-1, keepdims=True)
                + self.rms_eps
            )
            delta_scale = jnp.minimum(
                1.0,
                jnp.asarray(self.delta_rms_clip, dtype=jnp.float32) / delta_rms,
            )
            delta = delta_f32 * delta_scale
        return jnp.asarray(z + delta, dtype=self.compute_dtype)

    def __call__(self, z: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
        params = (
            jnp.asarray(self.norm_scale[...], dtype=self.compute_dtype),
            jnp.asarray(self.cond_w[...], dtype=self.compute_dtype),
            jnp.asarray(self.cond_b[...], dtype=self.compute_dtype),
            jnp.asarray(self.w_gate_up[...], dtype=self.compute_dtype),
            jnp.asarray(self.b_gate_up[...], dtype=self.compute_dtype),
            jnp.asarray(self.w_down[...], dtype=self.compute_dtype),
            jnp.asarray(self.b_down[...], dtype=self.compute_dtype),
        )

        def body(carry, layer_params):
            return self._layer(carry, condition, layer_params), None

        z, _ = jax.lax.scan(
            body,
            jnp.asarray(z, dtype=self.compute_dtype),
            params,
        )
        return z


class JointLatentSASAModel(nnx.Module):
    """Joint DFM and configurable projected-state JEPA model."""

    def __init__(
        self,
        encoder: BT4Model,
        config: JointLatentSASAConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = encoder
        self.config = config
        param_dtype = _parse_compute_dtype(config.param_dtype)
        compute_dtype = _parse_compute_dtype(config.compute_dtype)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.encoder_dim = int(encoder.embedding_size)
        self.z_dim = int(config.z_dim)
        validate_bt4_freeze_config(config)
        validate_bt4_future_target_stop_gradient_config(config)
        if config.bt4_policy_distill_coeff != 0.0:
            if config.action_vocab_size != _LEGACY_TO_CANONICAL_WHITE.shape[0]:
                raise ValueError(
                    "BT4 policy distillation requires action_vocab_size="
                    f"{_LEGACY_TO_CANONICAL_WHITE.shape[0]}, found "
                    f"{config.action_vocab_size}."
                )
            if not hasattr(encoder, "policy_head"):
                raise ValueError(
                    "BT4 policy distillation requires encoder.policy_head."
                )
        trainable_future_tail = int(
            config.bt4_future_target_trainable_tail_layers
        )
        if trainable_future_tail > 0:
            if not hasattr(encoder, "layers"):
                raise ValueError(
                    "A trainable future BT4 tail requires an encoder.layers "
                    "transformer stack"
                )
            encoder_layer_count = len(encoder.layers)
            if trainable_future_tail > encoder_layer_count:
                raise ValueError(
                    "bt4_future_target_trainable_tail_layers exceeds the "
                    f"loaded encoder depth: {trainable_future_tail} > "
                    f"{encoder_layer_count}"
                )
        if config.jepa_loss_type != "raw_mse":
            raise ValueError(
                "Joint projected-vector training supports raw_mse JEPA loss only."
            )
        if config.jepa_target_mode not in ("projected_bt4", "current_repeat"):
            raise ValueError(
                f"Unsupported jepa_target_mode: {config.jepa_target_mode!r}."
            )
        if config.jepa_rollout_mode not in (
            "recurrent",
            "direct_sequence",
        ):
            raise ValueError(
                "jepa_rollout_mode must be 'recurrent' or "
                f"'direct_sequence', found {config.jepa_rollout_mode!r}."
            )
        if (
            config.jepa_rollout_mode == "direct_sequence"
            and config.jepa_sampled_target_anchors
        ):
            raise ValueError(
                "jepa_rollout_mode='direct_sequence' does not support "
                "sampled-target anchors."
            )
        if config.jepa_feedback_mode not in (
            "none",
            "final_pass_adjoint",
        ):
            raise ValueError(
                "jepa_feedback_mode must be 'none' or "
                f"'final_pass_adjoint', found "
                f"{config.jepa_feedback_mode!r}."
            )
        if (
            config.jepa_feedback_mode == "final_pass_adjoint"
            and config.jepa_rollout_mode != "recurrent"
        ):
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires "
                "jepa_rollout_mode='recurrent'."
            )
        if (
            isinstance(config.jepa_target_sample_count, bool)
            or not isinstance(
                config.jepa_target_sample_count,
                (int, np.integer),
            )
            or not 0 <= config.jepa_target_sample_count <= config.horizon
        ):
            raise ValueError(
                "jepa_target_sample_count must be an integer in "
                f"[0, {config.horizon}], found "
                f"{config.jepa_target_sample_count!r}."
            )
        if config.jepa_target_sampling_unit not in (
            "batch_shared",
            "example_balanced",
        ):
            raise ValueError(
                "jepa_target_sampling_unit must be 'batch_shared' or "
                f"'example_balanced', found "
                f"{config.jepa_target_sampling_unit!r}."
            )
        if (
            config.jepa_target_sampling_unit == "example_balanced"
            and config.jepa_target_sample_count != 1
        ):
            raise ValueError(
                "jepa_target_sampling_unit='example_balanced' requires "
                "jepa_target_sample_count=1."
            )
        if (
            config.jepa_target_sampling_unit == "example_balanced"
            and config.jepa_sampled_target_anchors
        ):
            raise ValueError(
                "jepa_target_sampling_unit='example_balanced' does not "
                "support sampled-target anchors."
            )
        if (
            isinstance(config.jepa_projector_active_layers, bool)
            or not isinstance(
                config.jepa_projector_active_layers,
                (int, np.integer),
            )
            or not 0
            <= config.jepa_projector_active_layers
            <= config.projector_layers
        ):
            raise ValueError(
                "jepa_projector_active_layers must be an integer in "
                f"[0, {config.projector_layers}], found "
                f"{config.jepa_projector_active_layers!r}."
            )
        if (
            isinstance(config.dfm_active_layers, bool)
            or not isinstance(
                config.dfm_active_layers,
                (int, np.integer),
            )
            or not 0 <= config.dfm_active_layers <= config.dfm_layers
        ):
            raise ValueError(
                "dfm_active_layers must be an integer in "
                f"[0, {config.dfm_layers}], found "
                f"{config.dfm_active_layers!r}."
            )
        projector_heads = config.projector_num_heads
        if config.z_dim % projector_heads != 0:
            raise ValueError(
                f"z_dim={config.z_dim} must be divisible by "
                f"projector_num_heads={projector_heads}."
            )
        condition_dim = (
            config.jepa_condition_dim
            if config.jepa_condition_dim > 0
            else config.z_dim
        )
        if (
            config.jepa_feedback_mode == "final_pass_adjoint"
            and condition_dim != config.z_dim
        ):
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires the "
                "effective jepa_condition_dim to equal z_dim: "
                f"{condition_dim} != {config.z_dim}."
            )
        jepa_mlp_dim = (
            config.jepa_mlp_dim if config.jepa_mlp_dim > 0 else config.z_dim * 4
        )
        projector_mlp_dim = (
            config.projector_mlp_dim
            if config.projector_mlp_dim > 0
            else config.z_dim * 4
        )

        self.state_projector = StateVectorProjector(
            self.encoder_dim,
            config.z_dim,
            num_layers=config.projector_layers,
            active_layers=config.jepa_projector_active_layers,
            num_heads=projector_heads,
            mlp_dim=projector_mlp_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            use_qk_gain=config.use_qk_gain,
            use_qk_norm=config.use_qk_norm,
            use_xsa=config.use_xsa,
            scan_layers=config.scan_layers,
            remat_blocks=config.remat_blocks,
        )
        self.dfm_state_projector = LinearAdapter(
            self.encoder_dim,
            config.token_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.jepa_action_embed = TrainableEmbedding(
            config.action_vocab_size + 1,
            condition_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.jepa_hidden_adapter = LinearAdapter(
            config.token_dim,
            condition_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.jepa_transition = ConditionedVectorTransition(
            config.z_dim,
            condition_dim,
            num_layers=config.jepa_layers,
            mlp_dim=jepa_mlp_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            delta_rms_clip=config.jepa_delta_rms_clip,
        )
        self.jepa_state_norm = TrainableRMSNorm(
            config.z_dim,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            eps=JEPA_STATE_RMS_EPSILON,
        )
        self.value_wdl_head = VectorValueWDLHead(
            config.z_dim,
            max(config.z_dim, jepa_mlp_dim // 2),
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )

        self.action_embed = TrainableEmbedding(
            config.action_vocab_size + 1,
            config.token_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.time_embed1 = TrainableParam(
            jax.random.normal(rngs.params(), (1, config.token_dim)) / np.sqrt(1)
        )
        self.time_embed2 = TrainableParam(
            jax.random.normal(
                rngs.params(),
                (config.token_dim, config.token_dim),
            )
            / np.sqrt(config.token_dim)
        )
        self.time_bias = TrainableParam(jnp.zeros((config.token_dim,)))
        self.pos_embed = TrainableParam(
            jax.random.normal(
                rngs.params(),
                (config.horizon, config.token_dim),
            )
            / np.sqrt(config.token_dim)
        )

        self.dfm_blocks = TrainableTransformerStack(
            num_layers=config.dfm_layers,
            width=config.token_dim,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            use_qk_gain=config.use_qk_gain,
            use_qk_norm=config.use_qk_norm,
            use_xsa=config.use_xsa,
            scan_layers=config.scan_layers,
            remat_blocks=config.remat_blocks,
        )
        self.dfm_out_norm = TrainableRMSNorm(
            config.token_dim,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.out_proj = TrainableParam(
            jax.random.normal(
                rngs.params(),
                (config.token_dim, config.action_vocab_size),
                dtype=param_dtype,
            )
            / np.sqrt(config.token_dim)
        )
        self.out_bias = TrainableParam(
            jnp.zeros((config.action_vocab_size,), dtype=param_dtype)
        )

    def encode_bt4_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        tokens = jnp.asarray(
            self.encoder.encode_tokens(planes),
            dtype=self.compute_dtype,
        )
        if self.config.bt4_freeze_backbone:
            tokens = jax.lax.stop_gradient(tokens)
        return tokens

    def encode_future_bt4_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        """Encode a future target with the configured BT4 gradient boundary."""

        trainable_tail_layers = int(
            self.config.bt4_future_target_trainable_tail_layers
        )
        if trainable_tail_layers == 0:
            return jax.lax.stop_gradient(self.encode_bt4_tokens(planes))

        layer_count = len(self.encoder.layers)
        if trainable_tail_layers > layer_count:
            raise ValueError(
                "bt4_future_target_trainable_tail_layers exceeds the loaded "
                f"encoder depth: {trainable_tail_layers} > {layer_count}"
            )
        alpha = (
            float(math.pow(2.0 * layer_count, -0.25))
            if layer_count > 0
            else 1.0
        )
        tokens, batch_size = self.encoder.embedding(planes, alpha)
        tokens = tokens.reshape((batch_size, 64, self.encoder_dim))
        gradient_boundary = layer_count - trainable_tail_layers
        for layer_index, layer in enumerate(self.encoder.layers):
            if layer_index == gradient_boundary:
                tokens = jax.lax.stop_gradient(tokens)
            tokens = layer(tokens, alpha)
        return jnp.asarray(tokens, dtype=self.compute_dtype)

    def encode_current_jepa(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        """Return projected current-state JEPA vectors."""

        return self.jepa_latents(self.encode_bt4_tokens(current_planes))

    def jepa_latents(self, bt4_tokens: jnp.ndarray) -> jnp.ndarray:
        """Project already encoded BT4 tokens onto the JEPA state manifold."""

        return self.normalize_jepa_state(self.state_projector(bt4_tokens))

    def encode_current_targets(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        return self.encode_current_jepa(current_planes)

    def encode_future_targets(self, future_planes: jnp.ndarray) -> jnp.ndarray:
        batch_size, horizon, channels, height, width = future_planes.shape
        flat_planes = future_planes.reshape(
            (batch_size * horizon, channels, height, width)
        )
        target_vectors = self.state_projector(
            self.encode_bt4_tokens(flat_planes)
        )
        target_vectors = target_vectors.reshape(
            (batch_size, horizon, self.z_dim)
        )
        return self.normalize_jepa_state(target_vectors)

    def encode_current_and_future_targets(
        self,
        current_planes: jnp.ndarray,
        future_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        _, z_all = self.encode_current_and_future_tokens_and_vectors(
            current_planes,
            future_planes,
        )
        return z_all[:, 0], z_all[:, 1:]

    def encode_current_and_future_tokens_and_vectors(
        self,
        current_planes: jnp.ndarray,
        future_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, horizon, channels, height, width = future_planes.shape
        if self.config.bt4_future_target_stop_gradient:
            current_tokens = self.encode_bt4_tokens(current_planes)
            flat_future_planes = future_planes.reshape(
                (batch_size * horizon, channels, height, width)
            )
            future_tokens = self.encode_future_bt4_tokens(
                flat_future_planes
            ).reshape(
                (batch_size, horizon, 64, self.encoder_dim)
            )
            tokens = jnp.concatenate(
                (current_tokens[:, None, :, :], future_tokens),
                axis=1,
            )
            vectors = self.state_projector(
                tokens.reshape(
                    (batch_size * (horizon + 1), 64, self.encoder_dim)
                )
            ).reshape(
                (batch_size, horizon + 1, self.z_dim)
            )
            return tokens, self.normalize_jepa_state(vectors)

        all_planes = jnp.concatenate(
            (current_planes[:, None, :, :, :], future_planes),
            axis=1,
        )
        chunk_size = int(self.config.bt4_encode_chunk_size)
        if chunk_size == 1:
            time_major_planes = jnp.swapaxes(all_planes, 0, 1)

            def encode_one(planes_t):
                tokens_t = self.encode_bt4_tokens(planes_t)
                vectors_t = self.normalize_jepa_state(
                    self.state_projector(tokens_t)
                )
                return tokens_t, vectors_t

            encode_one_fn = jax.checkpoint(encode_one, prevent_cse=False)

            def scan_body(_, planes_t):
                tokens_t, vectors_t = encode_one_fn(planes_t)
                return None, (tokens_t, vectors_t)

            _, (tokens_t, vectors_t) = jax.lax.scan(
                scan_body,
                None,
                time_major_planes,
                unroll=1,
            )
            return (
                jnp.swapaxes(tokens_t, 0, 1),
                jnp.swapaxes(vectors_t, 0, 1),
            )

        flat_planes = all_planes.reshape(
            (batch_size * (horizon + 1), channels, height, width)
        )
        encoder_tokens = self.encode_bt4_tokens(flat_planes)
        tokens = encoder_tokens.reshape(
            (batch_size, horizon + 1, 64, self.encoder_dim)
        )
        vectors = self.state_projector(
            tokens.reshape(
                (batch_size * (horizon + 1), 64, self.encoder_dim)
            )
        )
        vectors = vectors.reshape(
            (batch_size, horizon + 1, self.z_dim)
        )
        return tokens, self.normalize_jepa_state(vectors)

    def normalize_jepa_state(self, z: jnp.ndarray) -> jnp.ndarray:
        """Shared bounded RMSNorm for the JEPA state manifold."""

        if self.config.jepa_state_fixed_unit_rms:
            stats_z = jnp.asarray(z, dtype=jnp.float32)
            inv_rms = jax.lax.rsqrt(
                jnp.mean(
                    jnp.square(stats_z),
                    axis=-1,
                    keepdims=True,
                )
                + self.jepa_state_norm.eps
            )
            return jnp.asarray(
                stats_z * inv_rms,
                dtype=self.compute_dtype,
            )
        z = jnp.asarray(z, dtype=self.compute_dtype)
        if not self.config.jepa_state_rmsnorm:
            return z
        stats_z = jnp.asarray(z, dtype=jnp.float32)
        inv_rms = jax.lax.rsqrt(
            jnp.mean(jnp.square(stats_z), axis=-1, keepdims=True)
            + self.jepa_state_norm.eps
        )
        scale = jnp.asarray(
            self.jepa_state_norm.scale[...],
            dtype=jnp.float32,
        )
        if self.config.jepa_state_rms_scale_max > 0.0:
            max_scale = jnp.asarray(
                self.config.jepa_state_rms_scale_max,
                dtype=jnp.float32,
            )
            scale = jnp.clip(scale, 1.0 / max_scale, max_scale)
        out = stats_z * inv_rms * scale
        return jnp.asarray(out, dtype=self.compute_dtype)

    def dfm_latents(self, bt4_tokens: jnp.ndarray) -> jnp.ndarray:
        return self.dfm_state_projector(
            jnp.asarray(bt4_tokens, dtype=self.compute_dtype)
        )

    def get_time_embedding(self, t: jnp.ndarray) -> jnp.ndarray:
        t = t[:, None]
        t_emb = t @ self.time_embed1[...]
        t_emb = jax.nn.relu(t_emb)
        return t_emb @ self.time_embed2[...] + self.time_bias[...]

    def planner_from_latents(
        self,
        z_dfm: jnp.ndarray,
        action_tokens: jnp.ndarray,
        t: jnp.ndarray,
        *,
        return_hidden: bool = False,
    ):
        horizon = action_tokens.shape[1]
        action_emb = self.action_embed(action_tokens)
        action_emb = (
            action_emb
            + jnp.asarray(self.pos_embed[...], dtype=self.compute_dtype)[
                None, :horizon, :
            ]
        )
        action_emb = action_emb + self.get_time_embedding(t)[:, None, :]
        seq = jnp.concatenate([z_dfm, action_emb], axis=1)
        seq = _apply_transformer_stack_prefix(
            self.dfm_blocks,
            seq,
            active_layers=self.config.dfm_active_layers,
            compute_dtype=self.compute_dtype,
        )
        state_hidden = seq[:, :64, :]
        action_hidden = seq[:, 64:, :]
        logits = self.logits_from_action_hidden(action_hidden)
        if return_hidden:
            return logits, {
                "state_tokens": state_hidden,
                "action_tokens": action_hidden,
            }
        return logits

    def logits_from_action_hidden(
        self,
        action_hidden: jnp.ndarray,
    ) -> jnp.ndarray:
        action_out = self.dfm_out_norm(action_hidden)
        return (
            action_out
            @ jnp.asarray(self.out_proj[...], dtype=self.compute_dtype)
            + jnp.asarray(self.out_bias[...], dtype=self.compute_dtype)
        )

    def jepa_rollout_from_latents(
        self,
        z0_jepa: jnp.ndarray,
        actions: jnp.ndarray,
        action_hidden: jnp.ndarray,
        *,
        z0_normalized: bool = False,
    ) -> jnp.ndarray:
        action_hidden_seq = jnp.transpose(
            jnp.asarray(action_hidden, dtype=self.compute_dtype),
            (1, 0, 2),
        )
        actions_seq = jnp.transpose(actions, (1, 0))

        def loop_body(z, inputs):
            action_idx, hidden = inputs
            condition = (
                self.jepa_action_embed(action_idx)
                + self.jepa_hidden_adapter(hidden)
            )
            next_z = self.jepa_transition(z, condition)
            next_z = self.normalize_jepa_state(next_z)
            return next_z, next_z

        z0 = (
            jnp.asarray(z0_jepa, dtype=self.compute_dtype)
            if z0_normalized
            else self.normalize_jepa_state(z0_jepa)
        )
        _, pred_seq = jax.lax.scan(
            loop_body,
            z0,
            (actions_seq, action_hidden_seq),
        )
        return jnp.transpose(pred_seq, (1, 0, 2))

    def jepa_step_from_latents(
        self,
        z0_jepa: jnp.ndarray,
        action: jnp.ndarray,
        action_hidden: jnp.ndarray,
        *,
        z0_normalized: bool = False,
    ) -> jnp.ndarray:
        """Run one proposal-conditioned recurrent JEPA transition."""

        expected_hidden_shape = (
            action.shape[0],
            self.config.token_dim,
        )
        if action_hidden.shape != expected_hidden_shape:
            raise ValueError(
                "action_hidden shape must be "
                f"{expected_hidden_shape}, found {action_hidden.shape}"
            )
        condition = (
            self.jepa_action_embed(action)
            + self.jepa_hidden_adapter(
                jnp.asarray(action_hidden, dtype=self.compute_dtype)
            )
        )
        z0 = (
            jnp.asarray(z0_jepa, dtype=self.compute_dtype)
            if z0_normalized
            else self.normalize_jepa_state(z0_jepa)
        )
        return self.normalize_jepa_state(
            self.jepa_transition(z0, condition)
        )

    def dfm_latents_with_jepa_feedback(
        self,
        z_dfm: jnp.ndarray,
        z0_jepa: jnp.ndarray,
        z1_jepa: jnp.ndarray,
        feedback_gate: jnp.ndarray,
    ) -> JepaFeedbackResult:
        """Apply the frozen, capped adjoint JEPA residual to DFM state tokens."""

        if self.config.jepa_feedback_mode != "final_pass_adjoint":
            raise ValueError(
                "JEPA feedback latents require "
                "jepa_feedback_mode='final_pass_adjoint'."
            )
        expected_jepa_shape = (z_dfm.shape[0], self.z_dim)
        if z0_jepa.shape != expected_jepa_shape:
            raise ValueError(
                f"z0_jepa must have shape {expected_jepa_shape}, "
                f"found {z0_jepa.shape}"
            )
        if z1_jepa.shape != expected_jepa_shape:
            raise ValueError(
                f"z1_jepa must have shape {expected_jepa_shape}, "
                f"found {z1_jepa.shape}"
            )
        if feedback_gate.shape != (z_dfm.shape[0],):
            raise ValueError(
                "feedback_gate must have shape "
                f"{(z_dfm.shape[0],)}, found {feedback_gate.shape}"
            )

        adapter_w = jnp.asarray(
            self.jepa_hidden_adapter.w[...],
            dtype=jnp.float32,
        )
        if adapter_w.shape[1] != self.z_dim:
            raise ValueError(
                "The JEPA hidden adapter output must equal z_dim for "
                "adjoint feedback: "
                f"{adapter_w.shape[1]} != {self.z_dim}."
            )
        delta = (
            jnp.asarray(z1_jepa, dtype=jnp.float32)
            - jnp.asarray(z0_jepa, dtype=jnp.float32)
        )
        variance_correction = jnp.sqrt(
            jnp.asarray(
                adapter_w.shape[0] / adapter_w.shape[1],
                dtype=jnp.float32,
            )
        )
        raw_feedback = delta @ jnp.swapaxes(adapter_w, -1, -2)
        raw_feedback = raw_feedback * variance_correction

        z_dfm_f32 = jnp.asarray(z_dfm, dtype=jnp.float32)
        delta_rms = jnp.sqrt(
            jnp.mean(jnp.square(delta), axis=-1)
        )
        raw_feedback_rms = jnp.sqrt(
            jnp.mean(jnp.square(raw_feedback), axis=-1)
        )
        state_rms = jnp.sqrt(
            jnp.mean(jnp.square(z_dfm_f32), axis=(1, 2))
        )
        max_feedback_rms = (
            JEPA_FEEDBACK_MAX_STATE_RMS_RATIO * state_rms
        )
        cap_scale = jnp.minimum(
            1.0,
            max_feedback_rms
            / jnp.maximum(raw_feedback_rms, JEPA_STATE_RMS_EPSILON),
        )
        gate = jnp.clip(
            jnp.asarray(feedback_gate, dtype=jnp.float32),
            0.0,
            1.0,
        )
        applied_feedback = (
            raw_feedback * cap_scale[:, None] * gate[:, None]
        )
        applied_feedback_rms = jnp.sqrt(
            jnp.mean(jnp.square(applied_feedback), axis=-1)
        )
        latents = z_dfm_f32 + applied_feedback[:, None, :]
        return JepaFeedbackResult(
            latents=jnp.asarray(latents, dtype=self.compute_dtype),
            delta_rms=delta_rms,
            raw_feedback_rms=raw_feedback_rms,
            applied_feedback_rms=applied_feedback_rms,
            state_rms=state_rms,
            cap_fraction=jnp.mean(
                (cap_scale < 1.0).astype(jnp.float32)
            ),
        )

    def jepa_direct_sequence_from_latents(
        self,
        z0_jepa: jnp.ndarray,
        actions: jnp.ndarray,
        action_hidden: jnp.ndarray,
        *,
        z0_normalized: bool = False,
    ) -> jnp.ndarray:
        """Predict every horizon directly from one current-state latent."""

        expected_hidden_shape = (
            actions.shape[0],
            actions.shape[1],
            self.config.token_dim,
        )
        if action_hidden.shape != expected_hidden_shape:
            raise ValueError(
                "action_hidden shape must be "
                f"{expected_hidden_shape}, found {action_hidden.shape}"
            )
        condition = (
            self.jepa_action_embed(actions)
            + self.jepa_hidden_adapter(
                jnp.asarray(action_hidden, dtype=self.compute_dtype)
            )
        )
        z0 = (
            jnp.asarray(z0_jepa, dtype=self.compute_dtype)
            if z0_normalized
            else self.normalize_jepa_state(z0_jepa)
        )
        z0_by_horizon = jnp.broadcast_to(
            z0[:, None, :],
            (actions.shape[0], actions.shape[1], self.z_dim),
        )
        pred_z = self.jepa_transition(z0_by_horizon, condition)
        return self.normalize_jepa_state(pred_z)

    def jepa_predictions_from_latents(
        self,
        z0_jepa: jnp.ndarray,
        actions: jnp.ndarray,
        action_hidden: jnp.ndarray,
        *,
        z0_normalized: bool = False,
    ) -> jnp.ndarray:
        """Dispatch to the checkpoint-authoritative JEPA prediction graph."""

        if self.config.jepa_rollout_mode == "recurrent":
            return self.jepa_rollout_from_latents(
                z0_jepa,
                actions,
                action_hidden,
                z0_normalized=z0_normalized,
            )
        if self.config.jepa_rollout_mode == "direct_sequence":
            return self.jepa_direct_sequence_from_latents(
                z0_jepa,
                actions,
                action_hidden,
                z0_normalized=z0_normalized,
            )
        raise ValueError(
            f"Unsupported jepa_rollout_mode: "
            f"{self.config.jepa_rollout_mode!r}"
        )

    def jepa_rollout_from_latents_with_anchors(
        self,
        z0_jepa: jnp.ndarray,
        actions: jnp.ndarray,
        action_hidden: jnp.ndarray,
        anchor_latents: jnp.ndarray,
        anchor_mask: jnp.ndarray,
        *,
        z0_normalized: bool = False,
    ) -> jnp.ndarray:
        """Free predictions whose downstream carry may use true latents.

        Prediction ``h`` is always produced before applying anchor ``h``.
        The anchor can therefore affect only predictions after ``h``.
        """

        expected_latent_shape = (
            actions.shape[0],
            actions.shape[1],
            self.z_dim,
        )
        if anchor_latents.shape != expected_latent_shape:
            raise ValueError(
                "anchor_latents shape must be "
                f"{expected_latent_shape}, found {anchor_latents.shape}"
            )
        if anchor_mask.shape != actions.shape:
            raise ValueError(
                "anchor_mask shape must match actions: "
                f"{anchor_mask.shape} != {actions.shape}"
            )

        action_hidden_seq = jnp.transpose(
            jnp.asarray(action_hidden, dtype=self.compute_dtype),
            (1, 0, 2),
        )
        actions_seq = jnp.transpose(actions, (1, 0))
        anchor_latents_seq = jnp.transpose(
            jnp.asarray(anchor_latents, dtype=self.compute_dtype),
            (1, 0, 2),
        )
        anchor_mask_seq = jnp.transpose(
            jnp.asarray(anchor_mask, dtype=jnp.bool_),
            (1, 0),
        )

        def loop_body(z, inputs):
            action_idx, hidden, anchor_z, use_anchor = inputs
            condition = (
                self.jepa_action_embed(action_idx)
                + self.jepa_hidden_adapter(hidden)
            )
            pred_z = self.jepa_transition(z, condition)
            pred_z = self.normalize_jepa_state(pred_z)
            next_carry = jnp.where(use_anchor[:, None], anchor_z, pred_z)
            return next_carry, pred_z

        z0 = (
            jnp.asarray(z0_jepa, dtype=self.compute_dtype)
            if z0_normalized
            else self.normalize_jepa_state(z0_jepa)
        )
        _, pred_seq = jax.lax.scan(
            loop_body,
            z0,
            (
                actions_seq,
                action_hidden_seq,
                anchor_latents_seq,
                anchor_mask_seq,
            ),
        )
        return jnp.transpose(pred_seq, (1, 0, 2))

    def jepa_teacher_forced_from_latents(
        self,
        z_context: jnp.ndarray,
        actions: jnp.ndarray,
        action_hidden: jnp.ndarray,
    ) -> jnp.ndarray:
        """One-step JEPA predictions using true previous latents as inputs."""

        z_context_seq = jnp.transpose(
            jnp.asarray(z_context, dtype=self.compute_dtype),
            (1, 0, 2),
        )
        action_hidden_seq = jnp.transpose(
            jnp.asarray(action_hidden, dtype=self.compute_dtype),
            (1, 0, 2),
        )
        actions_seq = jnp.transpose(actions, (1, 0))

        def loop_body(_, inputs):
            z_in, action_idx, hidden = inputs
            condition = (
                self.jepa_action_embed(action_idx)
                + self.jepa_hidden_adapter(hidden)
            )
            next_z = self.jepa_transition(z_in, condition)
            next_z = self.normalize_jepa_state(next_z)
            return None, next_z

        _, pred_seq = jax.lax.scan(
            loop_body,
            None,
            (z_context_seq, actions_seq, action_hidden_seq),
        )
        return jnp.transpose(pred_seq, (1, 0, 2))

    def value_wdl_from_pred(
        self,
        pred_z: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, horizon, z_dim = pred_z.shape
        value, wdl = self.value_wdl_head(
            pred_z.reshape((batch_size * horizon, z_dim))
        )
        return (
            value.reshape((batch_size, horizon)),
            wdl.reshape((batch_size, horizon, 3)),
        )


class EmaBt4Encoder(nnx.Module):
    """Headless clone of the BT4 token-producing trunk."""

    def __init__(self, online_encoder: BT4Model):
        self.embedding = nnx.clone(online_encoder.embedding)
        self.layers = nnx.clone(online_encoder.layers)
        self.embedding_size = int(online_encoder.embedding_size)

    def encode_tokens(
        self,
        planes: jnp.ndarray,
        alpha: float | None = None,
    ) -> jnp.ndarray:
        if alpha is None:
            alpha = (
                float(math.pow(2.0 * len(self.layers), -0.25))
                if len(self.layers) > 0
                else 1.0
            )
        x, batch_size = self.embedding(planes, alpha)
        x = x.reshape((batch_size, 64, self.embedding_size))
        for layer in self.layers:
            x = layer(x, alpha)
        return x


class EmaTargetModel(nnx.Module):
    """Non-optimized EMA state for the JEPA target-producing parameters.

    The checkpointed EMA master contains a headless FP32 BT4 token encoder,
    FP32 state projector, and optional FP32 JEPA state RMSNorm path. A
    physically independent native-dtype encoder mirror is derived from the
    FP32 encoder master for GPU-exact forward computation.
    """

    def __init__(self, online_model: JointLatentSASAModel):
        self.encoder_master = EmaBt4Encoder(online_model.encoder)
        self.encoder_compute = EmaBt4Encoder(online_model.encoder)
        self.state_projector = nnx.clone(online_model.state_projector)
        self.jepa_state_norm = nnx.clone(online_model.jepa_state_norm)
        self.config = online_model.config
        self.compute_dtype = online_model.compute_dtype
        self.z_dim = online_model.z_dim

        master_state = nnx.state(self.encoder_master)
        fp32_master_state = jax.tree.map(
            lambda value: jnp.array(
                value,
                dtype=(
                    jnp.float32
                    if jnp.issubdtype(
                        jnp.asarray(value).dtype,
                        jnp.inexact,
                    )
                    else jnp.asarray(value).dtype
                ),
                copy=True,
            ),
            master_state,
        )
        nnx.update(self.encoder_master, fp32_master_state)

        for component in (
            self.state_projector,
            self.jepa_state_norm,
        ):
            component_state = nnx.state(component)
            independent_state = jax.tree.map(
                lambda value: jnp.array(
                    value,
                    dtype=(
                        jnp.float32
                        if jnp.issubdtype(
                            jnp.asarray(value).dtype,
                            jnp.floating,
                        )
                        else jnp.asarray(value).dtype
                    ),
                    copy=True,
                ),
                component_state,
            )
            nnx.update(component, independent_state)
        compute_state = nnx.state(self.encoder_compute)
        independent_compute_state = jax.tree.map(
            lambda value: jnp.array(
                value,
                dtype=jnp.asarray(value).dtype,
                copy=True,
            ),
            compute_state,
        )
        nnx.update(self.encoder_compute, independent_compute_state)

    def encode_bt4_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            self.encoder_compute.encode_tokens(planes),
            dtype=self.compute_dtype,
        )

    def normalize_jepa_state(self, z: jnp.ndarray) -> jnp.ndarray:
        if self.config.jepa_state_fixed_unit_rms:
            stats_z = jnp.asarray(z, dtype=jnp.float32)
            inv_rms = jax.lax.rsqrt(
                jnp.mean(
                    jnp.square(stats_z),
                    axis=-1,
                    keepdims=True,
                )
                + self.jepa_state_norm.eps
            )
            return jnp.asarray(
                stats_z * inv_rms,
                dtype=self.compute_dtype,
            )
        z = jnp.asarray(z, dtype=self.compute_dtype)
        if not self.config.jepa_state_rmsnorm:
            return z
        stats_z = jnp.asarray(z, dtype=jnp.float32)
        inv_rms = jax.lax.rsqrt(
            jnp.mean(jnp.square(stats_z), axis=-1, keepdims=True)
            + self.jepa_state_norm.eps
        )
        scale = jnp.asarray(
            self.jepa_state_norm.scale[...],
            dtype=jnp.float32,
        )
        if self.config.jepa_state_rms_scale_max > 0.0:
            max_scale = jnp.asarray(
                self.config.jepa_state_rms_scale_max,
                dtype=jnp.float32,
            )
            scale = jnp.clip(scale, 1.0 / max_scale, max_scale)
        return jnp.asarray(stats_z * inv_rms * scale, dtype=self.compute_dtype)

    def encode_future_targets(self, future_planes: jnp.ndarray) -> jnp.ndarray:
        batch_size, horizon, channels, height, width = future_planes.shape
        if int(self.config.bt4_encode_chunk_size) == 1:
            time_major_planes = jnp.swapaxes(future_planes, 0, 1)

            def encode_one(planes_t):
                vectors_t = self.state_projector(
                    self.encode_bt4_tokens(planes_t)
                )
                return self.normalize_jepa_state(vectors_t)

            encode_one_fn = jax.checkpoint(
                encode_one,
                prevent_cse=False,
            )

            def scan_body(_, planes_t):
                return None, encode_one_fn(planes_t)

            _, vectors_t = jax.lax.scan(
                scan_body,
                None,
                time_major_planes,
                unroll=1,
            )
            return jnp.swapaxes(vectors_t, 0, 1)

        flat_planes = future_planes.reshape(
            (batch_size * horizon, channels, height, width)
        )
        vectors = self.state_projector(
            self.encode_bt4_tokens(flat_planes)
        )
        vectors = vectors.reshape((batch_size, horizon, self.z_dim))
        return self.normalize_jepa_state(vectors)


def _ema_target_component_pairs(
    target: EmaTargetModel,
    online: JointLatentSASAModel,
) -> tuple[tuple[nnx.Module, nnx.Module], ...]:
    return (
        (target.encoder_master, online.encoder),
        (target.state_projector, online.state_projector),
        (target.jepa_state_norm, online.jepa_state_norm),
    )


def assert_ema_encoder_compute_structure(
    target: EmaTargetModel,
) -> None:
    """Preflight that master and derived encoder state graphs still match."""

    compute_structure = jax.tree.structure(
        nnx.state(target.encoder_compute)
    )
    master_structure = jax.tree.structure(
        nnx.state(target.encoder_master)
    )
    if compute_structure != master_structure:
        raise ValueError("EMA encoder master/compute structures differ")


def refresh_ema_encoder_compute(
    target: EmaTargetModel,
) -> None:
    """Refresh the native encoder mirror from the FP32 EMA master."""

    assert_ema_encoder_compute_structure(target)
    compute_state = nnx.state(target.encoder_compute)
    master_state = nnx.state(target.encoder_master)
    refreshed = jax.tree.map(
        lambda compute_value, master_value: jnp.array(
            master_value,
            dtype=jnp.asarray(compute_value).dtype,
            copy=True,
        ),
        compute_state,
        master_state,
    )
    nnx.update(target.encoder_compute, refreshed)


def validate_ema_encoder_compute(
    target: EmaTargetModel,
) -> None:
    """Fail if the native compute mirror differs from ``cast(master)``."""

    assert_ema_encoder_compute_structure(target)
    compute_state = nnx.to_pure_dict(
        nnx.state(target.encoder_compute)
    )
    master_state = nnx.to_pure_dict(
        nnx.state(target.encoder_master)
    )
    for index, (compute_value, master_value) in enumerate(
        zip(
            jax.tree.leaves(compute_state),
            jax.tree.leaves(master_state),
            strict=True,
        )
    ):
        compute_array = np.asarray(compute_value)
        expected = np.asarray(master_value).astype(
            compute_array.dtype,
            copy=False,
        )
        if not np.array_equal(compute_array, expected):
            raise ValueError(
                "EMA encoder compute mirror differs from cast(master) "
                f"at leaf {index}"
            )


def ema_checkpoint_state(
    target: EmaTargetModel,
) -> dict[str, Any]:
    """Return only authoritative EMA state; the compute mirror is derived."""

    return {
        "encoder_master": nnx.state(
            target.encoder_master,
            TrainableParam,
        ),
        "jepa_state_norm": nnx.state(
            target.jepa_state_norm,
            TrainableParam,
        ),
        "state_projector": nnx.state(
            target.state_projector,
            TrainableParam,
        ),
    }


def sync_ema_target_from_online(
    target: EmaTargetModel,
    online: JointLatentSASAModel,
) -> None:
    """Copy a restored online target path into the FP32 EMA shadow exactly."""

    for target_component, online_component in _ema_target_component_pairs(
        target,
        online,
    ):
        target_state = nnx.state(target_component, TrainableParam)
        online_state = nnx.state(online_component, TrainableParam)
        target_structure = jax.tree.structure(target_state)
        if target_structure != jax.tree.structure(online_state):
            raise ValueError("EMA target/online parameter structures differ")
        copied = jax.tree.map(
            lambda target_value, online_value: jnp.array(
                online_value,
                dtype=jnp.asarray(target_value).dtype,
                copy=True,
            ),
            target_state,
            online_state,
        )
        nnx.update(target_component, copied)
    refresh_ema_encoder_compute(target)


def update_ema_target_after_optimizer(
    target: EmaTargetModel,
    online: JointLatentSASAModel,
    decay: float,
) -> None:
    """Apply ``target = decay*target + (1-decay)*online`` in FP32."""

    decay_array = jnp.asarray(decay, dtype=jnp.float32)
    one_minus_decay = jnp.asarray(1.0, dtype=jnp.float32) - decay_array
    for target_component, online_component in _ema_target_component_pairs(
        target,
        online,
    ):
        target_state = nnx.state(target_component, TrainableParam)
        online_state = nnx.state(online_component, TrainableParam)
        updated = jax.tree.map(
            lambda target_value, online_value: (
                decay_array * jnp.asarray(target_value, dtype=jnp.float32)
                + one_minus_decay
                * jnp.asarray(online_value, dtype=jnp.float32)
            ),
            target_state,
            online_state,
        )
        nnx.update(target_component, updated)
    refresh_ema_encoder_compute(target)


def joint_stage1_loss_fn(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    sigreg_axis_name: str | None = None,
    compute_fp32_legality: bool = False,
    positive_target_override: jax.Array | None = None,
    sample_future_targets: bool = False,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    actions = batch["action_indices"][:, : model.config.horizon]
    batch_size, horizon = actions.shape
    loss_horizon = horizon if model.config.loss_horizon <= 0 else min(model.config.loss_horizon, horizon)
    loss_horizon_mask = (jnp.arange(horizon) < loss_horizon).astype(jnp.float32)

    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(valid), 1.0)
    future_valid = jnp.asarray(batch["future_valid"], dtype=jnp.float32)[:, :horizon]
    future_planes = jnp.asarray(batch["future_planes"], dtype=jnp.float32)[:, :horizon]

    # Preserve the baseline DFM time/mask RNG streams. Target-horizon sampling
    # is derived only from the former SIGReg branch, and the remaining SIGReg
    # key is then used exactly as the candidate's projection/sampling source.
    rng_t, rng_mask, rng_sigreg = jax.random.split(rng, 3)
    configured_target_count = int(model.config.jepa_target_sample_count)
    target_sampling_active = bool(
        sample_future_targets
        and 0 < configured_target_count < horizon
    )
    per_example_target_sampling = bool(
        target_sampling_active
        and model.config.jepa_target_sampling_unit
        == "example_balanced"
    )
    selected_horizons_by_example = None
    if target_sampling_active:
        rng_target_horizons, rng_sigreg = jax.random.split(rng_sigreg)
        if per_example_target_sampling:
            selected_horizons_by_example = (
                balanced_example_target_horizons(
                    rng_target_horizons,
                    batch_size=batch_size,
                    horizon=horizon,
                )
            )
            batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
            target_future_planes = future_planes[
                batch_indices,
                selected_horizons_by_example,
            ][:, None]
            future_valid_for_loss = future_valid[
                batch_indices,
                selected_horizons_by_example,
            ][:, None]
            selected_horizons = jnp.arange(
                configured_target_count,
                dtype=jnp.int32,
            )
        else:
            selected_horizons = jnp.sort(
                jax.random.permutation(rng_target_horizons, horizon)[
                    :configured_target_count
                ]
            )
            target_future_planes = jnp.take(
                future_planes,
                selected_horizons,
                axis=1,
            )
            future_valid_for_loss = jnp.take(
                future_valid,
                selected_horizons,
                axis=1,
            )
    else:
        selected_horizons = jnp.arange(horizon, dtype=jnp.int32)
        target_future_planes = future_planes
        future_valid_for_loss = future_valid

    with jax.named_scope("joint_encode_project_current_future"):
        all_bt4_tokens, z_all = model.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            target_future_planes,
        )
    current_bt4_tokens = all_bt4_tokens[:, 0]
    z_jepa = z_all[:, 0]
    target_z = z_all[:, 1:]
    with jax.named_scope("joint_dfm_state_projector"):
        z_dfm = model.dfm_latents(current_bt4_tokens)

    t = jax.random.uniform(rng_t, shape=(batch_size,))
    training_time_power_active = bool(
        sample_future_targets
        and model.config.dfm_training_time_power != 1.0
    )
    if training_time_power_active:
        t = transform_dfm_training_time(
            t,
            power=model.config.dfm_training_time_power,
        )
    if "deterministic_t" in batch:
        t = jnp.full_like(t, batch["deterministic_t"])
    noisy_actions, is_masked = mask_actions(actions, 1.0 - t, model.config.action_vocab_size, rng_mask)
    force_first_mask_active = bool(
        sample_future_targets
        and model.config.dfm_force_first_action_mask
    )
    if force_first_mask_active:
        noisy_actions, is_masked = force_first_action_mask(
            noisy_actions,
            is_masked,
            mask_token_id=model.config.action_vocab_size,
        )

    feedback_active = (
        model.config.jepa_feedback_mode == "final_pass_adjoint"
    )
    with jax.named_scope("joint_dfm_noisy_planner"):
        if feedback_active:
            preliminary_logits, preliminary_hidden = (
                model.planner_from_latents(
                    z_dfm,
                    noisy_actions,
                    t,
                    return_hidden=True,
                )
            )
        else:
            preliminary_logits = model.planner_from_latents(
                z_dfm,
                noisy_actions,
                t,
            )
            preliminary_hidden = None

    feedback_result: JepaFeedbackResult | None = None
    root_proposal: RootProposal | None = None
    if feedback_active:
        assert preliminary_hidden is not None
        with jax.named_scope("joint_dfm_root_proposal"):
            root_legal_mask = root_legal_mask_from_indices(
                batch["legal_idx"][:, 0, :],
                batch["legal_count"][:, 0],
                action_vocab_size=model.config.action_vocab_size,
            )
            root_legal_valid = (
                jnp.asarray(batch["legal_count"][:, 0], dtype=jnp.int32)
                > 0
            )
            if "legal_masks_valid" in batch:
                root_legal_valid = root_legal_valid & (
                    jnp.asarray(
                        batch["legal_masks_valid"][:, 0],
                        dtype=jnp.bool_,
                    )
                )
            root_legal_valid = root_legal_valid & (valid > 0.0)
            root_proposal = proposal_from_root_logits(
                preliminary_logits[:, 0, :],
                noisy_actions[:, 0],
                root_legal_mask,
                root_legal_valid,
                mask_token_id=model.config.action_vocab_size,
            )
        with jax.named_scope("joint_jepa_proposal_step"):
            proposal_z = model.jepa_step_from_latents(
                z_jepa,
                root_proposal.action,
                preliminary_hidden["action_tokens"][:, 0, :],
                z0_normalized=True,
            )
        with jax.named_scope("joint_jepa_adjoint_feedback"):
            feedback_result = model.dfm_latents_with_jepa_feedback(
                z_dfm,
                z_jepa,
                proposal_z,
                root_proposal.feedback_gate,
            )
        with jax.named_scope("joint_dfm_feedback_planner"):
            logits = model.planner_from_latents(
                feedback_result.latents,
                noisy_actions,
                t,
            )
    else:
        logits = preliminary_logits
    with jax.named_scope("joint_dfm_ce_loss"):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
    bt4_policy_distillation: Bt4PolicyDistillationResult | None = None
    if model.config.bt4_policy_distill_coeff != 0.0:
        with jax.named_scope("joint_bt4_frozen_policy_head_teacher"):
            teacher_canonical_logits = model.encoder.policy_head(
                jax.lax.stop_gradient(current_bt4_tokens)
            )
        legal_metadata_valid = (
            jnp.asarray(batch["legal_count"][:, 0], dtype=jnp.int32) > 0
        )
        if "legal_masks_valid" in batch:
            legal_metadata_valid = legal_metadata_valid & jnp.asarray(
                batch["legal_masks_valid"][:, 0],
                dtype=jnp.bool_,
            )
        with jax.named_scope("joint_bt4_root_policy_distillation"):
            bt4_policy_distillation = (
                bt4_policy_distillation_from_logits(
                    teacher_canonical_logits,
                    logits[:, 0, :],
                    batch["current_planes"],
                    actions[:, 0],
                    batch["legal_idx"][:, 0, :],
                    batch["legal_count"][:, 0],
                    valid > 0.0,
                    is_masked[:, 0],
                    legal_metadata_valid,
                )
            )
    root_legal_conditional_ce: (
        RootLegalConditionalCeResult | None
    ) = None
    if model.config.root_legal_conditional_ce_coeff != 0.0:
        root_legal_metadata_valid = (
            jnp.asarray(batch["legal_count"][:, 0], dtype=jnp.int32) > 0
        )
        if "legal_masks_valid" in batch:
            root_legal_metadata_valid = (
                root_legal_metadata_valid
                & jnp.asarray(
                    batch["legal_masks_valid"][:, 0],
                    dtype=jnp.bool_,
                )
            )
        with jax.named_scope("joint_dfm_root_legal_conditional_ce"):
            root_legal_conditional_ce = (
                root_legal_conditional_ce_from_logits(
                    logits[:, 0, :],
                    actions[:, 0],
                    batch["legal_idx"][:, 0, :],
                    batch["legal_count"][:, 0],
                    valid > 0.0,
                    is_masked[:, 0],
                    root_legal_metadata_valid,
                )
            )
    ce_by_horizon = -jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
    loss_mask = jnp.asarray(is_masked, dtype=jnp.float32) * loss_horizon_mask[None, :]
    weighted_loss_mask = loss_mask * valid[:, None]
    dfm_ce_num_by_horizon = jnp.sum(ce_by_horizon * weighted_loss_mask, axis=0)
    dfm_ce_den_by_horizon = jnp.sum(weighted_loss_mask, axis=0)
    dfm_ce_loss_by_horizon = dfm_ce_num_by_horizon / jnp.maximum(dfm_ce_den_by_horizon, 1.0)
    dfm_ce_horizon_valid = (dfm_ce_den_by_horizon > 0).astype(jnp.float32) * loss_horizon_mask
    dfm_ce_loss = jnp.sum(dfm_ce_loss_by_horizon * dfm_ce_horizon_valid) / jnp.maximum(
        jnp.sum(dfm_ce_horizon_valid),
        1.0,
    )
    preliminary_dfm_ce_loss = dfm_ce_loss
    preliminary_dfm_ce_loss_by_horizon = dfm_ce_loss_by_horizon
    if feedback_active:
        preliminary_log_probs = jax.nn.log_softmax(
            jax.lax.stop_gradient(
                jnp.asarray(preliminary_logits, dtype=jnp.float32)
            ),
            axis=-1,
        )
        preliminary_ce_by_horizon = -jnp.take_along_axis(
            preliminary_log_probs,
            actions[..., None],
            axis=-1,
        )[..., 0]
        preliminary_dfm_ce_num_by_horizon = jnp.sum(
            preliminary_ce_by_horizon * weighted_loss_mask,
            axis=0,
        )
        preliminary_dfm_ce_loss_by_horizon = (
            preliminary_dfm_ce_num_by_horizon
            / jnp.maximum(dfm_ce_den_by_horizon, 1.0)
        )
        preliminary_dfm_ce_loss = jnp.sum(
            preliminary_dfm_ce_loss_by_horizon
            * dfm_ce_horizon_valid
        ) / jnp.maximum(
            jnp.sum(dfm_ce_horizon_valid),
            1.0,
        )
    if model.config.dfm_first_action_loss_share == 0.0:
        # Preserve the compatibility path's exact reduction arithmetic.
        dfm_objective_weight_by_horizon = None
        dfm_objective_ce_loss = dfm_ce_loss
    else:
        dfm_objective_weight_by_horizon = dfm_objective_horizon_weights(
            dfm_ce_horizon_valid,
            first_action_loss_share=(
                model.config.dfm_first_action_loss_share
            ),
        )
        dfm_objective_ce_loss = jnp.sum(
            dfm_ce_loss_by_horizon * dfm_objective_weight_by_horizon
        )
    dfm_mask_fraction_by_horizon = dfm_ce_den_by_horizon / denom

    probs = jnp.exp(log_probs)
    first_legal_mass = legal_mass_from_indices(
        probs[:, :1],
        batch["legal_idx"][:, :1],
        batch["legal_count"][:, :1],
    )[:, 0]
    first_legal_valid = jnp.ones_like(first_legal_mass)
    if "legal_masks_valid" in batch:
        first_legal_valid = jnp.asarray(batch["legal_masks_valid"], dtype=jnp.float32)[:, 0]
    first_legal_gate = jnp.asarray(is_masked, dtype=jnp.float32)[:, 0]
    if not model.config.legality_on_masked_only:
        first_legal_gate = jnp.ones_like(first_legal_gate)
    first_slot_valid = valid * first_legal_valid * loss_horizon_mask[0] * first_legal_gate
    first_legality_loss = (
        jnp.sum((1.0 - first_legal_mass) * first_slot_valid)
        / jnp.maximum(jnp.sum(first_slot_valid), 1.0)
    )
    if compute_fp32_legality:
        first_probs_fp32 = jax.nn.softmax(
            jnp.asarray(logits[:, :1], dtype=jnp.float32),
            axis=-1,
        )
        first_legal_mass_fp32_by_sample = jnp.clip(
            legal_mass_from_indices(
                first_probs_fp32,
                batch["legal_idx"][:, :1],
                batch["legal_count"][:, :1],
            )[:, 0],
            0.0,
            1.0,
        )
        first_legality_loss_fp32 = (
            jnp.sum((1.0 - first_legal_mass_fp32_by_sample) * first_slot_valid)
            / jnp.maximum(jnp.sum(first_slot_valid), 1.0)
        )
    else:
        first_legality_loss_fp32 = first_legality_loss
    horizon_legality_loss = jnp.asarray(0.0, dtype=jnp.float32)
    legality_loss = first_legality_loss
    weighted_legality_loss = model.config.first_legality_coeff * first_legality_loss

    clean_t = jnp.ones((batch_size,), dtype=jnp.float32)
    with jax.named_scope("joint_dfm_clean_planner_hidden"):
        _, clean_hidden = model.planner_from_latents(z_dfm, actions, clean_t, return_hidden=True)
    with jax.named_scope("joint_jepa_recurrent_rollout"):
        teacher_forcing = jnp.asarray(batch.get("jepa_teacher_forcing", 0.0), dtype=jnp.float32)

        def teacher_forced_rollout(_):
            return model.jepa_teacher_forced_from_latents(z_all[:, :horizon], actions, clean_hidden["action_tokens"])

        def free_rollout(_):
            return model.jepa_predictions_from_latents(
                z_jepa,
                actions,
                clean_hidden["action_tokens"],
                z0_normalized=True,
            )

        if (
            target_sampling_active
            and model.config.jepa_sampled_target_anchors
        ):
            dense_anchor_latents = jnp.zeros(
                (batch_size, horizon, model.config.z_dim),
                dtype=target_z.dtype,
            ).at[:, selected_horizons, :].set(target_z)
            selected_anchor_valid = (
                valid[:, None] * future_valid_for_loss
            ) > 0.0
            dense_anchor_mask = jnp.zeros(
                (batch_size, horizon),
                dtype=jnp.bool_,
            ).at[:, selected_horizons].set(selected_anchor_valid)
            pred_z = model.jepa_rollout_from_latents_with_anchors(
                z_jepa,
                actions,
                clean_hidden["action_tokens"],
                dense_anchor_latents,
                dense_anchor_mask,
                z0_normalized=True,
            )
        elif target_sampling_active:
            # Full teacher forcing requires all future latent states. Sampled
            # targets otherwise retain the exact free-rollout control path.
            pred_z = free_rollout(None)
        else:
            pred_z = jax.lax.cond(
                teacher_forcing > 0.5,
                teacher_forced_rollout,
                free_rollout,
                operand=None,
            )
    if per_example_target_sampling:
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        pred_for_loss = pred_z[
            batch_indices,
            selected_horizons_by_example,
        ][:, None]
    else:
        pred_for_loss = (
            jnp.take(pred_z, selected_horizons, axis=1)
            if target_sampling_active
            else pred_z
        )
    if positive_target_override is None:
        target_vectors = _jepa_positive_target_vectors(
            z_jepa,
            target_z,
            pred_for_loss,
            target_mode=model.config.jepa_target_mode,
            stop_gradient=model.config.jepa_target_stop_gradient,
        )
    else:
        if model.config.jepa_target_mode != "projected_bt4":
            raise ValueError(
                "EMA positive targets require jepa_target_mode='projected_bt4'."
            )
        if positive_target_override.shape != pred_for_loss.shape:
            raise ValueError(
                "EMA positive target shape differs from prediction shape: "
                f"{positive_target_override.shape} != {pred_for_loss.shape}"
            )
        target_vectors = jax.lax.stop_gradient(
            jnp.asarray(
                positive_target_override,
                dtype=pred_for_loss.dtype,
            )
        )
    with jax.named_scope("joint_jepa_raw_mse_loss"):
        sample_raw_mse = jnp.mean(
            (jnp.asarray(pred_for_loss, dtype=jnp.float32) - jnp.asarray(target_vectors, dtype=jnp.float32)) ** 2,
            axis=-1,
        )
    with jax.named_scope("joint_jepa_rms_norm_loss"):
        pred_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(pred_for_loss, dtype=jnp.float32)), axis=-1) + 1e-6)
        target_rms = jax.lax.stop_gradient(
            jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(target_vectors, dtype=jnp.float32)), axis=-1) + 1e-6)
        )
        sample_norm_loss = jnp.abs(jnp.log(pred_rms) - jnp.log(target_rms))
        if model.config.jepa_norm_loss_coeff == 1.0:
            sample_jepa = sample_raw_mse + sample_norm_loss
        elif model.config.jepa_norm_loss_coeff == 0.0:
            sample_jepa = sample_raw_mse
        else:
            sample_jepa = (
                sample_raw_mse
                + model.config.jepa_norm_loss_coeff * sample_norm_loss
            )
    if per_example_target_sampling:
        horizon_weights = model.config.jepa_gamma ** (
            selected_horizons_by_example.astype(jnp.float32)
        )
        jepa_mask = (
            future_valid_for_loss
            * valid[:, None]
            * horizon_weights[:, None]
        )
    else:
        horizon_weights = model.config.jepa_gamma ** selected_horizons.astype(jnp.float32)
        jepa_mask = future_valid_for_loss * valid[:, None] * horizon_weights[None, :]
    jepa_positive_loss = _weighted_horizon_mean(sample_jepa, jepa_mask)
    jepa_raw_mse = _weighted_horizon_mean(sample_raw_mse, jepa_mask)
    jepa_norm_loss = _weighted_horizon_mean(sample_norm_loss, jepa_mask)
    if positive_target_override is not None:
        positive_target_rms = jnp.sqrt(
            _weighted_horizon_mean(
                jnp.mean(
                    jnp.square(
                        jnp.asarray(target_vectors, dtype=jnp.float32)
                    ),
                    axis=-1,
                ),
                jepa_mask,
            )
        )
        positive_pred_rms = jnp.sqrt(
            _weighted_horizon_mean(
                jnp.mean(
                    jnp.square(
                        jnp.asarray(pred_for_loss, dtype=jnp.float32)
                    ),
                    axis=-1,
                ),
                jepa_mask,
            )
        )
        pred_target_cosine_by_sample = jnp.sum(
            jnp.asarray(pred_for_loss, dtype=jnp.float32)
            * jnp.asarray(target_vectors, dtype=jnp.float32),
            axis=-1,
        ) / jnp.maximum(
            jnp.linalg.norm(
                jnp.asarray(pred_for_loss, dtype=jnp.float32),
                axis=-1,
            )
            * jnp.linalg.norm(
                jnp.asarray(target_vectors, dtype=jnp.float32),
                axis=-1,
            ),
            1e-12,
        )
        positive_pred_target_cosine = _weighted_horizon_mean(
            pred_target_cosine_by_sample,
            jepa_mask,
        )
        online_target_rms = jnp.sqrt(
            _weighted_horizon_mean(
                jnp.mean(
                    jnp.square(
                        jnp.asarray(target_z, dtype=jnp.float32)
                    ),
                    axis=-1,
                ),
                jepa_mask,
            )
        )
        online_positive_target_mse = _weighted_horizon_mean(
            jnp.mean(
                jnp.square(
                    jnp.asarray(target_z, dtype=jnp.float32)
                    - jnp.asarray(target_vectors, dtype=jnp.float32)
                ),
                axis=-1,
            ),
            jepa_mask,
        )
    sigreg_z_all = z_all
    sigreg_pred_z = pred_z
    sigreg_valid = valid
    sigreg_target_future_valid = future_valid_for_loss
    sigreg_pred_future_valid = future_valid
    sigreg_projection_rng = rng_sigreg
    sigreg_example_count = model.config.jepa_sigreg_example_count
    if sigreg_example_count > batch_size:
        raise ValueError(
            "jepa_sigreg_example_count cannot exceed physical batch size: "
            f"{sigreg_example_count} > {batch_size}"
        )
    if 0 < sigreg_example_count < batch_size:
        sigreg_selection_rng, sigreg_projection_rng = jax.random.split(
            rng_sigreg
        )
        sigreg_indices = jax.random.permutation(
            sigreg_selection_rng,
            batch_size,
        )[:sigreg_example_count]
        sigreg_z_all = jnp.take(z_all, sigreg_indices, axis=0)
        sigreg_pred_z = jnp.take(pred_z, sigreg_indices, axis=0)
        sigreg_valid = jnp.take(valid, sigreg_indices, axis=0)
        sigreg_target_future_valid = jnp.take(
            future_valid_for_loss,
            sigreg_indices,
            axis=0,
        )
        sigreg_pred_future_valid = jnp.take(
            future_valid,
            sigreg_indices,
            axis=0,
        )
    effective_sigreg_example_count = sigreg_z_all.shape[0]

    jepa_sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
    jepa_sigreg_u_stat_discrepancy = jnp.asarray(
        0.0,
        dtype=jnp.float32,
    )
    if per_example_target_sampling:
        target_future_importance = jnp.asarray(
            horizon / configured_target_count,
            dtype=jnp.float32,
        )
    else:
        target_future_importance = jnp.asarray(
            horizon / selected_horizons.shape[0],
            dtype=jnp.float32,
        )
    if target_sampling_active:
        target_sigreg_future_weight = (
            sigreg_valid[:, None]
            * sigreg_target_future_valid
            * target_future_importance
        )
    else:
        # Preserve the legacy full-horizon graph exactly, including operation
        # ordering in its gradients.
        target_sigreg_future_weight = (
            sigreg_valid[:, None] * sigreg_target_future_valid
        )
    valid_all = jnp.concatenate(
        [sigreg_valid[:, None], target_sigreg_future_weight],
        axis=1,
    )
    sigreg_weight = valid_all.reshape((-1,))
    sigreg_valid_count = jnp.sum(sigreg_weight)
    if model.config.jepa_sigreg_coeff != 0.0:
        with jax.named_scope("joint_jepa_sigreg"):
            sigreg_tokens = jnp.asarray(
                sigreg_z_all,
                dtype=jnp.float32,
            ).reshape((-1, sigreg_z_all.shape[-1]))
            if model.config.jepa_sigreg_kind == "moments":
                jepa_sigreg_loss = _sigreg_moments_loss(sigreg_tokens, sample_weight=sigreg_weight)
            elif model.config.jepa_sigreg_kind == "quantile":
                jepa_sigreg_loss = _quantile_sigreg_loss(
                    sigreg_tokens,
                    d_proj=model.config.jepa_sigreg_proj_dim,
                    rng=sigreg_projection_rng,
                )
            elif model.config.jepa_sigreg_kind == "le_jepa":
                if model.config.jepa_sigreg_estimator == "u_stat":
                    sigreg_statistics = _le_jepa_sigreg_statistics(
                        sigreg_tokens,
                        proj_dim=model.config.jepa_sigreg_proj_dim,
                        rng=sigreg_projection_rng,
                        sample_weight=sigreg_weight,
                        axis_name=sigreg_axis_name,
                        compute_u_stat=True,
                    )
                    jepa_sigreg_loss = sigreg_statistics.official
                    jepa_sigreg_u_stat_discrepancy = (
                        sigreg_statistics.u_stat_discrepancy
                    )
                else:
                    jepa_sigreg_loss = _official_le_jepa_sigreg_loss(
                        sigreg_tokens,
                        proj_dim=model.config.jepa_sigreg_proj_dim,
                        rng=sigreg_projection_rng,
                        sample_weight=sigreg_weight,
                        axis_name=sigreg_axis_name,
                    )
            else:
                raise ValueError(f"Unsupported jepa_sigreg_kind: {model.config.jepa_sigreg_kind!r}")
    jepa_pred_sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
    jepa_pred_sigreg_u_stat_discrepancy = jnp.asarray(
        0.0,
        dtype=jnp.float32,
    )
    pred_sigreg_weight = (
        sigreg_pred_future_valid * sigreg_valid[:, None]
    ).reshape((-1,))
    pred_sigreg_valid_count = jnp.sum(pred_sigreg_weight)
    if model.config.jepa_pred_sigreg_coeff != 0.0:
        with jax.named_scope("joint_jepa_pred_sigreg"):
            pred_sigreg_tokens = jnp.asarray(
                sigreg_pred_z,
                dtype=jnp.float32,
            ).reshape((-1, sigreg_pred_z.shape[-1]))
            if model.config.jepa_sigreg_kind == "moments":
                jepa_pred_sigreg_loss = _sigreg_moments_loss(pred_sigreg_tokens, sample_weight=pred_sigreg_weight)
            elif model.config.jepa_sigreg_kind == "quantile":
                jepa_pred_sigreg_loss = _quantile_sigreg_loss(
                    pred_sigreg_tokens,
                    d_proj=model.config.jepa_sigreg_proj_dim,
                    rng=sigreg_projection_rng,
                )
            elif model.config.jepa_sigreg_kind == "le_jepa":
                if model.config.jepa_sigreg_estimator == "u_stat":
                    pred_sigreg_statistics = _le_jepa_sigreg_statistics(
                        pred_sigreg_tokens,
                        proj_dim=model.config.jepa_sigreg_proj_dim,
                        rng=sigreg_projection_rng,
                        sample_weight=pred_sigreg_weight,
                        axis_name=sigreg_axis_name,
                        compute_u_stat=True,
                    )
                    jepa_pred_sigreg_loss = (
                        pred_sigreg_statistics.official
                    )
                    jepa_pred_sigreg_u_stat_discrepancy = (
                        pred_sigreg_statistics.u_stat_discrepancy
                    )
                else:
                    jepa_pred_sigreg_loss = (
                        _official_le_jepa_sigreg_loss(
                            pred_sigreg_tokens,
                            proj_dim=model.config.jepa_sigreg_proj_dim,
                            rng=sigreg_projection_rng,
                            sample_weight=pred_sigreg_weight,
                            axis_name=sigreg_axis_name,
                        )
                    )
            else:
                raise ValueError(f"Unsupported jepa_sigreg_kind: {model.config.jepa_sigreg_kind!r}")

    target_variance_hinge: TargetVarianceHingeResult | None = None
    if model.config.jepa_target_variance_hinge_coeff != 0.0:
        with jax.named_scope("joint_jepa_target_variance_hinge"):
            target_variance_hinge = per_horizon_target_variance_hinge(
                target_z,
                valid[:, None] * future_valid,
                gamma=model.config.jepa_target_variance_hinge_gamma,
            )

    value_loss = jnp.asarray(0.0, dtype=jnp.float32)
    wdl_loss = jnp.asarray(0.0, dtype=jnp.float32)
    value_pred_mean = jnp.asarray(0.0, dtype=jnp.float32)
    value_target_mean = jnp.asarray(0.0, dtype=jnp.float32)
    wdl_diagnostics: dict[str, jnp.ndarray] | None = None
    if model.config.value_coeff != 0.0 or model.config.wdl_coeff != 0.0:
        with jax.named_scope("joint_value_wdl_heads"):
            value_pred, wdl_logits = model.value_wdl_from_pred(pred_z)
        if model.config.value_coeff != 0.0:
            value_targets = jnp.asarray(
                batch.get("value_targets", jnp.zeros_like(value_pred)),
                dtype=jnp.float32,
            )[:, :horizon]
            value_loss_by_horizon = jnp.square(
                value_pred - value_targets
            )
            value_loss = _weighted_horizon_mean(
                value_loss_by_horizon,
                future_valid * valid[:, None],
            )
            value_pred_mean = jnp.mean(
                jnp.asarray(value_pred, dtype=jnp.float32)
            )
            value_target_mean = jnp.mean(
                jnp.asarray(value_targets, dtype=jnp.float32)
            )

        if model.config.wdl_coeff != 0.0:
            if "wdl_targets" not in batch:
                raise ValueError(
                    "Nonzero wdl_coeff requires per-horizon wdl_targets."
                )
            raw_wdl_targets = jnp.asarray(
                batch["wdl_targets"],
                dtype=jnp.float32,
            )[:, :horizon]
            if raw_wdl_targets.shape != wdl_logits.shape:
                raise ValueError(
                    "wdl_targets must match predicted WDL logits: "
                    f"{raw_wdl_targets.shape} != {wdl_logits.shape}"
                )
            wdl_sum = jnp.sum(
                raw_wdl_targets,
                axis=-1,
                keepdims=True,
            )
            wdl_valid = (wdl_sum[..., 0] > 0).astype(jnp.float32)
            wdl_targets = raw_wdl_targets / jnp.maximum(wdl_sum, 1e-12)
            wdl_mask = future_valid * valid[:, None] * wdl_valid
            wdl_valid_count = jnp.sum(wdl_mask)
            wdl_den_by_horizon = jnp.sum(wdl_mask, axis=0)
            wdl_log_probs = jax.nn.log_softmax(wdl_logits, axis=-1)
            wdl_probs = jnp.exp(wdl_log_probs)
            wdl_loss_by_sample = -jnp.sum(
                wdl_targets * wdl_log_probs,
                axis=-1,
            )
            wdl_loss = _weighted_horizon_mean(
                wdl_loss_by_sample,
                wdl_mask,
            )
            wdl_loss_by_horizon = (
                jnp.sum(wdl_loss_by_sample * wdl_mask, axis=0)
                / jnp.maximum(wdl_den_by_horizon, 1.0)
            )
            wdl_correct = (
                jnp.argmax(wdl_logits, axis=-1)
                == jnp.argmax(wdl_targets, axis=-1)
            ).astype(jnp.float32)
            wdl_accuracy = _weighted_horizon_mean(
                wdl_correct,
                wdl_mask,
            )
            wdl_accuracy_by_horizon = (
                jnp.sum(wdl_correct * wdl_mask, axis=0)
                / jnp.maximum(wdl_den_by_horizon, 1.0)
            )
            wdl_expected_value = wdl_probs[..., 0] - wdl_probs[..., 2]
            wdl_target_value = wdl_targets[..., 0] - wdl_targets[..., 2]
            wdl_expected_value_square_error = jnp.square(
                wdl_expected_value - wdl_target_value
            )
            wdl_expected_value_mse = _weighted_horizon_mean(
                wdl_expected_value_square_error,
                wdl_mask,
            )
            wdl_expected_value_mse_by_horizon = (
                jnp.sum(
                    wdl_expected_value_square_error * wdl_mask,
                    axis=0,
                )
                / jnp.maximum(wdl_den_by_horizon, 1.0)
            )
            wdl_target_class_fraction = jnp.sum(
                wdl_targets * wdl_mask[..., None],
                axis=(0, 1),
            ) / jnp.maximum(wdl_valid_count, 1.0)
            wdl_label_is_one_hot = (
                jnp.isclose(wdl_sum[..., 0], 1.0, rtol=0.0, atol=1e-6)
                & jnp.isclose(
                    jnp.sum(jnp.square(raw_wdl_targets), axis=-1),
                    1.0,
                    rtol=0.0,
                    atol=1e-6,
                )
                & jnp.all(raw_wdl_targets >= 0.0, axis=-1)
            ).astype(jnp.float32)
            wdl_label_one_hot_fraction = _weighted_horizon_mean(
                wdl_label_is_one_hot,
                future_valid * valid[:, None],
            )
            wdl_diagnostics = {
                "wdl_weighted_loss": (
                    jnp.asarray(model.config.wdl_coeff, dtype=jnp.float32)
                    * wdl_loss
                ),
                "wdl_valid_count": wdl_valid_count,
                "wdl_label_one_hot_fraction": (
                    wdl_label_one_hot_fraction
                ),
                "wdl_accuracy": wdl_accuracy,
                "wdl_expected_value_mse": wdl_expected_value_mse,
                "wdl_target_win_fraction": wdl_target_class_fraction[0],
                "wdl_target_draw_fraction": wdl_target_class_fraction[1],
                "wdl_target_loss_fraction": wdl_target_class_fraction[2],
                "wdl_loss_by_horizon": wdl_loss_by_horizon,
                "wdl_accuracy_by_horizon": wdl_accuracy_by_horizon,
                "wdl_expected_value_mse_by_horizon": (
                    wdl_expected_value_mse_by_horizon
                ),
            }

    horizon_valid = future_valid_for_loss * valid[:, None]
    if per_example_target_sampling:
        assigned_horizons = selected_horizons_by_example
        horizon_num = jnp.zeros(
            (horizon,), dtype=jnp.float32
        ).at[assigned_horizons].add(
            sample_jepa[:, 0] * horizon_valid[:, 0]
        )
        horizon_raw_mse_num = jnp.zeros(
            (horizon,), dtype=jnp.float32
        ).at[assigned_horizons].add(
            sample_raw_mse[:, 0] * horizon_valid[:, 0]
        )
        horizon_norm_num = jnp.zeros(
            (horizon,), dtype=jnp.float32
        ).at[assigned_horizons].add(
            sample_norm_loss[:, 0] * horizon_valid[:, 0]
        )
        horizon_denom = jnp.zeros(
            (horizon,), dtype=jnp.float32
        ).at[assigned_horizons].add(horizon_valid[:, 0])
        target_assignment_count_by_horizon = jnp.zeros(
            (horizon,), dtype=jnp.float32
        ).at[assigned_horizons].add(
            jnp.ones((batch_size,), dtype=jnp.float32)
        )
        target_horizon_mask = (
            target_assignment_count_by_horizon > 0.0
        ).astype(jnp.float32)
    else:
        horizon_num = jnp.zeros((horizon,), dtype=jnp.float32).at[selected_horizons].add(
            jnp.sum(sample_jepa * horizon_valid, axis=0)
        )
        horizon_raw_mse_num = jnp.zeros((horizon,), dtype=jnp.float32).at[selected_horizons].add(
            jnp.sum(sample_raw_mse * horizon_valid, axis=0)
        )
        horizon_norm_num = jnp.zeros((horizon,), dtype=jnp.float32).at[selected_horizons].add(
            jnp.sum(sample_norm_loss * horizon_valid, axis=0)
        )
        horizon_denom = jnp.zeros((horizon,), dtype=jnp.float32).at[selected_horizons].add(jnp.sum(horizon_valid, axis=0))
        target_horizon_mask = jnp.zeros((horizon,), dtype=jnp.float32).at[selected_horizons].set(1.0)
    jepa_loss_by_horizon = horizon_num / jnp.maximum(horizon_denom, 1.0)
    jepa_raw_mse_by_horizon = horizon_raw_mse_num / jnp.maximum(horizon_denom, 1.0)
    jepa_norm_loss_by_horizon = horizon_norm_num / jnp.maximum(horizon_denom, 1.0)
    downstream_anchor_horizon_mask = target_horizon_mask * (
        jnp.arange(horizon, dtype=jnp.int32) < horizon - 1
    ).astype(jnp.float32)

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    zero_by_horizon = jnp.zeros((horizon,), dtype=jnp.float32)
    jepa_cosine_loss = zero
    jepa_normalized_mse = zero
    mean_token_cosine = zero
    mean_token_cosine_by_horizon = zero_by_horizon
    pred_token_norm = zero
    target_token_norm = zero
    identity_jepa_loss = zero
    identity_jepa_cosine_loss = zero
    identity_mean_token_cosine = zero
    shuffled_jepa_loss = jnp.asarray(0.0, dtype=jnp.float32)
    shuffled_jepa_cosine_loss = jnp.asarray(0.0, dtype=jnp.float32)
    shuffled_mean_token_cosine = jnp.asarray(0.0, dtype=jnp.float32)
    action_contrast_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if model.config.jepa_state_fixed_unit_rms:
        jepa_state_rms_scale_raw = jnp.ones(
            (model.config.z_dim,),
            dtype=jnp.float32,
        )
        jepa_state_rms_scale = jepa_state_rms_scale_raw
    else:
        jepa_state_rms_scale_raw = jnp.asarray(
            model.jepa_state_norm.scale[...],
            dtype=jnp.float32,
        )
        jepa_state_rms_scale = jepa_state_rms_scale_raw
        if model.config.jepa_state_rms_scale_max > 0.0:
            jepa_state_rms_scale_cap = jnp.asarray(
                model.config.jepa_state_rms_scale_max,
                dtype=jnp.float32,
            )
            jepa_state_rms_scale = jnp.clip(
                jepa_state_rms_scale_raw,
                1.0 / jepa_state_rms_scale_cap,
                jepa_state_rms_scale_cap,
            )

    unclipped_loss = (
        model.config.dfm_ce_coeff * dfm_objective_ce_loss
        + weighted_legality_loss
        + model.config.jepa_positive_coeff * jepa_positive_loss
        + model.config.jepa_sigreg_coeff * jepa_sigreg_loss
        + model.config.jepa_pred_sigreg_coeff * jepa_pred_sigreg_loss
        + model.config.value_coeff * value_loss
        + model.config.wdl_coeff * wdl_loss
        + model.config.jepa_action_contrast_coeff * action_contrast_loss
    )
    if bt4_policy_distillation is not None:
        unclipped_loss = (
            unclipped_loss
            + model.config.bt4_policy_distill_coeff
            * bt4_policy_distillation.loss
        )
    if root_legal_conditional_ce is not None:
        unclipped_loss = (
            unclipped_loss
            + model.config.root_legal_conditional_ce_coeff
            * root_legal_conditional_ce.loss
        )
    if target_variance_hinge is not None:
        unclipped_loss = (
            unclipped_loss
            + model.config.jepa_target_variance_hinge_coeff
            * target_variance_hinge.loss
        )
    loss, loss_clip_scale = _clip_loss_preserve_gradient(unclipped_loss, model.config.loss_clip_value)

    preds = jnp.argmax(logits, axis=-1)
    accuracy = (
        jnp.sum((preds == actions) * weighted_loss_mask)
        / jnp.maximum(jnp.sum(weighted_loss_mask), 1.0)
    )
    aux = {
        "loss": loss,
        "unclipped_loss": unclipped_loss,
        "loss_clip_value": jnp.asarray(model.config.loss_clip_value, dtype=jnp.float32),
        "loss_clip_scale": loss_clip_scale,
        "dfm_ce_loss": dfm_ce_loss,
        "dfm_ce_loss_by_horizon": dfm_ce_loss_by_horizon,
        "dfm_mask_fraction_by_horizon": dfm_mask_fraction_by_horizon,
        "legality_loss": legality_loss,
        "first_legality_loss": first_legality_loss,
        "first_legality_loss_fp32": first_legality_loss_fp32,
        "first_legal_mass_fp32": 1.0 - first_legality_loss_fp32,
        "horizon_legality_loss": horizon_legality_loss,
        "weighted_legality_loss": weighted_legality_loss,
        "first_legal_mass": 1.0 - first_legality_loss,
        "horizon_legal_mass": jnp.asarray(0.0, dtype=jnp.float32),
        "horizon_legality_evaluated": jnp.asarray(0.0, dtype=jnp.float32),
        "jepa_positive_loss": jepa_positive_loss,
        "jepa_cosine_loss": jepa_cosine_loss,
        "jepa_raw_mse": jepa_raw_mse,
        "jepa_norm_loss": jepa_norm_loss,
        "jepa_normalized_mse": jepa_normalized_mse,
        "jepa_sigreg_loss": jepa_sigreg_loss,
        "jepa_pred_sigreg_loss": jepa_pred_sigreg_loss,
        "jepa_sigreg_valid_count": sigreg_valid_count,
        "jepa_pred_sigreg_valid_count": pred_sigreg_valid_count,
        "jepa_teacher_forcing": teacher_forcing,
        "value_loss": value_loss,
        "wdl_loss": wdl_loss,
        "value_pred_mean": value_pred_mean,
        "value_target_mean": value_target_mean,
        "jepa_action_contrast_loss": action_contrast_loss,
        "jepa_shuffled_loss": shuffled_jepa_loss,
        "jepa_shuffled_cosine_loss": shuffled_jepa_cosine_loss,
        "jepa_shuffled_mean_token_cosine": shuffled_mean_token_cosine,
        "jepa_true_minus_shuffled": zero,
        "jepa_loss_by_horizon": jepa_loss_by_horizon,
        "jepa_raw_mse_by_horizon": jepa_raw_mse_by_horizon,
        "jepa_norm_loss_by_horizon": jepa_norm_loss_by_horizon,
        "jepa_target_horizon_mask": target_horizon_mask,
        "jepa_target_sample_count": jnp.asarray(
            configured_target_count
            if per_example_target_sampling
            else selected_horizons.shape[0],
            dtype=jnp.float32,
        ),
        "jepa_target_sample_fraction": jnp.asarray(
            (
                configured_target_count
                if per_example_target_sampling
                else selected_horizons.shape[0]
            )
            / horizon,
            dtype=jnp.float32,
        ),
        "jepa_target_mean_horizon": (
            jnp.mean(
                selected_horizons_by_example.astype(jnp.float32) + 1.0
            )
            if per_example_target_sampling
            else jnp.mean(selected_horizons.astype(jnp.float32) + 1.0)
        ),
        "mean_token_cosine": mean_token_cosine,
        "mean_token_cosine_by_horizon": mean_token_cosine_by_horizon,
        "pred_token_norm": pred_token_norm,
        "target_token_norm": target_token_norm,
        "z_state_norm": jnp.mean(jnp.linalg.norm(jnp.asarray(z_all, dtype=jnp.float32), axis=-1)),
        "z_state_std": jnp.std(jnp.asarray(z_all, dtype=jnp.float32)),
        "z_pred_norm": jnp.mean(jnp.linalg.norm(jnp.asarray(pred_z, dtype=jnp.float32), axis=-1)),
        "z_target_norm": jnp.mean(jnp.linalg.norm(jnp.asarray(target_z, dtype=jnp.float32), axis=-1)),
        "jepa_state_rms_scale_raw_max": jnp.max(jepa_state_rms_scale_raw),
        "jepa_state_rms_scale_raw_min": jnp.min(jepa_state_rms_scale_raw),
        "jepa_state_rms_scale_effective_max": jnp.max(jepa_state_rms_scale),
        "jepa_state_rms_scale_effective_min": jnp.min(jepa_state_rms_scale),
        "identity_jepa_loss": identity_jepa_loss,
        "identity_jepa_cosine_loss": identity_jepa_cosine_loss,
        "identity_mean_token_cosine": identity_mean_token_cosine,
        "jepa_loss_minus_identity": zero,
        "accuracy": accuracy,
        "mask_prob": jnp.mean(1.0 - t),
        "loss_horizon": jnp.asarray(loss_horizon, dtype=jnp.float32),
    }
    if feedback_active:
        assert feedback_result is not None
        assert root_proposal is not None
        aux.update(
            {
                "jepa_feedback_active": jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                ),
                "jepa_feedback_preliminary_planner_calls": jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                ),
                "jepa_feedback_conditioned_planner_calls": jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                ),
                "jepa_feedback_preliminary_dfm_ce_loss": (
                    preliminary_dfm_ce_loss
                ),
                "jepa_feedback_preliminary_dfm_ce_loss_by_horizon": (
                    preliminary_dfm_ce_loss_by_horizon
                ),
                "jepa_feedback_dfm_ce_gain": (
                    preliminary_dfm_ce_loss - dfm_ce_loss
                ),
                "jepa_feedback_dfm_ce_gain_by_horizon": (
                    preliminary_dfm_ce_loss_by_horizon
                    - dfm_ce_loss_by_horizon
                ),
                "jepa_feedback_proposal_confidence_mean": jnp.mean(
                    root_proposal.confidence
                ),
                "jepa_feedback_gate_mean": jnp.mean(
                    root_proposal.feedback_gate
                ),
                "jepa_feedback_gate_nonzero_fraction": jnp.mean(
                    (
                        root_proposal.feedback_gate > 0.0
                    ).astype(jnp.float32)
                ),
                "jepa_feedback_delta_rms_mean": jnp.mean(
                    feedback_result.delta_rms
                ),
                "jepa_feedback_raw_rms_mean": jnp.mean(
                    feedback_result.raw_feedback_rms
                ),
                "jepa_feedback_applied_rms_mean": jnp.mean(
                    feedback_result.applied_feedback_rms
                ),
                "jepa_feedback_state_rms_mean": jnp.mean(
                    feedback_result.state_rms
                ),
                "jepa_feedback_applied_to_state_rms_ratio": (
                    jnp.mean(
                        feedback_result.applied_feedback_rms
                        / jnp.maximum(
                            feedback_result.state_rms,
                            JEPA_STATE_RMS_EPSILON,
                        )
                    )
                ),
                "jepa_feedback_cap_fraction": (
                    feedback_result.cap_fraction
                ),
            }
        )
    if bt4_policy_distillation is not None:
        aux.update(
            {
                "bt4_policy_distill_loss": (
                    bt4_policy_distillation.loss
                ),
                "bt4_policy_distill_weighted_loss": (
                    jnp.asarray(
                        model.config.bt4_policy_distill_coeff,
                        dtype=jnp.float32,
                    )
                    * bt4_policy_distillation.loss
                ),
                "bt4_policy_distill_eligible_count": (
                    bt4_policy_distillation.eligible_count
                ),
                "bt4_policy_distill_eligible_fraction": (
                    bt4_policy_distillation.eligible_fraction
                ),
                "bt4_policy_distill_side_plane_valid_fraction": (
                    bt4_policy_distillation.side_plane_valid_fraction
                ),
                "bt4_policy_distill_mapping_complete_fraction": (
                    bt4_policy_distillation.mapping_complete_fraction
                ),
                "bt4_policy_distill_legal_mapping_coverage": (
                    bt4_policy_distillation.legal_mapping_coverage
                ),
                "bt4_policy_distill_teacher_finite_fraction": (
                    bt4_policy_distillation.teacher_finite_fraction
                ),
                "bt4_policy_distill_teacher_entropy": (
                    bt4_policy_distillation.teacher_entropy
                ),
                "bt4_policy_distill_teacher_played_nll": (
                    bt4_policy_distillation.teacher_played_nll
                ),
                "bt4_policy_distill_teacher_played_accuracy": (
                    bt4_policy_distillation.teacher_played_accuracy
                ),
                "bt4_policy_distill_teacher_student_top1_agreement": (
                    bt4_policy_distillation.teacher_student_top1_agreement
                ),
            }
        )
    if root_legal_conditional_ce is not None:
        aux.update(
            {
                "root_legal_conditional_ce_loss": (
                    root_legal_conditional_ce.loss
                ),
                "root_legal_conditional_ce_weighted_loss": (
                    jnp.asarray(
                        model.config.root_legal_conditional_ce_coeff,
                        dtype=jnp.float32,
                    )
                    * root_legal_conditional_ce.loss
                ),
                "root_legal_conditional_ce_candidate_count": (
                    root_legal_conditional_ce.candidate_count
                ),
                "root_legal_conditional_ce_eligible_count": (
                    root_legal_conditional_ce.eligible_count
                ),
                "root_legal_conditional_ce_eligible_fraction": (
                    root_legal_conditional_ce.eligible_fraction
                ),
                "root_legal_conditional_ce_legal_metadata_valid_fraction": (
                    root_legal_conditional_ce
                    .legal_metadata_valid_fraction
                ),
                "root_legal_conditional_ce_legal_count_valid_fraction": (
                    root_legal_conditional_ce
                    .legal_count_valid_fraction
                ),
                "root_legal_conditional_ce_played_in_legal_fraction": (
                    root_legal_conditional_ce
                    .played_in_legal_fraction
                ),
                "root_legal_conditional_ce_finite_fraction": (
                    root_legal_conditional_ce.finite_fraction
                ),
                "root_legal_conditional_ce_mean_legal_count": (
                    root_legal_conditional_ce.mean_legal_count
                ),
                "root_legal_conditional_ce_top1_accuracy": (
                    root_legal_conditional_ce.top1_accuracy
                ),
            }
        )
    if wdl_diagnostics is not None:
        aux.update(wdl_diagnostics)
    if dfm_objective_weight_by_horizon is not None:
        aux.update(
            {
                "dfm_objective_ce_loss": dfm_objective_ce_loss,
                "dfm_objective_weight_by_horizon": (
                    dfm_objective_weight_by_horizon
                ),
            }
        )
    if force_first_mask_active:
        aux["dfm_force_first_action_mask"] = jnp.asarray(
            1.0,
            dtype=jnp.float32,
        )
    if training_time_power_active:
        aux["dfm_training_time_power"] = jnp.asarray(
            model.config.dfm_training_time_power,
            dtype=jnp.float32,
        )
    if model.config.dfm_active_layers > 0:
        aux.update(
            {
                "dfm_configured_layers": jnp.asarray(
                    model.config.dfm_layers,
                    dtype=jnp.float32,
                ),
                "dfm_active_layers": jnp.asarray(
                    model.config.dfm_active_layers,
                    dtype=jnp.float32,
                ),
                "dfm_active_layer_fraction": jnp.asarray(
                    model.config.dfm_active_layers
                    / model.config.dfm_layers,
                    dtype=jnp.float32,
                ),
            }
        )
    if model.config.jepa_projector_active_layers > 0:
        aux.update(
            {
                "jepa_projector_configured_layers": jnp.asarray(
                    model.config.projector_layers,
                    dtype=jnp.float32,
                ),
                "jepa_projector_active_layers": jnp.asarray(
                    model.config.jepa_projector_active_layers,
                    dtype=jnp.float32,
                ),
                "jepa_projector_active_layer_fraction": jnp.asarray(
                    model.config.jepa_projector_active_layers
                    / model.config.projector_layers,
                    dtype=jnp.float32,
                ),
            }
        )
    if target_sampling_active:
        aux.update(
            {
                "jepa_target_sampling_active": jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                ),
                "jepa_target_future_importance_weight": (
                    target_future_importance
                ),
                "bt4_encoded_boards_per_example": jnp.asarray(
                    (
                        configured_target_count
                        if per_example_target_sampling
                        else selected_horizons.shape[0]
                    )
                    + 1,
                    dtype=jnp.float32,
                ),
                "jepa_prediction_horizon_count": jnp.asarray(
                    pred_z.shape[1],
                    dtype=jnp.float32,
                ),
            }
        )
        if per_example_target_sampling:
            aux["jepa_target_assignment_count_by_horizon"] = (
                target_assignment_count_by_horizon
            )
        if model.config.bt4_future_target_stop_gradient:
            trainable_tail_layers = int(
                model.config.bt4_future_target_trainable_tail_layers
            )
            partial_future_boards = (
                configured_target_count
                if trainable_tail_layers > 0
                else 0
            )
            stopped_future_boards = (
                configured_target_count
                if trainable_tail_layers == 0
                else 0
            )
            aux.update(
                {
                    "bt4_future_target_stop_gradient": jnp.asarray(
                        1.0,
                        dtype=jnp.float32,
                    ),
                    "bt4_trainable_encoded_boards_per_example": (
                        jnp.asarray(
                            1.0 + partial_future_boards,
                            dtype=jnp.float32,
                        )
                    ),
                    "bt4_stop_gradient_encoded_boards_per_example": (
                        jnp.asarray(
                            stopped_future_boards,
                            dtype=jnp.float32,
                        )
                    ),
                    "bt4_full_gradient_encoded_boards_per_example": (
                        jnp.asarray(1.0, dtype=jnp.float32)
                    ),
                    "bt4_partial_gradient_encoded_boards_per_example": (
                        jnp.asarray(
                            partial_future_boards,
                            dtype=jnp.float32,
                        )
                    ),
                    "bt4_future_target_trainable_tail_layers": (
                        jnp.asarray(
                            trainable_tail_layers,
                            dtype=jnp.float32,
                        )
                    ),
                    "bt4_future_target_detached_prefix_layers": (
                        jnp.asarray(
                            15 - trainable_tail_layers,
                            dtype=jnp.float32,
                        )
                    ),
                }
            )
        if model.config.jepa_sampled_target_anchors:
            eligible_anchor_slots = (
                valid[:, None] * future_valid[:, : max(horizon - 1, 0)]
            )
            used_anchor_slots = (
                eligible_anchor_slots
                * downstream_anchor_horizon_mask[None, : horizon - 1]
            )
            aux.update(
                {
                    "jepa_sampled_target_anchors_active": jnp.asarray(
                        1.0,
                        dtype=jnp.float32,
                    ),
                    "jepa_sampled_target_anchor_horizon_mask": (
                        downstream_anchor_horizon_mask
                    ),
                    "jepa_sampled_target_anchor_horizon_count": jnp.sum(
                        downstream_anchor_horizon_mask
                    ),
                    "jepa_sampled_target_anchor_example_count": jnp.sum(
                        used_anchor_slots
                    ),
                    "jepa_sampled_target_anchor_fraction": (
                        jnp.sum(used_anchor_slots)
                        / jnp.maximum(jnp.sum(eligible_anchor_slots), 1.0)
                    ),
                }
            )
    if target_variance_hinge is not None:
        aux.update(
            {
                "jepa_target_variance_hinge_loss": (
                    target_variance_hinge.loss
                ),
                "jepa_target_variance_valid_count_by_horizon": (
                    target_variance_hinge.valid_count_by_horizon
                ),
                "jepa_target_variance_eligible_by_horizon": (
                    target_variance_hinge.eligible_by_horizon
                ),
                "jepa_target_feature_std_mean_by_horizon": (
                    target_variance_hinge.feature_std_mean_by_horizon
                ),
                "jepa_target_feature_std_p05_by_horizon": (
                    target_variance_hinge.feature_std_p05_by_horizon
                ),
                "jepa_target_feature_std_median_by_horizon": (
                    target_variance_hinge.feature_std_median_by_horizon
                ),
                "jepa_target_variance_active_fraction_by_horizon": (
                    target_variance_hinge.active_fraction_by_horizon
                ),
                "jepa_target_variance_hinge_by_horizon": (
                    target_variance_hinge.hinge_by_horizon
                ),
            }
        )
    if model.config.jepa_state_fixed_unit_rms:
        aux.update(
            {
                "jepa_state_fixed_unit_rms": jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                ),
                "jepa_state_trainable_scale_used": jnp.asarray(
                    0.0,
                    dtype=jnp.float32,
                ),
            }
        )
    if model.config.jepa_sigreg_estimator == "u_stat":
        aux.update(
            {
                "jepa_sigreg_u_stat_discrepancy": (
                    jepa_sigreg_u_stat_discrepancy
                ),
                "jepa_pred_sigreg_u_stat_discrepancy": (
                    jepa_pred_sigreg_u_stat_discrepancy
                ),
            }
        )
    if model.config.jepa_sigreg_example_count > 0:
        aux.update(
            {
                "jepa_sigreg_example_count": jnp.asarray(
                    effective_sigreg_example_count,
                    dtype=jnp.float32,
                ),
                "jepa_sigreg_sampling_fraction": jnp.asarray(
                    effective_sigreg_example_count / batch_size,
                    dtype=jnp.float32,
                ),
            }
        )
    if positive_target_override is not None:
        aux.update(
            {
                "jepa_target_semantics_ema": jnp.asarray(
                    1.0,
                    dtype=jnp.float32,
                ),
                "jepa_positive_ema_target_rms": positive_target_rms,
                "jepa_online_future_target_rms": online_target_rms,
                "jepa_positive_pred_rms": positive_pred_rms,
                "jepa_pred_positive_ema_target_cosine": (
                    positive_pred_target_cosine
                ),
                "jepa_online_future_vs_ema_target_mse": (
                    online_positive_target_mse
                ),
                "z_online_future_target_norm": aux["z_target_norm"],
                "z_positive_ema_target_norm": jnp.mean(
                    jnp.linalg.norm(
                        jnp.asarray(
                            target_vectors,
                            dtype=jnp.float32,
                        ),
                        axis=-1,
                    )
                ),
            }
        )
    return loss, aux


def validate_learning_rate_schedule_config(
    config: JointLatentSASAConfig,
) -> None:
    """Reject ambiguous or inert learning-rate schedule settings."""

    for name in (
        "lr_warmup_steps",
        "lr_decay_start_steps",
        "lr_decay_steps",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer, found {value!r}"
            )
    if (
        isinstance(config.lr_min_ratio, bool)
        or not math.isfinite(config.lr_min_ratio)
        or not 0.0 < config.lr_min_ratio <= 1.0
    ):
        raise ValueError(
            "lr_min_ratio must be finite and in (0, 1], found "
            f"{config.lr_min_ratio!r}"
        )
    if config.lr_decay_steps == 0:
        if config.lr_decay_start_steps != 0:
            raise ValueError(
                "lr_decay_start_steps is inert when lr_decay_steps=0; "
                "keep it at 0"
            )
        if config.lr_min_ratio != 1.0:
            raise ValueError(
                "lr_min_ratio is inert when lr_decay_steps=0; keep it "
                "at 1.0"
            )
    elif config.lr_warmup_steps != 0:
        raise ValueError(
            "Simultaneous learning-rate warmup and decay are unsupported; "
            "set lr_warmup_steps=0 when lr_decay_steps is positive"
        )


def validate_bt4_freeze_config(config: JointLatentSASAConfig) -> None:
    """Reject ambiguous frozen-backbone configurations."""

    if not config.bt4_freeze_backbone:
        return
    if not config.unfreeze_bt4_encoder:
        raise ValueError(
            "bt4_freeze_backbone preserves the source model ABI and requires "
            "unfreeze_bt4_encoder=True"
        )
    if config.bt4_learning_rate != 0.0:
        raise ValueError(
            "bt4_freeze_backbone requires bt4_learning_rate=0.0, found "
            f"{config.bt4_learning_rate!r}"
        )


def validate_bt4_future_target_stop_gradient_config(
    config: JointLatentSASAConfig,
) -> None:
    """Restrict asymmetric BT4 gradients to the measured K=1 contract."""

    trainable_tail_layers = config.bt4_future_target_trainable_tail_layers
    if (
        isinstance(trainable_tail_layers, bool)
        or not isinstance(trainable_tail_layers, (int, np.integer))
        or not 0 <= trainable_tail_layers <= 15
    ):
        raise ValueError(
            "bt4_future_target_trainable_tail_layers must be an integer in "
            f"[0, 15], found {trainable_tail_layers!r}"
        )
    if not config.bt4_future_target_stop_gradient:
        if trainable_tail_layers != 0:
            raise ValueError(
                "bt4_future_target_trainable_tail_layers requires "
                "bt4_future_target_stop_gradient=True"
            )
        return
    requirements = {
        "bt4_freeze_backbone": (config.bt4_freeze_backbone, False),
        "unfreeze_bt4_encoder": (config.unfreeze_bt4_encoder, True),
        "jepa_target_semantics": (config.jepa_target_semantics, "online"),
        "jepa_target_mode": (config.jepa_target_mode, "projected_bt4"),
        "jepa_target_sample_count": (config.jepa_target_sample_count, 1),
        "jepa_target_sampling_unit": (
            config.jepa_target_sampling_unit,
            "example_balanced",
        ),
        "bt4_encode_chunk_size": (config.bt4_encode_chunk_size, 0),
        "jepa_sampled_target_anchors": (
            config.jepa_sampled_target_anchors,
            False,
        ),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in requirements.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "bt4_future_target_stop_gradient requires "
            + ", ".join(mismatches)
        )


def validate_legacy_init_for_config(
    config: JointLatentSASAConfig,
    init_mode: str,
) -> None:
    """Require fresh optimizer state for a changed optimizer ABI."""

    if config.bt4_freeze_backbone and init_mode != "model-only":
        raise ValueError(
            "bt4_freeze_backbone changes the optimizer ABI and requires "
            "--init model-only for legacy initialization"
        )


def learning_rate_schedules(config: JointLatentSASAConfig):
    """Return main and BT4 schedules sharing one optimizer-step contract."""

    validate_learning_rate_schedule_config(config)

    def schedule(peak: float):
        if config.lr_decay_steps > 0:
            cosine = optax.cosine_decay_schedule(
                init_value=peak,
                decay_steps=config.lr_decay_steps,
                alpha=config.lr_min_ratio,
            )
            if config.lr_decay_start_steps == 0:
                return cosine
            return optax.join_schedules(
                [optax.constant_schedule(peak), cosine],
                boundaries=[config.lr_decay_start_steps],
            )
        if config.lr_warmup_steps > 0:
            return optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=peak,
                        transition_steps=config.lr_warmup_steps,
                    ),
                    optax.constant_schedule(peak),
                ],
                boundaries=[config.lr_warmup_steps],
            )
        return optax.constant_schedule(peak)

    return schedule(config.learning_rate), schedule(config.bt4_learning_rate)


def _cosine_decay_ratio(
    *,
    update: int,
    start: int,
    decay_steps: int,
    minimum: float,
) -> float:
    relative_update = min(max(update - start, 0), decay_steps)
    progress = relative_update / decay_steps
    return minimum + (1.0 - minimum) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def learning_rate_schedule_contract(
    config: JointLatentSASAConfig,
) -> dict[str, Any]:
    """Describe the zero-based optimizer-update schedule in run metadata."""

    validate_learning_rate_schedule_config(config)
    if config.lr_decay_steps > 0:
        start = int(config.lr_decay_start_steps)
        decay_steps = int(config.lr_decay_steps)
        end = start + decay_steps
        boundary_updates = {0, start, start + decay_steps // 2, end - 1, end}
        if start > 0:
            boundary_updates.add(start - 1)
        ratios = {
            str(update): _cosine_decay_ratio(
                update=update,
                start=start,
                decay_steps=decay_steps,
                minimum=float(config.lr_min_ratio),
            )
            for update in sorted(boundary_updates)
        }
        family = "constant_then_cosine_decay"
    elif config.lr_warmup_steps > 0:
        warmup_steps = int(config.lr_warmup_steps)
        boundary_updates = {0, warmup_steps - 1, warmup_steps}
        ratios = {
            str(update): min(update / warmup_steps, 1.0)
            for update in sorted(boundary_updates)
        }
        family = "linear_warmup_then_constant"
    else:
        ratios = {"0": 1.0}
        family = "constant"
    return {
        "family": family,
        "counter": "zero_based_optimizer_update",
        "main_peak_learning_rate": float(config.learning_rate),
        "bt4_peak_learning_rate": float(config.bt4_learning_rate),
        "warmup_steps": int(config.lr_warmup_steps),
        "decay_start_update": int(config.lr_decay_start_steps),
        "decay_transition_steps": int(config.lr_decay_steps),
        "minimum_ratio": float(config.lr_min_ratio),
        "ratio_by_update": ratios,
        "learning_rate_by_update": {
            update: {
                "main": float(config.learning_rate) * ratio,
                "bt4": float(config.bt4_learning_rate) * ratio,
            }
            for update, ratio in ratios.items()
        },
    }


def _optimizer_partition_for_path(
    path: tuple[Any, ...],
    *,
    freeze_backbone: bool,
) -> str:
    parts = [str(getattr(item, "key", item)) for item in path]
    name = "/".join(parts)
    if name.startswith("encoder/embedding") or name.startswith(
        "encoder/layers"
    ):
        return "frozen_bt4" if freeze_backbone else "bt4"
    return "main"


def create_joint_optimizer(
    model: JointLatentSASAModel,
    config: JointLatentSASAConfig,
    *,
    freeze_backbone: bool | None = None,
) -> nnx.Optimizer:
    """Build either the configured or source-compatible optimizer ABI."""

    if freeze_backbone is None:
        freeze_backbone = config.bt4_freeze_backbone
    learning_rate, bt4_learning_rate = learning_rate_schedules(config)
    if config.use_muon:
        from chess_dfm_jax.nnx_bt4 import muon_adamw

        main_tx = muon_adamw(
            learning_rate=learning_rate,
            weight_decay=config.weight_decay,
        )
        if not freeze_backbone:
            bt4_tx = muon_adamw(
                learning_rate=bt4_learning_rate,
                weight_decay=config.weight_decay,
            )
    else:
        main_tx = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=config.weight_decay,
        )
        if not freeze_backbone:
            bt4_tx = optax.adamw(
                learning_rate=bt4_learning_rate,
                weight_decay=config.weight_decay,
            )

    def label_one(path, _value):
        return _optimizer_partition_for_path(
            path,
            freeze_backbone=freeze_backbone,
        )

    def label_tree(params):
        return jax.tree_util.tree_map_with_path(label_one, params)

    transforms = {"main": main_tx}
    if freeze_backbone:
        transforms["frozen_bt4"] = optax.set_to_zero()
    else:
        transforms["bt4"] = bt4_tx
    tx = optax.multi_transform(
        transforms,
        label_tree,
    )
    if config.skip_nonfinite_updates:
        tx = optax.apply_if_finite(
            tx,
            max_consecutive_errors=1_000_000_000,
        )
    if config.grad_clip_norm > 0.0:
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            tx,
        )
    optimizer = nnx.Optimizer(model, tx, wrt=TrainableParam)
    return optimizer


def create_joint_components(
    bt4_params: dict[str, Any],
    config: JointLatentSASAConfig,
    *,
    seed: int = 0,
) -> tuple[JointLatentSASAModel, nnx.Optimizer]:
    """Build the local checkpoint-compatible model and configured optimizer."""

    validate_bt4_freeze_config(config)
    validate_bt4_future_target_stop_gradient_config(config)
    encoder_dtype = _parse_compute_dtype(config.encoder_dtype)
    encoder = make_bt4_model(
        bt4_params,
        dtype=encoder_dtype,
        train_encoder=config.unfreeze_bt4_encoder,
    )
    model = JointLatentSASAModel(
        encoder,
        config,
        rngs=nnx.Rngs(seed),
    )
    optimizer = create_joint_optimizer(model, config)
    return model, optimizer


def release_construction_parameter_payload(
    mapped_params: dict[str, Any],
) -> None:
    """Release the construction-only NumPy BT4 tree before later heavy work."""

    if type(mapped_params) is not dict:
        raise TypeError(
            "Mapped construction parameters must be a plain dict, got "
            f"{type(mapped_params).__name__}"
        )
    mapped_params.clear()
    gc.collect()


def resolve_config(
    run_root: Path,
) -> tuple[JointLatentSASAConfig, dict[str, Any]]:
    """Read the legacy metadata while selecting only local model fields."""

    metadata_path = require_within_workspace(run_root / "checkpoint_state.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        raise TypeError(f"Missing config object in {metadata_path}")
    field_names = {
        field.name for field in dataclasses.fields(JointLatentSASAConfig)
    }
    config = JointLatentSASAConfig(
        **{
            key: value
            for key, value in raw_config.items()
            if key in field_names
        }
    )
    return config, metadata


def validate_objective_config(
    *,
    objective: str,
    config: JointLatentSASAConfig,
) -> None:
    validate_learning_rate_schedule_config(config)
    first_action_share = config.dfm_first_action_loss_share
    if (
        isinstance(first_action_share, bool)
        or not math.isfinite(first_action_share)
        or not 0.0 <= first_action_share <= 1.0
    ):
        raise ValueError(
            "dfm_first_action_loss_share must be finite and in [0, 1], "
            f"found {first_action_share!r}"
        )
    if first_action_share != 0.0:
        if objective != "normalized":
            raise ValueError(
                "dfm_first_action_loss_share requires --objective normalized."
            )
        effective_loss_horizon = (
            config.horizon
            if config.loss_horizon <= 0
            else min(config.loss_horizon, config.horizon)
        )
        if effective_loss_horizon < 2:
            raise ValueError(
                "dfm_first_action_loss_share requires at least two active "
                "loss horizons."
            )
    if not isinstance(config.dfm_force_first_action_mask, bool):
        raise ValueError(
            "dfm_force_first_action_mask must be boolean, found "
            f"{config.dfm_force_first_action_mask!r}"
        )
    if config.dfm_force_first_action_mask and objective != "normalized":
        raise ValueError(
            "dfm_force_first_action_mask requires --objective normalized."
        )
    training_time_power = config.dfm_training_time_power
    if (
        isinstance(training_time_power, bool)
        or not math.isfinite(training_time_power)
        or training_time_power <= 0.0
    ):
        raise ValueError(
            "dfm_training_time_power must be finite and positive, found "
            f"{training_time_power!r}"
        )
    if training_time_power != 1.0 and objective != "normalized":
        raise ValueError(
            "dfm_training_time_power requires --objective normalized."
        )
    if (
        isinstance(config.first_legality_coeff, bool)
        or not math.isfinite(config.first_legality_coeff)
        or config.first_legality_coeff < 0.0
    ):
        raise ValueError(
            "first_legality_coeff must be finite and non-negative, found "
            f"{config.first_legality_coeff!r}"
        )
    if (
        isinstance(config.bt4_policy_distill_coeff, bool)
        or not math.isfinite(config.bt4_policy_distill_coeff)
        or config.bt4_policy_distill_coeff < 0.0
    ):
        raise ValueError(
            "bt4_policy_distill_coeff must be finite and non-negative, "
            f"found {config.bt4_policy_distill_coeff!r}"
        )
    if (
        config.bt4_policy_distill_coeff != 0.0
        and objective != "normalized"
    ):
        raise ValueError(
            "bt4_policy_distill_coeff requires --objective normalized."
        )
    if (
        config.bt4_policy_distill_coeff != 0.0
        and config.action_vocab_size
        != _LEGACY_TO_CANONICAL_WHITE.shape[0]
    ):
        raise ValueError(
            "BT4 policy distillation requires action_vocab_size="
            f"{_LEGACY_TO_CANONICAL_WHITE.shape[0]}, found "
            f"{config.action_vocab_size}."
        )
    if (
        isinstance(config.root_legal_conditional_ce_coeff, bool)
        or not math.isfinite(
            config.root_legal_conditional_ce_coeff
        )
        or config.root_legal_conditional_ce_coeff < 0.0
    ):
        raise ValueError(
            "root_legal_conditional_ce_coeff must be finite and "
            "non-negative, found "
            f"{config.root_legal_conditional_ce_coeff!r}"
        )
    if (
        config.root_legal_conditional_ce_coeff != 0.0
        and objective != "normalized"
    ):
        raise ValueError(
            "root_legal_conditional_ce_coeff requires "
            "--objective normalized."
        )
    if (
        isinstance(config.wdl_coeff, bool)
        or not math.isfinite(config.wdl_coeff)
        or config.wdl_coeff < 0.0
    ):
        raise ValueError(
            "wdl_coeff must be finite and non-negative, found "
            f"{config.wdl_coeff!r}"
        )
    if config.wdl_coeff != 0.0 and objective != "normalized":
        raise ValueError(
            "wdl_coeff requires --objective normalized."
        )
    dfm_active_layers = config.dfm_active_layers
    if (
        isinstance(dfm_active_layers, bool)
        or not isinstance(dfm_active_layers, (int, np.integer))
        or not 0 <= dfm_active_layers <= config.dfm_layers
    ):
        raise ValueError(
            "dfm_active_layers must be an integer in "
            f"[0, {config.dfm_layers}], found "
            f"{dfm_active_layers!r}"
        )
    projector_active_layers = config.jepa_projector_active_layers
    if (
        isinstance(projector_active_layers, bool)
        or not isinstance(projector_active_layers, (int, np.integer))
        or not 0 <= projector_active_layers <= config.projector_layers
    ):
        raise ValueError(
            "jepa_projector_active_layers must be an integer in "
            f"[0, {config.projector_layers}], found "
            f"{projector_active_layers!r}"
        )
    if objective == "normalized" and config.jepa_sigreg_kind != "le_jepa":
        raise ValueError(
            "--objective normalized requires jepa_sigreg_kind='le_jepa'; "
            f"found {config.jepa_sigreg_kind!r}"
        )
    if (
        not math.isfinite(config.jepa_norm_loss_coeff)
        or config.jepa_norm_loss_coeff < 0.0
    ):
        raise ValueError(
            "jepa_norm_loss_coeff must be finite and non-negative, found "
            f"{config.jepa_norm_loss_coeff}"
        )
    if objective == "legacy" and config.jepa_norm_loss_coeff != 1.0:
        raise ValueError(
            "--objective legacy requires jepa_norm_loss_coeff=1.0 for "
            "exact compatibility; use --objective normalized for the "
            "no-norm ablation."
        )
    if config.jepa_sigreg_estimator not in ("v_stat", "u_stat"):
        raise ValueError(
            "jepa_sigreg_estimator must be 'v_stat' or 'u_stat', found "
            f"{config.jepa_sigreg_estimator!r}"
        )
    if (
        config.jepa_sigreg_estimator == "u_stat"
        and objective != "normalized"
    ):
        raise ValueError(
            "jepa_sigreg_estimator='u_stat' requires "
            "--objective normalized."
        )
    if (
        config.jepa_sigreg_estimator == "u_stat"
        and config.jepa_sigreg_kind != "le_jepa"
    ):
        raise ValueError(
            "jepa_sigreg_estimator='u_stat' requires "
            "jepa_sigreg_kind='le_jepa'."
        )
    if (
        isinstance(config.jepa_sigreg_example_count, bool)
        or not isinstance(
            config.jepa_sigreg_example_count,
            (int, np.integer),
        )
        or config.jepa_sigreg_example_count < 0
    ):
        raise ValueError(
            "jepa_sigreg_example_count must be a non-negative integer, "
            f"found {config.jepa_sigreg_example_count!r}"
        )
    if (
        config.jepa_sigreg_example_count > 0
        and objective != "normalized"
    ):
        raise ValueError(
            "Fixed-count SIGReg example sampling requires "
            "--objective normalized."
        )
    target_sample_count = config.jepa_target_sample_count
    if (
        isinstance(target_sample_count, bool)
        or not isinstance(target_sample_count, (int, np.integer))
        or not 0 <= target_sample_count <= config.horizon
    ):
        raise ValueError(
            "jepa_target_sample_count must be an integer in "
            f"[0, {config.horizon}], found {target_sample_count!r}"
        )
    target_sampling_unit = config.jepa_target_sampling_unit
    if target_sampling_unit not in (
        "batch_shared",
        "example_balanced",
    ):
        raise ValueError(
            "jepa_target_sampling_unit must be 'batch_shared' or "
            f"'example_balanced', found {target_sampling_unit!r}"
        )
    if (
        target_sampling_unit == "example_balanced"
        and target_sample_count != 1
    ):
        raise ValueError(
            "jepa_target_sampling_unit='example_balanced' requires "
            "jepa_target_sample_count=1."
        )
    if target_sampling_unit == "example_balanced" and not (
        0 < target_sample_count < config.horizon
    ):
        raise ValueError(
            "jepa_target_sampling_unit='example_balanced' requires a "
            "strict sampled-target count in (0, horizon)."
        )
    if type(config.jepa_sampled_target_anchors) is not bool:
        raise ValueError(
            "jepa_sampled_target_anchors must be a bool, found "
            f"{config.jepa_sampled_target_anchors!r}"
        )
    if config.jepa_sampled_target_anchors and not (
        0 < target_sample_count < config.horizon
    ):
        raise ValueError(
            "jepa_sampled_target_anchors requires a strict sampled-target "
            "count in (0, horizon)."
        )
    if (
        target_sampling_unit == "example_balanced"
        and config.jepa_sampled_target_anchors
    ):
        raise ValueError(
            "jepa_target_sampling_unit='example_balanced' does not "
            "support sampled-target anchors."
        )
    rollout_mode = config.jepa_rollout_mode
    if rollout_mode not in ("recurrent", "direct_sequence"):
        raise ValueError(
            "jepa_rollout_mode must be 'recurrent' or 'direct_sequence', "
            f"found {rollout_mode!r}"
        )
    if rollout_mode == "direct_sequence":
        if objective != "normalized":
            raise ValueError(
                "jepa_rollout_mode='direct_sequence' requires "
                "--objective normalized."
            )
        if (
            target_sample_count != 1
            or target_sampling_unit != "example_balanced"
        ):
            raise ValueError(
                "jepa_rollout_mode='direct_sequence' requires balanced "
                "per-example K=1 target sampling."
            )
        if config.jepa_sampled_target_anchors:
            raise ValueError(
                "jepa_rollout_mode='direct_sequence' does not support "
                "sampled-target anchors."
            )
        if config.jepa_teacher_forcing_steps != 0:
            raise ValueError(
                "jepa_rollout_mode='direct_sequence' does not support "
                "teacher-forced recurrent carries."
            )
    feedback_mode = config.jepa_feedback_mode
    if feedback_mode not in ("none", "final_pass_adjoint"):
        raise ValueError(
            "jepa_feedback_mode must be 'none' or 'final_pass_adjoint', "
            f"found {feedback_mode!r}"
        )
    if feedback_mode == "final_pass_adjoint":
        effective_condition_dim = (
            config.jepa_condition_dim
            if config.jepa_condition_dim > 0
            else config.z_dim
        )
        if objective != "normalized":
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires "
                "--objective normalized."
            )
        if rollout_mode != "recurrent":
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires "
                "jepa_rollout_mode='recurrent'."
            )
        if effective_condition_dim != config.z_dim:
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires the "
                "effective jepa_condition_dim to equal z_dim."
            )
        if (
            target_sample_count != 1
            or target_sampling_unit != "example_balanced"
        ):
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' requires "
                "balanced per-example K=1 target sampling."
            )
        if config.jepa_sampled_target_anchors:
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' does not "
                "support sampled-target anchors."
            )
        if config.jepa_teacher_forcing_steps != 0:
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' does not "
                "support teacher-forced recurrent carries."
            )
        if config.horizon != 8:
            raise ValueError(
                "jepa_feedback_mode='final_pass_adjoint' is frozen for "
                f"horizon=8, found horizon={config.horizon}."
            )
    if 0 < target_sample_count < config.horizon:
        if objective != "normalized":
            raise ValueError(
                "Sampled future targets require --objective normalized."
            )
        if config.jepa_target_semantics != "online":
            raise ValueError(
                "Sampled future targets currently require online targets; "
                "EMA target encoding is intentionally unsupported."
            )
        if config.jepa_teacher_forcing_steps != 0:
            raise ValueError(
                "Sampled future targets cannot be combined with teacher "
                "forcing because unencoded future states are unavailable."
            )
        if config.jepa_target_variance_hinge_coeff != 0.0:
            raise ValueError(
                "Sampled future targets do not yet support the per-horizon "
                "target variance hinge."
            )
    if config.jepa_target_semantics not in ("online", "ema"):
        raise ValueError(
            "jepa_target_semantics must be 'online' or 'ema', found "
            f"{config.jepa_target_semantics!r}"
        )
    if not 0.0 <= config.jepa_target_ema_decay < 1.0:
        raise ValueError(
            "jepa_target_ema_decay must be in [0, 1), found "
            f"{config.jepa_target_ema_decay}"
        )
    if np.float32(config.jepa_target_ema_decay) >= np.float32(1.0):
        raise ValueError(
            "jepa_target_ema_decay rounds to 1.0 in FP32 and would "
            "freeze the EMA target; choose a smaller decay."
        )
    if (
        config.jepa_target_semantics == "online"
        and config.jepa_target_ema_decay
        != JointLatentSASAConfig.jepa_target_ema_decay
    ):
        raise ValueError(
            "jepa_target_ema_decay is inert unless "
            "jepa_target_semantics='ema'; keep its default in online mode."
        )
    if config.jepa_target_semantics == "ema":
        if objective != "normalized":
            raise ValueError(
                "EMA targets require --objective normalized."
            )
        if config.jepa_target_mode != "projected_bt4":
            raise ValueError(
                "EMA targets require jepa_target_mode='projected_bt4'."
            )
        if config.jepa_target_stop_gradient:
            raise ValueError(
                "EMA targets are intrinsically detached; keep "
                "jepa_target_stop_gradient=False to avoid conflating "
                "target semantics."
            )
    variance_hinge_coeff = config.jepa_target_variance_hinge_coeff
    variance_hinge_gamma = config.jepa_target_variance_hinge_gamma
    if (
        not math.isfinite(variance_hinge_coeff)
        or variance_hinge_coeff < 0.0
    ):
        raise ValueError(
            "jepa_target_variance_hinge_coeff must be finite and "
            f"non-negative, found {variance_hinge_coeff}"
        )
    if (
        not math.isfinite(variance_hinge_gamma)
        or variance_hinge_gamma <= 0.0
    ):
        raise ValueError(
            "jepa_target_variance_hinge_gamma must be finite and "
            f"positive, found {variance_hinge_gamma}"
        )
    if (
        variance_hinge_coeff == 0.0
        and variance_hinge_gamma
        != JointLatentSASAConfig.jepa_target_variance_hinge_gamma
    ):
        raise ValueError(
            "jepa_target_variance_hinge_gamma is inert unless "
            "jepa_target_variance_hinge_coeff is nonzero; keep its "
            "default when the hinge is disabled."
        )
    if variance_hinge_coeff != 0.0 and objective != "normalized":
        raise ValueError(
            "Target variance hinge requires --objective normalized."
        )
    if config.jepa_state_fixed_unit_rms:
        if objective != "normalized":
            raise ValueError(
                "Fixed-unit-RMS JEPA state requires --objective normalized."
            )
        if config.jepa_state_rmsnorm:
            raise ValueError(
                "Fixed-unit-RMS JEPA state cannot be combined with the "
                "legacy trainable jepa_state_rmsnorm."
            )
        if config.jepa_target_semantics != "online":
            raise ValueError(
                "Fixed-unit-RMS JEPA state initially requires online "
                "targets and cannot be combined with EMA targets."
            )
        if config.jepa_target_stop_gradient:
            raise ValueError(
                "Fixed-unit-RMS JEPA state cannot be combined with "
                "target stop-gradient."
            )
        if variance_hinge_coeff != 0.0:
            raise ValueError(
                "Fixed-unit-RMS JEPA state cannot be combined with the "
                "target variance hinge."
            )
        if (
            config.jepa_state_rms_scale_max
            != JointLatentSASAConfig.jepa_state_rms_scale_max
        ):
            raise ValueError(
                "jepa_state_rms_scale_max is inert in fixed-unit-RMS "
                "mode; keep its default."
            )


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten scalar and per-horizon arrays for JSON metric records."""

    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.ndim == 0:
            flattened[key] = float(array)
            continue
        flat = array.reshape(-1)
        if key.endswith("_by_horizon") or key in {
            "dfm_ce_loss_by_horizon",
            "dfm_mask_fraction_by_horizon",
            "jepa_loss_by_horizon",
            "jepa_raw_mse_by_horizon",
            "jepa_norm_loss_by_horizon",
            "mean_token_cosine_by_horizon",
            "jepa_target_horizon_mask",
            "jepa_sampled_target_anchor_horizon_mask",
        }:
            for index, item in enumerate(flat):
                flattened[f"{key}_h{index + 1}"] = float(item)
            continue
        for index, item in enumerate(flat):
            flattened[f"{key}_{index}"] = float(item)
    return flattened


def reportable_stage1_aux(
    aux: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove known unevaluated compatibility placeholders from reports."""

    missing = sorted(UNEVALUATED_LEGACY_AUX_METRICS - set(aux))
    if missing:
        raise KeyError(
            "Stage-1 aux is missing expected legacy placeholder key(s): "
            + ", ".join(missing)
        )
    unexpectedly_evaluated = [
        key
        for key in sorted(UNEVALUATED_LEGACY_AUX_METRICS)
        if not np.all(np.asarray(aux[key]) == 0)
    ]
    if unexpectedly_evaluated:
        raise ValueError(
            "Legacy aux metric(s) marked unevaluated became nonzero; "
            "define their reporting semantics before emitting them: "
            + ", ".join(unexpectedly_evaluated)
        )
    return {
        key: value
        for key, value in aux.items()
        if key not in UNEVALUATED_LEGACY_AUX_METRICS
    }


class SigRegResult(NamedTuple):
    normalized: jax.Array
    official: jax.Array
    valid_count: jax.Array
    discrepancy: jax.Array
    u_stat_discrepancy: jax.Array


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as checkpoint JSON")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_normalize(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _abi_mapping_key(key: Any) -> dict[str, Any]:
    if isinstance(key, bool):
        return {"type": "bool", "value": key}
    if isinstance(key, int):
        return {"type": "int", "value": key}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    raise TypeError(f"Unsupported state mapping key {key!r} ({type(key).__name__})")


def _abi_path_text(path: list[dict[str, Any]]) -> str:
    if not path:
        return "<root>"
    parts = []
    for item in path:
        kind = item["type"]
        value = item["value"]
        parts.append(f"[{value}]" if kind in {"index", "int"} else f".{value}")
    return "".join(parts).lstrip(".")


def _state_schema_records(
    value: Any,
    *,
    path: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    path_list = list(path)
    if isinstance(value, Mapping):
        keyed = [(_abi_mapping_key(key), child) for key, child in value.items()]
        keyed.sort(key=lambda item: _canonical_json_bytes(item[0]))
        records = [
            {
                "path": path_list,
                "kind": "mapping",
                "keys": [key for key, _ in keyed],
            }
        ]
        for key, child in keyed:
            records.extend(_state_schema_records(child, path=path + (key,)))
        return records

    if isinstance(value, list):
        records = [{"path": path_list, "kind": "list", "length": len(value)}]
        for index, child in enumerate(value):
            key = {"type": "index", "value": index}
            records.extend(_state_schema_records(child, path=path + (key,)))
        return records

    if isinstance(value, tuple):
        records = [
            {
                "path": path_list,
                "kind": "tuple",
                "length": len(value),
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
            }
        ]
        for index, child in enumerate(value):
            key = {"type": "index", "value": index}
            records.extend(_state_schema_records(child, path=path + (key,)))
        return records

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"Object-valued state leaf at {_abi_path_text(path_list)}")
    return [
        {
            "path": path_list,
            "kind": "leaf",
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "nbytes": int(array.size * array.dtype.itemsize),
        }
    ]


def research_state_abi(value: Any) -> dict[str, Any]:
    """Return a canonical path/shape/dtype ABI for a model or optimizer tree."""

    schema = _state_schema_records(value)
    leaves = [record for record in schema if record["kind"] == "leaf"]
    return {
        "schema_version": 1,
        "sha256": _json_sha256(schema),
        "leaf_count": len(leaves),
        "nbytes": sum(int(record["nbytes"]) for record in leaves),
        "schema": schema,
    }


def _schema_by_path(abi: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _canonical_json_bytes(record["path"]).decode("utf-8"): record
        for record in abi["schema"]
    }


def assert_research_state_compatible(
    expected: Any,
    incoming: Any,
    *,
    label: str,
) -> None:
    """Reject any state path, container, shape, or dtype mismatch."""

    expected_abi = research_state_abi(expected)
    incoming_abi = research_state_abi(incoming)
    if expected_abi["sha256"] == incoming_abi["sha256"]:
        return

    expected_records = _schema_by_path(expected_abi)
    incoming_records = _schema_by_path(incoming_abi)
    missing = sorted(set(expected_records) - set(incoming_records))
    extra = sorted(set(incoming_records) - set(expected_records))
    changed = sorted(
        path
        for path in set(expected_records) & set(incoming_records)
        if expected_records[path] != incoming_records[path]
    )

    details: list[str] = []
    for kind, paths in (("missing", missing), ("extra", extra), ("changed", changed)):
        for encoded_path in paths[:3]:
            record = (expected_records if kind != "extra" else incoming_records)[encoded_path]
            details.append(f"{kind} {_abi_path_text(record['path'])}")
    if len(missing) + len(extra) + len(changed) > len(details):
        details.append("additional differences omitted")
    joined = "; ".join(details) or "schema digest differs"
    raise ValueError(
        f"{label} checkpoint ABI mismatch "
        f"(expected {expected_abi['sha256']}, incoming {incoming_abi['sha256']}): {joined}"
    )


def extract_research_train_state(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    ema_target: EmaTargetModel | None = None,
) -> dict[str, Any]:
    """Copy trainable model and optimizer state to a raw NumPy payload."""

    def to_host(value: Any) -> Any:
        if isinstance(value, jax.Array):
            return np.asarray(value)
        return value

    model_state = jax.tree.map(to_host, nnx.state(model, TrainableParam))
    optimizer_state = jax.tree.map(to_host, nnx.state(optimizer.opt_state))
    payload = {
        "step": np.asarray(int(optimizer.step[...]), dtype=np.int64),
        "model_trainable": dict(nnx.to_pure_dict(model_state)),
        "optimizer_state": dict(nnx.to_pure_dict(optimizer_state)),
    }
    if ema_target is not None:
        payload["ema_target"] = {
            name: dict(
                nnx.to_pure_dict(
                    jax.tree.map(to_host, component_state)
                )
            )
            for name, component_state in ema_checkpoint_state(
                ema_target
            ).items()
        }
    return payload


def strict_restore_research_payload(
    payload: dict[str, Any],
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    ema_target: EmaTargetModel | None = None,
) -> int:
    """Preflight both state trees, then restore them without partial fallback."""

    required = {"step", "model_trainable", "optimizer_state"}
    if ema_target is not None:
        required.add("ema_target")
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(f"Research state payload keys differ: missing={missing}, extra={extra}")

    step_array = np.asarray(payload["step"])
    if step_array.shape != () or not np.issubdtype(step_array.dtype, np.integer):
        raise ValueError(
            f"Research optimizer step must be an integer scalar, got "
            f"shape={step_array.shape}, dtype={step_array.dtype}"
        )
    optimizer_step = int(step_array)
    if optimizer_step < 0:
        raise ValueError(f"Research optimizer step must be non-negative, got {optimizer_step}")

    model_state = nnx.state(model, TrainableParam)
    optimizer_state = nnx.state(optimizer.opt_state)
    model_current = dict(nnx.to_pure_dict(model_state))
    optimizer_current = dict(nnx.to_pure_dict(optimizer_state))
    assert_research_state_compatible(
        model_current,
        payload["model_trainable"],
        label="model",
    )
    assert_research_state_compatible(
        optimizer_current,
        payload["optimizer_state"],
        label="optimizer",
    )
    ema_states: dict[str, Any] | None = None
    if ema_target is not None:
        ema_states = ema_checkpoint_state(ema_target)
        ema_current = {
            name: dict(nnx.to_pure_dict(component_state))
            for name, component_state in ema_states.items()
        }
        assert_research_state_compatible(
            ema_current,
            payload["ema_target"],
            label="EMA target",
        )
        assert_ema_encoder_compute_structure(ema_target)

    nnx.replace_by_pure_dict(model_state, payload["model_trainable"])
    nnx.replace_by_pure_dict(optimizer_state, payload["optimizer_state"])
    if ema_states is not None:
        for name, component_state in ema_states.items():
            nnx.replace_by_pure_dict(
                component_state,
                payload["ema_target"][name],
            )
    nnx.update(model, model_state)
    nnx.update(optimizer.opt_state, optimizer_state)
    if ema_states is not None:
        nnx.update(
            ema_target.encoder_master,
            ema_states["encoder_master"],
        )
        nnx.update(
            ema_target.jepa_state_norm,
            ema_states["jepa_state_norm"],
        )
        nnx.update(
            ema_target.state_projector,
            ema_states["state_projector"],
        )
        refresh_ema_encoder_compute(ema_target)
        validate_ema_encoder_compute(ema_target)
    optimizer.step[...] = jnp.asarray(optimizer_step, dtype=optimizer.step[...].dtype)
    return int(optimizer.step[...])


def _research_checkpoint_update(path: Path) -> int | None:
    match = RESEARCH_CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None or not path.is_dir():
        return None
    return int(match.group(1))


def _read_research_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    manifest_path = require_within_workspace(checkpoint_dir / "manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid research checkpoint manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Research checkpoint manifest must be an object: {manifest_path}")
    if manifest.get("format") != RESEARCH_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported research checkpoint format {manifest.get('format')!r}: {manifest_path}"
        )
    directory_update = _research_checkpoint_update(checkpoint_dir)
    if directory_update is None or int(manifest.get("research_update", -1)) != directory_update:
        raise ValueError(f"Research checkpoint directory/manifest update mismatch: {checkpoint_dir}")
    return manifest


def completed_research_checkpoints(checkpoint_root: Path) -> list[Path]:
    """List published checkpoints; hidden/incomplete temporary directories are ignored."""

    checkpoint_root = require_within_workspace(checkpoint_root)
    if not checkpoint_root.is_dir():
        return []
    completed: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        update = _research_checkpoint_update(path)
        if update is None or not (path / "manifest.json").is_file():
            continue
        try:
            manifest = _read_research_manifest(path)
        except ValueError:
            continue
        if not (path / "state.npz").is_file():
            continue
        completed.append((int(manifest["research_update"]), path))
    return [path for _, path in sorted(completed)]


def research_checkpoints_for_evaluation(path: Path) -> tuple[Path, list[Path]]:
    """Resolve one checkpoint or every completed checkpoint in a run/root.

    Unlike routine latest-checkpoint discovery, an evaluation sweep fails on a
    visible malformed ``update*`` directory. Hidden partial directories remain
    ignored because they are never published checkpoints.
    """

    candidate = require_within_workspace(path)
    if _research_checkpoint_update(candidate) is not None:
        _read_research_manifest(candidate)
        if not (candidate / "state.npz").is_file():
            raise ValueError(
                f"Research checkpoint is missing state.npz: {candidate}"
            )
        return candidate.parent, [candidate]

    checkpoint_root = (
        require_within_workspace(candidate / "checkpoints")
        if (candidate / "checkpoints").is_dir()
        else candidate
    )
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"Research checkpoint path is not a directory: {checkpoint_root}"
        )

    malformed: list[str] = []
    for child in checkpoint_root.iterdir():
        if child.name.startswith("."):
            continue
        if RESEARCH_CHECKPOINT_PATTERN.fullmatch(child.name) is None:
            continue
        if not child.is_dir():
            malformed.append(f"{child.name} (not a directory)")
            continue
        try:
            _read_research_manifest(child)
        except ValueError as exc:
            malformed.append(f"{child.name} ({exc})")
            continue
        if not (child / "state.npz").is_file():
            malformed.append(f"{child.name} (missing state.npz)")
    if malformed:
        raise ValueError(
            "Malformed published checkpoint(s) in evaluation root: "
            + "; ".join(malformed)
        )

    checkpoints = completed_research_checkpoints(checkpoint_root)
    if not checkpoints:
        raise FileNotFoundError(
            f"No completed research checkpoint under {checkpoint_root}"
        )
    return checkpoint_root, checkpoints


def latest_research_checkpoint(checkpoint_root: Path) -> Path | None:
    checkpoints = completed_research_checkpoints(checkpoint_root)
    return checkpoints[-1] if checkpoints else None


def resolve_research_checkpoint(path: Path) -> Path:
    """Resolve an exact update directory, a checkpoint root, or a run directory."""

    candidate = require_within_workspace(path)
    if _research_checkpoint_update(candidate) is not None:
        _read_research_manifest(candidate)
        return candidate
    if (candidate / "checkpoints").is_dir():
        candidate = require_within_workspace(candidate / "checkpoints")
    latest = latest_research_checkpoint(candidate)
    if latest is None:
        raise FileNotFoundError(f"No completed research checkpoint under {candidate}")
    return latest


def _load_research_npz(path: Path) -> dict[str, Any]:
    with np.load(require_within_workspace(path), allow_pickle=True) as data:
        payload: dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
                payload[key] = value.item()
            else:
                payload[key] = value
    return payload


def _fsync_path(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prune_research_checkpoints(checkpoint_root: Path, *, max_to_keep: int) -> list[Path]:
    """Prune old completed checkpoints and return the retained paths."""

    checkpoint_root = require_within_workspace(checkpoint_root)
    checkpoints = completed_research_checkpoints(checkpoint_root)
    if max_to_keep <= 0 or len(checkpoints) <= max_to_keep:
        return checkpoints
    for path in checkpoints[:-max_to_keep]:
        shutil.rmtree(path)
    _fsync_path(checkpoint_root)
    return checkpoints[-max_to_keep:]


def save_research_checkpoint(
    checkpoint_root: Path,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    research_update: int,
    next_data_cursor: int,
    resume_contract: dict[str, Any],
    lineage: dict[str, Any],
    max_to_keep: int = 2,
    extra: dict[str, Any] | None = None,
    ema_target: EmaTargetModel | None = None,
) -> Path:
    """Atomically publish a checksummed local research checkpoint."""

    checkpoint_root = require_within_workspace(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    research_update = int(research_update)
    next_data_cursor = int(next_data_cursor)
    if research_update < 0 or next_data_cursor < 0:
        raise ValueError("research_update and next_data_cursor must be non-negative")

    final_dir = checkpoint_root / f"update{research_update:08d}"
    if final_dir.exists():
        raise FileExistsError(f"Research checkpoint already exists: {final_dir}")
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".update{research_update:08d}.partial-",
            dir=checkpoint_root,
        )
    )
    try:
        payload = extract_research_train_state(
            model,
            optimizer,
            ema_target,
        )
        state_path = temporary_dir / "state.npz"
        np.savez(state_path, **payload)
        _fsync_path(state_path)
        state_size = state_path.stat().st_size
        state_sha256 = sha256_file(state_path)

        normalized_contract = _json_normalize(resume_contract)
        normalized_lineage = _json_normalize(lineage)
        manifest = {
            "format": RESEARCH_CHECKPOINT_FORMAT,
            "created_utc": datetime.now(UTC).isoformat(),
            "research_update": research_update,
            "optimizer_step": int(np.asarray(payload["step"])),
            "next_data_cursor": next_data_cursor,
            "resume_contract": normalized_contract,
            "resume_contract_sha256": _json_sha256(normalized_contract),
            "lineage": normalized_lineage,
            "model_abi": research_state_abi(payload["model_trainable"]),
            "optimizer_abi": research_state_abi(payload["optimizer_state"]),
            "state": {
                "filename": "state.npz",
                "size_bytes": state_size,
                "sha256": state_sha256,
            },
            "extra": _json_normalize(extra or {}),
        }
        if ema_target is not None:
            manifest["ema_target_abi"] = research_state_abi(
                payload["ema_target"]
            )
        write_json(temporary_dir / "manifest.json", manifest)
        _fsync_path(temporary_dir / "manifest.json")
        _fsync_path(temporary_dir)
        os.replace(temporary_dir, final_dir)
        _fsync_path(checkpoint_root)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    prune_research_checkpoints(checkpoint_root, max_to_keep=max_to_keep)
    return final_dir


def load_research_checkpoint(
    path: Path,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    expected_resume_contract: dict[str, Any] | None = None,
    ema_target: EmaTargetModel | None = None,
) -> dict[str, Any]:
    """Verify and strictly restore a completed local research checkpoint."""

    checkpoint_dir = resolve_research_checkpoint(path)
    manifest = _read_research_manifest(checkpoint_dir)
    state = manifest.get("state")
    if not isinstance(state, dict) or state.get("filename") != "state.npz":
        raise ValueError(f"Invalid state record in {checkpoint_dir / 'manifest.json'}")
    state_path = require_within_workspace(checkpoint_dir / "state.npz")
    if state_path.stat().st_size != int(state.get("size_bytes", -1)):
        raise ValueError(f"Research checkpoint state size mismatch: {state_path}")
    observed_sha256 = sha256_file(state_path)
    if observed_sha256 != state.get("sha256"):
        raise ValueError(f"Research checkpoint state checksum mismatch: {state_path}")

    contract = manifest.get("resume_contract")
    if _json_sha256(contract) != manifest.get("resume_contract_sha256"):
        raise ValueError(f"Research checkpoint resume contract checksum mismatch: {checkpoint_dir}")
    if expected_resume_contract is not None:
        normalized_expected = _json_normalize(expected_resume_contract)
        if _json_sha256(normalized_expected) != manifest["resume_contract_sha256"]:
            raise ValueError(
                "Research checkpoint resume contract mismatch "
                f"(expected {_json_sha256(normalized_expected)}, "
                f"incoming {manifest['resume_contract_sha256']})"
            )

    payload = _load_research_npz(state_path)
    payload_step = int(np.asarray(payload.get("step", -1)))
    if payload_step != int(manifest.get("optimizer_step", -1)):
        raise ValueError(
            f"Research optimizer step mismatch: manifest={manifest.get('optimizer_step')}, "
            f"payload={payload_step}"
        )
    payload_model_abi = research_state_abi(payload.get("model_trainable"))
    payload_optimizer_abi = research_state_abi(payload.get("optimizer_state"))
    if payload_model_abi != manifest.get("model_abi"):
        raise ValueError(f"Research checkpoint model ABI manifest mismatch: {checkpoint_dir}")
    if payload_optimizer_abi != manifest.get("optimizer_abi"):
        raise ValueError(f"Research checkpoint optimizer ABI manifest mismatch: {checkpoint_dir}")
    if ema_target is None:
        if "ema_target" in payload or "ema_target_abi" in manifest:
            raise ValueError(
                "Research checkpoint contains EMA target state but no EMA "
                "target model was supplied."
            )
    else:
        if "ema_target" not in payload:
            raise ValueError(
                "Research checkpoint is missing required EMA target state."
            )
        payload_ema_abi = research_state_abi(payload["ema_target"])
        if payload_ema_abi != manifest.get("ema_target_abi"):
            raise ValueError(
                f"Research checkpoint EMA target ABI manifest mismatch: "
                f"{checkpoint_dir}"
            )

    strict_restore_research_payload(
        payload,
        model,
        optimizer,
        ema_target,
    )
    restored = dict(manifest)
    restored["checkpoint_dir"] = str(checkpoint_dir)
    return restored


def load_research_checkpoint_for_evaluation(
    path: Path,
    *,
    model: nnx.Module,
    ema_target: EmaTargetModel | None = None,
) -> dict[str, Any]:
    """Verify a checkpoint and restore only state used by evaluation.

    Optimizer payloads remain unopened. The enclosing state file is still
    checksum-verified, and the model/EMA trees are ABI-preflighted before any
    in-memory mutation. This keeps a multi-checkpoint sweep read-only and avoids
    transferring unused optimizer moments to the accelerator.
    """

    checkpoint_dir = resolve_research_checkpoint(path)
    manifest = _read_research_manifest(checkpoint_dir)
    state = manifest.get("state")
    if not isinstance(state, dict) or state.get("filename") != "state.npz":
        raise ValueError(f"Invalid state record in {checkpoint_dir / 'manifest.json'}")
    state_path = require_within_workspace(checkpoint_dir / "state.npz")
    if state_path.stat().st_size != int(state.get("size_bytes", -1)):
        raise ValueError(f"Research checkpoint state size mismatch: {state_path}")
    if sha256_file(state_path) != state.get("sha256"):
        raise ValueError(f"Research checkpoint state checksum mismatch: {state_path}")

    contract = manifest.get("resume_contract")
    if _json_sha256(contract) != manifest.get("resume_contract_sha256"):
        raise ValueError(
            f"Research checkpoint resume contract checksum mismatch {checkpoint_dir}"
        )

    required_keys = {"step", "model_trainable", "optimizer_state"}
    if ema_target is not None:
        required_keys.add("ema_target")
    with np.load(state_path, allow_pickle=True) as data:
        observed_keys = set(data.files)
        if observed_keys != required_keys:
            raise ValueError(
                "Research state payload keys differ: "
                f"missing={sorted(required_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - required_keys)}"
            )

        step_array = np.asarray(data["step"])
        if step_array.shape != () or not np.issubdtype(step_array.dtype, np.integer):
            raise ValueError(
                "Research optimizer step must be an integer scalar, got "
                f"shape={step_array.shape}, dtype={step_array.dtype}"
            )
        optimizer_step = int(step_array)
        if optimizer_step != int(manifest.get("optimizer_step", -1)):
            raise ValueError(
                "Research optimizer step mismatch: "
                f"manifest={manifest.get('optimizer_step')}, "
                f"payload={optimizer_step}"
            )

        model_value = data["model_trainable"]
        model_payload = (
            model_value.item()
            if model_value.shape == () and model_value.dtype == object
            else model_value
        )
        ema_payload: Any | None = None
        if ema_target is not None:
            ema_value = data["ema_target"]
            ema_payload = (
                ema_value.item()
                if ema_value.shape == () and ema_value.dtype == object
                else ema_value
            )

    if research_state_abi(model_payload) != manifest.get("model_abi"):
        raise ValueError(f"Research checkpoint model ABI manifest mismatch: {checkpoint_dir}")
    model_state = nnx.state(model, TrainableParam)
    assert_research_state_compatible(
        dict(nnx.to_pure_dict(model_state)),
        model_payload,
        label="model",
    )

    ema_states: dict[str, Any] | None = None
    if ema_target is None:
        if "ema_target_abi" in manifest:
            raise ValueError(
                "Research checkpoint contains EMA target state but no EMA "
                "target model was supplied."
            )
    else:
        if ema_payload is None:
            raise ValueError(
                "Research checkpoint is missing required EMA target state."
            )
        if research_state_abi(ema_payload) != manifest.get("ema_target_abi"):
            raise ValueError(
                f"Research checkpoint EMA target ABI manifest mismatch: {checkpoint_dir}"
            )
        ema_states = ema_checkpoint_state(ema_target)
        assert_research_state_compatible(
            {
                name: dict(nnx.to_pure_dict(component_state))
                for name, component_state in ema_states.items()
            },
            ema_payload,
            label="EMA target",
        )
        assert_ema_encoder_compute_structure(ema_target)

    nnx.replace_by_pure_dict(model_state, model_payload)
    nnx.update(model, model_state)
    if ema_states is not None:
        assert ema_payload is not None
        for name, component_state in ema_states.items():
            nnx.replace_by_pure_dict(
                component_state,
                ema_payload[name],
            )
        nnx.update(
            ema_target.encoder_master,
            ema_states["encoder_master"],
        )
        nnx.update(
            ema_target.jepa_state_norm,
            ema_states["jepa_state_norm"],
        )
        nnx.update(
            ema_target.state_projector,
            ema_states["state_projector"],
        )
        refresh_ema_encoder_compute(ema_target)
        validate_ema_encoder_compute(ema_target)

    restored = dict(manifest)
    restored["checkpoint_dir"] = str(checkpoint_dir)
    return restored


def serialized_model_config(
    config: JointLatentSASAConfig,
) -> dict[str, Any]:
    """Serialize config without changing disabled-ablation contracts."""

    payload = dataclasses.asdict(config)
    if config.jepa_target_variance_hinge_coeff == 0.0:
        payload.pop("jepa_target_variance_hinge_coeff")
        payload.pop("jepa_target_variance_hinge_gamma")
    if not config.jepa_state_fixed_unit_rms:
        payload.pop("jepa_state_fixed_unit_rms")
    if not config.jepa_sampled_target_anchors:
        payload.pop("jepa_sampled_target_anchors")
    if (
        config.jepa_target_sampling_unit
        == JointLatentSASAConfig.jepa_target_sampling_unit
    ):
        payload.pop("jepa_target_sampling_unit")
    if (
        config.jepa_rollout_mode
        == JointLatentSASAConfig.jepa_rollout_mode
    ):
        payload.pop("jepa_rollout_mode")
    if (
        config.jepa_feedback_mode
        == JointLatentSASAConfig.jepa_feedback_mode
    ):
        payload.pop("jepa_feedback_mode")
    if config.jepa_projector_active_layers == 0:
        payload.pop("jepa_projector_active_layers")
    if config.dfm_active_layers == 0:
        payload.pop("dfm_active_layers")
    if config.bt4_policy_distill_coeff == 0.0:
        payload.pop("bt4_policy_distill_coeff")
    if config.root_legal_conditional_ce_coeff == 0.0:
        payload.pop("root_legal_conditional_ce_coeff")
    if config.dfm_first_action_loss_share == 0.0:
        payload.pop("dfm_first_action_loss_share")
    if not config.dfm_force_first_action_mask:
        payload.pop("dfm_force_first_action_mask")
    if config.dfm_training_time_power == 1.0:
        payload.pop("dfm_training_time_power")
    if not config.bt4_freeze_backbone:
        payload.pop("bt4_freeze_backbone")
    if not config.bt4_future_target_stop_gradient:
        payload.pop("bt4_future_target_stop_gradient")
    if config.bt4_future_target_trainable_tail_layers == 0:
        payload.pop("bt4_future_target_trainable_tail_layers")
    return payload


def build_research_resume_contract(
    *,
    config: Any,
    objective: str,
    sigreg_reference_count: float,
    batch_size: int,
    train_seed: int,
    train_provenance: dict[str, Any],
    models_dir: Path,
) -> dict[str, Any]:
    """Build the semantic contract that must remain fixed for exact continuation."""

    models_dir = require_within_workspace(models_dir)
    exported_model = require_within_workspace(models_dir / "BT4_exported.pb.gz")
    assets = load_asset_manifest()
    trajectory_asset = assets["trajectory_v3"]["archive"]
    source_files = (
        Path(__file__),
        REPO_ROOT / "research" / "prepare.py",
        REPO_ROOT / "chess_dfm_jax" / "nnx_bt4.py",
        REPO_ROOT / "chess_dfm_jax" / "data" / "trajectory_v3.py",
    )
    if config.jepa_feedback_mode == "final_pass_adjoint":
        source_files = source_files + (
            REPO_ROOT / "research" / "inference.py",
        )
    if config.bt4_policy_distill_coeff != 0.0:
        source_files = source_files + (
            REPO_ROOT / "chess_dfm_jax" / "policy.py",
        )
    objective_contract = {
        "name": objective,
        "jepa_target_semantics": config.jepa_target_semantics,
        "jepa_target_stop_gradient": bool(
            config.jepa_target_stop_gradient
        ),
        "jepa_norm_loss_coeff": float(config.jepa_norm_loss_coeff),
        "jepa_sigreg_estimator": config.jepa_sigreg_estimator,
        "jepa_sigreg_example_count": int(
            config.jepa_sigreg_example_count
        ),
        "target_sigreg_coeff": float(config.jepa_sigreg_coeff),
        "pred_sigreg_coeff": float(config.jepa_pred_sigreg_coeff),
        "target_sigreg_reference_count": float(sigreg_reference_count),
        "pred_sigreg_reference_count": float(sigreg_reference_count),
    }
    if config.jepa_rollout_mode == "direct_sequence":
        objective_contract["jepa_prediction_graph"] = {
            "mode": "direct_sequence",
            "current_state_source": "same_normalized_z0_for_every_horizon",
            "condition": (
                "jepa_action_embed(action_h)+"
                "jepa_hidden_adapter(clean_full_sequence_dfm_hidden_h)"
            ),
            "horizon_execution": "one_parallel_batch_horizon_tensor",
            "recurrent_prediction_carry": False,
            "full_action_sequence_visible_through_dfm_hidden": True,
            "parameter_state_abi": "unchanged",
            "optimizer_state_abi": "unchanged_source_compatible",
            "prediction_sigreg_horizons": "all",
            "collapse_diagnostics_dispatch": "same_configured_graph",
            "dfm_inference_affected": False,
        }
    if config.jepa_feedback_mode == "final_pass_adjoint":
        objective_contract["jepa_closed_loop_feedback"] = {
            "mode": "final_pass_adjoint",
            "proposal_source": (
                "legal_masked_preliminary_dfm_root_argmax_or_revealed_token"
            ),
            "teacher_action_visible_when_masked": False,
            "proposal_index_gradient": "stopped",
            "proposal_confidence_gradient": "stopped",
            "invalid_training_legality_feedback_gate": 0.0,
            "jepa_steps_per_feedback": 1,
            "feedback_latent": "normalized_z1_minus_normalized_z0",
            "bridge": "transpose(jepa_hidden_adapter.w)",
            "bridge_bias": "unused",
            "variance_correction": (
                "sqrt(token_dim/jepa_condition_dim)"
            ),
            "maximum_feedback_to_state_rms_ratio": (
                JEPA_FEEDBACK_MAX_STATE_RMS_RATIO
            ),
            "feedback_broadcast_tokens": 64,
            "training_noisy_planner_calls": 2,
            "training_dfm_ce_source": "feedback_conditioned_logits_only",
            "preliminary_dfm_ce": "detached_diagnostic_only",
            "inference_dfm_planner_calls": 8,
            "inference_feedback_after_pass": 7,
            "inference_feedback_applied_to_pass": 8,
            "inference_additional_bt4_encodes": 0,
            "inference_additional_jepa_projector_calls": 1,
            "inference_additional_jepa_transition_steps": 1,
            "model_state_abi": "unchanged",
            "optimizer_state_abi": "unchanged_source_compatible",
        }
    if config.bt4_policy_distill_coeff != 0.0:
        objective_contract["bt4_root_policy_distillation"] = {
            "coefficient": float(
                config.bt4_policy_distill_coeff
            ),
            "temperature": BT4_POLICY_DISTILL_TEMPERATURE,
            "teacher_head": "frozen_source_bt4_policy_head",
            "teacher_token_source": "online_current_bt4_tokens",
            "teacher_exact_raw_bt4_at_initialization_only": True,
            "teacher_token_gradient": "exact_zero",
            "teacher_logit_gradient": "exact_zero",
            "teacher_action_codec": "lc0_canonical_1858",
            "student_action_codec": "legacy_absolute_1858",
            "side_to_move_source": (
                "classical_112_auxiliary_plane_108"
            ),
            "canonical_to_legacy_mapping": (
                "white_identity_black_vertical_square_mirror"
            ),
            "invalid_mapping_sentinel": -1,
            "support": "stored_representable_root_legal_indices",
            "student_distribution": "legal_masked_root_softmax",
            "loss": "teacher_to_student_kl",
            "eligibility": (
                "valid*root_masked*legal_metadata_valid*"
                "uniform_binary_side_plane*complete_mapping*finite_teacher"
            ),
            "additional_bt4_encoder_calls": 0,
            "additional_training_policy_head_calls": 1,
            "inference_policy_head_calls": 0,
            "model_state_abi": "unchanged",
            "optimizer_state_abi": "unchanged_source_compatible",
            "checkpoint_state_abi": "unchanged",
        }
    if config.root_legal_conditional_ce_coeff != 0.0:
        objective_contract["root_legal_conditional_ce"] = {
            "coefficient": float(
                config.root_legal_conditional_ce_coeff
            ),
            "scope": "masked_played_root_action_only",
            "support": "stored_representable_root_legal_indices",
            "loss": (
                "logsumexp(stored_legal_logits)-played_action_logit"
            ),
            "reduction_dtype": "float32",
            "normalization": "eligible_rows",
            "eligibility": (
                "valid*root_masked*legal_metadata_valid*"
                "valid_count_and_indices*played_in_legal*finite_logits"
            ),
            "uniform_full_vocabulary_dfm_ce_unchanged": True,
            "first_legality_objective_unchanged": True,
            "direct_illegal_logit_gradient": "exact_zero",
            "legal_logit_gradient_sum": "exact_zero",
            "additional_encoder_calls": 0,
            "additional_planner_calls": 0,
            "additional_rng_draws": 0,
            "inference_affected": False,
            "model_state_abi": "unchanged",
            "optimizer_state_abi": "unchanged_source_compatible",
            "checkpoint_state_abi": "unchanged",
        }
    if config.dfm_first_action_loss_share != 0.0:
        tail_horizon_count = (
            config.horizon
            if config.loss_horizon <= 0
            else min(config.loss_horizon, config.horizon)
        ) - 1
        objective_contract["dfm_horizon_weighting"] = {
            "uniform_dfm_ce_reporting_unchanged": True,
            "training_objective_metric": "dfm_objective_ce_loss",
            "played_first_action_share": float(
                config.dfm_first_action_loss_share
            ),
            "remaining_share": float(
                1.0 - config.dfm_first_action_loss_share
            ),
            "remaining_active_horizon_count": int(tail_horizon_count),
            "each_remaining_horizon_share": float(
                (1.0 - config.dfm_first_action_loss_share)
                / tail_horizon_count
            ),
            "first_legality_coeff": float(
                config.first_legality_coeff
            ),
            "legality_to_first_action_ce_coefficient_ratio": float(
                config.first_legality_coeff
                / config.dfm_first_action_loss_share
            ),
            "selection_metric": "two_pool_dfm_ce_loss_by_horizon_h1",
            "inference_affected": False,
        }
    if config.dfm_force_first_action_mask:
        objective_contract["dfm_first_action_masking"] = {
            "scope": "training_only",
            "played_first_action_masked": True,
            "played_first_action_mask_token": int(
                config.action_vocab_size
            ),
            "remaining_action_masks": "unchanged_existing_rng_draws",
            "time_sampling": "unchanged",
            "target_actions": "unchanged",
            "uniform_dfm_ce_weighting": True,
            "first_legality_coeff": float(
                config.first_legality_coeff
            ),
            "evaluation_affected": False,
            "inference_affected": False,
            "model_state_abi": "unchanged",
            "optimizer_state_abi": "unchanged_source_compatible",
        }
    if config.dfm_training_time_power != 1.0:
        objective_contract["dfm_training_time_distribution"] = {
            "scope": "training_only",
            "base_draw": "u~Uniform(0,1)",
            "transform": "t=u**power",
            "power": float(config.dfm_training_time_power),
            "expected_time": float(
                1.0 / (config.dfm_training_time_power + 1.0)
            ),
            "expected_mask_probability": float(
                config.dfm_training_time_power
                / (config.dfm_training_time_power + 1.0)
            ),
            "rng_draws": "unchanged",
            "deterministic_validation_time": "unchanged",
            "evaluation_affected": False,
            "inference_affected": False,
            "model_state_abi": "unchanged",
            "optimizer_state_abi": "unchanged_source_compatible",
        }
    if config.jepa_sigreg_example_count > 0:
        objective_contract["jepa_sigreg_sampling"] = {
            "unit": "physical_batch_example",
            "without_replacement": True,
            "shared_target_prediction_indices": True,
            "resampled_each_update": True,
            "projection_rng_split_after_selection": True,
            "full_batch_fast_path_when_equal": True,
        }
    if 0 < config.jepa_target_sample_count < config.horizon:
        if config.jepa_target_sampling_unit == "example_balanced":
            objective_contract["future_target_sampling"] = {
                "scope": "training_only",
                "sample_count": int(config.jepa_target_sample_count),
                "per_example_target_count": int(
                    config.jepa_target_sample_count
                ),
                "population_horizons": int(config.horizon),
                "unit": "one_balanced_horizon_per_physical_example",
                "assignment_construction": (
                    "random_permutation_of_tiled_horizon_labels"
                ),
                "maximum_assignment_count_spread": 1,
                "resampled_each_update": True,
                "rng_derivation": "split_from_existing_sigreg_rng_branch",
                "dfm_time_and_mask_rng_unchanged": True,
                "positive_jepa_horizons": "per_example_assigned",
                "prediction_sigreg_horizons": "all",
                "target_sigreg_future_importance_weight": float(
                    config.horizon / config.jepa_target_sample_count
                ),
                "target_sigreg_current_importance_weight": 1.0,
                "evaluation_horizons": "all",
            }
        else:
            objective_contract["future_target_sampling"] = {
                "scope": "training_only",
                "sample_count": int(config.jepa_target_sample_count),
                "population_horizons": int(config.horizon),
                "unit": "one_shared_horizon_subset_per_physical_batch",
                "without_replacement": True,
                "sorted_after_selection": True,
                "resampled_each_update": True,
                "rng_derivation": "split_from_existing_sigreg_rng_branch",
                "dfm_time_and_mask_rng_unchanged": True,
                "positive_jepa_horizons": "sampled",
                "prediction_sigreg_horizons": "all",
                "target_sigreg_future_importance_weight": float(
                    config.horizon / config.jepa_target_sample_count
                ),
                "target_sigreg_current_importance_weight": 1.0,
                "evaluation_horizons": "all",
            }
    if config.jepa_sampled_target_anchors:
        objective_contract["sampled_target_rollout_anchors"] = {
            "scope": "training_only",
            "source": "same_sampled_online_future_targets",
            "selection_rng": "no_additional_rng",
            "timing": "after_prediction_h_before_transition_h_plus_1",
            "prediction_at_anchor_horizon": "free_pre_anchor_output",
            "positive_jepa_predictions": "pre_anchor",
            "prediction_sigreg_predictions": "all_pre_anchor_outputs",
            "carry_replacement_validity": "valid*future_valid",
            "target_gradient": "attached_online",
            "final_horizon_anchor": "inert_no_downstream_transition",
            "evaluation_rollout": "fully_free_running",
            "inference_future_latents": False,
        }
    if config.jepa_projector_active_layers > 0:
        objective_contract["jepa_projector_active_depth"] = {
            "configured_parameter_layers": int(config.projector_layers),
            "active_forward_layers": int(
                config.jepa_projector_active_layers
            ),
            "active_layer_indices": list(
                range(config.jepa_projector_active_layers)
            ),
            "inactive_layer_role": "checkpoint_abi_only",
            "inactive_layer_loss_gradient": "exact_zero",
            "source_model_restore": "exact_all_parameters",
            "training_evaluation_depth_shared": True,
            "checkpoint_storage_packed": False,
            "dfm_inference_path_affected": False,
        }
    if config.dfm_active_layers > 0:
        objective_contract["dfm_active_depth"] = {
            "configured_parameter_layers": int(config.dfm_layers),
            "active_forward_layers": int(config.dfm_active_layers),
            "active_layer_indices": list(
                range(config.dfm_active_layers)
            ),
            "inactive_layer_role": "checkpoint_abi_only",
            "inactive_layer_loss_gradient": "exact_zero",
            "source_model_restore": "exact_all_parameters",
            "training_evaluation_inference_depth_shared": True,
            "checkpoint_storage_packed": False,
            "dfm_inference_path_affected": True,
            "refinement_passes_changed": False,
        }
    if config.jepa_target_variance_hinge_coeff != 0.0:
        objective_contract["jepa_target_variance_hinge"] = {
            "coefficient": float(
                config.jepa_target_variance_hinge_coeff
            ),
            "gamma": float(config.jepa_target_variance_hinge_gamma),
            "epsilon": TARGET_VARIANCE_HINGE_EPSILON,
            "source": "attached_online_projected_future_targets",
            "sample_axis": "batch_independently_per_future_horizon",
            "variance_denominator": "weighted_population_count",
            "validity": "valid*future_valid",
            "minimum_valid_count_per_horizon": 2.0,
            "horizon_reduction": "equal_mean_over_eligible_horizons",
        }
    if config.jepa_state_fixed_unit_rms:
        objective_contract["jepa_state_manifold"] = {
            "kind": "fixed_unit_rms",
            "formula": "z/sqrt(mean(z^2)+epsilon)",
            "epsilon": JEPA_STATE_RMS_EPSILON,
            "statistics_dtype": "float32",
            "output_dtype": config.compute_dtype,
            "application_scope": [
                "projected_current_state",
                "projected_future_targets",
                "each_recurrent_jepa_prediction",
            ],
            "stored_scale_parameter": (
                "retained_for_checkpoint_abi_only"
            ),
            "stored_scale_parameter_read_by_objective": False,
            "stored_scale_parameter_loss_gradient": "exact_zero",
        }
    if config.jepa_target_semantics == "ema":
        objective_contract["jepa_target_ema"] = {
            "decay": float(config.jepa_target_ema_decay),
            "effective_decay_float32": float(
                np.float32(config.jepa_target_ema_decay)
            ),
            "initialization": "exact_copy_after_source_restore",
            "checkpointed_master_scope": [
                "encoder_master",
                "state_projector",
                "jepa_state_norm",
            ],
            "derived_runtime_scope": ["encoder_compute"],
            "floating_master_storage_dtype": "float32",
            "encoder_compute_storage_dtype": "online_native",
            "encoder_forward_source": "encoder_compute_only",
            "encoder_compute_checkpointed": False,
            "encoder_compute_refresh": [
                "after_source_sync",
                "after_each_post_optimizer_ema_update",
                "after_checkpoint_restore",
            ],
            "encoder_compute_equation": (
                "encoder_compute=cast_online_native(encoder_master)"
            ),
            "positive_target_gradient": "always_stopped",
            "online_target_sigreg_gradient": "attached",
            "update_order": "after_online_optimizer",
            "update_equation": (
                "target=decay*target+(1-decay)*online_updated"
            ),
        }
    optimizer_contract: dict[str, Any] = {
        "learning_rate_schedule": learning_rate_schedule_contract(
            config
        ),
    }
    if config.bt4_freeze_backbone:
        optimizer_contract["bt4_backbone_freeze"] = {
            "model_state_abi": "source_compatible_trainable_encoder_leaves",
            "forward_values": "unchanged",
            "gradient_boundary": "stop_gradient_after_bt4_tokens",
            "optimizer_partition": "frozen_bt4_set_to_zero",
            "optimizer_state_for_bt4": "none",
            "bt4_learning_rate": float(config.bt4_learning_rate),
            "legacy_initialization": (
                "model_only_after_compatibility_optimizer_validation"
            ),
            "legacy_exact_optimizer_restore": False,
            "research_checkpoint_resume": "exact_frozen_optimizer_abi",
            "evaluation_and_inference_values_affected": False,
        }
    if config.bt4_future_target_stop_gradient:
        trainable_tail_layers = int(
            config.bt4_future_target_trainable_tail_layers
        )
        detached_prefix_layers = 15 - trainable_tail_layers
        objective_contract["bt4_future_target_stop_gradient"] = {
            "scope": "training_gradient_routing_only",
            "encoder_calls_per_step": 2,
            "trainable_current_boards_per_example": 1,
            "stop_gradient_future_boards_per_example": int(
                config.jepa_target_sample_count
                if trainable_tail_layers == 0
                else 0
            ),
            "partial_gradient_future_boards_per_example": int(
                config.jepa_target_sample_count
                if trainable_tail_layers > 0
                else 0
            ),
            "current_bt4_token_gradient": "attached",
            "future_bt4_token_gradient": (
                "attached_through_tail"
                if trainable_tail_layers > 0
                else "exact_zero"
            ),
            "future_embedding_gradient": "exact_zero",
            "future_detached_prefix_layers": detached_prefix_layers,
            "future_trainable_tail_layers": trainable_tail_layers,
            "future_prefix_block_gradient": "exact_zero",
            "future_tail_block_gradient": (
                "attached"
                if trainable_tail_layers > 0
                else "not_applicable"
            ),
            "future_state_projector_gradient": "attached",
            "shared_state_projector": True,
            "model_state_abi": "unchanged",
            "optimizer_state_abi": "unchanged_source_compatible",
            "bt4_optimizer_partition": "trainable",
            "evaluation_forward_values_affected": False,
            "evaluation_horizons": "all",
            "inference_affected": False,
        }
    return {
        "architecture_source": ARCHITECTURE_SOURCE,
        "model_config": serialized_model_config(config),
        "objective": objective_contract,
        "optimizer": optimizer_contract,
        "data": {
            "batch_size": int(batch_size),
            "horizon": int(config.horizon),
            "view": "joint_latent_sasa",
            "train_seed": int(train_seed),
            "schedule": train_provenance,
            "source_archive_sha256": trajectory_asset["sha256"],
            "source_archive_size_bytes": int(trajectory_asset["size_bytes"]),
        },
        "assets": {
            "bt4_exported_path": str(exported_model),
            "bt4_exported_sha256": sha256_file(exported_model),
        },
        "code": {
            "files": {
                str(path.relative_to(REPO_ROOT)): sha256_file(path)
                for path in source_files
            },
        },
        "software": {
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "jax": jax.__version__,
            "flax": importlib.metadata.version("flax"),
            "numpy": np.__version__,
            "optax": importlib.metadata.version("optax"),
            "device_platform": jax.devices()[0].platform,
            "device_kind": jax.devices()[0].device_kind,
        },
    }


def normalized_le_jepa_sigreg(
    z: jax.Array,
    *,
    proj_dim: int,
    rng: jax.Array,
    reference_count: float = 1.0,
    sample_weight: jax.Array | None = None,
    t_max: float = 3.0,
    n_points: int = 17,
    estimator: str = "v_stat",
) -> SigRegResult:
    """Fixed-reference LeJEPA ECF discrepancy.

    ``official`` preserves the published Epps-Pulley statistic
    ``valid_count * discrepancy`` for diagnostics. ``normalized`` replaces the
    variable valid count with a fixed reference count.

    The default V-statistic is invariant to exact duplication and zero padding,
    but retains a finite-sample self-pair bias. ``estimator="u_stat"`` removes
    that diagonal term for independently sampled batches; it deliberately is
    not invariant to duplicating the same observations.

    This function operates on one physical batch. Epps-Pulley is non-additive,
    so callers must not average independently evaluated microbatch results.
    """

    if proj_dim < 1:
        raise ValueError(f"proj_dim must be positive, got {proj_dim}")
    if reference_count <= 0:
        raise ValueError(f"reference_count must be positive, got {reference_count}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if estimator not in ("v_stat", "u_stat"):
        raise ValueError(
            "estimator must be 'v_stat' or 'u_stat', found "
            f"{estimator!r}"
        )

    statistics = _le_jepa_sigreg_statistics(
        z,
        proj_dim=proj_dim,
        rng=rng,
        sample_weight=sample_weight,
        t_max=t_max,
        n_points=n_points,
        compute_u_stat=True,
    )
    selected = (
        statistics.v_stat_discrepancy
        if estimator == "v_stat"
        else statistics.u_stat_discrepancy
    )
    normalized = selected * jnp.asarray(
        reference_count,
        dtype=jnp.float32,
    )
    return SigRegResult(
        normalized,
        statistics.official,
        statistics.valid_count,
        statistics.v_stat_discrepancy,
        statistics.u_stat_discrepancy,
    )


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
    ema_target: EmaTargetModel | None = None,
) -> dict[str, jax.Array]:
    """Authoritative free-rollout collapse and action-dependence diagnostics."""

    actions = batch["action_indices"][:, : model.config.horizon]
    future_planes = batch["future_planes"][:, : model.config.horizon]
    all_tokens, z_all = model.encode_current_and_future_tokens_and_vectors(
        batch["current_planes"],
        future_planes,
    )
    positive_target_z = (
        z_all[:, 1:]
        if ema_target is None
        else jax.lax.stop_gradient(
            ema_target.encode_future_targets(future_planes)
        )
    )
    z_dfm = model.dfm_latents(all_tokens[:, 0])
    clean_t = jnp.ones((actions.shape[0],), dtype=jnp.float32)
    _, clean_hidden = model.planner_from_latents(
        z_dfm,
        actions,
        clean_t,
        return_hidden=True,
    )
    pred_z = model.jepa_predictions_from_latents(
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
        positive_target_z,
        z_all[:, 0],
        valid,
        rng=rng_baseline,
    )

    permutation = jax.random.permutation(rng_action, actions.shape[0])
    action_shuffled_pred = model.jepa_predictions_from_latents(
        z_all[:, 0],
        actions[permutation],
        clean_hidden["action_tokens"][permutation],
        z0_normalized=True,
    )
    action_shuffled_mse = jnp.mean(
        jnp.square(
            jnp.asarray(action_shuffled_pred, dtype=jnp.float32)
            - jnp.asarray(positive_target_z, dtype=jnp.float32)
        ),
        axis=-1,
    )
    denom = jnp.maximum(jnp.sum(valid, axis=0), 1.0)
    metrics["action_shuffled_mse_by_horizon"] = (
        jnp.sum(action_shuffled_mse * valid, axis=0) / denom
    )
    if ema_target is not None:
        online_ema_mse = jnp.mean(
            jnp.square(
                jnp.asarray(z_all[:, 1:], dtype=jnp.float32)
                - jnp.asarray(positive_target_z, dtype=jnp.float32)
            ),
            axis=-1,
        )
        metrics["online_ema_target_mse_by_horizon"] = (
            jnp.sum(online_ema_mse * valid, axis=0) / denom
        )
    return metrics


_compat_stage1_loss_and_grad = nnx.value_and_grad(
    joint_stage1_loss_fn,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)


def _compat_train_step_impl(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
):
    with jax.named_scope("joint_stage1_loss_and_grad"):
        (loss, aux), grads = _compat_stage1_loss_and_grad(
            model,
            batch,
            rng,
        )
    with jax.named_scope("joint_stage1_optimizer_update"):
        optimizer.update(model, grads)
    return loss, aux


@nnx.jit
def eval_joint_stage1_step(
    model: JointLatentSASAModel,
    batch: dict[str, jax.Array],
    rng: jax.Array,
):
    return joint_stage1_loss_fn(model, batch, rng)


@nnx.jit
def train_joint_stage1_step(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
):
    return _compat_train_step_impl(model, optimizer, batch, rng)


@nnx.jit(donate_argnums=(0, 1))
def train_joint_stage1_step_donated(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
):
    return _compat_train_step_impl(model, optimizer, batch, rng)


def normalized_stage1_loss_fn(
    model,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
    positive_target_override: jax.Array | None = None,
    sample_future_targets: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compatibility loss with finite-sample SIGReg and legal corrections.

    The V-statistic path converts the compatibility graph's official EP
    statistic to a fixed-reference discrepancy algebraically. The U-statistic
    path additionally removes the diagonal empirical-ECF self-pair term.
    """

    _, compatibility_aux = joint_stage1_loss_fn(
        model,
        batch,
        rng,
        compute_fp32_legality=True,
        positive_target_override=positive_target_override,
        sample_future_targets=sample_future_targets,
    )
    target_official = jnp.asarray(
        compatibility_aux["jepa_sigreg_loss"],
        dtype=jnp.float32,
    )
    pred_official = jnp.asarray(
        compatibility_aux["jepa_pred_sigreg_loss"],
        dtype=jnp.float32,
    )
    target_count = jnp.asarray(
        compatibility_aux["jepa_sigreg_valid_count"],
        dtype=jnp.float32,
    )
    pred_count = jnp.asarray(
        compatibility_aux["jepa_pred_sigreg_valid_count"],
        dtype=jnp.float32,
    )

    target_v_stat = target_official * (
        jnp.asarray(target_reference_count, dtype=jnp.float32) / jnp.maximum(target_count, 1.0)
    )
    pred_v_stat = pred_official * (
        jnp.asarray(pred_reference_count, dtype=jnp.float32) / jnp.maximum(pred_count, 1.0)
    )
    if model.config.jepa_sigreg_estimator == "u_stat":
        target_normalized = jnp.asarray(
            compatibility_aux["jepa_sigreg_u_stat_discrepancy"],
            dtype=jnp.float32,
        ) * jnp.asarray(target_reference_count, dtype=jnp.float32)
        pred_normalized = jnp.asarray(
            compatibility_aux[
                "jepa_pred_sigreg_u_stat_discrepancy"
            ],
            dtype=jnp.float32,
        ) * jnp.asarray(pred_reference_count, dtype=jnp.float32)
    else:
        target_normalized = target_v_stat
        pred_normalized = pred_v_stat

    corrected_first_legality = jnp.asarray(
        compatibility_aux["first_legality_loss_fp32"],
        dtype=jnp.float32,
    )
    corrected_legal_mass = 1.0 - corrected_first_legality
    corrected_weighted_legality = model.config.first_legality_coeff * corrected_first_legality

    unclipped = jnp.asarray(
        compatibility_aux["unclipped_loss"],
        dtype=jnp.float32,
    )
    unclipped = (
        unclipped
        - jnp.asarray(
            compatibility_aux["weighted_legality_loss"],
            dtype=jnp.float32,
        )
        - model.config.jepa_sigreg_coeff * target_official
        - model.config.jepa_pred_sigreg_coeff * pred_official
        + corrected_weighted_legality
        + model.config.jepa_sigreg_coeff * target_normalized
        + model.config.jepa_pred_sigreg_coeff * pred_normalized
    )
    loss, clip_scale = _clip_loss_preserve_gradient(unclipped, model.config.loss_clip_value)

    aux = dict(compatibility_aux)
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
    if model.config.jepa_sigreg_estimator == "u_stat":
        aux.update(
            {
                "jepa_sigreg_v_stat_loss": target_v_stat,
                "jepa_pred_sigreg_v_stat_loss": pred_v_stat,
                "jepa_sigreg_u_stat_loss": target_normalized,
                "jepa_pred_sigreg_u_stat_loss": pred_normalized,
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
        sample_future_targets=True,
    )
    components = [
        jnp.asarray(
            aux.get("dfm_objective_ce_loss", aux["dfm_ce_loss"]),
            dtype=jnp.float32,
        ),
        jnp.asarray(aux["jepa_positive_loss"], dtype=jnp.float32),
        jnp.asarray(aux["jepa_sigreg_loss"], dtype=jnp.float32),
        jnp.asarray(aux["jepa_pred_sigreg_loss"], dtype=jnp.float32),
        jnp.asarray(aux["first_legality_loss"], dtype=jnp.float32),
    ]
    if model.config.jepa_target_variance_hinge_coeff != 0.0:
        components.append(
            jnp.asarray(
                aux["jepa_target_variance_hinge_loss"],
                dtype=jnp.float32,
            )
        )
    if model.config.wdl_coeff != 0.0:
        components.append(
            jnp.asarray(aux["wdl_loss"], dtype=jnp.float32)
        )
    if model.config.bt4_policy_distill_coeff != 0.0:
        components.append(
            jnp.asarray(
                aux["bt4_policy_distill_loss"],
                dtype=jnp.float32,
            )
        )
    if model.config.root_legal_conditional_ce_coeff != 0.0:
        components.append(
            jnp.asarray(
                aux["root_legal_conditional_ce_loss"],
                dtype=jnp.float32,
            )
        )
    return jnp.stack(components)


def gradient_component_names(
    config: JointLatentSASAConfig,
) -> tuple[str, ...]:
    """Return the audit ABI, adding enabled-only research components."""

    enabled_components: tuple[str, ...] = ()
    if config.jepa_target_variance_hinge_coeff != 0.0:
        enabled_components += (
            TARGET_VARIANCE_HINGE_COMPONENT,
        )
    if config.wdl_coeff != 0.0:
        enabled_components += (WDL_COMPONENT,)
    if config.bt4_policy_distill_coeff != 0.0:
        enabled_components += (BT4_POLICY_DISTILL_COMPONENT,)
    if config.root_legal_conditional_ce_coeff != 0.0:
        enabled_components += (ROOT_LEGAL_CONDITIONAL_COMPONENT,)
    return GRADIENT_COMPONENT_NAMES + enabled_components


def normalized_stage1_training_loss_fn(
    model,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Training-only normalized objective with configured target sampling."""

    return normalized_stage1_loss_fn(
        model,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
        sample_future_targets=True,
    )


_normalized_loss_and_grad = nnx.value_and_grad(
    normalized_stage1_training_loss_fn,
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


def ema_normalized_stage1_loss_fn(
    model: JointLatentSASAModel,
    ema_target: EmaTargetModel,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Normalized objective against a detached pre-update EMA future target."""

    ema_future_target = jax.lax.stop_gradient(
        ema_target.encode_future_targets(
            batch["future_planes"][:, : model.config.horizon]
        )
    )
    return normalized_stage1_loss_fn(
        model,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
        positive_target_override=ema_future_target,
    )


_ema_normalized_loss_and_grad = nnx.value_and_grad(
    ema_normalized_stage1_loss_fn,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)


def _ema_normalized_train_step_impl(
    model: JointLatentSASAModel,
    ema_target: EmaTargetModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    with jax.named_scope("research_ema_target_loss_and_grad"):
        (loss, aux), grads = _ema_normalized_loss_and_grad(
            model,
            ema_target,
            batch,
            rng,
            target_reference_count,
            pred_reference_count,
        )
    with jax.named_scope("research_ema_online_optimizer_update"):
        optimizer.update(model, grads)
    with jax.named_scope("research_ema_target_post_optimizer_update"):
        update_ema_target_after_optimizer(
            ema_target,
            model,
            model.config.jepa_target_ema_decay,
        )
    return loss, aux


@nnx.jit
def eval_ema_normalized_stage1_step(
    model: JointLatentSASAModel,
    ema_target: EmaTargetModel,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    return ema_normalized_stage1_loss_fn(
        model,
        ema_target,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
    )


@nnx.jit
def train_ema_normalized_stage1_step(
    model: JointLatentSASAModel,
    ema_target: EmaTargetModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    return _ema_normalized_train_step_impl(
        model,
        ema_target,
        optimizer,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
    )


@nnx.jit(donate_argnums=(0, 1, 2))
def train_ema_normalized_stage1_step_donated(
    model: JointLatentSASAModel,
    ema_target: EmaTargetModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    rng: jax.Array,
    target_reference_count: float,
    pred_reference_count: float,
):
    return _ema_normalized_train_step_impl(
        model,
        ema_target,
        optimizer,
        batch,
        rng,
        target_reference_count,
        pred_reference_count,
    )


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Strictly continue a completed local research checkpoint or checkpoint root.",
    )
    parser.add_argument(
        "--eval-checkpoints",
        type=Path,
        help=(
            "Read-only evaluation sweep over one research checkpoint, a "
            "checkpoint root, or a run containing checkpoints/. The "
            "checkpoint contract defines the model and objective."
        ),
    )
    parser.add_argument(
        "--eval-root-legal-conditional-ce",
        action="store_true",
        help=(
            "For read-only --eval-checkpoints only, overlay coefficient "
            "1.0 so legacy checkpoints report the root legal-conditional "
            "CE diagnostic. Model state and forward logits are unchanged; "
            "the reported aggregate loss includes the unit diagnostic."
        ),
    )
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
    parser.add_argument(
        "--sigreg-estimator",
        choices=("v_stat", "u_stat"),
        help=(
            "Finite-sample estimator for normalized LeJEPA SIGReg. "
            "v_stat preserves duplication invariance; u_stat removes "
            "the diagonal self-pair bias."
        ),
    )
    parser.add_argument(
        "--sigreg-example-count",
        type=int,
        help=(
            "Randomly sample this many physical-batch examples for both "
            "target and prediction SIGReg; zero uses the full batch. "
            "Use a fixed positive count to keep estimator scale and cost "
            "portable across physical batches."
        ),
    )
    parser.add_argument(
        "--jepa-norm-loss-coeff",
        type=float,
        help=(
            "Coefficient on the per-state prediction/target log-RMS "
            "mismatch inside the positive JEPA loss. Use 0 only with "
            "--objective normalized to test raw MSE plus SIGReg."
        ),
    )
    parser.add_argument(
        "--jepa-target-stop-gradient",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Detach the projected/current-repeat target from the positive JEPA "
            "loss; omit to preserve the checkpoint config."
        ),
    )
    parser.add_argument(
        "--jepa-target-semantics",
        choices=("online", "ema"),
        help=(
            "Use the online projected future vector or a detached EMA teacher; "
            "omit to preserve the checked experiment config."
        ),
    )
    parser.add_argument(
        "--jepa-target-ema-decay",
        type=float,
        help=(
            "Post-optimizer EMA decay. This is active only with "
            "--jepa-target-semantics ema."
        ),
    )
    parser.add_argument(
        "--target-variance-hinge-coeff",
        type=float,
        help=(
            "Coefficient for the attached, per-future-horizon target "
            "feature-variance floor."
        ),
    )
    parser.add_argument(
        "--target-variance-hinge-gamma",
        type=float,
        help=(
            "Target feature standard-deviation floor. Active only when "
            "--target-variance-hinge-coeff is nonzero."
        ),
    )
    parser.add_argument(
        "--jepa-state-fixed-unit-rms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Project every JEPA state onto a fixed unit-RMS manifold "
            "using FP32 statistics and no trainable scale."
        ),
    )
    parser.add_argument("--sigreg-reference-count", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help=(
            "Override the restored main-model learning rate. Omit to preserve "
            "the source checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--bt4-learning-rate",
        type=float,
        help=(
            "Override the restored BT4 learning rate. Omit to preserve the "
            "source checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--train-batch-schedule",
        choices=("shard_major", "global_permutation"),
        default="shard_major",
        help=(
            "Use cache-friendly consecutive batches within shuffled shards, "
            "or globally permute all shard/batch slots each epoch."
        ),
    )
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
        "--eval-checkpoint-seeds",
        type=int,
        nargs="+",
        help=(
            "Validation seeds for --eval-checkpoints. Each seed fixes both "
            "the global validation positions and objective RNG; defaults to "
            "--val-seed."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help=(
            "Validation batch size. Omit to match --batch-size; set it "
            "explicitly to keep held-out positions fixed across training "
            "batch-size experiments."
        ),
    )
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
        "--compile-only",
        action="store_true",
        help=(
            "Populate the persistent cache from constructor-equivalent model "
            "and optimizer shapes, then exit before opening the legacy "
            "checkpoint. Requires --compile-ahead, --init model-only, "
            "--steps 0, --eval-batches 0, and no checkpoint request."
        ),
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
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save every N updates in this invocation; zero disables periodic saves.",
    )
    parser.add_argument(
        "--save-updates",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Save at these absolute research-update numbers; omit to disable. "
            "This supports sparse preregistered checkpoint schedules."
        ),
    )
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save final model and optimizer state (about 1.85 GB for the baseline).",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=2,
        help="Maximum completed checkpoints retained in this run segment; zero keeps all.",
    )
    return parser.parse_args(argv)


def apply_config_overrides(
    config: JointLatentSASAConfig,
    args: argparse.Namespace,
) -> JointLatentSASAConfig:
    """Apply explicit CLI objective overrides while preserving metadata defaults."""
    return dataclasses.replace(
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
        jepa_sigreg_estimator=(
            config.jepa_sigreg_estimator
            if args.sigreg_estimator is None
            else args.sigreg_estimator
        ),
        jepa_sigreg_example_count=(
            config.jepa_sigreg_example_count
            if args.sigreg_example_count is None
            else args.sigreg_example_count
        ),
        jepa_norm_loss_coeff=(
            config.jepa_norm_loss_coeff
            if args.jepa_norm_loss_coeff is None
            else args.jepa_norm_loss_coeff
        ),
        jepa_target_stop_gradient=(
            config.jepa_target_stop_gradient
            if args.jepa_target_stop_gradient is None
            else args.jepa_target_stop_gradient
        ),
        jepa_target_semantics=(
            config.jepa_target_semantics
            if args.jepa_target_semantics is None
            else args.jepa_target_semantics
        ),
        jepa_target_ema_decay=(
            config.jepa_target_ema_decay
            if args.jepa_target_ema_decay is None
            else args.jepa_target_ema_decay
        ),
        jepa_target_variance_hinge_coeff=(
            config.jepa_target_variance_hinge_coeff
            if args.target_variance_hinge_coeff is None
            else args.target_variance_hinge_coeff
        ),
        jepa_target_variance_hinge_gamma=(
            config.jepa_target_variance_hinge_gamma
            if args.target_variance_hinge_gamma is None
            else args.target_variance_hinge_gamma
        ),
        jepa_state_fixed_unit_rms=(
            config.jepa_state_fixed_unit_rms
            if args.jepa_state_fixed_unit_rms is None
            else args.jepa_state_fixed_unit_rms
        ),
        learning_rate=(
            config.learning_rate
            if args.learning_rate is None
            else args.learning_rate
        ),
        bt4_learning_rate=(
            config.bt4_learning_rate
            if args.bt4_learning_rate is None
            else args.bt4_learning_rate
        ),
    )


def apply_experiment_overrides(
    config: JointLatentSASAConfig,
    overrides: Mapping[str, Any] | None = None,
) -> JointLatentSASAConfig:
    """Apply the checked-in autoresearch edit surface to restored metadata."""

    selected = EXPERIMENT_OVERRIDES if overrides is None else overrides
    if not isinstance(selected, Mapping):
        raise TypeError(
            "EXPERIMENT_OVERRIDES must be a mapping of config field names "
            f"to values, got {type(selected).__name__}"
        )

    config_fields = {
        field.name: field
        for field in dataclasses.fields(JointLatentSASAConfig)
    }
    unknown = sorted(set(selected) - set(config_fields))
    if unknown:
        raise ValueError(
            "Unknown EXPERIMENT_OVERRIDES config field(s): "
            + ", ".join(unknown)
        )

    checked: dict[str, Any] = {}
    for name, value in selected.items():
        default = config_fields[name].default
        expected_type = type(default)
        if expected_type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"EXPERIMENT_OVERRIDES[{name!r}] must be float, "
                    f"got {type(value).__name__}"
                )
            checked[name] = float(value)
            continue
        if type(value) is not expected_type:
            raise TypeError(
                f"EXPERIMENT_OVERRIDES[{name!r}] must be "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )
        checked[name] = value

    return dataclasses.replace(config, **checked)


def validate_no_inert_config_overrides(
    config: JointLatentSASAConfig,
) -> None:
    """Reject legacy knobs that the local stage-1 graph cannot honor."""

    fields = JointLatentSASAConfig.__dataclass_fields__
    non_default = [
        (
            name,
            getattr(config, name),
            fields[name].default,
        )
        for name in _INERT_OR_DEPRECATED_CONFIG_FIELDS
        if getattr(config, name) != fields[name].default
    ]
    if non_default:
        details = ", ".join(
            f"{name}={value!r} (required default {default!r})"
            for name, value, default in non_default
        )
        raise ValueError(
            "Inert/deprecated research config fields cannot be set: "
            + details
        )


def checkpoint_evaluation_diagnostic_overlay(
    config: JointLatentSASAConfig,
    *,
    root_legal_conditional_ce: bool,
) -> tuple[JointLatentSASAConfig, dict[str, Any]]:
    """Add explicitly read-only metrics to a checkpoint-authoritative graph."""

    if not root_legal_conditional_ce:
        return config, {}
    overlay = {
        "root_legal_conditional_ce": {
            "coefficient": 1.0,
            "purpose": "read_only_metric_overlay",
            "model_state_mutation": False,
            "forward_logits_affected": False,
            "reported_aggregate_loss_includes_overlay": True,
        }
    }
    return (
        dataclasses.replace(
            config,
            root_legal_conditional_ce_coeff=1.0,
        ),
        overlay,
    )


def checkpoint_evaluation_contract(
    checkpoints: list[Path],
) -> tuple[JointLatentSASAConfig, str, float, dict[str, Any], str]:
    """Return the homogeneous, checkpoint-authoritative evaluation contract."""

    if not checkpoints:
        raise ValueError("Checkpoint evaluation requires at least one checkpoint")
    manifests = [_read_research_manifest(path) for path in checkpoints]
    contract = manifests[0].get("resume_contract")
    contract_sha256 = manifests[0].get("resume_contract_sha256")
    if not isinstance(contract, dict) or not isinstance(contract_sha256, str):
        raise ValueError(f"Missing research resume contract in {checkpoints[0]}")
    if _json_sha256(contract) != contract_sha256:
        raise ValueError(
            f"Research checkpoint resume contract checksum mismatch {checkpoints[0]}"
        )
    for path, manifest in zip(checkpoints[1:], manifests[1:], strict=True):
        incoming = manifest.get("resume_contract")
        incoming_sha256 = manifest.get("resume_contract_sha256")
        if not isinstance(incoming, dict) or _json_sha256(incoming) != incoming_sha256:
            raise ValueError(
                f"Research checkpoint resume contract checksum mismatch {path}"
            )
        if incoming_sha256 != contract_sha256:
            raise ValueError(
                "Checkpoint evaluation root mixes resume contracts: "
                f"{checkpoints[0].name}={contract_sha256}, "
                f"{path.name}={incoming_sha256}"
            )

    raw_config = contract.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint evaluation contract is missing model_config")
    fields = {field.name for field in dataclasses.fields(JointLatentSASAConfig)}
    unknown = sorted(set(raw_config) - fields)
    if unknown:
        raise ValueError(
            "Checkpoint model_config contains unsupported field(s): "
            + ", ".join(unknown)
        )
    config = JointLatentSASAConfig(**raw_config)
    validate_no_inert_config_overrides(config)

    objective_contract = contract.get("objective")
    if not isinstance(objective_contract, dict):
        raise ValueError("Checkpoint evaluation contract is missing objective")
    objective = objective_contract.get("name")
    if objective not in {"legacy", "normalized"}:
        raise ValueError(f"Unsupported checkpoint objective {objective!r}")
    expected_values = {
        "jepa_target_semantics": config.jepa_target_semantics,
        "jepa_target_stop_gradient": bool(config.jepa_target_stop_gradient),
        "jepa_norm_loss_coeff": float(config.jepa_norm_loss_coeff),
        "jepa_sigreg_estimator": config.jepa_sigreg_estimator,
        "jepa_sigreg_example_count": int(
            config.jepa_sigreg_example_count
        ),
        "target_sigreg_coeff": float(config.jepa_sigreg_coeff),
        "pred_sigreg_coeff": float(config.jepa_pred_sigreg_coeff),
    }
    legacy_objective_defaults = {
        "jepa_norm_loss_coeff": 1.0,
        "jepa_sigreg_estimator": "v_stat",
        "jepa_sigreg_example_count": 0,
    }
    mismatches = {
        key: (
            objective_contract.get(
                key,
                legacy_objective_defaults.get(key),
            ),
            expected,
        )
        for key, expected in expected_values.items()
        if objective_contract.get(
            key,
            legacy_objective_defaults.get(key),
        )
        != expected
    }
    if mismatches:
        raise ValueError(
            "Checkpoint model/objective contract mismatch: "
            + ", ".join(
                f"{key}={observed!r} (model_config {expected!r})"
                for key, (observed, expected) in mismatches.items()
            )
        )
    target_reference = objective_contract.get("target_sigreg_reference_count")
    pred_reference = objective_contract.get("pred_sigreg_reference_count")
    if (
        isinstance(target_reference, bool)
        or not isinstance(target_reference, (int, float))
        or not math.isfinite(float(target_reference))
        or float(target_reference) <= 0.0
    ):
        raise ValueError(
            "Checkpoint target SIGReg reference count must be finite and positive"
        )
    if pred_reference != target_reference:
        raise ValueError(
            "The local evaluator requires equal target/pred SIGReg reference "
            f"counts, got {target_reference!r} and {pred_reference!r}"
        )
    validate_objective_config(objective=str(objective), config=config)
    return config, str(objective), float(target_reference), contract, contract_sha256


def summarize_checkpoint_evaluation_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-seed validation records and select minimum DFM CE."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        validation = record.get("validation")
        if not isinstance(validation, dict):
            raise ValueError("Checkpoint evaluation record has no validation metrics")
        grouped.setdefault(int(record["research_update"]), []).append(record)

    by_checkpoint: list[dict[str, Any]] = []
    for update in sorted(grouped):
        update_records = sorted(
            grouped[update],
            key=lambda record: int(record["validation_seed"]),
        )
        metric_names = sorted(
            set.intersection(
                *(set(record["validation"]) for record in update_records)
            )
        )
        mean_validation = {
            name: float(
                np.mean(
                    [float(record["validation"][name]) for record in update_records],
                    dtype=np.float64,
                )
            )
            for name in metric_names
        }
        by_checkpoint.append(
            {
                "research_update": update,
                "optimizer_step": int(update_records[0]["optimizer_step"]),
                "next_data_cursor": int(update_records[0]["next_data_cursor"]),
                "checkpoint_dir": update_records[0]["checkpoint_dir"],
                "state_sha256": update_records[0]["state_sha256"],
                "validation_seeds": [
                    int(record["validation_seed"]) for record in update_records
                ],
                "validation_seconds_total": float(
                    sum(float(record["validation_seconds"]) for record in update_records)
                ),
                "mean_validation": mean_validation,
            }
        )

    best: dict[str, Any] | None = None
    comparable = [
        item for item in by_checkpoint if "dfm_ce_loss" in item["mean_validation"]
    ]
    if comparable:
        selected = min(
            comparable,
            key=lambda item: (
                float(item["mean_validation"]["dfm_ce_loss"]),
                int(item["research_update"]),
            ),
        )
        best = {
            "metric": "dfm_ce_loss",
            "mode": "min",
            "research_update": selected["research_update"],
            "checkpoint_dir": selected["checkpoint_dir"],
            "mean": selected["mean_validation"]["dfm_ce_loss"],
        }
        first_value = float(by_checkpoint[0]["mean_validation"]["dfm_ce_loss"])
        for item in by_checkpoint:
            if "dfm_ce_loss" in item["mean_validation"]:
                item["dfm_ce_loss_delta_from_first"] = (
                    float(item["mean_validation"]["dfm_ce_loss"]) - first_value
                )

    return {
        "checkpoint_count": len(by_checkpoint),
        "evaluation_count": len(records),
        "by_checkpoint": by_checkpoint,
        "best_checkpoint": best,
    }


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
    ema_target: EmaTargetModel | None = None,
) -> tuple[dict[str, float], float]:
    if count < 1:
        return {}, 0.0
    totals: dict[str, float] = {}
    started = time.perf_counter()
    for index in range(count):
        batch = batches.batch_at(index)
        batch["deterministic_t"] = np.asarray(deterministic_t, dtype=np.float32)
        rng = jax.random.fold_in(jax.random.PRNGKey(seed), index)
        if ema_target is not None:
            if objective != "normalized":
                raise ValueError("EMA target evaluation requires normalized objective")
            loss, aux = eval_ema_normalized_stage1_step(
                model,
                ema_target,
                batch,
                rng,
                sigreg_reference_count,
                sigreg_reference_count,
            )
        elif objective == "normalized":
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
        metrics = {
            "loss": float(loss),
            **flatten_metrics(reportable_stage1_aux(aux)),
        }
        if collapse_diagnostics:
            diagnostic_rng = jax.random.fold_in(rng, 0xC011A95E)
            diagnostics = diagnose_joint_latents(
                model,
                batch,
                diagnostic_rng,
                ema_target,
            )
            jax.block_until_ready(diagnostics)
            metrics.update(flatten_metrics(diagnostics))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / count for key, value in totals.items()}, time.perf_counter() - started


def run_checkpoint_evaluation(
    args: argparse.Namespace,
    *,
    commit: str,
    timestamp: str,
) -> int:
    """Evaluate a checkpoint series read-only with one model/JIT cache."""

    requested_path = require_within_workspace(args.eval_checkpoints)
    checkpoint_root, checkpoints = research_checkpoints_for_evaluation(
        requested_path
    )
    (
        config,
        objective,
        sigreg_reference_count,
        checkpoint_contract,
        checkpoint_contract_sha256,
    ) = checkpoint_evaluation_contract(checkpoints)
    checkpoint_model_config = config
    config, diagnostic_overlays = (
        checkpoint_evaluation_diagnostic_overlay(
            config,
            root_legal_conditional_ce=(
                args.eval_root_legal_conditional_ce
            ),
        )
    )
    validate_objective_config(
        objective=objective,
        config=config,
    )

    data_contract = checkpoint_contract.get("data")
    if not isinstance(data_contract, dict):
        raise ValueError("Checkpoint evaluation contract is missing data metadata")
    training_batch_size = int(data_contract.get("batch_size", 0))
    if training_batch_size < 1:
        raise ValueError("Checkpoint training batch size must be positive")
    eval_batch_size = (
        training_batch_size
        if args.eval_batch_size is None
        else args.eval_batch_size
    )
    validation_seeds = (
        [args.val_seed]
        if args.eval_checkpoint_seeds is None
        else list(args.eval_checkpoint_seeds)
    )
    if len(set(validation_seeds)) != len(validation_seeds):
        raise ValueError("--eval-checkpoint-seeds must not contain duplicates")
    for seed in validation_seeds:
        if seed < 0 or seed > np.iinfo(np.uint32).max:
            raise ValueError(
                f"Validation seeds must be in [0, 2**32 - 1], got {seed}"
            )

    source_label_path = (
        requested_path.parent
        if requested_path.name == "checkpoints"
        else requested_path
    )
    source_label = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", source_label_path.name
    ).strip("-") or "checkpoints"
    run_id = args.run_id or f"checkpoint-eval-{source_label}-{timestamp}"
    output_dir = require_within_workspace(
        args.output_dir
        or REPO_ROOT / "research" / "runs" / run_id
    )
    if output_dir == requested_path or output_dir.is_relative_to(requested_path):
        raise ValueError(
            "--output-dir for --eval-checkpoints must be outside the "
            f"evaluated checkpoint/run tree: {requested_path}"
        )

    assets = checkpoint_contract.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("Checkpoint evaluation contract is missing asset metadata")
    models_dir = require_within_workspace(args.models_dir)
    exported_model = require_within_workspace(
        models_dir / "BT4_exported.pb.gz"
    )
    expected_export_sha256 = assets.get("bt4_exported_sha256")
    observed_export_sha256 = sha256_file(exported_model)
    if observed_export_sha256 != expected_export_sha256:
        raise ValueError(
            "BT4 asset checksum differs from checkpoint contract: "
            f"expected {expected_export_sha256}, observed {observed_export_sha256}"
        )
    trajectory_asset = load_asset_manifest()["trajectory_v3"]["archive"]
    expected_trajectory_sha256 = data_contract.get("source_archive_sha256")
    if trajectory_asset["sha256"] != expected_trajectory_sha256:
        raise ValueError(
            "Trajectory asset checksum differs from checkpoint contract: "
            f"expected {expected_trajectory_sha256}, "
            f"observed {trajectory_asset['sha256']}"
        )

    data_root = require_within_workspace(args.data_root)
    validation_batches = {
        seed: FixedTrajectoryBatches(
            data_root / "val",
            batch_size=eval_batch_size,
            horizon=config.horizon,
            seed=seed,
            shuffle_files=True,
            batch_schedule="global_permutation",
        )
        for seed in validation_seeds
    }
    model_params = load_mapped_bt4_params(models_dir=models_dir)
    model, unused_optimizer = create_joint_components(
        model_params, config, seed=args.seed
    )
    del unused_optimizer
    gc.collect()
    ema_target = (
        EmaTargetModel(model)
        if config.jepa_target_semantics == "ema"
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in (
            vars(args) | {"output_dir": str(output_dir)}
        ).items()
    }
    run_config = {
        "format": "chess-dfm-checkpoint-evaluation-v1",
        "autoresearch_ready": AUTORESEARCH_READY,
        "architecture_source": ARCHITECTURE_SOURCE,
        "git_commit": commit,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "mode": "read_only_checkpoint_evaluation",
        "checkpoint_contract_authoritative": True,
        "checkpoint_root": str(checkpoint_root),
        "checkpoints": [str(path) for path in checkpoints],
        "checkpoint_count": len(checkpoints),
        "checkpoint_resume_contract": checkpoint_contract,
        "checkpoint_resume_contract_sha256": checkpoint_contract_sha256,
        "checkpoint_model_config": serialized_model_config(
            checkpoint_model_config
        ),
        "model_config": serialized_model_config(config),
        "diagnostic_overlays": diagnostic_overlays,
        "learning_rate_schedule": learning_rate_schedule_contract(config),
        "objective": objective,
        "sigreg_reference_count": sigreg_reference_count,
        "training_batch_size": training_batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_batches": args.eval_batches,
        "validation_examples_per_seed": eval_batch_size * args.eval_batches,
        "validation_seeds": validation_seeds,
        "validation_schedule": (
            "one global permutation per seed; batch indexes "
            "[0, eval_batches) reused for every checkpoint"
        ),
        "compilation_reuse": (
            "one model object and one nnx.jit cache for the entire process; "
            "the first evaluation may include compilation"
        ),
        "checkpoint_mutation": "none",
        "optimizer_state_restored": False,
        "args": args_payload,
        "val_data_by_seed": {
            str(seed): batches.provenance()
            for seed, batches in validation_batches.items()
        },
    }
    write_json(output_dir / "run_config.json", run_config)

    records: list[dict[str, Any]] = []
    metrics_path = output_dir / "checkpoint_metrics.jsonl"
    wall_started = time.perf_counter()
    with metrics_path.open("x", encoding="utf-8") as metrics_log:
        for checkpoint in checkpoints:
            restore_started = time.perf_counter()
            manifest = load_research_checkpoint_for_evaluation(
                checkpoint,
                model=model,
                ema_target=ema_target,
            )
            restore_seconds = time.perf_counter() - restore_started
            for validation_seed in validation_seeds:
                validation, validation_seconds = evaluate(
                    model,
                    validation_batches[validation_seed],
                    count=args.eval_batches,
                    seed=validation_seed,
                    deterministic_t=args.val_deterministic_t,
                    objective=objective,
                    sigreg_reference_count=sigreg_reference_count,
                    collapse_diagnostics=args.collapse_diagnostics,
                    ema_target=ema_target,
                )
                record = {
                    "checkpoint_dir": str(checkpoint),
                    "manifest_sha256": sha256_file(
                        checkpoint / "manifest.json"
                    ),
                    "state_sha256": manifest["state"]["sha256"],
                    "research_update": int(manifest["research_update"]),
                    "optimizer_step": int(manifest["optimizer_step"]),
                    "next_data_cursor": int(manifest["next_data_cursor"]),
                    "restore_seconds": restore_seconds,
                    "validation_seed": validation_seed,
                    "eval_batch_size": eval_batch_size,
                    "eval_batches": args.eval_batches,
                    "validation_examples": (
                        eval_batch_size * args.eval_batches
                    ),
                    "validation_seconds": validation_seconds,
                    "validation": validation,
                }
                records.append(record)
                metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_log.flush()
                os.fsync(metrics_log.fileno())

    aggregate = summarize_checkpoint_evaluation_records(records)
    summary = {
        **run_config,
        **aggregate,
        "wall_seconds": time.perf_counter() - wall_started,
        "metrics_path": str(metrics_path),
        "gpu_memory": gpu_memory_stats(),
    }
    write_json(output_dir / "checkpoint_summary.json", summary)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "checkpoint_count": summary["checkpoint_count"],
                "evaluation_count": summary["evaluation_count"],
                "best_checkpoint": summary["best_checkpoint"],
                "wall_seconds": summary["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def should_continue(*, updates: int, steps: int, deadline: float | None) -> bool:
    if steps > 0 and updates >= steps:
        return False
    if deadline is not None and updates > 0 and time.perf_counter() >= deadline:
        return False
    return steps > 0 or deadline is not None


def validate_save_updates(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return a strictly increasing sparse checkpoint schedule."""

    updates = tuple(values)
    if any(isinstance(update, bool) or update <= 0 for update in updates):
        raise ValueError("--save-updates values must be positive integers")
    if tuple(sorted(set(updates))) != updates:
        raise ValueError(
            "--save-updates values must be unique and strictly increasing"
        )
    return updates


def validate_compile_only_args(
    args: argparse.Namespace,
    *,
    save_updates: tuple[int, ...],
) -> None:
    """Fail closed unless compile-only is a zero-state, zero-execution mode."""

    if not args.compile_only:
        return
    if not args.compile_ahead:
        raise ValueError("--compile-only requires --compile-ahead")
    if args.init != "model-only":
        raise ValueError("--compile-only requires --init model-only")
    if args.objective != "normalized":
        raise ValueError(
            "--compile-only currently requires --objective normalized"
        )
    if args.steps != 0:
        raise ValueError("--compile-only requires --steps 0")
    if args.train_seconds != 0.0:
        raise ValueError("--compile-only requires --train-seconds 0")
    if args.eval_batches != 0:
        raise ValueError("--compile-only requires --eval-batches 0")
    if args.eval_only:
        raise ValueError("--compile-only cannot be combined with --eval-only")
    if args.gradient_audit:
        raise ValueError(
            "--compile-only cannot be combined with --gradient-audit"
        )
    if args.eval_checkpoints is not None:
        raise ValueError(
            "--compile-only cannot be combined with --eval-checkpoints"
        )
    if args.eval_root_legal_conditional_ce:
        raise ValueError(
            "--compile-only cannot be combined with "
            "--eval-root-legal-conditional-ce"
        )
    if args.resume_from is not None:
        raise ValueError(
            "--compile-only cannot be combined with --resume-from"
        )
    if args.save_every != 0 or save_updates or args.save_final:
        raise ValueError(
            "--compile-only forbids every checkpoint write request"
        )
    if args.gpu_monitor_interval_ms != 0:
        raise ValueError(
            "--compile-only requires --gpu-monitor-interval-ms 0"
        )
    if args.collapse_diagnostics:
        raise ValueError("--compile-only cannot run collapse diagnostics")
    if not args.donate:
        raise ValueError("--compile-only requires the ordinary donated graph")


def should_save_checkpoint(
    *,
    invocation_update: int,
    research_update: int,
    save_every: int,
    save_updates: tuple[int, ...],
) -> bool:
    return (
        save_every > 0 and invocation_update % save_every == 0
    ) or research_update in save_updates


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


class TrainingCompilation(NamedTuple):
    executable: Any
    seconds: float
    cost_analysis_raw: dict[str, float]
    memory_analysis: dict[str, int]


def compile_training_executable(
    train_fn: Any,
    *,
    objective: str,
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: Mapping[str, Any],
    rng: jax.Array,
    sigreg_reference_count: float,
    ema_target: EmaTargetModel | None,
) -> TrainingCompilation:
    """Lower and compile the canonical training call without executing it."""

    compile_args = training_call_args(
        objective=objective,
        model=model,
        optimizer=optimizer,
        batch=batch,
        rng=rng,
        sigreg_reference_count=sigreg_reference_count,
        ema_target=ema_target,
    )
    started = time.perf_counter()
    executable = train_fn.lower(*compile_args).compile()
    seconds = time.perf_counter() - started
    cost_analysis_raw = (
        normalize_cost_analysis(executable.cost_analysis())
        if hasattr(executable, "cost_analysis")
        else {}
    )
    memory_analysis = (
        normalize_memory_analysis(executable.memory_analysis())
        if hasattr(executable, "memory_analysis")
        else {}
    )
    return TrainingCompilation(
        executable=executable,
        seconds=seconds,
        cost_analysis_raw=cost_analysis_raw,
        memory_analysis=memory_analysis,
    )


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


def training_function(
    *,
    objective: str,
    donate: bool,
    target_semantics: str = "online",
):
    if target_semantics == "ema":
        if objective != "normalized":
            raise ValueError("EMA targets require normalized objective")
        return (
            train_ema_normalized_stage1_step_donated
            if donate
            else train_ema_normalized_stage1_step
        )
    if target_semantics != "online":
        raise ValueError(
            f"Unsupported JEPA target semantics {target_semantics!r}"
        )
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
    ema_target: EmaTargetModel | None = None,
) -> tuple[Any, ...]:
    if ema_target is not None:
        if objective != "normalized":
            raise ValueError("EMA targets require normalized objective")
        return (
            model,
            ema_target,
            optimizer,
            batch,
            rng,
            sigreg_reference_count,
            sigreg_reference_count,
        )
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


def training_state_footprint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
) -> dict[str, Any]:
    """Summarize model parameters and optimizer storage without full schemas."""

    _, trainable_state, _ = nnx.split(model, TrainableParam, ...)
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(trainable_state)
    group_counts = {name: 0 for name in GRADIENT_GROUP_NAMES}
    partition_counts = {
        "main": 0,
        "bt4": 0,
        "frozen_bt4": 0,
    }
    for path, leaf in path_leaves:
        count = int(np.prod(leaf.shape, dtype=np.int64))
        group = gradient_group_for_path(path)
        partition = _optimizer_partition_for_path(
            path,
            freeze_backbone=model.config.bt4_freeze_backbone,
        )
        group_counts[group] += count
        group_counts["all"] += count
        partition_counts[partition] += count

    def compact_abi(value: Any) -> dict[str, Any]:
        abi = research_state_abi(value)
        return {
            "sha256": abi["sha256"],
            "leaf_count": int(abi["leaf_count"]),
            "nbytes": int(abi["nbytes"]),
        }

    return {
        "model_trainable_parameter_count_by_group": group_counts,
        "model_parameter_count_by_optimizer_partition": partition_counts,
        "optimizer_active_parameter_count": (
            partition_counts["main"] + partition_counts["bt4"]
        ),
        "optimizer_frozen_parameter_count": partition_counts[
            "frozen_bt4"
        ],
        "model_trainable_state_abi": compact_abi(
            nnx.state(model, TrainableParam)
        ),
        "optimizer_state_abi": compact_abi(
            nnx.state(optimizer.opt_state)
        ),
    }


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

    component_names = gradient_component_names(model.config)
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
        component_count = len(component_names)

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

    component_count = len(component_names)
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
            for component, norm in zip(component_names, norms, strict=True)
        }
        cosines: dict[str, float | None] = {}
        for left in range(component_count):
            for right in range(left + 1, component_count):
                denom = norms[left] * norms[right]
                key = (
                    f"{component_names[left]}"
                    f"__{component_names[right]}"
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

    primary_coefficients = np.zeros(
        (component_count,),
        dtype=np.float64,
    )
    primary_coefficients[component_names.index("dfm_ce")] = 1.0
    primary_coefficients[component_names.index("jepa_positive")] = 1.0
    primary_coefficients[component_names.index("fp32_legality")] = (
        model.config.first_legality_coeff
    )
    if WDL_COMPONENT in component_names:
        primary_coefficients[component_names.index(WDL_COMPONENT)] = (
            model.config.wdl_coeff
        )
    if BT4_POLICY_DISTILL_COMPONENT in component_names:
        primary_coefficients[
            component_names.index(BT4_POLICY_DISTILL_COMPONENT)
        ] = model.config.bt4_policy_distill_coeff
    if ROOT_LEGAL_CONDITIONAL_COMPONENT in component_names:
        primary_coefficients[
            component_names.index(ROOT_LEGAL_CONDITIONAL_COMPONENT)
        ] = model.config.root_legal_conditional_ce_coeff
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

    result = {
        "component_names": list(component_names),
        "component_values": {
            name: float(value)
            for name, value in zip(
                component_names,
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
                component_names,
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
                component_names,
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
    if TARGET_VARIANCE_HINGE_COMPONENT in component_names:
        hinge_index = component_names.index(
            TARGET_VARIANCE_HINGE_COMPONENT
        )
        variance_reference_coefficients = np.zeros(
            (component_count,),
            dtype=np.float64,
        )
        variance_reference_coefficients[
            component_names.index("jepa_positive")
        ] = model.config.jepa_positive_coeff
        variance_reference_coefficients[
            component_names.index("target_sigreg")
        ] = model.config.jepa_sigreg_coeff
        variance_fractions = (0.03, 0.10, 0.30)
        variance_suggestions: dict[
            str,
            dict[str, float],
        ] = {}
        for group_index, group_name in enumerate(GRADIENT_GROUP_NAMES):
            gram = grams[group_index]
            reference_norm = float(
                np.sqrt(
                    max(
                        float(
                            variance_reference_coefficients
                            @ gram
                            @ variance_reference_coefficients
                        ),
                        0.0,
                    )
                )
            )
            hinge_norm = float(
                np.sqrt(
                    max(
                        float(gram[hinge_index, hinge_index]),
                        0.0,
                    )
                )
            )
            if reference_norm == 0.0 or hinge_norm == 0.0:
                continue
            variance_suggestions[group_name] = {
                f"{fraction:.2f}": (
                    fraction * reference_norm / hinge_norm
                )
                for fraction in variance_fractions
            }
        result.update(
            {
                "target_variance_hinge_reference": (
                    "configured_jepa_positive_plus_target_sigreg"
                ),
                "target_variance_hinge_reference_gradient_coefficients": {
                    name: float(value)
                    for name, value in zip(
                        component_names,
                        variance_reference_coefficients,
                        strict=True,
                    )
                },
                "suggested_target_variance_hinge_coefficients_by_gradient_fraction": (
                    variance_suggestions
                ),
            }
        )
    return result


def run_compile_only(
    args: argparse.Namespace,
    *,
    commit: str,
    timestamp: str,
    run_id: str,
    output_dir: Path,
    config: JointLatentSASAConfig,
    train_batches: FixedTrajectoryBatches,
    resume_contract: dict[str, Any],
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    ema_target: EmaTargetModel | None,
    checkpoint_step: int,
    source_checkpoint_path: Path,
) -> int:
    """Populate the training cache without decoding or executing source state."""

    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = require_within_workspace(
        os.environ["JAX_COMPILATION_CACHE_DIR"]
    )
    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in (
            vars(args) | {"output_dir": str(output_dir)}
        ).items()
    }
    state_footprint = training_state_footprint(model, optimizer)
    run_config = {
        "format": "chess-dfm-compile-only-v1",
        "mode": "compile_only_shape_equivalent",
        "autoresearch_ready": AUTORESEARCH_READY,
        "architecture_source": ARCHITECTURE_SOURCE,
        "git_commit": commit,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "args": args_payload,
        "model_config": serialized_model_config(config),
        "learning_rate_schedule": learning_rate_schedule_contract(config),
        "training_state_footprint": state_footprint,
        "checkpoint_step": checkpoint_step,
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_opened": False,
        "source_checkpoint_values_restored": False,
        "constructor_parameter_payload_released": True,
        "parameter_values_are_dynamic_compilation_inputs": True,
        "initial_optimizer_step": int(optimizer.step[...]),
        "initial_data_cursor": 0,
        "resume_contract": resume_contract,
        "resume_contract_sha256": _json_sha256(resume_contract),
        "train_data": train_batches.provenance(),
        "jax_compilation_cache_dir": str(cache_dir),
        "checkpoint_writes": 0,
        "updates": 0,
        "validation_batches": 0,
    }
    write_json(output_dir / "run_config.json", run_config)

    train_fn = training_function(
        objective=args.objective,
        donate=args.donate,
        target_semantics=config.jepa_target_semantics,
    )
    compile_batch = train_batches.batch_at(0)
    compile_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 0)
    compilation = compile_training_executable(
        train_fn,
        objective=args.objective,
        model=model,
        optimizer=optimizer,
        batch=compile_batch,
        rng=compile_rng,
        sigreg_reference_count=args.sigreg_reference_count,
        ema_target=ema_target,
    )
    optimizer_step_after_compile = int(optimizer.step[...])
    if optimizer_step_after_compile != run_config["initial_optimizer_step"]:
        raise RuntimeError(
            "Compile-only changed the optimizer step without execution: "
            f"{optimizer_step_after_compile} != "
            f"{run_config['initial_optimizer_step']}"
        )
    if compilation.cost_analysis_raw:
        write_json(
            output_dir / "compiler_cost_analysis.json",
            compilation.cost_analysis_raw,
        )
    report = {
        **run_config,
        "explicit_compile_seconds": compilation.seconds,
        "compiler_cost_analysis": compiler_cost_summary(
            compilation.cost_analysis_raw
        ),
        "compiler_memory_analysis": compilation.memory_analysis,
        "gpu_memory": gpu_memory_stats(),
        "completed": True,
        "optimizer_step_after_compile": optimizer_step_after_compile,
        "model_or_optimizer_executed": False,
        "checkpoint_path": None,
        "metrics_path": None,
    }
    write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "mode": report["mode"],
                "completed": report["completed"],
                "updates": report["updates"],
                "checkpoint_writes": report["checkpoint_writes"],
                "source_checkpoint_opened": report[
                    "source_checkpoint_opened"
                ],
                "explicit_compile_seconds": report[
                    "explicit_compile_seconds"
                ],
                "compiler_cost_analysis": report[
                    "compiler_cost_analysis"
                ],
                "compiler_memory_analysis": report[
                    "compiler_memory_analysis"
                ],
                "gpu_memory": report["gpu_memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    validate_environment()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"research/train.py requires GPU, found {jax.default_backend()!r}")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if (
        not args.compile_only
        and not args.eval_only
        and not args.gradient_audit
        and args.eval_checkpoints is None
        and args.steps == 0
        and args.train_seconds <= 0
    ):
        raise ValueError("Set --steps > 0 or --train-seconds > 0")
    if args.train_seconds < 0:
        raise ValueError("--train-seconds must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive")
    if args.sigreg_reference_count <= 0:
        raise ValueError("--sigreg-reference-count must be positive")
    for flag, value in (
        ("--learning-rate", args.learning_rate),
        ("--bt4-learning-rate", args.bt4_learning_rate),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{flag} must be finite and non-negative")
    if args.gpu_monitor_interval_ms < 0:
        raise ValueError("--gpu-monitor-interval-ms must be non-negative")
    if 0 < args.gpu_monitor_interval_ms < 50:
        raise ValueError("--gpu-monitor-interval-ms must be 0 or at least 50")
    if args.gradient_audit and args.eval_only:
        raise ValueError("--gradient-audit and --eval-only are mutually exclusive")
    if args.gradient_audit and args.eval_checkpoints is not None:
        raise ValueError(
            "--gradient-audit and --eval-checkpoints are mutually exclusive"
        )
    if args.gradient_audit and args.objective != "normalized":
        raise ValueError("--gradient-audit requires --objective normalized")
    if not 0.0 <= args.val_deterministic_t <= 1.0:
        raise ValueError("--val-deterministic-t must be in [0, 1]")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    save_updates = validate_save_updates(args.save_updates)
    validate_compile_only_args(args, save_updates=save_updates)
    if args.max_checkpoints < 0:
        raise ValueError("--max-checkpoints must be non-negative")
    if args.resume_from is not None and args.init != "exact":
        raise ValueError("--resume-from is an exact continuation and cannot use --init model-only")
    if args.resume_from is not None and args.eval_checkpoints is not None:
        raise ValueError(
            "--resume-from and --eval-checkpoints are mutually exclusive"
        )
    if (
        args.eval_root_legal_conditional_ce
        and args.eval_checkpoints is None
    ):
        raise ValueError(
            "--eval-root-legal-conditional-ce requires "
            "--eval-checkpoints"
        )
    if args.eval_checkpoints is not None:
        if args.eval_batches < 1:
            raise ValueError("--eval-checkpoints requires --eval-batches > 0")
        if args.train_seconds > 0:
            raise ValueError(
                "--eval-checkpoints cannot be combined with --train-seconds"
            )
        if args.save_every > 0 or save_updates or args.save_final:
            raise ValueError(
                "--eval-checkpoints cannot save or prune checkpoints"
            )
        if args.compile_ahead:
            raise ValueError(
                "--compile-ahead is a training option and cannot be combined "
                "with --eval-checkpoints"
            )
        if args.gpu_monitor_interval_ms > 0:
            raise ValueError(
                "--gpu-monitor-interval-ms is a training option and cannot "
                "be combined with --eval-checkpoints"
            )

    run_root = require_within_workspace(args.run_root)
    checkpoint_dir = require_within_workspace(args.checkpoint_dir)
    resume_from = (
        require_within_workspace(args.resume_from)
        if args.resume_from is not None
        else None
    )
    models_dir = require_within_workspace(args.models_dir)
    data_root = require_within_workspace(args.data_root)
    commit = git_commit()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.eval_checkpoints is not None:
        return run_checkpoint_evaluation(
            args,
            commit=commit,
            timestamp=timestamp,
        )
    run_id = args.run_id or f"compat-step265k-b{args.batch_size}-{timestamp}"
    output_dir = require_within_workspace(
        args.output_dir or REPO_ROOT / "research" / "runs" / run_id
    )
    metrics_path = output_dir / "metrics.jsonl"

    config, metadata = resolve_config(run_root)
    config = apply_experiment_overrides(config)
    config = apply_config_overrides(config, args)
    validate_no_inert_config_overrides(config)
    validate_objective_config(
        objective=args.objective,
        config=config,
    )
    validate_bt4_freeze_config(config)
    validate_bt4_future_target_stop_gradient_config(config)
    if resume_from is None:
        validate_legacy_init_for_config(config, args.init)
    if (
        args.gradient_audit
        and config.jepa_target_semantics == "ema"
    ):
        raise ValueError(
            "--gradient-audit currently measures online objective components "
            "and cannot be combined with EMA targets."
        )
    train_batches = FixedTrajectoryBatches(
        data_root / "train",
        batch_size=args.batch_size,
        horizon=config.horizon,
        seed=args.seed,
        shuffle_files=True,
        batch_schedule=args.train_batch_schedule,
    )
    eval_batch_size = (
        args.batch_size
        if args.eval_batch_size is None
        else args.eval_batch_size
    )
    val_batches = FixedTrajectoryBatches(
        data_root / "val",
        batch_size=eval_batch_size,
        horizon=config.horizon,
        seed=args.val_seed,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    resume_contract = build_research_resume_contract(
        config=config,
        objective=args.objective,
        sigreg_reference_count=args.sigreg_reference_count,
        batch_size=args.batch_size,
        train_seed=args.seed,
        train_provenance=train_batches.provenance(),
        models_dir=models_dir,
    )

    model_params = load_mapped_bt4_params(models_dir=models_dir)
    model, optimizer = create_joint_components(model_params, config, seed=args.seed)
    ema_target = (
        EmaTargetModel(model)
        if config.jepa_target_semantics == "ema"
        else None
    )
    if args.compile_only:
        release_construction_parameter_payload(model_params)
        del model_params
        checkpoint_step = int(metadata["latest_step"])
        source_checkpoint_path = require_within_workspace(
            checkpoint_dir / f"step{checkpoint_step:07d}" / "state.npz"
        )
        expected_source_path = require_within_workspace(
            DEFAULT_CHECKPOINT_DIR / "step0265000" / "state.npz"
        )
        if (
            checkpoint_step != 265_000
            or source_checkpoint_path != expected_source_path
        ):
            raise ValueError(
                "Compile-only supports only the checksum-pinned "
                "step-265,000 model-only source contract."
            )
        return run_compile_only(
            args,
            commit=commit,
            timestamp=timestamp,
            run_id=run_id,
            output_dir=output_dir,
            config=config,
            train_batches=train_batches,
            resume_contract=resume_contract,
            model=model,
            optimizer=optimizer,
            ema_target=ema_target,
            checkpoint_step=checkpoint_step,
            source_checkpoint_path=source_checkpoint_path,
        )
    restore_started = time.perf_counter()
    research_update = 0
    next_data_cursor = 0
    resumed_from: str | None = None
    if resume_from is not None:
        resume_manifest = load_research_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            expected_resume_contract=resume_contract,
            ema_target=ema_target,
        )
        checkpoint_step = int(resume_manifest["optimizer_step"])
        research_update = int(resume_manifest["research_update"])
        next_data_cursor = int(resume_manifest["next_data_cursor"])
        resumed_from = str(resume_manifest["checkpoint_dir"])
        lineage = {
            "kind": "research_resume",
            "parent_checkpoint": resumed_from,
            "parent_manifest_sha256": sha256_file(
                Path(resumed_from) / "manifest.json"
            ),
            "parent_run_id": resume_manifest.get("lineage", {}).get("run_id"),
            "run_id": run_id,
            "git_commit": commit,
        }
    else:
        checkpoint_step = int(metadata["latest_step"])
        source_checkpoint_path = require_within_workspace(
            checkpoint_dir / f"step{checkpoint_step:07d}" / "state.npz"
        )
        expected_source_path = require_within_workspace(
            DEFAULT_CHECKPOINT_DIR / "step0265000" / "state.npz"
        )
        if checkpoint_step != 265_000 or source_checkpoint_path != expected_source_path:
            raise ValueError(
                "The clean trainer imports only the checksum-pinned step-265,000 "
                "legacy source. Convert any other legacy state explicitly before use."
            )
        checkpoint_asset = load_asset_manifest()["checkpoint_step_265000"]
        source_init_mode = (
            "exact"
            if args.init == "exact" and not args.gradient_audit
            else "model-only"
        )
        import_optimizer = optimizer
        if config.bt4_freeze_backbone:
            import_optimizer = create_joint_optimizer(
                model,
                config,
                freeze_backbone=False,
            )
        import_result = import_legacy_checkpoint(
            source_checkpoint_path,
            model=model,
            optimizer=import_optimizer,
            expected_size_bytes=int(checkpoint_asset["state_npz_size_bytes"]),
            expected_sha256=str(checkpoint_asset["state_npz_sha256"]),
            expected_step=checkpoint_step,
            init_mode=source_init_mode,
        )
        if import_optimizer is not optimizer:
            del import_optimizer
        if ema_target is not None:
            sync_ema_target_from_online(ema_target, model)
        lineage = {
            "kind": "legacy_import",
            "source_checkpoint": str(source_checkpoint_path),
            "source_checkpoint_step": checkpoint_step,
            "source_checkpoint_size_bytes": import_result.source_size_bytes,
            "source_checkpoint_sha256": import_result.source_sha256,
            "source_model_abi_sha256": import_result.model_abi_sha256,
            "source_optimizer_abi_sha256": import_result.optimizer_abi_sha256,
            "init_mode": source_init_mode,
            "run_id": run_id,
            "git_commit": commit,
        }
        if config.bt4_freeze_backbone:
            lineage["legacy_optimizer_validation"] = (
                "source_compatible_temporary_optimizer"
            )
            lineage["training_optimizer"] = (
                "fresh_frozen_bt4_optimizer"
            )
    restore_seconds = time.perf_counter() - restore_started
    initial_optimizer_step = int(optimizer.step[...])
    initial_research_update = research_update
    initial_data_cursor = next_data_cursor
    state_footprint = training_state_footprint(model, optimizer)
    output_dir.mkdir(parents=True, exist_ok=False)

    if args.gradient_audit:
        if config.jepa_target_variance_hinge_coeff != 0.0:
            if config.jepa_sigreg_coeff == 0.0:
                raise ValueError(
                    "Target-variance-hinge --gradient-audit requires "
                    "nonzero --target-sigreg-coeff for its configured "
                    "JEPA-positive-plus-target-SIGReg reference."
                )
        elif (
            config.jepa_sigreg_coeff == 0.0
            or config.jepa_pred_sigreg_coeff == 0.0
        ):
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
            "unevaluated_legacy_aux_metrics": sorted(
                UNEVALUATED_LEGACY_AUX_METRICS
            ),
            "git_commit": commit,
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in (
                    vars(args) | {"output_dir": str(output_dir)}
                ).items()
            },
            "model_config": serialized_model_config(config),
            "learning_rate_schedule": learning_rate_schedule_contract(
                config
            ),
            "training_state_footprint": state_footprint,
            "checkpoint_step": checkpoint_step,
            "initial_optimizer_step": initial_optimizer_step,
            "initial_research_update": initial_research_update,
            "initial_data_cursor": initial_data_cursor,
            "resumed_from": resumed_from,
            "resume_contract": resume_contract,
            "resume_contract_sha256": _json_sha256(resume_contract),
            "restore_seconds": restore_seconds,
            "train_data": train_batches.provenance(),
            "audit": audit,
        }
        write_json(output_dir / "run_config.json", audit_config)
        write_json(output_dir / "gradient_audit.json", audit_config)
        audit_summary = {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "component_values": audit["component_values"],
            "gradient_norms": audit["gradient_norms"],
            "suggested_sigreg_coefficients": audit[
                "suggested_sigreg_coefficients_by_gradient_fraction"
            ],
            "wall_seconds": audit["wall_seconds"],
        }
        if (
            "suggested_target_variance_hinge_coefficients_by_gradient_fraction"
            in audit
        ):
            audit_summary["suggested_target_variance_hinge_coefficients"] = (
                audit[
                    "suggested_target_variance_hinge_coefficients_by_gradient_fraction"
                ]
            )
        print(
            json.dumps(
                audit_summary,
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
        ema_target=ema_target,
    )

    ema_target_metadata: dict[str, Any] | None = None
    if ema_target is not None:
        ema_checkpoint_abi = research_state_abi(
            ema_checkpoint_state(ema_target)
        )
        ema_master_abi = research_state_abi(
            nnx.state(ema_target.encoder_master)
        )
        ema_compute_abi = research_state_abi(
            nnx.state(ema_target.encoder_compute)
        )
        ema_target_metadata = {
            "checkpointed_trainable_state_abi": ema_checkpoint_abi,
            "runtime_variable_state_abi": research_state_abi(
                nnx.state(ema_target)
            ),
            "encoder_master_state_abi": ema_master_abi,
            "encoder_compute_mirror_state_abi": ema_compute_abi,
            "checkpointed_master_scope": [
                "encoder_master",
                "state_projector",
                "jepa_state_norm",
            ],
            "derived_runtime_scope": ["encoder_compute"],
            "floating_master_storage_dtype": "float32",
            "encoder_compute_storage_dtype": "online_native",
            "encoder_forward_source": "encoder_compute_only",
            "encoder_compute_checkpointed": False,
            "encoder_compute_refresh": [
                "after_source_sync",
                "after_each_post_optimizer_ema_update",
                "after_checkpoint_restore",
            ],
            "encoder_compute_equation": (
                "encoder_compute=cast_online_native(encoder_master)"
            ),
            "mirror_refresh_state_bandwidth_per_update": {
                "encoder_master_read_bytes": int(
                    ema_master_abi["nbytes"]
                ),
                "encoder_compute_write_bytes": int(
                    ema_compute_abi["nbytes"]
                ),
                "total_bytes": int(
                    ema_master_abi["nbytes"]
                    + ema_compute_abi["nbytes"]
                ),
            },
            "extra_forward": {
                "scope": "future_planes_only",
                "modules": ["bt4_encoder", "state_projector"],
                "encoded_boards_per_example": int(config.horizon),
                "compilation_static": True,
            },
            "positive_loss_target_source": "ema_future_vectors",
            "target_sigreg_source": (
                "online_z_all_current_and_future"
            ),
            "initialization": (
                "restored_checkpoint"
                if resumed_from is not None
                else "exact_copy_after_source_restore"
            ),
            "update_order": "after_online_optimizer",
            "hard_acceptance_metrics": [
                "steady_end_to_end_examples_per_second",
                "training_wall_examples_per_second",
                "steady_device_examples_per_second",
                "steady_encoded_boards_per_second",
                "gpu_memory.peak_bytes_in_use",
                "gpu_monitor.memory_used_mib_max",
            ],
        }

    run_config = {
        "autoresearch_ready": AUTORESEARCH_READY,
        "architecture_source": ARCHITECTURE_SOURCE,
        "unevaluated_legacy_aux_metrics": sorted(
            UNEVALUATED_LEGACY_AUX_METRICS
        ),
        "git_commit": commit,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "args": vars(args) | {"output_dir": str(output_dir)},
        "model_config": serialized_model_config(config),
        "learning_rate_schedule": learning_rate_schedule_contract(config),
        "training_state_footprint": state_footprint,
        "ema_target": ema_target_metadata,
        "checkpoint_step": checkpoint_step,
        "initial_optimizer_step": initial_optimizer_step,
        "initial_research_update": initial_research_update,
        "initial_data_cursor": initial_data_cursor,
        "resumed_from": resumed_from,
        "resume_contract": resume_contract,
        "resume_contract_sha256": _json_sha256(resume_contract),
        "train_data": train_batches.provenance(),
        "val_data": val_batches.provenance(),
    }
    run_config["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_config["args"].items()
    }
    write_json(output_dir / "run_config.json", run_config)

    train_fn = training_function(
        objective=args.objective,
        donate=args.donate,
        target_semantics=config.jepa_target_semantics,
    )
    executable = train_fn
    explicit_compile_seconds: float | None = None
    compiler_cost_analysis_raw: dict[str, float] = {}
    compiler_memory_analysis: dict[str, int] = {}
    if args.compile_ahead and not args.eval_only:
        compile_batch = train_batches.batch_at(next_data_cursor)
        compile_rng = jax.random.fold_in(
            jax.random.PRNGKey(args.seed),
            next_data_cursor,
        )
        compilation = compile_training_executable(
            train_fn,
            objective=args.objective,
            model=model,
            optimizer=optimizer,
            batch=compile_batch,
            rng=compile_rng,
            sigreg_reference_count=args.sigreg_reference_count,
            ema_target=ema_target,
        )
        executable = compilation.executable
        explicit_compile_seconds = compilation.seconds
        compiler_cost_analysis_raw = compilation.cost_analysis_raw
        compiler_memory_analysis = compilation.memory_analysis
        if compiler_cost_analysis_raw:
            write_json(
                output_dir / "compiler_cost_analysis.json",
                compiler_cost_analysis_raw,
            )

    updates = 0
    examples = 0
    first_fetch_seconds: float | None = None
    first_update_seconds: float | None = None
    steady_fetch_seconds: list[float] = []
    steady_update_seconds: list[float] = []
    steady_iteration_seconds: list[float] = []
    final_train_metrics: dict[str, float] = {}
    deadline: float | None = float("inf") if args.train_seconds > 0 else None
    training_wall_started = time.perf_counter()
    gpu_monitor: tuple[subprocess.Popen[str], IO[str], IO[str]] | None = None
    checkpoint_save_seconds = 0.0
    last_checkpoint_path: Path | None = None
    last_checkpoint_update: int | None = None

    def save_current_checkpoint() -> Path:
        nonlocal checkpoint_save_seconds, last_checkpoint_path, last_checkpoint_update
        save_started = time.perf_counter()
        saved = save_research_checkpoint(
            output_dir / "checkpoints",
            model=model,
            optimizer=optimizer,
            research_update=research_update,
            next_data_cursor=next_data_cursor,
            resume_contract=resume_contract,
            lineage=lineage,
            max_to_keep=args.max_checkpoints,
            extra={"last_train_metrics": final_train_metrics},
            ema_target=ema_target,
        )
        checkpoint_save_seconds += time.perf_counter() - save_started
        last_checkpoint_path = saved
        last_checkpoint_update = research_update
        return saved

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

                data_step = next_data_cursor
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
                    ema_target=ema_target,
                )
                loss, aux = executable(*call_args)
                jax.block_until_ready((loss, aux))
                update_seconds = time.perf_counter() - update_started

                updates += 1
                research_update += 1
                next_data_cursor += 1
                examples += args.batch_size
                if first_update_seconds is None:
                    first_fetch_seconds = fetch_seconds
                    first_update_seconds = update_seconds
                    if args.train_seconds > 0:
                        deadline = time.perf_counter() + args.train_seconds
                else:
                    steady_fetch_seconds.append(fetch_seconds)
                    steady_update_seconds.append(update_seconds)
                    steady_iteration_seconds.append(
                        fetch_seconds + update_seconds
                    )

                final_train_metrics = {
                    "loss": float(loss),
                    **flatten_metrics(reportable_stage1_aux(aux)),
                }
                record = {
                    "update": updates,
                    "research_update": research_update,
                    "optimizer_step": int(optimizer.step[...]),
                    "data_step": data_step,
                    "next_data_cursor": next_data_cursor,
                    "fetch_seconds": fetch_seconds,
                    "update_seconds": update_seconds,
                    "examples_per_second": args.batch_size / max(update_seconds, 1e-12),
                    "end_to_end_examples_per_second": (
                        args.batch_size
                        / max(fetch_seconds + update_seconds, 1e-12)
                    ),
                    **final_train_metrics,
                }
                metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_log.flush()
                if should_save_checkpoint(
                    invocation_update=updates,
                    research_update=research_update,
                    save_every=args.save_every,
                    save_updates=save_updates,
                ):
                    save_current_checkpoint()
    finally:
        if gpu_monitor is not None:
            stop_gpu_monitor(*gpu_monitor)

    if (
        args.save_final
        and updates > 0
        and last_checkpoint_update != research_update
    ):
        save_current_checkpoint()

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
        ema_target=ema_target,
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
    steady_fetch_seconds_mean = (
        float(np.mean(steady_fetch_seconds))
        if steady_fetch_seconds
        else None
    )
    steady_fetch_seconds_p50 = (
        float(np.quantile(steady_fetch_seconds, 0.50))
        if steady_fetch_seconds
        else None
    )
    steady_fetch_seconds_p95 = (
        float(np.quantile(steady_fetch_seconds, 0.95))
        if steady_fetch_seconds
        else None
    )
    steady_iteration_seconds_mean = (
        float(np.mean(steady_iteration_seconds))
        if steady_iteration_seconds
        else None
    )
    steady_iteration_seconds_p50 = (
        float(np.quantile(steady_iteration_seconds, 0.50))
        if steady_iteration_seconds
        else None
    )
    steady_iteration_seconds_p95 = (
        float(np.quantile(steady_iteration_seconds, 0.95))
        if steady_iteration_seconds
        else None
    )
    performance_seconds = steady_update_seconds_mean or first_update_seconds
    compiler_cost_analysis = compiler_cost_summary(compiler_cost_analysis_raw)
    performance = compiler_performance(compiler_cost_analysis, performance_seconds)
    online_target_horizons = (
        config.jepa_target_sample_count
        if 0 < config.jepa_target_sample_count < config.horizon
        else config.horizon
    )
    online_encoded_boards_per_example = online_target_horizons + 1
    encoded_boards_per_example = (
        online_encoded_boards_per_example
        + (
            config.horizon
            if config.jepa_target_semantics == "ema"
            else 0
        )
    )
    throughput = {
        "throughput_semantics": {
            "steady_device_examples_per_second": (
                "batch_size / accelerator update time; excludes host fetch"
            ),
            "steady_end_to_end_examples_per_second": (
                "batch_size / (host fetch + accelerator update)"
            ),
            "training_wall_examples_per_second": (
                "all examples / measured training-loop wall time; includes "
                "first update, monitoring, and checkpoint saves"
            ),
            "steady_examples_per_second": (
                "backward-compatible alias for "
                "steady_device_examples_per_second"
            ),
        },
        "steady_examples_per_second": (
            None
            if steady_update_seconds_mean is None
            else args.batch_size / steady_update_seconds_mean
        ),
        "steady_device_examples_per_second": (
            None
            if steady_update_seconds_mean is None
            else args.batch_size / steady_update_seconds_mean
        ),
        "steady_end_to_end_examples_per_second": (
            None
            if steady_iteration_seconds_mean is None
            else args.batch_size / steady_iteration_seconds_mean
        ),
        "training_wall_examples_per_second": (
            None
            if training_wall_seconds <= 0.0
            else examples / training_wall_seconds
        ),
        "steady_data_stall_fraction": (
            None
            if steady_fetch_seconds_mean is None
            or steady_iteration_seconds_mean is None
            else steady_fetch_seconds_mean / steady_iteration_seconds_mean
        ),
        "steady_encoded_boards_per_second": (
            None
            if steady_update_seconds_mean is None
            else args.batch_size
            * encoded_boards_per_example
            / steady_update_seconds_mean
        ),
        "encoded_boards_per_example": encoded_boards_per_example,
        "online_encoded_boards_per_example": (
            online_encoded_boards_per_example
        ),
        "online_future_target_horizons_per_example": online_target_horizons,
        "ema_teacher_encoded_boards_per_example": (
            config.horizon
            if config.jepa_target_semantics == "ema"
            else 0
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
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "last_checkpoint_path": (
            str(last_checkpoint_path) if last_checkpoint_path is not None else None
        ),
        "explicit_compile_seconds": explicit_compile_seconds,
        "first_fetch_seconds": first_fetch_seconds,
        "first_update_seconds": first_update_seconds,
        "compile_and_first_update_seconds": (
            None
            if first_update_seconds is None
            else first_update_seconds + (explicit_compile_seconds or 0.0)
        ),
        "steady_update_seconds_mean": steady_update_seconds_mean,
        "steady_update_seconds_p50": steady_update_seconds_p50,
        "steady_update_seconds_p95": steady_update_seconds_p95,
        "steady_fetch_seconds_mean": steady_fetch_seconds_mean,
        "steady_fetch_seconds_p50": steady_fetch_seconds_p50,
        "steady_fetch_seconds_p95": steady_fetch_seconds_p95,
        "steady_iteration_seconds_mean": steady_iteration_seconds_mean,
        "steady_iteration_seconds_p50": steady_iteration_seconds_p50,
        "steady_iteration_seconds_p95": steady_iteration_seconds_p95,
        "compiler_cost_analysis": compiler_cost_analysis,
        "compiler_memory_analysis": compiler_memory_analysis,
        "gpu_monitor": gpu_monitor_summary,
        "gpu_samples_path": str(gpu_samples_path) if gpu_samples_path.is_file() else None,
        **performance,
        **throughput,
        "updates": updates,
        "research_update": research_update,
        "next_data_cursor": next_data_cursor,
        "final_optimizer_step": int(optimizer.step[...]),
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
                "research_update": research_update,
                "next_data_cursor": next_data_cursor,
                "examples": examples,
                "last_checkpoint_path": report["last_checkpoint_path"],
                "initial_val_dfm_ce": initial_val.get("dfm_ce_loss"),
                "final_val_dfm_ce": final_val.get("dfm_ce_loss"),
                "explicit_compile_seconds": explicit_compile_seconds,
                "first_update_seconds": first_update_seconds,
                "steady_update_seconds_mean": report["steady_update_seconds_mean"],
                "steady_device_examples_per_second": report[
                    "steady_device_examples_per_second"
                ],
                "steady_end_to_end_examples_per_second": report[
                    "steady_end_to_end_examples_per_second"
                ],
                "training_wall_examples_per_second": report[
                    "training_wall_examples_per_second"
                ],
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
