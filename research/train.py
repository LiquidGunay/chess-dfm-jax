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


AUTORESEARCH_READY = False
ARCHITECTURE_SOURCE = "research_train_local_model_and_loss"

# AUTORESEARCH EDIT SURFACE: change model/objective knobs here. Checkpoint
# metadata is loaded first, then these values, then explicit CLI overrides.
EXPERIMENT_OVERRIDES: dict[str, Any] = {
    # "jepa_target_stop_gradient": True,
    # "jepa_target_semantics": "ema",
    # "jepa_target_ema_decay": 0.99,
    # "jepa_target_variance_hinge_coeff": 1.0,
    # "jepa_target_variance_hinge_gamma": 0.9,
    # "jepa_state_fixed_unit_rms": True,
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
GRADIENT_GROUP_NAMES = ("backbone", "dfm", "jepa", "other", "all")
TARGET_VARIANCE_HINGE_EPSILON = 1e-4
JEPA_STATE_RMS_EPSILON = 1e-6

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
    projector_num_heads: int = 8
    projector_mlp_dim: int = 0
    jepa_condition_dim: int = 0
    dfm_layers: int = 4
    jepa_layers: int = 4
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
    first_legality_coeff: float = 7.64
    horizon_legality_coeff: float = 0.0
    legality_on_masked_only: bool = True
    jepa_positive_coeff: float = 1.0
    jepa_loss_type: str = "raw_mse"
    jepa_target_mode: str = "projected_bt4"
    jepa_target_stop_gradient: bool = False
    jepa_target_semantics: str = "online"
    jepa_target_ema_decay: float = 0.99
    jepa_target_variance_hinge_coeff: float = 0.0
    jepa_target_variance_hinge_gamma: float = 0.9
    jepa_target_sample_count: int = 0
    jepa_gamma: float = 1.0
    jepa_sigreg_coeff: float = 0.1
    jepa_pred_sigreg_coeff: float = 0.0
    jepa_sigreg_kind: str = "le_jepa"
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
    skip_nonfinite_updates: bool = True
    unfreeze_bt4_encoder: bool = True
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
    "jepa_target_sample_count",
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


class TargetVarianceHingeResult(NamedTuple):
    loss: jax.Array
    valid_count_by_horizon: jax.Array
    eligible_by_horizon: jax.Array
    feature_std_mean_by_horizon: jax.Array
    feature_std_p05_by_horizon: jax.Array
    feature_std_median_by_horizon: jax.Array
    active_fraction_by_horizon: jax.Array
    hinge_by_horizon: jax.Array


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
    """LeJEPA Epps-Pulley SIGReg with random slices.

    This follows the official structure:
      1. Project latents onto random unit directions.
      2. Match each projected empirical characteristic function to N(0, 1).
      3. Average the Epps-Pulley statistic over slices.
    """
    z = jnp.asarray(z, dtype=jnp.float32)
    sample_count, dim = z.shape
    if sample_count == 0:
        return jnp.zeros((), dtype=jnp.float32)

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
    return jnp.where(
        global_sample_count > 0.0,
        jnp.mean(per_slice),
        jnp.zeros((), dtype=jnp.float32),
    )


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


class StateVectorProjector(nnx.Module):
    """Project BT4 square tokens to one global JEPA state vector."""

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        *,
        num_layers: int,
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
        seq = self.blocks(seq)
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
    """Joint DFM and recurrent projected-state JEPA model."""

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
        if config.jepa_loss_type != "raw_mse":
            raise ValueError(
                "Joint projected-vector training supports raw_mse JEPA loss only."
            )
        if config.jepa_target_mode not in ("projected_bt4", "current_repeat"):
            raise ValueError(
                f"Unsupported jepa_target_mode: {config.jepa_target_mode!r}."
            )
        if config.jepa_target_sample_count not in (0, config.horizon):
            raise ValueError(
                "Projected-vector JEPA currently uses full-horizon targets; "
                "set jepa_target_sample_count=0."
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
        return jnp.asarray(
            self.encoder.encode_tokens(planes),
            dtype=self.compute_dtype,
        )

    def encode_current_jepa(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        """Return projected current-state JEPA vectors."""

        return self.normalize_jepa_state(
            self.state_projector(self.encode_bt4_tokens(current_planes))
        )

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
        seq = self.dfm_blocks(seq)
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
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    actions = batch["action_indices"][:, : model.config.horizon]
    batch_size, horizon = actions.shape
    loss_horizon = horizon if model.config.loss_horizon <= 0 else min(model.config.loss_horizon, horizon)
    loss_horizon_mask = (jnp.arange(horizon) < loss_horizon).astype(jnp.float32)

    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(valid), 1.0)
    future_valid = jnp.asarray(batch["future_valid"], dtype=jnp.float32)[:, :horizon]
    future_planes = jnp.asarray(batch["future_planes"], dtype=jnp.float32)[:, :horizon]

    with jax.named_scope("joint_encode_project_current_future"):
        all_bt4_tokens, z_all = model.encode_current_and_future_tokens_and_vectors(
            batch["current_planes"],
            future_planes,
        )
    current_bt4_tokens = all_bt4_tokens[:, 0]
    z_jepa = z_all[:, 0]
    target_z = z_all[:, 1:]
    with jax.named_scope("joint_dfm_state_projector"):
        z_dfm = model.dfm_latents(current_bt4_tokens)

    rng_t, rng_mask, rng_sigreg = jax.random.split(rng, 3)
    t = jax.random.uniform(rng_t, shape=(batch_size,))
    if "deterministic_t" in batch:
        t = jnp.full_like(t, batch["deterministic_t"])
    noisy_actions, is_masked = mask_actions(actions, 1.0 - t, model.config.action_vocab_size, rng_mask)

    with jax.named_scope("joint_dfm_noisy_planner"):
        logits = model.planner_from_latents(z_dfm, noisy_actions, t)
    with jax.named_scope("joint_dfm_ce_loss"):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
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
            return model.jepa_rollout_from_latents(
                z_jepa,
                actions,
                clean_hidden["action_tokens"],
                z0_normalized=True,
            )

        pred_z = jax.lax.cond(teacher_forcing > 0.5, teacher_forced_rollout, free_rollout, operand=None)
    selected_horizons = jnp.arange(horizon, dtype=jnp.int32)
    pred_for_loss = pred_z
    future_valid_for_loss = future_valid
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
        sample_jepa = sample_raw_mse + sample_norm_loss
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
    jepa_sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
    valid_all = jnp.concatenate([valid[:, None], valid[:, None] * future_valid], axis=1)
    sigreg_weight = valid_all.reshape((-1,))
    sigreg_valid_count = jnp.sum(sigreg_weight)
    if model.config.jepa_sigreg_coeff != 0.0:
        with jax.named_scope("joint_jepa_sigreg"):
            sigreg_tokens = jnp.asarray(z_all, dtype=jnp.float32).reshape((-1, z_all.shape[-1]))
            if model.config.jepa_sigreg_kind == "moments":
                jepa_sigreg_loss = _sigreg_moments_loss(sigreg_tokens, sample_weight=sigreg_weight)
            elif model.config.jepa_sigreg_kind == "quantile":
                jepa_sigreg_loss = _quantile_sigreg_loss(
                    sigreg_tokens,
                    d_proj=model.config.jepa_sigreg_proj_dim,
                    rng=rng_sigreg,
                )
            elif model.config.jepa_sigreg_kind == "le_jepa":
                jepa_sigreg_loss = _official_le_jepa_sigreg_loss(
                    sigreg_tokens,
                    proj_dim=model.config.jepa_sigreg_proj_dim,
                    rng=rng_sigreg,
                    sample_weight=sigreg_weight,
                    axis_name=sigreg_axis_name,
                )
            else:
                raise ValueError(f"Unsupported jepa_sigreg_kind: {model.config.jepa_sigreg_kind!r}")
    jepa_pred_sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
    pred_sigreg_weight = (future_valid_for_loss * valid[:, None]).reshape((-1,))
    pred_sigreg_valid_count = jnp.sum(pred_sigreg_weight)
    if model.config.jepa_pred_sigreg_coeff != 0.0:
        with jax.named_scope("joint_jepa_pred_sigreg"):
            pred_sigreg_tokens = jnp.asarray(pred_z, dtype=jnp.float32).reshape((-1, pred_z.shape[-1]))
            if model.config.jepa_sigreg_kind == "moments":
                jepa_pred_sigreg_loss = _sigreg_moments_loss(pred_sigreg_tokens, sample_weight=pred_sigreg_weight)
            elif model.config.jepa_sigreg_kind == "quantile":
                jepa_pred_sigreg_loss = _quantile_sigreg_loss(
                    pred_sigreg_tokens,
                    d_proj=model.config.jepa_sigreg_proj_dim,
                    rng=rng_sigreg,
                )
            elif model.config.jepa_sigreg_kind == "le_jepa":
                jepa_pred_sigreg_loss = _official_le_jepa_sigreg_loss(
                    pred_sigreg_tokens,
                    proj_dim=model.config.jepa_sigreg_proj_dim,
                    rng=rng_sigreg,
                    sample_weight=pred_sigreg_weight,
                    axis_name=sigreg_axis_name,
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
    if model.config.value_coeff != 0.0 or model.config.wdl_coeff != 0.0:
        with jax.named_scope("joint_value_wdl_heads"):
            value_pred, wdl_logits = model.value_wdl_from_pred(pred_z)
        value_targets = jnp.asarray(batch.get("value_targets", jnp.zeros_like(value_pred)), dtype=jnp.float32)[:, :horizon]
        value_loss_by_horizon = jnp.square(value_pred - value_targets)
        value_loss = _weighted_horizon_mean(value_loss_by_horizon, future_valid * valid[:, None])
        wdl_targets = jnp.asarray(batch.get("wdl_targets", jnp.zeros_like(wdl_logits)), dtype=jnp.float32)[:, :horizon]
        wdl_sum = jnp.sum(wdl_targets, axis=-1, keepdims=True)
        wdl_valid = (wdl_sum[..., 0] > 0).astype(jnp.float32)
        wdl_targets = wdl_targets / jnp.maximum(wdl_sum, 1e-12)
        wdl_loss_by_horizon = -jnp.sum(wdl_targets * jax.nn.log_softmax(wdl_logits, axis=-1), axis=-1)
        wdl_loss = _weighted_horizon_mean(wdl_loss_by_horizon, future_valid * valid[:, None] * wdl_valid)
        value_pred_mean = jnp.mean(jnp.asarray(value_pred, dtype=jnp.float32))
        value_target_mean = jnp.mean(jnp.asarray(value_targets, dtype=jnp.float32))

    horizon_valid = future_valid_for_loss * valid[:, None]
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
    jepa_loss_by_horizon = horizon_num / jnp.maximum(horizon_denom, 1.0)
    jepa_raw_mse_by_horizon = horizon_raw_mse_num / jnp.maximum(horizon_denom, 1.0)
    jepa_norm_loss_by_horizon = horizon_norm_num / jnp.maximum(horizon_denom, 1.0)
    target_horizon_mask = jnp.zeros((horizon,), dtype=jnp.float32).at[selected_horizons].set(1.0)

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
        model.config.dfm_ce_coeff * dfm_ce_loss
        + weighted_legality_loss
        + model.config.jepa_positive_coeff * jepa_positive_loss
        + model.config.jepa_sigreg_coeff * jepa_sigreg_loss
        + model.config.jepa_pred_sigreg_coeff * jepa_pred_sigreg_loss
        + model.config.value_coeff * value_loss
        + model.config.wdl_coeff * wdl_loss
        + model.config.jepa_action_contrast_coeff * action_contrast_loss
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
        "jepa_target_sample_count": jnp.asarray(selected_horizons.shape[0], dtype=jnp.float32),
        "jepa_target_sample_fraction": jnp.asarray(selected_horizons.shape[0] / horizon, dtype=jnp.float32),
        "jepa_target_mean_horizon": jnp.mean(selected_horizons.astype(jnp.float32) + 1.0),
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



def create_joint_components(
    bt4_params: dict[str, Any],
    config: JointLatentSASAConfig,
    *,
    seed: int = 0,
) -> tuple[JointLatentSASAModel, nnx.Optimizer]:
    """Build the local checkpoint-compatible model and optimizer."""

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
    learning_rate = config.learning_rate
    bt4_learning_rate = config.bt4_learning_rate
    if config.lr_warmup_steps > 0:
        learning_rate = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=config.learning_rate,
                    transition_steps=config.lr_warmup_steps,
                ),
                optax.constant_schedule(config.learning_rate),
            ],
            boundaries=[config.lr_warmup_steps],
        )
        bt4_learning_rate = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=config.bt4_learning_rate,
                    transition_steps=config.lr_warmup_steps,
                ),
                optax.constant_schedule(config.bt4_learning_rate),
            ],
            boundaries=[config.lr_warmup_steps],
        )
    if config.use_muon:
        from chess_dfm_jax.nnx_bt4 import muon_adamw

        main_tx = muon_adamw(
            learning_rate=learning_rate,
            weight_decay=config.weight_decay,
        )
        bt4_tx = muon_adamw(
            learning_rate=bt4_learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        main_tx = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=config.weight_decay,
        )
        bt4_tx = optax.adamw(
            learning_rate=bt4_learning_rate,
            weight_decay=config.weight_decay,
        )

    def label_one(path, _value):
        parts = [str(getattr(item, "key", item)) for item in path]
        name = "/".join(parts)
        if name.startswith("encoder/embedding") or name.startswith(
            "encoder/layers"
        ):
            return "bt4"
        return "main"

    def label_tree(params):
        return jax.tree_util.tree_map_with_path(label_one, params)

    tx = optax.multi_transform(
        {"main": main_tx, "bt4": bt4_tx},
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
    return model, optimizer


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
    if objective == "normalized" and config.jepa_sigreg_kind != "le_jepa":
        raise ValueError(
            "--objective normalized requires jepa_sigreg_kind='le_jepa'; "
            f"found {config.jepa_sigreg_kind!r}"
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
    objective_contract = {
        "name": objective,
        "jepa_target_semantics": config.jepa_target_semantics,
        "jepa_target_stop_gradient": bool(
            config.jepa_target_stop_gradient
        ),
        "target_sigreg_coeff": float(config.jepa_sigreg_coeff),
        "pred_sigreg_coeff": float(config.jepa_pred_sigreg_coeff),
        "target_sigreg_reference_count": float(sigreg_reference_count),
        "pred_sigreg_reference_count": float(sigreg_reference_count),
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
    return {
        "architecture_source": ARCHITECTURE_SOURCE,
        "model_config": serialized_model_config(config),
        "objective": objective_contract,
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
        positive_target_z,
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
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compatibility loss with corrected count scaling and legal bounds.

    The compatibility forward graph's official EP statistics are converted to
    fixed-reference discrepancies algebraically, preserving their exact
    gradients while removing valid-count scaling.
    """

    _, compatibility_aux = joint_stage1_loss_fn(
        model,
        batch,
        rng,
        compute_fp32_legality=True,
        positive_target_override=positive_target_override,
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

    target_normalized = target_official * (
        jnp.asarray(target_reference_count, dtype=jnp.float32) / jnp.maximum(target_count, 1.0)
    )
    pred_normalized = pred_official * (
        jnp.asarray(pred_reference_count, dtype=jnp.float32) / jnp.maximum(pred_count, 1.0)
    )

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
    components = [
        jnp.asarray(aux["dfm_ce_loss"], dtype=jnp.float32),
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
    return jnp.stack(components)


def gradient_component_names(
    config: JointLatentSASAConfig,
) -> tuple[str, ...]:
    """Return the audit ABI, adding enabled-only research components."""

    if config.jepa_target_variance_hinge_coeff != 0.0:
        return GRADIENT_COMPONENT_NAMES + (
            TARGET_VARIANCE_HINGE_COMPONENT,
        )
    return GRADIENT_COMPONENT_NAMES


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
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save every N updates in this invocation; zero disables periodic saves.",
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
    if args.gradient_audit and args.objective != "normalized":
        raise ValueError("--gradient-audit requires --objective normalized")
    if not 0.0 <= args.val_deterministic_t <= 1.0:
        raise ValueError("--val-deterministic-t must be in [0, 1]")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    if args.max_checkpoints < 0:
        raise ValueError("--max-checkpoints must be non-negative")
    if args.resume_from is not None and args.init != "exact":
        raise ValueError("--resume-from is an exact continuation and cannot use --init model-only")

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
    val_batches = FixedTrajectoryBatches(
        data_root / "val",
        batch_size=args.batch_size,
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
        import_result = import_legacy_checkpoint(
            source_checkpoint_path,
            model=model,
            optimizer=optimizer,
            expected_size_bytes=int(checkpoint_asset["state_npz_size_bytes"]),
            expected_sha256=str(checkpoint_asset["state_npz_sha256"]),
            expected_step=checkpoint_step,
            init_mode=source_init_mode,
        )
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
    restore_seconds = time.perf_counter() - restore_started
    initial_optimizer_step = int(optimizer.step[...])
    initial_research_update = research_update
    initial_data_cursor = next_data_cursor
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
                "steady_examples_per_second",
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
        compile_args = training_call_args(
            objective=args.objective,
            model=model,
            optimizer=optimizer,
            batch=compile_batch,
            rng=compile_rng,
            sigreg_reference_count=args.sigreg_reference_count,
            ema_target=ema_target,
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
                    first_update_seconds = update_seconds
                    if args.train_seconds > 0:
                        deadline = time.perf_counter() + args.train_seconds
                else:
                    steady_update_seconds.append(update_seconds)

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
                    **final_train_metrics,
                }
                metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_log.flush()
                if args.save_every > 0 and updates % args.save_every == 0:
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
    performance_seconds = steady_update_seconds_mean or first_update_seconds
    compiler_cost_analysis = compiler_cost_summary(compiler_cost_analysis_raw)
    performance = compiler_performance(compiler_cost_analysis, performance_seconds)
    encoded_boards_per_example = (
        config.horizon + 1
        + (
            config.horizon
            if config.jepa_target_semantics == "ema"
            else 0
        )
    )
    throughput = {
        "steady_examples_per_second": (
            None
            if steady_update_seconds_mean is None
            else args.batch_size / steady_update_seconds_mean
        ),
        "steady_encoded_boards_per_second": (
            None
            if steady_update_seconds_mean is None
            else args.batch_size
            * encoded_boards_per_example
            / steady_update_seconds_mean
        ),
        "encoded_boards_per_example": encoded_boards_per_example,
        "online_encoded_boards_per_example": config.horizon + 1,
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
