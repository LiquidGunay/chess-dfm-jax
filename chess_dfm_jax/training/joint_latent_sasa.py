"""Joint Latent-SASA DFM + JEPA training components."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chess_dfm_jax.nnx_bt4 import (
    BT4Model,
    TrainableEmbedding,
    TrainableParam,
    TrainableRMSNorm,
    TrainableTransformerStack,
    make_bt4_model,
    rounded_swiglu_dim,
)
from chess_dfm_jax.training.dfm import mask_actions, refine_actions_from_latents
from chess_dfm_jax.training.jepa import _l2_normalize, _parse_compute_dtype, _sigreg_loss as _quantile_sigreg_loss


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
    jepa_state_rms_scale_max: float = 2.0
    jepa_teacher_forcing_steps: int = 0
    jepa_delta_rms_clip: float = 0.5
    remat_blocks: bool = True
    scan_layers: bool = False


def _weighted_horizon_mean(sample_values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(sample_values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _select_jepa_sample_loss(
    *,
    raw_mse: jnp.ndarray,
    cosine_distance: jnp.ndarray,
    normalized_mse: jnp.ndarray,
    loss_type: str,
) -> jnp.ndarray:
    if loss_type == "raw_mse":
        return raw_mse
    if loss_type == "cosine":
        return cosine_distance
    if loss_type == "normalized_mse":
        return normalized_mse
    raise ValueError(f"Unsupported jepa_loss_type: {loss_type!r}")


def _sigreg_moments_loss(z: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Cheap SigReg surrogate without per-dimension sorting."""
    z = jnp.asarray(z, dtype=jnp.float32)
    if sample_weight is None:
        sample_weight = jnp.ones((z.shape[0],), dtype=jnp.float32)
    sample_weight = jnp.maximum(jnp.asarray(sample_weight, dtype=jnp.float32), 0.0)
    denom = jnp.maximum(jnp.sum(sample_weight), 1.0)
    weight = sample_weight[:, None]
    mean = jnp.sum(z * weight, axis=0) / denom
    centered = z - mean
    variance = jnp.sum(jnp.square(centered) * weight, axis=0) / denom
    return jnp.mean(jnp.square(mean)) + jnp.mean(jnp.square(variance - 1.0))


def _sync_rng_for_pmap(rng: jnp.ndarray, axis_name: str | None) -> jnp.ndarray:
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
    directions = jax.random.normal(rng, (dim, proj_dim), dtype=jnp.float32)
    directions = directions / jnp.maximum(jnp.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    if sample_weight is None:
        sample_weight = jnp.ones((sample_count,), dtype=jnp.float32)
    sample_weight = jnp.maximum(jnp.asarray(sample_weight, dtype=jnp.float32), 0.0)

    projected = z @ directions

    t = jnp.linspace(0.0, t_max, n_points, dtype=jnp.float32)
    dt = jnp.asarray(t_max / max(n_points - 1, 1), dtype=jnp.float32)
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
        global_sample_count = jax.lax.psum(global_sample_count, axis_name=axis_name)
    denom = jnp.maximum(global_sample_count, 1.0)
    cos_mean = cos_sum / denom
    sin_mean = sin_sum / denom

    err = jnp.square(cos_mean - phi[None, :]) + jnp.square(sin_mean)
    per_slice = (err @ weights) * global_sample_count
    return jnp.where(global_sample_count > 0.0, jnp.mean(per_slice), jnp.zeros((), dtype=jnp.float32))


def _clip_loss_preserve_gradient(loss: jnp.ndarray, clip_value: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scale an oversized scalar loss without making its gradient exactly zero."""
    loss = jnp.asarray(loss, dtype=jnp.float32)
    if clip_value <= 0.0:
        return loss, jnp.asarray(1.0, dtype=jnp.float32)
    clip = jnp.asarray(clip_value, dtype=jnp.float32)
    stopped_loss = jax.lax.stop_gradient(jnp.maximum(loss, 1e-6))
    scale = jnp.minimum(1.0, clip / stopped_loss)
    return loss * scale, scale


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
            jax.random.normal(rngs.params(), (input_dim, z_dim), dtype=param_dtype) / np.sqrt(max(input_dim, 1))
        )
        self.in_bias = TrainableParam(jnp.zeros((z_dim,), dtype=param_dtype))
        self.cls = TrainableParam(jnp.zeros((1, z_dim), dtype=param_dtype))
        self.pos_embed = TrainableParam(
            jax.random.normal(rngs.params(), (65, z_dim), dtype=param_dtype) * (0.02 / np.sqrt(max(z_dim, 1)))
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
            tokens.reshape((batch * 64, self.input_dim)) @ jnp.asarray(self.in_proj[...], dtype=self.compute_dtype)
            + jnp.asarray(self.in_bias[...], dtype=self.compute_dtype)
        ).reshape((batch, 64, self.z_dim))
        cls = jnp.broadcast_to(jnp.asarray(self.cls[...], dtype=self.compute_dtype), (batch, 1, self.z_dim))
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
            jax.random.normal(rngs.params(), (input_dim, hidden_dim), dtype=param_dtype) / np.sqrt(max(input_dim, 1))
        )
        self.b1 = TrainableParam(jnp.zeros((hidden_dim,), dtype=param_dtype))
        self.value_w = TrainableParam(
            jax.random.normal(rngs.params(), (hidden_dim, 1), dtype=param_dtype) / np.sqrt(max(hidden_dim, 1))
        )
        self.value_b = TrainableParam(jnp.zeros((1,), dtype=param_dtype))
        self.wdl_w = TrainableParam(
            jax.random.normal(rngs.params(), (hidden_dim, 3), dtype=param_dtype) / np.sqrt(max(hidden_dim, 1))
        )
        self.wdl_b = TrainableParam(jnp.zeros((3,), dtype=param_dtype))

    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        z = jnp.asarray(z, dtype=self.compute_dtype)
        hidden = z @ jnp.asarray(self.w1[...], dtype=self.compute_dtype) + jnp.asarray(self.b1[...], dtype=self.compute_dtype)
        hidden = jax.nn.silu(hidden)
        value = hidden @ jnp.asarray(self.value_w[...], dtype=self.compute_dtype) + jnp.asarray(
            self.value_b[...], dtype=self.compute_dtype
        )
        wdl = hidden @ jnp.asarray(self.wdl_w[...], dtype=self.compute_dtype) + jnp.asarray(
            self.wdl_b[...], dtype=self.compute_dtype
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
        self.cond_w = TrainableParam(jnp.zeros((num_layers, condition_dim, 2 * z_dim), dtype=param_dtype))
        self.cond_b = TrainableParam(jnp.zeros((num_layers, 2 * z_dim), dtype=param_dtype))
        self.w_gate_up = TrainableParam(normal((num_layers, z_dim, 2 * self.swiglu_dim), 1.0 / np.sqrt(max(z_dim, 1))))
        self.b_gate_up = TrainableParam(jnp.zeros((num_layers, 2 * self.swiglu_dim), dtype=param_dtype))
        self.w_down = TrainableParam(
            normal((num_layers, self.swiglu_dim, z_dim), init_scale / np.sqrt(max(self.swiglu_dim, 1)))
        )
        self.b_down = TrainableParam(jnp.zeros((num_layers, z_dim), dtype=param_dtype))

    def _rms_norm(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        stats_x = jnp.asarray(x, dtype=jnp.float32)
        inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(stats_x), axis=-1, keepdims=True) + self.rms_eps)
        out = stats_x * inv_rms * jnp.asarray(scale, dtype=jnp.float32)
        return jnp.asarray(out, dtype=self.compute_dtype)

    def _layer(self, z: jnp.ndarray, condition: jnp.ndarray, params: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
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
            delta_rms = jnp.sqrt(jnp.mean(jnp.square(delta_f32), axis=-1, keepdims=True) + self.rms_eps)
            delta_scale = jnp.minimum(1.0, jnp.asarray(self.delta_rms_clip, dtype=jnp.float32) / delta_rms)
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

        z, _ = jax.lax.scan(body, jnp.asarray(z, dtype=self.compute_dtype), params)
        return z


class JointLatentSASAModel(nnx.Module):
    """Minimal joint model for positive hidden-state DFM/JEPA coupling."""

    def __init__(self, encoder: BT4Model, config: JointLatentSASAConfig, *, rngs: nnx.Rngs):
        self.encoder = encoder
        self.config = config
        param_dtype = _parse_compute_dtype(config.param_dtype)
        compute_dtype = _parse_compute_dtype(config.compute_dtype)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.encoder_dim = int(encoder.embedding_size)
        self.z_dim = int(config.z_dim)
        if config.jepa_loss_type != "raw_mse":
            raise ValueError("Joint projected-vector training supports raw_mse JEPA loss only.")
        if config.jepa_target_mode not in ("projected_bt4", "current_repeat"):
            raise ValueError(f"Unsupported jepa_target_mode: {config.jepa_target_mode!r}.")
        if config.jepa_target_sample_count not in (0, config.horizon):
            raise ValueError("Projected-vector JEPA currently uses full-horizon targets; set jepa_target_sample_count=0.")
        projector_heads = config.projector_num_heads
        if config.z_dim % projector_heads != 0:
            raise ValueError(f"z_dim={config.z_dim} must be divisible by projector_num_heads={projector_heads}.")
        condition_dim = config.jepa_condition_dim if config.jepa_condition_dim > 0 else config.z_dim
        jepa_mlp_dim = config.jepa_mlp_dim if config.jepa_mlp_dim > 0 else config.z_dim * 4
        projector_mlp_dim = config.projector_mlp_dim if config.projector_mlp_dim > 0 else config.z_dim * 4

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
        self.jepa_state_norm = TrainableRMSNorm(config.z_dim, param_dtype=param_dtype, compute_dtype=compute_dtype)
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
        self.time_embed1 = TrainableParam(jax.random.normal(rngs.params(), (1, config.token_dim)) / np.sqrt(1))
        self.time_embed2 = TrainableParam(
            jax.random.normal(rngs.params(), (config.token_dim, config.token_dim)) / np.sqrt(config.token_dim)
        )
        self.time_bias = TrainableParam(jnp.zeros((config.token_dim,)))
        self.pos_embed = TrainableParam(
            jax.random.normal(rngs.params(), (config.horizon, config.token_dim)) / np.sqrt(config.token_dim)
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
        self.dfm_out_norm = TrainableRMSNorm(config.token_dim, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.out_proj = TrainableParam(
            jax.random.normal(rngs.params(), (config.token_dim, config.action_vocab_size), dtype=param_dtype)
            / np.sqrt(config.token_dim)
        )
        self.out_bias = TrainableParam(jnp.zeros((config.action_vocab_size,), dtype=param_dtype))

    def encode_bt4_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self.encoder.encode_tokens(planes), dtype=self.compute_dtype)

    def encode_current_jepa(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        """Return projected current-state JEPA vectors."""
        return self.normalize_jepa_state(self.state_projector(self.encode_bt4_tokens(current_planes)))

    def encode_current_targets(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        return self.encode_current_jepa(current_planes)

    def encode_future_targets(self, future_planes: jnp.ndarray) -> jnp.ndarray:
        batch_size, horizon, channels, height, width = future_planes.shape
        flat_planes = future_planes.reshape((batch_size * horizon, channels, height, width))
        target_vectors = self.state_projector(self.encode_bt4_tokens(flat_planes))
        target_vectors = target_vectors.reshape((batch_size, horizon, self.z_dim))
        return self.normalize_jepa_state(target_vectors)

    def encode_current_and_future_targets(
        self,
        current_planes: jnp.ndarray,
        future_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        _, z_all = self.encode_current_and_future_tokens_and_vectors(current_planes, future_planes)
        return z_all[:, 0], z_all[:, 1:]

    def encode_current_and_future_tokens_and_vectors(
        self,
        current_planes: jnp.ndarray,
        future_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, horizon, channels, height, width = future_planes.shape
        all_planes = jnp.concatenate((current_planes[:, None, :, :, :], future_planes), axis=1)
        chunk_size = int(self.config.bt4_encode_chunk_size)
        if chunk_size == 1:
            time_major_planes = jnp.swapaxes(all_planes, 0, 1)

            def encode_one(planes_t):
                tokens_t = self.encode_bt4_tokens(planes_t)
                vectors_t = self.normalize_jepa_state(self.state_projector(tokens_t))
                return tokens_t, vectors_t

            # With trainable BT4, a plain scan still saves every horizon's
            # attention activations for backward. Remat keeps the scan memory
            # proportional to one board-state encode, at the cost of recompute.
            encode_one_fn = jax.checkpoint(encode_one, prevent_cse=False)

            def scan_body(_, planes_t):
                tokens_t, vectors_t = encode_one_fn(planes_t)
                return None, (tokens_t, vectors_t)

            _, (tokens_t, vectors_t) = jax.lax.scan(scan_body, None, time_major_planes, unroll=1)
            return jnp.swapaxes(tokens_t, 0, 1), jnp.swapaxes(vectors_t, 0, 1)

        flat_planes = all_planes.reshape((batch_size * (horizon + 1), channels, height, width))
        encoder_tokens = self.encode_bt4_tokens(flat_planes)
        tokens = encoder_tokens.reshape((batch_size, horizon + 1, 64, self.encoder_dim))
        vectors = self.state_projector(tokens.reshape((batch_size * (horizon + 1), 64, self.encoder_dim)))
        vectors = vectors.reshape((batch_size, horizon + 1, self.z_dim))
        return tokens, self.normalize_jepa_state(vectors)

    def normalize_jepa_state(self, z: jnp.ndarray) -> jnp.ndarray:
        """Shared bounded RMSNorm for the JEPA state manifold."""
        z = jnp.asarray(z, dtype=self.compute_dtype)
        if not self.config.jepa_state_rmsnorm:
            return z
        stats_z = jnp.asarray(z, dtype=jnp.float32)
        inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(stats_z), axis=-1, keepdims=True) + self.jepa_state_norm.eps)
        scale = jnp.asarray(self.jepa_state_norm.scale[...], dtype=jnp.float32)
        if self.config.jepa_state_rms_scale_max > 0.0:
            max_scale = jnp.asarray(self.config.jepa_state_rms_scale_max, dtype=jnp.float32)
            scale = jnp.clip(scale, 1.0 / max_scale, max_scale)
        out = stats_z * inv_rms * scale
        return jnp.asarray(out, dtype=self.compute_dtype)

    def dfm_latents(self, bt4_tokens: jnp.ndarray) -> jnp.ndarray:
        return self.dfm_state_projector(jnp.asarray(bt4_tokens, dtype=self.compute_dtype))

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
        action_emb = action_emb + jnp.asarray(self.pos_embed[...], dtype=self.compute_dtype)[None, :horizon, :]
        action_emb = action_emb + self.get_time_embedding(t)[:, None, :]
        seq = jnp.concatenate([z_dfm, action_emb], axis=1)
        seq = self.dfm_blocks(seq)
        state_hidden = seq[:, :64, :]
        action_hidden = seq[:, 64:, :]
        logits = self.logits_from_action_hidden(action_hidden)
        if return_hidden:
            return logits, {"state_tokens": state_hidden, "action_tokens": action_hidden}
        return logits

    def logits_from_action_hidden(self, action_hidden: jnp.ndarray) -> jnp.ndarray:
        action_out = self.dfm_out_norm(action_hidden)
        return (
            action_out @ jnp.asarray(self.out_proj[...], dtype=self.compute_dtype)
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
        action_hidden_seq = jnp.transpose(jnp.asarray(action_hidden, dtype=self.compute_dtype), (1, 0, 2))
        actions_seq = jnp.transpose(actions, (1, 0))

        def loop_body(z, inputs):
            action_idx, hidden = inputs
            condition = self.jepa_action_embed(action_idx) + self.jepa_hidden_adapter(hidden)
            next_z = self.jepa_transition(z, condition)
            next_z = self.normalize_jepa_state(next_z)
            return next_z, next_z

        z0 = jnp.asarray(z0_jepa, dtype=self.compute_dtype) if z0_normalized else self.normalize_jepa_state(z0_jepa)
        _, pred_seq = jax.lax.scan(loop_body, z0, (actions_seq, action_hidden_seq))
        return jnp.transpose(pred_seq, (1, 0, 2))

    def jepa_teacher_forced_from_latents(
        self,
        z_context: jnp.ndarray,
        actions: jnp.ndarray,
        action_hidden: jnp.ndarray,
    ) -> jnp.ndarray:
        """One-step JEPA predictions using true previous latents as inputs."""
        z_context_seq = jnp.transpose(jnp.asarray(z_context, dtype=self.compute_dtype), (1, 0, 2))
        action_hidden_seq = jnp.transpose(jnp.asarray(action_hidden, dtype=self.compute_dtype), (1, 0, 2))
        actions_seq = jnp.transpose(actions, (1, 0))

        def loop_body(_, inputs):
            z_in, action_idx, hidden = inputs
            condition = self.jepa_action_embed(action_idx) + self.jepa_hidden_adapter(hidden)
            next_z = self.jepa_transition(z_in, condition)
            next_z = self.normalize_jepa_state(next_z)
            return None, next_z

        _, pred_seq = jax.lax.scan(loop_body, None, (z_context_seq, actions_seq, action_hidden_seq))
        return jnp.transpose(pred_seq, (1, 0, 2))

    def value_wdl_from_pred(self, pred_z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, horizon, z_dim = pred_z.shape
        value, wdl = self.value_wdl_head(pred_z.reshape((batch_size * horizon, z_dim)))
        return value.reshape((batch_size, horizon)), wdl.reshape((batch_size, horizon, 3))


def legal_mass_from_indices(
    probs: jnp.ndarray,
    legal_idx: jnp.ndarray,
    legal_count: jnp.ndarray,
) -> jnp.ndarray:
    safe_idx = jnp.clip(jnp.asarray(legal_idx, dtype=jnp.int32), 0, probs.shape[-1] - 1)
    legal_probs = jnp.take_along_axis(probs, safe_idx, axis=-1)
    slot_valid = jnp.arange(safe_idx.shape[-1], dtype=jnp.int32) < jnp.asarray(legal_count, dtype=jnp.int32)[..., None]
    return jnp.sum(jnp.where(slot_valid, legal_probs, 0.0), axis=-1)


def sample_legal_prefix_candidates(
    actions: jnp.ndarray,
    legal_idx: jnp.ndarray,
    legal_count: jnp.ndarray,
    rng: jnp.ndarray,
    *,
    candidate_count: int,
    mask_token_id: int,
) -> dict[str, jnp.ndarray]:
    """Sample positive plus legal-prefix-corrupted candidate action chunks.

    Candidate 0 is always the stored positive sequence. Candidates 1..K-1
    replace one sampled horizon action with a different legal action and mask
    all later horizons. Invalid alternatives are marked in `candidate_valid`.
    """
    if candidate_count < 1:
        raise ValueError(f"candidate_count must be >= 1, got {candidate_count}.")
    batch_size, horizon = actions.shape
    positive = actions[:, None, :]
    positive_eval = jnp.full((batch_size, 1), horizon - 1, dtype=jnp.int32)
    positive_valid = jnp.ones((batch_size, 1), dtype=jnp.float32)
    if candidate_count == 1:
        return {
            "candidates": positive,
            "eval_horizon": positive_eval,
            "candidate_valid": positive_valid,
            "positive_index": jnp.zeros((batch_size,), dtype=jnp.int32),
        }

    negative_count = candidate_count - 1
    rng_h, rng_alt = jax.random.split(rng)
    corrupt_h = jax.random.randint(rng_h, (batch_size, negative_count), 0, horizon)
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    legal_at_h = legal_idx[batch_indices, corrupt_h, :]
    legal_count_h = legal_count[batch_indices, corrupt_h]
    true_at_h = actions[batch_indices, corrupt_h]

    slots = jnp.arange(legal_at_h.shape[-1], dtype=jnp.int32)
    slot_valid = slots[None, None, :] < legal_count_h[:, :, None]
    true_match = (legal_at_h == true_at_h[:, :, None]) & slot_valid
    true_found = jnp.any(true_match, axis=-1)
    true_pos = jnp.argmax(true_match, axis=-1)

    effective_slots = jnp.where(true_found, legal_count_h - 1, legal_count_h)
    raw_max = jnp.maximum(effective_slots, 1)
    raw = jnp.floor(jax.random.uniform(rng_alt, (batch_size, negative_count)) * raw_max).astype(jnp.int32)
    alt_pos = raw + jnp.where(true_found & (raw >= true_pos), 1, 0)
    alt_pos = jnp.clip(alt_pos, 0, legal_at_h.shape[-1] - 1)
    alt_action = jnp.take_along_axis(legal_at_h, alt_pos[:, :, None], axis=-1)[:, :, 0]
    neg_valid = ((effective_slots > 0) & (alt_action != true_at_h)).astype(jnp.float32)

    base = jnp.broadcast_to(actions[:, None, :], (batch_size, negative_count, horizon))
    positions = jnp.arange(horizon, dtype=jnp.int32)
    after_corruption = positions[None, None, :] > corrupt_h[:, :, None]
    at_corruption = positions[None, None, :] == corrupt_h[:, :, None]
    negatives = jnp.where(after_corruption, mask_token_id, base)
    negatives = jnp.where(at_corruption, alt_action[:, :, None], negatives)

    return {
        "candidates": jnp.concatenate([positive, negatives], axis=1),
        "eval_horizon": jnp.concatenate([positive_eval, corrupt_h], axis=1),
        "candidate_valid": jnp.concatenate([positive_valid, neg_valid], axis=1),
        "positive_index": jnp.zeros((batch_size,), dtype=jnp.int32),
    }


def joint_stage1_loss_fn(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    sigreg_axis_name: str | None = None,
    compute_fp32_legality: bool = False,
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
    if model.config.jepa_target_mode == "current_repeat":
        # Profiling-only target: removes future-state information from JEPA.
        # This is not a meaningful training objective.
        with jax.named_scope("joint_jepa_current_repeat_target"):
            target_vectors = jnp.broadcast_to(z_jepa[:, None, :], pred_for_loss.shape)
    else:
        target_vectors = target_z
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
    jepa_state_rms_scale_raw = jnp.asarray(model.jepa_state_norm.scale[...], dtype=jnp.float32)
    jepa_state_rms_scale = jepa_state_rms_scale_raw
    if model.config.jepa_state_rms_scale_max > 0.0:
        jepa_state_rms_scale_cap = jnp.asarray(model.config.jepa_state_rms_scale_max, dtype=jnp.float32)
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
    return loss, aux


def joint_jepa_positive_loss_fn(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Positive JEPA loss only, used for coupling-gradient diagnostics."""
    actions = batch["action_indices"][:, : model.config.horizon]
    _, horizon = actions.shape
    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    future_valid = jnp.asarray(batch["future_valid"], dtype=jnp.float32)[:, :horizon]
    future_planes = jnp.asarray(batch["future_planes"], dtype=jnp.float32)[:, :horizon]
    with jax.named_scope("joint_diag_encode_current_future_bt4"):
        all_bt4_tokens, z_all = model.encode_current_and_future_tokens_and_vectors(batch["current_planes"], future_planes)
    z_jepa = z_all[:, 0]
    target = z_all[:, 1:]
    with jax.named_scope("joint_diag_dfm_state_projector"):
        z_dfm = model.dfm_latents(all_bt4_tokens[:, 0])

    clean_t = jnp.ones((actions.shape[0],), dtype=jnp.float32)
    _, clean_hidden = model.planner_from_latents(z_dfm, actions, clean_t, return_hidden=True)
    pred_z = model.jepa_rollout_from_latents(
        z_jepa,
        actions,
        clean_hidden["action_tokens"],
        z0_normalized=True,
    )

    sample_raw_mse = jnp.mean(
        (jnp.asarray(pred_z, dtype=jnp.float32) - jnp.asarray(target, dtype=jnp.float32)) ** 2,
        axis=-1,
    )
    pred_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(pred_z, dtype=jnp.float32)), axis=-1) + 1e-6)
    target_rms = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(target, dtype=jnp.float32)), axis=-1) + 1e-6))
    sample_norm_loss = jnp.abs(jnp.log(pred_rms) - jnp.log(target_rms))
    pred_norm = _l2_normalize(jnp.asarray(pred_z, dtype=jnp.float32))
    target_norm = _l2_normalize(jnp.asarray(target, dtype=jnp.float32))
    cosine = jnp.sum(pred_norm * target_norm, axis=-1)
    sample_cosine_distance = 2.0 - 2.0 * cosine
    sample_normalized_mse = jnp.mean((pred_norm - target_norm) ** 2, axis=-1)
    sample_jepa = sample_raw_mse + sample_norm_loss
    horizon_weights = model.config.jepa_gamma ** jnp.arange(horizon, dtype=jnp.float32)
    jepa_mask = future_valid * valid[:, None] * horizon_weights[None, :]
    denom = jnp.maximum(jnp.sum(jepa_mask), 1.0)
    loss = jnp.sum(sample_jepa * jepa_mask) / denom
    cosine_loss = jnp.sum(sample_cosine_distance * jepa_mask) / denom
    mean_token_cosine = jnp.sum(cosine * jepa_mask) / denom
    raw_mse = jnp.sum(sample_raw_mse * jepa_mask) / denom
    norm_loss = jnp.sum(sample_norm_loss * jepa_mask) / denom
    normalized_mse = jnp.sum(sample_normalized_mse * jepa_mask) / denom
    return loss, {
        "jepa_positive_loss": loss,
        "jepa_cosine_loss": cosine_loss,
        "mean_token_cosine": mean_token_cosine,
        "jepa_raw_mse": raw_mse,
        "jepa_norm_loss": norm_loss,
        "jepa_normalized_mse": normalized_mse,
    }


def joint_jepa_action_baseline_diagnostics(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compare true-action JEPA rollout against shuffled-action and identity baselines."""
    actions = batch["action_indices"][:, : model.config.horizon]
    batch_size, horizon = actions.shape
    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    future_valid = jnp.asarray(batch["future_valid"], dtype=jnp.float32)[:, :horizon]
    future_planes = jnp.asarray(batch["future_planes"], dtype=jnp.float32)[:, :horizon]
    with jax.named_scope("joint_diag_encode_current_future_bt4"):
        all_bt4_tokens, z_all = model.encode_current_and_future_tokens_and_vectors(batch["current_planes"], future_planes)
    z_jepa = z_all[:, 0]
    target_vectors = z_all[:, 1:]
    with jax.named_scope("joint_diag_dfm_state_projector"):
        z_dfm = model.dfm_latents(all_bt4_tokens[:, 0])
    target_norm = _l2_normalize(jnp.asarray(target_vectors, dtype=jnp.float32))

    clean_t = jnp.ones((batch_size,), dtype=jnp.float32)
    _, true_hidden = model.planner_from_latents(z_dfm, actions, clean_t, return_hidden=True)
    true_pred = model.jepa_rollout_from_latents(
        z_jepa,
        actions,
        true_hidden["action_tokens"],
        z0_normalized=True,
    )

    permutation = jax.random.permutation(rng, batch_size)
    shuffled_actions = actions[permutation]
    _, shuffled_hidden = model.planner_from_latents(z_dfm, shuffled_actions, clean_t, return_hidden=True)
    shuffled_pred = model.jepa_rollout_from_latents(
        z_jepa,
        shuffled_actions,
        shuffled_hidden["action_tokens"],
        z0_normalized=True,
    )

    identity_pred = jnp.broadcast_to(z_jepa[:, None, :], target_vectors.shape)

    horizon_weights = model.config.jepa_gamma ** jnp.arange(horizon, dtype=jnp.float32)
    mask = future_valid * valid[:, None] * horizon_weights[None, :]
    denom = jnp.maximum(jnp.sum(mask), 1.0)

    def score(pred_vectors: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pred_norm = _l2_normalize(jnp.asarray(pred_vectors, dtype=jnp.float32))
        cosine = jnp.sum(pred_norm * target_norm, axis=-1)
        loss_by_sample = 2.0 - 2.0 * cosine
        raw_mse_by_sample = jnp.mean(
            (jnp.asarray(pred_vectors, dtype=jnp.float32) - jnp.asarray(target_vectors, dtype=jnp.float32)) ** 2,
            axis=-1,
        )
        loss = jnp.sum(loss_by_sample * mask) / denom
        raw_mse = jnp.sum(raw_mse_by_sample * mask) / denom
        mean_cosine = jnp.sum(cosine * mask) / denom
        return loss, raw_mse, mean_cosine

    true_loss, true_raw_mse, true_cosine = score(true_pred)
    shuffled_loss, shuffled_raw_mse, shuffled_cosine = score(shuffled_pred)
    identity_loss, identity_raw_mse, identity_cosine = score(identity_pred)
    return {
        "jepa_diag_true_loss": true_loss,
        "jepa_diag_true_raw_mse": true_raw_mse,
        "jepa_diag_true_cosine": true_cosine,
        "jepa_diag_shuffled_loss": shuffled_loss,
        "jepa_diag_shuffled_raw_mse": shuffled_raw_mse,
        "jepa_diag_shuffled_cosine": shuffled_cosine,
        "jepa_diag_identity_loss": identity_loss,
        "jepa_diag_identity_raw_mse": identity_raw_mse,
        "jepa_diag_identity_cosine": identity_cosine,
        "jepa_diag_true_minus_shuffled": true_loss - shuffled_loss,
        "jepa_diag_true_minus_identity": true_loss - identity_loss,
    }


def joint_contrastive_loss_fn(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Deprecated no-op: contrastive/ranking is removed from the active plan."""
    del model, batch, rng
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return zero, {
        "contrastive_loss": zero,
        "contrastive_accuracy": zero,
        "contrastive_positive_similarity": zero,
        "contrastive_negative_similarity": zero,
        "contrastive_similarity_margin": zero,
        "contrastive_valid_fraction": zero,
    }


def joint_stage2_loss_fn(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    sigreg_axis_name: str | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    rng_stage1, rng_contrast = jax.random.split(rng)
    stage1_loss, aux = joint_stage1_loss_fn(model, batch, rng_stage1, sigreg_axis_name=sigreg_axis_name)
    contrastive_loss, contrast_aux = joint_contrastive_loss_fn(model, batch, rng_contrast)
    total_loss = stage1_loss + model.config.contrastive_coeff * contrastive_loss
    out = dict(aux)
    out.update(contrast_aux)
    out["stage1_loss"] = stage1_loss
    out["loss"] = total_loss
    return total_loss, out


_joint_stage1_loss_and_grad = nnx.value_and_grad(
    joint_stage1_loss_fn,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)

_joint_jepa_positive_loss_and_grad = nnx.value_and_grad(
    joint_jepa_positive_loss_fn,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)

_joint_stage2_loss_and_grad = nnx.value_and_grad(
    joint_stage2_loss_fn,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)


def joint_stage1_loss_fn_data_parallel(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    return joint_stage1_loss_fn(model, batch, rng, sigreg_axis_name="data")


def joint_stage2_loss_fn_data_parallel(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    return joint_stage2_loss_fn(model, batch, rng, sigreg_axis_name="data")


_joint_stage1_data_parallel_loss_and_grad = nnx.value_and_grad(
    joint_stage1_loss_fn_data_parallel,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)

_joint_stage2_data_parallel_loss_and_grad = nnx.value_and_grad(
    joint_stage2_loss_fn_data_parallel,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)


def _train_joint_stage1_step_impl(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    with jax.named_scope("joint_stage1_loss_and_grad"):
        (loss, aux), grads = _joint_stage1_loss_and_grad(model, batch, rng)
    with jax.named_scope("joint_stage1_optimizer_update"):
        optimizer.update(model, grads)
    return loss, aux


def _train_joint_stage2_step_impl(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    with jax.named_scope("joint_stage2_loss_and_grad"):
        (loss, aux), grads = _joint_stage2_loss_and_grad(model, batch, rng)
    with jax.named_scope("joint_stage2_optimizer_update"):
        optimizer.update(model, grads)
    return loss, aux


def _pmean_metrics(metrics: dict[str, jnp.ndarray], axis_name: str) -> dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(lambda value: jax.lax.pmean(value, axis_name), metrics)


def _train_joint_stage1_step_data_parallel_impl(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    with jax.named_scope("joint_stage1_data_parallel_loss_and_grad"):
        (loss, aux), grads = _joint_stage1_data_parallel_loss_and_grad(model, batch, rng)
    with jax.named_scope("joint_stage1_data_parallel_pmean"):
        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        aux = _pmean_metrics(aux, axis_name="data")
    with jax.named_scope("joint_stage1_data_parallel_optimizer_update"):
        optimizer.update(model, grads)
    return loss, aux


def _train_joint_stage2_step_data_parallel_impl(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    with jax.named_scope("joint_stage2_data_parallel_loss_and_grad"):
        (loss, aux), grads = _joint_stage2_data_parallel_loss_and_grad(model, batch, rng)
    with jax.named_scope("joint_stage2_data_parallel_pmean"):
        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        aux = _pmean_metrics(aux, axis_name="data")
    with jax.named_scope("joint_stage2_data_parallel_optimizer_update"):
        optimizer.update(model, grads)
    return loss, aux


def _eval_joint_stage1_step_data_parallel_impl(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    loss, aux = joint_stage1_loss_fn(model, batch, rng, sigreg_axis_name="data")
    return jax.lax.pmean(loss, axis_name="data"), _pmean_metrics(aux, axis_name="data")


def _eval_joint_stage2_step_data_parallel_impl(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    loss, aux = joint_stage2_loss_fn(model, batch, rng, sigreg_axis_name="data")
    return jax.lax.pmean(loss, axis_name="data"), _pmean_metrics(aux, axis_name="data")


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.zeros((), dtype=jnp.float32)
    total = jnp.zeros((), dtype=jnp.float32)
    for leaf in leaves:
        array = jnp.asarray(leaf, dtype=jnp.float32)
        total = total + jnp.sum(jnp.square(array))
    return jnp.sqrt(total)


def joint_coupling_gradient_diagnostics(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Measure whether JEPA loss reaches the intended DFM coupling path."""
    (loss, aux), grads = _joint_jepa_positive_loss_and_grad(model, batch)
    pure = nnx.to_pure_dict(grads)
    dfm_path = {
        key: pure[key]
        for key in (
            "action_embed",
            "dfm_state_projector",
            "dfm_blocks",
            "dfm_out_norm",
            "out_bias",
            "out_proj",
            "pos_embed",
            "time_bias",
            "time_embed1",
            "time_embed2",
        )
        if key in pure
    }
    out = {
        "coupling_jepa_positive_loss": loss,
        "coupling_mean_token_cosine": aux["mean_token_cosine"],
        "coupling_jepa_raw_mse": aux["jepa_raw_mse"],
        "coupling_jepa_normalized_mse": aux["jepa_normalized_mse"],
        "coupling_grad_norm_dfm_path": _tree_l2_norm(dfm_path),
        "coupling_grad_norm_state_projector": _tree_l2_norm(pure.get("state_projector", {})),
        "coupling_grad_norm_jepa_path": _tree_l2_norm(
            {
                key: pure[key]
                for key in ("jepa_hidden_adapter", "jepa_action_embed", "jepa_transition")
                if key in pure
            }
        ),
        "coupling_grad_norm_bt4_encoder": _tree_l2_norm(pure.get("encoder", {})),
    }
    return out


@nnx.jit
def train_joint_stage1_step(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _train_joint_stage1_step_impl(model, optimizer, batch, rng)


@nnx.jit(donate_argnums=(0, 1))
def train_joint_stage1_step_donated(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _train_joint_stage1_step_impl(model, optimizer, batch, rng)


@nnx.jit
def eval_joint_stage1_step(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return joint_stage1_loss_fn(model, batch, rng)


@nnx.jit
def train_joint_stage2_step(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _train_joint_stage2_step_impl(model, optimizer, batch, rng)


@nnx.jit(donate_argnums=(0, 1))
def train_joint_stage2_step_donated(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _train_joint_stage2_step_impl(model, optimizer, batch, rng)


@nnx.pmap(axis_name="data", in_axes=(None, None, 0, 0), out_axes=0)
def train_joint_stage1_step_data_parallel(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _train_joint_stage1_step_data_parallel_impl(model, optimizer, batch, rng)


@nnx.pmap(axis_name="data", in_axes=(None, None, 0, 0), out_axes=0)
def train_joint_stage2_step_data_parallel(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _train_joint_stage2_step_data_parallel_impl(model, optimizer, batch, rng)


@nnx.pmap(axis_name="data", in_axes=(None, 0, 0), out_axes=0)
def eval_joint_stage1_step_data_parallel(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _eval_joint_stage1_step_data_parallel_impl(model, batch, rng)


@nnx.pmap(axis_name="data", in_axes=(None, 0, 0), out_axes=0)
def eval_joint_stage2_step_data_parallel(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return _eval_joint_stage2_step_data_parallel_impl(model, batch, rng)


@nnx.jit
def eval_joint_stage2_step(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return joint_stage2_loss_fn(model, batch, rng)


def refine_joint_dfm_actions_from_current(
    model: JointLatentSASAModel,
    current_planes: jnp.ndarray,
    legal_mask: jnp.ndarray,
    refinement_steps: int,
) -> jnp.ndarray:
    """Run iterative DFM sampling for the joint model with one BT4 encode."""
    bt4_tokens = model.encode_bt4_tokens(current_planes)
    z_dfm = model.dfm_latents(bt4_tokens)
    return refine_actions_from_latents(model, z_dfm, legal_mask, refinement_steps)


def create_joint_components(
    bt4_params: dict[str, Any],
    config: JointLatentSASAConfig,
    *,
    seed: int = 0,
) -> tuple[JointLatentSASAModel, nnx.Optimizer]:
    encoder_dtype = _parse_compute_dtype(config.encoder_dtype)
    encoder = make_bt4_model(bt4_params, dtype=encoder_dtype, train_encoder=config.unfreeze_bt4_encoder)
    model = JointLatentSASAModel(encoder, config, rngs=nnx.Rngs(seed))
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

        main_tx = muon_adamw(learning_rate=learning_rate, weight_decay=config.weight_decay)
        bt4_tx = muon_adamw(learning_rate=bt4_learning_rate, weight_decay=config.weight_decay)
    else:
        main_tx = optax.adamw(learning_rate=learning_rate, weight_decay=config.weight_decay)
        bt4_tx = optax.adamw(learning_rate=bt4_learning_rate, weight_decay=config.weight_decay)

    def label_one(path, _value):
        parts = [str(getattr(item, "key", item)) for item in path]
        name = "/".join(parts)
        if name.startswith("encoder/embedding") or name.startswith("encoder/layers"):
            return "bt4"
        return "main"

    def label_tree(params):
        return jax.tree_util.tree_map_with_path(label_one, params)

    tx = optax.multi_transform({"main": main_tx, "bt4": bt4_tx}, label_tree)
    if config.skip_nonfinite_updates:
        tx = optax.apply_if_finite(tx, max_consecutive_errors=1_000_000_000)
    if config.grad_clip_norm > 0.0:
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), tx)
    optimizer = nnx.Optimizer(model, tx, wrt=TrainableParam)
    return model, optimizer


__all__ = [
    "JointLatentSASAConfig",
    "JointLatentSASAModel",
    "create_joint_components",
    "eval_joint_stage1_step",
    "eval_joint_stage1_step_data_parallel",
    "eval_joint_stage2_step",
    "eval_joint_stage2_step_data_parallel",
    "joint_contrastive_loss_fn",
    "joint_coupling_gradient_diagnostics",
    "joint_jepa_action_baseline_diagnostics",
    "joint_jepa_positive_loss_fn",
    "joint_stage1_loss_fn",
    "joint_stage2_loss_fn",
    "legal_mass_from_indices",
    "refine_joint_dfm_actions_from_current",
    "sample_legal_prefix_candidates",
    "train_joint_stage1_step",
    "train_joint_stage1_step_data_parallel",
    "train_joint_stage1_step_donated",
    "train_joint_stage2_step",
    "train_joint_stage2_step_data_parallel",
    "train_joint_stage2_step_donated",
]
