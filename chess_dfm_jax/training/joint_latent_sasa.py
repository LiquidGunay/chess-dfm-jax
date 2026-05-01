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
    EncoderLayer,
    TrainableEmbedding,
    TrainableLayerNorm,
    TrainableParam,
    make_bt4_model,
)
from chess_dfm_jax.training.dfm import mask_actions
from chess_dfm_jax.training.jepa import TokenProjector, _l2_normalize, _parse_compute_dtype, _sigreg_loss


@dataclasses.dataclass
class JointLatentSASAConfig:
    token_dim: int = 256
    dfm_layers: int = 4
    jepa_layers: int = 2
    jepa_num_heads: int = 0
    jepa_mlp_dim: int = 0
    num_heads: int = 4
    mlp_dim: int = 1024
    learning_rate: float = 3e-4
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
    jepa_gamma: float = 0.9
    jepa_sigreg_coeff: float = 0.0
    jepa_action_contrast_coeff: float = 0.0
    jepa_action_contrast_margin: float = 0.05
    contrastive_coeff: float = 0.0
    contrastive_temperature: float = 0.1
    candidate_count: int = 1
    use_qk_gain: bool = False
    use_muon: bool = False


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


class JointLatentSASAModel(nnx.Module):
    """Minimal joint model for positive hidden-state DFM/JEPA coupling."""

    def __init__(self, encoder: BT4Model, config: JointLatentSASAConfig, *, rngs: nnx.Rngs):
        self.encoder = encoder
        self.config = config
        param_dtype = _parse_compute_dtype(config.param_dtype)
        compute_dtype = _parse_compute_dtype(config.compute_dtype)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.encoder_dim = int(encoder.embedding_size)
        if config.jepa_loss_type != "raw_mse":
            raise ValueError("Joint stage-1 hot training supports raw_mse JEPA loss only.")
        jepa_num_heads = config.jepa_num_heads if config.jepa_num_heads > 0 else config.num_heads
        if self.encoder_dim % jepa_num_heads != 0:
            raise ValueError(
                f"Raw-BT4 JEPA width {self.encoder_dim} must be divisible by jepa_num_heads={jepa_num_heads}."
            )
        jepa_mlp_dim = config.jepa_mlp_dim if config.jepa_mlp_dim > 0 else self.encoder_dim * 4

        self.shared_projector = TokenProjector(
            encoder.embedding_size,
            config.token_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.dfm_adapter = LinearAdapter(
            config.token_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.jepa_action_embed = TrainableEmbedding(
            config.action_vocab_size + 1,
            self.encoder_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.jepa_action_adapter = LinearAdapter(
            config.token_dim,
            self.encoder_dim,
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

        self.dfm_blocks = nnx.List(
            [
                EncoderLayer(
                    width=config.token_dim,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    rngs=rngs,
                    param_dtype=param_dtype,
                    compute_dtype=compute_dtype,
                    use_qk_gain=config.use_qk_gain,
                )
                for _ in range(config.dfm_layers)
            ]
        )
        self.jepa_blocks = nnx.List(
            [
                EncoderLayer(
                    width=self.encoder_dim,
                    num_heads=jepa_num_heads,
                    mlp_dim=jepa_mlp_dim,
                    rngs=rngs,
                    param_dtype=param_dtype,
                    compute_dtype=compute_dtype,
                    use_qk_gain=config.use_qk_gain,
                )
                for _ in range(config.jepa_layers)
            ]
        )
        self.dfm_out_norm = TrainableLayerNorm(config.token_dim, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.jepa_out_norm = TrainableLayerNorm(self.encoder_dim, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.out_proj = TrainableParam(
            jax.random.normal(rngs.params(), (config.token_dim, config.action_vocab_size), dtype=param_dtype)
            / np.sqrt(config.token_dim)
        )
        self.out_bias = TrainableParam(jnp.zeros((config.action_vocab_size,), dtype=param_dtype))

    def encode_shared(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        encoder_tokens = jax.lax.stop_gradient(self.encoder.encode_tokens(current_planes))
        return self.shared_projector(encoder_tokens)

    def encode_current_jepa(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        """Return anchored raw BT4 tokens for the JEPA stream."""
        encoder_tokens = self.encoder.encode_tokens(current_planes)
        return jax.lax.stop_gradient(jnp.asarray(encoder_tokens, dtype=self.compute_dtype))

    def encode_current_targets(self, current_planes: jnp.ndarray) -> jnp.ndarray:
        return self.encode_current_jepa(current_planes)

    def encode_future_targets(self, future_planes: jnp.ndarray) -> jnp.ndarray:
        batch_size, horizon, channels, height, width = future_planes.shape
        flat_planes = future_planes.reshape((batch_size * horizon, channels, height, width))
        encoder_tokens = self.encoder.encode_tokens(flat_planes)
        target_tokens = encoder_tokens.reshape((batch_size, horizon, 64, self.encoder_dim))
        return jax.lax.stop_gradient(jnp.asarray(target_tokens, dtype=self.compute_dtype))

    def encode_current_and_future_targets(
        self,
        current_planes: jnp.ndarray,
        future_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, horizon, channels, height, width = future_planes.shape
        all_planes = jnp.concatenate((current_planes[:, None, :, :, :], future_planes), axis=1)
        flat_planes = all_planes.reshape((batch_size * (horizon + 1), channels, height, width))
        encoder_tokens = self.encoder.encode_tokens(flat_planes)
        tokens = encoder_tokens.reshape((batch_size, horizon + 1, 64, self.encoder_dim))
        tokens = jax.lax.stop_gradient(jnp.asarray(tokens, dtype=self.compute_dtype))
        return tokens[:, 0], tokens[:, 1:]

    def dfm_latents(self, shared_tokens: jnp.ndarray) -> jnp.ndarray:
        return self.dfm_adapter(jnp.asarray(shared_tokens, dtype=self.compute_dtype))

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
        for block in self.dfm_blocks:
            seq = block(seq)
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
    ) -> jnp.ndarray:
        action_hidden_seq = jnp.transpose(jnp.asarray(action_hidden, dtype=self.compute_dtype), (1, 0, 2))
        actions_seq = jnp.transpose(actions, (1, 0))

        def loop_body(tokens, inputs):
            action_idx, hidden = inputs
            action_token = self.jepa_action_embed(action_idx) + self.jepa_action_adapter(hidden)
            seq = tokens + action_token[:, None, :]
            for block in self.jepa_blocks:
                seq = block(seq)
            next_tokens = self.jepa_out_norm(seq)
            return next_tokens, next_tokens

        _, pred_seq = jax.lax.scan(loop_body, z0_jepa, (actions_seq, action_hidden_seq))
        return jnp.transpose(pred_seq, (1, 0, 2, 3))


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
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    actions = batch["action_indices"][:, : model.config.horizon]
    batch_size, horizon = actions.shape
    loss_horizon = horizon if model.config.loss_horizon <= 0 else min(model.config.loss_horizon, horizon)
    loss_horizon_mask = (jnp.arange(horizon) < loss_horizon).astype(jnp.float32)

    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(valid), 1.0)
    future_valid = jnp.asarray(batch["future_valid"], dtype=jnp.float32)[:, :horizon]
    future_planes = jnp.asarray(batch["future_planes"], dtype=jnp.float32)[:, :horizon]

    current_tokens = model.encode_current_jepa(batch["current_planes"])
    shared_tokens = model.shared_projector(current_tokens)
    z_dfm = model.dfm_latents(shared_tokens)
    z_jepa = current_tokens

    rng_t, rng_mask = jax.random.split(rng, 2)
    t = jax.random.uniform(rng_t, shape=(batch_size,))
    if "deterministic_t" in batch:
        t = jnp.full_like(t, batch["deterministic_t"])
    noisy_actions, is_masked = mask_actions(actions, 1.0 - t, model.config.action_vocab_size, rng_mask)

    logits = model.planner_from_latents(z_dfm, noisy_actions, t)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce_by_horizon = -jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
    loss_mask = jnp.asarray(is_masked, dtype=jnp.float32) * loss_horizon_mask[None, :]
    sample_ce = jnp.sum(ce_by_horizon * loss_mask, axis=-1) / jnp.maximum(jnp.sum(loss_mask, axis=-1), 1e-5)
    dfm_ce_loss = jnp.sum(sample_ce * valid) / denom

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
    horizon_legality_loss = jnp.asarray(0.0, dtype=jnp.float32)
    legality_loss = first_legality_loss
    weighted_legality_loss = model.config.first_legality_coeff * first_legality_loss

    clean_t = jnp.ones((batch_size,), dtype=jnp.float32)
    _, clean_hidden = model.planner_from_latents(z_dfm, actions, clean_t, return_hidden=True)
    pred_tokens = model.jepa_rollout_from_latents(z_jepa, actions, clean_hidden["action_tokens"])
    target_tokens = model.encode_future_targets(future_planes)
    sample_jepa = jnp.mean(
        jnp.mean(
            (jnp.asarray(pred_tokens, dtype=jnp.float32) - jnp.asarray(target_tokens, dtype=jnp.float32)) ** 2,
            axis=-1,
        ),
        axis=-1,
    )
    horizon_weights = model.config.jepa_gamma ** jnp.arange(horizon, dtype=jnp.float32)
    jepa_mask = future_valid * valid[:, None] * horizon_weights[None, :]
    jepa_positive_loss = _weighted_horizon_mean(sample_jepa, jepa_mask)
    jepa_raw_mse = jepa_positive_loss
    jepa_sigreg_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if model.config.jepa_sigreg_coeff != 0.0:
        jepa_sigreg_loss = _sigreg_loss(jnp.asarray(pred_tokens, dtype=jnp.float32).reshape((-1, pred_tokens.shape[-1])))

    horizon_valid = future_valid * valid[:, None]
    horizon_denom = jnp.maximum(jnp.sum(horizon_valid, axis=0), 1.0)
    jepa_loss_by_horizon = jnp.sum(sample_jepa * horizon_valid, axis=0) / horizon_denom
    jepa_raw_mse_by_horizon = jepa_loss_by_horizon

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

    loss = (
        model.config.dfm_ce_coeff * dfm_ce_loss
        + weighted_legality_loss
        + model.config.jepa_positive_coeff * jepa_positive_loss
        + model.config.jepa_sigreg_coeff * jepa_sigreg_loss
        + model.config.jepa_action_contrast_coeff * action_contrast_loss
    )

    preds = jnp.argmax(logits, axis=-1)
    accuracy = (
        jnp.sum((preds == actions) * loss_mask * valid[:, None])
        / jnp.maximum(jnp.sum(loss_mask * valid[:, None]), 1.0)
    )
    aux = {
        "loss": loss,
        "dfm_ce_loss": dfm_ce_loss,
        "legality_loss": legality_loss,
        "first_legality_loss": first_legality_loss,
        "horizon_legality_loss": horizon_legality_loss,
        "weighted_legality_loss": weighted_legality_loss,
        "first_legal_mass": 1.0 - first_legality_loss,
        "horizon_legal_mass": jnp.asarray(0.0, dtype=jnp.float32),
        "horizon_legality_evaluated": jnp.asarray(0.0, dtype=jnp.float32),
        "jepa_positive_loss": jepa_positive_loss,
        "jepa_cosine_loss": jepa_cosine_loss,
        "jepa_raw_mse": jepa_raw_mse,
        "jepa_normalized_mse": jepa_normalized_mse,
        "jepa_sigreg_loss": jepa_sigreg_loss,
        "jepa_action_contrast_loss": action_contrast_loss,
        "jepa_shuffled_loss": shuffled_jepa_loss,
        "jepa_shuffled_cosine_loss": shuffled_jepa_cosine_loss,
        "jepa_shuffled_mean_token_cosine": shuffled_mean_token_cosine,
        "jepa_true_minus_shuffled": zero,
        "jepa_loss_by_horizon": jepa_loss_by_horizon,
        "jepa_raw_mse_by_horizon": jepa_raw_mse_by_horizon,
        "mean_token_cosine": mean_token_cosine,
        "mean_token_cosine_by_horizon": mean_token_cosine_by_horizon,
        "pred_token_norm": pred_token_norm,
        "target_token_norm": target_token_norm,
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

    shared_tokens = model.encode_shared(batch["current_planes"])
    z_dfm = model.dfm_latents(shared_tokens)
    z_jepa = model.encode_current_jepa(batch["current_planes"])

    clean_t = jnp.ones((actions.shape[0],), dtype=jnp.float32)
    _, clean_hidden = model.planner_from_latents(z_dfm, actions, clean_t, return_hidden=True)
    pred_tokens = model.jepa_rollout_from_latents(z_jepa, actions, clean_hidden["action_tokens"])
    target_tokens = model.encode_future_targets(future_planes)

    raw_mse_by_token = jnp.mean(
        (jnp.asarray(pred_tokens, dtype=jnp.float32) - jnp.asarray(target_tokens, dtype=jnp.float32)) ** 2,
        axis=-1,
    )
    pred_norm = _l2_normalize(jnp.asarray(pred_tokens, dtype=jnp.float32))
    target_norm = _l2_normalize(jnp.asarray(target_tokens, dtype=jnp.float32))
    cosine = jnp.sum(pred_norm * target_norm, axis=-1)
    token_distance = 2.0 - 2.0 * cosine
    normalized_mse_by_token = jnp.mean((pred_norm - target_norm) ** 2, axis=-1)
    sample_cosine_distance = jnp.mean(token_distance, axis=-1)
    sample_raw_mse = jnp.mean(raw_mse_by_token, axis=-1)
    sample_normalized_mse = jnp.mean(normalized_mse_by_token, axis=-1)
    sample_jepa = _select_jepa_sample_loss(
        raw_mse=sample_raw_mse,
        cosine_distance=sample_cosine_distance,
        normalized_mse=sample_normalized_mse,
        loss_type=model.config.jepa_loss_type,
    )
    horizon_weights = model.config.jepa_gamma ** jnp.arange(horizon, dtype=jnp.float32)
    jepa_mask = future_valid * valid[:, None] * horizon_weights[None, :]
    denom = jnp.maximum(jnp.sum(jepa_mask), 1.0)
    loss = jnp.sum(sample_jepa * jepa_mask) / denom
    cosine_loss = jnp.sum(sample_cosine_distance * jepa_mask) / denom
    mean_token_cosine = jnp.sum(jnp.mean(cosine, axis=-1) * jepa_mask) / denom
    raw_mse = jnp.sum(sample_raw_mse * jepa_mask) / denom
    normalized_mse = jnp.sum(sample_normalized_mse * jepa_mask) / denom
    return loss, {
        "jepa_positive_loss": loss,
        "jepa_cosine_loss": cosine_loss,
        "mean_token_cosine": mean_token_cosine,
        "jepa_raw_mse": raw_mse,
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

    shared_tokens = model.encode_shared(batch["current_planes"])
    z_dfm = model.dfm_latents(shared_tokens)
    z_jepa = model.encode_current_jepa(batch["current_planes"])
    target_tokens = model.encode_future_targets(future_planes)
    target_norm = _l2_normalize(jnp.asarray(target_tokens, dtype=jnp.float32))

    clean_t = jnp.ones((batch_size,), dtype=jnp.float32)
    _, true_hidden = model.planner_from_latents(z_dfm, actions, clean_t, return_hidden=True)
    true_pred = model.jepa_rollout_from_latents(z_jepa, actions, true_hidden["action_tokens"])

    permutation = jax.random.permutation(rng, batch_size)
    shuffled_actions = actions[permutation]
    _, shuffled_hidden = model.planner_from_latents(z_dfm, shuffled_actions, clean_t, return_hidden=True)
    shuffled_pred = model.jepa_rollout_from_latents(z_jepa, shuffled_actions, shuffled_hidden["action_tokens"])

    current_target_tokens = model.encode_current_targets(batch["current_planes"])
    identity_pred = jnp.broadcast_to(current_target_tokens[:, None, :, :], target_tokens.shape)

    horizon_weights = model.config.jepa_gamma ** jnp.arange(horizon, dtype=jnp.float32)
    mask = future_valid * valid[:, None] * horizon_weights[None, :]
    denom = jnp.maximum(jnp.sum(mask), 1.0)

    def score(pred_tokens: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pred_norm = _l2_normalize(jnp.asarray(pred_tokens, dtype=jnp.float32))
        cosine = jnp.sum(pred_norm * target_norm, axis=-1)
        loss_by_sample = jnp.mean(2.0 - 2.0 * cosine, axis=-1)
        raw_mse_by_sample = jnp.mean(
            jnp.mean((jnp.asarray(pred_tokens, dtype=jnp.float32) - jnp.asarray(target_tokens, dtype=jnp.float32)) ** 2, axis=-1),
            axis=-1,
        )
        loss = jnp.sum(loss_by_sample * mask) / denom
        raw_mse = jnp.sum(raw_mse_by_sample * mask) / denom
        mean_cosine = jnp.sum(jnp.mean(cosine, axis=-1) * mask) / denom
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
    """Contrast true action chunks against legal prefix corruptions."""
    actions = batch["action_indices"][:, : model.config.horizon]
    batch_size, horizon = actions.shape
    candidates = sample_legal_prefix_candidates(
        actions,
        batch["legal_idx"][:, :horizon],
        batch["legal_count"][:, :horizon],
        rng,
        candidate_count=model.config.candidate_count,
        mask_token_id=model.config.action_vocab_size,
    )
    cand_actions = candidates["candidates"]
    candidate_count = cand_actions.shape[1]
    eval_horizon = candidates["eval_horizon"]
    candidate_valid = candidates["candidate_valid"]

    shared_tokens = model.encode_shared(batch["current_planes"])
    z_dfm = model.dfm_latents(shared_tokens)
    z_jepa = model.encode_current_jepa(batch["current_planes"])
    z_dfm_cand = jnp.broadcast_to(z_dfm[:, None, :, :], (batch_size, candidate_count, 64, model.config.token_dim))
    z_jepa_cand = jnp.broadcast_to(z_jepa[:, None, :, :], (batch_size, candidate_count, 64, model.encoder_dim))
    flat_z_dfm = z_dfm_cand.reshape((batch_size * candidate_count, 64, model.config.token_dim))
    flat_z_jepa = z_jepa_cand.reshape((batch_size * candidate_count, 64, model.encoder_dim))
    flat_actions = cand_actions.reshape((batch_size * candidate_count, horizon))

    clean_t = jnp.ones((batch_size * candidate_count,), dtype=jnp.float32)
    _, hidden = model.planner_from_latents(flat_z_dfm, flat_actions, clean_t, return_hidden=True)
    flat_pred = model.jepa_rollout_from_latents(flat_z_jepa, flat_actions, hidden["action_tokens"])
    pred = flat_pred.reshape((batch_size, candidate_count, horizon, 64, model.encoder_dim))
    target = model.encode_future_targets(jnp.asarray(batch["future_planes"], dtype=jnp.float32)[:, :horizon])

    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    cand_idx = jnp.arange(candidate_count, dtype=jnp.int32)[None, :]
    pred_eval = pred[batch_idx, cand_idx, eval_horizon]
    target_eval = target[batch_idx, eval_horizon]
    pred_norm = _l2_normalize(jnp.asarray(pred_eval, dtype=jnp.float32))
    target_norm = _l2_normalize(jnp.asarray(target_eval, dtype=jnp.float32))
    similarity = jnp.mean(jnp.sum(pred_norm * target_norm, axis=-1), axis=-1)

    future_valid = jnp.asarray(batch["future_valid"], dtype=jnp.float32)[:, :horizon]
    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    target_valid = future_valid[batch_idx, eval_horizon]
    candidate_valid = candidate_valid * target_valid * valid[:, None]
    row_valid = (candidate_valid[:, 0] > 0) & (jnp.sum(candidate_valid[:, 1:], axis=1) > 0)
    row_valid_f = row_valid.astype(jnp.float32)

    masked_logits = jnp.where(
        candidate_valid > 0,
        similarity / jnp.maximum(model.config.contrastive_temperature, 1e-6),
        -1.0e9,
    )
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    sample_loss = -log_probs[:, 0]
    denom = jnp.maximum(jnp.sum(row_valid_f), 1.0)
    contrastive_loss = jnp.sum(sample_loss * row_valid_f) / denom

    pred_choice = jnp.argmax(masked_logits, axis=-1)
    accuracy = jnp.sum((pred_choice == 0).astype(jnp.float32) * row_valid_f) / denom
    positive_similarity = jnp.sum(similarity[:, 0] * row_valid_f) / denom
    negative_valid = candidate_valid[:, 1:] * row_valid_f[:, None]
    negative_similarity = jnp.sum(similarity[:, 1:] * negative_valid) / jnp.maximum(jnp.sum(negative_valid), 1.0)
    margin = positive_similarity - negative_similarity
    return contrastive_loss, {
        "contrastive_loss": contrastive_loss,
        "contrastive_accuracy": accuracy,
        "contrastive_positive_similarity": positive_similarity,
        "contrastive_negative_similarity": negative_similarity,
        "contrastive_similarity_margin": margin,
        "contrastive_valid_fraction": jnp.mean(row_valid_f),
    }


def joint_stage2_loss_fn(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    rng_stage1, rng_contrast = jax.random.split(rng)
    stage1_loss, aux = joint_stage1_loss_fn(model, batch, rng_stage1)
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
            "dfm_adapter",
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
        "coupling_grad_norm_shared_projector": _tree_l2_norm(pure.get("shared_projector", {})),
        "coupling_grad_norm_jepa_path": _tree_l2_norm(
            {
                key: pure[key]
                for key in ("jepa_action_adapter", "jepa_action_embed", "jepa_blocks", "jepa_out_norm")
                if key in pure
            }
        ),
        "coupling_grad_norm_target_projector": _tree_l2_norm(pure.get("target_projector", {})),
    }
    return out


@nnx.jit
def train_joint_stage1_step(
    model: JointLatentSASAModel,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    (loss, aux), grads = _joint_stage1_loss_and_grad(model, batch, rng)
    optimizer.update(model, grads)
    return loss, aux


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
    (loss, aux), grads = _joint_stage2_loss_and_grad(model, batch, rng)
    optimizer.update(model, grads)
    return loss, aux


@nnx.jit
def eval_joint_stage2_step(
    model: JointLatentSASAModel,
    batch: dict[str, jnp.ndarray],
    rng: jnp.ndarray,
):
    return joint_stage2_loss_fn(model, batch, rng)


def create_joint_components(
    bt4_params: dict[str, Any],
    config: JointLatentSASAConfig,
    *,
    seed: int = 0,
) -> tuple[JointLatentSASAModel, nnx.Optimizer]:
    encoder_dtype = _parse_compute_dtype(config.encoder_dtype)
    encoder = make_bt4_model(bt4_params, dtype=encoder_dtype)
    model = JointLatentSASAModel(encoder, config, rngs=nnx.Rngs(seed))
    if config.use_muon:
        from chess_dfm_jax.nnx_bt4 import muon_adamw

        tx = muon_adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    else:
        tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=TrainableParam)
    return model, optimizer


__all__ = [
    "JointLatentSASAConfig",
    "JointLatentSASAModel",
    "create_joint_components",
    "eval_joint_stage1_step",
    "eval_joint_stage2_step",
    "joint_contrastive_loss_fn",
    "joint_coupling_gradient_diagnostics",
    "joint_jepa_action_baseline_diagnostics",
    "joint_jepa_positive_loss_fn",
    "joint_stage1_loss_fn",
    "joint_stage2_loss_fn",
    "legal_mass_from_indices",
    "sample_legal_prefix_candidates",
    "train_joint_stage1_step",
    "train_joint_stage2_step",
]
