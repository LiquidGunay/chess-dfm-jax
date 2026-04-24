"""Token-level JEPA training components for LC0 BT4 models."""

from __future__ import annotations

import collections
import dataclasses
import os
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chess_dfm_jax.nnx_bt4 import BT4Model, EncoderLayer, TrainableParam, TrainableLayerNorm, TrainableEmbedding, make_bt4_model, swish
from chess_dfm_jax.policy import policy_index_to_move, move_to_policy_index

@dataclasses.dataclass
class JEPAConfig:
    token_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    encoder_dtype: str = "float16"
    head_param_dtype: str = "float32"
    head_compute_dtype: str = "float32"
    action_source: str = "best"
    action_vocab_size: int = 1858
    use_qk_gain: bool = False
    use_muon: bool = False
    sigreg_coeff: float = 1.0
    value_coeff: float = 1.0
    wdl_coeff: float = 1.0


class ActionMLP(nnx.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs, param_dtype=jnp.float32, compute_dtype=jnp.float32):
        self.embedding = TrainableEmbedding(vocab_size, embed_dim, rngs=rngs, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.dense1 = TrainableParam(jax.random.normal(rngs.params(), (embed_dim, hidden_dim), dtype=param_dtype)/np.sqrt(max(embed_dim, 1)))
        self.bias1 = TrainableParam(jnp.zeros((hidden_dim,), dtype=param_dtype))
        self.dense2 = TrainableParam(jax.random.normal(rngs.params(), (hidden_dim, output_dim), dtype=param_dtype)/np.sqrt(max(hidden_dim, 1)))
        self.bias2 = TrainableParam(jnp.zeros((output_dim,), dtype=param_dtype))
        self.compute_dtype = jnp.dtype(compute_dtype)

    def __call__(self, indices: jnp.ndarray) -> jnp.ndarray:
        x = self.embedding(indices)
        x = x @ jnp.asarray(self.dense1[...], dtype=self.compute_dtype) + jnp.asarray(self.bias1[...], dtype=self.compute_dtype)
        x = swish(x)
        x = x @ jnp.asarray(self.dense2[...], dtype=self.compute_dtype) + jnp.asarray(self.bias2[...], dtype=self.compute_dtype)
        return x


class ValuePredictionHead(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs, param_dtype=jnp.float32, compute_dtype=jnp.float32):
        self.dense1 = TrainableParam(jax.random.normal(rngs.params(), (input_dim, hidden_dim), dtype=param_dtype)/np.sqrt(max(input_dim, 1)))
        self.bias1 = TrainableParam(jnp.zeros((hidden_dim,), dtype=param_dtype))
        self.dense_q = TrainableParam(jax.random.normal(rngs.params(), (hidden_dim, 1), dtype=param_dtype)/np.sqrt(max(hidden_dim, 1)))
        self.bias_q = TrainableParam(jnp.zeros((1,), dtype=param_dtype))
        self.dense_wdl = TrainableParam(jax.random.normal(rngs.params(), (hidden_dim, 3), dtype=param_dtype)/np.sqrt(max(hidden_dim, 1)))
        self.bias_wdl = TrainableParam(jnp.zeros((3,), dtype=param_dtype))
        self.compute_dtype = jnp.dtype(compute_dtype)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = x @ jnp.asarray(self.dense1[...], dtype=self.compute_dtype) + jnp.asarray(self.bias1[...], dtype=self.compute_dtype)
        x = swish(x)
        q = x @ jnp.asarray(self.dense_q[...], dtype=self.compute_dtype) + jnp.asarray(self.bias_q[...], dtype=self.compute_dtype)
        wdl = x @ jnp.asarray(self.dense_wdl[...], dtype=self.compute_dtype) + jnp.asarray(self.bias_wdl[...], dtype=self.compute_dtype)
        return q.squeeze(-1), wdl


class TokenProjector(nnx.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
    ):
        self.w = TrainableParam(
            jax.random.normal(rngs.params(), (input_dim, output_dim), dtype=param_dtype)
            / np.sqrt(max(input_dim, 1))
        )
        self.compute_dtype = compute_dtype

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w = jnp.asarray(self.w[...], dtype=self.compute_dtype)
        return jnp.dot(x, w)


class TokenTransitionHead(nnx.Module):
    def __init__(self, encoder: BT4Model, config: JEPAConfig, *, rngs: nnx.Rngs):
        param_dtype = _parse_compute_dtype(config.head_param_dtype)
        compute_dtype = _parse_compute_dtype(config.head_compute_dtype)
        self.encoder = encoder
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.token_projector = TokenProjector(
            encoder.embedding_size,
            config.token_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.action_mlp = ActionMLP(
            vocab_size=config.action_vocab_size,
            embed_dim=128,
            hidden_dim=config.token_dim * 2,
            output_dim=config.token_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.value_head = ValuePredictionHead(
            input_dim=config.token_dim,
            hidden_dim=config.token_dim * 2,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        square_pos = jax.random.normal(rngs.params(), (64, config.token_dim), dtype=jnp.float32)
        square_pos = square_pos / np.sqrt(max(config.token_dim, 1))
        self.square_pos = TrainableParam(jnp.asarray(square_pos, dtype=jnp.dtype(param_dtype)))
        self.blocks = nnx.List(
            [
                EncoderLayer(
                    width=config.token_dim,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    rngs=rngs,
                    param_dtype=param_dtype,
                    compute_dtype=compute_dtype,
                    use_qk_gain=config.use_qk_gain,
                    use_xsa=config.use_xsa,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = TrainableLayerNorm(
            config.token_dim,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )

    def encode_state_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        encoder_tokens = self.encoder.encode_tokens(planes)
        projected = self.token_projector(encoder_tokens)
        square_pos = jnp.asarray(self.square_pos[...], dtype=self.compute_dtype)
        return jnp.asarray(projected, dtype=self.compute_dtype) + square_pos[None, :, :]

    def predict_next(self, tokens: jnp.ndarray, action_idx: jnp.ndarray) -> jnp.ndarray:
        # tokens: [B, 64, D]
        # action_idx: [B]
        action_token = self.action_mlp(action_idx)
        # Condition board tokens by adding action embedding: [B, 64, D]
        seq = tokens + action_token[:, None, :]
        for block in self.blocks:
            seq = block(seq)
        # Return predicted next tokens
        return self.output_norm(seq)

    def __call__(
        self,
        current_planes: jnp.ndarray,
        action_indices: jnp.ndarray,
        next_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # action_indices: [B, K] where K is horizon
        current_tokens = self.encode_state_tokens(current_planes)
        target_tokens = jax.lax.stop_gradient(self.encode_state_tokens(next_planes))
        
        # Unroll predictor for K steps
        def loop_body(tokens, action_idx):
            next_tokens = self.predict_next(tokens, action_idx)
            return next_tokens, None
            
        # Swap axes to [K, B] for scan
        actions_seq = jnp.transpose(action_indices, (1, 0))
        final_pred, _ = jax.lax.scan(loop_body, current_tokens, actions_seq)
        
        z_pool = jnp.mean(final_pred, axis=1)
        q_pred, wdl_pred = self.value_head(z_pool)
        
        return final_pred, target_tokens, q_pred, wdl_pred


def _l2_normalize(x: jnp.ndarray, axis: int = -1, epsilon: float = 1e-12) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), epsilon)


def _sigreg_loss(z: jnp.ndarray, d_proj: int = 128, rng: jnp.ndarray | None = None) -> jnp.ndarray:
    if rng is None:
        rng = jax.random.PRNGKey(0)
    batch_size, dim = z.shape
    if batch_size == 0:
        return jnp.zeros(())
    W = jax.random.normal(rng, (dim, d_proj))
    W = W / jnp.maximum(jnp.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    z_proj = jnp.matmul(z, W)
    z_proj_sorted = jnp.sort(z_proj, axis=0)
    p = (jnp.arange(batch_size, dtype=jnp.float32) + 0.5) / batch_size
    from jax.scipy.special import ndtri
    target_quantiles = ndtri(p)
    target_quantiles = jnp.expand_dims(target_quantiles, axis=-1)
    return jnp.mean(jnp.square(z_proj_sorted - target_quantiles))

def transition_jepa_loss(
    model: LC0JEPA,
    batch: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    # batch["action_indices"]: [B, K]
    pred_tokens, target_tokens, q_pred, wdl_pred = model(
        batch["current_planes"],
        batch["action_indices"],
        batch["next_planes"],
    )
    pred_tokens_norm = _l2_normalize(jnp.asarray(pred_tokens, dtype=jnp.float32))
    target_tokens_norm = _l2_normalize(jnp.asarray(target_tokens, dtype=jnp.float32))
    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    
    # Cosine similarity between 64 tokens
    cosine = jnp.sum(pred_tokens_norm * target_tokens_norm, axis=-1)
    token_distance = 2.0 - 2.0 * cosine
    sample_sim_loss = jnp.mean(token_distance, axis=-1)
    
    denom = jnp.maximum(valid.sum(), 1.0)
    sim_loss = jnp.sum(sample_sim_loss * valid) / denom
    mean_cosine = jnp.sum(jnp.mean(cosine, axis=-1) * valid) / denom
    
    # SigReg Loss
    z_flat = pred_tokens.reshape((-1, pred_tokens.shape[-1]))
    sigreg = _sigreg_loss(z_flat)
    
    # Value Loss (MSE)
    value_target = jnp.asarray(batch.get("value_target", jnp.zeros_like(q_pred)), dtype=jnp.float32)
    sample_val_loss = jnp.square(q_pred - value_target)
    val_loss = jnp.sum(sample_val_loss * valid) / denom
    
    # WDL Loss (Cross-Entropy/KL Divergence)
    wdl_target = jnp.asarray(batch.get("wdl_target", jnp.zeros_like(wdl_pred)), dtype=jnp.float32)
    wdl_target = wdl_target / jnp.maximum(jnp.sum(wdl_target, axis=-1, keepdims=True), 1e-12)
    wdl_log_probs = jax.nn.log_softmax(wdl_pred, axis=-1)
    sample_wdl_loss = -jnp.sum(wdl_target * wdl_log_probs, axis=-1)
    wdl_loss = jnp.sum(sample_wdl_loss * valid) / denom
    
    total_loss = sim_loss + model.sigreg_coeff * sigreg + model.value_coeff * val_loss + model.wdl_coeff * wdl_loss
    
    aux = {
        "loss": total_loss,
        "jepa_loss": sim_loss,
        "sigreg_loss": sigreg,
        "val_loss": val_loss,
        "wdl_loss": wdl_loss,
        "valid_fraction": valid.mean(),
        "mean_token_cosine": mean_cosine,
        "pred_token_norm": jnp.mean(jnp.linalg.norm(pred_tokens, axis=-1)),
        "target_token_norm": jnp.mean(jnp.linalg.norm(target_tokens, axis=-1)),
    }
    return total_loss, aux


class LC0JEPA(nnx.Module):
    def __init__(self, encoder: BT4Model, config: JEPAConfig, *, rngs: nnx.Rngs):
        self.encoder = encoder
        self.transition = TokenTransitionHead(encoder, config, rngs=rngs)
        self.sigreg_coeff = config.sigreg_coeff
        self.value_coeff = config.value_coeff
        self.wdl_coeff = config.wdl_coeff

    def encode_state_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        return self.transition.encode_state_tokens(planes)

    def __call__(
        self,
        current_planes: jnp.ndarray,
        action_indices: jnp.ndarray,
        next_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.transition(current_planes, action_indices, next_planes)


_loss_and_grad = nnx.value_and_grad(
    transition_jepa_loss,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)


@nnx.jit
def train_step(model: LC0JEPA, optimizer: nnx.Optimizer, batch: dict[str, jnp.ndarray]):
    (loss, aux), grads = _loss_and_grad(model, batch)
    optimizer.update(model, grads)
    return loss, aux


def build_synthetic_transition_batch(batch_size: int, horizon: int = 1) -> dict[str, jnp.ndarray]:
    return {
        "current_planes": jnp.zeros((batch_size, 112, 8, 8), dtype=jnp.float32),
        "action_indices": jnp.zeros((batch_size, horizon), dtype=jnp.int32),
        "next_planes": jnp.zeros((batch_size, 112, 8, 8), dtype=jnp.float32),
        "valid": jnp.ones((batch_size,), dtype=jnp.float32),
        "value_target": jnp.zeros((batch_size,), dtype=jnp.float32),
        "wdl_target": jnp.zeros((batch_size, 3), dtype=jnp.float32),
        "legal_mask": jnp.ones((batch_size, 1858), dtype=jnp.float32),
    }


def build_transition_batch(
    raw_batch: dict[str, Any],
    action_source: str = "best",
) -> dict[str, jnp.ndarray]:
    return raw_batch


def _parse_compute_dtype(dtype_str: str) -> jnp.dtype:
    if dtype_str == "float16":
        return jnp.float16
    return jnp.float32


def create_jepa_components(
    bt4_params: dict,
    config: JEPAConfig,
    *,
    seed: int = 0,
) -> tuple[LC0JEPA, nnx.Optimizer]:
    encoder_dtype = _parse_compute_dtype(config.encoder_dtype)
    encoder = make_bt4_model(bt4_params, dtype=encoder_dtype)
    model = LC0JEPA(encoder, config, rngs=nnx.Rngs(seed))
    from chess_dfm_jax.nnx_bt4 import muon_adamw
    if config.use_muon:
        tx = muon_adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    else:
        tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=TrainableParam)
    return model, optimizer

def extract_train_state(model: LC0JEPA, optimizer: nnx.Optimizer) -> dict[str, Any]:
    def _to_numpy(x):
        if isinstance(x, jax.Array):
            return np.asarray(x)
        return x

    state_model = nnx.state(model, TrainableParam)
    state_opt = nnx.state(optimizer.opt_state)
    
    state_model = jax.tree.map(_to_numpy, state_model)
    state_opt = jax.tree.map(_to_numpy, state_opt)

    return {
        "step": np.asarray(int(optimizer.step[...]), dtype=np.int64),
        "model_trainable": dict(nnx.to_pure_dict(state_model)),
        "optimizer_state": dict(nnx.to_pure_dict(state_opt)),
    }

def restore_train_state(payload: dict[str, Any], model: nnx.Module, optimizer: nnx.Optimizer | None = None) -> int:
    model_state = nnx.state(model, TrainableParam)
    nnx.replace_by_pure_dict(model_state, payload["model_trainable"])
    nnx.update(model, model_state)

    if optimizer is not None:
        opt_state = nnx.state(optimizer.opt_state)
        nnx.replace_by_pure_dict(opt_state, payload["optimizer_state"])
        nnx.update(optimizer.opt_state, opt_state)
        optimizer.step[...] = jnp.asarray(payload["step"], dtype=optimizer.step[...].dtype)
        return int(optimizer.step[...])
    return int(payload["step"])


__all__ = [
    "JEPAConfig",
    "LC0JEPA",
    "TokenTransitionHead",
    "TrainableParam",
    "build_synthetic_transition_batch",
    "build_transition_batch",
    "create_jepa_components",
    "extract_train_state",
    "restore_train_state",
    "train_step",
    "transition_jepa_loss",
]
