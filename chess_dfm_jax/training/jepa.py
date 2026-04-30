"""Token-level JEPA training components for LC0 BT4 models."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chess_dfm_jax.data.trajectory import (
    build_synthetic_trajectory_shard,
    terminal_target_indices,
    trajectory_shard_to_batch,
)
from chess_dfm_jax.nnx_bt4 import (
    BT4Model,
    EncoderLayer,
    TrainableEmbedding,
    TrainableLayerNorm,
    TrainableParam,
    make_bt4_model,
    swish,
)

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
    use_xsa: bool = False
    use_muon: bool = False
    terminal_only: bool = False
    sigreg_coeff: float = 0.01
    value_coeff: float = 0.0
    wdl_coeff: float = 0.0


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
        self.jepa_width = int(encoder.embedding_size)
        if self.jepa_width % config.num_heads != 0:
            raise ValueError(
                f"Raw-BT4 JEPA width {self.jepa_width} must be divisible by num_heads={config.num_heads}."
            )
        hidden_dim = max(config.token_dim * 2, self.jepa_width)
        self.action_mlp = ActionMLP(
            vocab_size=config.action_vocab_size,
            embed_dim=128,
            hidden_dim=hidden_dim,
            output_dim=self.jepa_width,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.value_head = ValuePredictionHead(
            input_dim=self.jepa_width,
            hidden_dim=hidden_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )
        self.blocks = nnx.List(
            [
                EncoderLayer(
                    width=self.jepa_width,
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
            self.jepa_width,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
        )

    def encode_state_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        encoder_tokens = self.encoder.encode_tokens(planes)
        return jax.lax.stop_gradient(jnp.asarray(encoder_tokens, dtype=self.compute_dtype))

    def encode_future_tokens(self, future_planes: jnp.ndarray) -> jnp.ndarray:
        batch_size, horizon, channels, height, width = future_planes.shape
        flat_planes = future_planes.reshape((batch_size * horizon, channels, height, width))
        flat_tokens = self.encode_state_tokens(flat_planes)
        token_dim = flat_tokens.shape[-1]
        return flat_tokens.reshape((batch_size, horizon, 64, token_dim))

    def predict_next(
        self,
        tokens: jnp.ndarray,
        action_idx: jnp.ndarray,
        action_hidden: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        action_token = self.action_mlp(action_idx)
        if action_hidden is not None:
            action_token = action_token + jnp.asarray(action_hidden, dtype=self.compute_dtype)
        seq = tokens + action_token[:, None, :]
        for block in self.blocks:
            seq = block(seq)
        return self.output_norm(seq)

    def rollout_from_latents(
        self,
        z0_jepa: jnp.ndarray,
        action_indices: jnp.ndarray,
        action_hidden: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Predict future latent tokens from current latents and action context."""
        current_tokens = jnp.asarray(z0_jepa, dtype=self.compute_dtype)
        if action_hidden is None:
            action_hidden_seq = jnp.zeros(
                (
                    action_indices.shape[1],
                    action_indices.shape[0],
                    current_tokens.shape[-1],
                ),
                dtype=self.compute_dtype,
            )
        else:
            action_hidden_seq = jnp.transpose(jnp.asarray(action_hidden, dtype=self.compute_dtype), (1, 0, 2))

        def loop_body(tokens, inputs):
            action_idx, hidden = inputs
            next_tokens = self.predict_next(tokens, action_idx, hidden)
            return next_tokens, next_tokens

        actions_seq = jnp.transpose(action_indices, (1, 0))
        _, pred_tokens_seq = jax.lax.scan(loop_body, current_tokens, (actions_seq, action_hidden_seq))
        return jnp.transpose(pred_tokens_seq, (1, 0, 2, 3))

    def predict_sequence(self, current_planes: jnp.ndarray, action_indices: jnp.ndarray) -> jnp.ndarray:
        current_tokens = self.encode_state_tokens(current_planes)
        return self.rollout_from_latents(current_tokens, action_indices)

    def __call__(
        self,
        current_planes: jnp.ndarray,
        action_indices: jnp.ndarray,
        future_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pred_tokens_seq = self.predict_sequence(current_planes, action_indices)
        target_tokens_seq = jax.lax.stop_gradient(self.encode_future_tokens(future_planes))
        z_pool = jnp.mean(pred_tokens_seq, axis=2)
        q_pred, wdl_pred = self.value_head(z_pool)
        return pred_tokens_seq, target_tokens_seq, q_pred, wdl_pred


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
    pred_tokens, target_tokens, q_pred, wdl_pred = model(
        batch["current_planes"],
        batch["action_indices"],
        batch["future_planes"],
    )
    pred_tokens_norm = _l2_normalize(jnp.asarray(pred_tokens, dtype=jnp.float32))
    target_tokens_norm = _l2_normalize(jnp.asarray(target_tokens, dtype=jnp.float32))
    future_valid = jnp.asarray(batch["future_valid"], dtype=jnp.float32)
    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    if model.terminal_only:
        terminal_index = jnp.asarray(batch["terminal_target_index"], dtype=jnp.int32)
        terminal_mask = jax.nn.one_hot(
            terminal_index, future_valid.shape[1], dtype=jnp.float32
        )
        loss_mask = terminal_mask * valid[:, None]
    else:
        loss_mask = future_valid * valid[:, None]

    cosine = jnp.sum(pred_tokens_norm * target_tokens_norm, axis=-1)
    token_distance = 2.0 - 2.0 * cosine
    sample_sim_loss = jnp.mean(token_distance, axis=-1)

    denom = jnp.maximum(loss_mask.sum(), 1.0)
    sim_loss = jnp.sum(sample_sim_loss * loss_mask) / denom
    mean_cosine = jnp.sum(jnp.mean(cosine, axis=-1) * loss_mask) / denom

    z_flat = pred_tokens.reshape((-1, pred_tokens.shape[-1]))
    sigreg = _sigreg_loss(z_flat)

    value_target = jnp.asarray(
        batch.get("value_targets", jnp.zeros_like(q_pred)), dtype=jnp.float32
    )
    sample_val_loss = jnp.square(q_pred - value_target)
    val_loss = jnp.sum(sample_val_loss * loss_mask) / denom

    wdl_target = jnp.asarray(batch.get("wdl_targets", jnp.zeros_like(wdl_pred)), dtype=jnp.float32)
    wdl_target = wdl_target / jnp.maximum(jnp.sum(wdl_target, axis=-1, keepdims=True), 1e-12)
    wdl_log_probs = jax.nn.log_softmax(wdl_pred, axis=-1)
    sample_wdl_loss = -jnp.sum(wdl_target * wdl_log_probs, axis=-1)
    wdl_loss = jnp.sum(sample_wdl_loss * loss_mask) / denom

    total_loss = (
        sim_loss
        + model.sigreg_coeff * sigreg
        + model.value_coeff * val_loss
        + model.wdl_coeff * wdl_loss
    )

    aux = {
        "loss": total_loss,
        "jepa_loss": sim_loss,
        "sigreg_loss": sigreg,
        "val_loss": val_loss,
        "wdl_loss": wdl_loss,
        "valid_fraction": loss_mask.mean(),
        "mean_token_cosine": mean_cosine,
        "terminal_only": jnp.asarray(1.0 if model.terminal_only else 0.0, dtype=jnp.float32),
        "pred_token_norm": jnp.mean(jnp.linalg.norm(pred_tokens, axis=-1)),
        "target_token_norm": jnp.mean(jnp.linalg.norm(target_tokens, axis=-1)),
    }
    return total_loss, aux


class LC0JEPA(nnx.Module):
    def __init__(self, encoder: BT4Model, config: JEPAConfig, *, rngs: nnx.Rngs):
        self.encoder = encoder
        self.transition = TokenTransitionHead(encoder, config, rngs=rngs)
        self.terminal_only = config.terminal_only
        self.sigreg_coeff = config.sigreg_coeff
        self.value_coeff = config.value_coeff
        self.wdl_coeff = config.wdl_coeff

    def encode_state_tokens(self, planes: jnp.ndarray) -> jnp.ndarray:
        return self.transition.encode_state_tokens(planes)

    def predict_sequence(
        self,
        current_planes: jnp.ndarray,
        action_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.transition.predict_sequence(current_planes, action_indices)

    def jepa_rollout_from_latents(
        self,
        z0_jepa: jnp.ndarray,
        action_indices: jnp.ndarray,
        action_hidden: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self.transition.rollout_from_latents(z0_jepa, action_indices, action_hidden)

    def __call__(
        self,
        current_planes: jnp.ndarray,
        action_indices: jnp.ndarray,
        future_planes: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.transition(current_planes, action_indices, future_planes)


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


@nnx.jit
def eval_jepa_step(model: LC0JEPA, batch: dict[str, jnp.ndarray]):
    return transition_jepa_loss(model, batch)


def build_synthetic_transition_batch(batch_size: int, horizon: int = 1) -> dict[str, jnp.ndarray]:
    shard = build_synthetic_trajectory_shard(batch_size=batch_size, horizon=horizon)
    batch = trajectory_shard_to_batch(shard)
    return {key: jnp.asarray(value) for key, value in batch.items()}


def build_transition_batch(
    raw_batch: dict[str, Any],
    action_source: str = "best",
) -> dict[str, jnp.ndarray]:
    del action_source
    batch = dict(raw_batch)
    if "action_indices" not in batch:
        if "action_idx" not in batch:
            raise KeyError("Missing action_indices/action_idx in transition batch.")
        action_idx = np.asarray(batch["action_idx"], dtype=np.int32)
        if action_idx.ndim == 1:
            batch["action_indices"] = action_idx[:, None]
        else:
            batch["action_indices"] = action_idx.astype(np.int32)
    else:
        batch["action_indices"] = np.asarray(batch["action_indices"], dtype=np.int32)

    if "action_idx" not in batch:
        batch["action_idx"] = np.asarray(batch["action_indices"][:, 0], dtype=np.int32)

    if "future_planes" not in batch:
        if "next_planes" not in batch:
            raise KeyError("Missing future_planes/next_planes in transition batch.")
        next_planes = np.asarray(batch["next_planes"], dtype=np.float32)
        batch["future_planes"] = next_planes[:, None, ...]

    future_planes = np.asarray(batch["future_planes"], dtype=np.float32)
    batch["future_planes"] = future_planes
    horizon = future_planes.shape[1]

    if "future_valid" not in batch:
        batch["future_valid"] = np.ones((future_planes.shape[0], horizon), dtype=np.float32)
    else:
        batch["future_valid"] = np.asarray(batch["future_valid"], dtype=np.float32)

    if "terminal_target_index" not in batch:
        batch["terminal_target_index"] = terminal_target_indices(
            np.asarray(batch["future_valid"], dtype=np.float32)
        )

    terminal_idx = np.asarray(batch["terminal_target_index"], dtype=np.int32)
    batch["next_planes"] = future_planes[np.arange(future_planes.shape[0]), terminal_idx]
    batch["valid"] = np.asarray(
        batch.get("valid", (np.asarray(batch["future_valid"]).sum(axis=1) > 0).astype(np.float32)),
        dtype=np.float32,
    )

    if "value_targets" not in batch:
        if "value_target" in batch:
            value_target = np.asarray(batch["value_target"], dtype=np.float32)
            value_targets = np.zeros((value_target.shape[0], horizon), dtype=np.float32)
            value_targets[np.arange(value_target.shape[0]), terminal_idx] = value_target
            batch["value_targets"] = value_targets
        else:
            batch["value_targets"] = np.zeros((future_planes.shape[0], horizon), dtype=np.float32)
    else:
        batch["value_targets"] = np.asarray(batch["value_targets"], dtype=np.float32)
    batch["value_target"] = np.asarray(
        batch["value_targets"][np.arange(future_planes.shape[0]), terminal_idx], dtype=np.float32
    )

    if "wdl_targets" not in batch:
        if "wdl_target" in batch:
            wdl_target = np.asarray(batch["wdl_target"], dtype=np.float32)
            wdl_targets = np.zeros((wdl_target.shape[0], horizon, 3), dtype=np.float32)
            wdl_targets[np.arange(wdl_target.shape[0]), terminal_idx] = wdl_target
            batch["wdl_targets"] = wdl_targets
        else:
            batch["wdl_targets"] = np.zeros((future_planes.shape[0], horizon, 3), dtype=np.float32)
    else:
        batch["wdl_targets"] = np.asarray(batch["wdl_targets"], dtype=np.float32)
    batch["wdl_target"] = np.asarray(
        batch["wdl_targets"][np.arange(future_planes.shape[0]), terminal_idx], dtype=np.float32
    )

    if "legal_mask" not in batch:
        if "legal_masks" in batch:
            batch["legal_mask"] = np.asarray(batch["legal_masks"], dtype=np.float32)[:, 0]
        else:
            batch["legal_mask"] = np.ones((future_planes.shape[0], 1858), dtype=np.float32)
    else:
        batch["legal_mask"] = np.asarray(batch["legal_mask"], dtype=np.float32)

    return {key: jnp.asarray(value) for key, value in batch.items()}


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


def create_jepa_components(
    bt4_params: dict,
    config: JEPAConfig,
    *,
    seed: int = 0,
) -> tuple[LC0JEPA, nnx.Optimizer]:
    if config.use_xsa:
        raise NotImplementedError("use_xsa is reserved but not implemented in EncoderLayer.")
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
    "eval_jepa_step",
    "extract_train_state",
    "restore_train_state",
    "train_step",
    "transition_jepa_loss",
]
