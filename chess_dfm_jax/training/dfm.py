"""Discrete Flow Matching (Categorical Diffusion) Planner."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chess_dfm_jax.nnx_bt4 import BT4Model, EncoderLayer, TrainableParam, TrainableLayerNorm, TrainableEmbedding, make_bt4_model
from chess_dfm_jax.training.jepa import _parse_compute_dtype

@dataclasses.dataclass
class DFMConfig:
    token_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    encoder_dtype: str = "float16"
    compute_dtype: str = "float32"
    action_vocab_size: int = 1858 # Real actions are 0..1857. MASK token is 1858.
    horizon: int = 8
    use_qk_gain: bool = False
    use_xsa: bool = False
    use_muon: bool = False
    loss_horizon: int = 0
    first_legality_loss_weight: float = 2.0
    horizon_legality_loss_weight: float = 0.0
    first_action_loss_weight: float = 0.0

class DFMDenoiser(nnx.Module):
    def __init__(self, encoder: BT4Model, config: DFMConfig, *, rngs: nnx.Rngs):
        if config.use_xsa:
            raise NotImplementedError("use_xsa is reserved but not implemented for DFM.")
        self.encoder = encoder
        self.config = config
        compute_dtype = _parse_compute_dtype(config.compute_dtype)
        self.compute_dtype = compute_dtype
        
        # We need an embedding for actions (including MASK token)
        self.action_embed = TrainableEmbedding(
            config.action_vocab_size + 1, # +1 for MASK
            config.token_dim,
            rngs=rngs,
            param_dtype=jnp.float32,
            compute_dtype=compute_dtype
        )
        
        # Time embedding for diffusion timestep t
        self.time_embed1 = TrainableParam(jax.random.normal(rngs.params(), (1, config.token_dim)) / np.sqrt(1))
        self.time_embed2 = TrainableParam(jax.random.normal(rngs.params(), (config.token_dim, config.token_dim)) / np.sqrt(config.token_dim))
        self.time_bias = TrainableParam(jnp.zeros((config.token_dim,)))
        
        # Positional embedding for the action sequence
        self.pos_embed = TrainableParam(jax.random.normal(rngs.params(), (config.horizon, config.token_dim)) / np.sqrt(config.token_dim))
        
        # Projection from encoder dimension to token_dim (if needed)
        self.z_proj = TrainableParam(jax.random.normal(rngs.params(), (encoder.embedding_size, config.token_dim)) / np.sqrt(encoder.embedding_size))
        self.z_bias = TrainableParam(jnp.zeros((config.token_dim,)))

        # Transformer blocks (DFM cannot use XSA because action tokens must attend to their own positional/time embeddings)
        self.blocks = nnx.List([
            EncoderLayer(
                width=config.token_dim,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
                rngs=rngs,
                param_dtype=jnp.float32,
                compute_dtype=compute_dtype,
                use_qk_gain=config.use_qk_gain,
            ) for _ in range(config.num_layers)
        ])
        
        self.out_norm = TrainableLayerNorm(config.token_dim, compute_dtype=compute_dtype)
        self.out_proj = TrainableParam(jax.random.normal(rngs.params(), (config.token_dim, config.action_vocab_size)) / np.sqrt(config.token_dim))
        self.out_bias = TrainableParam(jnp.zeros((config.action_vocab_size,)))

    def get_time_embedding(self, t: jnp.ndarray) -> jnp.ndarray:
        # t is [B] in [0, 1]
        t = t[:, None] # [B, 1]
        t_emb = t @ self.time_embed1[...]
        t_emb = jax.nn.relu(t_emb)
        t_emb = t_emb @ self.time_embed2[...] + self.time_bias[...]
        return t_emb

    def __call__(self, current_planes: jnp.ndarray, noisy_actions: jnp.ndarray, t: jnp.ndarray):
        # current_planes: [B, 112, 8, 8]
        # noisy_actions: [B, K]
        # t: [B]
        
        K = noisy_actions.shape[1]
        
        # 1. Encode board state (frozen)
        z = jax.lax.stop_gradient(self.encoder.encode_tokens(current_planes)) # [B, 64, encoder_dim]
        z = z @ jnp.asarray(self.z_proj[...], dtype=self.compute_dtype) + jnp.asarray(self.z_bias[...], dtype=self.compute_dtype)
        
        # 2. Embed actions and add positional encoding
        a_emb = self.action_embed(noisy_actions) # [B, K, token_dim]
        a_emb = a_emb + jnp.asarray(self.pos_embed[...], dtype=self.compute_dtype)[None, :K, :]
        
        # 3. Time embedding
        t_emb = self.get_time_embedding(t) # [B, token_dim]
        
        # Add time embedding to action embeddings (could also use AdaLN, but addition is standard)
        a_emb = a_emb + t_emb[:, None, :]
        
        # 4. Concatenate board tokens and action tokens
        # Sequence: [Z_1...Z_64, A_1...A_K]
        seq = jnp.concatenate([z, a_emb], axis=1) # [B, 64 + K, token_dim]
        
        # 5. Apply Transformer
        for block in self.blocks:
            seq = block(seq)
            
        # 6. Extract action tokens and project to vocab
        a_out = seq[:, 64:, :] # [B, K, token_dim]
        a_out = self.out_norm(a_out)
        logits = a_out @ jnp.asarray(self.out_proj[...], dtype=self.compute_dtype) + jnp.asarray(self.out_bias[...], dtype=self.compute_dtype)
        
        return logits # [B, K, vocab_size]


def mask_actions(actions: jnp.ndarray, mask_prob: jnp.ndarray, mask_token_id: int, rng: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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

def dfm_loss_fn(model: DFMDenoiser, batch: dict[str, jnp.ndarray], rng: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    # batch["action_indices"]: [B, max_K]
    actions = batch["action_indices"][:, :model.config.horizon]
    batch_size, K = actions.shape
    loss_horizon = K if model.config.loss_horizon <= 0 else min(model.config.loss_horizon, K)
    loss_horizon_mask = (jnp.arange(K) < loss_horizon).astype(jnp.float32)
    
    # 1. Sample t ~ U(0, 1)
    rng_t, rng_mask = jax.random.split(rng)
    t = jax.random.uniform(rng_t, shape=(batch_size,))
    
    if "deterministic_t" in batch:
        t = jnp.full_like(t, batch["deterministic_t"])
    
    # 2. Discrete Flow Matching Probability Path P_t(x|x_1)
    # Target state x_1 is the clean token. Prior state x_0 is the [MASK] token.
    # Linear probability path: P(x_t = M) = 1 - t, P(x_t = x_1) = t
    mask_prob = 1.0 - t
    
    # 3. Create noisy actions (x_t)
    mask_token_id = model.config.action_vocab_size
    noisy_actions, is_masked = mask_actions(actions, mask_prob, mask_token_id, rng_mask)
    
    # 4. Predict clean actions (the vector field target)
    logits = model(batch["current_planes"], noisy_actions, t)
    
    # 5. Discrete Flow Matching Objective
    # DFM minimizes the categorical cross-entropy between the predicted target and the true x_1,
    # evaluated only on the tokens that are currently masked (x_t = M).
    one_hot_targets = jax.nn.one_hot(actions, model.config.action_vocab_size)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    ce_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1) # [B, K]
    
    mask_float = jnp.asarray(is_masked, dtype=jnp.float32)
    # The true DFM loss can be weighted by 1 / (1-t) depending on the exact formulation, 
    # but uniform weighting across masked tokens is standard and stable.
    loss_mask_float = mask_float * loss_horizon_mask[None, :]
    loss_masked = jnp.sum(ce_loss * loss_mask_float, axis=-1) / jnp.maximum(jnp.sum(loss_mask_float, axis=-1), 1e-5)
    
    # Average over batch
    valid = jnp.asarray(batch["valid"], dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(valid), 1.0)
    masked_ce_loss = jnp.sum(loss_masked * valid) / denom
    loss = masked_ce_loss
    
    # 6. Legality losses.
    # The legacy first-step term keeps compatibility with existing runs. The optional
    # horizon term uses teacher-forced legal masks for every available target ply.
    if "legal_mask" in batch:
        legal_mask = jnp.asarray(batch["legal_mask"], dtype=jnp.float32) # [B, V]
        illegal_mask = 1.0 - legal_mask # [B, V]
        
        logits_0 = logits[:, 0, :] # [B, V]
        probs_0 = jax.nn.softmax(logits_0, axis=-1)
        
        illegal_prob_mass = jnp.sum(probs_0 * illegal_mask, axis=-1) # [B]
        legality_loss = jnp.sum(illegal_prob_mass * valid) / denom
        
        loss = loss + model.config.first_legality_loss_weight * legality_loss
    else:
        legality_loss = jnp.zeros(())

    if "legal_masks" in batch and model.config.horizon_legality_loss_weight != 0.0:
        legal_masks = jnp.asarray(batch["legal_masks"], dtype=jnp.float32)[:, :K, :]
        probs = jax.nn.softmax(logits, axis=-1)
        illegal_prob_mass_by_horizon = jnp.sum(probs * (1.0 - legal_masks), axis=-1) # [B, K]
        if "legal_masks_valid" in batch:
            legal_valid = jnp.asarray(batch["legal_masks_valid"], dtype=jnp.float32)[:, :K]
        else:
            legal_valid = jnp.ones((batch_size, K), dtype=jnp.float32)
        continuation_mask = (jnp.arange(K) > 0).astype(jnp.float32) * loss_horizon_mask
        horizon_valid = valid[:, None] * legal_valid * continuation_mask[None, :]
        horizon_legality_loss = (
            jnp.sum(illegal_prob_mass_by_horizon * horizon_valid)
            / jnp.maximum(jnp.sum(horizon_valid), 1.0)
        )
        loss = loss + model.config.horizon_legality_loss_weight * horizon_legality_loss
    else:
        horizon_legality_loss = jnp.zeros(())

    first_action_loss = jnp.sum(ce_loss[:, 0] * valid) / denom
    loss = loss + model.config.first_action_loss_weight * first_action_loss
    
    # Metrics
    preds = jnp.argmax(logits, axis=-1)
    accuracy_masked = jnp.sum((preds == actions) * loss_mask_float, axis=-1) / jnp.maximum(jnp.sum(loss_mask_float, axis=-1), 1e-5)
    mean_accuracy = jnp.sum(accuracy_masked * valid) / denom
    mean_mask_prob = jnp.mean(mask_prob)

    horizon_denom = jnp.maximum(jnp.sum(loss_mask_float * valid[:, None], axis=0), 1e-5)
    loss_by_horizon = jnp.sum(ce_loss * loss_mask_float * valid[:, None], axis=0) / horizon_denom
    accuracy_by_horizon = jnp.sum((preds == actions) * loss_mask_float * valid[:, None], axis=0) / horizon_denom
    mask_rate_by_horizon = jnp.sum(loss_mask_float * valid[:, None], axis=0) / jnp.maximum(denom, 1e-5)

    aux = {
        "loss": loss,
        "masked_ce_loss": masked_ce_loss,
        "legality_loss": legality_loss,
        "horizon_legality_loss": horizon_legality_loss,
        "first_action_loss": first_action_loss,
        "loss_horizon": jnp.asarray(loss_horizon, dtype=jnp.float32),
        "accuracy": mean_accuracy,
        "mask_prob": mean_mask_prob,
        "loss_by_horizon": loss_by_horizon,
        "accuracy_by_horizon": accuracy_by_horizon,
        "mask_rate_by_horizon": mask_rate_by_horizon,
    }
    
    return loss, aux

_dfm_loss_and_grad = nnx.value_and_grad(
    dfm_loss_fn,
    argnums=nnx.DiffState(0, TrainableParam),
    has_aux=True,
)

@nnx.jit
def train_dfm_step(model: DFMDenoiser, optimizer: nnx.Optimizer, batch: dict[str, jnp.ndarray], rng: jnp.ndarray):
    (loss, aux), grads = _dfm_loss_and_grad(model, batch, rng)
    optimizer.update(model, grads)
    return loss, aux


@nnx.jit
def eval_dfm_step(model: DFMDenoiser, batch: dict[str, jnp.ndarray], rng: jnp.ndarray):
    return dfm_loss_fn(model, batch, rng)

def create_dfm_components(
    bt4_params: dict,
    config: DFMConfig,
    *,
    seed: int = 0,
) -> tuple[DFMDenoiser, nnx.Optimizer]:
    encoder_dtype = _parse_compute_dtype(config.encoder_dtype)
    encoder = make_bt4_model(bt4_params, dtype=encoder_dtype)
    model = DFMDenoiser(encoder, config, rngs=nnx.Rngs(seed))
    
    if config.use_muon:
        from chess_dfm_jax.nnx_bt4 import muon_adamw
        tx = muon_adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    else:
        tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        
    optimizer = nnx.Optimizer(model, tx, wrt=TrainableParam)
    return model, optimizer
