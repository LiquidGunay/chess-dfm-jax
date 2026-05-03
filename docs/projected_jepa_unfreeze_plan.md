# Projected JEPA Unfreeze Plan

Status: implemented as the active joint Latent-SASA direction on 2026-05-02.

This plan replaces the raw frozen-BT4 token JEPA target with a projected state
vector target and unfreezes the BT4 encoder from the start at a much smaller
learning rate. The motivation is that the previous SigReg path did not
propagate into the moving projector/encoder path we actually needed to
regularize, so the projected state space could collapse or drift without the
regularizer correcting it.

## Architecture

For each H8 trajectory sample:

```text
current_planes: [B, 112, 8, 8]
future_planes:  [B, H, 112, 8, 8]
actions:        [B, H]
```

The BT4 encoder is now trainable:

```text
E_bt4(s) -> tokens [B, 64, 1024]
```

Only the BT4 input embedding and encoder blocks are trainable. BT4 policy/value
heads stay outside the joint objective. BT4 architectural code must remain
checkpoint-compatible; XSA, QK norm, RMSNorm, SwiGLU, and pre-norm changes apply
only to the trainable DFM/projector/JEPA stack.

All current and future boards are encoded in one batched call:

```text
[B, H + 1, 112, 8, 8] -> [B * (H + 1), 112, 8, 8]
BT4 -> [B * (H + 1), 64, 1024]
```

The state projector maps BT4 square tokens into one global JEPA state vector:

```text
P(tokens) -> z [B, z_dim]
```

Implementation details:

- Linear projection from `1024` to `z_dim` per square token.
- Learned CLS token plus learned 65-position embedding.
- Trainable transformer stack with RMSNorm, SwiGLU, pre-norm residuals, QK norm,
  and XSA.
- Final RMSNorm on the CLS output.
- Default `z_dim = 2048`; sweep `1024`, `2048`, and `4096` if memory allows.

The DFM stays token-based:

```text
BT4(s_t) [B, 64, 1024]
  -> dfm_state_projector -> [B, 64, token_dim]
  -> DFM(noisy_actions, t)
  -> action logits [B, H, 1858]
  -> action_hidden [B, H, token_dim]
```

The JEPA transition is vector recurrent:

```text
z_hat_0 = z_t
condition_h = action_embed(a_h) + hidden_adapter(dfm_action_hidden_h)
z_hat_{h+1} = T(z_hat_h, condition_h)
```

`T` is an AdaLN-style residual MLP over vectors:

```text
RMSNorm(z)
condition -> shift/scale, zero-initialized
gate_up = x @ W_gate_up
gate, up = split(gate_up)
delta = (silu(gate) * up) @ W_down
z_next = z + delta
```

The down projection is initialized near zero so the transition starts close to
identity. The horizon recurrence uses `jax.lax.scan`.

## Loss

The active Stage 1 loss is:

```text
L =
    dfm_ce_coeff        * L_dfm_masked_ce
  + first_legality_coeff * L_first_move_illegal_mass
  + jepa_positive_coeff * L_jepa_mse
  + jepa_sigreg_coeff   * L_sigreg
  + value_coeff         * L_value
  + wdl_coeff           * L_wdl
```

Defaults:

```text
dfm_ce_coeff = 1.0
first_legality_coeff = 7.64
horizon_legality_coeff = 0.0
jepa_positive_coeff = 1.0
jepa_sigreg_coeff = 0.1
value_coeff = 0.05
wdl_coeff = 0.05
jepa_gamma = 1.0
```

`L_dfm_masked_ce` is cross entropy only on masked/noised action-token positions
under the uniform DFM noise schedule.

`L_first_move_illegal_mass` is the probability mass assigned to illegal first
moves from the stored legal index list. Later-ply legality remains disabled in
the hot training loss because later legality depends on the generated prefix,
not on the dataset's true future prefix.

`L_jepa_mse` is symmetric projected-state MSE:

```text
mean_h valid_h * mean((z_hat_{t+h} - z_{t+h})^2)
```

The target projector and BT4 encoder are not stopped. This is intentional: the
projected latent space is a learned space, so SigReg must be able to regularize
the projected states that participate in the MSE.

`L_sigreg` is LeJEPA-style random-projection quantile SigReg over all current
and future projected states in the batch:

```text
z_all [B, H + 1, z_dim] -> [B * (H + 1), z_dim]
```

Default projection dimension is `128`.

`L_value` is MSE from predicted JEPA states to game-result-derived scalar value
targets. `L_wdl` is WDL cross entropy from predicted JEPA states to
game-result-derived WDL targets. Current LC0 PGN-derived v3 shards build these
targets from final PGN result and side-to-move at each future board.

Contrastive/ranking loss is removed from the active plan. It remains a future
reranking direction only after projected JEPA dynamics are demonstrably useful.

## Optimizer

Use two learning-rate groups:

```text
BT4 embedding/encoder: 1e-5
all other trainable params: 6e-4
```

Both groups use the same warmup schedule. The default warmup is `1000` steps.
Muon is enabled by default for square-ish 2D matrices; AdamW handles biases,
norms, embeddings, FFN expansion/contraction matrices, and other non-square or
non-2D leaves. Default global gradient clipping is `1.0`, and non-finite
updates are skipped.

Initial run knobs:

```text
horizon = 8 when memory allows, otherwise H4 smoke first
token_dim = 256
z_dim = 2048
projector_layers = 2
projector_num_heads = 8
dfm_layers = 4
jepa_layers = 4
num_heads = 4
mlp_dim = 1024
jepa_mlp_dim = 4 * z_dim
learning_rate = 6e-4
bt4_learning_rate = 1e-5
lr_warmup_steps = 1000
use_xsa = true
use_qk_norm = true
grad_clip_norm = 1.0
use_muon = true
```

## Metrics

Track:

- Total loss and every weighted component.
- DFM first/top-k action accuracy and legal top-1 before masking.
- First-move illegal probability mass.
- JEPA projected MSE by horizon.
- Shuffle/identity JEPA diagnostics as controls.
- SigReg value, projected-state norm, projected-state standard deviation.
- Value/WDL validation losses and calibration.
- Coupling gradient norms into the DFM action-hidden path, JEPA transition,
  state projector, and trainable BT4 encoder.

The first success criterion is not tournament strength. It is:

```text
projected JEPA improves over identity/shuffled controls
and does not collapse the projected state distribution
while DFM action metrics remain stable.
```

Only after that should we reintroduce candidate ranking/reranking.
