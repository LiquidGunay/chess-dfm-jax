# Joint Latent DFM/JEPA Stability Notes

This note records the concrete training issues hit while moving to the current
joint DFM/JEPA setup, the fixes that were applied, and the diagnostics that
remain important for interpreting runs.

## Current Objective

The current run family trains a joint model with:

- BT4 unfrozen from the start with a low learning rate (`1e-5`).
- New projector from BT4 tokens `[B, 64, 1024]` to a vector latent `z(t)` with
  dimension `num_proj` / `z_dim` (`1024` in the current runs).
- DFM consumes BT4 token latents and predicts eight action plies.
- JEPA consumes the current projected latent, true action embeddings, and DFM
  hidden action states to predict future projected latents.
- Main losses are DFM cross entropy, first-move legality, JEPA latent MSE, and
  SIGReg on target projected latents.
- JEPA positive loss is now the sum of per-coordinate latent MSE and an equal
  scalar RMS norm-matching term between predictions and stop-gradient targets.
- Value and WDL losses are disabled for the current long runs because their
  labels are too noisy and were making the total objective harder to interpret.

The scalar training loss is:

```text
loss =
  dfm_ce_coeff * dfm_ce_loss
  + first_legality_coeff * first_legality_loss
  + jepa_positive_coeff * jepa_positive_loss
  + jepa_sigreg_coeff * jepa_sigreg_loss
  + optional terms, currently disabled
```

For the current `20260503e` teacher-then-free rollout run:

```text
horizon = 8
batch_size = 256
devices = 4
steps = 221432
learning_rate = 6e-4
bt4_learning_rate = 1e-5
grad_clip_norm = 1.0
first_legality_coeff = 2.0
jepa_sigreg_kind = le_jepa
jepa_sigreg_proj_dim = 1024
jepa_sigreg_coeff = 0.1
jepa_pred_sigreg_coeff = 0.0
jepa_teacher_forcing_steps = 20000
value_coeff = 0.0
wdl_coeff = 0.0
loss_clip_value = 20.0
jepa_delta_rms_clip = 0.5
```

## Loss Scales

### DFM CE

`dfm_ce_loss` is natural-log cross entropy on masked action predictions:

```text
CE = -log p(true_action)
```

Approximate interpretation:

```text
CE 8 -> p(true) ~= 0.00034
CE 4 -> p(true) ~= 0.018
CE 2 -> p(true) ~= 0.135
CE 1 -> p(true) ~= 0.368
```

A CE drop of `0.693` doubles the probability assigned to the true move. A CE
drop of `1.0` multiplies the true-move probability by `e`.

`dfm_ce_loss_by_horizon` is the same loss broken out by ply. Later horizons are
expected to be harder, but all horizons stagnating together usually indicates a
model/data/objective issue rather than normal compounding difficulty.

### Legality

`first_legality_loss` is:

```text
1 - probability_mass_on_legal_moves
```

So:

```text
1.0 -> almost no mass on legal moves
0.5 -> half the probability mass is legal
0.1 -> 90% of probability mass is legal
0.01 -> effectively solved
```

`weighted_legality_loss = first_legality_coeff * first_legality_loss`. With the
current coefficient `2.0`, a legality loss of `0.1` contributes `0.2`.

Only the first move legality term is active in current training. The horizon
legality term is disabled because later generated-prefix legality belongs more
naturally in sampling/evaluation than in this supervised denoising loss.

### JEPA Positive Loss

`jepa_positive_loss` is:

```text
jepa_raw_mse + jepa_norm_loss
```

where:

```text
jepa_raw_mse = mean((pred_z - target_z) ** 2, axis=-1)
jepa_norm_loss = (rms(pred_z) - stop_gradient(rms(target_z))) ** 2
```

It is averaged over valid examples and horizons. `jepa_loss_by_horizon` is this
combined loss broken out by horizon; `jepa_raw_mse_by_horizon` and
`jepa_norm_loss_by_horizon` expose the two parts separately.

Because the target latent is intended to be approximately `N(0, I)`, a zero
prediction has:

```text
target_std ~= 1
target_norm ~= sqrt(D)
MSE(zero, target) ~= 1
```

For `D = 1024`, the target norm should be near `32`. Therefore:

```text
z_target_norm ~= 32
z_pred_norm ~= 32
z_state_std ~= 1
z_pred_std ~= 1
```

`jepa_raw_mse ~= 1` is ambiguous. It can mean useful partial alignment, or
it can mean a collapsed/low-norm predictor that is close to the zero baseline.
It must be interpreted together with `z_pred_norm`, `z_target_norm`,
`jepa_loss_by_horizon`, and cosine/shuffle diagnostics.

For prediction `p`, target `y`, latent dimension `D`, and cosine similarity `c`:

```text
MSE = (||p||^2 + ||y||^2 - 2 * p.y) / D
```

If alignment is poor, shrinking `||p||` can reduce MSE. This is why raw MSE does
not by itself force `z_pred_norm` to be useful.

## Major Issues and Resolutions

### SIGReg Gradients Were Not Controlling the Representation

Earlier projector/latent collapse was consistent with SIGReg not effectively
acting on the representation being used by JEPA. The current setup makes BT4
trainable from the start at low LR and applies SIGReg directly to the projected
target latents `z_all`.

Resolution:

- Unfreeze BT4 at `1e-5`.
- Keep non-BT4 modules at `6e-4`.
- Use Muon for eligible 2D matrices and AdamW for the rest.
- Apply SIGReg on target projected latents with gradients flowing through the
  projector and BT4.

### Official SIGReg Replaced the Quantile Approximation

The current `le_jepa` SIGReg implementation follows the official style used in
LeJEPA/LeWorldModel: random projection slices plus integrated characteristic
function discrepancy against a standard Gaussian.

Resolution:

- Default `jepa_sigreg_proj_dim` is now `1024`.
- Integration uses `17` points up to `t_max = 3`.
- Under `pmap`, projection RNG is synchronized across replicas and projected
  empirical characteristic function means are averaged across the data axis.

Scale interpretation for the official loss:

```text
raw_SIGReg ~= N_global * mean_integrated_slice_error
```

For current H8 B256 training:

```text
N_global = B * (H + 1) = 256 * 9 = 2304
```

Therefore:

```text
raw_SIGReg 50 -> per-slice discrepancy ~= 0.0217
raw_SIGReg 5  -> per-slice discrepancy ~= 0.00217
```

With `jepa_sigreg_coeff = 0.1`:

```text
raw 50 -> weighted contribution 5.0
raw 5  -> weighted contribution 0.5
raw 1  -> weighted contribution 0.1
```

The `sig10` run showed that a strong SIGReg term can keep norm/std near the
desired scale, but it dominated/clipped the total objective. The `sig0p1` runs
test whether a much weaker coefficient can still stabilize the latent without
overpowering CE/JEPA/legal once the task losses improve.

### Hard Output Normalization Conflicted With the Gaussian Target

Hard-normalizing the final projected latent makes the target manifold less like
an unconstrained isotropic Gaussian. The preferred target is a raw vector with
approximately zero mean, unit per-coordinate variance, and L2 norm near
`sqrt(D)`, controlled by SIGReg rather than explicit final normalization.

Resolution:

- Removed hard final RMSNorm from the state vector projector output.
- Kept normalization inside transformer/predictor blocks where it stabilizes
  computation without directly constraining the final target latent.
- Kept `--no-jepa-state-rmsnorm` for current runs so recurrent latents are not
  hard projected back onto a fixed norm manifold.

### JEPA Prediction Norm Can Collapse Under Raw MSE

With unit-variance targets, a low-norm predictor can sit near MSE `1.0`.
Teacher forcing can hide this because it trains one-step prediction from true
previous latents rather than requiring the recurrent state to stay useful when
fed its own predictions.

Resolution:

- Track `z_pred_norm`, `z_target_norm`, and `jepa_loss_by_horizon` together.
- Add a free-rollout diagnostic run with `jepa_teacher_forcing_steps = 0`.
- Add an equal-weight RMS norm-matching term inside `jepa_positive_loss` so a
  low-norm prediction pays an additional penalty even when raw MSE is near the
  zero-predictor baseline.
- Keep prediction SIGReg disabled for now (`jepa_pred_sigreg_coeff = 0.0`) so we
  can first see whether recurrent training alone fixes the low prediction norm.

Potential next fixes if free rollout still collapses:

- Add a small prediction SIGReg coefficient.
- Increase the prediction-scale regularizer if the norm-matching term is not enough.
- Adjust JEPA residual/delta scale or clipping.
- Use a SIGReg coefficient schedule after the latent std/norm stabilizes.

### Recurrent JEPA Explosions

Some long runs showed JEPA MSE spikes and exploding prediction norms. This can
come from recurrent compounding, unstable target latent scale, noisy value/WDL
heads, or too-strong gradients when the total loss is dominated by a single
term.

Resolution:

- Disable value and WDL losses.
- Clip global gradient norm at `1.0`.
- Clip scalar loss value at `20.0` while preserving gradient direction/scale.
- Add `jepa_delta_rms_clip` to bound per-step JEPA residual updates. The current
  candidate value is `0.5`, down from `2.0`, to make recurrent drift harder.
- Keep small/near-zero final predictor initialization so the transition starts
  close to identity.

### Invalid Future States in SIGReg

Trajectory batches provide `future_planes` and `future_valid` separately. Future
planes for invalid horizons are still present in the tensor, so treating
`z_all.reshape(-1, z_dim)` as fully valid can regularize invalid/terminal states
as if they were real targets.

Resolution:

- Target SIGReg is now weighted by:

```text
valid_all[:, 0] = valid
valid_all[:, h + 1] = valid * future_valid[:, h]
```

- Prediction SIGReg, when enabled, is weighted by `valid * future_valid`.
- The official characteristic-function SIGReg computes weighted empirical means
  without boolean indexing, preserving static JAX shapes under `jit`/`pmap`.

### Teacher Forcing vs Free Rollout

Teacher forcing trains:

```text
pred z(t+h+1) = transition(true z(t+h), action_h, dfm_hidden_h)
```

Free rollout trains:

```text
pred z(t+h+1) = transition(pred z(t+h), action_h, dfm_hidden_h)
```

Teacher forcing is useful for one-step local learning but can under-penalize
latent drift/collapse across recurrent horizons. Free rollout directly tests the
recurrent dynamics that will be used by the model.

Current decision:

- Run `20260503e` with `20000` teacher-forced steps, then free rollout for the
  remainder of two epochs.
- Use the equal-weight RMS norm term and lower JEPA delta RMS clip to address
  low-norm predictor collapse before adding prediction SIGReg.
- If it collapses/explodes, add a prediction-scale regularizer rather than hard
  normalizing target latents.

### BT4 Must Stay Architecturally Isolated

BT4 is the pretrained encoder and should not inherit experimental DFM/JEPA
architecture changes accidentally.

Resolution:

- Keep XSA/exclusive self-attention options limited to DFM/JEPA attention.
- Do not change BT4 attention semantics as part of latent DFM/JEPA experiments.
- BT4 can be unfrozen/trained with low LR, but architecture changes should not
  be applied to it unless explicitly intended.

### Value and WDL Losses Were Removed

Value and WDL targets are not reliable enough for this training phase. They can
create large, noisy gradients and obscure whether CE/JEPA/legal are improving.

Resolution:

- Current long runs use `--value-coeff 0.0 --wdl-coeff 0.0`.
- Model heads can remain present, but they are not part of the current training
  objective.

### Data Loading and Checkpointing

Host-side data loading and startup cache waits were significant TPU utilization
risks.

Resolution:

- Added cache warmup/wait helpers.
- Added Grain loader support for future scalable input pipelines.
- Added Orbax checkpoint support and tests so long runs can save/restore robustly.
- Current TPU runs use persistent disk-backed GCS cache and minimum startup cache
  thresholds for train/val shards.

### MFU and Multi-Chip Execution

Single-chip MFU plateaued around the mid-20% range, with profiling pointing more
to workload shape, host/input, optimizer, and small-kernel overhead than a single
obvious custom-kernel fix.

Resolution:

- Added/remat/layer-scan/buffer-donation oriented changes where appropriate.
- Moved the main long run to 4-chip data parallelism.
- Added pmap fixes for RNG and SIGReg aggregation.
- Kept custom full attention kernels out of the critical path; if custom kernels
  become useful later, narrow row-wise kernels are the safer first target.

## Current Run to Watch

The current teacher-then-free diagnostic queue is:

```text
sweeps/jph8v5p4sig0p1_normloss_dclip0p5_tf20k_20260503e.jsonl
```

The important metrics are:

```text
dfm_ce_loss
accuracy
first_legality_loss
jepa_positive_loss
jepa_loss_by_horizon
jepa_raw_mse
jepa_norm_loss
jepa_norm_loss_by_horizon
jepa_sigreg_loss
z_state_std
z_target_norm
z_pred_norm
loss_clip_scale
estimated_iteration_mfu
```

Pass criteria for continuing in this direction:

- CE decreases and accuracy increases.
- First legality loss decreases or stays controlled.
- `z_target_norm` stays near `32` for `D=1024`.
- `z_pred_norm` moves toward `32` rather than collapsing far below it.
- Later-horizon JEPA loss does not explode relative to early horizons.
- Loss clipping is not permanently active after early stabilization.

If these conditions hold, the next planned change is a SIGReg schedule: keep
`0.1` during initial latent stabilization, then decay toward `0.03` or `0.01`
once target latent std/norm are stable and task losses become the dominant
training signal.
