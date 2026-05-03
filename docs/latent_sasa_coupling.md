# Latent-SASA Coupling Work Plan

`docs/implementation_plan_gold.md` is the detailed design source. This file is
the repo-grounded implementation checklist: what exists today, what needs to be
refactored, and the guardrails for introducing DFM/JEPA coupling without turning
the two systems into loosely related baselines.

## Current Repo State

- `DFMDenoiser.__call__(current_planes, noisy_actions, t)` remains the
  standalone compatibility path. It now also exposes `encode_current`,
  `planner_from_latents`, `logits_from_action_hidden`, and `return_hidden=True`.
- `LC0JEPA` remains the standalone compatibility path. It now also exposes
  `jepa_rollout_from_latents(z0_jepa, actions, action_hidden)`.
- Standalone DFM and JEPA trainers are useful baselines and should keep working
  while the joint path is added.
- trajectory-v2 is the source-of-truth data contract. trajectory-v3 compact
  shards are a derived throughput format and currently load through `full`,
  `dfm_action`, `jepa_latent`, and `joint_latent_sasa` views.
- `JointLatentSASAModel` implements the active projected-vector Stage 1
  primitive: DFM CE, first-move legality loss, projected-vector JEPA MSE,
  SigReg, value/WDL heads, and JEPA conditioning through DFM action hidden
  states.
- Legal-prefix candidate generation and Stage 2 contrastive support exist, but
  contrastive/ranking is no longer in the active training loss. Reintroduce it
  only after projected JEPA dynamics beat identity/shuffled controls.
- The queue runner and raw NumPy checkpoint path are the active operational
  path. Async GCS upload is the next checkpointing upgrade; Orbax is not a
  prerequisite for joint training.
- `--init-checkpoint-uri` initializes a new run from an existing checkpoint
  without resuming that checkpoint directory. It uses non-strict model-state
  migration and a fresh optimizer so old Stage 1 checkpoints can branch into
  the current fused RMSNorm/SwiGLU stack.
- The first JEPA H2 baseline showed validation degradation after early steps, so
  do not scale JEPA by depth/width before adding better validation metrics,
  lower or scheduled learning rates, and the joint hidden-state path.

## Coupling Rules

- BT4 remains frozen for standalone DFM and legacy standalone JEPA baselines.
  Active joint Latent-SASA unfreezes BT4 input embedding and encoder layers
  from the start with `bt4_learning_rate=1e-5`; BT4 heads remain outside the
  joint objective.
- The joint model uses a compact DFM planning projector and a separate trainable
  state-vector projector. JEPA consumes `z_t = P(BT4(s_t)) [B,z_dim]` and
  predicts `z_{t+h} = P(BT4(s_{t+h})) [B,H,z_dim]`.
- The JEPA target projector/encoder path is not stopped in the active objective.
  SigReg must backpropagate into projected states to prevent collapse.
- The bridge is DFM action-token hidden states, not only action IDs. JEPA losses
  must have a gradient path into the DFM planner/action-token pathway.
- DFM may predict illegal actions and is trained with legality penalties. JEPA
  should only evaluate legal teacher-forced actions, legal prefix corruptions,
  or generated candidates with legal prefixes.
- Do not use clean candidate pass logits as DFM likelihoods for distillation.
  Candidate likelihoods for distillation must come from an all-mask or otherwise
  non-leaking scoring pass.
- Candidate evaluation must be vectorized with `[B, K, ...] -> [B*K, ...]`
  flattening. Do not loop over candidates in Python inside the train step.
- Changing `H`, candidate count `K`, latent width `D`, or legal `Lmax` is a new
  compiled experiment. Keep these static per queued run.

## Implementation Work Items

### W0: Preserve Baselines

- Keep `scripts/train_dfm.py` and `scripts/train_jepa.py` working as standalone
  baselines.
- Add any shared modules behind compatibility wrappers so current checkpoints
  and smoke tests are not broken.
- Continue to select DFM/JEPA baselines on held-out shards, not training loss.

Acceptance:

- Existing DFM and JEPA tests still pass.
- Existing trainer CLI entry points still construct models with old configs.

### W1: DFM Latent API Boundary

Status: initial API implemented.

Refactor DFM into separable pieces:

```text
encode_current(current_planes) -> encoder_tokens
dfm_state_projector(encoder_tokens) -> z_dfm
planner_from_latents(z_dfm, actions_or_noisy_actions, t, return_hidden=True)
    -> logits, {"state_tokens": ..., "action_tokens": ...}
```

Guidelines:

- `__call__` should remain as a compatibility wrapper.
- `hidden["action_tokens"]` shape must be `[B, H, D]`.
- Do not put target/future encoding logic into DFM.
- Standalone DFM keeps BT4 frozen. Joint training may unfreeze BT4 through the
  joint model's encoder path.

Tests:

- `planner_from_latents` returns logits `[B, H, 1858]`.
- hidden action tokens are finite and shape-stable for `H=1,2,4,8`.
- Old `model(current_planes, noisy_actions, t)` returns the same logits as the
  new path with internally encoded latents.

### W2: Shared Latent Components

Status: active joint module implemented with a compact DFM projector, a
projected-vector JEPA state projector, and DFM-hidden-state conditioning. The
previous raw-frozen-BT4 token path is now legacy; the earlier stopped-projector
path is deprecated because SigReg could not regularize the moving projected
state space.

Add reusable joint components:

```text
DFMTokenProjector
StateVectorProjector
JEPAActionAdapter
JEPAActionEmbedding
```

Guidelines:

- DFM may use a compact projected BT4 basis because its targets are action
  labels/legal masks.
- JEPA defines its state as `P(BT4(s)) [B,z_dim]`. Both current and future
  projected states participate in JEPA MSE and SigReg.
- DFM action hidden states are projected into JEPA condition width and injected into the
  JEPA rollout so JEPA loss still reaches the DFM action pathway.

Tests:

- State projector and BT4 encoder receive gradients from JEPA MSE and SigReg.
- DFM action-hidden path receives gradients from JEPA MSE.
- BT4 checkpoint-compatible architecture remains unchanged.

### W3: JEPA From Latents And Hidden States

Status: initial API implemented.

Add a JEPA API that does not re-encode current planes:

```text
jepa_rollout_from_latents(
    z0_jepa,
    actions,
    action_hidden=None,
) -> z_pred_1:H
```

Guidelines:

- Use `jax.lax.scan` for horizon rollout.
- The first implementation can combine action embedding and hidden state by
  addition or a small MLP, but action hidden state must be part of the path.
- Keep existing `predict_sequence(current_planes, actions)` as the standalone
  teacher-forced baseline wrapper.

Tests:

- Output shape is `[B, H, z_dim]`.
- JEPA consumes `hidden["action_tokens"]` without shape polymorphism.
- `L_jepa` has non-zero gradients into JEPA transition parameters.
- In joint mode, `L_jepa` has non-zero gradients into DFM action-token pathway.

### W4: Joint Batch View And Legal Indices

Status: implemented for trajectory-v2 and trajectory-v3 `.npz` shards.

Add an explicit `joint_latent_sasa` loader view.

Required batch fields:

```text
current_planes
action_indices
future_planes
future_valid
valid
legal_idx
legal_count
optional value_targets / wdl_targets
optional metadata for eval/debugging
```

Guidelines:

- For trajectory-v2, dense legal masks may be adapted to `legal_idx/count`.
- For trajectory-v3, prefer keeping compact `legal_idx/count`; only materialize
  dense masks for legacy DFM losses or debugging.
- `dfm_action` must continue to avoid materializing `planes_future`.
- `jepa_latent` must continue to avoid loading legal data unless requested.

Tests:

- `joint_latent_sasa` includes future planes and legal indices/counts.
- Dense legal masks reconstructed from `legal_idx/count` match the source mask.
- Selective views do not materialize columns they do not need.

### W5: Legal Prefix Candidates

Status: vectorized candidate sampling implemented; Stage 2 loss integration is
pending.

Implement legal prefix corruption candidate generation:

```text
positive: [a0, a1, ..., aH-1]
negative at h: [a0, ..., a{h-1}, b_h, MASK, ...]
```

where `b_h` is legal at stored state `s_h` and differs from `a_h`.

Guidelines:

- Start with `K=2 or 3` and one sampled evaluation horizon.
- If `legal_count <= 1`, mask the contrastive loss for that sample/candidate.
- Flatten candidates to `[B*K, ...]` before DFM/JEPA execution.

Tests:

- Legal alternative is never equal to the true action.
- `legal_count <= 1` is handled without invalid indexing.
- Flattened candidate execution matches a slow reference implementation.

### W6: Stage 1 Joint Trainer

Status: Stage 1 model/loss/train-step primitive and queue-compatible trainer
implemented. Local synthetic smoke verified checkpoint save/resume, validation,
W&B-disabled script mode, coupling-gradient diagnostics, and action-baseline
JEPA diagnostics. TPU H2/H4 smoke and short queued runs completed.

Add the first joint objective:

```text
L = 1.0 * L_dfm_ce
  + lambda_first * L_first_legal
  + 1.0 * L_jepa_positive
  + lambda_sigreg * L_jepa_sigreg
  + lambda_value * L_value
  + lambda_wdl * L_wdl
```

Current definitions:

- `L_dfm_ce` is masked-token CE over `loss_horizon`.
- `L_first_legal = 1 - sum_a p(a) 1[a legal at s0]` for horizon slot 0.
- `L_horizon_legal` is the same illegal-mass penalty for later teacher-forced
  horizon slots only. It is off by default for Stage 1.
- Legality defaults to masked-token positions only. Validation uses `t=0`, so
  all supervised slots are masked.
- `L_jepa_positive = mean((z_pred - z_target)^2)` over projected BT4 state
  vectors.
- `L_jepa_sigreg` is enabled by default with coefficient `0.1` and applies the
  LeJEPA/quantile random-projection regularizer to projected current/future
  state vectors.
- `L_action_contrast` is disabled in the active plan. Shuffled-action rollouts
  are diagnostics until projected JEPA beats identity/shuffled controls.
- JEPA diagnostics also log raw MSE, normalized MSE, vector norms, per-horizon
  losses, identity baseline, and shuffled-action baseline.

Initial config:

```text
H = 4 smoke, then H = 8
K = 1
token_dim = 256
z_dim = 2048
DFM layers = 4
JEPA layers = 4
value_coeff = 0.05
wdl_coeff = 0.05
```

Guidelines:

- Use the random-policy-balanced legality coefficient for first-ply legality
  unless a run explicitly tests another value.
- The active target space is projected BT4 state vectors.
  `--target-projector-mode` is accepted only for compatibility and is ignored
  by the joint trainer.
- Use `bt4_learning_rate=1e-5`, `learning_rate=6e-4`, and warmup before scaling
  JEPA depth.
- Report gradient norms from `L_jepa` into DFM action-token/pathway modules.

Acceptance:

- JEPA latent validation improves over identity/random-action controls.
- DFM action CE and legality do not collapse.
- Coupling gradient diagnostics are non-zero and finite.

### W7: Stage 2 Contrastive Coupling

Status: Stage 2 contrastive loss and trainer support exist, but this is no
longer the active next objective. Keep it as future work after projected JEPA
state dynamics are stable.

Add legal prefix negatives and contrastive future loss:

```text
L = L_dfm_ce
  + lambda_legal * L_legal
  + L_jepa_positive
  + 0.5 * L_contrast
```

Guidelines:

- Start with `H=4`, `K=3`, and `H_eval=1`.
- Use only legal alternatives from stored `legal_idx/count`.
- Do not add rank or distillation until contrastive candidate accuracy exceeds
  chance on held-out validation.

Acceptance:

- Positive similarity separates from legal-negative similarity.
- Contrastive candidate accuracy exceeds chance.
- First-action validation metrics do not degrade materially versus action-only.

Latest smoke result:

```text
run: joint-stage2-h4-k3-b64-v3-smoke-20260502d
init: Stage 1 checkpoint step 387072, non-strict migrated, fresh optimizer
data: v3 LC0 H8 minimum-cache smoke, 32 train shards / 32 val shards visible
steps: 200
single-chip throughput: about 0.24 s/step, about 26% model-FLOP MFU

validation step 100:
  val_contrastive_loss = 1.2755
  val_contrastive_accuracy = 0.3359
  val_jepa_raw_mse = 1.8177
  val_dfm_ce_loss = 5.7810
  val_first_legality_loss = 0.8545

validation step 200:
  val_contrastive_loss = 0.4978
  val_contrastive_accuracy = 0.7148
  val_jepa_raw_mse = 1.5724
  val_dfm_ce_loss = 5.6433
  val_first_legality_loss = 0.8115
```

Interpretation: Stage 2 wiring and legal-prefix contrastive signal are working.
This is not yet a model-quality result because the smoke used a tiny
minimum-cache shard window and only 200 steps.

### W8: Reranking And Rank Head

Add candidate reranking after Stage 2 works:

```text
score(chunk) = rank_head(pool(z_hat_1:H))
```

Optional later score:

```text
value_head(z_hat_H)
+ WDL score
- invalid prefix penalty
+ gamma * DFM all-mask logprob
```

Guidelines:

- Compare raw DFM proposals against JEPA-reranked proposals on the same held-out
  candidate set.
- Do not distill unless reranking beats raw DFM.
- Keep value/WDL losses off until target alignment is verified for the future
  state being scored.

Acceptance:

- Reranked DFM beats raw DFM on held-out first-action quality, full-prefix
  legality, or puzzle/action eval.

### W9: Distillation

Only after reranking helps, add all-mask distillation:

```text
q_i = stopgrad(softmax(score_i / tau))
L_distill = -sum_i q_i * logp_DFM(candidate_i | all-mask input)
```

Guidelines:

- Start with `lambda_distill = 0.05`.
- Never compute `logp_DFM` from the clean candidate pass.
- Track diversity and legality so distillation does not collapse the planner.

Acceptance:

- Raw DFM improves after distillation.
- Reranked DFM remains stronger than raw DFM.
- Diversity and legal-rate metrics do not collapse.

### W10: Loader And Checkpointing Upgrades

Keep these as systems work alongside objective development:

- Finish trajectory-v3 conversion and validation.
- Add deterministic windowed prefetch once full startup cache becomes too slow.
- Keep the loader boundary Grain-compatible, but do not migrate to Grain until
  compact-v3 throughput and sharding needs are measured.
- Add async GCS upload for completed local raw NumPy checkpoints and queue
  status. Keep local writes synchronous.

Acceptance:

- Same seed and step reproduce the same rows.
- No growing-cache replay artifact.
- Queue status reports latest local and latest uploaded checkpoint steps.

## Experiment Order

1. Preserve and evaluate current action-only DFM baselines.
2. Stabilize JEPA-only baselines with better validation, lower/scheduled LR, and
   identity/random-action controls.
3. Implement W1-W4 API/data boundaries with synthetic tests.
4. Run Stage 1 joint H1/H2 positive hidden-state coupling.
5. Add W5/W7 legal-prefix contrastive candidates.
6. Add W8 reranking only after contrastive metrics are useful.
7. Add W9 distillation only after reranking beats raw DFM.

## Do-Not-Do List

- Do not unfreeze BT4 for early joint experiments.
- Do not train JEPA on impossible illegal transitions.
- Do not treat action-ID-only JEPA conditioning as sufficient coupling.
- Do not launch broad Phase A legality sweeps before the clean full-dataset
  baseline.
- Do not start mixed TCEC+LC0 joint training from an unvalidated or mutable
  prefix.
- Do not claim multi-device TPU scaling until sharded arrays/pmap/pjit are
  explicitly implemented and measured.
