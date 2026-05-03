# PLAN.md — Latent-SASA Joint Pretraining for `chess-dfm-jax`

## 0. Goal

Train a coupled **DFM + JEPA latent planner** for chess trajectories, without simply cloning DiffuSearch’s discrete `S-A-S-A` state-token setup.

The central idea:

```text
Do not force the model to generate future board tokens.

Instead:

1. DFM learns to denoise / propose future action chunks.
2. JEPA learns to predict future BT4 latent states conditioned on those action chunks.
3. JEPA consumes DFM action-token hidden states, so future-consequence losses shape the DFM planner.
4. Later, JEPA/rank/value heads rerank DFM candidates.
5. Only after reranking works, distill the reranker’s preferences back into DFM.
```

Target representation:

```text
s0, a0, s1, a1, s2, ...
```

becomes:

```text
z0, a0, z1, a1, z2, ...
```

where:

```text
zh = projected BT4 state representation of board state sh
   = P(BT4(sh)) with shape [z_dim]
```

This is **Latent-SASA**, not discrete board-token SASA.

Implementation correction, 2026-05-02: the active repo plan is projected-vector
JEPA with BT4 unfreezing from the start. Earlier references in this document to
raw frozen BT4-token JEPA targets, EMA/stop-gradient target projectors, and
contrastive Stage 2 as the immediate next objective are superseded by
`docs/projected_jepa_unfreeze_plan.md`. The current objective uses
`z = P(BT4(s))`, backpropagates SigReg through the projected states, trains BT4
embedding/encoder params at `1e-5`, trains the rest at `6e-4`, and keeps
contrastive/ranking deferred until projected JEPA dynamics are stable.

---

## 1. Non-goals

Do not start with:

```text
- Discrete state-token S-ASA generation.
- Stockfish/top-k relabeling as a required dependency.
- Full MCTS.
- Chess-specific evaluation heuristics such as material bonuses, king-safety handcrafted scores, capture bonuses, etc.
- Feeding illegal actions into JEPA as normal transition inputs.
- Distillation before the JEPA reranker has proven useful.
- Changing BT4 architecture internals while loading the pinned BT4 checkpoint.
- Reintroducing contrastive/ranking before projected JEPA beats identity and
  shuffled-action controls.
```

Legal move constraints are allowed. They are environment constraints, not evaluation heuristics.

---

## 2. Existing data assumption

Use the existing H8 trajectory datasets as the source of truth.

Each sample conceptually provides:

```text
current board:
    s0 / planes_t

trajectory:
    actions a0:H

future states:
    s1:H / planes_future

validity:
    future_valid

legal actions:
    legal mask or legal_idx/legal_count at each stored horizon

optional:
    value targets
    WDL targets
    metadata
```

We do not need top-k actions in the dataset. Legal alternatives can be generated online from the stored legal action sets.

---

## 3. High-level architecture

### 3.1 Modules

```text
Frozen BT4 encoder:
    E(s) -> e ∈ R[64, 1024]

Shared online projector:
    P_online(e) -> u ∈ R[64, D]

DFM adapter:
    A_dfm(u) -> z_dfm ∈ R[64, D]

JEPA adapter:
    A_jepa(u) -> z_jepa ∈ R[64, D]

Target projector:
    P_target(e_future) -> z_target ∈ R[64, D]
    Usually EMA / stop-gradient target projector.

DFM planner trunk:
    planner_from_latents(z_dfm, action_tokens, t)
        -> action_logits
        -> state hidden tokens
        -> action hidden tokens

JEPA transition model:
    jepa_rollout_from_latents(
        z0_jepa,
        actions,
        dfm_action_hidden
    ) -> predicted future latents z_hat_1:H

Optional heads:
    value/WDL head
    rank head
```

### 3.2 Sharing policy

Use:

```text
BT4:
    frozen and shared

BT4 -> latent projector:
    shared online base projector

DFM / JEPA adapters:
    separate small adapters

Target projector:
    separate EMA / stop-gradient target projector

Action embedding:
    shared between DFM and JEPA

DFM action hidden states:
    passed into JEPA transition model
```

Avoid:

```text
fully separate DFM and JEPA projectors
```

because this makes the coupling too thin.

Also avoid:

```text
fully shared everything
```

because action CE gradients can distort the JEPA target space.

Recommended compromise:

```text
shared base latent space
+ small objective-specific adapters
+ EMA target projector
+ shared action embedding
+ JEPA consumes DFM action hidden states
```

---

## 4. Core coupling mechanism

The bridge should not merely be:

```text
action_id -> JEPA
```

The bridge should be:

```text
DFM action-token hidden states -> JEPA transition
```

The DFM should return hidden states:

```python
logits, hidden = planner_from_latents(
    z0_dfm,
    actions_or_noisy_actions,
    t,
    return_hidden=True,
)

hidden = {
    "state_tokens":  seq[:, :64, :],
    "action_tokens": seq[:, 64:, :],
}
```

Then JEPA uses:

```python
z_pred = jepa_rollout_from_latents(
    z0_jepa=z0_jepa,
    actions=actions,
    action_hidden=hidden["action_tokens"],
)
```

So the gradient path is:

```text
JEPA latent loss
    -> JEPA transition model
    -> DFM action hidden states
    -> DFM planner trunk
    -> DFM action embeddings
    -> shared state projector
```

This is the main mechanism that makes the DFM action representation consequence-aware.

---

## 5. Illegal action policy

Use different rules for DFM and JEPA.

### DFM denoising pass

The DFM predicts logits over all action IDs.

```text
DFM may assign probability to illegal actions.
Legality loss penalizes this.
Inference hard-masks illegal first moves.
```

### JEPA transition pass

JEPA should receive only valid legal transitions.

```text
Teacher-forced JEPA:
    use stored real legal actions.

Contrastive candidate JEPA:
    use legal alternatives from legal_idx/legal_count.

Generated DFM candidate JEPA:
    only evaluate legal prefixes.
    If a candidate becomes illegal, truncate or mark invalid.
```

Do not ask JEPA to predict a future after an impossible move.

---

## 6. Training passes

A full joint step may contain up to four passes, but they should be introduced gradually.

### Pass A — noisy DFM denoising pass

Purpose:

```text
Train the DFM action model.
```

Input:

```text
z0_dfm
noisy/masked action sequence
t
```

Output:

```text
action logits
optional hidden states
```

Losses:

```text
L_dfm_ce
L_legal
```

### Pass B — clean positive planning pass

Purpose:

```text
Make DFM hidden states useful for JEPA future prediction.
```

Input:

```text
z0_dfm
clean stored action sequence
planner_mode=True or t=1.0
```

Output:

```text
action logits
DFM action hidden states
```

JEPA consumes these hidden states and predicts:

```text
z_hat_1:H
```

Loss:

```text
L_jepa_positive
```

### Pass C — clean legal candidate pass

Purpose:

```text
Contrast real continuation against legal alternative actions.
```

Input:

```text
z0_dfm repeated across candidates
candidate action sequences
planner_mode=True or t=1.0
```

Output:

```text
candidate action hidden states
candidate predicted future latents
```

Losses:

```text
L_contrastive_future
optional L_rank
```

### Pass D — all-mask DFM scoring pass

Purpose:

```text
Distill JEPA/ranker preferences back into the DFM.
```

Only add this after reranking works.

Important:

```text
Do not compute candidate log-probabilities from the clean candidate pass,
because the DFM has seen the candidate tokens.
```

Instead use:

```text
input actions = [MASK, MASK, ..., MASK]
```

Then:

```text
candidate_logprob = sum_h log_softmax(mask_logits_h)[candidate_action_h]
```

Loss:

```text
L_distill = - Σ_i q_i log p_DFM(candidate_i | s0)
```

where:

```text
q_i = stopgrad(softmax(score_i / tau))
```

---

## 7. Candidate generation

No top-k dataset labels are required.

Use the stored true action sequence and legal action sets.

For each sample:

```text
positive candidate:
    [a0, a1, a2, a3]

negative candidate at horizon h:
    [a0, ..., a_{h-1}, b_h, MASK, ..., MASK]

where:
    b_h is legal at stored state s_h
    b_h != a_h
```

Evaluate only up to the first corrupted action.

Example for `H=4` and corruption at `h=2`:

```text
positive:
    [a0, a1, a2, a3]

negative:
    [a0, a1, b2, MASK]
```

Compare:

```text
positive predicted z3 to real z3
negative predicted z3 to real z3
```

The contrastive objective should make the positive candidate more similar to the real future latent than the legal corrupted candidate.

### 7.1 Candidate tensor shapes

Let:

```text
B = batch size
H = action horizon
K = number of candidates per position, including positive
D = latent dimension
```

Shapes:

```text
z0_dfm:       [B, 64, D]
z0_jepa:      [B, 64, D]
actions:      [B, H]
candidates:   [B, K, H]
eval_horizon: [B, K]
```

Flatten candidates for efficient device execution:

```text
z0_dfm_cand:  [B*K, 64, D]
z0_jepa_cand: [B*K, 64, D]
cand_actions: [B*K, H]
```

Never loop over candidates in Python.

---

## 8. Losses

### 8.1 DFM action CE

```text
L_dfm_ce = CE(predicted clean actions, target actions)
```

Apply only on selected masked/noised positions and valid horizons.

### 8.2 Legality loss

Prefer compact legal indices:

```text
legal_idx:   [B, H, Lmax]
legal_count: [B, H]
```

Given:

```text
probs: [B, H, V]
```

compute:

```python
legal_probs = take_along_axis(probs, legal_idx, axis=-1)
valid_slot = arange(Lmax) < legal_count[..., None]
legal_mass = sum(legal_probs * valid_slot, axis=-1)
```

Possible losses:

```text
L_legal = mean(1 - legal_mass)
```

or stronger:

```text
L_legal = mean(-log(legal_mass + eps))
```

Track first-ply legality and horizon legality separately.

### 8.3 Positive JEPA latent loss

Target branch:

```python
z_target = stopgrad(BT4(future_planes))
```

Prediction:

```python
z_pred = JEPA(z0_raw_bt4, actions, dfm_action_hidden)
```

Use raw BT4-token MSE as the primary loss. Cosine distance remains a diagnostic,
not the optimized objective, because normalized cosine can improve while raw
coordinate reconstruction gets worse if predicted token norms drift.

```text
L_jepa =
    Σ_h γ^h * mean((z_pred_h - z_target_h)^2)
```

Mask with `future_valid`.

### 8.4 Contrastive legal-alternative loss

For candidates:

```text
candidate 0 = positive
candidate 1..K-1 = legal corrupted candidates
```

Compute similarity against the real future target at the candidate’s evaluation horizon:

```text
sim_i = cosine(z_pred_candidate_i_at_eval_h, z_target_at_eval_h)
```

Then:

```text
L_contrast =
    CE(sim / temperature, label=positive_index)
```

Usually:

```text
positive_index = 0
```

This trains the model to understand that a different legal action from the same state usually leads to a different future latent.

### 8.5 Optional rank loss

A rank head scores each predicted candidate rollout:

```text
score_i = rank_head(z_pred_i)
```

Train:

```text
L_rank = CE(score, label=positive_index)
```

This is a behavior-cloning style candidate-ranking objective:

```text
real strong-engine/self-play continuation > random legal corruption
```

Use moderate weight because some random legal alternatives may also be strong.

### 8.6 Optional value/WDL loss

If value/WDL targets are available:

```text
L_value = MSE(value_head(z_pred_h), value_target_h)
L_wdl   = CE(wdl_head(z_pred_h), wdl_target_h)
```

This can later provide a better reranking score than pure positive-vs-corruption ranking.

### 8.7 Optional distillation loss

Only after reranking helps.

Candidate scores:

```text
score_i = JEPA/rank/value score for candidate i
q_i = stopgrad(softmax(score_i / tau))
```

DFM candidate log-probability from all-mask pass:

```text
logp_i = Σ_h log p_DFM(candidate_i_h | s0, all_mask_input)
```

Distillation:

```text
L_distill = - mean_B Σ_i q_i * logp_i
```

This means:

```text
“Given these candidates, make the cheap DFM policy more likely to emit the candidates that the latent critic prefers.”
```

---

## 9. Total loss by stage

### Stage 1 loss

```text
L =
    1.0 * L_dfm_ce
  + 2.0 * L_legal
  + 1.0 * L_jepa_positive
```

### Stage 2 loss

```text
L =
    1.0 * L_dfm_ce
  + 2.0 * L_legal
  + 1.0 * L_jepa_positive
  + 0.5 * L_contrast
```

### Stage 3 loss

```text
L =
    1.0 * L_dfm_ce
  + 2.0 * L_legal
  + 1.0 * L_jepa_positive
  + 0.5 * L_contrast
  + 0.25 * L_rank
  + optional value/WDL losses
```

### Stage 4 loss

```text
L =
    previous losses
  + 0.05 to 0.25 * L_distill
```

Do not add distillation at high weight initially.

---

## 10. Efficient implementation

### 10.1 Required refactor

Refactor current DFM so that state encoding and planner computation are separable.

Target API:

```python
e0 = bt4.encode_tokens(current_planes)
u0 = shared_projector(e0)

z0_dfm = dfm_adapter(u0)
z0_jepa = jepa_adapter(u0)

logits, hidden = planner_from_latents(
    z0_dfm,
    action_tokens,
    t,
    return_hidden=True,
)
```

JEPA API:

```python
z_pred = jepa_rollout_from_latents(
    z0_jepa,
    actions,
    action_hidden=hidden["action_tokens"],
)
```

Target branch:

```python
ef = bt4.encode_tokens(future_planes.reshape(B * H, ...))
zt = target_projector(ef).reshape(B, H, 64, D)
zt = stop_gradient(zt)
```

### 10.2 Candidate flattening

Never do:

```python
for candidate in candidates:
    run planner
    run jepa
```

Do:

```python
z0_dfm_cand = repeat(z0_dfm, K)      # [B*K, 64, D]
z0_jepa_cand = repeat(z0_jepa, K)    # [B*K, 64, D]
cand_actions = candidates.reshape(B*K, H)

logits_cand, hidden_cand = planner_from_latents(
    z0_dfm_cand,
    cand_actions,
    t=ones([B*K]),
    return_hidden=True,
)

z_pred_cand = jepa_rollout_from_latents(
    z0_jepa_cand,
    cand_actions,
    hidden_cand["action_tokens"],
)

z_pred_cand = z_pred_cand.reshape(B, K, H, 64, D)
```

### 10.3 Horizon loop

Use `jax.lax.scan` inside JEPA rollout.

Do not unroll the horizon loop in Python.

### 10.4 Static shapes

Keep these static per compiled training run:

```text
H
K
D
number of JEPA layers
number of DFM layers
Lmax for legal actions
```

Changing `H` or `K` should trigger a new compile.

---

## 11. Hardware cost model

Let:

```text
B = batch size
K = candidates per state
H_eval = number of evaluated JEPA transition steps
M = number of JEPA transition layers
N = number of DFM planner layers
S = JEPA state-token length, usually 64
L = DFM sequence length, roughly 64 + H
```

Approximate trainable compute ratio relative to one DFM pass:

```text
ratio ≈
    1                                  # noisy DFM pass
  + K                                  # clean candidate planner pass
  + K * H_eval * (M / N) * (S / L)     # JEPA rollout
  + optional 1                         # all-mask distillation pass
```

Since:

```text
S ≈ 64
L ≈ 64 + H
```

for small H:

```text
S / L ≈ 1
```

So approximate:

```text
ratio ≈ 1 + K + K * H_eval * M/N
```

Example starter setting:

```text
K = 3
H_eval = 1
M = 2
N = 8
```

Compute ratio:

```text
1 + 3 + 3 * 1 * 2/8 = 4.75x
```

This is why the first candidate contrastive stage should use:

```text
K = 2 or 3
H_eval = 1
JEPA layers = 1 or 2
```

Do not start with full H4 rerank/distill.

---

## 12. Data loading plan

### 12.1 Short-term

Use existing trajectory-v2 shards, but add mode-specific selective loading.

For DFM-only:

```text
load:
    planes_t
    actions[:, :H]
    legal information[:, :H]
    valid/future_valid

do not load:
    planes_future
```

For JEPA/joint:

```text
load:
    planes_t
    planes_future[:, :H]
    actions[:, :H]
    legal information[:, :H]
    future_valid[:, :H]
    optional value/WDL
```

### 12.2 Medium-term compact v3 format

Create a derived compact dataset:

```text
trajectory_v3_latent_sasa/
    manifest.json

    planes_t_bitpack.npy or planes_t_u8.npy
        [N, 112, 8, 8] or bitpacked equivalent

    planes_future_bitpack.npy or planes_future_u8.npy
        [N, H, 112, 8, 8] or bitpacked equivalent

    actions_u16.npy
        [N, H]

    future_valid_u8.npy
        [N, H]

    legal_idx_u16.npy
        [N, H, Lmax]

    legal_count_u16.npy
        [N, H]

    optional value_i16.npy
    optional wdl_u8.npy
    sample_id_u64.npy
    game_id_u64.npy
    ply_i32.npy
```

Use:

```text
legal_idx/legal_count
```

instead of dense float32 legal masks.

### 12.3 Loader requirements

The training loader should support:

```text
- deterministic distributed sampling
- column-selective reading
- sample-level shuffle buffer
- static-shape batches
- local SSD / persistent disk caching
- background prefetch
- no growing-cache replay artifact
- resumable epoch/step state
```

The loader should report:

```text
read_time
decode/unpack_time
host_to_device_time
train_step_time
queue_depth
examples_per_second
accelerator_idle_fraction
```

---

## 13. Candidate generation implementation details

Given:

```text
actions:      [B, H]
legal_idx:    [B, H, Lmax]
legal_count:  [B, H]
```

For each sample and candidate:

```text
1. Choose corruption horizon h among valid horizons.
2. Choose a legal alternative b_h at state s_h.
3. Ensure b_h != true action a_h.
4. Candidate = true prefix up to h-1 + b_h + MASK after h.
5. eval_horizon = h.
```

Edge case:

```text
If legal_count <= 1:
    no legal alternative exists.
    Either skip contrastive candidate for that sample or duplicate positive with zero loss mask.
```

Implementation should be vectorized in JAX if possible.

Pseudo-logic for excluding true action:

```python
true_a = actions[b, h]
legal = legal_idx[b, h, :legal_count[b, h]]

# find true action position if present
true_pos = argwhere(legal == true_a)

# sample raw index from [0, legal_count - 2]
raw = randint(0, legal_count - 1)

# skip true position
alt_pos = raw + (raw >= true_pos)

b_h = legal[alt_pos]
```

Fallback if true action is not found in legal list:

```text
sample any legal action and reject if equal to true action
```

---

## 14. Reranking

Reranking is a stronger inference/training-time mode.

Procedure:

```text
1. DFM proposes M candidate action chunks.
2. Filter/truncate illegal candidates.
3. For each candidate:
       run clean planner pass
       feed DFM action hidden states into JEPA
       predict future latents
       score predicted future with rank/value/WDL head
4. Aggregate scores by first move.
5. Pick the best first move.
```

Simple score:

```text
score(chunk) = rank_head(mean_pool(z_hat_1:H))
```

Later score:

```text
score(chunk) =
    value_head(z_hat_H)
  + alpha * WDL_good_logit
  - beta * invalid_prefix_penalty
  + gamma * logp_DFM(chunk)
```

Important eval:

```text
raw DFM move quality
vs
JEPA-reranked DFM move quality
```

If reranked DFM does not beat raw DFM, do not add distillation.

---

## 15. Distillation

Distillation is only Stage 4.

It converts expensive JEPA/ranker preferences into cheaper DFM probabilities.

For candidates:

```text
C1, C2, ..., CK
```

JEPA/ranker gives:

```text
score_i
```

Convert to target distribution:

```text
q_i = stopgrad(softmax(score_i / tau))
```

Compute DFM log-probabilities from all-mask pass:

```text
logp_i = log p_DFM(C_i | s0, all-mask input)
```

Loss:

```text
L_distill = -Σ_i q_i logp_i
```

Important leakage rule:

```text
Do not use clean candidate pass logits to compute logp_i.
The candidate was visible in that pass.
Use all-mask or appropriate denoising/scoring pass.
```

Purpose:

```text
Make raw DFM more likely to propose candidates that the latent critic prefers.
```

Success criterion:

```text
After distillation, fast/raw DFM improves, not just reranked DFM.
```

---

## 16. Training stages

### Stage 0 — loader and model API refactor

Implement:

```text
- selective loading
- legal_idx/legal_count support
- planner_from_latents API
- DFM hidden-state return
- jepa_rollout_from_latents API
- shared projector + separate adapters
- target projector with stop-gradient/EMA
```

No major research objective yet.

### Stage 1 — positive hidden-state JEPA coupling

Use:

```text
H = 1 or 2 first
K = 1
```

Loss:

```text
L_dfm_ce + L_legal + L_jepa_positive
```

Goal:

```text
Verify JEPA loss improves future-latent prediction and sends gradients into the DFM planner trunk.
```

### Stage 2 — legal contrastive prefix negatives

Use:

```text
H = 4 stored trajectory
H_eval = 1 sampled corruption horizon
K = 3 candidates:
    1 positive
    2 legal prefix corruptions
JEPA layers = 1 or 2 initially
```

Loss:

```text
L_dfm_ce
+ L_legal
+ L_jepa_positive
+ L_contrast
```

Goal:

```text
Make DFM hidden action states consequence-aware.
```

### Stage 3 — rank head and reranking eval

Add:

```text
L_rank
optional value/WDL losses
```

Run eval:

```text
raw DFM
vs
JEPA-reranked DFM
```

Goal:

```text
Show that JEPA/rank scoring improves move selection over raw DFM proposals.
```

### Stage 4 — distill reranker into DFM

Add all-mask DFM scoring pass.

Loss:

```text
L_distill
```

Use small weight:

```text
lambda_distill = 0.05 to 0.25
```

Goal:

```text
Improve fast/raw DFM using the critic’s preferences.
```

### Stage 5 — full-chunk candidates

Only after Stage 2–4 work.

Try:

```text
K = 4 to 8
H_eval = 2 to 4
generated DFM candidates
environment-validated legal prefixes
value/WDL-aware scoring
```

---

## 17. Metrics

### DFM metrics

```text
first_action_CE
horizon_CE
first_action_accuracy
top_k_accuracy
pre-mask legal mass
post-mask legal accuracy
policy entropy
```

### JEPA metrics

```text
latent cosine by horizon
latent normalized MSE by horizon
future_valid-masked latent loss
value/WDL error from predicted latents
```

### Contrastive/ranking metrics

```text
contrastive candidate accuracy
positive similarity
negative similarity
similarity margin
rank accuracy
```

### Coupling diagnostics

Track whether JEPA losses actually affect the DFM:

```text
gradient norm from L_jepa into DFM planner blocks
gradient norm from L_contrast into DFM planner blocks
gradient norm into action embeddings
gradient norm into shared projector
gradient norm into JEPA-only modules
```

Important diagnostic questions:

```text
1. Does JEPA loss produce non-trivial gradients on DFM action-token layers?
2. Does contrastive loss improve candidate discrimination?
3. Does JEPA reranking improve move quality?
4. Does distillation improve raw DFM after reranking works?
```

### Systems metrics

```text
examples/sec
tokens/sec
host input wait
device idle fraction
GCS/local read time
decode/unpack time
host-to-device transfer time
train step time
MFU estimate
```

---

## 18. Tests

### Model tests

```text
- planner_from_latents returns logits and hidden states with expected shapes.
- hidden["action_tokens"] has shape [B, H, D].
- JEPA consumes hidden["action_tokens"] without shape polymorphism.
- target branch is stop-gradient.
- EMA target projector updates correctly.
```

### Candidate tests

```text
- legal alternative is never equal to true action.
- legal_count <= 1 is handled safely.
- candidate positive index is always known.
- eval_horizon is valid.
- candidate flattening [B,K,...] -> [B*K,...] matches a slow reference loop.
```

### Leakage tests

```text
- distillation log-probs are computed from all-mask pass, not clean candidate pass.
- clean candidate logits are not used as policy likelihoods.
```

### Gradient tests

```text
- L_jepa has non-zero gradient into JEPA transition.
- L_jepa has non-zero gradient into DFM action hidden pathway.
- L_jepa does not gradient into frozen BT4.
- L_contrast has non-zero gradient into DFM planner trunk.
```

### Loader tests

```text
- same seed + same step gives same rows.
- distributed processes do not overlap incorrectly.
- resume preserves sample order.
- selective loading does not materialize unused columns.
- legal_idx/legal_count reconstructs legal mass correctly.
```

---

## 19. Starter config

Suggested first meaningful joint run:

```yaml
objective: joint_latent_sasa

data:
  source: existing_h8_trajectory_v2
  view: joint_latent_sasa
  horizon: 4
  use_future_planes: true
  legal_format: legal_idx_count
  selective_loading: true

model:
  bt4:
    frozen: true

  latent:
    token_dim: 512
    shared_projector: true
    dfm_adapter: true
    jepa_adapter: true
    target_projector: ema
    ema_decay: 0.99

  dfm_planner:
    num_layers: 4   # use 8 only after debugging
    num_heads: 8
    mlp_dim: 2048
    return_hidden: true

  jepa_transition:
    num_layers: 2
    num_heads: 8
    mlp_dim: 2048
    consumes_dfm_action_hidden: true

training:
  batch_size: hardware_dependent
  action_noise_schedule: uniform_or_absorbing
  candidate_count: 3
  candidate_mode: legal_prefix_corruption
  eval_horizon_mode: sampled_single_horizon
  use_rank_head: false
  use_distillation: false

loss_weights:
  dfm_ce: 1.0
  legality: 2.0
  jepa_positive: 1.0
  contrastive: 0.5
  rank: 0.0
  distill: 0.0
```

After this is stable:

```yaml
use_rank_head: true
loss_weights:
  rank: 0.25
```

Only after reranking helps:

```yaml
use_distillation: true
loss_weights:
  distill: 0.05
```

---

## 20. Success criteria

### Stage 1 success

```text
- JEPA latent cosine improves.
- Action CE does not collapse.
- Legal mass improves or remains stable.
- JEPA gradient reaches DFM planner/action hidden pathway.
```

### Stage 2 success

```text
- Contrastive candidate accuracy exceeds chance.
- Positive future similarity separates from legal negative similarity.
- First-action metrics do not degrade significantly.
```

### Stage 3 success

```text
- JEPA-reranked DFM beats raw DFM on heldout action quality or puzzle/action eval.
```

### Stage 4 success

```text
- Raw DFM improves after distillation.
- Reranked DFM remains stronger than raw DFM.
- Distillation does not collapse diversity or legality.
```

---

## 21. Core implementation order

1. Refactor DFM into:

   ```text
   encode_current
   planner_from_latents
   return_hidden=True
   ```

2. Refactor JEPA into:

   ```text
   jepa_rollout_from_latents(z0_jepa, actions, dfm_action_hidden)
   ```

3. Add shared projector + adapters:

   ```text
   P_online
   A_dfm
   A_jepa
   P_target
   ```

4. Add positive hidden-state JEPA loss.

5. Add legal prefix candidate generation.

6. Add candidate flattening and contrastive loss.

7. Add gradient/coupling diagnostics.

8. Add rank head and reranking eval.

9. Add all-mask distillation only after reranking works.

10. Compact the dataset/loading path once the objective is validated.

---

## 22. Summary

The main risk is that DFM and JEPA become two separate models with a thin action-ID bridge.

The fix is:

```text
DFM action hidden states must be the plan tokens consumed by JEPA.
```

The first serious objective should be:

```text
DFM CE
+ legality loss
+ positive JEPA latent prediction
+ legal-alternative contrastive future loss
```

Then, once JEPA is useful as a critic:

```text
rerank DFM candidates
```

Then, only after reranking helps:

```text
distill critic preferences back into the DFM using all-mask policy scoring
```

The final direction is therefore:

```text
Latent-SASA joint pretraining on existing strong-engine self-play data,
with DFM proposing action chunks and JEPA making those chunks consequence-aware.
```

