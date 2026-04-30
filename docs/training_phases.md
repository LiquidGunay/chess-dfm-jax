# Training Phases

Keep smoke tests, phase-wise sweeps, and large combined runs separated. The TPU
controller is shared infrastructure; the training command, checkpoint URI, W&B
group, and run ID define the experiment phase.

## Phase Order

1. `smoke`: smallest possible job that proves source packaging, TPU startup,
   data download, training, checkpoint upload, status upload, and cleanup.
2. `dfm`: action-only DFM baselines. Use `scripts/train_dfm.py` and checkpoint
   under `runs/dfm/<run_id>/checkpoints`.
3. `jepa`: teacher-forced latent future prediction baselines. Use
   `scripts/train_jepa.py` and checkpoint under `runs/jepa/<run_id>/checkpoints`.
4. `profile`: profiling-only jobs. Use `scripts/profile_jepa_tpu.py`; never
   share checkpoint directories with training jobs.
5. `joint`: Latent-SASA joint DFM+JEPA runs after loader and queue smoke tests.
6. `combined`: later mixed-data plan-scoring/reranking runs after LC0-only joint
   baselines are stable.

## Current Run State

As of 2026-04-29, the LC0-only Phase A diagnostic run on the 1M sample dataset
has completed:

- run ID: stored in ignored local launch/status config, not tracked in public docs
- phase: DFM curriculum Phase A
- hardware: single `v5litepod-1` TPU VM, one JAX process
- data: LC0 test80 H8 trajectory-v2 shards, 920 train shards and 55 validation
  shards
- cache policy: `--gcs-startup-cache-policy all`
- objective: first-ply legal policy learning before full H8 denoising
- loss: `CE(first_move) + 5 * illegal_prob_mass(first_move)`
- result: 30k steps, about 6.1 effective epochs over the LC0 train split

This run is intentionally LC0-only. The earlier mixed-data runs were useful for
system validation, but they were too exposed to repeated-cache artifacts and hard
TCEC positions to treat as final model-selection evidence.

Current live work:

- A larger LC0 test80 H8 build is running on a CPU data VM. It targets about
  10M samples from the next 640 LC0 archives and uses retry/backoff plus
  staggered workers to avoid LC0 archive rate limits.
- Two short TCEC Phase-A legality ablations are running for 15k steps with
  first-step legality weights 10 and 20. These are diagnostic runs while the
  larger LC0 data finishes, not final model-selection runs.

## BT4 and Projectors

BT4 stays frozen for DFM Phase A/B/C, TCEC diagnostics, and initial JEPA runs.
Only consider unfreezing after DFM has held-out legal/plausible action chunks
and JEPA beats identity/random-action baselines on changed-square metrics. The
first unfreeze should be adapter/LoRA-style or last-block-only with a smaller
learning rate; full BT4 fine-tuning is a late joint-refinement stage.

Standalone DFM keeps its compact current-state projector because its targets
are actions/legal masks. Standalone JEPA and joint JEPA use raw frozen BT4
tokens as current inputs and future targets.
Joint Latent-SASA should use:

- compact DFM BT4-to-planning-latent projector
- small DFM adapter
- JEPA action embedding plus DFM-action-hidden adapter into BT4 width
- raw `stopgrad(BT4(s_t))` JEPA input and raw `stopgrad(BT4(s_{t+h}))` targets
- DFM action-token hidden-state conditioning into the JEPA transition

## Phase A Loss

The Phase A DFM command uses fixed `t=0` masking for the supervised horizon:

```text
loss = CE(actions[:, 0])
     + first_legality_loss_weight * mean(sum_illegal_probs(step=0))
```

For the completed LC0 1M diagnostic run:

```text
first_action_loss_weight = 1.0
first_legality_loss_weight = 5.0
horizon_legality_loss_weight = 0.0
loss_horizon = 1
train_deterministic_t = 0.0
```

That diagnostic double-counted first-move CE because masked CE already includes
the first action when `loss_horizon=1` and `t=0`. The next clean LC0 10M Phase A
run should use:

```text
first_action_loss_weight = 0.0
loss_horizon = 1
train_deterministic_t = 0.0
learning_rate = 6e-4
```

Choose the legality coefficient by matching the random-policy scale:

```text
lambda_legal = log(1858) / (1 - avg_legal_moves / 1858)
```

The expected value is around `7.5-8.0`, but use the measured LC0 10M average
legal-move count before launch.

For the first queued baseline, default to `7.64` if the dataset average is not
available yet:

```text
loss = CE(a0) + 7.64 * illegal_prob_mass(a0)
```

For the first H4 action-only baseline:

```text
loss = CE(a0:a3)
     + 7.64 * illegal_prob_mass(a0)
     + 7.64 * teacher_forced_illegal_mass(a1:a3)
```

Validation should include fixed `t` slices, not just the training distribution,
so we can separate first-move legality, first-move accuracy, and denoising
behavior at harder noise levels.

## Latent-SASA Defaults

`docs/implementation_plan_gold.md` is the full design reference for coupling,
and `docs/latent_sasa_coupling.md` is the implementation checklist. The current
repo has standalone DFM and JEPA baselines plus Stage 1 and Stage 2 joint
trainer paths. The next missing research piece is reranking evaluation.

Initial JEPA baselines:

```text
H = 2, then H = 4
token_dim = 256
num_layers = 4
num_heads = 4
mlp_dim = 1024
learning_rate = 1e-4
loss = latent_cosine + 0.01 * sigreg
value_coeff = 0.0
wdl_coeff = 0.0
```

`scripts/train_jepa.py` now supports the same trainer-side GCS cache pattern as
DFM: `--gcs-train-prefix`, `--gcs-val-prefix`, cache-all startup, validation
batches, W&B credential preflight, and single-step GCS checkpoint resume. Use
the `jepa_latent` loader view for JEPA-only baselines so legal masks are not
loaded. JEPA logs approximate MFU from precomputed shape-based FLOP estimates;
actual BT4 current/future encodings are still computed at runtime.

Initial joint baselines:

```text
H = 2, then H = 4
token_dim = 256
num_layers = 4
num_heads = 4
mlp_dim = 1024
learning_rate = 3e-4
lambda_action = 1.0
lambda_first_legal = 7.64
lambda_horizon_legal = 0.0
lambda_jepa = 1.0
lambda_value = 0.0 initially
lambda_wdl = 0.0 initially
lambda_rank = 0.0 first, then 0.2
```

For queued Stage 1 runs after the raw-BT4 fix, use `learning_rate=6e-4`
and `legality_on_masked_only=true`. `--target-projector-mode` is deprecated
and ignored; JEPA input/targets are frozen raw BT4 tokens with shape
`[B, 64, 1024]` and `[B, H, 64, 1024]`. The legacy `--legality-coeff` flag is
only an alias for `--first-legality-coeff`.

Do not sweep batch size. Probe the largest batch that fits the allocated TPU
shape and keep the learning rate fixed for the first baseline queue.

## Joint Implementation Gates

Before launching non-smoke `joint` experiments, all of these must be true:

- Done: `LeelaChunkDataLoader` supports a `joint_latent_sasa` view with
  `current_planes`, `action_indices`, `future_planes`, `future_valid`, `valid`,
  and compact `legal_idx/legal_count`.
- Done: DFM exposes `planner_from_latents(..., return_hidden=True)` and returns
  `hidden["action_tokens"]` shaped `[B, H, D]`.
- Done: JEPA exposes `jepa_rollout_from_latents(z0_jepa, actions, action_hidden)` and
  can consume DFM action-token hidden states.
- Done: the joint model uses a compact DFM projector/adapter for action
  denoising, but JEPA consumes raw frozen BT4 current tokens and predicts raw
  frozen BT4 future tokens. There is no trainable JEPA target projector in the
  active path.
- Done: a one-step joint smoke test saves and restores a raw NumPy checkpoint under
  `runs/joint/<run_id>/checkpoints`.
- Done: coupling diagnostics show finite, non-zero JEPA-loss gradients into the DFM
  planner/action-token path and no gradients into frozen BT4.

Stage 1 joint loss:

```text
L = 1.0 * L_dfm_ce
  + lambda_first * L_first_legal
  + lambda_horizon * L_horizon_legal
  + 1.0 * L_jepa_positive
  + lambda_sigreg * L_jepa_sigreg
  + lambda_action_contrast * L_action_contrast
```

Use `H=1 or 2`, `K=1`, `token_dim=256`, DFM `L4`, JEPA `L1-L2`, and
`value_coeff=wdl_coeff=0.0` until latent dynamics is stable.

Current Stage 1 defaults:

```text
first_legality_coeff = 7.64
horizon_legality_coeff = 0.0
legality_on_masked_only = true
jepa_target_space = raw_bt4_tokens
jepa_num_heads = 8 for BT4 width 1024 unless the run explicitly overrides it
jepa_mlp_dim = 4096 by default, or a smaller explicit ablation
jepa_sigreg_coeff = 0.0 initially, then 0.01 ablation
jepa_action_contrast_coeff = 0.0 initially, then 0.1-0.5 ablation
```

`L_horizon_legal` applies only to later teacher-forced horizon slots and should
be treated as an ablation because it is not sampler legality. JEPA still trains
with normalized cosine distance, but the trainer logs raw MSE, normalized MSE,
token norms, per-horizon JEPA losses, identity baseline, and shuffled-action
baseline diagnostics. `L_action_contrast = max(margin + L_true_actions -
L_shuffled_actions, 0)` is the next ablation if the diagnostics show true and
shuffled action sequences scoring equally.

Stage 2 joint loss:

```text
L = L_dfm_ce
  + lambda_legal * L_legal
  + L_jepa_positive
  + 0.5 * L_contrast
```

Use legal-prefix corruptions from stored legal sets, start with `H=4`, `K=2-3`,
and flatten candidates to `[B*K, ...]`. If `legal_count <= 1`, mask that
candidate/sample out of the contrastive loss.

Stage 2 support is implemented locally and logs contrastive accuracy,
positive/negative similarity, margin, and valid fraction. Stage 3 adds
rank/reranking only after Stage 2 held-out contrastive accuracy is above chance.
Stage 4 adds all-mask distillation only after reranking beats raw DFM.
Distillation candidate log-probabilities must come from an all-mask pass, not
from the clean candidate pass where the candidate tokens were visible.

## Next Runs

Do not launch the combined TCEC+LC0 run until all of these are true:

- the exact-deduplicated prefix has a complete `manifest.json`
- at least one train, validation, and test shard passes trajectory-v2 rollout
  validation
- the launch spec points at the deduplicated prefix, not a partial raw prefix
- `--gcs-startup-cache-policy all` is explicit in the run command

After the larger LC0 dataset is complete and validated, continue in this order:

1. Phase A rerun: train on the full validated LC0 10M dataset with
   `loss_horizon=1`, fixed `t=0`, cache-all startup, `learning_rate=6e-4`, and
   the measured random-policy-balanced legality coefficient. Do not sweep Phase
   A before this run.
2. Phase B: branch from Phase A with `--init-checkpoint-uri`, increase
   `loss_horizon` to 2-4, and keep first-move legality pressure.
3. Phase C: train full H8 with scheduled or random `t` and held-out validation.
4. Phase D: evaluate sampler/refinement behavior, including first-move legal
   rate and full exact-rollout legal sequence rate.
5. Mixed-data run: repeat the curriculum on exact-deduplicated TCEC+LC0 after
   the dataset is validated.

## Naming

- Prefix run IDs with the phase, for example `dfm-k4-b256-lr1e3-...` or
  `jepa-h8-terminal-false-...`.
- Use one checkpoint directory per run ID. Do not point two live TPU workers at
  the same checkpoint URI.
- Use separate W&B groups: `smoke`, `dfm-sweep`, `jepa-sweep`, `profile-sweep`,
  and `combined-sweep`.
- Keep `.local.json` specs ignored. Public examples should stay placeholder-only.

## Smoke Guardrail

Before any sweep, run a 1-3 step smoke using the same training script and shard
family as the intended phase. The controller now validates specs before creating
TPU resources, uploads a fresh source snapshot by default, and deletes resources
after completion, failure, or allocation timeout.

```bash
uv run python -u scripts/run_tpu_spot_jepa.py \
  --job-spec docs/tpu_ondemand_smoke_current.local.json
```

The current known-good pattern is a small Spot TPU VM smoke with `scripts/train_dfm.py`,
`--steps 3`, `--batch-size 2`, and one trajectory shard. On April 27, 2026, this
path completed on a `v5litepod-1` Spot TPU in `us-east1-c`, wrote three
checkpoints, and cleaned up the TPU resource.

## Sweep Launch Safety

`scripts/sweep_manager.py` is intentionally fail-closed:

- It refuses to launch unless `--allow-launch` is set.
- It refuses launches without an explicit `--phase`.
- For `--phase dfm`, entries must run `scripts/train_dfm.py`.
- For `--phase jepa`, entries must run `scripts/train_jepa.py`.
- For `--phase profile`, entries must run `scripts/profile_jepa_tpu.py`.
- Failed experiments are not requeued unless `--requeue-on-failure` is set.
- For GCS-backed DFM training, use `--gcs-startup-cache-policy all` unless the
  experiment is explicitly testing streaming behavior.

For the next sweeps, prefer `scripts/run_experiment_queue.py` inside the TPU VM:

```bash
python scripts/run_experiment_queue.py \
  --queue-uri gs://bucket/queues/latent_sasa_baselines.jsonl \
  --workdir /tmp/chess_dfm_jax/repo \
  --status-dir /tmp/chess_dfm_jax/artifacts/experiment_queue \
  --status-uri gs://bucket/runs/queues/latent_sasa_baselines
```

The queue runner validates that experiment IDs are unique, phase commands point
at the expected training script, and no two queue entries write to the same
checkpoint URI. A code/data failure stops the queue by default. Use
`--keep-going` only for explicitly independent diagnostic queues. On relaunch it
syncs existing status files from `--status-uri` and skips entries already marked
`completed` unless `--rerun-completed` is set.

Checkpointing remains raw NumPy locally. The next implementation step is an
asynchronous GCS uploader for completed local checkpoints and queue status that
reports both local and uploaded latest steps. Do not migrate to Orbax until the
queue and loader paths are stable.

For curriculum branches, use `--init-checkpoint-uri` to initialize from the
latest or selected checkpoint of a prior run while writing to a new run ID.
GCS initialization and resume sync only the selected `stepXXXXXXX/state.npz`
instead of copying every checkpoint in the source directory.

If W&B logging is enabled, `scripts/train_dfm.py` checks for non-interactive
credentials before TPU initialization, model loading, checkpoint restore, or
cache warming. Provide `WANDB_API_KEY`, a restricted `~/.netrc`, set
`WANDB_MODE=offline`, or pass `--no-wandb`.

Example dry validation:

```bash
uv run python scripts/sweep_manager.py \
  --experiments hparam_sweep.jsonl \
  --project-id your-gcp-project \
  --phase dfm
```

Example launch, after setting local env/config:

```bash
export CHESS_DFM_BUCKET_PREFIX=your-bucket-prefix
export CHESS_DFM_SERVICE_ACCOUNT=trainer@your-gcp-project.iam.gserviceaccount.com
export CHESS_DFM_SOURCE_SNAPSHOT_URI=gs://your-bucket/source_snapshots/current.tar.gz

uv run python scripts/sweep_manager.py \
  --experiments hparam_sweep.jsonl \
  --project-id your-gcp-project \
  --phase dfm \
  --allow-launch
```

If the project only has the default compute service account and it has bucket
read/write permission, that is enough for smoke runs. A dedicated trainer service
account is still preferred for long sweeps because permissions are easier to audit.

## Monitoring

Use the marimo browser for both data and run status:

```bash
uvx marimo check notebooks/state_action_training_browser.py
uv run notebooks/state_action_training_browser.py
```

For interactive use, point `Local status config` at an ignored spec such as
`docs/tpu_ondemand_smoke_current.local.json` or `docs/tpu_spot_job_spec.local.json`.
The notebook reads `status.json`, checkpoint step, metadata, and W&B when those
fields are present. For DFM specs, it also extracts `--checkpoint-uri` from
`entry_command`.
