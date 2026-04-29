# Training Phases

Keep smoke tests, phase-wise sweeps, and large combined runs separated. The TPU
controller is shared infrastructure; the training command, checkpoint URI, W&B
group, and run ID define the experiment phase.

## Phase Order

1. `smoke`: smallest possible job that proves source packaging, TPU startup,
   data download, training, checkpoint upload, status upload, and cleanup.
2. `dfm`: action-only DFM sweeps. Use `scripts/train_dfm.py` and checkpoint
   under `runs/dfm/<run_id>/checkpoints`.
3. `jepa`: latent future prediction sweeps. Use `scripts/train_jepa.py` and
   checkpoint under `runs/jepa/<run_id>/checkpoints`.
4. `profile`: profiling-only jobs. Use `scripts/profile_jepa_tpu.py`; never
   share checkpoint directories with training jobs.
5. `combined`: later plan-scoring/reranking runs after DFM and JEPA sweeps have
   stable phase winners.

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

## Phase A Loss

The Phase A DFM command uses fixed `t=0` masking for the supervised horizon:

```text
loss = first_action_loss_weight * CE(actions[:, 0])
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

Validation should include fixed `t` slices, not just the training distribution,
so we can separate first-move legality, first-move accuracy, and denoising
behavior at harder noise levels.

## Next Runs

Do not launch the combined TCEC+LC0 run until all of these are true:

- the exact-deduplicated prefix has a complete `manifest.json`
- at least one train, validation, and test shard passes trajectory-v2 rollout
  validation
- the launch spec points at the deduplicated prefix, not a partial raw prefix
- `--gcs-startup-cache-policy all` is explicit in the run command

After the larger LC0 dataset is complete and validated, continue in this order:

1. Phase A rerun: train on the larger LC0 dataset with `loss_horizon=1`, fixed
   `t=0`, cache-all startup, and a legality weight chosen from the short TCEC
   ablations or a small LC0 smoke.
2. Phase B: resume from Phase A, increase `loss_horizon` to 2-4, and keep
   first-move legality pressure.
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
