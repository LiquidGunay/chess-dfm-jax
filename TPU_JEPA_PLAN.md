# TPU-Safe Token JEPA Plan

This file captures the current implementation plan and the choices already made for running JEPA training safely on Spot TPUs from this repo.

## Summary

The training path is built around a frozen LC0 BT4 encoder plus a trainable token-level JEPA head. BT4 emits one token per square. The JEPA head now consumes raw frozen BT4 tokens directly, conditions them on action tokens, runs a small transformer, and predicts the next state's raw frozen BT4 tokens. The training loss is anchored to BT4 token space, not a trainable projected target space.

The cloud path uses multi-host `v5litepod-16` Spot TPU VMs. A local controller uploads an immutable source snapshot to GCS, requests a queued resource with a startup script, and retries across an ordered zone list. Checkpointing uses Orbax async checkpoints to GCS so the next launch can resume after preemption. Multi-host coordination is handled via `jax.distributed.initialize()`.

## Architecture

- Frozen BT4 encoder is loaded from the pinned LC0 BT4 model files.
- JEPA trainable head:
  - raw current tokens: `64 x 1024`
  - action embedding: policy-index vocabulary to BT4 width `1024`
  - action token is added to the 64 board tokens at each rollout step
  - small transformer over `64` square tokens
  - output is the predicted next-state `64 x 1024` BT4 token tensor
- Default training config:
  - raw JEPA width is `1024`
  - `token_dim=256` is retained only as a compatibility/hidden-size knob
  - `num_layers=4`
  - `num_heads=8`
  - `mlp_dim=1024`
  - `action_source=best`
  - GPU path: encoder `float16`, head params `float32`, head compute `float32`
  - TPU path: encoder `bfloat16`, head params `float32`, head compute `bfloat16`

## Checkpointing and Resume

- Checkpoints use Orbax `CheckpointManager`.
- Only the trainable JEPA head and optimizer state are checkpointed.
- Frozen BT4 weights are never checkpointed; they are reloaded from the pinned model file.
- Checkpoint payload contains:
  - step
  - trainable JEPA state
  - optimizer state
- Default save cadence is every `100` steps.
- The trainer catches `SIGTERM` and writes a final checkpoint before exit.
- Resume uses the latest step in the checkpoint directory.
- One checkpoint directory must have exactly one active writer.

## Spot TPU Execution Model

- First production target: single-host `v5litepod-8` Spot TPU VM.
- Ordered fallback zones:
  - `us-central1-a`
  - `europe-west4-b`
- One regional bucket per region is used so source snapshots, checkpoints, and status files stay close to the TPU.
- Local controller responsibilities:
  - tar the repo into an immutable source snapshot
  - upload the snapshot to the same-region bucket
  - create the queued resource request
  - monitor resource state and job status
  - retry on allocation failure or preemption
- TPU startup script responsibilities:
  - download the exact source snapshot
  - install `jax[tpu]`
  - install the repo
  - sync model files and chunk data from GCS
  - launch `scripts/train_jepa.py --resume --checkpoint-uri gs://...`
  - write job status back to GCS

## Interfaces

- Local trainer: `scripts/train_jepa.py`
  - supports resume, checkpoint-uri override, token-level config, W&B ids/groups, and platform tags
- Local TPU controller: `scripts/run_tpu_spot_jepa.py`
  - consumes a JSON spec and manages queued-resource retries
- TPU profiling entrypoint: `scripts/profile_jepa_tpu.py`
  - saves a representative trace, a small arithmetic-intensity sweep, and device metadata into one artifact directory
- TPU controller spec: `docs/tpu_spot_job_spec.example.json`
- Main cloud helper module: `chess_dfm_jax/training/tpu_jobs.py`
- Notebook interfaces:
  - `notebooks/training_jepa.py`
  - `notebooks/analyze_jepa.py`

## Known Limits

- The first cloud path is single-host only.
- The current chunk loader’s sample-level shuffle path breaks board/move alignment for JEPA transition construction. Training now avoids that path by using file shuffling and `shuffle_buffer=0`.
- The TPU controller is implemented but has not yet been run end-to-end against the real GCP project from this machine.

## Files Added or Reworked

- `chess_dfm_jax/training/jepa.py`
- `chess_dfm_jax/training/checkpoints.py`
- `chess_dfm_jax/training/tpu_jobs.py`
- `scripts/train_jepa.py`
- `scripts/run_tpu_spot_jepa.py`
- `notebooks/training_jepa.py`
- `notebooks/analyze_jepa.py`
- `docs/tpu_spot_training.md`
- `docs/tpu_spot_job_spec.example.json`

## Validation Done

- Local compile pass succeeded for `chess_dfm_jax`, `scripts`, and `notebooks`.
- Local GPU smoke run succeeded:
  - `steps=2`
  - `batch_size=2`
  - loss decreased from about `1.19` to `0.84`
- Local resume path succeeded:
  - a run checkpointed at step `1`
  - a second invocation resumed and restored the optimizer step correctly

## Inputs Still Needed From The User For Real GCP Bring-Up

- final GCS bucket names for `us-central1` and `europe-west4`
- TPU VM service account email
- whether to use the default VPC/subnetworks or specific custom subnetworks
- exact GCS location for model files
- exact GCS location for chunk training data
- local machine GCP authentication method:
  - preferred: install `gcloud` and run ADC login
  - fallback: provide a service-account JSON key path
