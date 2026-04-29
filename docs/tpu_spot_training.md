# Spot TPU Training

The TPU-safe path in this repo targets single-host TPU VMs. The local controller
lives in `scripts/run_tpu_spot_jepa.py` and uploads an immutable source snapshot
to GCS before each launch attempt. Use this controller for smoke tests and
phase-specific sweeps, but keep the training commands and checkpoint namespaces
separate as described in `docs/training_phases.md`.

## Required GCP setup

- Enable the Cloud TPU API and Cloud Storage API in your target project.
- Create one regional bucket per fallback region. The default examples assume `us-central1` and `europe-west4`.
- Upload BT4 model files to a bucket path such as `gs://.../models/`.
- Upload trajectory-v2 shard data to a bucket path such as `gs://.../chunks/trajectory-v2-example/`.
- Use a TPU VM service account with storage read/write access to the chosen buckets.
  A project default compute service account is acceptable for smoke tests if it
  already has those permissions; a dedicated trainer service account is preferred
  for longer sweeps.
- Use Application Default Credentials on the local controller machine so the Python Google Cloud clients can create queued resources and upload artifacts.

## Job spec

Start from `docs/tpu_spot_job_spec.example.json` for a public-safe template.

If you want a private local copy for actual runs:

1. Copy `docs/tpu_spot_job_spec.local.example.json` to `docs/tpu_spot_job_spec.local.json`
2. Fill in your project, buckets, and service account
3. Keep the `.local.json` file untracked

Important fields:

- `zone_order`: ordered fallback zones for Spot capacity.
- `bucket_by_region`: same-region buckets for source snapshots, checkpoints, and status files.
- `models_uri_by_region`: per-region bucket directories containing `BT4_exported.pb.gz` and the other BT4 artifacts.
- `chunk_data_uri_by_region`: per-region bucket directories containing the chunk files.
- `train_args`: arguments forwarded directly to `scripts/train_jepa.py`.
- `entry_command`: optional full command. Use this for DFM smoke/sweep specs that
  call `scripts/train_dfm.py`; otherwise `train_args` renders a JEPA command.

## Launch

Run the controller from the repo root:

```bash
uv run python -u scripts/run_tpu_spot_jepa.py --job-spec docs/tpu_spot_job_spec.local.json
```

What the controller does:

1. Validates the job spec before creating TPU resources.
2. Creates a source tarball of the repo, excluding top-level `data/`, `models/`,
   `runs/`, `wandb/`, local config, caches, and `.venv/`.
3. Uploads that tarball to the same-region bucket for the current zone attempt.
4. Requests a queued resource with a TPU VM startup script.
5. The startup script installs `jax[tpu]`, installs the repo, downloads models
   and chunk data, and starts the rendered training command.
6. The trainer saves checkpoints every `save-every` steps and writes
   `status.json` on job start and exit.
7. The controller deletes the queued resource after completion, job failure,
   resource failure, or allocation timeout.

## Smoke Test

Use an ignored local spec for smoke tests:

```bash
uv run python -u scripts/run_tpu_spot_jepa.py \
  --job-spec docs/tpu_ondemand_smoke_current.local.json
```

The current verified smoke pattern is a 3-step DFM run on one small trajectory
shard. It proves TPU provisioning, source snapshot packaging, dependency install,
data/model download, training, checkpoint upload, `status.json`, artifact upload,
and cleanup before spending budget on a phase sweep.

## Limits and current assumptions

- The controller is written for **single-host** TPU jobs only.
- It assumes `gcloud storage cp` is available on the TPU VM runtime for startup-script file transfers.
- It assumes exactly one writer per checkpoint directory. Do not launch multiple trainers against the same checkpoint URI at the same time.
- The controller loop is designed for long-running batch jobs, not interactive SSH sessions.
- On-demand capacity can fail even for valid accelerator types. Prefer one
  supervised smoke attempt at a time, then expand to phase sweeps only after
  checkpoint and cleanup are verified.
