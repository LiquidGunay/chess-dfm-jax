# Spot TPU Training

The first TPU-safe path in this repo targets single-host `v5e-8` Spot TPU VMs. The local controller lives in `scripts/run_tpu_spot_jepa.py` and uploads an immutable source snapshot to GCS before each launch attempt.

## Required GCP setup

- Enable the Cloud TPU API and Cloud Storage API in project `project-b9551f07-5f68-491a-8a0`.
- Create one regional bucket per fallback region. The current controller expects one bucket in `us-central1` and one in `europe-west4`.
- Upload BT4 model files to a bucket path such as `gs://.../models/`.
- Upload the chunk training data directory to a bucket path such as `gs://.../chunks/training-run2--20251215-1017/`.
- Create a TPU VM service account with storage read/write access to the chosen buckets.
- Use Application Default Credentials on the local controller machine so the Python Google Cloud clients can create queued resources and upload artifacts.

## Job spec

Start from `docs/tpu_spot_job_spec.example.json` and fill in the bucket names and service account. The important fields are:

- `zone_order`: ordered fallback zones for Spot capacity.
- `bucket_by_region`: same-region buckets for source snapshots, checkpoints, and status files.
- `models_uri_by_region`: per-region bucket directories containing `BT4_exported.pb.gz` and the other BT4 artifacts.
- `chunk_data_uri_by_region`: per-region bucket directories containing the chunk files.
- `train_args`: arguments forwarded directly to `scripts/train_jepa.py`.

## Launch

Run the controller from the repo root:

```bash
../.venv/bin/python scripts/run_tpu_spot_jepa.py --job-spec docs/tpu_spot_job_spec.example.json
```

What the controller does:

1. Creates a source tarball of the repo, excluding `data/`, `models/`, `runs/`, `wandb/`, and `.venv/`.
2. Uploads that tarball to the same-region bucket for the current zone attempt.
3. Requests a Spot queued resource with a TPU VM startup script.
4. The startup script installs `jax[tpu]`, installs the repo, downloads models and chunk data, and starts `scripts/train_jepa.py --resume --checkpoint-uri gs://...`.
5. The trainer saves Orbax checkpoints every `save-every` steps and writes `status.json` on job start and exit.
6. If capacity is unavailable or the TPU is preempted, the local controller retries in the next zone and resumes from the latest checkpoint.

## Limits and current assumptions

- The controller is written for **single-host** TPU jobs only.
- It assumes `gcloud storage cp` is available on the TPU VM runtime for startup-script file transfers.
- It assumes exactly one writer per checkpoint directory. Do not launch multiple trainers against the same checkpoint URI at the same time.
- The controller loop is designed for long-running batch jobs, not interactive SSH sessions.
