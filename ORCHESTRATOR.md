# GCP Orchestrator & VM Configuration

This document explains how our experimental sweep runs are managed on Google Cloud using Spot (preemptible) TPUs, how the orchestrator VM works, and how to check on jobs.

## The Architecture

Because Spot TPUs are frequently preempted (reclaimed by Google when full-paying customers request capacity), it's impossible to run a long, uninterrupted training job on a single instance.

To solve this, we use a **headless orchestrator VM** running completely in the cloud (for example `your-orchestrator-vm` in `us-central1-a`).
This VM runs our `sweep_manager.py` script as a persistent background daemon.

1. **Job Distribution:** The orchestrator reads `hparam_sweep.jsonl` from a configured Google Cloud Storage path such as `gs://your-us-central2-bucket/configs/hparam_sweep.jsonl`.
2. **Quota Awareness & Floating:** It maintains a pool of 14 concurrent threads checking for available Spot TPU quota across `us-central2`, `us-central1`, and `europe-west4`. 
3. **Queue Bouncing (Preemption Recovery):** It requests a TPU via `gcloud compute tpus queued-resources create`. If a job gets preempted (`SUSPENDING` -> `FAILED`), the orchestrator cleanly deletes the dead queued resource, and puts a fresh request into the queue. If one zone is congested, the thread for an idle zone (e.g., `europe-west4`) can pick up the job and float it cross-region to dodge preemption.
4. **Resilience:** The TPUs save checkpoint state periodically to the configured regional bucket. If an instance dies, the next provisioned TPU resumes training from the exact same step seamlessly.

## How to Interact with the Orchestrator

The orchestrator is entirely autonomous, but you can interact with it and check its status.

### 1. Check Live TPUs in the Queue
To see which experiments have secured TPUs (`ACTIVE`/`PROVISIONING`) and which are waiting (`WAITING_FOR_RESOURCES`), run:

```bash
# Check US Central 2 (v4-16 pods)
gcloud compute tpus queued-resources list --zone=us-central2-b --project=your-gcp-project

# Check US Central 1 (v5litepod-16 pods)
gcloud compute tpus queued-resources list --zone=us-central1-a --project=your-gcp-project

# Check Europe West 4 (v5litepod-16 pods)
gcloud compute tpus queued-resources list --zone=europe-west4-b --project=your-gcp-project
```

### 2. Check the Orchestrator Logs
To see exactly what the orchestrator is doing (e.g. which runs it is currently spinning up or deleting after a preemption):

```bash
gcloud compute ssh your-orchestrator-vm --project=your-gcp-project --zone=us-central1-a --command="sudo cat /root/sweep_hparam.log | tail -n 50"
```

### 3. Check Live Training Progress (W&B)
To poll the Weights & Biases API and see the current loss and steps of the active sweep:

```bash
uv run python scripts/check_sweep.py
```

### 4. Updating the Sweep or Codebase
If you want to start new experiments or push a new code fix:
1. Update the code or `hparam_sweep.jsonl` locally.
2. Build a new `.tar.gz` source snapshot using `chess_dfm_jax.training.tpu_jobs.create_source_snapshot`.
3. Upload the snapshot to all regional buckets (for example `verified_source_example.tar.gz`).
4. Recreate the orchestrator VM or SSH into it and restart the `sweep_manager.py` daemon.
