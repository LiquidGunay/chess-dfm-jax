# TPU Spot Job Troubleshooting and Run Guide

This document captures the challenges faced while bringing up the Spot TPU execution pipeline for the JEPA model, the resolutions applied, and instructions on how to run jobs and monitor their progress.

## Problems Encountered and Resolutions

### 1. Spot Queue "Serving" vs. "Training" Quotas
**Problem:** We initially requested a single-host `v5litepod-4`. This was constantly hitting a hard 4-chip limit and failing or getting stuck in long queues. We discovered that GCP splits v5e quota into "Serving" (single-host, e.g., v5litepod-1/4/8) and "Training" (multi-host, e.g., v5litepod-16+). The user's 64-chip trial quota was exclusively for "Training".
**Resolution:** Updated the job spec to request a multi-host `v5litepod-16` instead, which successfully bypassed the 4-chip bottleneck and utilized the correct free-trial quota bucket.

### 2. Multi-Host JAX Initialization
**Problem:** Moving from a single-host (`v5litepod-4`) to a multi-host (`v5litepod-16`, which is 4 connected VMs) environment caused the training script to hang or fail because the separate hosts weren't communicating.
**Resolution:** Added `jax.distributed.initialize()` to the top of `scripts/profile_jepa_tpu.py` so the JAX runtimes on the 4 separate TPU VMs can form a cluster.

### 3. Controller Losing Queue Position
**Problem:** The `run_tpu_spot_jepa.py` controller had an `allocation_timeout_s` of 15 minutes. If a resource sat in the `WAITING_FOR_RESOURCES` or `PROVISIONING` state for more than 15 minutes, the controller would delete the request and recreate it, effectively throwing away our position at the back of the line continuously.
**Resolution:** Modified `chess_dfm_jax/training/tpu_jobs.py` to stop timing out jobs that are actively `WAITING_FOR_RESOURCES` or `PROVISIONING`. It now holds its position indefinitely until an instance is granted or the backend explicitly fails the request.

### 4. "Ghost" Resources Consuming Quota
**Problem:** If a Spot request was rejected due to zero capacity (or if the startup script failed and the job died), the Queued Resource transitioned to a `FAILED` state. However, a `FAILED` resource still consumes TPU quota in GCP. This blocked all subsequent retries.
**Resolution:** Patched the controller loop so that whenever a resource enters the `FAILED` or `SUSPENDED` state, the script explicitly calls `gcloud compute tpus queued-resources delete` to free up the quota *before* moving on to the next zone.

### 5. Python and Library Version Mismatches on TPU VMs
**Problem:** The Google-provided TPU VMs run Python 3.10. Our `pyproject.toml` required Python `>=3.11`. Additionally, the latest versions of libraries like `flax` and `orbax-checkpoint` require Python 3.11+. The startup script was crashing during `pip install`.
**Resolution:** Downgraded `requires-python` to `>=3.10` and pinned compatible library versions (`flax>=0.10.0`, `orbax-checkpoint>=0.10.0`, `marimo>=0.10.0`) in `pyproject.toml`.

### 6. Missing Startup Logs (Blind Failures)
**Problem:** When the TPU VMs failed to run the script (e.g., due to the Python mismatch), the VMs were immediately deleted by the controller as a "failed" job. The logs were lost on the ephemeral VMs, leaving us blind to the root cause.
**Resolution:** Modified the startup script template in `tpu_jobs.py` to tee all output to `/var/log/chess_dfm_jax-startup.log` and added a `gcloud storage cp` command to upload this log to the GCS `artifacts/` folder at the end of the script, regardless of success or failure.

### 7. GCS 404 Errors Crashing the Controller
**Problem:** When the controller checked the `status.json` file on GCS using `gcloud storage cat`, if the file didn't exist yet, it threw a "not found: 404" error. The Python wrapper didn't recognize this specific error string as a benign missing file and crashed.
**Resolution:** Added `"not found"` to the `missing_markers` array inside the `read_json` function in `tpu_jobs.py`.

### 8. Double-Nested Model Directories (FileNotFoundError)
**Problem:** The `gcloud storage cp --recursive` command was downloading the model directory into a nested structure (e.g., `repo/models/models/BT4...`). The profiling script was looking for them at the root of `repo/models/` and crashed with a `FileNotFoundError`.
**Resolution:** Appended the `/*` glob to the GCS URIs in the startup script template within `tpu_jobs.py` (e.g., `gs://bucket/models/*`). This forces GCS to place the files flatly inside the target directory.

---

## How to Run a Job

To launch a TPU job, you start the local spot controller with your desired job specification JSON file. The controller will automatically handle packaging your local code, uploading it to GCS, creating the TPU Queued Resource request, and polling for completion.

Because the controller might need to wait hours for Spot capacity, you should run it in the background (or using `tmux`/`screen`/`nohup`):

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the controller in the background, redirecting logs to a file
python -u scripts/run_tpu_spot_jepa.py --job-spec docs/tpu_profile_job_spec.local.json > spot_controller.log 2>&1 &
```

## How to Check Job Progress

Once the controller is running, you can monitor the lifecycle of your job through several layers:

### 1. The Orchestrator View (The Python Controller)
Check this to see if the controller is actively trying to get a TPU, waiting in the queue, or falling back to another zone.
```bash
tail -f spot_controller.log
```

### 2. The High-Level VM Status
Check this to see what the TPU VM is currently doing. The states are usually `booting` -> `running` -> `completed` (or `failed`).
```bash
# Check the primary zone (Europe)
gcloud storage cat gs://your-europe-west4-bucket/runs/jepa/jepa-profile-example/status.json

# Check the fallback zone (US-Central)
gcloud storage cat gs://your-us-central1-bucket/runs/jepa/jepa-profile-example/status.json
```

### 3. The Deep Dive (Startup Logs)
If the high-level status says `failed`, the controller is now configured to upload the exact console output of the TPU VM before it dies. Read this to find the Python Traceback or bash error:
```bash
gcloud storage cat gs://your-europe-west4-bucket/runs/jepa/jepa-profile-example/artifacts/startup.log
```

### 4. The Final Results
Once the job `status.json` says `completed`, your profiling data (MFU metrics, CSVs, TensorBoard traces) will be available in the artifacts folder:
```bash
gcloud storage ls gs://your-europe-west4-bucket/runs/jepa/jepa-profile-example/artifacts/profile_bundle/
```
