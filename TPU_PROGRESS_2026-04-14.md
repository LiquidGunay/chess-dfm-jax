# TPU Profiling Progress

## Current Progress

- GCP project in use: `project-b9551f07-5f68-491a-8a0`
- Active account: `gunaysoni@gmail.com`
- Existing `default` VPC/subnets are being used for the first TPU path.
- Regional buckets created:
  - `gs://gunay-chess-experiments-us-central1`
  - `gs://gunay-chess-experiments-europe-west4`
- Model files are uploaded in `us-central1`:
  - `gs://gunay-chess-experiments-us-central1/models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`
  - `gs://gunay-chess-experiments-us-central1/models/BT4.onnx`
  - `gs://gunay-chess-experiments-us-central1/models/BT4_exported.pb.gz`
- The Spot TPU controller now has a `gcloud` CLI fallback and no longer depends on local ADC being configured.
- TPU specs and docs were corrected to use the TPU names that actually exist in `us-central1-a`:
  - accelerator family: `v5litepod-*`
  - runtime: `tpu-ubuntu2204-base`
- Added TPU profiling entrypoint:
  - `scripts/profile_jepa_tpu.py`
- Added TPU profile job spec:
  - `docs/tpu_profile_job_spec.project.json`
- Added theory-backed JEPA profiling support:
  - `chess_dfm_jax/analysis/jepa_theory.py`
  - `scripts/profile_jepa_sweep.py`
  - `scripts/run_roofline.py` now accepts token/head/MLP sizing args

## What Happened During Bring-Up

- First real queued-resource request used `v5litepod-8` and failed with a quota error:
  - `Quota 'TPUV5sPreemptibleLitepodServingPerProjectPerZoneForTPUAPI' exhausted. Limit 4 in zone us-central1-a`
- The profile spec was reduced to `v5litepod-4`.
- A follow-up `v5litepod-4` request successfully uploaded the source snapshot to GCS:
  - `gs://gunay-chess-experiments-us-central1/source_snapshots/jepa-v5litepod4-profile/...`
- The controller then hit a transient `NOT_FOUND` immediately after `create`, while polling `describe`.
- The controller was patched so that an immediate `NOT_FOUND` from `describe` is treated as control-plane propagation instead of a hard failure.

## Current State Of The Code

- `chess_dfm_jax/training/tpu_jobs.py`
  - defaults updated to `v5litepod-8` and `tpu-ubuntu2204-base`
  - `gcloud` CLI fallback added for storage and queued-resource control
  - transient `NOT_FOUND` on queued-resource `describe` now maps to a temporary `CREATING` state
- `scripts/profile_jepa_tpu.py`
  - saves:
    - `device_info.json`
    - `single_point.csv`
    - `single_point.json`
    - `sweep_points.csv`
    - `sweep_points.json`
    - `SUMMARY.txt`
    - TensorBoard-compatible JAX trace under `tb_trace/`
- `docs/tpu_profile_job_spec.project.json`
  - currently configured for a profile-only `v5litepod-4` Spot request in `us-central1-a`

## Next Steps

1. Retry the `v5litepod-4` Spot TPU profile request with the patched controller.
2. If the queued resource becomes active, wait for the TPU VM startup script to finish and confirm:
   - `status.json` in GCS moves from `booting` to `running` to `completed`
   - profile artifacts appear under `gs://gunay-chess-experiments-us-central1/runs/jepa/jepa-v5litepod4-profile/artifacts/`
3. Pull back the first TPU profiling bundle and inspect:
   - arithmetic intensity by batch size and token width
   - achieved throughput
   - device count and topology
   - TensorBoard/Perfetto trace
4. After the unsharded TPU profile succeeds, add a true multi-device sharded profiling path so the sweep reflects the multiple TPU devices rather than a single-device JIT.
5. Once profiling numbers are stable, retune:
   - batch sizes
   - `token_dim`
   - `mlp_dim`
   - any shape choices that avoid wasteful TPU padding
