#!/usr/bin/env python3
"""Launch the same-run Phase A set2 -> set1+set2 continuation chain."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from chess_dfm_jax.tracking import load_env_file
from chess_dfm_jax.training.tpu_jobs import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "docs" / "dfm_lc010m_phasea_current.local.json"
ZONE = "us-west1-c"
REGION = "us-west1"
SET1 = "gs://gunay-chess-experiments-us-central1/data/trajectory_v2_lc0_test80_h8_10m_retry_20260429"
SET2 = "gs://gunay-chess-experiments-us-central1/data/trajectory_v2_lc0_test80_h8_10m_skip704_20260429"
RUN_ID = "dfm-lc010m-h8-phaseA-h1-lr6e4-leg764-20260429"
SET2_TPU_NAME = "dfm-phasea-resume-set2-20260430"
COMBINED_STAGE = "combined_same_vm"
CACHE_MOUNT = "/mnt/chess-dfm-cache"
CACHE_DIR = f"{CACHE_MOUNT}/gcs_cache"


def run_cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def train_command(
    *,
    run_id: str,
    bucket: str,
    train_prefix: str,
    val_prefix: str,
    steps: int,
    wandb_project: str,
) -> str:
    parts = [
        "/tmp/venv/bin/python",
        "-u",
        "scripts/train_dfm.py",
        "--backend",
        "tpu",
        "--steps",
        str(steps),
        "--batch-size",
        "192",
        "--horizon",
        "8",
        "--gcs-train-prefix",
        train_prefix,
        "--gcs-val-prefix",
        val_prefix,
        "--gcs-startup-cache-policy",
        "all",
        "--gcs-prefetch-workers",
        "16",
        "--gcs-cache-dir",
        CACHE_DIR,
        "--models-dir",
        "/tmp/chess_dfm_jax/repo/models",
        "--save-dir",
        "/tmp/chess_dfm_jax/artifacts",
        "--checkpoint-uri",
        f"{bucket}/runs/dfm/{run_id}/checkpoints",
        "--run-id",
        run_id,
        "--resume",
        "--token-dim",
        "640",
        "--num-layers",
        "8",
        "--num-heads",
        "10",
        "--mlp-dim",
        "2560",
        "--learning-rate",
        "6e-4",
        "--weight-decay",
        "1e-4",
        "--use-muon",
        "--encoder-dtype",
        "bfloat16",
        "--head-compute-dtype",
        "bfloat16",
        "--loss-horizon",
        "1",
        "--train-deterministic-t",
        "0.0",
        "--first-action-loss-weight",
        "0.0",
        "--first-legality-loss-weight",
        "7.64",
        "--horizon-legality-loss-weight",
        "0.0",
        "--val-batches",
        "32",
        "--val-every",
        "1000",
        "--val-deterministic-t",
        "0.0",
        "--val-t-values",
        "0.1,0.5,0.9",
        "--save-every",
        "1000",
        "--log-every",
        "20",
        "--peak-tflops",
        "197",
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        "dfm-lc0-phaseA",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def delete_tpu(*, project_id: str, name: str) -> None:
    run_cli(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            name,
            f"--project={project_id}",
            f"--zone={ZONE}",
            "--quiet",
        ],
        check=False,
    )


def launch_combined_on_same_tpu(base: dict, bucket: str, work_dir: Path) -> None:
    command = train_command(
        run_id=RUN_ID,
        bucket=bucket,
        train_prefix=f"{SET1}/train,{SET2}/train",
        val_prefix=f"{SET1}/val,{SET2}/val",
        steps=196980,
        wandb_project=base.get("wandb_project", "chess_dfm_jax-jepa-sweep"),
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    env_path = work_dir / "wandb.env"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        env_path.write_text(f"WANDB_API_KEY={shlex.quote(wandb_key)}\n", encoding="utf-8")
        run_cli(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "scp",
                str(env_path),
                f"{SET2_TPU_NAME}:/tmp/chess_dfm_wandb.env",
                f"--project={base['project_id']}",
                f"--zone={ZONE}",
            ]
        )
        env_path.unlink(missing_ok=True)

    status_uri = f"{bucket}/runs/dfm/{RUN_ID}/status.json"
    log_file = "/var/log/chess_dfm_jax-phasea-combined.log"
    remote_script = f"""#!/bin/bash
set -euo pipefail
export HOME=/root
CACHE_MOUNT={shlex.quote(CACHE_MOUNT)}
CACHE_DEVICE=""
for candidate in /dev/disk/by-id/google-persistent-disk-1 /dev/disk/by-id/scsi-0Google_PersistentDisk_persistent-disk-1; do
  if [ -e "$candidate" ]; then
    CACHE_DEVICE="$candidate"
    break
  fi
done
if [ -n "$CACHE_DEVICE" ]; then
  if ! sudo blkid "$CACHE_DEVICE" >/dev/null 2>&1; then
    sudo mkfs.ext4 -F -m 0 "$CACHE_DEVICE"
  fi
  sudo mkdir -p "$CACHE_MOUNT"
  if ! mountpoint -q "$CACHE_MOUNT"; then
    sudo mount -o discard,defaults "$CACHE_DEVICE" "$CACHE_MOUNT"
  fi
  sudo chmod 777 "$CACHE_MOUNT"
  mkdir -p "$CACHE_MOUNT/gcs_cache/train" "$CACHE_MOUNT/gcs_cache/val"
fi
if [ -d /tmp/chess_dfm_jax/gcs_cache/train ] && [ -d "$CACHE_MOUNT/gcs_cache/train" ]; then
  rsync -a --ignore-existing --include="*.npz" --exclude="*" /tmp/chess_dfm_jax/gcs_cache/train/ "$CACHE_MOUNT/gcs_cache/train/" || true
fi
if [ -d /tmp/chess_dfm_jax/gcs_cache/val ] && [ -d "$CACHE_MOUNT/gcs_cache/val" ]; then
  rsync -a --ignore-existing --include="*.npz" --exclude="*" /tmp/chess_dfm_jax/gcs_cache/val/ "$CACHE_MOUNT/gcs_cache/val/" || true
fi
if [ -f /tmp/chess_dfm_wandb.env ]; then
  set -a
  source /tmp/chess_dfm_wandb.env
  set +a
  rm -f /tmp/chess_dfm_wandb.env
fi
cd /tmp/chess_dfm_jax/repo
python3 - <<'PY' >/tmp/chess_dfm_jax_status.json
import json, time
print(json.dumps({{"state": "running", "stage": "{COMBINED_STAGE}", "run_id": "{RUN_ID}", "zone": "{ZONE}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log_file": "{log_file}"}}))
PY
gcloud storage cp /tmp/chess_dfm_jax_status.json {shlex.quote(status_uri)} --quiet || true
set +e
{command} >>{shlex.quote(log_file)} 2>&1
STATUS=$?
set -e
python3 - <<PY >/tmp/chess_dfm_jax_status.json
import json, time
status = int("$STATUS")
print(json.dumps({{"state": "completed" if status == 0 else "failed", "exit_code": status, "stage": "{COMBINED_STAGE}", "run_id": "{RUN_ID}", "zone": "{ZONE}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log_file": "{log_file}"}}))
PY
gcloud storage cp /tmp/chess_dfm_jax_status.json {shlex.quote(status_uri)} --quiet || true
exit "$STATUS"
"""
    remote_script_path = work_dir / "run_combined_same_vm.sh"
    remote_script_path.write_text(remote_script, encoding="utf-8")
    run_cli(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            str(remote_script_path),
            f"{SET2_TPU_NAME}:/tmp/run_combined_same_vm.sh",
            f"--project={base['project_id']}",
            f"--zone={ZONE}",
        ]
    )
    remote_script_path.unlink(missing_ok=True)
    run_cli(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            SET2_TPU_NAME,
            f"--project={base['project_id']}",
            f"--zone={ZONE}",
            "--command=chmod +x /tmp/run_combined_same_vm.sh && nohup /tmp/run_combined_same_vm.sh >/tmp/run_combined_same_vm.nohup 2>&1 &",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-s", type=int, default=120)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/chess_dfm_phasea_chain"))
    args = parser.parse_args()

    load_env_file()
    base = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    bucket = base["bucket_by_region"][REGION].rstrip("/")
    status_uri = f"{bucket}/runs/dfm/{RUN_ID}/status.json"
    combined_launched = False

    while True:
        status = read_json(status_uri) or {}
        state = status.get("state")
        stage = status.get("stage")
        print(json.dumps({"event": "poll", "run_id": RUN_ID, "state": state, "status": status}), flush=True)
        if state == "completed" and stage == COMBINED_STAGE:
            delete_tpu(project_id=base["project_id"], name=SET2_TPU_NAME)
            return 0
        if state == "completed" and not combined_launched:
            launch_combined_on_same_tpu(base, bucket, args.work_dir)
            combined_launched = True
        if state == "failed" and stage == COMBINED_STAGE:
            delete_tpu(project_id=base["project_id"], name=SET2_TPU_NAME)
            return 1
        if state == "failed":
            delete_tpu(project_id=base["project_id"], name=SET2_TPU_NAME)
            print("Phase A set2 continuation failed; not launching combined continuation.", flush=True)
            return 1
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
