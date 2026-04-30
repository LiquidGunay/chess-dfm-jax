#!/usr/bin/env python3
"""Launch the Phase A set2 -> set1+set2 continuation chain."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from chess_dfm_jax.tracking import load_env_file
from chess_dfm_jax.training.tpu_jobs import (
    TPUJobSpec,
    create_source_snapshot,
    read_json,
    render_startup_script,
    upload_file,
    upload_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "docs" / "dfm_lc010m_phasea_current.local.json"
ZONE = "us-west1-c"
REGION = "us-west1"
SET1 = "gs://gunay-chess-experiments-us-central1/data/trajectory_v2_lc0_test80_h8_10m_retry_20260429"
SET2 = "gs://gunay-chess-experiments-us-central1/data/trajectory_v2_lc0_test80_h8_10m_skip704_20260429"
PHASE2_RUN = "dfm-lc010m-phasea2-set2-20260430"
PHASE3_RUN = "dfm-lc020m-phasea3-set1set2-20260430"


def run_cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def train_command(
    *,
    run_id: str,
    bucket: str,
    train_prefix: str,
    val_prefix: str,
    steps: int,
    init_checkpoint_uri: str,
    init_checkpoint_step: int,
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
        "/tmp/chess_dfm_jax/gcs_cache",
        "--models-dir",
        "/tmp/chess_dfm_jax/repo/models",
        "--save-dir",
        "/tmp/chess_dfm_jax/artifacts",
        "--checkpoint-uri",
        f"{bucket}/runs/dfm/{run_id}/checkpoints",
        "--run-id",
        run_id,
        "--init-checkpoint-uri",
        init_checkpoint_uri,
        "--init-checkpoint-step",
        str(init_checkpoint_step),
        "--init-restore-optimizer",
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


def phase3_spec(base: dict, bucket: str) -> TPUJobSpec:
    command = train_command(
        run_id=PHASE3_RUN,
        bucket=bucket,
        train_prefix=f"{SET1}/train,{SET2}/train",
        val_prefix=f"{SET1}/val,{SET2}/val",
        steps=98490,
        init_checkpoint_uri=f"{bucket}/runs/dfm/{PHASE2_RUN}/checkpoints",
        init_checkpoint_step=49264,
        wandb_project=base.get("wandb_project", "chess_dfm_jax-jepa-sweep"),
    )
    return TPUJobSpec.from_dict(
        {
            **base,
            "run_id": PHASE3_RUN,
            "run_name": PHASE3_RUN,
            "run_family": "dfm",
            "zone_order": [ZONE],
            "accelerator_type": "v5litepod-1",
            "wandb_group": "dfm-lc0-phaseA",
            "workdir": "/tmp/chess_dfm_jax",
            "entry_command": command,
            "spot": True,
            "enable_external_ips": True,
        }
    )


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


def tpu_exists(*, project_id: str, name: str) -> bool:
    result = run_cli(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            f"--project={project_id}",
            f"--zone={ZONE}",
            "--format=json",
        ],
        check=False,
    )
    return result.returncode == 0


def launch_phase3(base: dict, bucket: str, work_dir: Path) -> None:
    spec = phase3_spec(base, bucket)
    if tpu_exists(project_id=spec.project_id, name=PHASE3_RUN):
        print(f"{PHASE3_RUN} already exists; not launching a duplicate.", flush=True)
        return

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    work_dir.mkdir(parents=True, exist_ok=True)
    source_archive = create_source_snapshot(PROJECT_ROOT, work_dir / "source.tar.gz")
    source_uri = spec.source_uri(ZONE, stamp)
    upload_file(source_archive, source_uri)
    upload_json(spec.to_dict(), f"{spec.run_root_uri(ZONE)}/job_spec.json")
    startup_path = work_dir / "startup_phasea3_set1set2.sh"
    startup_path.write_text(render_startup_script(spec, ZONE, source_uri), encoding="utf-8")
    try:
        run_cli(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "create",
                PHASE3_RUN,
                f"--project={spec.project_id}",
                f"--zone={ZONE}",
                "--accelerator-type=v5litepod-1",
                "--version=tpu-ubuntu2204-base",
                "--spot",
                f"--service-account={spec.service_account}",
                "--network=default",
                f"--metadata-from-file=startup-script={startup_path}",
                f"--labels=run_id={PHASE3_RUN},controller=chess_dfm_jax",
            ]
        )
    finally:
        startup_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-s", type=int, default=120)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/chess_dfm_phasea_chain"))
    args = parser.parse_args()

    load_env_file()
    base = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    bucket = base["bucket_by_region"][REGION].rstrip("/")
    phase2_status_uri = f"{bucket}/runs/dfm/{PHASE2_RUN}/status.json"

    while True:
        status = read_json(phase2_status_uri) or {}
        state = status.get("state")
        print(json.dumps({"event": "poll", "run_id": PHASE2_RUN, "state": state, "status": status}), flush=True)
        if state == "completed":
            delete_tpu(project_id=base["project_id"], name=PHASE2_RUN)
            launch_phase3(base, bucket, args.work_dir)
            return 0
        if state == "failed":
            delete_tpu(project_id=base["project_id"], name=PHASE2_RUN)
            print("Phase A2 failed; not launching Phase A3.", flush=True)
            return 1
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
