#!/usr/bin/env python3
"""Quota-Aware Sweep Manager for Massive Parallel TPU Profiling/Training."""

import argparse
import json
import os
import subprocess
import threading
import sys
import time
from queue import Empty, Queue
from dataclasses import dataclass

from chess_dfm_jax.tracking import load_env_file

# Known trial quotas (in total chips)
ZONES_QUOTA = {
    "us-central2-b": {"v4": 32},
    "us-central1-a": {"v5litepod": 64},
    "europe-west4-b": {"v5litepod": 64},
    "europe-west4-a": {"v6e": 64},
}

# Translating to slots of 16-chip pods
# (We use 16-chip pods to use the 'Training' quota as discovered earlier)
SLOTS = []
for zone, quota in ZONES_QUOTA.items():
    if "v5litepod" in quota:
        num_pods = quota["v5litepod"] // 16
        for i in range(num_pods):
            SLOTS.append({"zone": zone, "accelerator_type": "v5litepod-16"})
    elif "v6e" in quota:
        num_pods = quota["v6e"] // 16
        for i in range(num_pods):
            SLOTS.append({"zone": zone, "accelerator_type": "v6e-16"})
    elif "v4" in quota:
        num_pods = quota["v4"] // 16
        for i in range(num_pods):
            SLOTS.append({"zone": zone, "accelerator_type": "v4-16"})


@dataclass
class Experiment:
    experiment_id: str
    entry_command: str
    env: dict[str, str]


BLOCKED_SOURCE_MARKERS = (
    "verified_source_v27_dfm.tar.gz",
    "verified_source_example.tar.gz",
    "your-bucket-prefix",
)

BLOCKED_ENTRY_MARKERS = (
    "/tmp/lc0jaxhuman",
)

PHASE_ENTRY_MARKERS = {
    "dfm": "scripts/train_dfm.py",
    "jepa": "scripts/train_jepa.py",
    "profile": "scripts/profile_jepa_tpu.py",
}

def load_experiments(jsonl_path: str) -> list[Experiment]:
    experiments = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            experiments.append(Experiment(
                experiment_id=data["experiment_id"],
                entry_command=data["entry_command"],
                env=data.get("env", {})
            ))
    return experiments


def validate_experiments(experiments: list[Experiment], *, phase: str) -> None:
    for exp in experiments:
        for marker in BLOCKED_ENTRY_MARKERS:
            if marker in exp.entry_command:
                raise ValueError(
                    f"{exp.experiment_id} uses blocked legacy path {marker!r}; "
                    "update the entry_command before launching TPU resources."
                )
        expected_marker = PHASE_ENTRY_MARKERS.get(phase)
        if expected_marker and expected_marker not in exp.entry_command:
            raise ValueError(
                f"{exp.experiment_id} is in phase {phase!r} but does not run {expected_marker!r}."
            )


def validate_launch_config(source_snapshot_uri: str, bucket_prefix: str, service_account: str) -> None:
    if not bucket_prefix or bucket_prefix == "your-bucket-prefix":
        raise ValueError("A real bucket prefix or --bucket-uri is required before launch.")
    if not service_account or "your-gcp-project" in service_account:
        raise ValueError("A real service account is required before launch.")
    for marker in BLOCKED_SOURCE_MARKERS:
        if marker in source_snapshot_uri:
            raise ValueError(
                f"Refusing to launch with blocked source snapshot {source_snapshot_uri!r}; "
                "create a fresh snapshot from the current tree first."
            )


def bucket_for_region(region: str, *, bucket_prefix: str, bucket_uri: str) -> str:
    if bucket_uri:
        if not bucket_uri.startswith("gs://"):
            raise ValueError(f"--bucket-uri must be a gs:// URI, got {bucket_uri!r}.")
        return bucket_uri.rstrip("/")
    return f"gs://{bucket_prefix}-{region}"


def build_slots(*, zones: list[str], accelerator_type: str, max_parallel: int) -> list[dict[str, str]]:
    if zones:
        slots = [
            {"zone": zone, "accelerator_type": accelerator_type}
            for zone in zones
            for _ in range(max_parallel)
        ]
    else:
        slots = list(SLOTS)
    return slots[:max_parallel]


def worker(
    slot: dict,
    task_queue: Queue,
    project_id: str,
    workdir: str,
    source_snapshot_uri: str,
    bucket_prefix: str,
    bucket_uri: str,
    models_uri: str,
    chunk_data_uri: str,
    service_account: str,
    requeue_on_failure: bool,
    phase: str,
    spot: bool,
):
    zone = slot["zone"]
    accel = slot["accelerator_type"]
    
    while True:
        try:
            exp = task_queue.get_nowait()
        except Empty:
            return
        print(f"[{zone} | {accel}] Starting experiment {exp.experiment_id}")
        
        # Determine subnetwork (defaulting to the region's default subnetwork)
        region = zone.rsplit("-", 1)[0]
        subnetwork = f"projects/{project_id}/regions/{region}/subnetworks/default"
        
        bucket = bucket_for_region(region, bucket_prefix=bucket_prefix, bucket_uri=bucket_uri)
        resolved_models_uri = models_uri.rstrip("/") if models_uri else f"{bucket}/models"
        resolved_chunk_data_uri = chunk_data_uri.rstrip("/") if chunk_data_uri else ""

        # Create a specific job spec for this experiment.
        spec_dict = {
            "project_id": project_id,
            "run_id": exp.experiment_id,
            "run_name": exp.experiment_id,
            "run_family": phase,
            "zone_order": [zone],
            "bucket_by_region": {
                region: bucket
            },
            "wandb_project": "chess_dfm_jax-jepa-sweep",
            "wandb_group": f"{phase}-sweep",
            "accelerator_type": accel,
            "runtime_version": "tpu-ubuntu2204-base",
            "service_account": service_account,
            "network": "default",
            "subnetwork_by_zone": {
                zone: subnetwork
            },
            "enable_external_ips": True,
            "autocheckpoint_enabled": True,
            "allocation_timeout_s": 900,
            "poll_interval_s": 30,
            "workdir": "/tmp/chess_dfm_jax",
            "models_uri_by_region": {
                region: resolved_models_uri
            },
            "chunk_data_uri_by_region": {
                region: resolved_chunk_data_uri
            },
            "entry_command": exp.entry_command,
            "env": {**exp.env, "CHESS_DFM_SWEEP_PHASE": phase},
            "spot": spot
        }

        # Replace placeholder in command
        bucket_name = bucket.replace("gs://", "")
        cmd_str = exp.entry_command.replace("{REGION_BUCKET}", bucket_name)
        spec_dict["entry_command"] = cmd_str
        
        spec_path = os.path.join(workdir, f"spec_{exp.experiment_id}.json")
        with open(spec_path, "w") as f:
            json.dump(spec_dict, f, indent=2)
            
        # Run the spot controller synchronously
        cmd = [
            sys.executable, "-u", "scripts/run_tpu_spot_jepa.py",
            "--job-spec", spec_path,
        ]
        if source_snapshot_uri:
            cmd.extend(["--override-source-uri", source_snapshot_uri])
        
        print(f"[{zone} | {accel}] Executing: {' '.join(cmd)}")
        try:
            # Synchronously run the spot controller for this regional slot
            subprocess.run(cmd, check=False)
            
            # Post-execution verification: Check the persistent status file on GCS
            # region is already defined above
            status_uri = f"{bucket}/runs/{phase}/{exp.experiment_id}/status.json"
            
            check_status_cmd = ["gcloud", "storage", "cat", status_uri]
            status_result = subprocess.run(check_status_cmd, capture_output=True, text=True)
            
            is_completed = False
            if status_result.returncode == 0:
                try:
                    status_data = json.loads(status_result.stdout)
                    if status_data.get("state") == "completed":
                        is_completed = True
                except Exception:
                    pass
            
            if is_completed:
                 print(f"[{zone} | {accel}] Successfully finished experiment {exp.experiment_id}")
                 task_queue.task_done()
            else:
                 print(f"[{zone} | {accel}] Experiment {exp.experiment_id} interrupted or failed.")
                 task_queue.task_done()
                 if requeue_on_failure:
                     print(f"[{zone} | {accel}] Re-queuing {exp.experiment_id} because --requeue-on-failure is set.")
                     task_queue.put(exp)
                     time.sleep(60) # Cooldown for IP release
        except Exception as e:
            print(f"[{zone} | {accel}] Critical error running controller for {exp.experiment_id}: {e}")
            task_queue.task_done()
            if requeue_on_failure:
                task_queue.put(exp)
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, required=True, help="Path to experiments.jsonl")
    parser.add_argument("--project-id", type=str, default="your-gcp-project")
    parser.add_argument("--workdir", type=str, default="sweep_runs")
    parser.add_argument("--source-snapshot-uri", type=str, default=os.environ.get("CHESS_DFM_SOURCE_SNAPSHOT_URI", ""))
    parser.add_argument("--bucket-prefix", type=str, default=os.environ.get("CHESS_DFM_BUCKET_PREFIX", ""))
    parser.add_argument("--bucket-uri", type=str, default=os.environ.get("CHESS_DFM_BUCKET_URI", ""))
    parser.add_argument("--models-uri", type=str, default=os.environ.get("CHESS_DFM_MODELS_URI", ""))
    parser.add_argument("--chunk-data-uri", type=str, default=os.environ.get("CHESS_DFM_CHUNK_DATA_URI", ""))
    parser.add_argument(
        "--skip-chunk-sync",
        action="store_true",
        help="Do not copy chunk_data_uri at TPU startup; use trainer-side GCS streaming/cache.",
    )
    parser.add_argument("--service-account", type=str, default=os.environ.get("CHESS_DFM_SERVICE_ACCOUNT", ""))
    parser.add_argument("--zone", action="append", default=[], help="TPU zone to use. Repeat for ordered fallback zones.")
    parser.add_argument("--accelerator-type", type=str, default="v5litepod-1")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--allow-launch", action="store_true", help="Required to create TPU queued resources.")
    parser.add_argument("--on-demand", action="store_true", help="Use on-demand TPU capacity instead of spot/preemptible capacity.")
    parser.add_argument("--requeue-on-failure", action="store_true", help="Retry failed experiments. Disabled by default to avoid runaway TPU provisioning.")
    parser.add_argument(
        "--phase",
        type=str,
        default=os.environ.get("CHESS_DFM_SWEEP_PHASE", "unclassified"),
        choices=["smoke", "dfm", "jepa", "profile", "combined", "unclassified"],
        help="Explicit experiment phase used for W&B grouping and entry-command validation.",
    )
    args = parser.parse_args()
    load_env_file()

    bucket_prefix = args.bucket_prefix
    service_account = args.service_account
    
    if args.allow_launch and args.phase == "unclassified":
        raise SystemExit("--phase is required for launches so phase-wise sweeps stay separated.")

    phase_workdir = os.path.join(args.workdir, args.phase)
    os.makedirs(phase_workdir, exist_ok=True)
    experiments = load_experiments(args.experiments)
    validate_experiments(experiments, phase=args.phase)

    if not args.allow_launch:
        raise SystemExit("Refusing to create TPU resources without --allow-launch.")
    validate_launch_config(
        args.source_snapshot_uri,
        bucket_prefix or args.bucket_uri,
        service_account,
    )
    slots = build_slots(
        zones=args.zone,
        accelerator_type=args.accelerator_type,
        max_parallel=args.max_parallel,
    )
    
    print(
        f"Loaded {len(experiments)} {args.phase} experiments. "
        f"Distributing across {len(slots)} slot(s)."
    )
    
    task_queue = Queue()
    for exp in experiments:
        task_queue.put(exp)
        
    threads = []
    for slot in slots:
        t = threading.Thread(
            target=worker,
            args=(
                slot,
                task_queue,
                args.project_id,
                phase_workdir,
                args.source_snapshot_uri,
                bucket_prefix,
                args.bucket_uri,
                args.models_uri,
                "" if args.skip_chunk_sync else args.chunk_data_uri,
                service_account,
                args.requeue_on_failure,
                args.phase,
                not args.on_demand,
            ),
        )
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print("All experiments completed.")

if __name__ == "__main__":
    main()
