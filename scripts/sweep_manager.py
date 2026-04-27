#!/usr/bin/env python3
"""Quota-Aware Sweep Manager for Massive Parallel TPU Profiling/Training."""

import argparse
import json
import os
import subprocess
import threading
import sys
import time
from queue import Queue
from dataclasses import dataclass
from typing import Any

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

def worker(slot: dict, task_queue: Queue, project_id: str, workdir: str):
    zone = slot["zone"]
    accel = slot["accelerator_type"]
    
    while True:
        exp = task_queue.get()
        print(f"[{zone} | {accel}] Starting experiment {exp.experiment_id}")
        
        # Determine subnetwork (defaulting to the region's default subnetwork)
        region = zone.rsplit("-", 1)[0]
        subnetwork = f"projects/{project_id}/regions/{region}/subnetworks/default"
        
        # Create a specific job spec for this experiment
        spec_dict = {
            "project_id": project_id,
            "run_id": exp.experiment_id,
            "run_name": exp.experiment_id,
            "zone_order": [zone],
            "bucket_by_region": {
                region: f"gs://gunay-chess-experiments-{region}"
            },
            "wandb_project": "chess_dfm_jax-jepa-sweep",
            "wandb_group": "parallel-sweep",
            "accelerator_type": accel,
            "runtime_version": "tpu-ubuntu2204-base",
            "service_account": "534484336006-compute@developer.gserviceaccount.com",
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
                region: f"gs://gunay-chess-experiments-{region}/models"
            },
            "chunk_data_uri_by_region": {
                region: f"gs://gunay-chess-experiments-{region}/data/chunks"
            },
            "entry_command": exp.entry_command,
            "env": exp.env,
            "spot": True
        }

        # Replace placeholder in command
        bucket_name = spec_dict["bucket_by_region"][region].replace("gs://", "")
        cmd_str = exp.entry_command.replace("{REGION_BUCKET}", bucket_name)
        spec_dict["entry_command"] = cmd_str
        
        spec_path = os.path.join(workdir, f"spec_{exp.experiment_id}.json")
        with open(spec_path, "w") as f:
            json.dump(spec_dict, f, indent=2)
            
        # Run the spot controller synchronously
        log_path = os.path.join(workdir, f"log_{exp.experiment_id}.txt")
        cmd = [
            sys.executable, "-u", "scripts/run_tpu_spot_jepa.py",
            "--job-spec", spec_path,
            "--override-source-uri", f"gs://gunay-chess-experiments-{region}/fixed_snapshots/verified_source_v26_dfm.tar.gz"
        ]
        
        print(f"[{zone} | {accel}] Executing: {' '.join(cmd)}")
        try:
            # Synchronously run the spot controller for this regional slot
            subprocess.run(cmd, check=False)
            
            # Post-execution verification: Check the persistent status file on GCS
            # region is already defined above
            bucket = f"gs://gunay-chess-experiments-{region}"
            status_uri = f"{bucket}/runs/jepa/{exp.experiment_id}/status.json"
            
            check_status_cmd = ["/snap/google-cloud-cli/current/bin/gcloud", "storage", "cat", status_uri]
            status_result = subprocess.run(check_status_cmd, capture_output=True, text=True)
            
            is_completed = False
            if status_result.returncode == 0:
                try:
                    status_data = json.loads(status_result.stdout)
                    if status_data.get("state") == "completed":
                        is_completed = True
                except:
                    pass
            
            if is_completed:
                 print(f"[{zone} | {accel}] Successfully finished experiment {exp.experiment_id}")
                 task_queue.task_done()
            else:
                 print(f"[{zone} | {accel}] Experiment {exp.experiment_id} interrupted or failed. Re-queuing...")
                 task_queue.put(exp)
                 time.sleep(60) # Cooldown for IP release
        except Exception as e:
            print(f"[{zone} | {accel}] Critical error running controller for {exp.experiment_id}: {e}")
            task_queue.put(exp)
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, required=True, help="Path to experiments.jsonl")
    parser.add_argument("--project-id", type=str, default="project-b9551f07-5f68-491a-8a0")
    parser.add_argument("--workdir", type=str, default="sweep_runs")
    args = parser.parse_args()
    
    os.makedirs(args.workdir, exist_ok=True)
    experiments = load_experiments(args.experiments)
    
    print(f"Loaded {len(experiments)} experiments. Distributing across {len(SLOTS)} quota slots.")
    
    task_queue = Queue()
    for exp in experiments:
        task_queue.put(exp)
        
    threads = []
    for slot in SLOTS:
        t = threading.Thread(target=worker, args=(slot, task_queue, args.project_id, args.workdir))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print("All experiments completed.")

if __name__ == "__main__":
    main()
