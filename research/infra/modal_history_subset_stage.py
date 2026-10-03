"""HH-002 CPU-only contracted input staging; no GPU allocation."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess

import modal
from research.infra import modal_train_app as base
from research.infra.modal_history_stage import extra_volumes, image as base_image
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget

app = modal.App("chess-dfm-hh002-stage")
AUDIT = "research/analysis/hh001_history_20261003/contracted_selection_audit.json"
PATHS = ("research/hero_history_data.py", "research/hero_history_subset_data.py",
         "research/infra/modal_history_subset_stage.py", "chess_dfm_jax/encoding.py", AUDIT)
image = base_image.add_local_file(AUDIT, "/root/"+AUDIT, copy=True)


def identity(root):
    values = {p: base._sha256_file(Path(root)/p) for p in PATHS}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def spec():
    return ModalJobSpec(name="HH-002-input-stage", timeout_seconds=7200, startup_timeout_seconds=300,
                        retries=0, cpu_cores=Decimal(8), memory_gib=Decimal(16))


@app.function(image=image, cpu=8, memory=16384, timeout=7200, startup_timeout=300,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={base.BASE_INPUT_MOUNT:base.base_input_volume.with_mount_options(read_only=True),
                       base.TRAINING_INPUT_MOUNT:base.training_input_volume.with_mount_options(read_only=True),
                       base.TRAINING_RESULT_MOUNT:base.training_result_volume,
                       **{p:v.with_mount_options(read_only=True) for p,v in extra_volumes.items()}})
def stage(expected_source, git_commit):
    from research.hero_history_subset_data import build_subset
    from research.hero_history_data import sha256, write_json_once
    if identity("/root") != expected_source:
        raise ValueError("Staging source mismatch")
    result = Path(base.TRAINING_RESULT_MOUNT)
    output = result/"datasets"/f"hh002-history-{expected_source[:20]}"
    inventory = {}
    for mount in (base.BASE_INPUT_MOUNT, base.TRAINING_INPUT_MOUNT, base.TRAINING_RESULT_MOUNT, *extra_volumes):
        inventory[mount] = sum((Path(parent)/name).stat().st_size
                              for parent, _, files in os.walk(mount) for name in files)
    if (sum(inventory.values()) + 20*1024**3) / 1024**3 >= 768:
        raise RuntimeError("Storage review required")
    raw = base._verify_raw_bt4_volume()
    baseline = result/"runs"/base.HERO1_EXACT_PHASE1_RESULT_LABEL
    checkpoints = []
    for path in sorted((baseline/"checkpoints").glob("*/manifest.json")):
        record = json.loads(path.read_text())
        update = record.get("optimizer_update")
        if update not in {100,200,300,400,500,554}:
            continue
        state = path.parent/record["state"]["path"]
        if state.stat().st_size != record["state"]["size_bytes"] or sha256(state) != record["state"]["sha256"]:
            raise ValueError("Historical baseline checkpoint drift")
        checkpoints.append({"update":update, "path":str(path.parent), "state_sha256":sha256(state)})
    if len({r["update"] for r in checkpoints}) != len(checkpoints):
        raise ValueError("Duplicate baseline checkpoint updates")
    try:
        manifest = build_subset(base._training_dataset_root()/"trajectory_v3", output,
            result/"datasets/hero1-history-8db9ffbe43cdc9fb3712", Path("/root")/AUDIT,
            Path("/root/research/eval/hero_epoch_v1/position_indices.npz"))
        receipt = {"passed":True, "gpu_active":False, "dataset_root":str(output),
                   "source_tree_sha256":expected_source, "git_commit":git_commit,
                   "manifest_sha256":sha256(output/"manifest.json"), "raw_bt4_sha256":sha256(raw),
                   "train_examples":manifest["train_examples"], "validation_examples":manifest["validation_examples"],
                   "inventory_bytes":inventory, "baseline_checkpoints":checkpoints,
                   "baseline_training_metrics_sha256":sha256(baseline/"metrics.jsonl"),
                   "baseline_run_config_sha256":sha256(baseline/"run_config.json")}
        write_json_once(output/"preflight.json",receipt)
        return receipt
    finally:
        base.training_result_volume.commit()


@app.local_entrypoint()
def main(recorded_spend: str = "3.71525855"):
    root = Path(__file__).resolve().parents[2]
    spend = max(base._modal_launch_spend(), Decimal(recorded_spend))
    decision = enforce_modal_budget(estimate_modal_job(spec()), recorded_month_spend=spend, per_job_cap=Decimal("1.10"))
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    print(json.dumps({"stage_budget":decision, "source":identity(root)}),flush=True)
    print(json.dumps({"subset_preflight":stage.remote(identity(root),commit)}),flush=True)
