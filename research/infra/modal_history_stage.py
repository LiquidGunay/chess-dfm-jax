"""HH-001 CPU-only input reconstruction. This module cannot launch GPUs."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess

import modal

from research.infra import modal_train_app as base
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget

app = modal.App("chess-dfm-hero-history-stage")
PATHS = ("research/hero_history_data.py", "research/infra/modal_history_stage.py",
         "chess_dfm_jax/encoding.py", "research/eval/hero_epoch_v1/position_indices.npz")
RESULT_ROOT = Path(base.TRAINING_RESULT_MOUNT)
extra_volumes = {
    "/inventory/compile-cache": modal.Volume.from_name("chess-dfm-training-compile-cache", create_if_missing=False),
    "/inventory/interpretability-results": modal.Volume.from_name("chess-dfm-interpretability-results", create_if_missing=False),
}
image = base.gpu_image.add_local_file(
    "research/eval/hero_epoch_v1/position_indices.npz", "/root/research/eval/hero_epoch_v1/position_indices.npz", copy=True)


def source_identity(root):
    values = {name: base._sha256_file(Path(root)/name) for name in PATHS}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def cost_spec():
    return ModalJobSpec(name="HH-001-history-rebuild-cpu", timeout_seconds=7200,
                        startup_timeout_seconds=300, retries=0,
                        cpu_cores=Decimal(8), memory_gib=Decimal(16))


@app.function(image=image, cpu=8, memory=16384, timeout=7200, startup_timeout=300,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={base.BASE_INPUT_MOUNT: base.base_input_volume.with_mount_options(read_only=True),
                       base.TRAINING_INPUT_MOUNT: base.training_input_volume.with_mount_options(read_only=True),
                       base.TRAINING_RESULT_MOUNT: base.training_result_volume,
                       **{path:volume.with_mount_options(read_only=True) for path,volume in extra_volumes.items()}})
def rebuild(expected_source, git_commit):
    from research.hero_history_data import build_dataset, sha256, write_json_once
    if source_identity("/root") != expected_source:
        raise ValueError("CPU reconstruction source mismatch")
    output = RESULT_ROOT/"datasets"/f"hero1-history-{expected_source[:20]}"
    # Inventory every chess volume before creating a new training dataset.
    import os
    inventory = {}
    for mount in [base.BASE_INPUT_MOUNT, base.TRAINING_INPUT_MOUNT,
                  base.TRAINING_RESULT_MOUNT, *extra_volumes]:
        total = 0
        for parent, _, files in os.walk(mount):
            total += sum((Path(parent)/name).stat().st_size for name in files)
        inventory[mount] = total
    projected_gib = (sum(inventory.values()) + 20 * 1024**3) / 1024**3
    if projected_gib >= 768:
        raise RuntimeError("Storage review threshold reached; no new dataset")
    raw = base._verify_raw_bt4_volume()
    indices = Path("/root/research/eval/hero_epoch_v1/position_indices.npz")
    if sha256(indices) != "c20102b49358dda5c9d6828e703cf133b196b3bf3b018cb143290aa8c1dc3e96":
        raise ValueError("Frozen validation indices changed")
    print(json.dumps({"inventory_bytes": inventory, "projected_gib_with_20_gib_reserve": projected_gib}), flush=True)
    try:
        manifest = build_dataset(base._training_dataset_root()/"trajectory_v3", output, indices)
    finally:
        base.training_result_volume.commit()
    receipt = {"passed": True, "gpu_active": False, "dataset_root": str(output),
               "source_tree_sha256": expected_source, "git_commit": git_commit,
               "manifest_sha256": sha256(output/"manifest.json"),
               "raw_bt4_sha256": sha256(raw), "inventory_bytes": inventory,
               "train_examples": manifest["train_examples"],
               "validation_examples": manifest["validation_examples"],
               "output_bytes": sum(r["size_bytes"] for s in ("train","validation") for r in manifest[s]),
               "recovered_moves": sum(len(r["recovered_moves"]) for s in ("train","validation") for r in manifest[s]),
               "reference_parity_positions": sum(r["reference_parity_positions"] for s in ("train","validation") for r in manifest[s])}
    write_json_once(output/"preflight.json", receipt)
    base.training_result_volume.commit()
    return receipt


@app.local_entrypoint()
def main(recorded_spend: str = "0"):
    root = Path(__file__).resolve().parents[2]
    # Calling this entrypoint is a single CPU attempt, never automatic retries.
    spend = max(base._modal_launch_spend(), Decimal(recorded_spend))
    decision = enforce_modal_budget(estimate_modal_job(cost_spec()), recorded_month_spend=spend,
                                    per_job_cap=Decimal("1.10"))
    commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=root, text=True).strip()
    print(json.dumps({"cost_guard": decision, "source_tree_sha256": source_identity(root)}), flush=True)
    result = rebuild.remote(source_identity(root), commit)
    print(json.dumps({"history_preflight": result}), flush=True)
