"""Bounded companion to the history-corrected arena: matched policy loss."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

import modal

from research.infra.modal_arena_app import (
    arena_image, base_input_volume, training_result_volume,
    BASE_INPUT_MOUNT, TRAINING_RESULT_MOUNT, _verify_raw_bt4_volume,
    _validated_training_result, _arena_source_tree_sha256, _arena_job_spec,
)

app = modal.App("chess-dfm-common-policy-evaluation")
image = arena_image.add_local_file(".local/ev003_common_validation.npz", "/opt/common_validation.npz", copy=True)
SOURCES = ("research/evaluate_common_policy.py", "research/infra/modal_common_policy_app.py")


def source_digest(root: Path) -> str:
    return hashlib.sha256(b"".join((root / path).read_bytes() for path in SOURCES)).hexdigest()


@app.function(
    image=image, gpu="L4", cpu=4, memory=16384, timeout=2400,
    startup_timeout=600, retries=0, max_containers=1, single_use_containers=True,
    volumes={BASE_INPUT_MOUNT: base_input_volume.with_mount_options(read_only=True),
             TRAINING_RESULT_MOUNT: training_result_volume.with_mount_options(read_only=True)},
)
def evaluate_common(protocol: dict) -> dict:
    os.environ["CHESS_DFM_WORKSPACE_ROOT"] = TRAINING_RESULT_MOUNT
    os.environ["CHESS_DFM_TRUSTED_WORKSPACE_ROOTS"] = os.pathsep.join((BASE_INPUT_MOUNT, TRAINING_RESULT_MOUNT, "/root"))
    from research.evaluate_common_policy import evaluate, sha256

    if source_digest(Path("/root")) != protocol["evaluation_source_sha256"]:
        raise ValueError("Common validation code drift")
    if _arena_source_tree_sha256(Path("/root")) != protocol["arena_source_tree_sha256"]:
        raise ValueError("Model/encoding implementation drift")
    inputs = Path("/opt/common_validation.npz")
    if sha256(inputs) != protocol["inputs_sha256"]:
        raise ValueError("Common validation inputs drift")
    raw = _verify_raw_bt4_volume()
    if sha256(raw) != protocol["raw_bt4_sha256"]:
        raise ValueError("Raw model drift")
    checkpoints = {}
    for name, identity in protocol["models"].items():
        checked = _validated_training_result(identity["label"])
        if sha256(checked["root"] / "checkpoint" / "model.safetensors") != identity["state_sha256"]:
            raise ValueError("Checkpoint bytes drift")
        checkpoints[name] = checked["root"]
    return {**evaluate(inputs, checkpoints, raw), "protocol": protocol}


@app.local_entrypoint()
def main(protocol_path: str, recorded_spend: str):
    from research.interpretability.remote_cost import estimate_modal_job, enforce_modal_budget

    protocol = json.loads(Path(protocol_path).read_text())
    if source_digest(Path.cwd()) != protocol["evaluation_source_sha256"]:
        raise ValueError("Local evaluator changed after protocol freeze")
    spec, cap = _arena_job_spec(128, 1, 1)
    decision = enforce_modal_budget(estimate_modal_job(spec), recorded_month_spend=Decimal(recorded_spend), per_job_cap=cap)
    print(json.dumps({"cost_guard": decision}), flush=True)
    call = evaluate_common.spawn(protocol)
    print(json.dumps({"function_call_id": call.object_id}), flush=True)
