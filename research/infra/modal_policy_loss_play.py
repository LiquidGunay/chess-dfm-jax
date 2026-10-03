"""EV-007: bounded CPU preflight followed by inference-aligned policy losses."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

import modal

from research.infra import modal_arena_app as arena, modal_train_app as train
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget

app = modal.App("chess-dfm-ev007-policy-loss")
image = arena.arena_image.add_local_file(".local/ev003_common_validation.npz", "/opt/sequential_validation.npz", copy=True)
SOURCES = ("research/evaluate_policy_loss_play.py", "research/evaluate_common_policy.py",
           "research/infra/modal_policy_loss_play.py", "research/hero_history_batches.py",
           "research/hero_history_data.py", "research/hero_history_subset_data.py")


def source_identity(root):
    root = Path(root)
    return hashlib.sha256((arena._arena_source_tree_sha256(root)
        + train._tree_sha256(root, relative_paths=SOURCES)).encode()).hexdigest()


def job_spec(stage):
    return ModalJobSpec(name="EV-007-"+stage, timeout_seconds=600 if stage == "prepare" else 2400,
        startup_timeout_seconds=300 if stage == "prepare" else 600, retries=0,
        cpu_cores=Decimal(4), memory_gib=Decimal(8 if stage == "prepare" else 16),
        gpu=None if stage == "prepare" else "L4")


def bind_environment():
    os.environ["CHESS_DFM_WORKSPACE_ROOT"] = "/workspace"
    os.environ["CHESS_DFM_TRUSTED_WORKSPACE_ROOTS"] = os.pathsep.join(
        str(Path(p).resolve(strict=True)) for p in (arena.BASE_INPUT_MOUNT, arena.TRAINING_RESULT_MOUNT))


def output_root(protocol):
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return Path(arena.TRAINING_RESULT_MOUNT) / "validations" / ("ev007-"+digest[:20])


def verify_source(protocol):
    if source_identity("/root") != protocol["source_sha256"]:
        raise ValueError("Evaluator source drift")
    if arena._arena_source_tree_sha256(Path("/root")) != protocol["arena_source_sha256"]:
        raise ValueError("Arena/model source drift")


def verify_models(protocol):
    from research.input_contract import torch_hero_plane_history_mode
    raw = arena._verify_raw_bt4_volume()
    if arena._sha256_file(raw) != protocol["raw_bt4_sha256"]:
        raise ValueError("Raw checkpoint checksum drift")
    result = {}
    for name, identity in protocol["models"].items():
        if name == "raw_bt4":
            result[name] = {**identity, "root": None}
            continue
        checked = arena._validated_training_result(identity["label"])
        root = checked["root"]
        if arena._sha256_file(root / "checkpoint/model.safetensors") != identity["state_sha256"]:
            raise ValueError("Checkpoint checksum drift")
        if arena._sha256_file(root / "run_config.json") != identity["run_config_sha256"]:
            raise ValueError("Checkpoint config drift")
        if torch_hero_plane_history_mode(json.loads((root / "run_config.json").read_text())) != identity["plane_history_mode"]:
            raise ValueError("Checkpoint input-history contract drift")
        result[name] = {**identity, "root": str(root)}
    return raw, result


@app.function(image=image, cpu=4, memory=8192, timeout=600, startup_timeout=300,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={arena.BASE_INPUT_MOUNT: arena.base_input_volume.with_mount_options(read_only=True),
                       arena.TRAINING_RESULT_MOUNT: arena.training_result_volume})
def prepare(protocol, declared_cost):
    bind_environment()
    import numpy as np
    from research.evaluate_policy_loss_play import prepare_hero, validate_arrays
    from research.hero_history_data import write_json_once
    verify_source(protocol)
    _, models = verify_models(protocol)
    out = output_root(protocol); out.mkdir(parents=True, exist_ok=False)
    hero = prepare_hero(Path(arena.TRAINING_RESULT_MOUNT) / protocol["hero_data_relative"],
                        protocol["hero_data_manifest_sha256"], out / "hero_validation.npz")
    sequential = Path("/opt/sequential_validation.npz")
    if arena._sha256_file(sequential) != protocol["sequential_inputs_sha256"]:
        raise ValueError("Sequential input checksum drift")
    with np.load(sequential, allow_pickle=False) as archive:
        seq_checks = validate_arrays({k: archive[k] for k in archive.files}, 8192)
    result = {"passed": True, "gpu_active": False, "source_sha256": protocol["source_sha256"],
              "models": models, "raw_bt4_sha256": protocol["raw_bt4_sha256"],
              "pools": {"hero_validation": hero, "sequential_validation": {**seq_checks,
                  "file_sha256": protocol["sequential_inputs_sha256"]}},
              "output_relative": str(out.relative_to(arena.TRAINING_RESULT_MOUNT)),
              "declared_cost": declared_cost, "protocol": protocol}
    write_json_once(out / "preflight.json", result)
    arena.training_result_volume.commit()
    return result


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=2400, startup_timeout=600,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={arena.BASE_INPUT_MOUNT: arena.base_input_volume.with_mount_options(read_only=True),
                       arena.TRAINING_RESULT_MOUNT: arena.training_result_volume})
def evaluate(protocol, preflight_sha256, declared_cost):
    bind_environment()
    import torch
    from research.evaluate_policy_loss_play import evaluate as evaluate_policies
    from research.hero_history_data import write_json_once
    verify_source(protocol)
    out = output_root(protocol)
    if arena._sha256_file(out / "preflight.json") != preflight_sha256:
        raise ValueError("Preflight checksum drift")
    preflight = json.loads((out / "preflight.json").read_text())
    if not preflight["passed"] or preflight["gpu_active"] or preflight["protocol"] != protocol:
        raise ValueError("Missing valid CPU preflight")
    raw, models = verify_models(protocol)
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
        raise ValueError("Expected L4")
    pools = {"hero_validation": {"path": out / "hero_validation.npz", "positions": 8128},
             "sequential_validation": {"path": Path("/opt/sequential_validation.npz"), "positions": 8192}}
    for name, pool in pools.items():
        if arena._sha256_file(pool["path"]) != preflight["pools"][name]["file_sha256"]:
            raise ValueError("Prepared validation bytes changed")
    result_dir = out / "results"; result_dir.mkdir(exist_ok=False)
    result = evaluate_policies(pools, models, raw, result_dir)
    result.update(protocol=protocol, preflight_sha256=preflight_sha256, declared_cost=declared_cost,
                  output_relative=str(result_dir.relative_to(arena.TRAINING_RESULT_MOUNT)))
    write_json_once(result_dir / "result.json", result)
    arena.training_result_volume.commit()
    return {"result_relative": str((result_dir / "result.json").relative_to(arena.TRAINING_RESULT_MOUNT)),
            "result_sha256": arena._sha256_file(result_dir / "result.json"),
            "position_metrics_sha256": result["position_metrics_sha256"]}


@app.local_entrypoint()
def main(protocol_path: str, stage: str, recorded_spend: str, preflight_path: str = ""):
    from research.infra.modal_eval_matrix import atomic_json
    if stage not in ("prepare", "evaluate"):
        raise ValueError("Unknown stage")
    path = Path(protocol_path); protocol = json.loads(path.read_text())
    if source_identity(Path.cwd()) != protocol["source_sha256"]:
        raise ValueError("Commit and freeze evaluator source before launch")
    preflight_sha = ""
    if stage == "evaluate":
        preflight = json.loads(Path(preflight_path).read_text())
        remote = b"".join(arena.training_result_volume.read_file(preflight["output_relative"]+"/preflight.json"))
        if json.loads(remote) != preflight or not preflight["passed"] or preflight["gpu_active"]:
            raise ValueError("Complete matching CPU preflight required before GPU allocation")
        preflight_sha = hashlib.sha256(remote).hexdigest()
    decision = enforce_modal_budget(estimate_modal_job(job_spec(stage)),
        recorded_month_spend=Decimal(recorded_spend), per_job_cap=Decimal("0.07" if stage == "prepare" else "1.00"))
    print(json.dumps({"cost_guard": decision}), flush=True)
    receipt_path = path.parent / (stage+"_run.json")
    if receipt_path.exists():
        raise ValueError("An attempt already exists; no automatic retry")
    receipt = {"stage": stage, "status": "launching", "cost_guard": decision,
               "git_commit": arena._local_git_commit(Path.cwd()),
               "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_json(receipt_path, receipt)
    if stage == "prepare":
        result = prepare.remote(protocol, decision["declared_attempt_upper_bound"])
        atomic_json(path.parent / "preflight.json", result)
        receipt.update(status="complete", output_relative=result["output_relative"])
    else:
        call = evaluate.spawn(protocol, preflight_sha, decision["declared_attempt_upper_bound"])
        receipt.update(status="submitted", function_call_id=call.object_id)
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)
