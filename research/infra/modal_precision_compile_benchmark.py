"""GPU-002 budget-gated CPU preflight and single-GPU disposable sweep."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import modal
from research.infra import modal_train_app as base
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget
from research.precision_compile_benchmark import ARMS, write_json

app = modal.App("chess-dfm-gpu002-precision")
SOURCES = ("research/precision_compile_benchmark.py", "research/infra/modal_precision_compile_benchmark.py",
           "research/infra/modal_train_app.py", "research/interpretability/remote_cost.py")
VOLUMES = {base.BASE_INPUT_MOUNT: base.base_input_volume.with_mount_options(read_only=True),
           base.TRAINING_RESULT_MOUNT: base.training_result_volume}


def source_identity(root):
    return base._tree_sha256(Path(root), relative_paths=SOURCES)


def verify_source(protocol, root):
    if source_identity(root) != protocol["source_sha256"]:
        raise ValueError("GPU-002 source drift")
    for name, digest in protocol["frozen_training_sources"].items():
        if base._sha256_file(Path(root)/name) != digest:
            raise ValueError("Frozen training dependency changed: "+name)
    if protocol["arms"] != list(ARMS):
        raise ValueError("Arm specification drift")


def spec(gpu=False):
    return ModalJobSpec(name="GPU-002-"+("sweep" if gpu else "preflight"),
        timeout_seconds=4200 if gpu else 600, startup_timeout_seconds=600 if gpu else 300,
        retries=0, cpu_cores=Decimal(4), memory_gib=Decimal(32 if gpu else 16),
        gpu="L40S" if gpu else None)


def output_root(protocol):
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return Path(base.TRAINING_RESULT_MOUNT)/"benchmarks"/("gpu002-"+digest[:20])


def environment(scratch):
    env = os.environ.copy()
    env.update(CHESS_DFM_WORKSPACE_ROOT="/workspace",
        CHESS_DFM_TRUSTED_WORKSPACE_ROOTS=os.pathsep.join((base.BASE_INPUT_MOUNT, base.TRAINING_RESULT_MOUNT, str(scratch))),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_MAX_THREADS="2",
        TORCHINDUCTOR_COMPILE_THREADS="1")
    return env


@app.function(image=base.gpu_image, cpu=4, memory=16384, timeout=600, startup_timeout=300,
              retries=0, max_containers=1, single_use_containers=True, volumes=VOLUMES)
def preflight(protocol, commit, cost):
    scratch = Path(tempfile.mkdtemp(prefix="gpu002-preflight-"))
    os.environ.update(environment(scratch))
    verify_source(protocol, "/root")
    from research.hero_history_data import verify_dataset
    from research.precision_compile_benchmark import prepared_batch
    data = Path(base.TRAINING_RESULT_MOUNT)/protocol["dataset_relative"]
    if base._sha256_file(data/"manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs")
    verify_dataset(data)
    checkpoint = Path(base.TRAINING_RESULT_MOUNT)/protocol["checkpoint_relative"]
    if base._sha256_file(checkpoint/"manifest.json") != protocol["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint manifest differs")
    manifest = json.loads((checkpoint/"manifest.json").read_text())
    state = checkpoint/manifest["state"]["path"]
    if (manifest["optimizer_update"] != 500 or state.stat().st_size != manifest["state"]["size_bytes"]
        or base._sha256_file(state) != protocol["checkpoint_state_sha256"]):
        raise ValueError("Checkpoint state differs")
    audits = []
    for cursor in protocol["batch_cursors"]:
        _, audit, digest = prepared_batch(protocol, data, cursor)
        audits.append({"cursor": cursor, "input_sha256": digest, **audit})
    out = output_root(protocol); out.mkdir(parents=True, exist_ok=False)
    result = {"passed": True, "gpu_active": False, "full_dataset_hashes_verified": True,
        "protocol": protocol, "git_commit": commit, "declared_cost": cost, "audits": audits,
        "output_relative": str(out.relative_to(base.TRAINING_RESULT_MOUNT))}
    write_json(out/"preflight.json", result)
    base.training_result_volume.commit()
    return result


def stop_process_group(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


@app.function(image=base.gpu_image, gpu="L40S", cpu=4, memory=32768, timeout=4200, startup_timeout=600,
              retries=0, max_containers=1, single_use_containers=True, volumes=VOLUMES)
def sweep(protocol, preflight_sha, commit, cost):
    verify_source(protocol, "/root")
    out = output_root(protocol)
    if base._sha256_file(out/"preflight.json") != preflight_sha:
        raise ValueError("CPU preflight drift")
    pre = json.loads((out/"preflight.json").read_text())
    if pre["protocol"] != protocol or not pre["passed"] or pre["gpu_active"]:
        raise ValueError("Completed CPU verification required")
    scratch = Path(tempfile.mkdtemp(prefix="gpu002-sweep-"))
    env = environment(scratch)
    os.environ.update(env)
    import torch
    if "L40S" not in torch.cuda.get_device_name(0): raise ValueError("Unexpected GPU")
    write_json(scratch/"protocol.json", protocol)
    write_json(out/"launch.json", {"protocol": protocol, "git_commit": commit,
        "declared_cost": cost, "preflight_sha256": preflight_sha,
        "device": torch.cuda.get_device_name(0)})
    data = Path(base.TRAINING_RESULT_MOUNT)/protocol["dataset_relative"]
    checkpoint = Path(base.TRAINING_RESULT_MOUNT)/protocol["checkpoint_relative"]
    if (base._sha256_file(data/"manifest.json") != protocol["dataset_manifest_sha256"] or
        base._sha256_file(checkpoint/"manifest.json") != protocol["checkpoint_manifest_sha256"]):
        raise ValueError("Input manifest drift after CPU verification")
    base.training_result_volume.commit()
    records = []
    for arm in ARMS:
        target = out/arm["id"]; target.mkdir(exist_ok=False)
        arm_env = dict(env, TORCHINDUCTOR_CACHE_DIR=str(scratch/arm["id"]/"inductor"),
                       TRITON_CACHE_DIR=str(scratch/arm["id"]/"triton"))
        cmd = [sys.executable, "-m", "research.precision_compile_benchmark", "--protocol", str(scratch/"protocol.json"),
               "--arm", arm["id"], "--data", str(data), "--checkpoint", str(checkpoint),
               "--scratch", str(scratch), "--output", str(target)]
        write_json(target/"launch.json", {"command": cmd, "arm": arm, "cold_cache": True})
        print("Starting "+arm["id"], flush=True)
        began = time.monotonic(); status = "running"; last_commit = began
        with (target/"stdout.log").open("x") as log:
            process = subprocess.Popen(cmd, cwd="/root", env=arm_env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while process.poll() is None:
                    if time.monotonic()-began > arm["timeout"]:
                        status = "timeout"; break
                    memory_path = Path("/sys/fs/cgroup/memory.current")
                    if memory_path.exists() and int(memory_path.read_text()) > 30*1024**3:
                        status = "host_memory_guard"; break
                    if time.monotonic()-last_commit > 45:
                        print(arm["id"]+f" running {time.monotonic()-began:.0f}s", flush=True)
                        base.training_result_volume.commit(); last_commit = time.monotonic()
                    time.sleep(1)
            finally:
                stop_process_group(process)
        if status == "running": status = "complete" if process.returncode == 0 else "failed"
        result_path = target/"result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        record = {"arm": arm["id"], "process_status": status, "returncode": process.returncode,
                  "elapsed_seconds": time.monotonic()-began,
                  "result_status": result["status"] if result else None}
        if result:
            expected = [a["input_sha256"] for a in pre["audits"]][:len(result["batches"])]
            if [b["input_sha256"] for b in result["batches"]] != expected:
                raise ValueError("GPU batches differ from CPU-verified batches")
        write_json(target/"receipt.json", record); records.append(record)
        base.training_result_volume.commit()
        print(json.dumps(record), flush=True)
        if arm["id"] == "fp32_eager" and (status != "complete" or result["status"] != "complete"):
            break  # No comparison without a valid FP32 reference.
    final = {"records": records, "git_commit": commit, "declared_cost": cost,
             "output_relative": str(out.relative_to(base.TRAINING_RESULT_MOUNT)),
             "all_complete": len(records) == len(ARMS) and all(r["result_status"] == "complete" for r in records)}
    write_json(out/"sweep.json", final); base.training_result_volume.commit()
    return final


@app.local_entrypoint()
def main(protocol_path: str, stage: str):
    from research.infra.modal_eval_matrix import atomic_json
    path = Path(protocol_path); protocol = json.loads(path.read_text())
    verify_source(protocol, Path.cwd())
    if stage not in {"preflight", "sweep"}: raise ValueError("Unknown stage")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit inputs before launch")
    receipt_path = path.parent/(stage+"_run.json")
    if receipt_path.exists(): raise ValueError("Attempt already exists; no automatic retry")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    spend = max(Decimal(protocol["prior_conservative_floor_dollars"]), base._modal_launch_spend())
    if stage == "sweep":
        spend += Decimal(estimate_modal_job(spec())["declared_attempt_upper_bound_dollars"])
    decision = enforce_modal_budget(estimate_modal_job(spec(stage == "sweep")),
        recorded_month_spend=spend, per_job_cap=Decimal("3.20" if stage == "sweep" else ".08"))
    if stage == "preflight":
        atomic_json(receipt_path, {"status": "launching", "cost_guard": decision, "git_commit": commit})
        result = preflight.remote(protocol, commit, decision["declared_attempt_upper_bound"])
        atomic_json(path.parent/"preflight.json", result)
        atomic_json(receipt_path, {"status": "complete", "cost_guard": decision, "git_commit": commit})
    else:
        pre = json.loads((path.parent/"preflight.json").read_text())
        raw = b"".join(base.training_result_volume.read_file(pre["output_relative"]+"/preflight.json"))
        if json.loads(raw) != pre or pre["protocol"] != protocol or not pre["passed"] or pre["gpu_active"]:
            raise ValueError("Matching CPU receipt required")
        atomic_json(receipt_path, {"status": "launching", "cost_guard": decision, "git_commit": commit})
        call = sweep.spawn(protocol, hashlib.sha256(raw).hexdigest(), commit, decision["declared_attempt_upper_bound"])
        atomic_json(receipt_path, {"status": "submitted", "cost_guard": decision,
                                  "git_commit": commit, "call_id": call.object_id})
        print(json.dumps({"call_id": call.object_id, "output_relative": pre["output_relative"]}), flush=True)
