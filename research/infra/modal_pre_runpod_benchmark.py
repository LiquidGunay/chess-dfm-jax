"""GPU-003 budget-gated CPU verification and bounded single-L40S tests."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import modal
from research.infra import modal_train_app as base
from research.infra.modal_precision_compile_benchmark import stop_process_group
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget
from research.pre_runpod_benchmark import ARMS, choose_candidate, write_json

app = modal.App("chess-dfm-gpu003-pre-runpod")
SOURCES = ("research/pre_runpod_benchmark.py", "research/infra/modal_pre_runpod_benchmark.py",
    "research/precision_compile_benchmark.py", "research/infra/modal_precision_compile_benchmark.py",
    "research/infra/modal_train_app.py", "research/interpretability/remote_cost.py")
VOLUMES = {base.BASE_INPUT_MOUNT: base.base_input_volume.with_mount_options(read_only=True),
           base.TRAINING_RESULT_MOUNT: base.training_result_volume}


def source_identity(root): return base._tree_sha256(Path(root), relative_paths=SOURCES)


def verify_source(protocol, root):
    if source_identity(root) != protocol["source_sha256"]: raise ValueError("GPU-003 source drift")
    for name, digest in protocol["frozen_training_sources"].items():
        if base._sha256_file(Path(root)/name) != digest: raise ValueError("Frozen dependency changed: "+name)
    if protocol["arms"] != list(ARMS): raise ValueError("Arm drift")


def spec(gpu=False):
    return ModalJobSpec(name="GPU-003-"+("sweep" if gpu else "preflight"),
        timeout_seconds=5700 if gpu else 900, startup_timeout_seconds=600 if gpu else 300,
        retries=0, cpu_cores=Decimal(4), memory_gib=Decimal(32 if gpu else 16), gpu="L40S" if gpu else None)


def output_root(protocol):
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return Path(base.TRAINING_RESULT_MOUNT)/"benchmarks"/("gpu003-"+digest[:20])


def environment(scratch):
    env = os.environ.copy()
    env.update(CHESS_DFM_WORKSPACE_ROOT="/workspace",
        CHESS_DFM_TRUSTED_WORKSPACE_ROOTS=os.pathsep.join((base.BASE_INPUT_MOUNT, base.TRAINING_RESULT_MOUNT, str(scratch))),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_MAX_THREADS="2",
        TORCHINDUCTOR_COMPILE_THREADS="1")
    return env


@app.function(image=base.gpu_image, cpu=4, memory=16384, timeout=900, startup_timeout=300,
              retries=0, max_containers=1, single_use_containers=True, volumes=VOLUMES)
def preflight(protocol, commit, cost):
    scratch = Path(tempfile.mkdtemp(prefix="gpu003-preflight-"))
    os.environ.update(environment(scratch))
    verify_source(protocol, "/root")
    import torch
    torch.set_num_threads(2)
    from research.hero_history_data import verify_dataset
    from research.input_contract import audit_full_history_batch
    from research.pre_runpod_benchmark import batches, prepare
    from research.train_torch import _verified_training_checkpoint
    data = Path(base.TRAINING_RESULT_MOUNT)/protocol["dataset_relative"]
    if base._sha256_file(data/"manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest mismatch")
    verify_dataset(data)
    checkpoint = Path(base.TRAINING_RESULT_MOUNT)/protocol["checkpoint_relative"]
    if base._sha256_file(checkpoint/"manifest.json") != protocol["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint manifest mismatch")
    _, manifest = _verified_training_checkpoint(checkpoint)
    if (manifest["optimizer_update"] != 500 or manifest["next_data_cursor"] != 500 or
        manifest["state"]["sha256"] != protocol["checkpoint_state_sha256"]):
        raise ValueError("Wrong parent checkpoint")
    loader = batches(protocol, data)
    if loader.provenance() != manifest["resume_contract"]["data"]:
        raise ValueError("Parent data provenance mismatch")
    hashes, audits = {}, []
    for cursor in range(500, 550):
        audits.append({"cursor": cursor, **audit_full_history_batch(loader.batch_at(cursor), allow_chess960_initial=True)})
        _, hashes[str(cursor)] = prepare(loader, cursor)
        if cursor % 10 == 0: print("CPU audited training cursor "+str(cursor), flush=True)
    validation = batches(protocol, data, "validation")
    validation_audits = [audit_full_history_batch(validation.batch_at(i), allow_chess960_initial=True) for i in range(16)]
    out = output_root(protocol); out.mkdir(parents=True, exist_ok=False)
    result = {"passed": True, "gpu_active": False, "full_dataset_hashes_verified": True,
        "protocol": protocol, "git_commit": commit, "declared_cost": cost,
        "training_hashes": hashes, "history_audits": audits, "validation_audits": validation_audits,
        "output_relative": str(out.relative_to(base.TRAINING_RESULT_MOUNT))}
    write_json(out/"preflight.json", result); base.training_result_volume.commit()
    return result


def subprocess_arm(selected, protocol, scratch, out, pre, env, *, resume=False):
    label = selected["id"]+("_resume" if resume else "")
    target = out/label; target.mkdir(exist_ok=False)
    cmd = [sys.executable, "-m", "research.pre_runpod_benchmark", "--protocol", str(scratch/"protocol.json"),
        "--preflight", str(out/"preflight.json"), "--arm", selected["id"],
        "--data", str(Path(base.TRAINING_RESULT_MOUNT)/protocol["dataset_relative"]),
        "--checkpoint", str(Path(base.TRAINING_RESULT_MOUNT)/protocol["checkpoint_relative"]),
        "--scratch", str(scratch), "--output", str(target)]
    if resume: cmd += ["--resume", str(scratch/selected["id"]/"checkpoints"/"update00000545")]
    write_json(target/"launch.json", {"command": cmd, "arm": selected, "resume": resume,
        "cache_policy": "Fresh process and empty cache for each arm; resumed process reuses same-arm compiler cache"})
    child_env = dict(env, TORCHINDUCTOR_CACHE_DIR=str(scratch/selected["id"]/"inductor"),
                     TRITON_CACHE_DIR=str(scratch/selected["id"]/"triton"))
    begin = time.monotonic(); last_commit = begin; status = "running"
    with (target/"stdout.log").open("x") as log:
        process = subprocess.Popen(cmd, cwd="/root", env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-begin > (500 if resume else selected["timeout"]):
                    status = "timeout"; break
                memory_path = Path("/sys/fs/cgroup/memory.current")
                if memory_path.exists() and int(memory_path.read_text()) > 30*1024**3:
                    status = "host_memory_guard"; break
                if time.monotonic()-last_commit > 45:
                    print(label+f" running {time.monotonic()-begin:.0f}s", flush=True)
                    base.training_result_volume.commit(); last_commit = time.monotonic()
                time.sleep(1)
        finally: stop_process_group(process)
    if status == "running": status = "complete" if process.returncode == 0 else "failed"
    result = json.loads((target/"result.json").read_text()) if (target/"result.json").exists() else None
    record = {"arm": label, "process_status": status, "returncode": process.returncode,
        "elapsed_seconds": time.monotonic()-begin, "result_status": result["status"] if result else None}
    if result:
        expected_cursors = range(545 if resume else 500, 550)
        if [r["input_sha256"] for r in result["records"]] != [pre["training_hashes"][str(i)] for i in expected_cursors]:
            raise ValueError("Output input/RNG identities mismatch")
    write_json(target/"receipt.json", record); base.training_result_volume.commit()
    print(json.dumps(record), flush=True)
    return record, result


@app.function(image=base.gpu_image, gpu="L40S", cpu=4, memory=32768, timeout=5700, startup_timeout=600,
              retries=0, max_containers=1, single_use_containers=True, volumes=VOLUMES)
def sweep(protocol, preflight_sha, commit, cost):
    verify_source(protocol, "/root")
    out = output_root(protocol)
    if base._sha256_file(out/"preflight.json") != preflight_sha: raise ValueError("CPU preflight drift")
    pre = json.loads((out/"preflight.json").read_text())
    if pre["protocol"] != protocol or not pre["passed"] or pre["gpu_active"]: raise ValueError("CPU verification required")
    scratch = Path(tempfile.mkdtemp(prefix="gpu003-sweep-"))
    env = environment(scratch); os.environ.update(env)
    import torch
    if "L40S" not in torch.cuda.get_device_name(): raise ValueError("Unexpected GPU")
    for relative, digest in ((protocol["dataset_relative"]+"/manifest.json", protocol["dataset_manifest_sha256"]),
                             (protocol["checkpoint_relative"]+"/manifest.json", protocol["checkpoint_manifest_sha256"])):
        if base._sha256_file(Path(base.TRAINING_RESULT_MOUNT)/relative) != digest: raise ValueError("Input manifest drift")
    write_json(scratch/"protocol.json", protocol)
    write_json(out/"launch.json", {"protocol": protocol, "git_commit": commit,
        "preflight_sha256": preflight_sha, "declared_cost": cost, "device": torch.cuda.get_device_name()})
    base.training_result_volume.commit()
    records, results = [], {}
    for selected in ARMS:
        record, result = subprocess_arm(selected, protocol, scratch, out, pre, env)
        records.append(record)
        if result and record["process_status"] == "complete": results[selected["id"]] = result
        if selected["id"] == "production_control" and selected["id"] not in results: break
    winner = choose_candidate(results)
    resume_target = winner or ("production_control" if "production_control" in results else None)
    resume_evidence = None
    if resume_target:
        selected = next(a for a in ARMS if a["id"] == resume_target)
        record, result = subprocess_arm(selected, protocol, scratch, out, pre, env, resume=True)
        records.append(record)
        resume_evidence = {"target": resume_target, "completed": result is not None,
            "state_bitwise_equal": bool(result and result["final_state_hashes"] == results[resume_target]["final_state_hashes"]),
            "validation_equal": bool(result and result["validation"] == results[resume_target]["validation"])}
    final = {"records": records, "candidate": winner, "resume": resume_evidence,
        "output_relative": str(out.relative_to(base.TRAINING_RESULT_MOUNT)),
        "all_arms_complete": len(results) == len(ARMS), "git_commit": commit, "declared_cost": cost,
        "cuda_graphs": "Deferred: first establish full-step gains and safe resume. No capture implementation or automatic runtime adoption."}
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
    # No other jobs may consume unreserved budget in this bounded round.
    active = json.loads(subprocess.check_output(["modal", "container", "list", "--json"], text=True))
    if active: raise ValueError("Other active containers: account for them before launch")
    receipt_path = path.parent/(stage+"_run.json")
    if receipt_path.exists(): raise ValueError("Attempt exists; no automatic retry")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    spend = max(Decimal(protocol["prior_conservative_floor_dollars"]), base._modal_launch_spend())
    if stage == "sweep": spend += Decimal(estimate_modal_job(spec())["declared_attempt_upper_bound_dollars"])
    decision = enforce_modal_budget(estimate_modal_job(spec(stage == "sweep")), recorded_month_spend=spend,
                                   per_job_cap=Decimal("4.30" if stage == "sweep" else ".11"))
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
        atomic_json(receipt_path, {"status": "submitted", "cost_guard": decision, "git_commit": commit, "call_id": call.object_id})
        print(json.dumps({"call_id": call.object_id, "output_relative": pre["output_relative"]}), flush=True)
