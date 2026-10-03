"""Budget-gated VP-001 CPU preflight, fresh joint training and four-way arena."""
from decimal import Decimal
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import modal
from research.infra import modal_train_app as base, modal_arena_app as arena
from research.infra import proposal_a_epoch_data as data_tools
from research.infra.modal_history_stage import extra_volumes
from research.infra.modal_precision_compile_benchmark import stop_process_group
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget

app = modal.App("chess-dfm-vp001-joint-value")
image = base.gpu_image.uv_pip_install("wandb==0.19.11")
image = image.add_local_file(arena.TERMINAL_CONFIG_RELATIVE,"/root/"+arena.TERMINAL_CONFIG_RELATIVE,copy=True)
for asset in arena.ARENA_ASSET_NAMES:
    image = image.add_local_file("research/assets/arena/"+asset, "/opt/chess-assets/"+asset, copy=True)
SOURCES = ("research/train_torch.py","research/value_pilot.py","research/infra/modal_value_pilot.py",
    "research/pre_runpod_benchmark.py","research/evaluate_arena.py","research/evaluate_policy_loss_play.py",
    "research/evaluate_common_policy.py","research/infra/modal_train_app.py","research/infra/modal_arena_app.py",
    "research/infra/proposal_a_epoch_data.py","research/input_contract.py","chess_dfm_jax/data/lc0_sequential.py",
    "chess_dfm_jax/weights.py","research/interpretability/remote_cost.py")
VOLUMES = {base.BASE_INPUT_MOUNT:base.base_input_volume.with_mount_options(read_only=True),
    base.TRAINING_INPUT_MOUNT:base.training_input_volume.with_mount_options(read_only=True),
    base.TRAINING_RESULT_MOUNT:base.training_result_volume}
HERO = base.HERO1_EXACT_PHASE1_RESULT_LABEL
HERO_SHA = "e518083ee5476e7ff15e7c34caf9c0505f8fb7d330ea5a6882e4e9e5c1ef1912"


def identity(root):
    return hashlib.sha256((base._tree_sha256(Path(root), relative_paths=tuple(set(SOURCES+base.SOURCE_TREE_PATHS)))
        + arena._arena_source_tree_sha256(Path(root))).encode()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write("\n")


def spec(stage):
    gpu = {"preflight":None,"train":"L40S","evaluate":"L4","eval-preflight":None}[stage]
    return ModalJobSpec(name="VP-001-"+stage, timeout_seconds={"preflight":1800,"train":5400,"evaluate":9600,"eval-preflight":600}[stage],
        startup_timeout_seconds=600 if gpu else 300, retries=0,cpu_cores=Decimal(4),
        memory_gib=Decimal(32 if stage == "train" else 16),gpu=gpu)


def root(protocol):
    digest = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:20]
    return Path(base.TRAINING_RESULT_MOUNT)/"experiments"/("vp001-"+digest)


def label(protocol, arm):
    return "hero-training-vp001-"+arm.replace("_","-")+"-"+root(protocol).name[-20:]


def environment():
    os.environ.update(CHESS_DFM_WORKSPACE_ROOT="/workspace",
        CHESS_DFM_TRUSTED_WORKSPACE_ROOTS=os.pathsep.join((base.BASE_INPUT_MOUNT,base.TRAINING_INPUT_MOUNT,base.TRAINING_RESULT_MOUNT,"/tmp","/root")),
        OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",NUMEXPR_MAX_THREADS="2",TORCHINDUCTOR_COMPILE_THREADS="1")


def verify(protocol):
    environment()
    if identity("/root") != protocol["source_sha256"]:
        raise ValueError("VP-001 frozen source mismatch")


@app.function(image=image,cpu=4,memory=16384,timeout=1800,startup_timeout=300,retries=0,
    max_containers=1,single_use_containers=True,volumes={**VOLUMES,**extra_volumes})
def preflight(protocol):
    verify(protocol)
    from research.value_pilot import batches, config, prepare
    from research.pre_runpod_benchmark import prepared_digest
    from research.input_contract import audit_full_history_batch
    from research.evaluate_common_policy import prepare as prepare_policy
    import torch
    import numpy as np
    torch.set_num_threads(2)
    out = root(protocol); out.mkdir(parents=True,exist_ok=False)
    write(out/"protocol.json",protocol)
    inventory = {}
    for mount in (*VOLUMES, *extra_volumes):
        inventory[mount] = sum((Path(p)/n).stat().st_size for p,_,names in os.walk(mount) for n in names)
    if (sum(inventory.values())+20*1024**3)/1024**3 >= 768:
        raise RuntimeError("Projected storage reaches review threshold")
    data = base._training_dataset_root(data_tools.DATASET_LABEL)/"lc0_sequential"
    receipt = data_tools.verify_dataset(data)
    raw = base._verify_raw_bt4_volume()
    hero = arena._validated_training_result(HERO)
    if base._sha256_file(hero["root"]/"checkpoint/model.safetensors") != HERO_SHA:
        raise ValueError("Hero 1 checkpoint changed")
    arena._materialize_arena_assets()
    hashes = {}
    loader = batches(data)
    for i in range(1000):
        audit_full_history_batch(loader.batch_at(i))
        item = prepare(loader,i,config("native"))
        hashes[str(i)] = prepared_digest(item)
        if "current_wdl_target" not in item.compact_batch:
            raise ValueError("Current search target missing")
        if i % 100 == 0:
            print(json.dumps({"CPU_verified_updates":i,"GPU_active":False}),flush=True)
    val = batches(data,"validation")
    value_targets = []
    for i in range(128):
        batch = val.batch_at(i); audit_full_history_batch(batch)
        value_targets.append(batch["current_wdl_target"])
    with (out/"value_targets.npz").open("xb") as f:
        np.savez_compressed(f,wdl=np.concatenate(value_targets))
    pool = prepare_policy(data,out/"policy_validation.npz",out/"policy_validation.json")
    result = {"passed":True,"gpu_active":False,"training_hashes":hashes,"data":receipt,
        "training_provenance":loader.provenance(),"validation_provenance":val.provenance(),
        "raw_sha256":base._sha256_file(raw),"hero_sha256":HERO_SHA,"storage_bytes":inventory,
        "validation":pool,"value_targets_sha256":base._sha256_file(out/"value_targets.npz"),
        "source_sha256":protocol["source_sha256"]}
    write(out/"preflight.json",result); base.training_result_volume.commit()
    return {"output_relative":str(out.relative_to(base.TRAINING_RESULT_MOUNT)),
        "preflight_sha256":base._sha256_file(out/"preflight.json"),"passed":True}


@app.function(image=image,gpu="L40S",cpu=4,memory=32768,timeout=5400,startup_timeout=600,
    retries=0,max_containers=1,single_use_containers=True,volumes=VOLUMES,
    secrets=[modal.Secret.from_name("chess-dfm-wandb",required_keys=["WANDB_API_KEY"])])
def train(protocol, preflight_sha, cost, commit):
    verify(protocol)
    out = root(protocol)
    if base._sha256_file(out/"preflight.json") != preflight_sha:
        raise ValueError("CPU preflight checksum mismatch")
    pre = json.loads((out/"preflight.json").read_text())
    if not pre["passed"] or pre["gpu_active"]:
        raise ValueError("CPU verification required")
    scratch = Path(tempfile.mkdtemp(prefix="vp001-"))
    source_data = base._training_dataset_root(data_tools.DATASET_LABEL)/"lc0_sequential"
    data_tools.copy_verified_dataset(source_data,scratch/"data")
    raw = base._verify_raw_bt4_volume()
    receipts = []
    for arm in ("native","jepa_residual"):
        destination = Path(base.TRAINING_RESULT_MOUNT)/"runs"/label(protocol,arm)
        command = [sys.executable,"-u","-m","research.value_pilot","--arm",arm,"--data",str(scratch/"data"),
            "--raw",str(raw),"--output",str(destination),"--protocol",str(out/"protocol.json"),"--preflight",str(out/"preflight.json")]
        env = dict(os.environ,TORCHINDUCTOR_CACHE_DIR=str(scratch/arm/"inductor"),TRITON_CACHE_DIR=str(scratch/arm/"triton"),CHESS_DFM_GIT_COMMIT=commit)
        start = time.monotonic(); last_commit = start
        with (out/(arm+".log")).open("x") as log:
            child = subprocess.Popen(command,cwd="/root",env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    if time.monotonic()-start > 2550:
                        raise TimeoutError("Per-arm runtime cap reached")
                    if time.monotonic()-last_commit > 40:
                        base.training_result_volume.commit(); last_commit = time.monotonic()
                        print(json.dumps({"arm":arm,"seconds":round(last_commit-start),"log":str(out/(arm+".log"))}),flush=True)
                    time.sleep(1)
                if child.returncode:
                    raise RuntimeError("Training subprocess failed: "+arm)
            finally:
                stop_process_group(child); base.training_result_volume.commit()
        report = json.loads((destination/"report.json").read_text())
        if report["updates"] != 1000:
            raise ValueError("Incomplete pilot")
        write(destination/"modal_run.json",{"result_label":label(protocol,arm),"profile":"vp001","arm":arm,
            "source_sha256":protocol["source_sha256"],"git_commit":commit,"declared_attempt_upper_bound_dollars":cost})
        receipts.append({"arm":arm,"label":label(protocol,arm),"report":report})
        base.training_result_volume.commit()
    if receipts[0]["report"]["common_initial_parameters_sha256"] != receipts[1]["report"]["common_initial_parameters_sha256"]:
        raise ValueError("Common fresh initialization differs")
    write(out/"training_complete.json",{"status":"complete","runs":receipts})
    base.training_result_volume.commit()
    return {"status":"complete","output_relative":str(out.relative_to(base.TRAINING_RESULT_MOUNT)),"labels":[x["label"] for x in receipts]}


@app.function(image=image,cpu=4,memory=16384,timeout=600,startup_timeout=300,retries=0,
    max_containers=1,single_use_containers=True,volumes=VOLUMES)
def eval_preflight(protocol):
    verify(protocol)
    out = root(protocol)
    if not (out/"training_complete.json").is_file():
        raise ValueError("Training must complete first")
    from research.evaluate_arena import torch_hero_checkpoint_descriptor
    raw = arena._verify_raw_bt4_volume()
    models = {}
    for name, lab in (("native",label(protocol,"native")),("jepa_residual",label(protocol,"jepa_residual")),("hero1",HERO)):
        record = arena._validated_training_result(lab)
        if base._sha256_file(record["root"]/"checkpoint/model.safetensors") != record["state_sha256"]:
            raise ValueError("Evaluation checkpoint bytes changed")
        torch_hero_checkpoint_descriptor(record["root"], models_dir=raw.parent)
        models[name] = {"label":lab,"root":str(record["root"]),"state_sha256":record["state_sha256"],
            "plane_history_mode":"current_only_as_preprocessed" if name == "hero1" else "lc0_full_history_as_preprocessed"}
    models["raw_bt4"] = {"root":None,"label":"raw-bt4","state_sha256":base._sha256_file(raw),"plane_history_mode":"lc0_full_history_as_preprocessed"}
    pre = json.loads((out/"preflight.json").read_text())
    if base._sha256_file(out/"policy_validation.npz") != pre["validation"]["file_sha256"]:
        raise ValueError("Validation input changed")
    write(out/"evaluation_preflight.json",{"passed":True,"gpu_active":False,"models":models,
        "source_sha256":protocol["source_sha256"],"arena_source_sha256":arena._arena_source_tree_sha256(Path("/root"))})
    base.training_result_volume.commit()
    return {"sha256":base._sha256_file(out/"evaluation_preflight.json"),"models":models}


@app.function(image=image,gpu="L4",cpu=4,memory=16384,timeout=9600,startup_timeout=600,retries=0,
    max_containers=1,single_use_containers=True,volumes=VOLUMES)
def evaluate(protocol, preflight_sha, cost, commit):
    verify(protocol)
    out = root(protocol)
    if base._sha256_file(out/"evaluation_preflight.json") != preflight_sha:
        raise ValueError("Evaluation preflight drift")
    pre = json.loads((out/"evaluation_preflight.json").read_text())
    summaries = []
    for left,right in itertools.combinations(("native","jepa_residual","hero1","raw_bt4"),2):
        result = arena._run_arena(candidate_label=pre["models"][left]["label"],opponent_label=pre["models"][right]["label"],
            candidate_policy_mode="dfm",opponent_policy_mode="policy_only" if right == "raw_bt4" else "dfm",
            pair_count=128,candidate_refinement_passes=1,opponent_refinement_passes=1,opening_start_index=1664,
            expected_arena_source_tree_sha256=pre["arena_source_sha256"],git_commit=commit,
            declared_attempt_upper_bound_dollars=cost,raw_bt4_plane_history_mode="lc0_full_history_as_preprocessed")
        record = {"left":left,"right":right,"summary":result}
        write(out/("match-"+left+"-"+right+".json"),record); summaries.append(record)
        base.training_result_volume.commit()
    from research.evaluate_policy_loss_play import evaluate as policy_losses
    losses = policy_losses({"sequential":{"path":str(out/"policy_validation.npz"),"positions":8192}},
        pre["models"],arena._verify_raw_bt4_volume(),out)
    losses["experiment_id"] = "VP-001"
    write(out/"policy_losses.json",losses)
    write(out/"evaluation_complete.json",{"status":"complete","matches":summaries,"policy_losses":losses})
    base.training_result_volume.commit()
    return {"status":"complete","output_relative":str(out.relative_to(base.TRAINING_RESULT_MOUNT))}


@app.local_entrypoint()
def main(stage: str, protocol_path: str="research/analysis/vp001_joint_value_20261003/preregister.json"):
    path = Path(protocol_path)
    if stage == "freeze":
        protocol = {"experiment_id":"VP-001","source_sha256":identity(Path.cwd()),
            "initialization":"Fresh raw BT4 including native value; seed-0 Hero JEPA/DFM/projector; no Hero checkpoint resume",
            "arms":["native","jepa_residual"],"steps":1000,"batch_size":1024,"seed":0,
            "data_inventory_sha256":data_tools.INVENTORY_SHA256,"full_history":True,
            "value_targets":"Actual root_q/root_d at current and future positions; no successor fallback",
            "value_objective":"Both: coefficient .25 times uniform mean WDL CE over H0..H8. A: native H0 + original auxiliary futures. B: current-native logits (W/L swapped on odd horizons) plus auxiliary residual at H0..H8; residual output initialized zero.",
            "other_losses":"Original Hero DFM, root legal CE, legality, JEPA and SIGReg; no teacher KL or feedback",
            "optimizer":"Hero peaks 5e-4 and 5e-4/30, selective WD .01; FP32 weights/moments, BF16 compute; WSD original 566866-example warmup, original horizon and 20% cooldown. Stop in stable phase, preserve recovery at 554/1000.",
            "validation_updates":[0,250,554,1000],"validation_positions_intermediate":2048,"validation_positions_final":8192,
            "tournament":{"players":["native","jepa_residual","hero1","raw_bt4"],"pairs_per_match":128,
                "opening_start":1664,"matches":6,"games":1536,"policy":"one-pass DFM; raw native policy",
                "input_contract":"Full history for new models/raw, original current-only Hero1",
                "scope":"Searchless development round robin; no held-out or search-strength claim; fixed terminal update, no selection"},
            "budget":{"live_compute_floor":"7.81","old_completed_maxima_released":True,"training_round_cap":"4.25","evaluation_round_cap":"3.50","stop":"34","outer":"42.50"},
            "historical_correction":"PA-001 full-history inputs were correct, but current WDL targets were dropped by batch filters. New pipeline preserves real current targets and fails closed if absent; historical evidence is unchanged."}
        write(path,protocol); print(json.dumps(protocol)); return
    protocol = json.loads(path.read_text())
    if identity(Path.cwd()) != protocol["source_sha256"]:
        raise ValueError("Local source changed after freeze")
    # Completed historical maximum reservations are not actual usage. Reserve
    # this workflow's whole bounded phases until its terminal reconciliation.
    prior = Decimal(protocol["budget"]["live_compute_floor"])
    stages = ["preflight","train","eval-preflight","evaluate"]
    for previous in stages[:stages.index(stage)]:
        prior += Decimal(estimate_modal_job(spec(previous))["declared_attempt_upper_bound_dollars"])
    spend = max(base._modal_launch_spend(),prior)
    cap = {"preflight":".19","train":"4.00","eval-preflight":".08","evaluate":"3.20"}[stage]
    decision = enforce_modal_budget(estimate_modal_job(spec(stage)),recorded_month_spend=spend,per_job_cap=Decimal(cap))
    commit = base._local_git_commit(Path.cwd())
    write(path.parent/(stage+"-reservation.json"),decision)
    if stage == "preflight":
        result = preflight.remote(protocol)
    elif stage == "eval-preflight":
        result = eval_preflight.remote(protocol)
    elif stage == "train":
        pre = json.loads((path.parent/"preflight-result.json").read_text())
        call = train.spawn(protocol,pre["preflight_sha256"],decision["declared_attempt_upper_bound"],commit)
        result = {"call_id":call.object_id,"output_relative":str(root(protocol).relative_to(base.TRAINING_RESULT_MOUNT))}
    else:
        pre = json.loads((path.parent/"eval-preflight-result.json").read_text())
        call = evaluate.spawn(protocol,pre["sha256"],decision["declared_attempt_upper_bound"],commit)
        result = {"call_id":call.object_id}
    write(path.parent/(stage+"-result.json"),result); print(json.dumps(result),flush=True)
