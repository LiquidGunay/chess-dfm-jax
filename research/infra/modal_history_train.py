"""HH-002: one bounded training-and-loss-comparison attempt after CPU staging."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal
from research.infra import modal_train_app as base
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget

PLAN = "research/analysis/hh002_history_subset_20261003/plan.json"
AMENDMENT = "research/analysis/hh002_history_subset_20261003/validation_cost_amendment.json"
SOURCE_PATHS = ("research/train_torch.py", "research/prepare.py", "research/arena.py",
    "research/play_arena.py", "research/arena_history_trust.py", "research/input_contract.py",
    "research/hero_history_data.py", "research/hero_history_subset_data.py", "research/hero_history_batches.py",
    "research/hero_history_tracking.py", "research/analysis/eval_history_loss_baseline.py",
    "research/infra/modal_history_train.py", "research/infra/modal_train_app.py",
    "research/interpretability/remote_cost.py", PLAN, AMENDMENT)
app = modal.App("chess-dfm-hh002-train")
image = (base.gpu_image.uv_pip_install("wandb==0.19.11")
         .add_local_file(PLAN,"/root/"+PLAN,copy=True)
         .add_local_file(AMENDMENT,"/root/"+AMENDMENT,copy=True))
wandb_secret = os.environ.get("HH002_WANDB_SECRET", "")
if wandb_secret not in {"", "chess-dfm-wandb"}:
    raise ValueError("Unexpected W&B secret name")
secrets = [modal.Secret.from_name(wandb_secret, required_keys=["WANDB_API_KEY"])] if wandb_secret else []


def source_identity(root):
    return base._tree_sha256(Path(root), relative_paths=SOURCE_PATHS)


def spec():
    return ModalJobSpec(name="HH-002-training-and-loss-comparison", timeout_seconds=5400,
        startup_timeout_seconds=900,retries=0,cpu_cores=Decimal(4),memory_gib=Decimal(32),gpu="L40S")


def training_command(data_root, manifest_sha256, raw, output):
    return [sys.executable,"-m","research.train_torch","train","--recipe","hero",
        "--data-format","trajectory_v3_history","--history-manifest-sha256",manifest_sha256,
        "--data-root",str(data_root),"--raw-bt4-path",str(raw),"--output-dir",str(output),
        "--batch-size","1024","--steps","553","--seed","0","--data-start","0",
        "--threads","2","--remat-mode","bt4-projector","--attention-impl","sdpa-all",
        "--compile-regions","fresh","--prefetch-depth","1","--train-seconds","0",
        "--log-every","20","--gpu-monitor-interval-ms","100",
        "--validation-updates","100","200","300","400","500","553",
        "--validation-example-count","8128","--save-updates","500","553",
        "--max-checkpoints","3","--save-final"]


@app.function(image=image,gpu="L40S",cpu=4,memory=32768,timeout=5400,startup_timeout=900,
              retries=0,max_containers=1,single_use_containers=True,secrets=secrets,
              volumes={base.BASE_INPUT_MOUNT:base.base_input_volume.with_mount_options(read_only=True),
                       base.TRAINING_RESULT_MOUNT:base.training_result_volume})
def run(preflight_relative, preflight_sha256, source_sha256, git_commit, declared_cost):
    from research.hero_history_data import verify_dataset, sha256, write_json_once
    from research.hero_history_tracking import LossTracker, JsonlTail, import_historical_curves
    started = time.monotonic()
    root = Path(base.TRAINING_RESULT_MOUNT)
    preflight_path = root/preflight_relative
    if sha256(preflight_path) != preflight_sha256 or source_identity("/root") != source_sha256:
        raise ValueError("Frozen CPU preflight or training source mismatch")
    receipt = json.loads(preflight_path.read_text())
    if not receipt.get("passed") or receipt.get("gpu_active") is not False:
        raise ValueError("A complete CPU-only preflight is required")
    data = Path(receipt["dataset_root"])
    if sha256(data/"manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("Frozen dataset manifest drift")
    verify_dataset(data)
    raw = base._verify_raw_bt4_volume()
    label = "hh002-hero-history-"+hashlib.sha256((source_sha256+receipt["manifest_sha256"]).encode()).hexdigest()[:20]
    output, comparison = root/"runs"/label, root/"comparisons"/label
    comparison.mkdir(parents=True,exist_ok=False)
    if output.exists():
        raise FileExistsError("Never overwrite or automatically restart a training attempt")
    baseline = root/"runs"/base.HERO1_EXACT_PHASE1_RESULT_LABEL
    if sha256(baseline/"metrics.jsonl") != receipt["baseline_training_metrics_sha256"]:
        raise ValueError("Historical training curve drift")
    if sha256(baseline/"run_config.json") != receipt["baseline_run_config_sha256"]:
        raise ValueError("Historical model configuration drift")
    environment = os.environ.copy()
    cache = Path("/tmp")/label
    environment.update({"CHESS_DFM_WORKSPACE_ROOT":"/workspace",
        "CHESS_DFM_TRUSTED_WORKSPACE_ROOTS":os.pathsep.join([base.BASE_INPUT_MOUNT,base.TRAINING_RESULT_MOUNT]),
        "CHESS_DFM_GIT_COMMIT":git_commit,"CHESS_DFM_SOURCE_TREE_SHA256":source_sha256,
        "TORCHINDUCTOR_CACHE_DIR":str(cache/"inductor"),"TRITON_CACHE_DIR":str(cache/"triton"),
        "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","MKL_NUM_THREADS":"2","NUMEXPR_MAX_THREADS":"2"})
    launch = {"experiment_id":"HH-002","result_label":label,"source_tree_sha256":source_sha256,
              "git_commit":git_commit,"dataset_manifest_sha256":receipt["manifest_sha256"],
              "preflight_sha256":preflight_sha256,"declared_attempt_upper_bound_dollars":declared_cost,
              "stage_receipt":receipt,"plan":json.loads((Path("/root")/PLAN).read_text()),
              "amendment":json.loads((Path("/root")/AMENDMENT).read_text())}
    write_json_once(comparison/"launch.json",launch)
    base.training_result_volume.commit()
    tracker = LossTracker(comparison/"tracking", run_id="hh002-full-"+source_sha256[:12],
        name="Hero recipe — full history (reduced data)",config={
            "experiment_id":"HH-002","history_mode":"full_history","data_sha256":receipt["manifest_sha256"],
            "source_sha256":source_sha256,"plan":launch["plan"],"validation_protocol":launch["amendment"]})
    tails = {"train":JsonlTail(output/"metrics.jsonl"),"val":JsonlTail(output/"validation_metrics.jsonl")}
    last_commit = time.monotonic()
    def collect():
        for split, tail in tails.items():
            for record in tail.read():
                tracker.log(record,split)
    def execute(command, log_name, *, track=False, timeout=None):
        nonlocal last_commit
        process = subprocess.Popen(command,cwd="/root",env=environment,text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        began = time.monotonic()
        try:
            with (comparison/log_name).open("x") as log:
                for line in process.stdout:
                    log.write(line);log.flush()
                    print(line,end="",flush=True)
                    if track:
                        collect()
                    if time.monotonic()-last_commit > 60:
                        base.training_result_volume.commit();last_commit=time.monotonic()
                    if timeout is not None and time.monotonic()-began > timeout:
                        process.terminate()
                        raise TimeoutError("Comparison exceeded remaining bounded time")
            if process.wait() != 0:
                raise RuntimeError(f"Subprocess failed: {log_name}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    try:
        execute(training_command(data,receipt["manifest_sha256"],raw,output),"training.stdout.log",track=True)
        collect()
        tracker.finish()
        write_json_once(output/"modal_run.json",{**launch,"schema_version":"chess-dfm-modal-training-run-v1",
            "profile":"hh002","arm":"hero_full_history_subset","gpu":"L40S",
            "dataset":{"format":"trajectory_v3_history","data_root":str(data)},
            "raw_bt4_sha256":base.RAW_BT4_SHA256})
        base.training_result_volume.commit()
        baseline_status = "not_started_time_guard"
        if time.monotonic()-started < 3850:
            execute([sys.executable,"-m","research.analysis.eval_history_loss_baseline",
                "--preflight",str(preflight_path),"--raw",str(raw),"--baseline",str(baseline),
                "--output",str(comparison/"baseline_validation.jsonl")],"baseline.stdout.log",
                timeout=5300-(time.monotonic()-started))
            baseline_status = "complete" if len(receipt["baseline_checkpoints"]) == 6 else "available_checkpoints_only"
        baseline_tracking = import_historical_curves(comparison/"tracking",baseline,
            comparison/"baseline_validation.jsonl",run_id="hh002-old-"+source_sha256[:12],config={
                "history_mode":"current_only_as_preprocessed","role":"historical_reference_not_matched_training",
                "validation_data_sha256":receipt["manifest_sha256"],"validation_protocol":launch["amendment"],
                "checkpoints":receipt["baseline_checkpoints"]})
        result = {"training":"complete","result_label":label,"comparison":baseline_status,
                  "tracking":tracker.receipt,"baseline_tracking":baseline_tracking,
                  "seconds":time.monotonic()-started}
        write_json_once(comparison/"completion.json",result)
        return result
    except Exception as exc:
        write_json_once(comparison/"failure.json",{"type":type(exc).__name__,"message":str(exc),
            "training_report_present":(output/"report.json").is_file()})
        raise
    finally:
        collect()
        tracker.finish()
        base.training_result_volume.commit()


@app.local_entrypoint()
def main(preflight: str, recorded_spend: str = "4.76765855"):
    root = Path(__file__).resolve().parents[2]
    local = Path(preflight)
    receipt = json.loads(local.read_text())
    relative = str(Path(receipt["dataset_root"]).relative_to(base.TRAINING_RESULT_MOUNT)/"preflight.json")
    remote = b"".join(base.training_result_volume.read_file(relative))
    if json.loads(remote) != receipt or not receipt.get("passed"):
        raise ValueError("Local CPU receipt does not match completed remote preflight")
    spend = max(base._modal_launch_spend(),Decimal(recorded_spend))
    decision = enforce_modal_budget(estimate_modal_job(spec()),recorded_month_spend=spend,per_job_cap=Decimal("4.20"))
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip():
        raise RuntimeError("Commit launch inputs before GPU allocation")
    source = source_identity(root)
    print(json.dumps({"launch_budget":decision,"source_tree_sha256":source,"git_commit":commit}),flush=True)
    print(json.dumps({"hh002_result":run.remote(relative,hashlib.sha256(remote).hexdigest(),source,commit,
        decision["declared_attempt_upper_bound"])}),flush=True)
