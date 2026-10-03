"""CPU-only, immutable HH-002/Hero1 checkpoint and weight-movement audit."""
from decimal import Decimal
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import modal
from research.infra import modal_arena_app as arena
from research.infra import modal_train_app as train
from research.interpretability.remote_cost import ModalJobSpec, estimate_modal_job, enforce_modal_budget


CANDIDATE = "hh002-hero-history-0fe540075e5361da6c25"
BASELINE = train.HERO1_EXACT_PHASE1_RESULT_LABEL
SOURCE_FILES = ("research/infra/modal_hh002_audit.py",
                "research/interpretability/models.py", "research/interpretability/parameter_diff.py")
app = modal.App("chess-dfm-hh002-cpu-audit")


def source_identity(root):
    root = Path(root)
    return hashlib.sha256((arena._arena_source_tree_sha256(root)
                           + train._tree_sha256(root, relative_paths=SOURCE_FILES)).encode()).hexdigest()


def spec():
    return ModalJobSpec(name="HH-002-weight-audit-and-arena-preflight", timeout_seconds=1800,
                       startup_timeout_seconds=300, retries=0, cpu_cores=Decimal(4),
                       memory_gib=Decimal(16), gpu=None)


@app.function(image=arena.arena_image, cpu=4, memory=16384, timeout=1800, startup_timeout=300,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={arena.BASE_INPUT_MOUNT: arena.base_input_volume.with_mount_options(read_only=True),
                       arena.TRAINING_RESULT_MOUNT: arena.training_result_volume})
def run(expected_source, git_commit, declared_cost):
    import torch
    from research.input_contract import torch_hero_plane_history_mode
    from research.interpretability.models import BT4ComparisonModels, _encoder_layout
    from research.interpretability.parameter_diff import parameter_diff, NormAccumulator, _tensor_stats
    from research.train_torch import Config, JointModel, load_model_checkpoint, load_raw_bt4_hero_model
    from research.hero_history_data import write_json_once

    if source_identity("/root") != expected_source:
        raise ValueError("CPU audit source drift")
    started = time.monotonic()
    torch.set_num_threads(4)
    os.environ.update({"CHESS_DFM_WORKSPACE_ROOT": "/workspace",
        "CHESS_DFM_TRUSTED_WORKSPACE_ROOTS": os.pathsep.join([arena.BASE_INPUT_MOUNT, arena.TRAINING_RESULT_MOUNT])})
    raw = arena._verify_raw_bt4_volume()
    if arena._sha256_file(raw) != arena.RAW_BT4_SHA256:
        raise ValueError("Raw BT4 content drift")
    records, checked = {}, {}
    for label in (BASELINE, CANDIDATE):
        record = arena._validated_training_result(label)
        state = record["root"] / "checkpoint/model.safetensors"
        if arena._sha256_file(state) != record["state_sha256"]:
            raise ValueError("Checkpoint content drift")
        config_path = record["root"] / "run_config.json"
        payload = json.loads(config_path.read_text())
        records[label] = (record, payload)
        checked[label] = {"state_sha256": record["state_sha256"],
                          "run_config_sha256": arena._sha256_file(config_path),
                          "plane_history_mode": torch_hero_plane_history_mode(payload),
                          "updates": record["optimizer_update"]}
    effective_configs = {label: dataclasses.asdict(Config(**payload["config"]))
                         for label, (_, payload) in records.items()}
    if effective_configs[BASELINE] != effective_configs[CANDIDATE]:
        raise ValueError("Hero1 and HH-002 model recipes differ; comparison requires review")
    identity = hashlib.sha256(json.dumps({"models": checked, "source": expected_source}, sort_keys=True).encode()).hexdigest()
    output = Path(arena.TRAINING_RESULT_MOUNT) / "audits" / ("hh002-comparison-" + identity[:20])
    output.mkdir(parents=True, exist_ok=True)
    preflight = {"schema_version": "chess-dfm-arena-input-preflight-v1", "passed": True,
        "gpu_active": False, "models": checked, "raw_bt4_sha256": arena.RAW_BT4_SHA256,
        "arena_source_tree_sha256": arena._arena_source_tree_sha256(Path("/root")),
        "assets": {name: arena._sha256_file(Path("/opt/chess-assets") / name) for name in arena.ARENA_ASSET_NAMES}}
    preflight["effective_model_configs_identical"] = True
    preflight["legacy_config_note"] = "Missing historical optional fields use explicit current Config defaults; effective recipes are compared exactly."
    write_json_once(output / "input_preflight.json", preflight)
    arena.training_result_volume.commit()
    print(json.dumps({"input_preflight": preflight, "audit_relative": str(output.relative_to(arena.TRAINING_RESULT_MOUNT))}), flush=True)
    reports = {}
    for label, (record, payload) in records.items():
        config = Config(**payload["config"])
        initial, raw_manifest = load_raw_bt4_hero_model(device=torch.device("cpu"), raw_bt4_path=raw, config=config)
        trained = JointModel(config)
        manifest = load_model_checkpoint(checkpoint_dir=record["root"] / "checkpoint", model=trained)
        if manifest["state"]["sha256"] != checked[label]["state_sha256"]:
            raise ValueError("Restored checkpoint differs from preflight")
        # Check the initial encoder AND added heads against the actual recorded initialization.
        for key in ("combined_sha256", "fresh"):
            if raw_manifest[key] != payload["source"][key]:
                raise ValueError("Reconstructed initial parameters differ from recorded initialization")
        models = BT4ComparisonModels(initial.encoder, trained.encoder, dict(raw_manifest), dict(manifest),
                                     _encoder_layout(initial.encoder)[1])
        report = parameter_diff(models)
        init_named, final_named = dict(initial.named_parameters()), dict(trained.named_parameters())
        if init_named.keys() != final_named.keys():
            raise ValueError("Full-model parameter ABI differs")
        accumulators, leaves = {}, []
        for name in init_named:
            if name.startswith("encoder."):
                continue
            stats = _tensor_stats(init_named[name], final_named[name])
            group = name.split(".")[0]
            for key in ("all_added", group):
                accumulators.setdefault(key, NormAccumulator()).add(stats)
            accumulator = NormAccumulator(); accumulator.add(stats)
            leaves.append({"name": name, **accumulator.metrics()})
        report["added_heads_vs_recorded_fresh_initialization"] = {
            "groups": {name: value.metrics() for name, value in accumulators.items()}, "leaves": leaves}
        report["training_identity"] = {"label": label, **checked[label], "source": payload["source"]["combined_sha256"]}
        report["measurement_caveat"] = "Stored BF16 parameter movement, not FP32 optimizer-master movement. " \
            "This single-seed comparison has 553 versus 554 updates and a slightly contracted training set. " \
            "A larger norm does not imply better representations or stronger play."
        filename = "full_history_parameter_diff.json" if label == CANDIDATE else "hero1_parameter_diff.json"
        write_json_once(output / filename, report)
        reports[label] = {"path": filename, "sha256": arena._sha256_file(output / filename),
                          "groups": {k: report["group_metrics"][k] for k in ("all", "trunk", "policy_head")}}
        del models, initial, trained, init_named, final_named, report
        gc.collect()
    result = {"schema_version": "hh002-weight-comparison-v1", "passed": True,
              "audit_relative": str(output.relative_to(arena.TRAINING_RESULT_MOUNT)),
              "input_preflight": preflight, "reports": reports,
              "source_sha256": expected_source, "git_commit": git_commit,
              "declared_attempt_upper_bound_dollars": declared_cost,
              "seconds": time.monotonic() - started, "gpu_active": False}
    write_json_once(output / "completion.json", result)
    arena.training_result_volume.commit()
    return result


@app.local_entrypoint()
def main(recorded_spend: str = "8.95993055"):
    estimate = estimate_modal_job(spec())
    decision = enforce_modal_budget(estimate, recorded_month_spend=max(train._modal_launch_spend(), Decimal(recorded_spend)),
                                    per_job_cap=Decimal("0.20"))
    root = Path(__file__).resolve().parents[2]
    source = source_identity(root)
    print(json.dumps({"cost_guard": decision, "source_sha256": source}), flush=True)
    result = run.remote(source, arena._local_git_commit(root), decision["declared_attempt_upper_bound"])
    print(json.dumps({"audit_result": result}), flush=True)
