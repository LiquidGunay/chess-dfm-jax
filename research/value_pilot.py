"""VP-001 fresh joint Hero training; all scientific settings are explicit."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from chess_dfm_jax.data.lc0_sequential import SequentialBatches
from research import train_torch as t
from research.pre_runpod_benchmark import prepared_digest, state_hashes

ARMS = ("native", "jepa_residual")


def config(arm):
    if arm not in ARMS:
        raise ValueError("Unknown pilot arm")
    return dataclasses.replace(t.HERO_CONFIG, value_supervision_mode=arm,
        wdl_include_current_state=True, remat_bt4_blocks=True,
        remat_projector_blocks=True, remat_dfm_blocks=False,
        use_bt4_sdpa=True, use_head_sdpa=True)


def optimizer_policy():
    # Original Hero peak rates, decay and warmup; WSD stable phase at update 1000.
    return dataclasses.replace(t.LEGACY_OPTIMIZER_POLICY,
        schedule_kind="warmup_stable_linear_decay",
        wsd_decay_examples=round(t.HERO_CONFIG.lr_total_examples * .2))


def batches(root, split="train"):
    return SequentialBatches(root, split=split, batch_size=1024 if split == "train" else 64,
        horizon=8, seed=0 if split == "train" else 20000, shuffle_batches=True)


def prepare(loader, cursor, cfg):
    return t._prepare_training_step(loader, seed=0, update=cursor, data_cursor=cursor,
        batch_size=1024, config=cfg)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def append(path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True, allow_nan=False) + "\n")


def validate(model, loader, count=32):
    return t._evaluate_validation_pool(model, loader, count=count,
        seed=2026100301, device=torch.device("cuda"))


def run(arm, data, raw, output, protocol, preflight):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    cfg = config(arm)
    output.mkdir(parents=True, exist_ok=False)
    model, initialization = t.load_raw_bt4_hero_model(device=torch.device("cuda"),
        raw_bt4_path=raw, config=cfg)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("All pilot parameters must have FP32 storage")
    optimizer = t.MuonAdamW(model, cfg, policy=optimizer_policy(), examples_per_update=1024)
    loader, val_loader = batches(data), batches(data, "validation")
    common = hashlib.sha256()
    for name, p in model.named_parameters():
        if name in {"value_wdl_head.wdl_w", "value_wdl_head.wdl_b"}:
            continue
        common.update(name.encode()); common.update(p.detach().cpu().numpy().tobytes())
    partition = optimizer.partition_manifest()
    contract = {"schema_version":"vp001-fresh-joint-v1", "protocol":protocol,
        "config":dataclasses.asdict(cfg), "optimizer":dataclasses.asdict(optimizer_policy()),
        "data":loader.provenance(), "partition":partition, "arm":arm,
        "source_mapping_sha256":initialization["combined_sha256"]}
    write_json(output/"contract.json", contract)
    write_json(output/"optimizer_partition.json", partition)
    # Eager validation throughout; only training activates regional compilation.
    original = [(m, m.forward) for m in model.modules()]
    initial, seconds = validate(model, val_loader)
    append(output/"validation_metrics.jsonl", {"update":0,"examples":0,"seconds":seconds, **initial})
    compiled = t._apply_compile_regions(model, "fresh-bt4")
    compiled_forwards = [(m, m.forward) for m in model.modules()]
    run_config = {"config":dataclasses.asdict(cfg), "args":{"data_format":"lc0_sequential", "recipe":"hero",
        "batch_size":1024,"steps":1000,"seed":0}, "data":loader.provenance(),
        "input_contract":{"plane_history_mode":"lc0_full_history_as_preprocessed"},
        "compile_regions":compiled, "evaluation_compile_regions":["state_projector_blocks","dfm_blocks","jepa_transition"],
        "optimizer_policy":dataclasses.asdict(optimizer_policy()), "parameter_storage":"float32",
        "initialization":initialization, "common_initial_parameters_sha256":common.hexdigest(),
        "scientific_contract":contract}
    run_config.update(framework="torch",execution="regional-compile",torch_compile=True,
        recipe="hero",git_commit=os.environ.get("CHESS_DFM_GIT_COMMIT"))
    write_json(output/"run_config.json", run_config)
    tracker = None
    try:
        import wandb
        tracker = wandb.init(project="chess-dfm-history", group="VP-001-value-head-pilots",
            name="Fresh Hero — " + arm, id=output.name[-20:] + "-" + arm,
            dir=str(output), config={"arm":arm,"protocol":protocol,"model":dataclasses.asdict(cfg)},
            settings=wandb.Settings(disable_git=True, init_timeout=30))
        write_json(output/"wandb.json", {"url":tracker.url,"id":tracker.id})
        tracker.log({"update":0, **{"val/"+k:v for k,v in initial.items()}})
    except Exception as exc:
        write_json(output/"wandb_error.json", {"type":type(exc).__name__})
    model.train()
    began = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare, loader, 0, cfg)
        while optimizer.update < 1000:
            step_start = time.monotonic()
            cursor = optimizer.update
            item = future.result()
            digest = prepared_digest(item)
            if digest != preflight["training_hashes"][str(cursor)]:
                raise ValueError("Training input or RNG differs from CPU audit")
            future = pool.submit(prepare, loader, cursor+1, cfg) if cursor < 999 else None
            batch = t._torch_batch(item.compact_batch, torch.device("cuda"))
            if "current_wdl_target" not in batch:
                raise ValueError("Actual current target was lost")
            choices = t._choices_to_device(item.choices, torch.device("cuda"))
            loss, aux = t.loss_and_aux(model, batch, choices, compute_dtype=torch.bfloat16)
            loss.backward()
            opt = optimizer.step()
            torch.cuda.synchronize()
            if opt["optimizer_skipped_nonfinite"] or optimizer.update != cursor + 1:
                raise FloatingPointError("Nonfinite update")
            row = {**opt, **t._json_scalars(aux), "update":optimizer.update,
                "examples":optimizer.examples_seen,"loss":float(loss),"input_sha256":digest,
                "step_seconds":time.monotonic()-step_start}
            if not all(math.isfinite(v) for v in row.values() if isinstance(v, (int,float))):
                raise FloatingPointError("Nonfinite training metric")
            rows.append(row); append(output/"metrics.jsonl", row)
            if tracker and optimizer.update % 10 == 0:
                tracker.log({"update":optimizer.update, **{"train/"+k:v for k,v in row.items() if isinstance(v,(int,float))}})
            if optimizer.update % 50 == 0:
                print(json.dumps({"arm":arm,"update":optimizer.update,"loss":row["loss"],"elapsed_seconds":time.monotonic()-began}), flush=True)
            del loss, aux, batch, choices, item
            if optimizer.update in (250, 554, 1000):
                for module, forward in original:
                    module.forward = forward
                metrics, elapsed = validate(model, val_loader, 128 if optimizer.update == 1000 else 32)
                record = {"update":optimizer.update,"examples":optimizer.examples_seen,"seconds":elapsed,
                    "positions":8192 if optimizer.update == 1000 else 2048, **metrics}
                append(output/"validation_metrics.jsonl", record)
                if tracker:
                    tracker.log({"update":optimizer.update, **{"val/"+k:v for k,v in metrics.items()}})
                for module, forward in compiled_forwards:
                    module.forward = forward
                model.train()
            if optimizer.update in (554, 1000):
                t.save_training_checkpoint(output_dir=output, model=model, optimizer=optimizer,
                    source_mapping_sha256=initialization["combined_sha256"], next_data_cursor=optimizer.update,
                    resume_contract=contract)
    checkpoint = t.save_model_checkpoint(output_dir=output, model=model,
        source_mapping_sha256=initialization["combined_sha256"], optimizer_update=1000, data_cursor=1000)
    report = {"status":"complete","arm":arm,"updates":1000,"examples":1024000,
        "checkpoint":checkpoint,"wall_seconds":time.monotonic()-began,
        "last_metrics":rows[-1],"mean_last50":{k:float(np.mean([r[k] for r in rows[-50:]])) for k,v in rows[-1].items() if isinstance(v,(int,float))},
        "initial_validation":initial,"validation":metrics,"final_state_hashes":state_hashes(model,optimizer),
        "common_initial_parameters_sha256":common.hexdigest()}
    write_json(output/"report.json", report)
    if tracker:
        tracker.finish()
    print(json.dumps({"complete":arm,"state_sha256":checkpoint["state"]["sha256"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("arm","data","raw","output","protocol","preflight"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    run(a.arm, Path(a.data), Path(a.raw), Path(a.output),
        json.loads(Path(a.protocol).read_text()), json.loads(Path(a.preflight).read_text()))
