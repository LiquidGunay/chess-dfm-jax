"""GPU-003: disposable full-batch runtime forks, not production continuation.

The parent is restored strictly before any explicit precision/runtime changes.
Checkpoint experiments use their own contract; the production trainer and its
exact-resume checks are unchanged. Only compact evidence leaves the container.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from research.precision_compile_benchmark import config as parent_config, write_json


def arm(name, *, storage="checkpoint", dtype="bfloat16", compile="fresh",
        norm="eager", prefetch="step-start", remat=True, timeout=480):
    return dict(id=name, storage=storage, dtype=dtype, compile=compile, norm=norm,
                prefetch=prefetch, remat_bt4=remat, timeout=timeout)


ARMS = (
    arm("production_control"),
    arm("fp32_storage_control", storage="float32"),
    arm("bf16_compiled", storage="float32", compile="fresh-bt4", timeout=800),
    arm("fp16_compiled", storage="float32", dtype="float16", compile="fresh-bt4", timeout=800),
    arm("fused_norm", norm="eager-fused-backward"),
    arm("delayed_prefetch", prefetch="after-forward"),
    arm("fused_delayed", norm="eager-fused-backward", prefetch="after-forward"),
    arm("fp16_less_remat", storage="float32", dtype="float16", compile="fresh-bt4",
        remat=False, timeout=800),
)


def batches(protocol, data, split="train"):
    from research.hero_history_batches import HistoryTrajectoryBatches
    from research.train_torch import canonicalize_trajectory_batch
    return HistoryTrajectoryBatches(data, split=split, batch_size=1024 if split == "train" else 64,
        horizon=8, manifest_sha256=protocol["dataset_manifest_sha256"],
        canonicalize=canonicalize_trajectory_batch)


def prepared_digest(prepared):
    digest = hashlib.sha256()
    for name, array in sorted(prepared.compact_batch.items()):
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    for name, value in prepared.choices._asdict().items():
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare(loader, cursor):
    from research.train_torch import _prepare_training_step
    item = _prepare_training_step(loader, seed=0, update=cursor, data_cursor=cursor,
                                  batch_size=1024, config=parent_config())
    return item, prepared_digest(item)


def promote_state(model, optimizer):
    """Exact BF16->FP32 promotion, including moments, without resetting progress."""
    import torch
    from research.train_torch import MuonAdamW
    model.float()
    promoted = MuonAdamW(model, optimizer.config, policy=optimizer.policy, examples_per_update=1024)
    with torch.no_grad():
        for old, new in zip(optimizer.leaves, promoted.leaves, strict=True):
            if old.name != new.name or old.master_parameter is not None:
                raise ValueError("Unexpected parent optimizer ABI")
            new.first_moment.copy_(old.first_moment.float())
            if new.second_moment is not None:
                new.second_moment.copy_(old.second_moment.float())
    promoted.update, promoted.examples_seen = optimizer.update, optimizer.examples_seen
    return promoted


def child_contract(protocol, selected, config, partition, parent):
    return {"schema_version": "gpu003-disposable-fork-v1", "protocol": protocol,
        "arm": selected, "config": dataclasses.asdict(config), "optimizer_partition": partition,
        "parent_state_sha256": parent["state"]["sha256"],
        "parent_contract_sha256": parent["resume_contract_sha256"],
        "compute_dtype": selected["dtype"], "seed": 0, "batch_size": 1024,
        "rng": "trainer materialize_step_choices(seed=0, update=absolute pre-step counter); no global RNG consumption",
        "transition": "Strict parent restore, then declared runtime/storage fork. Original cosine schedule retained for this systems comparison."}


def state_hashes(model, optimizer):
    import torch
    groups = {"model": hashlib.sha256(), "optimizer": hashlib.sha256()}
    def add(group, name, tensor):
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()): raise FloatingPointError("Nonfinite final state: "+name)
        digest = groups[group]
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    for name, value in model.named_parameters(): add("model", name, value)
    for leaf in optimizer.leaves:
        add("optimizer", "first."+leaf.name, leaf.first_moment)
        if leaf.second_moment is not None: add("optimizer", "second."+leaf.name, leaf.second_moment)
        if leaf.master_parameter is not None: add("optimizer", "master."+leaf.name, leaf.master_parameter)
    return {**{key: value.hexdigest() for key, value in groups.items()},
            "update": optimizer.update, "examples_seen": optimizer.examples_seen}


def summarize(records):
    if [r["update"] for r in records] != list(range(501, 551)):
        raise ValueError("Expected fifty matched updates")
    if any(r["optimizer_skipped_nonfinite"] or not math.isfinite(r["loss"]) for r in records):
        raise ValueError("Nonfinite training")
    by_update = {r["update"]: r for r in records}
    window = [by_update[i] for i in range(511, 541)]
    durations = [r["end_to_end_seconds"] for r in window]
    return {"measured_updates": [511, 540], "updates": 30,
        "mean_seconds": statistics.mean(durations), "median_seconds": statistics.median(durations),
        "examples_per_second": 1024/statistics.mean(durations),
        "mean_forward_seconds": statistics.mean(r["forward_seconds"] for r in window),
        "mean_backward_seconds": statistics.mean(r["backward_seconds"] for r in window),
        "mean_optimizer_seconds": statistics.mean(r["optimizer_seconds"] for r in window),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in window),
        "scope": "Full batch 1024, CPU prepare/prefetch wait, transfer, forward/backward, unscale, optimizer, scalar/log writes. Excludes cold ten updates, checkpoint/hash audits, validation and initialization."}


def choose_candidate(results, minimum_speedup=1.10, max_ce_ratio=1.02):
    baseline = results.get("production_control")
    if not baseline or baseline.get("status") != "complete": return None
    ce_keys = ("root_legal_conditional_ce", "base_policy_root_legal_conditional_ce")
    eligible = []
    for name, row in results.items():
        if name == "production_control" or row.get("status") != "complete": continue
        speedup = baseline["timing"]["mean_seconds"]/row["timing"]["mean_seconds"]
        if speedup < minimum_speedup: continue
        if any(row["validation"][key] > baseline["validation"][key]*max_ce_ratio for key in ce_keys): continue
        eligible.append((row["timing"]["mean_seconds"], name))
    return min(eligible)[1] if eligible else None


def common_validation(model, protocol, data):
    import torch
    from research.train_torch import _evaluate_validation_pool
    return _evaluate_validation_pool(model, batches(protocol, data, "validation"),
        count=16, seed=2026072501, device=torch.device("cuda"))


def run(protocol, selected, data, checkpoint, scratch, output, preflight, resume=None):
    import torch
    from research import train_torch as trainer
    torch.set_num_threads(2); torch.manual_seed(0)
    # All arms use the production reduction settings, isolating storage and
    # compute in the three FP32-storage cells. Not GPU-002's strict oracle.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    cfg = dataclasses.replace(parent_config(), remat_bt4_blocks=selected["remat_bt4"])
    parent = json.loads((checkpoint/"manifest.json").read_text())
    model = trainer.JointModel(parent_config(), bt4_norm_impl="eager")
    optimizer = trainer.MuonAdamW(model, parent_config(), examples_per_update=1024)
    trainer.load_training_checkpoint(checkpoint_dir=checkpoint, model=model, optimizer=optimizer,
        expected_source_mapping_sha256=parent["source_mapping_sha256"],
        expected_resume_contract=parent["resume_contract"])
    if optimizer.update != 500 or optimizer.examples_seen != 512000:
        raise ValueError("Wrong parent counters")
    if selected["storage"] == "float32": optimizer = promote_state(model, optimizer)
    # Move existing parameter objects/moment tensors together after CPU restore.
    model.to("cuda")
    for leaf in optimizer.leaves:
        leaf.first_moment = leaf.first_moment.to("cuda")
        if leaf.second_moment is not None: leaf.second_moment = leaf.second_moment.to("cuda")
    named = dict(model.named_parameters())
    if any(named[leaf.name] is not leaf.parameter for leaf in optimizer.leaves):
        raise ValueError("Device move changed optimizer parameter identities")
    model.config = cfg
    # LayerNorm dispatch is runtime-only and does not change the state ABI.
    for module in model.modules():
        if isinstance(module, trainer.RawLayerNorm): module.implementation = selected["norm"]
    optimizer.config = cfg
    contract = child_contract(protocol, selected, cfg, optimizer.partition_manifest(), parent)
    write_json(output/"fork.json", contract)
    if resume:
        trainer.load_training_checkpoint(checkpoint_dir=resume, model=model, optimizer=optimizer,
            expected_source_mapping_sha256=parent["source_mapping_sha256"], expected_resume_contract=contract)
        if optimizer.update != 545: raise ValueError("Wrong interruption checkpoint")
    initial_validation = None
    if selected["id"] == "production_control" and not resume:
        initial_validation, _ = common_validation(model, protocol, data)
    model.train()
    original_forwards = [(module, module.forward) for module in model.modules()]
    compiled = trainer._apply_compile_regions(model, selected["compile"])
    dtype = getattr(torch, selected["dtype"])
    scale = 128.0 if dtype == torch.float16 else 1.0
    loader = batches(protocol, data)
    records = []
    start_update = optimizer.update
    checkpoint_manifest = None
    rounded_proxy = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare, loader, optimizer.update)
        while optimizer.update < 550:
            if optimizer.update == 510: torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            cursor = optimizer.update
            item, digest = future.result()
            wait_seconds = time.perf_counter()-begin
            if item.update != cursor or item.data_cursor != cursor or digest != preflight["training_hashes"][str(cursor)]:
                raise ValueError("Input/RNG identity differs from CPU verification")
            future = pool.submit(prepare, loader, cursor+1) if cursor < 549 and selected["prefetch"] == "step-start" else None
            batch = trainer._torch_batch(item.compact_batch, torch.device("cuda"))
            choices = trainer._choices_to_device(item.choices, torch.device("cuda"))
            torch.cuda.synchronize()
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
            loss, aux = trainer.loss_and_aux(model, batch, choices, compute_dtype=dtype)
            events[1].record()
            if cursor < 549 and selected["prefetch"] == "after-forward": future = pool.submit(prepare, loader, cursor+1)
            (loss*scale).backward()
            if scale != 1:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None: p.grad.div_(scale)
            events[2].record()
            # Observational proxy, outside measured window. Nonzero gradient is
            # not proof that a nonzero optimizer update was rounded away.
            before = {}
            if cursor == 500:
                before = {n: (p.detach().cpu().clone(), p.grad.detach().cpu().ne(0))
                    for n,p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None}
            opt = optimizer.step()
            events[3].record(); torch.cuda.synchronize()
            if opt["optimizer_skipped_nonfinite"] or optimizer.update != cursor+1:
                raise FloatingPointError("Nonfinite optimizer step")
            scalars = trainer._json_scalars(aux)
            if not math.isfinite(float(loss)) or not all(math.isfinite(x) for x in scalars.values()):
                raise FloatingPointError("Nonfinite loss metric")
            row = {"update": optimizer.update, "loss": float(loss), "input_sha256": digest,
                "data_wait_seconds": wait_seconds, "prepare_seconds": item.prepare_seconds,
                "forward_seconds": events[0].elapsed_time(events[1])/1000,
                "backward_seconds": events[1].elapsed_time(events[2])/1000,
                "optimizer_seconds": events[2].elapsed_time(events[3])/1000,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), **scalars, **opt}
            # The optimizer metric names the pre-step counter; completed update
            # above remains the trainer's absolute counter.
            with (output/"metrics.jsonl").open("a") as handle: handle.write(json.dumps(row, allow_nan=False)+"\n")
            row["end_to_end_seconds"] = time.perf_counter()-begin
            records.append(row)
            if before:
                proxy = defaultdict(lambda: [0, 0])
                for n,p in model.named_parameters():
                    if n not in before: continue
                    old, nonzero = before[n]
                    group = ".".join(n.split(".")[:3]) if n.startswith("encoder.layers.") else ".".join(n.split(".")[:2])
                    proxy[group][0] += int(((p.detach().cpu() == old) & nonzero).sum())
                    proxy[group][1] += int(nonzero.sum())
                rounded_proxy = {k: {"unchanged_with_nonzero_gradient": v[0], "nonzero_gradient_coordinates": v[1],
                    "fraction": v[0]/v[1] if v[1] else None} for k,v in proxy.items()}
                del before
            if optimizer.update == 545 and not resume:
                checkpoint_manifest = trainer.save_training_checkpoint(output_dir=scratch/selected["id"], model=model,
                    optimizer=optimizer, source_mapping_sha256=parent["source_mapping_sha256"],
                    next_data_cursor=optimizer.update, resume_contract=contract)
            if optimizer.update % 10 == 0: print(json.dumps({"arm": selected["id"], "update": optimizer.update, "loss": row["loss"]}), flush=True)
            del batch, choices, item, loss, aux
    final_hashes = state_hashes(model, optimizer)
    # One common eager BF16 evaluator, not separate compiled/dtype evaluators.
    for module, forward in original_forwards: module.forward = forward
    for module in model.modules():
        if isinstance(module, trainer.RawLayerNorm): module.implementation = "eager"
    validation, validation_seconds = common_validation(model, protocol, data)
    result = {"status": "complete", "arm": selected, "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(), "compiled_regions": compiled, "loss_scale": scale,
        "start_update": start_update, "end_update": optimizer.update, "records": records,
        "timing": summarize(records) if not resume else None,
        "initial_validation": initial_validation, "validation": validation, "validation_seconds": validation_seconds,
        "final_state_hashes": final_hashes, "recovery_checkpoint": checkpoint_manifest,
        "unchanged_weight_proxy_update501": rounded_proxy,
        "precision_note": "FP32-storage arms also promote optimizer moments; this is an explicit optimizer-state change, not a runtime-only optimization. All use production TF32/reduction settings."}
    write_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "preflight", "data", "checkpoint", "scratch", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--arm", choices=[a["id"] for a in ARMS], required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    selected = next(a for a in ARMS if a["id"] == args.arm)
    try:
        run(json.loads(args.protocol.read_text()), selected, args.data, args.checkpoint,
            args.scratch, args.output, json.loads(args.preflight.read_text()), args.resume)
    except Exception as exc:
        write_json(args.output/"failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__": main()
