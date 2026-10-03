"""GPU-002: fixed-weight, full-history precision/compilation diagnostic.

No optimizer steps or retained model writes. The six-cell arithmetic matrix
uses FP32 parameter/gradient storage; the production control preserves the
checkpoint's native mixed storage. These are forward/backward timings, not
end-to-end training throughput or playing-strength measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


ARMS = (
    {"id": "fp32_eager", "dtype": "float32", "compile": "none", "storage": "float32", "timeout": 240},
    {"id": "bf16_eager", "dtype": "bfloat16", "compile": "none", "storage": "float32", "timeout": 240},
    {"id": "fp16_eager", "dtype": "float16", "compile": "none", "storage": "float32", "timeout": 240},
    {"id": "fp32_compiled", "dtype": "float32", "compile": "fresh-bt4", "storage": "float32", "timeout": 900},
    {"id": "bf16_compiled", "dtype": "bfloat16", "compile": "fresh-bt4", "storage": "float32", "timeout": 900},
    {"id": "fp16_compiled", "dtype": "float16", "compile": "fresh-bt4", "storage": "float32", "timeout": 900},
    {"id": "production_control", "dtype": "bfloat16", "compile": "fresh", "storage": "checkpoint", "timeout": 360},
)


def config():
    from research.train_torch import HERO_CONFIG
    return dataclasses.replace(HERO_CONFIG, remat_bt4_blocks=True,
        remat_projector_blocks=True, remat_dfm_blocks=False,
        use_bt4_sdpa=True, use_head_sdpa=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def tensor_stats(reference, candidate):
    """Accumulate on CPU in float64, including explicit nonfinite failures."""
    import torch
    a, b = reference.detach().cpu().double(), candidate.detach().cpu().double()
    if a.shape != b.shape:
        raise ValueError("Tensor shape mismatch")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"finite": False, "elements": a.numel()}
    delta = b - a
    return {"finite": True, "elements": a.numel(), "dot": float((a*b).sum()),
            "reference_squared": float(a.square().sum()),
            "candidate_squared": float(b.square().sum()),
            "difference_squared": float(delta.square().sum()),
            "max_absolute": float(delta.abs().max()) if a.numel() else 0.0}


def finalize_stats(stats):
    if not stats["finite"]:
        return {**stats, "relative_l2": None, "cosine": None}
    a = math.sqrt(stats["reference_squared"])
    b = math.sqrt(stats["candidate_squared"])
    return {**stats, "relative_l2": math.sqrt(stats["difference_squared"])/max(a, 1e-30),
            "cosine": max(-1.0, min(1.0, stats["dot"]/(a*b))) if a*b > 0 else None}


def merge_stats(left, right):
    if left is None:
        return dict(right)
    if not left["finite"] or not right["finite"]:
        return {"finite": False, "elements": left["elements"]+right["elements"]}
    return {key: (max(left[key], right[key]) if key == "max_absolute" else
                  True if key == "finite" else left[key]+right[key]) for key in left}


def prepared_batch(protocol, data, cursor):
    import torch
    from research.hero_history_batches import HistoryTrajectoryBatches
    from research.input_contract import audit_full_history_batch
    from research.train_torch import canonicalize_trajectory_batch, _prepare_training_step
    batches = HistoryTrajectoryBatches(data, split="train", batch_size=protocol["batch_size"],
        horizon=8, manifest_sha256=protocol["dataset_manifest_sha256"],
        canonicalize=canonicalize_trajectory_batch)
    audit = audit_full_history_batch(batches.batch_at(cursor), allow_chess960_initial=True)
    prepared = _prepare_training_step(batches, seed=protocol["seed"], update=501,
        data_cursor=cursor, batch_size=protocol["batch_size"], config=config())
    digest = hashlib.sha256()
    for name, array in sorted(prepared.compact_batch.items()):
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    for name, value in vars(prepared.choices).items() if hasattr(prepared.choices, "__dict__") else prepared.choices._asdict().items():
        if isinstance(value, torch.Tensor):
            digest.update(name.encode()); digest.update(value.numpy().tobytes())
    return prepared, audit, digest.hexdigest()


def restore_model(checkpoint, arm):
    import torch
    from safetensors import safe_open
    from research.train_torch import JointModel
    model = JointModel(config(), bt4_norm_impl="eager")
    manifest = json.loads((checkpoint/"manifest.json").read_text())
    with safe_open(str(checkpoint/manifest["state"]["path"]), framework="pt", device="cpu") as state:
        expected = {"model."+name for name, _ in model.named_parameters()}
        if {key for key in state.keys() if key.startswith("model.")} != expected:
            raise ValueError("Checkpoint model leaf mismatch")
        for name, parameter in model.named_parameters():
            value = state.get_tensor("model."+name)
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("Checkpoint model ABI mismatch: "+name)
            with torch.no_grad(): parameter.copy_(value)
    if arm["storage"] == "float32": model.float()
    return model.to("cuda").train()


def gradient_comparison(model, scale, reference_path, save_reference=False):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from research.train_torch import _gradient_audit_group
    names = {name for name, p in model.named_parameters() if p.grad is not None}
    finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    if not finite:
        return {"finite": False, "groups": {}, "gradient_leaf_count": len(names)}
    if save_reference:
        save_file({name: p.grad.detach().float().cpu().contiguous()/scale
                   for name, p in model.named_parameters() if p.grad is not None}, str(reference_path))
    groups = {}
    with safe_open(str(reference_path), framework="pt", device="cpu") as reference:
        if set(reference.keys()) != names:
            raise ValueError("Gradient leaf sets differ")
        for name, p in model.named_parameters():
            if p.grad is None: continue
            stats = tensor_stats(reference.get_tensor(name), p.grad.detach().float().cpu()/scale)
            for group in ("all", _gradient_audit_group(name)):
                groups[group] = merge_stats(groups.get(group), stats)
    return {"finite": True, "gradient_leaf_count": len(names),
            "groups": {key: finalize_stats(value) for key, value in groups.items()}}


def forward_backward(model, batch, choices, dtype, scale, *, capture=None):
    import torch
    from research.train_torch import loss_and_aux
    model.zero_grad(set_to_none=True)
    start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    torch.cuda.synchronize(); began = time.perf_counter(); start.record()
    loss, aux = loss_and_aux(model, batch, choices, compute_dtype=dtype, capture=capture)
    middle.record(); (loss*scale).backward(); end.record(); torch.cuda.synchronize()
    result = {"wall_seconds": time.perf_counter()-began,
              "forward_seconds": start.elapsed_time(middle)/1000,
              "backward_seconds": middle.elapsed_time(end)/1000}
    return loss, aux, result


def trace_current(model, batch, dtype):
    """Mirror encode_current without hooks or a new grad-mode specialization."""
    from torch.utils.checkpoint import checkpoint
    encoder = model.encoder
    current = encoder.embedding(batch["current_planes"], encoder.alpha, dtype)
    captured = {"embedding": current.detach().float().cpu()}
    for i, layer in enumerate(encoder.layers):
        current = checkpoint(layer, current, encoder.alpha, dtype, use_reentrant=False)
        captured[f"block_{i:02d}"] = current.detach().float().cpu()
    captured["native_policy"] = encoder.policy_head(current, dtype).detach().float().cpu()
    return captured


def run_arm(protocol, arm, data, checkpoint, scratch, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from research.train_torch import _torch_batch, _choices_to_device, _json_scalars, _legal_root_actions
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    # Strict FP32 oracle; all matrix cells share reduction settings. Production
    # control separately restores the existing runtime's high/TF32 settings.
    production = arm["id"] == "production_control"
    torch.set_float32_matmul_precision("high" if production else "highest")
    torch.backends.cuda.matmul.allow_tf32 = production
    torch.backends.cudnn.allow_tf32 = production
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = production
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = production
    dtype = getattr(torch, arm["dtype"])
    scale = 128.0 if dtype == torch.float16 else 1.0
    model = restore_model(checkpoint, arm)
    from research.train_torch import _apply_compile_regions
    compiled = _apply_compile_regions(model, arm["compile"])
    storage = Counter()
    for parameter in model.parameters(): storage[str(parameter.dtype)] += parameter.numel()*parameter.element_size()
    result = {"arm": arm, "torch_version": torch.__version__, "device": torch.cuda.get_device_name(),
              "parameter_bytes_by_dtype": dict(storage), "loss_scale": scale,
              "tf32": production, "reduced_precision_reductions": production,
              "compiled_regions": compiled, "optimizer_steps": 0, "batches": []}
    reference = arm["id"] == "fp32_eager"
    for index, cursor in enumerate(protocol["batch_cursors"]):
        prepared, audit, digest = prepared_batch(protocol, data, cursor)
        batch, choices = _torch_batch(prepared.compact_batch, torch.device("cuda")), _choices_to_device(prepared.choices, torch.device("cuda"))
        capture = {}
        torch.cuda.reset_peak_memory_stats()
        loss, aux, elapsed = forward_backward(model, batch, choices, dtype, scale, capture=capture)
        finite_loss = bool(torch.isfinite(loss))
        gradients = gradient_comparison(model, scale, scratch/f"grad_{index}.safetensors", save_reference=reference)
        logits = capture["root_logits"].float().cpu()
        scalars = {"loss": float(loss.detach().cpu()), **_json_scalars(aux)}
        scalars_finite = all(math.isfinite(value) for value in scalars.values())
        record = {"cursor": cursor, "input_sha256": digest, "history_audit": audit,
                  "timing": elapsed, "scalars": {k: v if math.isfinite(v) else None for k,v in scalars.items()},
                  "gradients_vs_fp32_eager": gradients,
                  "finite": finite_loss and gradients["finite"] and scalars_finite and bool(torch.isfinite(logits).all()),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        if index == 0 and record["finite"] and arm["id"] != "production_control":
            eager_path = scratch/(arm["dtype"]+"_eager_grad.safetensors")
            record["gradients_vs_same_dtype_eager"] = gradient_comparison(
                model, scale, eager_path, save_reference=arm["compile"] == "none")
        if reference:
            save_file({"root_logits": logits}, str(scratch/f"logits_{index}.safetensors"))
            write_json(scratch/f"input_{index}.json", {"sha256": digest})
        if json.loads((scratch/f"input_{index}.json").read_text())["sha256"] != digest:
            raise ValueError("Matched batch/RNG identity changed")
        with safe_open(str(scratch/f"logits_{index}.safetensors"), framework="pt", device="cpu") as handle:
            ref_logits = handle.get_tensor("root_logits")
        record["root_logits_vs_fp32_eager"] = finalize_stats(tensor_stats(ref_logits, logits))
        a, valid = _legal_root_actions(ref_logits, prepared.compact_batch)
        b, other_valid = _legal_root_actions(logits, prepared.compact_batch)
        if not torch.equal(valid, other_valid): raise ValueError("Legal mask mismatch")
        record["legal_top1_agreement"] = float(((a == b) & valid).sum()/valid.sum().clamp_min(1)) if record["finite"] else None
        result["batches"].append(record)
        del loss, aux, capture
        if not record["finite"]:
            result["status"] = "nonfinite"
            write_json(output/"result.json", result)
            return result
        if index == 0:
            result["cold_first_forward_backward_seconds"] = elapsed["wall_seconds"]
            # Warm and measured work: constant weights, fixed batch and choices.
            for _ in range(protocol["warmup_repeats"]):
                warm_loss, warm_aux, _timing = forward_backward(model, batch, choices, dtype, scale)
                del warm_loss, warm_aux
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(protocol["measured_repeats"]):
                warm_loss, warm_aux, timing = forward_backward(model, batch, choices, dtype, scale)
                timings.append(timing)
                del warm_loss, warm_aux
            result["warm"] = {"repeats": timings, "mean_seconds": statistics.mean(t["wall_seconds"] for t in timings),
                "median_seconds": statistics.median(t["wall_seconds"] for t in timings),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            result["cold_minus_warm_seconds"] = elapsed["wall_seconds"]-result["warm"]["median_seconds"]
            model.zero_grad(set_to_none=True)
            trace = trace_current(model, batch, dtype)
            if reference: save_file(trace, str(scratch/"trace.safetensors"))
            with safe_open(str(scratch/"trace.safetensors"), framework="pt", device="cpu") as handle:
                result["layer_trace_vs_fp32_eager"] = {key: finalize_stats(tensor_stats(handle.get_tensor(key), value)) for key,value in trace.items()}
            a, valid = _legal_root_actions(trace["native_policy"], prepared.compact_batch)
            with safe_open(str(scratch/"trace.safetensors"), framework="pt", device="cpu") as handle:
                b, _ = _legal_root_actions(handle.get_tensor("native_policy"), prepared.compact_batch)
            result["native_policy_legal_top1_agreement"] = float(((a == b) & valid).sum()/valid.sum().clamp_min(1))
            # Full traces are compared above; only an eight-position slice is
            # retained to keep permanent diagnostic storage small.
            save_file({"root_logits": logits, **{k: v if k == "native_policy" else v[:8].contiguous()
                      for k,v in trace.items()}}, str(output/"diagnostic_tensors.safetensors"))
            del trace
        del batch, choices, prepared
        print(json.dumps({"arm": arm["id"], "batch": index, "finite": record["finite"],
                          "loss": record["scalars"]["loss"]}), flush=True)
    result["status"] = "complete"
    result["compile_counters"] = {k: dict(v) for k,v in torch._dynamo.utils.counters.items()}
    result["compiler_timing_report"] = torch._dynamo.utils.compile_times()
    write_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "arm", "data", "checkpoint", "scratch", "output"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    arm = next(a for a in ARMS if a["id"] == args.arm)
    run_arm(protocol, arm, Path(args.data), Path(args.checkpoint), Path(args.scratch), Path(args.output))


if __name__ == "__main__":
    main()
