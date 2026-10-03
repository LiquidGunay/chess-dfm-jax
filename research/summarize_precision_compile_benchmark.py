"""Summarize GPU-002 without equating numerical agreement with chess strength."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from research.precision_compile_benchmark import finalize_stats, tensor_stats, write_json


def summarize_arm(result, batch_size):
    batches = result["batches"]
    if result["status"] != "complete":
        return {"status": result["status"], "completed_batches": len(batches)}
    if not batches or any(not b["finite"] for b in batches):
        raise ValueError("Complete arm contains nonfinite or missing diagnostics")
    if result["optimizer_steps"] != 0:
        raise ValueError("Fixed-weight benchmark mutated optimizer")
    times = [x["wall_seconds"] for x in result["warm"]["repeats"]]
    if min(times) <= 0: raise ValueError("Nonpositive timing")
    elapsed = statistics.mean(times)
    if abs(elapsed-result["warm"]["mean_seconds"]) > 1e-9:
        raise ValueError("Warm timing summary mismatch")
    row = {"status": "complete", "cold_first_forward_backward_seconds": result["cold_first_forward_backward_seconds"],
        "cold_minus_warm_seconds": result["cold_minus_warm_seconds"],
        "mean_forward_backward_seconds": elapsed, "examples_per_second_forward_backward_only": batch_size/elapsed,
        "warm_min_seconds": min(times), "warm_max_seconds": max(times),
        "peak_allocated_gib": result["warm"]["peak_allocated_bytes"]/1024**3,
        "parameter_bytes_by_dtype": result["parameter_bytes_by_dtype"],
        "mean_training_batch_loss": statistics.mean(b["scalars"]["loss"] for b in batches),
        "root_legal_top1_agreement_to_fp32_eager": statistics.mean(b["legal_top1_agreement"] for b in batches),
        "native_policy_legal_top1_agreement_to_fp32_eager_first_batch": result["native_policy_legal_top1_agreement"],
        "first_batch_gradient_cosine_to_fp32_eager": batches[0]["gradients_vs_fp32_eager"]["groups"]["all"]["cosine"],
        "first_batch_gradient_relative_l2_to_fp32_eager": batches[0]["gradients_vs_fp32_eager"]["groups"]["all"]["relative_l2"],
        "first_batch_gradient_groups_to_fp32_eager": {k:{"cosine":v["cosine"], "relative_l2":v["relative_l2"]}
            for k,v in batches[0]["gradients_vs_fp32_eager"]["groups"].items()},
        "layer_trace_vs_fp32_eager": {k:{"relative_l2":v["relative_l2"], "cosine":v["cosine"]}
                                      for k,v in result["layer_trace_vs_fp32_eager"].items()}}
    same_dtype = batches[0].get("gradients_vs_same_dtype_eager")
    if same_dtype:
        row["first_batch_gradient_cosine_to_same_dtype_eager"] = same_dtype["groups"]["all"]["cosine"]
    return row


def build_summary(root):
    import torch
    from safetensors.torch import load_file
    # Fix CPU reduction ordering to the two-thread collection analysis. This
    # keeps retained float64 tensor comparisons reproducible across hosts.
    torch.set_num_threads(2)
    collected = json.loads((root/"completion.json").read_text())
    for name, identity in collected["artifacts"].items():
        path = root/name
        if path.stat().st_size != identity["size_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("Collected artifact changed: "+name)
    protocol = json.loads((root/"preregister.json").read_text())
    completed = json.loads((root/"sweep.json").read_text())
    rows, documents = {}, {}
    for record in completed["records"]:
        arm = record["arm"]
        path = root/arm/"result.json"
        if path.exists():
            result = json.loads(path.read_text())
            expected_arm = next(a for a in protocol["arms"] if a["id"] == arm)
            if result["arm"] != expected_arm: raise ValueError("Arm drift")
            expected_cursors = protocol["batch_cursors"][:len(result["batches"])]
            if [b["cursor"] for b in result["batches"]] != expected_cursors:
                raise ValueError("Batch selection differs")
            if result["status"] == "complete" and len(result["batches"]) != len(protocol["batch_cursors"]):
                raise ValueError("Complete arm missing a batch")
            rows[arm] = summarize_arm(result, protocol["batch_size"])
            documents[arm] = result
        else:
            rows[arm] = {"status": record["process_status"], "returncode": record["returncode"]}
    comparisons = {}
    for precision in ("fp32", "bf16", "fp16"):
        eager, compiled = precision+"_eager", precision+"_compiled"
        if any(rows.get(a, {}).get("status") != "complete" for a in (eager, compiled)): continue
        a, b = rows[eager], rows[compiled]
        time_saved = a["mean_forward_backward_seconds"]-b["mean_forward_backward_seconds"]
        warm_difference = b["cold_minus_warm_seconds"]-a["cold_minus_warm_seconds"]
        comparisons[precision] = {"warm_speedup": a["mean_forward_backward_seconds"]/b["mean_forward_backward_seconds"],
            "cold_extra_seconds": warm_difference,
            "approximate_break_even_repeats": max(0, warm_difference)/time_saved if time_saved > 0 else None,
            "gradient_cosine_compiled_vs_eager": b["first_batch_gradient_cosine_to_same_dtype_eager"],
            "compiled_closer_gradient_to_fp32_eager": b["first_batch_gradient_relative_l2_to_fp32_eager"] < a["first_batch_gradient_relative_l2_to_fp32_eager"]}
        x = load_file(str(root/eager/"diagnostic_tensors.safetensors"))
        y = load_file(str(root/compiled/"diagnostic_tensors.safetensors"))
        comparisons[precision]["retained_first_batch_tensors_compiled_vs_eager"] = {
            key: finalize_stats(tensor_stats(x[key], y[key])) for key in x}
    return {"experiment_id": "GPU-002", "all_complete": completed["all_complete"],
        "batch_size": protocol["batch_size"], "diagnostic_positions": protocol["batch_size"]*len(protocol["batch_cursors"]),
        "arms": rows, "compilation_comparisons": comparisons,
        "limits": ["Single L40S, fixed order, three training batches, ten fixed-weight measured repeats; no fleet-level inference.",
            "Timings exclude optimizer, data, transfers, checkpointing and validation; batch 64 is not production batch 1024.",
            "Cold-minus-warm estimates initialization overhead, not compiler-only time; full compiler timing reports are retained per arm.",
            "Main matrix has FP32 parameter storage. Production control has checkpoint-native mixed storage and its own TF32/reduction settings.",
            "FP32 reference starts from stored checkpoint values, so cannot recover original unquantized weights.",
            "Loss is composite training loss, not held-out validation or playing strength. Numerical closeness does not determine Elo.",
            "Native policy and layer diagnostics cover the first 64 positions; cross-arm retained layer tensor slices cover eight.",
            "Dtype can change SDPA kernel dispatch. Separate full-step production-batch testing is required before adopting any optimization."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = build_summary(args.root)
    write_json(args.root/"summary.json", summary)
    print(json.dumps({"all_complete": summary["all_complete"], "arms": {
        k: {n: v for n,v in row.items() if n != "layer_trace_vs_fp32_eager"} for k,row in summary["arms"].items()}}, indent=2))
