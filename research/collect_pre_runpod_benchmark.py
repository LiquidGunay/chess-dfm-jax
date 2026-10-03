"""Read-only collection and recomputation of terminal GPU-003 evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import modal
from research.collect_gpu_training_benchmark import preserve_bytes
from research.pre_runpod_benchmark import choose_candidate, summarize


def collect(root):
    protocol = json.loads((root/"preregister.json").read_text())
    submitted = json.loads((root/"sweep_run.json").read_text())
    try:
        complete = modal.FunctionCall.from_id(submitted["call_id"]).get(timeout=0)
    except TimeoutError:
        print(json.dumps({"status": "running", "call_id": submitted["call_id"]}), flush=True)
        return
    volume = modal.Volume.from_name("chess-dfm-training-results", environment_name="chess-dfm-research")
    pre = json.loads((root/"preflight.json").read_text())
    remote = complete["output_relative"]
    if remote != pre["output_relative"]: raise ValueError("Output identity differs")
    artifacts, results, replays = {}, {}, {}
    def fetch(name):
        raw = b"".join(volume.read_file(remote+"/"+name))
        preserve_bytes(root/name, raw)
        artifacts[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        return raw
    if json.loads(fetch("launch.json"))["protocol"] != protocol: raise ValueError("Protocol differs")
    if json.loads(fetch("sweep.json")) != complete: raise ValueError("Completion differs")
    allowed = {a["id"] for a in protocol["arms"]}
    for record in complete["records"]:
        name = record["arm"]
        base = name.removesuffix("_resume")
        if base not in allowed: raise ValueError("Unknown arm")
        for filename in ("launch.json", "receipt.json", "stdout.log"): fetch(name+"/"+filename)
        if record["process_status"] != "complete":
            for filename in ("failure.json", "fork.json"):
                try: fetch(name+"/"+filename)
                except FileNotFoundError: pass  # Timeout can precede these writes.
        if record["result_status"] is not None:
            result = json.loads(fetch(name+"/result.json"))
            fetch(name+"/fork.json"); fetch(name+"/metrics.jsonl")
            if result["end_update"] != 550: raise ValueError("Wrong terminal update")
            expected = range(545 if name.endswith("_resume") else 500, 550)
            if [r["input_sha256"] for r in result["records"]] != [pre["training_hashes"][str(i)] for i in expected]:
                raise ValueError("Input verification differs")
            if not name.endswith("_resume"):
                if summarize(result["records"]) != result["timing"]: raise ValueError("Timing recomputation differs")
                results[name] = result
            else:
                replays[base] = result
    if choose_candidate(results) != complete["candidate"]: raise ValueError("Selection recomputation differs")
    replay_diagnostics = None
    if complete["resume"]:
        target = complete["resume"]["target"]
        replay = replays.get(target)
        reference = results[target]
        expected = {"target": target, "completed": replay is not None,
            "state_bitwise_equal": bool(replay and replay["final_state_hashes"] == reference["final_state_hashes"]),
            "validation_equal": bool(replay and replay["validation"] == reference["validation"])}
        if complete["resume"] != expected: raise ValueError("Recovery check recomputation differs")
        if replay:
            original = {row["update"]:row for row in reference["records"]}
            keys = ("loss", "gradient_global_norm", "learning_rate", "bt4_learning_rate",
                    "root_legal_conditional_ce", "base_policy_root_legal_conditional_ce")
            replay_diagnostics = {
                "max_abs_training_metric_difference": {key:max(abs(row[key]-original[row["update"]][key])
                    for row in replay["records"]) for key in keys},
                "validation_abs_difference": {key:abs(replay["validation"][key]-reference["validation"][key])
                    for key in ("loss", "root_legal_conditional_ce", "base_policy_root_legal_conditional_ce")},
                "interpretation": "Descriptive differences only; do not replace the frozen bitwise replay criterion with a post-hoc tolerance. Without a repeated uninterrupted control, nondeterministic execution and checkpoint-specific effects are not separated."}
    baseline = results.get("production_control")
    rows = {}
    for name, result in results.items():
        rows[name] = {"timing": result["timing"],
            "speedup_vs_production": baseline["timing"]["mean_seconds"]/result["timing"]["mean_seconds"],
            "validation": {k:result["validation"][k] for k in ("loss", "root_legal_conditional_ce", "base_policy_root_legal_conditional_ce")},
            "first_update_seconds": result["records"][0]["end_to_end_seconds"],
            "final_state_hashes": result["final_state_hashes"],
            "unchanged_weight_proxy": result["unchanged_weight_proxy_update501"]}
    summary = {"experiment_id": "GPU-003", "arms": rows, "candidate": complete["candidate"],
        "resume": complete["resume"], "all_arms_complete": complete["all_arms_complete"],
        "replay_diagnostics": replay_diagnostics,
        "failures": [r for r in complete["records"] if r["process_status"] != "complete"],
        "scope": "Fifty-update warmup-prefix diagnostic; common teacher-forced/full-horizon validation, not playing strength or stable-phase WSD validation. FP32-storage arms also change optimizer moment storage."}
    preserve_bytes(root/"summary.json", (json.dumps(summary, indent=2, sort_keys=True)+"\n").encode())
    receipt = {"call_id": submitted["call_id"], "output_relative": remote, "artifacts": artifacts,
        "hash_basis": "Terminal bytes from authenticated Modal Volume, preserving existing evidence; timings and selection independently recomputed."}
    preserve_bytes(root/"completion.json", (json.dumps(receipt, indent=2, sort_keys=True)+"\n").encode())
    print(json.dumps({"status": "collected", "arms_complete": len(results), "candidate": complete["candidate"], "resume": complete["resume"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    collect(parser.parse_args().root)
