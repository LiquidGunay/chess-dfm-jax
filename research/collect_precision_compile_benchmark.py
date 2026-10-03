"""Read-only collection of terminal GPU-002 evidence (never launches compute)."""
import argparse
import hashlib
import json
from pathlib import Path

import modal
from research.collect_gpu_training_benchmark import preserve_bytes


def collect(root):
    protocol = json.loads((root/"preregister.json").read_text())
    submitted = json.loads((root/"sweep_run.json").read_text())
    call = modal.FunctionCall.from_id(submitted["call_id"])
    try:
        complete = call.get(timeout=0)
    except TimeoutError:
        print(json.dumps({"status": "running", "call_id": submitted["call_id"]}), flush=True)
        return
    volume = modal.Volume.from_name("chess-dfm-training-results", environment_name="chess-dfm-research")
    remote = complete["output_relative"]
    pre = json.loads((root/"preflight.json").read_text())
    if pre["output_relative"] != remote:
        raise ValueError("Completion output path differs")
    artifacts = {}
    names = ["launch.json", "sweep.json"]
    for record in complete["records"]:
        arm = record["arm"]
        if arm not in {a["id"] for a in protocol["arms"]}: raise ValueError("Unknown arm")
        names.extend(arm+"/"+name for name in ("launch.json", "receipt.json", "stdout.log"))
        if record["result_status"] is not None:
            names.append(arm+"/result.json")
            # Only complete first-batch diagnostics write retained tensor slices.
            if record["result_status"] == "complete":
                names.append(arm+"/diagnostic_tensors.safetensors")
    for name in names:
        target = root/name
        payload = b"".join(volume.read_file(remote+"/"+name))
        preserve_bytes(target, payload)
        artifacts[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    launch = json.loads((root/"launch.json").read_text())
    if launch["protocol"] != protocol or json.loads((root/"sweep.json").read_text()) != complete:
        raise ValueError("Protocol/completion mismatch")
    receipt = {"call_id": submitted["call_id"], "output_relative": remote,
               "all_complete": complete["all_complete"], "artifacts": artifacts,
               "hash_basis": "SHA256 of terminal bytes read from authenticated Modal Volume; collection preserves any existing local evidence."}
    preserve_bytes(root/"completion.json", (json.dumps(receipt, indent=2, sort_keys=True)+"\n").encode())
    print(json.dumps({"status": "collected", "all_complete": complete["all_complete"], "artifacts": len(artifacts)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    collect(parser.parse_args().root)
