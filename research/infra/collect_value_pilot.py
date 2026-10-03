"""Read-only VP-001 collector and fail-closed continuation after training.

The optional --continue flag launches only the frozen evaluation stages,
through their existing CPU verification and fresh billing gates. No retries.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import modal


def store(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Immutable collected receipt changed: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(data)


def json_store(path, obj):
    store(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode())


def collect(directory):
    volume = modal.Volume.from_name("chess-dfm-training-results", environment_name="chess-dfm-research")
    protocol = json.loads((directory / "preregister.json").read_text())
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:20]
    remote = "experiments/vp001-" + digest
    for name in ("protocol.json", "preflight.json", "training_complete.json", "evaluation_preflight.json", "evaluation_complete.json", "policy_losses.json", "position_metrics.npz"):
        try:
            data = b"".join(volume.read_file(remote + "/" + name))
        except FileNotFoundError:
            continue
        store(directory / "collected" / name, data)
    completed = directory / "collected/training_complete.json"
    if completed.exists():
        for run in json.loads(completed.read_text())["runs"]:
            for name in ("report.json", "run_config.json", "modal_run.json", "validation_metrics.jsonl", "metrics.jsonl", "optimizer_partition.json", "checkpoint/manifest.json", "wandb.json", "wandb_error.json"):
                try:
                    data = b"".join(volume.read_file("runs/" + run["label"] + "/" + name))
                except FileNotFoundError:
                    continue
                store(directory / "collected" / run["arm"] / name, data)
    evaluated = directory / "collected/evaluation_complete.json"
    if evaluated.exists():
        for match in json.loads(evaluated.read_text())["matches"]:
            result = match["summary"]
            arena_label = result.get("arena_label", result.get("result_label"))
            if arena_label:
                for name in ("state.json", "report.json", "games.jsonl", "games.pgn"):
                    try:
                        data = b"".join(volume.read_file("arenas/" + arena_label + "/" + name))
                    except FileNotFoundError:
                        continue
                    store(directory / "collected/matches" / (match["left"] + "-" + match["right"]) / name, data)


def wait_call(directory, stage):
    record = json.loads((directory / (stage + "-result.json")).read_text())
    call = modal.FunctionCall.from_id(record["call_id"])
    while True:
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            print(json.dumps({"stage": stage, "status": "running", "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
            time.sleep(45)
            continue
        json_store(directory / (stage + "-completion.json"), result)
        if result.get("status") != "complete":
            raise RuntimeError("Remote stage did not complete successfully")
        collect(directory)
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("research/analysis/vp001_joint_value_20261003"))
    parser.add_argument("--continue", dest="continue_evaluation", action="store_true")
    args = parser.parse_args()
    if not args.continue_evaluation:
        collect(args.directory)
        return
    wait_call(args.directory, "train")
    for stage in ("eval-preflight", "evaluate"):
        if (args.directory / (stage + "-reservation.json")).exists():
            raise FileExistsError("Evaluation already attempted; no automatic retry")
        command = ["/root/.local/bin/modal", "run", "--env", "chess-dfm-research"]
        if stage == "evaluate":
            command.append("--detach")
        command.extend(["research/infra/modal_value_pilot.py", "--stage", stage,
                        "--protocol-path", str(args.directory / "preregister.json")])
        subprocess.run(command, check=True)
    wait_call(args.directory, "evaluate")
    print("Frozen training and evaluation complete; compact evidence collected.", flush=True)


if __name__ == "__main__":
    main()
