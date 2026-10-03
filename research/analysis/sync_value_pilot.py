"""Upload completed VP-001 scalar logs directly to W&B from this machine.

No credential is sent to Modal. Dry-run by default. Separates 2048-position
validation curves from the 8192-position terminal validation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from research.analysis.import_wandb_logs import ensure_private_project, scalar_metrics

ENTITY, PROJECT = "gunays-independent", "chess-dfm-history"


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(directory, arm):
    root = directory / "collected" / arm
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "complete" or report["updates"] != 1000:
        raise ValueError("Only complete pilots may be imported")
    source = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("metrics.jsonl", "validation_metrics.jsonl", "run_config.json", "report.json")}
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    events = {}
    for row in records(root / "metrics.jsonl"):
        update = row["update"]
        if update in events or row["examples"] != update * 1024:
            raise ValueError("Training counter drift")
        events[update] = {"update": update, "examples": row["examples"], **scalar_metrics(row, "train")}
    if sorted(events) != list(range(1, 1001)):
        raise ValueError("Incomplete training curve")
    for row in records(root / "validation_metrics.jsonl"):
        update = row["update"]
        count = 8192 if update == 1000 else 2048
        event = events.setdefault(update, {"update": update, "examples": update * 1024})
        event.update(scalar_metrics(row, "val_" + str(count)))
    if sorted(events) != list(range(1001)):
        raise ValueError("Initial validation missing")
    return {"arm": arm, "id": json.loads((root / "wandb.json").read_text())["id"],
        "name": "Hero + " + ("native value" if arm == "native" else "JEPA value residual") + " | full history | 1000 updates",
        "events": [events[i] for i in sorted(events)], "source": source, "digest": digest}


def verify(remote, events):
    observed = list(remote.scan_history(page_size=1000))
    by_update = {int(row["update"]): row for row in observed if row.get("update") is not None}
    if len(by_update) != len(events):
        raise ValueError("W&B curve row count mismatch")
    for event in events:
        row = by_update[event["update"]]
        for key, value in event.items():
            if row.get(key) is None or not math.isclose(float(row[key]), float(value), rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("W&B scalar differs from authoritative JSONL: " + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dotenv", type=Path)
    args = parser.parse_args()
    runs = [prepare(args.directory, arm) for arm in ("native", "jepa_residual")]
    if not args.upload:
        print(json.dumps([{k: r[k] for k in ("arm", "id", "name", "digest")} for r in runs], indent=2))
        return
    if args.dotenv:
        from dotenv import dotenv_values
        os.environ["WANDB_API_KEY"] = dotenv_values(args.dotenv)["WANDB_API_KEY"]
    import wandb
    ensure_private_project(ENTITY, PROJECT)
    api = wandb.Api()
    receipts = []
    for spec in runs:
        identity = f"{ENTITY}/{PROJECT}/{spec['id']}"
        try:
            existing = api.run(identity)
        except wandb.errors.CommError:
            existing = None
        if existing and existing.config.get("import_manifest_sha256") != spec["digest"]:
            raise ValueError("Existing W&B run has a different source contract")
        if existing is None:
            run = wandb.init(entity=ENTITY, project=PROJECT, id=spec["id"], name=spec["name"],
                group="VP-001-value-head-pilots", job_type="fresh-joint-training",
                tags=["VP-001", "full-history", "fresh-BT4", spec["arm"]],
                config={"comparison_cohort": "VP-001", "arm": spec["arm"], "archive": False,
                    "import_schema": "vp001-canonical-jsonl-v1", "import_manifest_sha256": spec["digest"],
                    "source_files": spec["source"], "training_updates": 1000,
                    "validation_warning": "2048-position intermediate and 8192-position terminal metrics are separate series",
                    "protocol": json.loads((args.directory / "preregister.json").read_text())},
                settings=wandb.Settings(disable_git=True), save_code=False,
                dir=str(args.directory))
            run.define_metric("examples")
            for prefix in ("train", "val_2048", "val_8192"):
                run.define_metric(prefix + "/*", step_metric="examples")
            for event in spec["events"]:
                run.log(event, step=event["update"])
            run.finish()
        remote = wandb.Api().run(identity)
        verify(remote, spec["events"])
        receipts.append({"arm": spec["arm"], "url": remote.url, "id": spec["id"],
            "all_source_scalars_verified": True, "rows": len(spec["events"]), "source_sha256": spec["digest"]})
    destination = args.directory / "wandb_sync_receipt.json"
    with destination.open("x") as stream:
        json.dump({"runs": receipts, "credentials_kept_off_modal": True}, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
