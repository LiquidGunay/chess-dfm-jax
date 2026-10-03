"""Import canonical historical Torch JSONL logs; never rerun training.

Dry-run by default. Credentials are loaded only in memory, and only scalar
metrics plus curated provenance are uploaded. Stable IDs and content hashes
prevent accidental duplicate imports or overwriting changed evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("hero_v2_phase1", "proposal_a_tuning", "pa001_modal_20260929")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scalar_metrics(record, prefix):
    return {f"{prefix}/{key}": value for key, value in record.items()
            if isinstance(value, (int, float, bool)) and math.isfinite(value)
            and key not in {"update", "examples"}}


def pool_identity(record):
    pool = record["pool"]
    data = pool.get("data", {})
    identity = {"definition": pool.get("pool_definition"),
                "files": data.get("file_manifest_sha256"),
                "indices": pool.get("indices_sha256"),
                "count": record["evaluation_examples"]}
    return identity, f"val_original_{identity['count']}_{digest(identity)[:8]}"


def merged_events(training, validation):
    events = {}
    previous = 0
    for row in training:
        update = row["update"]
        if update <= previous or row["examples"] <= 0:
            raise ValueError("Training records must have unique increasing updates")
        previous = update
        events[update] = {"update": update, "examples": row["examples"],
                          **scalar_metrics(row, "train")}
    pools = {}
    for row in validation:
        identity, prefix = pool_identity(row)
        pools[prefix] = identity
        update = row["update"]
        event = events.setdefault(update, {"update": update, "examples": row["examples"]})
        if event["examples"] != row["examples"]:
            raise ValueError("Validation/training examples disagree at the same update")
        metrics = scalar_metrics(row["metrics"], prefix)
        if set(metrics) & set(event):
            raise ValueError("Duplicate validation records for an update/pool")
        event.update(metrics)
    return [events[k] for k in sorted(events)], pools


def display_identity(family, relative, config, modal, updates):
    """Keep diagnostic, exposure and dataset cohorts separate in the UI."""
    if family == "pa001_modal_20260929":
        return "Proposal A | full epoch | 7,774 updates", "PA-001-full-epoch"
    if relative.name == "hero1_exact":
        return "Hero 1 | no history | 554 updates", "hero-554-current-only"
    arm = modal.get("arm_config", {})
    model = config.get("config", {})
    if family == "proposal_a_tuning":
        lr = model.get("learning_rate")
        wd = model.get("weight_decay")
        mode = arm.get("weight_decay_mode", "default")
        return f"Proposal A | LR {lr:g} | decay {wd:g} ({mode})", f"proposal-a-{relative.parts[0]}"
    precision = {
        "fp32_master": "FP32 optimizer", "parameter": "BF16 optimizer",
        "encoder_fp32_master": "encoder FP32", "main_fp32_master": "heads FP32",
        "encoder_trunk_fp32_master": "trunk FP32",
    }.get(arm.get("optimizer_precision"), "default optimizer")
    ratio = arm.get("encoder_lr_ratio")
    parts = ["Hero v2", precision]
    if ratio:
        parts.append(f"encoder LR 1/{round(1 / ratio)}")
    if arm.get("legality_coeff") == 0:
        parts.append("no legality penalty")
    if arm.get("policy_distill_coeff"):
        parts.append("Hero1 distillation" if arm.get("policy_distill_teacher_result_label") else "policy distillation")
    if arm.get("dfm_closed_loop_mode") not in (None, "none"):
        parts.append("closed-loop")
    group = "diagnostics" if "diagnostics" in relative.parts else f"hero-{updates}-current-only"
    return " | ".join(parts), group


def inventory(root):
    results = []
    labels = set()
    for family in FAMILIES:
        for path in sorted((root / "research/analysis" / family).rglob("metrics.jsonl")):
            if "segments" in path.relative_to(root).parts:
                # The canonical parent already incorporates accepted continuation segments.
                continue
            parent = path.parent
            config_path = parent / "run_config.json"
            config = json.loads(config_path.read_text())
            modal_path = parent / "modal_run.json"
            modal = json.loads(modal_path.read_text()) if modal_path.exists() else {}
            label = modal.get("result_label") or Path(config["args"]["output_dir"]).name
            if label in labels:
                raise ValueError("Duplicate canonical result label")
            labels.add(label)
            training = read_records(path)
            validation = read_records(parent / "validation_metrics.jsonl")
            if not training:
                continue
            events, pools = merged_events(training, validation)
            files = {}
            for name in ("metrics.jsonl", "validation_metrics.jsonl", "run_config.json",
                         "modal_run.json", "stitch_manifest.json"):
                p = parent / name
                if p.exists():
                    files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
            fmt = config["args"].get("data_format") or "trajectory_v3"
            history = "full_history" if fmt == "lc0_sequential" else "current_only_as_preprocessed"
            relative = parent.relative_to(root / "research/analysis" / family)
            name, group = display_identity(family, relative, config, modal, training[-1]["update"])
            curated = {
                "import_schema": "historical-torch-jsonl-v1", "result_label": label,
                "original_created_utc": config.get("created_utc"),
                "framework": config.get("framework"), "recipe": config.get("recipe"),
                "original_git_commit": config.get("git_commit"),
                "original_source_sha256": modal.get("source_tree_sha256"),
                "model_recipe": config.get("config", {}), "data_format": fmt,
                "history_mode": history, "training_data": config.get("data", {}),
                "archive": True, "comparison_cohort": group,
                "run_role": "diagnostic" if group == "diagnostics" else "historical_training",
                "original_arm": modal.get("arm"),
                "training_updates": training[-1]["update"],
                "validation_pools": pools, "import_source_files": files,
                "import_manifest_sha256": digest(files),
                "comparison_caveat": "Original validation pools, not HH-002 matched validation. "
                    "Different data/recipes cannot be compared by total loss alone.",
            }
            results.append({"id": "local-" + digest(label)[:20], "name": name,
                            "group": group, "config": curated, "events": events,
                            "training_rows": len(training), "validation_rows": len(validation),
                            "last_update": training[-1]["update"],
                            "last_examples": training[-1]["examples"]})
    return results


def ensure_private_project(entity, project):
    import requests
    session = requests.Session()
    session.auth = ("api", os.environ["WANDB_API_KEY"])
    query = "query($e:String!,$p:String!) { project(name:$p,entityName:$e) { name access } }"
    variables = {"e": entity, "p": project}
    def call(query):
        response = session.post("https://api.wandb.ai/graphql",
                                json={"query": query, "variables": variables}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError("W&B project operation returned GraphQL errors")
        return payload["data"]
    existing = call(query)["project"]
    if existing is None:
        call('mutation($e:String!,$p:String!) { upsertModel(input:{entityName:$e,name:$p,access:"PRIVATE"}) { model { name access } } }')
        existing = call(query)["project"]
    if existing["access"].lower() != "private":
        raise ValueError("Refusing upload to a non-private project")


def upload(runs, *, entity, project, output):
    import wandb
    ensure_private_project(entity, project)
    api = wandb.Api(timeout=30)
    existing = {run.id: run for run in api.runs(f"{entity}/{project}")}
    receipts = []
    log_dir = ROOT / ".local/wandb-import"
    log_dir.mkdir(parents=True, exist_ok=True)
    for item in runs:
        prior = existing.get(item["id"])
        if prior:
            if (prior.config.get("import_manifest_sha256") != item["config"]["import_manifest_sha256"]
                    or not prior.summary.get("import/complete")):
                raise ValueError("Existing run differs or has an incomplete import; refusing overwrite")
            url, status = prior.url, "already_uploaded"
        else:
            run = wandb.init(entity=entity, project=project, id=item["id"], resume="never",
                             name=item["name"], group=item["group"], job_type="historical-log-import",
                             tags=["archive", "historical-import", "pytorch", item["group"],
                                   item["config"]["history_mode"]],
                             notes=item["config"]["comparison_caveat"], config=item["config"],
                             dir=str(log_dir), mode="online", save_code=False,
                             settings=wandb.Settings(disable_git=True, disable_code=True,
                                 x_disable_meta=True, x_disable_stats=True, console="off", init_timeout=60))
            try:
                run.define_metric("examples")
                run.define_metric("train/*", step_metric="examples")
                for prefix in item["config"]["validation_pools"]:
                    run.define_metric(prefix + "/*", step_metric="examples")
                for event in item["events"]:
                    run.log(event)
                run.summary.update({"import/complete": True,
                    "import/training_rows": item["training_rows"],
                    "import/validation_rows": item["validation_rows"],
                    "import/events": len(item["events"]),
                    "import/last_update": item["last_update"],
                    "import/last_examples": item["last_examples"]})
                url = run.url
            finally:
                run.finish()
            api.flush()
            remote = api.run(f"{entity}/{project}/{item['id']}")
            if (not remote.summary.get("import/complete")
                    or remote.summary.get("import/events") != len(item["events"])
                    or remote.summary.get("update") != item["last_update"]):
                raise RuntimeError("Remote import verification failed")
            status = "uploaded_verified"
        receipt = {k: item[k] for k in ("id", "name", "group", "training_rows",
                                       "validation_rows", "last_update", "last_examples")}
        receipt.update(url=url, status=status, config=item["config"])
        receipts.append(receipt)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"entity": entity, "project": project,
                                      "runs": receipts}, indent=2) + "\n")
        print(json.dumps({k: receipt[k] for k in ("id", "name", "url", "status")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dotenv", type=Path)
    parser.add_argument("--inventory", type=Path, help="Save a credential-free proposed upload inventory")
    parser.add_argument("--entity", default="gunays-independent")
    parser.add_argument("--project", default="chess-dfm-torch")
    parser.add_argument("--output", type=Path, default=ROOT / "research/analysis/wandb_sync_20261003/import_receipt.json")
    args = parser.parse_args()
    runs = inventory(ROOT)
    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps({
            "status": "prepared_not_uploaded",
            "entity": args.entity, "project": args.project, "required_privacy": "private",
            "payload": "Scalar metrics and curated model/data configuration/provenance; no checkpoints, datasets, code, or credentials",
            "runs": [{k: v for k, v in run.items() if k != "events"} for run in runs],
        }, indent=2) + "\n")
    print(json.dumps({"run_count": len(runs), "training_rows": sum(r["training_rows"] for r in runs),
                      "validation_rows": sum(r["validation_rows"] for r in runs)}), flush=True)
    if args.upload:
        if args.dotenv:
            from dotenv import dotenv_values
            key = dotenv_values(args.dotenv).get("WANDB_API_KEY")
            if key:
                os.environ["WANDB_API_KEY"] = key
        if not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("WANDB_API_KEY is not configured")
        upload(runs, entity=args.entity, project=args.project, output=args.output)
    else:
        for run in runs:
            print(json.dumps({k: run[k] for k in ("id", "name", "group", "training_rows", "validation_rows")}))


if __name__ == "__main__":
    main()
