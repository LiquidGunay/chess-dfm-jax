#!/usr/bin/env python3
"""Export the W&B history needed to reconstruct the final joint-training lineage.

The export is deliberately bounded to three known runs and a fixed metric allowlist.
It reads WANDB_API_KEY from /mountpoint/.exp/.env when the variable is not already
present, but never writes or prints the credential.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import shutil
from pathlib import Path
from typing import Any

import wandb


ENTITY = "gunays-independent"
PROJECT = "chess_dfm_jax-joint"
RUNS = {
    "initial_b256": (
        "joint-h8-v5p4-b256-rawproj-officialsig1024-normloss-"
        "dclip0p5-sig0p1-tf20k-free-2full-20260503e"
    ),
    "continuation_b512": "jph8-b512-ln-sig005-lr3e4-init221432-8ep-v5e4-20260506a",
    "final_reused_run": (
        "jph8-b512-ln-sig005-lr3e4-init221432-8ep-v5e64-euw4-20260506a"
    ),
}

HISTORY_KEYS = [
    "_step",
    "_runtime",
    "_timestamp",
    "step",
    "data_parallel_devices",
    "data_parallel_local_devices",
    "data_parallel_per_device_batch_size",
    "process_batch_size",
    "process_count",
    "examples_per_second",
    "iteration_examples_per_second",
    "learning_rate",
    "bt4_learning_rate",
    "loss",
    "dfm_ce_loss",
    "dfm_ce_loss_by_horizon_h1",
    "dfm_ce_loss_by_horizon_h2",
    "dfm_ce_loss_by_horizon_h3",
    "dfm_ce_loss_by_horizon_h4",
    "dfm_ce_loss_by_horizon_h5",
    "dfm_ce_loss_by_horizon_h6",
    "dfm_ce_loss_by_horizon_h7",
    "dfm_ce_loss_by_horizon_h8",
    "accuracy",
    "first_legal_mass",
    "first_legality_loss",
    "jepa_positive_loss",
    "jepa_raw_mse",
    "jepa_norm_loss",
    "jepa_sigreg_loss",
    "jepa_pred_sigreg_loss",
    "jepa_teacher_forcing",
    "jepa_target_sample_count",
    "val_loss",
    "val_dfm_ce_loss",
    "val_dfm_ce_loss_by_horizon_h1",
    "val_dfm_ce_loss_by_horizon_h2",
    "val_dfm_ce_loss_by_horizon_h3",
    "val_dfm_ce_loss_by_horizon_h4",
    "val_dfm_ce_loss_by_horizon_h5",
    "val_dfm_ce_loss_by_horizon_h6",
    "val_dfm_ce_loss_by_horizon_h7",
    "val_dfm_ce_loss_by_horizon_h8",
    "val_accuracy",
    "val_first_legal_mass",
    "val_first_legality_loss",
    "val_jepa_positive_loss",
    "val_jepa_raw_mse",
    "val_jepa_norm_loss",
    "val_jepa_sigreg_loss",
    "val_jepa_pred_sigreg_loss",
    "val_jepa_target_sample_count",
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def clean_mapping(mapping: Any) -> dict[str, Any]:
    return json.loads(json.dumps(dict(mapping), default=json_default))


def export_run(api: wandb.Api, label: str, run_id: str, raw_dir: Path) -> dict[str, Any]:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    run_dir = raw_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "label": label,
        "entity": ENTITY,
        "project": PROJECT,
        "id": run.id,
        "name": run.name,
        "display_name": run.display_name,
        "url": run.url,
        "state": run.state,
        "created_at": getattr(run, "created_at", None),
        "heartbeat_at": getattr(run, "heartbeat_at", None),
        "config": clean_mapping(run.config),
        "summary": clean_mapping(run.summary),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n"
    )

    history_path = run_dir / "history.jsonl.gz"
    row_count = 0
    min_step: int | None = None
    max_step: int | None = None
    with gzip.open(history_path, "wt", encoding="utf-8") as handle:
        for row in run.scan_history(keys=HISTORY_KEYS, page_size=10_000):
            cleaned = json.loads(json.dumps(row, default=json_default))
            handle.write(json.dumps(cleaned, separators=(",", ":"), sort_keys=True) + "\n")
            row_count += 1
            step = cleaned.get("step", cleaned.get("_step"))
            if isinstance(step, (int, float)):
                int_step = int(step)
                min_step = int_step if min_step is None else min(min_step, int_step)
                max_step = int_step if max_step is None else max(max_step, int_step)

    log_status: dict[str, Any] = {"available": False}
    try:
        output_file = run.file("output.log")
        if output_file.size and output_file.size > 0:
            downloaded = Path(
                output_file.download(root=str(run_dir / "download"), replace=True).name
            )
            destination = run_dir / "output.log"
            shutil.move(str(downloaded), destination)
            shutil.rmtree(run_dir / "download", ignore_errors=True)
            log_status = {
                "available": True,
                "bytes": destination.stat().st_size,
            }
    except Exception as exc:  # W&B raises varying exceptions for absent files.
        log_status = {"available": False, "error_type": type(exc).__name__}

    return {
        "label": label,
        "id": run.id,
        "url": run.url,
        "state": run.state,
        "history_rows": row_count,
        "min_step": min_step,
        "max_step": max_step,
        "output_log": log_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
    )
    args = parser.parse_args()

    load_dotenv(Path("/mountpoint/.exp/.env"))
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not available")

    args.output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)
    inventory = [
        export_run(api, label, run_id, args.output)
        for label, run_id in RUNS.items()
    ]
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    for item in inventory:
        print(
            f"{item['label']}: rows={item['history_rows']}, "
            f"steps={item['min_step']}..{item['max_step']}, "
            f"state={item['state']}"
        )


if __name__ == "__main__":
    main()
