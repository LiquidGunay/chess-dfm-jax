from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.infra.pull_modal_training_compact import (
    COMPACT_FILES,
    REQUIRED_COMPACT_FILES,
    parse_volume_inventory,
    require_result_label,
    validate_compact_run,
)


def test_result_label_matches_training_contract() -> None:
    label = "hero-training-phase1-v2_fp32_1_12-l40s-abc123"
    assert require_result_label(label) == label
    for unsafe in ("candidate", "hero-training-../x", "hero-training-UPPER"):
        with pytest.raises(ValueError):
            require_result_label(unsafe)


def test_inventory_requires_core_files_and_accepts_optional_validation() -> None:
    label = "hero-training-phase1-control-l40s-abc"
    records = [
        {
            "filename": f"runs/{label}/{name}",
            "type": "file",
            "size": index + 1,
        }
        for index, name in enumerate(COMPACT_FILES)
    ]
    records.append(
        {
            "filename": f"runs/{label}/checkpoint/model.safetensors",
            "type": "file",
            "size": 10_000_000,
        }
    )
    inventory = parse_volume_inventory(json.dumps(records), result_label=label)
    assert set(inventory) == set(COMPACT_FILES)
    assert "model.safetensors" not in inventory

    for record in records:
        record["size"] = "1.0 KiB"
    display_inventory = parse_volume_inventory(
        json.dumps(records), result_label=label
    )
    assert all(size is None for size in display_inventory.values())

    without_validation = [
        record
        for record in records
        if not str(record["filename"]).endswith("/validation_metrics.jsonl")
    ]
    diagnostic_inventory = parse_volume_inventory(
        json.dumps(without_validation),
        result_label=label,
    )
    assert set(diagnostic_inventory) == set(REQUIRED_COMPACT_FILES)

    without_report = [
        record
        for record in records
        if not str(record["filename"]).endswith("/report.json")
    ]
    with pytest.raises(ValueError, match="missing compact files"):
        parse_volume_inventory(json.dumps(without_report), result_label=label)


def test_validate_compact_run_rejects_segment_only_metrics(tmp_path: Path) -> None:
    payloads = {
        "metrics.jsonl": '{"update": 1}\n{"update": 2}\n',
        "run_config.json": "{}\n",
        "report.json": '{"updates": 2}\n',
        "modal_run.json": "{}\n",
        "loss_summary.json": "{}\n",
        "optimizer_partition.json": "{}\n",
        "validation_metrics.jsonl": '{"update": 2}\n',
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(payload, encoding="utf-8")
    sizes = {name: (tmp_path / name).stat().st_size for name in COMPACT_FILES}
    validate_compact_run(tmp_path, expected_sizes=sizes)

    (tmp_path / "metrics.jsonl").write_text(
        '{"update": 2}\n{"update": 3}\n', encoding="utf-8"
    )
    sizes["metrics.jsonl"] = (tmp_path / "metrics.jsonl").stat().st_size
    with pytest.raises(ValueError, match="starting at one"):
        validate_compact_run(tmp_path, expected_sizes=sizes)


def test_validate_compact_run_allows_validation_free_diagnostic(
    tmp_path: Path,
) -> None:
    payloads = {
        "metrics.jsonl": "{\"update\": 1}\n{\"update\": 2}\n",
        "run_config.json": "{}\n",
        "report.json": "{\"updates\": 2}\n",
        "modal_run.json": "{\"profile\": \"diagnostic\"}\n",
        "loss_summary.json": "{}\n",
        "optimizer_partition.json": "{}\n",
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(payload, encoding="utf-8")
    sizes = {
        name: (tmp_path / name).stat().st_size
        for name in REQUIRED_COMPACT_FILES
    }

    validate_compact_run(tmp_path, expected_sizes=sizes)

    (tmp_path / "modal_run.json").write_text(
        "{\"profile\": \"phase1\"}\n",
        encoding="utf-8",
    )
    sizes["modal_run.json"] = (tmp_path / "modal_run.json").stat().st_size
    with pytest.raises(ValueError, match="no validation metrics"):
        validate_compact_run(tmp_path, expected_sizes=sizes)
