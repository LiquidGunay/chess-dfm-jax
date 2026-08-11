from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.stitch_training_segments import stitch_training_segments
from research.train_torch import _LOSS_SUMMARY_METRICS


def _record(update: int, *, runtime: float) -> dict[str, object]:
    record: dict[str, object] = {
        "update": update,
        "examples": update * 1024,
        "optimizer_skipped_nonfinite": False,
        "step_seconds": runtime,
        "elapsed_seconds": runtime * update,
        "segment_update": update,
        "segment_examples": update * 1024,
    }
    record.update({name: float(update) for name in _LOSS_SUMMARY_METRICS})
    return record


def _write_segment(
    root: Path,
    records: list[dict[str, object]],
    *,
    resume_update: int | None,
    terminal: bool,
) -> None:
    root.mkdir(parents=True)
    args = {
        "batch_size": 1024,
        "max_checkpoints": 7 if resume_update is None else 2,
        "resume_checkpoint": (
            None if resume_update is None else f"/remote/update{resume_update:08d}"
        ),
        "save_updates": [2] if resume_update is None else [4],
    }
    run_config = {
        "framework": "torch",
        "recipe": "hero_v2",
        "config": {"z_dim": 128},
        "args": args,
        "created_utc": "fresh" if resume_update is None else "resumed",
        "restore_seconds": 1.0 if resume_update is None else 2.0,
        "resume": (
            {"enabled": False}
            if resume_update is None
            else {"enabled": True, "optimizer_update": resume_update}
        ),
    }
    (root / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )
    (root / "optimizer_partition.json").write_text(
        json.dumps({"partition": "same"}),
        encoding="utf-8",
    )
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    if terminal:
        (root / "report.json").write_text(
            json.dumps(
                {
                    "updates": 4,
                    "checkpoint": {"state": {"sha256": "a" * 64}},
                    "live_validation": {"records": []},
                }
            ),
            encoding="utf-8",
        )
        (root / "modal_run.json").write_text("{}\n", encoding="utf-8")
        (root / "validation_metrics.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )


def _write_continuation_segment(
    root: Path,
    records: list[dict[str, object]],
    *,
    target_updates: int,
    validation_updates: list[int],
    resume_checkpoint: Path | None,
    resume_update: int | None,
    terminal: bool,
    nested_stitch: bool = False,
) -> Path:
    root.mkdir(parents=True)
    source_mapping_sha256 = "d" * 64
    checkpoint_state = b"continuation-test-state"
    checkpoint_state_sha256 = hashlib.sha256(checkpoint_state).hexdigest()
    live_validation = {
        "enabled": True,
        "manifest_path": "/workspace/frozen-validation/manifest.json",
        "fast_validation": {"batch_size": 64, "pool": "fast"},
        "updates": validation_updates,
        "observed_percentages": [float(update) for update in validation_updates],
    }
    optimizer_partition = {
        "schema_version": "test-partition-v1",
        "partition": "same",
    }
    resume_contract = {
        "schema_version": "torch-training-resume-contract-v1",
        "framework": "torch",
        "recipe": "hero_v2",
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_mapping_sha256": source_mapping_sha256,
        "config": {"z_dim": 128},
        "data": {"identity": "same"},
        "optimizer_policy": {"precision": "fp32_master"},
        "optimizer_partition": optimizer_partition,
        "schedule": {"target_updates": target_updates, "batch_size": 1024},
        "runtime": {"live_validation": live_validation},
    }
    resume_contract_sha256 = hashlib.sha256(
        json.dumps(
            resume_contract,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if resume_update is None:
        resume: dict[str, object] = {"enabled": False}
    else:
        assert resume_checkpoint is not None
        resume = {
            "enabled": True,
            "checkpoint_dir": str(resume_checkpoint.resolve()),
            "optimizer_update": resume_update,
            "next_data_cursor": resume_update,
            "checkpoint_state_sha256": checkpoint_state_sha256,
        }
    if nested_stitch:
        resume = {
            "enabled": True,
            "checkpoint_dir": "/archived/internal-retry/update00000001",
            "optimizer_update": 1,
            "next_data_cursor": 1,
            "checkpoint_state_sha256": "e" * 64,
        }
    run_config = {
        "framework": "torch",
        "recipe": "hero_v2",
        "git_commit": "a" * 40,
        "config": {"z_dim": 128},
        "data": {"identity": "same"},
        "optimizer_policy": {"precision": "fp32_master"},
        "optimizer_partition": optimizer_partition,
        "source": {"combined_sha256": source_mapping_sha256},
        "args": {
            "batch_size": 1024,
            "steps": target_updates,
            "validation_updates": validation_updates,
        },
        "live_validation": live_validation,
        "resume_contract": resume_contract,
        "resume_contract_sha256": resume_contract_sha256,
        "resume": resume,
    }
    (root / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )
    (root / "optimizer_partition.json").write_text(
        json.dumps(optimizer_partition),
        encoding="utf-8",
    )
    metrics_payload = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    (root / "metrics.jsonl").write_text(metrics_payload, encoding="utf-8")

    final_update = int(records[-1]["update"])
    checkpoint_dir = root / "checkpoints" / f"update{final_update:08d}"
    checkpoint_dir.mkdir(parents=True)
    state_path = checkpoint_dir / "state.safetensors"
    state_path.write_bytes(checkpoint_state)
    checkpoint_manifest = {
        "format": "chess-dfm-torch-training-v2",
        "model_only": False,
        "optimizer_resume_supported": True,
        "optimizer_update": final_update,
        "next_data_cursor": final_update,
        "resume_contract": resume_contract,
        "resume_contract_sha256": resume_contract_sha256,
        "state": {
            "path": state_path.name,
            "size_bytes": state_path.stat().st_size,
            "sha256": checkpoint_state_sha256,
        },
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(checkpoint_manifest),
        encoding="utf-8",
    )
    if nested_stitch:
        (root / "stitch_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "chess-dfm-training-segment-stitch-v1",
                    "authoritative_training_curve": "metrics.jsonl",
                    "terminal_update": final_update,
                    "merged_record_count": len(records),
                    "merged_update_range": [1, final_update],
                    "output_sha256": {
                        "metrics.jsonl": hashlib.sha256(
                            metrics_payload.encode("utf-8")
                        ).hexdigest()
                    },
                }
            ),
            encoding="utf-8",
        )
    if terminal:
        (root / "report.json").write_text(
            json.dumps(
                {
                    "updates": final_update,
                    "next_data_cursor": final_update,
                    "checkpoint": {"state": {"sha256": "f" * 64}},
                    "recovery_checkpoints": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "modal_run.json").write_text("{}\n", encoding="utf-8")
        (root / "validation_metrics.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
    return checkpoint_dir


def test_stitch_accepts_exact_scientific_replay_and_uses_later_runtime(
    tmp_path: Path,
) -> None:
    first = tmp_path / "attempt000"
    terminal = tmp_path / "terminal"
    output = tmp_path / "compact"
    _write_segment(
        first,
        [_record(update, runtime=1.0) for update in (1, 2, 3)],
        resume_update=None,
        terminal=False,
    )
    _write_segment(
        terminal,
        [_record(update, runtime=2.0) for update in (3, 4)],
        resume_update=2,
        terminal=True,
    )

    manifest = stitch_training_segments(
        (("attempt000", first), ("terminal", terminal)),
        terminal_segment="terminal",
        output_dir=output,
    )

    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["update"] for record in records] == [1, 2, 3, 4]
    assert records[2]["step_seconds"] == 2.0
    assert manifest["overlap_updates"] == [3]
    assert manifest["update_owner_ranges"] == [
        {"name": "attempt000", "updates": [1, 2]},
        {"name": "terminal", "updates": [3, 4]},
    ]
    assert manifest["authoritative_training_curve"] == "metrics.jsonl"
    assert manifest["authoritative_loss_summary"] == "loss_summary.json"
    assert manifest["terminal_report_curve_scope"] == (
        "terminal_segment_only_unmodified"
    )
    assert json.loads((output / "loss_summary.json").read_text())[
        "terminal"
    ]["update"] == 4


def test_stitch_rejects_scientific_replay_or_config_drift(tmp_path: Path) -> None:
    first = tmp_path / "attempt000"
    terminal = tmp_path / "terminal"
    _write_segment(
        first,
        [_record(update, runtime=1.0) for update in (1, 2, 3)],
        resume_update=None,
        terminal=False,
    )
    replay = [_record(update, runtime=2.0) for update in (3, 4)]
    replay[0]["loss"] = -1.0
    _write_segment(
        terminal,
        replay,
        resume_update=2,
        terminal=True,
    )
    with pytest.raises(ValueError, match="Scientific replay drift"):
        stitch_training_segments(
            (("attempt000", first), ("terminal", terminal)),
            terminal_segment="terminal",
            output_dir=tmp_path / "bad-replay",
        )

    replay[0]["loss"] = 3.0
    (terminal / "metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in replay),
        encoding="utf-8",
    )
    config = json.loads((terminal / "run_config.json").read_text())
    config["config"]["z_dim"] = 256
    (terminal / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Training config drift"):
        stitch_training_segments(
            (("attempt000", first), ("terminal", terminal)),
            terminal_segment="terminal",
            output_dir=tmp_path / "bad-config",
        )


@pytest.mark.parametrize("nested_predecessor", (False, True))
def test_stitch_cross_profile_continuation_binds_checkpoint_and_full_curve(
    tmp_path: Path,
    nested_predecessor: bool,
) -> None:
    predecessor = tmp_path / "phase1"
    terminal = tmp_path / "phase2"
    predecessor_checkpoint = _write_continuation_segment(
        predecessor,
        [_record(update, runtime=1.0) for update in (1, 2)],
        target_updates=2,
        validation_updates=[2],
        resume_checkpoint=None,
        resume_update=None,
        terminal=False,
        nested_stitch=nested_predecessor,
    )
    _write_continuation_segment(
        terminal,
        [_record(update, runtime=2.0) for update in (3, 4)],
        target_updates=4,
        validation_updates=[3, 4],
        resume_checkpoint=predecessor_checkpoint,
        resume_update=2,
        terminal=True,
    )

    output = tmp_path / "compact"
    manifest = stitch_training_segments(
        (("predecessor", predecessor), ("terminal", terminal)),
        terminal_segment="terminal",
        output_dir=output,
        continuation_predecessor_segment="predecessor",
    )

    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["update"] for record in records] == [1, 2, 3, 4]
    assert manifest["continuation_predecessor_segment"] == "predecessor"
    assert manifest["resume_bindings"] == [
        {
            "source_segment": "predecessor",
            "destination_segment": "terminal",
            "optimizer_update": 2,
            "next_data_cursor": 2,
            "checkpoint_state_sha256": hashlib.sha256(
                b"continuation-test-state"
            ).hexdigest(),
            "checkpoint_manifest_sha256": hashlib.sha256(
                (predecessor_checkpoint / "manifest.json").read_bytes()
            ).hexdigest(),
        }
    ]
    predecessor_record = manifest["segments"][0]
    assert predecessor_record["config_validation"] == (
        "normalized_exact_resume_contract"
    )
    assert bool(
        predecessor_record["nested_stitch_manifest_sha256"]
    ) is nested_predecessor

    terminal_config_path = terminal / "run_config.json"
    terminal_config = json.loads(terminal_config_path.read_text())
    terminal_config["resume"]["checkpoint_state_sha256"] = "0" * 64
    terminal_config_path.write_text(
        json.dumps(terminal_config),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="Continuation checkpoint provenance drift",
    ):
        stitch_training_segments(
            (("predecessor", predecessor), ("terminal", terminal)),
            terminal_segment="terminal",
            output_dir=tmp_path / "bad-binding",
            continuation_predecessor_segment="predecessor",
        )
