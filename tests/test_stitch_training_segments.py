from __future__ import annotations

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
