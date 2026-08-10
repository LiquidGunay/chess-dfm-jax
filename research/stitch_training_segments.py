"""Build a complete, provenance-bound metric curve from exact-resume segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "chess-dfm-training-segment-stitch-v1"
TERMINAL_COPY_FILES = (
    "run_config.json",
    "report.json",
    "modal_run.json",
    "optimizer_partition.json",
    "validation_metrics.jsonl",
)
OPERATIONAL_METRIC_KEYS = frozenset(
    {
        "backward_cuda_seconds",
        "checkpoint_write_seconds",
        "data_prefetch_hidden_seconds",
        "data_prepare_seconds",
        "data_seconds",
        "data_wait_seconds",
        "elapsed_seconds",
        "examples_per_second_end_to_end",
        "examples_per_second_step",
        "forward_cuda_seconds",
        "gpu_memory_allocated_bytes",
        "gpu_memory_reserved_bytes",
        "gpu_peak_memory_allocated_bytes",
        "gpu_peak_memory_reserved_bytes",
        "host_overhead_in_step_seconds",
        "optimizer_cuda_seconds",
        "recovery_checkpoint_path",
        "segment_examples",
        "segment_update",
        "step_seconds",
        "transfer_seconds",
    }
)
_ALLOWED_CONFIG_DIFFERENCE_PREFIXES = (
    ("created_utc",),
    ("restore_seconds",),
    ("resume",),
    ("args", "max_checkpoints"),
    ("args", "resume_checkpoint"),
    ("args", "save_updates"),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read metric segment {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected metric object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"Metric segment is empty: {path}")
    updates = [int(record.get("update", -1)) for record in records]
    if updates != list(range(updates[0], updates[-1] + 1)) or updates[0] < 1:
        raise ValueError(f"Metric segment is not positive and contiguous: {path}")
    for record, update in zip(records, updates, strict=True):
        if int(record.get("examples", -1)) < 1:
            raise ValueError(f"Metric update {update} has invalid examples")
        if bool(record.get("optimizer_skipped_nonfinite", False)):
            raise ValueError(f"Metric update {update} skipped a non-finite step")
    return records


def _difference_paths(
    left: Any,
    right: Any,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[tuple[str, ...]] = []
        for key in sorted(set(left) | set(right)):
            key_path = prefix + (str(key),)
            if key not in left or key not in right:
                differences.append(key_path)
            else:
                differences.extend(
                    _difference_paths(left[key], right[key], key_path)
                )
        return differences
    return [] if left == right else [prefix]


def _allowed_config_difference(path: tuple[str, ...]) -> bool:
    return any(
        path[: len(allowed)] == allowed
        for allowed in _ALLOWED_CONFIG_DIFFERENCE_PREFIXES
    )


def _validate_run_config(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    segment_name: str,
) -> list[str]:
    differences = _difference_paths(reference, candidate)
    forbidden = [path for path in differences if not _allowed_config_difference(path)]
    if forbidden:
        rendered = [".".join(path) for path in forbidden[:20]]
        raise ValueError(
            f"Training config drift in segment {segment_name}: {rendered}"
        )
    return [".".join(path) for path in differences]


def _semantic_metric_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in OPERATIONAL_METRIC_KEYS
    }


def _atomic_materialize(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _copy_terminal_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_name(f".{destination.name}.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def stitch_training_segments(
    segments: Sequence[tuple[str, Path]],
    *,
    terminal_segment: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify and merge chronologically ordered exact-resume metric segments.

    Later segments win replayed operational telemetry. Their scientific fields
    must be byte-equivalent after JSON decoding before any overlap is accepted.
    """

    if len(segments) < 2:
        raise ValueError("A stitch requires at least two metric segments")
    names = [name for name, _ in segments]
    if len(set(names)) != len(names):
        raise ValueError("Metric segment names must be unique")
    if terminal_segment not in names:
        raise ValueError("Terminal segment is not present in the segment list")
    if names[-1] != terminal_segment:
        raise ValueError("Terminal segment must be chronologically last")

    terminal_dir = dict(segments)[terminal_segment]
    terminal_report = _load_object(terminal_dir / "report.json")
    terminal_config = _load_object(terminal_dir / "run_config.json")
    terminal_update = int(terminal_report.get("updates", -1))
    if terminal_update < 1:
        raise ValueError("Terminal report has an invalid update count")
    terminal_partition_sha256 = _sha256_file(
        terminal_dir / "optimizer_partition.json"
    )

    merged: dict[int, dict[str, Any]] = {}
    merged_owner: dict[int, str] = {}
    overlap_updates: set[int] = set()
    overlap_semantics: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    for name, directory in segments:
        records = _load_records(directory / "metrics.jsonl")
        config = _load_object(directory / "run_config.json")
        config_differences = _validate_run_config(
            terminal_config,
            config,
            segment_name=name,
        )
        partition_sha256 = _sha256_file(directory / "optimizer_partition.json")
        if partition_sha256 != terminal_partition_sha256:
            raise ValueError(f"Optimizer partition drift in segment {name}")
        resume = config.get("resume")
        resume = resume if isinstance(resume, Mapping) else {}
        resume_enabled = resume.get("enabled") is True
        expected_start = (
            int(resume.get("optimizer_update", -1)) + 1
            if resume_enabled
            else 1
        )
        first_update = int(records[0]["update"])
        if first_update != expected_start:
            raise ValueError(
                f"Segment {name} starts at {first_update}, expected {expected_start}"
            )
        for record in records:
            update = int(record["update"])
            if update in merged:
                previous_semantic = _semantic_metric_record(merged[update])
                current_semantic = _semantic_metric_record(record)
                if previous_semantic != current_semantic:
                    changed = _difference_paths(previous_semantic, current_semantic)
                    raise ValueError(
                        f"Scientific replay drift at update {update}: "
                        f"{['.'.join(path) for path in changed[:20]]}"
                    )
                overlap_updates.add(update)
                overlap_semantics.append(current_semantic)
            merged[update] = record
            merged_owner[update] = name
        segment_records.append(
            {
                "name": name,
                "metrics_sha256": _sha256_file(directory / "metrics.jsonl"),
                "run_config_sha256": _sha256_file(directory / "run_config.json"),
                "optimizer_partition_sha256": partition_sha256,
                "record_count": len(records),
                "update_range": [
                    int(records[0]["update"]),
                    int(records[-1]["update"]),
                ],
                "resume_enabled": resume_enabled,
                "resume_update": (
                    int(resume["optimizer_update"])
                    if resume_enabled
                    else None
                ),
                "allowed_config_differences": config_differences,
            }
        )

    expected_updates = list(range(1, terminal_update + 1))
    observed_updates = sorted(merged)
    if observed_updates != expected_updates:
        missing = sorted(set(expected_updates) - set(observed_updates))
        extra = sorted(set(observed_updates) - set(expected_updates))
        raise ValueError(
            f"Stitched curve is incomplete: missing={missing[:20]}, extra={extra[:20]}"
        )
    records = [merged[update] for update in expected_updates]
    examples = [int(record["examples"]) for record in records]
    if any(right <= left for left, right in zip(examples, examples[1:])):
        raise ValueError("Stitched examples are not strictly increasing")

    from research.train_torch import _summarize_training_records

    metrics_payload = b"".join(
        json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
        for record in records
    )
    loss_summary = _summarize_training_records(records)
    loss_summary_payload = (
        json.dumps(loss_summary, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    for name in TERMINAL_COPY_FILES:
        source = terminal_dir / name
        if not source.is_file():
            raise ValueError(f"Terminal segment is missing compact artifact {name}")
        _copy_terminal_file(source, output_dir / name)
    _atomic_materialize(output_dir / "metrics.jsonl", metrics_payload)
    _atomic_materialize(output_dir / "loss_summary.json", loss_summary_payload)

    checkpoint = terminal_report.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    state = checkpoint.get("state")
    state = state if isinstance(state, Mapping) else {}
    output_hashes = {
        name: _sha256_file(output_dir / name)
        for name in (*TERMINAL_COPY_FILES, "metrics.jsonl", "loss_summary.json")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_rule": (
            "chronological_later_segment_wins_only_after_exact_nonoperational_"
            "overlap_match"
        ),
        "operational_metric_keys_ignored_in_overlap": sorted(
            OPERATIONAL_METRIC_KEYS
        ),
        "authoritative_training_curve": "metrics.jsonl",
        "authoritative_loss_summary": "loss_summary.json",
        "terminal_report_curve_scope": "terminal_segment_only_unmodified",
        "terminal_report_retained_for": [
            "validation",
            "checkpoint_provenance",
        ],
        "segments": segment_records,
        "terminal_segment": terminal_segment,
        "terminal_update": terminal_update,
        "terminal_model_state_sha256": state.get("sha256"),
        "merged_record_count": len(records),
        "merged_update_range": [1, terminal_update],
        "overlap_updates": sorted(overlap_updates),
        "overlap_record_count": len(overlap_updates),
        "overlap_semantic_sha256": _sha256_bytes(
            _canonical_json_bytes(overlap_semantics)
        ),
        "update_owner_ranges": [
            {
                "name": name,
                "updates": [
                    min(update for update, owner in merged_owner.items() if owner == name),
                    max(update for update, owner in merged_owner.items() if owner == name),
                ],
            }
            for name in names
            if name in set(merged_owner.values())
        ],
        "output_sha256": output_hashes,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    _atomic_materialize(output_dir / "stitch_manifest.json", manifest_payload)
    return manifest


def _parse_segment(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("segment must use NAME=PATH") from exc
    if not name or Path(name).name != name or not path:
        raise argparse.ArgumentTypeError("segment must use a safe NAME=PATH")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment",
        action="append",
        type=_parse_segment,
        required=True,
        help="Chronological metric segment as NAME=PATH; repeat for each segment.",
    )
    parser.add_argument("--terminal-segment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = stitch_training_segments(
        args.segment,
        terminal_segment=args.terminal_segment,
        output_dir=args.output,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
