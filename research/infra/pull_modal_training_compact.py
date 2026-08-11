"""Atomically pull only compact, terminal Modal training evidence.

This intentionally refuses checkpoint directories and incomplete runs.  It is
the local storage boundary for the Hero-v2 search: model and optimizer states
remain on the remote Volume while the small sufficient statistics are hashed
and retained in the workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_MODAL_BIN = "/root/.local/bin/modal"
DEFAULT_ENVIRONMENT = "chess-dfm-research"
DEFAULT_VOLUME = "chess-dfm-training-results"
REQUIRED_COMPACT_FILES = (
    "metrics.jsonl",
    "run_config.json",
    "report.json",
    "modal_run.json",
    "loss_summary.json",
    "optimizer_partition.json",
)
OPTIONAL_COMPACT_FILES = ("validation_metrics.jsonl",)
COMPACT_FILES = REQUIRED_COMPACT_FILES + OPTIONAL_COMPACT_FILES


def require_result_label(value: str) -> str:
    safe = all(
        character.islower() or character.isdigit() or character in "-_"
        for character in value
    )
    if (
        not value.startswith("hero-training-")
        or not 1 <= len(value) <= 200
        or Path(value).name != value
        or not safe
    ):
        raise ValueError("Unsafe Hero training result label")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_volume_inventory(
    payload: str, *, result_label: str
) -> dict[str, int | None]:
    try:
        records = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Modal Volume inventory is not valid JSON") from exc
    if not isinstance(records, list):
        raise ValueError("Modal Volume inventory must be a list")

    prefix = f"runs/{result_label}/"
    inventory: dict[str, int | None] = {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("type") != "file":
            continue
        remote_name = record.get("filename")
        if not isinstance(remote_name, str) or not remote_name.startswith(prefix):
            continue
        name = remote_name.removeprefix(prefix)
        if "/" in name or name not in COMPACT_FILES:
            continue
        raw_size = record.get("size")
        if isinstance(raw_size, int):
            inventory[name] = raw_size
        elif isinstance(raw_size, str):
            # Modal CLI 1.1 currently emits display sizes such as "17.7 KiB"
            # even under --json. Presence remains authoritative; downloaded
            # bytes are validated structurally and content-hashed below.
            inventory[name] = None

    missing = sorted(set(REQUIRED_COMPACT_FILES) - set(inventory))
    if missing:
        raise ValueError(
            f"Modal training result is incomplete; missing compact files: {missing}"
        )
    return inventory


def validate_compact_run(
    directory: Path, *, expected_sizes: Mapping[str, int | None]
) -> None:
    missing = [
        name for name in REQUIRED_COMPACT_FILES if not (directory / name).is_file()
    ]
    if missing:
        raise ValueError(f"Compact result is missing required files: {missing}")
    for name, expected_size in expected_sizes.items():
        if name not in COMPACT_FILES:
            raise ValueError(f"Unexpected compact result file: {name}")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Compact result is missing {name}")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"Downloaded size differs from Volume inventory for {name}")

    try:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        modal_run = json.loads((directory / "modal_run.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Compact result has invalid terminal JSON") from exc
    if not isinstance(report, Mapping) or not isinstance(modal_run, Mapping):
        raise ValueError("Terminal report and Modal record must be JSON objects")

    updates: list[int] = []
    with (directory / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                update = int(record["update"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed training metric at line {line_number}") from exc
            updates.append(update)
    if not updates or updates != list(range(1, updates[-1] + 1)):
        raise ValueError("Training metrics must contain contiguous updates starting at one")
    if int(report.get("updates", -1)) != updates[-1]:
        raise ValueError("Terminal report update does not match compact metrics")

    validation_path = directory / "validation_metrics.jsonl"
    validation_lines = (
        [
            line
            for line in validation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if validation_path.is_file()
        else []
    )
    profile = modal_run.get("profile")
    validation_optional_profiles = {"smoke", "benchmark", "diagnostic"}
    if not validation_lines and profile not in validation_optional_profiles:
        raise ValueError("Terminal run has no validation metrics")


def _run_checked(command: Sequence[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Modal command exceeded {timeout_seconds:g} seconds") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise RuntimeError(
            f"Modal command exited with status {completed.returncode}: {detail}"
        )
    return completed


def pull_compact_result(
    *,
    result_label: str,
    output_dir: Path,
    modal_bin: str = DEFAULT_MODAL_BIN,
    environment: str = DEFAULT_ENVIRONMENT,
    volume: str = DEFAULT_VOLUME,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    result_label = require_result_label(result_label)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    remote_root = f"runs/{result_label}"
    inventory_result = _run_checked(
        (
            modal_bin,
            "volume",
            "ls",
            "--env",
            environment,
            "--json",
            volume,
            remote_root,
        ),
        timeout_seconds=timeout_seconds,
    )
    inventory = parse_volume_inventory(
        inventory_result.stdout,
        result_label=result_label,
    )
    download_names = tuple(name for name in COMPACT_FILES if name in inventory)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-",
            dir=output_dir.parent,
        )
    )
    try:
        for name in download_names:
            _run_checked(
                (
                    modal_bin,
                    "volume",
                    "get",
                    "--env",
                    environment,
                    "--force",
                    volume,
                    f"{remote_root}/{name}",
                    str(temporary / name),
                ),
                timeout_seconds=timeout_seconds,
            )
        validate_compact_run(temporary, expected_sizes=inventory)
        files = {
            name: {
                "sha256": sha256_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in download_names
        }
        manifest = {
            "schema_version": "chess-dfm-modal-training-compact-pull-v1",
            "result_label": result_label,
            "environment": environment,
            "volume": volume,
            "remote_root": remote_root,
            "checkpoint_files_downloaded": 0,
            "files": files,
            "total_size_bytes": sum(record["size_bytes"] for record in files.values()),
        }
        (temporary / "compact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_label")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--modal-bin", default=DEFAULT_MODAL_BIN)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    manifest = pull_compact_result(
        result_label=args.result_label,
        output_dir=args.output_dir,
        modal_bin=args.modal_bin,
        environment=args.environment,
        volume=args.volume,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
