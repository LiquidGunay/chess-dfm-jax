"""Audit the workspace against the frozen local-GPU storage contract.

This command is deliberately read-only. It fails on missing required assets,
unselected checkpoint state, restored source archives, dataset drift, or cache
growth beyond the declared budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.prepare import REPO_ROOT, require_within_workspace  # noqa: E402


RETENTION_MANIFEST_PATH = Path(__file__).with_name("storage_retention.json")


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Repository path must be a non-empty string: {raw_path!r}")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"Repository path must be relative: {raw_path}")
    root = repo_root.resolve()
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Repository path escapes {root}: {raw_path}") from exc
    return resolved


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for directory, _, filenames in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            total += (directory_path / filename).stat(follow_symlinks=False).st_size
    return total


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def audit_storage(
    *,
    manifest_path: Path = RETENTION_MANIFEST_PATH,
    repo_root: Path = REPO_ROOT,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable report for the exact-retention contract."""

    repo_root = repo_root.resolve()
    manifest = _load_json(manifest_path.resolve())
    errors: list[str] = []
    report: dict[str, Any] = {
        "cache_usage_bytes": {},
        "dataset": {},
        "errors": errors,
        "manifest": str(manifest_path.resolve()),
        "repo_root": str(repo_root),
        "required_file_count": 0,
        "schema_version": manifest.get("schema_version"),
        "state_files": {},
        "verify_hashes": bool(verify_hashes),
    }

    if manifest.get("schema_version") != 1:
        errors.append(
            f"Unsupported retention schema: expected 1, found {manifest.get('schema_version')!r}"
        )

    required_entries = manifest.get("required_files")
    if not isinstance(required_entries, list):
        errors.append("required_files must be a list")
        required_entries = []
    required_paths: set[str] = set()
    report["required_file_count"] = len(required_entries)
    for index, entry in enumerate(required_entries):
        if not isinstance(entry, dict):
            errors.append(f"required_files[{index}] must be an object")
            continue
        raw_path = entry.get("path")
        try:
            path = _resolve_repo_path(repo_root, raw_path)
        except (TypeError, ValueError) as exc:
            errors.append(f"required_files[{index}]: {exc}")
            continue
        required_paths.add(raw_path)
        if not path.is_file():
            errors.append(f"Missing required file: {raw_path}")
            continue
        expected_size = entry.get("size_bytes")
        observed_size = path.stat().st_size
        if not isinstance(expected_size, int) or observed_size != expected_size:
            errors.append(
                f"Size mismatch for {raw_path}: expected {expected_size!r}, "
                f"found {observed_size}"
            )
        if verify_hashes:
            expected_sha256 = entry.get("sha256")
            observed_sha256 = _sha256_file(path)
            if observed_sha256 != expected_sha256:
                errors.append(
                    f"SHA-256 mismatch for {raw_path}: expected {expected_sha256!r}, "
                    f"found {observed_sha256}"
                )

    allowed_raw = manifest.get("allowed_state_files")
    if not isinstance(allowed_raw, list) or not all(
        isinstance(path, str) for path in allowed_raw
    ):
        errors.append("allowed_state_files must be a list of paths")
        allowed_states: set[str] = set()
    else:
        allowed_states = set(allowed_raw)
    unpinned_allowed_states = sorted(allowed_states - required_paths)
    for raw_path in unpinned_allowed_states:
        errors.append(f"Allowed state is not a pinned required file: {raw_path}")

    observed_states: set[str] = set()
    state_roots = manifest.get("state_search_roots")
    if not isinstance(state_roots, list):
        errors.append("state_search_roots must be a list")
        state_roots = []
    for raw_root in state_roots:
        try:
            root = _resolve_repo_path(repo_root, raw_root)
        except (TypeError, ValueError) as exc:
            errors.append(f"Invalid state search root: {exc}")
            continue
        if not root.exists():
            continue
        for state_path in root.rglob("state.npz"):
            observed_states.add(state_path.relative_to(repo_root).as_posix())
    missing_states = sorted(allowed_states - observed_states)
    extra_states = sorted(observed_states - allowed_states)
    for raw_path in missing_states:
        errors.append(f"Missing allowed state: {raw_path}")
    for raw_path in extra_states:
        errors.append(f"Unselected checkpoint state is still present: {raw_path}")
    report["state_files"] = {
        "allowed": sorted(allowed_states),
        "extra": extra_states,
        "missing": missing_states,
        "observed": sorted(observed_states),
    }

    absent_entries = manifest.get("must_be_absent")
    if not isinstance(absent_entries, list):
        errors.append("must_be_absent must be a list")
        absent_entries = []
    for raw_path in absent_entries:
        try:
            path = _resolve_repo_path(repo_root, raw_path)
        except (TypeError, ValueError) as exc:
            errors.append(f"Invalid must_be_absent path: {exc}")
            continue
        if path.exists() or path.is_symlink():
            errors.append(f"Redundant archive or staging path is present: {raw_path}")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("dataset must be an object")
    else:
        try:
            dataset_root = _resolve_repo_path(repo_root, dataset.get("root"))
            dataset_manifest_path = _resolve_repo_path(
                repo_root, dataset.get("manifest_path")
            )
            dataset_manifest = _load_json(dataset_manifest_path)
            expected_chunks = dataset.get("chunks_written")
            if not isinstance(expected_chunks, dict):
                raise TypeError("dataset.chunks_written must be an object")
            observed_chunks: dict[str, int] = {}
            observed_payload_size = 0
            for split, expected_count in expected_chunks.items():
                split_paths = sorted((dataset_root / split).glob("*.npz"))
                observed_chunks[split] = len(split_paths)
                observed_payload_size += sum(path.stat().st_size for path in split_paths)
                if len(split_paths) != expected_count:
                    errors.append(
                        f"Dataset shard-count mismatch for {split}: "
                        f"expected {expected_count}, found {len(split_paths)}"
                    )
            observed_split_dirs = sorted(
                path.name for path in dataset_root.iterdir() if path.is_dir()
            )
            if observed_split_dirs != sorted(expected_chunks):
                errors.append(
                    "Dataset split mismatch: "
                    f"expected {sorted(expected_chunks)}, found {observed_split_dirs}"
                )
            expected_payload_size = dataset.get("payload_size_bytes")
            if observed_payload_size != expected_payload_size:
                errors.append(
                    "Dataset payload-size mismatch: "
                    f"expected {expected_payload_size}, found {observed_payload_size}"
                )
            expected_schema = dataset.get("schema_version")
            observed_schema = dataset_manifest.get("schema_version")
            if observed_schema != expected_schema:
                errors.append(
                    f"Dataset schema mismatch: expected {expected_schema!r}, "
                    f"found {observed_schema!r}"
                )
            if dataset_manifest.get("chunks_written") != expected_chunks:
                errors.append("Dataset manifest chunk counts do not match retention contract")
            if dataset_manifest.get("output_size_bytes") != expected_payload_size:
                errors.append("Dataset manifest payload size does not match retention contract")
            report["dataset"] = {
                "chunks_written": observed_chunks,
                "payload_size_bytes": observed_payload_size,
                "schema_version": observed_schema,
            }
        except (FileNotFoundError, NotADirectoryError, TypeError, ValueError) as exc:
            errors.append(f"Dataset audit failed: {exc}")

    cache_budgets = manifest.get("cache_budgets_bytes")
    if not isinstance(cache_budgets, dict):
        errors.append("cache_budgets_bytes must be an object")
        cache_budgets = {}
    for raw_path, budget in cache_budgets.items():
        try:
            path = _resolve_repo_path(repo_root, raw_path)
        except (TypeError, ValueError) as exc:
            errors.append(f"Invalid cache path: {exc}")
            continue
        observed_size = _directory_size_bytes(path)
        report["cache_usage_bytes"][raw_path] = {
            "budget": budget,
            "observed": observed_size,
        }
        if not isinstance(budget, int) or observed_size > budget:
            errors.append(
                f"Cache budget exceeded for {raw_path}: "
                f"budget {budget!r}, found {observed_size}"
            )

    report["ok"] = not errors
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Read and SHA-256 every pinned file; the default checks structure and sizes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_within_workspace(REPO_ROOT)
    report = audit_storage(verify_hashes=args.verify_hashes)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
