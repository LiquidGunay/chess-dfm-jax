"""Atomic, content-addressed files for reproducible interpretation runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_target(path: Path) -> Path:
    resolved = path.resolve()
    if resolved in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"Refusing unsafe artifact path: {resolved}")
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    target = _safe_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: Any) -> None:
    rendered = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(path, rendered)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rendered = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(path, rendered)


def write_text_atomic(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def content_identity(payloads: Mapping[str, Any]) -> str:
    """Hash named logical payloads independently of pretty-print formatting."""

    return sha256_bytes(canonical_json_bytes(dict(payloads)))


def write_checksums(run_dir: Path) -> dict[str, str]:
    """Hash every retained regular file except the checksum ledger itself."""

    directory = _safe_target(run_dir)
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checksums.sha256" and not path.name.endswith(".partial")
    )
    records = {path.relative_to(directory).as_posix(): sha256_file(path) for path in files}
    lines = "".join(f"{digest}  {name}\n" for name, digest in records.items())
    write_text_atomic(directory / "checksums.sha256", lines)
    return records


def verify_checksums(run_dir: Path) -> dict[str, str]:
    directory = _safe_target(run_dir)
    ledger = directory / "checksums.sha256"
    if not ledger.is_file():
        raise ValueError("Run bundle has no checksums.sha256")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum line {line_number}") from exc
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Malformed SHA256 on line {line_number}")
        target = (directory / relative).resolve()
        try:
            target.relative_to(directory)
        except ValueError as exc:
            raise ValueError(f"Checksum path escapes run directory: {relative}") from exc
        if relative in expected:
            raise ValueError(f"Duplicate checksum path: {relative}")
        expected[relative] = digest
    actual_names = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if actual_names != set(expected):
        raise ValueError("Run bundle file inventory differs from checksum ledger")
    for relative, digest in expected.items():
        if sha256_file(directory / relative) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
    return expected
