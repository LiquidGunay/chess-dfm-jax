from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.interpretability.artifacts import (
    content_identity,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


def test_run_bundle_writes_and_verifies_complete_inventory(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "manifest.json", {"b": 2, "a": 1})
    write_jsonl_atomic(tmp_path / "examples.jsonl", [{"row": 1}, {"row": 2}])
    write_text_atomic(tmp_path / "run.log", "complete\n")
    checksums = write_checksums(tmp_path)

    assert set(checksums) == {"examples.jsonl", "manifest.json", "run.log"}
    assert verify_checksums(tmp_path) == checksums
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"a": 1, "b": 2}
    assert content_identity({"b": 2, "a": 1}) == content_identity({"a": 1, "b": 2})


def test_bundle_verifier_rejects_drift_and_untracked_files(tmp_path: Path) -> None:
    write_text_atomic(tmp_path / "result.txt", "first")
    write_checksums(tmp_path)
    write_text_atomic(tmp_path / "result.txt", "second")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(tmp_path)

    write_text_atomic(tmp_path / "result.txt", "first")
    write_checksums(tmp_path)
    write_text_atomic(tmp_path / "extra.txt", "untracked")
    with pytest.raises(ValueError, match="inventory"):
        verify_checksums(tmp_path)
