from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research.storage_audit import audit_storage


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_contract(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    dataset_root = repo_root / "data" / "trajectory_v3"
    shard_payloads = {
        "train": [b"train-0", b"train-1"],
        "val": [b"val-0"],
        "test": [b"test-0"],
    }
    payload_size = 0
    for split, payloads in shard_payloads.items():
        for index, payload in enumerate(payloads):
            _write(dataset_root / split / f"chunk_{index:06d}.npz", payload)
            payload_size += len(payload)

    chunks_written = {
        split: len(payloads) for split, payloads in shard_payloads.items()
    }
    dataset_manifest = {
        "chunks_written": chunks_written,
        "output_size_bytes": payload_size,
        "schema_version": "trajectory-v3",
    }
    dataset_manifest_payload = (
        json.dumps(dataset_manifest, sort_keys=True).encode("utf-8") + b"\n"
    )
    _write(dataset_root / "manifest.json", dataset_manifest_payload)

    state_paths = [
        "checkpoints/source/checkpoints/step/state.npz",
        "research/runs/control/checkpoints/update/state.npz",
        "research/runs/corrected/checkpoints/update/state.npz",
    ]
    required_files = [
        {
            "path": "data/trajectory_v3/manifest.json",
            "role": "trajectory_manifest",
            "sha256": _sha256(dataset_manifest_payload),
            "size_bytes": len(dataset_manifest_payload),
        }
    ]
    for index, raw_path in enumerate(state_paths):
        payload = f"state-{index}".encode("utf-8")
        _write(repo_root / raw_path, payload)
        required_files.append(
            {
                "path": raw_path,
                "role": f"state_{index}",
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )

    contract = {
        "allowed_state_files": state_paths,
        "cache_budgets_bytes": {".local/cache/jax": 1024},
        "dataset": {
            "chunks_written": chunks_written,
            "manifest_path": "data/trajectory_v3/manifest.json",
            "payload_size_bytes": payload_size,
            "root": "data/trajectory_v3",
            "schema_version": "trajectory-v3",
        },
        "must_be_absent": ["data/source/archive.tar"],
        "required_files": required_files,
        "schema_version": 1,
        "state_search_roots": ["checkpoints", "research/runs"],
    }
    contract_path = repo_root / "research" / "storage_retention.json"
    _write(
        contract_path,
        json.dumps(contract, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return repo_root, contract_path


def test_storage_audit_accepts_exact_contract(tmp_path: Path) -> None:
    repo_root, contract_path = _make_contract(tmp_path)

    report = audit_storage(
        manifest_path=contract_path,
        repo_root=repo_root,
        verify_hashes=True,
    )

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["dataset"]["chunks_written"] == {
        "test": 1,
        "train": 2,
        "val": 1,
    }
    assert len(report["state_files"]["observed"]) == 3


def test_storage_audit_rejects_extra_state_archive_and_cache(tmp_path: Path) -> None:
    repo_root, contract_path = _make_contract(tmp_path)
    _write(
        repo_root / "research/runs/rejected/checkpoints/update/state.npz",
        b"unselected",
    )
    _write(
        repo_root
        / "research/runs/rejected/checkpoints/update/state.safetensors",
        b"unselected-recovery",
    )
    _write(repo_root / "data/source/archive.tar", b"redundant")
    _write(repo_root / ".local/cache/jax/oversized", b"x" * 1025)

    report = audit_storage(
        manifest_path=contract_path,
        repo_root=repo_root,
    )

    assert report["ok"] is False
    assert any("Unselected checkpoint state" in error for error in report["errors"])
    assert len(report["state_files"]["extra"]) == 2
    assert any("Redundant archive" in error for error in report["errors"])
    assert any("Cache budget exceeded" in error for error in report["errors"])
