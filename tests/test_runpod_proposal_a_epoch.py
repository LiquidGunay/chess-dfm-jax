from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

import research.infra.runpod_proposal_a_epoch as runpod_control
from research.infra.runpod_epoch_bootstrap import _load_manifest
from research.infra.runpod_proposal_a_epoch import (
    DEFAULT_PREREGISTER,
    _json_bytes,
    _sha256_file,
    build_launch_manifest,
    build_pod_create_command,
    validate_preregister,
)
from research.infra.runpod_proposal_a_worker import (
    build_training_command,
    find_latest_valid_recovery,
)


def _launch_manifest() -> dict[str, object]:
    preregister = validate_preregister()
    return build_launch_manifest(
        preregister=preregister,
        preregister_sha256=_sha256_file(DEFAULT_PREREGISTER),
        git_binding={
            "branch": "research/local-gpu-autoresearch",
            "commit": "a" * 40,
            "remote": "https://github.com/LiquidGunay/chess-dfm-jax.git",
            "upstream": "origin/research/local-gpu-autoresearch",
        },
        volume_id="volume123",
        s3_endpoint="https://s3api-us-ga-2.runpod.io/",
        created_utc="2026-08-10T20:00:00+00:00",
    )


def test_one_epoch_preregister_verifies_all_bound_evidence() -> None:
    payload = validate_preregister()

    assert payload["model_contract"]["main_learning_rate"] == 5e-5
    assert payload["model_contract"]["encoder_learning_rate_ratio"] == 1.0
    assert payload["model_contract"]["weight_decay_mode"] == "decoupled"
    assert payload["schedule_contract"]["steps"] == 7_774
    assert payload["data_contract"]["validation_scheduler"]["evaluation_examples"] == 8_192


def test_launch_manifest_embeds_a_canonical_preregister_and_no_credentials(
    tmp_path: Path,
) -> None:
    manifest = _launch_manifest()
    canonical_sha = hashlib.sha256(_json_bytes(manifest["preregister"])).hexdigest()

    assert manifest["preregister_sha256"] == canonical_sha
    assert manifest["preregister_file_sha256"] == _sha256_file(DEFAULT_PREREGISTER)
    assert manifest["network_volume"]["id"] == "volume123"
    assert manifest["paths"]["dataset"].startswith("/workspace/")
    serialized = json.dumps(manifest)
    assert "RUNPOD_API_KEY" not in serialized
    assert "SECRET_ACCESS_KEY" not in serialized

    path = tmp_path / "launch_manifest.json"
    path.write_bytes(_json_bytes(manifest))
    assert _load_manifest(path) == manifest


def test_pod_command_is_secure_l40s_volume_bound_and_absolutely_capped() -> None:
    preregister = validate_preregister()
    receipt = {
        "gpu_started": False,
        "prefix": "chess-dfm/proposal-a/v1",
        "volume_id": "volume123",
    }

    command = build_pod_create_command(
        stage_receipt=receipt,
        preregister=preregister,
        runpodctl=Path("/safe/runpodctl"),
        now=datetime(2026, 8, 10, 20, 0, tzinfo=UTC),
    )

    assert command[0] == "/safe/runpodctl"
    assert command[command.index("--cloud-type") + 1] == "SECURE"
    assert command[command.index("--gpu-id") + 1] == "NVIDIA L40S"
    assert "A100" not in command
    assert command[command.index("--network-volume-id") + 1] == "volume123"
    assert command[command.index("--terminate-after") + 1] == "2026-08-11T05:00:00Z"
    docker_args = command[command.index("--docker-args") + 1]
    assert "runpod_epoch_bootstrap.py" in docker_args
    assert "launch_manifest.json" in docker_args


def test_launch_deletes_a_created_pod_when_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_receipt = tmp_path / "stage.json"
    stage_receipt.write_text(
        json.dumps(
            {
                "gpu_started": False,
                "prefix": "chess-dfm/proposal-a/v1",
                "volume_id": "volume123",
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, ...]] = []

    def fake_runpodctl_json(
        runpodctl: Path,
        arguments: tuple[str, ...],
    ) -> dict[str, object]:
        del runpodctl
        calls.append(arguments)
        if arguments[:2] == ("pod", "create"):
            return {"id": "pod123"}
        if arguments == ("pod", "get", "pod123"):
            return {
                "id": "pod123",
                "gpu": {"displayName": "NVIDIA A100 80GB PCIe"},
                "costPerHr": "1.69",
            }
        if arguments == ("pod", "delete", "pod123"):
            return {"id": "pod123", "desiredStatus": "TERMINATED"}
        raise AssertionError(arguments)

    monkeypatch.setattr(
        runpod_control,
        "_runpodctl_json",
        fake_runpodctl_json,
    )

    with pytest.raises(RuntimeError, match="unexpected GPU"):
        runpod_control.launch(
            stage_receipt_path=stage_receipt,
            output=tmp_path / "launch.json",
            execute=True,
            runpodctl=Path("/safe/runpodctl"),
        )

    assert calls[-1] == ("pod", "delete", "pod123")
    assert not (tmp_path / "launch.json").exists()


def test_resume_command_keeps_only_future_sparse_checkpoints(tmp_path: Path) -> None:
    manifest = _launch_manifest()
    recovery = {
        "update": 6_219,
        "checkpoint_dir": "/workspace/checkpoint-6219",
    }

    command = build_training_command(
        manifest,
        dataset_root=tmp_path / "data",
        raw_bt4_path=tmp_path / "BT4.pb.gz",
        attempt_output=tmp_path / "attempt-0001",
        recovery=recovery,
    )

    save_index = command.index("--save-updates")
    assert command[save_index + 1 : save_index + 2] == ["7000"]
    assert command[save_index + 2] == "--save-final"
    assert command[command.index("--resume-checkpoint") + 1] == ("/workspace/checkpoint-6219")
    validation_index = command.index("--validation-updates")
    assert command[validation_index + 1 : validation_index + 3] == [
        "6219",
        "7774",
    ]

    after_7000 = build_training_command(
        manifest,
        dataset_root=tmp_path / "data",
        raw_bt4_path=tmp_path / "BT4.pb.gz",
        attempt_output=tmp_path / "attempt-0002",
        recovery={
            "update": 7_000,
            "checkpoint_dir": "/workspace/checkpoint-7000",
        },
    )
    assert "--save-updates" not in after_7000
    assert "--save-final" in after_7000


def _write_recovery(
    root: Path,
    *,
    attempt: int,
    update: int,
    state: bytes,
    valid_sha: bool,
) -> None:
    checkpoint = root / f"attempt-{attempt:04d}" / "checkpoints" / f"update{update:08d}"
    checkpoint.mkdir(parents=True)
    state_path = checkpoint / "state.safetensors"
    state_path.write_bytes(state)
    digest = hashlib.sha256(state).hexdigest()
    if not valid_sha:
        digest = "0" * 64
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "format": "chess-dfm-torch-training-v2",
                "model_only": False,
                "optimizer_resume_supported": True,
                "optimizer_update": update,
                "state": {
                    "path": state_path.name,
                    "size_bytes": len(state),
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )


def test_recovery_selection_skips_a_newer_checksum_invalid_state(
    tmp_path: Path,
) -> None:
    _write_recovery(
        tmp_path,
        attempt=0,
        update=1_000,
        state=b"old-valid",
        valid_sha=True,
    )
    _write_recovery(
        tmp_path,
        attempt=1,
        update=2_500,
        state=b"new-valid",
        valid_sha=True,
    )
    _write_recovery(
        tmp_path,
        attempt=2,
        update=4_000,
        state=b"partial",
        valid_sha=False,
    )

    selected = find_latest_valid_recovery(tmp_path)

    assert selected is not None
    assert selected["update"] == 2_500
    assert selected["state_sha256"] == hashlib.sha256(b"new-valid").hexdigest()
