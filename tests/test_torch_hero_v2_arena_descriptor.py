from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from research.evaluate_arena import (
    _canonical_json_bytes,
    torch_hero_checkpoint_descriptor,
)
from research.prepare import REPO_ROOT
from research.train_torch import HERO_CONFIG


@pytest.fixture
def workspace_tmp() -> Path:
    root = REPO_ROOT / ".local" / "tmp" / "torch-arena-descriptor-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def test_torch_hero_v2_descriptor_binds_run_and_recovery_recipe(
    workspace_tmp: Path,
) -> None:
    models_dir = workspace_tmp / "models"
    models_dir.mkdir()
    (models_dir / "BT4_exported.pb.gz").write_bytes(b"fixture-bt4")
    run_root = workspace_tmp / "torch-hero-v2-run"
    checkpoint_dir = run_root / "checkpoints" / "update00000554"
    checkpoint_dir.mkdir(parents=True)
    state_path = checkpoint_dir / "state.safetensors"
    state_path.write_bytes(b"fixture-hero-v2-recovery-state")
    expected_config = dataclasses.asdict(
        dataclasses.replace(
            HERO_CONFIG,
            remat_bt4_blocks=True,
            remat_projector_blocks=True,
            remat_dfm_blocks=False,
            use_bt4_sdpa=True,
            use_head_sdpa=True,
        )
    )
    compile_regions = [
        "state_projector_blocks",
        "dfm_blocks",
        "jepa_transition",
    ]
    git_commit = "d" * 40
    resume_contract = {
        "schema_version": "torch-training-resume-contract-v1",
        "framework": "torch",
        "recipe": "hero_v2",
        "git_commit": git_commit,
        "config": expected_config,
        "runtime": {"compiled_regions": compile_regions},
    }
    manifest = {
        "format": "chess-dfm-torch-training-v1",
        "model_only": False,
        "optimizer_resume_supported": True,
        "optimizer_update": 554,
        "source_mapping_sha256": "a" * 64,
        "resume_contract": resume_contract,
        "resume_contract_sha256": hashlib.sha256(
            _canonical_json_bytes(resume_contract)
        ).hexdigest(),
        "state": {
            "path": "state.safetensors",
            "size_bytes": state_path.stat().st_size,
            "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            "model_leaf_count": 462,
        },
    }
    manifest_path = checkpoint_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_config = {
        "framework": "torch",
        "execution": "regional-compile",
        "torch_compile": True,
        "recipe": "hero_v2",
        "git_commit": git_commit,
        "compile_regions": compile_regions,
        "config": expected_config,
    }
    run_config_path = run_root / "run_config.json"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    descriptor = torch_hero_checkpoint_descriptor(
        checkpoint_dir,
        models_dir=models_dir,
    )

    assert descriptor.descriptor["lineage"]["recipe"] == "hero_v2"
    assert descriptor.descriptor["torch_run_config"]["recipe"] == "hero_v2"

    resume_contract["recipe"] = "hero"
    manifest["resume_contract"] = resume_contract
    manifest["resume_contract_sha256"] = hashlib.sha256(
        _canonical_json_bytes(resume_contract)
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="Torch hero recovery checkpoint training contract mismatch",
    ):
        torch_hero_checkpoint_descriptor(
            checkpoint_dir,
            models_dir=models_dir,
        )

    run_config["recipe"] = "baseline"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="Torch hero run execution contract mismatch",
    ):
        torch_hero_checkpoint_descriptor(
            checkpoint_dir,
            models_dir=models_dir,
        )
