from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.prepare import (
    ASSET_MANIFEST_PATH,
    REPO_ROOT,
    WORKSPACE_ROOT,
    WorkspacePaths,
    load_asset_manifest,
    require_within_workspace,
)


def test_repository_is_inside_workspace() -> None:
    assert require_within_workspace(REPO_ROOT) == REPO_ROOT
    assert REPO_ROOT.is_relative_to(WORKSPACE_ROOT)


def test_workspace_paths_are_inside_workspace() -> None:
    paths = WorkspacePaths.from_repo()
    for path in (
        paths.repo,
        paths.local,
        paths.data,
        paths.models,
        paths.checkpoints,
        paths.artifacts,
        paths.runs,
    ):
        assert path.is_relative_to(WORKSPACE_ROOT)


def test_path_guard_rejects_external_paths() -> None:
    with pytest.raises(ValueError, match="escapes"):
        require_within_workspace(Path("/tmp/chess-dfm-jax"))


def test_asset_manifest_has_immutable_digests() -> None:
    manifest = load_asset_manifest()
    assert manifest["trajectory_v3"]["archive"]["size_bytes"] == 11_531_386_880
    assert len(manifest["trajectory_v3"]["archive"]["sha256"]) == 64
    assert manifest["models"]["archive"]["size_bytes"] == 1_459_712_000
    assert len(manifest["models"]["archive"]["sha256"]) == 64
    assert manifest["checkpoint_step_265000"]["part_count"] == 22
    assert len(manifest["checkpoint_step_265000"]["archive_sha256"]) == 64

    parsed_directly = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert parsed_directly == manifest
