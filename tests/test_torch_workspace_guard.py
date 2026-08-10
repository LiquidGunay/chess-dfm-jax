from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import research.train_torch as train_torch
import research.prepare as prepare


def test_trusted_resolved_volume_root_is_accepted_without_weakening_escape_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    volume_root = tmp_path / "physical-volume"
    volume_root.mkdir()
    mounted_root = tmp_path / "mounted-volume"
    mounted_root.symlink_to(volume_root, target_is_directory=True)
    result = mounted_root / "runs" / "result"

    monkeypatch.setattr(train_torch, "_WORKSPACE_ROOT", repo_root)
    monkeypatch.setattr(
        train_torch,
        "_TRUSTED_WORKSPACE_ROOTS",
        (mounted_root.resolve(strict=True),),
    )
    assert train_torch._require_workspace(result) == result.resolve()


def test_trusted_volume_root_reaches_real_fixed_trajectory_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume_root = tmp_path / "physical-data-volume"
    split = volume_root / "train"
    split.mkdir(parents=True)
    np.savez(
        split / "chunk0000.npz",
        batch_size=np.asarray(4, dtype=np.int32),
    )
    mounted_root = tmp_path / "mounted-data-volume"
    mounted_root.symlink_to(volume_root, target_is_directory=True)
    monkeypatch.setattr(
        prepare,
        "TRUSTED_WORKSPACE_ROOTS",
        (mounted_root.resolve(strict=True),),
    )

    batches = prepare.FixedTrajectoryBatches(
        mounted_root / "train",
        batch_size=2,
        horizon=1,
        seed=0,
        shuffle_files=False,
    )
    assert batches.samples_per_shard == 4
    assert batches.steps_per_epoch == 2

    with pytest.raises(ValueError, match="Path escapes trusted workspace roots"):
        prepare.require_within_workspace(tmp_path / "untrusted")
