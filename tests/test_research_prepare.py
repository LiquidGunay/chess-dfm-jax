from __future__ import annotations

import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import pytest

from chess_dfm_jax.data.trajectory import build_synthetic_trajectory_shard
import chess_dfm_jax.data.trajectory_v3 as trajectory_v3_module
from chess_dfm_jax.data.trajectory_v3 import (
    trajectory_v3_from_v2_npz,
    trajectory_v3_to_batch,
)
import research.prepare as prepare
from research.prepare import (
    ASSET_MANIFEST_PATH,
    FixedTrajectoryBatches,
    REPO_ROOT,
    WORKSPACE_ROOT,
    WorkspacePaths,
    validate_runtime_packages,
    load_asset_manifest,
    require_within_workspace,
)


def write_trajectory_split(
    tmp_path: Path,
    monkeypatch,
    *,
    shard_count: int = 2,
    samples_per_shard: int = 6,
    horizon: int = 4,
) -> Path:
    split_dir = tmp_path / (
        f"trajectory-split-{len(list(tmp_path.glob('trajectory-split-*'))):03d}"
    )
    split_dir.mkdir()
    for shard_index in range(shard_count):
        shard = build_synthetic_trajectory_shard(
            batch_size=samples_per_shard,
            horizon=horizon,
        )
        payload = {
            "schema_version": np.asarray("trajectory-v2"),
            "planes_t": shard.planes_t,
            "actions": shard.actions,
            "planes_future": shard.planes_future,
            "future_valid": shard.future_valid,
            "legal_masks": shard.legal_masks,
            "value_targets": (
                shard.value_targets
                + np.float32(shard_index)
            ),
            "wdl_targets": shard.wdl_targets,
        }
        compact = trajectory_v3_from_v2_npz(
            payload,
            legal_lmax=128,
        )
        np.savez_compressed(
            split_dir / f"shard-{shard_index:03d}.npz",
            **compact,
        )
    monkeypatch.setattr(
        prepare,
        "require_within_workspace",
        lambda path: Path(path).resolve(),
    )
    return split_dir


def full_decoded_shard(
    path: Path,
    *,
    horizon: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return trajectory_v3_to_batch(
            payload,
            view="joint_latent_sasa",
            horizon=horizon,
            include_metadata=False,
        )


def assert_batch_equal(
    observed: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    assert observed.keys() == expected.keys()
    for key in observed:
        np.testing.assert_array_equal(
            observed[key],
            expected[key],
            err_msg=key,
        )


def make_batch_schedule(
    tmp_path: Path,
    monkeypatch,
    *,
    seed: int,
    shuffle_files: bool,
    batch_schedule: str = "shard_major",
    shard_count: int = 5,
    samples_per_shard: int = 8,
    batch_size: int = 2,
) -> FixedTrajectoryBatches:
    split_dir = tmp_path / (
        f"split-{len(list(tmp_path.glob('split-*'))):03d}"
    )
    split_dir.mkdir()
    for shard_index in range(shard_count):
        np.savez(
            split_dir / f"shard-{shard_index:03d}.npz",
            batch_size=np.asarray(
                samples_per_shard,
                dtype=np.int32,
            ),
        )
    monkeypatch.setattr(
        prepare,
        "require_within_workspace",
        lambda path: Path(path).resolve(),
    )
    schedule = FixedTrajectoryBatches(
        split_dir,
        batch_size=batch_size,
        horizon=2,
        seed=seed,
        shuffle_files=shuffle_files,
        batch_schedule=batch_schedule,
    )
    path_to_index = {
        path: index
        for index, path in enumerate(schedule.paths)
    }

    def load_identity_shard(path: Path) -> dict[str, np.ndarray]:
        shard_index = path_to_index[path]
        return {
            "identity": (
                shard_index * samples_per_shard
                + np.arange(samples_per_shard, dtype=np.int32)
            )
        }

    monkeypatch.setattr(
        schedule,
        "_load_shard",
        load_identity_shard,
    )

    def decode_identity_rows(
        path: Path,
        *,
        row_slice: slice | None,
    ) -> dict[str, np.ndarray]:
        batch = load_identity_shard(path)
        if row_slice is None:
            return batch
        return {
            key: value[row_slice]
            for key, value in batch.items()
        }

    monkeypatch.setattr(
        schedule,
        "_decode_shard",
        decode_identity_rows,
    )
    return schedule


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


def test_runtime_package_guard_fails_closed(monkeypatch) -> None:
    real_version = importlib.metadata.version

    def drifted_version(package: str) -> str:
        if package == "jax":
            return "999.0.0"
        return real_version(package)

    monkeypatch.setattr(importlib.metadata, "version", drifted_version)
    with pytest.raises(RuntimeError, match="jax: expected 0.10.1, found 999.0.0"):
        validate_runtime_packages()


def test_shard_major_default_preserves_legacy_schedule(
    tmp_path,
    monkeypatch,
) -> None:
    schedule = make_batch_schedule(
        tmp_path,
        monkeypatch,
        seed=17,
        shuffle_files=True,
    )
    assert schedule.batch_schedule == "shard_major"
    expected_shards = list(range(len(schedule.paths)))
    random.Random(17).shuffle(expected_shards)
    expected = [
        (shard_index, batch_in_shard)
        for shard_index in expected_shards
        for batch_in_shard in range(schedule.batches_per_shard)
    ]
    observed = [
        schedule._slot_for_step(step)
        for step in range(schedule.steps_per_epoch)
    ]

    assert observed == expected
    provenance = schedule.provenance()
    assert provenance["batch_schedule"] == "shard_major"
    assert provenance["seed_effective"] is True

    unshuffled = make_batch_schedule(
        tmp_path,
        monkeypatch,
        seed=999,
        shuffle_files=False,
    )
    assert [
        unshuffled._slot_for_step(step)
        for step in range(unshuffled.steps_per_epoch)
    ] == [
        (shard_index, batch_in_shard)
        for shard_index in range(len(unshuffled.paths))
        for batch_in_shard in range(
            unshuffled.batches_per_shard
        )
    ]
    assert unshuffled.provenance()["seed_effective"] is False


def test_global_permutation_has_exact_epoch_coverage(
    tmp_path,
    monkeypatch,
) -> None:
    schedule = make_batch_schedule(
        tmp_path,
        monkeypatch,
        seed=23,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    observed = [
        schedule._slot_for_step(step)
        for step in range(schedule.steps_per_epoch)
    ]
    expected = {
        (shard_index, batch_in_shard)
        for shard_index in range(len(schedule.paths))
        for batch_in_shard in range(schedule.batches_per_shard)
    }

    assert len(observed) == schedule.steps_per_epoch
    assert len(set(observed)) == schedule.steps_per_epoch
    assert set(observed) == expected
    assert len({shard for shard, _ in observed[:8]}) > 1
    assert len({block for _, block in observed[:8]}) > 1
    provenance = schedule.provenance()
    assert provenance["batch_schedule"] == "global_permutation"
    assert provenance["shuffle_files"] is True
    assert provenance["seed_effective"] is True


def test_global_permutation_is_stateless_and_seeded(
    tmp_path,
    monkeypatch,
) -> None:
    first = make_batch_schedule(
        tmp_path,
        monkeypatch,
        seed=31,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    restarted = make_batch_schedule(
        tmp_path,
        monkeypatch,
        seed=31,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    different_seed = make_batch_schedule(
        tmp_path,
        monkeypatch,
        seed=32,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    query_steps = [
        7,
        0,
        first.steps_per_epoch + 3,
        11,
        first.steps_per_epoch - 1,
        2,
    ]

    for step in query_steps:
        np.testing.assert_array_equal(
            first.batch_at(step)["identity"],
            restarted.batch_at(step)["identity"],
        )
        assert (
            first._slot_for_step(step)
            == restarted._slot_for_step(step)
        )
    first_epoch = [
        first._slot_for_step(step)
        for step in range(first.steps_per_epoch)
    ]
    restarted_epoch = [
        restarted._slot_for_step(step)
        for step in range(restarted.steps_per_epoch)
    ]
    different_epoch = [
        different_seed._slot_for_step(step)
        for step in range(different_seed.steps_per_epoch)
    ]
    assert first_epoch == restarted_epoch
    assert first_epoch != different_epoch


def test_global_permutation_invalid_combinations_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    with pytest.raises(
        ValueError,
        match="global_permutation.*requires shuffle_files=True",
    ):
        make_batch_schedule(
            tmp_path,
            monkeypatch,
            seed=1,
            shuffle_files=False,
            batch_schedule="global_permutation",
        )
    with pytest.raises(ValueError, match="batch_schedule must be"):
        make_batch_schedule(
            tmp_path,
            monkeypatch,
            seed=1,
            shuffle_files=True,
            batch_schedule="not-a-schedule",
        )


def test_shard_major_default_decodes_and_caches_one_full_shard(
    tmp_path,
    monkeypatch,
) -> None:
    split_dir = write_trajectory_split(tmp_path, monkeypatch)
    schedule = FixedTrajectoryBatches(
        split_dir,
        batch_size=2,
        horizon=3,
        seed=7,
        shuffle_files=False,
    )
    real_decoder = trajectory_v3_module.trajectory_v3_to_batch
    row_slices = []

    def record_decoder(data, **kwargs):
        row_slices.append(kwargs.get("row_slice"))
        return real_decoder(data, **kwargs)

    monkeypatch.setattr(
        trajectory_v3_module,
        "trajectory_v3_to_batch",
        record_decoder,
    )

    full = full_decoded_shard(schedule.paths[0], horizon=3)
    first = schedule.batch_at(0)
    second = schedule.batch_at(1)

    assert schedule.batch_schedule == "shard_major"
    assert row_slices == [None]
    assert schedule._cached_path == schedule.paths[0]
    assert schedule._cached_batch is not None
    assert_batch_equal(
        first,
        {key: value[0:2] for key, value in full.items()},
    )
    assert_batch_equal(
        second,
        {key: value[2:4] for key, value in full.items()},
    )


def test_global_permutation_decodes_only_requested_rows(
    tmp_path,
    monkeypatch,
) -> None:
    split_dir = write_trajectory_split(tmp_path, monkeypatch)
    schedule = FixedTrajectoryBatches(
        split_dir,
        batch_size=2,
        horizon=3,
        seed=19,
        shuffle_files=True,
        batch_schedule="global_permutation",
    )
    real_decoder = trajectory_v3_module.trajectory_v3_to_batch
    row_slices = []

    def record_decoder(data, **kwargs):
        row_slices.append(kwargs.get("row_slice"))
        return real_decoder(data, **kwargs)

    monkeypatch.setattr(
        trajectory_v3_module,
        "trajectory_v3_to_batch",
        record_decoder,
    )

    for step in (0, 1):
        shard_index, batch_in_shard = schedule._slot_for_step(step)
        start = batch_in_shard * schedule.batch_size
        end = start + schedule.batch_size
        full = full_decoded_shard(
            schedule.paths[shard_index],
            horizon=3,
        )
        observed = schedule.batch_at(step)
        assert_batch_equal(
            observed,
            {
                key: value[start:end]
                for key, value in full.items()
            },
        )

    assert row_slices == [
        slice(
            schedule._slot_for_step(step)[1] * schedule.batch_size,
            (
                schedule._slot_for_step(step)[1] + 1
            )
            * schedule.batch_size,
        )
        for step in (0, 1)
    ]
    assert schedule._cached_path is None
    assert schedule._cached_batch is None
