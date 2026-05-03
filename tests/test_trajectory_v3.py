from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.data.trajectory import build_synthetic_trajectory_shard  # noqa: E402
from chess_dfm_jax.data.trajectory import trajectory_joint_batch_from_npz  # noqa: E402
from chess_dfm_jax.data.leela import LeelaChunkDataLoader  # noqa: E402
from chess_dfm_jax.data.grain_loader import (  # noqa: E402
    GrainUnavailableError,
    create_grain_trajectory_loader,
    grain_available,
)
from chess_dfm_jax.data.trajectory_v3 import (  # noqa: E402
    legal_indices_to_masks,
    legal_masks_to_indices,
    pack_planes,
    trajectory_v3_from_v2_npz,
    trajectory_v3_to_batch,
    unpack_planes,
)


def test_pack_planes_roundtrip():
    shard = build_synthetic_trajectory_shard(batch_size=2, horizon=4)
    packed = pack_planes(shard.planes_t)
    assert packed.shape == (2, 896)
    restored = unpack_planes(packed)
    np.testing.assert_allclose(restored, shard.planes_t)


def test_legal_indices_roundtrip():
    shard = build_synthetic_trajectory_shard(batch_size=2, horizon=4)
    legal_idx, legal_count = legal_masks_to_indices(shard.legal_masks, lmax=128)
    assert legal_idx.shape == (2, 4, 128)
    assert legal_count.shape == (2, 4)
    restored = legal_indices_to_masks(legal_idx, legal_count)
    np.testing.assert_allclose(restored, shard.legal_masks)


def test_trajectory_v3_roundtrip_full_and_dfm_views():
    shard = build_synthetic_trajectory_shard(batch_size=2, horizon=4)
    payload = {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
        "value_targets": shard.value_targets,
        "wdl_targets": shard.wdl_targets,
        "fen_t": shard.fen_t,
    }
    compact = trajectory_v3_from_v2_npz(payload, legal_lmax=128)
    full = trajectory_v3_to_batch(compact, view="full")
    np.testing.assert_allclose(full["current_planes"], shard.planes_t)
    np.testing.assert_allclose(full["future_planes"], shard.planes_future)
    np.testing.assert_allclose(full["legal_masks"], shard.legal_masks)
    np.testing.assert_array_equal(full["action_indices"], shard.actions)

    dfm = trajectory_v3_to_batch(compact, view="dfm_action", horizon=2)
    assert dfm["action_indices"].shape == (2, 2)
    assert dfm["legal_masks"].shape == (2, 2, 1858)
    assert "future_planes" not in dfm

    jepa = trajectory_v3_to_batch(compact, view="jepa_latent", horizon=2)
    assert jepa["action_indices"].shape == (2, 2)
    assert jepa["future_planes"].shape == (2, 2, 112, 8, 8)
    assert "legal_masks" not in jepa

    joint_v2 = trajectory_joint_batch_from_npz(payload, horizon=2)
    assert joint_v2["action_indices"].shape == (2, 2)
    assert joint_v2["future_planes"].shape == (2, 2, 112, 8, 8)
    assert joint_v2["legal_idx"].shape == (2, 2, 128)
    assert joint_v2["legal_count"].shape == (2, 2)
    assert "legal_masks" not in joint_v2
    restored_v2 = legal_indices_to_masks(
        joint_v2["legal_idx"].astype(np.uint16),
        joint_v2["legal_count"].astype(np.uint16),
    )
    np.testing.assert_allclose(restored_v2, shard.legal_masks[:, :2])

    joint_v3 = trajectory_v3_to_batch(compact, view="joint_latent_sasa", horizon=2)
    assert joint_v3["action_indices"].shape == (2, 2)
    assert joint_v3["future_planes"].shape == (2, 2, 112, 8, 8)
    assert joint_v3["legal_idx"].shape == (2, 2, 128)
    assert joint_v3["legal_count"].shape == (2, 2)
    assert "legal_masks" not in joint_v3


def test_leela_loader_reads_trajectory_v3_dfm_action_view():
    shard = build_synthetic_trajectory_shard(batch_size=3, horizon=4)
    payload = {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
    }
    compact = trajectory_v3_from_v2_npz(payload, legal_lmax=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = Path(tmpdir) / "chunk_000000.npz"
        np.savez_compressed(chunk_path, **compact)

        loader = LeelaChunkDataLoader(
            [str(chunk_path)],
            batch_size=2,
            horizon=2,
            batch_view="dfm_action",
        )
        batch = next(iter(loader))
        assert batch["current_planes"].shape == (2, 112, 8, 8)
        assert batch["action_indices"].shape == (2, 2)
        assert batch["legal_masks"].shape == (2, 2, 1858)
        assert "future_planes" not in batch


def test_leela_loader_reads_trajectory_v3_joint_view():
    shard = build_synthetic_trajectory_shard(batch_size=3, horizon=4)
    payload = {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
    }
    compact = trajectory_v3_from_v2_npz(payload, legal_lmax=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = Path(tmpdir) / "chunk_000000.npz"
        np.savez_compressed(chunk_path, **compact)

        loader = LeelaChunkDataLoader(
            [str(chunk_path)],
            batch_size=2,
            horizon=2,
            batch_view="joint_latent_sasa",
        )
        batch = next(iter(loader))
        assert batch["current_planes"].shape == (2, 112, 8, 8)
        assert batch["action_indices"].shape == (2, 2)
        assert batch["future_planes"].shape == (2, 2, 112, 8, 8)
        assert batch["legal_idx"].shape == (2, 2, 128)
        assert batch["legal_count"].shape == (2, 2)
        assert "legal_masks" not in batch


def test_grain_loader_reports_clear_error_when_dependency_missing():
    if grain_available():
        pytest.skip("Grain is installed in this environment.")
    with pytest.raises(GrainUnavailableError, match="Grain data loading was requested"):
        create_grain_trajectory_loader(
            ["missing.npz"],
            batch_size=2,
            horizon=2,
            batch_view="joint_latent_sasa",
        )
