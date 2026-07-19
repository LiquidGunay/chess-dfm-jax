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
import chess_dfm_jax.data.trajectory_v3 as trajectory_v3_module  # noqa: E402
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


def _synthetic_v2_payload(*, batch_size: int = 5, horizon: int = 4):
    shard = build_synthetic_trajectory_shard(
        batch_size=batch_size,
        horizon=horizon,
    )
    return shard, {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
        "legal_masks_valid": np.ones_like(shard.future_valid),
        "value_targets": shard.value_targets,
        "wdl_targets": shard.wdl_targets,
        "source": shard.source,
        "game_id": shard.game_id,
        "ply": shard.ply,
        "result": shard.result,
        "fen_t": shard.fen_t,
        "input_format": shard.input_format,
        "actions_uci": shard.actions_uci,
    }


def _assert_exact_row_slice(
    full: dict[str, np.ndarray],
    sliced: dict[str, np.ndarray],
    *,
    row_slice: slice,
    source_batch_size: int,
) -> None:
    assert sliced.keys() == full.keys()
    for key, full_value in full.items():
        expected = np.asarray(full_value)
        if expected.ndim > 0 and expected.shape[0] == source_batch_size:
            expected = expected[row_slice]
        np.testing.assert_array_equal(sliced[key], expected, err_msg=key)


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


@pytest.mark.parametrize("plane_codec", ["uint8", "packbits"])
@pytest.mark.parametrize(
    "view",
    ["full", "dfm_action", "jepa_latent", "joint_latent_sasa"],
)
def test_trajectory_v3_row_slice_matches_full_decode_exactly(
    plane_codec,
    view,
):
    _, payload = _synthetic_v2_payload()
    compact = trajectory_v3_from_v2_npz(
        payload,
        legal_lmax=128,
        plane_codec=plane_codec,
    )
    compact["source"] = np.asarray("scalar-shard-source")
    row_slice = slice(1, 5, 2)

    full = trajectory_v3_to_batch(
        compact,
        view=view,
        horizon=3,
        include_metadata=True,
    )
    sliced = trajectory_v3_to_batch(
        compact,
        view=view,
        horizon=3,
        include_metadata=True,
        row_slice=row_slice,
    )

    _assert_exact_row_slice(
        full,
        sliced,
        row_slice=row_slice,
        source_batch_size=5,
    )
    assert np.asarray(sliced["source"]).ndim == 0


def test_trajectory_v3_row_slice_expands_only_selected_rows(monkeypatch):
    _, payload = _synthetic_v2_payload(batch_size=6)
    compact = trajectory_v3_from_v2_npz(
        payload,
        legal_lmax=128,
        plane_codec="uint8",
    )
    decoded_plane_shapes = []
    legal_shapes = []
    real_decode_planes = trajectory_v3_module.decode_planes_u8
    real_legal_masks = trajectory_v3_module.legal_indices_to_masks

    def record_decode_planes(encoded):
        decoded_plane_shapes.append(np.asarray(encoded).shape)
        return real_decode_planes(encoded)

    def record_legal_masks(legal_idx, legal_count, *, vocab_size=1858):
        legal_shapes.append(
            (
                np.asarray(legal_idx).shape,
                np.asarray(legal_count).shape,
            )
        )
        return real_legal_masks(
            legal_idx,
            legal_count,
            vocab_size=vocab_size,
        )

    monkeypatch.setattr(
        trajectory_v3_module,
        "decode_planes_u8",
        record_decode_planes,
    )
    monkeypatch.setattr(
        trajectory_v3_module,
        "legal_indices_to_masks",
        record_legal_masks,
    )

    batch = trajectory_v3_to_batch(
        compact,
        view="full",
        horizon=3,
        row_slice=slice(2, 4),
    )

    assert batch["current_planes"].shape[0] == 2
    assert batch["future_planes"].shape[0] == 2
    assert decoded_plane_shapes == [
        (2, 112, 8, 8),
        (2, 3, 112, 8, 8),
    ]
    assert legal_shapes == [((2, 3, 128), (2, 3))]


@pytest.mark.parametrize(
    ("row_slice", "error_type", "message"),
    [
        (1, TypeError, "row_slice must be a slice or None"),
        (slice(None, None, 0), ValueError, "row_slice step cannot be zero"),
    ],
)
def test_trajectory_v3_row_slice_rejects_invalid_selectors(
    row_slice,
    error_type,
    message,
):
    _, payload = _synthetic_v2_payload(batch_size=2)
    compact = trajectory_v3_from_v2_npz(payload)

    with pytest.raises(error_type, match=message):
        trajectory_v3_to_batch(compact, row_slice=row_slice)


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
