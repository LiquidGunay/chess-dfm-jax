from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from research.interpretability.pilot_corpus import (
    DEFAULT_OUTPUT_DIR,
    PILOT_SCHEMA,
    _read_member_bytes,
    decode_pilot_batch,
    index_validation_members,
    load_pilot_corpus,
    sha256_file,
    write_deterministic_npz,
)


def test_real_pilot_artifact_strict_load_and_decode() -> None:
    manifest_path = DEFAULT_OUTPUT_DIR / "manifest.json"
    arrays, manifest = load_pilot_corpus(manifest_path)
    batch, decoded_manifest = decode_pilot_batch(manifest_path)

    assert manifest == decoded_manifest
    assert manifest["schema_version"] == PILOT_SCHEMA
    assert manifest["selection"]["count"] == 128
    assert len(manifest["source_shards"]) == 122
    assert (
        manifest["manifest_integrity"]["sha256"]
        == "9b38b1cb42d187555e2fff2bf83754213bfe0f57420cb037b6d3080fede64c2e"
    )
    assert (
        manifest["data"]["sha256"]
        == "cce0247810039fbe9ebfec863febe1f1b3c3e8d2c92b896c51714afca8df1121"
    )
    assert arrays["planes_t_u8"].shape == (128, 112, 8, 8)
    assert arrays["planes_t_u8"].dtype == np.dtype(np.uint8)
    assert arrays["planes_future_u8"].shape == (128, 8, 112, 8, 8)
    assert arrays["global_index_u32"].tolist() == sorted(arrays["global_index_u32"].tolist())
    assert sorted(arrays["selection_rank_u16"].tolist()) == list(range(128))

    assert batch["current_planes"].shape == (128, 112, 8, 8)
    assert batch["current_planes"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        batch["current_planes"],
        arrays["planes_t_u8"].astype(np.float32),
    )
    assert batch["legal_idx"].shape == (128, 8, 128)
    assert batch["legal_count"].shape == (128, 8)
    assert np.isfinite(batch["current_planes"]).all()
    for row in range(128):
        legal = batch["legal_idx"][row, 0, : batch["legal_count"][row, 0]]
        assert int(batch["action_idx"][row]) in legal


def test_deterministic_npz_preserves_scalar_rank(tmp_path: Path) -> None:
    arrays = {
        "matrix": np.arange(12, dtype=np.float32).reshape(3, 4),
        "scalar": np.asarray(7, dtype=np.int32),
        "text": np.asarray("pilot"),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)

    assert sha256_file(first) == sha256_file(second)
    with np.load(first, allow_pickle=False) as payload:
        assert payload["scalar"].shape == ()
        assert int(payload["scalar"]) == 7
        np.testing.assert_array_equal(payload["matrix"], arrays["matrix"])


def _write_test_tar(path: Path) -> bytes:
    payload = b"member-payload"
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        directory = tarfile.TarInfo("./val/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)

        member = tarfile.TarInfo("./val/chunk_000000.npz")
        member.size = len(payload)
        member.mode = 0o644
        archive.addfile(member, fileobj=__import__("io").BytesIO(payload))
    return payload


def test_tar_segment_index_validates_headers_and_offsets(tmp_path: Path) -> None:
    segment = tmp_path / "segment.tar"
    expected_payload = _write_test_tar(segment)
    global_start = 512 * 11
    members = index_validation_members(segment, global_start=global_start)

    member = members["./val/chunk_000000.npz"]
    assert member.global_header_offset == global_start + member.local_header_offset
    assert member.global_payload_offset == global_start + member.local_payload_offset
    assert _read_member_bytes(segment, member) == expected_payload

    corrupted = bytearray(segment.read_bytes())
    corrupted[0] ^= 1
    segment.write_bytes(corrupted)
    with pytest.raises(ValueError, match="Tar checksum mismatch"):
        index_validation_members(segment, global_start=global_start)


def test_pilot_loader_rejects_manifest_and_data_drift(tmp_path: Path) -> None:
    source_manifest = DEFAULT_OUTPUT_DIR / "manifest.json"
    source_data = DEFAULT_OUTPUT_DIR / "positions.npz"
    manifest_path = tmp_path / "manifest.json"
    data_path = tmp_path / "positions.npz"
    shutil.copyfile(source_manifest, manifest_path)
    shutil.copyfile(source_data, data_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["positions"][0]["ply"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical checksum drift"):
        load_pilot_corpus(manifest_path)

    shutil.copyfile(source_manifest, manifest_path)
    corrupted = bytearray(data_path.read_bytes())
    corrupted[-1] ^= 1
    data_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="data checksum drift"):
        load_pilot_corpus(manifest_path)
