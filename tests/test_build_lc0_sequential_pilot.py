from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_lc0_sequential_pilot import (
    PILOT_CHUNK_INDICES,
    build_pilot,
)


def test_pilot_builder_selects_exact_chunks_and_is_immutable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "lc0-sequential-v1",
                "source_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    for index in PILOT_CHUNK_INDICES:
        split_root = source / "chunks" / f"chunk-{index:05d}" / "train"
        split_root.mkdir(parents=True)
        (split_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "lc0-sequential-v1",
                    "split": "train",
                    "position_count": 3,
                    "trainable_start_count": 2,
                }
            ),
            encoding="utf-8",
        )
        (split_root / "planes_u64.npy").write_bytes(bytes((index % 256,)))

    output = tmp_path / "pilot"
    marker = build_pilot(
        source_root=source,
        output_root=output,
        hardlink=False,
    )

    assert marker["selected_chunk_indices"] == list(PILOT_CHUNK_INDICES)
    assert marker["split_totals"]["train"]["trainable_starts"] == (
        2 * len(PILOT_CHUNK_INDICES)
    )
    assert marker["inventory_file_count"] == 1 + 2 * len(PILOT_CHUNK_INDICES)
    assert len(marker["inventory_sha256"]) == 64
    assert (output / ".stage.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        build_pilot(
            source_root=source,
            output_root=output,
            hardlink=False,
        )
