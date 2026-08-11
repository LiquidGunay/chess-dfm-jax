"""Deterministic, content-addressed corpus for the first BT4 interpretation pilot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.train_torch import canonicalize_trajectory_batch

from chess_dfm_jax.data.trajectory_v3 import trajectory_v3_to_batch


_REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_SCHEMA = "bt4-interpretability-pilot-corpus-v1"
DEFAULT_COUNT = 128
DEFAULT_INDEX_ARRAY = "fast_val_global_index"
DEFAULT_EVAL_MANIFEST = _REPO_ROOT / "research/eval/hero_epoch_v1/manifest.json"
DEFAULT_INDEX_PATH = _REPO_ROOT / "research/eval/hero_epoch_v1/position_indices.npz"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "research/eval/interpretability_pilot_v1"
DEFAULT_SOURCE_ARCHIVE_ID = "14jgdmEZsBVMFtQ7Vvjm3PTWU2QlE45KV"
DEFAULT_SOURCE_ARCHIVE_NAME = "trajectory_v3_lc0_test80_h8_sets1_3_20260430.tar"
DEFAULT_SOURCE_ARCHIVE_SHA256 = "d8feffa259580563c8097fa4a604ca415469dd80da66dabda170ac9ce929b968"
DEFAULT_SOURCE_ARCHIVE_SIZE = 11_531_386_880
DEFAULT_SEGMENT_GLOBAL_START = 571_904_512
DEFAULT_SEGMENT_GLOBAL_END = 1_201_040_111
SOURCE_ROWS_PER_SHARD = 1024
SOURCE_HORIZON = 8
SOURCE_SPLIT = "val"

_SOURCE_ROW_KEYS = (
    "actions_u16",
    "actions_uci",
    "fen_t",
    "future_valid_u8",
    "game_id",
    "input_format",
    "legal_count_u16",
    "legal_idx_u16",
    "planes_future_u8",
    "planes_t_u8",
    "ply",
    "result",
    "source",
    "value_targets",
    "wdl_targets",
)
_SOURCE_SCALAR_KEYS = (
    "batch_size",
    "horizon",
    "legal_codec",
    "plane_codec",
    "schema_version",
    "source_schema_version",
    "source_uri",
)


@dataclass(frozen=True)
class TarMember:
    name: str
    local_header_offset: int
    global_header_offset: int
    local_payload_offset: int
    global_payload_offset: int
    size_bytes: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"Refusing unsafe output path: {resolved}")
    return resolved


def _parse_tar_octal(field: bytes, *, label: str) -> int:
    raw = field.split(b"\0", 1)[0].strip() or b"0"
    try:
        return int(raw, 8)
    except ValueError as exc:
        raise ValueError(f"Invalid tar {label}: {raw!r}") from exc


def _validate_tar_header(header: bytes, *, local_offset: int) -> None:
    if len(header) != 512:
        raise ValueError(f"Short tar header at local offset {local_offset}")
    if header[257:262] != b"ustar":
        raise ValueError(f"Unsupported tar header at local offset {local_offset}")
    expected = _parse_tar_octal(header[148:156], label="checksum")
    actual = sum(header[:148]) + 8 * ord(" ") + sum(header[156:])
    if actual != expected:
        raise ValueError(
            f"Tar checksum mismatch at local offset {local_offset}: {actual} != {expected}"
        )


def iter_tar_segment_members(
    segment_path: Path,
    *,
    global_start: int,
) -> Iterator[TarMember]:
    """Iterate complete members in a 512-aligned, possibly truncated tar range."""

    if global_start < 0 or global_start % 512:
        raise ValueError("Tar segment global start must be a non-negative multiple of 512")
    segment_size = segment_path.stat().st_size
    local_offset = 0
    with segment_path.open("rb") as handle:
        while local_offset + 512 <= segment_size:
            handle.seek(local_offset)
            header = handle.read(512)
            if header == bytes(512):
                local_offset += 512
                continue
            _validate_tar_header(header, local_offset=local_offset)
            name = header[:100].split(b"\0", 1)[0].decode("utf-8")
            prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8")
            if prefix:
                name = f"{prefix}/{name}"
            size_bytes = _parse_tar_octal(header[124:136], label="size")
            local_payload_offset = local_offset + 512
            padded_size = ((size_bytes + 511) // 512) * 512
            next_offset = local_payload_offset + padded_size
            if local_payload_offset + size_bytes > segment_size:
                break
            yield TarMember(
                name=name,
                local_header_offset=local_offset,
                global_header_offset=global_start + local_offset,
                local_payload_offset=local_payload_offset,
                global_payload_offset=global_start + local_payload_offset,
                size_bytes=size_bytes,
            )
            local_offset = next_offset


def index_validation_members(
    segment_path: Path,
    *,
    global_start: int,
) -> dict[str, TarMember]:
    members: dict[str, TarMember] = {}
    saw_validation = False
    for member in iter_tar_segment_members(segment_path, global_start=global_start):
        if member.name == "./val/":
            saw_validation = True
            continue
        if member.name.startswith("./val/") and member.name.endswith(".npz"):
            saw_validation = True
            if member.name in members:
                raise ValueError(f"Duplicate tar member: {member.name}")
            members[member.name] = member
            continue
        if saw_validation and member.name.startswith("./train/"):
            break
    if not saw_validation or not members:
        raise ValueError("Tar segment contains no validation shards")
    return members


def _read_member_bytes(segment_path: Path, member: TarMember) -> bytes:
    with segment_path.open("rb") as handle:
        handle.seek(member.local_payload_offset)
        payload = handle.read(member.size_bytes)
    if len(payload) != member.size_bytes:
        raise ValueError(f"Short payload for {member.name}")
    return payload


def _array_scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _validate_source_shard(
    payload: Mapping[str, np.ndarray],
    *,
    member_name: str,
) -> None:
    required = set(_SOURCE_ROW_KEYS) | set(_SOURCE_SCALAR_KEYS)
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{member_name} is missing source arrays: {missing}")
    if int(np.asarray(payload["batch_size"]).item()) != SOURCE_ROWS_PER_SHARD:
        raise ValueError(f"{member_name} has an unexpected batch size")
    if int(np.asarray(payload["horizon"]).item()) != SOURCE_HORIZON:
        raise ValueError(f"{member_name} has an unexpected horizon")
    expected_scalars = {
        "schema_version": "trajectory-v3",
        "source_schema_version": "trajectory-v2",
        "plane_codec": "uint8",
        "legal_codec": "indices-u16-lmax-128",
    }
    for key, expected in expected_scalars.items():
        observed = _array_scalar_text(np.asarray(payload[key]))
        if observed != expected:
            raise ValueError(f"{member_name} {key} drift: {observed!r} != {expected!r}")
    for key in _SOURCE_ROW_KEYS:
        array = np.asarray(payload[key])
        if array.ndim == 0 or array.shape[0] != SOURCE_ROWS_PER_SHARD:
            raise ValueError(f"{member_name} {key} has invalid shape {array.shape}")
        if array.dtype.hasobject:
            raise ValueError(f"{member_name} {key} has forbidden object dtype")


def _npy_bytes(array: np.ndarray) -> bytes:
    normalized = np.asarray(array)
    if normalized.dtype.hasobject:
        raise ValueError("Pilot arrays may not use object dtype")
    if normalized.ndim > 0 and not normalized.flags.c_contiguous:
        normalized = np.ascontiguousarray(normalized)
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        normalized,
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    resolved = _require_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    with temporary.open("wb") as raw_handle:
        with zipfile.ZipFile(
            raw_handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    _npy_bytes(np.asarray(arrays[name])),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    os.replace(temporary, resolved)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    resolved = _require_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    base_payload = dict(payload)
    base_payload.pop("manifest_integrity", None)
    payload = {
        **base_payload,
        "manifest_integrity": {
            "algorithm": "sha256-canonical-json-without-manifest_integrity",
            "sha256": _sha256_bytes(_canonical_json_bytes(base_payload)),
        },
    }
    temporary = resolved.with_name(f".{resolved.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, resolved)


def _load_frozen_indices(
    *,
    eval_manifest_path: Path,
    position_indices_path: Path,
    index_array: str,
    count: int,
    source_archive_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str, str]:
    manifest_bytes = eval_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "chess-dfm-hero-eval-manifest-v1":
        raise ValueError("Unsupported Hero evaluation manifest")
    if manifest.get("immutable_after_creation") is not True:
        raise ValueError("Hero evaluation manifest is not immutable")
    if manifest["dataset"]["archive_sha256"] != source_archive_sha256:
        raise ValueError("Hero evaluation archive hash differs from the source archive")
    if index_array != manifest["position_indices"]["fast_validation"]["array"]:
        raise ValueError("Pilot must be a prefix of the frozen fast-validation pool")
    definition = manifest["position_indices"]["fast_validation"]
    if definition["split"] != SOURCE_SPLIT:
        raise ValueError("Frozen pilot indices must come from the validation split")
    expected_size = int(manifest["position_indices"]["size_bytes"])
    expected_sha256 = manifest["position_indices"]["sha256"]
    if position_indices_path.stat().st_size != expected_size:
        raise ValueError("Frozen position index size drift")
    observed_indices_sha256 = sha256_file(position_indices_path)
    if observed_indices_sha256 != expected_sha256:
        raise ValueError("Frozen position index checksum drift")
    with np.load(position_indices_path, allow_pickle=False) as payload:
        if index_array not in payload.files:
            raise KeyError(f"Missing frozen index array {index_array}")
        full_indices = np.asarray(payload[index_array], dtype=np.uint32)
    if not 0 < count <= full_indices.size:
        raise ValueError(f"Pilot count must be in [1, {full_indices.size}], got {count}")
    selected_in_source_order = full_indices[:count].copy()
    if np.unique(selected_in_source_order).size != count:
        raise ValueError("Frozen pilot prefix contains duplicate indices")
    selected_sorted = np.sort(selected_in_source_order)
    return (
        selected_in_source_order,
        selected_sorted,
        manifest,
        _sha256_bytes(manifest_bytes),
        observed_indices_sha256,
    )


def build_pilot_corpus(
    *,
    source_segment_path: Path,
    source_segment_global_start: int,
    source_segment_global_end: int,
    source_archive_id: str,
    source_archive_name: str,
    source_archive_sha256: str,
    source_archive_size: int,
    eval_manifest_path: Path,
    position_indices_path: Path,
    output_dir: Path,
    count: int = DEFAULT_COUNT,
    index_array: str = DEFAULT_INDEX_ARRAY,
) -> dict[str, Any]:
    """Build the immutable pilot and return its verified manifest."""

    segment_size = source_segment_path.stat().st_size
    expected_segment_size = source_segment_global_end - source_segment_global_start + 1
    if segment_size != expected_segment_size:
        raise ValueError(f"Source segment size drift: {segment_size} != {expected_segment_size}")
    (
        selected_in_source_order,
        selected_sorted,
        eval_manifest,
        eval_manifest_sha256,
        position_indices_sha256,
    ) = _load_frozen_indices(
        eval_manifest_path=eval_manifest_path,
        position_indices_path=position_indices_path,
        index_array=index_array,
        count=count,
        source_archive_sha256=source_archive_sha256,
    )
    source_rank = {
        int(global_index): rank
        for rank, global_index in enumerate(selected_in_source_order.tolist())
    }
    members = index_validation_members(
        source_segment_path,
        global_start=source_segment_global_start,
    )
    required_shards = sorted(
        {int(global_index) // SOURCE_ROWS_PER_SHARD for global_index in selected_sorted}
    )
    rows_by_shard: dict[int, list[int]] = {shard: [] for shard in required_shards}
    for global_index in selected_sorted:
        shard = int(global_index) // SOURCE_ROWS_PER_SHARD
        rows_by_shard[shard].append(int(global_index) % SOURCE_ROWS_PER_SHARD)

    row_records: dict[int, dict[str, np.ndarray]] = {}
    shard_records: list[dict[str, Any]] = []
    member_hashes: dict[int, str] = {}
    member_names: dict[int, str] = {}
    source_uris: dict[int, str] = {}
    for shard in required_shards:
        member_name = f"./val/chunk_{shard:06d}.npz"
        try:
            member = members[member_name]
        except KeyError as exc:
            raise KeyError(f"Required validation member not found: {member_name}") from exc
        member_bytes = _read_member_bytes(source_segment_path, member)
        member_sha256 = _sha256_bytes(member_bytes)
        member_hashes[shard] = member_sha256
        member_names[shard] = member_name
        with np.load(io.BytesIO(member_bytes), allow_pickle=False) as payload:
            _validate_source_shard(payload, member_name=member_name)
            source_uris[shard] = _array_scalar_text(np.asarray(payload["source_uri"]))
            for row in rows_by_shard[shard]:
                global_index = shard * SOURCE_ROWS_PER_SHARD + row
                row_records[global_index] = {
                    key: np.asarray(payload[key][row]).copy() for key in _SOURCE_ROW_KEYS
                }
        shard_records.append(
            {
                "member_name": member_name,
                "member_sha256": member_sha256,
                "size_bytes": member.size_bytes,
                "global_header_offset": member.global_header_offset,
                "global_payload_offset": member.global_payload_offset,
                "selected_rows": rows_by_shard[shard],
            }
        )

    ordered_indices = [int(value) for value in selected_sorted.tolist()]
    arrays: dict[str, np.ndarray] = {
        key: np.stack([row_records[index][key] for index in ordered_indices])
        for key in _SOURCE_ROW_KEYS
    }
    arrays.update(
        {
            "batch_size": np.asarray(count, dtype=np.int32),
            "global_index_u32": np.asarray(ordered_indices, dtype=np.uint32),
            "horizon": np.asarray(SOURCE_HORIZON, dtype=np.int32),
            "legal_codec": np.asarray("indices-u16-lmax-128"),
            "pilot_schema": np.asarray(PILOT_SCHEMA),
            "plane_codec": np.asarray("uint8"),
            "selection_rank_u16": np.asarray(
                [source_rank[index] for index in ordered_indices],
                dtype=np.uint16,
            ),
            "source_archive_sha256": np.asarray(source_archive_sha256),
            "source_member": np.asarray(
                [member_names[index // SOURCE_ROWS_PER_SHARD] for index in ordered_indices]
            ),
            "source_member_sha256": np.asarray(
                [member_hashes[index // SOURCE_ROWS_PER_SHARD] for index in ordered_indices]
            ),
            "source_row_u16": np.asarray(
                [index % SOURCE_ROWS_PER_SHARD for index in ordered_indices],
                dtype=np.uint16,
            ),
            "source_schema_version": np.asarray("trajectory-v2"),
            "source_shard_u16": np.asarray(
                [index // SOURCE_ROWS_PER_SHARD for index in ordered_indices],
                dtype=np.uint16,
            ),
            "source_uri": np.asarray(
                [source_uris[index // SOURCE_ROWS_PER_SHARD] for index in ordered_indices]
            ),
            "schema_version": np.asarray("trajectory-v3"),
        }
    )

    plane_hashes: list[str] = []
    position_ids: list[str] = []
    position_rows: list[dict[str, Any]] = []
    for row_index, global_index in enumerate(ordered_indices):
        plane_sha256 = _sha256_bytes(
            np.ascontiguousarray(arrays["planes_t_u8"][row_index]).tobytes()
        )
        identity = (
            f"{source_archive_sha256}\0{SOURCE_SPLIT}\0{global_index}\0{plane_sha256}"
        ).encode("utf-8")
        position_id = _sha256_bytes(identity)
        plane_hashes.append(plane_sha256)
        position_ids.append(position_id)
        shard = global_index // SOURCE_ROWS_PER_SHARD
        position_rows.append(
            {
                "position_id": position_id,
                "global_index": global_index,
                "selection_rank": source_rank[global_index],
                "source_shard_ordinal": shard,
                "source_row": global_index % SOURCE_ROWS_PER_SHARD,
                "source_member": member_names[shard],
                "source_member_sha256": member_hashes[shard],
                "plane_sha256": plane_sha256,
                "fen": str(arrays["fen_t"][row_index]),
                "game_id": str(arrays["game_id"][row_index]),
                "ply": int(arrays["ply"][row_index]),
            }
        )
    arrays["plane_sha256"] = np.asarray(plane_hashes)
    arrays["position_id"] = np.asarray(position_ids)

    output_dir = _require_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "positions.npz"
    manifest_path = output_dir / "manifest.json"
    write_deterministic_npz(data_path, arrays)
    array_inventory = {
        name: {
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
        }
        for name, array in sorted(arrays.items())
    }
    manifest_payload: dict[str, Any] = {
        "schema_version": PILOT_SCHEMA,
        "immutable_after_creation": True,
        "scientific_use": {
            "allowed": [
                "engineering validation",
                "development-layer selection",
                "metric implementation checks",
            ],
            "forbidden": [
                "terminal confirmatory claims",
                "blind-test reporting",
            ],
        },
        "selection": {
            "source_pool": "Hero fast validation",
            "source_array": index_array,
            "rule": f"first {count} frozen indices, then sorted for materialization",
            "count": count,
            "split": SOURCE_SPLIT,
            "source_order_indices_sha256": _sha256_bytes(
                np.ascontiguousarray(selected_in_source_order).tobytes()
            ),
            "sorted_indices_sha256": _sha256_bytes(np.ascontiguousarray(selected_sorted).tobytes()),
        },
        "source_archive": {
            "drive_file_id": source_archive_id,
            "name": source_archive_name,
            "sha256": source_archive_sha256,
            "size_bytes": source_archive_size,
            "range": {
                "global_start": source_segment_global_start,
                "global_end": source_segment_global_end,
                "size_bytes": segment_size,
                "sha256": sha256_file(source_segment_path),
            },
            "verification_scope": (
                "Full archive identity is pinned by its retained SHA256 sidecar; "
                "the locally fetched range is independently hashed and every "
                "consumed member is tar-checksummed and content-hashed."
            ),
        },
        "hero_evaluation_manifest": {
            "path": str(eval_manifest_path.resolve().relative_to(_REPO_ROOT.resolve())),
            "sha256": eval_manifest_sha256,
            "position_indices_path": str(
                position_indices_path.resolve().relative_to(_REPO_ROOT.resolve())
            ),
            "position_indices_sha256": position_indices_sha256,
            "split_inventory_sha256": eval_manifest["dataset"][SOURCE_SPLIT][
                "filename_size_manifest_sha256"
            ],
        },
        "source_shards": shard_records,
        "data": {
            "path": data_path.name,
            "sha256": sha256_file(data_path),
            "size_bytes": data_path.stat().st_size,
            "format": "deterministic-npz; fixed ZIP timestamp; no pickle",
        },
        "arrays": array_inventory,
        "positions": position_rows,
    }
    _write_manifest(manifest_path, manifest_payload)
    _arrays, verified_manifest = load_pilot_corpus(manifest_path)
    return verified_manifest


def load_pilot_corpus(
    manifest_path: Path = DEFAULT_OUTPUT_DIR / "manifest.json",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Strict-load and re-verify an immutable pilot corpus."""

    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PILOT_SCHEMA:
        raise ValueError("Unsupported pilot corpus schema")
    if manifest.get("immutable_after_creation") is not True:
        raise ValueError("Pilot corpus is not marked immutable")
    integrity = manifest.get("manifest_integrity")
    if not isinstance(integrity, dict):
        raise ValueError("Pilot manifest has no integrity record")
    base_manifest = dict(manifest)
    base_manifest.pop("manifest_integrity")
    expected_manifest_sha256 = _sha256_bytes(_canonical_json_bytes(base_manifest))
    if integrity.get("sha256") != expected_manifest_sha256:
        raise ValueError("Pilot manifest canonical checksum drift")

    data_record = manifest["data"]
    data_path = (manifest_path.parent / data_record["path"]).resolve(strict=True)
    try:
        data_path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("Pilot data path escapes its manifest directory") from exc
    if data_path.stat().st_size != int(data_record["size_bytes"]):
        raise ValueError("Pilot data size drift")
    if sha256_file(data_path) != data_record["sha256"]:
        raise ValueError("Pilot data checksum drift")

    inventory = manifest["arrays"]
    with np.load(data_path, allow_pickle=False) as payload:
        if set(payload.files) != set(inventory):
            raise ValueError("Pilot NPZ member inventory drift")
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    for name, record in inventory.items():
        array = arrays[name]
        if array.dtype.hasobject:
            raise ValueError(f"Pilot array {name} has forbidden object dtype")
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Pilot array ABI drift for {name}")

    count = int(manifest["selection"]["count"])
    global_indices = arrays["global_index_u32"]
    if global_indices.shape != (count,) or np.any(np.diff(global_indices) <= 0):
        raise ValueError("Pilot global indices are not strictly increasing")
    if len(manifest["positions"]) != count:
        raise ValueError("Pilot position inventory count drift")
    for row, position in enumerate(manifest["positions"]):
        global_index = int(global_indices[row])
        if int(position["global_index"]) != global_index:
            raise ValueError(f"Pilot position order drift at row {row}")
        plane_sha256 = _sha256_bytes(np.ascontiguousarray(arrays["planes_t_u8"][row]).tobytes())
        if plane_sha256 != position["plane_sha256"] or plane_sha256 != str(
            arrays["plane_sha256"][row]
        ):
            raise ValueError(f"Pilot plane checksum drift at row {row}")
        identity = (
            f"{manifest['source_archive']['sha256']}\0{SOURCE_SPLIT}\0"
            f"{global_index}\0{plane_sha256}"
        ).encode("utf-8")
        position_id = _sha256_bytes(identity)
        if position_id != position["position_id"] or position_id != str(arrays["position_id"][row]):
            raise ValueError(f"Pilot position identity drift at row {row}")
    return arrays, manifest


def decode_pilot_batch(
    manifest_path: Path = DEFAULT_OUTPUT_DIR / "manifest.json",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Decode stored planes and canonicalize policy actions/legal indices."""

    arrays, manifest = load_pilot_corpus(manifest_path)
    decoded = trajectory_v3_to_batch(
        arrays,
        view="joint_latent_sasa",
        horizon=SOURCE_HORIZON,
        include_metadata=True,
    )
    return canonicalize_trajectory_batch(decoded), manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-segment", type=Path, required=True)
    parser.add_argument(
        "--source-segment-global-start",
        type=int,
        default=DEFAULT_SEGMENT_GLOBAL_START,
    )
    parser.add_argument(
        "--source-segment-global-end",
        type=int,
        default=DEFAULT_SEGMENT_GLOBAL_END,
    )
    parser.add_argument("--source-archive-id", default=DEFAULT_SOURCE_ARCHIVE_ID)
    parser.add_argument("--source-archive-name", default=DEFAULT_SOURCE_ARCHIVE_NAME)
    parser.add_argument(
        "--source-archive-sha256",
        default=DEFAULT_SOURCE_ARCHIVE_SHA256,
    )
    parser.add_argument(
        "--source-archive-size",
        type=int,
        default=DEFAULT_SOURCE_ARCHIVE_SIZE,
    )
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--position-indices", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--index-array", default=DEFAULT_INDEX_ARRAY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_pilot_corpus(
        source_segment_path=args.source_segment,
        source_segment_global_start=args.source_segment_global_start,
        source_segment_global_end=args.source_segment_global_end,
        source_archive_id=args.source_archive_id,
        source_archive_name=args.source_archive_name,
        source_archive_sha256=args.source_archive_sha256,
        source_archive_size=args.source_archive_size,
        eval_manifest_path=args.eval_manifest,
        position_indices_path=args.position_indices,
        output_dir=args.output_dir,
        count=args.count,
        index_array=args.index_array,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "manifest_integrity_sha256": manifest["manifest_integrity"]["sha256"],
                "data_sha256": manifest["data"]["sha256"],
                "positions": manifest["selection"]["count"],
                "source_shards": len(manifest["source_shards"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
