#!/usr/bin/env python3
"""Read-only dense BT4 drift evaluation for the frozen four-model set.

The evaluator deliberately keeps checkpoint loading, activation capture, and
metric computation outside ``research/train.py``.  It reads the exact stored
trajectory planes referenced by the pinned development pool, evaluates every
normative BT4 hook in FP32, and removes its temporary activation files before
publishing a compact immutable result directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import chess
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.data.trajectory_v3 import (  # noqa: E402
    decode_planes_u8,
    unpack_planes,
)
from chess_dfm_jax.encoding import encode_board  # noqa: E402
from chess_dfm_jax.nnx_bt4 import (  # noqa: E402
    BT4TrainableParam,
    make_bt4_model,
)
from chess_dfm_jax.policy import (  # noqa: E402
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_VOCAB_SIZE,
    LC0_CANONICAL_1858_INPUT_FORMAT,
    encode_action,
    legal_action_mask,
)
from research.prepare import (  # noqa: E402
    REPO_ROOT,
    load_asset_manifest,
    require_within_workspace,
    sha256_file,
    write_json,
)


HOOK_NAMES = (
    "hook_attn_in",
    "hook_attn_out",
    "resid_mid_after_ln",
    "hook_mlp_out",
    "resid_post_after_ln",
)
VIEW_NAMES = ("square_tokens", "board_pooled")
CAPTURE_SCHEMA = "bt4-dense-capture-fp32-v1"
RESULT_SCHEMA = "bt4-dense-representation-stage1-v2"
DEFAULT_POOL = REPO_ROOT / "artifacts" / "arena" / "development-ply12-n128-v1.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "trajectory_v3"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "source" / "extracted"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "representations" / "dense-stage1-four-model-v2"
RECOVERED_STATE = (
    REPO_ROOT
    / "checkpoints"
    / "source"
    / "step0265000"
    / "checkpoints"
    / "step0265000"
    / "state.npz"
)
COMPATIBILITY_CHECKPOINT = (
    REPO_ROOT
    / "research"
    / "runs"
    / "baseline-freshopt-online-target576-b128-lr3e5-bt4lr1e6-30m-v2"
    / "checkpoints"
    / "update00000300"
)
CORRECTED_CHECKPOINT = (
    REPO_ROOT
    / "research"
    / "runs"
    / "corrected-nonorm-target5p76-pred1-fixed64-b128-30m-v2"
    / "checkpoints"
    / "update00000400"
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    kind: str
    state_path: Path | None
    expected_state_sha256: str | None
    expected_state_size: int | None
    expected_step: int | None
    checkpoint_manifest: Path | None = None


@dataclass(frozen=True)
class FrozenCorpus:
    planes: np.ndarray
    legal_masks: np.ndarray
    target_actions: np.ndarray
    entries: tuple[dict[str, Any], ...]
    contract: dict[str, Any]


@dataclass
class ActivationStatistics:
    centered: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_rms: np.ndarray
    coordinate_variance: np.ndarray
    summary: dict[str, Any]


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = require_within_workspace(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _progress(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, **payload},
            sort_keys=True,
        ),
        flush=True,
    )


def _array_tree_leaves(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], np.ndarray]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            yield from _array_tree_leaves(value[key], (*path, str(key)))
        return
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"Object leaf at {'/'.join(path)} is not an array")
    yield path, array


def encoder_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an encoder tree with paths, shapes, dtypes, and exact bytes."""

    digest = hashlib.sha256()
    for path, array in _array_tree_leaves(payload):
        contiguous = np.ascontiguousarray(array)
        digest.update("/".join(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _load_manifest_state(checkpoint_dir: Path) -> tuple[Path, str, int, int]:
    checkpoint_dir = require_within_workspace(checkpoint_dir)
    manifest_path = require_within_workspace(checkpoint_dir / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "chess-dfm-research-checkpoint-v1":
        raise ValueError(f"Unexpected checkpoint format in {manifest_path}")
    state = manifest.get("state")
    if not isinstance(state, dict) or state.get("filename") != "state.npz":
        raise ValueError(f"Invalid state record in {manifest_path}")
    state_path = require_within_workspace(checkpoint_dir / "state.npz")
    return (
        state_path,
        str(state["sha256"]),
        int(state["size_bytes"]),
        int(manifest["optimizer_step"]),
    )


def default_model_specs() -> tuple[ModelSpec, ...]:
    assets = load_asset_manifest()
    source_asset = assets["checkpoint_step_265000"]
    compatibility = _load_manifest_state(COMPATIBILITY_CHECKPOINT)
    corrected = _load_manifest_state(CORRECTED_CHECKPOINT)
    return (
        ModelSpec(
            model_id="raw_bt4",
            kind="raw_source_backbone",
            state_path=None,
            expected_state_sha256=None,
            expected_state_size=None,
            expected_step=None,
        ),
        ModelSpec(
            model_id="recovered_step265000",
            kind="recovered_joint_checkpoint",
            state_path=RECOVERED_STATE,
            expected_state_sha256=str(source_asset["state_npz_sha256"]),
            expected_state_size=int(source_asset["state_npz_size_bytes"]),
            expected_step=265_000,
        ),
        ModelSpec(
            model_id="compatibility_v2_update300",
            kind="norm_on_control",
            state_path=compatibility[0],
            expected_state_sha256=compatibility[1],
            expected_state_size=compatibility[2],
            expected_step=compatibility[3],
            checkpoint_manifest=COMPATIBILITY_CHECKPOINT / "manifest.json",
        ),
        ModelSpec(
            model_id="corrected_v2_update400",
            kind="corrected_no_norm_baseline",
            state_path=corrected[0],
            expected_state_sha256=corrected[1],
            expected_state_size=corrected[2],
            expected_step=corrected[3],
            checkpoint_manifest=CORRECTED_CHECKPOINT / "manifest.json",
        ),
    )


def _load_verified_encoder_payload(spec: ModelSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    if spec.state_path is None:
        raise ValueError("Raw source has no checkpoint encoder payload")
    state_path = require_within_workspace(spec.state_path)
    stat = state_path.stat()
    if stat.st_size != spec.expected_state_size:
        raise ValueError(
            f"State size mismatch for {state_path}: "
            f"expected {spec.expected_state_size}, found {stat.st_size}"
        )
    observed_sha256 = sha256_file(state_path)
    if observed_sha256 != spec.expected_state_sha256:
        raise ValueError(
            f"State SHA-256 mismatch for {state_path}: "
            f"expected {spec.expected_state_sha256}, found {observed_sha256}"
        )

    with np.load(state_path, allow_pickle=True) as archive:
        expected_keys = {"step", "model_trainable", "optimizer_state"}
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"Unexpected state keys in {state_path}: {sorted(archive.files)}"
            )
        step = int(np.asarray(archive["step"]).item())
        if step != spec.expected_step:
            raise ValueError(
                f"Checkpoint step mismatch for {state_path}: "
                f"expected {spec.expected_step}, found {step}"
            )
        model_value = archive["model_trainable"]
        if model_value.shape != () or model_value.dtype != object:
            raise TypeError(f"model_trainable is not a scalar object in {state_path}")
        model_payload = model_value.item()

    if not isinstance(model_payload, dict):
        raise TypeError(f"model_trainable is not a dictionary in {state_path}")
    encoder = model_payload.get("encoder")
    if not isinstance(encoder, dict) or set(encoder) != {"embedding", "layers"}:
        raise ValueError(f"Checkpoint encoder tree is malformed in {state_path}")
    metadata = {
        "state_path": str(state_path),
        "state_sha256": observed_sha256,
        "state_size_bytes": stat.st_size,
        "checkpoint_step": step,
        "encoder_payload_sha256": encoder_payload_sha256(encoder),
    }
    del model_payload
    return encoder, metadata


def _cast_payload_like(destination: Any, source: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(destination, Mapping):
        if not isinstance(source, Mapping) or set(destination) != set(source):
            raise ValueError(f"Encoder tree keys differ at {'/'.join(path)}")
        return {
            key: _cast_payload_like(
                destination[key],
                source[key],
                (*path, str(key)),
            )
            for key in destination
        }
    destination_array = np.asarray(destination)
    source_array = np.asarray(source)
    if source_array.shape != destination_array.shape:
        raise ValueError(
            f"Encoder shape differs at {'/'.join(path)}: "
            f"expected {destination_array.shape}, found {source_array.shape}"
        )
    try:
        source_is_inexact = bool(
            jnp.issubdtype(jnp.dtype(source_array.dtype), jnp.inexact)
        )
    except TypeError:
        source_is_inexact = False
    if not source_is_inexact:
        raise TypeError(
            f"Encoder leaf at {'/'.join(path)} is not floating point: "
            f"{source_array.dtype}"
        )
    if not np.all(np.isfinite(source_array.astype(np.float32))):
        raise ValueError(f"Encoder leaf at {'/'.join(path)} is nonfinite")
    return jnp.asarray(source_array, dtype=destination_array.dtype)


def make_fp32_backbone(
    mapped_bt4_params: dict[str, Any],
    spec: ModelSpec,
) -> tuple[Any, dict[str, Any]]:
    """Construct the source model and optionally replace only its BT4 trunk."""

    model = make_bt4_model(
        mapped_bt4_params,
        dtype=jnp.float32,
        train_encoder=True,
    )
    if spec.state_path is None:
        state = nnx.to_pure_dict(nnx.state(model, BT4TrainableParam))
        return model, {
            "encoder_payload_sha256": encoder_payload_sha256(state),
            "checkpoint_step": None,
            "state_path": None,
            "state_sha256": None,
            "state_size_bytes": None,
        }

    payload, metadata = _load_verified_encoder_payload(spec)
    encoder_state = nnx.state(model, BT4TrainableParam)
    destination = nnx.to_pure_dict(encoder_state)
    cast_payload = _cast_payload_like(destination, payload)
    nnx.replace_by_pure_dict(encoder_state, cast_payload)
    nnx.update(model, encoder_state)
    restored = nnx.to_pure_dict(nnx.state(model, BT4TrainableParam))
    restored_digest = encoder_payload_sha256(restored)
    if restored_digest != metadata["encoder_payload_sha256"]:
        # The checkpoint stores BF16 online trunk weights. Casting those exact
        # values to FP32 changes dtype bytes but not values, so hash a casted
        # checkpoint tree before declaring a mismatch.
        cast_digest = encoder_payload_sha256(cast_payload)
        if restored_digest != cast_digest:
            raise ValueError(
                f"Restored FP32 encoder differs from cast checkpoint for {spec.model_id}"
            )
    metadata["fp32_encoder_payload_sha256"] = restored_digest
    del payload, cast_payload, restored
    gc.collect()
    return model, metadata


def _decode_current_planes(archive: Mapping[str, Any], row: int) -> np.ndarray:
    plane_codec = str(np.asarray(archive.get("plane_codec", "uint8")).item())
    if plane_codec == "uint8":
        return decode_planes_u8(np.asarray(archive["planes_t_u8"][row]))
    if plane_codec == "packbits":
        return unpack_planes(np.asarray(archive["planes_t_pack"][row]))
    raise ValueError(f"Unsupported plane codec {plane_codec!r}")


def load_frozen_corpus(
    pool_path: Path,
    data_root: Path,
    *,
    example_count: int,
) -> FrozenCorpus:
    """Load exact stored planes referenced by the pinned arena pool."""

    pool_path = require_within_workspace(pool_path)
    data_root = require_within_workspace(data_root)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    openings = pool.get("openings")
    if not isinstance(openings, list) or len(openings) < example_count:
        raise ValueError(
            f"Pool {pool_path} has fewer than {example_count} openings"
        )
    if example_count < 2:
        raise ValueError("Dense representation corpus requires at least two examples")

    planes: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[int] = []
    entries: list[dict[str, Any]] = []
    game_groups: set[str] = set()
    for corpus_index, opening in enumerate(openings[:example_count]):
        if not isinstance(opening, dict):
            raise TypeError(f"Opening {corpus_index} is not an object")
        relative_shard = Path(str(opening["source_shard"]))
        if relative_shard.is_absolute() or ".." in relative_shard.parts:
            raise ValueError(f"Unsafe source shard {relative_shard}")
        shard_path = require_within_workspace(data_root / relative_shard)
        row = int(opening["source_row"])
        with np.load(shard_path, allow_pickle=False) as archive:
            source_count = int(np.asarray(archive["actions_u16"]).shape[0])
            if not 0 <= row < source_count:
                raise IndexError(f"Row {row} is outside {shard_path}")
            stored_fen = str(np.asarray(archive["fen_t"][row]).item())
            expected_fen = str(opening["fen"])
            if stored_fen != expected_fen:
                raise ValueError(
                    f"FEN mismatch for {relative_shard}:{row}: "
                    f"pool={expected_fen!r}, stored={stored_fen!r}"
                )
            stored_ply = int(np.asarray(archive["ply"][row]).item())
            if stored_ply != int(opening["ply"]):
                raise ValueError(f"Ply mismatch for {relative_shard}:{row}")
            input_format = str(np.asarray(archive["input_format"][row]).item())
            if input_format != LC0_CANONICAL_1858_INPUT_FORMAT:
                raise ValueError(
                    f"Unexpected input format {input_format!r} in {shard_path}"
                )
            current_planes = np.asarray(
                _decode_current_planes(archive, row),
                dtype=np.float32,
            )
            first_action_uci = str(np.asarray(archive["actions_uci"][row, 0]).item())

        board = chess.Board(stored_fen)
        reconstructed = np.asarray(
            encode_board(
                board,
                [],
                planes_layout="nchw",
                input_format=LC0_CANONICAL_1858_INPUT_FORMAT,
            ),
            dtype=np.float32,
        )
        if not np.array_equal(current_planes, reconstructed):
            raise ValueError(
                f"Stored planes differ from audited current-only encoding at "
                f"{relative_shard}:{row}"
            )
        move = chess.Move.from_uci(first_action_uci)
        if not board.is_legal(move):
            raise ValueError(
                f"Stored first action {first_action_uci} is illegal at {stored_fen}"
            )
        legal_mask = legal_action_mask(
            board,
            codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
        )
        target = encode_action(
            move,
            board=board,
            codec_id=ACTION_CODEC_LC0_CANONICAL_1858,
        )
        if not legal_mask[target]:
            raise ValueError(f"Encoded target is absent from canonical legal mask")

        # Persisted ``game_id`` is a known constant placeholder and one source
        # archive shard contains many games.  The arena pool instead selects
        # exactly one ply-12 row per source game and gives that selected row a
        # unique content hash.  That immutable selection identity is therefore
        # the honest bootstrap group available in this dataset.
        game_group = str(opening["selection_hash"])
        if game_group in game_groups:
            raise ValueError(
                "Pinned dense corpus contains repeated source-game identifiers: "
                f"{game_group}"
            )
        game_groups.add(game_group)
        plane_bytes = np.ascontiguousarray(current_planes).tobytes(order="C")
        entry = {
            "corpus_index": corpus_index,
            "selection_hash": str(opening["selection_hash"]),
            "source_shard": relative_shard.as_posix(),
            "source_row": row,
            "source_game": game_group,
            "source_uri": str(opening.get("source_uri", "")),
            "fen": stored_fen,
            "ply": stored_ply,
            "first_action_uci": first_action_uci,
            "canonical_target_action": int(target),
            "canonical_legal_count": int(np.count_nonzero(legal_mask)),
            "stored_plane_sha256": hashlib.sha256(plane_bytes).hexdigest(),
        }
        planes.append(current_planes)
        masks.append(np.asarray(legal_mask, dtype=bool))
        targets.append(int(target))
        entries.append(entry)

    entries_digest = _json_sha256(entries)
    contract = {
        "pool_path": str(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "data_root": str(data_root),
        "dataset_manifest_path": str(data_root / "manifest.json"),
        "dataset_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "selection": "first N entries in immutable hash-ranked development pool",
        "example_count": example_count,
        "source_game_count": len(game_groups),
        "source_game_grouping": "unique hash-selected ply-12 row; persisted game_id is unusable",
        "stored_planes_primary": True,
        "fen_reconstruction_role": "exact_current_only_audit_only",
        "entries_sha256": entries_digest,
        "input_format": LC0_CANONICAL_1858_INPUT_FORMAT,
        "action_codec_id": ACTION_CODEC_LC0_CANONICAL_1858,
    }
    return FrozenCorpus(
        planes=np.stack(planes).astype(np.float32, copy=False),
        legal_masks=np.stack(masks),
        target_actions=np.asarray(targets, dtype=np.int32),
        entries=tuple(entries),
        contract=contract,
    )


@nnx.jit
def _capture_fp32_hooks(model: Any, planes: jax.Array) -> tuple[jax.Array, jax.Array]:
    tokens, captures = model.encode_tokens_with_captures(planes)
    stacked = jnp.stack(
        (
            captures.hook_attn_in,
            captures.hook_attn_out,
            captures.resid_mid_after_ln,
            captures.hook_mlp_out,
            captures.resid_post_after_ln,
        ),
        axis=0,
    )
    return stacked, model.policy_head(tokens)


def capture_model(
    model: Any,
    planes: np.ndarray,
    capture_path: Path,
    policy_path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    capture_path = require_within_workspace(capture_path)
    policy_path = require_within_workspace(policy_path)
    example_count = int(planes.shape[0])
    if example_count % batch_size != 0:
        raise ValueError(
            f"Capture batch size {batch_size} must divide {example_count}"
        )
    layer_count = len(model.layers)
    width = int(model.embedding_size)
    captures = np.lib.format.open_memmap(
        capture_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(HOOK_NAMES), layer_count, example_count, 64, width),
    )
    policy = np.lib.format.open_memmap(
        policy_path,
        mode="w+",
        dtype=np.float32,
        shape=(example_count, ACTION_VOCAB_SIZE),
    )
    started = time.perf_counter()
    first_batch_seconds: float | None = None
    for start in range(0, example_count, batch_size):
        stop = start + batch_size
        batch_started = time.perf_counter()
        hook_values, logits = _capture_fp32_hooks(
            model,
            jnp.asarray(planes[start:stop], dtype=jnp.float32),
        )
        hook_values, logits = jax.device_get(
            jax.block_until_ready((hook_values, logits))
        )
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - batch_started
        if hook_values.shape != (
            len(HOOK_NAMES),
            layer_count,
            batch_size,
            64,
            width,
        ):
            raise ValueError(f"Unexpected hook capture shape {hook_values.shape}")
        if logits.shape != (batch_size, ACTION_VOCAB_SIZE):
            raise ValueError(f"Unexpected policy shape {logits.shape}")
        if not np.all(np.isfinite(hook_values)) or not np.all(np.isfinite(logits)):
            raise ValueError("FP32 representation capture contains nonfinite values")
        captures[:, :, start:stop] = np.asarray(hook_values, dtype=np.float32)
        policy[start:stop] = np.asarray(logits, dtype=np.float32)
    captures.flush()
    policy.flush()
    elapsed = time.perf_counter() - started
    del captures, policy
    return {
        "schema": CAPTURE_SCHEMA,
        "example_count": example_count,
        "capture_batch_size": batch_size,
        "layer_count": layer_count,
        "hook_count": len(HOOK_NAMES),
        "width": width,
        "first_batch_compile_and_capture_seconds": first_batch_seconds,
        "total_capture_seconds": elapsed,
        "examples_per_second_including_first_compile": example_count / elapsed,
        "capture_size_bytes": capture_path.stat().st_size,
        "policy_size_bytes": policy_path.stat().st_size,
    }


def activation_matrix(activation: np.ndarray, view: str) -> np.ndarray:
    values = np.asarray(activation, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (64, 1024):
        raise ValueError(f"Expected [board,64,1024] activation, got {values.shape}")
    if view == "square_tokens":
        return values.reshape((-1, values.shape[-1]))
    if view == "board_pooled":
        return np.mean(values, axis=1, dtype=np.float32)
    raise ValueError(f"Unsupported representation view {view!r}")


def _effective_rank_metrics(eigenvalues: np.ndarray) -> dict[str, Any]:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "stable_rank": 0.0,
            "numerical_rank": 0,
            "pca99_dimension": 0,
        }
    probabilities = values / total
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    maximum = float(np.max(values))
    descending = values[::-1]
    pca99 = int(np.searchsorted(np.cumsum(descending), 0.99 * total) + 1)
    return {
        "effective_rank": float(np.exp(entropy)),
        "participation_ratio": float(total * total / np.sum(np.square(values))),
        "stable_rank": float(total / maximum),
        "numerical_rank": int(np.count_nonzero(values > maximum * 1e-6)),
        "pca99_dimension": pca99,
    }


def activation_statistics(matrix: np.ndarray) -> ActivationStatistics:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"Activation matrix must be [N,D] with N>=2, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Activation matrix contains nonfinite values")
    coordinate_mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    centered = np.asarray(values - coordinate_mean[None, :], dtype=np.float32)
    coordinate_rms = np.sqrt(
        np.mean(np.square(values, dtype=np.float32), axis=0, dtype=np.float64)
    ).astype(np.float32)
    coordinate_variance = np.mean(
        np.square(centered, dtype=np.float32),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    covariance = np.asarray(
        (centered.T @ centered) / np.float32(values.shape[0] - 1),
        dtype=np.float32,
    )
    covariance = np.asarray((covariance + covariance.T) * 0.5, dtype=np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-5, 1e-7)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(
            f"Centered covariance is not positive semidefinite: min={np.min(eigenvalues)}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0).astype(np.float32)
    summary = {
        "sample_count": int(values.shape[0]),
        "feature_count": int(values.shape[1]),
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "coordinate_mean_l2": float(np.linalg.norm(coordinate_mean)),
        "coordinate_mean_abs_mean": float(np.mean(np.abs(coordinate_mean))),
        "centered_variance_mean": float(np.mean(coordinate_variance)),
        "centered_variance_p05": float(np.quantile(coordinate_variance, 0.05)),
        "centered_variance_median": float(np.median(coordinate_variance)),
        "centered_variance_min": float(np.min(coordinate_variance)),
        "top_covariance_eigenvalue": float(eigenvalues[-1]),
        **_effective_rank_metrics(eigenvalues),
    }
    return ActivationStatistics(
        centered=centered,
        covariance=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors.astype(np.float32, copy=False),
        coordinate_mean=coordinate_mean,
        coordinate_rms=coordinate_rms,
        coordinate_variance=coordinate_variance,
        summary=summary,
    )


def _pca_basis(
    statistics: ActivationStatistics,
    retained_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(statistics.eigenvalues, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("Cannot form PCA basis for zero-variance activations")
    descending_values = values[::-1]
    count = int(
        np.searchsorted(
            np.cumsum(descending_values),
            retained_variance * total,
        )
        + 1
    )
    maximum = float(descending_values[0])
    positive_count = int(np.count_nonzero(descending_values > maximum * 1e-8))
    count = max(1, min(count, positive_count))
    basis = np.asarray(statistics.eigenvectors[:, ::-1][:, :count], dtype=np.float32)
    return basis, descending_values[:count].astype(np.float32)


def _pwcca_weighted_mean(centered: np.ndarray, canonical: np.ndarray, correlations: np.ndarray) -> float:
    if canonical.shape[1] == 0:
        return 0.0
    q, _ = np.linalg.qr(np.asarray(canonical, dtype=np.float32), mode="reduced")
    weights = np.sum(np.abs(q.T @ np.asarray(centered, dtype=np.float32)), axis=1)
    weights = np.asarray(weights[: correlations.shape[0]], dtype=np.float64)
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        return float(np.mean(correlations))
    return float(np.sum(weights * correlations) / denominator)


def paired_representation_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    retained_variance: float,
) -> tuple[dict[str, Any], ActivationStatistics, ActivationStatistics, np.ndarray]:
    source_stats = activation_statistics(source)
    candidate_stats = activation_statistics(candidate)
    if source_stats.centered.shape != candidate_stats.centered.shape:
        raise ValueError("Paired activation matrices have different shapes")
    sample_count = source_stats.centered.shape[0]
    cross = np.asarray(
        (source_stats.centered.T @ candidate_stats.centered)
        / np.float32(sample_count - 1),
        dtype=np.float32,
    )
    numerator = float(np.sum(np.square(cross), dtype=np.float64))
    source_norm = float(
        np.sqrt(np.sum(np.square(source_stats.covariance), dtype=np.float64))
    )
    candidate_norm = float(
        np.sqrt(np.sum(np.square(candidate_stats.covariance), dtype=np.float64))
    )
    linear_cka = numerator / max(source_norm * candidate_norm, 1e-30)

    if sample_count < source_stats.centered.shape[1]:
        _, source_r = np.linalg.qr(source_stats.centered.T, mode="reduced")
        _, candidate_r = np.linalg.qr(candidate_stats.centered.T, mode="reduced")
        singular_values = np.linalg.svd(
            source_r @ candidate_r.T,
            compute_uv=False,
        ) / np.float32(sample_count - 1)
    else:
        singular_values = np.linalg.svd(cross, compute_uv=False)
    nuclear_cross = float(np.sum(singular_values, dtype=np.float64))
    source_energy = float(np.trace(source_stats.covariance))
    candidate_energy = float(np.trace(candidate_stats.covariance))
    procrustes_sse = max(
        source_energy + candidate_energy - 2.0 * nuclear_cross,
        0.0,
    )
    procrustes_r2 = 1.0 - procrustes_sse / max(candidate_energy, 1e-30)

    source_basis, source_eigenvalues = _pca_basis(
        source_stats,
        retained_variance,
    )
    candidate_basis, candidate_eigenvalues = _pca_basis(
        candidate_stats,
        retained_variance,
    )
    whitened_cross = (
        source_basis.T @ cross @ candidate_basis
    ) / np.sqrt(
        source_eigenvalues[:, None] * candidate_eigenvalues[None, :]
    )
    left, correlations, right_t = np.linalg.svd(
        whitened_cross,
        full_matrices=False,
    )
    correlations = np.clip(correlations, 0.0, 1.0).astype(np.float32)
    source_directions = (
        source_basis / np.sqrt(source_eigenvalues)[None, :]
    ) @ left
    candidate_directions = (
        candidate_basis / np.sqrt(candidate_eigenvalues)[None, :]
    ) @ right_t.T
    source_canonical = source_stats.centered @ source_directions
    candidate_canonical = candidate_stats.centered @ candidate_directions
    angles = np.degrees(np.arccos(np.clip(correlations, -1.0, 1.0)))
    mean_shift = candidate_stats.coordinate_mean - source_stats.coordinate_mean
    metrics = {
        "linear_cka": float(np.clip(linear_cka, 0.0, 1.0 + 1e-5)),
        "orthogonal_procrustes_r2": float(procrustes_r2),
        "orthogonal_procrustes_relative_error": float(
            math.sqrt(procrustes_sse / max(candidate_energy, 1e-30))
        ),
        "svcca_mean_correlation": float(np.mean(correlations)),
        "svcca_min_correlation": float(np.min(correlations)),
        "pwcca_source_weighted_correlation": _pwcca_weighted_mean(
            source_stats.centered,
            source_canonical,
            correlations,
        ),
        "pwcca_candidate_weighted_correlation": _pwcca_weighted_mean(
            candidate_stats.centered,
            candidate_canonical,
            correlations,
        ),
        "principal_angle_degrees_min": float(np.min(angles)),
        "principal_angle_degrees_median": float(np.median(angles)),
        "principal_angle_degrees_mean": float(np.mean(angles)),
        "principal_angle_degrees_max": float(np.max(angles)),
        "cca_dimension": int(correlations.shape[0]),
        "source_pca_dimension": int(source_basis.shape[1]),
        "candidate_pca_dimension": int(candidate_basis.shape[1]),
        "coordinate_mean_shift_l2": float(np.linalg.norm(mean_shift)),
        "coordinate_mean_shift_relative": float(
            np.linalg.norm(mean_shift)
            / max(np.linalg.norm(source_stats.coordinate_mean), 1e-30)
        ),
    }
    return metrics, source_stats, candidate_stats, correlations


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed_label: str,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("Bootstrap input must be a finite vector with at least two values")
    rng = np.random.default_rng(_stable_seed(seed_label))
    means = np.empty((samples,), dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def board_drift_intervals(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    view: str,
    bootstrap_samples: int,
    seed_prefix: str,
) -> dict[str, Any]:
    source_values = np.asarray(source, dtype=np.float32)
    candidate_values = np.asarray(candidate, dtype=np.float32)
    if view == "board_pooled":
        source_values = np.mean(source_values, axis=1, dtype=np.float32)
        candidate_values = np.mean(candidate_values, axis=1, dtype=np.float32)
    else:
        source_values = source_values.reshape((source_values.shape[0], -1))
        candidate_values = candidate_values.reshape((candidate_values.shape[0], -1))
    source_flat = source_values.reshape((source_values.shape[0], -1))
    candidate_flat = candidate_values.reshape((candidate_values.shape[0], -1))
    differences = np.linalg.norm(candidate_flat - source_flat, axis=1)
    source_norms = np.linalg.norm(source_flat, axis=1)
    relative_l2 = differences / np.maximum(source_norms, 1e-30)
    cosine = np.sum(source_flat * candidate_flat, axis=1) / np.maximum(
        source_norms * np.linalg.norm(candidate_flat, axis=1),
        1e-30,
    )
    return {
        "board_cluster_relative_l2": bootstrap_mean_interval(
            relative_l2,
            samples=bootstrap_samples,
            seed_label=f"{seed_prefix}:relative_l2",
        ),
        "board_cluster_cosine": bootstrap_mean_interval(
            cosine,
            samples=bootstrap_samples,
            seed_label=f"{seed_prefix}:cosine",
        ),
    }


def _uniform_square_sketch(board_count: int) -> np.ndarray:
    board_indices = np.arange(board_count, dtype=np.int64)[:, None]
    ranks = np.arange(8, dtype=np.int64)[None, :]
    files = (ranks + board_indices) % 8
    return ranks * 8 + files


def _normalized_gram_stack(
    captures: np.ndarray,
    *,
    hook_index: int,
    view: str,
) -> np.ndarray:
    board_count = captures.shape[2]
    square_sketch = _uniform_square_sketch(board_count)
    board_index = np.arange(board_count, dtype=np.int64)[:, None]
    grams: list[np.ndarray] = []
    for layer_index in range(captures.shape[1]):
        activation = np.asarray(
            captures[hook_index, layer_index],
            dtype=np.float32,
        )
        if view == "board_pooled":
            matrix = np.mean(activation, axis=1, dtype=np.float32)
        elif view == "square_tokens":
            matrix = activation[board_index, square_sketch].reshape(
                (-1, activation.shape[-1])
            )
        else:
            raise ValueError(view)
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        gram = np.asarray(matrix @ matrix.T, dtype=np.float32)
        gram = gram - np.mean(gram, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        gram = gram - np.mean(gram, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        gram = gram + np.float32(np.mean(gram, dtype=np.float64))
        norm = float(np.linalg.norm(gram))
        grams.append((gram / max(norm, 1e-30)).reshape(-1))
    return np.stack(grams)


def layer_correspondence_matrix(
    source_captures: np.ndarray,
    candidate_captures: np.ndarray,
    *,
    hook_index: int,
    view: str,
) -> np.ndarray:
    source_grams = _normalized_gram_stack(
        source_captures,
        hook_index=hook_index,
        view=view,
    )
    candidate_grams = _normalized_gram_stack(
        candidate_captures,
        hook_index=hook_index,
        view=view,
    )
    matrix = source_grams @ candidate_grams.T
    return np.clip(matrix, 0.0, 1.0 + 1e-5).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted, dtype=np.float64)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _legal_probabilities(logits: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    masked = np.where(legal_masks, logits, -np.inf)
    maximum = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(legal_masks, np.exp(masked - maximum), 0.0)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _topk_indices(probabilities: np.ndarray, k: int) -> np.ndarray:
    partial = np.argpartition(-probabilities, kth=k - 1, axis=-1)[:, :k]
    values = np.take_along_axis(probabilities, partial, axis=-1)
    order = np.argsort(-values, axis=-1, stable=True)
    return np.take_along_axis(partial, order, axis=-1)


def policy_model_metrics(
    logits: np.ndarray,
    legal_masks: np.ndarray,
    targets: np.ndarray,
    *,
    bootstrap_samples: int,
    seed_prefix: str,
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    all_probabilities = _softmax(logits)
    probabilities = _legal_probabilities(logits, legal_masks)
    rows = np.arange(logits.shape[0])
    target_probabilities = probabilities[rows, targets]
    nll = -np.log(np.maximum(target_probabilities, 1e-300))
    legal_mass = np.sum(all_probabilities * legal_masks, axis=-1)
    top1 = np.argmax(probabilities, axis=-1)
    top5 = _topk_indices(probabilities, 5)
    top1_correct = (top1 == targets).astype(np.float64)
    top5_correct = np.any(top5 == targets[:, None], axis=-1).astype(np.float64)
    return {
        "target_nll": bootstrap_mean_interval(
            nll,
            samples=bootstrap_samples,
            seed_label=f"{seed_prefix}:target_nll",
        ),
        "top1_accuracy": bootstrap_mean_interval(
            top1_correct,
            samples=bootstrap_samples,
            seed_label=f"{seed_prefix}:top1",
        ),
        "top5_accuracy": bootstrap_mean_interval(
            top5_correct,
            samples=bootstrap_samples,
            seed_label=f"{seed_prefix}:top5",
        ),
        "raw_legal_probability_mass": bootstrap_mean_interval(
            legal_mass,
            samples=bootstrap_samples,
            seed_label=f"{seed_prefix}:legal_mass",
        ),
    }


def policy_pair_metrics(
    source_logits: np.ndarray,
    candidate_logits: np.ndarray,
    legal_masks: np.ndarray,
    *,
    bootstrap_samples: int,
    seed_prefix: str,
) -> dict[str, Any]:
    source = _legal_probabilities(np.asarray(source_logits, dtype=np.float64), legal_masks)
    candidate = _legal_probabilities(np.asarray(candidate_logits, dtype=np.float64), legal_masks)
    midpoint = 0.5 * (source + candidate)
    tiny = 1e-300
    source_kl = np.sum(source * (np.log(np.maximum(source, tiny)) - np.log(np.maximum(candidate, tiny))), axis=-1)
    candidate_kl = np.sum(candidate * (np.log(np.maximum(candidate, tiny)) - np.log(np.maximum(source, tiny))), axis=-1)
    js = 0.5 * (
        np.sum(source * (np.log(np.maximum(source, tiny)) - np.log(np.maximum(midpoint, tiny))), axis=-1)
        + np.sum(candidate * (np.log(np.maximum(candidate, tiny)) - np.log(np.maximum(midpoint, tiny))), axis=-1)
    )
    source_top1 = np.argmax(source, axis=-1)
    candidate_top1 = np.argmax(candidate, axis=-1)
    source_top5 = _topk_indices(source, 5)
    candidate_top5 = _topk_indices(candidate, 5)
    top5_jaccard = np.asarray(
        [
            len(set(left.tolist()) & set(right.tolist()))
            / len(set(left.tolist()) | set(right.tolist()))
            for left, right in zip(source_top5, candidate_top5, strict=True)
        ],
        dtype=np.float64,
    )
    metrics = {
        "source_to_candidate_kl": source_kl,
        "candidate_to_source_kl": candidate_kl,
        "jensen_shannon": js,
        "top1_agreement": (source_top1 == candidate_top1).astype(np.float64),
        "top5_set_jaccard": top5_jaccard,
    }
    return {
        name: bootstrap_mean_interval(
            values,
            samples=bootstrap_samples,
            seed_label=f"{seed_prefix}:{name}",
        )
        for name, values in metrics.items()
    }


def _independent_model_metrics(
    captures: np.ndarray,
    *,
    model_index: int,
    model_id: str,
    moments: np.ndarray,
    spectra: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view_index, view in enumerate(VIEW_NAMES):
        for hook_index, hook_name in enumerate(HOOK_NAMES):
            for layer_index in range(captures.shape[1]):
                matrix = activation_matrix(
                    captures[hook_index, layer_index],
                    view,
                )
                statistics = activation_statistics(matrix)
                moments[model_index, view_index, hook_index, layer_index, 0] = statistics.coordinate_mean
                moments[model_index, view_index, hook_index, layer_index, 1] = statistics.coordinate_rms
                moments[model_index, view_index, hook_index, layer_index, 2] = statistics.coordinate_variance
                spectra[model_index, view_index, hook_index, layer_index] = statistics.eigenvalues
                rows.append(
                    {
                        "model_id": model_id,
                        "view": view,
                        "hook": hook_name,
                        "layer": layer_index,
                        **statistics.summary,
                    }
                )
                del statistics, matrix
    return rows


def _paired_metrics(
    source_captures: np.ndarray,
    candidate_captures: np.ndarray,
    *,
    source_id: str,
    candidate_id: str,
    pair_index: int,
    cca_spectra: np.ndarray,
    retained_variance: float,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired_rows: list[dict[str, Any]] = []
    correspondence_rows: list[dict[str, Any]] = []
    for view_index, view in enumerate(VIEW_NAMES):
        for hook_index, hook_name in enumerate(HOOK_NAMES):
            correspondence = layer_correspondence_matrix(
                source_captures,
                candidate_captures,
                hook_index=hook_index,
                view=view,
            )
            correspondence_rows.append(
                {
                    "source_model_id": source_id,
                    "candidate_model_id": candidate_id,
                    "view": view,
                    "view_detail": (
                        "all 128 board-pooled samples"
                        if view == "board_pooled"
                        else "uniform deterministic 8-of-64 square sketch per board"
                    ),
                    "hook": hook_name,
                    "linear_cka_matrix": correspondence.tolist(),
                    "best_candidate_layer_by_source_layer": np.argmax(
                        correspondence,
                        axis=1,
                    ).astype(int).tolist(),
                    "best_source_layer_by_candidate_layer": np.argmax(
                        correspondence,
                        axis=0,
                    ).astype(int).tolist(),
                }
            )
            for layer_index in range(source_captures.shape[1]):
                source_activation = np.asarray(
                    source_captures[hook_index, layer_index],
                    dtype=np.float32,
                )
                candidate_activation = np.asarray(
                    candidate_captures[hook_index, layer_index],
                    dtype=np.float32,
                )
                source_matrix = activation_matrix(source_activation, view)
                candidate_matrix = activation_matrix(candidate_activation, view)
                metrics, _, _, correlations = paired_representation_metrics(
                    source_matrix,
                    candidate_matrix,
                    retained_variance=retained_variance,
                )
                cca_spectra[
                    pair_index,
                    view_index,
                    hook_index,
                    layer_index,
                    : correlations.shape[0],
                ] = correlations
                seed_prefix = (
                    f"{source_id}:{candidate_id}:{view}:{hook_name}:{layer_index}"
                )
                metrics.update(
                    board_drift_intervals(
                        source_activation,
                        candidate_activation,
                        view=view,
                        bootstrap_samples=bootstrap_samples,
                        seed_prefix=seed_prefix,
                    )
                )
                paired_rows.append(
                    {
                        "source_model_id": source_id,
                        "candidate_model_id": candidate_id,
                        "view": view,
                        "hook": hook_name,
                        "layer": layer_index,
                        **metrics,
                    }
                )
                del (
                    source_activation,
                    candidate_activation,
                    source_matrix,
                    candidate_matrix,
                    correlations,
                )
    return paired_rows, correspondence_rows


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _artifact_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(args: argparse.Namespace) -> Path:
    if not args.allow_cpu and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"Dense representation capture requires GPU, found {jax.default_backend()!r}"
        )
    if not 0.5 <= args.cca_retained_variance <= 1.0:
        raise ValueError("--cca-retained-variance must be in [0.5, 1.0]")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    output_dir = require_within_workspace(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Representation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-",
            dir=output_dir.parent,
        )
    )
    work_dir = require_within_workspace(work_dir)
    capture_dir = require_within_workspace(work_dir / "temporary-captures")
    capture_dir.mkdir()

    try:
        corpus = load_frozen_corpus(
            args.pool,
            args.data_root,
            example_count=args.examples,
        )
        write_json(work_dir / "corpus.json", {
            "contract": corpus.contract,
            "entries": list(corpus.entries),
        })
        _progress(
            "corpus_loaded",
            example_count=args.examples,
            entries_sha256=corpus.contract["entries_sha256"],
        )
        specs = default_model_specs()
        raw_model_path = require_within_workspace(
            args.models_dir / "BT4_exported.pb.gz"
        )
        raw_model_sha256 = sha256_file(raw_model_path)
        if raw_model_sha256 != "61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651":
            raise ValueError(f"Raw BT4 asset drift: {raw_model_path}")
        mapped_params = load_mapped_bt4_params(models_dir=str(args.models_dir))

        model_count = len(specs)
        pair_specs = tuple(
            (source_index, candidate_index)
            for source_index in range(model_count)
            for candidate_index in range(source_index + 1, model_count)
        )
        layer_count = 15
        feature_count = 1024
        moments = np.empty(
            (model_count, len(VIEW_NAMES), len(HOOK_NAMES), layer_count, 3, feature_count),
            dtype=np.float32,
        )
        spectra = np.empty(
            (model_count, len(VIEW_NAMES), len(HOOK_NAMES), layer_count, feature_count),
            dtype=np.float32,
        )
        cca_spectra = np.full(
            (len(pair_specs), len(VIEW_NAMES), len(HOOK_NAMES), layer_count, feature_count),
            np.nan,
            dtype=np.float32,
        )
        model_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        correspondence_rows: list[dict[str, Any]] = []
        model_records: list[dict[str, Any]] = []
        policy_model_rows: list[dict[str, Any]] = []
        policy_pair_rows: list[dict[str, Any]] = []
        capture_handles: dict[str, np.ndarray] = {}
        policy_handles: dict[str, np.ndarray] = {}
        capture_paths: dict[str, Path] = {}
        policy_paths: dict[str, Path] = {}

        raw_capture_path = capture_dir / "raw_bt4-captures.npy"
        raw_policy_path = capture_dir / "raw_bt4-policy.npy"
        _progress("model_capture_started", model_id=specs[0].model_id)
        raw_model, raw_metadata = make_fp32_backbone(mapped_params, specs[0])
        raw_capture = capture_model(
            raw_model,
            corpus.planes,
            raw_capture_path,
            raw_policy_path,
            batch_size=args.capture_batch_size,
        )
        del raw_model
        gc.collect()
        _progress(
            "model_capture_completed",
            model_id=specs[0].model_id,
            seconds=raw_capture["total_capture_seconds"],
        )
        raw_captures = np.load(raw_capture_path, mmap_mode="r")
        raw_policy = np.load(raw_policy_path, mmap_mode="r")
        capture_handles[specs[0].model_id] = raw_captures
        policy_handles[specs[0].model_id] = raw_policy
        capture_paths[specs[0].model_id] = raw_capture_path
        policy_paths[specs[0].model_id] = raw_policy_path
        model_rows.extend(
            _independent_model_metrics(
                raw_captures,
                model_index=0,
                model_id=specs[0].model_id,
                moments=moments,
                spectra=spectra,
            )
        )
        _progress("model_metrics_completed", model_id=specs[0].model_id)
        raw_policy_metrics = policy_model_metrics(
            raw_policy,
            corpus.legal_masks,
            corpus.target_actions,
            bootstrap_samples=args.bootstrap_samples,
            seed_prefix="raw_bt4",
        )
        policy_model_rows.append({"model_id": "raw_bt4", **raw_policy_metrics})
        model_records.append(
            {
                "model_id": specs[0].model_id,
                "kind": specs[0].kind,
                "parameter_compute_dtype": "float32",
                "fixed_head": "raw_bt4_source_policy_head",
                **raw_metadata,
                "capture": raw_capture,
            }
        )

        for model_index, spec in enumerate(specs[1:], start=1):
            candidate_capture_path = capture_dir / f"{spec.model_id}-captures.npy"
            candidate_policy_path = capture_dir / f"{spec.model_id}-policy.npy"
            _progress("model_capture_started", model_id=spec.model_id)
            model, metadata = make_fp32_backbone(mapped_params, spec)
            capture = capture_model(
                model,
                corpus.planes,
                candidate_capture_path,
                candidate_policy_path,
                batch_size=args.capture_batch_size,
            )
            del model
            gc.collect()
            _progress(
                "model_capture_completed",
                model_id=spec.model_id,
                seconds=capture["total_capture_seconds"],
            )
            candidate_captures = np.load(candidate_capture_path, mmap_mode="r")
            candidate_policy = np.load(candidate_policy_path, mmap_mode="r")
            capture_handles[spec.model_id] = candidate_captures
            policy_handles[spec.model_id] = candidate_policy
            capture_paths[spec.model_id] = candidate_capture_path
            policy_paths[spec.model_id] = candidate_policy_path
            model_rows.extend(
                _independent_model_metrics(
                    candidate_captures,
                    model_index=model_index,
                    model_id=spec.model_id,
                    moments=moments,
                    spectra=spectra,
                )
            )
            _progress("model_metrics_completed", model_id=spec.model_id)
            candidate_policy_metrics = policy_model_metrics(
                candidate_policy,
                corpus.legal_masks,
                corpus.target_actions,
                bootstrap_samples=args.bootstrap_samples,
                seed_prefix=spec.model_id,
            )
            policy_model_rows.append(
                {"model_id": spec.model_id, **candidate_policy_metrics}
            )
            model_record = {
                "model_id": spec.model_id,
                "kind": spec.kind,
                "parameter_compute_dtype": "float32",
                "checkpoint_online_storage_dtype": "bfloat16",
                "fixed_head": "raw_bt4_source_policy_head",
                **metadata,
                "capture": capture,
            }
            if spec.checkpoint_manifest is not None:
                model_record["checkpoint_manifest_path"] = str(spec.checkpoint_manifest)
                model_record["checkpoint_manifest_sha256"] = sha256_file(spec.checkpoint_manifest)
            model_records.append(model_record)
            gc.collect()

        for pair_index, (source_index, candidate_index) in enumerate(pair_specs):
            source_id = specs[source_index].model_id
            candidate_id = specs[candidate_index].model_id
            new_pair_rows, new_correspondence = _paired_metrics(
                capture_handles[source_id],
                capture_handles[candidate_id],
                source_id=source_id,
                candidate_id=candidate_id,
                pair_index=pair_index,
                cca_spectra=cca_spectra,
                retained_variance=args.cca_retained_variance,
                bootstrap_samples=args.bootstrap_samples,
            )
            pair_rows.extend(new_pair_rows)
            correspondence_rows.extend(new_correspondence)
            policy_pair_rows.append(
                {
                    "source_model_id": source_id,
                    "candidate_model_id": candidate_id,
                    **policy_pair_metrics(
                        policy_handles[source_id],
                        policy_handles[candidate_id],
                        corpus.legal_masks,
                        bootstrap_samples=args.bootstrap_samples,
                        seed_prefix=f"{source_id}:{candidate_id}",
                    ),
                }
            )
            _progress(
                "pair_metrics_completed",
                source_model_id=source_id,
                candidate_model_id=candidate_id,
            )

        _write_jsonl(work_dir / "per_model_metrics.jsonl", model_rows)
        _write_jsonl(work_dir / "pair_metrics.jsonl", pair_rows)
        _write_jsonl(work_dir / "layer_correspondence.jsonl", correspondence_rows)
        write_json(work_dir / "policy_metrics.json", {
            "per_model": policy_model_rows,
            "pairwise": policy_pair_rows,
        })
        np.savez_compressed(
            work_dir / "coordinate_moments.npz",
            model_ids=np.asarray([spec.model_id for spec in specs]),
            views=np.asarray(VIEW_NAMES),
            hooks=np.asarray(HOOK_NAMES),
            statistic_names=np.asarray(("mean", "rms", "centered_variance")),
            values=moments,
        )
        np.savez_compressed(
            work_dir / "covariance_spectra.npz",
            model_ids=np.asarray([spec.model_id for spec in specs]),
            views=np.asarray(VIEW_NAMES),
            hooks=np.asarray(HOOK_NAMES),
            eigenvalues=spectra,
        )
        np.savez_compressed(
            work_dir / "cca_spectra.npz",
            source_model_ids=np.asarray(
                [specs[source_index].model_id for source_index, _ in pair_specs]
            ),
            candidate_model_ids=np.asarray(
                [specs[candidate_index].model_id for _, candidate_index in pair_specs]
            ),
            views=np.asarray(VIEW_NAMES),
            hooks=np.asarray(HOOK_NAMES),
            canonical_correlations=cca_spectra,
        )

        final_hook_rows = {
            f"{row['source_model_id']}__to__{row['candidate_model_id']}": row
            for row in pair_rows
            if row["view"] == "square_tokens"
            and row["hook"] == "resid_post_after_ln"
            and row["layer"] == 14
        }
        summary = {
            "schema_version": RESULT_SCHEMA,
            "gate": "dense_stage1_core_complete_sparse_interpretation_still_blocked",
            "model_ids": [spec.model_id for spec in specs],
            "pair_count": len(pair_specs),
            "corpus": corpus.contract,
            "final_trunk_square_token_drift": final_hook_rows,
            "policy": {
                "per_model": policy_model_rows,
                "pairwise": policy_pair_rows,
            },
            "limitations": [
                "Cross-framework TransformerLens parity remains pending.",
                "Published sparse artifacts remain gated and were not loaded.",
                "Dense linear probes are a separate Stage-1 follow-up.",
                "Full 15x15 square-token correspondence uses the declared uniform token sketch; corresponding-layer metrics use all square tokens.",
            ],
        }
        write_json(work_dir / "summary.json", summary)
        write_json(work_dir / "models.json", {"models": model_records})

        del raw_captures, raw_policy, candidate_captures, candidate_policy
        capture_handles.clear()
        policy_handles.clear()
        gc.collect()
        for path in (*capture_paths.values(), *policy_paths.values()):
            path.unlink()
        if any(capture_dir.iterdir()):
            raise RuntimeError("Temporary capture directory is not empty")
        capture_dir.rmdir()

        durable_names = (
            "corpus.json",
            "models.json",
            "per_model_metrics.jsonl",
            "pair_metrics.jsonl",
            "layer_correspondence.jsonl",
            "policy_metrics.json",
            "coordinate_moments.npz",
            "covariance_spectra.npz",
            "cca_spectra.npz",
            "summary.json",
        )
        files = [
            _artifact_file_record(work_dir / name)
            for name in durable_names
        ]
        manifest = {
            "schema_version": RESULT_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [
                {
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                    "id": device.id,
                }
                for device in jax.devices()
            ],
            "fp32_capture": True,
            "hook_names": list(HOOK_NAMES),
            "view_names": list(VIEW_NAMES),
            "cca_retained_variance": args.cca_retained_variance,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "one pinned source game / board",
            "model_records_sha256": _json_sha256(model_records),
            "corpus_contract_sha256": _json_sha256(corpus.contract),
            "raw_bt4_path": str(raw_model_path),
            "raw_bt4_sha256": raw_model_sha256,
            "temporary_activation_captures_retained": False,
            "files": files,
        }
        write_json(work_dir / "manifest.json", manifest)
        os.replace(work_dir, output_dir)
        _progress("result_published", output_dir=str(output_dir))
        return output_dir
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--capture-batch-size", type=int, default=4)
    parser.add_argument("--cca-retained-variance", type=float, default=0.99)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a deliberately small diagnostic run without a GPU.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = run(args)
    print(json.dumps({"output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
