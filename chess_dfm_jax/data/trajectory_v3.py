"""Compact trajectory-v3 helpers for Latent-SASA data."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from chess_dfm_jax.data.trajectory import (
    TRAJECTORY_V2,
    infer_trajectory_schema,
    terminal_target_indices,
)


TRAJECTORY_V3 = "trajectory-v3"
PLANE_SHAPE = (112, 8, 8)
PLANE_SIZE = 112 * 8 * 8
PACKED_PLANE_SIZE = PLANE_SIZE // 8
ACTION_VOCAB_SIZE = 1858
LEGAL_PAD = np.iinfo(np.uint16).max


def _require_binary_planes(planes: np.ndarray, *, name: str, strict: bool) -> np.ndarray:
    array = np.asarray(planes)
    if array.shape[-3:] != PLANE_SHAPE:
        raise ValueError(f"{name} must end with shape {PLANE_SHAPE}, got {array.shape}.")
    if strict:
        finite = np.isfinite(array)
        binary = np.isclose(array, 0.0) | np.isclose(array, 1.0)
        if not bool(np.all(finite & binary)):
            raise ValueError(f"{name} contains non-binary or non-finite plane values.")
    return (array > 0.5).astype(np.uint8, copy=False)


def pack_planes(planes: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Pack binary `[... ,112,8,8]` planes to `[... ,896]` uint8 bytes."""
    binary = _require_binary_planes(planes, name="planes", strict=strict)
    prefix = binary.shape[:-3]
    flat = binary.reshape((-1, PLANE_SIZE))
    packed = np.packbits(flat, axis=1, bitorder="big")
    return packed.reshape((*prefix, PACKED_PLANE_SIZE)).astype(np.uint8, copy=False)


def unpack_planes(packed: np.ndarray) -> np.ndarray:
    """Unpack `[... ,896]` uint8 bytes to float32 `[... ,112,8,8]` planes."""
    array = np.asarray(packed, dtype=np.uint8)
    if array.shape[-1] != PACKED_PLANE_SIZE:
        raise ValueError(f"packed planes must end with {PACKED_PLANE_SIZE}, got {array.shape}.")
    prefix = array.shape[:-1]
    flat = array.reshape((-1, PACKED_PLANE_SIZE))
    bits = np.unpackbits(flat, axis=1, count=PLANE_SIZE, bitorder="big")
    return bits.reshape((*prefix, *PLANE_SHAPE)).astype(np.float32)


def encode_planes_u8(planes: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Encode integral `[... ,112,8,8]` planes as uint8 without changing shape."""
    array = np.asarray(planes)
    if array.shape[-3:] != PLANE_SHAPE:
        raise ValueError(f"planes must end with shape {PLANE_SHAPE}, got {array.shape}.")
    if strict:
        finite = np.isfinite(array)
        rounded = np.rint(array)
        integral = np.isclose(array, rounded)
        in_range = (rounded >= 0) & (rounded <= np.iinfo(np.uint8).max)
        if not bool(np.all(finite & integral & in_range)):
            raise ValueError("planes contain non-integral, non-finite, or out-of-uint8-range values.")
    return np.rint(array).astype(np.uint8)


def decode_planes_u8(planes: np.ndarray) -> np.ndarray:
    array = np.asarray(planes, dtype=np.uint8)
    if array.shape[-3:] != PLANE_SHAPE:
        raise ValueError(f"uint8 planes must end with shape {PLANE_SHAPE}, got {array.shape}.")
    return array.astype(np.float32)


def legal_masks_to_indices(
    legal_masks: np.ndarray,
    *,
    lmax: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense legal masks `[B,H,1858]` to padded uint16 legal lists."""
    masks = np.asarray(legal_masks) > 0
    if masks.ndim != 3 or masks.shape[-1] != ACTION_VOCAB_SIZE:
        raise ValueError(f"legal_masks must have shape [B,H,{ACTION_VOCAB_SIZE}], got {masks.shape}.")
    if lmax <= 0 or lmax >= LEGAL_PAD:
        raise ValueError(f"lmax must be in [1, {LEGAL_PAD - 1}], got {lmax}.")
    batch, horizon, _ = masks.shape
    indices = np.full((batch, horizon, lmax), LEGAL_PAD, dtype=np.uint16)
    counts = np.zeros((batch, horizon), dtype=np.uint16)
    for sample_idx in range(batch):
        for horizon_idx in range(horizon):
            legal = np.flatnonzero(masks[sample_idx, horizon_idx])
            if legal.size > lmax:
                raise ValueError(
                    "Legal move count exceeds lmax at "
                    f"sample={sample_idx}, horizon={horizon_idx}: {legal.size} > {lmax}."
                )
            counts[sample_idx, horizon_idx] = np.uint16(legal.size)
            indices[sample_idx, horizon_idx, : legal.size] = legal.astype(np.uint16)
    return indices, counts


def legal_indices_to_masks(
    legal_idx: np.ndarray,
    legal_count: np.ndarray,
    *,
    vocab_size: int = ACTION_VOCAB_SIZE,
) -> np.ndarray:
    """Reconstruct dense float32 masks from padded legal-index lists."""
    indices = np.asarray(legal_idx, dtype=np.uint16)
    counts = np.asarray(legal_count, dtype=np.uint16)
    if indices.ndim != 3:
        raise ValueError(f"legal_idx must have rank 3, got {indices.shape}.")
    if counts.shape != indices.shape[:2]:
        raise ValueError(f"legal_count must have shape {indices.shape[:2]}, got {counts.shape}.")
    masks = np.zeros((*indices.shape[:2], vocab_size), dtype=np.float32)
    for sample_idx in range(indices.shape[0]):
        for horizon_idx in range(indices.shape[1]):
            count = int(counts[sample_idx, horizon_idx])
            if count == 0:
                continue
            values = indices[sample_idx, horizon_idx, :count]
            valid = values[values != LEGAL_PAD].astype(np.int32)
            masks[sample_idx, horizon_idx, valid] = 1.0
    return masks


def trajectory_v3_from_v2_npz(
    data: Mapping[str, np.ndarray],
    *,
    legal_lmax: int = 128,
    strict_binary_planes: bool = False,
    plane_codec: str = "uint8",
    include_metadata: bool = True,
) -> dict[str, np.ndarray]:
    """Convert a trajectory-v2 mapping to a compact per-shard trajectory-v3 mapping."""
    schema = infer_trajectory_schema(data)
    if schema != TRAJECTORY_V2:
        raise ValueError(f"trajectory-v3 conversion requires trajectory-v2 shards, got {schema}.")

    planes_t = np.asarray(data["planes_t"], dtype=np.float32)
    planes_future = np.asarray(data["planes_future"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int32)
    future_valid = np.asarray(data.get("future_valid", np.ones(actions.shape, dtype=np.float32)))
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [B,H], got {actions.shape}.")
    batch_size, horizon = actions.shape
    if planes_t.shape != (batch_size, *PLANE_SHAPE):
        raise ValueError(f"planes_t must have shape {(batch_size, *PLANE_SHAPE)}, got {planes_t.shape}.")
    if planes_future.shape != (batch_size, horizon, *PLANE_SHAPE):
        raise ValueError(
            f"planes_future must have shape {(batch_size, horizon, *PLANE_SHAPE)}, got {planes_future.shape}."
        )
    if future_valid.shape != (batch_size, horizon):
        raise ValueError(f"future_valid must have shape {(batch_size, horizon)}, got {future_valid.shape}.")
    if np.any(actions < 0) or np.any(actions >= ACTION_VOCAB_SIZE):
        raise ValueError("actions contain indices outside the LC0 1858-action vocabulary.")

    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRAJECTORY_V3),
        "source_schema_version": np.asarray(str(np.asarray(data.get("schema_version", TRAJECTORY_V2)).item())),
        "plane_codec": np.asarray(plane_codec),
        "legal_codec": np.asarray(f"indices-u16-lmax-{legal_lmax}"),
        "batch_size": np.asarray(batch_size, dtype=np.int32),
        "horizon": np.asarray(horizon, dtype=np.int32),
        "actions_u16": actions.astype(np.uint16),
        "future_valid_u8": (future_valid > 0).astype(np.uint8),
    }
    if plane_codec == "uint8":
        output["planes_t_u8"] = encode_planes_u8(planes_t, strict=True)
        output["planes_future_u8"] = encode_planes_u8(planes_future, strict=True)
    elif plane_codec == "packbits":
        output["planes_t_pack"] = pack_planes(planes_t, strict=strict_binary_planes)
        output["planes_future_pack"] = pack_planes(planes_future, strict=strict_binary_planes)
    else:
        raise ValueError(f"Unsupported trajectory-v3 plane codec: {plane_codec}")

    if "legal_masks" in data:
        legal_idx, legal_count = legal_masks_to_indices(data["legal_masks"], lmax=legal_lmax)
        output["legal_idx_u16"] = legal_idx
        output["legal_count_u16"] = legal_count
    if "legal_masks_valid" in data:
        output["legal_valid_u8"] = (np.asarray(data["legal_masks_valid"]) > 0).astype(np.uint8)
    if "value_targets" in data:
        output["value_targets"] = np.asarray(data["value_targets"], dtype=np.float32)
    if "wdl_targets" in data:
        output["wdl_targets"] = np.asarray(data["wdl_targets"], dtype=np.float32)

    if include_metadata:
        for key in ("source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"):
            if key in data:
                output[key] = np.asarray(data[key])
    return output


def trajectory_v3_to_batch(
    data: Mapping[str, np.ndarray],
    *,
    view: str = "full",
    horizon: int | None = None,
    include_metadata: bool = False,
) -> dict[str, np.ndarray]:
    """Decode a trajectory-v3 shard to the existing training batch contract."""
    schema = str(np.asarray(data["schema_version"]).item())
    if schema != TRAJECTORY_V3:
        raise ValueError(f"Expected {TRAJECTORY_V3}, got {schema}.")
    actions = np.asarray(data["actions_u16"], dtype=np.int32)
    shard_horizon = actions.shape[1]
    view_horizon = shard_horizon if horizon is None or horizon <= 0 else min(horizon, shard_horizon)
    actions = actions[:, :view_horizon]
    future_valid = np.asarray(data["future_valid_u8"], dtype=np.float32)[:, :view_horizon]
    batch_size = actions.shape[0]

    plane_codec = str(np.asarray(data.get("plane_codec", "uint8")).item())
    if plane_codec == "packbits":
        current_planes = unpack_planes(data["planes_t_pack"])
    elif plane_codec == "uint8":
        current_planes = decode_planes_u8(data["planes_t_u8"])
    else:
        raise ValueError(f"Unsupported trajectory-v3 plane codec: {plane_codec}")

    batch: dict[str, np.ndarray] = {
        "current_planes": current_planes,
        "action_indices": actions,
        "action_idx": actions[:, 0],
        "future_valid": future_valid,
        "terminal_target_index": terminal_target_indices(future_valid),
        "valid": (future_valid.sum(axis=1) > 0).astype(np.float32),
    }

    if "legal_idx_u16" in data and "legal_count_u16" in data:
        legal_masks = legal_indices_to_masks(
            np.asarray(data["legal_idx_u16"], dtype=np.uint16)[:, :view_horizon],
            np.asarray(data["legal_count_u16"], dtype=np.uint16)[:, :view_horizon],
        )
        batch["legal_masks"] = legal_masks
        batch["legal_mask"] = legal_masks[:, 0]
    else:
        batch["legal_mask"] = np.ones((batch_size, ACTION_VOCAB_SIZE), dtype=np.float32)
    if "legal_valid_u8" in data:
        batch["legal_masks_valid"] = np.asarray(data["legal_valid_u8"], dtype=np.float32)[:, :view_horizon]

    if view == "full":
        if plane_codec == "packbits":
            future_planes = unpack_planes(data["planes_future_pack"])[:, :view_horizon]
        else:
            future_planes = decode_planes_u8(data["planes_future_u8"])[:, :view_horizon]
        terminal_idx = batch["terminal_target_index"]
        batch["future_planes"] = future_planes
        batch["next_planes"] = future_planes[np.arange(batch_size), terminal_idx]
        if "value_targets" in data:
            value_targets = np.asarray(data["value_targets"], dtype=np.float32)[:, :view_horizon]
            batch["value_targets"] = value_targets
            batch["value_target"] = value_targets[np.arange(batch_size), terminal_idx]
        if "wdl_targets" in data:
            wdl_targets = np.asarray(data["wdl_targets"], dtype=np.float32)[:, :view_horizon]
            batch["wdl_targets"] = wdl_targets
            batch["wdl_target"] = wdl_targets[np.arange(batch_size), terminal_idx]
    elif view != "dfm_action":
        raise ValueError(f"Unsupported trajectory-v3 view: {view}")

    if include_metadata:
        for key in ("source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"):
            if key in data:
                batch[key] = np.asarray(data[key])
    return batch


__all__ = [
    "ACTION_VOCAB_SIZE",
    "LEGAL_PAD",
    "PACKED_PLANE_SIZE",
    "PLANE_SHAPE",
    "TRAJECTORY_V3",
    "decode_planes_u8",
    "encode_planes_u8",
    "legal_indices_to_masks",
    "legal_masks_to_indices",
    "pack_planes",
    "trajectory_v3_from_v2_npz",
    "trajectory_v3_to_batch",
    "unpack_planes",
]
