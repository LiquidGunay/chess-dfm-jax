"""Trajectory shard helpers for state-action future rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import legal_move_mask, move_to_policy_index, policy_index_to_move


TRAJECTORY_V1 = "trajectory-v1"
TRAJECTORY_V1_ADAPTED = "trajectory-v1-adapted"
TRAJECTORY_V2 = "trajectory-v2"
DEFAULT_INPUT_FORMAT = "INPUT_CLASSICAL_112_PLANE"
ACTION_VOCAB_SIZE = 1858
DEFAULT_LEGAL_LMAX = 128
LEGAL_PAD = np.iinfo(np.uint16).max


@dataclass
class TrajectoryShard:
    schema_version: str
    planes_t: np.ndarray
    actions: np.ndarray
    planes_future: np.ndarray
    future_valid: np.ndarray
    legal_masks: np.ndarray | None = None
    legal_masks_valid: np.ndarray | None = None
    value_targets: np.ndarray | None = None
    wdl_targets: np.ndarray | None = None
    source: np.ndarray | None = None
    game_id: np.ndarray | None = None
    ply: np.ndarray | None = None
    result: np.ndarray | None = None
    fen_t: np.ndarray | None = None
    input_format: np.ndarray | None = None
    actions_uci: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return int(self.planes_t.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])


def _expand_optional(values: np.ndarray | None, *, batch_size: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 0:
        return np.repeat(array[None], batch_size, axis=0)
    return array


def _maybe_get(data: Mapping[str, np.ndarray], key: str, default=None):
    return data[key] if key in data else default


def _normalize_actions(actions: np.ndarray) -> np.ndarray:
    array = np.asarray(actions, dtype=np.int32)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected actions with rank 1 or 2, got shape {array.shape}.")
    return array


def infer_trajectory_schema(data: Mapping[str, np.ndarray]) -> str:
    if "planes_future" in data:
        return TRAJECTORY_V2
    if "planes_target" in data:
        return TRAJECTORY_V1
    raise KeyError("Shard is missing trajectory keys: expected planes_future or planes_target.")


def validate_trajectory_shard(shard: TrajectoryShard) -> None:
    """Validate the trajectory-v2 shape and masking contract."""
    batch_size, horizon = shard.actions.shape
    if shard.planes_t.shape != (batch_size, 112, 8, 8):
        raise ValueError(f"planes_t must have shape {(batch_size, 112, 8, 8)}, got {shard.planes_t.shape}.")
    if shard.planes_future.shape != (batch_size, horizon, 112, 8, 8):
        raise ValueError(
            "planes_future must have shape "
            f"{(batch_size, horizon, 112, 8, 8)}, got {shard.planes_future.shape}."
        )
    if shard.future_valid.shape != (batch_size, horizon):
        raise ValueError(
            f"future_valid must have shape {(batch_size, horizon)}, got {shard.future_valid.shape}."
        )

    valid_future = np.asarray(shard.future_valid, dtype=np.float32) > 0
    if np.any(valid_future):
        zero_valid = np.isclose(np.abs(shard.planes_future).sum(axis=(2, 3, 4)), 0.0) & valid_future
        if np.any(zero_valid):
            bad = np.argwhere(zero_valid)[0]
            raise ValueError(
                "planes_future contains a zero-filled valid horizon at "
                f"sample={int(bad[0])}, horizon={int(bad[1])}."
            )

    if shard.legal_masks is not None:
        if shard.legal_masks.shape != (batch_size, horizon, 1858):
            raise ValueError(
                f"legal_masks must have shape {(batch_size, horizon, 1858)}, got {shard.legal_masks.shape}."
            )
        if shard.legal_masks_valid is not None and shard.legal_masks_valid.shape != (batch_size, horizon):
            raise ValueError(
                "legal_masks_valid must have shape "
                f"{(batch_size, horizon)}, got {shard.legal_masks_valid.shape}."
            )
        mask_valid = (
            np.asarray(shard.legal_masks_valid, dtype=np.float32) > 0
            if shard.legal_masks_valid is not None
            else valid_future
        )
        for sample_idx, horizon_idx in np.argwhere(mask_valid):
            action_idx = int(shard.actions[sample_idx, horizon_idx])
            if action_idx < 0 or action_idx >= shard.legal_masks.shape[-1]:
                raise ValueError(
                    f"Action index out of legal-mask range at sample={sample_idx}, horizon={horizon_idx}: {action_idx}."
                )
            if shard.legal_masks[sample_idx, horizon_idx, action_idx] <= 0:
                raise ValueError(
                    f"Recorded action is illegal at sample={sample_idx}, horizon={horizon_idx}: {action_idx}."
                )

    if shard.value_targets is not None and shard.value_targets.shape != (batch_size, horizon):
        raise ValueError(
            f"value_targets must have shape {(batch_size, horizon)}, got {shard.value_targets.shape}."
        )
    if shard.wdl_targets is not None and shard.wdl_targets.shape != (batch_size, horizon, 3):
        raise ValueError(
            f"wdl_targets must have shape {(batch_size, horizon, 3)}, got {shard.wdl_targets.shape}."
        )

    for name in ("source", "game_id", "ply", "result", "fen_t", "input_format"):
        values = getattr(shard, name)
        if values is not None and np.asarray(values).shape[0] != batch_size:
            raise ValueError(f"{name} must have batch dimension {batch_size}, got {np.asarray(values).shape}.")

    if shard.actions_uci is not None and np.asarray(shard.actions_uci).shape[:2] != (batch_size, horizon):
        raise ValueError(
            f"actions_uci must have leading shape {(batch_size, horizon)}, got {np.asarray(shard.actions_uci).shape}."
        )


def trajectory_shard_from_npz(data: Mapping[str, np.ndarray]) -> TrajectoryShard:
    schema = infer_trajectory_schema(data)
    planes_t = np.asarray(data["planes_t"], dtype=np.float32)
    actions = _normalize_actions(data["actions"])
    batch_size, horizon = actions.shape

    if schema == TRAJECTORY_V2:
        planes_future = np.asarray(data["planes_future"], dtype=np.float32)
        future_valid = np.asarray(
            _maybe_get(data, "future_valid", np.ones((batch_size, horizon), dtype=np.float32)),
            dtype=np.float32,
        )
        legal_masks = (
            np.asarray(data["legal_masks"], dtype=np.float32) if "legal_masks" in data else None
        )
        legal_masks_valid = (
            np.asarray(data["legal_masks_valid"], dtype=np.float32)
            if "legal_masks_valid" in data
            else None
        )
        value_targets = (
            np.asarray(data["value_targets"], dtype=np.float32)
            if "value_targets" in data
            else None
        )
        wdl_targets = (
            np.asarray(data["wdl_targets"], dtype=np.float32) if "wdl_targets" in data else None
        )
        schema_version = str(np.asarray(_maybe_get(data, "schema_version", TRAJECTORY_V2)).item())
    else:
        planes_target = np.asarray(data["planes_target"], dtype=np.float32)
        planes_future = np.zeros((batch_size, horizon, 112, 8, 8), dtype=np.float32)
        planes_future[:, -1] = planes_target
        future_valid = np.zeros((batch_size, horizon), dtype=np.float32)
        future_valid[:, -1] = 1.0

        legal_masks = None
        legal_masks_valid = None
        if "legal_mask" in data:
            legal_masks = np.zeros((batch_size, horizon, 1858), dtype=np.float32)
            legal_masks[:, 0] = np.asarray(data["legal_mask"], dtype=np.float32)
            legal_masks_valid = np.zeros((batch_size, horizon), dtype=np.float32)
            legal_masks_valid[:, 0] = 1.0

        value_targets = None
        if "value_target" in data:
            value_targets = np.zeros((batch_size, horizon), dtype=np.float32)
            value_targets[:, -1] = np.asarray(data["value_target"], dtype=np.float32)

        wdl_targets = None
        if "wdl_target" in data:
            wdl_targets = np.zeros((batch_size, horizon, 3), dtype=np.float32)
            wdl_targets[:, -1] = np.asarray(data["wdl_target"], dtype=np.float32)

        schema_version = TRAJECTORY_V1_ADAPTED

    shard = TrajectoryShard(
        schema_version=schema_version,
        planes_t=planes_t,
        actions=actions,
        planes_future=planes_future,
        future_valid=future_valid,
        legal_masks=legal_masks,
        legal_masks_valid=legal_masks_valid,
        value_targets=value_targets,
        wdl_targets=wdl_targets,
        source=_expand_optional(_maybe_get(data, "source"), batch_size=batch_size),
        game_id=_expand_optional(_maybe_get(data, "game_id"), batch_size=batch_size),
        ply=_expand_optional(_maybe_get(data, "ply"), batch_size=batch_size),
        result=_expand_optional(_maybe_get(data, "result"), batch_size=batch_size),
        fen_t=_expand_optional(_maybe_get(data, "fen_t"), batch_size=batch_size),
        input_format=_expand_optional(_maybe_get(data, "input_format"), batch_size=batch_size),
        actions_uci=_expand_optional(_maybe_get(data, "actions_uci"), batch_size=batch_size),
    )
    validate_trajectory_shard(shard)
    return shard


def load_trajectory_shard(path: str | Path) -> TrajectoryShard:
    with np.load(path, allow_pickle=False) as data:
        return trajectory_shard_from_npz(data)


def terminal_target_indices(future_valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(future_valid, dtype=np.float32)
    reversed_idx = np.argmax(valid[:, ::-1] > 0, axis=1)
    has_valid = np.any(valid > 0, axis=1)
    last_valid = valid.shape[1] - 1 - reversed_idx
    return np.where(has_valid, last_valid, 0).astype(np.int32)


def legal_masks_to_indices(
    legal_masks: np.ndarray,
    *,
    lmax: int = DEFAULT_LEGAL_LMAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense legal masks `[B,H,1858]` to padded compact legal lists."""
    masks = np.asarray(legal_masks) > 0
    if masks.ndim != 3 or masks.shape[-1] != ACTION_VOCAB_SIZE:
        raise ValueError(f"legal_masks must have shape [B,H,{ACTION_VOCAB_SIZE}], got {masks.shape}.")
    if lmax <= 0 or lmax >= LEGAL_PAD:
        raise ValueError(f"lmax must be in [1, {LEGAL_PAD - 1}], got {lmax}.")
    batch_size, horizon, _ = masks.shape
    indices = np.full((batch_size, horizon, lmax), LEGAL_PAD, dtype=np.uint16)
    counts = np.zeros((batch_size, horizon), dtype=np.uint16)
    for sample_idx in range(batch_size):
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


def trajectory_shard_to_batch(
    shard: TrajectoryShard,
    *,
    include_metadata: bool = False,
) -> dict[str, np.ndarray]:
    batch_size = shard.batch_size
    terminal_idx = terminal_target_indices(shard.future_valid)
    terminal_planes = shard.planes_future[np.arange(batch_size), terminal_idx]
    valid = (shard.future_valid.sum(axis=1) > 0).astype(np.float32)

    batch: dict[str, np.ndarray] = {
        "current_planes": np.asarray(shard.planes_t, dtype=np.float32),
        "action_indices": np.asarray(shard.actions, dtype=np.int32),
        "action_idx": np.asarray(shard.actions[:, 0], dtype=np.int32),
        "future_planes": np.asarray(shard.planes_future, dtype=np.float32),
        "future_valid": np.asarray(shard.future_valid, dtype=np.float32),
        "terminal_target_index": terminal_idx.astype(np.int32),
        "next_planes": np.asarray(terminal_planes, dtype=np.float32),
        "valid": valid,
    }

    if shard.legal_masks is not None:
        batch["legal_masks"] = np.asarray(shard.legal_masks, dtype=np.float32)
        batch["legal_mask"] = np.asarray(shard.legal_masks[:, 0], dtype=np.float32)
    else:
        batch["legal_mask"] = np.ones((batch_size, 1858), dtype=np.float32)

    if shard.legal_masks_valid is not None:
        batch["legal_masks_valid"] = np.asarray(shard.legal_masks_valid, dtype=np.float32)

    if shard.value_targets is not None:
        value_targets = np.asarray(shard.value_targets, dtype=np.float32)
        batch["value_targets"] = value_targets
        batch["value_target"] = value_targets[np.arange(batch_size), terminal_idx]
    else:
        batch["value_targets"] = np.zeros((batch_size, shard.horizon), dtype=np.float32)
        batch["value_target"] = np.zeros((batch_size,), dtype=np.float32)

    if shard.wdl_targets is not None:
        wdl_targets = np.asarray(shard.wdl_targets, dtype=np.float32)
        batch["wdl_targets"] = wdl_targets
        batch["wdl_target"] = wdl_targets[np.arange(batch_size), terminal_idx]
    else:
        batch["wdl_targets"] = np.zeros((batch_size, shard.horizon, 3), dtype=np.float32)
        batch["wdl_target"] = np.zeros((batch_size, 3), dtype=np.float32)

    if include_metadata:
        for name in ("source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"):
            value = getattr(shard, name)
            if value is not None:
                batch[name] = np.asarray(value)

    return batch


def trajectory_action_batch_from_npz(
    data: Mapping[str, np.ndarray],
    *,
    horizon: int | None = None,
    include_metadata: bool = False,
) -> dict[str, np.ndarray]:
    """Build a DFM action batch without materializing future board planes.

    Dense trajectory-v2 shards store `planes_future`, which dominates host memory
    and load time. DFM action training does not need those boards, so this view
    reads only current planes, actions, validity, and legal masks.
    """
    schema = infer_trajectory_schema(data)
    planes_t = np.asarray(data["planes_t"], dtype=np.float32)
    actions = _normalize_actions(data["actions"])
    batch_size, shard_horizon = actions.shape
    view_horizon = shard_horizon if horizon is None or horizon <= 0 else min(horizon, shard_horizon)
    actions = actions[:, :view_horizon]

    if schema == TRAJECTORY_V2:
        if "future_valid" in data:
            future_valid = np.asarray(data["future_valid"], dtype=np.float32)[:, :view_horizon]
        else:
            future_valid = np.ones((batch_size, view_horizon), dtype=np.float32)
        legal_masks = (
            np.asarray(data["legal_masks"], dtype=np.float32)[:, :view_horizon]
            if "legal_masks" in data
            else None
        )
        legal_masks_valid = (
            np.asarray(data["legal_masks_valid"], dtype=np.float32)[:, :view_horizon]
            if "legal_masks_valid" in data
            else None
        )
    else:
        future_valid = np.zeros((batch_size, view_horizon), dtype=np.float32)
        if view_horizon > 0:
            future_valid[:, -1] = 1.0
        legal_masks = None
        legal_masks_valid = None
        if "legal_mask" in data and view_horizon > 0:
            legal_masks = np.zeros((batch_size, view_horizon, 1858), dtype=np.float32)
            legal_masks[:, 0] = np.asarray(data["legal_mask"], dtype=np.float32)
            legal_masks_valid = np.zeros((batch_size, view_horizon), dtype=np.float32)
            legal_masks_valid[:, 0] = 1.0

    if planes_t.shape != (batch_size, 112, 8, 8):
        raise ValueError(f"planes_t must have shape {(batch_size, 112, 8, 8)}, got {planes_t.shape}.")
    if future_valid.shape != (batch_size, view_horizon):
        raise ValueError(
            f"future_valid must have shape {(batch_size, view_horizon)}, got {future_valid.shape}."
        )
    if legal_masks is not None:
        if legal_masks.shape != (batch_size, view_horizon, 1858):
            raise ValueError(
                f"legal_masks must have shape {(batch_size, view_horizon, 1858)}, got {legal_masks.shape}."
            )
        if legal_masks_valid is not None and legal_masks_valid.shape != (batch_size, view_horizon):
            raise ValueError(
                "legal_masks_valid must have shape "
                f"{(batch_size, view_horizon)}, got {legal_masks_valid.shape}."
            )
        mask_valid = (
            np.asarray(legal_masks_valid, dtype=np.float32) > 0
            if legal_masks_valid is not None
            else np.asarray(future_valid, dtype=np.float32) > 0
        )
        for sample_idx, horizon_idx in np.argwhere(mask_valid):
            action_idx = int(actions[sample_idx, horizon_idx])
            if action_idx < 0 or action_idx >= legal_masks.shape[-1]:
                raise ValueError(
                    f"Action index out of legal-mask range at sample={sample_idx}, horizon={horizon_idx}: {action_idx}."
                )
            if legal_masks[sample_idx, horizon_idx, action_idx] <= 0:
                raise ValueError(
                    f"Recorded action is illegal at sample={sample_idx}, horizon={horizon_idx}: {action_idx}."
                )

    valid = (future_valid.sum(axis=1) > 0).astype(np.float32)
    batch: dict[str, np.ndarray] = {
        "current_planes": planes_t,
        "action_indices": actions.astype(np.int32),
        "action_idx": actions[:, 0].astype(np.int32),
        "future_valid": future_valid,
        "terminal_target_index": terminal_target_indices(future_valid),
        "valid": valid,
    }
    if legal_masks is not None:
        batch["legal_masks"] = legal_masks
        batch["legal_mask"] = legal_masks[:, 0]
    else:
        batch["legal_mask"] = np.ones((batch_size, 1858), dtype=np.float32)
    if legal_masks_valid is not None:
        batch["legal_masks_valid"] = legal_masks_valid
    if include_metadata:
        for name in ("source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"):
            if name in data:
                batch[name] = _expand_optional(data[name], batch_size=batch_size)
    return batch


def trajectory_latent_batch_from_npz(
    data: Mapping[str, np.ndarray],
    *,
    horizon: int | None = None,
    include_metadata: bool = False,
) -> dict[str, np.ndarray]:
    """Build a JEPA latent batch without materializing unused legal masks."""
    schema = infer_trajectory_schema(data)
    planes_t = np.asarray(data["planes_t"], dtype=np.float32)
    actions = _normalize_actions(data["actions"])
    batch_size, shard_horizon = actions.shape
    view_horizon = shard_horizon if horizon is None or horizon <= 0 else min(horizon, shard_horizon)
    actions = actions[:, :view_horizon]

    if schema == TRAJECTORY_V2:
        planes_future = np.asarray(data["planes_future"], dtype=np.float32)[:, :view_horizon]
        future_valid = np.asarray(
            _maybe_get(data, "future_valid", np.ones((batch_size, shard_horizon), dtype=np.float32)),
            dtype=np.float32,
        )[:, :view_horizon]
        value_targets = (
            np.asarray(data["value_targets"], dtype=np.float32)[:, :view_horizon]
            if "value_targets" in data
            else np.zeros((batch_size, view_horizon), dtype=np.float32)
        )
        wdl_targets = (
            np.asarray(data["wdl_targets"], dtype=np.float32)[:, :view_horizon]
            if "wdl_targets" in data
            else np.zeros((batch_size, view_horizon, 3), dtype=np.float32)
        )
    else:
        planes_target = np.asarray(data["planes_target"], dtype=np.float32)
        planes_future = np.zeros((batch_size, view_horizon, 112, 8, 8), dtype=np.float32)
        future_valid = np.zeros((batch_size, view_horizon), dtype=np.float32)
        if view_horizon > 0:
            planes_future[:, -1] = planes_target
            future_valid[:, -1] = 1.0
        value_targets = np.zeros((batch_size, view_horizon), dtype=np.float32)
        if "value_target" in data and view_horizon > 0:
            value_targets[:, -1] = np.asarray(data["value_target"], dtype=np.float32)
        wdl_targets = np.zeros((batch_size, view_horizon, 3), dtype=np.float32)
        if "wdl_target" in data and view_horizon > 0:
            wdl_targets[:, -1] = np.asarray(data["wdl_target"], dtype=np.float32)

    if planes_t.shape != (batch_size, 112, 8, 8):
        raise ValueError(f"planes_t must have shape {(batch_size, 112, 8, 8)}, got {planes_t.shape}.")
    if planes_future.shape != (batch_size, view_horizon, 112, 8, 8):
        raise ValueError(
            f"planes_future must have shape {(batch_size, view_horizon, 112, 8, 8)}, got {planes_future.shape}."
        )
    if future_valid.shape != (batch_size, view_horizon):
        raise ValueError(
            f"future_valid must have shape {(batch_size, view_horizon)}, got {future_valid.shape}."
        )

    terminal_idx = terminal_target_indices(future_valid)
    valid = (future_valid.sum(axis=1) > 0).astype(np.float32)
    batch: dict[str, np.ndarray] = {
        "current_planes": planes_t,
        "action_indices": actions.astype(np.int32),
        "action_idx": actions[:, 0].astype(np.int32),
        "future_planes": planes_future,
        "future_valid": future_valid,
        "terminal_target_index": terminal_idx.astype(np.int32),
        "next_planes": planes_future[np.arange(batch_size), terminal_idx],
        "valid": valid,
        "value_targets": value_targets,
        "value_target": value_targets[np.arange(batch_size), terminal_idx],
        "wdl_targets": wdl_targets,
        "wdl_target": wdl_targets[np.arange(batch_size), terminal_idx],
    }
    if include_metadata:
        for name in ("source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"):
            if name in data:
                batch[name] = _expand_optional(data[name], batch_size=batch_size)
    return batch


def trajectory_joint_batch_from_npz(
    data: Mapping[str, np.ndarray],
    *,
    horizon: int | None = None,
    legal_lmax: int = DEFAULT_LEGAL_LMAX,
    include_metadata: bool = False,
) -> dict[str, np.ndarray]:
    """Build a joint Latent-SASA batch with future states and compact legal sets."""
    schema = infer_trajectory_schema(data)
    if schema != TRAJECTORY_V2:
        raise ValueError("joint_latent_sasa view requires trajectory-v2 shards.")

    planes_t = np.asarray(data["planes_t"], dtype=np.float32)
    actions = _normalize_actions(data["actions"])
    batch_size, shard_horizon = actions.shape
    view_horizon = shard_horizon if horizon is None or horizon <= 0 else min(horizon, shard_horizon)
    actions = actions[:, :view_horizon]

    planes_future = np.asarray(data["planes_future"], dtype=np.float32)[:, :view_horizon]
    future_valid = np.asarray(
        _maybe_get(data, "future_valid", np.ones((batch_size, shard_horizon), dtype=np.float32)),
        dtype=np.float32,
    )[:, :view_horizon]
    if "legal_masks" not in data:
        raise KeyError("joint_latent_sasa view requires legal_masks for trajectory-v2 shards.")
    legal_masks = np.asarray(data["legal_masks"], dtype=np.float32)[:, :view_horizon]
    legal_idx, legal_count = legal_masks_to_indices(legal_masks, lmax=legal_lmax)
    legal_valid = (
        np.asarray(data["legal_masks_valid"], dtype=np.float32)[:, :view_horizon]
        if "legal_masks_valid" in data
        else np.ones((batch_size, view_horizon), dtype=np.float32)
    )

    if planes_t.shape != (batch_size, 112, 8, 8):
        raise ValueError(f"planes_t must have shape {(batch_size, 112, 8, 8)}, got {planes_t.shape}.")
    if planes_future.shape != (batch_size, view_horizon, 112, 8, 8):
        raise ValueError(
            f"planes_future must have shape {(batch_size, view_horizon, 112, 8, 8)}, got {planes_future.shape}."
        )
    if future_valid.shape != (batch_size, view_horizon):
        raise ValueError(
            f"future_valid must have shape {(batch_size, view_horizon)}, got {future_valid.shape}."
        )

    terminal_idx = terminal_target_indices(future_valid)
    valid = (future_valid.sum(axis=1) > 0).astype(np.float32)
    value_targets = (
        np.asarray(data["value_targets"], dtype=np.float32)[:, :view_horizon]
        if "value_targets" in data
        else np.zeros((batch_size, view_horizon), dtype=np.float32)
    )
    wdl_targets = (
        np.asarray(data["wdl_targets"], dtype=np.float32)[:, :view_horizon]
        if "wdl_targets" in data
        else np.zeros((batch_size, view_horizon, 3), dtype=np.float32)
    )
    batch: dict[str, np.ndarray] = {
        "current_planes": planes_t,
        "action_indices": actions.astype(np.int32),
        "action_idx": actions[:, 0].astype(np.int32),
        "future_planes": planes_future,
        "future_valid": future_valid,
        "terminal_target_index": terminal_idx.astype(np.int32),
        "next_planes": planes_future[np.arange(batch_size), terminal_idx],
        "valid": valid,
        "legal_idx": legal_idx.astype(np.int32),
        "legal_count": legal_count.astype(np.int32),
        "legal_masks_valid": legal_valid,
        "value_targets": value_targets,
        "value_target": value_targets[np.arange(batch_size), terminal_idx],
        "wdl_targets": wdl_targets,
        "wdl_target": wdl_targets[np.arange(batch_size), terminal_idx],
    }
    if include_metadata:
        for name in ("source", "game_id", "ply", "result", "fen_t", "input_format", "actions_uci"):
            if name in data:
                batch[name] = _expand_optional(data[name], batch_size=batch_size)
    return batch


def rollout_from_fen(
    fen: str,
    actions: np.ndarray,
    *,
    input_format: str = DEFAULT_INPUT_FORMAT,
) -> dict[str, np.ndarray]:
    if chess is None:  # pragma: no cover
        raise ImportError("python-chess is required for rollout validation.")

    board = chess.Board(fen)
    legal_masks = []
    future_planes = []
    actions_uci = []
    for action_idx in np.asarray(actions, dtype=np.int32):
        legal_masks.append(legal_move_mask(board, "lc0_1858").astype(np.float32))
        move = policy_index_to_move(int(action_idx), "lc0_1858")
        if move not in board.legal_moves:
            raise ValueError(f"Illegal action {action_idx} for board {fen}.")
        actions_uci.append(move.uci())
        board.push(move)
        future_planes.append(encode_board(board, [], input_format=input_format).astype(np.float32))

    return {
        "legal_masks": np.stack(legal_masks, axis=0).astype(np.float32),
        "planes_future": np.stack(future_planes, axis=0).astype(np.float32),
        "actions_uci": np.asarray(actions_uci),
    }


def build_synthetic_trajectory_shard(batch_size: int = 2, horizon: int = 4) -> TrajectoryShard:
    if chess is None:  # pragma: no cover
        raise ImportError("python-chess is required for synthetic trajectory construction.")

    lines = [
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"],
        ["d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "c8f5", "f1d3", "e7e6"],
        ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5"],
    ]
    if horizon > min(len(line) for line in lines):
        raise ValueError(f"Synthetic trajectories support horizon <= {min(len(line) for line in lines)}, got {horizon}.")

    line_payloads = []
    for line_idx, opening in enumerate(lines):
        board = chess.Board()
        selected = opening[:horizon]
        action_ids = np.asarray(
            [move_to_policy_index(chess.Move.from_uci(uci), "lc0_1858") for uci in selected],
            dtype=np.int32,
        )
        rollout = rollout_from_fen(board.fen(), action_ids)
        draw_wdl = np.zeros((horizon, 3), dtype=np.float32)
        draw_wdl[:, 1] = 1.0
        line_payloads.append(
            {
                "planes_t": encode_board(board, [], input_format=DEFAULT_INPUT_FORMAT).astype(np.float32),
                "actions": action_ids,
                "planes_future": rollout["planes_future"],
                "legal_masks": rollout["legal_masks"],
                "value_targets": np.zeros((horizon,), dtype=np.float32),
                "wdl_targets": draw_wdl,
                "future_valid": np.ones((horizon,), dtype=np.float32),
                "source": "synthetic",
                "game_id": f"synthetic-line-{line_idx}",
                "ply": 0,
                "result": "1/2-1/2",
                "fen_t": board.fen(),
                "actions_uci": rollout["actions_uci"],
                "input_format": DEFAULT_INPUT_FORMAT,
            }
        )

    planes_t = []
    actions = []
    planes_future = []
    legal_masks = []
    value_targets = []
    wdl_targets = []
    future_valid = []
    source = []
    game_id = []
    ply = []
    result = []
    fen_t = []
    actions_uci = []
    input_format = []

    for sample_idx in range(batch_size):
        payload = line_payloads[sample_idx % len(line_payloads)]
        planes_t.append(payload["planes_t"])
        actions.append(payload["actions"])
        planes_future.append(payload["planes_future"])
        legal_masks.append(payload["legal_masks"])
        value_targets.append(payload["value_targets"])
        wdl_targets.append(payload["wdl_targets"])
        future_valid.append(payload["future_valid"])
        source.append(payload["source"])
        game_id.append(payload["game_id"])
        ply.append(payload["ply"])
        result.append(payload["result"])
        fen_t.append(payload["fen_t"])
        actions_uci.append(payload["actions_uci"])
        input_format.append(payload["input_format"])

    return TrajectoryShard(
        schema_version=TRAJECTORY_V2,
        planes_t=np.stack(planes_t, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.int32),
        planes_future=np.stack(planes_future, axis=0).astype(np.float32),
        future_valid=np.stack(future_valid, axis=0).astype(np.float32),
        legal_masks=np.stack(legal_masks, axis=0).astype(np.float32),
        value_targets=np.stack(value_targets, axis=0).astype(np.float32),
        wdl_targets=np.stack(wdl_targets, axis=0).astype(np.float32),
        source=np.asarray(source),
        game_id=np.asarray(game_id),
        ply=np.asarray(ply, dtype=np.int32),
        result=np.asarray(result),
        fen_t=np.asarray(fen_t),
        input_format=np.asarray(input_format),
        actions_uci=np.stack(actions_uci, axis=0),
    )


__all__ = [
    "ACTION_VOCAB_SIZE",
    "DEFAULT_INPUT_FORMAT",
    "DEFAULT_LEGAL_LMAX",
    "LEGAL_PAD",
    "TRAJECTORY_V1",
    "TRAJECTORY_V1_ADAPTED",
    "TRAJECTORY_V2",
    "TrajectoryShard",
    "build_synthetic_trajectory_shard",
    "infer_trajectory_schema",
    "load_trajectory_shard",
    "legal_masks_to_indices",
    "rollout_from_fen",
    "terminal_target_indices",
    "trajectory_action_batch_from_npz",
    "trajectory_joint_batch_from_npz",
    "trajectory_latent_batch_from_npz",
    "trajectory_shard_from_npz",
    "trajectory_shard_to_batch",
    "validate_trajectory_shard",
]
