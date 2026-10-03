"""Checkpoint-specific board-input contracts, independent of model architecture."""

from collections.abc import Mapping
from typing import Any


CURRENT_ONLY = "current_only_as_preprocessed"
LC0_FULL_HISTORY = "lc0_full_history_as_preprocessed"
PLANE_HISTORY_MODES = frozenset({CURRENT_ONLY, LC0_FULL_HISTORY})


def require_training_history(data_format: str, *, allow_legacy_current_only: bool = False) -> str:
    """New training must preserve history; legacy reproduction is explicit."""
    if data_format in {"lc0_sequential", "trajectory_v3_history"}:
        return LC0_FULL_HISTORY
    if data_format == "trajectory_v3" and allow_legacy_current_only:
        return CURRENT_ONLY
    if data_format == "trajectory_v3":
        raise ValueError(
            "Legacy trajectory_v3 has zeroed board history. New training requires "
            "--data-format lc0_sequential and a full-history dataset. Only explicit "
            "historical reproduction may use --allow-legacy-current-only."
        )
    raise ValueError(f"Unsupported training data format: {data_format!r}")


def audit_full_history_batch(batch: Mapping[str, Any], *, allow_chess960_initial: bool = False) -> dict[str, int]:
    """Check actual input bytes, not just the dataset's name.

    Every valid future target is after at least one move, including targets
    of an initial-position example. Classical LC0 must retain its predecessor.
    """
    import numpy as np
    import chess
    from chess_dfm_jax.encoding import encode_board

    current = np.asarray(batch["current_planes"])
    future = np.asarray(batch["future_planes"])
    valid = np.asarray(batch["future_valid"], dtype=bool)
    if (current.ndim != 4 or future.ndim != 5 or current.shape[0] != future.shape[0]
            or current.shape[-3:] != (112, 8, 8) or future.shape[-3:] != (112, 8, 8)):
        raise ValueError("History audit requires classical 112-plane inputs")
    if not np.isfinite(current).all() or not np.isfinite(future).all():
        raise ValueError("History audit rejects nonfinite input planes")
    if future.shape[:2] != valid.shape or not valid.any():
        raise ValueError("History audit requires valid future targets")
    present = np.any(future[..., 13:104, :, :] != 0, axis=(-3, -2, -1))
    if np.any(valid & ~present):
        raise ValueError("Full-history training data contains a zero-history future target")
    current_present = np.any(current[:, 13:104] != 0, axis=(1, 2, 3))
    # The real initial position has no predecessor. Do not mistake that valid
    # boundary for a broken dataset; any other zero-history current row fails.
    initial = encode_board(chess.Board(), [])
    initial_rows = np.all(current == initial, axis=(1, 2, 3))
    if allow_chess960_initial:
        # Only genuine initial arrangements qualify; never accept an arbitrary
        # FEN as a history-free root. This opt-in is for the retained mixed Hero
        # corpus, whose complete prefixes were separately replayed on CPU.
        for row in np.flatnonzero(~current_present & ~initial_rows):
            for index in range(960):
                if np.array_equal(current[row], encode_board(chess.Board.from_chess960_pos(index), [])):
                    initial_rows[row] = True
                    break
    if np.any(~current_present & ~initial_rows):
        raise ValueError("Full-history training data contains a zero-history noninitial current position")
    return {
        "examples": int(current.shape[0]),
        "current_with_history": int(current_present.sum()),
        "valid_future_targets": int(valid.sum()),
        "future_targets_with_history": int((present & valid).sum()),
    }


def torch_hero_plane_history_mode(run_config: Mapping[str, Any]) -> str:
    """Infer the immutable preprocessing contract from recorded training data.

    Legacy Hero runs predate ``data_format`` and used trajectory-v3. Never
    infer preprocessing from a model's display name or its parameter shapes.
    """
    args = run_config.get("args", {})
    data = run_config.get("data", {})
    data_format = args.get("data_format")
    schema = data.get("schema_version")
    explicit = run_config.get("input_contract", {}).get("plane_history_mode")
    if data_format == "trajectory_v3_history" or schema == "hero-trajectory-full-history-v1":
        if (data_format != "trajectory_v3_history" or schema != "hero-trajectory-full-history-v1"
                or explicit != LC0_FULL_HISTORY or not data.get("manifest_sha256")):
            raise ValueError("Incomplete or conflicting reconstructed-history provenance")
        return LC0_FULL_HISTORY
    if data_format == "lc0_sequential" or schema == "lc0-sequential-v1":
        if data_format not in {None, "lc0_sequential"} or schema not in {
            None, "lc0-sequential-v1"
        }:
            raise ValueError("Conflicting training input-history provenance")
        if explicit not in {None, LC0_FULL_HISTORY}:
            raise ValueError("Explicit input history conflicts with training provenance")
        return LC0_FULL_HISTORY
    if data_format not in {None, "trajectory_v3"} or schema not in {
        None, "trajectory-v3"
    }:
        raise ValueError("Unsupported training input-history provenance")
    if explicit not in {None, CURRENT_ONLY}:
        raise ValueError("Explicit input history conflicts with training provenance")
    return CURRENT_ONLY
