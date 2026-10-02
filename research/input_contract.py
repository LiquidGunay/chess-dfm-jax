"""Checkpoint-specific board-input contracts, independent of model architecture."""

from collections.abc import Mapping
from typing import Any


CURRENT_ONLY = "current_only_as_preprocessed"
LC0_FULL_HISTORY = "lc0_full_history_as_preprocessed"
PLANE_HISTORY_MODES = frozenset({CURRENT_ONLY, LC0_FULL_HISTORY})


def torch_hero_plane_history_mode(run_config: Mapping[str, Any]) -> str:
    """Infer the immutable preprocessing contract from recorded training data.

    Legacy Hero runs predate ``data_format`` and used trajectory-v3. Never
    infer preprocessing from a model's display name or its parameter shapes.
    """
    args = run_config.get("args", {})
    data = run_config.get("data", {})
    data_format = args.get("data_format")
    schema = data.get("schema_version")
    if data_format == "lc0_sequential" or schema == "lc0-sequential-v1":
        if data_format not in {None, "lc0_sequential"} or schema not in {
            None, "lc0-sequential-v1"
        }:
            raise ValueError("Conflicting training input-history provenance")
        return LC0_FULL_HISTORY
    if data_format not in {None, "trajectory_v3"} or schema not in {
        None, "trajectory-v3"
    }:
        raise ValueError("Unsupported training input-history provenance")
    return CURRENT_ONLY
