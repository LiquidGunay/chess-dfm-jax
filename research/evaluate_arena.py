#!/usr/bin/env python3
"""Resumable in-process relative-Elo arena for local research checkpoints.

This command deliberately reports only checkpoint-relative chess strength.
It does not produce human, Lichess, or absolute Elo.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.nnx_bt4 import make_bt4_model
from chess_dfm_jax.policy import (
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    legal_action_mask,
)
from research.arena import (
    GSPRTConfig,
    GSPRTState,
    action_codec_capability,
    load_opening_pool,
    make_color_reversed_pairs,
    pair_aware_score_elo_interval,
    pair_score_for_model,
    pentanomial_normalized_elo_diagnostics,
    pentanomial_stats,
)
from research.arena_history_trust import (
    HISTORY_VALIDATION_FULL_REPLAY,
    HISTORY_VALIDATION_SCHEMA,
    HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
)
from research.import_legacy import import_legacy_checkpoint
from research.local_policy import (
    PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
    STATIC_INFERENCE_BATCHING_SCHEMA,
    STATIC_INFERENCE_PADDING_MODE,
    LocalDFMPolicy,
)
from research.raw_bt4_policy import (
    RAW_BT4_POLICY_COMPUTE_DTYPE,
    LocalBT4Policy,
)
from research.play_arena import (
    ArenaGameplayResult,
    BatchedArenaPolicy,
    LoadedOpeningHistories,
    histories_for_pairs,
    load_opening_history_sidecar,
    play_arena_pairs,
)
from research.prepare import (
    REPO_ROOT,
    load_asset_manifest,
    require_within_workspace,
    sha256_file,
)


ARENA_RUN_SCHEMA = "chess-dfm-relative-arena-run-v3"
ARENA_BLOCK_SCHEMA = "chess-dfm-relative-arena-block-v3"
RELATIVE_ELO_SCOPE = "checkpoint_pool_relative_only"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "source" / "extracted"
DEFAULT_SOURCE_RUN_ROOT = REPO_ROOT / "checkpoints" / "source" / "step0265000"


def _descriptor_uses_jepa_at_inference(
    descriptor: Mapping[str, Any],
) -> bool:
    """Return whether a model descriptor enables model-internal JEPA feedback."""
    model_config = descriptor.get("model_config")
    if not isinstance(model_config, Mapping):
        return False
    return bool(
        model_config.get("jepa_feedback_mode", "none")
        == "final_pass_adjoint"
        or model_config.get(
            "dfm_condition_on_current_jepa_state",
            False,
        )
        or model_config.get("dfm_jepa_fusion_mode", "none") != "none"
        or model_config.get("dfm_closed_loop_mode", "none") != "none"
    )


@dataclasses.dataclass(frozen=True)
class FrozenTier:
    name: str
    pool_path: Path
    pool_sha256: str
    histories_path: Path
    history_manifest_sha256: str
    default_pair_count: int
    default_block_pairs: int
    default_additional_ply_cap: int | None
    promotion_eligible: bool
    available: bool
    unavailable_reason: str | None


FROZEN_TIERS = {
    "hero_correctness": FrozenTier(
        name="hero_correctness",
        pool_path=REPO_ROOT
        / "research"
        / "assets"
        / "arena"
        / "promotion-ply12-n2048-v3.json",
        pool_sha256=(
            "e750e87643c482d28b4201668bd355234fe1fb27f2a690bf784706b1ddd37459"
        ),
        histories_path=REPO_ROOT
        / "research"
        / "assets"
        / "arena"
        / "promotion-ply12-n2048-histories-v3.json",
        history_manifest_sha256=(
            "d8781efb5d76066bcf2ce2e9ab2897f3261b20f7c4488c90992a8eadcc918a4b"
        ),
        default_pair_count=16,
        default_block_pairs=16,
        default_additional_ply_cap=16,
        promotion_eligible=False,
        available=True,
        unavailable_reason=None,
    ),
    "hero_development": FrozenTier(
        name="hero_development",
        pool_path=REPO_ROOT
        / "research"
        / "assets"
        / "arena"
        / "promotion-ply12-n2048-v3.json",
        pool_sha256=(
            "e750e87643c482d28b4201668bd355234fe1fb27f2a690bf784706b1ddd37459"
        ),
        histories_path=REPO_ROOT
        / "research"
        / "assets"
        / "arena"
        / "promotion-ply12-n2048-histories-v3.json",
        history_manifest_sha256=(
            "d8781efb5d76066bcf2ce2e9ab2897f3261b20f7c4488c90992a8eadcc918a4b"
        ),
        default_pair_count=2048,
        default_block_pairs=16,
        default_additional_ply_cap=256,
        promotion_eligible=False,
        available=True,
        unavailable_reason=None,
    ),
    "correctness": FrozenTier(
        name="correctness",
        pool_path=REPO_ROOT / "artifacts" / "arena" / "development-ply12-n128-v1.json",
        pool_sha256=("451784d106bc25b06a3891e910213220275d93eae434badddfe1a38b2e58138a"),
        histories_path=REPO_ROOT
        / "artifacts"
        / "arena"
        / "development-ply12-n128-histories-v1.json",
        history_manifest_sha256=(
            "496eace967e101fea28c9c26d6f9517c311d9e7f82a127510d9852397578a36c"
        ),
        default_pair_count=16,
        default_block_pairs=16,
        default_additional_ply_cap=16,
        promotion_eligible=False,
        available=True,
        unavailable_reason=None,
    ),
    "development": FrozenTier(
        name="development",
        pool_path=REPO_ROOT / "artifacts" / "arena" / "development-ply12-n128-v1.json",
        pool_sha256=("451784d106bc25b06a3891e910213220275d93eae434badddfe1a38b2e58138a"),
        histories_path=REPO_ROOT
        / "artifacts"
        / "arena"
        / "development-ply12-n128-histories-v1.json",
        history_manifest_sha256=(
            "496eace967e101fea28c9c26d6f9517c311d9e7f82a127510d9852397578a36c"
        ),
        default_pair_count=128,
        default_block_pairs=16,
        default_additional_ply_cap=None,
        promotion_eligible=False,
        available=True,
        unavailable_reason=None,
    ),
    "promotion": FrozenTier(
        name="promotion",
        pool_path=REPO_ROOT
        / "research"
        / "assets"
        / "arena"
        / "promotion-ply12-n2048-v3.json",
        pool_sha256=("e750e87643c482d28b4201668bd355234fe1fb27f2a690bf784706b1ddd37459"),
        histories_path=REPO_ROOT
        / "research"
        / "assets"
        / "arena"
        / "promotion-ply12-n2048-histories-v3.json",
        history_manifest_sha256=(
            "d8781efb5d76066bcf2ce2e9ab2897f3261b20f7c4488c90992a8eadcc918a4b"
        ),
        default_pair_count=2048,
        default_block_pairs=1,
        default_additional_ply_cap=None,
        promotion_eligible=True,
        available=True,
        unavailable_reason=None,
    ),
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _with_payload_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = _json_sha256(result)
    return result


def _validate_payload_sha256(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    observed = payload.get("payload_sha256")
    value = dict(payload)
    value.pop("payload_sha256", None)
    expected = _json_sha256(value)
    if observed != expected:
        raise ValueError(f"{label} payload digest mismatch: expected {expected}, found {observed}")


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = require_within_workspace(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_path(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_provenance() -> dict[str, Any]:
    paths = (
        "research/train.py",
        "research/train_torch.py",
        "research/evaluate_torch_migration_parity.py",
        "research/inference.py",
        "research/local_policy.py",
        "research/raw_bt4_policy.py",
        "research/arena_history_trust.py",
        "research/arena.py",
        "research/play_arena.py",
        "research/evaluate_arena.py",
        "chess_dfm_jax/policy.py",
        "chess_dfm_jax/encoding.py",
    )
    return {
        "git_commit": _git_commit(),
        "files": {relative: sha256_file(REPO_ROOT / relative) for relative in paths},
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    source = require_within_workspace(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object: {source}")
    return value


@dataclasses.dataclass(frozen=True)
class ResearchCheckpointDescriptor:
    checkpoint_dir: Path
    manifest: dict[str, Any]
    config: Any
    descriptor: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class TorchResearchCheckpointDescriptor:
    checkpoint_dir: Path
    manifest: dict[str, Any]
    run_config: dict[str, Any]
    descriptor: dict[str, Any]


def _validated_torch_hero_model_config(
    recorded_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a recorded hero config without forcing incumbent equality."""

    from research.train_torch import Config

    field_names = {field.name for field in dataclasses.fields(Config)}
    recorded_names = set(recorded_config)
    unknown = recorded_names - field_names
    missing = field_names - recorded_names
    legacy_missing = {
        "jepa_feedback_mode",
        "dfm_condition_on_current_jepa_state",
        "dfm_jepa_fusion_mode",
        "policy_passthrough_mode",
        "dfm_state_source",
        "wdl_include_current_state",
        "dfm_closed_loop_mode",
    }
    if unknown:
        raise ValueError(
            "Torch hero run has unrecognized model config keys: "
            f"{sorted(unknown)}"
        )
    if missing - legacy_missing:
        raise ValueError(
            "Torch hero run is missing model config keys: "
            f"{sorted(missing - legacy_missing)}"
        )
    try:
        config = Config(**dict(recorded_config))
    except TypeError as exc:
        raise ValueError("Torch hero run has an invalid model config.") from exc

    defaults = dataclasses.asdict(Config())
    normalized = dataclasses.asdict(config)
    for name, default in defaults.items():
        value = normalized[name]
        if type(default) is bool:
            valid_type = type(value) is bool
        elif type(default) is int:
            valid_type = type(value) is int
        elif type(default) is float:
            valid_type = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )
        elif isinstance(default, str):
            valid_type = isinstance(value, str)
        else:
            valid_type = value is None or type(value) is bool
        if not valid_type:
            raise ValueError(
                f"Torch hero run has an invalid {name!r} value."
            )

    positive_dimensions = (
        "horizon",
        "token_dim",
        "z_dim",
        "projector_layers",
        "projector_heads",
        "projector_mlp_dim",
        "dfm_layers",
        "dfm_heads",
        "dfm_mlp_dim",
        "jepa_layers",
        "jepa_mlp_dim",
    )
    if any(normalized[name] < 1 for name in positive_dimensions):
        raise ValueError(
            "Torch hero run has a non-positive model dimension or depth."
        )
    if (
        normalized["token_dim"] % normalized["projector_heads"] != 0
        or normalized["token_dim"] % normalized["dfm_heads"] != 0
    ):
        raise ValueError(
            "Torch hero attention heads must divide the token dimension."
        )
    if normalized["horizon"] != 8:
        raise ValueError("Torch hero Arena requires horizon eight.")
    if normalized["action_codec"] != ACTION_CODEC_LC0_CANONICAL_1858:
        raise ValueError("Torch hero Arena requires the canonical codec.")
    if normalized["use_bt4_policy_residual"] is not True:
        raise ValueError("Torch hero Arena requires the BT4 policy residual.")
    if normalized["jepa_feedback_mode"] not in {
        "none",
        "final_pass_adjoint",
    }:
        raise ValueError("Torch hero run has an invalid JEPA feedback mode.")
    required_runtime = {
        "remat_bt4_blocks": True,
        "remat_projector_blocks": True,
        "remat_dfm_blocks": False,
        "use_bt4_sdpa": True,
        "use_head_sdpa": True,
    }
    if any(
        normalized[name] is not expected
        for name, expected in required_runtime.items()
    ):
        raise ValueError("Torch hero runtime config mismatch.")

    for name in legacy_missing - recorded_names:
        normalized.pop(name)
    if normalized != dict(recorded_config):
        raise ValueError("Torch hero model config is not canonically typed.")
    return normalized


def research_checkpoint_descriptor(
    path: Path,
    *,
    models_dir: Path,
) -> ResearchCheckpointDescriptor:
    from research.train import (
        JointLatentSASAConfig,
        RESEARCH_CHECKPOINT_FORMAT,
        resolve_research_checkpoint,
    )

    checkpoint_dir = resolve_research_checkpoint(path)
    manifest_path = require_within_workspace(checkpoint_dir / "manifest.json")
    manifest = _load_json_object(
        manifest_path,
        label="research checkpoint manifest",
    )
    if manifest.get("format") != RESEARCH_CHECKPOINT_FORMAT:
        raise ValueError("Unsupported research checkpoint format.")
    contract = manifest.get("resume_contract")
    if not isinstance(contract, dict):
        raise ValueError("Research checkpoint has no resume contract.")
    if _json_sha256(contract) != manifest.get("resume_contract_sha256"):
        raise ValueError("Research checkpoint resume-contract digest mismatch.")
    raw_config = contract.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Research checkpoint has no model_config.")
    config = JointLatentSASAConfig(**raw_config)

    state = manifest.get("state")
    if not isinstance(state, dict) or state.get("filename") != "state.npz":
        raise ValueError("Research checkpoint has an invalid state record.")
    state_path = require_within_workspace(checkpoint_dir / "state.npz")
    if not state_path.is_file() or state_path.stat().st_size != int(state.get("size_bytes", -1)):
        raise ValueError("Research checkpoint state size mismatch.")

    model_path = require_within_workspace(models_dir / "BT4_exported.pb.gz")
    assets = contract.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("Research checkpoint has no asset contract.")
    if require_within_workspace(assets.get("bt4_exported_path", "")) != model_path:
        raise ValueError("Research checkpoint names a different BT4 constructor asset.")
    model_sha256 = sha256_file(model_path)
    if assets.get("bt4_exported_sha256") != model_sha256:
        raise ValueError("Research checkpoint BT4 constructor digest mismatch.")
    descriptor = {
        "kind": "research",
        "action_codec_id": ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        "plane_history_mode": PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "research_update": int(manifest["research_update"]),
        "optimizer_step": int(manifest["optimizer_step"]),
        "resume_contract_sha256": manifest["resume_contract_sha256"],
        "state": {
            "path": str(state_path),
            "size_bytes": int(state["size_bytes"]),
            "sha256": str(state["sha256"]),
        },
        "lineage": manifest.get("lineage"),
        "model_abi_sha256": manifest.get("model_abi", {}).get("sha256"),
        "model_config": raw_config,
        "bt4_constructor": {
            "path": str(model_path),
            "sha256": model_sha256,
        },
    }
    return ResearchCheckpointDescriptor(
        checkpoint_dir=checkpoint_dir,
        manifest=manifest,
        config=config,
        descriptor=descriptor,
    )


def torch_research_checkpoint_descriptor(
    path: Path,
    *,
    models_dir: Path,
) -> TorchResearchCheckpointDescriptor:
    """Describe a strict model-only checkpoint from the eager-Torch trainer."""

    from research.train_torch import CONFIG

    candidate = require_within_workspace(path)
    if (candidate / "checkpoint" / "manifest.json").is_file():
        checkpoint_dir = require_within_workspace(candidate / "checkpoint")
        run_root = candidate
    elif (candidate / "manifest.json").is_file():
        checkpoint_dir = candidate
        run_root = require_within_workspace(candidate.parent)
    else:
        raise FileNotFoundError(
            f"No eager-Torch checkpoint manifest under {candidate}"
        )

    manifest_path = require_within_workspace(checkpoint_dir / "manifest.json")
    manifest = _load_json_object(
        manifest_path,
        label="eager-Torch checkpoint manifest",
    )
    if manifest.get("format") != "chess-dfm-torch-model-v1":
        raise ValueError("Unsupported eager-Torch checkpoint format.")
    if manifest.get("model_only") is not True:
        raise ValueError("Eager-Torch arena checkpoint must be model-only.")
    state = manifest.get("state")
    if not isinstance(state, dict) or state.get("path") != "model.safetensors":
        raise ValueError("Eager-Torch checkpoint has an invalid state record.")
    state_path = require_within_workspace(checkpoint_dir / "model.safetensors")
    if (
        not state_path.is_file()
        or state_path.stat().st_size != int(state.get("size_bytes", -1))
    ):
        raise ValueError("Eager-Torch checkpoint state size mismatch.")

    run_config_path = require_within_workspace(run_root / "run_config.json")
    run_config = _load_json_object(
        run_config_path,
        label="eager-Torch run config",
    )
    if (
        run_config.get("framework") != "torch"
        or run_config.get("execution") != "eager"
        or run_config.get("torch_compile") is not False
    ):
        raise ValueError("Eager-Torch run execution contract mismatch.")
    expected_config = dataclasses.asdict(CONFIG)
    if run_config.get("config") != expected_config:
        raise ValueError(
            "Eager-Torch checkpoint config differs from the checked-out "
            "one-file trainer; evaluate it at its recorded git commit."
        )

    model_path = require_within_workspace(models_dir / "BT4_exported.pb.gz")
    if not model_path.is_file():
        raise ValueError(f"Raw BT4 asset does not exist: {model_path}")
    descriptor = {
        "kind": "torch_research",
        "action_codec_id": ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        "plane_history_mode": PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "research_update": int(manifest["optimizer_update"]),
        "optimizer_resume_supported": bool(
            manifest.get("optimizer_resume_supported", False)
        ),
        "source_mapping_sha256": manifest.get("source_mapping_sha256"),
        "state": {
            "path": str(state_path),
            "size_bytes": int(state["size_bytes"]),
            "sha256": str(state["sha256"]),
            "leaf_count": int(state["leaf_count"]),
        },
        "lineage": run_config.get("source"),
        "torch_run_config": {
            "path": str(run_config_path),
            "sha256": sha256_file(run_config_path),
            "git_commit": run_config.get("git_commit"),
            "framework": run_config.get("framework"),
            "execution": run_config.get("execution"),
            "torch_compile": run_config.get("torch_compile"),
        },
        "model_config": expected_config,
        "jax_materialization": (
            "exact_shape_dtype_value_checked_from_model_safetensors"
        ),
        "bt4_constructor": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
    }
    return TorchResearchCheckpointDescriptor(
        checkpoint_dir=checkpoint_dir,
        manifest=manifest,
        run_config=run_config,
        descriptor=descriptor,
    )


def torch_hero_checkpoint_descriptor(
    path: Path,
    *,
    models_dir: Path,
) -> TorchResearchCheckpointDescriptor:
    """Describe a canonical-codec checkpoint from the clean hero recipe."""

    candidate = require_within_workspace(path)
    if (candidate / "checkpoint" / "manifest.json").is_file():
        checkpoint_dir = require_within_workspace(candidate / "checkpoint")
    elif (candidate / "manifest.json").is_file():
        checkpoint_dir = candidate
    else:
        raise FileNotFoundError(
            f"No Torch hero checkpoint manifest under {candidate}"
        )

    manifest_path = require_within_workspace(checkpoint_dir / "manifest.json")
    manifest = _load_json_object(
        manifest_path,
        label="Torch hero checkpoint manifest",
    )
    checkpoint_format = manifest.get("format")
    if checkpoint_format == "chess-dfm-torch-model-v1":
        if manifest.get("model_only") is not True:
            raise ValueError("Torch hero model checkpoint must be model-only.")
        expected_state_name = "model.safetensors"
        model_leaf_count_key = "leaf_count"
        checkpoint_storage_kind = "model_only"
        run_root = require_within_workspace(checkpoint_dir.parent)
    elif checkpoint_format == "chess-dfm-torch-training-v1":
        if (
            manifest.get("model_only") is not False
            or manifest.get("optimizer_resume_supported") is not True
        ):
            raise ValueError(
                "Torch hero recovery checkpoint has an invalid resume contract."
            )
        if checkpoint_dir.parent.name != "checkpoints":
            raise ValueError(
                "Torch hero recovery checkpoint must be under checkpoints/."
            )
        expected_state_name = "state.safetensors"
        model_leaf_count_key = "model_leaf_count"
        checkpoint_storage_kind = "training_recovery"
        run_root = require_within_workspace(checkpoint_dir.parent.parent)
    else:
        raise ValueError("Unsupported Torch hero checkpoint format.")
    state = manifest.get("state")
    if not isinstance(state, dict) or state.get("path") != expected_state_name:
        raise ValueError("Torch hero checkpoint has an invalid state record.")
    state_path = require_within_workspace(checkpoint_dir / expected_state_name)
    if (
        not state_path.is_file()
        or state_path.stat().st_size != int(state.get("size_bytes", -1))
    ):
        raise ValueError("Torch hero checkpoint state size mismatch.")
    if sha256_file(state_path) != str(state.get("sha256", "")):
        raise ValueError("Torch hero checkpoint state checksum mismatch.")

    run_config_path = require_within_workspace(run_root / "run_config.json")
    run_config = _load_json_object(
        run_config_path,
        label="Torch hero run config",
    )
    recorded_config = run_config.get("config")
    if not isinstance(recorded_config, dict):
        raise ValueError("Torch hero run has no recorded model config.")
    expected_config = _validated_torch_hero_model_config(recorded_config)
    recorded_sigreg_count = recorded_config.get("sigreg_example_count")
    if (
        type(recorded_sigreg_count) is not int
        or recorded_sigreg_count < 1
    ):
        raise ValueError(
            "Torch hero run has an invalid SIGReg example count."
        )
    recorded_wdl_coeff = recorded_config.get("wdl_coeff")
    if (
        isinstance(recorded_wdl_coeff, bool)
        or not isinstance(recorded_wdl_coeff, (int, float))
        or not math.isfinite(float(recorded_wdl_coeff))
        or float(recorded_wdl_coeff) < 0.0
    ):
        raise ValueError(
            "Torch hero run has an invalid WDL coefficient."
        )
    recorded_feedback_mode = recorded_config.get(
        "jepa_feedback_mode",
        "none",
    )
    if recorded_feedback_mode not in {
        "none",
        "final_pass_adjoint",
    }:
        raise ValueError(
            "Torch hero run has an invalid JEPA feedback mode."
        )
    expected_compile_regions = [
        "state_projector_blocks",
        "dfm_blocks",
        "jepa_transition",
    ]
    if (
        run_config.get("framework") != "torch"
        or run_config.get("execution") != "regional-compile"
        or run_config.get("torch_compile") is not True
        or run_config.get("recipe") != "hero"
        or run_config.get("compile_regions") != expected_compile_regions
    ):
        raise ValueError("Torch hero run execution contract mismatch.")
    if checkpoint_storage_kind == "training_recovery":
        resume_contract = manifest.get("resume_contract")
        if not isinstance(resume_contract, dict):
            raise ValueError(
                "Torch hero recovery checkpoint has no resume contract."
            )
        if _json_sha256(resume_contract) != manifest.get(
            "resume_contract_sha256"
        ):
            raise ValueError(
                "Torch hero recovery resume-contract digest mismatch."
            )
        runtime = resume_contract.get("runtime")
        if (
            resume_contract.get("framework") != "torch"
            or resume_contract.get("recipe") != "hero"
            or resume_contract.get("git_commit") != run_config.get("git_commit")
            or resume_contract.get("config") != expected_config
            or not isinstance(runtime, dict)
            or runtime.get("compiled_regions") != expected_compile_regions
        ):
            raise ValueError(
                "Torch hero recovery checkpoint training contract mismatch."
            )

    model_path = require_within_workspace(models_dir / "BT4_exported.pb.gz")
    if not model_path.is_file():
        raise ValueError(f"Raw BT4 asset does not exist: {model_path}")
    descriptor = {
        "kind": "torch_hero",
        "action_codec_id": ACTION_CODEC_LC0_CANONICAL_1858,
        "plane_history_mode": PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "research_update": int(manifest["optimizer_update"]),
        "optimizer_resume_supported": bool(
            manifest.get("optimizer_resume_supported", False)
        ),
        "checkpoint_storage_kind": checkpoint_storage_kind,
        "source_mapping_sha256": manifest.get("source_mapping_sha256"),
        "state": {
            "path": str(state_path),
            "size_bytes": int(state["size_bytes"]),
            "sha256": str(state["sha256"]),
            "leaf_count": int(state[model_leaf_count_key]),
        },
        "lineage": {
            "recipe": "hero",
            "training_git_commit": run_config.get("git_commit"),
        },
        "torch_run_config": {
            "path": str(run_config_path),
            "sha256": sha256_file(run_config_path),
            "git_commit": run_config.get("git_commit"),
            "framework": run_config.get("framework"),
            "execution": run_config.get("execution"),
            "torch_compile": run_config.get("torch_compile"),
            "compile_regions": expected_compile_regions,
        },
        "model_config": expected_config,
        "arena_materialization": "native_torch_exact_safetensors",
        "bt4_constructor": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
    }
    return TorchResearchCheckpointDescriptor(
        checkpoint_dir=checkpoint_dir,
        manifest=manifest,
        run_config=run_config,
        descriptor=descriptor,
    )


def source_checkpoint_descriptor(
    *,
    source_run_root: Path,
    models_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    from research.train import resolve_config

    run_root = require_within_workspace(source_run_root)
    config, metadata = resolve_config(run_root)
    asset = load_asset_manifest()["checkpoint_step_265000"]
    step = int(metadata.get("latest_step", -1))
    if step != 265_000:
        raise ValueError(f"Source checkpoint must be step 265000, found {step}.")
    state_path = require_within_workspace(
        run_root / "checkpoints" / f"step{step:07d}" / "state.npz"
    )
    expected_size = int(asset["state_npz_size_bytes"])
    if state_path.stat().st_size != expected_size:
        raise ValueError("Source checkpoint size mismatch.")
    model_path = require_within_workspace(models_dir / "BT4_exported.pb.gz")
    descriptor = {
        "kind": "source",
        "action_codec_id": ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        "plane_history_mode": PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
        "run_root": str(run_root),
        "checkpoint_step": step,
        "state": {
            "path": str(state_path),
            "size_bytes": expected_size,
            "sha256": str(asset["state_npz_sha256"]),
        },
        "checkpoint_metadata_sha256": sha256_file(run_root / "checkpoint_state.json"),
        "model_config": dataclasses.asdict(config),
        "bt4_constructor": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
    }
    return config, descriptor


def raw_bt4_descriptor(
    *,
    models_dir: Path,
) -> dict[str, Any]:
    """Describe the immutable original BT4 policy checkpoint and semantics."""

    model_path = require_within_workspace(models_dir / "BT4_exported.pb.gz")
    if not model_path.is_file():
        raise ValueError(f"Raw BT4 asset does not exist: {model_path}")
    return {
        "kind": "raw_bt4",
        "action_codec_id": ACTION_CODEC_LC0_CANONICAL_1858,
        "plane_history_mode": PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
        "compute_dtype": RAW_BT4_POLICY_COMPUTE_DTYPE,
        "search": "none_deterministic_greedy_policy_head",
        "bt4_checkpoint": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
    }


def _torch_hero_model_id(
    *,
    role: str,
    descriptor: Mapping[str, Any],
) -> str:
    prefixes = {
        "candidate": "candidate-torch-hero",
        "opponent": "incumbent-torch-hero",
    }
    try:
        prefix = prefixes[role]
        state_sha256 = str(descriptor["state"]["sha256"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Torch hero descriptor has no state SHA-256.") from exc
    if len(state_sha256) != 64:
        raise ValueError("Torch hero descriptor state SHA-256 is invalid.")
    return f"{prefix}-{state_sha256[:12]}"


def _dtype(name: str) -> Any:
    values = {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported encoder dtype: {name!r}") from exc


def _create_model_only(
    bt4_params: dict[str, Any],
    config: Any,
    *,
    seed: int,
) -> Any:
    from research.train import JointLatentSASAModel

    encoder = make_bt4_model(
        bt4_params,
        dtype=_dtype(config.encoder_dtype),
        train_encoder=config.unfreeze_bt4_encoder,
    )
    return JointLatentSASAModel(
        encoder,
        config,
        rngs=nnx.Rngs(seed),
    )


def load_research_policy(
    checkpoint: ResearchCheckpointDescriptor,
    *,
    bt4_params: dict[str, Any],
    model_id: str,
    seed: int,
    refinement_passes: int,
    collect_diagnostics: bool,
    inference_batch_size: int,
) -> tuple[LocalDFMPolicy, dict[str, Any]]:
    from research.train import (
        EmaTargetModel,
        load_research_checkpoint_for_evaluation,
    )

    started = time.perf_counter()
    model = _create_model_only(
        bt4_params,
        checkpoint.config,
        seed=seed,
    )
    ema_target = EmaTargetModel(model) if "ema_target_abi" in checkpoint.manifest else None
    restored = load_research_checkpoint_for_evaluation(
        checkpoint.checkpoint_dir,
        model=model,
        ema_target=ema_target,
    )
    # The EMA teacher is a training-only target and is intentionally not
    # retained by the DFM inference policy.
    del ema_target
    elapsed = time.perf_counter() - started
    return (
        LocalDFMPolicy(
            model=model,
            model_id=model_id,
            refinement_passes=refinement_passes,
            trace_top_k=1,
            collect_diagnostics=collect_diagnostics,
            inference_batch_size=inference_batch_size,
            history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        ),
        {
            "load_seconds": elapsed,
            "checkpoint_dir": restored["checkpoint_dir"],
        },
    )


def load_torch_research_policy(
    checkpoint: TorchResearchCheckpointDescriptor,
    *,
    bt4_params: dict[str, Any],
    model_id: str,
    seed: int,
    refinement_passes: int,
    collect_diagnostics: bool,
    inference_batch_size: int,
) -> tuple[LocalDFMPolicy, dict[str, Any]]:
    """Materialize a strict Torch model-only state into the frozen JAX arena."""

    from research.evaluate_torch_migration_parity import (
        _apply_and_verify_jax_checkpoint,
        _checkpoint_tree,
        _jax_config,
    )
    from research.train import JointLatentSASAModel

    started = time.perf_counter()
    tree, manifest, tree_summary, ordered_names = _checkpoint_tree(
        checkpoint.checkpoint_dir
    )
    if manifest != checkpoint.manifest:
        raise ValueError("Eager-Torch checkpoint manifest changed during load.")
    config = _jax_config("bfloat16")
    encoder = make_bt4_model(
        bt4_params,
        dtype=jnp.bfloat16,
        attention_impl="manual",
        train_encoder=True,
    )
    model = JointLatentSASAModel(
        encoder,
        config,
        rngs=nnx.Rngs(seed),
    )
    roundtrip = _apply_and_verify_jax_checkpoint(
        model,
        tree,
        ordered_names=ordered_names,
        expected_summary=tree_summary,
    )
    del tree
    gc.collect()
    elapsed = time.perf_counter() - started
    return (
        LocalDFMPolicy(
            model=model,
            model_id=model_id,
            refinement_passes=refinement_passes,
            trace_top_k=1,
            collect_diagnostics=collect_diagnostics,
            inference_batch_size=inference_batch_size,
            history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        ),
        {
            "load_seconds": elapsed,
            "checkpoint_dir": str(checkpoint.checkpoint_dir),
            "checkpoint_state_sha256": manifest["state"]["sha256"],
            "jax_roundtrip": roundtrip,
        },
    )


def load_source_policy(
    *,
    config: Any,
    descriptor: Mapping[str, Any],
    bt4_params: dict[str, Any],
    model_id: str,
    seed: int,
    refinement_passes: int,
    collect_diagnostics: bool,
    inference_batch_size: int,
) -> tuple[LocalDFMPolicy, dict[str, Any]]:
    from research.train import create_joint_components

    started = time.perf_counter()
    model, optimizer = create_joint_components(
        bt4_params,
        config,
        seed=seed,
    )
    state = descriptor["state"]
    imported = import_legacy_checkpoint(
        state["path"],
        model=model,
        optimizer=optimizer,
        expected_size_bytes=int(state["size_bytes"]),
        expected_sha256=str(state["sha256"]),
        expected_step=265_000,
        init_mode="model-only",
    )
    del optimizer
    elapsed = time.perf_counter() - started
    return (
        LocalDFMPolicy(
            model=model,
            model_id=model_id,
            refinement_passes=refinement_passes,
            trace_top_k=1,
            collect_diagnostics=collect_diagnostics,
            inference_batch_size=inference_batch_size,
            history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        ),
        {
            "load_seconds": elapsed,
            "source_step": imported.source_step,
            "source_sha256": imported.source_sha256,
            "optimizer_restored": imported.optimizer_restored,
        },
    )


def _validate_torch_hero_refinement_passes(
    config: Any,
    refinement_passes: int,
) -> None:
    if refinement_passes < 1:
        raise ValueError("Torch hero refinement passes must be positive")
    feedback_mode = getattr(config, "jepa_feedback_mode", "none")
    if feedback_mode not in {"none", "final_pass_adjoint"}:
        raise ValueError(
            f"Unsupported Torch hero JEPA feedback mode: {feedback_mode!r}"
        )
    if feedback_mode == "final_pass_adjoint" and (
        refinement_passes != 8 or config.horizon != 8
    ):
        raise ValueError(
            "Torch hero final-pass JEPA feedback requires exactly eight "
            "refinement passes and horizon eight"
        )


def load_torch_hero_policy(
    checkpoint: TorchResearchCheckpointDescriptor,
    *,
    model_id: str,
    policy_mode: str,
    refinement_passes: int,
    inference_batch_size: int,
) -> tuple[BatchedArenaPolicy, dict[str, Any]]:
    """Load the canonical hero checkpoint as its native Torch policy."""

    import torch

    from research.train_torch import (
        Config,
        TorchHeroArenaPolicy,
        _apply_compile_regions,
        load_checkpoint_model_for_evaluation,
        load_raw_bt4_hero_model,
    )

    if policy_mode not in {"dfm", "policy_only"}:
        raise ValueError(
            "Torch hero checkpoint policy mode must be 'dfm' or 'policy_only'."
        )
    config = Config(**checkpoint.descriptor["model_config"])
    if policy_mode == "dfm":
        _validate_torch_hero_refinement_passes(config, refinement_passes)
    started = time.perf_counter()
    device = torch.device("cuda")
    model_path = require_within_workspace(
        checkpoint.descriptor["bt4_constructor"]["path"]
    )
    model, initialization = load_raw_bt4_hero_model(
        device=device,
        raw_bt4_path=model_path,
        config=config,
    )
    restored = load_checkpoint_model_for_evaluation(
        checkpoint_dir=checkpoint.checkpoint_dir,
        model=model,
    )
    if restored != checkpoint.manifest:
        raise ValueError("Torch hero checkpoint manifest changed during load.")
    if (
        restored.get("source_mapping_sha256")
        != initialization["combined_sha256"]
    ):
        raise ValueError("Torch hero checkpoint raw-BT4 mapping drift.")
    compiled_regions = _apply_compile_regions(model, "fresh")
    expected_regions = checkpoint.run_config["compile_regions"]
    if compiled_regions != expected_regions:
        raise ValueError("Torch hero compile-region contract drift.")
    model.eval()
    elapsed = time.perf_counter() - started
    return (
        TorchHeroArenaPolicy(
            model=model,
            model_id=model_id,
            policy_mode=policy_mode,
            inference_batch_size=inference_batch_size,
            refinement_passes=refinement_passes,
        ),
        {
            "load_seconds": elapsed,
            "framework": "torch",
            "device": torch.cuda.get_device_name(device),
            "checkpoint_dir": str(checkpoint.checkpoint_dir),
            "checkpoint_state_sha256": restored["state"]["sha256"],
            "raw_initialization_sha256": initialization["combined_sha256"],
            "compile_regions": compiled_regions,
            "policy_mode": policy_mode,
            "policy_head": (
                "hero_checkpoint_bt4_policy_only"
                if policy_mode == "policy_only"
                else "hero_checkpoint_bt4_plus_dfm"
            ),
        },
    )


def load_torch_raw_bt4_hero_policy(
    *,
    model_path: Path,
    model_config: Mapping[str, Any],
    model_id: str,
    inference_batch_size: int,
) -> tuple[BatchedArenaPolicy, dict[str, Any]]:
    """Load raw BT4 through the same native Torch encoder and canonical codec."""

    import torch

    from research.train_torch import (
        Config,
        TorchHeroArenaPolicy,
        load_raw_bt4_hero_model,
    )

    started = time.perf_counter()
    device = torch.device("cuda")
    config = Config(**dict(model_config))
    model, initialization = load_raw_bt4_hero_model(
        device=device,
        raw_bt4_path=require_within_workspace(model_path),
        config=config,
    )
    model.eval()
    elapsed = time.perf_counter() - started
    return (
        TorchHeroArenaPolicy(
            model=model,
            model_id=model_id,
            policy_mode="raw_bt4",
            inference_batch_size=inference_batch_size,
            refinement_passes=config.horizon,
        ),
        {
            "load_seconds": elapsed,
            "framework": "torch",
            "device": torch.cuda.get_device_name(device),
            "policy_head": "original_bt4",
            "raw_initialization_sha256": initialization["combined_sha256"],
        },
    )


def load_native_torch_hero_pair(
    *,
    candidate_checkpoint: TorchResearchCheckpointDescriptor,
    opponent_checkpoint: TorchResearchCheckpointDescriptor | None,
    candidate_record: Mapping[str, Any],
    opponent_record: Mapping[str, Any],
    candidate_id: str,
    opponent_id: str,
    candidate_policy_mode: str,
    opponent_policy_mode: str,
    candidate_refinement_passes: int,
    opponent_refinement_passes: int,
    inference_batch_size: int,
) -> tuple[
    BatchedArenaPolicy,
    dict[str, Any],
    BatchedArenaPolicy,
    dict[str, Any],
    dict[str, Any],
]:
    """Load a candidate and either a hero incumbent or raw BT4 in Torch."""

    candidate_policy, candidate_load = load_torch_hero_policy(
        candidate_checkpoint,
        model_id=candidate_id,
        policy_mode=candidate_policy_mode,
        refinement_passes=candidate_refinement_passes,
        inference_batch_size=inference_batch_size,
    )
    if opponent_checkpoint is None:
        if opponent_record.get("kind") != "raw_bt4":
            raise ValueError("Native Torch opponent descriptor is not raw BT4.")
        opponent_policy, opponent_load = load_torch_raw_bt4_hero_policy(
            model_path=Path(
                opponent_record["bt4_checkpoint"]["path"]
            ),
            model_config=candidate_record["model_config"],
            model_id=opponent_id,
            inference_batch_size=inference_batch_size,
        )
    else:
        if opponent_record.get("kind") != "torch_hero":
            raise ValueError(
                "Native Torch incumbent descriptor is not a Torch hero."
            )
        opponent_policy, opponent_load = load_torch_hero_policy(
            opponent_checkpoint,
            model_id=opponent_id,
            policy_mode=opponent_policy_mode,
            refinement_passes=opponent_refinement_passes,
            inference_batch_size=inference_batch_size,
        )
    candidate_device = candidate_load.get("device")
    opponent_device = opponent_load.get("device")
    if (
        candidate_load.get("framework") != "torch"
        or opponent_load.get("framework") != "torch"
        or not isinstance(candidate_device, str)
        or opponent_device != candidate_device
    ):
        raise ValueError(
            "Native Torch Arena policies did not load on one shared device."
        )
    accelerator_session = {
        "framework": "torch",
        "torch_devices": [candidate_device],
    }
    return (
        candidate_policy,
        candidate_load,
        opponent_policy,
        opponent_load,
        accelerator_session,
    )


def load_raw_bt4_policy(
    *,
    bt4_params: dict[str, Any],
    model_id: str,
    inference_batch_size: int,
) -> tuple[LocalBT4Policy, dict[str, Any]]:
    """Construct the immutable BF16 raw-BT4 policy-head opponent."""

    started = time.perf_counter()
    model = make_bt4_model(
        bt4_params,
        dtype=jnp.bfloat16,
        train_encoder=False,
    )
    elapsed = time.perf_counter() - started
    return (
        LocalBT4Policy(
            model=model,
            model_id=model_id,
            inference_batch_size=inference_batch_size,
            history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        ),
        {
            "load_seconds": elapsed,
            "compute_dtype": RAW_BT4_POLICY_COMPUTE_DTYPE,
            "policy_head": "original_bt4",
            "checkpoint_restore": "constructor_asset_weights",
        },
    )


@dataclasses.dataclass
class PolicyTracker:
    positions: int = 0
    legal_moves: int = 0
    representable_legal_actions: int = 0
    incomplete_coverage_positions: int = 0
    calls: int = 0
    call_seconds: float = 0.0
    max_call_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "mean_call_seconds": (self.call_seconds / self.calls if self.calls else None),
            "representable_fraction": (
                self.representable_legal_actions / self.legal_moves if self.legal_moves else None
            ),
        }


@dataclasses.dataclass(frozen=True)
class TrackingPolicy:
    delegate: BatchedArenaPolicy
    tracker: PolicyTracker

    @property
    def model_id(self) -> str:
        return str(self.delegate.model_id)

    @property
    def action_codec_id(self) -> str:
        return str(self.delegate.action_codec_id)

    @property
    def history_validation_mode(self) -> str:
        return str(
            getattr(
                self.delegate,
                "history_validation_mode",
                HISTORY_VALIDATION_FULL_REPLAY,
            )
        )

    @property
    def trusted_arena_history_schema(self) -> str | None:
        value = getattr(self.delegate, "trusted_arena_history_schema", None)
        return None if value is None else str(value)

    def _select_with_tracking(
        self,
        boards: Sequence[chess.Board],
        invoke: Any,
    ) -> Any:
        for board in boards:
            legal_count = board.legal_moves.count()
            mask = legal_action_mask(
                board,
                codec_id=self.action_codec_id,
            )
            representable_count = int(np.count_nonzero(mask))
            self.tracker.positions += 1
            self.tracker.legal_moves += legal_count
            self.tracker.representable_legal_actions += representable_count
            self.tracker.incomplete_coverage_positions += int(representable_count != legal_count)
        started = time.perf_counter()
        try:
            return invoke()
        finally:
            elapsed = time.perf_counter() - started
            self.tracker.calls += 1
            self.tracker.call_seconds += elapsed
            self.tracker.max_call_seconds = max(
                self.tracker.max_call_seconds,
                elapsed,
            )

    def select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Sequence[chess.Board]],
    ) -> Any:
        return self._select_with_tracking(
            boards,
            lambda: self.delegate.select_actions(boards, histories),
        )

    def select_actions_from_trusted_arena(
        self,
        boards: Sequence[chess.Board],
        endpoints: Sequence[Any],
    ) -> Any:
        capability = getattr(
            self.delegate,
            "select_actions_from_trusted_arena",
            None,
        )
        if not callable(capability):
            raise TypeError("Tracked policy lacks trusted arena history capability.")
        return self._select_with_tracking(
            boards,
            lambda: capability(boards, endpoints),
        )


def _warm_policy(
    policy: BatchedArenaPolicy,
    *,
    opening_pool: Mapping[str, Any],
    loaded_histories: LoadedOpeningHistories,
    batch_size: int,
) -> float:
    if batch_size < 1:
        raise ValueError("Warmup batch size must be positive.")
    boards: list[chess.Board] = []
    histories: list[tuple[chess.Board, ...]] = []
    for index in range(batch_size):
        fens = loaded_histories.histories_by_opening_index[index]
        history = tuple(chess.Board(fen) for fen in fens)
        if history[0].fen(en_passant="legal") != chess.Board().fen(en_passant="legal"):
            raise ValueError("Warmup history does not begin at standard chess.")
        if history[-1].fen(en_passant="legal") != opening_pool["openings"][index]["fen"]:
            raise ValueError("Warmup history does not match opening pool.")
        boards.append(history[-1])
        histories.append(history)
    started = time.perf_counter()
    result = policy.select_actions(boards, histories)
    actions = np.asarray(result.action_indices)
    if actions.shape != (batch_size,):
        raise ValueError("Warmup policy returned the wrong action shape.")
    return time.perf_counter() - started


def _static_inference_batch_size(
    *,
    block_pairs: int,
    policy_batch_size_cap: int,
) -> int:
    """Choose one physical shape that bounds every per-model block batch."""

    if block_pairs < 1 or policy_batch_size_cap < 1:
        raise ValueError("Static inference batching requires positive bounds.")
    return min(block_pairs, policy_batch_size_cap)


def _validate_static_inference_contract(
    contract: Mapping[str, Any],
    *,
    policies: Sequence[BatchedArenaPolicy],
) -> int:
    try:
        run = contract["run"]
        batching = contract["inference_batching"]
        physical_batch_size = int(batching["physical_batch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Arena contract has no valid static inference batching.") from exc
    expected_size = _static_inference_batch_size(
        block_pairs=int(run["block_pairs"]),
        policy_batch_size_cap=int(run["policy_batch_size_cap"]),
    )
    expected = {
        "schema_version": STATIC_INFERENCE_BATCHING_SCHEMA,
        "physical_batch_size": expected_size,
        "padding_mode": STATIC_INFERENCE_PADDING_MODE,
        "padding_source": "first_validated_active_encoded_row",
        "active_rows": "ordered_prefix",
        "output_handling": "slice_to_active_rows_before_semantic_validation",
        "host_validation_scope": "active_rows_only",
        "metrics_basis": "real_rows_only",
        "timing_basis": "physical_padded_call_wall_time",
        "runtime_rng": "none_deterministic_greedy_inference",
    }
    if dict(batching) != expected:
        raise ValueError("Arena static inference batching contract mismatch.")
    if physical_batch_size != expected_size:
        raise ValueError("Arena physical inference batch size is not run-derived.")
    for policy in policies:
        if getattr(policy, "inference_batching_schema", None) != (STATIC_INFERENCE_BATCHING_SCHEMA):
            raise ValueError(
                f"Policy {policy.model_id!r} has no compatible static batching schema."
            )
        if getattr(policy, "inference_padding_mode", None) != (STATIC_INFERENCE_PADDING_MODE):
            raise ValueError(f"Policy {policy.model_id!r} has no compatible static padding mode.")
        if getattr(policy, "inference_batch_size", None) != physical_batch_size:
            raise ValueError(
                f"Policy {policy.model_id!r} does not use frozen physical batch "
                f"size {physical_batch_size}."
            )
    return physical_batch_size


def _validate_history_validation_contract(
    contract: Mapping[str, Any],
    *,
    policies: Sequence[BatchedArenaPolicy],
) -> None:
    expected = {
        "schema_version": HISTORY_VALIDATION_SCHEMA,
        "public_policy_default": HISTORY_VALIDATION_FULL_REPLAY,
        "arena_hot_path": HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
        "opening_validation": "full_exact_replay_sidecar_and_selected_pairs",
        "authoritative_state": "full_stack_with_checked_legal_pushes",
        "policy_board_copy": "stack_false",
        "policy_payload": "sealed_constant_size_state_attestation",
        "repetition_authority": "runner_claim_draw_boundary_check",
        "complexity": "constant_per_policy_row",
    }
    try:
        observed = dict(contract["history_validation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Arena contract has no valid history validation mode.") from exc
    if observed != expected:
        raise ValueError("Arena history validation contract mismatch.")
    for policy in policies:
        if getattr(policy, "history_validation_mode", None) != (
            HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT
        ):
            raise ValueError(
                f"Policy {policy.model_id!r} does not use the pinned trusted "
                "arena endpoint history mode."
            )
        if getattr(policy, "trusted_arena_history_schema", None) != (
            HISTORY_VALIDATION_SCHEMA
        ) or not callable(getattr(policy, "select_actions_from_trusted_arena", None)):
            raise ValueError(
                f"Policy {policy.model_id!r} lacks the pinned trusted arena endpoint capability."
            )


def _gameplay_payload_digest(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    observed = value.pop("payload_sha256", None)
    expected = _json_sha256(value)
    if observed != expected:
        raise ValueError("Arena gameplay payload digest mismatch.")
    return expected


def _outcomes_from_gameplay(
    gameplay: Mapping[str, Any],
) -> list[Any]:
    from research.arena import GameOutcome

    outcomes = []
    for game in gameplay["games"]:
        outcomes.append(
            GameOutcome(
                pair_id=game["pair_id"],
                opening_index=int(game["opening_index"]),
                game_in_pair=int(game["game_in_pair"]),
                fen=game["fen"],
                white_model=game["white_model"],
                black_model=game["black_model"],
                result=game["result"],
                termination=game["termination"],
                ply_count=int(game["ply_count"]),
                winner_model=game["winner_model"],
                loser_model=game["loser_model"],
            )
        )
    return outcomes


def _pair_scores_from_gameplay(
    gameplay: Mapping[str, Any],
    *,
    candidate_model_id: str,
) -> list[float]:
    outcomes = _outcomes_from_gameplay(gameplay)
    if len(outcomes) % 2:
        raise ValueError("Arena block contains an incomplete pair.")
    return [
        pair_score_for_model(
            outcomes[offset : offset + 2],
            model_id=candidate_model_id,
        )
        for offset in range(0, len(outcomes), 2)
    ]


def _validate_real_row_tracking(
    gameplay_result: ArenaGameplayResult,
    *,
    trackers: Mapping[str, PolicyTracker],
) -> None:
    by_model = {stats.model_id: stats for stats in gameplay_result.model_stats}
    if set(by_model) != set(trackers):
        raise RuntimeError("Arena policy tracking model IDs disagree with gameplay.")
    for model_id, tracker in trackers.items():
        stats = by_model[model_id]
        if tracker.positions != stats.positions_evaluated:
            raise RuntimeError(f"Policy tracker for {model_id!r} counted non-gameplay positions.")
        if tracker.calls != stats.policy_calls:
            raise RuntimeError(f"Policy tracker for {model_id!r} disagrees on call count.")


def _empty_aggregate(model_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "pair_scores": [],
        "gameplay_wall_seconds": 0.0,
        "termination_counts": {},
        "fault_counts": {},
        "models": {
            model_id: {
                "games": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "points": 0.0,
                "fault_losses": 0,
                "policy_calls": 0,
                "positions_evaluated": 0,
                "coverage": dataclasses.asdict(PolicyTracker()),
            }
            for model_id in model_ids
        },
    }


def _add_counts(
    destination: dict[str, int],
    values: Mapping[str, Any],
) -> None:
    for key, value in values.items():
        destination[str(key)] = destination.get(str(key), 0) + int(value)


def _merge_block(
    aggregate: dict[str, Any],
    block: Mapping[str, Any],
    *,
    candidate_model_id: str,
) -> None:
    gameplay = block["gameplay"]
    _gameplay_payload_digest(gameplay)
    pair_scores = _pair_scores_from_gameplay(
        gameplay,
        candidate_model_id=candidate_model_id,
    )
    if pair_scores != block["pair_scores"]:
        raise ValueError("Arena block pair-score digest boundary mismatch.")
    aggregate["pair_scores"].extend(pair_scores)
    aggregate["gameplay_wall_seconds"] += float(block["timing"]["gameplay_wall_seconds"])
    stats = gameplay["stats"]
    _add_counts(aggregate["termination_counts"], stats["termination_counts"])
    _add_counts(aggregate["fault_counts"], stats["fault_counts"])
    tracking = block["policy_tracking"]
    for model in stats["models"]:
        model_id = model["model_id"]
        target = aggregate["models"][model_id]
        for key in (
            "games",
            "wins",
            "draws",
            "losses",
            "fault_losses",
            "policy_calls",
            "positions_evaluated",
        ):
            target[key] += int(model[key])
        target["points"] += float(model["points"])
        coverage = target["coverage"]
        observed = tracking[model_id]
        for key in (
            "positions",
            "legal_moves",
            "representable_legal_actions",
            "incomplete_coverage_positions",
            "calls",
        ):
            coverage[key] += int(observed[key])
        for key in ("call_seconds",):
            coverage[key] += float(observed[key])
        coverage["max_call_seconds"] = max(
            float(coverage["max_call_seconds"]),
            float(observed["max_call_seconds"]),
        )


def _finalize_aggregate(
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    result = json.loads(json.dumps(aggregate))
    pair_scores = result["pair_scores"]
    pair_count = len(pair_scores)
    result["pair_count"] = pair_count
    result["game_count"] = 2 * pair_count
    result["cap_draw_count"] = int(result["termination_counts"].get("ply_cap", 0))
    result["cap_draw_rate"] = (
        result["cap_draw_count"] / result["game_count"] if result["game_count"] else None
    )
    for model in result["models"].values():
        coverage = model["coverage"]
        coverage["mean_call_seconds"] = (
            coverage["call_seconds"] / coverage["calls"] if coverage["calls"] else None
        )
        coverage["representable_fraction"] = (
            coverage["representable_legal_actions"] / coverage["legal_moves"]
            if coverage["legal_moves"]
            else None
        )
    if pair_count:
        stats = pentanomial_stats(pair_scores)
        result["pentanomial"] = stats.as_dict()
        result["pair_aware_logistic_interval"] = dataclasses.asdict(
            pair_aware_score_elo_interval(stats)
        )
        result["normalized_elo_diagnostics"] = pentanomial_normalized_elo_diagnostics(
            stats.counts
        ).as_dict()
    else:
        result["pentanomial"] = None
        result["pair_aware_logistic_interval"] = None
        result["normalized_elo_diagnostics"] = None
    return result


def _read_block(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    source = require_within_workspace(path)
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"Arena block file checksum mismatch: {source}")
    block = _load_json_object(source, label="arena block")
    if block.get("schema_version") != ARENA_BLOCK_SCHEMA:
        raise ValueError("Unsupported arena block schema.")
    _validate_payload_sha256(block, label="Arena block")
    return block


def rebuild_aggregate(
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], GSPRTState | None]:
    contract = state["contract"]
    model_ids = (
        contract["models"]["candidate"]["model_id"],
        contract["models"]["opponent"]["model_id"],
    )
    aggregate = _empty_aggregate(model_ids)
    expected_start = 0
    gsprt = (
        GSPRTState(GSPRTConfig.normalized_promotion())
        if contract["tier"]["promotion_eligible"]
        else None
    )
    for reference in state["blocks"]:
        if int(reference["start_pair"]) != expected_start:
            raise ValueError("Arena blocks are not contiguous.")
        block = _read_block(
            Path(reference["path"]),
            expected_sha256=reference["sha256"],
        )
        if block["contract_sha256"] != state["contract_sha256"]:
            raise ValueError("Arena block belongs to a different run.")
        if int(block["start_pair"]) != expected_start:
            raise ValueError("Arena block/reference start mismatch.")
        _merge_block(
            aggregate,
            block,
            candidate_model_id=model_ids[0],
        )
        expected_start += int(block["pair_count"])
        if gsprt is not None:
            for pair_score in block["pair_scores"]:
                if gsprt.terminal:
                    raise ValueError(
                        "Persisted promotion arena continued after a terminal pair boundary."
                    )
                gsprt = gsprt.update(float(pair_score))
    return aggregate, gsprt


def _state_payload(
    *,
    contract: Mapping[str, Any],
    created_at_utc: str,
    blocks: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    gsprt: GSPRTState | None,
    sessions: Sequence[Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    finalized = _finalize_aggregate(aggregate)
    promotion = gsprt.as_dict() if gsprt is not None else None
    return _with_payload_sha256(
        {
            "schema_version": ARENA_RUN_SCHEMA,
            "relative_elo_scope": RELATIVE_ELO_SCOPE,
            "absolute_elo_claim": False,
            "created_at_utc": created_at_utc,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "contract": contract,
            "contract_sha256": _json_sha256(contract),
            "blocks": list(blocks),
            "aggregate": finalized,
            "normalized_promotion_state": promotion,
            "candidate_promoted": bool(
                promotion is not None and promotion["decision"] == "accept_h1"
            ),
            "sessions": list(sessions),
        }
    )


def load_run_state(
    output_dir: Path,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = require_within_workspace(output_dir / "state.json")
    state = _load_json_object(state_path, label="arena run state")
    if state.get("schema_version") != ARENA_RUN_SCHEMA:
        raise ValueError("Unsupported arena run-state schema.")
    _validate_payload_sha256(state, label="Arena run state")
    expected_contract_sha256 = _json_sha256(expected_contract)
    if (
        state.get("contract_sha256") != expected_contract_sha256
        or _json_sha256(state.get("contract")) != expected_contract_sha256
    ):
        raise ValueError("Arena resume contract mismatch.")
    referenced = {Path(item["path"]).name for item in state["blocks"]}
    blocks_dir = require_within_workspace(output_dir / "blocks")
    observed = (
        {
            path.name
            for path in blocks_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        if blocks_dir.is_dir()
        else set()
    )
    if observed != referenced:
        raise ValueError(
            "Arena block directory differs from state manifest: "
            f"missing={sorted(referenced - observed)}, "
            f"orphaned={sorted(observed - referenced)}"
        )
    aggregate, gsprt = rebuild_aggregate(state)
    if _finalize_aggregate(aggregate) != state["aggregate"]:
        raise ValueError("Arena aggregate does not match immutable blocks.")
    persisted_gsprt = state.get("normalized_promotion_state")
    if (gsprt.as_dict() if gsprt is not None else None) != persisted_gsprt:
        raise ValueError("Arena promotion state does not match pair blocks.")
    return state


def run_blocks(
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
    opening_pool: Mapping[str, Any],
    loaded_histories: LoadedOpeningHistories,
    candidate_policy: BatchedArenaPolicy,
    opponent_policy: BatchedArenaPolicy,
    resume: bool,
    session_setup: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_static_inference_contract(
        contract,
        policies=(candidate_policy, opponent_policy),
    )
    _validate_history_validation_contract(
        contract,
        policies=(candidate_policy, opponent_policy),
    )
    destination = require_within_workspace(output_dir)
    state_path = require_within_workspace(destination / "state.json")
    blocks_dir = require_within_workspace(destination / "blocks")
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"Arena run already exists; pass --resume: {destination}")
        state = load_run_state(
            destination,
            expected_contract=contract,
        )
        if state["status"] != "running":
            return state
        blocks = list(state["blocks"])
        sessions = list(state["sessions"])
        created_at = str(state["created_at_utc"])
        aggregate, gsprt = rebuild_aggregate(state)
    else:
        if resume:
            raise FileNotFoundError(f"No arena run exists to resume: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        blocks_dir.mkdir(parents=True, exist_ok=True)
        blocks = []
        sessions = []
        created_at = datetime.now(UTC).isoformat()
        aggregate = _empty_aggregate([candidate_policy.model_id, opponent_policy.model_id])
        gsprt = (
            GSPRTState(GSPRTConfig.normalized_promotion())
            if contract["tier"]["promotion_eligible"]
            else None
        )

    sessions.append(dict(session_setup))
    running = _state_payload(
        contract=contract,
        created_at_utc=created_at,
        blocks=blocks,
        aggregate=aggregate,
        gsprt=gsprt,
        sessions=sessions,
        status="running",
    )
    atomic_write_json(state_path, running)

    pair_target = int(contract["run"]["pair_count"])
    block_pairs = int(contract["run"]["block_pairs"])
    candidate_id = candidate_policy.model_id
    opponent_id = opponent_policy.model_id
    while len(aggregate["pair_scores"]) < pair_target:
        if gsprt is not None and gsprt.terminal:
            break
        start = len(aggregate["pair_scores"])
        count = min(block_pairs, pair_target - start)
        fens = [opening["fen"] for opening in opening_pool["openings"][start : start + count]]
        pairs = make_color_reversed_pairs(
            fens,
            model_a=candidate_id,
            model_b=opponent_id,
            start_index=start,
        )
        histories = histories_for_pairs(pairs, loaded_histories)
        candidate_tracker = PolicyTracker()
        opponent_tracker = PolicyTracker()
        policies = {
            candidate_id: TrackingPolicy(
                delegate=candidate_policy,
                tracker=candidate_tracker,
            ),
            opponent_id: TrackingPolicy(
                delegate=opponent_policy,
                tracker=opponent_tracker,
            ),
        }
        started = time.perf_counter()
        gameplay_result: ArenaGameplayResult = play_arena_pairs(
            pairs,
            opening_histories=histories,
            policies=policies,
            additional_ply_cap=int(contract["run"]["additional_ply_cap"]),
            policy_timeout_seconds=float(contract["run"]["policy_timeout_seconds"]),
            policy_batch_size_cap=int(contract["run"]["policy_batch_size_cap"]),
        )
        gameplay_seconds = time.perf_counter() - started
        _validate_real_row_tracking(
            gameplay_result,
            trackers={
                candidate_id: candidate_tracker,
                opponent_id: opponent_tracker,
            },
        )
        gameplay = gameplay_result.as_dict()
        pair_scores = _pair_scores_from_gameplay(
            gameplay,
            candidate_model_id=candidate_id,
        )
        block = _with_payload_sha256(
            {
                "schema_version": ARENA_BLOCK_SCHEMA,
                "contract_sha256": _json_sha256(contract),
                "start_pair": start,
                "pair_count": count,
                "pair_scores": pair_scores,
                "timing": {
                    "gameplay_wall_seconds": gameplay_seconds,
                },
                "policy_tracking": {
                    candidate_id: candidate_tracker.as_dict(),
                    opponent_id: opponent_tracker.as_dict(),
                },
                "gameplay": gameplay,
            }
        )
        block_path = require_within_workspace(
            blocks_dir / f"block-{start:08d}-pairs-{count:04d}.json"
        )
        if block_path.exists():
            raise FileExistsError(f"Refusing to overwrite arena block: {block_path}")
        atomic_write_json(block_path, block)
        reference = {
            "path": str(block_path),
            "sha256": sha256_file(block_path),
            "start_pair": start,
            "pair_count": count,
            "payload_sha256": block["payload_sha256"],
        }
        blocks.append(reference)
        _merge_block(
            aggregate,
            block,
            candidate_model_id=candidate_id,
        )
        if gsprt is not None:
            for pair_score in pair_scores:
                if gsprt.terminal:
                    raise RuntimeError(
                        "Promotion block crossed a terminal pair boundary; "
                        "promotion requires --block-pairs 1."
                    )
                gsprt = gsprt.update(pair_score)
        running = _state_payload(
            contract=contract,
            created_at_utc=created_at,
            blocks=blocks,
            aggregate=aggregate,
            gsprt=gsprt,
            sessions=sessions,
            status="running",
        )
        atomic_write_json(state_path, running)

    completed_pairs = len(aggregate["pair_scores"])
    terminal = gsprt is not None and gsprt.terminal
    status = (
        "promotion_terminal"
        if terminal
        else "complete"
        if completed_pairs == pair_target
        else "running"
    )
    final = _state_payload(
        contract=contract,
        created_at_utc=created_at,
        blocks=blocks,
        aggregate=aggregate,
        gsprt=gsprt,
        sessions=sessions,
        status=status,
    )
    atomic_write_json(state_path, final)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument(
        "--candidate",
        type=Path,
        help="Standard JAX research checkpoint.",
    )
    candidate.add_argument(
        "--candidate-torch",
        type=Path,
        help="Strict model-only checkpoint from research/train_torch.py.",
    )
    candidate.add_argument(
        "--candidate-torch-hero",
        type=Path,
        help=(
            "Canonical-codec hero checkpoint from the regional-compiled "
            "research/train_torch.py recipe."
        ),
    )
    candidate.add_argument(
        "--candidate-raw-bt4",
        action="store_true",
        help=(
            "Use a separately identified copy of raw BT4 as the candidate. "
            "This is the update-zero self-play/codec correctness control."
        ),
    )
    opponent = parser.add_mutually_exclusive_group()
    opponent.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="Research incumbent checkpoint; omit to use source step 265000.",
    )
    opponent.add_argument(
        "--opponent-source",
        action="store_true",
        help="Explicitly select source step 265000 (the default).",
    )
    opponent.add_argument(
        "--opponent-raw-bt4",
        action="store_true",
        help="Use the original board-aware canonical BT4 policy head.",
    )
    opponent.add_argument(
        "--opponent-torch-hero",
        type=Path,
        help=(
            "Use an independently loaded canonical-codec Torch hero "
            "checkpoint as the incumbent."
        ),
    )
    parser.add_argument(
        "--tier",
        choices=tuple(FROZEN_TIERS),
        default="correctness",
    )
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--block-pairs", type=int)
    parser.add_argument("--additional-ply-cap", type=int)
    parser.add_argument("--refinement-passes", type=int, default=8)
    parser.add_argument(
        "--candidate-refinement-passes",
        type=int,
        help=(
            "Override --refinement-passes for the candidate only. This is "
            "useful for matched Hero pass-count ablations."
        ),
    )
    parser.add_argument(
        "--opponent-refinement-passes",
        type=int,
        help=(
            "Override --refinement-passes for the opponent only. This is "
            "useful for matched Hero pass-count ablations."
        ),
    )
    parser.add_argument(
        "--candidate-torch-hero-policy-mode",
        choices=("dfm", "policy_only"),
        default="dfm",
        help=(
            "Use the full DFM policy or only the BT4 policy head stored in "
            "the candidate Hero checkpoint."
        ),
    )
    parser.add_argument(
        "--opponent-torch-hero-policy-mode",
        choices=("dfm", "policy_only"),
        default="dfm",
        help=(
            "Use the full DFM policy or only the BT4 policy head stored in "
            "the opponent Hero checkpoint."
        ),
    )
    parser.add_argument("--policy-batch-size-cap", type=int, default=64)
    parser.add_argument("--policy-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--collect-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect full refinement traces; disabled for lean arena play.",
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=DEFAULT_SOURCE_RUN_ROOT,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _resolved_run_options(
    args: argparse.Namespace,
    tier: FrozenTier,
) -> tuple[int, int, int]:
    if not tier.available:
        raise ValueError(f"{tier.name} tier is unavailable: {tier.unavailable_reason}")
    pair_count = tier.default_pair_count if args.pair_count is None else int(args.pair_count)
    block_pairs = tier.default_block_pairs if args.block_pairs is None else int(args.block_pairs)
    additional_ply_cap = (
        tier.default_additional_ply_cap
        if args.additional_ply_cap is None
        else int(args.additional_ply_cap)
    )
    if pair_count < 1 or pair_count > tier.default_pair_count:
        raise ValueError(f"{tier.name} pair_count must be in [1, {tier.default_pair_count}].")
    if block_pairs < 1 or block_pairs > pair_count:
        raise ValueError("block_pairs must be in [1, pair_count].")
    if tier.promotion_eligible and block_pairs != 1:
        raise ValueError(
            "Promotion requires --block-pairs 1 so the GSPRT can stop at "
            "the exact completed-pair boundary."
        )
    if additional_ply_cap is None:
        raise ValueError(
            f"{tier.name} has no frozen strength-valid ply cap yet; pass "
            "--additional-ply-cap explicitly and report its cap-draw rate."
        )
    if additional_ply_cap < 1:
        raise ValueError("additional_ply_cap must be positive.")
    return pair_count, block_pairs, additional_ply_cap


def _resolved_refinement_passes(
    args: argparse.Namespace,
) -> tuple[int, int]:
    default = int(args.refinement_passes)
    candidate = (
        default
        if args.candidate_refinement_passes is None
        else int(args.candidate_refinement_passes)
    )
    opponent = (
        default
        if args.opponent_refinement_passes is None
        else int(args.opponent_refinement_passes)
    )
    if default < 1:
        raise ValueError("refinement_passes must be positive.")
    if candidate < 1:
        raise ValueError("candidate_refinement_passes must be positive.")
    if opponent < 1:
        raise ValueError("opponent_refinement_passes must be positive.")
    return candidate, opponent


def _validate_native_torch_hero_mode(
    args: argparse.Namespace,
) -> bool:
    candidate_is_hero = args.candidate_torch_hero is not None
    opponent_is_hero = args.opponent_torch_hero is not None
    if (
        args.candidate_torch_hero_policy_mode != "dfm"
        and not candidate_is_hero
    ):
        raise ValueError(
            "--candidate-torch-hero-policy-mode requires "
            "--candidate-torch-hero."
        )
    if (
        args.opponent_torch_hero_policy_mode != "dfm"
        and not opponent_is_hero
    ):
        raise ValueError(
            "--opponent-torch-hero-policy-mode requires "
            "--opponent-torch-hero."
        )
    if opponent_is_hero and not candidate_is_hero:
        raise ValueError(
            "--opponent-torch-hero requires --candidate-torch-hero so both "
            "policies use one native Torch runtime and canonical codec."
        )
    if candidate_is_hero and not (
        opponent_is_hero or args.opponent_raw_bt4
    ):
        raise ValueError(
            "--candidate-torch-hero requires --opponent-torch-hero or "
            "--opponent-raw-bt4."
        )
    if candidate_is_hero and args.collect_diagnostics:
        raise ValueError(
            "Torch hero Arena supports only the lean diagnostics-off path."
        )
    return candidate_is_hero


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tier = FROZEN_TIERS[args.tier]
    pair_count, block_pairs, additional_ply_cap = _resolved_run_options(
        args,
        tier,
    )
    (
        candidate_refinement_passes,
        opponent_refinement_passes,
    ) = _resolved_refinement_passes(args)
    if args.policy_batch_size_cap < 1:
        raise ValueError("policy_batch_size_cap must be positive.")
    if not math.isfinite(args.policy_timeout_seconds) or args.policy_timeout_seconds <= 0:
        raise ValueError("policy_timeout_seconds must be positive and finite.")
    inference_batch_size = _static_inference_batch_size(
        block_pairs=block_pairs,
        policy_batch_size_cap=int(args.policy_batch_size_cap),
    )

    models_dir = require_within_workspace(args.models_dir)
    output_dir = require_within_workspace(args.output_dir)
    opening_pool = load_opening_pool(
        tier.pool_path,
        expected_pool_sha256=tier.pool_sha256,
    )
    loaded_histories = load_opening_history_sidecar(
        tier.histories_path,
        opening_pool=opening_pool,
        expected_manifest_sha256=tier.history_manifest_sha256,
    )

    native_torch_hero = _validate_native_torch_hero_mode(args)

    if args.candidate_raw_bt4:
        candidate_descriptor = None
        candidate_record = raw_bt4_descriptor(models_dir=models_dir)
        candidate_id = (
            "candidate-raw-bt4-"
            + candidate_record["bt4_checkpoint"]["sha256"][:12]
        )
    elif native_torch_hero:
        candidate_descriptor = torch_hero_checkpoint_descriptor(
            args.candidate_torch_hero,
            models_dir=models_dir,
        )
        candidate_record = candidate_descriptor.descriptor
        candidate_id = _torch_hero_model_id(
            role="candidate",
            descriptor=candidate_record,
        )
    elif args.candidate_torch is None:
        candidate_descriptor = research_checkpoint_descriptor(
            args.candidate,
            models_dir=models_dir,
        )
        candidate_record = candidate_descriptor.descriptor
        candidate_id = (
            "candidate-"
            + candidate_record["state"]["sha256"][:12]
        )
    else:
        candidate_descriptor = torch_research_checkpoint_descriptor(
            args.candidate_torch,
            models_dir=models_dir,
        )
        candidate_record = candidate_descriptor.descriptor
        candidate_id = (
            "candidate-torch-"
            + candidate_record["state"]["sha256"][:12]
        )
    opponent_hero_descriptor = None
    if args.opponent_torch_hero is not None:
        opponent_hero_descriptor = torch_hero_checkpoint_descriptor(
            args.opponent_torch_hero,
            models_dir=models_dir,
        )
        opponent_descriptor = opponent_hero_descriptor.descriptor
        opponent_config = None
        opponent_id = _torch_hero_model_id(
            role="opponent",
            descriptor=opponent_descriptor,
        )
    elif args.opponent_raw_bt4:
        opponent_descriptor = raw_bt4_descriptor(models_dir=models_dir)
        opponent_config = None
        opponent_id = (
            "raw-bt4-" + opponent_descriptor["bt4_checkpoint"]["sha256"][:12]
        )
    elif args.opponent_checkpoint is None:
        opponent_config, opponent_descriptor = source_checkpoint_descriptor(
            source_run_root=args.source_run_root,
            models_dir=models_dir,
        )
        opponent_id = "source-" + opponent_descriptor["state"]["sha256"][:12]
    else:
        incumbent_descriptor = research_checkpoint_descriptor(
            args.opponent_checkpoint,
            models_dir=models_dir,
        )
        opponent_descriptor = incumbent_descriptor.descriptor
        opponent_config = None
        opponent_id = "incumbent-" + opponent_descriptor["state"]["sha256"][:12]
    if candidate_id == opponent_id:
        raise ValueError("Candidate and opponent model IDs must be distinct.")

    contract = {
        "relative_elo_scope": RELATIVE_ELO_SCOPE,
        "absolute_elo_claim": False,
        "tier": {
            "name": tier.name,
            "promotion_eligible": tier.promotion_eligible,
            "pool_path": str(tier.pool_path),
            "pool_file_sha256": sha256_file(tier.pool_path),
            "pool_sha256": tier.pool_sha256,
            "ordered_fens_sha256": opening_pool["ordered_fens_sha256"],
            "histories_path": str(tier.histories_path),
            "history_file_sha256": loaded_histories.file_sha256,
            "history_manifest_sha256": (loaded_histories.manifest_sha256),
        },
        "models": {
            "candidate": {
                "model_id": candidate_id,
                **candidate_record,
                "arena_policy_mode": (
                    args.candidate_torch_hero_policy_mode
                    if native_torch_hero
                    else "default"
                ),
            },
            "opponent": {
                "model_id": opponent_id,
                **opponent_descriptor,
                "arena_policy_mode": (
                    args.opponent_torch_hero_policy_mode
                    if args.opponent_torch_hero is not None
                    else "default"
                ),
            },
        },
        "run": {
            "pair_count": pair_count,
            "block_pairs": block_pairs,
            "additional_ply_cap": additional_ply_cap,
            "refinement_passes": int(args.refinement_passes),
            "candidate_refinement_passes": candidate_refinement_passes,
            "opponent_refinement_passes": opponent_refinement_passes,
            "candidate_torch_hero_policy_mode": (
                args.candidate_torch_hero_policy_mode
            ),
            "opponent_torch_hero_policy_mode": (
                args.opponent_torch_hero_policy_mode
            ),
            "policy_batch_size_cap": int(args.policy_batch_size_cap),
            "policy_timeout_seconds": float(args.policy_timeout_seconds),
            "collect_diagnostics": bool(args.collect_diagnostics),
            "seed": int(args.seed),
            "opening_start_index": 0,
            "deterministic_greedy_policy": True,
            "jepa_used_at_inference": (
                (
                    args.candidate_torch_hero_policy_mode != "policy_only"
                    and _descriptor_uses_jepa_at_inference(candidate_record)
                )
                or (
                    args.opponent_torch_hero_policy_mode != "policy_only"
                    and _descriptor_uses_jepa_at_inference(
                        opponent_descriptor
                    )
                )
            ),
        },
        "inference_batching": {
            "schema_version": STATIC_INFERENCE_BATCHING_SCHEMA,
            "physical_batch_size": inference_batch_size,
            "padding_mode": STATIC_INFERENCE_PADDING_MODE,
            "padding_source": "first_validated_active_encoded_row",
            "active_rows": "ordered_prefix",
            "output_handling": "slice_to_active_rows_before_semantic_validation",
            "host_validation_scope": "active_rows_only",
            "metrics_basis": "real_rows_only",
            "timing_basis": "physical_padded_call_wall_time",
            "runtime_rng": "none_deterministic_greedy_inference",
        },
        "history_validation": {
            "schema_version": HISTORY_VALIDATION_SCHEMA,
            "public_policy_default": HISTORY_VALIDATION_FULL_REPLAY,
            "arena_hot_path": HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
            "opening_validation": "full_exact_replay_sidecar_and_selected_pairs",
            "authoritative_state": "full_stack_with_checked_legal_pushes",
            "policy_board_copy": "stack_false",
            "policy_payload": "sealed_constant_size_state_attestation",
            "repetition_authority": "runner_claim_draw_boundary_check",
            "complexity": "constant_per_policy_row",
        },
        "promotion_gate": (
            GSPRTState(GSPRTConfig.normalized_promotion()).as_dict()["config"]
            if tier.promotion_eligible
            else None
        ),
        "codec_capabilities": {
            "candidate": action_codec_capability(
                str(candidate_record["action_codec_id"])
            ),
            "opponent": action_codec_capability(
                str(opponent_descriptor["action_codec_id"])
            ),
        },
        "code": _code_provenance(),
    }

    if (output_dir / "state.json").exists() and args.resume:
        existing = load_run_state(
            output_dir,
            expected_contract=contract,
        )
        if existing["status"] != "running":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    if native_torch_hero:
        assert isinstance(
            candidate_descriptor,
            TorchResearchCheckpointDescriptor,
        )
        (
            candidate_policy,
            candidate_load,
            opponent_policy,
            opponent_load,
            accelerator_session,
        ) = load_native_torch_hero_pair(
            candidate_checkpoint=candidate_descriptor,
            opponent_checkpoint=opponent_hero_descriptor,
            candidate_record=candidate_record,
            opponent_record=opponent_descriptor,
            candidate_id=candidate_id,
            opponent_id=opponent_id,
            candidate_policy_mode=args.candidate_torch_hero_policy_mode,
            opponent_policy_mode=args.opponent_torch_hero_policy_mode,
            candidate_refinement_passes=candidate_refinement_passes,
            opponent_refinement_passes=opponent_refinement_passes,
            inference_batch_size=inference_batch_size,
        )
    else:
        bt4_params = load_mapped_bt4_params(models_dir=models_dir)
        if args.candidate_raw_bt4:
            candidate_policy, candidate_load = load_raw_bt4_policy(
                bt4_params=bt4_params,
                model_id=candidate_id,
                inference_batch_size=inference_batch_size,
            )
        elif args.candidate_torch is None:
            assert isinstance(
                candidate_descriptor,
                ResearchCheckpointDescriptor,
            )
            candidate_policy, candidate_load = load_research_policy(
                candidate_descriptor,
                bt4_params=bt4_params,
                model_id=candidate_id,
                seed=args.seed,
                refinement_passes=candidate_refinement_passes,
                collect_diagnostics=args.collect_diagnostics,
                inference_batch_size=inference_batch_size,
            )
        else:
            assert isinstance(
                candidate_descriptor,
                TorchResearchCheckpointDescriptor,
            )
            candidate_policy, candidate_load = load_torch_research_policy(
                candidate_descriptor,
                bt4_params=bt4_params,
                model_id=candidate_id,
                seed=args.seed,
                refinement_passes=candidate_refinement_passes,
                collect_diagnostics=args.collect_diagnostics,
                inference_batch_size=inference_batch_size,
            )
        if args.opponent_raw_bt4:
            opponent_policy, opponent_load = load_raw_bt4_policy(
                bt4_params=bt4_params,
                model_id=opponent_id,
                inference_batch_size=inference_batch_size,
            )
        elif args.opponent_checkpoint is None:
            opponent_policy, opponent_load = load_source_policy(
                config=opponent_config,
                descriptor=opponent_descriptor,
                bt4_params=bt4_params,
                model_id=opponent_id,
                seed=args.seed + 1,
                refinement_passes=opponent_refinement_passes,
                collect_diagnostics=args.collect_diagnostics,
                inference_batch_size=inference_batch_size,
            )
        else:
            opponent_policy, opponent_load = load_research_policy(
                incumbent_descriptor,
                bt4_params=bt4_params,
                model_id=opponent_id,
                seed=args.seed + 1,
                refinement_passes=opponent_refinement_passes,
                collect_diagnostics=args.collect_diagnostics,
                inference_batch_size=inference_batch_size,
            )
        accelerator_session = {
            "framework": "jax",
            "jax_devices": [
                f"{device.platform}:{device.id}" for device in jax.devices()
            ],
        }

    warm_batch = inference_batch_size
    candidate_warm = _warm_policy(
        candidate_policy,
        opening_pool=opening_pool,
        loaded_histories=loaded_histories,
        batch_size=warm_batch,
    )
    opponent_warm = _warm_policy(
        opponent_policy,
        opening_pool=opening_pool,
        loaded_histories=loaded_histories,
        batch_size=warm_batch,
    )
    session_setup = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        **accelerator_session,
        "candidate_load": candidate_load,
        "opponent_load": opponent_load,
        "warmup": {
            "active_batch_size": warm_batch,
            "physical_batch_size": inference_batch_size,
            "candidate_seconds": candidate_warm,
            "opponent_seconds": opponent_warm,
        },
    }
    final = run_blocks(
        output_dir=output_dir,
        contract=contract,
        opening_pool=opening_pool,
        loaded_histories=loaded_histories,
        candidate_policy=candidate_policy,
        opponent_policy=opponent_policy,
        resume=args.resume,
        session_setup=session_setup,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": final["status"],
                "relative_elo_scope": final["relative_elo_scope"],
                "pair_count": final["aggregate"]["pair_count"],
                "cap_draw_rate": final["aggregate"]["cap_draw_rate"],
                "pentanomial": final["aggregate"]["pentanomial"],
                "normalized_promotion_state": final["normalized_promotion_state"],
                "candidate_promoted": final["candidate_promoted"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
