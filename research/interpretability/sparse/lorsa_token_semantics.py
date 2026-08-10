"""Exact token-level semantic pilot for the reviewed published LoRSA.

This runner is intentionally downstream of a sealed Raw-to-Hero LoRSA transfer
run.  It refuses to execute unless that source artifact passes checksum,
identity, conversion, source-compatibility, and Hero-transfer gates.  It then
uses only development positions from the public-v2 corpus, makes a deterministic
group-disjoint Raw-fit/heldout split, and streams full ``[B, 64, 16384]`` LoRSA
features into CPU sufficient statistics.  Dense features are never written to
disk or retained across batches.

Feature/concept pairs are selected exactly once on Raw-fit positions.  The
frozen pairs are evaluated on heldout Raw and heldout Hero activations.  Results
are exploratory descriptive associations, including negative and null results;
they are not evidence of feature causality, model equivalence, or internal
search.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from research.interpretability.artifacts import (
    canonical_json_bytes,
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_text_atomic,
)
from research.interpretability.chess_concepts import PIECE_CODE_NAMES
from research.interpretability.chessbench import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_DIR,
    load_public_corpus,
)
from research.interpretability.models import (
    BT4ComparisonModels,
    load_bt4_comparison_models,
)
from research.interpretability.sparse.feature_semantics import (
    SparseConceptSpec,
    SparseFeatureConceptAccumulator,
    evaluate_sparse_feature_concepts,
)
from research.interpretability.sparse.lorsa import (
    LoadedConvertedLoRSA,
    LowRankSparseAttention,
    load_converted_lorsa,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_REVIEWED_LORSA_DIR = (
    _REPO_ROOT
    / ".local/artifacts/lorsa/converted/JacklE0niden_lc0-BT4-lorsa"
    / "1ec52be63cce41017b3853edbe9421b643a1476d/k_30_e_16/L14_reviewed_v1"
)
DEFAULT_RAW_BT4 = _REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz"
DEFAULT_HERO_CHECKPOINT = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint"
DEFAULT_LORSA_MANIFEST = _REVIEWED_LORSA_DIR / "manifest.json"
DEFAULT_LORSA_WEIGHTS = _REVIEWED_LORSA_DIR / "lorsa.safetensors"
DEFAULT_SOURCE_TRANSFER = (
    _REPO_ROOT / "research/analysis/published_lorsa_l14_transfer_dev512_v3"
)
DEFAULT_OUTPUT = (
    _REPO_ROOT / "research/analysis/published_lorsa_l14_token_semantics_dev512_v3"
)
SOURCE_TRANSFER_RUN_SCHEMA = "bt4-published-lorsa-transfer-run-v1"
SOURCE_TRANSFER_METRICS_SCHEMA = "bt4-published-lorsa-transfer-metrics-v1"
TOKEN_SEMANTICS_RUN_SCHEMA = "bt4-lorsa-token-semantics-run-v2"
TOKEN_SEMANTICS_METRICS_SCHEMA = "bt4-lorsa-token-semantics-metrics-v2"
TOKEN_SEMANTICS_SELECTION_SCHEMA = "bt4-lorsa-token-semantics-selection-v2"
TOKEN_SEMANTICS_POSITIONS_SCHEMA = "bt4-lorsa-token-semantics-positions-v2"
FORBIDDEN_FEATURE_SUFFIXES = {
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}


@dataclass(frozen=True)
class TokenConceptBinding:
    """Bind one exact public-v2 square array to a categorical concept."""

    spec: SparseConceptSpec
    array: str
    attack_count_bins: bool = False

    def validate(self) -> None:
        self.spec.validate()
        if not self.array:
            raise ValueError("Token concepts require a corpus array")
        if self.attack_count_bins and len(self.spec.class_names) != 5:
            raise ValueError("Attack-count concepts require exactly five bins")


@dataclass(frozen=True)
class DevelopmentSelection:
    fit_rows: np.ndarray
    heldout_rows: np.ndarray
    metadata: dict[str, Any]


def exact_token_concepts() -> tuple[TokenConceptBinding, ...]:
    """Return the preregistered exact, engine-free token concept family."""

    binary = ("absent", "present")
    attack_bins = ("zero", "one", "two", "three", "four_or_more")
    concepts = (
        TokenConceptBinding(
            SparseConceptSpec("piece_code", tuple(PIECE_CODE_NAMES)),
            "piece_code_i8",
        ),
        TokenConceptBinding(
            SparseConceptSpec("legal_origin", binary), "legal_origin_u8"
        ),
        TokenConceptBinding(
            SparseConceptSpec("legal_destination", binary), "legal_destination_u8"
        ),
        TokenConceptBinding(
            SparseConceptSpec("attacked_undefended_ours", binary),
            "attacked_undefended_ours_u8",
        ),
        TokenConceptBinding(
            SparseConceptSpec("attacked_undefended_theirs", binary),
            "attacked_undefended_theirs_u8",
        ),
        TokenConceptBinding(
            SparseConceptSpec("pinned_ours", binary), "pinned_ours_u8"
        ),
        TokenConceptBinding(
            SparseConceptSpec("pinned_theirs", binary), "pinned_theirs_u8"
        ),
        TokenConceptBinding(
            SparseConceptSpec("capture_destination", binary),
            "capture_destination_u8",
        ),
        TokenConceptBinding(
            SparseConceptSpec("checking_destination", binary),
            "checking_destination_u8",
        ),
        TokenConceptBinding(
            SparseConceptSpec("our_king_zone", binary), "our_king_zone_u8"
        ),
        TokenConceptBinding(
            SparseConceptSpec("their_king_zone", binary), "their_king_zone_u8"
        ),
        TokenConceptBinding(
            SparseConceptSpec("attack_count_ours", attack_bins),
            "attack_count_ours_u8",
            attack_count_bins=True,
        ),
        TokenConceptBinding(
            SparseConceptSpec("attack_count_theirs", attack_bins),
            "attack_count_theirs_u8",
            attack_count_bins=True,
        ),
    )
    for concept in concepts:
        concept.validate()
    return concepts


def concept_contract(
    concepts: Sequence[TokenConceptBinding] | None = None,
) -> dict[str, Any]:
    selected = tuple(concepts or exact_token_concepts())
    return {
        "schema_version": "bt4-lorsa-exact-token-concepts-v1",
        "engine_calls": False,
        "orientation": (
            "BT4 side-to-move token frame; black-to-move squares are vertically mirrored"
        ),
        "attack_count_binning": "min(exact python-chess attacker count, 4)",
        "concepts": [
            {
                "name": binding.spec.name,
                "source_array": binding.array,
                "class_names": list(binding.spec.class_names),
                "attack_count_bins": binding.attack_count_bins,
            }
            for binding in selected
        ],
        "definitions": {
            "attacked_undefended": (
                "occupied square has at least one enemy attacker and zero friendly attackers"
            ),
            "legal_origin_destination": "union over exact legal moves in the position",
            "capture_check_destination": (
                "union of destinations of exact legal captures/checking moves"
            ),
            "king_zone": "king square plus its adjacent king-attack squares",
            "pinned": "exact absolute pin under python-chess board rules",
        },
    }


def labels_for_rows(
    puzzles: Mapping[str, np.ndarray],
    rows: np.ndarray,
    concepts: Sequence[TokenConceptBinding] | None = None,
) -> dict[str, Tensor]:
    """Materialize only selected exact labels; never inspect unselected concepts."""

    selected = tuple(concepts or exact_token_concepts())
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("Semantic rows must be a one-dimensional integer array")
    labels: dict[str, Tensor] = {}
    for binding in selected:
        binding.validate()
        if binding.array not in puzzles:
            raise ValueError(f"Public-v2 corpus is missing {binding.array!r}")
        source = np.asarray(puzzles[binding.array])
        if source.ndim != 2 or source.shape[1] != 64:
            raise ValueError(f"Corpus concept {binding.array!r} must have shape [N, 64]")
        if source.dtype.kind not in "iub":
            raise ValueError(f"Corpus concept {binding.array!r} must be integer-valued")
        value = source[rows].astype(np.int64, copy=True)
        if binding.attack_count_bins:
            if bool((value < 0).any()):
                raise ValueError("Exact attack counts cannot be negative")
            np.minimum(value, 4, out=value)
        width = len(binding.spec.class_names)
        if bool(((value < 0) | (value >= width)).any()):
            raise ValueError(f"Labels for {binding.spec.name!r} violate class bounds")
        labels[binding.spec.name] = torch.from_numpy(value)
    return labels


def _rank(seed: int, namespace: str, value: str) -> tuple[str, str]:
    payload = f"{seed}\0{namespace}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), value


def _string_identity(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(sorted(values))).hexdigest()


def select_group_disjoint_development_rows(
    puzzles: Mapping[str, np.ndarray],
    *,
    position_count: int,
    fit_fraction: float,
    seed: int,
) -> DevelopmentSelection:
    """Select deterministic development rows with strict group disjointness."""

    if type(position_count) is not int or position_count < 2:
        raise ValueError("position_count must be an integer of at least two")
    if not math.isfinite(fit_fraction) or not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must lie strictly between zero and one")
    required = {"split_u8", "position_id", "group_id"}
    missing = sorted(required - set(puzzles))
    if missing:
        raise ValueError("Public-v2 puzzles are missing: " + ", ".join(missing))
    split = np.asarray(puzzles["split_u8"])
    position_ids = np.asarray(puzzles["position_id"])
    group_ids = np.asarray(puzzles["group_id"])
    if split.ndim != 1 or position_ids.shape != split.shape or group_ids.shape != split.shape:
        raise ValueError("Split, position, and group arrays must be aligned vectors")
    development = np.flatnonzero(split == 1)
    if development.size < position_count:
        raise ValueError("Development split is smaller than the requested semantic pilot")

    groups: dict[str, list[int]] = {}
    seen_positions: set[str] = set()
    for row_value in development:
        row = int(row_value)
        position_id = str(position_ids[row])
        group_id = str(group_ids[row])
        if not position_id or not group_id:
            raise ValueError("Development positions require non-empty identities")
        if position_id in seen_positions:
            raise ValueError("Development position IDs must be unique")
        seen_positions.add(position_id)
        groups.setdefault(group_id, []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two development groups are required")

    fit_count = int(math.floor(position_count * fit_fraction))
    fit_count = min(max(fit_count, 1), position_count - 1)
    heldout_count = position_count - fit_count
    ordered_groups = sorted(groups, key=lambda value: _rank(seed, "group", value))
    prefix = 0
    candidates: list[tuple[int, int, int]] = []
    for cut in range(1, len(ordered_groups)):
        prefix += len(groups[ordered_groups[cut - 1]])
        suffix = int(development.size) - prefix
        if prefix >= fit_count and suffix >= heldout_count:
            candidates.append((abs(prefix - fit_count), cut, prefix))
    if not candidates:
        raise ValueError(
            "No deterministic group cut can satisfy the requested fit/heldout counts"
        )
    _, cut, fit_available = min(candidates)
    fit_groups = set(ordered_groups[:cut])
    heldout_groups = set(ordered_groups[cut:])
    if fit_groups & heldout_groups:
        raise AssertionError("Group partition unexpectedly overlaps")

    def choose(group_set: set[str], count: int, namespace: str) -> np.ndarray:
        candidates_for_role = [
            row for group in group_set for row in groups[group]
        ]
        candidates_for_role.sort(
            key=lambda row: _rank(seed, namespace, str(position_ids[row]))
        )
        return np.asarray(candidates_for_role[:count], dtype=np.int64)

    fit_rows = choose(fit_groups, fit_count, "raw-fit-position")
    heldout_rows = choose(heldout_groups, heldout_count, "heldout-position")
    if fit_rows.size != fit_count or heldout_rows.size != heldout_count:
        raise AssertionError("Deterministic position selection returned the wrong size")
    selected_fit_groups = {str(group_ids[row]) for row in fit_rows}
    selected_heldout_groups = {str(group_ids[row]) for row in heldout_rows}
    if selected_fit_groups & selected_heldout_groups:
        raise AssertionError("Selected semantic positions leak groups across splits")
    if not bool((split[fit_rows] == 1).all() and (split[heldout_rows] == 1).all()):
        raise AssertionError("Semantic selection escaped the development split")

    fit_ids = [str(position_ids[row]) for row in fit_rows]
    heldout_ids = [str(position_ids[row]) for row in heldout_rows]
    metadata = {
        "method": (
            "SHA-256(seed, namespace, identity) ordering; constrained group cut; "
            "fixed per-partition position counts"
        ),
        "seed": seed,
        "requested_position_count": position_count,
        "fit_fraction_requested": fit_fraction,
        "fit_position_count": int(fit_rows.size),
        "heldout_position_count": int(heldout_rows.size),
        "fit_group_count": len(selected_fit_groups),
        "heldout_group_count": len(selected_heldout_groups),
        "fit_partition_available_positions": fit_available,
        "heldout_partition_available_positions": int(development.size) - fit_available,
        "fit_group_identity_sha256": _string_identity(sorted(selected_fit_groups)),
        "heldout_group_identity_sha256": _string_identity(
            sorted(selected_heldout_groups)
        ),
        "fit_position_identity_sha256": _string_identity(fit_ids),
        "heldout_position_identity_sha256": _string_identity(heldout_ids),
        "group_disjoint": True,
        "selection_source_model": "Raw",
        "selection_split": "Raw-fit development groups",
        "evaluation_splits": [
            "Raw heldout development groups",
            "Hero heldout development groups",
        ],
        "test_rows_selected": 0,
    }
    return DevelopmentSelection(fit_rows, heldout_rows, metadata)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Transfer example line {line_number} is not an object")
            rows.append(value)
    return rows


def validate_sealed_source_transfer(
    transfer_dir: Path,
    *,
    expected_identities: Mapping[str, str],
    puzzles: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Independently verify the source transfer bundle and every scientific gate."""

    required_identities = {
        "corpus_integrity",
        "lorsa_weights",
        "lorsa_manifest",
        "raw",
        "hero",
    }
    if set(expected_identities) != required_identities:
        raise ValueError("Expected source identities differ from the exact gate contract")
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in expected_identities.values()
    ):
        raise ValueError("Expected source identities must be lowercase SHA-256 values")

    root = transfer_dir.resolve(strict=True)
    checksums = verify_checksums(root)
    expected_files = {"metrics.json", "examples.jsonl", "manifest.json", "run.log"}
    manifest = _load_json_object(root / "manifest.json", label="Transfer manifest")
    metrics = _load_json_object(root / "metrics.json", label="Transfer metrics")
    examples = _load_jsonl_objects(root / "examples.jsonl")
    payloads = manifest.get("payload_identities", {})
    lorsa = manifest.get("lorsa", {})
    lattice = manifest.get("model_lattice", {})
    corpus = manifest.get("public_corpus_manifest", {})
    experiment = manifest.get("experiment", {})
    interpretation = metrics.get("interpretation_contract", {})
    artifact_gate = metrics.get("artifact_contract_gate", {})
    source_gate = metrics.get("source_compatibility_gate", {})
    source_stage = metrics.get("source_stage", {})
    transfer_stage = metrics.get("transfer_stage", {})
    if not all(
        isinstance(value, Mapping)
        for value in (
            payloads,
            lorsa,
            lattice,
            corpus,
            experiment,
            interpretation,
            artifact_gate,
            source_gate,
            source_stage,
            transfer_stage,
        )
    ):
        raise ValueError("Transfer bundle has malformed nested contracts")

    source_files = manifest.get("source_files")
    source_files_bound_in_run_id = "source_files" in payloads
    source_files_manifest_well_formed = (
        isinstance(source_files, Mapping)
        and bool(source_files)
        and all(
            isinstance(path, str)
            and bool(path)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for path, digest in source_files.items()
        )
    )
    source_files_identity_matches = (
        source_files_bound_in_run_id
        and source_files_manifest_well_formed
        and payloads.get("source_files") == content_identity(dict(source_files))
    )

    position_ids = [str(example.get("position_id", "")) for example in examples]
    groups_match = True
    development_only = True
    if not all(position_ids) or len(position_ids) != len(set(position_ids)):
        groups_match = False
        development_only = False
    if puzzles is not None and development_only:
        split = np.asarray(puzzles.get("split_u8"))
        corpus_position = np.asarray(puzzles.get("position_id"))
        corpus_group = np.asarray(puzzles.get("group_id"))
        if (
            split.ndim != 1
            or corpus_position.shape != split.shape
            or corpus_group.shape != split.shape
        ):
            raise ValueError("Public corpus identity arrays are malformed")
        development_rows = np.flatnonzero(split == 1)
        lookup = {
            str(corpus_position[row]): (int(row), str(corpus_group[row]))
            for row in development_rows
        }
        for example, position_id in zip(examples, position_ids, strict=True):
            if position_id not in lookup:
                development_only = False
                groups_match = False
                continue
            _, group_id = lookup[position_id]
            groups_match &= str(example.get("group_id", "")) == group_id

    metrics_identity = content_identity(metrics)
    positions_identity = hashlib.sha256("\n".join(position_ids).encode()).hexdigest()
    run_identity = content_identity(dict(payloads)) if isinstance(payloads, Mapping) else ""
    criteria = {
        "source_files_bound_in_run_id": source_files_bound_in_run_id,
        "source_files_manifest_well_formed": source_files_manifest_well_formed,
        "source_files_identity_matches": source_files_identity_matches,
        "checksum_inventory_exact": set(checksums) == expected_files,
        "manifest_schema_exact": manifest.get("schema_version")
        == SOURCE_TRANSFER_RUN_SCHEMA,
        "metrics_schema_exact": metrics.get("schema_version")
        == SOURCE_TRANSFER_METRICS_SCHEMA,
        "metrics_identity_matches": payloads.get("metrics") == metrics_identity,
        "positions_identity_matches": payloads.get("positions") == positions_identity,
        "run_identity_matches": manifest.get("run_id") == f"sha256:{run_identity}",
        "position_counts_match": len(examples) == metrics.get("position_count")
        == experiment.get("position_count"),
        "examples_unique_and_complete": development_only and groups_match,
        "artifact_contract_passed": artifact_gate.get("passed") is True,
        "source_compatibility_passed": source_gate.get("passed") is True,
        "source_stage_complete": source_stage.get("status") == "complete",
        "hero_transfer_complete": transfer_stage.get("status") == "complete",
        "development_split_only": interpretation.get("development_split_only") is True,
        "source_test_split_unopened": interpretation.get("test_split_opened") is False,
        "source_lorsa_frozen": interpretation.get("lorsa_frozen") is True
        and lorsa.get("parameters_frozen") is True,
        "hero_transfer_declared_interpretable": (
            interpretation.get("hero_transfer_interpretable") is True
        ),
        "corpus_identity_matches": corpus.get("integrity_sha256")
        == expected_identities["corpus_integrity"],
        "lorsa_payload_identity_matches": payloads.get("lorsa")
        == expected_identities["lorsa_weights"],
        "lorsa_before_after_frozen": lorsa.get("weights_sha256_before")
        == lorsa.get("weights_sha256_after")
        == lorsa.get("weights_sha256")
        == expected_identities["lorsa_weights"],
        "lorsa_manifest_identity_matches": lorsa.get("manifest_sha256")
        == expected_identities["lorsa_manifest"],
        "raw_identity_matches": payloads.get("raw")
        == lattice.get("raw_asset_sha256")
        == expected_identities["raw"],
        "hero_identity_matches": payloads.get("hero")
        == lattice.get("hero_state_sha256")
        == expected_identities["hero"],
        "layer_exact": experiment.get("layer") == 14,
        "feature_width_exact": lorsa.get("architecture", {}).get("n_ov_heads") == 16_384,
        "token_width_exact": lorsa.get("architecture", {}).get("n_ctx") == 64,
        "main_environment_pickle_excluded": (
            lorsa.get("main_environment_pickle_loaded") is False
        ),
    }
    return {
        "schema_version": "bt4-lorsa-source-transfer-verification-v1",
        "source_files_bound_in_run_id": source_files_bound_in_run_id,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "source_transfer_run_id": manifest.get("run_id"),
        "source_transfer_metrics_identity": metrics_identity,
        "source_transfer_positions_identity": positions_identity,
        "source_transfer_checksum_identity": content_identity(checksums),
        "position_count": len(examples),
        "failure_action": "abort before loading models or evaluating semantic features",
    }


def local_source_identities(
    *,
    corpus_manifest: Mapping[str, Any],
    raw_bt4: Path,
    hero_checkpoint: Path,
    lorsa_manifest: Path,
    lorsa_weights: Path,
) -> dict[str, str]:
    """Hash the exact local assets used by the semantic run."""

    integrity = corpus_manifest.get("manifest_integrity", {}).get("sha256")
    if not isinstance(integrity, str):
        raise ValueError("Public-v2 corpus has no manifest integrity SHA-256")
    hero_manifest = _load_json_object(
        hero_checkpoint.resolve(strict=True) / "manifest.json",
        label="Hero checkpoint manifest",
    )
    state = hero_manifest.get("state", {})
    if not isinstance(state, Mapping):
        raise ValueError("Hero checkpoint manifest has no state object")
    state_path = (hero_checkpoint.resolve() / str(state.get("path", ""))).resolve(
        strict=True
    )
    try:
        state_path.relative_to(hero_checkpoint.resolve())
    except ValueError as exc:
        raise ValueError("Hero checkpoint state path escapes its directory") from exc
    hero_sha = sha256_file(state_path)
    if state.get("sha256") != hero_sha:
        raise ValueError("Hero checkpoint state SHA-256 drift")
    return {
        "corpus_integrity": integrity,
        "lorsa_weights": sha256_file(lorsa_weights.resolve(strict=True)),
        "lorsa_manifest": sha256_file(lorsa_manifest.resolve(strict=True)),
        "raw": sha256_file(raw_bt4.resolve(strict=True)),
        "hero": hero_sha,
    }


def _module_dtype(module: LowRankSparseAttention) -> torch.dtype:
    dtypes = {parameter.dtype for parameter in module.parameters()}
    if len(dtypes) != 1:
        raise ValueError("LoRSA parameters have mixed dtypes")
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise ValueError("LoRSA parameters must be floating-point")
    return dtype


@torch.inference_mode()
def _stream_arm(
    *,
    models: BT4ComparisonModels,
    loaded_lorsa: LoadedConvertedLoRSA,
    puzzles: Mapping[str, np.ndarray],
    rows: np.ndarray,
    accumulator: SparseFeatureConceptAccumulator,
    concepts: Sequence[TokenConceptBinding],
    arm: str,
    layer: int,
    batch_size: int,
    device: torch.device,
    progress_every: int,
    log: Any,
) -> dict[str, Any]:
    if arm not in {"RR", "HH"}:
        raise ValueError("Token semantics supports only native Raw and Hero arms")
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("Streaming rows must be a non-empty vector")
    checkpoint_dtype = _module_dtype(loaded_lorsa.module)
    observed_feature_shape: list[int | str] | None = None
    for start in range(0, int(rows.size), batch_size):
        selected = rows[start : start + batch_size]
        planes = torch.from_numpy(
            np.asarray(puzzles["current_planes_u8"])[selected]
        ).to(device)
        labels = labels_for_rows(puzzles, selected, concepts)
        dense = models.policy_logits_with_captures(
            planes,
            arm=arm,
            compute_dtype=torch.float32,
            capture_layers=(layer,),
        )
        if dense.captures is None:
            raise AssertionError("Requested attention input capture was not returned")
        states = dense.captures.hook_attn_in[0]
        if torch.is_autocast_enabled(states.device.type):
            raise RuntimeError("Autocast must be disabled at the LoRSA boundary")
        sparse = loaded_lorsa.module(states.to(dtype=checkpoint_dtype))
        features = sparse.features
        expected_shape = (selected.size, 64, accumulator.n_features)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"LoRSA feature shape {tuple(features.shape)} differs from {expected_shape}"
            )
        accumulator.update(features, labels)
        observed_feature_shape = ["batch", 64, accumulator.n_features]
        del features, sparse, states, dense, labels, planes
        completed = min(start + batch_size, int(rows.size))
        if completed == rows.size or completed % progress_every == 0:
            log(f"{arm} streamed {completed}/{rows.size} positions")
    return {
        "arm": arm,
        "position_count": int(rows.size),
        "feature_shape": observed_feature_shape,
        "checkpoint_dtype": str(checkpoint_dtype),
        "features_persisted": False,
        "activity": accumulator.activity_metrics(),
    }


def _pair_id(row: Mapping[str, Any]) -> str:
    return (
        f"{row['concept']}:{int(row['class_index'])}:"
        f"feature_{int(row['feature_index'])}"
    )


def select_evidence_gated_feature_concepts(
    accumulator: SparseFeatureConceptAccumulator,
    *,
    top_per_direction_per_class: int,
    minimum_feature_support: int,
    minimum_positive_joint_support: int,
    minimum_negative_expected_joint_support: float,
    smoothing: float,
) -> list[dict[str, Any]]:
    """Select separate positive/enrichment and negative/exclusion evidence.

    A positive lift is eligible only when the class is actually observed often
    enough while the feature is active.  A negative lift is eligible only when
    the class would have been expected often enough under the corpus baseline;
    this prevents tiny expected counts from becoming extreme smoothed
    exclusions.  All ranking and gating uses Raw-fit statistics only.
    """

    if type(top_per_direction_per_class) is not int or top_per_direction_per_class <= 0:
        raise ValueError("top_per_direction_per_class must be a positive integer")
    if type(minimum_feature_support) is not int or minimum_feature_support <= 0:
        raise ValueError("minimum_feature_support must be a positive integer")
    if type(minimum_positive_joint_support) is not int or minimum_positive_joint_support <= 0:
        raise ValueError("minimum_positive_joint_support must be a positive integer")
    if (
        not math.isfinite(minimum_negative_expected_joint_support)
        or minimum_negative_expected_joint_support <= 0
    ):
        raise ValueError("minimum_negative_expected_joint_support must be positive")

    selected: list[dict[str, Any]] = []
    for concept in accumulator.concepts:
        effects = accumulator._effect_tensors(concept.name, smoothing=smoothing)
        lift = effects["log2_lift"]
        activation_lift = effects["activation_log2_lift"]
        support = effects["active_support"]
        joint = effects["joint_support"]
        baseline = effects["baseline"]
        class_tokens = effects["class_tokens"]
        assert isinstance(lift, Tensor)
        assert isinstance(activation_lift, Tensor)
        assert isinstance(support, Tensor)
        assert isinstance(joint, Tensor)
        assert isinstance(baseline, Tensor)
        assert isinstance(class_tokens, Tensor)
        eligible_features = torch.nonzero(
            support >= minimum_feature_support, as_tuple=False
        ).flatten().tolist()
        for class_index, class_name in enumerate(concept.class_names):
            if int(class_tokens[class_index]) == 0:
                continue
            candidates: dict[str, list[tuple[float, int]]] = {
                "positive": [],
                "negative": [],
            }
            for feature_index in eligible_features:
                value = float(lift[feature_index, class_index])
                direction = "positive" if value > 0 else "negative" if value < 0 else "zero"
                if direction == "zero":
                    continue
                observed_joint = int(joint[feature_index, class_index])
                expected_joint = float(
                    support[feature_index].double() * baseline[class_index]
                )
                if direction == "positive" and observed_joint < minimum_positive_joint_support:
                    continue
                if (
                    direction == "negative"
                    and expected_joint < minimum_negative_expected_joint_support
                ):
                    continue
                candidates[direction].append((abs(value), feature_index))
            for direction in ("positive", "negative"):
                candidates[direction].sort(key=lambda item: (-item[0], item[1]))
                for _, feature_index in candidates[direction][
                    :top_per_direction_per_class
                ]:
                    value = float(lift[feature_index, class_index])
                    observed_joint = int(joint[feature_index, class_index])
                    expected_joint = float(
                        support[feature_index].double() * baseline[class_index]
                    )
                    selected.append(
                        {
                            "concept": concept.name,
                            "class_index": class_index,
                            "class_name": class_name,
                            "feature_index": feature_index,
                            "fit_log2_lift": value,
                            "fit_activation_log2_lift": float(
                                activation_lift[feature_index, class_index]
                            ),
                            "fit_direction": direction,
                            "fit_feature_support": int(support[feature_index]),
                            "fit_joint_support": observed_joint,
                            "fit_baseline_rate": float(baseline[class_index]),
                            "fit_expected_joint_support": expected_joint,
                            "fit_evidence_gate": True,
                            "fit_evidence_quantity": (
                                "observed_joint_support"
                                if direction == "positive"
                                else "expected_joint_support_under_baseline"
                            ),
                        }
                    )
    return selected


def evaluate_evidence_gated_feature_concepts(
    accumulator: SparseFeatureConceptAccumulator,
    selection: Sequence[Mapping[str, Any]],
    *,
    minimum_feature_support: int,
    minimum_positive_joint_support: int,
    minimum_negative_expected_joint_support: float,
    smoothing: float,
) -> dict[str, Any]:
    """Evaluate frozen pairs with direction-appropriate heldout evidence gates."""

    report = evaluate_sparse_feature_concepts(
        accumulator,
        selection,
        minimum_evaluation_support=minimum_feature_support,
        smoothing=smoothing,
    )
    effect_cache = {
        concept.name: accumulator._effect_tensors(concept.name, smoothing=smoothing)
        for concept in accumulator.concepts
    }
    supported = 0
    replicated = 0
    for row in report["pairs"]:
        effects = effect_cache[str(row["concept"])]
        baseline = effects["baseline"]
        assert isinstance(baseline, Tensor)
        class_index = int(row["class_index"])
        feature_support = int(row["evaluation_feature_support"])
        observed_joint = int(row["evaluation_joint_support"])
        expected_joint = float(feature_support * baseline[class_index])
        direction = str(row["fit_direction"])
        if direction == "positive":
            evidence_value = float(observed_joint)
            evidence_threshold = float(minimum_positive_joint_support)
            evidence_quantity = "observed_joint_support"
        elif direction == "negative":
            evidence_value = expected_joint
            evidence_threshold = float(minimum_negative_expected_joint_support)
            evidence_quantity = "expected_joint_support_under_baseline"
        else:
            raise ValueError("Frozen fit direction must be positive or negative")
        feature_support_gate = bool(row["evaluation_support_gate"])
        evidence_gate = evidence_value >= evidence_threshold
        strict_gate = feature_support_gate and evidence_gate
        evaluation_lift = float(row["evaluation_log2_lift"])
        direction_replicated = strict_gate and (
            (evaluation_lift > 0 and direction == "positive")
            or (evaluation_lift < 0 and direction == "negative")
        )
        row["evaluation_feature_support_gate"] = feature_support_gate
        row["evaluation_baseline_rate"] = float(baseline[class_index])
        row["evaluation_expected_joint_support"] = expected_joint
        row["evaluation_evidence_quantity"] = evidence_quantity
        row["evaluation_evidence_value"] = evidence_value
        row["evaluation_evidence_threshold"] = evidence_threshold
        row["evaluation_evidence_gate"] = evidence_gate
        row["evaluation_support_gate"] = strict_gate
        row["direction_replicated"] = direction_replicated
        supported += int(strict_gate)
        replicated += int(direction_replicated)
    report["schema_version"] = "bt4-sparse-feature-semantic-heldout-evidence-gated-v2"
    report["supported_pair_count"] = supported
    report["direction_replicated_count"] = replicated
    report["direction_replication_rate_supported"] = (
        replicated / supported if supported else None
    )
    report["minimum_evaluation_support"] = minimum_feature_support
    report["minimum_positive_joint_support"] = minimum_positive_joint_support
    report["minimum_negative_expected_joint_support"] = (
        minimum_negative_expected_joint_support
    )
    return report


def _evaluation_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key.startswith("evaluation_") or key == "direction_replicated"
    }


def _evaluation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("pairs", [])
    supported = [row for row in rows if row["evaluation_support_gate"]]
    positive = sum(row["evaluation_log2_lift"] > 0 for row in supported)
    negative = sum(row["evaluation_log2_lift"] < 0 for row in supported)
    zero = sum(row["evaluation_log2_lift"] == 0 for row in supported)
    return {
        "selected_pair_count": report["selected_pair_count"],
        "supported_pair_count": report["supported_pair_count"],
        "unsupported_pair_count": report["selected_pair_count"]
        - report["supported_pair_count"],
        "direction_replicated_count": report["direction_replicated_count"],
        "direction_nonreplicated_supported_count": report["supported_pair_count"]
        - report["direction_replicated_count"],
        "direction_replication_rate_supported": report[
            "direction_replication_rate_supported"
        ],
        "supported_positive_lift_count": positive,
        "supported_negative_lift_count": negative,
        "supported_zero_lift_count": zero,
        "minimum_evaluation_support": report["minimum_evaluation_support"],
        "minimum_positive_joint_support": report["minimum_positive_joint_support"],
        "minimum_negative_expected_joint_support": report[
            "minimum_negative_expected_joint_support"
        ],
        "activity": report["activity"],
    }


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 2 or len(first) != len(second):
        return None
    x = torch.tensor(first, dtype=torch.float64)
    y = torch.tensor(second, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    if denominator == 0:
        return None
    return float(torch.dot(x, y) / denominator)


def combine_frozen_evaluations(
    selection: Sequence[Mapping[str, Any]],
    raw_report: Mapping[str, Any],
    hero_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair heldout Raw/Hero results without ever reselecting on either."""

    raw_rows = {_pair_id(row): row for row in raw_report["pairs"]}
    hero_rows = {_pair_id(row): row for row in hero_report["pairs"]}
    expected = [_pair_id(row) for row in selection]
    if len(expected) != len(set(expected)):
        raise ValueError("Frozen semantic selection contains duplicate pairs")
    if set(raw_rows) != set(expected) or set(hero_rows) != set(expected):
        raise ValueError("Heldout reports differ from the frozen Raw-fit selection")

    rows: list[dict[str, Any]] = []
    both_supported = 0
    direction_agreement = 0
    raw_lifts: list[float] = []
    hero_lifts: list[float] = []
    absolute_deltas: list[float] = []
    for frozen in selection:
        pair_id = _pair_id(frozen)
        raw = raw_rows[pair_id]
        hero = hero_rows[pair_id]
        raw_lift = float(raw["evaluation_log2_lift"])
        hero_lift = float(hero["evaluation_log2_lift"])
        supported = bool(
            raw["evaluation_support_gate"] and hero["evaluation_support_gate"]
        )
        agrees = supported and ((raw_lift >= 0) == (hero_lift >= 0))
        both_supported += int(supported)
        direction_agreement += int(agrees)
        if supported:
            raw_lifts.append(raw_lift)
            hero_lifts.append(hero_lift)
            absolute_deltas.append(abs(hero_lift - raw_lift))
        rows.append(
            {
                "pair_id": pair_id,
                **dict(frozen),
                "raw_heldout": _evaluation_fields(raw),
                "hero_heldout": _evaluation_fields(hero),
                "both_evaluation_support_gates": supported,
                "raw_hero_evaluation_direction_agreement": agrees,
                "hero_minus_raw_evaluation_log2_lift": hero_lift - raw_lift,
            }
        )
    return rows, {
        "both_supported_pair_count": both_supported,
        "raw_hero_direction_agreement_count": direction_agreement,
        "raw_hero_direction_agreement_rate_supported": (
            direction_agreement / both_supported if both_supported else None
        ),
        "raw_hero_heldout_lift_pearson_supported": _pearson(raw_lifts, hero_lifts),
        "mean_absolute_hero_minus_raw_lift_supported": (
            sum(absolute_deltas) / len(absolute_deltas) if absolute_deltas else None
        ),
        "interpretation": (
            "A descriptive model-diff summary on pairs frozen from Raw fit; agreement is "
            "not semantic equivalence and disagreement is an explicit negative result."
        ),
    }


def build_semantic_reports(
    *,
    raw_fit: SparseFeatureConceptAccumulator,
    raw_heldout: SparseFeatureConceptAccumulator,
    hero_heldout: SparseFeatureConceptAccumulator,
    top_per_direction_per_class: int,
    minimum_fit_support: int,
    minimum_positive_fit_joint_support: int,
    minimum_negative_fit_expected_joint_support: float,
    minimum_evaluation_support: int,
    minimum_positive_evaluation_joint_support: int,
    minimum_negative_evaluation_expected_joint_support: float,
    smoothing: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select on Raw fit and produce frozen heldout Raw/Hero reports."""

    selected = select_evidence_gated_feature_concepts(
        raw_fit,
        top_per_direction_per_class=top_per_direction_per_class,
        minimum_feature_support=minimum_fit_support,
        minimum_positive_joint_support=minimum_positive_fit_joint_support,
        minimum_negative_expected_joint_support=(
            minimum_negative_fit_expected_joint_support
        ),
        smoothing=smoothing,
    )
    frozen = [{"pair_id": _pair_id(row), **row} for row in selected]
    raw_report = evaluate_evidence_gated_feature_concepts(
        raw_heldout,
        frozen,
        minimum_feature_support=minimum_evaluation_support,
        minimum_positive_joint_support=minimum_positive_evaluation_joint_support,
        minimum_negative_expected_joint_support=(
            minimum_negative_evaluation_expected_joint_support
        ),
        smoothing=smoothing,
    )
    hero_report = evaluate_evidence_gated_feature_concepts(
        hero_heldout,
        frozen,
        minimum_feature_support=minimum_evaluation_support,
        minimum_positive_joint_support=minimum_positive_evaluation_joint_support,
        minimum_negative_expected_joint_support=(
            minimum_negative_evaluation_expected_joint_support
        ),
        smoothing=smoothing,
    )
    pairs, cross_model = combine_frozen_evaluations(frozen, raw_report, hero_report)
    raw_summary = _evaluation_summary(raw_report)
    hero_summary = _evaluation_summary(hero_report)
    selected_negative = sum(row["fit_direction"] == "negative" for row in frozen)
    selected_positive = sum(row["fit_direction"] == "positive" for row in frozen)
    status = "complete_descriptive" if frozen else "complete_no_selected_pairs"
    metrics = {
        "schema_version": TOKEN_SEMANTICS_METRICS_SCHEMA,
        "status": status,
        "selection": {
            "source_model": "Raw",
            "source_partition": "group-disjoint fit development positions",
            "selected_pair_count": len(frozen),
            "selected_positive_fit_pair_count": selected_positive,
            "selected_negative_fit_pair_count": selected_negative,
            "separate_positive_negative_slots": True,
            "top_per_direction_per_class": top_per_direction_per_class,
            "minimum_fit_feature_support": minimum_fit_support,
            "minimum_positive_fit_joint_support": (
                minimum_positive_fit_joint_support
            ),
            "minimum_negative_fit_expected_joint_support": (
                minimum_negative_fit_expected_joint_support
            ),
            "minimum_positive_evaluation_joint_support": (
                minimum_positive_evaluation_joint_support
            ),
            "minimum_negative_evaluation_expected_joint_support": (
                minimum_negative_evaluation_expected_joint_support
            ),
            "smoothing": smoothing,
            "activity": raw_fit.activity_metrics(),
        },
        "raw_heldout": raw_summary,
        "hero_heldout": hero_summary,
        "raw_hero_model_diff": cross_model,
        "negative_and_null_results": {
            "no_pairs_met_raw_fit_selection_gate": len(frozen) == 0,
            "raw_heldout_unsupported_pair_count": raw_summary["unsupported_pair_count"],
            "raw_heldout_direction_nonreplicated_supported_count": raw_summary[
                "direction_nonreplicated_supported_count"
            ],
            "raw_heldout_zero_lift_supported_count": raw_summary[
                "supported_zero_lift_count"
            ],
            "hero_heldout_unsupported_pair_count": hero_summary[
                "unsupported_pair_count"
            ],
            "hero_heldout_direction_nonreplicated_supported_count": hero_summary[
                "direction_nonreplicated_supported_count"
            ],
            "hero_heldout_zero_lift_supported_count": hero_summary[
                "supported_zero_lift_count"
            ],
            "raw_hero_direction_disagreement_supported_count": (
                cross_model["both_supported_pair_count"]
                - cross_model["raw_hero_direction_agreement_count"]
            ),
            "null_fields_are_json_null_not_imputed": True,
        },
        "claim_gates": {
            "exact_token_level_descriptive_association_measured": bool(frozen),
            "raw_heldout_direction_replication_observed": (
                raw_summary["direction_replicated_count"] > 0
            ),
            "hero_heldout_direction_transfer_observed": (
                hero_summary["direction_replicated_count"] > 0
            ),
            "feature_causality_permitted": False,
            "raw_hero_semantic_equivalence_permitted": False,
            "internal_search_claim_permitted": False,
            "confirmatory_test_claim_permitted": False,
        },
        "interpretation_contract": {
            "exploratory": True,
            "descriptive_only": True,
            "selection_only_on_raw_fit": True,
            "frozen_selection_on_both_heldout_arms": True,
            "direction_appropriate_evidence_gates": True,
            "test_split_selected_or_evaluated": False,
            "multiple_testing_adjusted": False,
            "occurrence_log2_lift_estimand": (
                "class prevalence among active feature tokens divided by token baseline"
            ),
            "activation_log2_lift_unit_warning": (
                "Activation-weighted lift is descriptive and depends on the published "
                "LoRSA decoder normalization/amplitude units; it is not a calibrated "
                "probability or a causal effect."
            ),
            "causal_followup_required": (
                "ablation/insertion, specificity, dose-response, and matched random controls"
            ),
        },
    }
    selection_payload = {
        "schema_version": TOKEN_SEMANTICS_SELECTION_SCHEMA,
        "selection_source": "Raw fit only",
        "evaluation_reselection": False,
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    return metrics, selection_payload


def _git_record() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "git_available": False,
            "commit": None,
            "worktree_clean": None,
            "status_sha256": None,
        }
    return {
        "git_available": True,
        "commit": commit,
        "worktree_clean": not status,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        }
    return result


def _source_file_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("feature_semantics.py"),
        Path(__file__).with_name("lorsa.py"),
        _REPO_ROOT / "research/interpretability/chess_concepts.py",
        _REPO_ROOT / "research/interpretability/chessbench.py",
        _REPO_ROOT / "research/interpretability/models.py",
        _REPO_ROOT / "research/train_torch.py",
    )
    return {
        path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path) for path in paths
    }


def _positions_payload(
    puzzles: Mapping[str, np.ndarray], selection: DevelopmentSelection
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for role, selected in (
        ("raw_fit", selection.fit_rows),
        ("heldout_raw_and_hero", selection.heldout_rows),
    ):
        for row in selected:
            rows.append(
                {
                    "role": role,
                    "position_id": str(puzzles["position_id"][row]),
                    "group_id": str(puzzles["group_id"][row]),
                    "public_split": "development",
                }
            )
    return {
        "schema_version": TOKEN_SEMANTICS_POSITIONS_SCHEMA,
        "split": selection.metadata,
        "positions": rows,
        "planes_persisted": False,
        "concept_labels_persisted": False,
        "test_positions": [],
    }


def _output_staging(output: Path) -> tuple[Path, Path]:
    target = output.resolve()
    if target in {Path("/"), Path.home().resolve(), _REPO_ROOT.resolve()}:
        raise ValueError(f"Unsafe output path: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    staging.mkdir()
    return target, staging


def _assert_storage_neutral(directory: Path) -> None:
    forbidden = [
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_FEATURE_SUFFIXES
    ]
    if forbidden:
        raise ValueError("Semantic run persisted a forbidden tensor file: " + ", ".join(forbidden))


def seal_semantic_run(
    *,
    output: Path,
    metrics: Mapping[str, Any],
    selection: Mapping[str, Any],
    positions: Mapping[str, Any],
    manifest: Mapping[str, Any],
    logs: Sequence[str],
) -> dict[str, Any]:
    target, staging = _output_staging(output)
    try:
        write_json_atomic(staging / "metrics.json", dict(metrics))
        write_json_atomic(staging / "selection.json", dict(selection))
        write_json_atomic(staging / "positions.json", dict(positions))
        write_json_atomic(staging / "manifest.json", dict(manifest))
        write_text_atomic(staging / "run.log", "\n".join(logs) + "\n")
        _assert_storage_neutral(staging)
        write_checksums(staging)
        verify_checksums(staging)
        os.replace(staging, target)
        checksums = verify_checksums(target)
        _assert_storage_neutral(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    retained = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    return {
        "output": str(target),
        "run_id": manifest["run_id"],
        "status": metrics["status"],
        "selected_pair_count": selection["pair_count"],
        "retained_bytes": retained,
        "checksums": checksums,
    }


def _runtime_identity_gate(
    models: BT4ComparisonModels,
    *,
    expected: Mapping[str, str],
    transfer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = models.descriptor()
    source_descriptor = transfer_manifest.get("model_lattice", {})
    criteria = {
        "raw_runtime_identity_matches": descriptor["raw_asset_sha256"]
        == expected["raw"],
        "hero_runtime_identity_matches": descriptor["hero_state_sha256"]
        == expected["hero"],
        "encoder_layout_matches_source_transfer": descriptor["encoder_layout_sha256"]
        == source_descriptor.get("encoder_layout_sha256"),
        "no_mixed_checkpoint_written": descriptor["on_disk_mixed_checkpoints"] is False,
    }
    return {"passed": all(criteria.values()), "criteria": criteria, "descriptor": descriptor}


def run_token_semantics(args: argparse.Namespace) -> dict[str, Any]:
    """Run the exact local Raw-fit -> heldout Raw/Hero semantic pilot."""

    if args.batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("batch_size and progress_every must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.provider_dollars != 0:
        raise ValueError("This local semantic pilot requires provider_dollars=0")
    if args.output.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {args.output.resolve()}")
    torch.set_num_threads(args.cpu_threads)
    concepts = exact_token_concepts()
    logs: list[str] = []

    def log(message: str) -> None:
        timestamped = f"{datetime.now(UTC).isoformat()} {message}"
        logs.append(timestamped)
        print(timestamped, flush=True)

    started_utc = datetime.now(UTC)
    started = time.monotonic()
    log("loading and verifying public-v2 corpus")
    corpora, corpus_manifest = load_public_corpus(args.corpus_manifest)
    puzzles = corpora["puzzles"]
    identities = local_source_identities(
        corpus_manifest=corpus_manifest,
        raw_bt4=args.raw_bt4,
        hero_checkpoint=args.hero_checkpoint,
        lorsa_manifest=args.lorsa_manifest,
        lorsa_weights=args.lorsa_weights,
    )
    source_gate = validate_sealed_source_transfer(
        args.source_transfer,
        expected_identities=identities,
        puzzles=puzzles,
    )
    if not source_gate["passed"]:
        failed = [name for name, passed in source_gate["criteria"].items() if not passed]
        raise RuntimeError("Sealed source-transfer gate failed: " + ", ".join(failed))
    log(f"source-transfer gate passed: {source_gate['source_transfer_run_id']}")

    split = select_group_disjoint_development_rows(
        puzzles,
        position_count=args.position_count,
        fit_fraction=args.fit_fraction,
        seed=args.seed,
    )
    positions_payload = _positions_payload(puzzles, split)
    log(
        "selected group-disjoint development positions: "
        f"fit={split.fit_rows.size}, heldout={split.heldout_rows.size}"
    )

    loaded_lorsa = load_converted_lorsa(args.lorsa_manifest, args.lorsa_weights)
    models = load_bt4_comparison_models(
        raw_bt4_path=args.raw_bt4,
        hero_checkpoint_dir=args.hero_checkpoint,
        device=device,
    )
    loaded_lorsa.module.to(device).eval().requires_grad_(False)
    models.raw_encoder.requires_grad_(False)
    models.hero_encoder.requires_grad_(False)
    transfer_manifest = _load_json_object(
        args.source_transfer / "manifest.json", label="Transfer manifest"
    )
    runtime_gate = _runtime_identity_gate(
        models,
        expected=identities,
        transfer_manifest=transfer_manifest,
    )
    if not runtime_gate["passed"]:
        failed = [name for name, passed in runtime_gate["criteria"].items() if not passed]
        raise RuntimeError("Runtime model identity gate failed: " + ", ".join(failed))
    if loaded_lorsa.manifest["manifest_sha256"] != identities["lorsa_manifest"]:
        raise RuntimeError("Loaded LoRSA manifest identity drift")
    if loaded_lorsa.manifest["weights_sha256"] != identities["lorsa_weights"]:
        raise RuntimeError("Loaded LoRSA weights identity drift")
    n_features = loaded_lorsa.module.config.n_ov_heads
    if n_features != 16_384 or loaded_lorsa.module.config.n_ctx != 64:
        raise RuntimeError("Reviewed LoRSA token/feature ABI drift")
    specs = tuple(binding.spec for binding in concepts)
    raw_fit = SparseFeatureConceptAccumulator(n_features, specs)
    raw_heldout = SparseFeatureConceptAccumulator(n_features, specs)
    hero_heldout = SparseFeatureConceptAccumulator(n_features, specs)

    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    weights_before = sha256_file(args.lorsa_weights)
    try:
        raw_fit_stream = _stream_arm(
            models=models,
            loaded_lorsa=loaded_lorsa,
            puzzles=puzzles,
            rows=split.fit_rows,
            accumulator=raw_fit,
            concepts=concepts,
            arm="RR",
            layer=args.layer,
            batch_size=args.batch_size,
            device=device,
            progress_every=args.progress_every,
            log=log,
        )
        raw_heldout_stream = _stream_arm(
            models=models,
            loaded_lorsa=loaded_lorsa,
            puzzles=puzzles,
            rows=split.heldout_rows,
            accumulator=raw_heldout,
            concepts=concepts,
            arm="RR",
            layer=args.layer,
            batch_size=args.batch_size,
            device=device,
            progress_every=args.progress_every,
            log=log,
        )
        hero_heldout_stream = _stream_arm(
            models=models,
            loaded_lorsa=loaded_lorsa,
            puzzles=puzzles,
            rows=split.heldout_rows,
            accumulator=hero_heldout,
            concepts=concepts,
            arm="HH",
            layer=args.layer,
            batch_size=args.batch_size,
            device=device,
            progress_every=args.progress_every,
            log=log,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32

    weights_after = sha256_file(args.lorsa_weights)
    if weights_after != weights_before:
        raise RuntimeError("LoRSA weights changed during frozen semantic evaluation")
    metrics, selection_payload = build_semantic_reports(
        raw_fit=raw_fit,
        raw_heldout=raw_heldout,
        hero_heldout=hero_heldout,
        top_per_direction_per_class=args.top_per_direction_per_class,
        minimum_fit_support=args.minimum_fit_support,
        minimum_positive_fit_joint_support=args.minimum_positive_fit_joint_support,
        minimum_negative_fit_expected_joint_support=(
            args.minimum_negative_fit_expected_joint_support
        ),
        minimum_evaluation_support=args.minimum_evaluation_support,
        minimum_positive_evaluation_joint_support=(
            args.minimum_positive_evaluation_joint_support
        ),
        minimum_negative_evaluation_expected_joint_support=(
            args.minimum_negative_evaluation_expected_joint_support
        ),
        smoothing=args.smoothing,
    )
    metrics["layer"] = args.layer
    metrics["position_split"] = split.metadata
    metrics["concept_contract"] = concept_contract(concepts)
    metrics["source_transfer_gate"] = source_gate
    metrics["runtime_model_identity_gate"] = runtime_gate
    metrics["streaming"] = {
        "raw_fit": raw_fit_stream,
        "raw_heldout": raw_heldout_stream,
        "hero_heldout": hero_heldout_stream,
        "dense_feature_batches_immediately_discarded": True,
        "features_persisted": False,
        "only_cpu_sufficient_statistics_retained": True,
    }
    metrics["claim_gates"]["source_transfer_artifact_passed"] = True
    metrics["claim_gates"]["runtime_model_identities_match"] = True
    metrics["claim_gates"]["test_split_selected_or_evaluated"] = False

    elapsed = time.monotonic() - started
    resource_usage: dict[str, Any] = {"wall_seconds": elapsed}
    if device.type == "cuda":
        resource_usage.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    source_files = _source_file_hashes()
    payloads = {
        "metrics": content_identity(metrics),
        "selection": content_identity(selection_payload),
        "positions": content_identity(positions_payload),
        "source_transfer_run": source_gate["source_transfer_run_id"],
        "source_transfer_checksums": source_gate["source_transfer_checksum_identity"],
        "corpus": identities["corpus_integrity"],
        "lorsa": identities["lorsa_weights"],
        "raw": identities["raw"],
        "hero": identities["hero"],
        "source_files": content_identity(source_files),
    }
    run_id = content_identity(payloads)
    manifest = {
        "schema_version": TOKEN_SEMANTICS_RUN_SCHEMA,
        "run_id": f"sha256:{run_id}",
        "started_utc": started_utc.isoformat(),
        "elapsed_seconds": elapsed,
        "git": _git_record(),
        "environment": _environment(device),
        "resource_usage": resource_usage,
        "experiment": {
            "layer": args.layer,
            "position_count": args.position_count,
            "fit_fraction": args.fit_fraction,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "compute_dtype": "float32",
            "autocast": False,
            "tf32": False,
            "exploratory": True,
            "confirmatory": False,
        },
        "cost": {
            "backend": args.backend,
            "provider_dollars": args.provider_dollars,
        },
        "storage_contract": {
            "features_persisted": False,
            "feature_summaries_persisted": False,
            "planes_persisted": False,
            "concept_label_tensors_persisted": False,
            "cpu_sufficient_statistics_persisted": False,
            "retained_files": [
                "metrics.json",
                "selection.json",
                "positions.json",
                "manifest.json",
                "run.log",
                "checksums.sha256",
            ],
            "forbidden_tensor_suffixes": sorted(FORBIDDEN_FEATURE_SUFFIXES),
        },
        "test_split_contract": {
            "corpus_container_contains_reserved_test_rows": True,
            "test_rows_selected": 0,
            "test_concepts_evaluated": False,
            "test_confirmatory_claims_permitted": False,
        },
        "source_transfer_gate": source_gate,
        "runtime_model_identity_gate": runtime_gate,
        "source_identities": identities,
        "lorsa": {
            "manifest_sha256": loaded_lorsa.manifest["manifest_sha256"],
            "weights_sha256_before": weights_before,
            "weights_sha256_after": weights_after,
            "parameters_frozen": not any(
                parameter.requires_grad for parameter in loaded_lorsa.module.parameters()
            ),
        },
        "payload_identities": payloads,
        "source_files": source_files,
    }
    logs.append(f"run_id sha256:{run_id}")
    result = seal_semantic_run(
        output=args.output,
        metrics=metrics,
        selection=selection_payload,
        positions=positions_payload,
        manifest=manifest,
        logs=logs,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bt4", type=Path, default=DEFAULT_RAW_BT4)
    parser.add_argument(
        "--hero-checkpoint", type=Path, default=DEFAULT_HERO_CHECKPOINT
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_DIR / "manifest.json",
    )
    parser.add_argument("--lorsa-manifest", type=Path, default=DEFAULT_LORSA_MANIFEST)
    parser.add_argument("--lorsa-weights", type=Path, default=DEFAULT_LORSA_WEIGHTS)
    parser.add_argument("--source-transfer", type=Path, default=DEFAULT_SOURCE_TRANSFER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--position-count", type=int, default=512)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--top-per-direction-per-class", type=int, default=5)
    parser.add_argument("--minimum-fit-support", type=int, default=20)
    parser.add_argument("--minimum-positive-fit-joint-support", type=int, default=5)
    parser.add_argument(
        "--minimum-negative-fit-expected-joint-support", type=float, default=5.0
    )
    parser.add_argument("--minimum-evaluation-support", type=int, default=10)
    parser.add_argument(
        "--minimum-positive-evaluation-joint-support", type=int, default=2
    )
    parser.add_argument(
        "--minimum-negative-evaluation-expected-joint-support",
        type=float,
        default=2.0,
    )
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument(
        "--backend", default="local GTX 1660 Ti zero-provider-cost semantic run"
    )
    parser.add_argument("--provider-dollars", type=float, default=0.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        corpora, corpus_manifest = load_public_corpus(args.corpus_manifest)
        identities = local_source_identities(
            corpus_manifest=corpus_manifest,
            raw_bt4=args.raw_bt4,
            hero_checkpoint=args.hero_checkpoint,
            lorsa_manifest=args.lorsa_manifest,
            lorsa_weights=args.lorsa_weights,
        )
        gate = validate_sealed_source_transfer(
            args.source_transfer,
            expected_identities=identities,
            puzzles=corpora["puzzles"],
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0 if gate["passed"] else 1
    result = run_token_semantics(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
