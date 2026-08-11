"""Exploratory causal validation of published LoRSA features on Raw and Hero BT4.

This runner consumes the immutable exact-token semantic study, freezes a small
post-hoc feature shortlist, calibrates matched random controls on Raw-fit only,
and evaluates decoder-contribution ablations and concept-token insertions on the
group-disjoint held-out development positions.  It never opens the reserved test
split, modifies a checkpoint, or persists dense activations.

The intervention target is the published LoRSA surrogate at layer 14.  Results
therefore establish causal effects inside that surrogate replacement path; they
do not by themselves prove that the native dense BT4 attention branch contains
the same causal variable.
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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from research.interpretability.artifacts import (
    canonical_json_bytes,
    content_identity,
    sha256_bytes,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from research.interpretability.chessbench import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_DIR,
    load_public_corpus,
)
from research.interpretability.models import BT4ComparisonModels, load_bt4_comparison_models
from research.interpretability.patching import legal_log_probabilities
from research.interpretability.sparse.lorsa import (
    LoadedConvertedLoRSA,
    load_converted_lorsa,
)
from research.interpretability.sparse.lorsa_token_semantics import (
    TOKEN_SEMANTICS_METRICS_SCHEMA,
    TOKEN_SEMANTICS_POSITIONS_SCHEMA,
    TOKEN_SEMANTICS_RUN_SCHEMA,
    TOKEN_SEMANTICS_SELECTION_SCHEMA,
    exact_token_concepts,
    labels_for_rows,
    local_source_identities,
    validate_sealed_source_transfer,
)
from research.interpretability.sparse.transfer_pilot import policy_replacement_metrics
from research.train_torch import BT4Encoder


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
DEFAULT_SEMANTIC_SOURCE = (
    _REPO_ROOT / "research/analysis/published_lorsa_l14_token_semantics_dev512_v3"
)
DEFAULT_OUTPUT = (
    _REPO_ROOT / "research/analysis/published_lorsa_l14_causal_dev256_v2"
)

CAUSAL_RUN_SCHEMA = "bt4-lorsa-causal-validation-run-v2"
CAUSAL_METRICS_SCHEMA = "bt4-lorsa-causal-validation-metrics-v2"
CAUSAL_DESIGN_SCHEMA = "bt4-lorsa-causal-validation-design-v2"
CAUSAL_POSITIONS_SCHEMA = "bt4-lorsa-causal-validation-positions-v2"
CAUSAL_RECORD_SCHEMA = "bt4-lorsa-causal-validation-record-v2"
FORBIDDEN_TENSOR_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".safetensors"}
TARGET_SCALE_DOSES = (0.0, 0.5, 1.5, 2.0)
TARGET_INJECTION_DOSES = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class CausalTarget:
    pair_id: str
    role: str
    rationale: str


# This shortlist is intentionally small and frozen in source before intervention
# outcomes are observed.  It is post-hoc with respect to the descriptive study.
CAUSAL_TARGETS: tuple[CausalTarget, ...] = (
    CausalTarget(
        "piece_code:2:feature_10843",
        "strong_transfer",
        "high-lift own-knight feature replicated in Raw and Hero",
    ),
    CausalTarget(
        "piece_code:5:feature_5429",
        "strong_transfer",
        "high-lift own-queen feature replicated in Raw and Hero",
    ),
    CausalTarget(
        "attacked_undefended_ours:1:feature_10784",
        "strong_transfer_polysemantic",
        "tactical feature replicated in both models and also associated with own queen",
    ),
    CausalTarget(
        "pinned_ours:1:feature_6661",
        "model_difference",
        "Raw-fit negative association reverses sign in Hero held-out",
    ),
    CausalTarget(
        "attack_count_theirs:2:feature_4516",
        "model_difference",
        "Raw held-out fails while Hero shows positive attack-count association",
    ),
    CausalTarget(
        "pinned_theirs:0:feature_5851",
        "null_nonreplication",
        "near-zero descriptive effect retained as a negative control",
    ),
)


@dataclass
class CalibrationAccumulator:
    """Bounded Raw-fit sufficient statistics for feature matching."""

    support_tokens: np.ndarray
    support_positions: np.ndarray
    activation_sum: np.ndarray
    activation_square_sum: np.ndarray
    target_position_coactivity: np.ndarray
    position_count: int = 0

    @classmethod
    def create(cls, n_features: int, target_features: Sequence[int]) -> CalibrationAccumulator:
        if n_features <= 0 or not target_features:
            raise ValueError("Calibration dimensions must be positive")
        return cls(
            support_tokens=np.zeros(n_features, dtype=np.int64),
            support_positions=np.zeros(n_features, dtype=np.int64),
            activation_sum=np.zeros(n_features, dtype=np.float64),
            activation_square_sum=np.zeros(n_features, dtype=np.float64),
            target_position_coactivity=np.zeros(
                (len(target_features), n_features), dtype=np.int64
            ),
        )

    def update(self, features: Tensor, target_features: Sequence[int]) -> None:
        if features.ndim != 3 or features.shape[-1] != self.support_tokens.size:
            raise ValueError("Calibration features must be [batch, token, feature]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Calibration features contain nonfinite values")
        positive = features.detach().float().clamp_min(0)
        active = positive > 0
        active_positions = active.any(dim=1)
        self.support_tokens += active.sum(dim=(0, 1)).cpu().numpy().astype(np.int64)
        self.support_positions += active_positions.sum(dim=0).cpu().numpy().astype(np.int64)
        self.activation_sum += positive.sum(dim=(0, 1)).cpu().double().numpy()
        self.activation_square_sum += positive.square().sum(dim=(0, 1)).cpu().double().numpy()
        for target_offset, feature_index in enumerate(target_features):
            target_active = active_positions[:, feature_index]
            if bool(target_active.any()):
                coactivity = active_positions[target_active].sum(dim=0)
                self.target_position_coactivity[target_offset] += (
                    coactivity.cpu().numpy().astype(np.int64)
                )
        self.position_count += int(features.shape[0])

    def feature_record(self, feature_index: int, decoder_norm: np.ndarray) -> dict[str, Any]:
        support = int(self.support_tokens[feature_index])
        position_support = int(self.support_positions[feature_index])
        mean = float(self.activation_sum[feature_index] / max(support, 1))
        rms = float(math.sqrt(self.activation_square_sum[feature_index] / max(support, 1)))
        return {
            "feature_index": int(feature_index),
            "raw_fit_active_token_count": support,
            "raw_fit_active_position_count": position_support,
            "raw_fit_mean_positive_activation": mean,
            "raw_fit_rms_positive_activation": rms,
            "injection_reference_amplitude": mean,
            "decoder_norm": float(decoder_norm[feature_index]),
        }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_sealed_semantic_source(
    semantic_dir: Path,
    *,
    expected_identities: Mapping[str, str],
    puzzles: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Verify the immutable descriptive dependency without requiring current source equality."""

    root = semantic_dir.resolve(strict=True)
    checksums = verify_checksums(root)
    manifest = _load_json_object(root / "manifest.json", label="Semantic manifest")
    metrics = _load_json_object(root / "metrics.json", label="Semantic metrics")
    selection = _load_json_object(root / "selection.json", label="Semantic selection")
    positions = _load_json_object(root / "positions.json", label="Semantic positions")
    payloads = manifest.get("payload_identities", {})
    source_files = manifest.get("source_files", {})
    source_identities = manifest.get("source_identities", {})
    if not all(
        isinstance(value, Mapping)
        for value in (payloads, source_files, source_identities)
    ):
        raise ValueError("Semantic source has malformed provenance mappings")

    position_rows = positions.get("positions", [])
    pair_rows = selection.get("pairs", [])
    if not isinstance(position_rows, list) or not isinstance(pair_rows, list):
        raise ValueError("Semantic source has malformed position or selection rows")
    fit = [row for row in position_rows if row.get("role") == "raw_fit"]
    heldout = [
        row for row in position_rows if row.get("role") == "heldout_raw_and_hero"
    ]
    all_ids = [str(row.get("position_id", "")) for row in position_rows]
    fit_groups = {str(row.get("group_id", "")) for row in fit}
    heldout_groups = {str(row.get("group_id", "")) for row in heldout}
    source_files_well_formed = bool(source_files) and all(
        isinstance(path, str)
        and bool(path)
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for path, digest in source_files.items()
    )

    corpus_rows_match = True
    if puzzles is not None:
        split = np.asarray(puzzles.get("split_u8"))
        corpus_positions = np.asarray(puzzles.get("position_id"))
        corpus_groups = np.asarray(puzzles.get("group_id"))
        if split.ndim != 1 or corpus_positions.shape != split.shape or corpus_groups.shape != split.shape:
            raise ValueError("Public corpus identity arrays are malformed")
        lookup = {
            str(corpus_positions[index]): (int(split[index]), str(corpus_groups[index]))
            for index in range(split.size)
        }
        for row in position_rows:
            identity = str(row.get("position_id", ""))
            corpus_rows_match &= identity in lookup
            if identity in lookup:
                public_split, group_id = lookup[identity]
                corpus_rows_match &= public_split == 1
                corpus_rows_match &= group_id == str(row.get("group_id", ""))

    selected_pairs = {str(row.get("pair_id", "")): row for row in pair_rows}
    target_pairs_exact = all(target.pair_id in selected_pairs for target in CAUSAL_TARGETS)
    criteria = {
        "checksum_inventory_exact": set(checksums)
        == {"manifest.json", "metrics.json", "positions.json", "run.log", "selection.json"},
        "run_schema_exact": manifest.get("schema_version") == TOKEN_SEMANTICS_RUN_SCHEMA,
        "metrics_schema_exact": metrics.get("schema_version")
        == TOKEN_SEMANTICS_METRICS_SCHEMA,
        "selection_schema_exact": selection.get("schema_version")
        == TOKEN_SEMANTICS_SELECTION_SCHEMA,
        "positions_schema_exact": positions.get("schema_version")
        == TOKEN_SEMANTICS_POSITIONS_SCHEMA,
        "metrics_identity_matches": payloads.get("metrics") == content_identity(metrics),
        "selection_identity_matches": payloads.get("selection")
        == content_identity(selection),
        "positions_identity_matches": payloads.get("positions")
        == content_identity(positions),
        "source_files_bound_and_well_formed": source_files_well_formed
        and payloads.get("source_files") == content_identity(dict(source_files)),
        "run_identity_matches": manifest.get("run_id")
        == f"sha256:{content_identity(dict(payloads))}",
        "source_identities_match_local_assets": dict(source_identities)
        == dict(expected_identities),
        "descriptive_run_complete": metrics.get("status") == "complete_descriptive",
        "selection_raw_fit_only": selection.get("selection_source") == "Raw fit only"
        and selection.get("evaluation_reselection") is False,
        "source_and_runtime_gates_passed": metrics.get("claim_gates", {}).get(
            "source_transfer_artifact_passed"
        )
        is True
        and metrics.get("claim_gates", {}).get("runtime_model_identities_match") is True,
        "test_split_unopened": metrics.get("claim_gates", {}).get(
            "test_split_selected_or_evaluated"
        )
        is False
        and positions.get("test_positions") == [],
        "position_count_exact": len(position_rows) == 512
        and len(fit) == 256
        and len(heldout) == 256,
        "positions_unique_and_development_only": len(all_ids) == len(set(all_ids))
        and all(all_ids)
        and all(row.get("public_split") == "development" for row in position_rows),
        "groups_disjoint": bool(fit_groups)
        and bool(heldout_groups)
        and fit_groups.isdisjoint(heldout_groups),
        "corpus_rows_match": corpus_rows_match,
        "causal_targets_present_exactly_once": target_pairs_exact
        and len(selected_pairs) == len(pair_rows),
    }
    return {
        "schema_version": "bt4-lorsa-semantic-source-verification-v1",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "semantic_run_id": manifest.get("run_id"),
        "semantic_checksum_identity": content_identity(checksums),
        "source_transfer_run_id": payloads.get("source_transfer_run"),
        "fit_position_count": len(fit),
        "heldout_position_count": len(heldout),
        "historical_source_files_identity": content_identity(dict(source_files))
        if source_files_well_formed
        else None,
        "current_source_equality_required": False,
        "reason_current_source_equality_not_required": (
            "the sealed run binds its historical source map; downstream code evolution is "
            "bound independently in the causal run"
        ),
        "failure_action": "abort before model loading or intervention",
    }


def choose_matched_random_controls(
    calibration: CalibrationAccumulator,
    decoder_norm: np.ndarray,
    *,
    target_features: Sequence[int],
    count: int,
    seed: int,
) -> dict[int, list[dict[str, Any]]]:
    """Choose seeded random controls from a nearest-neighbour matching pool."""

    if count <= 0:
        raise ValueError("Matched control count must be positive")
    feature_count = calibration.support_tokens.size
    if decoder_norm.shape != (feature_count,):
        raise ValueError("Decoder norms do not align with calibration features")
    excluded = set(int(index) for index in target_features)
    valid = (
        (calibration.support_tokens > 0)
        & (calibration.support_positions > 0)
        & (calibration.activation_sum > 0)
        & np.isfinite(decoder_norm)
        & (decoder_norm > 0)
    )
    for index in excluded:
        valid[index] = False
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < count:
        raise ValueError("Too few active features for matched controls")

    mean_activation = calibration.activation_sum / np.maximum(
        calibration.support_tokens, 1
    )
    records: dict[int, list[dict[str, Any]]] = {}
    for target_offset, target_index in enumerate(target_features):
        target_position_support = int(calibration.support_positions[target_index])
        intersections = calibration.target_position_coactivity[target_offset]
        unions = (
            target_position_support + calibration.support_positions - intersections
        )
        jaccard = intersections / np.maximum(unions, 1)
        log_support_delta = np.abs(
            np.log2((calibration.support_tokens + 1) / (calibration.support_tokens[target_index] + 1))
        )
        log_position_delta = np.abs(
            np.log2(
                (calibration.support_positions + 1)
                / (calibration.support_positions[target_index] + 1)
            )
        )
        log_amplitude_delta = np.abs(
            np.log2(
                np.maximum(mean_activation, 1e-12)
                / max(float(mean_activation[target_index]), 1e-12)
            )
        )
        log_decoder_delta = np.abs(
            np.log2(
                np.maximum(decoder_norm, 1e-12)
                / max(float(decoder_norm[target_index]), 1e-12)
            )
        )
        score = (
            log_support_delta
            + log_position_delta
            + log_amplitude_delta
            + log_decoder_delta
            + 2.0 * (1.0 - jaccard)
        )
        ordered = valid_indices[np.argsort(score[valid_indices], kind="stable")]
        pool_size = min(max(32, count), int(ordered.size))
        pool = ordered[:pool_size]
        digest = hashlib.sha256(f"{seed}:{target_index}:matched-controls".encode()).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        weights = np.exp(-(score[pool] - float(score[pool].min())))
        weights /= weights.sum()
        selected = generator.choice(pool, size=count, replace=False, p=weights)
        rows: list[dict[str, Any]] = []
        for index in selected:
            record = calibration.feature_record(int(index), decoder_norm)
            record.update(
                {
                    "match_score": float(score[index]),
                    "raw_fit_position_jaccard_with_target": float(jaccard[index]),
                    "target_feature_index": int(target_index),
                    "random_pool_size": pool_size,
                    "random_pool_rank": int(np.flatnonzero(ordered == index)[0]),
                }
            )
            rows.append(record)
        records[int(target_index)] = rows
    return records


def _dose_name(value: float) -> str:
    return str(value).replace(".", "p")


def build_decoder_interventions(
    baseline_reconstruction: Tensor,
    features: Tensor,
    decoder: Tensor,
    *,
    target_feature: int,
    controls: Sequence[int],
    injection_token_mask: Tensor,
    reference_amplitudes: Mapping[int, float],
) -> tuple[list[str], Tensor]:
    """Construct target interventions and energy/location-matched random directions."""

    if baseline_reconstruction.ndim != 3 or features.ndim != 3 or decoder.ndim != 2:
        raise ValueError("Intervention tensors have invalid ranks")
    batch, token_count, model_width = baseline_reconstruction.shape
    if features.shape[:2] != (batch, token_count) or decoder.shape[1] != model_width:
        raise ValueError("Intervention tensor shapes do not align")
    if injection_token_mask.shape != (batch, token_count):
        raise ValueError("Injection token mask does not align")
    indices = [int(target_feature), *(int(value) for value in controls)]
    if len(indices) != len(set(indices)) or any(
        index < 0 or index >= features.shape[-1] for index in indices
    ):
        raise ValueError("Target and control feature indices must be unique and in range")
    if set(indices) != set(reference_amplitudes):
        raise ValueError("Reference amplitudes must cover the target and controls exactly")

    baseline = baseline_reconstruction.float()
    feature_values = features.float()
    decoder_values = decoder.float()
    target_values = feature_values[..., target_feature]
    target_decoder = decoder_values[target_feature]
    target_decoder_norm = target_decoder.norm()
    if not bool(torch.isfinite(target_decoder_norm)) or float(target_decoder_norm) <= 0:
        raise ValueError("Target decoder direction has invalid norm")
    injection_mask = injection_token_mask.to(target_values.dtype)
    reference = float(reference_amplitudes[target_feature])
    names: list[str] = ["noop"]
    values: list[Tensor] = [baseline]

    def matched_direction(index: int) -> Tensor:
        direction = decoder_values[index]
        norm = direction.norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise ValueError("Matched-control decoder direction has invalid norm")
        return direction * (target_decoder_norm / norm)

    def target_scaled(scale: float) -> Tensor:
        contribution = target_values.unsqueeze(-1) * target_decoder
        return baseline + (scale - 1.0) * contribution

    def injection_delta(dose: float) -> Tensor:
        delta = (reference * dose - target_values).clamp_min(0)
        return delta * injection_mask

    def target_injected(dose: float) -> Tensor:
        return baseline + injection_delta(dose).unsqueeze(-1) * target_decoder

    for dose in TARGET_SCALE_DOSES:
        names.append(f"target_scale_{_dose_name(dose)}")
        values.append(target_scaled(dose))
    for index in controls:
        names.append(f"control_ablate_feature_{index}")
        values.append(
            baseline - target_values.unsqueeze(-1) * matched_direction(index)
        )
    for dose in TARGET_INJECTION_DOSES:
        names.append(f"target_inject_{_dose_name(dose)}")
        values.append(target_injected(dose))
    for index in controls:
        names.append(f"control_inject_feature_{index}")
        values.append(
            baseline + injection_delta(1.0).unsqueeze(-1) * matched_direction(index)
        )
    result = torch.stack(values)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Decoder interventions produced nonfinite values")
    return names, result


def last_layer_policy_logits(
    encoder: BT4Encoder,
    attention_inputs: Tensor,
    attention_outputs: Tensor,
    *,
    layer_index: int,
    compute_dtype: torch.dtype,
) -> Tensor:
    """Run the exact post-attention suffix for many layer-14 interventions."""

    if layer_index != len(encoder.layers) - 1:
        raise ValueError("Fast suffix evaluation is valid only at the final encoder layer")
    if attention_inputs.ndim != 3 or attention_outputs.ndim != 4:
        raise ValueError("Suffix inputs must be [batch, token, model] and [variant, ...]")
    variant_count, batch, token_count, width = attention_outputs.shape
    if attention_inputs.shape != (batch, token_count, width):
        raise ValueError("Suffix attention inputs and outputs do not align")
    layer = encoder.layers[layer_index]
    head = encoder.policy_head
    if head is None:
        raise ValueError("Causal validation requires the native policy head")
    states = attention_inputs.unsqueeze(0).expand(variant_count, -1, -1, -1)
    flat_states = states.reshape(variant_count * batch, token_count, width)
    flat_attention = attention_outputs.reshape(variant_count * batch, token_count, width)
    resid_mid = layer.ln_attn(
        flat_attention * encoder.alpha + flat_states,
        compute_dtype,
    )
    flat = resid_mid.reshape(variant_count * batch * token_count, width)
    mlp = layer.ffn2(F.mish(layer.ffn1(flat, compute_dtype)), compute_dtype)
    mlp = mlp.reshape(variant_count * batch, token_count, width)
    resid_post = layer.ln_ffn(mlp * encoder.alpha + resid_mid, compute_dtype)
    logits = head(resid_post, compute_dtype)
    return logits.reshape(variant_count, batch, -1)


def _policy_probabilities(
    logits: Tensor, legal: Tensor, legal_count: Tensor
) -> tuple[Tensor, Tensor]:
    logp, mask = legal_log_probabilities(logits, legal, legal_count)
    return logp.exp().masked_fill(~mask, 0.0), mask


def _projection(delta: Tensor, direction: Tensor, mask: Tensor) -> Tensor:
    delta = delta.masked_fill(~mask, 0.0).double()
    direction = direction.masked_fill(~mask, 0.0).double()
    denominator = direction.square().sum(dim=1)
    direction_tv = 0.5 * direction.abs().sum(dim=1)
    numerator = (delta * direction).sum(dim=1)
    estimable = (denominator > 1e-18) & (direction_tv >= 1e-7)
    return torch.where(estimable, numerator / denominator, torch.nan)


def _deterministic_injection_mask(
    concept_mask: Tensor,
    position_ids: Sequence[str],
    pair_id: str,
    seed: int,
) -> Tensor:
    if concept_mask.ndim != 2 or concept_mask.shape[0] != len(position_ids):
        raise ValueError("Concept mask and position identities do not align")
    result = torch.zeros_like(concept_mask, dtype=torch.bool)
    for offset, position_id in enumerate(position_ids):
        candidates = torch.nonzero(concept_mask[offset].cpu(), as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        digest = hashlib.sha256(
            f"{seed}:{pair_id}:{position_id}:injection-token".encode()
        ).digest()
        selected = int.from_bytes(digest[:8], "big") % int(candidates.numel())
        result[offset, int(candidates[selected])] = True
    return result


def _metric_row(metrics: Mapping[str, Tensor], offset: int) -> dict[str, Any]:
    return {
        "legal_policy_js": float(metrics["js_divergence"][offset]),
        "legal_policy_total_variation": float(metrics["total_variation"][offset]),
        "top1_preserved": bool(metrics["top1_agreement"][offset]),
        "target_move_log_odds_change": float(metrics["target_log_odds_change"][offset]),
    }


@torch.inference_mode()
def _evaluate_arm_batch(
    *,
    models: BT4ComparisonModels,
    loaded_lorsa: LoadedConvertedLoRSA,
    puzzles: Mapping[str, np.ndarray],
    rows: np.ndarray,
    target_rows: Mapping[str, Mapping[str, Any]],
    target_specs: Sequence[CausalTarget],
    control_plan: Mapping[int, Sequence[Mapping[str, Any]]],
    calibration_records: Mapping[int, Mapping[str, Any]],
    arm: str,
    layer: int,
    seed: int,
    device: torch.device,
    validate_suffix: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray], float]:
    planes = torch.from_numpy(np.asarray(puzzles["current_planes_u8"])[rows]).to(device)
    legal = torch.from_numpy(np.asarray(puzzles["legal_idx_u16"])[rows]).to(device)
    legal_count = torch.from_numpy(np.asarray(puzzles["legal_count_u16"])[rows]).to(device)
    targets = torch.from_numpy(np.asarray(puzzles["target_action_u16"])[rows]).to(device)
    position_ids = [str(np.asarray(puzzles["position_id"])[row]) for row in rows]
    group_ids = [str(np.asarray(puzzles["group_id"])[row]) for row in rows]
    labels = labels_for_rows(puzzles, rows, exact_token_concepts())

    dense = models.policy_logits_with_captures(
        planes,
        arm=arm,
        compute_dtype=torch.float32,
        capture_layers=(layer,),
    )
    if dense.captures is None:
        raise AssertionError("Causal runner lost the requested layer capture")
    states = dense.captures.hook_attn_in[0]
    checkpoint_dtype = next(loaded_lorsa.module.parameters()).dtype
    sparse = loaded_lorsa.module(states.to(dtype=checkpoint_dtype))
    baseline_reconstruction = sparse.reconstruction.to(dtype=torch.float32)
    encoder = models.raw_encoder if arm == "RR" else models.hero_encoder
    baseline_logits = last_layer_policy_logits(
        encoder,
        states,
        baseline_reconstruction.unsqueeze(0),
        layer_index=layer,
        compute_dtype=torch.float32,
    )[0]
    suffix_error = 0.0
    if validate_suffix:
        full = models.policy_logits_with_captures(
            planes,
            arm=arm,
            compute_dtype=torch.float32,
            capture_layers=(layer,),
            attention_output_overrides={layer: baseline_reconstruction},
        )
        suffix_error = float((full.logits - baseline_logits).abs().max())
        if suffix_error > 1e-5:
            raise RuntimeError(
                f"Fast final-layer suffix differs from full override by {suffix_error}"
            )

    native_to_lorsa = policy_replacement_metrics(
        dense.logits, baseline_logits, legal, legal_count, targets
    )
    records: list[dict[str, Any]] = []
    effect_vectors: dict[tuple[str, str], np.ndarray] = {}
    decoder = loaded_lorsa.module.W_O

    for target_spec in target_specs:
        target = target_rows[target_spec.pair_id]
        feature_index = int(target["feature_index"])
        control_rows = control_plan[feature_index]
        controls = [int(row["feature_index"]) for row in control_rows]
        concept_mask_cpu = labels[str(target["concept"])] == int(target["class_index"])
        injection_mask_cpu = _deterministic_injection_mask(
            concept_mask_cpu, position_ids, target_spec.pair_id, seed
        )
        injection_mask = injection_mask_cpu.to(device)
        reference_amplitudes = {
            feature_index: float(
                calibration_records[feature_index]["injection_reference_amplitude"]
            )
        }
        reference_amplitudes.update(
            {
                int(row["feature_index"]): float(row["injection_reference_amplitude"])
                for row in control_rows
            }
        )
        variant_names, reconstructions = build_decoder_interventions(
            baseline_reconstruction,
            sparse.features,
            decoder,
            target_feature=feature_index,
            controls=controls,
            injection_token_mask=injection_mask,
            reference_amplitudes=reference_amplitudes,
        )
        variant_logits = last_layer_policy_logits(
            encoder,
            states,
            reconstructions,
            layer_index=layer,
            compute_dtype=torch.float32,
        )
        if variant_names[0] != "noop":
            raise AssertionError("The in-batch numerical no-op must be the first variant")
        causal_baseline_logits = variant_logits[0]
        causal_baseline_probability, probability_mask = _policy_probabilities(
            causal_baseline_logits, legal, legal_count
        )
        noop_to_reference = policy_replacement_metrics(
            baseline_logits, causal_baseline_logits, legal, legal_count, targets
        )
        metric_vectors: dict[str, dict[str, Tensor]] = {}
        probability_vectors: dict[str, Tensor] = {}
        for variant_offset, name in enumerate(variant_names):
            logits = variant_logits[variant_offset]
            metric_vectors[name] = policy_replacement_metrics(
                causal_baseline_logits, logits, legal, legal_count, targets
            )
            probability, candidate_mask = _policy_probabilities(logits, legal, legal_count)
            if not torch.equal(candidate_mask, probability_mask):
                raise AssertionError("Legal support changed across interventions")
            probability_vectors[name] = probability

        ablation_direction = (
            probability_vectors["target_scale_0p0"] - causal_baseline_probability
        )
        injection_direction = (
            probability_vectors["target_inject_1p0"] - causal_baseline_probability
        )
        projections: dict[str, Tensor] = {}
        for name in (
            "target_scale_0p0",
            "target_scale_0p5",
            "target_scale_1p5",
            "target_scale_2p0",
        ):
            projections[name] = _projection(
                probability_vectors[name] - causal_baseline_probability,
                ablation_direction,
                probability_mask,
            )
        for name in (
            "target_inject_0p5",
            "target_inject_1p0",
            "target_inject_2p0",
        ):
            projections[name] = _projection(
                probability_vectors[name] - causal_baseline_probability,
                injection_direction,
                probability_mask,
            )

        feature_values = sparse.features[..., feature_index].float()
        feature_active = feature_values > 0
        concept_mask = concept_mask_cpu.to(device)
        ablation_delta = ablation_direction.masked_fill(~probability_mask, 0.0)
        for offset, position_id in enumerate(position_ids):
            variant_payload: dict[str, Any] = {}
            for name in variant_names:
                row = _metric_row(metric_vectors[name], offset)
                if name in projections:
                    projection = float(projections[name][offset])
                    row["dose_projection"] = projection if math.isfinite(projection) else None
                variant_payload[name] = row
            active_count = int(feature_active[offset].sum())
            active_on_concept = int(
                (feature_active[offset] & concept_mask[offset]).sum()
            )
            activation_mass = float(feature_values[offset].sum())
            on_concept_mass = float(
                (feature_values[offset] * concept_mask[offset]).sum()
            )
            record = {
                "schema_version": CAUSAL_RECORD_SCHEMA,
                "position_id": position_id,
                "group_id": group_ids[offset],
                "public_split": "development",
                "role": "heldout_raw_and_hero",
                "arm": arm,
                "pair_id": target_spec.pair_id,
                "target_role": target_spec.role,
                "concept": target["concept"],
                "class_index": int(target["class_index"]),
                "class_name": target["class_name"],
                "feature_index": feature_index,
                "concept_token_count": int(concept_mask[offset].sum()),
                "injection_opportunity": bool(injection_mask[offset].any()),
                "injection_token": (
                    int(torch.nonzero(injection_mask[offset], as_tuple=False).flatten()[0])
                    if bool(injection_mask[offset].any())
                    else None
                ),
                "feature_active_token_count": active_count,
                "feature_active_on_concept_token_count": active_on_concept,
                "feature_activation_mass": activation_mass,
                "feature_on_concept_activation_mass": on_concept_mass,
                "native_to_lorsa_baseline": _metric_row(native_to_lorsa, offset),
                "batched_noop_to_reference": _metric_row(noop_to_reference, offset),
                "interventions": variant_payload,
            }
            records.append(record)
            effect_vectors[(position_id, target_spec.pair_id)] = (
                ablation_delta[offset].detach().cpu().double().numpy()
            )

    return records, effect_vectors, suffix_error


def _finite(values: Sequence[float | None]) -> np.ndarray:
    result = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    return result[np.isfinite(result)]


def summarize_values(values: Sequence[float | None]) -> dict[str, Any]:
    array = _finite(values)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def paired_bootstrap_interval(
    differences: Sequence[float], *, samples: int, seed: int
) -> dict[str, Any]:
    values = _finite(differences)
    if values.size == 0 or samples <= 0:
        raise ValueError("Paired bootstrap requires finite differences and positive samples")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 1_000
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return {
        "count": int(values.size),
        "mean_difference": float(values.mean()),
        "bootstrap_samples": samples,
        "confidence": 0.95,
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
    }


def paired_sign_flip_pvalue(
    differences: Sequence[float], *, samples: int, seed: int
) -> float:
    values = _finite(differences)
    if values.size == 0 or samples <= 0:
        raise ValueError("Sign-flip test requires finite differences and positive samples")
    observed = abs(float(values.mean()))
    generator = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    chunk = 1_000
    while completed < samples:
        count = min(chunk, samples - completed)
        signs = generator.integers(0, 2, size=(count, values.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        simulated = np.abs((signs * values).mean(axis=1))
        exceed += int((simulated >= observed).sum())
        completed += count
    return float((exceed + 1) / (samples + 1))


def _unavailable_paired_interval(*, samples: int) -> dict[str, Any]:
    return {
        "count": 0,
        "mean_difference": None,
        "bootstrap_samples": samples,
        "confidence": 0.95,
        "lower": None,
        "upper": None,
        "estimable": False,
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    if not pvalues or any(not 0 <= value <= 1 for value in pvalues.values()):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * pvalues[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _variant_values(
    rows: Sequence[Mapping[str, Any]], variant: str, metric: str
) -> list[float]:
    return [float(row["interventions"][variant][metric]) for row in rows]


def _control_mean(
    row: Mapping[str, Any], controls: Sequence[int], *, prefix: str, metric: str
) -> float:
    return float(
        np.mean(
            [
                float(row["interventions"][f"{prefix}_feature_{index}"][metric])
                for index in controls
            ]
        )
    )


def build_causal_summary(
    records: Sequence[Mapping[str, Any]],
    effect_vectors: Mapping[str, Mapping[tuple[str, str], np.ndarray]],
    *,
    target_specs: Sequence[CausalTarget],
    target_rows: Mapping[str, Mapping[str, Any]],
    control_plan: Mapping[int, Sequence[Mapping[str, Any]]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate causal outcomes without retaining logits or activations."""

    arms: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    row_lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in records:
        row_lookup[(str(row["arm"]), str(row["pair_id"]), str(row["position_id"]))] = row

    for arm in ("RR", "HH"):
        arm_summary: dict[str, Any] = {}
        for target_offset, spec in enumerate(target_specs):
            target = target_rows[spec.pair_id]
            feature_index = int(target["feature_index"])
            controls = [
                int(row["feature_index"]) for row in control_plan[feature_index]
            ]
            selected = [
                row
                for row in records
                if row["arm"] == arm and row["pair_id"] == spec.pair_id
            ]
            if not selected:
                raise ValueError(f"No causal records for {arm} {spec.pair_id}")
            active = [row for row in selected if int(row["feature_active_token_count"]) > 0]
            opportunities = [row for row in selected if row["injection_opportunity"]]
            target_ablation = _variant_values(
                active, "target_scale_0p0", "legal_policy_total_variation"
            )
            control_ablation = [
                _control_mean(
                    row,
                    controls,
                    prefix="control_ablate",
                    metric="legal_policy_total_variation",
                )
                for row in active
            ]
            ablation_difference = np.asarray(target_ablation) - np.asarray(control_ablation)
            target_injection = _variant_values(
                opportunities, "target_inject_1p0", "legal_policy_total_variation"
            )
            control_injection = [
                _control_mean(
                    row,
                    controls,
                    prefix="control_inject",
                    metric="legal_policy_total_variation",
                )
                for row in opportunities
            ]
            injection_difference = np.asarray(target_injection) - np.asarray(control_injection)
            test_key = f"{arm}:{spec.pair_id}:active_ablation_specificity"
            pvalue = (
                paired_sign_flip_pvalue(
                    ablation_difference,
                    samples=bootstrap_samples,
                    seed=seed
                    + 1009 * target_offset
                    + (0 if arm == "RR" else 100_000),
                )
                if ablation_difference.size
                else 1.0
            )
            raw_pvalues[test_key] = pvalue
            variants: dict[str, Any] = {}
            first_interventions = selected[0]["interventions"]
            for variant in sorted(first_interventions):
                variants[variant] = {
                    metric: summarize_values(_variant_values(selected, variant, metric))
                    for metric in (
                        "legal_policy_js",
                        "legal_policy_total_variation",
                        "target_move_log_odds_change",
                    )
                }
                projections = [
                    row["interventions"][variant].get("dose_projection")
                    for row in selected
                ]
                if any(value is not None for value in projections):
                    variants[variant]["dose_projection"] = summarize_values(projections)

            scale_projection_medians = {
                name: summarize_values(
                    [
                        row["interventions"][name].get("dose_projection")
                        for row in active
                    ]
                )["median"]
                for name in (
                    "target_scale_0p5",
                    "target_scale_1p5",
                    "target_scale_2p0",
                )
            }
            injection_projection_medians = {
                name: summarize_values(
                    [
                        row["interventions"][name].get("dose_projection")
                        for row in opportunities
                    ]
                )["median"]
                for name in (
                    "target_inject_0p5",
                    "target_inject_1p0",
                    "target_inject_2p0",
                )
            }
            scale_directional = (
                scale_projection_medians["target_scale_0p5"] is not None
                and scale_projection_medians["target_scale_1p5"] is not None
                and scale_projection_medians["target_scale_2p0"] is not None
                and scale_projection_medians["target_scale_0p5"] > 0
                and scale_projection_medians["target_scale_1p5"] < 0
                and scale_projection_medians["target_scale_2p0"]
                < scale_projection_medians["target_scale_1p5"]
            )
            injection_directional = (
                injection_projection_medians["target_inject_0p5"] is not None
                and injection_projection_medians["target_inject_1p0"] is not None
                and injection_projection_medians["target_inject_2p0"] is not None
                and 0 < injection_projection_medians["target_inject_0p5"]
                < injection_projection_medians["target_inject_1p0"]
                < injection_projection_medians["target_inject_2p0"]
            )
            arm_summary[spec.pair_id] = {
                "target": {**asdict(spec), **dict(target)},
                "position_count": len(selected),
                "natural_feature_active_position_count": len(active),
                "concept_injection_opportunity_position_count": len(opportunities),
                "native_to_lorsa_baseline": {
                    metric: summarize_values(
                        [row["native_to_lorsa_baseline"][metric] for row in selected]
                    )
                    for metric in ("legal_policy_js", "legal_policy_total_variation")
                },
                "variants": variants,
                "active_position_ablation_specificity": {
                    "estimand": "target ablation TV minus mean matched-control ablation TV",
                    "target": summarize_values(target_ablation),
                    "matched_control_mean": summarize_values(control_ablation),
                    "paired_difference": (
                        paired_bootstrap_interval(
                            ablation_difference,
                            samples=bootstrap_samples,
                            seed=seed
                            + 2003 * target_offset
                            + (0 if arm == "RR" else 200_000),
                        )
                        if ablation_difference.size
                        else _unavailable_paired_interval(samples=bootstrap_samples)
                    ),
                    "sign_flip_pvalue_unadjusted": pvalue,
                    "multiple_testing_family": "12 Raw/Hero target active-position ablation tests",
                },
                "concept_token_injection_specificity": {
                    "estimand": "target insertion TV minus mean matched-control insertion TV",
                    "target": summarize_values(target_injection),
                    "matched_control_mean": summarize_values(control_injection),
                    "paired_difference": (
                        paired_bootstrap_interval(
                            injection_difference,
                            samples=bootstrap_samples,
                            seed=seed
                            + 3001 * target_offset
                            + (0 if arm == "RR" else 300_000),
                        )
                        if injection_difference.size
                        else _unavailable_paired_interval(samples=bootstrap_samples)
                    ),
                    "multiple_testing_adjusted": False,
                },
                "dose_response": {
                    "summary_statistic": "median after a 1e-7 TV estimability floor",
                    "scale_projection_medians": scale_projection_medians,
                    "scale_directional": scale_directional,
                    "injection_projection_medians": injection_projection_medians,
                    "injection_directional": injection_directional,
                },
            }
        arms[arm] = arm_summary

    adjusted = holm_adjust(raw_pvalues)
    causal_pass_count = 0
    for arm, pairs in arms.items():
        for pair_id, result in pairs.items():
            test_key = f"{arm}:{pair_id}:active_ablation_specificity"
            specificity = result["active_position_ablation_specificity"]
            specificity["sign_flip_pvalue_holm"] = adjusted[test_key]
            interval = specificity["paired_difference"]
            gate = {
                "active_position_count_ge_10": result[
                    "natural_feature_active_position_count"
                ]
                >= 10,
                "specificity_mean_positive": interval["mean_difference"] is not None
                and interval["mean_difference"] > 0,
                "specificity_95pct_bootstrap_lower_gt_0": interval["lower"] is not None
                and interval["lower"] > 0,
                "holm_adjusted_p_lt_0p05": adjusted[test_key] < 0.05,
                "ablation_dose_directional": result["dose_response"][
                    "scale_directional"
                ],
            }
            gate["passed"] = all(gate.values())
            result["exploratory_causal_feature_policy_gate"] = gate
            causal_pass_count += int(gate["passed"])

    cross_model: dict[str, Any] = {}
    for target_offset, spec in enumerate(target_specs):
        raw_rows = {
            str(row["position_id"]): row
            for row in records
            if row["arm"] == "RR" and row["pair_id"] == spec.pair_id
        }
        hero_rows = {
            str(row["position_id"]): row
            for row in records
            if row["arm"] == "HH" and row["pair_id"] == spec.pair_id
        }
        shared = sorted(set(raw_rows) & set(hero_rows))
        if len(shared) != len(raw_rows) or len(shared) != len(hero_rows):
            raise ValueError("Raw/Hero causal positions do not align")
        raw_tv = np.asarray(
            [
                raw_rows[position]["interventions"]["target_scale_0p0"][
                    "legal_policy_total_variation"
                ]
                for position in shared
            ],
            dtype=np.float64,
        )
        hero_tv = np.asarray(
            [
                hero_rows[position]["interventions"]["target_scale_0p0"][
                    "legal_policy_total_variation"
                ]
                for position in shared
            ],
            dtype=np.float64,
        )
        shared_active = [
            position
            for position in shared
            if int(raw_rows[position]["feature_active_token_count"]) > 0
            or int(hero_rows[position]["feature_active_token_count"]) > 0
        ]
        active_indices = np.asarray(
            [shared.index(position) for position in shared_active], dtype=np.int64
        )
        active_difference = (hero_tv - raw_tv)[active_indices]
        cosines: list[float] = []
        for position in shared_active:
            raw_vector = effect_vectors["RR"][(position, spec.pair_id)]
            hero_vector = effect_vectors["HH"][(position, spec.pair_id)]
            denominator = float(np.linalg.norm(raw_vector) * np.linalg.norm(hero_vector))
            if denominator > 1e-16:
                cosines.append(float(np.dot(raw_vector, hero_vector) / denominator))
        cross_model[spec.pair_id] = {
            "position_count": len(shared),
            "active_union_position_count": len(shared_active),
            "hero_minus_raw_ablation_total_variation": paired_bootstrap_interval(
                hero_tv - raw_tv,
                samples=bootstrap_samples,
                seed=seed + 5003 * target_offset,
            ),
            "hero_minus_raw_ablation_total_variation_active_union": (
                paired_bootstrap_interval(
                    active_difference,
                    samples=bootstrap_samples,
                    seed=seed + 6007 * target_offset,
                )
                if active_difference.size
                else _unavailable_paired_interval(samples=bootstrap_samples)
            ),
            "raw_hero_policy_delta_cosine_active_union": summarize_values(cosines),
            "interpretation": (
                "effect magnitude and direction inside the same frozen LoRSA decoder; "
                "differences reflect encoder/head context, not retrained sparse features"
            ),
        }

    return {
        "arms": arms,
        "cross_model": cross_model,
        "multiple_testing": {
            "method": "Holm family-wise correction",
            "family_size": len(raw_pvalues),
            "unadjusted": raw_pvalues,
            "adjusted": adjusted,
        },
        "exploratory_causal_gate_pass_count": causal_pass_count,
        "exploratory_causal_gate_total": len(target_specs) * 2,
    }


def _source_file_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("lorsa.py"),
        Path(__file__).with_name("lorsa_token_semantics.py"),
        _REPO_ROOT / "research/interpretability/artifacts.py",
        _REPO_ROOT / "research/interpretability/chessbench.py",
        _REPO_ROOT / "research/interpretability/models.py",
        _REPO_ROOT / "research/interpretability/patching.py",
        _REPO_ROOT / "research/train_torch.py",
    )
    return {
        path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path) for path in paths
    }


def _git_record() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "worktree_clean": not bool(status),
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


def _resolve_semantic_rows(
    semantic_dir: Path,
    puzzles: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]], dict[str, Any]]:
    positions = _load_json_object(
        semantic_dir / "positions.json", label="Semantic positions"
    )
    selection = _load_json_object(
        semantic_dir / "selection.json", label="Semantic selection"
    )
    corpus_ids = np.asarray(puzzles["position_id"])
    lookup = {str(identity): index for index, identity in enumerate(corpus_ids)}
    fit: list[int] = []
    heldout: list[int] = []
    for record in positions["positions"]:
        position_id = str(record["position_id"])
        if position_id not in lookup:
            raise ValueError(f"Semantic position {position_id} is absent from the corpus")
        destination = fit if record["role"] == "raw_fit" else heldout
        destination.append(int(lookup[position_id]))
    pair_lookup = {str(row["pair_id"]): row for row in selection["pairs"]}
    targets = {target.pair_id: pair_lookup[target.pair_id] for target in CAUSAL_TARGETS}
    return (
        np.asarray(fit, dtype=np.int64),
        np.asarray(heldout, dtype=np.int64),
        targets,
        positions,
    )


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


def seal_causal_run(
    *,
    output: Path,
    metrics: Mapping[str, Any],
    design: Mapping[str, Any],
    positions: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    logs: Sequence[str],
) -> dict[str, Any]:
    target, staging = _output_staging(output)
    try:
        write_json_atomic(staging / "metrics.json", dict(metrics))
        write_json_atomic(staging / "design.json", dict(design))
        write_json_atomic(staging / "positions.json", dict(positions))
        write_jsonl_atomic(staging / "interventions.jsonl", records)
        write_json_atomic(staging / "manifest.json", dict(manifest))
        write_text_atomic(staging / "run.log", "\n".join(logs) + "\n")
        forbidden = [
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.suffix.lower() in FORBIDDEN_TENSOR_SUFFIXES
        ]
        if forbidden:
            raise ValueError("Causal run persisted forbidden tensors: " + ", ".join(forbidden))
        write_checksums(staging)
        verify_checksums(staging)
        os.replace(staging, target)
        checksums = verify_checksums(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output": str(target),
        "run_id": manifest["run_id"],
        "status": metrics["status"],
        "record_count": len(records),
        "retained_bytes": sum(
            path.stat().st_size for path in target.rglob("*") if path.is_file()
        ),
        "checksums": checksums,
    }


def run_causal_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.layer != 14:
        raise ValueError("Published LoRSA causal validation is pinned to layer 14")
    if args.batch_size <= 0 or args.control_count <= 0 or args.cpu_threads <= 0:
        raise ValueError("Batch size, control count, and CPU threads must be positive")
    if args.bootstrap_samples <= 0 or args.progress_every <= 0:
        raise ValueError("Bootstrap samples and progress interval must be positive")
    if args.max_heldout_positions < 0:
        raise ValueError("Held-out position limit must be non-negative")
    if args.provider_dollars != 0:
        raise ValueError("The local causal run requires provider_dollars=0")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.output.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {args.output.resolve()}")

    started = time.monotonic()
    started_utc = datetime.now(UTC)
    logs: list[str] = []

    def log(message: str) -> None:
        line = f"{time.monotonic() - started:10.3f}s {message}"
        logs.append(line)
        print(line, flush=True)

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
    transfer_gate = validate_sealed_source_transfer(
        args.source_transfer, expected_identities=identities, puzzles=puzzles
    )
    if not transfer_gate["passed"]:
        failed = [name for name, passed in transfer_gate["criteria"].items() if not passed]
        raise RuntimeError("Source-transfer dependency failed: " + ", ".join(failed))
    semantic_gate = validate_sealed_semantic_source(
        args.semantic_source, expected_identities=identities, puzzles=puzzles
    )
    if not semantic_gate["passed"]:
        failed = [name for name, passed in semantic_gate["criteria"].items() if not passed]
        raise RuntimeError("Semantic dependency failed: " + ", ".join(failed))
    if semantic_gate["source_transfer_run_id"] != transfer_gate["source_transfer_run_id"]:
        raise RuntimeError("Semantic and independently verified transfer runs differ")
    log(f"sealed dependency chain passed: {semantic_gate['semantic_run_id']}")

    fit_rows, heldout_rows, target_rows, source_positions = _resolve_semantic_rows(
        args.semantic_source, puzzles
    )
    if args.max_heldout_positions:
        heldout_rows = heldout_rows[: args.max_heldout_positions]
    if heldout_rows.size == 0:
        raise ValueError("Causal validation selected no held-out positions")

    log("strict-loading Raw/Hero models and reviewed LoRSA")
    models = load_bt4_comparison_models(
        raw_bt4_path=args.raw_bt4,
        hero_checkpoint_dir=args.hero_checkpoint,
        device=device,
    ).eval()
    loaded_lorsa = load_converted_lorsa(args.lorsa_manifest, args.lorsa_weights)
    loaded_lorsa.module.to(device).eval().requires_grad_(False)
    models.raw_encoder.requires_grad_(False)
    models.hero_encoder.requires_grad_(False)
    descriptor = models.descriptor()
    expected_descriptor = _load_json_object(
        args.semantic_source / "manifest.json", label="Semantic manifest"
    )["runtime_model_identity_gate"]["descriptor"]
    runtime_criteria = {
        "raw_identity_matches": descriptor["raw_asset_sha256"] == identities["raw"],
        "hero_identity_matches": descriptor["hero_state_sha256"] == identities["hero"],
        "encoder_layout_matches_semantic_source": descriptor["encoder_layout_sha256"]
        == expected_descriptor["encoder_layout_sha256"],
        "no_mixed_checkpoint_written": descriptor["on_disk_mixed_checkpoints"] is False,
    }
    if not all(runtime_criteria.values()):
        raise RuntimeError("Runtime model identity gate failed")

    target_features = [int(target_rows[target.pair_id]["feature_index"]) for target in CAUSAL_TARGETS]
    if len(target_features) != len(set(target_features)):
        raise RuntimeError("Causal target features must be unique")
    decoder_norm = loaded_lorsa.module.decoder_norm.detach().float().cpu().numpy()
    calibration = CalibrationAccumulator.create(
        loaded_lorsa.module.config.n_ov_heads, target_features
    )
    checkpoint_dtype = next(loaded_lorsa.module.parameters()).dtype
    log(f"Raw-fit calibration over {fit_rows.size} frozen positions")
    try:
        with torch.inference_mode():
            for start in range(0, fit_rows.size, args.batch_size):
                selected = fit_rows[start : start + args.batch_size]
                planes = torch.from_numpy(
                    np.asarray(puzzles["current_planes_u8"])[selected]
                ).to(device)
                dense = models.policy_logits_with_captures(
                    planes,
                    arm="RR",
                    compute_dtype=torch.float32,
                    capture_layers=(args.layer,),
                )
                if dense.captures is None:
                    raise AssertionError("Calibration capture disappeared")
                sparse = loaded_lorsa.module(
                    dense.captures.hook_attn_in[0].to(dtype=checkpoint_dtype)
                )
                calibration.update(sparse.features, target_features)
                del sparse, dense, planes
                completed = min(start + args.batch_size, fit_rows.size)
                if completed == fit_rows.size or completed % args.progress_every == 0:
                    log(f"Raw-fit calibration {completed}/{fit_rows.size}")

        controls = choose_matched_random_controls(
            calibration,
            decoder_norm,
            target_features=target_features,
            count=args.control_count,
            seed=args.seed,
        )
        calibration_records = {
            index: calibration.feature_record(index, decoder_norm)
            for index in target_features
        }
        log("frozen matched-control plan; beginning held-out interventions")

        records: list[dict[str, Any]] = []
        effects: dict[str, dict[tuple[str, str], np.ndarray]] = {"RR": {}, "HH": {}}
        suffix_errors = {"RR": 0.0, "HH": 0.0}
        for start in range(0, heldout_rows.size, args.batch_size):
            selected = heldout_rows[start : start + args.batch_size]
            for arm in ("RR", "HH"):
                batch_records, batch_effects, suffix_error = _evaluate_arm_batch(
                    models=models,
                    loaded_lorsa=loaded_lorsa,
                    puzzles=puzzles,
                    rows=selected,
                    target_rows=target_rows,
                    target_specs=CAUSAL_TARGETS,
                    control_plan=controls,
                    calibration_records=calibration_records,
                    arm=arm,
                    layer=args.layer,
                    seed=args.seed,
                    device=device,
                    validate_suffix=start == 0,
                )
                records.extend(batch_records)
                effects[arm].update(batch_effects)
                suffix_errors[arm] = max(suffix_errors[arm], suffix_error)
            completed = min(start + args.batch_size, heldout_rows.size)
            if completed == heldout_rows.size or completed % args.progress_every == 0:
                log(f"held-out Raw/Hero interventions {completed}/{heldout_rows.size}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32

    log("computing paired bootstrap, sign-flip, Holm, and model-diff summaries")
    summary = build_causal_summary(
        records,
        effects,
        target_specs=CAUSAL_TARGETS,
        target_rows=target_rows,
        control_plan=controls,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    target_design = []
    for spec in CAUSAL_TARGETS:
        target = target_rows[spec.pair_id]
        feature_index = int(target["feature_index"])
        target_design.append(
            {
                **asdict(spec),
                "semantic_selection_row": target,
                "calibration": calibration_records[feature_index],
                "matched_random_controls": controls[feature_index],
            }
        )
    design = {
        "schema_version": CAUSAL_DESIGN_SCHEMA,
        "selection_timing": {
            "posthoc_to_descriptive_study": True,
            "frozen_before_intervention_outcomes": True,
            "confirmatory": False,
        },
        "targets": target_design,
        "matched_control_count_per_target": args.control_count,
        "matching_variables": [
            "Raw-fit active-token support",
            "Raw-fit active-position support",
            "Raw-fit mean positive activation",
            "LoRSA decoder norm",
            "Raw-fit position coactivity Jaccard",
        ],
        "matching_randomization": (
            "seeded weighted sample without replacement from the 32 nearest eligible features"
        ),
        "target_scale_doses": list(TARGET_SCALE_DOSES),
        "target_injection_doses": list(TARGET_INJECTION_DOSES),
        "insertion_site": (
            "one deterministic target-class token per position, shared across Raw/Hero and controls"
        ),
        "insertion_amplitude": "Raw-fit mean positive target activation; frozen for both arms",
        "matched_control_intervention": {
            "activation_scalar": "exact target-feature scalar",
            "token_support": "exact target-feature tokens for ablation; exact insertion token",
            "decoder_norm": "control direction rescaled to exact target decoder norm",
            "only_changed_dimension": "LoRSA decoder direction",
        },
        "primary_estimand": (
            "target-direction ablation legal-policy TV minus mean scalar/token/norm-matched "
            "random decoder-direction TV on target-active positions"
        ),
        "statistics": {
            "paired_bootstrap_samples": args.bootstrap_samples,
            "paired_sign_flip_samples": args.bootstrap_samples,
            "multiple_testing": "Holm across 12 Raw/Hero target ablation tests",
        },
        "claim_boundary": (
            "causal within frozen LoRSA layer-14 decoder replacement; not native-feature or input-"
            "concept causality"
        ),
    }
    position_payload = {
        "schema_version": CAUSAL_POSITIONS_SCHEMA,
        "source_semantic_positions_identity": content_identity(source_positions),
        "position_count": int(heldout_rows.size),
        "positions": [
            {
                "position_id": str(np.asarray(puzzles["position_id"])[row]),
                "group_id": str(np.asarray(puzzles["group_id"])[row]),
                "public_split": "development",
                "role": "heldout_raw_and_hero",
            }
            for row in heldout_rows
        ],
        "test_positions": [],
        "planes_persisted": False,
    }
    noop_tv_max = max(
        float(row["batched_noop_to_reference"]["legal_policy_total_variation"])
        for row in records
    )
    metrics = {
        "schema_version": CAUSAL_METRICS_SCHEMA,
        "status": "complete_exploratory_causal_surrogate",
        "position_count": int(heldout_rows.size),
        "record_count": len(records),
        "target_count": len(CAUSAL_TARGETS),
        "arms": ["RR", "HH"],
        "summary": summary,
        "dependency_gates": {
            "source_transfer": transfer_gate,
            "semantic_source": semantic_gate,
            "runtime_models": {"passed": True, "criteria": runtime_criteria},
        },
        "implementation_gates": {
            "fast_suffix_max_abs_error": suffix_errors,
            "fast_suffix_atol": 1e-5,
            "all_fast_suffix_checks_passed": all(
                value <= 1e-5 for value in suffix_errors.values()
            ),
            "batched_noop_max_legal_policy_total_variation": noop_tv_max,
            "batched_noop_total_variation_atol": 1e-6,
            "batched_noop_numerical_floor_passed": noop_tv_max <= 1e-6,
            "matched_control_scalar_token_decoder_norm_exact": True,
            "lorsa_parameters_frozen": not any(
                parameter.requires_grad for parameter in loaded_lorsa.module.parameters()
            ),
            "dense_features_persisted": False,
        },
        "claim_gates": {
            "lorsa_decoder_feature_policy_causality_measured": True,
            "exploratory_specific_causal_gate_pass_count": summary[
                "exploratory_causal_gate_pass_count"
            ],
            "native_dense_bt4_feature_causality_permitted": False,
            "input_chess_concept_causality_permitted": False,
            "confirmatory_test_claim_permitted": False,
            "reserved_test_split_selected_or_evaluated": False,
        },
        "negative_results_retained": True,
    }
    elapsed = time.monotonic() - started
    resources: dict[str, Any] = {"wall_seconds": elapsed}
    if device.type == "cuda":
        resources.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    source_files = _source_file_hashes()
    payloads = {
        "metrics": content_identity(metrics),
        "design": content_identity(design),
        "positions": content_identity(position_payload),
        "interventions": sha256_bytes(canonical_json_bytes(list(records))),
        "semantic_source_run": semantic_gate["semantic_run_id"],
        "semantic_source_checksums": semantic_gate["semantic_checksum_identity"],
        "source_transfer_run": transfer_gate["source_transfer_run_id"],
        "corpus": identities["corpus_integrity"],
        "lorsa": identities["lorsa_weights"],
        "raw": identities["raw"],
        "hero": identities["hero"],
        "source_files": content_identity(source_files),
    }
    run_id = content_identity(payloads)
    manifest = {
        "schema_version": CAUSAL_RUN_SCHEMA,
        "run_id": f"sha256:{run_id}",
        "started_utc": started_utc.isoformat(),
        "elapsed_seconds": elapsed,
        "git": _git_record(),
        "environment": _environment(device),
        "resource_usage": resources,
        "experiment": {
            "layer": args.layer,
            "fit_position_count": int(fit_rows.size),
            "heldout_position_count": int(heldout_rows.size),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "compute_dtype": "float32",
            "lorsa_checkpoint_dtype": str(checkpoint_dtype),
            "tf32": False,
            "autocast": False,
            "exploratory": True,
            "confirmatory": False,
        },
        "cost": {"backend": args.backend, "provider_dollars": args.provider_dollars},
        "storage_contract": {
            "dense_activations_persisted": False,
            "policy_logits_persisted": False,
            "planes_persisted": False,
            "retained_files": [
                "metrics.json",
                "design.json",
                "positions.json",
                "interventions.jsonl",
                "manifest.json",
                "run.log",
                "checksums.sha256",
            ],
            "forbidden_tensor_suffixes": sorted(FORBIDDEN_TENSOR_SUFFIXES),
        },
        "dependency_gates": metrics["dependency_gates"],
        "source_identities": identities,
        "payload_identities": payloads,
        "source_files": source_files,
    }
    logs.append(f"run_id sha256:{run_id}")
    result = seal_causal_run(
        output=args.output,
        metrics=metrics,
        design=design,
        positions=position_payload,
        records=records,
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
    parser.add_argument("--hero-checkpoint", type=Path, default=DEFAULT_HERO_CHECKPOINT)
    parser.add_argument(
        "--corpus-manifest", type=Path, default=DEFAULT_CORPUS_DIR / "manifest.json"
    )
    parser.add_argument("--lorsa-manifest", type=Path, default=DEFAULT_LORSA_MANIFEST)
    parser.add_argument("--lorsa-weights", type=Path, default=DEFAULT_LORSA_WEIGHTS)
    parser.add_argument("--source-transfer", type=Path, default=DEFAULT_SOURCE_TRANSFER)
    parser.add_argument("--semantic-source", type=Path, default=DEFAULT_SEMANTIC_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--control-count", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-heldout-positions", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--backend", default="local GTX 1660 Ti zero-provider-cost causal run"
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
        transfer = validate_sealed_source_transfer(
            args.source_transfer,
            expected_identities=identities,
            puzzles=corpora["puzzles"],
        )
        semantic = validate_sealed_semantic_source(
            args.semantic_source,
            expected_identities=identities,
            puzzles=corpora["puzzles"],
        )
        result = {
            "passed": transfer["passed"]
            and semantic["passed"]
            and transfer["source_transfer_run_id"]
            == semantic["source_transfer_run_id"],
            "source_transfer": transfer,
            "semantic_source": semantic,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    result = run_causal_validation(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
