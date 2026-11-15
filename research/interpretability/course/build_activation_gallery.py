"""Build a compact, provenance-bound gallery of real Raw/Hero activations.

The gallery is deliberately small enough for local teaching use.  It stores
eight public development positions, both encoders' post-layer residual streams,
four policy-lens trajectories, selected trained-probe predictions, and
target-action gradient summaries.  It never writes either source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import numpy as np
import safetensors
import torch
from safetensors import safe_open
from torch import Tensor

from chess_dfm_jax.policy import decode_lc0_canonical_1858
from research.interpretability.artifacts import sha256_file
from research.interpretability.attribution import action_objective
from research.interpretability.chessbench import load_public_corpus
from research.interpretability.logit_lens import cross_head_lenses
from research.interpretability.models import ARM_IDS, load_bt4_comparison_models
from research.interpretability.pilot_corpus import write_deterministic_npz
from research.interpretability.validate_probe_result import validate_probe_result


_REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "bt4-real-activation-gallery-v1"
DEFAULT_CORPUS = _REPO_ROOT / "research/eval/interpretability_public_v2/manifest.json"
DEFAULT_RAW_BT4 = _REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz"
DEFAULT_HERO = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint"
DEFAULT_PROBES = _REPO_ROOT / "research/analysis/modal_probe_all_layers_dev_v1"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "curriculum/interpretability/artifacts"
DEFAULT_DATA_NAME = "real_activation_gallery_v1.npz"
DEFAULT_MANIFEST_NAME = "real_activation_gallery_v1.json"
LAYERS = tuple(range(15))
MODEL_NAMES = ("raw", "hero")
PROBE_TASKS = ("legal_destination", "opponent_attack_count", "piece_code")
PROBE_RUN_SCHEMA = "bt4-raw-hero-probe-lookahead-run-v1"
PROBE_METADATA_SCHEMA = "bt4-raw-hero-probe-metadata-v1"
SCIENTIFIC_CODE_SOURCES = {
    "gallery_builder": "research/interpretability/course/build_activation_gallery.py",
    "comparison_model_loader": "research/interpretability/models.py",
    "torch_model_definition": "research/train_torch.py",
    "bt4_weight_loader": "chess_dfm_jax/weights.py",
    "bt4_proto_schema": "chess_dfm_jax/proto/net_pb2.py",
    "board_encoding": "chess_dfm_jax/encoding.py",
    "action_codec": "chess_dfm_jax/policy.py",
    "action_codec_move_table": "chess_dfm_jax/policy_moves.txt",
    "policy_attention_map": "chess_dfm_jax/policy_attn_map.txt",
    "corpus_loader": "research/interpretability/chessbench.py",
    "concept_labels": "research/interpretability/chess_concepts.py",
    "artifact_integrity": "research/interpretability/artifacts.py",
    "deterministic_npz": "research/interpretability/pilot_corpus.py",
    "policy_lens": "research/interpretability/logit_lens.py",
    "attribution_objective": "research/interpretability/attribution.py",
    "legal_policy_objective": "research/interpretability/patching.py",
    "probe_result_validator": "research/interpretability/validate_probe_result.py",
    "probe_run_selector": "research/interpretability/probe_pilot.py",
    "probe_metrics_code": "research/interpretability/probes.py",
    "lookahead_probe": "research/interpretability/lookahead.py",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _selected_rows(puzzles: Mapping[str, np.ndarray], count: int) -> np.ndarray:
    development = np.flatnonzero(np.asarray(puzzles["split_u8"]) == 1)
    if not 0 < count <= development.size:
        raise ValueError(f"count must lie in [1, {development.size}]")
    identifiers = np.asarray(puzzles["position_id"])[development]
    return development[np.argsort(identifiers, kind="stable")[:count]]


def _array_inventory(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def _token_cka(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    numerator = float(np.sum(left_gram * right_gram))
    denominator = float(
        np.linalg.norm(left_gram, ord="fro")
        * np.linalg.norm(right_gram, ord="fro")
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def _layer_metrics(raw: np.ndarray, hero: np.ndarray) -> dict[str, list[float]]:
    if raw.shape != hero.shape or raw.ndim != 4:
        raise ValueError("Expected paired [position, layer, square, channel] activations")
    result = {
        "raw_rms": [],
        "hero_rms": [],
        "delta_rms": [],
        "symmetric_relative_l2": [],
        "mean_token_cosine": [],
        "linear_cka_over_tokens": [],
    }
    for layer in range(raw.shape[1]):
        left = raw[:, layer].reshape(-1, raw.shape[-1]).astype(np.float64)
        right = hero[:, layer].reshape(-1, hero.shape[-1]).astype(np.float64)
        delta = right - left
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        cosine = np.divide(
            np.sum(left * right, axis=1),
            left_norm * right_norm,
            out=np.zeros_like(left_norm),
            where=(left_norm * right_norm) > 0.0,
        )
        denominator = 0.5 * (np.linalg.norm(left) + np.linalg.norm(right))
        result["raw_rms"].append(float(np.sqrt(np.mean(left * left))))
        result["hero_rms"].append(float(np.sqrt(np.mean(right * right))))
        result["delta_rms"].append(float(np.sqrt(np.mean(delta * delta))))
        result["symmetric_relative_l2"].append(
            0.0 if denominator == 0.0 else float(np.linalg.norm(delta) / denominator)
        )
        result["mean_token_cosine"].append(float(cosine.mean()))
        result["linear_cka_over_tokens"].append(_token_cka(left, right))
    return result


def _top_delta_channels(raw: np.ndarray, hero: np.ndarray, *, count: int) -> np.ndarray:
    if count <= 0 or count > raw.shape[-1]:
        raise ValueError("Invalid channel selection count")
    score = np.mean(np.abs(hero.astype(np.float32) - raw.astype(np.float32)), axis=(0, 2))
    order = np.argsort(-score, axis=1, kind="stable")
    return order[:, :count].astype(np.int16)


def _centered_singular_values(
    activations: np.ndarray,
    *,
    layers: tuple[int, ...],
) -> np.ndarray:
    """Return exact finite-gallery spectra over position-square observations."""

    values = np.asarray(activations, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError("Activation spectra require [position, layer, square, channel]")
    spectra = []
    for layer in layers:
        matrix = values[:, layer].reshape(-1, values.shape[-1]).astype(np.float64)
        matrix -= matrix.mean(axis=0, keepdims=True)
        spectra.append(np.linalg.svd(matrix, compute_uv=False, full_matrices=False))
    return np.stack(spectra).astype(np.float32)


def _decode_legal_moves(
    fens: np.ndarray,
    legal_indices: np.ndarray,
    legal_counts: np.ndarray,
) -> np.ndarray:
    result = np.full(legal_indices.shape, "", dtype="<U5")
    for row, fen in enumerate(fens.astype(str)):
        board = chess.Board(fen)
        for slot in range(int(legal_counts[row])):
            action = int(legal_indices[row, slot])
            move = decode_lc0_canonical_1858(board, action)
            if not board.is_legal(move):
                raise ValueError(f"Decoded action {action} is illegal for gallery row {row}")
            result[row, slot] = move.uci()
    return result


def _probe_prediction(
    handle: Any,
    prefix: str,
    activations: Tensor,
) -> Tensor:
    mean = handle.get_tensor(f"{prefix}.feature_mean")
    scale = handle.get_tensor(f"{prefix}.feature_scale")
    weight = handle.get_tensor(f"{prefix}.module.layers.weight")
    bias = handle.get_tensor(f"{prefix}.module.layers.bias")
    standardized = (activations.detach().float().cpu() - mean) / scale
    logits = standardized @ weight.T + bias
    task = prefix.rsplit(".", 1)[-1]
    if task == "piece_code":
        return torch.softmax(logits, dim=-1)
    if task == "legal_destination":
        return torch.sigmoid(logits).squeeze(-1)
    target_mean = handle.get_tensor(f"{prefix}.target_mean").item()
    target_scale = handle.get_tensor(f"{prefix}.target_scale").item()
    return logits.squeeze(-1) * target_scale + target_mean


def _all_probe_predictions(
    raw: Tensor,
    hero: Tensor,
    probe_path: Path,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    # Input captures are [layer, position, square, channel].
    with safe_open(probe_path, framework="pt", device="cpu") as handle:
        for model_name, values in (("raw", raw), ("hero", hero)):
            for task in PROBE_TASKS:
                predictions = []
                for layer in LAYERS:
                    prefix = f"concept.{model_name}.L{layer}.{task}"
                    prediction = _probe_prediction(handle, prefix, values[layer])
                    predictions.append(prediction)
                stacked = torch.stack(predictions, dim=1)
                arrays[f"{model_name}_probe_{task}_f16"] = (
                    stacked.numpy().astype(np.float16)
                )
    return arrays


def _validated_probe_contract(
    probe_dir: Path,
    *,
    models: Any,
    corpus_record: Mapping[str, Any],
    independent_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind probe bytes to the run's model, corpus, split, and probe ABI."""

    manifest_path = probe_dir / "manifest.json"
    metadata_path = probe_dir / "probe_metadata.json"
    weights_path = probe_dir / "probe_weights.safetensors"
    checksums_path = probe_dir / "checksums.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PROBE_RUN_SCHEMA:
        raise ValueError("Unsupported retained probe-run schema")
    if metadata.get("schema_version") != PROBE_METADATA_SCHEMA:
        raise ValueError("Unsupported retained probe-metadata schema")

    ledger: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.strip()
        if len(digest) != 64 or filename in ledger:
            raise ValueError("Malformed or duplicate retained probe checksum")
        ledger[filename] = digest
    probe_hashes = {
        "manifest.json": sha256_file(manifest_path),
        "probe_metadata.json": sha256_file(metadata_path),
        "probe_weights.safetensors": sha256_file(weights_path),
    }
    for filename, digest in probe_hashes.items():
        if ledger.get(filename) != digest:
            raise ValueError(f"Retained probe checksum drift: {filename}")

    payload_identities = manifest["payload_identities"]
    expected_run_id = "sha256:" + hashlib.sha256(
        _canonical_bytes(payload_identities)
    ).hexdigest()
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("Retained probe run identity drift")
    if payload_identities.get("weights") != probe_hashes["probe_weights.safetensors"]:
        raise ValueError("Probe-run identity does not bind the supplied weights")

    descriptor = models.descriptor()
    model_record = manifest["model_lattice"]
    model_identity_keys = (
        "raw_asset_sha256",
        "raw_mapping_sha256",
        "hero_state_sha256",
        "encoder_layout_sha256",
    )
    for key in model_identity_keys:
        if model_record.get(key) != descriptor.get(key):
            raise ValueError(f"Probe/model compatibility failed for {key}")
    if payload_identities.get("raw") != descriptor["raw_asset_sha256"]:
        raise ValueError("Probe payload identity uses a different Raw checkpoint")
    if payload_identities.get("hero") != descriptor["hero_state_sha256"]:
        raise ValueError("Probe payload identity uses a different Hero checkpoint")

    corpus_identity = corpus_record["manifest_integrity"]["sha256"]
    probe_corpus = manifest["public_corpus_manifest"]
    if probe_corpus.get("integrity_sha256") != corpus_identity:
        raise ValueError("Probe weights were fit against a different public corpus")
    if payload_identities.get("splits") != probe_corpus.get("split_position_sha256"):
        raise ValueError("Probe payload identity does not bind its recorded data splits")
    split_policy = "train fit + train selection; development evaluation; test untouched"
    if probe_corpus.get("split_policy") != split_policy:
        raise ValueError("Probe split policy drift")

    experiment = manifest["experiment"]
    if experiment.get("activation_boundary") != "resid_post_after_ln":
        raise ValueError("Probe activation boundary is incompatible with the gallery")
    if tuple(experiment.get("layers", ())) != LAYERS:
        raise ValueError("Probe run does not cover all gallery layers")
    if set(experiment.get("concepts", ())) != set(PROBE_TASKS):
        raise ValueError("Probe task inventory is incompatible with the gallery")
    if experiment.get("test_split_opened") is not False:
        raise ValueError("Probe run opened the test split")

    expected_probes = {
        f"concept.{model}.L{layer}.{task}"
        for model in MODEL_NAMES
        for layer in LAYERS
        for task in PROBE_TASKS
    }
    observed_probes = {
        name for name in metadata["probes"] if name.startswith("concept.")
    }
    if observed_probes != expected_probes:
        raise ValueError("Probe metadata concept inventory drift")
    for name in sorted(expected_probes):
        task = name.rsplit(".", 1)[-1]
        record = metadata["probes"][name]
        probe = record["probe"]
        expected_output = 13 if task == "piece_code" else 1
        if (
            probe.get("schema_version") != "bt4-concept-probe-v1"
            or probe.get("architecture") != "linear"
            or int(probe.get("input_width", -1)) != 1024
            or int(probe.get("output_width", -1)) != expected_output
            or probe.get("normalization_fit") != "training split only"
        ):
            raise ValueError(f"Probe ABI or fitting contract drift: {name}")
        classes = record.get("classes")
        if task == "piece_code" and classes != list(range(13)):
            raise ValueError(f"Piece-code class order drift: {name}")
        if task != "piece_code" and classes is not None:
            raise ValueError(f"Unexpected class inventory for scalar probe: {name}")

    validation_integrity = independent_validation["integrity"]
    required_true = (
        "corpus_integrity_match",
        "deterministic_split_hashes_match",
        "evaluation_ids_match",
        "evaluation_rows_all_development",
        "pickle_free_weight_metadata",
    )
    if not all(validation_integrity.get(key) is True for key in required_true):
        raise ValueError("Independent probe validation failed an integrity gate")
    if validation_integrity.get("test_rows_evaluated") is not False:
        raise ValueError("Independent probe validation reports an opened test split")
    recomputation = independent_validation["recomputation"]
    if (
        float(recomputation["maximum_absolute_metric_difference"]) > 1e-8
        or recomputation.get("paired_points_and_deltas_match") is not True
        or recomputation.get("paired_intervals_contain_point_estimates") is not True
    ):
        raise ValueError("Independent probe prediction recomputation failed")
    if independent_validation["run"]["run_id"] != manifest["run_id"]:
        raise ValueError("Independent validation refers to a different probe run")

    return {
        "source_run_id": manifest["run_id"],
        "source_manifest_schema_version": manifest["schema_version"],
        "source_manifest_sha256": probe_hashes["manifest.json"],
        "source_metadata_schema_version": metadata["schema_version"],
        "source_metadata_sha256": probe_hashes["probe_metadata.json"],
        "source_weights_sha256": probe_hashes["probe_weights.safetensors"],
        "source_checksums_sha256": sha256_file(checksums_path),
        "activation_boundary": experiment["activation_boundary"],
        "layers": list(LAYERS),
        "split_policy": split_policy,
        "test_split_opened": False,
        "public_corpus_integrity": corpus_identity,
        "model_identities": {key: descriptor[key] for key in model_identity_keys},
        "independent_validation": {
            "schema_version": independent_validation["schema_version"],
            "validation_sha256": independent_validation["validation_sha256"],
            "maximum_absolute_metric_difference": recomputation[
                "maximum_absolute_metric_difference"
            ],
            "evaluation_rows_all_development": True,
            "test_rows_evaluated": False,
        },
        "compatibility_gate": {
            "passed": True,
            "checks": [
                "checksum ledger and weight payload identity",
                "Raw/Hero checkpoint and encoder-layout identities",
                "public-corpus and deterministic split identities",
                "all-layer residual boundary and concept-task inventory",
                "training-only normalization and exact probe ABI",
                "independent prediction-level metric reproduction",
                "development evaluation with test split untouched",
            ],
        },
    }


def _native_target_gradients(
    encoder: Any,
    planes: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
    targets: Tensor,
    *,
    selected_layers: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head = encoder.policy_head
    if head is None:
        raise ValueError("Target-gradient extraction requires a native policy head")
    x = encoder.embedding(planes, encoder.alpha, torch.float32).detach().requires_grad_(True)
    retained: list[Tensor] = []
    for layer_index, layer in enumerate(encoder.layers):
        x = layer(x, encoder.alpha, torch.float32)
        if layer_index in selected_layers:
            retained.append(x)
    logits = head(x, torch.float32)
    objective = action_objective(
        logits,
        targets,
        kind="target_log_odds",
        legal_indices=legal_indices,
        legal_counts=legal_counts,
    )
    gradients = torch.autograd.grad(objective.sum(), retained, create_graph=False)
    square_norm = torch.stack(
        [gradient.float().square().sum(dim=-1).sqrt() for gradient in gradients],
        dim=1,
    )
    gradient_x_state = torch.stack(
        [(gradient.float() * state.float()).sum(dim=-1) for gradient, state in zip(gradients, retained, strict=True)],
        dim=1,
    )
    gradient_rms = torch.stack(
        [gradient.float().square().mean(dim=(-2, -1)).sqrt() for gradient in gradients],
        dim=1,
    )
    return (
        square_norm.detach().cpu().numpy().astype(np.float32),
        gradient_x_state.detach().cpu().numpy().astype(np.float32),
        gradient_rms.detach().cpu().numpy().astype(np.float32),
    )


def _position_summaries(
    arrays: Mapping[str, np.ndarray],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    arms = arrays["arm_ids"].astype(str)
    lens_logits = arrays["lens_logits_f16"].astype(np.float32)
    records: list[dict[str, Any]] = []
    for position in range(arrays["position_id"].size):
        legal_count = int(arrays["legal_count_u16"][position])
        legal = arrays["legal_idx_u16"][position, :legal_count].astype(np.int64)
        moves = arrays["legal_move_uci"][position, :legal_count].astype(str)
        target = int(arrays["target_action_u16"][position])
        arm_records: dict[str, Any] = {}
        for arm_index, arm in enumerate(arms):
            final = lens_logits[arm_index, position, -1]
            full_prob = np.exp(final - np.max(final))
            full_prob /= full_prob.sum()
            legal_logits = final[legal]
            legal_prob = np.exp(legal_logits - np.max(legal_logits))
            legal_prob /= legal_prob.sum()
            order = np.argsort(-legal_prob, kind="stable")[:top_k]
            target_slots = np.flatnonzero(legal == target)
            if target_slots.size != 1:
                raise ValueError("Gallery target must occur exactly once in legal actions")
            arm_records[str(arm)] = {
                "legal_mass_before_masking": float(full_prob[legal].sum()),
                "target_probability": float(legal_prob[target_slots[0]]),
                "top_legal_moves": [
                    {
                        "move": str(moves[index]),
                        "action": int(legal[index]),
                        "probability": float(legal_prob[index]),
                    }
                    for index in order
                ],
            }
        records.append(
            {
                "position_index": position,
                "position_id": str(arrays["position_id"][position]),
                "puzzle_id": str(arrays["puzzle_id"][position]),
                "fen": str(arrays["root_fen"][position]),
                "solution_uci": str(arrays["solution_uci"][position]),
                "themes": str(arrays["themes"][position]),
                "target_action": target,
                "target_move": str(moves[target_slots[0]]),
                "legal_count": legal_count,
                "arms": arm_records,
            }
        )
    return records


def build_activation_gallery(
    *,
    corpus_manifest: Path = DEFAULT_CORPUS,
    raw_bt4: Path = DEFAULT_RAW_BT4,
    hero_checkpoint: Path = DEFAULT_HERO,
    probe_dir: Path = DEFAULT_PROBES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    count: int = 8,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    started = time.monotonic()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if count <= 0 or count > 32:
        raise ValueError("The teaching gallery count must lie in [1, 32]")
    corpus_manifest = corpus_manifest.resolve(strict=True)
    raw_bt4 = raw_bt4.resolve(strict=True)
    hero_checkpoint = hero_checkpoint.resolve(strict=True)
    probe_dir = probe_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / DEFAULT_DATA_NAME
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME

    torch.manual_seed(20260811)
    np.random.seed(20260811)
    if target_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(target_device)

    corpus, corpus_record = load_public_corpus(corpus_manifest)
    probe_validation = validate_probe_result(probe_dir, corpus_manifest)
    puzzles = corpus["puzzles"]
    rows = _selected_rows(puzzles, count)
    planes = torch.from_numpy(np.asarray(puzzles["current_planes_u8"])[rows]).to(target_device)
    legal_indices = torch.from_numpy(np.asarray(puzzles["legal_idx_u16"])[rows]).to(target_device)
    legal_counts = torch.from_numpy(np.asarray(puzzles["legal_count_u16"])[rows]).to(target_device)
    targets = torch.from_numpy(np.asarray(puzzles["target_action_u16"])[rows]).to(target_device)

    models = load_bt4_comparison_models(
        raw_bt4_path=raw_bt4,
        hero_checkpoint_dir=hero_checkpoint,
        device=target_device,
    )
    for encoder in (models.raw_encoder, models.hero_encoder):
        encoder.requires_grad_(False)
    probe_contract = _validated_probe_contract(
        probe_dir,
        models=models,
        corpus_record=corpus_record,
        independent_validation=probe_validation,
    )

    with torch.inference_mode():
        lattice = models.lattice_logits(
            planes,
            compute_dtype=torch.float32,
            capture_layers=LAYERS,
        )
        if lattice.raw_captures is None or lattice.hero_captures is None:
            raise AssertionError("All-layer capture was not returned")
        raw_capture = lattice.raw_captures.resid_post_after_ln
        hero_capture = lattice.hero_captures.resid_post_after_ln
        raw_head = models.raw_encoder.policy_head
        hero_head = models.hero_encoder.policy_head
        if raw_head is None or hero_head is None:
            raise AssertionError("Comparison head disappeared")
        lenses = cross_head_lenses(
            lattice.raw_captures,
            lattice.hero_captures,
            raw_head,
            hero_head,
            compute_dtype=torch.float32,
        )
        final_errors = {
            arm: float((lenses[arm][-1] - lattice.logits[arm]).abs().max().item())
            for arm in ARM_IDS
        }
        parity_atol = 1e-5
        if any(error > parity_atol for error in final_errors.values()):
            raise RuntimeError(f"Final policy-lens parity failed: {final_errors}")
        probe_arrays = _all_probe_predictions(
            raw_capture,
            hero_capture,
            probe_dir / "probe_weights.safetensors",
        )

    selected_gradient_layers = (0, 7, 14)
    raw_gradient = _native_target_gradients(
        models.raw_encoder,
        planes,
        legal_indices,
        legal_counts,
        targets,
        selected_layers=selected_gradient_layers,
    )
    hero_gradient = _native_target_gradients(
        models.hero_encoder,
        planes,
        legal_indices,
        legal_counts,
        targets,
        selected_layers=selected_gradient_layers,
    )

    raw_np = raw_capture.detach().permute(1, 0, 2, 3).cpu().numpy().astype(np.float16)
    hero_np = hero_capture.detach().permute(1, 0, 2, 3).cpu().numpy().astype(np.float16)
    spectrum_layers = (0, 7, 14)
    fens = np.asarray(puzzles["root_fen"])[rows].astype("<U72")
    legal_np = np.asarray(puzzles["legal_idx_u16"])[rows].astype(np.uint16)
    legal_count_np = np.asarray(puzzles["legal_count_u16"])[rows].astype(np.uint16)
    arrays: dict[str, np.ndarray] = {
        "arm_ids": np.asarray(ARM_IDS, dtype="<U2"),
        "layer_indices_i16": np.asarray(LAYERS, dtype=np.int16),
        "gradient_layer_indices_i16": np.asarray(selected_gradient_layers, dtype=np.int16),
        "spectrum_layer_indices_i16": np.asarray(spectrum_layers, dtype=np.int16),
        "position_id": np.asarray(puzzles["position_id"])[rows].astype("<U64"),
        "puzzle_id": np.asarray(puzzles["puzzle_id"])[rows].astype("<U5"),
        "root_fen": fens,
        "solution_uci": np.asarray(puzzles["solution_uci"])[rows].astype("<U126"),
        "themes": np.asarray(puzzles["themes"])[rows].astype("<U117"),
        "split_u8": np.asarray(puzzles["split_u8"])[rows].astype(np.uint8),
        "piece_code_i8": np.asarray(puzzles["piece_code_i8"])[rows].astype(np.int8),
        "legal_destination_u8": np.asarray(puzzles["legal_destination_u8"])[rows].astype(np.uint8),
        "attack_count_theirs_u8": np.asarray(puzzles["attack_count_theirs_u8"])[rows].astype(np.uint8),
        "legal_idx_u16": legal_np,
        "legal_count_u16": legal_count_np,
        "legal_move_uci": _decode_legal_moves(fens, legal_np, legal_count_np),
        "target_action_u16": np.asarray(puzzles["target_action_u16"])[rows].astype(np.uint16),
        "raw_resid_post_f16": raw_np,
        "hero_resid_post_f16": hero_np,
        "top_delta_channel_i16": _top_delta_channels(raw_np, hero_np, count=8),
        "raw_centered_singular_values_f32": _centered_singular_values(
            raw_np,
            layers=spectrum_layers,
        ),
        "hero_centered_singular_values_f32": _centered_singular_values(
            hero_np,
            layers=spectrum_layers,
        ),
        "lens_logits_f16": np.stack(
            [lenses[arm].permute(1, 0, 2).detach().cpu().numpy() for arm in ARM_IDS],
            axis=0,
        ).astype(np.float16),
        "raw_target_gradient_square_norm_f32": raw_gradient[0],
        "raw_target_gradient_x_state_f32": raw_gradient[1],
        "raw_target_gradient_rms_f32": raw_gradient[2],
        "hero_target_gradient_square_norm_f32": hero_gradient[0],
        "hero_target_gradient_x_state_f32": hero_gradient[1],
        "hero_target_gradient_rms_f32": hero_gradient[2],
        **probe_arrays,
    }
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("Gallery contains nonfinite floating values")
    write_deterministic_npz(data_path, arrays)
    data_sha256 = sha256_file(data_path)
    layer_metrics = _layer_metrics(raw_np.astype(np.float32), hero_np.astype(np.float32))

    source_files = {
        source_id: {
            "path": relative,
            "sha256": sha256_file(_REPO_ROOT / relative),
        }
        for source_id, relative in SCIENTIFIC_CODE_SOURCES.items()
    }
    source_files.update({
        "public_corpus_manifest": {
            "path": str(corpus_manifest.relative_to(_REPO_ROOT)),
            "sha256": sha256_file(corpus_manifest),
        },
        "public_corpus_checksums": {
            "path": str((corpus_manifest.parent / "checksums.sha256").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(corpus_manifest.parent / "checksums.sha256"),
        },
        "public_corpus_puzzles": {
            "path": str(
                (corpus_manifest.parent / corpus_record["puzzles"]["data"]["path"])
                .relative_to(_REPO_ROOT)
            ),
            "sha256": corpus_record["puzzles"]["data"]["sha256"],
        },
        "public_corpus_engine_positions": {
            "path": str(
                (
                    corpus_manifest.parent
                    / corpus_record["engine_positions"]["data"]["path"]
                ).relative_to(_REPO_ROOT)
            ),
            "sha256": corpus_record["engine_positions"]["data"]["sha256"],
        },
        "raw_bt4": {
            "path": str(raw_bt4.relative_to(_REPO_ROOT)),
            "sha256": models.raw_manifest["raw_asset"]["sha256"],
        },
        "hero_checkpoint_manifest": {
            "path": str((hero_checkpoint / "manifest.json").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(hero_checkpoint / "manifest.json"),
        },
        "hero_checkpoint_state": {
            "path": str((hero_checkpoint / "model.safetensors").relative_to(_REPO_ROOT)),
            "sha256": models.hero_manifest["state"]["sha256"],
        },
        "probe_run_manifest": {
            "path": str((probe_dir / "manifest.json").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(probe_dir / "manifest.json"),
        },
        "probe_metadata": {
            "path": str((probe_dir / "probe_metadata.json").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(probe_dir / "probe_metadata.json"),
        },
        "probe_checksums": {
            "path": str((probe_dir / "checksums.sha256").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(probe_dir / "checksums.sha256"),
        },
        "probe_weights": {
            "path": str((probe_dir / "probe_weights.safetensors").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(probe_dir / "probe_weights.safetensors"),
        },
        "probe_run_metrics": {
            "path": str((probe_dir / "metrics.json").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(probe_dir / "metrics.json"),
        },
        "probe_predictions": {
            "path": str((probe_dir / "predictions.npz").relative_to(_REPO_ROOT)),
            "sha256": sha256_file(probe_dir / "predictions.npz"),
        },
    })
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scientific_status": (
            "frozen development teaching gallery; position-level inspection, not a population estimate"
        ),
        "selection": {
            "split": "development",
            "rule": "lowest position_id in the immutable public-v2 development split",
            "position_count": count,
            "position_ids": arrays["position_id"].astype(str).tolist(),
        },
        "representation": {
            "boundary": "resid_post_after_ln",
            "shape_semantics": "[position, layer, side-to-move square token, channel]",
            "stored_dtype": "float16",
            "source_compute_dtype": "float32",
            "channel_basis_warning": (
                "A residual channel is a basis coordinate, not a demonstrated chess feature."
            ),
            "black_token_orientation": (
                "BT4 square tokens are vertically mirrored for black to move; course views map back to absolute board squares."
            ),
        },
        "policy_lens": {
            "boundary": "resid_post_after_ln",
            "readout": "native nonlinear attention-policy head at every layer",
            "arm_semantics": "first letter encoder, second letter policy head",
            "final_parity_max_abs_error": final_errors,
            "final_parity_atol": parity_atol,
            "stored_dtype": "float16 after float32 parity validation",
        },
        "target_gradient": {
            "objective": "target log-odds on the fixed original legal-action support",
            "layers": list(selected_gradient_layers),
            "models": ["raw with raw head", "hero with hero head"],
            "boundary": "resid_post_after_ln",
            "interpretation": (
                "per-square norm and gradient-times-state reductions of one full "
                "endpoint-specific VJP; not a fitted corpus-average J-Lens and not "
                "a causal intervention"
            ),
        },
        "activation_spectrum": {
            "layers": list(spectrum_layers),
            "observations": count * 64,
            "operation": (
                "exact singular values of centered position-square residual rows "
                "within the finite gallery"
            ),
            "interpretation": (
                "descriptive variance geometry; effective rank is not evidence of "
                "superposition, feature count, or monosemanticity"
            ),
        },
        "probes": {
            "tasks": list(PROBE_TASKS),
            "weights_used_for_extraction_only": True,
            "predictions_stored_after_full-precision_probe_evaluation": True,
            **probe_contract,
        },
        "summary": {
            "paired_layer_metrics": layer_metrics,
            "positions": _position_summaries(arrays),
        },
        "data": {
            "path": DEFAULT_DATA_NAME,
            "sha256": data_sha256,
            "size_bytes": data_path.stat().st_size,
            "arrays": _array_inventory(arrays),
        },
        "sources": source_files,
        "source_public_corpus_integrity": corpus_record["manifest_integrity"]["sha256"],
        "models": models.descriptor(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python_chess": chess.__version__,
            "safetensors": safetensors.__version__,
            "device": str(target_device),
            "device_name": (
                torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda"
                else platform.processor() or "CPU"
            ),
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(target_device))
                if target_device.type == "cuda"
                else None
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "redistribution": {
            "checkpoints_bundled": False,
            "bulk_corpus_bundled": False,
            "artifact_class": "derived activation tensors and predictions",
            "standalone_release_gate": (
                "Human review of source-model and source-position redistribution terms is required."
            ),
        },
    }
    identity_payload = {
        "schema_version": payload["schema_version"],
        "selection": payload["selection"],
        "data_sha256": data_sha256,
        "sources": source_files,
    }
    payload["run_id"] = "sha256:" + hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest()
    base = dict(payload)
    payload["manifest_integrity"] = {
        "algorithm": "sha256-canonical-json-without-manifest_integrity",
        "sha256": hashlib.sha256(_canonical_bytes(base)).hexdigest(),
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--raw-bt4", type=Path, default=DEFAULT_RAW_BT4)
    parser.add_argument("--hero-checkpoint", type=Path, default=DEFAULT_HERO)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_activation_gallery(
        corpus_manifest=args.corpus_manifest,
        raw_bt4=args.raw_bt4,
        hero_checkpoint=args.hero_checkpoint,
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        count=args.count,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "positions": manifest["selection"]["position_count"],
                "data_sha256": manifest["data"]["sha256"],
                "size_bytes": manifest["data"]["size_bytes"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
