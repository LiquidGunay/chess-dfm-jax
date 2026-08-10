"""Run a guarded Raw-vs-Hero pilot spanning the chess-interpretability literature.

The run is deliberately development-only.  It combines depth-wise policy
readout, exact attention decomposition, bidirectional causal patching, paired
representation deltas, and action-specific attribution in one immutable
bundle.  Large confirmatory sweeps use the same functions through Modal only
after this local correctness gate passes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from torch import Tensor

from research.interpretability.artifacts import (
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from research.interpretability.attention import (
    attention_internals,
    attention_move_alignment,
    future_move_attention_alignment,
    qk_smolgen_balance,
)
from research.interpretability.attribution import (
    integrated_gradients,
    input_gradient_attribution,
    legal_piece_removal_variants,
    sarfa_scores,
    smoothgrad,
)
from research.interpretability.chess_concepts import canonical_square, reconstruct_puzzle
from research.interpretability.chessbench import (
    DEFAULT_EXTERNAL_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_DIR,
    load_public_corpus,
)
from research.interpretability.logit_lens import (
    cross_head_lenses,
    first_emergence_layer,
    logit_lens_metrics,
    summarize_lens_metrics,
    validate_final_lens,
)
from research.interpretability.models import ARM_IDS, ArmId, load_bt4_comparison_models
from research.interpretability.paired_diff import DeltaCovarianceAccumulator
from research.interpretability.patching import (
    legal_log_probabilities,
    patch_comparison_boundary,
    summarize_patch_metrics,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_BT4 = _REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz"
DEFAULT_HERO_CHECKPOINT = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint"
DEFAULT_PUZZLE_SOURCE = DEFAULT_EXTERNAL_DIR / "puzzles.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "research/analysis/raw_hero_literature_pilot_v1"
ALL_LAYERS = tuple(range(15))
DEFAULT_CAUSAL_LAYERS = (0, 7, 14)
PATCH_DIRECTIONS: tuple[tuple[str, ArmId, ArmId], ...] = (
    ("hero_to_raw_raw_head", "HR", "RR"),
    ("raw_to_hero_raw_head", "RR", "HR"),
    ("hero_to_raw_hero_head", "HH", "RH"),
    ("raw_to_hero_hero_head", "RH", "HH"),
)


def bootstrap_mean_interval(
    values: Tensor | np.ndarray | Sequence[float],
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260808,
) -> dict[str, float | int]:
    """Deterministic percentile bootstrap interval for a finite sample mean."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a non-empty finite vector")
    if type(samples) is not int or samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must lie in (0, 1)")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, array.size, size=(samples, array.size))
    means = array[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
    }


def legal_policy_js(
    first_logits: Tensor,
    second_logits: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
) -> Tensor:
    """Jensen-Shannon divergence on each position's exact legal support."""

    first_logp, mask = legal_log_probabilities(first_logits, legal_indices, legal_counts)
    second_logp, second_mask = legal_log_probabilities(
        second_logits,
        legal_indices,
        legal_counts,
    )
    if not torch.equal(mask, second_mask):  # pragma: no cover - shared arguments
        raise AssertionError("Legal masks unexpectedly differ")
    first = first_logp.exp().masked_fill(~mask, 0.0)
    second = second_logp.exp().masked_fill(~mask, 0.0)
    midpoint = 0.5 * (first + second)
    midpoint_log = torch.where(mask, midpoint.clamp_min(1e-30).log(), 0.0)
    return (
        0.5
        * (
            torch.where(mask, first * (first_logp - midpoint_log), 0.0).sum(dim=1)
            + torch.where(mask, second * (second_logp - midpoint_log), 0.0).sum(dim=1)
        )
    ).clamp_min(0.0)


def map_agreement(first: Tensor, second: Tensor, *, top_k: int = 8) -> dict[str, Any]:
    """Compare two finite non-negative square saliency maps."""

    first = first.detach().cpu().double().reshape(-1)
    second = second.detach().cpu().double().reshape(-1)
    if first.shape != second.shape or first.numel() == 0:
        raise ValueError("Saliency maps must be aligned non-empty vectors")
    if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
        raise ValueError("Saliency maps must be finite")
    if bool((first < 0).any()) or bool((second < 0).any()):
        raise ValueError("Saliency magnitudes must be non-negative")
    if type(top_k) is not int or not 1 <= top_k <= first.numel():
        raise ValueError("top_k lies outside the map")

    first_centered = first - first.mean()
    second_centered = second - second.mean()
    pearson_denominator = first_centered.norm() * second_centered.norm()
    cosine_denominator = first.norm() * second.norm()
    first_top = set(torch.topk(first, top_k).indices.tolist())
    second_top = set(torch.topk(second, top_k).indices.tolist())
    return {
        "pearson": (
            float(torch.dot(first_centered, second_centered) / pearson_denominator)
            if float(pearson_denominator) > 0.0
            else 0.0
        ),
        "cosine": (
            float(torch.dot(first, second) / cosine_denominator)
            if float(cosine_denominator) > 0.0
            else 0.0
        ),
        "top_k": top_k,
        "top_k_jaccard": len(first_top & second_top) / len(first_top | second_top),
        "argmax_agreement": int(first.argmax()) == int(second.argmax()),
    }


def _finite_vector_summary(value: Tensor) -> dict[str, Any]:
    vector = value.detach().cpu().double().reshape(-1)
    finite = torch.isfinite(vector)
    retained = vector[finite]
    return {
        "count": int(vector.numel()),
        "finite_count": int(finite.sum()),
        "mean": float(retained.mean()) if retained.numel() else None,
        "median": float(retained.median()) if retained.numel() else None,
        "minimum": float(retained.min()) if retained.numel() else None,
        "maximum": float(retained.max()) if retained.numel() else None,
    }


def _head_summary(value: Tensor, *, largest: bool = True) -> dict[str, Any]:
    """Compact position-by-head statistics without retaining attention tensors."""

    if value.ndim != 2 or value.shape[1] != 32:
        raise ValueError("Head statistics must have shape [position, 32]")
    means = torch.nanmean(value.detach().cpu().float(), dim=0)
    if not bool(torch.isfinite(means).all()):
        raise ValueError("Every attention head needs at least one finite observation")
    order = torch.argsort(means, descending=largest)
    return {
        "per_head_mean": means.tolist(),
        "top_5_heads": [{"head": int(head), "mean": float(means[head])} for head in order[:5]],
        "global_mean": float(means.mean()),
    }


def _square_map_record(
    magnitude: Tensor,
    *,
    origin: int,
    destination: int,
    board: chess.Board,
) -> dict[str, Any]:
    value = magnitude.detach().cpu().double().reshape(-1)
    if value.numel() != 64 or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("Square attribution must be a finite non-negative 64-vector")
    total = value.sum()
    normalized = value / total if float(total) > 0.0 else torch.zeros_like(value)
    top = torch.argsort(normalized, descending=True)[:8]

    def square_record(token: int) -> dict[str, Any]:
        absolute = token if board.turn == chess.WHITE else chess.square_mirror(token)
        return {
            "token": token,
            "token_name": chess.square_name(token),
            "absolute_name": chess.square_name(absolute),
            "mass": float(normalized[token]),
        }

    return {
        "normalized_mass": normalized.tolist(),
        "total_magnitude": float(total),
        "origin_destination_mass": float(normalized[origin] + normalized[destination]),
        "top_8_mass": float(normalized[top].sum()),
        "top_8": [square_record(int(token)) for token in top],
    }


def _selected_rows(puzzles: Mapping[str, np.ndarray], count: int) -> np.ndarray:
    development = np.flatnonzero(puzzles["split_u8"] == 1)
    if development.size < count:
        raise ValueError("Public development split is smaller than requested pilot")
    identifiers = puzzles["position_id"][development]
    return development[np.argsort(identifiers, kind="stable")[:count]]


def _puzzle_source_rows(path: Path, puzzle_ids: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.resolve(strict=True).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"PuzzleId", "PGN", "FEN", "Moves"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Puzzle CSV does not contain the required reconstruction columns")
        for row in reader:
            puzzle_id = row["PuzzleId"]
            if puzzle_id in puzzle_ids:
                if puzzle_id in result:
                    raise ValueError(f"Duplicate puzzle id in source: {puzzle_id}")
                result[puzzle_id] = row
                if len(result) == len(puzzle_ids):
                    break
    missing = sorted(puzzle_ids - set(result))
    if missing:
        raise ValueError(f"Puzzle source is missing selected ids: {missing}")
    return result


def _git_record() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
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
        "status": status.splitlines(),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return result


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


def _merge_tensor_chunks(
    chunks: Mapping[str, list[Tensor]], *, dimension: int
) -> dict[str, Tensor]:
    if not chunks or any(not values for values in chunks.values()):
        raise ValueError("Metric chunks are incomplete")
    return {name: torch.cat(values, dim=dimension) for name, values in chunks.items()}


def _attribution_methods(
    models: Any,
    *,
    arm: ArmId,
    planes: Tensor,
    baseline: Tensor,
    targets: Tensor,
    legal: Tensor,
    legal_count: Tensor,
    ig_steps: int,
    smoothgrad_samples: int,
    seed: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    def forward(value: Tensor) -> Tensor:
        return models.policy_logits(value, arm=arm, compute_dtype=torch.float32).logits

    raw_gradient = input_gradient_attribution(
        forward,
        planes,
        targets,
        legal_indices=legal,
        legal_counts=legal_count,
    )
    gradient_x_reference = input_gradient_attribution(
        forward,
        planes,
        targets,
        legal_indices=legal,
        legal_counts=legal_count,
        multiply_by_input=True,
        reference=baseline,
    )
    integrated = integrated_gradients(
        forward,
        planes,
        targets,
        baseline.unsqueeze(0),
        legal_indices=legal,
        legal_counts=legal_count,
        steps=ig_steps,
    )
    smoothed = smoothgrad(
        forward,
        planes,
        targets,
        legal_indices=legal,
        legal_counts=legal_count,
        samples=smoothgrad_samples,
        noise_std=0.05,
        seed=seed,
    )
    maps = {
        "input_gradient": raw_gradient.square_magnitude,
        "gradient_x_reference": gradient_x_reference.square_magnitude,
        "integrated_gradients": integrated.square_magnitude,
        "smoothgrad": smoothed.square_magnitude,
    }
    diagnostics = {
        "objective": "target_log_odds_on_fixed_original_legal_support",
        "integrated_gradients_steps": ig_steps,
        "smoothgrad_samples": smoothgrad_samples,
        "smoothgrad_noise_std": 0.05,
        "integrated_gradients_completeness": {
            "residual": integrated.completeness_residual.detach().cpu().tolist(),
            "relative_error": integrated.relative_completeness_error.detach().cpu().tolist(),
        },
    }
    return maps, diagnostics


def run_literature_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if args.position_count <= 0 or args.batch_size <= 0 or args.attribution_count <= 0:
        raise ValueError("position_count, batch_size, and attribution_count must be positive")
    if args.attribution_count > args.position_count:
        raise ValueError("attribution_count cannot exceed position_count")
    causal_layers = tuple(args.causal_layers)
    if not causal_layers or tuple(sorted(set(causal_layers))) != causal_layers:
        raise ValueError("causal_layers must be unique and increasing")
    if causal_layers[0] < 0 or causal_layers[-1] >= len(ALL_LAYERS):
        raise ValueError("causal_layers lie outside the BT4 encoder")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    target, staging = _output_staging(args.output)
    started = time.monotonic()
    started_utc = datetime.now(UTC)
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"{time.monotonic() - started:10.3f}s {message}"
        print(line, flush=True)
        log_lines.append(line)

    try:
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)

        log("strict-loading corrected public corpus v2")
        corpus, corpus_manifest = load_public_corpus(args.corpus_manifest)
        puzzles = corpus["puzzles"]
        rows = _selected_rows(puzzles, args.position_count)
        log("strict-loading Raw/Hero comparison lattice")
        models = load_bt4_comparison_models(
            raw_bt4_path=args.raw_bt4,
            hero_checkpoint_dir=args.hero_checkpoint,
            device=device,
        )
        for encoder in (models.raw_encoder, models.hero_encoder):
            encoder.requires_grad_(False)

        lens_chunks: dict[str, dict[str, list[Tensor]]] = {arm: {} for arm in ARM_IDS}
        final_parity: dict[str, dict[str, Any]] = {
            arm: {"passed": True, "maximum_absolute_error": 0.0, "rms_errors": []}
            for arm in ARM_IDS
        }
        attention_chunks: dict[str, dict[int, dict[str, list[Tensor]]]] = {
            name: {layer: {} for layer in causal_layers} for name in ("raw", "hero")
        }
        attention_pattern_js: dict[int, list[Tensor]] = {layer: [] for layer in causal_layers}
        attention_contribution_delta: dict[int, list[Tensor]] = {
            layer: [] for layer in causal_layers
        }
        attention_reconstruction_error: dict[str, dict[int, float]] = {
            name: {layer: 0.0 for layer in causal_layers} for name in ("raw", "hero")
        }
        delta_accumulators = {layer: DeltaCovarianceAccumulator(1024) for layer in causal_layers}
        behavioral_js: list[Tensor] = []

        log("running all-layer cross-head lenses, attention decomposition, and delta atlas")
        for start in range(0, rows.size, args.batch_size):
            selected = rows[start : start + args.batch_size]
            planes = torch.from_numpy(puzzles["current_planes_u8"][selected]).to(device)
            legal = torch.from_numpy(puzzles["legal_idx_u16"][selected]).to(device)
            legal_count = torch.from_numpy(puzzles["legal_count_u16"][selected]).to(device)
            targets = torch.from_numpy(puzzles["target_action_u16"][selected]).to(device)
            origins = torch.from_numpy(puzzles["target_origin_token_i16"][selected]).to(device)
            destinations = torch.from_numpy(puzzles["target_destination_token_i16"][selected]).to(
                device
            )
            future_origins = torch.from_numpy(puzzles["future_origin_token_root_i16"][selected]).to(
                device
            )
            future_destinations = torch.from_numpy(
                puzzles["future_destination_token_root_i16"][selected]
            ).to(device)
            future_valid = torch.from_numpy(puzzles["future_valid_u8"][selected]).to(device)
            with torch.inference_mode():
                lattice = models.lattice_logits(
                    planes,
                    compute_dtype=torch.float32,
                    capture_layers=ALL_LAYERS,
                )
                if lattice.raw_captures is None or lattice.hero_captures is None:
                    raise AssertionError("All-layer run lost encoder captures")
                raw_head = models.raw_encoder.policy_head
                hero_head = models.hero_encoder.policy_head
                if raw_head is None or hero_head is None:
                    raise AssertionError("Comparison policy head disappeared")
                lenses = cross_head_lenses(
                    lattice.raw_captures,
                    lattice.hero_captures,
                    raw_head,
                    hero_head,
                    compute_dtype=torch.float32,
                )
                for arm in ARM_IDS:
                    metrics = logit_lens_metrics(
                        lenses[arm],
                        legal,
                        legal_count,
                        targets,
                    )
                    for name, value in metrics.items():
                        lens_chunks[arm].setdefault(name, []).append(value.cpu())
                    parity = validate_final_lens(
                        lenses[arm],
                        lattice.logits[arm],
                        atol=args.parity_atol,
                    )
                    final_parity[arm]["passed"] &= parity["passed"]
                    final_parity[arm]["maximum_absolute_error"] = max(
                        final_parity[arm]["maximum_absolute_error"],
                        parity["maximum_absolute_error"],
                    )
                    final_parity[arm]["rms_errors"].append(parity["rms_error"])

                raw_hero_js = legal_policy_js(
                    lattice.logits["RR"],
                    lattice.logits["HR"],
                    legal,
                    legal_count,
                )
                behavioral_js.append(raw_hero_js.cpu())
                capture_offset = {
                    layer: lattice.raw_captures.layer_indices.index(layer)
                    for layer in causal_layers
                }
                for layer in causal_layers:
                    offset = capture_offset[layer]
                    raw_internal = attention_internals(
                        models.raw_encoder.layers[layer],
                        lattice.raw_captures.hook_attn_in[offset],
                        compute_dtype=torch.float32,
                    )
                    hero_internal = attention_internals(
                        models.hero_encoder.layers[layer],
                        lattice.hero_captures.hook_attn_in[offset],
                        compute_dtype=torch.float32,
                    )
                    for name, internal in (("raw", raw_internal), ("hero", hero_internal)):
                        alignment = attention_move_alignment(
                            internal.pattern,
                            origins,
                            destinations,
                        )
                        future_alignment = future_move_attention_alignment(
                            internal.pattern,
                            future_origins,
                            future_destinations,
                            future_valid,
                        )
                        balance = qk_smolgen_balance(internal)
                        values = {
                            **alignment,
                            **{f"balance_{key}": value for key, value in balance.items()},
                            "future_symmetric_endpoint_mass": future_alignment[
                                "symmetric_endpoint_mass"
                            ],
                            "future_origin_to_destination_top1": future_alignment[
                                "origin_to_destination_top1"
                            ],
                        }
                        for metric, value in values.items():
                            attention_chunks[name][layer].setdefault(metric, []).append(
                                value.detach().cpu()
                            )
                        attention_reconstruction_error[name][layer] = max(
                            attention_reconstruction_error[name][layer],
                            internal.contribution_reconstruction_max_abs_error,
                        )
                    midpoint = 0.5 * (raw_internal.pattern + hero_internal.pattern)
                    raw_log = raw_internal.pattern.clamp_min(1e-30).log()
                    hero_log = hero_internal.pattern.clamp_min(1e-30).log()
                    midpoint_log = midpoint.clamp_min(1e-30).log()
                    pattern_js = 0.5 * (
                        (raw_internal.pattern * (raw_log - midpoint_log)).sum(dim=-1)
                        + (hero_internal.pattern * (hero_log - midpoint_log)).sum(dim=-1)
                    ).mean(dim=-1)
                    attention_pattern_js[layer].append(pattern_js.cpu())
                    contribution_delta = (
                        (
                            raw_internal.projected_head_contributions
                            - hero_internal.projected_head_contributions
                        )
                        .float()
                        .square()
                        .mean(dim=(-2, -1))
                        .sqrt()
                    )
                    attention_contribution_delta[layer].append(contribution_delta.cpu())
                    delta_accumulators[layer].update(
                        lattice.raw_captures.resid_post_after_ln[offset],
                        lattice.hero_captures.resid_post_after_ln[offset],
                        behavior_delta=raw_hero_js,
                    )
            log(f"dense methods processed {min(start + args.batch_size, rows.size)}/{rows.size}")

        lens_metrics: dict[str, Any] = {}
        for arm in ARM_IDS:
            merged = _merge_tensor_chunks(lens_chunks[arm], dimension=1)
            summary = summarize_lens_metrics(merged, ALL_LAYERS)
            probability_emergence = first_emergence_layer(
                merged["target_probability"],
                ALL_LAYERS,
                threshold=0.5,
            )
            top1_emergence = first_emergence_layer(
                merged["top1_accuracy"],
                ALL_LAYERS,
                threshold=1.0,
            )
            summary["first_target_probability_ge_0.5"] = _finite_vector_summary(
                probability_emergence.float()
            )
            summary["first_top1_occurrence"] = _finite_vector_summary(top1_emergence.float())
            summary["final_target_probability_bootstrap"] = bootstrap_mean_interval(
                merged["target_probability"][-1],
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            parity = final_parity[arm]
            parity["rms_error"] = float(np.mean(parity.pop("rms_errors")))
            summary["final_lens_parity"] = parity
            lens_metrics[arm] = summary

        attention_metrics: dict[str, Any] = {"raw": {}, "hero": {}, "raw_hero": {}}
        for name in ("raw", "hero"):
            for layer in causal_layers:
                merged = _merge_tensor_chunks(attention_chunks[name][layer], dimension=0)
                per_layer: dict[str, Any] = {
                    "contribution_reconstruction_max_abs_error": (
                        attention_reconstruction_error[name][layer]
                    )
                }
                for metric, value in merged.items():
                    if value.ndim == 2:
                        per_layer[metric] = _head_summary(value)
                    elif value.ndim == 3:
                        per_ply: list[dict[str, Any]] = []
                        for ply in range(value.shape[1]):
                            ply_values = value[:, ply]
                            valid_positions = torch.isfinite(ply_values).all(dim=1)
                            if bool(valid_positions.any()):
                                report = _head_summary(ply_values[valid_positions])
                                report["available_position_count"] = int(valid_positions.sum())
                                report["available"] = True
                            else:
                                report = {
                                    "available": False,
                                    "available_position_count": 0,
                                    "reason": "no selected solution line reaches this ply",
                                }
                            per_ply.append(report)
                        per_layer[metric] = {"per_future_ply": per_ply}
                    else:
                        raise ValueError(f"Unexpected attention metric shape for {metric}")
                attention_metrics[name][str(layer)] = per_layer
        for layer in causal_layers:
            pattern_js = torch.cat(attention_pattern_js[layer])
            contribution = torch.cat(attention_contribution_delta[layer])
            attention_metrics["raw_hero"][str(layer)] = {
                "pattern_js": _head_summary(pattern_js),
                "projected_head_contribution_delta_rms": _head_summary(contribution),
            }

        delta_metrics: dict[str, Any] = {}
        for layer, accumulator in delta_accumulators.items():
            summary, _subspace = accumulator.finalize(top_component_count=16)
            delta_metrics[str(layer)] = summary

        log("running bidirectional whole-residual causal patching and identity control")
        patch_chunks: dict[str, dict[int, dict[str, list[Tensor]]]] = {
            name: {layer: {} for layer in causal_layers}
            for name, _source, _dest in PATCH_DIRECTIONS
        }
        noop_chunks: dict[str, list[Tensor]] = {}
        for start in range(0, rows.size, args.batch_size):
            selected = rows[start : start + args.batch_size]
            planes = torch.from_numpy(puzzles["current_planes_u8"][selected]).to(device)
            legal = torch.from_numpy(puzzles["legal_idx_u16"][selected]).to(device)
            legal_count = torch.from_numpy(puzzles["legal_count_u16"][selected]).to(device)
            targets = torch.from_numpy(puzzles["target_action_u16"][selected]).to(device)
            for name, source_arm, destination_arm in PATCH_DIRECTIONS:
                for layer in causal_layers:
                    result = patch_comparison_boundary(
                        models,
                        planes,
                        source_arm=source_arm,
                        destination_arm=destination_arm,
                        layer=layer,
                        boundary="resid_post_after_ln",
                        compute_dtype=torch.float32,
                        legal_indices=legal,
                        legal_counts=legal_count,
                        targets=targets,
                    )
                    for metric, value in result.metrics.items():
                        patch_chunks[name][layer].setdefault(metric, []).append(value.cpu())
            noop = patch_comparison_boundary(
                models,
                planes,
                source_arm="RR",
                destination_arm="RR",
                layer=causal_layers[len(causal_layers) // 2],
                boundary="resid_post_after_ln",
                compute_dtype=torch.float32,
                legal_indices=legal,
                legal_counts=legal_count,
                targets=targets,
            )
            for metric, value in noop.metrics.items():
                noop_chunks.setdefault(metric, []).append(value.cpu())
            log(f"causal methods processed {min(start + args.batch_size, rows.size)}/{rows.size}")

        patch_metrics: dict[str, Any] = {}
        for name, _source, _destination in PATCH_DIRECTIONS:
            patch_metrics[name] = {}
            for layer in causal_layers:
                merged = _merge_tensor_chunks(patch_chunks[name][layer], dimension=0)
                report = summarize_patch_metrics(merged)
                projection = merged["distribution_delta_projection"]
                projection = projection[torch.isfinite(projection)]
                if projection.numel():
                    report["distribution_delta_projection_bootstrap"] = bootstrap_mean_interval(
                        projection,
                        samples=args.bootstrap_samples,
                        seed=args.seed + layer,
                    )
                patch_metrics[name][str(layer)] = report
        noop_metrics = summarize_patch_metrics(_merge_tensor_chunks(noop_chunks, dimension=0))

        log("running exact action attribution and chess-aware SARFA perturbations")
        selected_puzzle_ids = {
            str(puzzles["puzzle_id"][row]) for row in rows[: args.attribution_count]
        }
        source_rows = _puzzle_source_rows(args.puzzle_source, selected_puzzle_ids)
        attribution_examples: list[dict[str, Any]] = []
        attribution_agreements: dict[str, list[float]] = {}
        reconstruction_passed = True
        completeness_errors: list[float] = []
        for attribution_offset, row in enumerate(rows[: args.attribution_count]):
            puzzle_id = str(puzzles["puzzle_id"][row])
            source = source_rows[puzzle_id]
            reconstructed = reconstruct_puzzle(
                puzzle_id=puzzle_id,
                pgn=source["PGN"],
                fen_before_opponent=source["FEN"],
                moves=source["Moves"],
            )
            root_planes = puzzles["current_planes_u8"][row]
            from chess_dfm_jax.encoding import encode_board

            rebuilt = encode_board(reconstructed.board, reconstructed.history)
            exact_reconstruction = bool(np.array_equal(rebuilt, root_planes))
            reconstruction_passed &= exact_reconstruction
            planes = (
                torch.from_numpy(root_planes).to(device=device, dtype=torch.float32).unsqueeze(0)
            )
            baseline_row = rows[(attribution_offset + 1) % rows.size]
            baseline = (
                torch.from_numpy(puzzles["current_planes_u8"][baseline_row])
                .to(
                    device=device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
            )
            legal = torch.from_numpy(puzzles["legal_idx_u16"][row]).to(device).unsqueeze(0)
            legal_count = torch.as_tensor(
                [int(puzzles["legal_count_u16"][row])],
                device=device,
            )
            targets = torch.as_tensor(
                [int(puzzles["target_action_u16"][row])],
                device=device,
            )
            origin = int(puzzles["target_origin_token_i16"][row])
            destination = int(puzzles["target_destination_token_i16"][row])
            example: dict[str, Any] = {
                "position_id": str(puzzles["position_id"][row]),
                "puzzle_id": puzzle_id,
                "root_fen": str(puzzles["root_fen"][row]),
                "exact_plane_reconstruction": exact_reconstruction,
                "baseline_position_id": str(puzzles["position_id"][baseline_row]),
                "target_origin_token": origin,
                "target_destination_token": destination,
                "arms": {},
                "raw_hero_map_agreement": {},
            }
            maps_by_arm: dict[str, dict[str, Tensor]] = {}
            diagnostics_by_arm: dict[str, Any] = {}
            for arm in ("RR", "HH"):
                maps, diagnostics = _attribution_methods(
                    models,
                    arm=arm,
                    planes=planes,
                    baseline=baseline,
                    targets=targets,
                    legal=legal,
                    legal_count=legal_count,
                    ig_steps=args.ig_steps,
                    smoothgrad_samples=args.smoothgrad_samples,
                    seed=args.seed + attribution_offset,
                )
                maps_by_arm[arm] = maps
                diagnostics_by_arm[arm] = diagnostics
                completeness_errors.extend(
                    np.asarray(diagnostics["integrated_gradients_completeness"]["relative_error"])
                    .reshape(-1)
                    .tolist()
                )
                example["arms"][arm] = {
                    "diagnostics": diagnostics,
                    "maps": {
                        method: _square_map_record(
                            value[0],
                            origin=origin,
                            destination=destination,
                            board=reconstructed.board,
                        )
                        for method, value in maps.items()
                    },
                }

            removal = legal_piece_removal_variants(
                reconstructed.board,
                reconstructed.history,
                target_move=reconstructed.solution_moves[0],
            )
            eligible_absolute = np.flatnonzero(removal.eligible)
            if eligible_absolute.size == 0:
                raise ValueError(f"Puzzle {puzzle_id} has no eligible SARFA perturbation")
            variants = torch.from_numpy(removal.planes[eligible_absolute]).to(
                device=device,
                dtype=torch.float32,
            )
            eligible = torch.ones(
                (1, eligible_absolute.size),
                device=device,
                dtype=torch.bool,
            )
            target_legal = (
                torch.from_numpy(removal.target_legal[eligible_absolute])
                .to(
                    device=device,
                    dtype=torch.bool,
                )
                .unsqueeze(0)
            )
            sarfa_maps: dict[str, Tensor] = {}
            for arm in ("RR", "HH"):
                with torch.inference_mode():
                    original = models.policy_logits(
                        planes,
                        arm=arm,
                        compute_dtype=torch.float32,
                    ).logits
                    chunks: list[Tensor] = []
                    for start in range(0, variants.shape[0], args.perturbation_batch_size):
                        chunks.append(
                            models.policy_logits(
                                variants[start : start + args.perturbation_batch_size],
                                arm=arm,
                                compute_dtype=torch.float32,
                            ).logits
                        )
                    perturbed = torch.cat(chunks).unsqueeze(0)
                    sarfa = sarfa_scores(
                        original,
                        perturbed,
                        legal,
                        legal_count,
                        targets,
                        eligible=eligible,
                        target_legal=target_legal,
                    )
                token_map = torch.zeros(64, device=device, dtype=torch.float32)
                for index, absolute in enumerate(eligible_absolute.tolist()):
                    token = canonical_square(reconstructed.board, absolute)
                    token_map[token] = sarfa["saliency"][0, index]
                sarfa_maps[arm] = token_map.unsqueeze(0)
                example["arms"][arm]["maps"]["sarfa_piece_removal"] = _square_map_record(
                    token_map,
                    origin=origin,
                    destination=destination,
                    board=reconstructed.board,
                )
                example["arms"][arm]["sarfa"] = {
                    "eligible_piece_count": int(eligible_absolute.size),
                    "target_invalidated_count": int((~target_legal).sum()),
                    "contract": removal.contract,
                }

            for method in (*maps_by_arm["RR"], "sarfa_piece_removal"):
                raw_map = (
                    sarfa_maps["RR"]
                    if method == "sarfa_piece_removal"
                    else maps_by_arm["RR"][method]
                )
                hero_map = (
                    sarfa_maps["HH"]
                    if method == "sarfa_piece_removal"
                    else maps_by_arm["HH"][method]
                )
                agreement = map_agreement(raw_map, hero_map)
                example["raw_hero_map_agreement"][method] = agreement
                for key, value in agreement.items():
                    if isinstance(value, bool):
                        numeric = float(value)
                    elif isinstance(value, (int, float)) and key != "top_k":
                        numeric = float(value)
                    else:
                        continue
                    attribution_agreements.setdefault(f"{method}.{key}", []).append(numeric)
            attribution_examples.append(example)
            del planes, baseline, variants, maps_by_arm, sarfa_maps
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            log(f"attribution processed {attribution_offset + 1}/{args.attribution_count}")

        attribution_metrics = {
            "position_count": args.attribution_count,
            "exact_plane_reconstruction_gate": reconstruction_passed,
            "integrated_gradients_relative_completeness_error": (
                bootstrap_mean_interval(
                    completeness_errors,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            ),
            "raw_hero_map_agreement": {
                name: bootstrap_mean_interval(
                    values,
                    samples=args.bootstrap_samples,
                    seed=args.seed + index,
                )
                for index, (name, values) in enumerate(sorted(attribution_agreements.items()))
            },
        }

        final_layer = str(causal_layers[-1])
        final_patch_errors: dict[str, float | None] = {}
        for name, _source, _destination in PATCH_DIRECTIONS:
            summary = patch_metrics[name][final_layer]["distribution_delta_projection"]
            lower = summary["minimum"]
            upper = summary["maximum"]
            final_patch_errors[name] = (
                None
                if lower is None or upper is None
                else max(abs(float(lower) - 1.0), abs(float(upper) - 1.0))
            )
        attention_max_error = max(
            error
            for per_model in attention_reconstruction_error.values()
            for error in per_model.values()
        )
        noop_js = noop_metrics["baseline_to_patched_js"]["maximum"]
        gates = {
            "all_final_lens_parity": all(
                lens_metrics[arm]["final_lens_parity"]["passed"] for arm in ARM_IDS
            ),
            "attention_head_sum_reconstruction_le_tolerance": (
                attention_max_error <= args.attention_reconstruction_atol
            ),
            "identity_patch_js_le_tolerance": (
                noop_js is not None and float(noop_js) <= args.identity_patch_js_atol
            ),
            "final_layer_patch_reaches_source": all(
                error is not None and error <= args.final_patch_projection_atol
                for error in final_patch_errors.values()
            ),
            "exact_puzzle_plane_reconstruction": reconstruction_passed,
            "all_reported_metrics_finite": True,
        }
        gates["passed"] = all(gates.values())
        metrics = {
            "schema_version": "bt4-raw-hero-literature-pilot-metrics-v1",
            "scope": "exploratory development-only pilot",
            "position_count": int(rows.size),
            "attribution_position_count": args.attribution_count,
            "logit_lens": lens_metrics,
            "attention": attention_metrics,
            "paired_delta": delta_metrics,
            "causal_patching": patch_metrics,
            "identity_patch_control": noop_metrics,
            "attribution": attribution_metrics,
            "behavior": {
                "raw_vs_hero_fixed_raw_head_policy_js": bootstrap_mean_interval(
                    torch.cat(behavioral_js),
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            },
            "correctness_gates": {
                **gates,
                "observed": {
                    "attention_reconstruction_max_abs_error": attention_max_error,
                    "identity_patch_max_js": noop_js,
                    "final_patch_projection_absolute_error": final_patch_errors,
                },
            },
        }
        write_json_atomic(staging / "metrics.json", metrics)
        write_jsonl_atomic(staging / "attribution_examples.jsonl", attribution_examples)

        elapsed = time.monotonic() - started
        position_digest = hashlib.sha256(
            "\n".join(str(puzzles["position_id"][row]) for row in rows).encode()
        ).hexdigest()
        payloads = {
            "metrics": content_identity(metrics),
            "attribution_examples": content_identity({"rows": attribution_examples}),
            "positions": position_digest,
            "raw": models.raw_manifest["raw_asset"]["sha256"],
            "hero": models.hero_manifest["state"]["sha256"],
        }
        run_id = content_identity(payloads)
        manifest = {
            "schema_version": "bt4-raw-hero-literature-pilot-run-v1",
            "run_id": f"sha256:{run_id}",
            "started_utc": started_utc.isoformat(),
            "elapsed_seconds": elapsed,
            "git": _git_record(),
            "environment": _environment(device),
            "model_lattice": models.descriptor(),
            "public_corpus_manifest": {
                "path": str(args.corpus_manifest.resolve()),
                "integrity_sha256": corpus_manifest["manifest_integrity"]["sha256"],
                "split": "development only",
                "selection": "lowest position_id in development split",
                "position_digest_sha256": position_digest,
            },
            "puzzle_source": {
                "path": str(args.puzzle_source.resolve()),
                "sha256": sha256_file(args.puzzle_source),
            },
            "experiment": {
                "compute_dtype": "float32",
                "all_lens_layers": list(ALL_LAYERS),
                "causal_attention_delta_layers": list(causal_layers),
                "causal_boundary": "resid_post_after_ln",
                "patch_directions": [name for name, _source, _dest in PATCH_DIRECTIONS],
                "position_count": int(rows.size),
                "batch_size": args.batch_size,
                "attribution_count": args.attribution_count,
                "ig_steps": args.ig_steps,
                "smoothgrad_samples": args.smoothgrad_samples,
                "seed": args.seed,
                "confirmatory": False,
                "test_split_opened": False,
                "correctness_tolerances": {
                    "final_lens_absolute": args.parity_atol,
                    "attention_reconstruction_absolute": args.attention_reconstruction_atol,
                    "identity_patch_js": args.identity_patch_js_atol,
                    "final_patch_projection_absolute": args.final_patch_projection_atol,
                },
            },
            "cost": {"backend": "local GTX 1660 Ti", "provider_dollars": 0.0},
            "payload_identities": payloads,
            "source_files": {
                path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("attention.py"),
                    Path(__file__).with_name("attribution.py"),
                    Path(__file__).with_name("logit_lens.py"),
                    Path(__file__).with_name("paired_diff.py"),
                    Path(__file__).with_name("patching.py"),
                    _REPO_ROOT / "research/train_torch.py",
                )
            },
        }
        write_json_atomic(staging / "manifest.json", manifest)
        log(f"correctness gates passed={gates['passed']}")
        log_lines.append(f"run_id sha256:{run_id}")
        write_text_atomic(staging / "run.log", "\n".join(log_lines) + "\n")
        write_checksums(staging)
        verify_checksums(staging)
        os.replace(staging, target)
        checksums = verify_checksums(target)
        return {
            "output": str(target),
            "run_id": f"sha256:{run_id}",
            "metrics_sha256": checksums["metrics.json"],
            "gates_passed": gates["passed"],
            "elapsed_seconds": elapsed,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bt4", type=Path, default=DEFAULT_RAW_BT4)
    parser.add_argument("--hero-checkpoint", type=Path, default=DEFAULT_HERO_CHECKPOINT)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_DIR / "manifest.json",
    )
    parser.add_argument("--puzzle-source", type=Path, default=DEFAULT_PUZZLE_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--position-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--attribution-count", type=int, default=2)
    parser.add_argument("--perturbation-batch-size", type=int, default=4)
    parser.add_argument("--causal-layers", type=int, nargs="+", default=DEFAULT_CAUSAL_LAYERS)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--smoothgrad-samples", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--parity-atol", type=float, default=1e-5)
    parser.add_argument("--attention-reconstruction-atol", type=float, default=1e-4)
    parser.add_argument("--identity-patch-js-atol", type=float, default=1e-7)
    parser.add_argument("--final-patch-projection-atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_literature_pilot(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
