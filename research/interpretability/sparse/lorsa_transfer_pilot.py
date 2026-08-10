"""Audit a published LoRSA on Raw BT4, then freeze it for Raw -> Hero transfer.

The experiment is deliberately sequential.  It first proves that the converted
artifact is structurally compatible with, reconstructs, and safely replaces the
Raw BT4 attention branch.  Hero is evaluated only after every Raw source gate
passes.  No LoRSA parameter is trained or modified by this runner.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from research.interpretability.artifacts import (
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from research.interpretability.attention import attention_internals
from research.interpretability.chessbench import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_DIR,
    load_public_corpus,
)
from research.interpretability.models import BT4ComparisonModels, load_bt4_comparison_models
from research.interpretability.sparse.lorsa import (
    LoRSAConfig,
    LowRankSparseAttention,
    initialize_lorsa,
    load_converted_lorsa,
)
from research.interpretability.sparse.transcoder import SparseSupportAccumulator
from research.interpretability.sparse.transfer_pilot import (
    policy_replacement_metrics,
    summarize_policy_metrics,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_BT4 = _REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz"
DEFAULT_HERO_CHECKPOINT = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint"
DEFAULT_LORSA_DIR = _REPO_ROOT / "data/external/leela_saes/lorsa/L14"
DEFAULT_OUTPUT = _REPO_ROOT / "research/analysis/published_lorsa_l14_transfer_pilot_v1"
DEFAULT_LAYER = 14
UPSTREAM_COMMIT = "f946a5736f40397f5c834104aeac6e8e6ff753bd"


class ReconstructionAccumulator:
    """Exact float64 sufficient statistics for chunked reconstruction metrics."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.element_count = 0
        self.trailing_shape: tuple[int, ...] | None = None
        self.target_sum: Tensor | None = None
        self.sse = 0.0
        self.target_square_sum = 0.0
        self.reconstruction_square_sum = 0.0
        self.cross_sum = 0.0

    def update(self, target: Tensor, reconstruction: Tensor) -> None:
        if target.shape != reconstruction.shape or target.ndim < 1 or target.shape[0] == 0:
            raise ValueError("Reconstruction chunks must be aligned and non-empty")
        target64 = target.detach().double().cpu()
        reconstruction64 = reconstruction.detach().double().cpu()
        if not bool(torch.isfinite(target64).all() and torch.isfinite(reconstruction64).all()):
            raise ValueError("Reconstruction chunks contain nonfinite values")
        trailing = tuple(target64.shape[1:])
        if self.trailing_shape is None:
            self.trailing_shape = trailing
            self.target_sum = torch.zeros(trailing, dtype=torch.float64)
        elif trailing != self.trailing_shape:
            raise ValueError("Reconstruction chunk trailing shapes differ")
        if self.target_sum is None:  # pragma: no cover - guarded above
            raise AssertionError("Reconstruction accumulator was not initialized")
        error = reconstruction64 - target64
        self.sample_count += int(target64.shape[0])
        self.element_count += int(target64.numel())
        self.target_sum += target64.sum(dim=0)
        self.sse += float(error.square().sum())
        self.target_square_sum += float(target64.square().sum())
        self.reconstruction_square_sum += float(reconstruction64.square().sum())
        self.cross_sum += float((target64 * reconstruction64).sum())

    def finalize(self) -> dict[str, float]:
        if self.sample_count == 0 or self.target_sum is None:
            raise ValueError("Cannot finalize an empty reconstruction accumulator")
        target_energy = max(self.target_square_sum, 1e-24)
        centered_energy = max(
            self.target_square_sum
            - float(self.target_sum.square().sum()) / float(self.sample_count),
            1e-24,
        )
        cosine_denominator = math.sqrt(
            max(self.target_square_sum, 1e-24)
            * max(self.reconstruction_square_sum, 1e-24)
        )
        return {
            "mse": self.sse / self.element_count,
            "normalized_mse": self.sse / target_energy,
            "relative_l2": math.sqrt(self.sse / target_energy),
            "cosine": self.cross_sum / cosine_denominator,
            "explained_variance": 1.0 - self.sse / centered_energy,
            "target_rms": math.sqrt(self.target_square_sum / self.element_count),
            "reconstruction_rms": math.sqrt(
                self.reconstruction_square_sum / self.element_count
            ),
        }


def per_position_reconstruction_metrics(
    target: Tensor,
    reconstruction: Tensor,
) -> dict[str, Tensor]:
    if target.shape != reconstruction.shape or target.ndim < 2:
        raise ValueError("Per-position reconstruction tensors must align")
    target64 = target.detach().double().reshape(target.shape[0], -1)
    reconstruction64 = reconstruction.detach().double().reshape(target.shape[0], -1)
    error_energy = (reconstruction64 - target64).square().sum(dim=1)
    target_energy = target64.square().sum(dim=1).clamp_min(1e-24)
    return {
        "normalized_mse": error_energy / target_energy,
        "cosine": F.cosine_similarity(target64, reconstruction64, dim=1),
    }


def _distribution_js(first: Tensor, second: Tensor) -> Tensor:
    if first.shape != second.shape or first.shape[-1] < 2:
        raise ValueError("Probability distributions must have aligned support")
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(
            dim=-1
        )
        + (
            second
            * (second.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())
        ).sum(dim=-1)
    )


def compare_attention_patterns(first: Tensor, second: Tensor) -> dict[str, Tensor]:
    """Compare patterns even when native and LoRSA head counts differ.

    Head-mean JS is always defined.  Matched-head JS is emitted only when head
    counts match (for Raw/Hero comparisons of the same architecture).
    """

    if (
        first.ndim != 4
        or second.ndim != 4
        or first.shape[0] != second.shape[0]
        or first.shape[-2:] != second.shape[-2:]
    ):
        raise ValueError("Attention patterns must align on batch/query/key dimensions")
    if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
        raise ValueError("Attention patterns contain nonfinite values")
    first_mean = first.float().mean(dim=1)
    second_mean = second.float().mean(dim=1)
    result = {
        "head_mean_js": _distribution_js(first_mean, second_mean).mean(dim=-1),
        "first_mean_query_entropy": (
            -(first.float() * first.float().clamp_min(1e-30).log())
            .sum(dim=-1)
            .mean(dim=(-1, -2))
        ),
        "second_mean_query_entropy": (
            -(second.float() * second.float().clamp_min(1e-30).log())
            .sum(dim=-1)
            .mean(dim=(-1, -2))
        ),
    }
    if first.shape[1] == second.shape[1]:
        result["matched_head_js"] = _distribution_js(first.float(), second.float()).mean(
            dim=(-1, -2)
        )
    return result


def compact_feature_summary(features: Tensor, *, count: int) -> dict[str, Any] | None:
    """Return bounded position-level feature evidence for later semantic audits."""

    if type(count) is not int or count < 0:
        raise ValueError("Feature summary count must be non-negative")
    if count == 0:
        return None
    if features.ndim != 2 or not bool(torch.isfinite(features).all()):
        raise ValueError("Feature summary expects one finite [token, feature] matrix")
    positive = features.detach().float().cpu().clamp_min(0)
    strength = positive.sum(dim=0)
    selected_count = min(count, int(strength.numel()))
    values, indices = torch.topk(strength, selected_count, sorted=True)
    rows: list[dict[str, Any]] = []
    for value, index in zip(values, indices, strict=True):
        feature = positive[:, index]
        peak_value, peak_token = feature.max(dim=0)
        rows.append(
            {
                "feature": int(index),
                "summed_activation": float(value),
                "maximum_activation": float(peak_value),
                "active_token_count": int((feature > 0).sum()),
                "peak_token": int(peak_token),
            }
        )
    return {
        "reported_feature_count": selected_count,
        "mean_token_l0": float((positive > 0).sum(dim=1).float().mean()),
        "top_features": rows,
    }


def artifact_contract_gate(manifest: Mapping[str, Any], *, layer: int) -> dict[str, Any]:
    """Fail-closed structural parity gate for the reviewed conversion manifest."""

    hook = manifest.get("hook_contract", {})
    normalization = manifest.get("normalization", {})
    conversion = manifest.get("conversion", {})
    upstream = manifest.get("upstream", {})
    architecture = manifest.get("architecture", {})
    expected_input = f"blocks.{layer}.hook_attn_in"
    expected_output = f"blocks.{layer}.hook_attn_out"
    criteria = {
        "reviewed_conversion": conversion.get("conversion_reviewed") is True,
        "main_environment_pickle_not_loaded": (
            conversion.get(
                "main_environment_pickle_loaded",
                manifest.get("main_environment_pickle_loaded"),
            )
            is False
        ),
        "source_pickle_declared_isolated": (
            conversion.get("source_pickle_deserialized_in_isolated_environment") is True
        ),
        "layer_matches": hook.get("layer") == layer,
        "input_hook_matches": hook.get("input") == expected_input,
        "output_hook_matches": hook.get("target") == expected_output,
        "sequence_shape_matches": hook.get("sequence_shape")
        == ["batch", 64, 1024],
        "normalization_folded": normalization.get("folded") is True,
        "upstream_commit_pinned": upstream.get("code_commit") == UPSTREAM_COMMIT,
        "bt4_architecture_matches": architecture.get("d_model") == 1024
        and architecture.get("n_ctx") == 64,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "expected": {
            "layer": layer,
            "input_hook": expected_input,
            "output_hook": expected_output,
            "sequence_shape": ["batch", 64, 1024],
            "upstream_commit": UPSTREAM_COMMIT,
        },
    }


def source_compatibility_gate(
    reconstruction: Mapping[str, float],
    policy: Mapping[str, Any],
    artifact_contract: Mapping[str, Any],
    *,
    native_recompute_max_abs_error: float,
    position_count: int,
) -> dict[str, Any]:
    """Preregistered exploratory Raw gate; every criterion must pass."""

    criteria = {
        "artifact_contract_passed": artifact_contract.get("passed") is True,
        "native_attention_recompute_max_abs_error_le_1e-5": (
            native_recompute_max_abs_error <= 1e-5
        ),
        "normalized_mse_le_0.25": reconstruction["normalized_mse"] <= 0.25,
        "cosine_ge_0.85": reconstruction["cosine"] >= 0.85,
        "mean_policy_js_le_0.02": policy["js_divergence"]["mean"] <= 0.02,
        "top1_agreement_ge_0.80": policy["top1_agreement"]["mean"] >= 0.80,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "scope": f"exploratory {position_count}-position Raw source compatibility pilot",
        "failure_action": "skip Hero transfer and prohibit published-feature claims",
        "interpretation": (
            "Passing establishes bounded source compatibility, not semantic validity of every "
            "feature or equivalence to the native attention heads."
        ),
    }


def _summarize_vectors(rows: Mapping[str, Sequence[Tensor]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, chunks in sorted(rows.items()):
        value = torch.cat([chunk.detach().double().cpu().reshape(-1) for chunk in chunks])
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Metric {name} is empty or nonfinite")
        result[name] = {
            "count": int(value.numel()),
            "mean": float(value.mean()),
            "median": float(value.median()),
            "p95": float(torch.quantile(value, 0.95)),
            "minimum": float(value.min()),
            "maximum": float(value.max()),
        }
    return result


def _support_jaccard_per_position(first: Tensor, second: Tensor) -> Tensor:
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("Feature supports must align as [batch, token, feature]")
    first_support = first > 0
    second_support = second > 0
    union = (first_support | second_support).sum(dim=-1)
    intersection = (first_support & second_support).sum(dim=-1)
    per_token = torch.where(union > 0, intersection.float() / union, 1.0)
    return per_token.mean(dim=-1)


def _git_record() -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    try:
        repository = run("rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        return {
            "git_available": False,
            "repository_available": None,
            "commit": None,
            "branch": None,
            "worktree_clean": None,
            "status_sha256": None,
            "status": [],
            "reason": "git executable is unavailable in the runtime image",
        }
    if repository.returncode != 0 or repository.stdout.strip() != "true":
        return {
            "git_available": True,
            "repository_available": False,
            "commit": None,
            "branch": None,
            "worktree_clean": None,
            "status_sha256": None,
            "status": [],
            "reason": "runtime source tree has no Git repository metadata",
        }
    status_result = run("status", "--short")
    commit_result = run("rev-parse", "HEAD")
    branch_result = run("branch", "--show-current")
    if any(result.returncode != 0 for result in (status_result, commit_result, branch_result)):
        raise RuntimeError("Git repository became unreadable while recording provenance")
    status = status_result.stdout.strip()
    return {
        "git_available": True,
        "repository_available": True,
        "commit": commit_result.stdout.strip() or None,
        "branch": branch_result.stdout.strip() or None,
        "worktree_clean": not bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "status": status.splitlines(),
    }


def _source_file_hashes() -> dict[str, str]:
    """Hash the complete deterministic source inventory for scientific runs.

    The runner imports several interpretation modules transitively, so a short
    hand-maintained list is not a sufficient provenance boundary. Bind every
    Python source under ``research/interpretability``, the top-level model
    package, and the explicit research entrypoints and guards used by local and
    remote runs. The allowlist excludes analysis artifacts, notebooks, caches,
    credentials, and environment state by construction.
    """

    interpretation_root = _REPO_ROOT / "research/interpretability"
    chess_package_root = _REPO_ROOT / "chess_dfm_jax"
    for source_root in (interpretation_root, chess_package_root):
        if not source_root.is_dir():
            raise FileNotFoundError(f"Execution source root is missing: {source_root}")
    paths = [
        path
        for path in interpretation_root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    paths.extend(
        path
        for path in chess_package_root.glob("*.py")
        if "__pycache__" not in path.parts
    )
    paths.extend(
        (
            _REPO_ROOT / "research/__init__.py",
            _REPO_ROOT / "research/train_torch.py",
            _REPO_ROOT / "research/arena_history_trust.py",
            _REPO_ROOT / "research/prepare.py",
            _REPO_ROOT / "research/resource_guard.py",
            _REPO_ROOT / "research/run_torch_gpu.sh",
        )
    )
    if not paths:
        raise FileNotFoundError("Execution source inventory is empty")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Execution source inventory is missing: "
            + ", ".join(str(path) for path in missing)
        )
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError(
            "Execution source inventory cannot contain symlinks: "
            + ", ".join(str(path) for path in symlinks)
        )
    repository_root = _REPO_ROOT.resolve(strict=True)
    escaped: list[Path] = []
    for path in paths:
        try:
            path.resolve(strict=True).relative_to(repository_root)
        except ValueError:
            escaped.append(path)
    if escaped:
        raise ValueError(
            "Execution source path escapes repository: "
            + ", ".join(str(path) for path in escaped)
        )
    ordered = sorted(
        set(paths), key=lambda path: path.relative_to(_REPO_ROOT).as_posix()
    )
    return {
        path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path) for path in ordered
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


def _selected_rows(puzzles: Mapping[str, np.ndarray], count: int) -> np.ndarray:
    development = np.flatnonzero(puzzles["split_u8"] == 1)
    if development.size < count:
        raise ValueError("Public development split is smaller than requested LoRSA pilot")
    identifiers = puzzles["position_id"][development]
    return development[np.argsort(identifiers, kind="stable")[:count]]


def _batch_inputs(
    puzzles: Mapping[str, np.ndarray],
    selected: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.from_numpy(puzzles["current_planes_u8"][selected]).to(device),
        torch.from_numpy(puzzles["legal_idx_u16"][selected]).to(device),
        torch.from_numpy(puzzles["legal_count_u16"][selected]).to(device),
        torch.from_numpy(puzzles["target_action_u16"][selected]).to(device),
    )


def _module_dtype(module: LowRankSparseAttention) -> torch.dtype:
    dtypes = {parameter.dtype for parameter in module.parameters()}
    if len(dtypes) != 1:
        raise ValueError(f"LoRSA parameters have mixed dtypes: {sorted(map(str, dtypes))}")
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise ValueError("LoRSA checkpoint parameters must be floating point")
    return dtype


def _evaluate_arm(
    models: BT4ComparisonModels,
    lorsa: LowRankSparseAttention,
    planes: Tensor,
    legal: Tensor,
    legal_count: Tensor,
    targets: Tensor,
    *,
    arm: str,
    layer: int,
    replacement: bool,
) -> dict[str, Any]:
    dense = models.policy_logits_with_captures(
        planes,
        arm=arm,
        compute_dtype=torch.float32,
        capture_layers=(layer,),
    )
    if dense.captures is None:
        raise AssertionError("LoRSA experiment lost requested attention captures")
    states = dense.captures.hook_attn_in[0]
    native_target = dense.captures.hook_attn_out[0]
    if torch.is_autocast_enabled(states.device.type):
        raise RuntimeError("Autocast must be disabled at the LoRSA dtype boundary")
    checkpoint_dtype = _module_dtype(lorsa)
    sparse_input = states.to(dtype=checkpoint_dtype)
    if sparse_input.dtype != checkpoint_dtype or sparse_input.device != states.device:
        raise AssertionError("Explicit LoRSA input cast failed")
    encoder = models.raw_encoder if arm[0] == "R" else models.hero_encoder
    native = attention_internals(
        encoder.layers[layer],
        states,
        compute_dtype=torch.float32,
    )
    native_error = float((native.projected_output - native_target).abs().max())
    sparse = lorsa(sparse_input)
    if sparse.reconstruction.dtype != checkpoint_dtype:
        raise RuntimeError("LoRSA forward silently changed the checkpoint dtype")
    reconstruction_target = native_target.to(dtype=checkpoint_dtype)
    intervention_output = sparse.reconstruction.to(dtype=native_target.dtype)
    if intervention_output.dtype != native_target.dtype:
        raise AssertionError("Explicit dense-branch restoration cast failed")
    dtype_contract = {
        "dense_capture_dtype": str(states.dtype),
        "lorsa_checkpoint_dtype": str(checkpoint_dtype),
        "reconstruction_metric_dtype": str(reconstruction_target.dtype),
        "policy_intervention_dtype": str(intervention_output.dtype),
        "autocast_enabled": False,
        "explicit_input_cast": states.dtype != checkpoint_dtype,
        "explicit_intervention_cast": sparse.reconstruction.dtype != native_target.dtype,
    }
    result: dict[str, Any] = {
        "dense": dense,
        "states": states,
        "native_target": native_target,
        "reconstruction_target": reconstruction_target,
        "intervention_output": intervention_output,
        "native_attention": native,
        "native_recompute_max_abs_error": native_error,
        "sparse": sparse,
        "dtype_contract": dtype_contract,
    }
    if replacement:
        replaced = models.policy_logits_with_captures(
            planes,
            arm=arm,
            compute_dtype=torch.float32,
            capture_layers=(layer,),
            attention_output_overrides={layer: intervention_output},
        )
        result["replacement"] = replaced
        result["policy"] = policy_replacement_metrics(
            dense.logits,
            replaced.logits,
            legal,
            legal_count,
            targets,
        )
    return result


def _append_metrics(
    destination: dict[str, list[Tensor]],
    values: Mapping[str, Tensor],
) -> None:
    for name, value in values.items():
        destination.setdefault(name, []).append(value.detach().cpu())


def _example_identity(puzzles: Mapping[str, np.ndarray], row: int) -> dict[str, Any]:
    return {
        "position_id": str(puzzles["position_id"][row]),
        "puzzle_id": str(puzzles["puzzle_id"][row]),
        "group_id": str(puzzles["group_id"][row]),
        "target_action": int(puzzles["target_action_u16"][row]),
    }



def _feature_observation_boundary(
    *,
    phase: str,
    selected_rows: np.ndarray,
    puzzles: Mapping[str, np.ndarray],
    raw_features: Tensor | None = None,
    hero_features: Tensor | None = None,
) -> None:
    """Stable no-op boundary for a future streaming feature-semantic accumulator.

    Dense feature tensors deliberately remain device-local and are never persisted.
    At this point the corresponding corpus rows (planes, IDs, moves, and any future
    board labels) are still available. A semantic consumer can be wired here
    without changing the experiment's compact artifact format.
    """

    if phase not in {"raw_source", "paired_transfer"}:
        raise ValueError("Unknown feature observation phase")
    if selected_rows.ndim != 1 or selected_rows.size == 0:
        raise ValueError("Feature observation rows must be a non-empty vector")
    if "position_id" not in puzzles:
        raise ValueError("Feature observation requires position identities")
    batch = int(selected_rows.size)
    for name, value in (("raw", raw_features), ("hero", hero_features)):
        if value is not None and (value.ndim != 3 or value.shape[0] != batch):
            raise ValueError(f"{name} feature observation batch does not align")
    # TODO: attach the streaming feature-semantic accumulator here. Keeping this
    # boundary dependency-free avoids coupling the scientific runner to notebooks.

def _seal_run(
    *,
    target: Path,
    staging: Path,
    metrics: dict[str, Any],
    examples: Sequence[Mapping[str, Any]],
    manifest: dict[str, Any],
    log_lines: Sequence[str],
) -> dict[str, Any]:
    write_json_atomic(staging / "metrics.json", metrics)
    write_jsonl_atomic(staging / "examples.jsonl", examples)
    write_json_atomic(staging / "manifest.json", manifest)
    write_text_atomic(staging / "run.log", "\n".join(log_lines) + "\n")
    write_checksums(staging)
    verify_checksums(staging)
    os.replace(staging, target)
    checksums = verify_checksums(target)
    return {
        "output": str(target),
        "run_id": manifest["run_id"],
        "metrics_sha256": checksums["metrics.json"],
        "source_gate_passed": metrics["source_compatibility_gate"]["passed"],
        "transfer_status": metrics["transfer_stage"]["status"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    if args.position_count <= 0 or args.batch_size <= 0 or args.cpu_threads <= 0:
        raise ValueError("position_count, batch_size, and cpu_threads must be positive")
    if not 0 <= args.layer < 15:
        raise ValueError("layer must lie in [0, 14]")
    if not 0 <= args.feature_summary_count <= 128:
        raise ValueError("feature_summary_count must lie in [0, 128]")
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

        log("strict-loading corrected public corpus development split")
        corpus, corpus_manifest = load_public_corpus(args.corpus_manifest)
        puzzles = corpus["puzzles"]
        rows = _selected_rows(puzzles, args.position_count)
        log("strict-loading Raw/Hero BT4 comparison models")
        models = load_bt4_comparison_models(
            raw_bt4_path=args.raw_bt4,
            hero_checkpoint_dir=args.hero_checkpoint,
            device=device,
        )
        log("strict-loading reviewed converted LoRSA safetensors")
        loaded = load_converted_lorsa(args.lorsa_manifest, args.lorsa_weights)
        lorsa = loaded.module.to(device).eval()
        for parameter in lorsa.parameters():
            parameter.requires_grad_(False)
        if lorsa.config.d_model != 1024 or lorsa.config.n_ctx != 64:
            raise ValueError("Converted LoRSA does not implement the BT4 sequence ABI")
        contract = artifact_contract_gate(loaded.manifest, layer=args.layer)
        weights_before = sha256_file(args.lorsa_weights)

        raw_reconstruction = ReconstructionAccumulator()
        raw_policy_rows: dict[str, list[Tensor]] = {}
        raw_pattern_rows: dict[str, list[Tensor]] = {}
        raw_native_error = 0.0
        examples: dict[str, dict[str, Any]] = {}
        dtype_contract: dict[str, Any] | None = None

        log("phase 1/2: auditing Raw source reconstruction and policy replacement")
        for start in range(0, rows.size, args.batch_size):
            selected = rows[start : start + args.batch_size]
            planes, legal, legal_count, targets = _batch_inputs(puzzles, selected, device)
            with torch.inference_mode():
                raw = _evaluate_arm(
                    models,
                    lorsa,
                    planes,
                    legal,
                    legal_count,
                    targets,
                    arm="RR",
                    layer=args.layer,
                    replacement=True,
                )
                if dtype_contract is None:
                    dtype_contract = raw["dtype_contract"]
                elif raw["dtype_contract"] != dtype_contract:
                    raise RuntimeError("LoRSA dtype contract changed across Raw batches")
                raw_reconstruction.update(
                    raw["reconstruction_target"],
                    raw["sparse"].reconstruction,
                )
                raw_native_error = max(
                    raw_native_error,
                    raw["native_recompute_max_abs_error"],
                )
                _append_metrics(raw_policy_rows, raw["policy"])
                raw_pattern = compare_attention_patterns(
                    raw["native_attention"].pattern,
                    raw["sparse"].pattern,
                )
                _append_metrics(raw_pattern_rows, raw_pattern)
                _feature_observation_boundary(
                    phase="raw_source",
                    selected_rows=selected,
                    puzzles=puzzles,
                    raw_features=raw["sparse"].features,
                )
                per_reconstruction = per_position_reconstruction_metrics(
                    raw["reconstruction_target"],
                    raw["sparse"].reconstruction,
                )
            for offset, row in enumerate(selected):
                identity = _example_identity(puzzles, int(row))
                identity.update(
                    {
                        "raw": {
                            "reconstruction_normalized_mse": float(
                                per_reconstruction["normalized_mse"][offset]
                            ),
                            "reconstruction_cosine": float(
                                per_reconstruction["cosine"][offset]
                            ),
                            "policy_js": float(raw["policy"]["js_divergence"][offset]),
                            "policy_top1_preserved": bool(
                                raw["policy"]["top1_agreement"][offset]
                            ),
                            "native_vs_lorsa_head_mean_pattern_js": float(
                                raw_pattern["head_mean_js"][offset]
                            ),
                            "features": compact_feature_summary(
                                raw["sparse"].features[offset],
                                count=args.feature_summary_count,
                            ),
                        }
                    }
                )
                examples[identity["position_id"]] = identity
            log(f"Raw source audit processed {min(start + args.batch_size, rows.size)}/{rows.size}")

        raw_reconstruction_summary = raw_reconstruction.finalize()
        raw_policy_summary = summarize_policy_metrics(
            {name: torch.cat(chunks) for name, chunks in raw_policy_rows.items()}
        )
        source_gate = source_compatibility_gate(
            raw_reconstruction_summary,
            raw_policy_summary,
            contract,
            native_recompute_max_abs_error=raw_native_error,
            position_count=int(rows.size),
        )
        raw_stage = {
            "status": "complete",
            "reconstruction": raw_reconstruction_summary,
            "policy_replacement": raw_policy_summary,
            "attention_pattern": _summarize_vectors(raw_pattern_rows),
            "native_attention_recompute_max_abs_error": raw_native_error,
        }

        if not source_gate["passed"]:
            log("Raw source gate failed; Hero transfer skipped by fail-closed policy")
            transfer_stage: dict[str, Any] = {
                "status": "skipped_source_gate_failed",
                "reason": source_gate["failure_action"],
            }
        else:
            log("phase 2/2: freezing LoRSA and measuring Raw -> Hero transfer")
            hero_reconstruction = ReconstructionAccumulator()
            hero_policy_rows: dict[str, list[Tensor]] = {}
            hero_pattern_rows: dict[str, list[Tensor]] = {}
            cross_pattern_rows: dict[str, list[Tensor]] = {}
            hero_native_error = 0.0
            support = SparseSupportAccumulator(lorsa.config.n_ov_heads)
            support_position_rows: list[Tensor] = []
            for start in range(0, rows.size, args.batch_size):
                selected = rows[start : start + args.batch_size]
                planes, legal, legal_count, targets = _batch_inputs(puzzles, selected, device)
                with torch.inference_mode():
                    raw = _evaluate_arm(
                        models,
                        lorsa,
                        planes,
                        legal,
                        legal_count,
                        targets,
                        arm="RR",
                        layer=args.layer,
                        replacement=False,
                    )
                    hero = _evaluate_arm(
                        models,
                        lorsa,
                        planes,
                        legal,
                        legal_count,
                        targets,
                        arm="HH",
                        layer=args.layer,
                        replacement=True,
                    )
                    if raw["dtype_contract"] != dtype_contract:
                        raise RuntimeError("LoRSA Raw dtype contract changed during transfer")
                    if hero["dtype_contract"] != dtype_contract:
                        raise RuntimeError("Raw/Hero LoRSA dtype contracts differ")
                    hero_reconstruction.update(
                        hero["reconstruction_target"],
                        hero["sparse"].reconstruction,
                    )
                    hero_native_error = max(
                        hero_native_error,
                        hero["native_recompute_max_abs_error"],
                    )
                    _append_metrics(hero_policy_rows, hero["policy"])
                    hero_pattern = compare_attention_patterns(
                        hero["native_attention"].pattern,
                        hero["sparse"].pattern,
                    )
                    _append_metrics(hero_pattern_rows, hero_pattern)
                    cross_lorsa = compare_attention_patterns(
                        raw["sparse"].pattern,
                        hero["sparse"].pattern,
                    )
                    cross_native = compare_attention_patterns(
                        raw["native_attention"].pattern,
                        hero["native_attention"].pattern,
                    )
                    _append_metrics(
                        cross_pattern_rows,
                        {f"lorsa_{key}": value for key, value in cross_lorsa.items()},
                    )
                    _append_metrics(
                        cross_pattern_rows,
                        {f"native_{key}": value for key, value in cross_native.items()},
                    )
                    support.update(raw["sparse"].features, hero["sparse"].features)
                    support_position = _support_jaccard_per_position(
                        raw["sparse"].features,
                        hero["sparse"].features,
                    )
                    support_position_rows.append(support_position.cpu())
                    _feature_observation_boundary(
                        phase="paired_transfer",
                        selected_rows=selected,
                        puzzles=puzzles,
                        raw_features=raw["sparse"].features,
                        hero_features=hero["sparse"].features,
                    )
                    per_reconstruction = per_position_reconstruction_metrics(
                        hero["reconstruction_target"],
                        hero["sparse"].reconstruction,
                    )
                for offset, row in enumerate(selected):
                    position_id = str(puzzles["position_id"][row])
                    examples[position_id]["hero"] = {
                        "reconstruction_normalized_mse": float(
                            per_reconstruction["normalized_mse"][offset]
                        ),
                        "reconstruction_cosine": float(per_reconstruction["cosine"][offset]),
                        "policy_js": float(hero["policy"]["js_divergence"][offset]),
                        "policy_top1_preserved": bool(
                            hero["policy"]["top1_agreement"][offset]
                        ),
                        "native_vs_lorsa_head_mean_pattern_js": float(
                            hero_pattern["head_mean_js"][offset]
                        ),
                        "features": compact_feature_summary(
                            hero["sparse"].features[offset],
                            count=args.feature_summary_count,
                        ),
                    }
                    examples[position_id]["raw_hero_transfer"] = {
                        "support_jaccard": float(support_position[offset]),
                        "lorsa_matched_head_pattern_js": float(
                            cross_lorsa["matched_head_js"][offset]
                        ),
                        "native_matched_head_pattern_js": float(
                            cross_native["matched_head_js"][offset]
                        ),
                    }
                log(
                    "Hero transfer processed "
                    f"{min(start + args.batch_size, rows.size)}/{rows.size}"
                )

            hero_reconstruction_summary = hero_reconstruction.finalize()
            hero_policy_summary = summarize_policy_metrics(
                {name: torch.cat(chunks) for name, chunks in hero_policy_rows.items()}
            )
            transfer_stage = {
                "status": "complete",
                "hero": {
                    "reconstruction": hero_reconstruction_summary,
                    "policy_replacement": hero_policy_summary,
                    "attention_pattern": _summarize_vectors(hero_pattern_rows),
                    "native_attention_recompute_max_abs_error": hero_native_error,
                },
                "raw_hero": {
                    "support_transfer": support.finalize(),
                    "support_jaccard_per_position": _summarize_vectors(
                        {"value": support_position_rows}
                    )["value"],
                    "attention_pattern": _summarize_vectors(cross_pattern_rows),
                    "degradation": {
                        "normalized_mse_delta_hero_minus_raw": (
                            hero_reconstruction_summary["normalized_mse"]
                            - raw_reconstruction_summary["normalized_mse"]
                        ),
                        "normalized_mse_ratio_hero_over_raw": (
                            hero_reconstruction_summary["normalized_mse"]
                            / max(raw_reconstruction_summary["normalized_mse"], 1e-12)
                        ),
                        "policy_js_delta_hero_minus_raw": (
                            hero_policy_summary["js_divergence"]["mean"]
                            - raw_policy_summary["js_divergence"]["mean"]
                        ),
                    },
                },
            }

        weights_after = sha256_file(args.lorsa_weights)
        if weights_after != weights_before:
            raise RuntimeError("LoRSA safetensors changed during frozen transfer")
        elapsed = time.monotonic() - started
        if dtype_contract is None:
            raise AssertionError("LoRSA dtype contract was not observed")
        resource_usage: dict[str, Any] = {"wall_seconds": elapsed}
        if device.type == "cuda":
            resource_usage.update(
                {
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
        metrics = {
            "schema_version": "bt4-published-lorsa-transfer-metrics-v1",
            "layer": args.layer,
            "position_count": int(rows.size),
            "artifact_contract_gate": contract,
            "source_stage": raw_stage,
            "source_compatibility_gate": source_gate,
            "transfer_stage": transfer_stage,
            "interpretation_contract": {
                "development_split_only": True,
                "test_split_opened": False,
                "exploratory": True,
                "lorsa_frozen": True,
                "hero_transfer_interpretable": source_gate["passed"],
                "attention_head_mean_caveat": (
                    "Native BT4 has 32 heads and LoRSA has a separately learned head count; "
                    "native-vs-LoRSA pattern JS therefore compares their head-mean distributions."
                ),
            },
        }
        source_files = _source_file_hashes()
        payloads = {
            "metrics": content_identity(metrics),
            "source_files": content_identity(source_files),
            "positions": hashlib.sha256(
                "\n".join(str(puzzles["position_id"][row]) for row in rows).encode()
            ).hexdigest(),
            "lorsa": weights_before,
            "raw": models.raw_manifest["raw_asset"]["sha256"],
            "hero": models.hero_manifest["state"]["sha256"],
            "base_input_bundle": args.base_input_bundle_sha256,
            "lorsa_input_bundle": args.lorsa_input_bundle_sha256,
        }
        run_id = content_identity(payloads)
        manifest = {
            "schema_version": "bt4-published-lorsa-transfer-run-v1",
            "run_id": f"sha256:{run_id}",
            "started_utc": started_utc.isoformat(),
            "elapsed_seconds": elapsed,
            "resource_usage": resource_usage,
            "git": _git_record(),
            "environment": _environment(device),
            "model_lattice": models.descriptor(),
            "public_corpus_manifest": {
                "path": str(args.corpus_manifest.resolve()),
                "integrity_sha256": corpus_manifest["manifest_integrity"]["sha256"],
                "split": "development only",
                "selection": "lowest position_id in development split",
            },
            "lorsa": {
                **loaded.manifest,
                "weights_sha256_before": weights_before,
                "weights_sha256_after": weights_after,
                "parameters_frozen": True,
            },
            "experiment": {
                "compute_dtype": "float32",
                "layer": args.layer,
                "hook_in": "hook_attn_in",
                "hook_out": "hook_attn_out",
                "replacement_point": "before alpha, residual addition, and attention LN",
                "sequential_source_gate": True,
                "position_count": int(rows.size),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "feature_summary_count": args.feature_summary_count,
                "dtype_contract": dtype_contract,
                "confirmatory": False,
            },
            "cost": {
                "backend": args.backend,
                "provider_dollars": args.provider_dollars,
                "declared_attempt_upper_bound_dollars": (
                    args.declared_attempt_upper_bound_dollars
                ),
                "input_bundle_sha256": args.input_bundle_sha256,
                "base_input_bundle_sha256": args.base_input_bundle_sha256,
                "lorsa_input_bundle_sha256": args.lorsa_input_bundle_sha256,
            },
            "payload_identities": payloads,
            "source_files": source_files,
        }
        log(f"source compatibility gate passed={source_gate['passed']}")
        log_lines.append(f"run_id sha256:{run_id}")
        return _seal_run(
            target=target,
            staging=staging,
            metrics=metrics,
            examples=[examples[str(puzzles["position_id"][row])] for row in rows],
            manifest=manifest,
            log_lines=log_lines,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise metrics, gating, sparse transfer, and artifact sealing without BT4 assets."""

    target, staging = _output_staging(args.output)
    started = time.monotonic()
    seed = args.seed
    generator = torch.Generator(device="cpu").manual_seed(seed)
    config = LoRSAConfig(
        d_model=8,
        n_ctx=4,
        n_qk_heads=2,
        d_qk_head=2,
        n_ov_heads=8,
        top_k=2,
        attn_scale=math.sqrt(2),
        use_smolgen=False,
    )
    module = LowRankSparseAttention(config)
    initialize_lorsa(module, seed=seed)
    module.eval()
    raw_states = torch.randn((3, 4, 8), generator=generator)
    hero_states = raw_states + 0.01 * torch.randn((3, 4, 8), generator=generator)
    with torch.inference_mode():
        raw = module(raw_states)
        hero = module(hero_states)
    raw_target = raw.reconstruction + 1e-4 * torch.randn(
        raw.reconstruction.shape,
        generator=generator,
    )
    hero_target = hero.reconstruction + 2e-4 * torch.randn(
        hero.reconstruction.shape,
        generator=generator,
    )
    raw_accumulator = ReconstructionAccumulator()
    hero_accumulator = ReconstructionAccumulator()
    raw_accumulator.update(raw_target[:1], raw.reconstruction[:1])
    raw_accumulator.update(raw_target[1:], raw.reconstruction[1:])
    hero_accumulator.update(hero_target, hero.reconstruction)
    dense_logits = torch.randn((3, 7), generator=generator)
    replacement_logits = dense_logits + 1e-4 * torch.randn((3, 7), generator=generator)
    legal = torch.arange(7).repeat(3, 1)
    legal_count = torch.full((3,), 7)
    targets = torch.tensor([0, 1, 2])
    policy = summarize_policy_metrics(
        policy_replacement_metrics(
            dense_logits,
            replacement_logits,
            legal,
            legal_count,
            targets,
        )
    )
    support = SparseSupportAccumulator(config.n_ov_heads)
    support.update(raw.features, hero.features)
    contract = {
        "passed": True,
        "criteria": {"synthetic_fixture": True},
        "scientific_artifact_contract": False,
    }
    source_gate = source_compatibility_gate(
        raw_accumulator.finalize(),
        policy,
        contract,
        native_recompute_max_abs_error=0.0,
        position_count=3,
    )
    patterns = compare_attention_patterns(raw.pattern, hero.pattern)
    metrics = {
        "schema_version": "bt4-lorsa-synthetic-smoke-metrics-v1",
        "artifact_contract_gate": contract,
        "source_stage": {
            "status": "complete",
            "reconstruction": raw_accumulator.finalize(),
            "policy_replacement": policy,
        },
        "source_compatibility_gate": source_gate,
        "transfer_stage": {
            "status": "complete",
            "hero": {"reconstruction": hero_accumulator.finalize()},
            "raw_hero": {
                "support_transfer": support.finalize(),
                "attention_pattern": _summarize_vectors(
                    {name: [value] for name, value in patterns.items()}
                ),
            },
        },
        "scientific_result": False,
    }
    examples = []
    for index in range(3):
        examples.append(
            {
                "synthetic_position": index,
                "raw_features": compact_feature_summary(
                    raw.features[index], count=min(args.feature_summary_count, 8)
                ),
                "hero_features": compact_feature_summary(
                    hero.features[index], count=min(args.feature_summary_count, 8)
                ),
                "support_jaccard": float(
                    _support_jaccard_per_position(
                        raw.features[index : index + 1],
                        hero.features[index : index + 1],
                    )[0]
                ),
            }
        )
    run_id = content_identity({"seed": seed, "metrics": metrics})
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": "bt4-lorsa-synthetic-smoke-run-v1",
        "run_id": f"sha256:{run_id}",
        "started_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "environment": _environment(torch.device("cpu")),
        "experiment": {
            "synthetic_smoke": True,
            "scientific_result": False,
            "paid_compute": False,
            "seed": seed,
            "architecture": vars(config),
        },
    }
    logs = ["synthetic fixture only; no BT4 or published checkpoint loaded"]
    return _seal_run(
        target=target,
        staging=staging,
        metrics=metrics,
        examples=examples,
        manifest=manifest,
        log_lines=logs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bt4", type=Path, default=DEFAULT_RAW_BT4)
    parser.add_argument("--hero-checkpoint", type=Path, default=DEFAULT_HERO_CHECKPOINT)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_DIR / "manifest.json",
    )
    parser.add_argument(
        "--lorsa-manifest",
        type=Path,
        default=DEFAULT_LORSA_DIR / "manifest.json",
    )
    parser.add_argument(
        "--lorsa-weights",
        type=Path,
        default=DEFAULT_LORSA_DIR / "lorsa.safetensors",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--position-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--feature-summary-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--backend", default="local GPU")
    parser.add_argument("--provider-dollars", type=float)
    parser.add_argument("--declared-attempt-upper-bound-dollars", type=float)
    parser.add_argument("--input-bundle-sha256")
    parser.add_argument("--base-input-bundle-sha256")
    parser.add_argument("--lorsa-input-bundle-sha256")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a tiny non-scientific CPU fixture without model/checkpoint assets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_synthetic_smoke(args) if args.synthetic_smoke else run_transfer(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
