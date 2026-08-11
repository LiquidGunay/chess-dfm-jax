"""Run the first published-transcoder Raw BT4 -> Hero transfer gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from research.interpretability.chessbench import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_DIR,
    load_public_corpus,
)
from research.interpretability.models import load_bt4_comparison_models
from research.interpretability.patching import legal_log_probabilities, target_log_odds
from research.interpretability.sparse.transcoder import (
    SparseSupportAccumulator,
    load_published_transcoder,
    reconstruction_metrics,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_BT4 = _REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz"
DEFAULT_HERO_CHECKPOINT = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint"
DEFAULT_TC_DIR = _REPO_ROOT / "data/external/leela_saes/k_30_e_16/L14"
DEFAULT_OUTPUT = _REPO_ROOT / "research/analysis/published_tc_l14_transfer_pilot_v1"
LAYER = 14


def policy_replacement_metrics(
    dense_logits: Tensor,
    replacement_logits: Tensor,
    legal_indices: Tensor,
    legal_counts: Tensor,
    targets: Tensor,
) -> dict[str, Tensor]:
    """Per-position legal-policy damage from replacing a dense branch."""

    dense_logp, mask = legal_log_probabilities(
        dense_logits,
        legal_indices,
        legal_counts,
        targets=targets,
    )
    replacement_logp, replacement_mask = legal_log_probabilities(
        replacement_logits,
        legal_indices,
        legal_counts,
        targets=targets,
    )
    if not torch.equal(mask, replacement_mask):
        raise AssertionError("Legal masks differ across native/replacement policies")
    dense_probability = dense_logp.exp().masked_fill(~mask, 0.0)
    replacement_probability = replacement_logp.exp().masked_fill(~mask, 0.0)
    midpoint = 0.5 * (dense_probability + replacement_probability)
    midpoint_log = torch.where(mask, midpoint.clamp_min(1e-30).log(), 0.0)
    js = 0.5 * (
        torch.where(mask, dense_probability * (dense_logp - midpoint_log), 0.0).sum(dim=1)
        + torch.where(
            mask,
            replacement_probability * (replacement_logp - midpoint_log),
            0.0,
        ).sum(dim=1)
    )
    dense_odds = target_log_odds(dense_logits, legal_indices, legal_counts, targets)
    replacement_odds = target_log_odds(
        replacement_logits,
        legal_indices,
        legal_counts,
        targets,
    )
    return {
        "js_divergence": js,
        "total_variation": 0.5 * (dense_probability - replacement_probability).abs().sum(dim=1),
        "top1_agreement": dense_logp.argmax(dim=1) == replacement_logp.argmax(dim=1),
        "target_log_odds_change": replacement_odds - dense_odds,
    }


def summarize_policy_metrics(rows: dict[str, Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in sorted(rows.items()):
        value = values.detach().cpu()
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"Policy metric {name} must be a non-empty vector")
        if value.dtype == torch.bool:
            result[name] = {
                "count": int(value.numel()),
                "true_count": int(value.sum()),
                "mean": float(value.float().mean()),
            }
        else:
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Policy metric {name} contains nonfinite values")
            result[name] = {
                "count": int(value.numel()),
                "mean": float(value.double().mean()),
                "median": float(value.double().median()),
                "maximum_absolute": float(value.double().abs().max()),
            }
    return result


def source_compatibility_gate(
    reconstruction: dict[str, float],
    policy: dict[str, Any],
    *,
    position_count: int | None = None,
) -> dict[str, Any]:
    """Preregistered exploratory source gate; every criterion must pass."""

    criteria = {
        "normalized_mse_le_0.25": reconstruction["normalized_mse"] <= 0.25,
        "cosine_ge_0.85": reconstruction["cosine"] >= 0.85,
        "mean_policy_js_le_0.02": policy["js_divergence"]["mean"] <= 0.02,
        "top1_agreement_ge_0.80": policy["top1_agreement"]["mean"] >= 0.80,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "scope": (
            "exploratory source compatibility pilot"
            if position_count is None
            else f"exploratory {position_count}-position source compatibility pilot"
        ),
        "interpretation": (
            "A failure blocks published-feature semantic claims; it does not measure Hero drift."
        ),
    }


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


def _selected_rows(puzzles: dict[str, np.ndarray], count: int) -> np.ndarray:
    development = np.flatnonzero(puzzles["split_u8"] == 1)
    if development.size < count:
        raise ValueError("Public development split is smaller than requested pilot")
    identifiers = puzzles["position_id"][development]
    return development[np.argsort(identifiers, kind="stable")[:count]]


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    if args.position_count <= 0 or args.batch_size <= 0:
        raise ValueError("position_count and batch_size must be positive")
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
        log("strict-loading raw/Hero BT4 comparison models")
        models = load_bt4_comparison_models(
            raw_bt4_path=args.raw_bt4,
            hero_checkpoint_dir=args.hero_checkpoint,
            device=device,
        )
        log("strict-loading and normalization-folding published L14 transcoder")
        loaded_tc = load_published_transcoder(args.tc_config, args.tc_weights)
        transcoder = loaded_tc.module.to(device).eval()

        reconstruction_targets: dict[str, list[Tensor]] = {"raw": [], "hero": []}
        reconstructions: dict[str, list[Tensor]] = {"raw": [], "hero": []}
        support_accumulator = SparseSupportAccumulator(transcoder.d_sae)
        policy_rows: dict[str, dict[str, list[Tensor]]] = {
            "raw": {},
            "hero": {},
        }
        examples: list[dict[str, Any]] = []
        for start in range(0, rows.size, args.batch_size):
            selected = rows[start : start + args.batch_size]
            planes = torch.from_numpy(puzzles["current_planes_u8"][selected]).to(device)
            legal = torch.from_numpy(puzzles["legal_idx_u16"][selected]).to(device)
            legal_count = torch.from_numpy(puzzles["legal_count_u16"][selected]).to(device)
            targets = torch.from_numpy(puzzles["target_action_u16"][selected]).to(device)
            with torch.inference_mode():
                raw_dense = models.policy_logits_with_captures(
                    planes,
                    arm="RR",
                    compute_dtype=torch.float32,
                    capture_layers=(LAYER,),
                )
                hero_dense = models.policy_logits_with_captures(
                    planes,
                    arm="HH",
                    compute_dtype=torch.float32,
                    capture_layers=(LAYER,),
                )
                if raw_dense.captures is None or hero_dense.captures is None:
                    raise AssertionError("Sparse transfer run lost layer captures")
                raw_input = raw_dense.captures.resid_mid_after_ln[0]
                hero_input = hero_dense.captures.resid_mid_after_ln[0]
                raw_target = raw_dense.captures.hook_mlp_out[0]
                hero_target = hero_dense.captures.hook_mlp_out[0]
                raw_tc = transcoder(raw_input)
                hero_tc = transcoder(hero_input)
                raw_replaced = models.policy_logits_with_captures(
                    planes,
                    arm="RR",
                    compute_dtype=torch.float32,
                    capture_layers=(LAYER,),
                    mlp_output_overrides={LAYER: raw_tc.reconstruction},
                )
                hero_replaced = models.policy_logits_with_captures(
                    planes,
                    arm="HH",
                    compute_dtype=torch.float32,
                    capture_layers=(LAYER,),
                    mlp_output_overrides={LAYER: hero_tc.reconstruction},
                )
                per_model = {
                    "raw": (raw_dense, raw_replaced, raw_target, raw_tc),
                    "hero": (hero_dense, hero_replaced, hero_target, hero_tc),
                }
                batch_policy: dict[str, dict[str, Tensor]] = {}
                for name, (dense, replaced, native_target, sparse) in per_model.items():
                    reconstruction_targets[name].append(native_target.float().cpu())
                    reconstructions[name].append(sparse.reconstruction.float().cpu())
                    batch_policy[name] = policy_replacement_metrics(
                        dense.logits,
                        replaced.logits,
                        legal,
                        legal_count,
                        targets,
                    )
                    for metric, values in batch_policy[name].items():
                        policy_rows[name].setdefault(metric, []).append(values.cpu())
                support_accumulator.update(raw_tc.features, hero_tc.features)
            for offset, row in enumerate(selected):
                examples.append(
                    {
                        "position_id": str(puzzles["position_id"][row]),
                        "puzzle_id": str(puzzles["puzzle_id"][row]),
                        "group_id": str(puzzles["group_id"][row]),
                        "target_action": int(puzzles["target_action_u16"][row]),
                        "raw_policy_js": float(batch_policy["raw"]["js_divergence"][offset]),
                        "hero_policy_js": float(batch_policy["hero"]["js_divergence"][offset]),
                        "raw_top1_preserved": bool(batch_policy["raw"]["top1_agreement"][offset]),
                        "hero_top1_preserved": bool(batch_policy["hero"]["top1_agreement"][offset]),
                    }
                )
            log(f"processed {min(start + args.batch_size, rows.size)}/{rows.size} positions")

        reconstruction = {
            name: reconstruction_metrics(
                torch.cat(reconstruction_targets[name]),
                torch.cat(reconstructions[name]),
            )
            for name in ("raw", "hero")
        }
        support = support_accumulator.finalize()
        policy = {
            name: summarize_policy_metrics(
                {metric: torch.cat(chunks) for metric, chunks in policy_rows[name].items()}
            )
            for name in ("raw", "hero")
        }
        gate = source_compatibility_gate(
            reconstruction["raw"],
            policy["raw"],
            position_count=int(rows.size),
        )
        transfer = {
            "normalized_mse_delta_hero_minus_raw": (
                reconstruction["hero"]["normalized_mse"] - reconstruction["raw"]["normalized_mse"]
            ),
            "normalized_mse_ratio_hero_over_raw": (
                reconstruction["hero"]["normalized_mse"]
                / max(reconstruction["raw"]["normalized_mse"], 1e-12)
            ),
            "policy_js_delta_hero_minus_raw": (
                policy["hero"]["js_divergence"]["mean"] - policy["raw"]["js_divergence"]["mean"]
            ),
        }
        metrics = {
            "schema_version": "bt4-published-transcoder-transfer-metrics-v1",
            "layer": LAYER,
            "artifact": "k_30_e_16",
            "position_count": int(rows.size),
            "reconstruction": reconstruction,
            "policy_replacement": policy,
            "support_transfer": support,
            "transfer_degradation": transfer,
            "source_compatibility_gate": gate,
        }
        write_json_atomic(staging / "metrics.json", metrics)
        write_jsonl_atomic(staging / "examples.jsonl", examples)
        elapsed = time.monotonic() - started
        payloads = {
            "metrics": content_identity(metrics),
            "positions": hashlib.sha256(
                "\n".join(str(puzzles["position_id"][row]) for row in rows).encode()
            ).hexdigest(),
            "transcoder": loaded_tc.manifest["weights_sha256"],
            "raw": models.raw_manifest["raw_asset"]["sha256"],
            "hero": models.hero_manifest["state"]["sha256"],
        }
        run_id = content_identity(payloads)
        manifest = {
            "schema_version": "bt4-published-transcoder-transfer-run-v1",
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
            },
            "transcoder": loaded_tc.manifest,
            "experiment": {
                "compute_dtype": "float32",
                "layer": LAYER,
                "hook_in": "resid_mid_after_ln",
                "hook_out": "hook_mlp_out",
                "replacement_point": "before alpha, residual addition, and LN2",
                "position_count": int(rows.size),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "confirmatory": False,
            },
            "cost": {
                "backend": "local GTX 1660 Ti",
                "provider_dollars": 0.0,
            },
            "payload_identities": payloads,
            "source_files": {
                path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("transcoder.py"),
                    _REPO_ROOT / "research/interpretability/models.py",
                    _REPO_ROOT / "research/train_torch.py",
                )
            },
        }
        write_json_atomic(staging / "manifest.json", manifest)
        log(f"source compatibility gate passed={gate['passed']}")
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
            "source_gate_passed": gate["passed"],
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
    parser.add_argument("--tc-config", type=Path, default=DEFAULT_TC_DIR / "config.json")
    parser.add_argument(
        "--tc-weights",
        type=Path,
        default=DEFAULT_TC_DIR / "sae_weights.safetensors",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--position-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_transfer(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
