"""Run the immutable 128-position raw/Hero activation and behavior pilot."""

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

from research.interpretability.activation_diff import (
    HOOK_FIELDS,
    ActivationDiffAccumulator,
)
from research.interpretability.artifacts import (
    content_identity,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from research.interpretability.behavior_diff import (
    evaluate_lattice_batch,
    summarize_behavior_records,
)
from research.interpretability.models import (
    ARM_IDS,
    ComparisonLatticeOutput,
    load_bt4_comparison_models,
)
from research.interpretability.pilot_corpus import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CORPUS_DIR,
    decode_pilot_batch,
    write_deterministic_npz,
)
from research.train_torch import BT4EncoderCapture


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_BT4 = _REPO_ROOT / "models/source/extracted/BT4_exported.pb.gz"
DEFAULT_HERO_CHECKPOINT = _REPO_ROOT / "research/runs/torch_hero_epoch_v1/checkpoint"
DEFAULT_OUTPUT = _REPO_ROOT / "research/analysis/raw_hero_pilot_v1"
DEFAULT_PARAMETER_DIFF = _REPO_ROOT / "research/analysis/raw_hero_parameter_diff_20260807.json"
DEFAULT_MATRIX_SPECTRA = _REPO_ROOT / "research/analysis/raw_hero_matrix_spectra_20260807.json"


def _parse_layers(value: str) -> tuple[int, ...]:
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Layer range is reversed: {item}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    layers = tuple(sorted(selected))
    if not layers or layers[0] < 0 or layers[-1] >= 15:
        raise ValueError("Layers must select at least one index in [0, 14]")
    return layers


def _compute_dtype(value: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    try:
        return mapping[value]
    except KeyError as exc:  # pragma: no cover - argparse guards this
        raise ValueError(f"Unknown compute dtype {value}") from exc


def _capture_tensors(capture: BT4EncoderCapture) -> list[tuple[str, torch.Tensor]]:
    return [(hook, getattr(capture, hook)) for hook in HOOK_FIELDS]


def _assert_lattice_equal(
    first: ComparisonLatticeOutput,
    second: ComparisonLatticeOutput,
    *,
    label: str,
    compare_captures: bool,
) -> None:
    for arm in ARM_IDS:
        if not torch.equal(first.logits[arm], second.logits[arm]):
            difference = float(
                (first.logits[arm].float() - second.logits[arm].float()).abs().max().item()
            )
            raise RuntimeError(f"{label} {arm} logits differ (max abs {difference})")
    for name in ("raw_tokens", "hero_tokens"):
        if not torch.equal(getattr(first, name), getattr(second, name)):
            raise RuntimeError(f"{label} {name} differ")
    if not compare_captures:
        return
    for model_name in ("raw_captures", "hero_captures"):
        first_capture = getattr(first, model_name)
        second_capture = getattr(second, model_name)
        if first_capture is None or second_capture is None:
            raise RuntimeError(f"{label} expected {model_name}")
        if first_capture.layer_indices != second_capture.layer_indices:
            raise RuntimeError(f"{label} capture layer order differs")
        for hook, first_tensor in _capture_tensors(first_capture):
            if not torch.equal(first_tensor, getattr(second_capture, hook)):
                raise RuntimeError(f"{label} {model_name}.{hook} differs")


def _preflight(
    models: Any,
    planes: torch.Tensor,
    *,
    layers: tuple[int, ...],
    dtype_names: Sequence[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for dtype_name in dtype_names:
            compute_dtype = _compute_dtype(dtype_name)
            first = models.lattice_logits(
                planes,
                compute_dtype=compute_dtype,
                capture_layers=layers,
            )
            second = models.lattice_logits(
                planes,
                compute_dtype=compute_dtype,
                capture_layers=layers,
            )
            _assert_lattice_equal(
                first,
                second,
                label=f"{dtype_name} repeatability",
                compare_captures=True,
            )
            ordinary = models.lattice_logits(
                planes,
                compute_dtype=compute_dtype,
                capture_layers=None,
            )
            _assert_lattice_equal(
                first,
                ordinary,
                label=f"{dtype_name} capture parity",
                compare_captures=False,
            )
            logits_digest = hashlib.sha256()
            for arm in ARM_IDS:
                logits = first.logits[arm]
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite preflight logits for {dtype_name}/{arm}")
                logits_digest.update(arm.encode("ascii"))
                logits_digest.update(logits.detach().float().cpu().numpy().tobytes())
            results[dtype_name] = {
                "repeatability": "bit-exact logits, tokens, and all selected captures",
                "capture_parity": "bit-exact lattice logits and terminal tokens",
                "finite_logits": True,
                "logits_fp32_sha256": logits_digest.hexdigest(),
                "position_count": planes.shape[0],
            }
            del first, second, ordinary
            if planes.device.type == "cuda":
                torch.cuda.empty_cache()
    return results


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
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status": status.splitlines(),
    }


def _source_hashes() -> dict[str, str]:
    files = [
        _REPO_ROOT / "research/train_torch.py",
        *sorted((_REPO_ROOT / "research/interpretability").glob("*.py")),
    ]
    return {
        path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path)
        for path in files
        if path.is_file()
    }


def _related_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record: dict[str, Any] = {
        "path": path.relative_to(_REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return record
    record["schema_version"] = payload.get("schema_version")
    if "selection" in payload:
        record["selection"] = payload["selection"]
    return record


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return result


def _output_staging(output: Path) -> tuple[Path, Path]:
    target = output.resolve()
    if target in {Path("/"), Path.home().resolve(), _REPO_ROOT.resolve()}:
        raise ValueError(f"Refusing unsafe output directory: {target}")
    if target.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    return target, staging


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    started_utc = datetime.now(UTC)
    layers = _parse_layers(args.layers)
    if args.max_positions <= 0 or args.batch_size <= 0:
        raise ValueError("max_positions and batch_size must be positive")
    if args.sketch_width <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("sketch_width and bootstrap_replicates must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    target, staging = _output_staging(args.output)
    log_lines: list[str] = []

    def log(message: str) -> None:
        elapsed = time.monotonic() - started
        line = f"{elapsed:10.3f}s {message}"
        log_lines.append(line)
        print(line, flush=True)

    try:
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
        log("strict-loading immutable pilot corpus")
        batch, corpus_manifest = decode_pilot_batch(args.corpus_manifest)
        corpus_count = int(corpus_manifest["selection"]["count"])
        position_count = min(args.max_positions, corpus_count)
        if position_count != args.max_positions:
            log(f"requested {args.max_positions} positions; corpus contains {corpus_count}")
        positions = corpus_manifest["positions"][:position_count]
        game_ids = np.asarray([str(position["game_id"]) for position in positions])
        log(f"strict-loading raw and Hero models on {device}")
        models = load_bt4_comparison_models(
            raw_bt4_path=args.raw_bt4,
            hero_checkpoint_dir=args.hero_checkpoint,
            device=device,
        )
        compute_dtype = _compute_dtype(args.compute_dtype)
        first_planes = torch.from_numpy(batch["current_planes"][:1]).to(device)
        preflight_dtype_names = tuple(
            name.strip() for name in args.preflight_dtypes.split(",") if name.strip()
        )
        invalid_preflight = set(preflight_dtype_names) - {
            "bfloat16",
            "float32",
            "float16",
        }
        if invalid_preflight:
            raise ValueError(f"Invalid preflight dtypes: {sorted(invalid_preflight)}")
        log(f"running real-weight parity preflight for {preflight_dtype_names}")
        preflight = _preflight(
            models,
            first_planes,
            layers=layers,
            dtype_names=preflight_dtype_names,
        )
        del first_planes

        activation = ActivationDiffAccumulator(
            layer_indices=layers,
            sketch_width=args.sketch_width,
            projection_seed=args.projection_seed,
        )
        behavior_records: list[dict[str, Any]] = []
        log(
            f"starting paired pass: {position_count} positions, batch={args.batch_size}, "
            f"dtype={args.compute_dtype}, layers={layers}"
        )
        with torch.inference_mode():
            for start in range(0, position_count, args.batch_size):
                end = min(position_count, start + args.batch_size)
                planes = torch.from_numpy(batch["current_planes"][start:end]).to(device)
                output = models.lattice_logits(
                    planes,
                    compute_dtype=compute_dtype,
                    capture_layers=layers,
                )
                if output.raw_captures is None or output.hero_captures is None:
                    raise AssertionError("Pilot lattice unexpectedly omitted captures")
                behavior_records.extend(
                    evaluate_lattice_batch(
                        output.logits,
                        batch["legal_idx"][start:end, 0, :],
                        batch["legal_count"][start:end, 0],
                        batch["action_idx"][start:end],
                    )
                )
                activation.update(output.raw_captures, output.hero_captures)
                del output, planes
                if end == position_count or end % args.progress_every < args.batch_size:
                    log(f"processed {end}/{position_count} positions")
        if len(behavior_records) != position_count:
            raise AssertionError("Behavior record count drift")
        log("finalizing game-cluster bootstrap and representation statistics")
        behavior_summary = summarize_behavior_records(
            behavior_records,
            game_ids,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        activation_summary = activation.finalize(
            game_ids,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        position_metric_arrays = activation.position_metric_arrays()
        selected_layers = activation_summary["selected_layers_for_expensive_development_work"][
            "layers"
        ]

        related = {
            "parameter_diff": _related_artifact(args.parameter_diff),
            "matrix_spectra": _related_artifact(args.matrix_spectra),
        }
        metrics_core = {
            "schema_version": "bt4-raw-hero-pilot-metrics-v1",
            "scientific_status": (
                "exploratory development pilot; eligible for tooling validation "
                "and expensive-layer selection, not terminal confirmatory claims"
            ),
            "behavior": behavior_summary,
            "activation": activation_summary,
            "related_model_diff_artifacts": related,
        }
        metrics_identity = content_identity(metrics_core)
        metrics = {"content_sha256": metrics_identity, **metrics_core}
        write_json_atomic(staging / "metrics.json", metrics)

        position_npz = {
            **position_metric_arrays,
            "position_id": np.asarray([position["position_id"] for position in positions]),
            "game_id": game_ids,
            "hook_fields": np.asarray(HOOK_FIELDS),
            "layer_indices": np.asarray(layers, dtype=np.int16),
            "schema_version": np.asarray("bt4-activation-position-metrics-v1"),
        }
        write_deterministic_npz(staging / "activation_position_metrics.npz", position_npz)
        residual_hook_offset = HOOK_FIELDS.index("resid_post_after_ln")
        layer_offsets = {layer: layers.index(layer) for layer in selected_layers}
        examples: list[dict[str, Any]] = []
        for row, (position, behavior) in enumerate(zip(positions, behavior_records, strict=True)):
            activation_example = {
                str(layer): {
                    metric: float(
                        position_metric_arrays[metric][
                            row,
                            residual_hook_offset,
                            layer_offsets[layer],
                        ]
                    )
                    for metric in position_metric_arrays
                }
                for layer in selected_layers
            }
            examples.append(
                {
                    "position_id": position["position_id"],
                    "global_index": position["global_index"],
                    "game_id": position["game_id"],
                    "ply": position["ply"],
                    "fen": position["fen"],
                    "behavior": behavior,
                    "selected_resid_post_activation_metrics": activation_example,
                }
            )
        write_jsonl_atomic(staging / "examples.jsonl", examples)
        log("writing and verifying atomic run bundle")
        completed_utc = datetime.now(UTC)
        environment = _environment(device)
        if device.type == "cuda":
            environment["gpu"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            environment["gpu"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        payloads = {
            "metrics_content_sha256": metrics_identity,
            "activation_position_metrics_sha256": sha256_file(
                staging / "activation_position_metrics.npz"
            ),
            "examples_sha256": sha256_file(staging / "examples.jsonl"),
            "corpus_manifest_integrity_sha256": corpus_manifest["manifest_integrity"]["sha256"],
            "raw_asset_sha256": models.descriptor()["raw_asset_sha256"],
            "hero_state_sha256": models.descriptor()["hero_state_sha256"],
        }
        run_identity = content_identity(payloads)
        manifest = {
            "schema_version": "bt4-interpretability-run-manifest-v1",
            "run_id": f"sha256:{run_identity}",
            "created_utc": started_utc.isoformat(),
            "completed_utc": completed_utc.isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "status": "complete",
            "scientific_status": metrics_core["scientific_status"],
            "hypotheses": {
                "primary": (
                    "Hero may differ from raw BT4 in parameters, corresponding "
                    "representations, and legal-policy behavior"
                ),
                "null": (
                    "the development corpus may resolve no meaningful encoder delta; "
                    "head-only or no resolved change remains valid"
                ),
            },
            "aggregation_unit": "position; bootstrap clusters are source game_id",
            "exclusions": "none; fail closed on nonfinite values or ABI drift",
            "stopping_rule": f"exactly the first {position_count} rows of the immutable corpus",
            "inputs": {
                "corpus_manifest": str(args.corpus_manifest.resolve()),
                "corpus_manifest_integrity_sha256": corpus_manifest["manifest_integrity"]["sha256"],
                "corpus_data_sha256": corpus_manifest["data"]["sha256"],
                "models": models.descriptor(),
            },
            "configuration": {
                "positions": position_count,
                "batch_size": args.batch_size,
                "compute_dtype": args.compute_dtype,
                "layers": list(layers),
                "sketch_width": args.sketch_width,
                "projection_seed": args.projection_seed,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "cpu_threads": args.cpu_threads,
            },
            "preflight": preflight,
            "source": {"git": _git_record(), "files_sha256": _source_hashes()},
            "environment": environment,
            "cost": {
                "backend": "local",
                "estimated_dollars": 0.0,
                "actual_provider_dollars": 0.0,
            },
            "payload_identities": payloads,
            "run_identity_contract": (
                "SHA256 of canonical payload identities; timestamps and paths excluded"
            ),
        }
        write_json_atomic(staging / "manifest.json", manifest)
        log_lines.append(f"run_id sha256:{run_identity}")
        write_text_atomic(staging / "run.log", "\n".join(log_lines) + "\n")
        write_checksums(staging)
        verify_checksums(staging)
        os.replace(staging, target)
        checksums = verify_checksums(target)
        return {
            "output": str(target),
            "run_id": f"sha256:{run_identity}",
            "metrics_sha256": checksums["metrics.json"],
            "positions": position_count,
            "selected_layers": selected_layers,
            "elapsed_seconds": manifest["elapsed_seconds"],
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parameter-diff", type=Path, default=DEFAULT_PARAMETER_DIFF)
    parser.add_argument("--matrix-spectra", type=Path, default=DEFAULT_MATRIX_SPECTRA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--preflight-dtypes", default="bfloat16,float32")
    parser.add_argument("--layers", default="0-14")
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sketch-width", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_pilot(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
