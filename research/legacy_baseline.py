#!/usr/bin/env python3
"""Create a strict local-GPU fingerprint of the legacy joint checkpoint.

This is a parity oracle, not the editable autoresearch trainer. It intentionally
uses the legacy model and loss implementation with deterministic inputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.data.leela import LeelaChunkDataLoader  # noqa: E402
from chess_dfm_jax.training.checkpoints import load_training_checkpoint  # noqa: E402
from chess_dfm_jax.training.joint_latent_sasa import (  # noqa: E402
    JointLatentSASAConfig,
    create_joint_components,
    eval_joint_stage1_step,
)
from research.prepare import (  # noqa: E402
    REPO_ROOT,
    require_within_workspace,
    sha256_file,
    write_json,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "checkpoints" / "source" / "step0265000"
DEFAULT_CHECKPOINT_DIR = DEFAULT_RUN_ROOT / "checkpoints"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "source" / "extracted"
DEFAULT_VAL_DIR = REPO_ROOT / "data" / "trajectory_v3" / "val"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "baselines" / "legacy_step265000_loss.json"


def resolve_config(run_root: Path) -> tuple[JointLatentSASAConfig, dict[str, Any]]:
    metadata_path = require_within_workspace(run_root / "checkpoint_state.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        raise TypeError(f"Missing config object in {metadata_path}")
    field_names = {field.name for field in dataclasses.fields(JointLatentSASAConfig)}
    config = JointLatentSASAConfig(
        **{key: value for key, value in raw_config.items() if key in field_names}
    )
    return config, metadata


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.ndim == 0:
            flattened[key] = float(array)
            continue
        flat = array.reshape(-1)
        if key.endswith("_by_horizon") or key in {
            "dfm_ce_loss_by_horizon",
            "dfm_mask_fraction_by_horizon",
            "jepa_loss_by_horizon",
            "jepa_raw_mse_by_horizon",
            "jepa_norm_loss_by_horizon",
            "mean_token_cosine_by_horizon",
            "jepa_target_horizon_mask",
        }:
            for index, item in enumerate(flat):
                flattened[f"{key}_h{index + 1}"] = float(item)
            continue
        for index, item in enumerate(flat):
            flattened[f"{key}_{index}"] = float(item)
    return flattened


def digest_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-t", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = require_within_workspace(args.run_root)
    checkpoint_dir = require_within_workspace(args.checkpoint_dir)
    models_dir = require_within_workspace(args.models_dir)
    val_dir = require_within_workspace(args.val_dir)
    output = require_within_workspace(args.output)

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Legacy baseline requires GPU, found {jax.default_backend()!r}")
    if not 0.0 <= args.deterministic_t <= 1.0:
        raise ValueError("--deterministic-t must be in [0, 1].")

    config, metadata = resolve_config(run_root)
    val_shards = sorted(val_dir.glob("*.npz"))
    if not val_shards:
        raise FileNotFoundError(f"No validation shards under {val_dir}")
    first_shard = require_within_workspace(val_shards[0])

    mapped_params = load_mapped_bt4_params(models_dir=models_dir)
    model, optimizer = create_joint_components(mapped_params, config, seed=args.seed)

    restore_started = time.perf_counter()
    payload = load_training_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=None,
        step=int(metadata["latest_step"]),
        strict=True,
    )
    restore_seconds = time.perf_counter() - restore_started
    payload_step = int(payload["step"])
    payload_keys = sorted(payload)
    del payload
    del optimizer

    loader = LeelaChunkDataLoader(
        [str(first_shard)],
        batch_size=args.batch_size,
        seed=args.seed,
        horizon=config.horizon,
        shuffle_files=False,
        drop_last=True,
        include_metadata=False,
        batch_view="joint_latent_sasa",
    )
    batch = next(iter(loader))
    batch = dict(batch)
    batch["deterministic_t"] = np.asarray(args.deterministic_t, dtype=np.float32)
    rng = jax.random.PRNGKey(args.seed)

    compile_started = time.perf_counter()
    loss, aux = eval_joint_stage1_step(model, batch, rng)
    jax.block_until_ready((loss, aux))
    compile_and_first_eval_seconds = time.perf_counter() - compile_started

    steady_started = time.perf_counter()
    repeated_loss, repeated_aux = eval_joint_stage1_step(model, batch, rng)
    jax.block_until_ready((repeated_loss, repeated_aux))
    steady_eval_seconds = time.perf_counter() - steady_started

    metrics = {"loss": float(loss), **flatten_metrics(aux)}
    repeated_metrics = {"loss": float(repeated_loss), **flatten_metrics(repeated_aux)}
    if metrics != repeated_metrics:
        differing = sorted(key for key in metrics if metrics[key] != repeated_metrics.get(key))
        raise RuntimeError(f"Deterministic repeated evaluation differed: {differing}")

    report = {
        "kind": "legacy_joint_stage1_gpu_fingerprint",
        "checkpoint_step": payload_step,
        "checkpoint_payload_keys": payload_keys,
        "strict_model_restore": True,
        "git_commit": _git_commit(),
        "config": dataclasses.asdict(config),
        "batch": {
            "batch_size": args.batch_size,
            "deterministic_t": args.deterministic_t,
            "seed": args.seed,
            "validation_shard": str(first_shard),
            "validation_shard_size_bytes": first_shard.stat().st_size,
            "validation_shard_sha256": sha256_file(first_shard),
            "keys": sorted(batch),
            "key_digest": digest_strings(sorted(batch)),
        },
        "timing": {
            "checkpoint_restore_seconds": restore_seconds,
            "compile_and_first_eval_seconds": compile_and_first_eval_seconds,
            "steady_eval_seconds": steady_eval_seconds,
        },
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [device.device_kind for device in jax.devices()],
        },
        "metrics": metrics,
    }
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
