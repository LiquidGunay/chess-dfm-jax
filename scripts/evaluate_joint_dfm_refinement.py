#!/usr/bin/env python3
"""Evaluate cached-latent DFM refinement quality for a joint Latent-SASA checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files  # noqa: E402
from chess_dfm_jax.paths import default_bt4_paths  # noqa: E402
from chess_dfm_jax.policy import policy_index_to_move  # noqa: E402
from chess_dfm_jax.training.checkpoints import (  # noqa: E402
    latest_checkpoint_step,
    load_training_checkpoint,
    read_checkpoint_metadata,
)
from chess_dfm_jax.training.dfm import refine_actions_from_latents  # noqa: E402
from chess_dfm_jax.training.joint_latent_sasa import (  # noqa: E402
    JointLatentSASAConfig,
    create_joint_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Local or gs:// joint checkpoint directory.")
    parser.add_argument("--models-dir", default=None, help="Path to BT4 model files.")
    parser.add_argument("--chunk-dir", default=None, help="Validation trajectory chunk directory.")
    parser.add_argument("--gcs-val-prefix", default="", help="Optional GCS prefix containing validation .npz shards.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--out-json", default=None, help="Optional path to write metrics JSON.")
    parser.add_argument("--refinement-steps", default="1,2,4,8,16", help="Comma-separated sampler step counts.")
    parser.add_argument("--backend", choices=["cpu", "tpu"], default="cpu")
    return parser.parse_args()


def _metadata_config(checkpoint_dir: Path) -> dict[str, Any]:
    metadata = read_checkpoint_metadata(checkpoint_dir.parent / "checkpoint_state.json")
    config = metadata.get("config", {})
    return config if isinstance(config, dict) else {}


def resolve_joint_config(checkpoint_dir: Path) -> JointLatentSASAConfig:
    metadata = _metadata_config(checkpoint_dir)
    if not metadata:
        raise ValueError(
            f"Could not find checkpoint metadata at {checkpoint_dir.parent / 'checkpoint_state.json'}; "
            "pass a checkpoint directory whose parent contains checkpoint_state.json."
        )
    field_names = {field.name for field in dataclasses.fields(JointLatentSASAConfig)}
    kwargs = {key: value for key, value in metadata.items() if key in field_names}
    return JointLatentSASAConfig(**kwargs)


def sync_gcs_tree(source_uri: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/snap/google-cloud-cli/current/bin/gcloud",
            "storage",
            "cp",
            "--recursive",
            f"{source_uri.rstrip('/')}/*",
            str(destination),
        ],
        check=True,
    )
    return destination


def sync_gcs_checkpoint(checkpoint_uri: str, destination: Path) -> Path:
    checkpoint_dir = sync_gcs_tree(checkpoint_uri, destination)
    parent_uri = checkpoint_uri.rstrip("/").rsplit("/", 1)[0]
    metadata_destination = destination.parent / "checkpoint_state.json"
    subprocess.run(
        [
            "/snap/google-cloud-cli/current/bin/gcloud",
            "storage",
            "cp",
            f"{parent_uri}/checkpoint_state.json",
            str(metadata_destination),
        ],
        check=False,
    )
    return checkpoint_dir


def parse_refinement_steps(value: str) -> list[int]:
    steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not steps:
        raise ValueError("--refinement-steps must contain at least one integer.")
    if any(step < 1 for step in steps):
        raise ValueError(f"All refinement steps must be >= 1, got {steps}.")
    return steps


def exact_sequence_legal(actions: np.ndarray, fen: str) -> tuple[bool, int]:
    if chess is None:
        return False, 0
    board = chess.Board(str(fen))
    for horizon_idx, action_idx in enumerate(np.asarray(actions, dtype=np.int32)):
        try:
            move = policy_index_to_move(int(action_idx), "lc0_1858")
        except Exception:
            return False, horizon_idx
        if move not in board.legal_moves:
            return False, horizon_idx
        board.push(move)
    return True, len(actions)


def _empty_sampler_totals(horizon: int) -> dict[str, Any]:
    return {
        "sample_count": 0.0,
        "first_move_legal_hits": 0.0,
        "first_move_match_hits": 0.0,
        "action_token_match_hits": 0.0,
        "action_token_match_count": 0.0,
        "horizon_hits": np.zeros((horizon,), dtype=np.float64),
        "horizon_counts": np.zeros((horizon,), dtype=np.float64),
        "teacher_forced_legal_hits": 0.0,
        "teacher_forced_legal_count": 0.0,
        "exact_available": 0.0,
        "exact_legal": 0.0,
        "target_exact_legal": 0.0,
        "sampling_time_s": 0.0,
    }


def _update_sampler_totals(
    totals: dict[str, Any],
    *,
    pred_actions: np.ndarray,
    target_actions: np.ndarray,
    batch: dict[str, np.ndarray],
    valid: np.ndarray,
) -> None:
    weight = float(np.sum(valid))
    if weight <= 0:
        return

    compare_horizon = min(pred_actions.shape[1], target_actions.shape[1], totals["horizon_hits"].shape[0])
    pred_actions = pred_actions[:, :compare_horizon]
    target_actions = target_actions[:, :compare_horizon]

    first_pred = pred_actions[:, 0]
    first_legal = np.asarray(batch["legal_mask"], dtype=bool)[np.arange(len(first_pred)), first_pred]
    token_matches = pred_actions == target_actions

    totals["sample_count"] += weight
    totals["first_move_legal_hits"] += float(np.sum(first_legal & valid))
    totals["first_move_match_hits"] += float(np.sum((first_pred == target_actions[:, 0]) & valid))
    totals["action_token_match_hits"] += float(np.sum(token_matches * valid[:, None]))
    totals["action_token_match_count"] += float(np.sum(valid) * compare_horizon)

    for horizon_idx in range(compare_horizon):
        totals["horizon_hits"][horizon_idx] += float(np.sum(token_matches[:, horizon_idx] & valid))
        totals["horizon_counts"][horizon_idx] += weight

    if "legal_masks" in batch:
        masks = np.asarray(batch["legal_masks"], dtype=bool)[:, :compare_horizon]
        valid_masks = np.asarray(batch.get("legal_masks_valid", np.ones(masks.shape[:2])), dtype=np.float32)
        for sample_idx in range(pred_actions.shape[0]):
            if not valid[sample_idx]:
                continue
            for horizon_idx in range(compare_horizon):
                if valid_masks[sample_idx, horizon_idx] <= 0:
                    continue
                totals["teacher_forced_legal_count"] += 1.0
                totals["teacher_forced_legal_hits"] += float(masks[sample_idx, horizon_idx, pred_actions[sample_idx, horizon_idx]])

    if "fen_t" in batch:
        fens = np.asarray(batch["fen_t"])
        totals["exact_available"] += weight
        for sample_idx, fen in enumerate(fens):
            if not valid[sample_idx]:
                continue
            is_legal, _ = exact_sequence_legal(pred_actions[sample_idx], str(fen))
            target_legal, _ = exact_sequence_legal(target_actions[sample_idx], str(fen))
            totals["exact_legal"] += float(is_legal)
            totals["target_exact_legal"] += float(target_legal)


def _finalize_sampler_metrics(totals: dict[str, Any]) -> dict[str, float | bool]:
    sample_count = max(float(totals["sample_count"]), 1.0)
    metrics: dict[str, float | bool] = {
        "sample_count": float(totals["sample_count"]),
        "first_move_legal_rate": float(totals["first_move_legal_hits"] / sample_count),
        "first_move_match_rate": float(totals["first_move_match_hits"] / sample_count),
        "action_token_match_rate": float(totals["action_token_match_hits"] / max(totals["action_token_match_count"], 1.0)),
        "sampling_time_s": float(totals["sampling_time_s"]),
    }
    for horizon_idx, count in enumerate(totals["horizon_counts"]):
        if count > 0:
            metrics[f"action_token_match_h{horizon_idx + 1}"] = float(totals["horizon_hits"][horizon_idx] / count)
    if totals["teacher_forced_legal_count"] > 0:
        metrics["teacher_forced_token_legal_rate"] = float(
            totals["teacher_forced_legal_hits"] / totals["teacher_forced_legal_count"]
        )
    metrics["full_sequence_legal_available"] = bool(totals["exact_available"] > 0 and chess is not None)
    if totals["exact_available"] > 0 and chess is not None:
        metrics["full_sequence_legal_rate"] = float(totals["exact_legal"] / totals["exact_available"])
        metrics["target_full_sequence_legal_rate"] = float(totals["target_exact_legal"] / totals["exact_available"])
    return metrics


def evaluate(args: argparse.Namespace, checkpoint_dir: Path) -> dict[str, Any]:
    if args.backend == "tpu":
        jax.distributed.initialize(initialization_timeout=1200)

    chunk_paths = discover_chunk_files(args.chunk_dir)
    if not chunk_paths:
        raise ValueError(f"No chunk files found in {args.chunk_dir}.")

    refinement_steps = parse_refinement_steps(args.refinement_steps)
    config = resolve_joint_config(checkpoint_dir)
    model_paths = default_bt4_paths(args.models_dir)
    params = load_mapped_bt4_params(models_dir=model_paths["models_dir"])
    model, optimizer = create_joint_components(params, config, seed=args.seed)
    payload = load_training_checkpoint(checkpoint_dir, model=model, optimizer=optimizer, step=args.step, strict=False)
    loaded_step = args.step or latest_checkpoint_step(checkpoint_dir)

    loader = LeelaChunkDataLoader(
        chunk_paths,
        batch_size=args.batch_size,
        seed=args.seed,
        horizon=config.horizon,
        shuffle_files=False,
        drop_last=False,
        include_metadata=True,
        batch_view="dfm_action",
    )

    totals_by_steps = {steps: _empty_sampler_totals(config.horizon) for steps in refinement_steps}
    batches_seen = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.batches:
            break
        batches_seen += 1

        current_planes = jnp.asarray(batch["current_planes"])
        legal_mask = jnp.asarray(batch["legal_mask"], dtype=bool)
        valid = np.asarray(batch.get("valid", np.ones((len(batch["current_planes"]),))), dtype=np.float32) > 0
        target_actions = np.asarray(batch["action_indices"], dtype=np.int32)[:, : config.horizon]

        encode_start = time.perf_counter()
        bt4_tokens = model.encode_bt4_tokens(current_planes)
        z_dfm = model.dfm_latents(bt4_tokens)
        jax.block_until_ready(z_dfm)
        encode_time_s = time.perf_counter() - encode_start

        for steps in refinement_steps:
            sample_start = time.perf_counter()
            pred_actions = refine_actions_from_latents(model, z_dfm, legal_mask, steps)
            pred_actions_np = np.asarray(jax.block_until_ready(pred_actions))
            # Report the standalone cost for this refinement count: one cached
            # BT4 encode plus the requested number of DFM planner passes.
            totals_by_steps[steps]["sampling_time_s"] += encode_time_s + (time.perf_counter() - sample_start)
            _update_sampler_totals(
                totals_by_steps[steps],
                pred_actions=pred_actions_np,
                target_actions=target_actions,
                batch=batch,
                valid=valid,
            )

    if batches_seen == 0:
        raise ValueError("No validation batches were evaluated.")

    return {
        "checkpoint_step": loaded_step,
        "checkpoint_keys": sorted(payload.keys()),
        "batch_count": batches_seen,
        "config": dataclasses.asdict(config),
        "refinement_caches_bt4": True,
        "refinement_steps": refinement_steps,
        "metrics_by_refinement_steps": {
            str(steps): _finalize_sampler_metrics(totals) for steps, totals in totals_by_steps.items()
        },
    }


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="joint-dfm-refine-eval-") as tmp:
        tmp_path = Path(tmp)
        checkpoint_dir = (
            sync_gcs_checkpoint(args.checkpoint_dir, tmp_path / "checkpoints")
            if args.checkpoint_dir.startswith("gs://")
            else Path(args.checkpoint_dir)
        )
        if args.gcs_val_prefix:
            args.chunk_dir = str(sync_gcs_tree(args.gcs_val_prefix, tmp_path / "val_shards"))
        metrics = evaluate(args, checkpoint_dir)

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
