#!/usr/bin/env python3
"""Evaluate DFM training loss and final refined action-sequence quality."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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
from chess_dfm_jax.training.dfm import DFMConfig, create_dfm_components, eval_dfm_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Local or gs:// DFM checkpoint directory.")
    parser.add_argument("--models-dir", default=None, help="Path to BT4 model files.")
    parser.add_argument("--chunk-dir", default=None, help="Validation trajectory-v2 chunk directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--out-json", default=None, help="Optional path to write metrics JSON.")
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument("--deterministic-t", type=float, default=0.5)
    parser.add_argument("--backend", choices=["cpu", "tpu"], default="cpu")
    parser.add_argument("--token-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--mlp-dim", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--encoder-dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--head-compute-dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--use-qk-gain", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-muon", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _metadata_config(checkpoint_dir: Path) -> dict[str, Any]:
    metadata = read_checkpoint_metadata(checkpoint_dir.parent / "checkpoint_state.json")
    config = metadata.get("config", {})
    return config if isinstance(config, dict) else {}


def _value_from_args_or_metadata(args: argparse.Namespace, metadata: dict[str, Any], name: str, default: Any) -> Any:
    arg_value = getattr(args, name)
    if arg_value is not None:
        return arg_value
    return metadata.get(name, default)


def resolve_dfm_config(args: argparse.Namespace, checkpoint_dir: Path) -> DFMConfig:
    metadata = _metadata_config(checkpoint_dir)
    return DFMConfig(
        token_dim=int(_value_from_args_or_metadata(args, metadata, "token_dim", 512)),
        num_layers=int(_value_from_args_or_metadata(args, metadata, "num_layers", 4)),
        num_heads=int(_value_from_args_or_metadata(args, metadata, "num_heads", 8)),
        mlp_dim=int(_value_from_args_or_metadata(args, metadata, "mlp_dim", 2048)),
        learning_rate=float(_value_from_args_or_metadata(args, metadata, "learning_rate", 3e-4)),
        weight_decay=float(_value_from_args_or_metadata(args, metadata, "weight_decay", 1e-4)),
        encoder_dtype=str(_value_from_args_or_metadata(args, metadata, "encoder_dtype", "float16")),
        compute_dtype=str(_value_from_args_or_metadata(args, metadata, "head_compute_dtype", "float32")),
        horizon=int(_value_from_args_or_metadata(args, metadata, "horizon", 1)),
        use_qk_gain=bool(_value_from_args_or_metadata(args, metadata, "use_qk_gain", False)),
        use_muon=bool(_value_from_args_or_metadata(args, metadata, "use_muon", False)),
    )


def sync_gcs_checkpoint(checkpoint_uri: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    source = checkpoint_uri.rstrip("/") + "/*"
    subprocess.run(
        ["/snap/google-cloud-cli/current/bin/gcloud", "storage", "cp", "--recursive", source, str(destination)],
        check=True,
    )
    return destination


def numeric_jax_batch(batch: dict[str, np.ndarray], deterministic_t: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            out[key] = array
    if 0.0 <= deterministic_t <= 1.0:
        out["deterministic_t"] = np.asarray(deterministic_t, dtype=np.float32)
    return out


def _refine_step_logits(model, planes: jnp.ndarray, actions: jnp.ndarray, t: jnp.ndarray, legal_mask: jnp.ndarray):
    logits = model(planes, actions, t)
    masked_first = jnp.where(legal_mask[:, None, :], logits[:, 0:1, :], -1e9)
    if logits.shape[1] == 1:
        return masked_first
    return jnp.concatenate([masked_first, logits[:, 1:, :]], axis=1)


def refine_actions(model, planes: np.ndarray, legal_mask: np.ndarray, refinement_steps: int) -> np.ndarray:
    batch_size = planes.shape[0]
    horizon = model.config.horizon
    mask_token = model.config.action_vocab_size
    x = jnp.full((batch_size, horizon), mask_token, dtype=jnp.int32)
    planes_jnp = jnp.asarray(planes)
    legal_mask_jnp = jnp.asarray(legal_mask, dtype=bool)

    for i in range(refinement_steps):
        t = jnp.full((batch_size,), i / refinement_steps, dtype=jnp.float32)
        logits = _refine_step_logits(model, planes_jnp, x, t, legal_mask_jnp)
        probs = jax.nn.softmax(logits, axis=-1)
        max_probs = jnp.max(probs, axis=-1)
        preds = jnp.argmax(logits, axis=-1)

        num_unmasked_target = (horizon * (i + 1)) // refinement_steps
        max_probs_all = jnp.where(x == mask_token, max_probs, 2.0)
        kth_idx = max(0, horizon - num_unmasked_target)
        thresholds = jnp.sort(max_probs_all, axis=-1)[:, kth_idx : kth_idx + 1]
        x = jnp.where(max_probs_all >= thresholds, preds, x)

    return np.asarray(jax.block_until_ready(x))


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


def add_weighted(totals: dict[str, float], key: str, value: float, weight: float) -> None:
    totals[key] = totals.get(key, 0.0) + float(value) * float(weight)


def evaluate(args: argparse.Namespace, checkpoint_dir: Path) -> dict[str, Any]:
    if args.backend == "tpu":
        jax.distributed.initialize(initialization_timeout=1200)

    chunk_paths = discover_chunk_files(args.chunk_dir)
    if not chunk_paths:
        raise ValueError(f"No chunk files found in {args.chunk_dir}.")

    config = resolve_dfm_config(args, checkpoint_dir)
    model_paths = default_bt4_paths(args.models_dir)
    params = load_mapped_bt4_params(models_dir=model_paths["models_dir"])
    model, _optimizer = create_dfm_components(params, config, seed=args.seed)
    payload = load_training_checkpoint(checkpoint_dir, model=model, step=args.step)
    loaded_step = args.step or latest_checkpoint_step(checkpoint_dir)

    loader = LeelaChunkDataLoader(
        chunk_paths,
        batch_size=args.batch_size,
        seed=args.seed,
        horizon=config.horizon,
        shuffle_files=False,
        drop_last=False,
        include_metadata=True,
    )

    rng = jax.random.PRNGKey(args.seed)
    totals: dict[str, float] = {}
    horizon_hits = np.zeros((config.horizon,), dtype=np.float64)
    horizon_counts = np.zeros((config.horizon,), dtype=np.float64)
    sample_count = 0
    exact_available = 0
    exact_legal = 0
    target_exact_legal = 0
    teacher_forced_legal_hits = 0
    teacher_forced_legal_count = 0
    batches_seen = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.batches:
            break
        batches_seen += 1

        eval_batch = numeric_jax_batch(batch, args.deterministic_t)
        rng, step_rng = jax.random.split(rng)
        loss, aux = eval_dfm_step(model, eval_batch, step_rng)
        jax.block_until_ready((loss, aux))

        valid = np.asarray(batch.get("valid", np.ones((len(batch["current_planes"]),))), dtype=np.float32) > 0
        weight = float(np.sum(valid))
        if weight <= 0:
            continue

        add_weighted(totals, "loss", float(loss), weight)
        for key, value in aux.items():
            add_weighted(totals, key, float(value), weight)

        pred_actions = refine_actions(
            model,
            np.asarray(batch["current_planes"], dtype=np.float32),
            np.asarray(batch["legal_mask"], dtype=bool),
            args.refinement_steps,
        )
        target_actions = np.asarray(batch["action_indices"], dtype=np.int32)[:, : config.horizon]
        compare_horizon = min(pred_actions.shape[1], target_actions.shape[1], config.horizon)
        pred_actions = pred_actions[:, :compare_horizon]
        target_actions = target_actions[:, :compare_horizon]

        first_pred = pred_actions[:, 0]
        first_legal = np.asarray(batch["legal_mask"], dtype=bool)[np.arange(len(first_pred)), first_pred]
        add_weighted(totals, "first_move_legal_rate", float(np.mean(first_legal[valid])), weight)
        add_weighted(totals, "first_move_match_rate", float(np.mean((first_pred == target_actions[:, 0])[valid])), weight)
        token_matches = pred_actions == target_actions
        add_weighted(totals, "action_token_match_rate", float(np.mean(token_matches[valid])), weight)

        for horizon_idx in range(compare_horizon):
            hits = token_matches[:, horizon_idx] & valid
            horizon_hits[horizon_idx] += float(np.sum(hits))
            horizon_counts[horizon_idx] += weight

        if "legal_masks" in batch:
            masks = np.asarray(batch["legal_masks"], dtype=bool)[:, :compare_horizon]
            valid_masks = np.asarray(batch.get("legal_masks_valid", np.ones(masks.shape[:2])), dtype=np.float32)
            for sample_idx in range(pred_actions.shape[0]):
                if not valid[sample_idx]:
                    continue
                for horizon_idx in range(compare_horizon):
                    if valid_masks[sample_idx, horizon_idx] <= 0:
                        continue
                    teacher_forced_legal_count += 1
                    teacher_forced_legal_hits += int(masks[sample_idx, horizon_idx, pred_actions[sample_idx, horizon_idx]])

        if "fen_t" in batch:
            fens = np.asarray(batch["fen_t"])
            exact_available += int(np.sum(valid))
            for sample_idx, fen in enumerate(fens):
                if not valid[sample_idx]:
                    continue
                is_legal, _ = exact_sequence_legal(pred_actions[sample_idx], str(fen))
                exact_legal += int(is_legal)
                target_legal, _ = exact_sequence_legal(target_actions[sample_idx], str(fen))
                target_exact_legal += int(target_legal)

        sample_count += int(weight)

    if sample_count == 0:
        raise ValueError("No validation samples were evaluated.")

    metrics: dict[str, Any] = {
        "checkpoint_step": loaded_step,
        "checkpoint_keys": sorted(payload.keys()),
        "sample_count": sample_count,
        "batch_count": batches_seen,
        "config": config.__dict__,
        "refinement_steps": args.refinement_steps,
        "deterministic_t": args.deterministic_t,
    }
    metrics.update({key: value / sample_count for key, value in totals.items()})
    for horizon_idx, count in enumerate(horizon_counts):
        if count > 0:
            metrics[f"action_token_match_h{horizon_idx + 1}"] = float(horizon_hits[horizon_idx] / count)
    if teacher_forced_legal_count > 0:
        metrics["teacher_forced_token_legal_rate"] = float(teacher_forced_legal_hits / teacher_forced_legal_count)
    metrics["full_sequence_legal_available"] = exact_available > 0 and chess is not None
    if exact_available > 0 and chess is not None:
        metrics["full_sequence_legal_rate"] = float(exact_legal / exact_available)
        metrics["target_full_sequence_legal_rate"] = float(target_exact_legal / exact_available)
    return metrics


def main() -> int:
    args = parse_args()
    if args.checkpoint_dir.startswith("gs://"):
        with tempfile.TemporaryDirectory(prefix="dfm-eval-ckpt-") as tmp:
            checkpoint_dir = sync_gcs_checkpoint(args.checkpoint_dir, Path(tmp) / "checkpoints")
            metrics = evaluate(args, checkpoint_dir)
    else:
        metrics = evaluate(args, Path(args.checkpoint_dir))

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
