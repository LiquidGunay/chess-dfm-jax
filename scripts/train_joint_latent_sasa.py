#!/usr/bin/env python3
"""Run Stage 1 Latent-SASA joint DFM+JEPA training."""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.bt4_theory import estimate_bt4_theory  # noqa: E402
from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.data.gcs_cache import GCSShardCache  # noqa: E402
from chess_dfm_jax.data.grain_loader import (  # noqa: E402
    GrainUnavailableError,
    create_grain_trajectory_batch_loader,
    create_grain_trajectory_loader,
)
from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files  # noqa: E402
from chess_dfm_jax.data.trajectory import build_synthetic_trajectory_shard, trajectory_joint_batch_from_npz  # noqa: E402
from chess_dfm_jax.paths import default_bt4_paths, project_root  # noqa: E402
from chess_dfm_jax.training.checkpoints import (  # noqa: E402
    create_checkpoint_manager,
    latest_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
    wait_for_checkpoint_completion,
)
from chess_dfm_jax.training.joint_latent_sasa import (  # noqa: E402
    JointLatentSASAConfig,
    create_joint_components,
    eval_joint_stage1_step,
    eval_joint_stage2_step,
    joint_coupling_gradient_diagnostics,
    joint_jepa_action_baseline_diagnostics,
    eval_joint_stage1_step_data_parallel,
    eval_joint_stage2_step_data_parallel,
    train_joint_stage1_step,
    train_joint_stage1_step_data_parallel,
    train_joint_stage1_step_donated,
    train_joint_stage2_step,
    train_joint_stage2_step_data_parallel,
    train_joint_stage2_step_donated,
)
from chess_dfm_jax.tracking import has_wandb_credentials, init_wandb_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--chunk-dir", type=str, default=None)
    parser.add_argument("--val-chunk-dir", type=str, default=None)
    parser.add_argument("--gcs-train-prefix", type=str, default="")
    parser.add_argument("--gcs-val-prefix", type=str, default="")
    parser.add_argument("--gcs-cache-dir", type=str, default="/tmp/chess_dfm_jax/gcs_cache")
    parser.add_argument("--gcs-prefetch-interval-s", type=int, default=60)
    parser.add_argument("--gcs-prefetch-workers", type=int, default=2)
    parser.add_argument("--gcs-max-cached-train-shards", type=int, default=0)
    parser.add_argument("--gcs-min-train-shards", type=int, default=1)
    parser.add_argument("--gcs-min-val-shards", type=int, default=1)
    parser.add_argument("--gcs-startup-cache-policy", choices=["all", "minimum"], default="all")
    parser.add_argument(
        "--loader-prefetch-batches",
        type=int,
        default=2,
        help="Background host batches to prepare ahead of the TPU step. Set 0 to disable.",
    )
    parser.add_argument(
        "--data-loader",
        type=str,
        default="leela",
        choices=["leela", "grain"],
        help="Data loading backend for trajectory .npz shards. Grain is optional and not used by default.",
    )
    parser.add_argument("--grain-workers", type=int, default=0, help="Grain worker count when --data-loader=grain.")
    parser.add_argument(
        "--grain-shard-cache-size",
        type=int,
        default=2,
        help="Decoded shard LRU size inside the optional Grain random-access source.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=10_000)
    parser.add_argument("--backend", type=str, default="tpu", choices=["tpu", "cpu"])
    parser.add_argument("--wandb-project", type=str, default="chess_dfm_jax-joint")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="joint-latent-sasa")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--z-dim", type=int, default=2048)
    parser.add_argument("--projector-layers", type=int, default=2)
    parser.add_argument("--projector-num-heads", type=int, default=8)
    parser.add_argument("--projector-mlp-dim", type=int, default=0)
    parser.add_argument("--jepa-condition-dim", type=int, default=0)
    parser.add_argument("--dfm-layers", type=int, default=4)
    parser.add_argument("--jepa-layers", type=int, default=4)
    parser.add_argument("--jepa-num-heads", type=int, default=0, help="Deprecated for vector JEPA; kept for checkpoint metadata compatibility.")
    parser.add_argument("--jepa-mlp-dim", type=int, default=0, help="Vector JEPA MLP width; defaults to 4x --z-dim.")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--bt4-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--param-dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--compute-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use-qk-gain", action="store_true")
    parser.add_argument(
        "--use-qk-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RMS-normalize Q and K per head before attention logits in trainable DFM/JEPA stacks.",
    )
    parser.add_argument(
        "--use-xsa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Exclusive Self-Attention in trainable DFM/JEPA stacks.",
    )
    parser.add_argument(
        "--use-muon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Muon for square-ish 2D matrices and AdamW fallback for the rest.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Global gradient clipping norm. 0 disables clipping.")
    parser.add_argument("--lr-warmup-steps", type=int, default=1000, help="Linearly warm learning rate from 0 to --learning-rate over N steps.")
    parser.add_argument(
        "--skip-nonfinite-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optimizer updates whose transformed gradients contain NaN/Inf.",
    )
    parser.add_argument(
        "--unfreeze-bt4-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train BT4 input embedding and encoder layers with --bt4-learning-rate.",
    )
    parser.add_argument(
        "--bt4-encode-chunk-size",
        type=int,
        default=0,
        help=(
            "Chunk size over current+future BT4 target encodes. 0 flattens all horizons into one "
            "large encoder batch; 1 scans one board state per sample at a time to reduce HBM when "
            "BT4 is trainable."
        ),
    )
    parser.add_argument(
        "--jepa-state-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply one learned RMSNorm to the JEPA recurrent input state and every recurrent output state.",
    )
    parser.add_argument(
        "--jepa-state-rms-scale-max",
        type=float,
        default=2.0,
        help="Clip the effective learned JEPA state RMSNorm scale to [1/max, max]. 0 disables clipping.",
    )
    parser.add_argument(
        "--jepa-teacher-forcing-steps",
        type=int,
        default=0,
        help="Use true z(t+h) as the JEPA transition input for this many initial training steps, then free-roll out.",
    )
    parser.add_argument(
        "--jepa-delta-rms-clip",
        type=float,
        default=2.0,
        help=(
            "Clip each JEPA transition residual delta to this per-sample RMS before adding it to the "
            "latent state. This stabilizes recurrent rollout without hard-normalizing z itself. 0 disables."
        ),
    )
    parser.add_argument(
        "--remat-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rematerialize trainable DFM/JEPA transformer blocks during backward pass.",
    )
    parser.add_argument(
        "--scan-layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use pure lax.scan over stacked DFM/JEPA layer parameters. "
            "Default is off for shallow single-chip speed; enable for larger-depth memory/compile experiments."
        ),
    )
    parser.add_argument(
        "--donate-train-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Donate model and optimizer buffers to the jitted train step.",
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--loss-horizon", type=int, default=0)
    parser.add_argument("--dfm-ce-coeff", type=float, default=1.0)
    parser.add_argument("--legality-coeff", type=float, default=None, help="Deprecated alias for --first-legality-coeff.")
    parser.add_argument("--first-legality-coeff", type=float, default=None)
    parser.add_argument(
        "--horizon-legality-coeff",
        type=float,
        default=0.0,
        help="Deprecated no-op for joint training. Later generated-prefix legality belongs in eval/sampling.",
    )
    parser.add_argument("--legality-on-masked-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jepa-positive-coeff", type=float, default=1.0)
    parser.add_argument(
        "--jepa-loss-type",
        type=str,
        default="raw_mse",
        choices=["raw_mse"],
        help="JEPA term optimized by --jepa-positive-coeff.",
    )
    parser.add_argument(
        "--jepa-target-mode",
        type=str,
        default="projected_bt4",
        choices=["projected_bt4", "current_repeat"],
        help=(
            "JEPA target source. projected_bt4 is the real training objective. "
            "current_repeat is a profiling-only diagnostic."
        ),
    )
    parser.add_argument(
        "--jepa-target-sample-count",
        type=int,
        default=0,
        help=(
            "Number of future horizons to supervise per step. 0 means all horizons. "
            "Use 1 for stochastic target-horizon sampling to reduce future BT4 target encodes."
        ),
    )
    parser.add_argument("--jepa-gamma", type=float, default=1.0)
    parser.add_argument("--jepa-sigreg-coeff", type=float, default=0.1)
    parser.add_argument(
        "--jepa-pred-sigreg-coeff",
        type=float,
        default=0.0,
        help="SigReg coefficient on recurrent JEPA predictions, separate from projected-BT4 target SigReg.",
    )
    parser.add_argument(
        "--jepa-sigreg-kind",
        type=str,
        default="le_jepa",
        choices=["le_jepa", "moments", "quantile"],
        help="SigReg implementation. le_jepa uses fixed random projections and Gaussian quantile matching.",
    )
    parser.add_argument("--jepa-sigreg-proj-dim", type=int, default=1024)
    parser.add_argument("--value-coeff", type=float, default=0.0)
    parser.add_argument("--wdl-coeff", type=float, default=0.0)
    parser.add_argument(
        "--loss-clip-value",
        type=float,
        default=0.0,
        help="If >0, rescale oversized scalar losses to this value while preserving gradient direction.",
    )
    parser.add_argument("--jepa-action-contrast-coeff", type=float, default=0.0)
    parser.add_argument("--jepa-action-contrast-margin", type=float, default=0.05)
    parser.add_argument(
        "--target-projector-mode",
        type=str,
        default=None,
        choices=["shared", "separate"],
        help="Deprecated no-op. Joint JEPA now targets projected BT4 state vectors.",
    )
    parser.add_argument("--contrastive-coeff", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--train-deterministic-t", type=float, default=-1.0)
    parser.add_argument("--val-deterministic-t", type=float, default=0.0)
    parser.add_argument("--legal-lmax", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--diagnostics-every", type=int, default=0)
    parser.add_argument("--max-to-keep", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-format",
        type=str,
        default="raw",
        choices=["raw", "raw+orbax"],
        help="Checkpoint backend. raw is existing .npz; raw+orbax also writes an Orbax mirror.",
    )
    parser.add_argument(
        "--async-orbax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Orbax AsyncCheckpointer when --checkpoint-format includes orbax and supported.",
    )
    parser.add_argument(
        "--init-checkpoint-uri",
        type=str,
        default="",
        help=(
            "Optional checkpoint root to initialize model/optimizer state from while "
            "writing checkpoints under this run's --checkpoint-uri. This is for "
            "curriculum branching; use --resume for ordinary same-run continuation."
        ),
    )
    parser.add_argument(
        "--init-checkpoint-step",
        type=int,
        default=0,
        help="Checkpoint step to load from --init-checkpoint-uri. 0 means latest.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--checkpoint-uri", type=str, default=None)
    parser.add_argument("--peak-tflops", type=float, default=197.0)
    parser.add_argument(
        "--data-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shard each global batch over local devices with replicated model/optimizer state and pmean gradients.",
    )
    parser.add_argument(
        "--data-parallel-devices",
        type=int,
        default=0,
        help="Number of local devices to use when --data-parallel is enabled. 0 means all local devices.",
    )
    parser.add_argument("--profile-dir", type=str, default="", help="Local TensorBoard trace directory.")
    parser.add_argument("--profile-uri", type=str, default="", help="Optional gs:// prefix for uploading trace artifacts.")
    parser.add_argument("--profile-start-step", type=int, default=20, help="1-indexed step at which to start JAX tracing.")
    parser.add_argument("--profile-steps", type=int, default=0, help="Number of train steps to capture in a JAX trace.")
    parser.add_argument("--disable-final-checkpoint", action="store_true", help="Skip forced final checkpoint; useful for profiling-only runs.")
    return parser.parse_args()


def resolve_run_name(args: argparse.Namespace) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"joint-sasa-h{args.horizon}-d{args.token_dim}-{timestamp}"


def flatten_aux_metrics(metrics: dict[str, object]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.ndim == 0:
            flat[key] = float(array)
        elif array.ndim == 1:
            for idx, item in enumerate(array):
                flat[f"{key}_h{idx + 1}"] = float(item)
        else:
            flat[key] = float(np.mean(array))
    return flat


def add_validation_prefix(metrics: dict[str, float], *, prefix: str = "val_") -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def split_train_validation_chunks(
    chunk_paths: list[str],
    *,
    val_fraction: float,
) -> tuple[list[str], list[str]]:
    if val_fraction <= 0.0 or not chunk_paths:
        return chunk_paths, []
    if val_fraction >= 1.0:
        raise ValueError("--val-fraction must be < 1.0 so at least one training shard remains.")
    val_count = max(1, int(round(len(chunk_paths) * val_fraction)))
    val_count = min(val_count, len(chunk_paths) - 1)
    return chunk_paths[:-val_count], chunk_paths[-val_count:]


def make_gcs_cache(
    *,
    prefix: str,
    cache_dir: Path,
    poll_interval_s: int,
    download_workers: int,
    max_cached_shards: int = 0,
    seed: int = 0,
) -> GCSShardCache | None:
    if not prefix:
        return None
    cache = GCSShardCache(
        prefix,
        cache_dir,
        poll_interval_s=poll_interval_s,
        download_workers=download_workers,
        max_cached_shards=max_cached_shards,
        seed=seed,
    )
    cache.start()
    return cache


def _gcloud_binary() -> str:
    snap_path = Path("/snap/google-cloud-cli/current/bin/gcloud")
    return str(snap_path) if snap_path.exists() else "gcloud"


def sync_checkpoint_uri(checkpoint_uri: str, destination: Path, *, step: int | None = None) -> Path:
    if not checkpoint_uri.startswith("gs://"):
        return Path(checkpoint_uri)
    destination.mkdir(parents=True, exist_ok=True)
    sync_step = step if step is not None else latest_checkpoint_step(checkpoint_uri)
    if sync_step is None:
        return destination
    step_name = f"step{int(sync_step):07d}"
    (destination / step_name).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _gcloud_binary(),
            "storage",
            "cp",
            f"{checkpoint_uri.rstrip('/')}/{step_name}/state.npz",
            str(destination / step_name / "state.npz"),
        ],
        check=True,
    )
    return destination


def _run_root_from_checkpoint_uri(checkpoint_uri: str) -> str:
    uri = checkpoint_uri.rstrip("/")
    if uri.endswith("/checkpoints"):
        return uri[: -len("/checkpoints")]
    return uri.rsplit("/", 1)[0]


def sync_run_sidecars(output_dir: Path, checkpoint_uri: str | None) -> None:
    """Upload non-checkpoint run artifacts next to the GCS checkpoint root."""
    if not checkpoint_uri or jax.process_index() != 0:
        return
    run_root = _run_root_from_checkpoint_uri(checkpoint_uri)
    for name in ("metrics.jsonl", "checkpoint_state.json", "run_config.json"):
        path = output_dir / name
        if not path.exists():
            continue
        subprocess.run(
            [_gcloud_binary(), "storage", "cp", str(path), f"{run_root}/{name}"],
            check=False,
        )


def sync_profile_dir(profile_dir: Path, profile_uri: str) -> None:
    if not profile_uri or jax.process_index() != 0:
        return
    subprocess.run(
        [_gcloud_binary(), "storage", "cp", "--recursive", str(profile_dir), profile_uri.rstrip("/") + "/"],
        check=False,
    )


def build_synthetic_joint_batch(batch_size: int, horizon: int, legal_lmax: int) -> dict[str, np.ndarray]:
    cache_key = (int(batch_size), int(horizon), int(legal_lmax))
    cached = _SYNTHETIC_JOINT_BATCH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    shard = build_synthetic_trajectory_shard(batch_size=batch_size, horizon=horizon)
    payload = {
        "schema_version": np.asarray("trajectory-v2"),
        "planes_t": shard.planes_t,
        "actions": shard.actions,
        "planes_future": shard.planes_future,
        "future_valid": shard.future_valid,
        "legal_masks": shard.legal_masks,
        "value_targets": shard.value_targets,
        "wdl_targets": shard.wdl_targets,
    }
    batch = trajectory_joint_batch_from_npz(payload, horizon=horizon, legal_lmax=legal_lmax)
    _SYNTHETIC_JOINT_BATCH_CACHE[cache_key] = batch
    return batch


_SYNTHETIC_JOINT_BATCH_CACHE: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}


def estimate_joint_step_flops(
    *,
    batch_size: int,
    horizon: int,
    token_dim: int,
    dfm_layers: int,
    jepa_layers: int,
    jepa_width: int,
    jepa_mlp_dim: int,
    mlp_dim: int,
    bt4_encoder_forward_flops_per_batch: float = 0.0,
    bt4_encoder_forward_count: int | float | None = None,
    stage: str = "stage1",
    candidate_count: int = 1,
) -> dict[str, float]:
    seq_len = 64 + horizon
    state_len = 64
    dfm_swiglu_dim = max(1, int(round((2.0 / 3.0) * mlp_dim)))
    jepa_swiglu_dim = max(1, int(round((2.0 / 3.0) * jepa_mlp_dim)))

    def dfm_forward(batch: int) -> float:
        return batch * dfm_layers * (
            4.0 * seq_len * token_dim * token_dim
            + 3.0 * seq_len * token_dim * dfm_swiglu_dim
            + 2.0 * seq_len * seq_len * token_dim
        )

    def jepa_forward(batch: int) -> float:
        return batch * horizon * jepa_layers * (
            4.0 * state_len * jepa_width * jepa_width
            + 3.0 * state_len * jepa_width * jepa_swiglu_dim
            + 2.0 * state_len * state_len * jepa_width
        )

    # Stage 1 has two DFM passes: noisy-action CE and clean-action hidden state coupling.
    stage1_dfm_train = 3.0 * dfm_forward(batch_size) * 2.0
    stage1_jepa_train = 3.0 * jepa_forward(batch_size)
    contrastive_dfm_train = 0.0
    contrastive_jepa_train = 0.0
    if stage == "stage2" and candidate_count > 0:
        flat_candidate_batch = batch_size * candidate_count
        contrastive_dfm_train = 3.0 * dfm_forward(flat_candidate_batch)
        contrastive_jepa_train = 3.0 * jepa_forward(flat_candidate_batch)
    dfm_train = stage1_dfm_train + contrastive_dfm_train
    jepa_train = stage1_jepa_train + contrastive_jepa_train
    encoder_forward_count = float(horizon + 1 if bt4_encoder_forward_count is None else bt4_encoder_forward_count)
    bt4_encoder_forward = encoder_forward_count * float(bt4_encoder_forward_flops_per_batch)
    trainable_train = dfm_train + jepa_train
    return {
        "estimated_stage1_dfm_train_flops_per_step": float(stage1_dfm_train),
        "estimated_stage1_jepa_train_flops_per_step": float(stage1_jepa_train),
        "estimated_contrastive_dfm_train_flops_per_step": float(contrastive_dfm_train),
        "estimated_contrastive_jepa_train_flops_per_step": float(contrastive_jepa_train),
        "estimated_dfm_train_flops_per_step": float(dfm_train),
        "estimated_jepa_train_flops_per_step": float(jepa_train),
        "estimated_trainable_step_flops": float(trainable_train),
        "estimated_bt4_encoder_forward_flops_per_step": float(bt4_encoder_forward),
        "estimated_bt4_encoder_forward_count": float(encoder_forward_count),
        "estimated_total_step_flops": float(trainable_train + bt4_encoder_forward),
        "estimated_jepa_width": float(jepa_width),
        "estimated_jepa_mlp_dim": float(jepa_mlp_dim),
        "estimated_dfm_swiglu_dim": float(dfm_swiglu_dim),
        "estimated_jepa_swiglu_dim": float(jepa_swiglu_dim),
    }


def shard_batch_for_data_parallel(
    batch: dict[str, np.ndarray],
    *,
    device_count: int,
    global_batch_size: int,
) -> dict[str, np.ndarray]:
    if device_count <= 1:
        return batch
    if global_batch_size % device_count != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"data_parallel_devices={device_count}."
        )
    per_device = global_batch_size // device_count
    sharded: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        array = np.asarray(value)
        if array.ndim == 0:
            if key in ("deterministic_t", "jepa_teacher_forcing"):
                sharded[key] = np.broadcast_to(array, (device_count,))
                continue
            raise ValueError(f"Cannot shard scalar batch leaf {key!r} for data parallel training.")
        if array.shape[0] != global_batch_size:
            raise ValueError(
                f"Cannot shard batch leaf {key!r}: leading dim {array.shape[0]} "
                f"!= global batch size {global_batch_size}."
            )
        sharded[key] = array.reshape((device_count, per_device, *array.shape[1:]))
    return sharded


def unreplicate_data_parallel_tree(tree, *, device_count: int):
    if device_count <= 1:
        return tree

    def first_replica(value):
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == device_count:
            return array[0]
        return value

    return jax.tree_util.tree_map(first_replica, tree)


def evaluate_validation_batches(
    *,
    model,
    loader_obj: LeelaChunkDataLoader | None,
    val_batches: int,
    rng: jax.Array,
    batch_size: int,
    horizon: int,
    legal_lmax: int,
    deterministic_t: float,
    stage: str,
    data_parallel_devices: int = 1,
) -> tuple[dict[str, float], jax.Array]:
    loader = iter(loader_obj) if loader_obj is not None else None
    totals: dict[str, float] = {}
    fetch_time_total = 0.0
    eval_time_total = 0.0
    count = 0
    for _ in range(val_batches):
        fetch_start = time.perf_counter()
        if loader is None:
            batch = build_synthetic_joint_batch(batch_size, horizon, legal_lmax)
        else:
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(loader_obj)
                batch = next(loader)
        fetch_time_total += time.perf_counter() - fetch_start
        if 0.0 <= deterministic_t <= 1.0:
            batch = dict(batch)
            batch["deterministic_t"] = np.asarray(deterministic_t, dtype=np.float32)
        rng, eval_rng = jax.random.split(rng)
        eval_start = time.perf_counter()
        if data_parallel_devices > 1:
            dp_batch = shard_batch_for_data_parallel(
                batch,
                device_count=data_parallel_devices,
                global_batch_size=batch_size,
            )
            dp_rng = jax.random.split(eval_rng, data_parallel_devices)
            if stage == "stage2":
                loss, aux = eval_joint_stage2_step_data_parallel(model, dp_batch, dp_rng)
            else:
                loss, aux = eval_joint_stage1_step_data_parallel(model, dp_batch, dp_rng)
            jax.block_until_ready((loss, aux))
            loss = unreplicate_data_parallel_tree(loss, device_count=data_parallel_devices)
            aux = unreplicate_data_parallel_tree(aux, device_count=data_parallel_devices)
        elif stage == "stage2":
            loss, aux = eval_joint_stage2_step(model, batch, eval_rng)
        else:
            loss, aux = eval_joint_stage1_step(model, batch, eval_rng)
        if data_parallel_devices <= 1:
            jax.block_until_ready((loss, aux))
        eval_time_total += time.perf_counter() - eval_start
        batch_metrics = {"loss": float(loss)}
        batch_metrics.update(flatten_aux_metrics(aux))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        return {}, rng
    averaged = {key: value / count for key, value in totals.items()}
    averaged["data_fetch_time_s"] = fetch_time_total / count
    averaged["eval_step_time_s"] = eval_time_total / count
    return averaged, rng


def main() -> int:
    args = parse_args()
    if not args.no_wandb and not has_wandb_credentials():
        raise SystemExit(
            "W&B logging is enabled but no non-interactive credentials were found. "
            "Set WANDB_API_KEY, create a restricted ~/.netrc entry for api.wandb.ai, "
            "set WANDB_MODE=offline, or pass --no-wandb."
        )

    if args.backend == "tpu":
        jax.distributed.initialize(initialization_timeout=1200)

    data_parallel_devices = 1
    if args.data_parallel:
        available_devices = jax.local_device_count()
        requested_devices = args.data_parallel_devices if args.data_parallel_devices > 0 else available_devices
        if requested_devices < 1:
            raise ValueError("--data-parallel-devices must be positive or 0 for all local devices.")
        if requested_devices > available_devices:
            raise ValueError(
                f"Requested {requested_devices} data-parallel devices, but only "
                f"{available_devices} local devices are available."
            )
        if args.batch_size % requested_devices != 0:
            raise ValueError(
                f"--batch-size={args.batch_size} must be divisible by "
                f"data_parallel_devices={requested_devices}."
            )
        data_parallel_devices = requested_devices

    run_name = args.run_id or resolve_run_name(args)
    save_root = Path(args.save_dir) if args.save_dir else (project_root() / "runs" / "joint")
    output_dir = save_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoint_root = output_dir / "checkpoints"
    local_checkpoint_root.mkdir(parents=True, exist_ok=True)
    gcs_checkpoint_root = args.checkpoint_uri
    metadata_path = output_dir / "checkpoint_state.json"

    model_paths = default_bt4_paths(args.models_dir)
    params = load_mapped_bt4_params(models_dir=model_paths["models_dir"])
    first_legality_coeff = (
        args.first_legality_coeff
        if args.first_legality_coeff is not None
        else args.legality_coeff
        if args.legality_coeff is not None
        else 7.64
    )
    config = JointLatentSASAConfig(
        token_dim=args.token_dim,
        z_dim=args.z_dim,
        projector_layers=args.projector_layers,
        projector_num_heads=args.projector_num_heads,
        projector_mlp_dim=args.projector_mlp_dim,
        jepa_condition_dim=args.jepa_condition_dim,
        dfm_layers=args.dfm_layers,
        jepa_layers=args.jepa_layers,
        jepa_num_heads=args.jepa_num_heads,
        jepa_mlp_dim=args.jepa_mlp_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        learning_rate=args.learning_rate,
        bt4_learning_rate=args.bt4_learning_rate,
        weight_decay=args.weight_decay,
        encoder_dtype=args.encoder_dtype,
        param_dtype=args.param_dtype,
        compute_dtype=args.compute_dtype,
        horizon=args.horizon,
        loss_horizon=args.loss_horizon,
        dfm_ce_coeff=args.dfm_ce_coeff,
        first_legality_coeff=first_legality_coeff,
        horizon_legality_coeff=args.horizon_legality_coeff,
        legality_on_masked_only=args.legality_on_masked_only,
        jepa_positive_coeff=args.jepa_positive_coeff,
        jepa_loss_type=args.jepa_loss_type,
        jepa_target_mode=args.jepa_target_mode,
        jepa_target_sample_count=args.jepa_target_sample_count,
        jepa_gamma=args.jepa_gamma,
        jepa_sigreg_coeff=args.jepa_sigreg_coeff,
        jepa_pred_sigreg_coeff=args.jepa_pred_sigreg_coeff,
        jepa_sigreg_kind=args.jepa_sigreg_kind,
        jepa_sigreg_proj_dim=args.jepa_sigreg_proj_dim,
        value_coeff=args.value_coeff,
        wdl_coeff=args.wdl_coeff,
        loss_clip_value=args.loss_clip_value,
        jepa_action_contrast_coeff=args.jepa_action_contrast_coeff,
        jepa_action_contrast_margin=args.jepa_action_contrast_margin,
        contrastive_coeff=args.contrastive_coeff,
        contrastive_temperature=args.contrastive_temperature,
        candidate_count=args.candidate_count,
        use_qk_gain=args.use_qk_gain,
        use_qk_norm=args.use_qk_norm,
        use_xsa=args.use_xsa,
        use_muon=args.use_muon,
        grad_clip_norm=args.grad_clip_norm,
        lr_warmup_steps=args.lr_warmup_steps,
        skip_nonfinite_updates=args.skip_nonfinite_updates,
        unfreeze_bt4_encoder=args.unfreeze_bt4_encoder,
        bt4_encode_chunk_size=args.bt4_encode_chunk_size,
        jepa_state_rmsnorm=args.jepa_state_rmsnorm,
        jepa_state_rms_scale_max=args.jepa_state_rms_scale_max,
        jepa_teacher_forcing_steps=args.jepa_teacher_forcing_steps,
        jepa_delta_rms_clip=args.jepa_delta_rms_clip,
        remat_blocks=args.remat_blocks,
        scan_layers=args.scan_layers,
    )
    model, optimizer = create_joint_components(params, config, seed=args.seed)
    checkpoint_manager = create_checkpoint_manager(
        local_checkpoint_root,
        save_interval_steps=args.save_every,
        max_to_keep=args.max_to_keep,
        checkpoint_format=args.checkpoint_format,
        async_orbax=args.async_orbax,
    )

    start_step = 0
    if args.resume and args.init_checkpoint_uri:
        raise SystemExit("--resume and --init-checkpoint-uri are mutually exclusive.")
    if args.init_checkpoint_uri:
        init_dir = output_dir / "init_checkpoint"
        init_step_arg = args.init_checkpoint_step if args.init_checkpoint_step > 0 else None
        init_source = sync_checkpoint_uri(args.init_checkpoint_uri, init_dir, step=init_step_arg)
        init_step = init_step_arg if init_step_arg is not None else latest_checkpoint_step(init_source)
        if init_step is None:
            raise FileNotFoundError(f"No checkpoint found under --init-checkpoint-uri={args.init_checkpoint_uri!r}.")
        load_training_checkpoint(init_source, model=model, optimizer=None, step=init_step, strict=False)
        print(
            "Initialized model from checkpoint with non-strict migration: "
            f"source={args.init_checkpoint_uri} step={init_step}; optimizer=fresh"
        )
        sys.stdout.flush()
    if args.resume:
        resume_step = latest_checkpoint_step(local_checkpoint_root)
        source_root = local_checkpoint_root
        if resume_step is None and gcs_checkpoint_root and jax.process_index() == 0:
            sync_checkpoint_uri(gcs_checkpoint_root, local_checkpoint_root)
            resume_step = latest_checkpoint_step(local_checkpoint_root)
        if resume_step is not None:
            load_training_checkpoint(source_root, model=model, optimizer=optimizer, step=resume_step)
            start_step = int(resume_step)
            print(f"Resumed from step: {start_step}")
        else:
            print("No checkpoint found. Starting from scratch.")
        sys.stdout.flush()

    stop_requested = {"flag": False, "signal": None}

    def handle_signal(signum, _frame):
        stop_requested["flag"] = True
        stop_requested["signal"] = signal.Signals(signum).name
        print(f"signal_received={stop_requested['signal']}")
        sys.stdout.flush()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    train_cache = make_gcs_cache(
        prefix=args.gcs_train_prefix,
        cache_dir=Path(args.gcs_cache_dir) / "train",
        poll_interval_s=args.gcs_prefetch_interval_s,
        download_workers=args.gcs_prefetch_workers,
        max_cached_shards=args.gcs_max_cached_train_shards,
        seed=args.seed,
    )
    val_cache = make_gcs_cache(
        prefix=args.gcs_val_prefix,
        cache_dir=Path(args.gcs_cache_dir) / "val",
        poll_interval_s=args.gcs_prefetch_interval_s,
        download_workers=args.gcs_prefetch_workers,
        seed=args.val_seed,
    )
    if train_cache is not None:
        if args.gcs_startup_cache_policy == "all":
            train_cache.wait_for_all_visible()
        else:
            train_cache.wait_for_minimum(args.gcs_min_train_shards)
    if val_cache is not None and args.val_batches > 0:
        val_cache.wait_for_minimum(args.gcs_min_val_shards)

    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else None
    chunk_paths = (
        train_cache.local_paths()
        if train_cache is not None
        else discover_chunk_files(str(chunk_dir))
        if chunk_dir and chunk_dir.exists()
        else []
    )
    val_chunk_paths: list[str] = []
    if val_cache is not None:
        val_chunk_paths = val_cache.local_paths()
    elif args.val_chunk_dir:
        val_dir = Path(args.val_chunk_dir)
        val_chunk_paths = discover_chunk_files(str(val_dir)) if val_dir.exists() else []
        if not val_chunk_paths:
            raise ValueError(f"No validation chunk files found in {args.val_chunk_dir}.")
    elif args.val_fraction > 0.0:
        chunk_paths, val_chunk_paths = split_train_validation_chunks(chunk_paths, val_fraction=args.val_fraction)

    if args.chunk_dir and args.chunk_dir != "synthetic" and train_cache is None and not chunk_paths:
        raise ValueError(f"No chunk files found in {args.chunk_dir}. Aborting to prevent silent synthetic fallback.")

    data_source = "synthetic"
    loader = None
    loader_obj = None
    val_loader_obj = None
    if chunk_paths and (args.chunk_dir != "synthetic" or train_cache is not None):
        data_source = args.gcs_train_prefix or str(chunk_dir)
        if args.data_loader == "grain":
            if train_cache is not None:
                chunk_paths = train_cache.local_paths()
            try:
                loader_obj = create_grain_trajectory_batch_loader(
                    chunk_paths,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    horizon=args.horizon,
                    batch_view="joint_latent_sasa",
                    legal_lmax=args.legal_lmax,
                    shuffle=True,
                    drop_last=True,
                    worker_count=args.grain_workers,
                    prefetch_batches=args.loader_prefetch_batches,
                    shard_cache_size=args.grain_shard_cache_size,
                )
            except GrainUnavailableError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            loader_obj = LeelaChunkDataLoader(
                chunk_paths,
                batch_size=args.batch_size,
                seed=args.seed,
                horizon=args.horizon,
                batch_view="joint_latent_sasa",
                legal_lmax=args.legal_lmax,
                chunk_paths_provider=train_cache.local_paths if train_cache is not None else None,
                shuffle_files=True if train_cache is not None else False,
                drop_last=True,
                prefetch_batches=args.loader_prefetch_batches,
            )
        loader = iter(loader_obj)
    if val_chunk_paths:
        if args.data_loader == "grain":
            if val_cache is not None:
                val_chunk_paths = val_cache.local_paths()
            try:
                val_loader_obj = create_grain_trajectory_batch_loader(
                    val_chunk_paths,
                    batch_size=args.batch_size,
                    seed=args.val_seed,
                    horizon=args.horizon,
                    batch_view="joint_latent_sasa",
                    legal_lmax=args.legal_lmax,
                    shuffle=False,
                    drop_last=False,
                    worker_count=args.grain_workers,
                    prefetch_batches=max(0, min(args.loader_prefetch_batches, 2)),
                    shard_cache_size=args.grain_shard_cache_size,
                )
            except GrainUnavailableError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            val_loader_obj = LeelaChunkDataLoader(
                val_chunk_paths,
                batch_size=args.batch_size,
                seed=args.val_seed,
                horizon=args.horizon,
                batch_view="joint_latent_sasa",
                legal_lmax=args.legal_lmax,
                shuffle_files=False,
                drop_last=False,
                chunk_paths_provider=val_cache.local_paths if val_cache is not None else None,
                prefetch_batches=max(0, min(args.loader_prefetch_batches, 2)),
            )

    run_config = config.__dict__.copy()
    run_config.update(vars(args))
    run_config["first_legality_coeff"] = first_legality_coeff
    run_config["jepa_target_space"] = (
        "projected_bt4_vectors" if args.jepa_target_mode == "projected_bt4" else "current_projected_vector_repeated_diagnostic"
    )
    run_config["value_wdl_target_source"] = "trajectory shard value_targets/wdl_targets; LC0 PGN v3 targets are final-game-result derived"
    run_config.update(
        {
            "model_family": "joint_latent_sasa",
            "objective_stage": args.stage,
            "batch_view": "joint_latent_sasa",
            "data_loader": args.data_loader,
            "data_parallel": bool(args.data_parallel),
            "data_parallel_devices": data_parallel_devices,
            "data_parallel_per_device_batch_size": args.batch_size // data_parallel_devices,
            "checkpoint_format": args.checkpoint_format,
            "train_chunk_count": len(chunk_paths),
            "val_chunk_count": len(val_chunk_paths),
            "gcs_train_prefix": args.gcs_train_prefix,
            "gcs_val_prefix": args.gcs_val_prefix,
        }
    )
    target_encoder_count = 0 if args.jepa_target_mode == "current_repeat" else args.horizon
    encoder_forward_count = 1 + target_encoder_count
    if args.stage == "stage2":
        # Stage 2 also rolls out contrastive candidates. Current implementation encodes
        # candidate current boards plus all candidate future boards for valid negatives.
        encoder_forward_count += args.horizon + 1
    flops = estimate_joint_step_flops(
        batch_size=args.batch_size,
        horizon=args.horizon,
        token_dim=args.token_dim,
        dfm_layers=args.dfm_layers,
        jepa_layers=args.jepa_layers,
        jepa_width=args.z_dim,
        jepa_mlp_dim=args.jepa_mlp_dim if args.jepa_mlp_dim > 0 else args.z_dim * 4,
        mlp_dim=args.mlp_dim,
        bt4_encoder_forward_flops_per_batch=estimate_bt4_theory(
            params,
            batch_size=args.batch_size,
        ).encoder_forward_flops * (3.0 if args.unfreeze_bt4_encoder else 1.0),
        bt4_encoder_forward_count=encoder_forward_count,
        stage=args.stage,
        candidate_count=args.candidate_count,
    )
    run_config.update(flops)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    sync_run_sidecars(output_dir, gcs_checkpoint_root)

    run = None
    if jax.process_index() == 0 and not args.no_wandb:
        run = init_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            run_id=run_name,
            resume="allow" if args.resume else None,
            config=run_config,
        )

    print(f"backend={jax.default_backend()} process_index={jax.process_index()} process_count={jax.process_count()} device={jax.devices()[0]}")
    print(
        "data_parallel="
        f"{bool(args.data_parallel)} devices={data_parallel_devices} "
        f"per_device_batch={args.batch_size // data_parallel_devices}"
    )
    print(f"data_source={data_source}")
    print(f"models_dir={model_paths['models_dir']}")
    print(f"output_dir={output_dir}")
    print(f"checkpoint_uri={gcs_checkpoint_root}")
    print(f"horizon={args.horizon} stage={args.stage} batch_view=joint_latent_sasa")
    print(f"train_chunk_count={len(chunk_paths)} val_chunk_count={len(val_chunk_paths)}")
    sys.stdout.flush()

    metrics_log = (output_dir / "metrics.jsonl").open("a", encoding="utf-8")
    last_metrics: dict[str, float] | None = None
    completed_step = start_step
    rng = jax.random.PRNGKey(args.seed + jax.process_index())
    val_rng = jax.random.PRNGKey(args.val_seed + jax.process_index())
    val_every = args.val_every if args.val_every > 0 else args.save_every
    profile_dir = Path(args.profile_dir) if args.profile_dir else output_dir / "tb_trace"
    profile_enabled = args.profile_steps > 0
    profile_started = False
    profile_active = False

    try:
        for step in range(start_step, args.steps):
            iteration_start = time.perf_counter()
            rng, step_rng = jax.random.split(rng)
            fetch_start = time.perf_counter()
            if loader is not None:
                try:
                    batch = next(loader)
                except StopIteration:
                    loader = iter(loader_obj)
                    batch = next(loader)
            else:
                batch = build_synthetic_joint_batch(args.batch_size, args.horizon, args.legal_lmax)
            data_fetch_time = time.perf_counter() - fetch_start
            next_step = step + 1
            host_batch_start = time.perf_counter()
            if 0.0 <= args.train_deterministic_t <= 1.0:
                batch = dict(batch)
                batch["deterministic_t"] = np.asarray(args.train_deterministic_t, dtype=np.float32)
            if args.jepa_teacher_forcing_steps > 0:
                batch = dict(batch)
                batch["jepa_teacher_forcing"] = np.asarray(
                    1.0 if next_step <= args.jepa_teacher_forcing_steps else 0.0,
                    dtype=np.float32,
                )
            host_batch_time = time.perf_counter() - host_batch_start

            if (
                profile_enabled
                and not profile_started
                and next_step >= args.profile_start_step
                and jax.process_index() == 0
            ):
                profile_dir.mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(str(profile_dir), create_perfetto_trace=True)
                profile_started = True
                profile_active = True
                print(f"jax_profile_start step={next_step} dir={profile_dir}")
                sys.stdout.flush()

            train_step_start = time.perf_counter()
            with jax.profiler.StepTraceAnnotation("train_joint_latent_sasa", step_num=next_step):
                if data_parallel_devices > 1:
                    batch_for_step = shard_batch_for_data_parallel(
                        batch,
                        device_count=data_parallel_devices,
                        global_batch_size=args.batch_size,
                    )
                    step_rng_for_step = jax.random.split(step_rng, data_parallel_devices)
                    if args.stage == "stage2":
                        loss, aux = train_joint_stage2_step_data_parallel(
                            model,
                            optimizer,
                            batch_for_step,
                            step_rng_for_step,
                        )
                    else:
                        loss, aux = train_joint_stage1_step_data_parallel(
                            model,
                            optimizer,
                            batch_for_step,
                            step_rng_for_step,
                        )
                elif args.stage == "stage2":
                    train_fn = train_joint_stage2_step_donated if args.donate_train_state else train_joint_stage2_step
                    loss, aux = train_fn(model, optimizer, batch, step_rng)
                else:
                    train_fn = train_joint_stage1_step_donated if args.donate_train_state else train_joint_stage1_step
                    loss, aux = train_fn(model, optimizer, batch, step_rng)
            jax.block_until_ready((loss, aux))
            if data_parallel_devices > 1:
                loss = unreplicate_data_parallel_tree(loss, device_count=data_parallel_devices)
                aux = unreplicate_data_parallel_tree(aux, device_count=data_parallel_devices)
            step_time = time.perf_counter() - train_step_start
            completed_step = step + 1

            metrics = {
                "step": completed_step,
                "loss": float(loss),
                "step_time_s": step_time,
                "train_step_time_s": step_time,
                "data_fetch_time_s": data_fetch_time,
                "host_batch_time_s": host_batch_time,
                "loader_prefetch_batches": float(args.loader_prefetch_batches),
                "data_parallel_devices": float(data_parallel_devices),
                "data_parallel_per_device_batch_size": float(args.batch_size // data_parallel_devices),
            }
            metrics.update(flatten_aux_metrics(aux))
            if profile_active:
                metrics["jax_profile_active"] = 1.0
            examples_per_second = args.batch_size / max(step_time, 1e-12)
            metrics["examples_per_second"] = examples_per_second
            metrics.update(flops)
            total_tflops = flops["estimated_total_step_flops"] / max(step_time, 1e-12) / 1e12
            metrics["estimated_total_tflops"] = total_tflops
            metrics["estimated_total_mfu"] = total_tflops / args.peak_tflops if args.peak_tflops > 0 else 0.0
            if train_cache is not None:
                metrics.update({f"gcs_train_cache_{key}": value for key, value in train_cache.stats().items()})
            if val_cache is not None:
                metrics.update({f"gcs_val_cache_{key}": value for key, value in val_cache.stats().items()})

            if args.diagnostics_every > 0 and data_parallel_devices == 1 and (
                completed_step % args.diagnostics_every == 0 or completed_step == args.steps
            ):
                diagnostics_start = time.perf_counter()
                diag = joint_coupling_gradient_diagnostics(model, batch)
                jax.block_until_ready(diag)
                metrics.update(flatten_aux_metrics(diag))
                baseline_diag = joint_jepa_action_baseline_diagnostics(
                    model,
                    batch,
                    jax.random.fold_in(step_rng, completed_step),
                )
                jax.block_until_ready(baseline_diag)
                metrics.update(flatten_aux_metrics(baseline_diag))
                metrics["diagnostics_time_s"] = time.perf_counter() - diagnostics_start
            else:
                metrics["diagnostics_time_s"] = 0.0

            if args.val_batches > 0 and (completed_step % val_every == 0 or completed_step == args.steps):
                validation_start = time.perf_counter()
                val_metrics, val_rng = evaluate_validation_batches(
                    model=model,
                    loader_obj=val_loader_obj,
                    val_batches=args.val_batches,
                    rng=val_rng,
                    batch_size=args.batch_size,
                    horizon=args.horizon,
                    legal_lmax=args.legal_lmax,
                    deterministic_t=args.val_deterministic_t,
                    stage=args.stage,
                    data_parallel_devices=data_parallel_devices,
                )
                metrics.update(add_validation_prefix(val_metrics))
                metrics["validation_time_s"] = time.perf_counter() - validation_start
            else:
                metrics["validation_time_s"] = 0.0

            total_step_time = time.perf_counter() - iteration_start
            metrics["total_step_time_s"] = total_step_time
            metrics["iteration_examples_per_second"] = args.batch_size / max(total_step_time, 1e-12)
            metrics["estimated_iteration_total_tflops"] = (
                flops["estimated_total_step_flops"] / max(total_step_time, 1e-12) / 1e12
            )
            metrics["estimated_iteration_mfu"] = (
                metrics["estimated_iteration_total_tflops"] / args.peak_tflops if args.peak_tflops > 0 else 0.0
            )
            metrics["host_overhead_time_s"] = max(0.0, total_step_time - step_time)
            metrics["host_overhead_fraction"] = metrics["host_overhead_time_s"] / max(total_step_time, 1e-12)

            last_metrics = metrics
            metrics_log.write(json.dumps(metrics) + "\n")
            metrics_log.flush()

            if step % args.log_every == 0 or completed_step == args.steps:
                log_parts = [
                    f"step={completed_step}",
                    f"loss={metrics['loss']:.6f}",
                    f"dfm_ce={metrics.get('dfm_ce_loss', 0.0):.6f}",
                    f"legal={metrics.get('legality_loss', 0.0):.6f}",
                    f"first_legal={metrics.get('first_legality_loss', 0.0):.6f}",
                    f"horizon_legal={metrics.get('horizon_legality_loss', 0.0):.6f}",
                    f"jepa={metrics.get('jepa_positive_loss', 0.0):.6f}",
                    f"tf={metrics.get('jepa_teacher_forcing', 0.0):.0f}",
                    f"sigreg={metrics.get('jepa_sigreg_loss', 0.0):.6f}",
                    f"pred_sigreg={metrics.get('jepa_pred_sigreg_loss', 0.0):.6f}",
                    f"value={metrics.get('value_loss', 0.0):.6f}",
                    f"wdl={metrics.get('wdl_loss', 0.0):.6f}",
                    f"act_contrast={metrics.get('jepa_action_contrast_loss', 0.0):.6f}",
                    f"raw_mse={metrics.get('jepa_raw_mse', 0.0):.6f}",
                    f"z_std={metrics.get('z_state_std', 0.0):.4f}",
                    f"acc={metrics.get('accuracy', 0.0):.4f}",
                ]
                if "contrastive_loss" in metrics:
                    log_parts.extend(
                        [
                            f"contrast={metrics.get('contrastive_loss', 0.0):.6f}",
                            f"contrast_acc={metrics.get('contrastive_accuracy', 0.0):.4f}",
                            f"contrast_margin={metrics.get('contrastive_similarity_margin', 0.0):.6f}",
                            f"contrast_valid={metrics.get('contrastive_valid_fraction', 0.0):.4f}",
                        ]
                    )
                log_parts.extend(
                    [
                        f"step_time_s={metrics['step_time_s']:.3f}",
                        f"fetch_s={metrics.get('data_fetch_time_s', 0.0):.3f}",
                        f"host_frac={metrics.get('host_overhead_fraction', 0.0):.2f}",
                        f"ex_per_s={metrics['examples_per_second']:.1f}",
                        f"total_mfu={metrics['estimated_total_mfu']:.4f}",
                        f"iter_mfu={metrics.get('estimated_iteration_mfu', 0.0):.4f}",
                    ]
                )
                print(" ".join(log_parts))
                sys.stdout.flush()

            if run is not None and jax.process_index() == 0:
                run.log(metrics, step=completed_step)

            if profile_active and completed_step >= args.profile_start_step + args.profile_steps - 1:
                jax.profiler.stop_trace()
                profile_active = False
                print(f"jax_profile_stop step={completed_step} dir={profile_dir}")
                sys.stdout.flush()
                sync_profile_dir(profile_dir, args.profile_uri)

            should_save = (not args.disable_final_checkpoint) and (
                (completed_step % args.save_every == 0) or (completed_step == args.steps)
            )
            if should_save:
                save_training_checkpoint(
                    checkpoint_manager,
                    model=model,
                    optimizer=optimizer,
                    step=completed_step,
                    metrics=metrics,
                    metadata_path=metadata_path,
                    config=run_config,
                    extra={"last_metrics": metrics, "checkpoint_schema": f"joint_latent_sasa_{args.stage}"},
                )
                if jax.process_index() == 0 and gcs_checkpoint_root:
                    wait_for_checkpoint_completion(checkpoint_manager)
                    subprocess.run(
                        f"/snap/google-cloud-cli/current/bin/gcloud storage cp --recursive {shlex.quote(str(local_checkpoint_root))}/* {shlex.quote(gcs_checkpoint_root)}/",
                        shell=True,
                    )
                    sync_run_sidecars(output_dir, gcs_checkpoint_root)

            if stop_requested["flag"]:
                print(f"stopping_after_step={completed_step} stop_signal={stop_requested['signal']}")
                sys.stdout.flush()
                break
    finally:
        if profile_active:
            jax.profiler.stop_trace()
            profile_active = False
            sync_profile_dir(profile_dir, args.profile_uri)
        if train_cache is not None:
            train_cache.stop()
        if val_cache is not None:
            val_cache.stop()
        metrics_log.close()
        latest_saved_step = checkpoint_manager.latest_step()
        if not args.disable_final_checkpoint and completed_step > start_step and latest_saved_step != completed_step:
            save_training_checkpoint(
                checkpoint_manager,
                model=model,
                optimizer=optimizer,
                step=completed_step,
                metrics=last_metrics,
                metadata_path=metadata_path,
                config=run_config,
                extra={"last_metrics": last_metrics, "forced": True, "checkpoint_schema": f"joint_latent_sasa_{args.stage}"},
                force=True,
            )
        wait_for_checkpoint_completion(checkpoint_manager)
        if jax.process_index() == 0 and gcs_checkpoint_root:
            subprocess.run(
                f"/snap/google-cloud-cli/current/bin/gcloud storage cp --recursive {shlex.quote(str(local_checkpoint_root))}/* {shlex.quote(gcs_checkpoint_root)}/",
                shell=True,
            )
            sync_run_sidecars(output_dir, gcs_checkpoint_root)
        if run is not None:
            print(f"wandb_url={run.url}")
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
