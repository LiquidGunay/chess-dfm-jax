#!/usr/bin/env python3
"""Run DFM action-denoising training with optional W&B logging and raw NumPy checkpoints."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import subprocess
import shlex
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.bt4_theory import estimate_bt4_theory  # noqa: E402
from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.data.gcs_cache import GCSShardCache  # noqa: E402
from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files  # noqa: E402
from chess_dfm_jax.paths import default_bt4_paths, project_root  # noqa: E402
from chess_dfm_jax.training.checkpoints import (  # noqa: E402
    create_checkpoint_manager,
    latest_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
)
from chess_dfm_jax.training.dfm import (  # noqa: E402
    DFMConfig,
    create_dfm_components,
    eval_dfm_step,
    train_dfm_step,
)
from chess_dfm_jax.training.jepa import (  # noqa: E402
    build_synthetic_transition_batch,
)
from chess_dfm_jax.tracking import has_wandb_credentials, init_wandb_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to train.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Total batch size across all devices.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to BT4 models.")
    parser.add_argument("--chunk-dir", type=str, default=None, help="Path to Leela chunks.")
    parser.add_argument("--val-chunk-dir", type=str, default=None, help="Optional path to held-out validation chunks.")
    parser.add_argument("--gcs-train-prefix", type=str, default="", help="Optional GCS prefix containing train .npz shards to cache asynchronously.")
    parser.add_argument("--gcs-val-prefix", type=str, default="", help="Optional GCS prefix containing fixed validation .npz shards to cache asynchronously.")
    parser.add_argument("--gcs-cache-dir", type=str, default="/tmp/chess_dfm_jax/gcs_cache", help="Local cache root for GCS shard streaming.")
    parser.add_argument("--gcs-prefetch-interval-s", type=int, default=60)
    parser.add_argument("--gcs-prefetch-workers", type=int, default=2)
    parser.add_argument("--gcs-max-cached-train-shards", type=int, default=0, help="0 means cache all visible train shards.")
    parser.add_argument("--gcs-min-train-shards", type=int, default=1, help="Block startup until this many train shards are cached.")
    parser.add_argument("--gcs-min-val-shards", type=int, default=1, help="Block validation startup until this many val shards are cached.")
    parser.add_argument(
        "--gcs-startup-cache-policy",
        choices=["all", "minimum"],
        default="all",
        help=(
            "For GCS training data, 'all' blocks startup until every visible shard is local. "
            "'minimum' preserves the older growing-cache behavior."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Fraction of discovered training chunk files to reserve for validation when --val-chunk-dir is not set.",
    )
    parser.add_argument("--val-batches", type=int, default=0, help="Number of validation batches to evaluate. 0 disables validation.")
    parser.add_argument("--val-every", type=int, default=0, help="Run validation every N training steps. Defaults to --save-every when validation is enabled.")
    parser.add_argument("--val-seed", type=int, default=10_000, help="Seed for validation data ordering and DFM masks.")
    parser.add_argument(
        "--val-deterministic-t",
        type=float,
        default=-1.0,
        help="If in [0,1], evaluate validation at a fixed diffusion t instead of random t.",
    )
    parser.add_argument(
        "--val-t-values",
        type=str,
        default="",
        help=(
            "Optional comma-separated fixed t values for additional validation slices, "
            "for example '0.0,0.1,0.5,0.9'. Metrics are logged as val_tXX_*."
        ),
    )
    parser.add_argument("--backend", type=str, default="tpu", choices=["tpu", "cpu"], help="JAX backend.")
    parser.add_argument("--wandb-project", type=str, default="chess_dfm_jax-jepa", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity.")
    parser.add_argument("--wandb-group", type=str, default="jepa-training", help="W&B group.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--token-dim", type=int, default=512, help="DFM token dimension.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of DFM transformer layers.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--mlp-dim", type=int, default=2048, help="MLP hidden dimension.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--loss-horizon",
        type=int,
        default=0,
        help="Only train/evaluate masked CE over the first N action positions. 0 means full model horizon.",
    )
    parser.add_argument("--first-legality-loss-weight", type=float, default=2.0)
    parser.add_argument("--horizon-legality-loss-weight", type=float, default=0.0)
    parser.add_argument("--first-action-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--train-deterministic-t",
        type=float,
        default=-1.0,
        help="If in [0,1], train every batch at a fixed flow time t instead of sampling t uniformly.",
    )
    parser.add_argument("--encoder-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--head-param-dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help=argparse.SUPPRESS)
    parser.add_argument("--head-compute-dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--action-source", type=str, default="best", choices=["best", "played"])
    parser.add_argument("--use-qk-gain", action="store_true", help="Use QK gain scaling.")
    parser.add_argument("--use-xsa", action="store_true", help="Use Exclusive Self-Attention.")
    parser.add_argument("--use-muon", action="store_true", help="Use Muon optimizer.")
    parser.add_argument("--save-dir", type=str, default=None, help="Local path for saves.")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--max-to-keep", type=int, default=3, help="Max checkpoints to keep.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run ID.")
    parser.add_argument("--checkpoint-uri", type=str, default=None, help="Explicit checkpoint GCS URI.")
    parser.add_argument(
        "--init-checkpoint-uri",
        type=str,
        default=None,
        help=(
            "Load model weights from this checkpoint directory before training a new run. "
            "Optimizer state is not restored unless --init-restore-optimizer is set."
        ),
    )
    parser.add_argument(
        "--init-restore-optimizer",
        action="store_true",
        help="With --init-checkpoint-uri, also restore optimizer state while keeping a fresh run/checkpoint directory.",
    )
    parser.add_argument("--init-checkpoint-step", type=int, default=None, help="Checkpoint step for --init-checkpoint-uri. Defaults to latest.")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon.")
    parser.add_argument("--job-spec", type=str, default=None, help="Optional job spec JSON.")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=197.0,
        help="Approximate device peak TFLOP/s used for estimated MFU metrics. Set 0 to disable.",
    )
    return parser.parse_args()


def resolve_run_name(args: argparse.Namespace) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"dfm-{args.num_layers}l-{args.token_dim}d-{timestamp}"


def estimate_dfm_forward_flops(*, batch_size: int, horizon: int, token_dim: int, num_layers: int, num_heads: int, mlp_dim: int, vocab_size: int = 1858, encoder_width: int = 1024, board_tokens: int = 64) -> float:
    """Approximate DFM head forward FLOPs, excluding the frozen BT4 encoder."""
    del num_heads  # Head count changes sharding, not the dense FLOP count in this estimate.
    seq = board_tokens + horizon
    flops = 0.0
    flops += 2.0 * batch_size * board_tokens * encoder_width * token_dim  # z projection
    flops += 2.0 * batch_size * token_dim * token_dim  # timestep MLP
    for _ in range(num_layers):
        flops += 8.0 * batch_size * seq * token_dim * token_dim  # qkv + output projections
        flops += 4.0 * batch_size * seq * seq * token_dim  # attention score + value mixing
        flops += 4.0 * batch_size * seq * token_dim * mlp_dim  # MLP up/down
    flops += 2.0 * batch_size * horizon * token_dim * vocab_size  # action logits
    return flops


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


def add_validation_prefix(metrics: dict[str, float], *, prefix: str = "val_") -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def parse_val_t_values(raw: str) -> list[float]:
    if not raw.strip():
        return []
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--val-t-values entries must be in [0, 1], got {value}.")
        values.append(value)
    return values


def t_metric_prefix(t_value: float) -> str:
    return f"val_t{int(round(t_value * 100)):03d}_"


def flatten_aux_metrics(metrics: dict[str, object]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.ndim == 0:
            flat[key] = float(array)
            continue
        if array.ndim == 1:
            for idx, item in enumerate(array):
                flat[f"{key}_h{idx + 1}"] = float(item)
            continue
        flat[key] = float(np.mean(array))
    return flat


def evaluate_validation_batches(
    *,
    model,
    loader_obj: LeelaChunkDataLoader,
    val_batches: int,
    rng: jax.Array,
    deterministic_t: float,
) -> tuple[dict[str, float], jax.Array]:
    loader = iter(loader_obj)
    totals: dict[str, float] = {}
    count = 0
    for _ in range(val_batches):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(loader_obj)
            batch = next(loader)
        if 0.0 <= deterministic_t <= 1.0:
            batch = dict(batch)
            batch["deterministic_t"] = np.asarray(deterministic_t, dtype=np.float32)
        rng, step_rng = jax.random.split(rng)
        loss, aux = eval_dfm_step(model, batch, step_rng)
        jax.block_until_ready((loss, aux))
        batch_metrics = {"loss": float(loss)}
        batch_metrics.update(flatten_aux_metrics(aux))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        return {}, rng
    return {key: value / count for key, value in totals.items()}, rng


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


def sync_checkpoint_uri(checkpoint_uri: str, destination: Path, *, step: int | None = None) -> Path:
    """Sync one checkpoint step from a local or GCS checkpoint directory to a local path."""
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
            "/snap/google-cloud-cli/current/bin/gcloud",
            "storage",
            "cp",
            f"{checkpoint_uri.rstrip('/')}/{step_name}/state.npz",
            str(destination / step_name / "state.npz"),
        ],
        check=True,
    )
    return destination


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

    run_name = args.run_id or resolve_run_name(args)
    
    # Unify output directory logic
    save_root = Path(args.save_dir) if args.save_dir else (project_root() / "runs" / "dfm")
    output_dir = save_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    local_checkpoint_root = output_dir / "checkpoints"
    local_checkpoint_root.mkdir(parents=True, exist_ok=True)
    gcs_checkpoint_root = args.checkpoint_uri
    
    ckpt_paths = {
        "run_dir": output_dir,
        "checkpoint_dir": local_checkpoint_root,
        "metadata_path": output_dir / "checkpoint_state.json",
    }
    
    # Initialize components
    model_paths = default_bt4_paths(args.models_dir)
    params = load_mapped_bt4_params(models_dir=model_paths["models_dir"])
    config = DFMConfig(
        token_dim=args.token_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        encoder_dtype=args.encoder_dtype,
        compute_dtype=args.head_compute_dtype,
        horizon=args.horizon,
        use_qk_gain=args.use_qk_gain,
        use_xsa=args.use_xsa,
        use_muon=args.use_muon,
        loss_horizon=args.loss_horizon,
        first_legality_loss_weight=args.first_legality_loss_weight,
        horizon_legality_loss_weight=args.horizon_legality_loss_weight,
        first_action_loss_weight=args.first_action_loss_weight,
    )
    model, optimizer = create_dfm_components(params, config, seed=args.seed)

    checkpoint_manager = create_checkpoint_manager(
        local_checkpoint_root,
        save_interval_steps=args.save_every,
        max_to_keep=args.max_to_keep,
    )

    start_step = 0
    resumed = False
    if args.resume:
        resume_step = latest_checkpoint_step(local_checkpoint_root)
        source_root = local_checkpoint_root
        
        if jax.process_index() == 0:
            print(f"DEBUG: Checking for checkpoints in local: {local_checkpoint_root}")
            sys.stdout.flush()

        if resume_step is None and gcs_checkpoint_root:
            if jax.process_index() == 0:
                print(f"No local checkpoints. Syncing from {gcs_checkpoint_root}...")
                sys.stdout.flush()
                try:
                    sync_checkpoint_uri(gcs_checkpoint_root, local_checkpoint_root)
                    resume_step = latest_checkpoint_step(local_checkpoint_root)
                    source_root = local_checkpoint_root
                    print(f"DEBUG: After sync, latest_checkpoint_step({local_checkpoint_root}) -> {resume_step}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Warning: Failed to sync from GCS: {e}")
                    sys.stdout.flush()

        if resume_step is not None:
            try:
                load_training_checkpoint(source_root, model=model, optimizer=optimizer, step=resume_step)
                start_step = int(resume_step)
                if jax.process_index() == 0:
                    print(f"Resumed from step: {start_step}")
                    sys.stdout.flush()
            except Exception as e:
                if jax.process_index() == 0:
                    print(f"Warning: Could not resume from {source_root}: {e}")
                    print("Starting from scratch.")
                    sys.stdout.flush()
        else:
            if jax.process_index() == 0:
                    print("No checkpoint found. Starting from scratch.")
                    sys.stdout.flush()

        resumed = start_step > 0

    if args.init_checkpoint_uri and not resumed:
        init_checkpoint_root = sync_checkpoint_uri(
            args.init_checkpoint_uri,
            output_dir / "init_checkpoint",
            step=args.init_checkpoint_step,
        )
        init_step = args.init_checkpoint_step or latest_checkpoint_step(init_checkpoint_root)
        if init_step is None:
            raise FileNotFoundError(f"No checkpoint found under {args.init_checkpoint_uri}.")
        load_training_checkpoint(
            init_checkpoint_root,
            model=model,
            optimizer=optimizer if args.init_restore_optimizer else None,
            step=init_step,
        )
        if jax.process_index() == 0:
            print(f"Initialized model weights from {args.init_checkpoint_uri} step={init_step}")
            if args.init_restore_optimizer:
                print("Optimizer state was restored; this is a continuation branch with a fresh run ID.")
            else:
                print("Optimizer state was not restored; this is a fresh optimizer branch.")
            sys.stdout.flush()

    stop_requested = {"flag": False, "signal": None}
    def handle_signal(signum, _frame):
        stop_requested["flag"] = True
        stop_requested["signal"] = signal.Signals(signum).name
        print(f"signal_received={stop_requested['signal']}")
        sys.stdout.flush()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Data loading with horizon support
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
        val_chunk_dir = Path(args.val_chunk_dir)
        val_chunk_paths = discover_chunk_files(str(val_chunk_dir)) if val_chunk_dir.exists() else []
        if not val_chunk_paths:
            raise ValueError(f"No validation chunk files found in {args.val_chunk_dir}.")
    elif args.val_fraction > 0.0:
        chunk_paths, val_chunk_paths = split_train_validation_chunks(
            chunk_paths,
            val_fraction=args.val_fraction,
        )
    
    if args.chunk_dir and args.chunk_dir != "synthetic" and train_cache is None and not chunk_paths:
        raise ValueError(f"No chunk files found in {args.chunk_dir}. Aborting to prevent silent synthetic fallback.")
        
    data_source = "synthetic"
    loader = None
    val_loader_obj = None
    if chunk_paths and (args.chunk_dir != "synthetic" or train_cache is not None):
        data_source = args.gcs_train_prefix or str(chunk_dir)
        loader_obj = LeelaChunkDataLoader(
            chunk_paths,
            batch_size=args.batch_size,
            seed=args.seed,
            horizon=args.horizon,
            action_source=args.action_source,
            batch_view="dfm_action",
            chunk_paths_provider=train_cache.local_paths if train_cache is not None else None,
            shuffle_files=True if train_cache is not None else False,
            drop_last=True,
        )
        loader = iter(loader_obj)
    if val_chunk_paths:
        val_loader_obj = LeelaChunkDataLoader(
            val_chunk_paths,
            batch_size=args.batch_size,
            seed=args.val_seed,
            horizon=args.horizon,
            action_source=args.action_source,
            batch_view="dfm_action",
            shuffle_files=False,
            drop_last=False,
            chunk_paths_provider=val_cache.local_paths if val_cache is not None else None,
        )

    run = None
    run_config = config.__dict__.copy()
    run_config.update(vars(args))
    estimated_dfm_forward_flops = estimate_dfm_forward_flops(
        batch_size=args.batch_size,
        horizon=args.horizon,
        token_dim=args.token_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
    )
    estimated_dfm_train_flops = 3.0 * estimated_dfm_forward_flops
    estimated_bt4_encoder_flops = float(estimate_bt4_theory(params, batch_size=args.batch_size).encoder_forward_flops)
    estimated_total_step_flops = estimated_bt4_encoder_flops + estimated_dfm_train_flops
    run_config.update(
        {
            "estimated_bt4_encoder_forward_flops": estimated_bt4_encoder_flops,
            "estimated_dfm_forward_flops": estimated_dfm_forward_flops,
            "estimated_dfm_train_flops_per_step": estimated_dfm_train_flops,
            "estimated_total_step_flops": estimated_total_step_flops,
            "estimated_mfu_peak_tflops": args.peak_tflops,
            "estimated_mfu_note": "Approximate FLOPs. total_mfu includes frozen BT4 encoder forward + DFM forward/backward/update rule-of-thumb; dfm_mfu excludes BT4.",
            "train_chunk_count": len(chunk_paths),
            "val_chunk_count": len(val_chunk_paths),
            "gcs_train_prefix": args.gcs_train_prefix,
            "gcs_val_prefix": args.gcs_val_prefix,
        }
    )
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
    print(f"data_source={data_source}")
    print(f"models_dir={model_paths['models_dir']}")
    print(f"output_dir={output_dir}")
    print(f"checkpoint_uri={gcs_checkpoint_root}")
    print(f"horizon={args.horizon}")
    print(f"train_chunk_count={len(chunk_paths)} val_chunk_count={len(val_chunk_paths)}")
    if train_cache is not None:
        print(f"gcs_train_cache_stats={train_cache.stats()}")
    if val_cache is not None:
        print(f"gcs_val_cache_stats={val_cache.stats()}")
    sys.stdout.flush()

    metrics_log = (output_dir / "metrics.jsonl").open("a", encoding="utf-8")
    last_metrics: dict[str, float] | None = None
    completed_step = start_step

    rng = jax.random.PRNGKey(args.seed + jax.process_index())
    val_rng = jax.random.PRNGKey(args.val_seed + jax.process_index())
    val_every = args.val_every if args.val_every > 0 else args.save_every
    val_t_values = parse_val_t_values(args.val_t_values)

    try:
        for step in range(start_step, args.steps):
            rng, step_rng = jax.random.split(rng)
            if loader is not None:
                try:
                    batch = next(loader)
                except StopIteration:
                    loader = iter(loader_obj)
                    batch = next(loader)
            else:
                batch = build_synthetic_transition_batch(args.batch_size, horizon=args.horizon)
            if 0.0 <= args.train_deterministic_t <= 1.0:
                batch = dict(batch)
                batch["deterministic_t"] = np.asarray(args.train_deterministic_t, dtype=np.float32)

            step_start = time.perf_counter()
            loss, aux = train_dfm_step(model, optimizer, batch, step_rng)
            jax.block_until_ready((loss, aux))
            step_time = time.perf_counter() - step_start
            completed_step = step + 1
            metrics = {"step": completed_step, "loss": float(loss), "step_time_s": step_time}
            if train_cache is not None:
                metrics.update({f"gcs_train_cache_{key}": value for key, value in train_cache.stats().items()})
            if val_cache is not None:
                metrics.update({f"gcs_val_cache_{key}": value for key, value in val_cache.stats().items()})
            examples_per_second = args.batch_size / max(step_time, 1e-12)
            action_tokens_per_second = (args.batch_size * args.horizon) / max(step_time, 1e-12)
            estimated_dfm_tflops = estimated_dfm_train_flops / max(step_time, 1e-12) / 1e12
            estimated_total_tflops = estimated_total_step_flops / max(step_time, 1e-12) / 1e12
            metrics.update(
                {
                    "examples_per_second": examples_per_second,
                    "action_tokens_per_second": action_tokens_per_second,
                    "estimated_bt4_encoder_forward_flops": estimated_bt4_encoder_flops,
                    "estimated_dfm_train_flops_per_step": estimated_dfm_train_flops,
                    "estimated_total_step_flops": estimated_total_step_flops,
                    "estimated_dfm_tflops": estimated_dfm_tflops,
                    "estimated_total_tflops": estimated_total_tflops,
                    "estimated_dfm_mfu": (
                        estimated_dfm_tflops / args.peak_tflops
                        if args.peak_tflops and args.peak_tflops > 0
                        else 0.0
                    ),
                    "estimated_total_mfu": (
                        estimated_total_tflops / args.peak_tflops
                        if args.peak_tflops and args.peak_tflops > 0
                        else 0.0
                    ),
                    # Backward-compatible names for existing dashboards.
                    "estimated_train_flops_per_step": estimated_total_step_flops,
                    "estimated_tflops": estimated_total_tflops,
                    "estimated_mfu": (
                        estimated_total_tflops / args.peak_tflops
                        if args.peak_tflops and args.peak_tflops > 0
                        else 0.0
                    ),
                }
            )
            metrics.update(flatten_aux_metrics(aux))
            if (
                val_loader_obj is not None
                and args.val_batches > 0
                and (completed_step % val_every == 0 or completed_step == args.steps)
            ):
                val_metrics, val_rng = evaluate_validation_batches(
                    model=model,
                    loader_obj=val_loader_obj,
                    val_batches=args.val_batches,
                    rng=val_rng,
                    deterministic_t=args.val_deterministic_t,
                )
                metrics.update(add_validation_prefix(val_metrics))
                for t_value in val_t_values:
                    val_metrics_t, val_rng = evaluate_validation_batches(
                        model=model,
                        loader_obj=val_loader_obj,
                        val_batches=args.val_batches,
                        rng=val_rng,
                        deterministic_t=t_value,
                    )
                    metrics.update(add_validation_prefix(val_metrics_t, prefix=t_metric_prefix(t_value)))
            last_metrics = metrics
            metrics_log.write(json.dumps(metrics) + "\n")
            metrics_log.flush()

            if step % args.log_every == 0 or completed_step == args.steps:
                print(
                    " ".join([
                        f"step={completed_step}",
                        f"loss={metrics['loss']:.6f}",
                        f"legality_loss={metrics.get('legality_loss', 0.0):.6f}",
                        f"accuracy={metrics.get('accuracy', 0.0):.4f}",
                        f"mask_prob={metrics.get('mask_prob', 0.0):.4f}",
                        f"step_time_s={metrics['step_time_s']:.3f}",
                        f"ex_per_s={metrics['examples_per_second']:.1f}",
                        f"dfm_mfu={metrics['estimated_dfm_mfu']:.4f}",
                        f"total_mfu={metrics['estimated_total_mfu']:.4f}",
                        *(
                            [
                                f"val_loss={metrics['val_loss']:.6f}",
                                f"val_legality_loss={metrics.get('val_legality_loss', 0.0):.6f}",
                            ]
                            if "val_loss" in metrics
                            else []
                        ),
                    ])
                )
                sys.stdout.flush()

            if run is not None and jax.process_index() == 0:
                run.log(metrics, step=completed_step)

            if checkpoint_manager is not None:
                was_before = (step // args.save_every)
                is_now = (completed_step // args.save_every)
                should_save = (is_now > was_before) or (completed_step == args.steps)
                
                if should_save:
                    if jax.process_index() == 0:
                        print(f"DEBUG: Saving checkpoint at step {completed_step} to {local_checkpoint_root}")
                        sys.stdout.flush()
                    try:
                        save_training_checkpoint(
                            checkpoint_manager,
                            model=model,
                            optimizer=optimizer,
                            step=completed_step,
                            metrics=metrics,
                            metadata_path=ckpt_paths["metadata_path"],
                            config=run_config,
                            extra={"last_metrics": metrics},
                        )
                        if jax.process_index() == 0:
                            print(f"DEBUG: Successfully saved local checkpoint at step {completed_step}")
                            sys.stdout.flush()
                            if gcs_checkpoint_root:
                                print(f"DEBUG: Syncing to {gcs_checkpoint_root}...")
                                sys.stdout.flush()
                                subprocess.run(
                                    f"/snap/google-cloud-cli/current/bin/gcloud storage cp --recursive {shlex.quote(str(local_checkpoint_root))}/* {shlex.quote(gcs_checkpoint_root)}/",
                                    shell=True
                                )
                    except Exception as e:
                        if jax.process_index() == 0:
                            print(f"Warning: Failed to save checkpoint at step {completed_step}: {e}")
                            sys.stdout.flush()

            if stop_requested["flag"]:
                print(f"stopping_after_step={completed_step} stop_signal={stop_requested['signal']}")
                sys.stdout.flush()
                break

    finally:
        if train_cache is not None:
            train_cache.stop()
        if val_cache is not None:
            val_cache.stop()
        metrics_log.close()
        # Final forced local save if needed
        latest_saved_step = checkpoint_manager.latest_step()
        if completed_step > start_step and latest_saved_step != completed_step:
            save_training_checkpoint(
                checkpoint_manager,
                model=model,
                optimizer=optimizer,
                step=completed_step,
                metrics=last_metrics,
                metadata_path=ckpt_paths["metadata_path"],
                config=run_config,
                extra={"last_metrics": last_metrics, "forced": True},
                force=True,
            )
        
        # Final sync to GCS before exit
        if jax.process_index() == 0 and gcs_checkpoint_root:
            print(f"DEBUG: Final sync of checkpoints to {gcs_checkpoint_root}")
            sys.stdout.flush()
            subprocess.run(
                f"/snap/google-cloud-cli/current/bin/gcloud storage cp --recursive {shlex.quote(str(local_checkpoint_root))}/* {shlex.quote(gcs_checkpoint_root)}/",
                shell=True
            )
        
        if run is not None:
            print(f"wandb_url={run.url}")
            run.finish()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
