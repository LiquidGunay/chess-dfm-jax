#!/usr/bin/env python3
"""Run token-level JEPA training with optional W&B logging and raw NumPy checkpoints."""

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
from chess_dfm_jax.analysis.jepa_theory import estimate_jepa_theory  # noqa: E402
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
from chess_dfm_jax.training.jepa import (  # noqa: E402
    JEPAConfig,
    build_synthetic_transition_batch,
    build_transition_batch,
    create_jepa_components,
    eval_jepa_step,
    train_step,
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
    parser.add_argument("--gcs-train-prefix", type=str, default="", help="Optional GCS prefix containing train .npz shards.")
    parser.add_argument("--gcs-val-prefix", type=str, default="", help="Optional GCS prefix containing validation .npz shards.")
    parser.add_argument("--gcs-cache-dir", type=str, default="/tmp/chess_dfm_jax/gcs_cache", help="Local cache root for GCS shards.")
    parser.add_argument("--gcs-prefetch-interval-s", type=int, default=60)
    parser.add_argument("--gcs-prefetch-workers", type=int, default=2)
    parser.add_argument("--gcs-max-cached-train-shards", type=int, default=0, help="0 means cache all visible train shards.")
    parser.add_argument("--gcs-min-train-shards", type=int, default=1)
    parser.add_argument("--gcs-min-val-shards", type=int, default=1)
    parser.add_argument(
        "--gcs-startup-cache-policy",
        choices=["all", "minimum"],
        default="all",
        help="For GCS data, 'all' blocks until every visible train shard is local.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=10_000)
    parser.add_argument("--backend", type=str, default="tpu", choices=["tpu", "cpu"], help="JAX backend.")
    parser.add_argument("--wandb-project", type=str, default="chess_dfm_jax-jepa", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity.")
    parser.add_argument("--wandb-group", type=str, default="jepa-training", help="W&B group.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--token-dim", type=int, default=512, help="JEPA token dimension.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of JEPA predictor layers.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--mlp-dim", type=int, default=2048, help="MLP hidden dimension.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--encoder-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--head-param-dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--head-compute-dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--action-source", type=str, default="best", choices=["best", "played"])
    parser.add_argument("--use-qk-gain", action="store_true", help="Use QK gain scaling.")
    parser.add_argument("--use-xsa", action="store_true", help="Use Exclusive Self-Attention.")
    parser.add_argument("--use-muon", action="store_true", help="Use Muon optimizer.")
    parser.add_argument("--terminal-only", action="store_true", help="Only supervise the final rollout state.")
    parser.add_argument("--sigreg-coeff", type=float, default=0.01)
    parser.add_argument("--value-coeff", type=float, default=0.0)
    parser.add_argument("--wdl-coeff", type=float, default=0.0)
    parser.add_argument("--save-dir", type=str, default=None, help="Local path for saves.")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--max-to-keep", type=int, default=3, help="Max checkpoints to keep.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run ID.")
    parser.add_argument("--checkpoint-uri", type=str, default=None, help="Explicit checkpoint GCS URI.")
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
    return f"jepa-{args.num_layers}l-{args.token_dim}d-{timestamp}"


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


def evaluate_validation_batches(
    *,
    model,
    loader_obj: LeelaChunkDataLoader,
    val_batches: int,
) -> dict[str, float]:
    loader = iter(loader_obj)
    totals: dict[str, float] = {}
    count = 0
    for _ in range(val_batches):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(loader_obj)
            batch = next(loader)
        batch = build_transition_batch(batch)
        loss, aux = eval_jepa_step(model, batch)
        jax.block_until_ready((loss, aux))
        batch_metrics = {"loss": float(loss)}
        batch_metrics.update(flatten_aux_metrics(aux))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


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
    save_root = Path(args.save_dir) if args.save_dir else (project_root() / "runs" / "jepa")
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
    config = JEPAConfig(
        token_dim=args.token_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        encoder_dtype=args.encoder_dtype,
        head_param_dtype=args.head_param_dtype,
        head_compute_dtype=args.head_compute_dtype,
        action_source=args.action_source,
        use_qk_gain=args.use_qk_gain,
        use_xsa=args.use_xsa,
        use_muon=args.use_muon,
        terminal_only=args.terminal_only,
        sigreg_coeff=args.sigreg_coeff,
        value_coeff=args.value_coeff,
        wdl_coeff=args.wdl_coeff,
    )
    model, optimizer = create_jepa_components(params, config, seed=args.seed)

    checkpoint_manager = create_checkpoint_manager(
        local_checkpoint_root,
        save_interval_steps=args.save_every,
        max_to_keep=args.max_to_keep,
    )

    start_step = 0
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
            batch_view="jepa_latent",
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
            batch_view="jepa_latent",
            shuffle_files=False,
            drop_last=False,
            chunk_paths_provider=val_cache.local_paths if val_cache is not None else None,
        )

    run = None
    run_config = config.__dict__.copy()
    run_config.update(vars(args))
    estimated_bt4_encoder_forward_flops = float(
        estimate_bt4_theory(params, batch_size=args.batch_size).encoder_forward_flops
    )
    estimated_jepa_head_forward_flops = float(
        estimate_jepa_theory(
            batch_size=args.batch_size,
            token_dim=args.token_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
        ).forward_flops
    )
    estimated_jepa_head_train_flops = 3.0 * estimated_jepa_head_forward_flops * max(args.horizon, 1)
    estimated_encoder_flops = estimated_bt4_encoder_forward_flops * (max(args.horizon, 1) + 1)
    estimated_total_step_flops = estimated_encoder_flops + estimated_jepa_head_train_flops
    run_config.update(
        {
            "estimated_bt4_encoder_forward_flops_per_state": estimated_bt4_encoder_forward_flops,
            "estimated_bt4_encoder_forward_flops_per_step": estimated_encoder_flops,
            "estimated_jepa_head_forward_flops_h1": estimated_jepa_head_forward_flops,
            "estimated_jepa_head_train_flops_per_step": estimated_jepa_head_train_flops,
            "estimated_total_step_flops": estimated_total_step_flops,
            "estimated_mfu_peak_tflops": args.peak_tflops,
            "estimated_mfu_note": "Approximate FLOPs. total_mfu includes frozen BT4 encoder forward for current+future states plus JEPA head forward/backward/update rule-of-thumb.",
            "train_chunk_count": len(chunk_paths),
            "val_chunk_count": len(val_chunk_paths),
            "gcs_train_prefix": args.gcs_train_prefix,
            "gcs_val_prefix": args.gcs_val_prefix,
            "batch_view": "jepa_latent",
        }
    )
    if jax.process_index() == 0 and not args.no_wandb:
        run = init_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
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

    try:
        for step in range(start_step, args.steps):
            if loader is not None:
                try:
                    batch = next(loader)
                except StopIteration:
                    loader = iter(loader_obj)
                    batch = next(loader)
            else:
                batch = build_synthetic_transition_batch(args.batch_size, horizon=args.horizon)
            batch = build_transition_batch(batch)

            step_start = time.perf_counter()
            loss, aux = train_step(model, optimizer, batch)
            step_time = time.perf_counter() - step_start
            completed_step = step + 1
            metrics = {"step": completed_step, "loss": float(loss), "step_time_s": step_time}
            examples_per_second = args.batch_size / step_time if step_time > 0 else 0.0
            metrics.update(
                {
                    "examples_per_second": examples_per_second,
                    "estimated_bt4_encoder_forward_flops_per_step": estimated_encoder_flops,
                    "estimated_jepa_head_train_flops_per_step": estimated_jepa_head_train_flops,
                    "estimated_total_step_flops": estimated_total_step_flops,
                }
            )
            if args.peak_tflops > 0 and step_time > 0:
                metrics.update(
                    {
                        "estimated_jepa_head_tflops": estimated_jepa_head_train_flops / step_time / 1e12,
                        "estimated_total_tflops": estimated_total_step_flops / step_time / 1e12,
                        "estimated_jepa_head_mfu": (estimated_jepa_head_train_flops / step_time / 1e12)
                        / args.peak_tflops,
                        "estimated_total_mfu": (estimated_total_step_flops / step_time / 1e12)
                        / args.peak_tflops,
                    }
                )
            metrics.update({key: float(value) for key, value in aux.items()})
            if train_cache is not None:
                metrics.update({f"gcs_train_cache_{key}": value for key, value in train_cache.stats().items()})
            if val_cache is not None:
                metrics.update({f"gcs_val_cache_{key}": value for key, value in val_cache.stats().items()})
            val_every = args.val_every or args.save_every
            if val_loader_obj is not None and args.val_batches > 0 and (
                completed_step % val_every == 0 or completed_step == args.steps
            ):
                val_metrics = evaluate_validation_batches(
                    model=model,
                    loader_obj=val_loader_obj,
                    val_batches=args.val_batches,
                )
                metrics.update(add_validation_prefix(val_metrics))
            last_metrics = metrics
            metrics_log.write(json.dumps(metrics) + "\n")
            metrics_log.flush()

            if step % args.log_every == 0 or completed_step == args.steps:
                print(
                    " ".join([
                        f"step={completed_step}",
                        f"loss={metrics['loss']:.6f}",
                        f"jepa_loss={metrics['jepa_loss']:.6f}",
                        f"valid={metrics['valid_fraction']:.3f}",
                        f"token_cos={metrics['mean_token_cosine']:.4f}",
                        f"step_time_s={metrics['step_time_s']:.3f}",
                        f"ex_per_s={metrics['examples_per_second']:.1f}",
                        f"total_mfu={metrics.get('estimated_total_mfu', 0.0):.4f}",
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
