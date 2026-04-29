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
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files
from chess_dfm_jax.paths import default_bt4_paths, project_root
from chess_dfm_jax.training.checkpoints import (
    checkpoint_paths,
    create_checkpoint_manager,
    latest_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
    wait_for_checkpoint_completion,
)
from chess_dfm_jax.training.jepa import (
    JEPAConfig,
    build_synthetic_transition_batch,
    build_transition_batch,
    create_jepa_components,
    train_step,
)
from chess_dfm_jax.tracking import init_wandb_run, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to train.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Total batch size across all devices.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to BT4 models.")
    parser.add_argument("--chunk-dir", type=str, default=None, help="Path to Leela chunks.")
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
    return parser.parse_args()


def resolve_run_name(args: argparse.Namespace) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"jepa-{args.num_layers}l-{args.token_dim}d-{timestamp}"


def main() -> int:
    args = parse_args()
    
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
                    subprocess.run(["/snap/google-cloud-cli/current/bin/gcloud", "storage", "cp", "-r", f"{gcs_checkpoint_root}/*", str(local_checkpoint_root)], check=True)
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
                print(f"No checkpoint found. Starting from scratch.")
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
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else None
    chunk_paths = discover_chunk_files(str(chunk_dir)) if chunk_dir and chunk_dir.exists() else []
    
    if args.chunk_dir and args.chunk_dir != "synthetic" and not chunk_paths:
        raise ValueError(f"No chunk files found in {args.chunk_dir}. Aborting to prevent silent synthetic fallback.")
        
    data_source = "synthetic"
    loader = None
    if chunk_paths and args.chunk_dir != "synthetic":
        data_source = str(chunk_dir)
        loader_obj = LeelaChunkDataLoader(
            chunk_paths,
            batch_size=args.batch_size,
            seed=args.seed,
            horizon=args.horizon,
            action_source=args.action_source,
            drop_last=True,
        )
        loader = iter(loader_obj)

    run = None
    run_config = config.__dict__.copy()
    run_config.update(vars(args))
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

            step_start = time.perf_counter()
            loss, aux = train_step(model, optimizer, batch)
            step_time = time.perf_counter() - step_start
            completed_step = step + 1
            metrics = {"step": completed_step, "loss": float(loss), "step_time_s": step_time}
            metrics.update({key: float(value) for key, value in aux.items()})
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
