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

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.data.gcs_cache import GCSShardCache  # noqa: E402
from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files  # noqa: E402
from chess_dfm_jax.data.trajectory import build_synthetic_trajectory_shard, trajectory_joint_batch_from_npz  # noqa: E402
from chess_dfm_jax.paths import default_bt4_paths, project_root  # noqa: E402
from chess_dfm_jax.training.checkpoints import (  # noqa: E402
    create_checkpoint_manager,
    latest_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
)
from chess_dfm_jax.training.joint_latent_sasa import (  # noqa: E402
    JointLatentSASAConfig,
    create_joint_components,
    eval_joint_stage1_step,
    eval_joint_stage2_step,
    joint_coupling_gradient_diagnostics,
    joint_jepa_action_baseline_diagnostics,
    train_joint_stage1_step,
    train_joint_stage2_step,
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
    parser.add_argument("--dfm-layers", type=int, default=4)
    parser.add_argument("--jepa-layers", type=int, default=2)
    parser.add_argument("--jepa-num-heads", type=int, default=0, help="JEPA heads for raw BT4 tokens; defaults to --num-heads.")
    parser.add_argument("--jepa-mlp-dim", type=int, default=0, help="JEPA MLP width for raw BT4 tokens; defaults to 4x BT4 width.")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--param-dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--compute-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use-qk-gain", action="store_true")
    parser.add_argument("--use-muon", action="store_true")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--loss-horizon", type=int, default=0)
    parser.add_argument("--dfm-ce-coeff", type=float, default=1.0)
    parser.add_argument("--legality-coeff", type=float, default=None, help="Deprecated alias for --first-legality-coeff.")
    parser.add_argument("--first-legality-coeff", type=float, default=None)
    parser.add_argument("--horizon-legality-coeff", type=float, default=0.0)
    parser.add_argument("--legality-on-masked-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jepa-positive-coeff", type=float, default=1.0)
    parser.add_argument("--jepa-gamma", type=float, default=0.9)
    parser.add_argument("--jepa-sigreg-coeff", type=float, default=0.0)
    parser.add_argument("--jepa-action-contrast-coeff", type=float, default=0.0)
    parser.add_argument("--jepa-action-contrast-margin", type=float, default=0.05)
    parser.add_argument(
        "--target-projector-mode",
        type=str,
        default=None,
        choices=["shared", "separate"],
        help="Deprecated no-op. Joint JEPA now targets raw frozen BT4 tokens.",
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
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--checkpoint-uri", type=str, default=None)
    parser.add_argument("--peak-tflops", type=float, default=197.0)
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


def build_synthetic_joint_batch(batch_size: int, horizon: int, legal_lmax: int) -> dict[str, np.ndarray]:
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
    return trajectory_joint_batch_from_npz(payload, horizon=horizon, legal_lmax=legal_lmax)


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
) -> dict[str, float]:
    seq_len = 64 + horizon
    state_len = 64
    dfm_layer_forward = batch_size * dfm_layers * (
        4.0 * seq_len * token_dim * token_dim
        + 2.0 * seq_len * token_dim * mlp_dim
        + 2.0 * seq_len * seq_len * token_dim
    )
    jepa_layer_forward = batch_size * horizon * jepa_layers * (
        4.0 * state_len * jepa_width * jepa_width
        + 2.0 * state_len * jepa_width * jepa_mlp_dim
        + 2.0 * state_len * state_len * jepa_width
    )
    dfm_train = 3.0 * dfm_layer_forward * 2.0
    jepa_train = 3.0 * jepa_layer_forward
    return {
        "estimated_dfm_train_flops_per_step": float(dfm_train),
        "estimated_jepa_train_flops_per_step": float(jepa_train),
        "estimated_total_step_flops": float(dfm_train + jepa_train),
        "estimated_jepa_width": float(jepa_width),
        "estimated_jepa_mlp_dim": float(jepa_mlp_dim),
    }


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
) -> tuple[dict[str, float], jax.Array]:
    loader = iter(loader_obj) if loader_obj is not None else None
    totals: dict[str, float] = {}
    count = 0
    for _ in range(val_batches):
        if loader is None:
            batch = build_synthetic_joint_batch(batch_size, horizon, legal_lmax)
        else:
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(loader_obj)
                batch = next(loader)
        if 0.0 <= deterministic_t <= 1.0:
            batch = dict(batch)
            batch["deterministic_t"] = np.asarray(deterministic_t, dtype=np.float32)
        rng, eval_rng = jax.random.split(rng)
        if stage == "stage2":
            loss, aux = eval_joint_stage2_step(model, batch, eval_rng)
        else:
            loss, aux = eval_joint_stage1_step(model, batch, eval_rng)
        jax.block_until_ready((loss, aux))
        batch_metrics = {"loss": float(loss)}
        batch_metrics.update(flatten_aux_metrics(aux))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        return {}, rng
    return {key: value / count for key, value in totals.items()}, rng


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
        dfm_layers=args.dfm_layers,
        jepa_layers=args.jepa_layers,
        jepa_num_heads=args.jepa_num_heads,
        jepa_mlp_dim=args.jepa_mlp_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        learning_rate=args.learning_rate,
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
        jepa_gamma=args.jepa_gamma,
        jepa_sigreg_coeff=args.jepa_sigreg_coeff,
        jepa_action_contrast_coeff=args.jepa_action_contrast_coeff,
        jepa_action_contrast_margin=args.jepa_action_contrast_margin,
        contrastive_coeff=args.contrastive_coeff,
        contrastive_temperature=args.contrastive_temperature,
        candidate_count=args.candidate_count,
        use_qk_gain=args.use_qk_gain,
        use_muon=args.use_muon,
    )
    model, optimizer = create_joint_components(params, config, seed=args.seed)
    checkpoint_manager = create_checkpoint_manager(
        local_checkpoint_root,
        save_interval_steps=args.save_every,
        max_to_keep=args.max_to_keep,
    )

    start_step = 0
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
        )
        loader = iter(loader_obj)
    if val_chunk_paths:
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
        )

    run_config = config.__dict__.copy()
    run_config.update(vars(args))
    run_config["first_legality_coeff"] = first_legality_coeff
    run_config["jepa_target_space"] = "raw_bt4_tokens"
    run_config.update(
        {
            "model_family": "joint_latent_sasa",
            "objective_stage": args.stage,
            "batch_view": "joint_latent_sasa",
            "train_chunk_count": len(chunk_paths),
            "val_chunk_count": len(val_chunk_paths),
            "gcs_train_prefix": args.gcs_train_prefix,
            "gcs_val_prefix": args.gcs_val_prefix,
        }
    )
    flops = estimate_joint_step_flops(
        batch_size=args.batch_size,
        horizon=args.horizon,
        token_dim=args.token_dim,
        dfm_layers=args.dfm_layers,
        jepa_layers=args.jepa_layers,
        jepa_width=int(params["embedding_size"]),
        jepa_mlp_dim=args.jepa_mlp_dim if args.jepa_mlp_dim > 0 else int(params["embedding_size"]) * 4,
        mlp_dim=args.mlp_dim,
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
                batch = build_synthetic_joint_batch(args.batch_size, args.horizon, args.legal_lmax)
            if 0.0 <= args.train_deterministic_t <= 1.0:
                batch = dict(batch)
                batch["deterministic_t"] = np.asarray(args.train_deterministic_t, dtype=np.float32)

            step_start = time.perf_counter()
            if args.stage == "stage2":
                loss, aux = train_joint_stage2_step(model, optimizer, batch, step_rng)
            else:
                loss, aux = train_joint_stage1_step(model, optimizer, batch, step_rng)
            jax.block_until_ready((loss, aux))
            step_time = time.perf_counter() - step_start
            completed_step = step + 1

            metrics = {"step": completed_step, "loss": float(loss), "step_time_s": step_time}
            metrics.update(flatten_aux_metrics(aux))
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

            if args.diagnostics_every > 0 and (
                completed_step % args.diagnostics_every == 0 or completed_step == args.steps
            ):
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

            if args.val_batches > 0 and (completed_step % val_every == 0 or completed_step == args.steps):
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
                )
                metrics.update(add_validation_prefix(val_metrics))

            last_metrics = metrics
            metrics_log.write(json.dumps(metrics) + "\n")
            metrics_log.flush()

            if step % args.log_every == 0 or completed_step == args.steps:
                print(
                    " ".join(
                        [
                            f"step={completed_step}",
                            f"loss={metrics['loss']:.6f}",
                            f"dfm_ce={metrics.get('dfm_ce_loss', 0.0):.6f}",
                            f"legal={metrics.get('legality_loss', 0.0):.6f}",
                            f"first_legal={metrics.get('first_legality_loss', 0.0):.6f}",
                            f"horizon_legal={metrics.get('horizon_legality_loss', 0.0):.6f}",
                            f"jepa={metrics.get('jepa_positive_loss', 0.0):.6f}",
                            f"sigreg={metrics.get('jepa_sigreg_loss', 0.0):.6f}",
                            f"act_contrast={metrics.get('jepa_action_contrast_loss', 0.0):.6f}",
                            f"raw_mse={metrics.get('jepa_raw_mse', 0.0):.6f}",
                            f"cos={metrics.get('mean_token_cosine', 0.0):.4f}",
                            f"acc={metrics.get('accuracy', 0.0):.4f}",
                            f"step_time_s={metrics['step_time_s']:.3f}",
                            f"ex_per_s={metrics['examples_per_second']:.1f}",
                            f"total_mfu={metrics['estimated_total_mfu']:.4f}",
                        ]
                    )
                )
                sys.stdout.flush()

            if run is not None and jax.process_index() == 0:
                run.log(metrics, step=completed_step)

            should_save = (completed_step % args.save_every == 0) or (completed_step == args.steps)
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
        if train_cache is not None:
            train_cache.stop()
        if val_cache is not None:
            val_cache.stop()
        metrics_log.close()
        latest_saved_step = checkpoint_manager.latest_step()
        if completed_step > start_step and latest_saved_step != completed_step:
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
