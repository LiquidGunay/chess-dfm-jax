import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
# Current Projected-JEPA / Joint Latent-SASA Big Run

This notebook documents the Stage 1 joint training loop used by the big
projected-JEPA / latent-SASA run:

```text
joint-h8-v5p4-b256-rawproj-officialsig1024-normloss-dclip0p5-sig0p1-tf20k-free-2full-20260503e
```

The run command lives in
`sweeps/jph8v5p4sig0p1_normloss_dclip0p5_tf20k_20260503e.jsonl`.
The architecture and losses below describe the committed implementation used by
that run. If a local uncommitted log-ratio norm-loss edit is present, ignore it
for this analysis: the big run used the squared JEPA RMS matching term described
below.

The configuration is fixed in notebook cells so it is inspectable before
anything touches TPU, GCS, W&B, model files, or checkpoints.

Safety default: `RUN_TRAINING = False`. Running all cells only validates imports
and prints the exact in-notebook configuration. Do not set `RUN_TRAINING = True`
unless you intentionally want to launch the real training script.
"""
    )
    return


@app.cell
def _():
    import contextlib
    import copy
    import importlib
    import json
    import os
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    return REPO_ROOT, contextlib, copy, importlib, json, os, sys


@app.cell
def _(mo):
    mo.md(
        r"""
## Architecture

The model is `JointLatentSASAModel`, built by
`chess_dfm_jax.training.joint_latent_sasa.create_joint_components`.

Data enters as trajectory shards in the `joint_latent_sasa` loader view:

```text
current_planes        [B, 112, 8, 8]
action_indices        [B, H]
future_planes         [B, H, 112, 8, 8]
future_valid          [B, H]
legal_idx/count       [B, H, Lmax], [B, H]
value_targets         [B, H]
wdl_targets           [B, H, 3]
valid                 [B]
```

The big H8 projected setup uses:

```text
BT4 encoder:          current/future board planes -> [B, 64, 1024] square tokens
state projector:      BT4 square tokens + CLS -> projected z [B, 1024]
DFM path:             current BT4 tokens -> 256-d square latents -> H=8 action logits
DFM hidden path:      clean true-action pass -> H=8 action hidden states
JEPA path:            recurrent transition over z, true actions, and DFM hidden states
JEPA target:          projected BT4 future z targets, not raw BT4 tokens
value/WDL heads:      present but disabled in the big run
```

BT4 embedding and encoder layers are unfrozen, but they use a smaller learning
rate than the newly trained DFM/projector/JEPA parameters. Current plus eight
future board states are encoded with `bt4_encode_chunk_size=1` to control HBM
while BT4 is trainable.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## Fixed In-Notebook Config

These values mirror the full big-run command in
`sweeps/jph8v5p4sig0p1_normloss_dclip0p5_tf20k_20260503e.jsonl`:

- `horizon = 8`, `batch_size = 256`, 4-way data parallel
- `steps = 221432`
- `z_dim = 1024`, `token_dim = 256`
- state projector: 2 layers, 8 heads, MLP 4096
- DFM: 4 layers, 4 heads, MLP 1024
- JEPA recurrent transition: 4 AdaLN/SwiGLU layers, MLP 4096
- non-BT4 LR `6e-4`, BT4 LR `1e-5`, warmup 1000, grad clip 1.0
- Muon enabled, QK norm enabled, XSA enabled
- `jepa_delta_rms_clip = 0.5`, learned JEPA state RMSNorm disabled
- official LeJEPA SIGReg on projected BT4 target latents:
  `jepa_sigreg_coeff = 0.1`, `jepa_sigreg_proj_dim = 1024`
- predicted-latent SIGReg disabled: `jepa_pred_sigreg_coeff = 0.0`
- teacher forcing for the first 20000 steps, then free rollout through step
  221432
- first legality coefficient 2.0, loss clip 20.0, value/WDL disabled
- key differences from `20260503c`: delta clip changed from 2.0 to 0.5 and
  teacher forcing changed from 110716 steps to 20000 steps
- full v3 LC0 H8 train/validation prefixes, with the requested local GCS cache
  path and W&B enabled
"""
    )
    return


@app.cell
def _():
    RUN_TRAINING = False
    STEPS = 221432

    CURRENT = {
        "backend": "tpu",
        "stage": "stage1",
        "steps": STEPS,
        "batch_size": 256,
        "seed": 0,
        "models_dir": "/tmp/chess_dfm_jax/repo/models",
        "save_dir": "/tmp/chess_dfm_jax/artifacts",
        "run_id": "joint-h8-v5p4-b256-rawproj-officialsig1024-normloss-dclip0p5-sig0p1-tf20k-free-2full-20260503e",
        "checkpoint_uri": "gs://gunay-chess-experiments-us-central1/runs/joint/joint-h8-v5p4-b256-rawproj-officialsig1024-normloss-dclip0p5-sig0p1-tf20k-free-2full-20260503e/checkpoints",
        "resume": False,
        "gcs_train_prefix": "gs://gunay-chess-experiments-us-central1/data/trajectory_v3_lc0_test80_h8_sets1_3_20260430/train",
        "gcs_val_prefix": "gs://gunay-chess-experiments-us-central1/data/trajectory_v3_lc0_test80_h8_sets1_3_20260430/val",
        "gcs_cache_dir": "/tmp/chess_dfm_jax/gcs_cache",
        "gcs_startup_cache_policy": "minimum",
        "gcs_min_train_shards": 27679,
        "gcs_min_val_shards": 1524,
        "gcs_prefetch_workers": 32,
        "gcs_prefetch_interval_s": 60,
        "gcs_max_cached_train_shards": 0,
        "loader_prefetch_batches": 4,
        "val_batches": 32,
        "val_every": 5000,
        "val_seed": 10_000,
        "val_deterministic_t": 0.0,
        "wandb_project": "chess_dfm_jax-joint",
        "wandb_group": "joint-projected-jepa-h8-v5p4-20260503e",
        "no_wandb": False,
        "token_dim": 256,
        "z_dim": 1024,
        "projector_layers": 2,
        "projector_num_heads": 8,
        "projector_mlp_dim": 4096,
        "jepa_condition_dim": 1024,
        "dfm_layers": 4,
        "jepa_layers": 4,
        "jepa_num_heads": 4,
        "jepa_mlp_dim": 4096,
        "num_heads": 4,
        "mlp_dim": 1024,
        "learning_rate": 6e-4,
        "bt4_learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "encoder_dtype": "bfloat16",
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
        "use_qk_gain": False,
        "use_qk_norm": True,
        "use_xsa": True,
        "use_muon": True,
        "grad_clip_norm": 1.0,
        "lr_warmup_steps": 1000,
        "skip_nonfinite_updates": True,
        "unfreeze_bt4_encoder": True,
        "bt4_encode_chunk_size": 1,
        "jepa_state_rmsnorm": False,
        "jepa_state_rms_scale_max": 2.0,
        "jepa_teacher_forcing_steps": 20000,
        "jepa_delta_rms_clip": 0.5,
        "remat_blocks": True,
        "scan_layers": False,
        "donate_train_state": True,
        "horizon": 8,
        "loss_horizon": 0,
        "dfm_ce_coeff": 1.0,
        "first_legality_coeff": 2.0,
        "horizon_legality_coeff": 0.0,
        "legality_on_masked_only": True,
        "jepa_positive_coeff": 1.0,
        "jepa_loss_type": "raw_mse",
        "jepa_target_mode": "projected_bt4",
        "jepa_target_sample_count": 0,
        "jepa_gamma": 1.0,
        "jepa_sigreg_coeff": 0.1,
        "jepa_pred_sigreg_coeff": 0.0,
        "jepa_sigreg_kind": "le_jepa",
        "jepa_sigreg_proj_dim": 1024,
        "value_coeff": 0.0,
        "wdl_coeff": 0.0,
        "loss_clip_value": 20.0,
        "jepa_action_contrast_coeff": 0.0,
        "jepa_action_contrast_margin": 0.05,
        "contrastive_coeff": 0.0,
        "contrastive_temperature": 0.1,
        "candidate_count": 1,
        "train_deterministic_t": -1.0,
        "legal_lmax": 128,
        "save_every": 5000,
        "log_every": 20,
        "diagnostics_every": 0,
        "max_to_keep": 5,
        "peak_tflops": 788.0,
        "data_parallel": True,
        "data_parallel_devices": 4,
        "profile_steps": 0,
        "disable_final_checkpoint": False,
    }

    return CURRENT, RUN_TRAINING, STEPS


@app.cell
def _(CURRENT, mo):
    mo.md(
        f"""
## Data Flow And Model Shapes

For global `B={CURRENT["batch_size"]}` and `H={CURRENT["horizon"]}`:

```text
current_planes  [256, 112, 8, 8]
future_planes   [256, 8, 112, 8, 8]
BT4 tokens      [256, 64, 1024] per board
all BT4 tokens  [256, 9, 64, 1024] after current + H future boards
z_all           [256, 9, 1024]
z_t             [256, 1024]
target_z        [256, 8, 1024]
z_dfm           [256, 64, 256]
DFM logits      [256, 8, 1858]
DFM hidden      [256, 8, 256]
JEPA pred_z     [256, 8, 1024]
value_pred      disabled by default
wdl_logits      disabled by default
```

`z` creation is active and trainable: BT4 outputs `[B,64,1024]`; the state
projector creates vector `z [B,z_dim]` by prepending a CLS token to the 64
projected square tokens and reading the CLS output. DFM consumes current BT4
square tokens projected to compact `token_dim` and runs two passes: a noisy
diffusion pass for action CE and a clean true-action pass to expose action hidden
states. JEPA consumes current `z`, true action embeddings, and clean DFM action
hidden states to predict future projected `z` targets.

With 4-way data parallel, each local device sees batch 64. Metrics are pmean'd
across the `data` axis before logging.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## Exact Losses

Stage 1 joint training optimizes:

```text
total =
    dfm_ce_coeff * dfm_ce
  + first_legality_coeff * first_legality_loss
  + horizon_legality_coeff * horizon_legality_loss
  + jepa_positive_coeff * (jepa_raw_mse + jepa_norm_loss)
  + jepa_sigreg_coeff * jepa_sigreg_loss
  + jepa_pred_sigreg_coeff * jepa_pred_sigreg_loss
  + value_coeff * value_loss
  + wdl_coeff * wdl_loss
  + jepa_action_contrast_coeff * jepa_action_contrast_loss
```

If `loss_clip_value > 0`, the optimized scalar is
rescaled down to `loss_clip_value` while preserving gradient direction;
`unclipped_loss` and `loss_clip_scale` are logged separately.

With the big-run defaults:

```text
total =
    1.0  * dfm_ce
  + 2.0  * first_legality_loss
  + 0.0  * horizon_legality_loss
  + 1.0  * (jepa_raw_mse + jepa_norm_loss)
  + 0.1  * jepa_sigreg_loss
  + 0.0  * jepa_pred_sigreg_loss
```

Loss meanings:

- `dfm_ce`: cross-entropy on stored true actions, but only at action positions
  masked by the diffusion/denoising corruption and inside `loss_horizon`.
- `first_legality_loss`: `1 - legal_probability_mass` for the first action
  slot, gated to masked examples by default.
- `horizon_legality_loss`: currently always zeroed in the joint hot path because
  stored later legal sets belong to the dataset's true prefix, not arbitrary
  generated prefixes.
- `jepa_raw_mse`: mean squared error between predicted future projected vectors
  and projected BT4 target vectors, weighted by `future_valid`.
- `jepa_norm_loss`: the committed big-run RMS matching term:
  `sample_norm_loss = (rms(pred_z) - stop_gradient(rms(target_z))) ** 2`.
  This is not the later aborted log-ratio norm-loss idea.
- `jepa_positive_loss`: `jepa_raw_mse + jepa_norm_loss`, averaged over valid
  horizons with `jepa_gamma = 1.0`.
- `jepa_sigreg_loss`: official LeJEPA Epps-Pulley random-projection SIGReg over
  projected BT4 current/future target latents `z_all`, with gradients enabled
  through the projector and unfrozen BT4 encoder. It is not applied to `pred_z`
  in the big run because `jepa_pred_sigreg_coeff = 0.0`.
- `value_loss` and `wdl_loss`: disabled by default because the current
  trajectory value/WDL targets are not reliable enough to use as training
  objectives. When both coefficients are zero, the hot path skips the
  value/WDL heads and logs these metrics as zero.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## Optimizer And LR Groups

`create_joint_components` builds one Optax transform with two parameter groups:

- `bt4`: `encoder/embedding` and `encoder/layers`, learning rate `1e-5`.
- `main`: state projector, DFM, JEPA transition, action embeddings/adapters,
  value/WDL head, learning rate `6e-4`.

Both groups use the same warmup shape: linear warmup from 0 to the target LR for
1000 steps, then constant LR. With `use_muon=True`, the repo's Muon+AdamW
hybrid is used for each group. With `grad_clip_norm=1.0`, global norm clipping
wraps the multi-transform. `skip_nonfinite_updates=True` skips updates whose
transformed gradients contain NaN/Inf.
"""
    )
    return


@app.cell
def _(CURRENT, REPO_ROOT, copy, importlib, json):
    from chess_dfm_jax.training.joint_latent_sasa import JointLatentSASAConfig

    train_joint_latent_sasa = importlib.import_module("scripts.train_joint_latent_sasa")

    MODEL_CONFIG = JointLatentSASAConfig(
        token_dim=CURRENT["token_dim"],
        z_dim=CURRENT["z_dim"],
        projector_layers=CURRENT["projector_layers"],
        projector_num_heads=CURRENT["projector_num_heads"],
        projector_mlp_dim=CURRENT["projector_mlp_dim"],
        jepa_condition_dim=CURRENT["jepa_condition_dim"],
        dfm_layers=CURRENT["dfm_layers"],
        jepa_layers=CURRENT["jepa_layers"],
        jepa_num_heads=CURRENT["jepa_num_heads"],
        jepa_mlp_dim=CURRENT["jepa_mlp_dim"],
        num_heads=CURRENT["num_heads"],
        mlp_dim=CURRENT["mlp_dim"],
        learning_rate=CURRENT["learning_rate"],
        bt4_learning_rate=CURRENT["bt4_learning_rate"],
        weight_decay=CURRENT["weight_decay"],
        encoder_dtype=CURRENT["encoder_dtype"],
        param_dtype=CURRENT["param_dtype"],
        compute_dtype=CURRENT["compute_dtype"],
        horizon=CURRENT["horizon"],
        loss_horizon=CURRENT["loss_horizon"],
        dfm_ce_coeff=CURRENT["dfm_ce_coeff"],
        first_legality_coeff=CURRENT["first_legality_coeff"],
        horizon_legality_coeff=CURRENT["horizon_legality_coeff"],
        legality_on_masked_only=CURRENT["legality_on_masked_only"],
        jepa_positive_coeff=CURRENT["jepa_positive_coeff"],
        jepa_loss_type=CURRENT["jepa_loss_type"],
        jepa_target_mode=CURRENT["jepa_target_mode"],
        jepa_target_sample_count=CURRENT["jepa_target_sample_count"],
        jepa_gamma=CURRENT["jepa_gamma"],
        jepa_sigreg_coeff=CURRENT["jepa_sigreg_coeff"],
        jepa_sigreg_kind=CURRENT["jepa_sigreg_kind"],
        jepa_sigreg_proj_dim=CURRENT["jepa_sigreg_proj_dim"],
        value_coeff=CURRENT["value_coeff"],
        wdl_coeff=CURRENT["wdl_coeff"],
        jepa_action_contrast_coeff=CURRENT["jepa_action_contrast_coeff"],
        jepa_action_contrast_margin=CURRENT["jepa_action_contrast_margin"],
        contrastive_coeff=CURRENT["contrastive_coeff"],
        contrastive_temperature=CURRENT["contrastive_temperature"],
        candidate_count=CURRENT["candidate_count"],
        use_qk_gain=CURRENT["use_qk_gain"],
        use_qk_norm=CURRENT["use_qk_norm"],
        use_xsa=CURRENT["use_xsa"],
        use_muon=CURRENT["use_muon"],
        grad_clip_norm=CURRENT["grad_clip_norm"],
        lr_warmup_steps=CURRENT["lr_warmup_steps"],
        skip_nonfinite_updates=CURRENT["skip_nonfinite_updates"],
        unfreeze_bt4_encoder=CURRENT["unfreeze_bt4_encoder"],
        bt4_encode_chunk_size=CURRENT["bt4_encode_chunk_size"],
        jepa_state_rmsnorm=CURRENT["jepa_state_rmsnorm"],
        jepa_state_rms_scale_max=CURRENT["jepa_state_rms_scale_max"],
        jepa_teacher_forcing_steps=CURRENT["jepa_teacher_forcing_steps"],
        jepa_delta_rms_clip=CURRENT["jepa_delta_rms_clip"],
        remat_blocks=CURRENT["remat_blocks"],
        scan_layers=CURRENT["scan_layers"],
    )

    DRY_RUN_SUMMARY = {
        "repo_root": str(REPO_ROOT),
        "training_entrypoint": str(train_joint_latent_sasa.__file__),
        "import_ok": True,
        "model_config": copy.deepcopy(MODEL_CONFIG.__dict__),
        "runtime_config": copy.deepcopy(CURRENT),
    }

    print(json.dumps(DRY_RUN_SUMMARY, indent=2, sort_keys=True))

    return DRY_RUN_SUMMARY, JointLatentSASAConfig, MODEL_CONFIG, train_joint_latent_sasa


@app.cell
def _(mo):
    mo.md(
        r"""
## Metrics To Watch

The real script writes `metrics.jsonl` under
`save_dir/run_id/metrics.jsonl` and prints compact progress lines. Important
fields are:

- `loss`, `unclipped_loss`, `loss_clip_scale`
- `dfm_ce_loss`, `dfm_ce_loss_by_horizon_h1..h8`, `accuracy`, `mask_prob`
- `first_legality_loss`, `first_legal_mass`, `horizon_legality_evaluated`
- `jepa_positive_loss`, `jepa_raw_mse`, `jepa_raw_mse_by_horizon_h1..h8`
- `jepa_norm_loss`, `jepa_norm_loss_by_horizon_h1..h8`
- `jepa_sigreg_loss`, `jepa_pred_sigreg_loss`, `jepa_teacher_forcing`
- `z_state_std`, `z_state_norm`, `z_pred_norm`, `z_target_norm`
- `value_loss`, `wdl_loss`, `value_pred_mean`, `value_target_mean`
- `examples_per_second`, `step_time_s`, `host_overhead_fraction`,
  `estimated_total_tflops`, `estimated_total_mfu`, `estimated_iteration_mfu`
- `val_*` copies of the same loss/metric fields every 5000 steps

If `diagnostics_every > 0`, the script also runs coupling-gradient and
true-vs-shuffled/identity JEPA diagnostics. This notebook keeps diagnostics off
because the big run used `diagnostics_every = 0`.
"""
    )
    return


@app.cell
def _(CURRENT, copy):
    def _flag_name(key: str) -> str:
        return "--" + key.replace("_", "-")

    def config_to_argv(config: dict) -> list[str]:
        """Convert the fixed notebook config to the existing script's argv shape."""
        args = ["train_joint_latent_sasa.py"]
        skip_keys = {
            "checkpoint_uri",
            "resume",
            "no_wandb",
            "use_qk_gain",
            "use_qk_norm",
            "use_xsa",
            "use_muon",
            "skip_nonfinite_updates",
            "unfreeze_bt4_encoder",
            "jepa_state_rmsnorm",
            "remat_blocks",
            "scan_layers",
            "donate_train_state",
            "legality_on_masked_only",
            "data_parallel",
            "disable_final_checkpoint",
        }
        for key, value in config.items():
            if key in skip_keys:
                continue
            if value is None or value == "":
                continue
            args.extend([_flag_name(key), str(value)])

        boolean_flag_map = {
            "checkpoint_uri": "--checkpoint-uri",
            "resume": "--resume",
            "no_wandb": "--no-wandb",
            "use_qk_gain": "--use-qk-gain",
            "use_qk_norm": "--use-qk-norm",
            "use_xsa": "--use-xsa",
            "use_muon": "--use-muon",
            "skip_nonfinite_updates": "--skip-nonfinite-updates",
            "unfreeze_bt4_encoder": "--unfreeze-bt4-encoder",
            "jepa_state_rmsnorm": "--jepa-state-rmsnorm",
            "remat_blocks": "--remat-blocks",
            "scan_layers": "--scan-layers",
            "donate_train_state": "--donate-train-state",
            "legality_on_masked_only": "--legality-on-masked-only",
            "data_parallel": "--data-parallel",
            "disable_final_checkpoint": "--disable-final-checkpoint",
        }
        false_boolean_flag_map = {
            "use_qk_norm": "--no-use-qk-norm",
            "use_xsa": "--no-use-xsa",
            "use_muon": "--no-use-muon",
            "skip_nonfinite_updates": "--no-skip-nonfinite-updates",
            "unfreeze_bt4_encoder": "--no-unfreeze-bt4-encoder",
            "jepa_state_rmsnorm": "--no-jepa-state-rmsnorm",
            "remat_blocks": "--no-remat-blocks",
            "scan_layers": "--no-scan-layers",
            "donate_train_state": "--no-donate-train-state",
            "legality_on_masked_only": "--no-legality-on-masked-only",
            "data_parallel": "--no-data-parallel",
        }
        for key, flag in boolean_flag_map.items():
            if key not in config:
                continue
            value = bool(config[key])
            if value:
                if key == "checkpoint_uri":
                    args.extend([flag, str(config[key])])
                else:
                    args.append(flag)
            elif key in false_boolean_flag_map:
                args.append(false_boolean_flag_map[key])

        return args

    TRAIN_ARGV = config_to_argv(copy.deepcopy(CURRENT))
    print(" ".join(TRAIN_ARGV))

    return TRAIN_ARGV, config_to_argv


@app.cell
def _(mo):
    mo.md(
        r"""
## Training Loop

The real loop is not duplicated here. The next cell calls
`scripts.train_joint_latent_sasa.main()` with a temporary `sys.argv` built from
the fixed notebook dictionary above.

This bridge exists because the current script owns:

- JAX distributed initialization for TPU backend
- GCS shard cache startup and shutdown
- `LeelaChunkDataLoader(..., batch_view="joint_latent_sasa")`
- BT4 parameter loading and `JointLatentSASAConfig` construction
- checkpoint save/resume and sidecar config/metrics files
- train/eval step dispatch through `train_joint_stage1_step_data_parallel`
  for the 4-device big run

The bridge is intentionally small and transparent: inspect `TRAIN_ARGV` above to
see exactly what the existing entrypoint receives.
"""
    )
    return


@app.cell
def _(RUN_TRAINING, TRAIN_ARGV, contextlib, sys, train_joint_latent_sasa):
    @contextlib.contextmanager
    def _temporary_argv(argv: list[str]):
        old_argv = sys.argv[:]
        sys.argv = argv[:]
        try:
            yield
        finally:
            sys.argv = old_argv

    if RUN_TRAINING:
        with _temporary_argv(TRAIN_ARGV):
            TRAINING_EXIT_CODE = train_joint_latent_sasa.main()
        print(f"training_exit_code={TRAINING_EXIT_CODE}")
    else:
        TRAINING_EXIT_CODE = None
        print("RUN_TRAINING is False; training was not started.")

    return (TRAINING_EXIT_CODE,)


@app.cell
def _(CURRENT, TRAINING_EXIT_CODE, mo):
    output_dir = f'{CURRENT["save_dir"].rstrip("/")}/{CURRENT["run_id"]}'
    mo.md(
        f"""
## Outputs

When `RUN_TRAINING` is `True`, the script writes the normal run artifacts under:

```text
{output_dir}
```

Expected files include:

- `run_config.json`
- `metrics.jsonl`
- `checkpoint_state.json`
- `checkpoints/stepXXXXXXX/state.npz`

Current training exit code: `{TRAINING_EXIT_CODE}`.
"""
    )
    return


if __name__ == "__main__":
    app.run()
