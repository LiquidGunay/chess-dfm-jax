# chess-dfm-jax

Standalone BT4 training scaffold extracted from the larger `schutpaper` repo.

This folder is meant to be copied to a GPU or TPU VM and used as a clean starting
point for:

- manual BT4 reproduction in JAX
- encoder/backbone experimentation
- token-level JEPA training on LC0 self-play data
- parity checks against the shipped ONNX oracle
- roofline and step profiling
- preemption-aware Spot TPU batch training with raw NumPy checkpoints
- trajectory-v2 DFM training on exact multi-ply chess rollouts
- queued TPU experiment execution for multiple ablations per provisioned worker
- Latent-SASA joint DFM+JEPA pretraining
- dataset QA/deduplication and run monitoring through marimo notebooks

## Local GPU autoresearch status

The clean A10G research path lives on `research/local-gpu-autoresearch`; the
historical TPU implementation is preserved on
`legacy/tpu-joint-latent-sasa`. The detailed contract and measured evidence are
in `docs/local_gpu_autoresearch_plan.md` and
`docs/local_gpu_baseline.md`.

Unattended autoresearch is not enabled: `AUTORESEARCH_READY = False`,
`research/results.tsv` remains header-only, and no continuation checkpoint has
received an Elo evaluation or promotion. The current provisional efficiency
candidate imports the step-265,000 model only, starts a fresh zero-warmup
optimizer at main/BT4 rates `3e-5`/`1e-6`, and trains at batch 128 while
holding validation at 64 × 64 examples. Its small, noisy CE gain comes with
legal-mass regression and a narrow target-scale gate failure, so it is a
baseline candidate rather than a frozen result.

## Layout

- `chess_dfm_jax/`: local package with encoder, policy map, weights loader, BT4 reference forward, NNX BT4 encoder/model, chunk loader, PGN sequence loader, roofline helpers, raw NumPy checkpoints, TPU controller helpers, and W&B utilities.
- `notebooks/`: pedagogical marimo notebooks.
- `scripts/`: runnable CLIs for parity checks, roofline measurement, and chunk inspection.
- `docs/`: workflow notes for manual reproduction, roofline analysis, and data loading.

## Quickstart

```bash
cd chess-dfm-jax
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

If you want to reuse the parent repo environment on this machine:

```bash
source ../.venv/bin/activate
```

Optional experiment tracking:

```bash
echo "WANDB_API_KEY=..." > .env
```

## Model paths

The scaffold looks for BT4 files in this order:

1. `LC0JAXHUMAN_MODELS_DIR`
2. `./models/`
3. `../models/`

Expected filenames:

- `BT4.onnx`
- `BT4_exported.pb.gz`
- `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`

## Main notebooks

- `notebooks/lc0_bt4_jax_repro.py`: manual forward pass scaffold with parity checks.
- `notebooks/leela_data_pipeline.py`: widget-driven LC0 data browser for chunk samples, plane inspection, board views, and policy-head summaries.
- `notebooks/state_action_training_browser.py`: trajectory-v2 shard browser with exact rollout checks and local/GCS/W&B training-status inspection.
- `notebooks/trajectory_dedup_browser.py`: duplicate-statistics browser for trajectory-v2 datasets.
- `notebooks/training_roofline.py`: timing, cost analysis, and roofline workflow for reference forward plus NNX encoder forward/backward.
- `notebooks/training_jepa.py`: frozen-BT4, token-level JEPA scaffold with one token per square and an action-conditioned transformer.
- `notebooks/analyze_jepa.py`: inspect saved JEPA runs with training curves, per-square cosine heatmaps, and a post hoc two-ply probe on held-out PGN sequences.
- `notebooks/play_bt4.py`: play against the greedy BT4 policy head on GPU.

## Main scripts

- `python scripts/compare_logits.py`
- `python scripts/compare_logits.py --forward-fn chess_dfm_jax.nnx_bt4:bt4_forward_fp16`
- `python scripts/inspect_leela_chunks.py --chunk-dir /path/to/chunks`
- `python scripts/run_roofline.py --batch-size 8`
- `python scripts/run_roofline.py --target nnx_encoder_forward --compute-dtype float16 --batch-size 8`
- `python scripts/run_roofline.py --target nnx_encoder_backward --compute-dtype float16 --batch-size 8`
- `python scripts/run_roofline.py --target jepa_train --compute-dtype float16 --batch-size 8`
- `python scripts/profile_jepa_tpu.py --out-dir artifacts/profile_bundle`
- `python scripts/train_jepa.py --steps 10 --chunk-dir /path/to/chunks`
- `python scripts/train_jepa.py --steps 50000 --run-name local-jepa --resume --checkpoint-uri runs/jepa/local-jepa/checkpoints`
- `python scripts/run_tpu_spot_jepa.py --job-spec docs/tpu_spot_job_spec.example.json`
- `python scripts/run_experiment_queue.py --queue experiments.jsonl --dry-run`
- `uvx marimo check notebooks/state_action_training_browser.py`
- `uv run notebooks/state_action_training_browser.py`
- `uvx marimo check notebooks/trajectory_dedup_browser.py`
- `python scripts/trajectory_dedup.py stats --input-prefix lc0=/path/or/gs-prefix --output-json dedup_stats.json`

## Current DFM Status

As of 2026-04-29, the main training path is the DFM action denoiser on
trajectory-v2 shards:

- LC0-only H8 data is ready and validated at
  `data/trajectory_v2_lc0_test80_h8_1m`: 920 train shards, 55 validation
  shards, 49 test shards, and 1,048,576 total samples.
- TCEC S20-S28 H8 data is ready at
  `data/trajectory_v2_tcec_s20_s28_standard_h8_4m`: 2,283,113 total samples.
- Combined TCEC+LC0 exact deduplication is supported by
  `scripts/trajectory_dedup.py`. Exact `position + full action sequence`
  duplicates are removed while repeated positions with different continuations
  are kept for policy diversity.
- `scripts/train_dfm.py` defaults to `--gcs-startup-cache-policy all` for GCS
  datasets. Training blocks until every visible train shard is cached locally,
  preventing the earlier growing-cache replay artifact.

The clean LC0-only Phase A baseline uses:

- model: DFM `token_dim=640`, `num_layers=8`, `num_heads=10`, `mlp_dim=2560`
- optimizer: Muon where configured by the trainer, learning rate `6e-4`
- data: LC0 H8 train/val trajectory-v2 shards
- objective: first-ply curriculum with all actions masked at `t=0`
- loss: `CE(first_move) + 7.64 * illegal_prob_mass(first_move)` by default,
  with the legality coefficient recomputed from dataset average legal-move
  count for major new datasets

Use `docs/training_phases.md` for the exact phase plan and launch guardrails.

## Latent-SASA direction

The next research target is Latent-SASA joint pretraining:

- BT4 embedding/encoder are unfrozen in the active joint path with a small
  `1e-5` learning rate.
- A trainable BT4-token-to-state-vector projector maps `64 x 1024` BT4 tokens
  into one `z_dim` JEPA state vector.
- DFM denoises explicit action chunks.
- JEPA predicts projected BT4 future state vectors for those action chunks.
- JEPA consumes DFM action-token hidden states, not only action IDs, so
  future-latent losses shape the DFM planner representation.
- Value/WDL heads score predicted projected futures. Candidate ranking is
  deferred until projected JEPA dynamics are stable.

Standalone DFM and JEPA remain baselines. The detailed design lives in
`docs/implementation_plan_gold.md`; the repo-grounded implementation checklist
is `docs/latent_sasa_coupling.md`. The first joint APIs are in place:
`planner_from_latents(..., return_hidden=True)`,
`jepa_rollout_from_latents(..., action_hidden=...)`, a `joint_latent_sasa`
loader view, legal-prefix candidates, a Stage 1 joint loss primitive, and a
queue-compatible Stage 1 trainer with checkpoint save/resume. The current
active design is documented in `docs/projected_jepa_unfreeze_plan.md`;
contrastive/ranking is no longer in the active training loss. Grain is deferred until the compact loader schema is stable; the current
implementation starts with a custom deterministic, column-selective loader.

## JEPA architecture

The active joint model trains BT4 encoder layers, a state projector, DFM, and
JEPA together:

- BT4 encoder produces `64 x 1024` square tokens.
- The state projector maps those tokens to `z_t [B, z_dim]`.
- DFM consumes projected square tokens and returns action logits plus
  `action_hidden [B, H, token_dim]`.
- JEPA consumes true action embeddings plus DFM action hidden states and predicts
  future projected vectors `z_{t+1:t+H}`.
- The model predicts value and WDL from predicted projected future vectors.
- `token_dim` remains the compact DFM/planning width; `z_dim` is the JEPA state
  vector width.

`scripts/train_jepa.py` defaults:

- raw JEPA token width is BT4 width `1024`
- `token_dim=512` remains a compatibility/hidden-size knob for the action MLP
- `num_layers=4`
- `num_heads=8`
- `mlp_dim=2048`
- `action_source=best`
- GPU default: encoder `float16`, head params `float32`, head compute `float32`
- TPU default: encoder `bfloat16`, head params `float32`, head compute `bfloat16`

The Spot TPU example spec uses a smaller `token_dim=256, mlp_dim=1024` configuration for cost-controlled training. Two-ply probes are analysis-only and are not part of the training loss.

The JEPA trainer uses the `jepa_latent` loader view for trajectory shards. That
view keeps `planes_t`, `actions`, `planes_future`, `future_valid`, and optional
value/WDL targets, but skips dense legal masks because latent transition
training does not consume them.

## Checkpoints and resume

`scripts/train_jepa.py` and `scripts/train_dfm.py` use raw NumPy checkpoint
directories, not Orbax. Each checkpoint stores trainable model state and, for
normal resume, optimizer state.

- Local runs default to `runs/jepa/<run-name>/checkpoints/` or
  `runs/dfm/<run-name>/checkpoints/`.
- Resume uses `--resume` and restores the latest step from the checkpoint directory.
- Curriculum branches can use `--init-checkpoint-uri` to initialize model
  weights from a prior run while writing checkpoints under a new run ID; GCS
  initialization copies only the selected checkpoint step.
- GCS resume in both DFM and JEPA syncs only the selected checkpoint step, not
  the whole checkpoint tree.
- The trainer handles `SIGTERM` and writes a final checkpoint before exiting.
- Only trainable head/projector state and optimizer state are checkpointed;
  frozen BT4 weights are reloaded from the pinned model file.

Do not run two trainers against the same checkpoint directory at the same time.
The next checkpointing upgrade is asynchronous GCS upload of completed local
checkpoints; local checkpoint writes remain synchronous and raw NumPy.

## Spot TPU flow

The first cloud path is single-host TPU VMs.

- Use `docs/tpu_spot_job_spec.example.json` as the controller spec template.
- The local controller uploads an immutable source snapshot to GCS.
- The TPU VM startup script installs `jax[tpu]`, installs the repo, downloads models/data, and runs `scripts/train_jepa.py --resume`.
- For sweeps, run `scripts/run_experiment_queue.py` as the TPU entry command so
  several experiments execute on one provisioned VM.
- Raw NumPy checkpoints go to a regional GCS path so a later zone retry can
  resume safely.

See `docs/tpu_spot_training.md` for the setup details.

## Docs

- `VISION.md`: long-range research direction for latent chess planning.
- `ROADMAP.md`: current implementation priorities and acceptance checks.
- `docs/data_loading.md`: trajectory-v2 shard contract and loader behavior.

## Suggested workflow

1. Finish the TODO cells in `notebooks/lc0_bt4_jax_repro.py` until manual outputs match the reference and ONNX.
2. Use `scripts/compare_logits.py --forward-fn chess_dfm_jax.nnx_bt4:bt4_forward_fp16` for a GPU-oriented low-precision parity check, or `bt4_forward` for the fp32 baseline.
3. Use `scripts/run_roofline.py --target nnx_encoder_forward` and `--target nnx_encoder_backward` with `--compute-dtype float16` or `float32` to measure encoder arithmetic intensity before building a full training loop.
4. Prototype chunk batching in `notebooks/leela_data_pipeline.py`.
5. Start with `notebooks/training_jepa.py` for a token-level JEPA smoke test on GPU.
6. Use `notebooks/play_bt4.py` when you want a quick human-vs-policy sanity check without search.
7. Use `scripts/train_jepa.py` for tracked local runs with W&B and raw NumPy checkpoints.
8. Use `notebooks/analyze_jepa.py` to inspect `metrics.jsonl`, per-square cosine heatmaps, and the post hoc two-ply probe.
9. Use `scripts/profile_jepa_tpu.py` when you want a TPU-oriented trace plus a small arithmetic-intensity sweep in one artifact bundle.
10. Use `scripts/run_tpu_spot_jepa.py` with a filled job spec when you are ready to move the same training path to Spot TPU VMs.
11. Keep smoke, DFM, JEPA, profile, and combined runs separated as described in `docs/training_phases.md`.
12. Plug the JEPA `train_step` into `scripts/run_roofline.py` as described in `docs/roofline_analysis.md`.
