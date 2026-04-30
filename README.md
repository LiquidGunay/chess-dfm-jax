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

- BT4 stays frozen initially.
- A shared online BT4-to-planning-latent base feeds small DFM and JEPA adapters.
- DFM denoises explicit action chunks.
- JEPA predicts BT4-derived future latents for those action chunks.
- Later ranking/value heads score real chunks above corrupted legal chunks.

Standalone DFM and JEPA remain baselines. Grain is deferred until the compact
loader schema is stable; the current implementation starts with a custom
deterministic, column-selective loader.

## JEPA architecture

The trainable model keeps BT4 frozen and trains only a small transition head:

- BT4 encoder produces `64 x 1024` square tokens.
- A trainable projector maps those tokens to `64 x token_dim`.
- A learned action embedding conditions the current square tokens at each rollout step.
- A small transformer unrolls a predicted token sequence over an action chunk.
- The model predicts future projected BT4 tokens, value targets, and WDL targets for each horizon step.

`scripts/train_jepa.py` defaults:

- `token_dim=512`
- `num_layers=4`
- `num_heads=8`
- `mlp_dim=2048`
- `action_source=best`
- GPU default: encoder `float16`, head params `float32`, head compute `float32`
- TPU default: encoder `bfloat16`, head params `float32`, head compute `bfloat16`

The Spot TPU example spec uses a smaller `token_dim=256, mlp_dim=1024` configuration for cost-controlled training. Two-ply probes are analysis-only and are not part of the training loss.

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
