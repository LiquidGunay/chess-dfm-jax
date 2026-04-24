# LC0 BT4 Reproduction Plan

`lc0jax-human/` is now a standalone scaffold, not just a scratch notebook.

## Goal

Reproduce the LC0 BT4 forward pass in plain JAX, verify it against the shipped
ONNX oracle, and leave yourself with a clean starting point for JEPA / DFM style
training on top of the LC0 trunk.

## Local layout

- `chess_dfm_jax/encoding.py`: standalone LC0 plane encoder.
- `chess_dfm_jax/policy.py`: policy index mapping and legality masks.
- `chess_dfm_jax/weights.py`: `.pb.gz` loader plus BT4 weight mapping.
- `chess_dfm_jax/reference_bt4.py`: the trusted BT4 reference forward.
- `chess_dfm_jax/nnx_bt4.py`: reusable `flax.nnx` BT4 encoder and full-model forward path.
- `chess_dfm_jax/data/leela.py`: Leela self-play chunk parsing and batching.
- `chess_dfm_jax/analysis/roofline.py`: JAX timing and cost-analysis helpers.
- `notebooks/lc0_bt4_jax_repro.py`: pedagogical manual-forward notebook.
- `notebooks/leela_data_pipeline.py`: chunk loading and JEPA / DFM batch views.
- `notebooks/training_roofline.py`: roofline workflow notebook.
- `notebooks/play_bt4.py`: interactive greedy-play notebook for policy sanity checks.
- `scripts/compare_logits.py`: CLI parity harness.
- `scripts/run_roofline.py`: CLI roofline harness.

## Model files

The scaffold searches for BT4 assets in this order:

1. `LC0JAXHUMAN_MODELS_DIR`
2. `./models/`
3. `../models/`

Expected filenames:

- `BT4.onnx`
- `BT4_exported.pb.gz`
- `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`

## Recommended workflow

1. Start in `notebooks/lc0_bt4_jax_repro.py`.
2. Replace the reference-backed helper cells with your own implementations stage by stage.
3. Use `scripts/compare_logits.py` whenever you want a fast non-notebook parity check.
4. Once parity is stable, move to `notebooks/leela_data_pipeline.py` and decide what your transition-JEPA / DFM batch contract should look like.
5. After you have a real `train_step`, use `docs/roofline_analysis.md` and `scripts/run_roofline.py` to profile it.

## BT4 forward blueprint

Keep the implementation split along these stages:

1. Input preprocessing and embedding projection.
2. Input FFN and normalization.
3. Encoder stack: attention + smolgen + FFN repeated 15 times.
4. Policy head.
5. Value / WDL head.
6. Moves-left head.

That keeps the reusable trunk explicit, which is what you want later for JEPA,
DFM, probing, or alternate heads.

## Parity checklist

1. Encode a small FEN batch with `chess_dfm_jax.encoding.encode_board()`.
2. Map weights with `chess_dfm_jax.weights.map_bt4_weights()`.
3. Run your forward function.
4. Run the ONNX oracle.
5. Compare policy, WDL, and moves-left outputs.
6. If needed, add legality masking and inspect top-k agreement.

Example:

```bash
source ../.venv/bin/activate
python scripts/compare_logits.py --legal-mask
```

To compare the reusable NNX module instead of the reference forward:

```bash
python scripts/compare_logits.py --forward-fn chess_dfm_jax.nnx_bt4:bt4_forward --legal-mask
```

## Training-stack direction

The current scaffold is intentionally opinionated about boundaries:

- Chess-domain preprocessing stays in `encoding.py` and `policy.py`.
- The trusted inference path stays in `reference_bt4.py`.
- Chunk parsing and batching stay in `data/leela.py`.
- Profiling logic stays in `analysis/roofline.py`.

That way your actual trainable stack can grow cleanly around a reusable trunk
instead of around one monolithic notebook.
