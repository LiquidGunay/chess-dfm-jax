# Roofline Workflow

Use the roofline tools only after the step you are profiling is numerically sane.
Do parity first, then profile.

## What the scaffold gives you

- `chess_dfm_jax.analysis.roofline.measure_roofline_point()`: JIT-compile a step, read JAX cost analysis, and time it.
- `scripts/run_roofline.py`: CLI wrapper around that helper.
- `notebooks/training_roofline.py`: interactive walkthrough.

## Fast path: reference forward

```bash
source ../.venv/bin/activate
python scripts/run_roofline.py --batch-size 8 --repeat 20
```

This measures the reference BT4 forward pass and prints:

- seconds per step
- steps per second
- FLOPs per step, if JAX cost analysis provides them
- bytes accessed per step, if JAX cost analysis provides them
- arithmetic intensity
- achieved TFLOPs and GB/s

## NNX encoder forward and backward

The scaffold also exposes `flax.nnx` profiling targets for the reusable BT4
encoder:

```bash
python scripts/run_roofline.py --target nnx_encoder_forward --compute-dtype float16 --batch-size 8 --repeat 20
python scripts/run_roofline.py --target nnx_encoder_backward --compute-dtype float16 --batch-size 8 --repeat 20
```

Notes:

- `nnx_encoder_forward` profiles `BT4Model.encode_tokens()`
- `nnx_encoder_backward` profiles forward + backward on a simple surrogate loss
  `mean(square(tokens))`
- `--compute-dtype` lets you switch between `float16`, `bfloat16`, and `float32`
- the backward point is meant to estimate encoder training cost before a real
  JEPA objective exists

Once the first JEPA scaffold is in place, you can also profile the actual train
step:

```bash
python scripts/run_roofline.py --target jepa_train --compute-dtype float16 --batch-size 8 --repeat 20
```

The JEPA target also accepts the token-head dimensions:

```bash
python scripts/run_roofline.py \
  --target jepa_train \
  --compute-dtype bfloat16 \
  --batch-size 128 \
  --token-dim 512 \
  --num-layers 4 \
  --num-heads 8 \
  --mlp-dim 2048 \
  --repeat 10
```

To sweep several TPU-friendly shapes and write a CSV:

```bash
python scripts/profile_jepa_sweep.py \
  --batch-sizes 64,128,256 \
  --token-dims 256,384,512 \
  --mlp-dims 1024,1536,2048 \
  --compute-dtype bfloat16 \
  --head-compute-dtype bfloat16 \
  --out-csv artifacts/jepa_sweep.csv \
  --out-json artifacts/jepa_sweep.json
```

## Theory-side checks

`notebooks/training_roofline.py` now also reports:

- total BT4 parameter count
- parameter bytes for fp16 and fp32
- a matmul-dominant encoder forward FLOP estimate
- a full forward FLOP estimate
- a rule-of-thumb encoder backward FLOP estimate (`3x` encoder forward)

Treat those theoretical FLOPs as architecture-level estimates. The JAX cost
analysis remains the main runtime-facing number.

## Add your own training step later

The CLI supports a factory target:

```bash
python scripts/run_roofline.py --target factory --factory mypkg.profile_target:make_target
```

Your factory should return a dict with:

```python
{
    "name": "train_step_bs256",
    "step_fn": train_step,
    "args": (state, batch),
    "kwargs": {},
}
```

The only hard requirement is that `step_fn(*args, **kwargs)` returns JAX arrays
or pytrees of JAX arrays so `block_until_ready()` works.

## Peak hardware numbers

Pass hardware peaks explicitly when you know them:

```bash
python scripts/run_roofline.py \
  --batch-size 8 \
  --peak-tflops 989 \
  --peak-gbps 3350
```

The script then reports the roofline bound and a rough efficiency ratio.

Keep the numbers hardware-specific. Do not hard-code GPU or TPU peaks into the
project.

## Trace collection

To collect a JAX trace for TensorBoard:

```bash
python scripts/run_roofline.py --trace-dir artifacts/trace --repeat 20
```

Then inspect the trace with your usual TensorBoard or profiling workflow.

On TPU, the same trace directory can be viewed with TensorBoard's profile
plugin or with Perfetto after exporting the trace from the TensorBoard UI.

For a TPU-oriented bundle that writes device metadata, a representative trace
point, and a small JEPA sweep into one directory, use:

```bash
python scripts/profile_jepa_tpu.py --out-dir artifacts/profile_bundle
```

## Practical guidance

- Use the same batch contract for profiling that you intend to train with.
- Warm up first so compilation does not pollute timing.
- Profile both a small and a large batch size; roofline behavior changes with occupancy.
- Treat cost-analysis FLOPs as estimates, not ground truth.
- Re-run profiling after major architectural changes, not after every tiny notebook edit.
