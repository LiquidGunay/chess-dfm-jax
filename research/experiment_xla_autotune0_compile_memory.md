# Experiment 025: bounded XLA autotune-off compilation

Status: preregistered on 2026-07-23. This is a systems experiment, not a
model, objective, data, batch-size, or inference experiment.

## Question and evidence

Experiments 021--024 changed four different small parts of the training graph,
yet every new batch-128 graph was stopped safely during cold XLA compilation
at `10.944--11.095 GB` process-group RSS. The fixed guard ceiling is
`7,516,192,768` bytes. None reached an optimizer update or wrote model state.
Previously cached incumbent graphs still execute under the guard, so cold
compilation is the immediate bottleneck.

The installed stack is JAX/JAXlib `0.10.1`. A guarded device-initialization
probe confirms that its CUDA plugin accepts
`xla_gpu_force_compilation_parallelism` and
`xla_gpu_enable_llvm_module_compilation_parallelism`. However, the effective
bare `CompileOptions` value of the latter is already `false`, and setting it
to `true` is reflected by the same API. The plugin describes the false value
as single-threaded LLVM-module compilation. Therefore explicitly setting
`false`, with or without a forced thread count of one, is a known no-op and
must not consume a cold-compile trial.

The installed XLA help reports stable GPU autotuning level `4`, where level
`0` disables GEMM and convolution autotuning. The hypothesis is that compiling
fewer candidate kernels will reduce peak host compiler memory enough to fit
the existing guard. This can change selected kernels and runtime speed, so it
may be adopted only after numerical and throughput checks.

Upstream references:

- <https://docs.jax.dev/en/latest/xla_flags.html>
- <https://openxla.org/xla/flags_guidance>

## Immutable candidate

Run exactly one cold-cache candidate with:

```text
XLA_FLAGS=--xla_gpu_autotune_level=0
JAX_COMPILATION_CACHE_DIR=.local/cache/jax/experiment025-autotune0-v1
```

Use a previously nonexistent cache subdirectory so a successful result cannot
be a hit from the incumbent cache. Keep the active accepted one-block-tail
configuration and all of the following fixed:

- source step-265,000 model-only initialization and fresh optimizer;
- balanced per-example K=1 future-target sampling;
- stopped future-target prefix with only the final BT4 block trainable;
- both projector blocks, all four DFM blocks, and recurrent JEPA;
- uniform eight-horizon CE and first-legality coefficient `2.0`;
- RMS norm coefficient `0.0`, target SIGReg `5.76`, and `z_pred` SIGReg
  `1.0`;
- 64-example V-statistic SIGReg samples;
- train/eval batch sizes `128/64`;
- main/BT4 peak learning rates `3e-5/1e-6`;
- the `400/800/0.1` cosine schedule, seed 0, validation seed 10,000, and
  global train permutation; and
- no feedback, teacher, WDL, root conditional CE, checkpoint write, or model
  state change.

The exact smoke performs one update with explicit compile-ahead and has a
900-second wall timeout. It runs only through `research/run_gpu.sh`, with the
unchanged two-CPU, 8 GiB launch, 3 GiB runtime, 7 GiB group-RSS, 30 GiB disk,
exclusive-lock, and zero-checkpoint contracts.

## Decision gates

The candidate passes compilation viability only if it:

1. starts from the fresh dedicated cache;
2. completes initial validation, explicit training compilation, and one
   optimizer update within 900 seconds;
3. exits normally with finite, unclipped metrics and the exact immutable
   model/loss/data contract;
4. writes zero checkpoints and no `state.npz`;
5. peaks at no more than `7,247,757,312` bytes group RSS, leaving 256 MiB of
   headroom below the immutable guard; and
6. never crosses the existing host-memory or disk floors.

The four default-level cold-compile failures are the preregistered control;
do not spend another unsafe default-level cold compile merely to reproduce
them on the exact incumbent graph.

If compilation viability passes:

1. compare initial validation and the one-update report with the retained
   accepted one-block-tail smoke under the same seed and data. Require
   identical discrete/configuration fields and absolute difference at most
   `1e-6` for reported floating losses and policy metrics;
2. run one checkpoint-free cached 30-update profile under autotune level 0,
   still through the guard;
3. require at least 95% of the retained incumbent profile's end-to-end
   examples/s, no more than 2% peak-JAX-HBM regression, finite unclipped
   updates, and the same compiler cost-analysis fields; and
4. only then make autotune level 0 the guarded launcher default and add tests
   for its fail-closed environment composition.

If any gate fails, do not adopt the flag, do not alter the guard, batch size,
autotune level, model, or loss as a rescue within this experiment. Remove the
dedicated cache and any empty/partial run directory after preserving compact
text/JSON evidence. Write no result row because this is not a 30-minute model
experiment. Resume model autoresearch only after a separate compilation path
is preregistered and shown safe.
