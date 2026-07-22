# Experiment 012: unchunked fused BT4 encode with full gradients

Status: preregistered on 2026-07-22 before changing the active override or
measuring the candidate on GPU.

## Question and hypothesis

Experiment 011 reproducibly raises fixed-run throughput by about 32% while
stopping only future-target BT4 gradients, but it also changes the encoder
execution topology. The balanced-K1 incumbent uses `bt4_encode_chunk_size=1`,
which scans over current and future boards as two checkpointed encoder calls.
Experiment 011 instead makes two explicit asymmetric calls. XLA reports 3.72%
more static FLOPs and fixed-run peak HBM rises 32.11%, despite the large speed
gain. The gain therefore cannot yet be attributed cleanly to removed future
backward work.

This experiment isolates execution topology. It keeps every gradient attached
and changes only the existing encode setting from the scanned path to the
unchunked path, which concatenates the current board and one balanced sampled
future board and encodes the resulting physical batch of 256 boards in one
BT4 call. The primary hypothesis is that larger fused kernels recover a useful
fraction of Experiment 011's throughput while preserving the incumbent's
policy behavior. The main failure mode is A10G OOM because full backward must
retain activations for both boards simultaneously.

## Candidate contract

Relative to balanced-K1, activate exactly:

```python
"bt4_encode_chunk_size": 0,
```

Keep `bt4_future_target_stop_gradient=False` and
`bt4_freeze_backbone=False`. The candidate semantics are:

- Concatenate current and one balanced per-example sampled future board,
  flatten to `2 * batch_size`, and execute one BT4 encoder call.
- Preserve gradients from both current and future token outputs into every
  BT4 encoder parameter. Preserve the shared state projector and every
  downstream gradient exactly as the incumbent objective defines them.
- Keep the source-compatible model and optimizer ABI, BT4 optimizer moments,
  learning-rate schedule, parameter dtypes, and checkpoint format unchanged.
- Preserve mathematically equivalent forward values; bound any batch-shape or
  scan-lowering roundoff in CPU preflight.
- Keep full-horizon evaluation and eight-pass searchless inference unchanged.
  Inference encodes only current boards and is unaffected by this training-only
  batching choice.

No loss change is allowed. RMS norm matching remains off; target SIGReg is
`5.76`; z_pred SIGReg is `1.0`; the V-statistic uses exactly 64 physical-batch
examples. Keep balanced K=1 target assignment, both projector blocks, all four
DFM blocks, eight DFM/JEPA horizons, batch/eval batch `128/64`, main/BT4 peak
rates `3e-5/1e-6`, and the `400/800/0.1` cosine schedule.

## Frozen controls

Balanced-K1 v1/update 1,581 remains the strength and systems control:

- two-pool CE/accuracy/legal mass
  `4.4979399741/0.1100921631/0.6497235410`;
- cached profile `117.3798961` end-to-end and `149.0971679` device
  examples/s;
- fixed-run throughput `117.0610578` examples/s;
- fixed-run peak JAX HBM `9,574,246,400` bytes; and
- selected state SHA-256
  `26c3621d1f9c35aa1401ef5e2e9d35ffdf59607d0c4369730cc6dd207b65c7c7`.

Experiment 011 is an attribution control only. Its profile is
`154.2508/216.7118` end-to-end/device examples/s and peak HBM is
`12,540,134,656` bytes. Its exact fixed runs are rejected because repeat legal
mass misses the frozen floor; no state is retained or used for initialization.

## Staged gates

Before any fixed-time run:

1. CPU tests prove incumbent-path stability, candidate numerical forward
   equivalence, nonzero encoder gradients from current-only and future-only
   losses, unchanged model/optimizer ABI, unchanged DFM RNG/action values,
   full-horizon evaluation, and the active one-knob override.
2. A one-update real-checkpoint A10G smoke at batch 128 must compile and finish
   without OOM, report finite unclipped loss, two encoded boards per example,
   balanced 16-per-horizon assignments, eight predictions, SIGReg counts
   `576/512`, and peak HBM below the device limit. Save no checkpoint.
3. Only after the smoke passes, run the cached 30-update profile with 100 ms
   hardware monitoring. No-regression throughput is `117.3798961`
   end-to-end examples/s. The preregistered useful-effect gate is 10%, or
   `129.1178857` examples/s. An OOM or throughput below the useful-effect gate
   ends the experiment without spending a 30-minute run.

Interpretation is frozen in advance: performance close to Experiment 011
means fused execution explains most of its systems gain; performance close to
balanced-K1 means detached future backward explains most; an OOM means the
asymmetric detached graph enabled an otherwise infeasible high-memory
schedule. Report peak HBM, device throughput, compiler FLOPs/bytes, utilization,
power, and data stalls in every case.

## Fixed-time decision if the profile passes

Start from step 265,000 model-only with a fresh optimizer; seed 0 and global
permutation; 1,800 steady seconds; validation seeds 10,000 and 20,000 with 64
batches of 64; sparse saves at 400, 800, 1200, 1600, and terminal; at most five
temporary states. Select minimum two-pool CE.

Acceptance requires CE below `4.4969695099`, accuracy at least
`0.1080921631`, legal mass at least `0.6467235410`, finite metrics, positive
prediction beating zero and action-shuffled controls at every horizon,
effective-rank mean/minimum at least `30.0/28.5`, feature-std p05 mean/minimum
at least `0.63/0.61`, target RMS at least `0.90`, and prediction/target RMS
ratio in `[0.94, 1.02]`. A passing primary requires one exact repeat; that
repeat must beat balanced-K1 CE and pass every non-CE gate before any arena.
Rejected runs retain no state. Keep the GPU single-tenant and all artifacts,
caches, and temporary files below `/mountpoint/.exp`.
