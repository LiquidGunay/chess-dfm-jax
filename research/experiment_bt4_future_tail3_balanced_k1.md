# Experiment 013: three-block future-target BT4 gradient tail

Status: preregistered on 2026-07-22; implementation and measurements have not
started.

## Question and hypothesis

Experiment 011 showed that removing future-target BT4 backward work raises
fixed-run throughput by about 32%, but its exact repeat narrowly missed the
legal-mass floor. Experiment 012 showed that fused execution alone explains
about 44% of that absolute profile gain; the remaining gain comes from the
future-gradient boundary under the measured lowerings. The all-or-nothing
boundary may therefore discard a useful alignment signal along with its
backward cost.

This experiment keeps the current-board BT4 path fully trainable and restores
future-target gradients only through the final three of BT4's 15 transformer
blocks. The embedding and first 12 blocks run forward for the future board,
then their activation is detached before blocks 12--14. The shared state
projector remains attached. The primary hypothesis is that a small late
gradient path recovers repeat-stable legal policy behavior while retaining
most of Experiment 011's throughput advantage.

The exact depth is fixed before implementation. Every BT4 block owns
`11,078,912` trainable parameters including its copied Smolgen matrix. The
three-block tail therefore exposes `33,236,736` of the encoder's
`195,305,728` parameters (`17.02%`) to future-target gradients. The detached
future prefix contains the `29,122,048`-parameter embedding and 12 blocks,
or `162,068,992` parameters (`82.98%`). Three blocks are the final fifth of
the trunk: large enough to provide a meaningful representation-alignment
path, while preserving roughly four fifths of the block-level future
backward saving. The compiler profile, rather than this linear estimate, is
the authoritative systems measurement.

## Candidate contract

Add a default-off integer setting and activate exactly:

```python
"bt4_future_target_stop_gradient": True,
"bt4_future_target_trainable_tail_layers": 3,
```

Candidate semantics:

- Encode the current board in one ordinary BT4 call with gradients through the
  embedding and all 15 blocks.
- Encode the one balanced sampled future board in a separate call. Run the
  embedding and blocks 0--11, apply `stop_gradient` to the boundary
  activation, and run blocks 12--14 with gradients attached.
- Do not detach the future token output or the shared state projector.
  Positive JEPA MSE and target SIGReg update the projector and final three BT4
  blocks, but cannot update the future embedding or first 12 blocks.
- Preserve exact forward mathematics, parameter paths, model state,
  optimizer state, checkpoint ABI, learning-rate groups, and resume
  compatibility. All BT4 parameters retain optimizer moments because the
  current-board path still trains the full encoder.
- Keep full-horizon validation and eight-pass searchless inference unchanged.
  Inference encodes only current boards and never uses this training-only
  future boundary.
- General support is bounded to an integer tail depth in `[0, 15]` when the
  existing future-target stop-gradient mode is enabled. Depth zero retains
  Experiment 011's fully detached future encode. Unsupported types, ranges,
  full-backbone freeze, non-online target semantics, target counts other than
  balanced per-example K=1, or incompatible chunking fail closed.
- Reports and resume contracts identify the boundary, prefix/tail depths,
  attached projector, encoded-board counts, and unchanged optimizer ABI.

The loss is not an experiment variable. RMS norm matching stays disabled;
target SIGReg remains `5.76`; `z_pred` SIGReg remains `1.0`; and normalized
SIGReg retains its fixed 64-example V-statistic sample. Keep both projector
blocks, all four DFM blocks, all eight prediction horizons, physical/eval
batches `128/64`, main/BT4 peak learning rates `3e-5/1e-6`, the
`400/800/0.1` cosine schedule, global-permutation data, and eight inference
passes.

## Frozen controls

Balanced-K1 v1/update 1,581 remains the strength control:

- two-pool CE/accuracy/legal mass
  `4.4979399741/0.1100921631/0.6497235410`;
- cached profile `117.3798961/149.0971679` end-to-end/device examples/s;
- fixed-run throughput `117.0610578` examples/s and peak JAX HBM
  `9,574,246,400` bytes;
- accepted exact-repeat CE separation `0.0009704642`; and
- state SHA-256
  `26c3621d1f9c35aa1401ef5e2e9d35ffdf59607d0c4369730cc6dd207b65c7c7`.

Experiment 011 is the zero-tail mechanistic control. Its profile reaches
`154.2508/216.7118` examples/s at `12,540,134,656` peak bytes, and its exact
fixed runs select CE `4.4942136/4.4963869`; it is rejected because repeat
legal mass is `0.6463479`. Experiment 012 is the full-gradient systems
control. Its profile reaches `133.6284/177.1029` examples/s at
`16,241,594,112` peak bytes; it is rejected because primary CE `4.4975412`
does not clear repeat noise. Neither rejected state is retained or used for
initialization.

## Staged gates

Before a fixed-time run:

1. CPU tests must prove default-path exactness; candidate forward equivalence;
   nonzero future-only gradient in blocks 12--14 and the shared projector;
   exact-zero future-only gradient in the embedding and blocks 0--11;
   nonzero current-only encoder gradients across the trunk; unchanged
   model/optimizer ABI, DFM values, RNG streams, full-horizon evaluation, and
   inference; explicit report/resume semantics; and fail-closed invalid
   configurations.
2. A one-update real-checkpoint A10G smoke at batch 128 must compile and finish
   with finite unclipped values. It must report one full-gradient current
   encode plus one 12-prefix-detached/3-tail-attached future encode per
   example, balanced `16`-per-horizon assignments, eight predictions, SIGReg
   counts `576/512`, and peak HBM below the device limit. Save no checkpoint.
3. A cached 30-update profile with 100 ms hardware monitoring must reach at
   least `129.1178857` end-to-end examples/s, 10% above balanced-K1, and peak
   below the fused full-gradient profile's `16,241,594,112` bytes. Report
   device throughput, compiler FLOPs/bytes, utilization, power, and data
   stalls. Failure ends the experiment without a 30-minute run.

The useful expected region is between the zero-tail and full-gradient
controls. A result near zero-tail throughput means three tail blocks are
cheap; a result near the fused control means their backward graph dominates
despite their depth. The measured profile governs continuation.

## Fixed-time decision if systems gates pass

Start from recovered step 265,000 model-only with a fresh optimizer, seed 0,
and one single-tenant NVIDIA A10G. Train for 1,800 steady-state seconds after
compile/first update. Evaluate temporary checkpoints on the identical
seed-10,000 and seed-20,000 pools, each 64 batches of 64 examples, and select
minimum mean DFM CE. Save only updates 800, 1600, and terminal, with at most
three temporary states; retain only a final qualifying state and delete all
others after compact evidence is recorded.

Acceptance requires:

- two-pool DFM CE below `4.4969695099`;
- accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

A passing primary requires one exact repeat. The repeat must beat the
balanced-K1 incumbent CE and pass every non-CE gate before a frozen 128-pair
arena against balanced-K1/update 1,581. A failure gets no arena, pass-count
sweep, or SAE refit. Keep all artifacts, caches, and temporary files below
`/mountpoint/.exp`, and never overlap GPU workloads.
