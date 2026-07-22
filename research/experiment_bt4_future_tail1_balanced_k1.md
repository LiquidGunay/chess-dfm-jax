# Experiment 014: one-block future-target BT4 gradient tail

Status: preregistered on 2026-07-22; implementation activation and
measurements have not started.

## Question and hypothesis

The zero-tail candidate in Experiment 011 is the strongest and fastest member
of the asymmetric family, but its exact repeat misses legal mass by
`0.0003756`. The three-block tail in Experiment 013 restores terminal legal
mass, yet its CE `4.4977803` improves the incumbent by only `0.0001597`, inside
accepted repeat noise. Full-gradient fused execution in Experiment 012 has
similar CE and legal behavior to the three-block result. More future-gradient
capacity therefore does not appear to improve the fixed-time policy metric,
but a very small attached path may still supply the alignment signal missing
from the zero-tail repeat.

This experiment attaches only the final BT4 block on the future branch. The
future embedding and blocks 0--13 run forward, their activation is detached,
and block 14 plus the shared state projector remain attached. The current
board continues to train the complete encoder. The primary hypothesis is that
the smallest possible block-level future gradient preserves zero-tail's CE
and throughput behavior while repairing its narrow repeat legal-mass miss.

This is the final preregistered tail-depth interpolation, not the first point
in a sweep. One block owns `11,078,912` trainable parameters, `5.67%` of the
`195,305,728`-parameter encoder. The detached future prefix contains
`184,226,816` parameters (`94.33%`). If this candidate fails either its
primary or repeat gate, close the tail-depth line rather than testing depths
2 or 4--15.

## Candidate contract

The implementation and checkpoint-compatible boundary from Experiment 013
remain unchanged. Activate exactly:

```python
"bt4_encode_chunk_size": 0,
"bt4_future_target_stop_gradient": True,
"bt4_future_target_trainable_tail_layers": 1,
```

Candidate semantics:

- Encode the current board with gradients through the embedding and all 15
  transformer blocks.
- Encode the one balanced sampled future board separately. Run the embedding
  and blocks 0--13, detach the boundary activation, then run block 14 with
  gradients attached.
- Keep the future token output and shared state projector attached. Positive
  JEPA MSE and target SIGReg update block 14 and the projector, but cannot
  update the future prefix.
- Preserve exact forward mathematics, parameter paths, model/optimizer state,
  checkpoint ABI, learning-rate partitions, and resume compatibility. Every
  BT4 parameter retains optimizer moments because the current path remains
  fully trainable.
- Keep full-horizon validation and eight-pass searchless inference unchanged;
  inference never uses the training-only future branch.
- Reports must identify one full-gradient current board, one partial-gradient
  future board, detached/attached depths `14/1`, and the attached projector.
  Invalid combinations continue to fail closed.

The objective is frozen, not retuned: RMS norm matching coefficient `0.0`,
target SIGReg `5.76`, `z_pred` SIGReg `1.0`, and a fixed 64-example normalized
SIGReg sample. Keep balanced per-example K=1 assignment, both projector
blocks, all four DFM blocks, eight horizons, train/eval batch `128/64`, peak
main/BT4 rates `3e-5/1e-6`, the `400/800/0.1` cosine schedule, global data
permutation, and eight inference passes.

## Frozen controls

Balanced-K1/update 1,581 remains the strength control at two-pool
CE/accuracy/legal mass `4.4979399741/0.1100921631/0.6497235410`, cached
profile `117.3798961` examples/s, fixed-run throughput `117.0610578`
examples/s, and accepted repeat CE separation `0.0009704642`.

The asymmetric controls are:

- zero tail: profile/fixed throughput `154.2508/154.6220`, primary/repeat CE
  `4.4942136/4.4963869`, and repeat legal mass `0.6463479`; rejected at the
  repeat legal gate;
- three-block tail: profile/fixed throughput `150.5611/149.5310`, terminal CE
  `4.4977803`, and legal mass `0.6467812`; rejected at the primary CE gate;
  and
- fused full gradient: profile/fixed throughput `133.6284/134.6191`, terminal
  CE `4.4975412`, and legal mass `0.6468241`; rejected at the primary CE gate.

No rejected state is retained or used for initialization.

## Staged gates

Before the fixed-time run:

1. Focused CPU tests must prove the active depth is one; forward values and
   model/optimizer ABI remain unchanged; future-only gradients are exact zero
   in the embedding and blocks 0--13 and nonzero in block 14 plus the shared
   projector; current-only gradients reach the complete trunk; evaluation and
   inference remain unchanged; routing/resume metadata are explicit; and
   invalid depths fail closed.
2. A one-update real-checkpoint A10G smoke at batch 128 must finish with finite
   unclipped values, detached/attached depths `14/1`, balanced
   16-per-horizon assignments, eight predictions, SIGReg counts `576/512`, no
   checkpoint write, and peak HBM below the device limit.
3. A checkpoint-free cached 30-update profile must clear the original useful
   systems floor of `129.1178857` examples/s and stay below the fused
   full-gradient peak of `16,241,594,112` bytes. Record comparisons to both
   zero-tail and three-tail, compiler FLOPs/bytes, utilization, power, and
   data stalls. Failure ends the experiment without a fixed run.

## Fixed-time decision if systems gates pass

Start from recovered step 265,000 model-only with a fresh optimizer and seed
0. Train for 1,800 steady-state seconds on one single-tenant A10G. Evaluate
temporary updates 800, 1600, and terminal on the identical seed-10,000 and
seed-20,000 pools, each 64 batches of 64 examples, and select minimum mean DFM
CE. Keep at most those three temporary states and delete every rejected state
after preserving compact evidence.

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

A passing primary requires one exact repeat. The repeat must beat incumbent CE
and pass every non-CE gate before the frozen 128-pair arena. A failure gets no
arena, pass-count sweep, or SAE refit and closes the tail-depth line. Keep all
artifacts, caches, and temporary files below `/mountpoint/.exp`, and never
overlap GPU workloads.
