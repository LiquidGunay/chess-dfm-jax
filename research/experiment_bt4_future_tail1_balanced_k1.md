# Experiment 014: one-block future-target BT4 gradient tail

Status: completed and accepted as the offline incumbent on 2026-07-22. The
frozen 128-pair arena was inconclusive and contained one symmetric legacy-codec
coverage fault; this is not an Elo promotion.

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

## Outcome

The implementation and every pre-run systems gate passed. Forty focused CPU
tests plus lint prove the active `14/1` detached/attached split, zero
future-only gradients through the embedding and blocks 0--13, nonzero
future-only gradients through block 14 and the shared projector, complete
current-board gradients, unchanged checkpoint/optimizer ABI, explicit routing
metadata, and fail-closed invalid configurations.

The real-checkpoint batch-128 smoke completed one unclipped update with exactly
16 examples assigned to every horizon, eight predictions, target/prediction
SIGReg counts `576/512`, and no state write. Cold compilation took `173.230`
seconds and peak JAX HBM was `12,866,583,552` bytes.

The cached 30-update profile reached `153.5362` end-to-end and `215.7788`
device examples/s at `12,875,020,288` bytes peak JAX HBM. It is only `0.46%`
slower than the zero-tail profile and `1.98%` faster than the three-block tail.
The compiler reports `14.2575` TFLOP and `133.447` GB per update. Mean/p50/p95
GPU utilization was `59.93/90/100%`, mean/p95 power was `169.63/196.91` W,
and the measured data-stall fraction was `28.85%`.

The primary 30-minute run compiled and completed its first update in `14.6871`
seconds, then processed `265,216` examples in 2,072 updates at `153.0754`
end-to-end and `213.2427` device examples/s. Peak JAX HBM was
`12,973,654,528` bytes. The frozen two-pool scan was:

| Update | DFM CE | Accuracy | Legal mass |
|---:|---:|---:|---:|
| 800 | 4.5034462 | 0.1087341 | 0.6441181 |
| 1,600 | 4.4980034 | 0.1097870 | 0.6477664 |
| 2,072 | **4.4950091** | **0.1104584** | **0.6488690** |

Selected update 2,072 improves the balanced-K1 incumbent by `0.0029309` CE
and clears the repeat-noise-aware ceiling by `0.0019604`. Its prediction
effective-rank mean/minimum is `30.9728/29.1101`, feature-std p05
mean/minimum is `0.64977/0.63551`, mean target RMS is `0.92525`, and the
prediction/target RMS ratio is `0.97148`. Positive JEPA prediction beats zero,
identity, shuffled-target, and action-shuffled controls at every horizon; its
mean JEPA/identity ratio is `0.14678`.

The exact repeat processed `266,240` examples in 2,080 updates at `153.4504`
end-to-end examples/s. Its selected terminal checkpoint reaches CE/accuracy/
legal mass `4.4971840/0.1103058/0.6468926`, independently beats the incumbent,
and passes every non-CE gate. Its effective-rank mean/minimum is
`30.9981/29.1399`, feature-std p05 mean/minimum is `0.64985/0.63608`, mean
target RMS is `0.92619`, and the prediction/target RMS ratio is `0.97041`.
The primary/repeat CE separation is `0.0021749`; the preregistered repeat gate
requires an independent incumbent beat rather than matching the primary
effect size, and it passes.

The preregistered primary checkpoint then scored `0.49609375` over 128
color-reversed development pairs against balanced-K1/update 1,581: 2 wins,
250 draws, and 4 losses across 256 games, pentanomial `[0, 4, 122, 2, 0]`.
Descriptive logistic Elo is `-2.71`, with a pair-aware 95% interval of
`[-87.96, +82.20]`; seven games hit the 256-ply cap. Mean policy-call time was
`43.73` ms for the candidate and `43.76` ms for the incumbent.

One candidate loss was a frozen `no_representable_move` fault. Both models use
the same declared `legacy_absolute_1858` codec, which cannot represent black
promotions; in the terminal position all eight legal moves were black
promotions. The acting-model-loses policy charged the fault to the candidate,
and the reported score already includes it. This is not an unmasked illegal
model choice, but it is a known symmetric action-space limitation and prevents
interpreting the development screen as promotion evidence.

Accept primary update 2,072 as the new offline incumbent because both exact
runs clear their frozen offline gates. The arena is indistinguishable from a
tie and does not authorize promotion. Retain only the primary selected state
for this experiment; its SHA-256 is
`69b11b51e7f98885deecefca541e1b8408795367bd60e89aa3f279b56e953edb`.
Delete the repeat state and all nonselected states after preserving compact
reports, scans, diagnostics, and arena evidence. The tail-depth line is now
closed: one block is the accepted boundary, and depths 2 or 4--15 are not a
priority. RMS norm matching remains off and target/`z_pred` SIGReg remain
fixed at `5.76/1.0`.
