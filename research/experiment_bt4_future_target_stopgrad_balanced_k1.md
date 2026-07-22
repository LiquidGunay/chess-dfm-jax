# Experiment 011: stop-gradient future BT4 target branch

Status: preregistered on 2026-07-22 before implementing the asymmetric encode
path or measuring it on GPU.

## Question and hypothesis

Experiment 010 shows that encoder backward/optimizer work is expensive: a
fully frozen BT4 backbone raises fixed-run throughput by `58.59%`, cuts peak
JAX HBM by `66.02%`, and improves two-pool CE to `4.4965047017`. It also lowers
legal mass to `0.6445921361`, below the frozen floor. This suggests that
current-board action gradients into BT4 matter, while it remains unknown
whether gradients through the separately encoded future target are worth
their cost.

This experiment keeps BT4 fully trainable from the current-board DFM,
legality, recurrent-JEPA, and current-state SIGReg paths, but stops gradients
at the future-target BT4 token output. The state projector remains attached on
both branches, so target SIGReg and positive JEPA loss still train the shared
projector. The primary hypothesis is that this asymmetric target branch
recovers the incumbent's legal mass while removing enough encoder backward
work to process more examples and improve fixed-time CE. The failure modes are
that target-side backbone gradients are important representation alignment,
or that shared-parameter optimizer work and the current-board backward pass
dominate enough to hide the compute saving.

## Candidate contract

Add a default-off configuration switch and activate exactly:

```python
"bt4_future_target_stop_gradient": True,
```

Restore `bt4_freeze_backbone=False` and BT4 peak learning rate `1e-6`. The
candidate semantics are:

- Encode current boards and the one balanced sampled future in separate BT4
  calls, then concatenate their token outputs before the unchanged shared
  state projector.
- Preserve current-board token gradients into every BT4 embedding/layer
  parameter. Apply `stop_gradient` only to the future BT4 token output.
- Keep the future target attached to the state projector. Positive JEPA MSE
  and target SIGReg therefore update projector parameters exactly as before;
  they simply do not update BT4 through the future branch.
- Keep the full source-compatible model and optimizer ABI. All BT4 parameters
  retain optimizer moments and may update from current-board gradients.
- Preserve exact forward values for identical parameters and inputs.
  Evaluation remains full-horizon; inference is unchanged and does not encode
  future boards.
- Initially support only online targets, balanced per-example K=1 sampling,
  no encode chunking, no sampled-target anchors, and no full-backbone freeze.
  Unsupported combinations fail closed.
- Serialize the switch only when enabled. Reports and resume contracts identify
  the two encode calls, stop-gradient boundary, trainable/frozen encoded-board
  counts, attached projector, unchanged optimizer ABI, and unchanged
  evaluation/inference semantics.

No loss coefficient changes: RMS norm matching stays off, target SIGReg stays
`5.76`, prediction SIGReg stays `1.0`, and the fixed 64-example V-statistic
sample remains active. Keep both projector blocks, all four DFM blocks, all
eight DFM/JEPA prediction horizons, the 400/800/10%-floor cosine schedule,
batch 128, and eight-pass searchless inference.

## Frozen controls and run configuration

The strength control remains balanced-K1 v1/update 1,581:

- state SHA-256:
  `26c3621d1f9c35aa1401ef5e2e9d35ffdf59607d0c4369730cc6dd207b65c7c7`;
- two-pool CE/accuracy/legal mass
  `4.4979399741/0.1100921631/0.6497235410`;
- cached profile `117.3798961` end-to-end and `149.0971679` device
  examples/s;
- fixed-run throughput `117.0610578` examples/s; and
- accepted repeat separation `0.0009704642`.

Full freeze is a mechanistic control only: profile/fixed throughput
`186.7543/185.6428`, selected CE/accuracy/legal mass
`4.4965047017/0.1097259521/0.6445921361`, and peak HBM `3,253,244,672` bytes.
It is rejected and supplies no initialization or acceptance baseline.

Start from the recovered step-265,000 model with a fresh optimizer. Freeze one
NVIDIA A10G and no concurrent workload; physical/evaluation batches `128/64`;
train/data seed 0 with `global_permutation`; peak main/BT4 rates
`3e-5/1e-6`; schedule start/decay/floor `400/800/0.1`; normalized loss
coefficients `0.0/5.76/1.0`; 1,800 seconds of steady-state training after
compile/first update; validation seeds 10,000 and 20,000, each 64 batches of
64 examples; and sparse saves at updates 400, 800, 1200, 1600, and terminal
with at most five temporary states.

## Correctness and performance gates

Before the fixed run:

1. CPU tests prove default exactness; candidate forward equality; current-only
   BT4 gradients; exact-zero future-branch BT4 gradients; attached future
   projector gradients; unchanged model/optimizer ABI; full-horizon
   evaluation; unchanged DFM RNG and inference; explicit reporting/resume
   semantics; and fail-closed unsupported combinations.
2. A one-update real-checkpoint A10G smoke proves nonzero BT4 parameter change,
   a finite unclipped update, full model depths, exactly one trainable current
   plus one stop-gradient future encode per example, balanced `16`-per-horizon
   assignment, SIGReg counts `576/512`, and no write outside
   `/mountpoint/.exp`.
3. A cached 30-update profile must not regress below `117.3798961`
   end-to-end examples/s. The expected useful effect is at least 10%
   (`129.1178857` examples/s); a smaller gain is recorded rather than assumed.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical seed-10,000 and seed-20,000
full-horizon pools and select minimum mean DFM CE. Acceptance requires:

- two-pool DFM CE below `4.4969695099`, more than the incumbent repeat
  separation better than balanced-K1 v1;
- accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

Compare the complete learning curve and systems profile against both balanced
K1 runs and full freeze. If the primary passes, repeat it exactly; the repeat
must beat balanced-K1 v1 CE and pass every non-CE gate before the frozen
128-pair arena against balanced-K1 v1/update 1,581. A failure gets no repeat,
arena, pass-count sweep, or SAE refit. Retain only a qualifying selected state;
otherwise delete every candidate state after preserving compact evidence.
