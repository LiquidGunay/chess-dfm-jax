# Experiment 019: first-action mask with low-time-biased corruption

Status: completed and rejected on 2026-07-22. Both exact runs pass every
offline gate and the candidate beats update 2,072 head to head, but its frozen
raw-BT4 score regresses below the required anchor. No candidate state is
retained, the accepted update 2,072 remains incumbent, and the corruption line
is closed.

## Question and hypothesis

Experiment 018 always masks the played first action and improves selected H1
CE from `2.8470488` to `2.8417612` while also improving uniform CE and passing
accuracy, legality, and every latent gate. It misses the frozen H1 ceiling by
only `0.0015028`. The DFM action transformer is bidirectional over the noisy
trajectory, however, so on a typical uniform-time training example roughly
half of actions 2--8 remain ground truth. Frozen validation and the first
inference pass begin with every action masked.

This experiment retains the forced H1 mask and transforms the existing
uniform draw `u` into training time `t=u^2`. Because masking probability is
`1-t`, its expectation rises from `1/2` to `2/3`, while every time in `(0,1)`
still has support and the time embedding remains consistent with the actual
mask probability. The hypothesis is that reducing ground-truth future-action
leakage closes Experiment 018's small H1 margin without sacrificing its
uniform CE or legal mass.

## Candidate contract

Add one default-off scalar and activate exactly:

```python
"dfm_force_first_action_mask": True,
"dfm_training_time_power": 2.0,
```

- Draw the same `u ~ Uniform(0,1)` from the existing `rng_t`, then use
  `t=u^2` only in normalized training. Do not consume or reorder RNG values.
- Generate the existing full action-mask random matrix unchanged, compare it
  against `1-t`, then force only H1 to the mask token as in Experiment 018.
  For the same random draws, every H2--H8 mask selected by `t=u` must remain
  selected by `t=u^2`; additional masks may be selected.
- Leave deterministic validation time, all evaluation forwards, inference,
  target actions, data order, legal masks, labels, metrics, and codec
  unchanged.
- Keep uniform eight-horizon DFM CE, source first-legality coefficient `2.0`,
  the accepted one-block future-gradient tail, balanced per-example K=1
  targets, both projector blocks, all four DFM and JEPA blocks, eight
  prediction horizons, train/eval batch `128/64`, peak main/BT4 rates
  `3e-5/1e-6`, the `400/800/0.1` cosine schedule, global data permutation,
  and eight-pass searchless inference.
- Keep RMS norm matching disabled, target SIGReg `5.76`, `z_pred` SIGReg
  `1.0`, and the fixed 64-example normalized SIGReg sample with effective
  target/prediction counts `576/512`.
- Preserve model, optimizer, and checkpoint state ABI. Record both training-
  only corruption controls in serialized config and resume metadata. Default
  power `1.0` must return the original uniform draw without adding arithmetic
  and preserve exact loss, aux, gradient, and RNG behavior.

Initialize from recovered step 265,000 model-only with a fresh optimizer and
seed 0. Do not initialize from the rejected Experiment 018 state.

## Staged gates

1. Focused CPU tests must prove exact `u^2`, identity at power `1.0`, unchanged
   RNG streams, H1 mask fraction/count `1.0/batch`, H2--H8 mask superset
   behavior, unchanged targets and target/SIGReg assignments, uniform CE,
   evaluation parity, unchanged state ABI/inference, serialization/resume
   metadata, and fail-closed finite-positive power validation.
2. A one-update real-checkpoint A10G smoke at batch 128 must be finite and
   unclipped with active time power `2.0`, H1 mask fraction `1.0`, no
   weighted-CE branch, source legality coefficient `2.0`, balanced
   16-per-horizon targets, SIGReg counts `576/512`, future routing `14/1`, no
   state write, and peak HBM below the device limit.
3. A checkpoint-free cached 30-update profile must reach at least `150.0`
   end-to-end examples/s and stay below `13,100,000,000` peak JAX HBM bytes.
   Record compiler work, utilization, power, and data stalls.

## Fixed-time decision

Train for 1,800 steady-state seconds and keep temporary updates 800, 1600, and
terminal only. Evaluate all three on unchanged seed-10,000 and seed-20,000
pools, each 64 batches of 64 examples. Select minimum two-pool H1 CE, breaking
an exact tie with lower uniform CE.

Primary acceptance requires:

- H1 CE below `2.8402584158`;
- uniform DFM CE no greater than `4.4971840288`;
- aggregate accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

A passing primary requires one exact repeat. The repeat must beat accepted
primary H1 CE `2.8470487520`, stay within the uniform-CE ceiling, and pass
every non-CE gate. Only then run frozen 128-pair cap-256 arenas against
accepted update 2,072 and raw BT4. An Elo-aligned offline incumbent requires
point score above `50%` against update 2,072 and above `36.5234375%` against
raw BT4. These development screens cannot authorize Elo promotion.

Failure at any earlier gate gets no repeat, arena, pass-count sweep, or SAE
work. Retain only a qualifying primary selected state and compact evidence;
delete all nonselected and repeat states. A primary or repeat failure closes
this corruption line rather than authorizing a time-power sweep. Keep all
mutable files below `/mountpoint/.exp`, and never overlap GPU workloads.

## Outcome

All implementation, CPU, smoke, and profile gates passed. The one-update
batch-128 smoke cold-compiled in `170.421 s`, used `12,865,654,016` peak JAX
HBM bytes, and confirmed active power `2.0`, H1 mask fraction `1.0`, observed
mean mask probability `0.6830`, uniform CE, source legality coefficient
`2.0`, balanced 16-per-horizon targets, SIGReg counts `576/512`, future
routing `14/1`, finite unclipped loss, and no state write. The 30-update
profile reached `151.563` end-to-end and `213.172` device examples/s at
`12,865,649,664` peak HBM bytes. XLA estimated `14.2575e12` FLOP and
`133.447e9` bytes per update; measured data stalls were `28.90%`, GPU
utilization mean/median/p95 was `61.88/88/100%`, and power mean/p95 was
`168.35/197.26 W`.

The primary fixed run completed 2,063 updates and 264,064 examples at
`152.663` steady end-to-end and `212.616` device examples/s, with
`12,967,065,856` peak HBM bytes. Two-pool H1-first selection chose its
terminal checkpoint:

| Update | H1 CE | Uniform CE | Accuracy | Legal mass |
|---:|---:|---:|---:|---:|
| 800 | 2.8761915 | 4.5040907 | 0.1092072 | 0.6451670 |
| 1600 | 2.8593588 | 4.4977851 | 0.1103516 | 0.6480522 |
| **2063** | **2.8398960** | **4.4907011** | **0.1103363** | **0.6487805** |

Selected update 2,063 beats the frozen H1 ceiling by `0.0003624` and passes
every other primary gate. Prediction effective-rank mean/min is
`30.9886/29.1578`, feature-std p05 mean/min is `0.64984/0.63625`, mean target
RMS is `0.92603`, and per-horizon prediction/target RMS ratio is
`0.96474--0.97640`. Positive JEPA prediction beats zero and action-shuffled
controls at every horizon.

The exact repeat completed 2,067 updates and 264,576 examples at `152.645`
steady end-to-end and `212.327` device examples/s, with `12,939,756,288` peak
HBM bytes. Its H1-first terminal checkpoint independently passes at H1/uniform
CE `2.8451609/4.4927080`, accuracy `0.1099701`, and legal mass `0.6475881`.
It beats the accepted-repeat H1 ceiling by `0.0018878`; effective-rank
mean/min is `30.9918/29.1289`, feature-std p05 mean/min is
`0.64989/0.63636`, mean target RMS is `0.92560`, RMS ratio is
`0.96549--0.97701`, and both trivial controls lose at every horizon.

The primary then scores `0.5078125` in the frozen 128-pair cap-256 arena
against accepted update 2,072: 7 wins, 246 draws, and 3 losses, pentanomial
`[0, 3, 118, 7, 0]`, with no faults and four cap draws. Descriptive logistic
Elo is `+5.43`, with pair-aware 95% interval `[-79.33, +90.86]`. Mean policy
call time is `42.90` ms for the candidate and `43.03` ms for the incumbent.
This passes the frozen `>50%` point-score gate but is not a promotion claim.

Against original raw BT4, the same checkpoint scores only `0.34765625`: 0
wins, 178 draws, and 78 losses, pentanomial `[13, 52, 63, 0, 0]`, with no cap
draws. Descriptive Elo is `-109.33`, with pair-aware interval
`[-212.25, -22.48]`. Candidate/raw mean policy-call time is
`42.51/33.36` ms. One candidate loss is the known
`no_representable_move` black-promotion fault of the legacy codec; the score
already includes it. The result misses the frozen raw-BT4 anchor
`0.365234375` by `0.017578125`.

The experiment is therefore rejected as an Elo-aligned incumbent despite its
repeat-stable offline gain and positive incumbent point score. Delete primary
state SHA-256 `4e6cef86137d276007e0a5d00a3546e4a9c88f68a31db630acaf38389c18f6b4`
and repeat state SHA-256
`900c6ae72ed2e85f6e9349b891d60903fe6c132b275c01e2710fe14a8f325b34`,
retain compact reports/scans/arena evidence only, restore forced H1 masking
off and training-time power `1.0`, and keep update 2,072 as the sole current
incumbent. Do not sweep the corruption power or run SAE work from this
candidate. RMS norm matching remains disabled and target/`z_pred` SIGReg stay
fixed at `5.76/1.0`.
