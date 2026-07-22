# Experiment 019: first-action mask with low-time-biased corruption

Status: preregistered on 2026-07-22; activation and measurements have not
started.

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
