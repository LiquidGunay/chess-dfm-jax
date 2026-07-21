# Experiment 008: one sampled future target

Status: preregistered on 2026-07-21 before changing the active override or
measuring the candidate on GPU.

## Question and hypothesis

Experiment 001 established that sampling two of eight future BT4 targets is a
large training-efficiency win over encoding all eight. Experiment 006 then
made the K=2 fixed-time trajectory stable with cosine warmdown. This experiment
asks whether sampling one future target per update is a better point on the
compute-versus-estimator-variance frontier.

The primary hypothesis is that removing one more trainable future BT4 encode
raises cached A10G throughput by at least 5% and yields enough additional
fixed-time updates to improve full-horizon policy CE beyond the accepted K=2
repeat envelope. The main risk is that a size-one Monte Carlo target gives the
positive JEPA loss and target-SIGReg gradient too much horizon variance,
damaging policy learning or latent health despite higher throughput.

This is a training-only target-sampling experiment. It keeps all eight DFM
action horizons, all eight recurrent JEPA predictions, both projector blocks,
all four DFM blocks, the eight-pass searchless inference budget, and the
normalized loss coefficients `0.0/5.76/1.0` for
norm/target-SIGReg/prediction-SIGReg.

## Candidate contract

Relative to the accepted full-model K=2 cosine-warmdown implementation, change
only:

```python
"jepa_target_sample_count": 1
```

The existing sampled-target implementation must retain these semantics:

- On each training update, uniformly sample one horizon from `{0, ..., 7}`
  using the existing target-selection RNG branch shared across the physical
  batch. DFM time and mask RNG streams remain unchanged.
- Encode the current board and only that future board, exactly two BT4 encodes
  per example. Compute DFM predictions and the recurrent free JEPA rollout for
  all eight horizons, then gather the selected prediction for positive JEPA
  loss.
- The size-one positive loss is an unbiased Monte Carlo estimate of the
  uniform eight-horizon mean; do not retune its coefficient.
- Target SIGReg includes the current latent with weight one and the sampled
  future latent with importance weight `8/1`. With the fixed 64-example
  SIGReg sample, its reported effective count remains `576`. Prediction
  SIGReg still consumes all eight predictions and reports count `512`.
- Validation, checkpoint selection, collapse diagnostics, arena inference,
  and any later representation evaluation use every future/prediction
  horizon. No sampled future board is available at inference.
- Sampled target anchors, teacher forcing, EMA targets, target stop-gradient,
  RMS norm matching, and all dormant loss heads remain disabled.

The accepted optimizer schedule remains constant at peak main/BT4 rates
through update 400, cosine-decays over updates 400--1200 to a 10% floor, and
then remains at that floor. Reports and strict checkpoint contracts must make
the size-one sampling and unchanged schedule explicit.

## Frozen control and run configuration

The offline incumbent is K=2 cosine-warmdown v1/update 1261:

- state SHA-256:
  `7bac3c49875ab35c3121c4471b65a40c08ce169a10915d4ebbb55661617cdc1a`;
- two-pool DFM CE `4.5007607210`;
- accuracy `0.1094207764`;
- legal mass `0.6473310636`;
- cached profile `93.2987` examples/s;
- 30-minute throughput `92.8796` examples/s; and
- exact-repeat selected-checkpoint CE separation `0.0014834367`.

Start from the recovered step-265,000 model with a fresh optimizer. Freeze one
NVIDIA A10G with no concurrent workload; physical/evaluation batches `128/64`;
train/data seed 0 with `global_permutation`; peak main/BT4 learning rates
`3e-5/1e-6`; schedule start/decay/floor `400/800/0.1`; normalized loss
coefficients `0.0/5.76/1.0`; V-statistic SIGReg on a fixed 64-example sample;
1,800 seconds of steady-state training after compile/first update; validation
seeds 10,000 and 20,000, each 64 batches of 64 examples; and saves at updates
400, 800, 1200, and terminal with at most four temporary states.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests prove the active override is one, size-one selection is
   deterministic for a fixed key, exactly two boards per example are encoded,
   one positive horizon is active, future importance is eight, target and
   prediction SIGReg counts remain `576/512` at the production shape, and
   evaluation remains full-horizon.
2. A one-update real-checkpoint A10G smoke has finite loss/gradients, full
   architecture depths, two encoded boards per example, eight predictions,
   one sampled target, SIGReg counts `576/512`, no clipping/skipped update, and
   no write outside `/mountpoint/.exp`.
3. A cached 30-update profile must reach at least `97.9636` end-to-end
   examples/s, 5% above the accepted K=2 cosine profile. Below that threshold,
   do not spend the fixed 30-minute budget until the discrepancy is diagnosed.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical seed-10,000 and seed-20,000
full-horizon pools and select minimum mean DFM CE. Acceptance requires:

- two-pool DFM CE below `4.4992772844`, more than the accepted
  `0.0014834367` repeat separation better than the incumbent;
- accuracy at least `0.1074207764`;
- legal mass at least `0.6443310636`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

Separately report the update count, examples, throughput, and the selected
checkpoint's deltas from both accepted K=2 repeats. If the first run passes,
repeat it exactly and require the repeat to beat the cosine incumbent CE while
passing every non-CE gate before the frozen 128-pair arena. A failure gets no
repeat, arena, pass-count sweep, or SAE refit. Retain only a qualifying
selected state; otherwise delete all candidate states after preserving compact
evidence.
