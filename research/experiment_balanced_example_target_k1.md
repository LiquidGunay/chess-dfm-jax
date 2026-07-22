# Experiment 009: balanced per-example K=1 targets

Status: completed and accepted as the offline incumbent on 2026-07-22. The
frozen 128-pair arena was positive but inconclusive; this is not an Elo
promotion.

## Question and hypothesis

Experiment 008 shows that K=1 target sampling is computationally attractive:
it raises cached throughput by `24.88%` and processes `40,320` more examples
than K=2 in 30 minutes. Its shared-horizon sampler chooses one future horizon
for the entire physical batch, however, so every update estimates the
eight-horizon JEPA objective from only one horizon. The terminal checkpoint
improves the accepted incumbent by `0.0008786097` CE but does not exceed
repeat noise.

This experiment tests the specific variance mechanism. Each example still
encodes exactly one future board, but horizon assignments are balanced across
the physical batch so every horizon receives equal supervision on every
batch-128 update. The primary hypothesis is that lower horizon-estimator
variance retains at least 95% of shared K=1 throughput and improves
full-horizon policy CE beyond the accepted K=2 repeat envelope. The failure
mode is that shared sampling provides useful coherent per-update tasks, or
that mixing target horizons inside one encoder batch changes gradients without
improving generalization.

This is a target-estimator experiment only. It preserves the accepted 10%
cosine schedule, both projector blocks, all four DFM blocks, all eight DFM and
JEPA prediction horizons, eight-pass searchless inference, and normalized
loss coefficients `0.0/5.76/1.0` for
norm/target-SIGReg/prediction-SIGReg.

## Candidate contract

Add a default-off target-sampling unit and activate exactly:

```python
"jepa_target_sample_count": 1,
"jepa_target_sampling_unit": "example_balanced",
```

The historical/default value is `"batch_shared"`; K=2 and all existing
checkpoints must retain their exact behavior. The candidate semantics are:

- Derive target assignments from the existing target-selection RNG branch,
  leaving DFM diffusion-time/mask RNG and the remaining SIGReg RNG unchanged.
- Construct one horizon label per physical example by randomly permuting a
  tiled `{0, ..., 7}` label vector. Horizon counts differ by at most one for
  arbitrary batch sizes and are exactly 16 each for batch 128.
- Gather and encode one future board per example. Together with the current
  board, training still performs exactly two BT4 encodes per example.
- Run the recurrent JEPA predictor and DFM objective over all eight horizons,
  then gather each example's assigned prediction and target for positive JEPA
  MSE. The balanced sample mean is the equal-horizon population mean at batch
  128; do not retune its coefficient.
- Target SIGReg uses each sampled future with importance weight eight plus the
  current state with weight one. The fixed 64-example SIGReg subsample must
  continue to report effective target/prediction counts `576/512`.
- Prediction SIGReg remains attached to all eight free-rollout predictions.
  RMS norm matching remains disabled, so prediction SIGReg continues to
  regularize `z_pred`.
- Training metrics scatter positive losses back to their assigned horizons
  and report the per-horizon sample mask/count semantics. Evaluation,
  checkpoint selection, collapse diagnostics, and inference remain full
  horizon and must not sample future states.
- Balanced per-example sampling initially requires `sample_count=1`, online
  targets, no teacher forcing, no sampled-target anchors, and no target
  variance hinge. Unsupported combinations fail closed.
- Reports and resume contracts identify the unit, balance rule, per-example
  count, importance weight, RNG derivation, and unchanged evaluation scope.

## Frozen controls and run configuration

The acceptance control is K=2 cosine-warmdown v1/update 1261:

- state SHA-256:
  `7bac3c49875ab35c3121c4471b65a40c08ce169a10915d4ebbb55661617cdc1a`;
- two-pool DFM CE `4.5007607210`;
- accuracy `0.1094207764`;
- legal mass `0.6473310636`;
- cached profile `93.2987` examples/s; and
- accepted-repeat CE separation `0.0014834367`.

The mechanistic compute control is shared-horizon K=1: cached profile
`116.5113` examples/s, 30-minute throughput `116.5422`, and selected terminal
CE/accuracy/legal mass `4.4998821113/0.1096038818/0.6480329307`.

Start from the recovered step-265,000 model with a fresh optimizer. Freeze one
NVIDIA A10G and no concurrent workload; physical/evaluation batches `128/64`;
train/data seed 0 with `global_permutation`; peak main/BT4 rates
`3e-5/1e-6`; schedule start/decay/floor `400/800/0.1`; normalized loss
coefficients `0.0/5.76/1.0`; V-statistic SIGReg on a fixed 64-example sample;
1,800 seconds of steady-state training after compile/first update; validation
seeds 10,000 and 20,000, each 64 batches of 64 examples; and saves at updates
400, 800, 1200, and terminal with at most four temporary states.

## Correctness and performance gates

Before the fixed run:

1. CPU tests prove default batch-shared exactness, deterministic balanced
   assignments for a fixed key, count spread at most one, all horizons present
   at batch 128, two encoded boards/example, one target/example, unchanged DFM
   time/mask outputs, correct per-horizon scatters, full prediction coverage,
   preserved SIGReg mass, full-horizon evaluation, and fail-closed unsupported
   combinations.
2. A one-update real-checkpoint A10G smoke has finite loss/gradients, full
   architecture depths, exactly 16 examples assigned to each horizon, two
   encoded boards/example, eight predictions, SIGReg counts `576/512`, no
   clipping/skipped update, and no write outside `/mountpoint/.exp`.
3. A cached 30-update profile must reach at least `110.6858` end-to-end
   examples/s, 95% of shared K=1. Below that threshold, do not spend the
   30-minute budget until the discrepancy is diagnosed.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical seed-10,000 and seed-20,000
full-horizon pools and select minimum mean DFM CE. Acceptance requires:

- two-pool DFM CE below `4.4992772844`, more than the accepted
  `0.0014834367` repeat separation better than the K=2 incumbent;
- accuracy at least `0.1074207764`;
- legal mass at least `0.6443310636`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

Separately compare its complete learning curve with shared K=1 and both K=2
warmdown runs. If the first run passes, repeat it exactly and require the
repeat to beat the K=2 incumbent CE while passing every non-CE gate before the
frozen 128-pair arena. A failure gets no repeat, arena, pass-count sweep, or
SAE refit. Retain only a qualifying selected state; otherwise delete every
candidate state after preserving compact evidence.

## Outcome

The implementation and all pre-run gates passed. Focused CPU coverage contains
72 passing tests. The real-checkpoint smoke used the full two-block projector
and four-block DFM, assigned exactly 16 examples to each horizon, encoded two
BT4 boards per example, produced all eight predictions, reported target and
prediction SIGReg counts `576/512`, and completed an unclipped optimizer
update. Peak smoke HBM was `9,462,135,296` bytes.

The cached 30-update profile reached `117.3799` end-to-end examples/s and
`149.0972` device examples/s at peak HBM `9,453,743,360` bytes. It clears the
frozen `110.6858` gate by `6.05%` and is slightly faster than the
`116.5113` shared-horizon K=1 compute control.

The primary 30-minute run compiled in `17.5010` seconds and processed
`202,368` examples in 1,581 updates at `117.0611` steady-state examples/s.
The frozen two-pool checkpoint scan was:

| Update | DFM CE | Accuracy | Legal mass |
|---:|---:|---:|---:|
| 400 | 4.5102302 | 0.1084442 | 0.6449483 |
| 800 | 4.5044888 | 0.1090546 | 0.6460078 |
| 1,200 | 4.5045125 | 0.1098633 | 0.6469114 |
| 1,581 | **4.4979400** | **0.1100922** | **0.6497235** |

The terminal checkpoint beats the K=2 cosine-warmdown incumbent by
`0.0028207470` CE and clears the repeat-noise-aware ceiling by `0.0013373103`.
Its prediction effective-rank mean/minimum is `31.0093/29.1222`, feature-std
p05 mean/minimum is `0.65017/0.63619`, mean target RMS is `0.92492`, and the
prediction/target RMS ratio is `0.97215`. Positive prediction beats zero and
action-shuffled controls at every horizon.

The exact repeat processed `201,600` examples in 1,575 updates at `116.4586`
examples/s. Its selected terminal checkpoint reaches CE `4.4969695`, accuracy
`0.1105804`, and legal mass `0.6477513`; it again passes every policy and
latent gate. The two selected CEs differ by `0.0009704642`, inside the accepted
K=2 repeat separation, and both independently beat that incumbent.

The preregistered primary checkpoint then scored `0.51171875` over 128
color-reversed development pairs against K=2 cosine warmdown: 8 wins, 246
draws, and 2 losses across 256 games, pentanomial `[0, 2, 118, 8, 0]`, with
six cap draws and zero faults. Descriptive logistic Elo is `+8.14`, with a
pair-aware 95% interval of `[-76.48, +93.77]`. Mean policy-call time was
`42.05 ms` for the candidate and `42.72 ms` for K=2. This is a positive but
inconclusive model-pool-relative screen, not an absolute-Elo claim or
promotion.

Accept primary update 1,581 as the new offline incumbent. Its state SHA-256 is
`26c3621d1f9c35aa1401ef5e2e9d35ffdf59607d0c4369730cc6dd207b65c7c7`.
Retain that one candidate state and compact evidence only; all other primary
and repeat states were deleted.
