# Experiment 001: two sampled future targets

Status: preregistered on 2026-07-21, before implementation or candidate
training.

## Question and hypothesis

The corrected baseline encodes the current board and all eight future boards
through the trainable BT4 trunk on every update. The first post-baseline
experiment asks whether uniformly sampling two future target horizons per
training update can preserve useful JEPA supervision while substantially
increasing the number of updates and examples processed in a fixed 30-minute
A10G budget.

The testable hypothesis is:

1. reducing future BT4 target encodes from eight to two improves steady-state
   end-to-end throughput by at least 15% relative to the corrected v2 rate of
   `41.35 examples/s`; and
2. the stochastic target objective plus additional fixed-time updates produces
   a checkpoint whose frozen full-horizon validation DFM cross-entropy beats
   the corrected repeat noise without crossing a policy, legality, or latent
   health gate.

This is a training-efficiency and objective-sampling experiment. It does not
change the eight-pass searchless inference budget, DFM action horizon, data,
model width/depth, optimizer, learning rates, or loss coefficients.

## Candidate contract

The sole model/objective override is:

```python
"jepa_target_sample_count": 2
```

The implementation must satisfy all of the following:

- Sampling is training-only. Evaluation, checkpoint selection, latent
  diagnostics, and arena inference retain all eight future horizons.
- On every training update, choose one uniformly random size-two subset of
  `{0, ..., 7}` without replacement, shared across the physical batch and
  sorted before gathering. Derive its RNG from the existing SIGReg RNG branch
  so the DFM diffusion-time and masking RNG streams remain unchanged.
- Encode the current board and only the two selected future boards. Run the
  recurrent JEPA predictor and the DFM action objective over all eight
  horizons, then gather the two corresponding predictions for positive JEPA
  loss.
- The positive JEPA horizon mean uses the same coefficient `1.0`. Uniform
  target selection is its Monte Carlo estimator; there is no coefficient
  retuning in this experiment.
- Target SIGReg uses the current state and the selected future targets. Each
  selected future has importance weight `8 / 2`, while the current state has
  weight one, preserving the full target-mixture mass and reported valid count.
  Prediction SIGReg continues to use predictions at all eight horizons.
- The fixed SIGReg physical-example sample remains 64, shared between target
  and prediction statistics. Target/prediction coefficients remain `5.76` and
  `1.0`, respectively; estimator remains the V-statistic.
- Sampling is unsupported with EMA targets or teacher forcing in this first
  implementation and must fail closed if either is enabled.
- `jepa_target_sample_count=0` retains the exact full-horizon control path.
  Existing full-path tests must pass without changed tolerances.
- Reports and checkpoint contracts record the sampling unit, RNG derivation,
  replacement rule, importance weights, and the fact that validation is full
  horizon.

## Frozen control and run configuration

The control is corrected no-norm v2/update 400:

- state SHA-256:
  `cbeb1bafa74d0983ae4c0cad2a33c405c3507b2428c5bcebbe4fb7f8df44aa61`;
- two-pool mean DFM CE on seeds 10,000 and 20,000: `4.5102692712`;
- two-pool accuracy: `0.1081848145`;
- two-pool legal mass: `0.6426193411`;
- four-pool mean DFM CE: `4.529214`;
- repeated selected-checkpoint four-pool CE separation: `0.000637`; and
- steady-state end-to-end throughput: `41.35 examples/s`.

The candidate starts from the same recovered step-265,000 model weights with a
fresh optimizer, rather than continuing from corrected v2. Freeze:

- one NVIDIA A10G and no concurrent GPU workload;
- physical batch 128 and validation batch 64;
- train seed/data seed 0 with `global_permutation`;
- main/BT4 learning rates `3e-5` / `1e-6`, constant with zero warmup;
- no RMS norm loss, target SIGReg `5.76`, prediction SIGReg `1.0`, fixed
  64-example SIGReg sample;
- 1,800 seconds of steady-state training after compile/first update;
- validation seeds 10,000 and 20,000, 64 batches of 64 examples each;
- eight DFM refinement passes for all strength measurements; and
- save every 400 updates, save the terminal state, and retain at most four
  candidate states during selection.

After selection, retain only the selected state if the candidate is accepted.
If rejected, retain no candidate state. Keep compact reports, metrics, and the
decision either way.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests prove deterministic without-replacement selection, full-path
   compatibility, two-target shapes/scatters, preserved target-mixture valid
   mass, full prediction SIGReg coverage, and full-horizon evaluation.
2. A one-update real-checkpoint A10G smoke has finite loss/gradients, exactly
   two sampled target horizons, no write outside the workspace, and lower
   compiler-estimated BT4 work/memory than the control graph.
3. A short cached A10G profile must show at least 15% higher end-to-end
   throughput than `41.35 examples/s`. Below that threshold, do not spend the
   30-minute candidate budget until the implementation is diagnosed.

## Checkpoint selection and decision

Evaluate every retained candidate checkpoint on the identical seed-10,000 and
seed-20,000 full-horizon pools. Select minimum mean DFM CE. If checkpoints or
the control are within `0.000637`, evaluate every tied item on the already
predeclared seeds 30,000 and 40,000 and select on the four-pool mean. Do not add
another pool after inspecting candidate-only results.

Advance the selected candidate as the new offline incumbent only when all are
true:

- full-horizon mean DFM CE improves over the matched control by more than
  `0.000637` after any required four-pool tie-break;
- accuracy is no more than `0.002` below the matched control;
- legal mass is no more than `0.003` below the matched control;
- every metric and gradient is finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean prediction effective rank is at least `30.0`, and the minimum-horizon
  effective rank is at least `28.5`;
- mean/minimum-horizon prediction feature-standard-deviation fifth percentile
  are at least `0.63` / `0.61`;
- mean target RMS is at least `0.90`; and
- the prediction/target RMS ratio lies in `[0.94, 1.02]`.

Throughput success alone permits keeping the sampled-target implementation as
a default-off research option, but not promoting its checkpoint. An offline
winner proceeds to the frozen relative-strength screen before it is called a
stronger model. Any pass-count sweep remains deferred until then.
