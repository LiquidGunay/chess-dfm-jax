# Experiment 001: two sampled future targets

Status: completed and accepted as the offline incumbent on 2026-07-21. The
relative-strength screen is positive in point estimate but inconclusive; this
checkpoint is not Elo-promoted.

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

## Outcome

Commits `afad342` and `c126e70` implement and test the frozen sampling
contract. The one-update A10G smoke encoded exactly three boards per example,
sampled exactly two future horizons without replacement, retained all eight
prediction horizons, and reported target/prediction SIGReg valid counts of
`576/512`. Runtime peak memory was `9,474,431,232` bytes. XLA's aggregate cost
estimate did not expose the expected encoder-work reduction: it reported
`13.5661e12` FLOPs/update versus `13.5223e12` for the control, while runtime
memory was lower. The measured cached profile was therefore the decisive
performance check.

The 30-update profile reached `92.1149` end-to-end examples/s, versus the
frozen `41.35` control rate: `2.23x` throughput and well above the `15%` gate.
The fixed 1,800-second run processed `161,408` examples in `1,261` updates,
sustained `92.9745` end-to-end examples/s, and peaked at `9,576,191,232` bytes
of HBM. No update was skipped, every recorded metric was finite, and no loss
clip activated.

All four retained checkpoints were evaluated on the full eight-horizon seed
10,000 and 20,000 pools. Update 800 won with:

- mean DFM CE `4.5054920968`, improving the matched control by `0.0047771744`
  or `7.50x` the `0.000637` repeat-noise threshold;
- accuracy `0.1095886230`, versus control `0.1081848145`;
- legal mass `0.6455246028`, versus control `0.6426193411`;
- mean/minimum prediction effective rank `31.0497/29.1594`;
- mean/minimum prediction feature-standard-deviation fifth percentile
  `0.65010/0.63620`;
- mean target RMS `0.92656` and prediction/target RMS ratio `0.97114`; and
- positive JEPA MSE below zero and action-shuffled prediction MSE at every
  horizon. Its mean JEPA/identity MSE ratio is `0.145870`, the value recorded
  in `research/results.tsv`.

The eight-pass 128-pair cap-256 development arena against corrected
v2/update 400 scored `0.505859375` for the candidate, with pentanomial counts
`[0,1,123,4,0]`, descriptive logistic Elo `+4.0717`, and pair-aware 95%
interval `[-80.7651,+89.4067]`. Ten of 256 games were cap draws. One
candidate-side `no_representable_move` loss came from the already documented
legacy promotion-codec limitation; the incumbent had no fault. This screen is
consistent with the offline gain but does not resolve chess strength.

Update 800 is accepted as the new offline incumbent. Its state SHA-256 is
`f42af6bef64c532376e0532d2370b1cf2bd6356d494e08738355359bb506abd2`.
Updates 400, 1200, and 1261 were deleted after selection, and the exact
retained state/manifest pair is enforced by `research/storage_retention.json`.
The pass-count and SAE-refit studies remain deferred until a larger
offline effect makes the strength evidence less ambiguous.

### Exact v2 repeat

An exact second 1,800-second run from the same source model, seed, data order,
optimizer, and objective processed `160,128` examples in `1,251` updates at
`92.1929` steady end-to-end examples/s, with peak HBM `9,583,339,776` bytes.
Its state differs from v1, confirming that the GPU path is not bitwise
repeatable. The two-pool scan independently selected update 800:

- DFM CE `4.5077912323`, improving the frozen control by `0.0024780389` or
  `3.89x` the acceptance threshold;
- accuracy `0.1079254150` and legal mass `0.6440788107`, both inside their
  gates;
- mean/minimum prediction effective rank `30.9897/29.0984`;
- mean/minimum prediction feature-std p05 `0.64930/0.63585`; and
- target RMS `0.92547` with prediction/target RMS ratio `0.97171`.

Prediction again beats zero and action-shuffled baselines at all eight
horizons. Thus the direction of the offline improvement and the throughput
gain replicate independently. The v1/v2 selected CE separation is
`0.0022991356`, however, which is `3.61x` the old `0.000637` corrected-control
repeat envelope. Treat K=2 as independently accepted twice but not tightly
repeat-stable in effect size. All four v2 states were deleted after the audit;
the compact run, selection, and collapse reports remain, while the better
arena-tested v1/update-800 state stays the sole retained K=2 checkpoint.
