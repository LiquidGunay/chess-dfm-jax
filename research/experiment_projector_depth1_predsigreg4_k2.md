# Experiment 004: stronger prediction SIGReg for one-block projector

Status: preregistered on 2026-07-21 before implementation or candidate GPU
measurement.

## Question and hypothesis

Experiment 003 showed that executing one of two stored JEPA projector blocks
is `6.70%` faster in the cached training profile and leaves policy CE
effectively tied with the K=2 incumbent, but contracts the predicted latent:
mean/minimum effective rank fell to `21.5274/19.5168`, mean/minimum
feature-std p05 to `0.49511/0.47037`, and the prediction/target RMS ratio to
`0.86034`. This happened with the RMS matching loss disabled and normalized
prediction SIGReg coefficient `1.0` enabled.

This experiment asks whether prediction SIGReg alone can prevent that
architecture-specific contraction. Relative to Experiment 003, change only
the prediction SIGReg coefficient from `1.0` to `4.0`. Keep the one-block
projector and every other model, loss, optimizer, data, and evaluation setting
fixed. The primary hypothesis is that the stronger coefficient restores every
latent-health gate and improves full-horizon policy CE beyond the K=2 repeat
envelope. The secondary hypothesis is that the coefficient change retains the
one-block training-speed gain because it adds no new tensor operation.

This is a conditional rescue experiment, not a new claim that `4.0` is the
general K=2 optimum. If it fails, abandon the shallow-projector line rather
than continuing an architecture-specific coefficient sweep.

## Coefficient rationale

At the Experiment 003 terminal seed-10,000 audit, the unweighted scalar terms
were:

| term | raw value | coefficient | weighted contribution |
| --- | ---: | ---: | ---: |
| positive JEPA MSE | 0.28325 | 1.0 | 0.28325 |
| target SIGReg | 0.07327 | 5.76 | 0.42201 |
| prediction SIGReg | 0.08710 | 1.0 | 0.08710 |

The regularizer of the collapsed population was therefore about three to five
times smaller than the other representation terms after training. A
prediction coefficient of `4.0` makes its post-drop contribution about
`0.3484`, between positive JEPA and target SIGReg. The calibration uses the
settled loss rather than initialization because normalized SIGReg falls
quickly early in training. This equalizes the representation auxiliaries, not
the much larger policy CE scalar, and does not claim equality of gradient
norms.

## Candidate contract

Relative to Experiment 003, the sole objective override is:

```python
"jepa_pred_sigreg_coeff": 4.0
```

The complete active candidate is one projector block plus that coefficient.
It must satisfy all of the following:

- Keep `jepa_norm_loss_coeff=0.0`. Log RMS mismatch only as a diagnostic; do
  not restore or partially mix in the norm loss.
- Keep target SIGReg coefficient `5.76`, normalized V-statistic estimator,
  fixed 64-example sampling, and target/prediction valid counts `576/512`.
- Keep two sampled future targets, no sampled-target anchors, all eight
  free-running predictions, and full-horizon validation.
- Execute projector block 0 for every online projector consumer while storing
  the exact two-block source-compatible parameter ABI. Block 1 remains
  checkpoint-only and has exactly zero loss gradient.
- Preserve the exact DFM forward values and RNG streams relative to
  Experiment 003 for identical parameters, batch, and root key. Only the
  scalar loss and gradients contributed by prediction SIGReg may change.
- Keep the source step-265,000 model-only initialization with a fresh
  optimizer. Do not initialize, partially restore, or add a parameter.
- Record the effective `0.0/5.76/4.0` norm/target/prediction coefficients in
  reports and resume contracts. Unsupported legacy-objective combinations
  continue to fail before compilation.

## Frozen run configuration

The policy-quality control remains Experiment 001 K=2 v1/update 800:

- state SHA-256:
  `f42af6bef64c532376e0532d2370b1cf2bd6356d494e08738355359bb506abd2`;
- two-pool DFM CE `4.5054920968`;
- accuracy `0.1095886230`;
- legal mass `0.6455246028`;
- cached profile `92.1149 examples/s`; and
- exact-repeat selected-checkpoint CE gap `0.0022991356`.

Experiment 003 is the conditional latent/speed control: terminal CE
`4.5055253655`, mean/minimum rank `21.5274/19.5168`, mean/minimum p05
`0.49511/0.47037`, RMS ratio `0.86034`, and cached profile `98.2906
examples/s`.

Freeze one NVIDIA A10G with no concurrent GPU workload, physical/evaluation
batches `128/64`, train/data seed 0 with `global_permutation`, main/BT4
learning rates `3e-5/1e-6`, zero warmup, and 1,800 seconds of steady-state
training after compile/first update. Evaluate seeds 10,000 and 20,000 with 64
batches of 64 examples each. Keep eight DFM refinement passes for any strength
measurement. Save updates 400, 800, 1200, and terminal, with at most four
temporary states.

## Correctness and performance gates

Before the 30-minute run:

1. Focused CPU tests prove one-block source ABI parity, zero gradients for
   stored block 1, coefficient/config serialization, exact DFM forward parity,
   and the unchanged normalized SIGReg populations.
2. A real-checkpoint A10G smoke has finite loss and gradients, no clipping or
   skipped update, active projector depth `1/2`, three encoded boards per
   example, eight predictions, and target/prediction counts `576/512`.
3. A cached 30-update profile must remain at least `87.5092 examples/s`, 95%
   of K=2. Report whether it remains within 5% of the matched one-block profile
   `98.2906`; speed alone cannot accept the candidate.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical two validation pools and
select minimum mean DFM CE. Always run the seed-10,000 collapse audit on the
selected checkpoint, even if CE fails, because latent rescue is the explicit
question. Acceptance requires all of the following:

- two-pool DFM CE below `4.5031929612`, more than `0.0022991356` better than
  retained K=2 v1;
- accuracy at least `0.1075886230`;
- legal mass at least `0.6425246028`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

Also report the latent deltas against Experiment 003 so a failed policy trial
still answers whether stronger prediction SIGReg rescued collapse. If the
first run passes, repeat it exactly and require the repeat to beat K=2 CE and
pass every health gate before any frozen 128-pair arena. A failure gets no
repeat, arena, pass-count sweep, SAE refit, or further one-block coefficient
tuning. Retain only a qualifying selected state; otherwise delete every
candidate state after preserving compact evidence.
