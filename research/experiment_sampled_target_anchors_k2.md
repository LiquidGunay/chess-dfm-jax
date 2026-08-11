# Experiment 002: sampled-target rollout anchors

Status: preregistered on 2026-07-21 before implementation or candidate GPU
measurement.

## Question and hypothesis

The current offline incumbent trains the recurrent JEPA predictor against two
uniformly sampled future targets per update, but every recurrent transition is
free-running. This experiment asks whether reusing those same encoded targets
as sparse teacher-forcing anchors improves the learned action policy by
reducing compounding latent rollout error, without adding BT4 encoder work.

The testable hypothesis is that sparse anchors improve frozen full-horizon
validation DFM cross-entropy by more than the observed `0.0022991356` gap
between the two exact K=2 repeats, while preserving policy accuracy, legal
mass, latent health, and at least 95% of the matched K=2 profile throughput.

This is a training-only recurrence intervention. It does not change the two
sampled positive-target horizons, target or prediction SIGReg populations,
loss coefficients, eight-pass searchless inference budget, model dimensions,
optimizer, data, or validation and arena protocols.

## Candidate contract

Relative to Experiment 001, the sole new model/objective override is:

```python
"jepa_sampled_target_anchors": True
```

The implementation must satisfy all of the following:

- Continue to sample one shared, sorted, size-two subset of the eight future
  horizons without replacement from the existing SIGReg RNG branch. Do not
  add or consume another random stream.
- Continue to encode exactly the current board and those two sampled future
  boards per training example.
- Produce the prediction at every horizon from the recurrent state that would
  ordinarily enter that transition. Score and retain that prediction before
  applying any anchor.
- After producing prediction `h`, replace the recurrent carry used to predict
  `h + 1` with the encoded target latent for `h` exactly when `h` was sampled
  and that example's current/future validity masks are true. Otherwise carry
  the prediction. A sampled horizon 8 target is still a positive target but is
  inert as an anchor because there is no ninth transition.
- Use the attached online target latent, matching the legacy teacher-forcing
  gradient semantics. Downstream losses may therefore backpropagate through
  the anchor into the online projector and BT4 trunk. EMA targets remain
  unsupported.
- Positive JEPA loss remains restricted to the two sampled, pre-anchor
  predictions. Prediction SIGReg remains over all eight pre-anchor prediction
  outputs. Target SIGReg and its future importance weight remain unchanged.
- DFM diffusion-time and mask RNG streams, DFM logits, DFM CE, and legality
  values must be exactly unchanged for the same parameters, batch, and root
  key; only gradients through the JEPA branch may differ.
- Evaluation, checkpoint selection, collapse diagnostics, inference, and arena
  play must always use an unanchored eight-horizon free rollout. No future
  board or future latent is available at inference.
- The disabled setting must retain the exact Experiment 001 path. Reports and
  resume contracts must record the training-only anchor timing, validity,
  target attachment, and free-running evaluation semantics.
- The enabled setting requires normalized training with a strict sampled
  target count in `(0, horizon)`, online targets, and zero legacy full-target
  teacher-forcing steps; unsupported combinations fail before compilation.

## Frozen control and run configuration

The matched control is Experiment 001 K=2 v1/update 800:

- state SHA-256:
  `f42af6bef64c532376e0532d2370b1cf2bd6356d494e08738355359bb506abd2`;
- two-pool mean DFM CE on seeds 10,000 and 20,000: `4.5054920968`;
- two-pool accuracy: `0.1095886230`;
- two-pool legal mass: `0.6455246028`;
- steady 30-minute end-to-end throughput: `92.9745 examples/s`;
- cached 30-step profile throughput: `92.1149 examples/s`; and
- exact-repeat selected-checkpoint CE gap: `0.0022991356`.

The candidate starts from the same recovered step-265,000 model weights with a
fresh optimizer, not from the retained K=2 checkpoint. Freeze:

- one NVIDIA A10G with no concurrent GPU workload;
- physical batch 128 and validation batch 64;
- train/data seed 0 with `global_permutation`;
- main/BT4 learning rates `3e-5` / `1e-6`, constant with zero warmup;
- target sample count 2, no RMS norm loss, target SIGReg `5.76`, prediction
  SIGReg `1.0`, and a fixed 64-example SIGReg sample;
- 1,800 seconds of steady-state training after compile/first update;
- validation seeds 10,000 and 20,000, each 64 batches of 64 examples;
- eight DFM refinement passes for strength measurements; and
- checkpoint saves every 400 updates plus terminal, retaining at most four
  states until selection.

After selection, retain only a qualifying selected state. If rejected, delete
all candidate states after preserving compact reports, metrics, and the
decision.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests prove that pre-anchor predictions through the first sampled
   horizon equal free rollout, only later predictions can change, invalid
   targets do not anchor, final-horizon anchors are inert, the disabled path is
   unchanged, target sampling/RNG and three-board encoding are unchanged, and
   full-horizon evaluation is free-running.
2. A one-update real-checkpoint A10G smoke has finite loss and gradients,
   exactly two sampled targets, all eight prediction outputs, and no write
   outside `/mountpoint/.exp`.
3. A cached short profile must reach at least `87.5092 examples/s`, 95% of the
   matched `92.1149 examples/s` profile. Below that threshold, diagnose before
   spending the 30-minute budget.

## Checkpoint selection and decision

Evaluate every saved candidate checkpoint on the identical seed-10,000 and
seed-20,000 full-horizon pools and select minimum mean DFM CE. Candidate
acceptance requires all of the following:

- two-pool DFM CE below `4.5031929612`, an improvement greater than the full
  `0.0022991356` K=2 repeat gap relative to the better retained control;
- accuracy at least `0.1075886230`;
- legal mass at least `0.6425246028`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum-horizon prediction effective rank at least `30.0/28.5`;
- mean/minimum-horizon prediction feature-standard-deviation fifth percentile
  at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

If the first run passes, repeat the exact candidate once because the A10G path
is not bitwise deterministic. The direction must independently pass all
non-CE health gates and beat the K=2 control CE; report repeat dispersion
explicitly. A passing first run may proceed to the frozen 128-pair cap-256
relative-strength screen against the retained K=2 offline incumbent, but only
the preregistered normalized-Elo GSPRT can promote chess strength. Do not run a
pass-count sweep or SAE refit from this experiment unless strength is later
resolved.

## Outcome

Commits `4a74004` and `97285fb` preregister and implement the sparse-anchor
contract. Ten focused target-sampling/anchor tests pass, and the full touched
CPU set contributed 63 passing tests. The one-update A10G smoke encoded three
boards per example, retained eight predictions, reported target/prediction
SIGReg counts `576/512`, and used the sampled horizons as downstream anchors
only after their predictions. Explicit first compile took `146.55` seconds
and peak HBM was `9,474,441,728` bytes.

The cached 30-update profile reached `93.4206` end-to-end examples/s, above
both the `87.5092` gate and the matched K=2 rate of `92.1149`. The fixed
1,800-second run completed 1,253 updates and 160,384 examples at `92.6631`
steady end-to-end examples/s, with peak HBM `9,576,349,184` bytes and mean GPU
utilization `72.81%`. All recorded values remained finite and no update was
skipped.

The full-horizon two-seed checkpoint scan produced:

| update | mean DFM CE | accuracy | legal mass |
| ---: | ---: | ---: | ---: |
| 400 | 4.5121365078 | 0.1076354980 | 0.6425860464 |
| 800 | 4.5095584393 | 0.1081085205 | 0.6443018941 |
| 1200 | 4.5244152509 | 0.1069793701 | 0.6373525830 |
| 1253 | **4.5064137187** | **0.1091308594** | **0.6438060440** |

The terminal checkpoint wins within the candidate run, but is `0.0009216219`
worse than retained K=2 v1 (`4.5054920968`) and `0.0032207575` above the
preregistered acceptance ceiling (`4.5031929612`). It therefore fails the
primary gate. The result lies between the two K=2 repeat outcomes, so the
honest interpretation is no resolved policy improvement rather than strong
evidence of a large regression.

The already-computed seed-10,000 free-rollout latent audit has mean/minimum
effective rank `30.8915/28.9334`, mean target RMS `0.90344`, and
prediction/target RMS ratio `0.97554`. Prediction MSE beats zero and
action-shuffled controls at every horizon. Mean feature-standard-deviation
fifth percentile is `0.63736`, but the final-horizon minimum is `0.60744`,
slightly below the frozen `0.61` floor. Thus the candidate also fails one
latent-tail gate even though it does not exhibit broad rank or RMS collapse.

The experiment is rejected. It is not repeated, arena-tested, or promoted.
All four 1.85 GB candidate states were deleted; compact manifests, metrics,
profile telemetry, the scan, and this decision remain. The anchor capability
stays available default-off, while the active experiment surface returns to
the unanchored K=2 incumbent. Future experiments continue with no RMS-matching
loss and prediction SIGReg coefficient `1.0`; changing that coefficient is a
separate experiment.
