# Experiment 003: one active JEPA projector block

Status: preregistered on 2026-07-21 before implementation or candidate GPU
measurement.

## Question and hypothesis

The K=2 offline incumbent executes both 2,048-wide transformer blocks in the
BT4-token-to-JEPA-state projector for the current board and each of two sampled
future targets. This experiment asks whether one active projector block is a
better fixed-time auxiliary representation: cheaper to train, less prone to
overfitting the short horizon objective, and still expressive enough to
regularize the shared BT4 trunk and recurrent JEPA transition.

The primary hypothesis is that one active block improves full-horizon held-out
DFM cross-entropy beyond the observed K=2 repeat gap while preserving every
policy, legality, and latent-health gate. The secondary hypothesis is at least
a 10% cached-profile throughput gain, from `92.1149` to
`101.3264 examples/s`. Failure to reach that speed target is recorded but does
not by itself reject a policy-quality winner; a regression below 95% of the
matched K=2 rate does block the 30-minute run.

This changes only the JEPA auxiliary projector. The DFM action path does not
execute this projector at searchless inference, so no move-latency improvement
is claimed here. Inference remains fixed at eight DFM refinement passes.

## Candidate contract

Relative to the unanchored K=2 incumbent, the sole new model/objective override
is:

```python
"jepa_projector_active_layers": 1
```

The implementation must satisfy all of the following:

- Continue to allocate and checksum the source-compatible two-block projector
  parameter tree, restore every model parameter exactly from the pinned
  step-265,000 checkpoint, and start a fresh optimizer. This experiment does
  not introduce random architecture-specific parameters or partial restore.
- Execute only stored projector block 0 after the input projection, CLS token,
  and positional embedding. Stored block 1 is checkpoint-ABI-only: it is not
  read by the forward objective and its loss gradient is exactly zero.
- Apply the same active-depth rule to every online projector consumer:
  current-state encoding, sampled/full future targets, full-horizon
  validation, collapse diagnostics, and any value/WDL head input. Do not let
  training and evaluation use different projector depths.
- `jepa_projector_active_layers=0` means all configured projector blocks and
  must preserve the exact K=2 control graph and numerics. Positive values must
  be integers in `[1, projector_layers]`; unsupported values fail before model
  construction or compilation.
- Keep sampled-target count 2, no rollout anchors, all eight recurrent
  predictions, full-horizon evaluation, target/prediction SIGReg populations,
  fixed 64-example SIGReg sampling, and coefficients `5.76/1.0` unchanged.
- Keep RMS-matching coefficient `0.0`; its logged value remains diagnostic
  only. Do not add a norm loss to compensate for a changed projector scale.
- Preserve DFM diffusion/mask RNG, logits, CE, and legality exactly for the
  same parameters, batch, and root key. Only JEPA/projector values and their
  shared-backbone gradients may differ.
- Reports and resume contracts record configured versus active projector
  depth, that inactive layers retain storage ABI only, and that the rule is
  identical in training and evaluation.

An accepted result may later be packed into a true one-block checkpoint in a
separate migration. This experiment deliberately keeps storage size fixed so
initialization and optimizer-policy changes do not confound active depth.

## Frozen control and run configuration

The matched control remains Experiment 001 K=2 v1/update 800:

- state SHA-256:
  `f42af6bef64c532376e0532d2370b1cf2bd6356d494e08738355359bb506abd2`;
- two-pool DFM CE `4.5054920968`;
- accuracy `0.1095886230`;
- legal mass `0.6455246028`;
- cached profile `92.1149 examples/s`;
- 30-minute throughput `92.9745 examples/s`; and
- exact-repeat selected-checkpoint CE gap `0.0022991356`.

The candidate starts from the same recovered step-265,000 model weights with a
fresh optimizer. Freeze:

- one NVIDIA A10G and no concurrent GPU workload;
- physical/evaluation batches `128/64`;
- train/data seed 0 with `global_permutation`;
- main/BT4 learning rates `3e-5/1e-6`, constant with zero warmup;
- K=2 future-target sampling, no sampled-target anchors, no RMS norm loss,
  target SIGReg `5.76`, prediction SIGReg `1.0`, V-statistic, and fixed
  64-example SIGReg sampling;
- 1,800 seconds of steady-state training after compile/first update;
- validation seeds 10,000 and 20,000, each 64 batches of 64 examples;
- eight DFM refinement passes for any strength measurement; and
- saves at updates 400, 800, 1200, and terminal with at most four temporary
  candidate states.

After selection, retain only a qualifying selected state. Delete all candidate
states after recording compact evidence if the experiment is rejected.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests prove default-path exactness, exact source-state ABI parity,
   one-block output equivalence to explicit block-0 execution, exact-zero
   loss gradients for block 1, shared training/evaluation depth, unchanged DFM
   values/RNG, K=2 encoder/prediction counts, serialization, validation, and
   resume semantics.
2. A one-update real-checkpoint A10G smoke has finite loss/gradients, active
   depth `1/2`, three encoded boards per example, eight recurrent predictions,
   target/prediction SIGReg counts `576/512`, and no write outside the
   workspace.
3. A cached 30-update profile must reach at least `87.5092 examples/s` (95% of
   K=2). The speed hypothesis succeeds at or above `101.3264 examples/s`.

## Checkpoint selection and decision

Evaluate all saved checkpoints on the identical seed-10,000 and seed-20,000
full-horizon pools and select minimum mean DFM CE. Acceptance requires:

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

If the first run passes, repeat it exactly and require the repeat to beat the
K=2 control CE while passing every non-CE health gate. A passing first run may
then receive the frozen 128-pair cap-256 incumbent screen; only the normalized-
Elo GSPRT can promote strength. A failure gets no repeat, arena, pass-count
sweep, or SAE refit.
