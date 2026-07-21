# Experiment 005: three active DFM planner blocks

Status: preregistered on 2026-07-21 before implementation or candidate GPU
measurement.

## Question and hypothesis

The K=2 incumbent executes all four stored DFM transformer blocks in both
training planner calls and in every searchless refinement pass. This
experiment asks whether three active blocks preserve or improve chess policy
quality after fixed-time fine-tuning while reducing planner compute. Unlike
JEPA projector depth, DFM depth affects both training and inference.

The primary hypothesis is that a three-block planner improves full-horizon
held-out DFM cross-entropy beyond the K=2 repeat envelope while preserving
policy, legality, and latent-health gates. The performance hypothesis is that
removing one planner block improves cached training throughput and matched
eight-pass inference throughput, especially at batch 64 where the cached BT4
encoder is less dominant.

This changes planner depth only. It keeps both JEPA projector blocks, two
sampled future targets, prediction SIGReg `1.0`, no RMS matching, and eight
searchless refinement passes. The rejected shallow-projector experiments do
not supply parameters or initialization to this candidate.

## Candidate contract

Relative to Experiment 001 K=2, the sole model override is:

```python
"dfm_active_layers": 3
```

The implementation must satisfy all of the following:

- Continue to allocate, restore, checksum, and serialize the exact
  source-compatible four-layer DFM parameter tree. Execute stored layers 0, 1,
  and 2 in order; stored layer 3 is checkpoint-ABI-only and receives exactly
  zero loss gradient.
- `dfm_active_layers=0` means all configured layers and preserves the exact
  source/K=2 control graph and numerics. Positive values must be integers in
  `[1, dfm_layers]`; unsupported values fail before construction or compile.
- Apply the active depth identically to the noisy DFM training planner, the
  clean planner hidden states that condition JEPA, validation, checkpoint
  selection, collapse diagnostics, checked local inference, and arena play.
  Training and inference must not silently use different depths.
- Preserve diffusion-time and mask RNG streams for the same batch and root
  key. DFM logits, clean hidden states, JEPA predictions, and their gradients
  may change because the planner itself changes; BT4/projector target vectors
  for fixed parameters must not.
- Restore every source model parameter exactly and start a fresh optimizer.
  Do not introduce a random parameter, partial restore, packed checkpoint, or
  optimizer continuation.
- Keep K=2 future-target sampling, no anchors, both active JEPA projector
  blocks, all eight recurrent predictions, target/prediction SIGReg
  populations `576/512`, fixed 64-example normalized V-statistic SIGReg,
  norm/target/prediction coefficients `0.0/5.76/1.0`, and full-horizon
  evaluation.
- Record configured versus active DFM depth, the exact active block indices,
  storage-only inactive blocks, and shared train/eval/inference semantics in
  reports and strict resume contracts.

An accepted state may later be packed into a true three-block checkpoint in a
separate migration. This experiment deliberately keeps the storage ABI fixed
so depth is not confounded with initialization or serialization changes.

## Frozen control and run configuration

The matched policy control remains Experiment 001 K=2 v1/update 800:

- state SHA-256:
  `f42af6bef64c532376e0532d2370b1cf2bd6356d494e08738355359bb506abd2`;
- two-pool DFM CE `4.5054920968`;
- accuracy `0.1095886230`;
- legal mass `0.6455246028`;
- cached training profile `92.1149 examples/s`;
- 30-minute throughput `92.9745 examples/s`; and
- exact-repeat selected-checkpoint CE gap `0.0022991356`.

The candidate starts from the same recovered step-265,000 model weights with
a fresh optimizer. Freeze one NVIDIA A10G and no concurrent GPU workload;
physical/evaluation batches `128/64`; train/data seed 0 with
`global_permutation`; main/BT4 learning rates `3e-5/1e-6`; zero warmup; 1,800
seconds of steady-state training after compile/first update; and validation
seeds 10,000 and 20,000, each 64 batches of 64 examples. Save updates 400,
800, 1200, and terminal with at most four temporary candidate states.

Inference remains deterministic greedy, legality-masked, and fixed at eight
DFM refinement passes. Report matched batch-1 and batch-64 latency using the
same planes, legal mask, warmup count, measured-call count, dtype, and A10G.
The existing full-depth source profile is context only; candidate decisions
must use a newly matched control measurement in the same profiling session.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests prove default full-depth exactness, source-state ABI parity,
   explicit three-layer prefix equivalence, exactly zero loss gradient for
   layer 3, shared training/evaluation/inference depth, unchanged target
   projection and RNG, serialization, validation, and resume semantics.
2. A real-checkpoint A10G smoke has finite loss and gradients, configured/
   active depth `4/3`, three encoded boards per example, eight JEPA
   predictions, SIGReg counts `576/512`, no clipping/skipped update, and no
   write outside `/mountpoint/.exp`.
3. A cached 30-update training profile must reach at least `87.5092`
   examples/s, 95% of K=2. Report the gain or loss against `92.1149`.
4. A matched source-weight inference profile must not regress batch-64
   eight-pass positions/s by more than 5%. The secondary speed hypothesis
   succeeds at a 5% or larger gain. Batch-1 p50/p95 and positions/s are
   reported but are not a hard gate because cached BT4 encoding dominates.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical seed-10,000 and seed-20,000
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

If the first run passes, repeat it exactly and require the repeat to beat K=2
CE while passing every non-CE gate. Then measure candidate checkpoint
inference against the retained K=2 checkpoint and, if still qualified, run the
frozen 128-pair cap-256 incumbent arena. Only the normalized-Elo GSPRT can
promote chess strength. A failure gets no repeat, arena, refinement-pass sweep,
or SAE refit. Retain only a qualifying selected state; otherwise delete all
candidate states after preserving compact evidence.
