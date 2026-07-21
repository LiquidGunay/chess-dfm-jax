# Experiment 006: fixed-time cosine learning-rate warmdown

Status: preregistered on 2026-07-21 before implementation or candidate GPU
measurement.

## Question and hypothesis

Both exact K=2 runs select update 800 under the constant main/BT4 learning
rates `3e-5/1e-6`. Their update-1200 checkpoints regress sharply and their
terminal checkpoints remain worse than update 800:

| run | update 400 CE | update 800 CE | update 1200 CE | terminal CE |
| --- | ---: | ---: | ---: | ---: |
| K=2 v1 | 4.5099766254 | **4.5054920968** | 4.5286737829 | 4.5071463250 |
| K=2 v2 | 4.5138616990 | **4.5077912323** | 4.5376210772 | 4.5146793593 |

This experiment asks whether late fixed-time degradation is reduced by
warming both learning rates down after the reproducible early-improvement
phase. The primary hypothesis is that a single preregistered cosine schedule
improves selected full-horizon CE beyond the K=2 repeat envelope while
preserving policy, legality, latent health, and throughput. The secondary
hypothesis is that update 1200 no longer exhibits the large policy/legal
regression seen in both constant-rate runs.

This is an optimizer-schedule experiment only. It restores the complete K=2
architecture: both JEPA projector blocks, all four DFM blocks, two sampled
future targets, eight recurrent predictions, and eight searchless refinement
passes.

## Candidate contract

Relative to Experiment 001 K=2, the sole coherent schedule override is:

```python
"lr_decay_start_steps": 400
"lr_decay_steps": 800
"lr_min_ratio": 0.1
```

Apply it identically as a multiplicative schedule to the main and BT4 peak
rates:

- optimizer updates 0 through 399: ratio `1.0`;
- updates 400 through 1199: Optax cosine decay from `1.0` to `0.1` over 800
  schedule steps;
- update 800: ratio `0.55` exactly;
- update 1200 and later: ratio `0.1`;
- no warmup (`lr_warmup_steps=0`).

The implementation must satisfy all of the following:

- Keep the existing constant-schedule path bitwise exact when
  `lr_decay_steps=0`, with `lr_decay_start_steps=0` and `lr_min_ratio=1.0`.
- Require integer non-negative start/decay steps and finite minimum ratio in
  `(0, 1]`. A positive start or non-unit minimum is inert and rejected when
  decay is disabled. The first implementation rejects simultaneous warmup and
  decay rather than guessing their composition.
- Use one optimizer update counter for both schedules. The BT4/main ratio
  remains exactly `1e-6 / 3e-5`; weight decay, Muon/Adam partitioning,
  clipping, and fresh optimizer initialization are unchanged.
- Restore every source model parameter exactly and start from a fresh
  optimizer. The schedule must not change model or optimizer-state ABI and
  must be bound into config serialization and strict resume contracts.
- Keep full four-layer DFM and two-layer JEPA projector execution, K=2 target
  sampling, no anchors, target/prediction SIGReg counts `576/512`, fixed
  64-example normalized V-statistic SIGReg, and norm/target/prediction
  coefficients `0.0/5.76/1.0`.
- Preserve every model forward value, RNG stream, loss scalar, and gradient at
  optimizer update 0 relative to K=2. Only post-gradient parameter updates may
  diverge according to the frozen schedule.
- Record peak rates, schedule family, boundaries, floor ratio, and exact
  boundary values in reports and resume contracts.

## Frozen control and run configuration

The matched control remains K=2 v1/update 800:

- state SHA-256:
  `f42af6bef64c532376e0532d2370b1cf2bd6356d494e08738355359bb506abd2`;
- two-pool DFM CE `4.5054920968`;
- accuracy `0.1095886230`;
- legal mass `0.6455246028`;
- cached profile `92.1149 examples/s`;
- 30-minute throughput `92.9745 examples/s`; and
- exact-repeat selected-checkpoint CE gap `0.0022991356`.

The candidate starts from the recovered step-265,000 model with a fresh
optimizer. Freeze one NVIDIA A10G and no concurrent workload;
physical/evaluation batches `128/64`; train/data seed 0 with
`global_permutation`; peak main/BT4 rates `3e-5/1e-6`; 1,800 seconds of
steady-state training after compile/first update; validation seeds 10,000 and
20,000, each 64 batches of 64 examples; and saves at updates 400, 800, 1200,
and terminal with at most four temporary states.

The update-based schedule is deliberately independent of measured wall time.
If hardware throughput changes enough that terminal occurs before update 1200,
the run is invalid for this hypothesis and must be diagnosed rather than
silently redefining the schedule.

## Correctness and performance gates

Before the 30-minute run:

1. CPU tests prove exact schedule ratios at updates 0, 399, 400, 800, 1199,
   1200, and later; proportional main/BT4 values; default constant-path
   exactness; validation of inert/unsupported combinations; config and resume
   binding; unchanged optimizer-state ABI; and unchanged update-0 model/loss/
   gradient values.
2. A real-checkpoint A10G smoke has finite loss and gradients, all architecture
   depths at their full defaults, three encoded boards per example, eight
   predictions, SIGReg counts `576/512`, no clipping/skipped update, and no
   write outside `/mountpoint/.exp`.
3. A cached 30-update profile must reach at least `87.5092 examples/s`, 95% of
   K=2. Schedule evaluation should have no material throughput cost; report
   the measured delta.

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

Separately report update-1200 CE/accuracy/legal deltas against both constant-
rate runs to test the warmdown hypothesis without using that secondary
analysis for checkpoint selection. If the first run passes, repeat it exactly
and require the repeat to beat K=2 CE while passing every non-CE gate before
the frozen 128-pair arena. A failure gets no repeat, arena, pass-count sweep,
or SAE refit. Retain only a qualifying selected state; otherwise delete all
candidate states after preserving compact evidence.
