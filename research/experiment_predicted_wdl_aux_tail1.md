# Experiment 020: outcome supervision on predicted JEPA states

Status: preregistered on 2026-07-22; activation and measurements have not
started.

## Question and hypothesis

Experiment 019 shows that lower first-action and uniform imitation CE can
repeat and win a direct point estimate against the current DFM incumbent while
still regressing against original raw BT4. The policy therefore needs a chess-
outcome signal, not another scalar corruption or CE-weight sweep.

The approved trajectory-v3 shards already contain per-horizon scalar
`value_targets` in `{-1,0,1}` and one-hot `wdl_targets`. These are final game
outcomes from the side to move after each played action. They are not engine
evaluations and do not provide counterfactual values for unplayed moves, but
they can test whether free-rollout predicted states retain outcome-relevant
information. The checkpoint-compatible `value_wdl_head` exists but has never
received outcome supervision.

This experiment trains the WDL branch on every valid free-rollout `z_pred`
with coefficient `0.25`. An untrained three-class CE starts near `ln(3)`, so
the weighted contribution is about `0.275`: comparable to the JEPA and target-
SIGReg components rather than large enough to dominate the roughly 4.5 policy
objective. Gradients pass through the WDL head, recurrent JEPA transition,
clean DFM action hidden states, and shared backbone. The hypothesis is that
this stronger action/latent/outcome coupling improves measured chess strength
without sacrificing the accepted imitation, legality, or latent-health gates.

## Frozen label baselines

Across the exact 64-by-64 validation pools and all eight valid horizons:

| Seed | Win fraction | Draw fraction | Loss fraction | Constant-prior CE | Majority accuracy |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.346680 | 0.306641 | 0.346680 | 1.096986 | 0.346680 |
| 20,000 | 0.279175 | 0.441650 | 0.279175 | 1.073341 | 0.441650 |
| Pooled | 0.312927 | 0.374146 | 0.312927 | 1.094934 | 0.374146 |

The win/loss columns use the stored side-to-move convention. Root-side signs
therefore alternate with horizon parity and must never be averaged without
conversion.

## Candidate contract

Activate exactly:

```python
"wdl_coeff": 0.25,
```

- Keep scalar `value_coeff=0.0`; isolate categorical WDL supervision.
- Apply the existing shared WDL head to all eight free-rollout `pred_z`
  states. Use only valid one-hot labels, normalize their class sum as the
  existing loss path does, and average over examples and horizons.
- Keep the balanced per-example K=1 positive JEPA target estimator. WDL uses
  all valid predicted horizons; it does not change target/SIGReg sampling.
- Record WDL CE, top-1 accuracy, expected-value MSE, valid count, target class
  fractions, and CE/accuracy by horizon. Metric additions must not alter the
  coefficient-zero forward, loss, gradient, RNG, or checkpoint behavior.
- Do not use WDL at inference in this experiment. Eight-pass deterministic
  DFM inference, legality masking, arena openings, and codecs remain exact.
  Fixed-candidate generation/ranking is a separate experiment only after a
  head generalizes and the policy gates repeat.
- Keep forced H1 masking off and training-time power `1.0`; Experiment 019
  closed corruption tuning.
- Keep uniform DFM CE, source first-legality coefficient `2.0`, the accepted
  one-block future-gradient tail, both projector blocks, all four DFM and JEPA
  blocks, horizon eight, train/eval batch `128/64`, peak main/BT4 rates
  `3e-5/1e-6`, the `400/800/0.1` cosine schedule, global data permutation,
  and current-only eight-pass searchless inference.
- Keep RMS norm matching disabled, target SIGReg `5.76`, `z_pred` SIGReg
  `1.0`, and the fixed 64-example normalized SIGReg sample with effective
  target/prediction counts `576/512`.
- Preserve the model and checkpoint state ABI. The existing WDL parameters
  are already part of the recovered checkpoint tree; only their loss becomes
  active. Serialize the coefficient and WDL metric contract fail-closed.

Initialize from recovered step 265,000 model-only with a fresh optimizer and
seed 0. Do not continue from Experiment 019 or the accepted update 2,072.

## Staged gates

1. Focused CPU tests must prove coefficient-zero parity, exact WDL reduction,
   side-to-move label convention, finite metrics, nonzero WDL gradients through
   the WDL head, JEPA transition, DFM hidden path, and BT4 current path,
   unchanged target/SIGReg assignment, unchanged inference/state ABI, active
   config serialization, and fail-closed invalid coefficient/label shapes.
2. A one-update real-checkpoint A10G smoke at batch 128 must be finite and
   unclipped with WDL coefficient `0.25`, nonzero WDL loss/valid count, fixed
   norm/target/prediction coefficients `0.0/5.76/1.0`, uniform CE, source
   legality coefficient `2.0`, balanced 16-per-horizon JEPA targets, SIGReg
   counts `576/512`, future routing `14/1`, and no state write.
3. A checkpoint-free cached 30-update profile must reach at least `145.0`
   end-to-end examples/s and stay below `13,100,000,000` peak JAX HBM bytes.
   Record compiler work, utilization, power, and data stalls.

## Fixed-time decision

Train for 1,800 steady-state seconds and keep temporary updates 800, 1600, and
terminal only. Evaluate all three on unchanged seed-10,000 and seed-20,000
pools, each 64 batches of 64 examples. Among checkpoints with pooled WDL CE no
greater than `1.05` and pooled WDL accuracy at least `0.40`, select minimum
two-pool H1 CE, breaking an exact tie with lower uniform CE. If no checkpoint
qualifies on WDL, select the lowest WDL CE for diagnosis and reject.

Primary acceptance requires:

- pooled WDL CE no greater than `1.05`, below the empirical constant-prior CE
  on each seed, and pooled WDL accuracy at least `0.40`;
- H1 CE no greater than the worse accepted control run, `2.8538390882`;
- uniform DFM CE no greater than `4.4971840288`;
- aggregate action accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94,1.02]`.

A passing primary requires one exact repeat satisfying the same WDL, policy,
and latent gates. Only then run frozen 128-pair cap-256 arenas against accepted
update 2,072 and raw BT4. An outcome-aware offline incumbent requires point
score above `50%` against update 2,072 and above `36.5234375%` against raw
BT4. These development screens cannot authorize Elo promotion.

Failure at any earlier gate gets no later arena or candidate-ranking test.
Retain only a primary state that passes both arena point gates and compact
evidence; delete all nonselected and repeat states. A failure restores
`wdl_coeff=0.0` and closes this single-coefficient joint-auxiliary test rather
than authorizing a WDL-weight sweep. Keep every mutable file below
`/mountpoint/.exp`, and never overlap GPU workloads.
