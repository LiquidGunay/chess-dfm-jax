# Experiment 018: inference-aligned first-action masking

Status: completed and rejected at the primary H1-margin gate on 2026-07-22.
No repeat or arena was run, no candidate state is retained, and the active
surface has returned to the accepted one-block-tail incumbent.

## Question and hypothesis

The fixed training corruption samples one `t ~ Uniform(0, 1)` per example
and masks each trajectory action with probability `1 - t`. The played first
action is therefore supervised on roughly half of each physical batch. Frozen
validation sets `t=0`, and eight-pass searchless inference starts from fully
masked actions. The model is selected on horizon-1 quality, but its first
action currently receives only about 64 masked examples per batch-128 update.

Experiments 015--017 show that scalar first-action reweighting cannot satisfy
the H1, uniform-CE, and legal-mass gates simultaneously. This experiment
returns to the accepted uniform objective and source legality coefficient,
then forces only the played first action to the mask token during training.
The other seven action masks and all predeclared RNG streams remain unchanged.
The hypothesis is that 128 inference-aligned H1 examples per update reduce
gradient variance and improve H1 CE without the legality tradeoff caused by
loss reweighting.

## Candidate contract

Add one default-off model/config flag and activate exactly:

```python
"dfm_force_first_action_mask": True,
```

- Draw the existing full `[batch, horizon]` random mask exactly as before,
  then overwrite only `is_masked[:, 0] = True` and
  `noisy_actions[:, 0] = action_vocab_size` in training. Do not consume or
  reorder RNG values.
- Do not alter the target action, horizons 2--8 masks/noisy actions, sampled
  `t`, validation, inference, legal masks, labels, metrics, or codec.
- Keep uniform eight-horizon DFM CE (`dfm_first_action_loss_share=0`) and the
  source first-legality coefficient `2.0`. Because legality is masked-only,
  its H1 training population becomes all valid examples without changing its
  coefficient or reduction definition.
- Keep the accepted one-block future-gradient tail, balanced per-example K=1
  targets, both projector blocks, all four DFM and JEPA blocks, eight
  prediction horizons, train/eval batch `128/64`, peak main/BT4 rates
  `3e-5/1e-6`, the `400/800/0.1` cosine schedule, global data permutation,
  and eight-pass searchless inference.
- Keep RMS norm matching disabled, target SIGReg `5.76`, `z_pred` SIGReg
  `1.0`, and the fixed 64-example normalized SIGReg sample with effective
  target/prediction counts `576/512`.
- Preserve model, optimizer, and checkpoint state ABI. Record the training-
  only corruption change in serialized config and resume metadata; default
  off must preserve the accepted graph's exact loss, aux, gradient, and RNG
  behavior.

Initialize from recovered step 265,000 model-only with a fresh optimizer and
seed 0. Do not initialize from rejected Experiment 015--017 states.

## Staged gates

1. Focused CPU tests must prove H1 mask fraction/count exactly `1.0/batch`,
   exact mask-token replacement, unchanged target actions and H2--H8 masks,
   unchanged target/SIGReg RNG outcomes, uniform CE weighting, default-off
   loss/aux/gradient parity, evaluation parity, unchanged state ABI and
   inference, metadata, and fail-closed objective validation.
2. A one-update real-checkpoint A10G smoke at batch 128 must be finite and
   unclipped with H1 mask fraction `1.0`, no weighted-CE aux branch, source
   legality coefficient `2.0`, balanced 16-per-horizon targets, SIGReg counts
   `576/512`, future routing `14/1`, no state write, and peak HBM below the
   device limit.
3. A checkpoint-free cached 30-update profile must reach at least `150.0`
   end-to-end examples/s and stay below `13,100,000,000` peak JAX HBM bytes.
   Record compiler work, utilization, power, and data stalls.

## Fixed-time decision

Train for 1,800 steady-state seconds and keep temporary updates 800, 1600, and
terminal only. Evaluate all three on unchanged seed-10,000 and seed-20,000
pools, each 64 batches of 64 examples. Select minimum two-pool H1 CE, breaking
an exact tie with lower uniform CE.

Primary acceptance requires:

- H1 CE below `2.8402584158`;
- uniform DFM CE no greater than `4.4971840288`;
- aggregate accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

A passing primary requires one exact repeat. The repeat must beat accepted
primary H1 CE `2.8470487520`, stay within the uniform-CE ceiling, and pass
every non-CE gate. Only then run frozen 128-pair cap-256 arenas against
accepted update 2,072 and raw BT4. An Elo-aligned offline incumbent requires
point score above `50%` against update 2,072 and above `36.5234375%` against
raw BT4. These development screens cannot authorize Elo promotion.

Failure at any earlier gate gets no repeat, arena, pass-count sweep, or SAE
work. Retain only a qualifying primary selected state and compact evidence;
delete all nonselected and repeat states. Keep all mutable files below
`/mountpoint/.exp`, and never overlap GPU workloads.

## Outcome

All CPU and staged GPU gates passed. The one-update smoke cold-compiled in
`170.274 s`, used `12,874,133,504` peak JAX HBM bytes, and confirmed H1 mask
fraction `1.0`, unchanged sampled H2--H8 fractions, no weighted-CE branch,
source legality coefficient `2.0`, balanced 16-per-horizon targets, SIGReg
counts `576/512`, future routing `14/1`, finite unclipped loss, and no state
write. The 30-update profile reached `154.153` end-to-end and `215.054`
device examples/s at `12,865,748,736` peak HBM bytes. XLA estimated
`14.2575e12` FLOP and `133.447e9` bytes per update; measured data stalls were
`28.32%`, GPU utilization mean/median/p95 was `60.49/91/100%`, and power
mean/p95 was `170.64/199.06 W`.

The fixed run completed `2,069` updates and `264,832` examples at `152.678`
end-to-end and `212.793` device examples/s, with `12,965,360,640` peak HBM
bytes. Two-pool H1-first selection chose the terminal checkpoint:

| Update | H1 CE | Uniform CE | Accuracy | Legal mass |
|---:|---:|---:|---:|---:|
| 800 | 2.8699970 | 4.5040965 | 0.1088715 | 0.6426268 |
| 1600 | 2.8563285 | 4.4963654 | 0.1098633 | 0.6462354 |
| **2069** | **2.8417612** | **4.4927525** | **0.1095734** | **0.6483661** |

The selected checkpoint clears uniform CE, accuracy, legal mass, and every
latent gate. Prediction effective-rank mean/min is `30.9865/29.0841`,
feature-std p05 mean/min is `0.64975/0.63553`, mean target RMS is `0.92561`,
and prediction/target RMS ratio is `0.97110`; positive JEPA prediction beats
zero, identity, and action-shuffled controls at every horizon. H1 CE improves
the accepted primary by `0.0052875`, but misses the preregistered
repeat-noise-aware ceiling by `0.0015028`.

The candidate is therefore rejected without repeat or arena. Delete all
three state payloads and restore the accepted surface. The close result
supports training/inference corruption alignment but not acceptance. A
separately preregistered follow-up may reduce future-action leakage by biasing
training time toward more-masked contexts while keeping H1 forced and the
loss fixed. RMS norm matching remains disabled; target and `z_pred` SIGReg
remain `5.76/1.0`.
