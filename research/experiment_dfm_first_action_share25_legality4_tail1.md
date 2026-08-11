# Experiment 016: first-action share with restored legality ratio

Status: completed and rejected at the primary gate on 2026-07-22. No repeat
or arena was run, no candidate state is retained, and the active surface has
returned to the unweighted one-block-tail incumbent.

## Question and hypothesis

Experiment 015 reallocates DFM CE from a uniform `1/8` per horizon to `1/4`
for the played first action and `3/28` for each later action. The selected
checkpoint improves held-out horizon-1 CE from `2.8470488` to `2.7167628` and
uniform CE from `4.4950091` to `4.4811102`, while preserving accuracy,
throughput, and latent health. It fails because legal mass falls from
`0.6488690` to `0.6371051`.

The source configuration's first-legality coefficient is `2.0`. Relative to
horizon-1 CE, its coefficient ratio is `2/(1/8)=16` under uniform CE but only
`2/(1/4)=8` under the 25%-share candidate. This experiment changes only that
coefficient from `2.0` to `4.0`, restoring the original ratio
`4/(1/4)=16`. The hypothesis is that scale-matched legality pressure recovers
the frozen legal-mass floor while preserving most of the unusually large
horizon-1 and uniform CE gains.

## Candidate contract

Activate exactly:

```python
"dfm_first_action_loss_share": 0.25,
"first_legality_coeff": 4.0,
```

The first-action weighting implementation and its default-off compatibility
path from Experiment 015 remain unchanged. Relative to that measured
candidate, `first_legality_coeff: 2.0 -> 4.0` is the sole scientific change.

- Preserve uniform `dfm_ce_loss` reporting and use the active
  `dfm_objective_ce_loss` weights `[0.25, 3/28, ..., 3/28]` for training.
- Compute the same first-action legal-mass loss over the same masks and
  multiply it by `4.0`. Do not change the legality definition, tolerance,
  masking rule, action labels, or inference legality mask.
- Keep the accepted one-block future-gradient tail, balanced per-example K=1
  targets, both projector blocks, all four DFM and JEPA blocks, eight
  prediction horizons, train/eval batch `128/64`, peak main/BT4 rates
  `3e-5/1e-6`, the `400/800/0.1` cosine schedule, global data permutation,
  and eight-pass searchless inference.
- Keep RMS norm matching disabled, target SIGReg `5.76`, `z_pred` SIGReg
  `1.0`, and the fixed 64-example normalized SIGReg sample with effective
  target/prediction counts `576/512`.
- Preserve model/optimizer/checkpoint ABI. Bind both active scalars, exact
  horizon weights, and the restored ratio into run and resume metadata.

Initialize from recovered step 265,000 model-only with a fresh optimizer and
seed 0. Do not initialize from any Experiment 015 state; all such state
payloads were rejected and deleted.

## Frozen controls

| Control | H1 CE | Uniform CE | Accuracy | Legal mass | Examples/s |
|---|---:|---:|---:|---:|---:|
| Accepted one-block tail/u2072 | 2.8470488 | 4.4950091 | 0.1104584 | 0.6488690 | 153.075 |
| Accepted exact repeat/u2080 | 2.8538391 | 4.4971840 | 0.1103058 | 0.6468926 | 153.450 |
| Share-25 rejected/u2077 | 2.7167628 | 4.4811102 | 0.1101532 | 0.6371051 | 153.297 |

The primary horizon-1 ceiling remains `2.8402584158`, one accepted exact-run
separation below the retained primary. The uniform-CE non-regression ceiling
remains `4.4971840288`, and the legal-mass floor remains `0.6467235410`.
Neither threshold is relaxed because the preceding candidate was promising.

## Staged gates

Before the fixed run:

1. Focused CPU tests must prove the active 25% share, coefficient `4.0`,
   restored ratio 16, unchanged uniform metrics, correct weighted-loss
   arithmetic, default-off parity, unchanged state ABI/inference, metadata,
   and fail-closed configuration validation.
2. A one-update real-checkpoint A10G smoke at batch 128 must be finite and
   unclipped with exact horizon weights, coefficient `4.0`, balanced
   16-per-horizon targets, SIGReg counts `576/512`, future routing `14/1`, no
   state write, and peak HBM below the device limit.
3. A checkpoint-free cached 30-update profile must reach at least `150.0`
   end-to-end examples/s and stay below `13,100,000,000` bytes peak JAX HBM.
   Record compiler work, utilization, power, and data stalls.

## Fixed-time decision

Train for 1,800 steady-state seconds and keep temporary updates 800, 1600, and
terminal only. Evaluate all three on the unchanged seed-10,000 and seed-20,000
pools, each 64 batches of 64 examples. Select minimum two-pool horizon-1 CE,
breaking an exact tie with lower uniform CE.

Primary acceptance requires:

- horizon-1 CE below `2.8402584158`;
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
primary horizon-1 CE `2.8470487520`, stay within the uniform-CE ceiling, and
pass every non-CE gate. Only then run frozen 128-pair cap-256 arenas against
accepted update 2,072 and raw BT4. An Elo-aligned offline incumbent requires a
point score above `50%` against update 2,072 and above the current
`36.5234375%` raw-BT4 score. These development screens cannot authorize Elo
promotion.

Failure at any earlier gate gets no later arena, pass-count sweep, or SAE
work. Retain only a qualifying primary selected state and compact evidence;
delete all nonselected and repeat states. Keep all mutable files below
`/mountpoint/.exp`, and never overlap GPU workloads.

## Outcome

All preflight gates passed. The one-update smoke cold-compiled in
`173.040 s`, used `12,866,199,552` peak JAX HBM bytes, and confirmed the
exact 25%/`3/28` weights, coefficient `4.0`, ratio 16, balanced 16-per-horizon
targets, SIGReg counts `576/512`, future routing `14/1`, and no state write.
The cached 30-update profile reached `153.908` end-to-end and `214.315`
device examples/s at `12,866,245,120` peak HBM bytes. XLA estimated
`14.2575e12` FLOP and `133.447e9` bytes per update; measured data stalls were
`28.19%`, GPU utilization mean/median/p95 was `56.25/65/100%`, and power
mean/p95 was `170.67/198.30 W`.

The fixed run completed `2,062` updates and `263,936` examples at `152.325`
end-to-end and `212.685` device examples/s, with `12,943,827,456` peak HBM
bytes. Two-pool H1-first selection chose the terminal checkpoint:

| Update | H1 CE | Uniform CE | Accuracy | Legal mass |
|---:|---:|---:|---:|---:|
| 800 | 2.8701953 | 4.5107981 | 0.1087189 | 0.6412415 |
| 1600 | 2.8625372 | 4.5031024 | 0.1096039 | 0.6467257 |
| **2062** | **2.8581280** | **4.5007452** | **0.1104431** | **0.6460591** |

The selected checkpoint misses the H1 ceiling by `0.0178696`, the uniform-CE
ceiling by `0.0035611`, and the legal-mass floor by `0.0006645`; accuracy
passes. Latent health also passes: prediction effective-rank mean/min is
`31.0318/29.1502`, feature-std p05 mean/min is `0.65004/0.63642`, mean target
RMS is `0.92620`, and prediction/target RMS ratio is `0.97067`. Positive JEPA
prediction beats zero, identity, and action-shuffled controls at every
horizon.

Restoring the nominal legality/H1 coefficient ratio almost restores legal
mass, but it also erases the large H1 and uniform-CE gains from Experiment
015. The tradeoff is therefore not explained by that scalar ratio alone. The
primary is rejected without repeat or arena, all three state payloads are
deleted, and the loss remains fixed at norm/target/`z_pred` coefficients
`0.0/5.76/1.0`.
