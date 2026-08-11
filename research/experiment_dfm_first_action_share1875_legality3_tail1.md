# Experiment 017: midpoint first-action share with matched legality

Status: completed and rejected at the primary gate on 2026-07-22. No repeat
or arena was run, no candidate state is retained, and scalar first-action
share/coefficient tuning is closed.

## Question and hypothesis

The accepted one-block-tail objective assigns `1/8` of uniform DFM CE to the
played first action and uses first-legality coefficient `2.0`. Experiment 015
moves the first-action share to `1/4` while leaving legality at `2.0`; it gains
substantially on H1 and uniform CE but loses legal mass. Experiment 016 keeps
the `1/4` share and raises legality to `4.0`; it nearly restores legal mass but
erases the CE gain. The three relevant selected checkpoints are:

| Objective | H1 CE | Uniform CE | Accuracy | Legal mass |
|---|---:|---:|---:|---:|
| `1/8`, legality 2 | 2.8470488 | 4.4950091 | 0.1104584 | 0.6488690 |
| `1/4`, legality 2 | 2.7167628 | 4.4811102 | 0.1101532 | 0.6371051 |
| `1/4`, legality 4 | 2.8581280 | 4.5007452 | 0.1104431 | 0.6460591 |

This experiment tests the exact midpoint on the ratio-preserving line:
first-action share `3/16 = 0.1875` and legality coefficient `3.0`. Its
legality-to-H1 coefficient ratio remains 16, and the seven later actions each
receive `13/112` of DFM CE. The hypothesis is that the midpoint retains a
measurable part of Experiment 015's H1 improvement while preserving the
accepted legal-mass floor. This is the final scalar interpolation; failure
closes first-action-share/coefficient tuning rather than triggering a sweep.

## Candidate contract

Activate exactly:

```python
"dfm_first_action_loss_share": 0.1875,
"first_legality_coeff": 3.0,
```

- Preserve uniform `dfm_ce_loss` reporting and train with exact objective
  weights `[3/16, 13/112, ..., 13/112]`, which sum to one.
- Do not change legality masks, action labels, metric definitions, tolerance,
  inference masking, or the `legacy_absolute_1858` codec.
- Keep the accepted one-block future-gradient tail, balanced per-example K=1
  targets, both projector blocks, all four DFM and JEPA blocks, eight
  prediction horizons, train/eval batch `128/64`, peak main/BT4 rates
  `3e-5/1e-6`, the `400/800/0.1` cosine schedule, global data permutation,
  and eight-pass searchless inference.
- Keep RMS norm matching disabled, target SIGReg `5.76`, `z_pred` SIGReg
  `1.0`, and the fixed 64-example normalized SIGReg sample with effective
  target/prediction counts `576/512`.
- Preserve model, optimizer, and checkpoint ABI. Bind both active scalars,
  exact horizon weights, and ratio 16 into run and resume metadata.

Initialize from recovered step 265,000 model-only with a fresh optimizer and
seed 0. No rejected Experiment 015 or 016 state may be used.

## Staged gates

1. Focused CPU tests must prove the exact `3/16` and `13/112` weights,
   coefficient `3.0`, ratio 16, unchanged uniform metrics, correct objective
   arithmetic, default-off parity, unchanged state ABI/inference, metadata,
   and fail-closed validation.
2. A one-update real-checkpoint A10G smoke at batch 128 must be finite and
   unclipped with exact weights, coefficient `3.0`, balanced 16-per-horizon
   targets, SIGReg counts `576/512`, future routing `14/1`, no state write,
   and peak HBM below `13,100,000,000` bytes.
3. Do not spend a separate 30-update profile. Experiments 015 and 016 execute
   the identical compiled graph and measured `153.297/153.908` end-to-end
   examples/s at `12.96/12.87` GB peak HBM. Changing only finite scalar
   constants and same-shaped horizon weights does not alter the graph shape.

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

The CPU/config gates passed. The one-update A10G smoke cold-compiled in
`173.121 s`, used `12,866,224,128` peak JAX HBM bytes, and confirmed exact
float32 weights `[0.1875, 0.11607143, ..., 0.11607143]`, coefficient `3.0`,
ratio 16, balanced 16-per-horizon targets, SIGReg counts `576/512`, future
routing `14/1`, finite unclipped loss, and no state write. Per preregistration,
the shape-identical Experiment 016 profile was reused rather than spending a
second 30-update GPU profile.

The fixed run completed `2,043` updates and `261,504` examples at `151.079`
end-to-end and `212.031` device examples/s, with `12,973,376,512` peak HBM
bytes. Two-pool H1-first selection chose the terminal checkpoint:

| Update | H1 CE | Uniform CE | Accuracy | Legal mass |
|---:|---:|---:|---:|---:|
| 800 | 2.8623868 | 4.5057547 | 0.1087952 | 0.6448959 |
| 1600 | 2.8597316 | 4.5016250 | 0.1097260 | 0.6469449 |
| **2043** | **2.8555801** | **4.4992252** | **0.1106110** | **0.6483664** |

The selected checkpoint clears accuracy and legal mass, but misses the H1
ceiling by `0.0153216` and the uniform-CE ceiling by `0.0020412`. Latent
health passes: prediction effective-rank mean/min is `30.9981/29.1104`,
feature-std p05 mean/min is `0.65003/0.63692`, mean target RMS is `0.92481`,
and prediction/target RMS ratio is `0.97215`. Positive JEPA prediction beats
zero, identity, and action-shuffled controls at every horizon.

The midpoint restores legal mass but does not produce the required action or
uniform-CE improvement. Reject without repeat or arena, delete all three
state payloads, restore the unweighted one-block-tail surface, and close
scalar first-action share/coefficient interpolation. RMS norm matching
remains disabled; target and `z_pred` SIGReg remain fixed at `5.76/1.0`.
