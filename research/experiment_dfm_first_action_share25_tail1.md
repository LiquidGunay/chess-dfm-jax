# Experiment 015: 25% first-action DFM objective share

Status: completed and rejected at the primary legal-mass gate on 2026-07-22.
No repeat or arena was run, and no candidate state is retained.

## Question and hypothesis

The accepted one-block-tail checkpoint improves uniform eight-horizon DFM CE
in two exact runs, but its frozen raw-BT4 arena score is `36.523%`, compared
with `38.086%` for corrected v2/update 400 on the identical roots. The
difference is small and is not a direct model-versus-model test, but it shows
that lower uniform trajectory CE has not yet produced higher measured chess
strength.

Training currently gives each of eight action horizons one eighth of DFM CE,
whereas searchless gameplay emits only the first action and discards the other
seven. This experiment doubles the aggregate objective allocation to the
played action: horizon 1 receives weight `0.25`, and horizons 2--8 each receive
`0.75 / 7 = 3/28`. The weights sum to one, so this changes horizon allocation
without changing the DFM CE coefficient. The hypothesis is that direct
first-action supervision improves the held-out horizon-1 CE and the matched
raw-BT4 arena score while retaining future-action modeling, uniform CE,
legality, latent health, and one-block-tail throughput.

## Candidate contract

Add one default-off scalar to the editable training surface and activate
exactly:

```python
"dfm_first_action_loss_share": 0.25,
```

Candidate semantics:

- `0.0` preserves the existing uniform objective exactly. An active value is
  a fraction in `(0, 1]`; this experiment uses `0.25` with all eight horizons.
- Compute the existing masked per-horizon CE means unchanged. Preserve
  `dfm_ce_loss` as the uniform eight-horizon validation and reporting metric.
- Add `dfm_objective_ce_loss`: horizon 1 has normalized weight `0.25`, and
  each other active horizon has normalized weight `3/28`. The weighted values
  sum to one over active, valid horizons.
- Use `dfm_objective_ce_loss`, not the uniform reporting metric, in the
  training objective and the DFM gradient-component audit. Report the exact
  effective horizon weights and both CE scalars.
- Do not change action corruption, masks, labels, logits, legality loss,
  checkpoint/model/optimizer state, inference, or the uniform validation
  metric. The config and effective weights must be bound into reports and the
  resume contract.
- Keep the accepted one-block future-gradient tail, balanced per-example K=1
  future targets, both projector blocks, all four DFM and JEPA blocks, eight
  prediction horizons, batch `128`, evaluation batch `64`, peak main/BT4
  rates `3e-5/1e-6`, `400/800/0.1` cosine schedule, global data permutation,
  and eight-pass searchless inference.
- Keep RMS norm matching disabled. Target SIGReg remains `5.76`, `z_pred`
  SIGReg remains `1.0`, and normalized SIGReg continues to use a fixed
  64-example sample with effective target/prediction counts `576/512`.

The training initialization remains recovered step 265,000 model-only with a
fresh optimizer and seed 0. It does not continue from the accepted update
2,072 state.

## Frozen controls

The one-block-tail primary/repeat selected checkpoints are the direct control:

| Run | Uniform DFM CE | Horizon-1 CE | Accuracy | Legal mass | Examples/s |
|---|---:|---:|---:|---:|---:|
| Primary/update 2,072 | 4.4950091206 | 2.8470487520 | 0.1104583740 | 0.6488690404 | 153.0754 |
| Repeat/update 2,080 | 4.4971840288 | 2.8538390882 | 0.1103057861 | 0.6468925704 | 153.4504 |

Their horizon-1 CE separation is `0.0067903362`. The repeat-noise-aware
primary horizon-1 ceiling is therefore `2.8402584158`. The worse accepted
repeat supplies a uniform-CE non-regression ceiling of `4.4971840288`.

The accepted checkpoint's frozen 128-pair scores are `49.609%` against the
balanced-K1 incumbent and `36.523%` against raw BT4. The latter has
descriptive logistic Elo `-96.02` and pair-aware interval
`[-195.33,-10.24]`. These are development screens, not absolute Elo or
promotion claims.

## Staged gates

Before the fixed run:

1. Focused CPU tests must prove bitwise/exact default-off loss and gradient
   parity, the active `[0.25, 3/28, ..., 3/28]` weights, a unit-sum effective
   mask, correct objective gradients, unchanged uniform metrics, unchanged
   model/optimizer ABI and inference, resume/report metadata, and fail-closed
   invalid shares.
2. A real-checkpoint batch-128 one-update A10G smoke must finish unclipped and
   finite with the active weights, balanced 16-per-horizon target assignments,
   eight predictions, SIGReg counts `576/512`, detached/attached future depths
   `14/1`, no state write, and peak HBM below the device limit.
3. A checkpoint-free cached 30-update profile must reach at least `150.0`
   end-to-end examples/s and stay below `13,100,000,000` bytes peak JAX HBM.
   Record compiler FLOPs/bytes, utilization, power, and data stalls. Failure
   ends the experiment before fixed-time training.

## Fixed-time decision

Train for 1,800 steady-state seconds. Keep temporary updates 800, 1600, and
terminal only. Evaluate all three on the unchanged seed-10,000 and seed-20,000
pools, each 64 batches of 64 examples. Select minimum mean horizon-1 DFM CE,
breaking an exact tie with lower uniform DFM CE; the scanner's existing
uniform `best_checkpoint` field is evidence only and does not override this
preregistered selection rule.

Primary acceptance requires:

- horizon-1 CE below `2.8402584158`;
- uniform eight-horizon DFM CE no greater than `4.4971840288`;
- aggregate accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

A passing primary requires one exact repeat. The repeat must beat the accepted
primary control's horizon-1 CE `2.8470487520`, stay within the uniform-CE
non-regression ceiling, and pass every non-CE gate. Only then run the frozen
128-pair cap-256 arenas against current update 2,072 and raw BT4. Accept an
Elo-aligned offline incumbent only if its point score exceeds `50%` against
update 2,072 and exceeds the current `36.5234375%` raw-BT4 score. These screens
do not authorize promotion; only the separate normalized-Elo promotion GSPRT
can do that.

A failure at any earlier gate gets no later arena, pass-count sweep, or SAE
work. Retain only a qualifying primary selected state and compact evidence;
delete all nonselected and repeat states. Keep all files, caches, and temporary
objects under `/mountpoint/.exp`, and never overlap GPU workloads.

## Outcome

The implementation and all pre-run gates passed. Eighty-six focused CPU tests
plus lint cover exact default-off loss/auxiliary-tree/gradient/optimizer/state
parity, the active `[0.25, 3/28, ..., 3/28]` weights, unit-sum and gradient
semantics, unchanged uniform metrics and inference, serialization, routing,
and fail-closed validation.

The real-checkpoint batch-128 smoke completed one finite, unclipped update with
the exact eight weights, balanced 16-per-horizon future assignments, target/
prediction SIGReg counts `576/512`, future detached/attached depths `14/1`,
and no state write. Cold compilation took `167.867` seconds and peak JAX HBM
was `12,866,224,128` bytes.

The cached 30-update profile passed at `154.7626` end-to-end and `214.7296`
device examples/s with peak JAX HBM `12,865,913,600` bytes. Compiler work is
unchanged at `14.2575` TFLOP and `133.447` GB per update. Mean/p50/p95 GPU
utilization was `64.27/96.5/100%`, mean/p95 power was `171.04/197.76` W, and
the data-stall fraction was `27.93%`.

The fixed run compiled and completed its first update in `15.5681` seconds,
then processed `265,856` examples in 2,077 updates at `153.2968` end-to-end
and `213.2939` device examples/s. Peak JAX HBM was `12,957,817,344` bytes.
The preregistered two-pool scan was:

| Update | Horizon-1 CE | Uniform CE | Accuracy | Legal mass |
|---:|---:|---:|---:|---:|
| 800 | 2.7410193 | 4.4907841 | 0.1093750 | 0.6338664 |
| 1,600 | 2.7239352 | 4.4828804 | 0.1101990 | 0.6372061 |
| 2,077 | **2.7167628** | **4.4811102** | **0.1101532** | **0.6371051** |

Terminal update 2,077 is the selected checkpoint. It clears the horizon-1 CE
ceiling by `0.1234956`, improves the accepted primary by `0.1302859`, and
clears the uniform-CE non-regression ceiling by `0.0160738`. Accuracy also
passes. Legal mass `0.6371051`, however, misses the frozen `0.6467235` floor by
`0.0096185`, so the primary is rejected.

Latent health is not the failure. Prediction effective-rank mean/minimum is
`30.9855/29.1275`, feature-std p05 mean/minimum is `0.64959/0.63593`, mean
target RMS is `0.92495`, and the prediction/target RMS ratio is `0.97167`.
Positive JEPA prediction beats zero, identity, shuffled-target, and
action-shuffled controls at all eight horizons; mean JEPA/identity ratio is
`0.14649`.

Per the frozen contract, run no repeat or arena. Delete all update-800,
update-1,600, and update-2,077 state payloads after preserving manifests,
reports, metrics, scans, and diagnostics. Return the active training surface
to the accepted unweighted one-block-tail incumbent. RMS norm matching remains
off, target SIGReg remains `5.76`, and `z_pred` SIGReg remains `1.0`.

The result isolates a useful loss-allocation effect. The source checkpoint's
first-legality coefficient is `2.0`: under uniform CE its ratio to horizon-1
CE weight is `2/(1/8)=16`, whereas the 25%-share candidate changes that ratio
to `2/(1/4)=8`. A separately preregistered coefficient-4 rescue would restore
the original relative legality pressure while testing whether the large CE
gain can survive. This follow-up must retain every current gate and is not
authorization to weaken the legal-mass threshold.
