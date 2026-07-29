# Experiment: three-arm 10%-epoch WDL/closed-loop scaling

Status: preregistered and started on 2026-07-29.

## Question

At 1,024 updates (`1,048,576` examples, `3.6996%` of one trajectory
epoch), current-state WDL preserves a strong one-pass policy, predicted-JEPA
closed loop is the only tested mechanism with positive recurrent refinement,
and their combination does not yet compose. This experiment asks whether
those conclusions persist after `2,768` updates (`2,834,432` examples,
`10.0004%` of an epoch), and whether their early loss and Elo trajectories
are informative about longer training.

The three arms are:

1. **current-WDL:** `wdl_include_current_state=true`;
2. **closed-loop:** `dfm_closed_loop_mode=predicted_jepa_tokens`; and
3. **combined:** both settings above.

Every arm starts afresh from the same raw BT4 checkpoint plus the same seed-0
JEPA/DFM initialization. None continues from the jointly pretrained legacy
checkpoint, the Hero-1024 checkpoint, or the one-epoch Hero checkpoint.

## Frozen training contract

All arms use the one-epoch Hero recipe and remain prefixes of a possible
one-epoch run:

- physical batch `1,024`, data cursor zero, global-permutation seed zero;
- main peak LR `5e-4`, BT4 peak LR `5e-4 / 30`;
- linear warmup for `566,866` examples, then cosine decay over the full
  `28,343,296`-example epoch to a `0.001` LR ratio;
- selective weight decay `0.01` and global gradient clipping at `1.0`;
- DFM CE / legality / JEPA-positive / target-SIGReg / prediction-SIGReg /
  WDL coefficients `1 / 2 / 1 / 2 / 2 / 0.25`;
- fixed count-64 SIGReg statistics for both latent losses;
- canonical LC0 action codec, horizon eight, and root-only BT4 policy-logit
  residual;
- SDPA-all, compiled fresh regions, rematerialized BT4 and projector,
  eager-fused LayerNorm backward, after-forward depth-one prefetch, two CPU
  threads, and the existing GPU/RAM/disk guard.

The 10% endpoint remains near peak LR (`4.9183e-4` main,
`1.6394e-5` BT4). The cosine schedule is deliberately not compressed into
the short run. Physical batch and fixed SIGReg estimator count remain
unchanged, so no SIGReg coefficient adjustment is made.

## Measurements

The same immutable 8,192-example validation pool is evaluated from the live
model at updates:

| Update | Examples | Epoch fraction |
| ---: | ---: | ---: |
| `554` | `567,296` | `2.0015%` |
| `1,024` | `1,048,576` | `3.6996%` |
| `1,384` | `1,417,216` | `5.0002%` |
| `2,076` | `2,125,824` | `7.5003%` |
| `2,768` | `2,834,432` | `10.0004%` |

These evaluations serialize no model snapshots. Each arm retains exactly one
terminal model-plus-optimizer recovery checkpoint, which is also directly
loadable by offline evaluation and Arena code. The requested stop step is a
mutable process boundary rather than part of resume compatibility, so a
selected arm can continue under the same full-epoch LR schedule.

At 10%, record full frozen-validation policy, legality, WDL, JEPA, SIGReg,
latent norm, and latent rank metrics. Then use a common, predeclared set of
256 complete color-reversed opening pairs to measure:

- one pass versus Hero-1024 one pass;
- eight passes versus Hero-1024 one pass;
- eight passes versus the same checkpoint at one pass; and
- the stronger DFM setting versus the same checkpoint's policy-only path.

Selection uses greedy legal action choice, inference batch cap 16, additional
ply cap 256, complete action-codec coverage, and zero tolerated policy
faults. One-pass WDL, closed-loop, combined, and Hero-1024 also receive a
connected round robin on the same openings. These are development Elo
estimates, not promotion tests.

## Curve analysis

The completed one-epoch Hero supplies a same-recipe shape prior, not a
cross-recipe quality guarantee. Refit its frozen-validation trajectory using
all ten 10%-spaced milestones and compare floor-plus-exponential,
floor-plus-power, and monotone interpolation fits in integrated LR mass.
Estimate each candidate's residual/shared-shape continuation from its
2--10% observations, with leave-one-milestone-out error and a curve-family
envelope. Report predictions as uncertain hypotheses, never as promotion
evidence.

Fit one-pass and eight-pass chess strength separately. With only one short
Elo endpoint per arm, do not claim an Elo scaling law; use the Hero
half/full-epoch results only as a qualitative prior. The known SIGReg-256
counterexample means lower multitask or validation loss cannot universally
stand in for Elo across recipes.

## Decision

Do not reject an arm solely because recurrent passes hurt at 10%. The
one-epoch Hero itself transitions from early pass degradation to strong
positive refinement. At this stage compare:

- one-pass strength;
- eight-pass strength and eight-versus-one refinement direction;
- frozen DFM CE and per-horizon behavior;
- WDL calibration and latent health; and
- training/inference cost.

After all three matched 10% runs and Arenas complete, choose at most one arm
for actual 25%, 50%, and 100% milestones. A projected curve alone cannot
authorize that extension.
