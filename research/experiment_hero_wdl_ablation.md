# Experiment: remove the predicted-state WDL auxiliary

## Question

Does the `0.25` categorical final-outcome auxiliary on free-running JEPA
predictions improve early searchless chess strength, or does it spend shared
gradient capacity on a noisy target that is not used at inference?

This is the first post-hero “go big, then ablate” experiment. It is not an
action-only model: uniform eight-horizon DFM CE, root legality/ranking, raw
JEPA MSE, target SIGReg, and prediction SIGReg all remain active.

## Motivation

The WDL labels are played-game final outcomes from the side to move at each
horizon. They contain useful chess information, but no LC0 search value or
counterfactual action ranking. The WDL head is training-only in the current
eight-pass decoder. At fresh initialization its weighted scalar is about
`0.295`; its gradients reach the JEPA transition, state projector, and raw
BT4 backbone, although the zero-initialized conditioning path initially
delays gradients into the DFM action stack.

The matched SIGReg sample-count experiment also shows why this ablation needs
an Arena signal: count 256 improved JEPA MSE by 12% and DFM CE slightly while
substantially lowering paired Arena score. A representation-side loss gain is
not sufficient evidence of chess strength.

## Preregistered comparison

Reuse `torch_autoresearch_sigreg64_u1024_v1` as the control. Run one new
candidate from the same raw BT4 parameters and identically seeded fresh
modules:

| Setting | Control | Candidate |
|---|---:|---:|
| WDL coefficient | `0.25` | `0.0` |
| SIGReg example count | `64` | `64` |
| Target / prediction SIGReg | `2.0 / 2.0` | `2.0 / 2.0` |
| Physical batch | `1,024` | `1,024` |
| Optimizer updates | `1,024` | `1,024` |
| Examples | `1,048,576` | `1,048,576` |
| Seed / data start | `0 / 0` | `0 / 0` |

Every other model, loss, optimizer, example-based LR schedule, canonical
codec, stochastic choice, and selected systems setting remains exactly
equal. Use the same fused-backward LayerNorm, regional compilation,
after-forward prefetch, and exact canonicalizer. Write every train component
to `metrics.jsonl` and exactly one terminal model-only checkpoint.

The control was produced at the immediately preceding default-neutral code
revision. Before launch, focused tests must prove that omitting the new
explicit CLI override still resolves to the exact frozen hero config and that
the Arena descriptor accepts the recorded coefficient while rejecting any
other unrecorded config drift.

## Frozen evaluation

Evaluate the candidate on the same 8,192-example fast validation pool. The
evaluator deliberately uses the standard count-64, WDL-`0.25` scoring
contract for both checkpoints; this makes the candidate's untouched WDL head
a diagnostic without adding WDL gradients during its training. Compare:

- uniform and per-horizon DFM CE and action accuracy;
- root legal-conditional CE/top-1 and legal mass;
- JEPA MSE and trivial/action-shuffled controls;
- prediction/target norms, feature tails, effective/stable rank, and cosine;
- diagnostic WDL CE, accuracy, calibration, and expected-value MSE; and
- throughput, peak HBM, non-finite skips, and every terminal/last-64 loss.

Then run the identical first 128 color-reversed opening pairs against raw BT4
with eight refinement passes, inference batch cap 16, and additional-ply cap
256. Require zero faults and zero cap draws. Compare per-opening pair scores
directly with the retained control rather than treating two independent Elo
headlines as the primary statistic.

## Decision

Arena is the primary screen. Advance WDL-off only if:

1. its paired Arena score is directionally non-worse than the control;
2. frozen DFM CE is no more than `0.005` worse, root legal CE is no more than
   `0.01` worse, action accuracy loses less than `0.1` percentage point, and
   legal mass loses less than `0.2` percentage point;
3. horizon-8 prediction and target effective/stable rank each retain at least
   95% of control, with no feature-tail collapse; and
4. all numerical, legality, and execution gates pass.

The diagnostic WDL metrics are not a promotion gate: they should worsen when
the head is untrained and quantify what the ablation removes.

A positive screen must be repeated from scratch and extended to at least 10%
of an epoch before it replaces the WDL-on recipe. A negative screen restores
`wdl_coeff=0.25` and closes the zero-versus-`0.25` ablation; do not infer a
better intermediate coefficient or open a sweep.

After all evidence and checkpoint hashes are sealed, retain only the model
needed for the active recipe. Never overlap its training, validation, Arena,
or any SAE workload on the single A10G.
