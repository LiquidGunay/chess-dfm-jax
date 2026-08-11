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

## Result

Reject WDL-off and keep `wdl_coeff=0.25`.

The candidate completed all 1,024 updates / 1,048,576 examples with zero
non-finite skips at `254.881` examples/s. This is only `0.12%` faster than
the matched control, with 9.8 MB less peak allocated HBM, so systems variance
does not explain the scientific result. The only model/loss configuration
difference is the preregistered WDL coefficient.

Frozen 8,192-example validation was mixed but failed four model-health gates:

| Metric | WDL `0.25` | WDL off | Candidate change |
|---|---:|---:|---:|
| DFM CE | `5.706299` | `5.716290` | `+0.009991` |
| Action accuracy | `0.072433` | `0.072754` | `+0.000320` |
| First-action legal mass | `0.554204` | `0.551965` | `-0.002239` |
| Root legal-conditional CE | `2.119581` | `2.148177` | `+0.028596` |
| JEPA MSE | `0.133190` | `0.127578` | `-0.005612` |
| Diagnostic WDL CE | `0.771990` | `1.167484` | `+0.395494` |

The JEPA MSE improvement did not preserve latent diversity. At horizon 8,
WDL-off retained only `84.90% / 86.51%` of control prediction
effective/stable rank and `89.19% / 89.28%` of control target
effective/stable rank. Feature-tail statistics did not collapse, but the
rank-retention gate failed.

Arena is decisive. On the identical 128 color-reversed opening pairs,
WDL-off scored `0.626953` against raw BT4 versus `0.908203` for WDL `0.25`.
The paired score delta is `-0.281250`; WDL-off was worse/equal/better on
`97 / 27 / 4` pairs. The descriptive paired-t 95% interval is
`[-0.318466, -0.244034]`. Both runs had zero policy faults and zero cap
draws.

This screen shows that the training-only final-outcome auxiliary is doing
substantial useful representation shaping or regularization by 3.70% of an
epoch, despite not being queried at inference. It does not establish that
`0.25` is the optimal coefficient. Per preregistration, do not open a
coefficient sweep from this negative zero-versus-`0.25` result.

The rejected candidate model was sealed as SHA-256
`12d54ce16b3997d3dad722a15c69011406065456af17f67b7e1cd95db8fdff8a`
and then its 712,339,608-byte tensor file was deleted. Its manifest, complete
training trace, frozen validation, Arena state, and scalar comparison remain;
the active WDL-on checkpoint
`05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9`
is retained.

The immutable comparison and plot are
`research/analysis/hero_wdl_ablation_20260727.json` and `.png`.
