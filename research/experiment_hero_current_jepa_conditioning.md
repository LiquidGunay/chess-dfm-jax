# Experiment: current-JEPA-state conditioning of DFM

Status: completed and not promoted on 2026-07-28. The preregistered offline
gates failed, while centered chess was inconclusive and the post-hoc
pass-scaling audit was favorable against raw BT4. Keep both matched
checkpoints until a disjoint confirmation closes the decision.

## Question and hypothesis

The retained hero model has one directional state/action coupling:

```text
current BT4 state -> DFM action hidden -> recurrent JEPA prediction
```

Its learned current JEPA state does not condition DFM action prediction.
The rejected final-pass feedback experiment tried to close the loop with a
model-generated action and predicted state change after seven planner calls.
That mechanism added an exposure-sensitive inference loop and harmed frozen
validation/rank metrics. The confirmed pass-count study then showed that this
checkpoint is strongest with one planner call.

This experiment instead exposes the already available current JEPA state to
the one-pass DFM policy:

```text
z_jepa = state_projector(current_bt4_tokens)
bridge = zero_initialized_linear(z_jepa)       # 1024 -> 256, no bias
z_dfm' = z_dfm + broadcast_over_64_tokens(bridge)
```

The hypothesis is that a target-safe, learnable residual lets action
supervision shape the JEPA representation and lets the policy consume that
representation, improving direct chess strength without recurrent predicted-
state feedback.

## One-change model contract

Add one default-off config field and activate exactly:

```python
dfm_condition_on_current_jepa_state = True
```

When active:

- add one no-bias `RawLinear(z_dim, token_dim)` bridge, exactly
  `1024 * 256 = 262,144` FP32 parameters;
- initialize its weight to exact zero without consuming the shared fresh-
  module RNG stream, so every parameter shared with the control has identical
  initialization;
- add the bridge output to every one of the 64 current DFM state tokens before
  every noisy/clean planner call in training, frozen validation, and Arena
  inference;
- compute the current state projector at inference, but add no JEPA transition
  and no additional DFM planner call;
- allow DFM gradients through the bridge into the current JEPA state
  projector as soon as the zero bridge begins to learn; and
- record bridge-weight RMS, applied-conditioning RMS, unconditioned DFM-state
  RMS, and their ratio in every training row and frozen evaluation.

The bridge consumes only the current board. It must not read the played
target action, sampled future target, WDL label, predicted future state, or
legality target. Keep `jepa_feedback_mode = "none"`.

Default-off construction must preserve the exact existing parameter/state
ABI and inference values. Existing checkpoints predate the new config key;
descriptor validation may supply only its frozen default `False` when that
single key is absent, while still rejecting unknown keys and every other
missing config field. A candidate checkpoint must fail to load unless the
active bridge is requested.

Keep every loss and coefficient fixed:

```text
DFM CE / root legal CE / illegal mass:
  1.0 / 0.08707768618513193 / 2.0

JEPA positive / target SIGReg / prediction SIGReg:
  1.0 / 2.0 / 2.0

WDL:
  0.25

SIGReg examples:
  64
```

Also freeze raw-BT4 initialization, batch `1,024`, eight-action training
horizon, target sample count one, optimizer, example-based LR schedule,
seed/data order, canonical codec, rematerialization, SDPA, fused-backward
LayerNorm, regional compilation, and after-forward prefetch.

## Correctness and systems gates

Before candidate training, focused CPU tests must prove:

- the explicit override is hero-only and default-off;
- disabled conditioning returns the exact original tensor and creates no
  parameter;
- active zero initialization returns exact original DFM latents at update
  zero, consumes no shared initialization RNG, and gives the bridge a nonzero
  gradient;
- after a nonzero bridge weight, gradients reach `z_jepa`;
- active residual broadcasting, shapes, dtype, and RMS diagnostics are exact;
- the training and full-horizon evaluation paths use identical conditioning;
- one-pass Arena inference computes and consumes the current JEPA state while
  preserving strict root legality;
- old and active checkpoint/config descriptors fail closed correctly; and
- all existing feedback-off/pass-count/Arena tests still pass.

Then run one guarded, checkpoint-free 20-update batch-1,024 systems smoke.
It must:

- complete 20/20 finite updates with zero skips;
- have nonzero bridge-weight and conditioning RMS by the terminal update;
- sustain at least `241.84` examples/s over warm updates, 95% of the retained
  control's `254.570` examples/s;
- stay below physical A10G HBM and all host guard limits; and
- write no checkpoint.

A systems failure ends this candidate rather than changing batch size,
compilation, rematerialization, or the scientific graph.

### Systems-gate result

Commit `9740c58` passed 71 focused tests plus Ruff and bytecode compilation.
The guarded run
`research/runs/torch_autoresearch_jepa_condition_smoke_b1024_v1` completed
20/20 finite updates with no optimizer skips and no checkpoint:

| Measurement | Result | Gate |
|---|---:|---:|
| Mean warm step throughput, updates 2--20 | `256.509` examples/s | `>=241.84` |
| Minimum warm step throughput, updates 2--20 | `256.199` examples/s | descriptive |
| Peak allocated HBM | `18,482,722,304` bytes | fits A10G |
| Peak guarded process-group RSS | `2,499,313,664` bytes | `<7 GiB` |
| Minimum host available memory | `8,653,017,088` bytes | `>3 GiB` |
| Terminal bridge-weight RMS | `0.0001171450` | nonzero |
| Terminal conditioning RMS | `0.0386254` | nonzero |
| Terminal conditioning/state RMS ratio | `0.0236672` | diagnostic |
| Checkpoint writes | `0` | `0` |

GPU utilization reached 100%; the guard completed normally after `111.78`
seconds. The architecture therefore clears the frozen systems gate without a
batch-size, compiler, rematerialization, or objective change.

## Matched training and frozen evaluation

If the smoke passes, train exactly 1,024 updates / 1,048,576 examples from
raw BT4 plus identically seeded fresh modules. Write every per-update scalar,
one loss summary, one run report, and exactly one terminal model-only
checkpoint. Do not continue from the retained model.

Evaluate on the identical frozen 8,192-example fast pool with batch 64 and
the active conditioning flag. Compare against the retained control:

| Metric | Retained control |
|---|---:|
| DFM CE | `5.7062988095` |
| Accuracy | `0.0724334717` |
| Legal mass | `0.5542044039` |
| Root legal CE | `2.1195805371` |
| JEPA positive MSE | `0.1331897546` |
| Horizon-8 prediction effective/stable rank | `11.1652365 / 4.6844682` |
| Horizon-8 target effective/stable rank | `10.6858968 / 4.7534403` |

## Primary chess screen and decision

Play the candidate directly against the retained count-64 checkpoint on the
same first 128 color-reversed `hero_development` opening pairs:

```text
refinement passes / policy batch cap / additional-ply cap / seed:
  1 / 16 / 256 / 0
```

Both models use their own recorded configs and independently verified
checkpoints in the native Torch runtime. Arena score against the incumbent is
the primary ranker; raw BT4 is not rerun for this first screen.

Advance only if:

1. candidate paired score is strictly greater than `0.5`, with zero faults,
   zero cap draws, normal terminations, and complete action coverage;
2. frozen DFM CE is no more than `0.005` worse, root legal CE no more than
   `0.01` worse, accuracy loses less than `0.001` absolute, and legal mass
   loses less than `0.002` absolute;
3. horizon-8 prediction and target effective/stable ranks each retain at
   least 95% of control, with no feature-tail collapse;
4. conditioning and bridge-weight RMS are finite and nonzero at the end of
   training and in frozen evaluation;
5. training sustains at least `241.84` examples/s and candidate mean Arena
   policy-call time is no more than 1.25 times the incumbent; and
6. every loss, gradient, checkpoint, codec, legality, and resource gate
   passes.

A positive screen requires an exact fresh repeat and then an extension to at
least 10% of an epoch before it replaces the incumbent. Confirm a winner on
disjoint predeclared opening pairs before promotion. A negative result closes
this zero-initialized current-state broadcast design; it does not reject
other learned state/action coupling mechanisms.

After sealing all hashes and compact evidence, retain only the model needed
for the active recipe. Never overlap training, validation, Arena, or SAE work
on the single A10G.

## Result

The candidate completed all 1,024 updates / 1,048,576 matched examples from
raw BT4 plus fresh modules with zero optimizer skips. It took `4,123.12`
seconds at `254.316` examples/s versus `4,119.01` seconds and `254.570`
examples/s for the control. Peak allocated HBM increased by only `2,393,600`
bytes. The bridge is systems-neutral for training.

The last-64 training window was already negative:

| Metric | Matched 1,024 control | JEPA-conditioned | Candidate change |
|---|---:|---:|---:|
| Total loss | `7.449589` | `7.539384` | `+0.089795` |
| DFM CE | `5.702513` | `5.751280` | `+0.048767` |
| Accuracy | `0.073498` | `0.070585` | `-0.002913` |
| Legal mass | `0.547733` | `0.529534` | `-0.018199` |
| Root legal CE | `2.119340` | `2.107607` | `-0.011732` |
| JEPA MSE | `0.134350` | `0.135860` | `+0.001510` |

The learned residual did not stay small. Over the final 64 updates, its mean
RMS was `0.774432` against unconditioned DFM-state RMS `1.616753`, a ratio of
`0.480475`; bridge-weight RMS was `0.009474`. Norms remained finite and
matched, so this is not numerical collapse.

Frozen 8,192-example validation confirms the policy regression:

| Metric | Matched 1,024 control | JEPA-conditioned | Candidate change |
|---|---:|---:|---:|
| DFM CE | `5.706299` | `5.740033` | `+0.033734` |
| Accuracy | `0.072433` | `0.069931` | `-0.002502` |
| Legal mass | `0.554204` | `0.537391` | `-0.016813` |
| Root legal CE | `2.119581` | `2.109003` | `-0.010577` |
| JEPA MSE | `0.133190` | `0.132550` | `-0.000640` |
| Target / prediction SIGReg | `0.063517 / 0.064247` | `0.069144 / 0.066891` | worse |
| WDL CE | `0.771990` | `0.791962` | `+0.019972` |

The tiny JEPA-MSE and root-ranking improvements do not compensate for worse
policy quality. At horizon 8, prediction effective/stable rank retains only
`94.16% / 94.03%`; target effective/stable rank retains
`95.29% / 94.59%`. Prediction and target feature-standard-deviation p05
retain `95.56% / 90.81%`, so feature tails remain present but representation
rank fails the 95% gate.

The centered direct Arena is directionally positive but inconclusive:

| Metric | Result |
|---|---:|
| Score / points | `0.509766` / `130.5 of 256` |
| W/D/L | `66 / 129 / 61` |
| Pentanomial | `[5, 33, 52, 28, 10]` |
| Better / equal / worse pairs | `38 / 52 / 38` |
| Descriptive paired-t 95% interval | `[0.467119, 0.552412]` |
| Two-sided paired-t p-value | `0.651227` |
| Descriptive logistic Elo | `+6.79` |
| Candidate / incumbent call | `31.552 / 29.495 ms` |
| Candidate latency ratio | `1.06977` |
| Faults / cap draws / abnormal terminations | `0 / 0 / 0` |

The candidate passes the strict point-score direction and latency gates, but
the Arena uncertainty includes a material loss and gain. It fails the
preregistered DFM-CE, accuracy, legal-mass, and rank gates, so it is not
promoted from this screen. The matched control checkpoint
`05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9`
remains the incumbent.

The offline result warns that the shared additive residual is blunt: it grows
to roughly half of the per-token state scale, slightly helps JEPA MSE/root
ranking, and harms held-out action prediction and latent rank. It does not
establish that this candidate is weaker at chess, nor that direct
state/action coupling is generally harmful. A future variant should still
consider preserving token structure or using a controlled
gate/cross-attention mechanism.

The candidate checkpoint was deterministically replayed and checksum-verified
as
`eb20c698fea1eabd4ebf71c13a1c195204736183f7b173079bd34f2a6e9a995f`
with all 52 scientific scalars matching at all 1,024 updates. Its
`713,388,280`-byte tensor is retained pending the follow-up decision.
Compact evidence and the comparison plot are:

- `research/analysis/current_jepa_conditioning_20260728.json`
- `research/analysis/current_jepa_conditioning_20260728.png`

## Post-hoc inference-pass audit

The original `0.509766` Arena opponent was the matched 1,024-update control,
not the one-epoch Hero. Its W/D/L `66/129/61` and descriptive interval
`[0.467119,0.552412]` are a centered near tie, not a rejection of chess
strength.

On the same first 128 development pairs against raw BT4:

| Passes | Matched control | JEPA-conditioned | Candidate minus control |
|---:|---:|---:|---:|
| 1 | `0.960938` | `0.986328` | `+0.025391` |
| 2 | `0.937500` | `0.943359` | `+0.005859` |
| 4 | `0.925781` | `0.958984` | `+0.033203` |
| 8 | `0.908203` | `0.945312` | `+0.037109` |
| 16 | `0.910156` | `0.933594` | `+0.023438` |

The candidate is descriptively better at every count, with paired intervals
excluding zero at passes 1, 4, and 8. It nevertheless falls from one to
sixteen passes by `-0.052734`, so it has not learned beneficial iterative
refinement after only 1,024 updates. Because this is a reused, ceiling-prone
raw-BT4 pool and direct head-to-head play is tied, reclassify the architecture
as **not promoted / scientifically inconclusive**. Require disjoint centered
confirmation before either a longer extension or closure.

Full replay, pass, and initialization evidence is in
`research/analysis/posthero_pass_bootstrap_audit_20260728.json`.
