# Experiment: current-JEPA-state conditioning of DFM

Status: implementation and systems gate passed on 2026-07-28; the matched
1,024-update candidate is authorized but has not yet started.

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
