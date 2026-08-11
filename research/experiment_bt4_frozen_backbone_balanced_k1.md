# Experiment 010: frozen BT4 backbone with balanced K=1 targets

Status: completed and rejected on 2026-07-22. The candidate substantially
improves throughput and DFM CE but fails the frozen legal-mass floor, so it
receives no repeat or arena.

## Question and hypothesis

The accepted balanced per-example K=1 model still differentiates and updates
the complete BT4 encoder. The backbone contains `195,305,728` of
`274,149,702` trainable parameters (`71.24%`), while its peak learning rate is
only `1e-6`. The raw BT4 policy is also the strongest available chess anchor.
It is therefore plausible that the expensive encoder backward pass buys
little short-run adaptation, and that preserving the self-play representation
while giving the DFM/JEPA heads more updates in the same wall time improves
the fixed-time policy metric.

This experiment freezes the BT4 backbone as one coherent change. It preserves
forward values and checkpoint-visible encoder parameters, stops gradients at
the encoded tokens, gives encoder leaves no optimizer updates or optimizer
state, and sets the reported BT4 learning rate to zero. Everything downstream
remains trainable. The primary hypothesis is that this both improves training
throughput and lowers 30-minute full-horizon DFM CE beyond the balanced-K1
repeat envelope. The failure modes are that even the small BT4 update is
important for coupling the chess representation to the DFM/JEPA objectives,
or that host input and downstream compute hide most of the expected speedup.

## Candidate contract

Add a default-off configuration switch and activate exactly:

```python
"bt4_freeze_backbone": True,
"bt4_learning_rate": 0.0,
```

The candidate semantics are:

- Keep `unfreeze_bt4_encoder=True` and retain the complete source-compatible
  model-state ABI. Freezing is an autoresearch runtime/optimizer policy, not a
  shape-changing checkpoint conversion.
- Apply `stop_gradient` once to the BT4 token output used by both current and
  sampled-future paths. Forward token values, validation, and inference must
  be identical to the unfrozen graph for identical parameters and inputs.
- Exclude BT4 embedding/layer leaves from optimizer transformations and
  optimizer state. Keep all non-BT4 parameters, including both state
  projectors, the four-block DFM, and recurrent JEPA predictor, trainable.
- Set the BT4 schedule identically to zero at every update. A nonzero BT4
  learning rate with the freeze switch is invalid and fails closed.
- Preserve strict source provenance. Model-only initialization first verifies
  the legacy source model and optimizer against a compatibility optimizer,
  restores the complete model, and then trains with a fresh frozen-backbone
  optimizer. Exact legacy optimizer restoration is unsupported and must fail
  closed. Exact research-checkpoint resume uses the frozen optimizer ABI.
- Serialize the switch only when enabled. Bind the stop-gradient location,
  optimizer exclusion, zero learning-rate schedule, source-import bridge, and
  unchanged forward/inference semantics into reports and resume contracts.
- Report model/optimizer parameter counts by group so the claimed optimizer
  state reduction is machine-checkable.

No other model, loss, data, or schedule setting changes. In particular, keep
balanced per-example K=1 target sampling; all eight DFM/JEPA prediction
horizons; two projector and four DFM blocks; eight-pass searchless inference;
the 400/800/10%-floor cosine schedule; raw JEPA MSE; RMS norm matching off;
target SIGReg `5.76`; prediction SIGReg `1.0`; and the fixed 64-example
V-statistic SIGReg sample.

## Frozen controls and run configuration

The primary control is balanced-K1 v1/update 1,581:

- state SHA-256:
  `26c3621d1f9c35aa1401ef5e2e9d35ffdf59607d0c4369730cc6dd207b65c7c7`;
- two-pool DFM CE `4.4979399741`;
- accuracy `0.1100921631`;
- legal mass `0.6497235410`;
- cached profile `117.3798961` end-to-end and `149.0971679` device
  examples/s;
- fixed-run throughput `117.0610578` examples/s;
- peak fixed-run HBM `9,574,246,400` bytes; and
- exact-repeat selected CE separation `0.0009704642`.

Start from the recovered step-265,000 model with a fresh optimizer. Freeze one
NVIDIA A10G and no concurrent workload; physical/evaluation batches `128/64`;
train/data seed 0 with `global_permutation`; main learning rate `3e-5`; zero
BT4 learning rate; schedule start/decay/floor `400/800/0.1`; loss coefficients
`0.0/5.76/1.0`; 1,800 seconds of steady-state training after compile/first
update; validation seeds 10,000 and 20,000, each 64 batches of 64 examples;
and saves at updates 400, 800, 1200, 1600, and terminal with at most five
temporary states.

## Correctness and performance gates

Before the fixed run:

1. CPU tests prove default-off serialization and forward/gradient parity;
   candidate forward equality; exact-zero BT4 gradients and parameter deltas;
   nonzero downstream gradients and updates; absence of encoder optimizer
   moments; strict model-only legacy import through the compatibility ABI;
   fail-closed exact legacy initialization; exact frozen-checkpoint resume;
   and complete reporting/resume semantics.
2. A one-update real-checkpoint A10G smoke proves bitwise-unchanged encoder
   leaves, a successful downstream optimizer update, finite metrics, full
   architecture depths, balanced `16`-per-horizon assignment, SIGReg counts
   `576/512`, and no write outside `/mountpoint/.exp`.
3. A cached 30-update profile must not regress below the control's
   `117.3798961` end-to-end examples/s. The expected useful effect is at least
   5% (`123.2488909` examples/s); a smaller gain is recorded rather than
   silently attributed to freezing. Diagnose any regression before spending
   the fixed-run budget.

## Checkpoint selection and decision

Evaluate every saved checkpoint on the identical seed-10,000 and seed-20,000
full-horizon pools and select minimum mean DFM CE. Strength-incumbent
acceptance requires:

- two-pool DFM CE below `4.4969695099`, more than the accepted
  `0.0009704642` repeat separation better than balanced-K1 v1;
- accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beats zero and action-shuffled predictions at every
  horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

Also compare the full learning curve, examples processed, compiler estimate,
peak HBM, and encoder/optimizer-state bytes with both balanced-K1 runs. If the
primary passes, repeat it exactly; the repeat must beat balanced-K1 v1 CE and
pass every non-CE gate before the frozen 128-pair arena against balanced-K1
v1/update 1,581. If CE stays within the incumbent repeat envelope while
throughput improves, record a compute result but do not replace the strength
incumbent or run the arena. A failure gets no repeat, arena, pass-count sweep,
or SAE refit. Retain only a qualifying selected state; otherwise delete every
candidate state after preserving compact evidence.

## Outcome

The implementation and all pre-run correctness gates passed. The focused CPU
suite has 76 passing tests. A real-checkpoint A10G smoke preserved the full
two-block projector and four-block DFM, assigned exactly 16 examples to every
horizon, encoded two boards per example, produced all eight predictions,
reported SIGReg counts `576/512`, and completed a finite unclipped downstream
update. Its peak JAX HBM was `3,164,207,872` bytes. At both update 400 and
terminal update 2,504, all 404 source BT4 encoder leaves were bitwise equal to
initialization with maximum absolute delta `0.0`.

The cached 30-update profile reached `186.7543` end-to-end and `281.1704`
device examples/s, `59.10%` and `88.58%` above the balanced-K1 control. Peak
JAX HBM fell to `3,155,814,144` bytes. XLA's static estimate fell from
`13.5645` to `6.4374` TFLOP/update, a `52.54%` reduction. The profile clears
both the no-regression gate and the preregistered 5% useful-effect target.

The fixed run compiled in `8.5523` seconds and processed `320,512` examples
in 2,504 updates at `185.6428` steady end-to-end and `282.4400` device
examples/s. Relative to balanced-K1 v1, this is `58.59%` more end-to-end
throughput and `58.38%` more examples within the same 1,800 steady seconds.
Peak JAX HBM was `3,253,244,672` bytes, `66.02%` lower than the incumbent.
The frozen optimizer covers `78,843,974` parameters and carries no state for
the `195,305,728` BT4 parameters. Each temporary checkpoint was
`1,290,641,865` bytes, `30.30%` smaller than the trainable-backbone state.

Hardware samples expose the new bottleneck: GPU utilization was `55.00%`
mean, `68%` median, and `100%` p95, while the measured data-stall fraction was
`34.27%`. Mean/p95 power was `177.93/215.62 W` and p95 SM clock was
`1710 MHz`. A future frozen or partially frozen graph should therefore retune
batching/input delivery separately rather than attributing host stalls to the
model kernel.

The frozen two-pool checkpoint scan was:

| Update | DFM CE | Accuracy | Legal mass |
|---:|---:|---:|---:|
| 400 | 4.5099108 | 0.1090088 | 0.6431585 |
| 800 | 4.5032279 | 0.1088867 | 0.6420830 |
| 1,200 | 4.5014677 | 0.1095886 | 0.6440504 |
| 1,600 | 4.4986572 | 0.1096039 | 0.6443900 |
| 2,504 | **4.4965047** | **0.1097260** | **0.6445921** |

Terminal update 2,504 wins monotonically on mean CE. It improves the
balanced-K1 incumbent by `0.0014352724` and clears the strict CE ceiling by
`0.0004648082`; accuracy also passes. Its latent audit passes every collapse
gate: prediction effective-rank mean/minimum `30.8243/28.9355`, feature-std
p05 mean/minimum `0.64926/0.63581`, mean target RMS `0.92239`, and
prediction/target RMS ratio `0.97462`. Positive prediction beats zero and
action-shuffled controls at every horizon, and JEPA/identity MSE is `0.13913`.

The decisive failure is legal mass. Selected legal mass `0.6445921` is
`0.0051314` below the incumbent and misses the frozen `0.6467235` floor by
`0.0021314`. Reject the candidate without repeat or arena. This is evidence
that some BT4 adaptation matters for the action distribution even though full
freezing is an excellent systems optimization and improves CE. All five
candidate states were deleted after recording hashes and compact evidence;
no frozen-backbone state is retained.
