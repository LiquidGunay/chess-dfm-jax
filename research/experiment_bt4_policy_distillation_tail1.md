# Experiment 023: frozen BT4 policy-head distillation at the DFM root

Status: preregistered and implemented on 2026-07-23. Codec/data/model audits,
the guarded CPU gate, and the no-update coefficient calibration are complete;
the training smoke and later accelerator gates have not started.

## Question and hypothesis

The accepted one-block-tail model is materially weaker than original raw BT4:
its frozen 128-pair point score is `36.5234375%`, despite improving the
trajectory imitation objective. Raw BT4's policy head is already loaded inside
the research model, but the existing loss never calls it. The head parameters
and attention mapping are source `nnx.Param` leaves rather than
`TrainableParam` leaves, so they receive no gradient, optimizer update, weight
decay, or research-checkpoint storage.

This experiment asks whether the stronger source policy can supervise the
DFM's root ranking during training without another BT4 encoder call and
without adding inference work. The existing current-board BT4 tokens feed the
frozen source policy head. Its legal distribution is converted from native
`lc0_canonical_1858` coordinates to the recovered DFM's
`legacy_absolute_1858` coordinates, detached, and matched by the legal-masked
DFM root distribution.

The teacher is exact original raw BT4 at source initialization. The current
BT4 encoder remains trainable, however, so its tokens evolve during training.
The precise description after the first update is therefore **frozen source
head on online tokens**, not a fully frozen raw-BT4 teacher. A truly frozen
teacher would require a second 195M-parameter encoder forward or precomputed
policy targets. Both are outside this bounded experiment and would work
against the local iteration-speed goal.

The hypothesis is that the cheap frozen-head target transfers useful
within-legal-move ranking into the eight-pass DFM and improves its direct raw
BT4 arena score, while the played-action CE and legality terms preserve the
trajectory objective and represented legal mass.

## Completed data and codec audit

Trajectory-v3 stores `INPUT_CLASSICAL_112_PLANE` tensors. In that format the
piece history is side-to-move oriented, while auxiliary plane index `108` is
exactly all zero for white to move and all one for black to move. A
deterministic audit of four shards in each of train, validation, and test
checked 12,288 rows:

- all input-format metadata was exactly classical 112-plane;
- plane 108 was spatially constant, binary, and agreed with `fen_t`;
- all 98,304 valid stored action labels matched `legacy_absolute_1858`;
- every stored root legal set matched the representable board-legal set;
- 173,364 sampled legal moves changed index under black canonicalization; and
- 191 legal promotions were intentionally absent from the incomplete legacy
  support.

The audit handled valid Chess960/DFRC metadata rows with Chess960 castling
semantics. Runtime distillation does not reconstruct boards or FENs: it uses
plane 108 plus the stored legal indices, so Chess960 castling does not create
a hot-path ambiguity.

For white, canonical-to-legacy mapping is identity. For black, vertically
mirror the from and to squares while preserving the queen/rook/bishop
promotion suffix. Some black-side entries have no inverse because the legacy
list contains only canonically oriented promotion slots. They receive an
invalid sentinel and must be excluded by the stored legacy legal mask. Any
valid stored legal slot with an invalid mapping makes that example ineligible
and is reported; the audited data contract expects zero such slots.

The implementation passes 146 focused tests under the resource guard in
172.04 seconds. The guard observed `4,344,844,288` bytes peak process-group
RSS and at least `7,996,817,408` bytes host `MemAvailable`. Tests cover the
exact mapping and promotion sentinels, loss and gating mathematics, stopped
teacher/nonzero student gradients, played-label independence, immutable
policy-head and checkpoint state, active serialization/resume provenance,
default-off legacy parity, checkpoint handling, SIGReg, target sampling, and
the accepted one-block future-tail path. Lint is clean.

## Frozen target and loss

Add one default-off scalar:

```python
"bt4_policy_distill_coeff": 0.0,
```

When it is positive, compute exactly:

```text
teacher_canonical =
    source_policy_head(stop_gradient(current_bt4_tokens))

teacher_legacy[a] =
    teacher_canonical[canonical_index(side_to_move, legacy_index=a)]

q = softmax(mask_to_stored_root_legal(teacher_legacy))
p = softmax(mask_to_stored_root_legal(dfm_root_logits))

root_policy_kl = sum_a q[a] * (log q[a] - log p[a])
```

Use temperature `1.0`. Stop gradients through the teacher logits,
distribution, side-to-move decision, and index mapping. The policy head and
its mapping remain nontrainable source state. The student is the existing
final noisy-planner root output; no second student forward is allowed.

Apply the KL only when the root action is masked, the sample is valid, stored
root legality is marked valid and nonempty, plane 108 is a uniform binary
plane, and every stored root legal action has a valid canonical mapping.
Invalid rows use a finite numerical fallback but have exactly zero KL weight.
The student softmax is legal-masked, so this term teaches ranking only; the
existing first-legality term remains solely responsible for total legal mass.
The played root label never enters the teacher distribution.

Report raw and weighted KL, eligible count/fraction, side-plane validity,
mapping coverage, teacher entropy, teacher played-action NLL/top-1 accuracy,
and teacher/student legal top-1 agreement. Coefficient zero must preserve the
original graph, arithmetic, RNG, gradients, state ABI, reports, and inference
behavior exactly.

## One-shot coefficient calibration

This is not a coefficient sweep. Before any optimizer update, evaluate the
source-initialized model on the unchanged seed-10,000 and seed-20,000 pools,
each 64 batches of 64 examples, with a calibration coefficient of `1.0`.
Let `K0` be pooled eligible root KL. Reject if it is nonfinite, nonpositive,
or if eligibility is not 100%.

Freeze the training coefficient to:

```text
lambda = clip(0.25 / K0, 0.05, 1.0)
```

and persist the measured `K0`, exact coefficient, and weighted source
contribution before the one-update training smoke. The target contribution
`0.25` is preregistered because it matches the source-scale JEPA positive
term (about `0.35`) and weighted target-SIGReg term (about `0.25`) while
remaining small relative to roughly `4.5` DFM CE. Do not round or retune the
coefficient after seeing training, validation, or arena results.

### Calibration result

Commit `537ec36` evaluated both frozen source-initialized pools with
coefficient `1.0`, deterministic time zero, 64 batches of 64 examples, zero
optimizer updates, and zero checkpoint writes:

| Seed | Root KL | Eligibility | Mapping coverage | Teacher/student top-1 | Teacher played accuracy |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.8583731093 | 1.0 | 0.9999999935 | 0.5656738281 | 0.4885253906 |
| 20,000 | 0.8729048609 | 1.0 | 0.9999999981 | 0.5588378906 | 0.4985351562 |
| Pooled | **0.8656389851** | **1.0** | **0.9999999958** | **0.5622558594** | **0.4935302734** |

The tiny mapping-coverage difference from one is validation-aggregation
floating-point roundoff; every row passed the complete-mapping gate. Pooled
teacher played NLL/entropy are `1.6699445089/1.8876852738`.

The preregistered formula therefore fixes:

```text
lambda = 0.25 / 0.8656389850657433
       = 0.2888039983331077
```

The coefficient is inside the unclipped interval and its pooled weighted
source contribution is exactly `0.25` in host FP64. It is now immutable for
the smoke, profile, fixed-time run, and repeat. Fourteen guarded targeted
checks, including an exact checked-in coefficient/formula assertion and all
active baseline controls, pass after freezing it; lint remains clean.

Seed 10,000 used `6,145,294,336` bytes peak guarded process-group RSS and
`3,809,939,968` bytes peak JAX HBM; seed 20,000 used
`6,097,801,216`/`3,837,368,832` bytes. Minimum host `MemAvailable` was
`5,897,994,240` and `6,033,047,552` bytes respectively. Both processes
exited normally with no checkpoint.

## Fixed controls

Keep:

- recovered step 265,000 model-only initialization and a fresh optimizer;
- seed 0 and global data permutation;
- horizon 8 and exactly eight searchless inference passes;
- train/evaluation batch `128/64`;
- main/BT4 peak learning rates `3e-5/1e-6`;
- cosine schedule `400/800/0.1`;
- one-block future-target BT4 gradient tail;
- balanced one-of-eight future target per physical example;
- both projector blocks, all four DFM blocks, and all four recurrent JEPA
  transition blocks;
- uniform eight-horizon DFM CE and source first-legality coefficient `2.0`;
- normalized V-statistic SIGReg on a fixed 64 examples; and
- loss coefficients norm/target/`z_pred` `0.0/5.76/1.0`.

Do not force the first mask, bias training time, enable WDL/value loss, change
DFM passes, or use the policy head at inference. Preserve model, optimizer,
and checkpoint state ABIs. Source-code provenance must include the action-codec
mapping implementation.

The accepted offline incumbent is
`future-tail1-balanced-k1-cosine-b128-30m-v1/update 2,072`. Its primary/repeat
two-pool uniform CE is `4.4950091/4.4971840`, accepted horizon-one CE ceiling
is `2.8538390882`, legal mass is `0.6488690/0.6468926`, direct raw-BT4 point
score is `36.5234375%`, and matched recurrent profile throughput is
`153.5362` examples/s.

## Staged gates

Before fixed-time training:

1. Focused guarded CPU tests must prove the exact white/black mapping,
   promotion sentinels, plane-108 decoding and invalid-row gating, stored-legal
   masking, exact KL reduction, target-label independence, stopped teacher
   gradients, nonzero student/DFM/backbone gradients, immutable policy-head
   state, coefficient-zero parity, unchanged state ABI and inference,
   serialization/resume semantics, and fail-closed invalid coefficients.
2. Run the guarded two-pool calibration with no optimizer update and no
   checkpoint. Eligibility must be 100%; teacher metrics, `K0`, the formula,
   and selected coefficient must be persisted before training.
3. A guarded real-checkpoint batch-128 one-update smoke must finish finite and
   unclipped, report the frozen coefficient and nonzero eligible KL, preserve
   balanced 16-per-horizon targets, SIGReg counts `576/512`, future routing
   `14/1`, fixed loss coefficients `0.0/5.76/1.0`, and write no state.
4. The launcher must enforce the existing exclusive GPU lock, two host CPUs,
   8 GiB launch and 3 GiB runtime `MemAvailable` floors, 7 GiB process-group
   RSS ceiling, 30 GiB disk reserve, and maximum two checkpoint writes. Any
   guard abort ends the experiment; do not rescue it with a smaller batch,
   altered graph, or relaxed guard.
5. A guarded checkpoint-free cached 30-update profile must reach at least
   `145.0` end-to-end examples/s and stay below `13,100,000,000` peak JAX HBM
   bytes. Record compiler work, utilization, power, data stalls, and
   distillation metrics.

Failure at any earlier gate gets no later profile, fixed run, checkpoint,
repeat, or arena.

## Fixed-time and strength decision

If every pre-run gate passes, train for 1,800 steady-state seconds after
compilation. Disable periodic saves and write at most update 1,200 plus the
terminal state. Evaluate both states on unchanged seed-10,000 and
seed-20,000 pools, each 64 batches of 64 examples.

Among states passing all policy and latent gates, select minimum pooled
teacher/student root KL, breaking an exact tie with lower horizon-one CE.
Primary acceptance requires:

- pooled root KL no greater than `0.90 * K0`;
- teacher/student legal top-1 agreement above its pooled source value;
- pooled horizon-one CE no greater than `2.8538390882`;
- uniform DFM CE no greater than `4.4971840288`;
- aggregate action accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- 100% distillation eligibility and every metric/gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94,1.02]`.

A passing primary requires one exact repeat satisfying the same systems,
distillation, policy, and latent gates. Only then run frozen 128-pair cap-256
arenas against update 2,072 and original raw BT4. An Elo-aligned distilled
incumbent requires point score above `50%` against update 2,072 and above
`36.5234375%` against raw BT4. These development screens cannot authorize
absolute Elo or formal promotion.

Retain a state only after both repeated offline and arena point gates pass.
Otherwise delete all candidate states after preserving compact reports,
hashes, telemetry, and the decision. A failure restores coefficient zero and
closes this one-shot distillation test rather than authorizing a weight or
temperature sweep. Keep every mutable file, cache, compiler artifact,
temporary file, and checkpoint under `/mountpoint/.exp`, and never overlap
GPU workloads.
