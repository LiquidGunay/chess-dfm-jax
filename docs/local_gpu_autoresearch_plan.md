# Local GPU Autoresearch Plan

Status: implementation in progress. The plan was approved on 2026-07-18 and
revised on 2026-07-19; unattended research remains disabled with
`AUTORESEARCH_READY = False`. V2/update 300 is the repeat-qualified
compatibility control, not a frozen objective. GPU execution is available
again.

The agreed critical path is now:

1. replace the per-state prediction/target RMS-matching term with calibrated
   prediction SIGReg while retaining raw JEPA MSE and target SIGReg;
2. profile the resulting compiled graph across feasible physical batch sizes
   before selecting the training batch;
3. calibrate both SIGReg coefficients on that selected graph and batch, then
   run short causal objective screens;
4. repeat-qualify a corrected 30-minute baseline;
5. measure relative Elo against the recovered DFM/JEPA model and raw BT4
   anchor under identical searchless inference budgets; and
6. begin the SAE/representation study only after those strength anchors exist.

The A10G is single-tenant throughout this sequence. Training, profiling,
arena evaluation, and SAE work do not run concurrently.

This document is the implementation contract for turning the existing
TPU/cloud-oriented BT4 + DFM + JEPA experiment into a fast, measurable,
single-A10G research loop.

The historical implementation is preserved by the local branch
`legacy/tpu-joint-latent-sasa`. New work happens on
`research/local-gpu-autoresearch`.

## Research goals

1. Measure how strong a fixed-compute, searchless chess policy can become when
   DFM action refinement is trained jointly with a JEPA state predictor.
2. Compare the representations learned by the resulting BT4 backbone with the
   original BT4 representations and the published BT4 transcoders/Lorsa
   dictionaries from
   [Leela-SAEs](https://github.com/JacklE0niden/Leela-SAEs).
3. Make architecture and optimization experiments cheap enough to run in a
   Karpathy-style autoresearch loop.
4. Record enough intermediate state to study collapse, feature emergence,
   phase transitions, grokking-like behavior, and double descent without
   relying on post-hoc anecdotes.

## Definitions

### Searchless

The model may use a fixed number of DFM denoising/refinement passes and a fixed
number of sampled candidate trajectories. It may not expand a game tree.

Every strength result must therefore include:

- DFM refinement count;
- number of sampled trajectories, if greater than one;
- batch-one move latency;
- batched throughput; and
- peak GPU memory.

### Current coupling

The current implementation is coupled during training, but not closed-loop at
inference:

```text
current state -> DFM action logits/hidden -> JEPA predicted future states
                                      \---- JEPA gradients update DFM
```

The JEPA prediction is not fed back into DFM logits. The clean DFM hidden states
used by JEPA are produced with the ground-truth action trajectory. At inference,
only DFM is used.

This is the first baseline and must be described as:

> DFM policy trained with a JEPA auxiliary objective.

A later experiment will explicitly close the loop:

```text
DFM refinement k -> proposed action trajectory
                 -> JEPA predicted latent trajectory
                 -> DFM refinement k+1 conditioned on predicted latents
```

This experiment does not require new labels.

## Workspace boundary

All mutable state must live under `/mountpoint/.exp`.

The research path will set and verify at least:

- `TMPDIR`
- `TEMP`
- `TMP`
- `XDG_CACHE_HOME`
- `UV_CACHE_DIR`
- `PIP_CACHE_DIR`
- `HF_HOME`
- `JAX_COMPILATION_CACHE_DIR`
- `WANDB_DIR`
- `WANDB_CACHE_DIR`
- `WANDB_CONFIG_DIR`
- `MPLCONFIGDIR`

The default local layout is:

```text
/mountpoint/.exp/chess-dfm-jax/
  .venv/
  .local/
    cache/
    tmp/
  data/
    trajectory_v3/
  models/
    source/
  checkpoints/
    source/
    recovered/
  artifacts/
    baselines/
    profiles/
    arena/
  research/
```

Downloaded source archives and source checkpoints are immutable. Recovery and
conversion always write a new copy.

## Known baseline assets

The approved local baseline uses only the existing Drive assets:

- trajectory-v3 LC0 H8 archive: about 11.53 GB;
- joint checkpoint at step 265,000: about 2.12 GB of state data; and
- base model archive: about 1.46 GB.

The trajectory manifest contains approximately:

- 28,343,296 training samples;
- 1,560,576 validation samples;
- 1,559,552 test samples; and
- 31,463,424 samples total.

Each sample contains current and future board planes, eight actions, future
validity, legal-move indices, FEN, ply, game result, and WDL/value labels. The
LC0 PGN WDL/value labels are final-game-result labels, not engine evaluations.
They are available for later ablation but remain disabled in the reproduction
baseline.

The Drive split manifest initially appeared to describe only `state.npz`, but
the verified tar also contains the original `checkpoint_state.json`,
`run_config.json`, and metrics history. No metadata reconstruction is required.
Strict structure and output verification is still mandatory because the
checkpoint does not record the training git SHA.

The joint checkpoint contains the fine-tuned BT4 embedding and encoder, but not
the frozen policy/value/moves-left heads or policy mapping. The base
`BT4_exported.pb.gz` is therefore required both to instantiate the current model
and to evaluate raw BT4.

## Phase 0: reproduce before simplifying

### Deliverables

- Workspace-local Python environment and cache contract.
- Verified archive checksums and manifests.
- Recovered checkpoint metadata written beside, not into, the source state.
- Evaluation of the legacy checkpoint on a fixed validation subset.
- A machine-readable baseline report.

### Baseline report

The report must include:

- exact git commit and checkpoint digest;
- JAX, CUDA, driver, and GPU details;
- effective model configuration;
- each scalar loss component;
- DFM action accuracy by horizon;
- first-move legal mass;
- prediction and target latent RMS;
- per-horizon JEPA MSE;
- compile time, step time, samples/s, boards/s, and peak HBM; and
- any missing or reconstructed metadata.

### Gate

Do not alter architecture or loss behavior until the recovered checkpoint can
be evaluated reproducibly and the reference output is saved.

## Phase 1: one editable research file

The clean path follows the separation used by
[autoresearch](https://github.com/karpathy/autoresearch):

```text
research/
  prepare.py      # fixed data/checkpoint/evaluation support
  train.py        # the one editable experimental surface
  program.md      # rules for an autoresearch agent
  results.tsv     # append-only result ledger
```

`research/train.py` owns:

- experiment configuration;
- projector, DFM, and vector-JEPA definitions;
- loss construction;
- optimizer construction;
- train and validation steps;
- the fixed-duration loop; and
- the final result record.

Stable board encoding, policy-index conversion, archive reading, checkpoint
adaptation, GPU profiling helpers, and arena rules remain outside the editable
surface.

The clean path initially excludes:

- TPU and multi-host initialization;
- GCS streaming;
- cloud job queues;
- deprecated stage-2 contrastive paths;
- placeholder metrics;
- legacy token-JEPA implementations; and
- implicit `/tmp` writes.

### Parity gate

- Load or explicitly convert the recovered checkpoint.
- Match reference outputs within documented BF16 tolerance.
- Complete a 100-step GPU smoke train.
- Resume deterministically.
- Avoid recompilation during steady-state static-shape training.
- Leave no files outside the workspace boundary.

## Phase 2: fix the objective and collapse measurements

### Batch-invariant SIGReg

The official LeJEPA Epps-Pulley statistic multiplies its empirical discrepancy
by the valid sample count. Target and prediction SIGReg also see different
numbers of vectors in this model.

Keep the official statistic as a diagnostic, but train with:

```text
unscaled_sigreg = official_ep / valid_count
```

This is the empirical-characteristic-function discrepancy used in LeJEPA's
finite-sample analysis and in the later world-model formulation, rather than
the sample-count-scaled goodness-of-fit test statistic. An optional fixed
`reference_count` is only a coefficient-translation convenience:

```text
training_sigreg = unscaled_sigreg * reference_count
```

It makes exact duplication and padding invariant. It does not remove the
finite-sample bias and variance of the V-statistic for independently sampled
batches. Measure that effect across physical batch sizes before freezing the
loss; if material, compare the unbiased U-statistic correction as a controlled
follow-up.

Required tests:

- duplicating every example leaves normalized loss and gradient unchanged;
- invalid horizons do not affect the statistic;
- target and prediction reference scales are explicit; and
- the raw official statistic remains available for comparison.

Epps-Pulley is non-additive across independently evaluated microbatches.
Gradient accumulation must not average per-microbatch SIGReg losses. The first
correct implementation either:

1. disables gradient accumulation while SIGReg is active; or
2. aggregates global cosine/sine/count sufficient statistics before forming
   the loss, using a differentiable two-pass or equivalent implementation.

Only the second implementation may claim invariance to microbatch partitioning,
and it requires a dedicated loss-and-gradient parity test.

Before selecting a coefficient, measure unweighted gradient norms and pairwise
gradient cosines for DFM CE, JEPA prediction, target SIGReg, prediction SIGReg,
and legality on the backbone, DFM, and JEPA parameter groups. Sweep coefficients
that make the auxiliary gradient roughly 1%, 3%, 10%, and 30% of the main
gradient. The legacy coefficient is not transferred blindly after removing the
sample-count multiplier.

The first 3%-scale target point (`0.40` with reference count one) was
numerically stable but allowed target and prediction RMS to contract while raw
JEPA MSE fell. The next baseline calibration must include the 10% and 30%
gradient points and the batch-64 legacy-equivalent strength (`5.76` with
reference count one). Selection uses scale-independent prediction quality and
latent-scale/rank gates, not raw MSE alone.

### Prediction-collapse ablations

The compatibility loss adds the following per-state, per-horizon term directly
to raw JEPA MSE:

```text
abs(log(rms(z_pred)) - stop_gradient(log(rms(z_target))))
```

This term prevents prediction scale from separating from target scale, but it
does not prevent joint target/prediction contraction. The recovered checkpoint
and repeat-qualified compatibility control currently have healthy prediction
variance/rank and a strong action-shuffle gap, so the norm term may contribute
to prediction health but has not been isolated as its cause.

The corrected-objective hypothesis is:

```text
loss = policy_ce
     + raw_jepa_mse
     + target_sigreg_coeff * normalized_sigreg(z_target)
     + pred_sigreg_coeff * normalized_sigreg(z_pred)
     + legality
```

Remove the RMS-matching term while leaving target attachment, rollout,
initialization, and all other loss semantics unchanged. This tests absolute
distributional regularization of both target and prediction latents instead of
relative norm matching.

The earlier prediction-SIGReg `0.57` rejection does not test this replacement:
that run retained the norm term and added a roughly 3%-of-JEPA-group-gradient
regularizer. Keep it rejected as an add-on to the compatibility objective, but
do not transfer `0.57` to the no-norm objective. Removing the norm term changes
the primary gradient, so both target and prediction SIGReg coefficients must be
audited again after selecting the physical batch.

Use three sequential, matched short screens rather than paying for a new full
grid:

1. norm on, prediction SIGReg off: compatibility control;
2. norm off, prediction SIGReg off: expose unregularized collapse pressure;
3. norm off, prediction SIGReg on: test the calibrated replacement.

Record raw and weighted scalar contributions, per-component gradient norms and
cosines, and their trajectories over the first 50--100 updates. Coefficients
should keep weighted terms in a comparable order of magnitude while preventing
auxiliary gradients from dominating the primary gradient. A rapidly falling
SIGReg value is expected and must be judged over the early trajectory, not only
at update zero.

The current implementation pools valid prediction tokens across horizons for
prediction SIGReg. Retain that implementation for the first isolated test, but
gate and report scale, rank, and action coupling separately at every horizon.
If pooled SIGReg hides a horizon-specific failure, compare per-horizon SIGReg
as a subsequent controlled change. If Gaussian marginal shape is not the
failure mode, prefer a targeted action-coupling loss; a per-horizon
VICReg-style standard-deviation hinge remains the first fallback for actual
low-variance collapse.

The current future target is produced by the same trainable encoder/projector
and is not stop-gradient. That makes joint target/predictor scale contraction a
real shortcut. Preserve this behavior for the compatibility baseline, then
compare a stop-gradient target and an EMA target encoder as explicit research
ablations; do not silently change target-gradient semantics while calibrating
SIGReg. The controlled EMA `0.99` implementation now passes the provisional
scale/rank gates, but it regresses fixed-slice policy and coupling metrics and
costs about `17%` throughput, so it remains an available ablation rather than
the promoted baseline. The per-horizon variance-hinge comparison is also
complete: coefficient `1.75` was the smallest measured point that crossed the
old provisional 95% target-RMS-retention threshold, while `3.75` fully
stabilized scale and coupling. Both regress DFM CE and legal mass, so neither
is promoted. The 95% threshold is retained only as historical calibration
context, not as the corrected objective's acceptance gate. Measured details
are in `docs/local_gpu_baseline.md`.

Those EMA and hinge comparisons used the original first-shard validation
slice. They remain mechanistic calibration evidence, but not representative
quality or promotion evidence after the validation-schedule audit.

Fixed unit-RMS states are now implemented as a default-off ablation with FP32
per-state statistics, no batch statistics, and no trainable scale. They remove
uniform target/prediction shrinkage without rank collapse or measurable
throughput cost. On the corrected matched 32-batch global validation sample,
however, unit RMS `+1.50` does not improve policy over online `5.76`; both
configurations substantially regress DFM CE, accuracy, and legal mass over 100
updates. Neither is the clean baseline.

Commit `8c2dcb6` fixes validation selection by deterministically permuting all
`(shard, batch_in_shard)` slots. Before this commit, two-batch studies always
read the first 128 examples of the first validation shard, and the 16-batch
expansion only exhausted that same shard. Those studies are now explicitly
forensic/calibration-only. Every new objective comparison must use the same
seeded global validation slots for control and candidate; a sampler or metric
change requires a matched control rerun.

The continuation audit adds two further constraints. The source TPU optimizer
was trained at global batch `8,192`; local batch 64 is 128 times smaller, and
legacy `--init exact` restores model/optimizer state but not the source input
cursor or PRNG stream. The source history also places step 265,000 after the
best of its nine recorded validation points. Exact-optimizer continuation at
the source learning rates, a ten-times-lower exact-optimizer continuation, and
a policy-only continuation all regress matched global policy metrics.
Policy-only was reverted.

The leading initialization candidate is therefore model-only import with a
fresh optimizer, constant main/BT4 rates `3e-5`/`1e-6`, and zero warmup.
The short batch-128 repeats first exposed `0.00394` final-CE dispersion and a
target-scale miss. A completed 30-minute v1 then produced 557 updates and six
saved checkpoints. A read-only scan on seeds 10,000 and 20,000 selected update
300 over the near-tied update 400 by primary CE; updates 500 and 557
regressed.

Two additional pools at seeds 30,000 and 40,000 preserve that tie-break.
Across all four pools, update 300 changes DFM CE
`4.550008280 → 4.529262789` (`-0.020745492`), accuracy
`0.106750488 → 0.107345581`, and legal mass
`0.642926642 → 0.644614464`. CE improves at every horizon. Update 400 is
`0.000801284` worse in aggregate CE but better on aggregate accuracy/legal
mass, and update 300's action metrics do not improve on every individual pool.

The identical 30-minute v2 repeat also produced 557 updates and independently
selected update 300. Across the same four pools, v2/u300 changes DFM CE
`4.550008280 → 4.529473042` (`-0.020535238`), accuracy
`0.106750488 → 0.108810425`, and legal mass
`0.642926642 → 0.646496401`; every horizon improves. Its primary gain differs
from v1/u300 by only `0.000210253`. V2/u300 is therefore the
repeat-qualified offline baseline.

The full latent audit does not show prediction collapse: mean effective rank
changes `31.4294 → 31.0318` with a minimum of `29.1529`, mean fifth-percentile
feature standard deviation changes `0.6876 → 0.6644` with a minimum of
`0.6515`, and the prediction/target RMS ratio changes `0.9600 → 0.9853`.
Positive prediction beats zero and action-shuffled controls at every horizon.
The v1 endpoint's absolute target RMS nevertheless retained only `94.21%` of
its initial value; scale behavior remains a research target rather than a
reason to relabel the selected checkpoint as promoted.

This is a baseline-qualification result, not an accepted autoresearch
experiment. It has no `research/results.tsv` row, promotion-eligible Elo
result, or promotion decision. A matched 30-minute prediction-SIGReg `0.57`
experiment was rejected: its best two-pool CE is `4.511721` versus incumbent
v2/u300 `4.510334`, and matched update-400 policy, legality, and JEPA metrics
regress for only tiny rank/variance gains. Prediction-SIGReg therefore stays
`0.0` in the compatibility control; its role as a replacement for the norm
term remains untested.

Static-batch arena pilots at caps 64, 128, and 256 eliminated timeout faults;
only cap 256 completed all 32 games without cap adjudication, making it the
first pilot-valid limit. Its 16-pair point estimate is `0` descriptive Elo
with a deliberately broad `[-287.451, +287.451]` interval, not a strength
decision. Commit `c2d5efb` replaces repeated hot-path history replay with
sealed O(1) endpoint validation and passes 83 CPU tests with exact gameplay
payload parity. GPU access is restored; real-checkpoint GPU parity, optimized
timing, and the full 128-pair screen remain unmeasured. Unattended search
remains disabled.

This scale/shape split is also motivated by
[VISReg](https://arxiv.org/abs/2606.02572), which argues that sketching
regularizers can have weak gradients near collapse and retains a separate
VICReg-style variance term for scale control. It is supporting evidence for
the comparison, not an equivalent objective: VISReg uses sliced Wasserstein
for distribution shape, whereas this baseline retains Epps-Pulley SIGReg.

Collapse measurements are computed per horizon before aggregation:

- target and prediction RMS;
- mean, minimum-quantile, and median feature standard deviation;
- covariance spectrum, effective rank, and participation ratio;
- delta RMS and cosine similarity;
- raw MSE and norm loss;
- zero, identity, shuffled-target, and action-shuffled baselines;
- teacher-forced versus free-state-rollout gap; and
- gradient norms by loss and module.

### Gate

- Loss and gradients are invariant to batch layout within numerical tolerance.
- Removing the explicit valid-count multiplier does not justify treating
  finite-sample SIGReg as independent of the sampled batch. Its value and
  gradient dispersion are measured across candidate physical batch sizes.
- Quality comparisons use seeded globally permuted validation slots; a
  first-shard slice is only a correctness/calibration instrument.
- Target and predicted latents remain inside an empirically calibrated
  representation-health envelope based on repeated-control variability:
  absolute RMS by horizon, prediction/target RMS ratio, effective rank,
  low-tail feature standard deviation, and normalized SIGReg discrepancy.
- The old 95% mean-target-RMS retention threshold is a diagnostic only, not a
  hard promotion gate. Aggregate RMS must not hide horizon- or
  feature-specific failure.
- JEPA beats trivial baselines at useful horizons.
- Positive predictions beat zero and action-shuffled predictions at every
  useful horizon.
- Policy CE, accuracy, and legal mass remain within predeclared
  repeated-control tolerances.
- Legal probability mass is accumulated in FP32 and bounded to `[0, 1]`; the
  legality penalty cannot become negative through BF16 summation error.
- Loss clipping is not permanently active.
- Validation metrics contain no zero placeholders.

Once this gate passes, the loss definition is frozen for the initial
architecture search.

## Phase 3: profile and optimize the A10G path

The first profile must measure the active compiled graph. Handwritten estimates
for the older token-JEPA are not accepted.

Annotate:

- shard reading and decompression;
- host-to-device transfer;
- current/future BT4 encoding;
- DFM noisy pass;
- DFM clean hidden pass;
- JEPA rollout;
- target and prediction regularization;
- backward pass; and
- optimizer update.

Report:

- compile/startup time separately;
- p50 and p95 steady-state step time;
- examples/s and encoded boards/s;
- peak and reserved HBM;
- compiler-reported FLOPs;
- achieved dense-BF16 TFLOP/s relative to the A10G reference peak;
- host and device idle time; and
- batch-one and batched inference latency.

The local CUPTI/JAX trace path currently faults with
`CUDA_ERROR_ILLEGAL_ADDRESS` on this driver combination. Until that is resolved,
use XLA cost/memory analysis plus 100 ms `nvidia-smi` utilization, memory,
power, and clock samples. Do not let an unavailable profiler block
wall-clock/throughput optimization or silently omit the limitation.

Batch 128 is the measured incumbent, not the selected final batch. First
implement the no-norm objective with nonzero target and prediction SIGReg so
the profile includes the intended compiled work. Then profile physical batches
64, 128, 192, and 256, stopping at OOM, compile instability, or a clear
throughput plateau. For every point, separate compilation from steady state
and record end-to-end examples/s, device examples/s, input stall, HBM,
utilization, power, and clocks.

Select the throughput knee rather than the largest batch that fits. Then verify
learning efficiency at fixed wall time, because a larger batch produces fewer
optimizer updates and may require a separate learning-rate comparison. Keep
validation fixed at 64-example batches over identical positions and finite
sample partitions. Exact SIGReg coefficients are calibrated only after this
physical batch is selected; a representative nonzero coefficient is sufficient
to include the correct work in the hardware profile.

Optimization sequence:

1. no-norm target-plus-prediction-SIGReg compiled graph;
2. physical batch feasibility and end-to-end throughput;
3. selected-batch learning rate and fixed-time learning efficiency;
4. real gradient accumulation, only with correct global SIGReg statistics;
5. input prefetch and pinned-transfer behavior;
6. future-encoder chunking/rematerialization;
7. sampled future targets with `K=1,2,8`;
8. BF16/FP32 boundaries;
9. scan/unroll choices; and
10. only then architecture changes.

The main expected lever is avoiding nine trainable BT4 encodes per sample on
every step. Future-horizon sampling must be unbiased and compared at fixed
wall-clock and fixed-example budgets.

The first host-path optimization is complete. Commit `bf18a5b` slices encoded
rows before plane/legal expansion for globally permuted batches and reports
device-only, fetch-inclusive, and whole-loop throughput separately. On the
A10G, batch 64 sustains about `33.0` fetch-inclusive examples/s with a
`16.7–16.9%` input-stall fraction. Batch 128 sustains about `41.4`
examples/s, roughly 25% faster, at a `7.5%` input-stall fraction and about
`9.70 GB` peak JAX live memory. Commit `455a806` decouples validation batch
size so training-batch experiments can retain the same positions and
finite-sample partition.

## Phase 4: inference and relative Elo

External Stockfish, puzzle, and opening datasets are not required.

The approved state/action shards also contain per-horizon scalar
`value_targets` in `{-1,0,1}` and one-hot `wdl_targets`. They are final-outcome
labels rather than search evaluations, but they are sufficient for a later
controlled value/WDL-head training and candidate-ranking experiment. The
compatibility baseline leaves those coefficients at zero, and the recovered
head has never been outcome-supervised.

Each horizon label is from the side to move after that action, so its root-side
sign alternates with horizon parity. Train or evaluate with this convention
made explicit; never average raw per-horizon scalar values. The first clean
value experiment freezes/detaches the representation and trains only the head,
compares target-latent and predicted-latent inputs, validates label shape and
one-hotness fail-closed, and initially ranks with calibrated WDL expected score.
These played-continuation outcomes are not counterfactual action values, so any
candidate-ranking benefit still requires paired-arena validation.

### Frequent offline evaluation

- fixed globally permuted validation DFM cross-entropy, with the schedule and
  effective seed recorded in the report;
- first-move and per-horizon top-1/top-k accuracy;
- legal mass;
- JEPA positive and collapse metrics;
- compilation, throughput, and memory; and
- the full DFM refinement trajectory.

The refinement trace records:

- entropy and legal mass at every pass;
- KL/JS divergence between consecutive passes;
- rank and probability path of the final selected move;
- top-k turnover;
- action-sequence edit distance; and
- the first pass at which the final move appears.

### Batched arena

Use fixed early-game FENs from a held-out trajectory split. Play every FEN with
colors swapped. Initial anchors are:

- raw BT4, when its policy checkpoint is available;
- recovered step-265,000 model;
- previous promoted model; and
- current candidate.

Before any games, make action semantics explicit:

- existing trajectory labels, legal masks, and the recovered DFM use
  `legacy_absolute_1858`;
- the native BT4 policy head uses board-aware `lc0_canonical_1858`; and
- every model manifest records its codec ID.

The canonical codec mirrors black moves into side-to-move coordinates, treats
knight promotion as the ordinary from-to slot, handles queen/rook/bishop
suffixes, and decodes by uniquely matching board-legal moves. Golden
white/black pairs, every promotion type, legal-mask cardinality, round trips,
and mirrored-board properties must pass before the arena is trusted. Do not
permute recovered weights: white is identity while black requires a
board-dependent transformation.

The frequent arena is an in-process, batched GPU evaluator rather than a serial
UCI tournament. UCI remains a correctness and interoperability path.

Each opening is played as a color-reversed pair. The pair is the sampling unit;
record pentanomial `LL/LD/DD-or-LW/WD/WW` outcomes. An illegal move, timeout,
exception, or non-finite result is a loss—never a silent fallback to raw BT4.
Use normal chess outcomes and a symmetric ply-cap draw; report the cap rate.

Use a frozen, hash-selected development pool and a separate promotion pool from
held-out trajectory game/FEN groups. No external Stockfish opening or puzzle
dataset is needed. Store pool checksums and selection seeds.

The held-out shard audit found that persisted `game_id` is a constant
placeholder and therefore cannot define groups safely. Pool construction
instead uses exactly `ply == 12`, which yields one position after six full
moves per source game, rejects invalid or terminal standard-chess FENs before
canonicalization, deduplicates exact six-field FENs, and hash-ranks the
remainder. The real inventory is:

- validation: 15,101 ply-12 rows, 14,553 valid standard positions, and 12,297
  unique FENs;
- test: 15,040 ply-12 rows, 14,484 valid standard positions, and 12,227 unique
  FENs; and
- 1,404 FENs overlap the two splits, leaving 10,823 root-FEN candidates after
  excluding the entire valid validation universe but before history validation.

About 3.6% of the ply-12 rows have Chess960-like castling metadata invalid
under standard-chess rules and are excluded rather than repaired. The
full-history reconstruction audit found another 36 test roots whose preceding
game was nonstandard even though the ply-12 root itself parsed as standard.
Those complete histories are excluded too, leaving 10,787 promotion candidates.
The development pool is 128 validation FENs; its first 16 form the correctness
tier. The initial promotion pool contains 2,048 test FENs after
excluding every valid validation candidate and all 36 nonstandard histories,
not merely the selected development subset.

Exact replay subsequently found a distinct root-state defect that the original
root-FEN terminality check could not see: selected v2 promotion entry 1,245 was
already claim-draw terminal by threefold repetition. Commit `5a49df9`
regenerates the ordered pool from the immutable shards, rejecting eight
hash-ranked roots during exact-history filtering: four histories that do not
start from standard chess, three with invalid standard-history FENs, and the
threefold-terminal root. V3 differs from v2 by exactly one selected removal
and one source-derived replacement. Commit `12147e8` pins the repaired pool
and sidecar in the evaluator; no position is skipped or substituted at arena
runtime.

Every arena root has a full standard-initial-position-to-root history sidecar.
The gameplay runner replays it exactly so repetition and draw-claim state are
real rather than inferred from a root FEN. This history is deliberately not fed
to the recovered model: the source trajectory preprocessor encoded current and
future positions with `encode_board(board, [])`. Current-only encoding exactly
matches the stored planes for all 2,176 selected development and promotion
roots; a 224-row audit found that history-aware encoding mismatched every stored
example. The model adapter therefore records
`plane_history_mode=current_only_as_preprocessed`. History-aware BT4 inputs are
a future distribution-changing experiment, not a silent evaluator correction.

Report model-pool relative logistic and normalized Elo with pair-aware 95%
uncertainty, game count, score breakdown, FEN set digest, refinement count,
candidate count, latency, and games/s. Never label it as human or Lichess Elo.

The first strength milestone is not promotion. It is a decision-useful relative
Elo estimate under an identical searchless inference budget against both:

- the recovered step-265,000 DFM/JEPA source checkpoint, which anchors local
  training progress; and
- the original self-play BT4 policy, which anchors the representation study.

Validate the raw-BT4 policy adapter and board-aware codec before treating that
second comparison as evidence. Use the correctness tier first, then the
128-pair screen for checkpoints that pass the offline gates. The broad 16-pair
interval is pipeline evidence only.

Arena tiers:

1. 16 opening pairs for correctness only;
2. 128 pairs / 256 games as a quick large-effect gate; and
3. a promotion GSPRT on the fresh pool with `H0=0`, `H1=+20 normalized Elo`,
   `alpha=beta=0.05`, checked after complete pairs and capped at 4,096 games.

The promotion implementation uses the official Fishtest
constrained-multinomial normalized-Elo likelihood, including the pentanomial
pair-to-game `sqrt(2)` conversion. It is distinct from descriptive logistic Elo
and is the only state allowed to report `promotion_eligible=true`.

Commit `a2ea5ed` adds the resumable checkpoint-versus-checkpoint command with
strict checkpoint/code/pool/history contracts, immutable checksummed
pair-blocks, atomic state publication, pair-boundary stopping, codec/fault
accounting, and a lean diagnostics-off inference path. Correctness and
development remain available. Promotion is explicitly `available=false` and
fails before model loading until the root-threefold defect is repaired. A
strength-valid long-game cap also remains to be calibrated.

Inconclusive is not evidence of equality. Small improvements should be combined
through validation and repeated training runs before paying for a promotion
arena.

The first implementation benchmark determines how many games fit in the
promotion budget. No fixed game count is assumed before measuring it.

## Phase 5: 30-minute autoresearch loop

Compilation and setup are measured separately. Each quick experiment receives
30 minutes of steady-state training on the A10G.

Every result records:

- git commit;
- complete configuration;
- seed and data-order digest;
- start and end checkpoints;
- validation DFM CE;
- action accuracy and legality;
- JEPA/collapse gate metrics;
- examples processed;
- examples/s;
- compile time;
- peak HBM; and
- keep/reject/confirm disposition.

Until a baseline is frozen, use `eval_batch_size=64`, `eval_batches=64`, and
the same 4,096 globally permuted positions for matched comparisons. Seed
10,000 is the development slice and seed 20,000 is the first independent
confirmation slice. A change to training batch size must not silently change
validation batch size, count, positions, or finite-sample partition.
Checkpoint selection now uses the mean of those two pools; when updates 300
and 400 were nearly tied, seeds 30,000 and 40,000 were added as an explicitly
matched tie-break rather than consulted one candidate at a time.

Promotion policy:

1. smoke compile and short correctness run;
2. repeat the candidate baseline until validation noise is measured;
3. complete a matched 30-minute baseline-qualification run;
4. open readiness only if its repeats pass every offline gate;
5. give each subsequent quick experiment one 30-minute run;
6. repeat or extend ideas exceeding baseline noise without failing a gate;
7. run batched relative-Elo evaluation; and
8. run full-data training only for confirmed candidates.

Primary optimization signal is held-out DFM CE, with action metrics and hard
JEPA-collapse gates. The mutable weighted training loss is not by itself a
promotion metric.

Initial experiment order:

1. remove RMS norm matching and replace it with calibrated prediction SIGReg;
2. select physical batch size and learning rate from active-graph profiling;
3. freeze a repeat-qualified 30-minute corrected baseline;
4. establish relative Elo against the recovered model and raw BT4;
5. future-target sampling;
6. teacher-forced versus free-state rollout schedules;
7. projector width, depth, and latent dimension;
8. DFM width, depth, and refinement training;
9. recurrent versus direct multi-horizon JEPA;
10. optimizer and BT4 learning-rate/freeze schedules; and
11. closed-loop latent-conditioned DFM refinement.

## Phase 6: representation and SAE study

The exact tensor, hook, source-parity, metric, and intervention contract is
[the BT4 sparse-replacement and representation study](sae_representation_plan.md).
The published Leela artifacts are MLP transcoders and LoRSA attention branch
replacements, not final-trunk SAEs. They operate on raw pre-`alpha` branch
outputs and their corresponding inputs. Source-model FP32 reconstruction and
cross-framework hook parity must pass before they can support claims about
backbone drift.

This phase is deliberately off the GPU critical path until the corrected model
has a decision-useful relative Elo estimate against the recovered DFM/JEPA
checkpoint and the raw BT4 anchor. Do not run SAE training concurrently with
model training, profiling, or arena evaluation; the single A10G performs one
of these workloads at a time. Elo-aligned checkpoints, rather than merely the
lowest-loss checkpoint, determine the first representation comparison set.

Use an identical fixed board/trajectory corpus and consistent hook semantics for:

- original self-play BT4;
- recovered step-265,000 model;
- intermediate promoted checkpoints;
- final JEPA-trained model; and
- scratch or alternative-pretraining controls.

First compare:

- linear CKA;
- SVCCA/Procrustes alignment;
- layer correspondence;
- covariance spectra and effective rank; and
- policy behavior.

Then apply the published BT4 transcoders/Lorsa dictionaries:

- reconstruction fidelity and explained variance;
- activation sparsity and dead features;
- decoder/feature alignment across checkpoints;
- concept selectivity;
- causal effects on policy logits; and
- features gained, lost, split, or merged.

Published dictionaries are transfer probes. A degradation in reconstruction
after backbone training is itself a result, but a fair final comparison also
requires matched dictionaries trained separately for each backbone.

The first pass therefore progresses from dense alignment, to fixed published
artifact transfer, to causal branch replacement, and only then to matched
refits. The representation contract freezes input-plane bytes, square order,
layer/hook semantics, residual scale and epsilon, dtype, sparse normalization,
corpus, and game-level resampling. It also records the unresolved mismatch
between the upstream BT4 config epsilon and the custom layer's default.

The current `z_pred` is not an input to DFM action inference. Representation
results may explain how JEPA gradients reshape the shared BT4/DFM system, but
must not be described as evidence for future-latent-conditioned action choice.

Save checkpoints on a log-spaced schedule so representation changes can be
aligned with validation loss, relative Elo, collapse recovery, and feature
emergence.

## Later work

Only after the baseline and promotion harness are stable:

- compare self-play BT4 initialization against scratch or matched alternative
  pretraining;
- run controlled size/data/time/regularization grids with repeated seeds;
- train a WDL/value head from the available final-result targets;
- rank fixed-count DFM candidates using predicted value;
- investigate stronger action/latent contrastive coupling; and
- add self-play RL with replay, opponents, and checkpoint promotion.

Claims of grokking, phase transitions, or double descent require predefined
sweeps, held-out data, and repeated seeds.

## Primary references

- [LeJEPA paper](https://arxiv.org/abs/2511.08544) and
  [official Epps-Pulley implementation](https://github.com/galilai-group/lejepa/blob/main/lejepa/univariate/epps_pulley.py)
- [LeWorldModel](https://arxiv.org/abs/2603.19312), the closest published
  predictive-world-model use of unscaled SIGReg
- [VISReg](https://arxiv.org/abs/2606.02572) for explicitly separating
  variance/scale control from distribution-shape regularization
- [VICReg](https://arxiv.org/abs/2105.04906) and its
  [official implementation](https://github.com/facebookresearch/vicreg)
- [RankMe](https://arxiv.org/abs/2210.02885) for entropy-based effective rank
- [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
- [Fishtest mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
  and the
  [official normalized-LLR implementation](https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/LLRcalc.py)

## Immediate implementation checklist

- [x] Preserve the historical commit on `legacy/tpu-joint-latent-sasa`.
- [x] Create `research/local-gpu-autoresearch`.
- [x] Establish and test the workspace-local environment contract.
- [x] Ground and download the approved Drive assets.
- [x] Verify archive and checkpoint digests.
- [x] Recover the original checkpoint metadata and sidecars.
- [x] Produce the first strict local-GPU baseline fingerprint.
- [x] Create the one-editable-file compatibility scaffold.
- [x] Move the experimental model/optimizer into `research/train.py` and pass
  exact CPU and real-checkpoint GPU parity.
- [x] Move the loss into `research/train.py`, pass exact loss/gradient and
  real-checkpoint GPU parity, and
  remove the final legacy model/loss import.
- [x] Add initial duplication/padding invariance and collapse tests.
- [x] Capture the first active-model GPU profile.
- [x] Recompute legality from the original logits in FP32 with correct gradients.
- [x] Calibrate normalized SIGReg coefficients by per-module gradient norms.
- [x] Reject target SIGReg `0.40` after a fixed-slice stability run exposed
  target/prediction scale contraction.
- [x] Reject the calibrated 10%-gradient target SIGReg point `1.32` after the
  same scale shortcut persisted.
- [x] Evaluate target SIGReg `3.96` and batch-64 legacy-equivalent `5.76`;
  retain `5.76` as the best compatibility point but reject both for the clean
  scale-stability gate.
- [x] Add and real-GPU verify strict atomic checkpoint/resume with deterministic
  data-cursor continuation.
- [x] Replace the unsafe initial legacy loader with a checksummed, restricted,
  exact-ABI pinned-source import boundary.
- [x] Pin and preflight the complete local-GPU runtime before every run.
- [x] Run and reject the simple positive-target stop-gradient ablation: it
  preserves scale but weakens prediction/action/policy quality.
- [x] Run and reject calibrated 10%-gradient prediction SIGReg: prediction
  rank was already healthy and policy quality regressed.
- [x] Implement, exact-parity test, and profile the controlled EMA target; it
  passes scale/rank gates, regresses policy/coupling on the forensic
  first-shard slice, and costs about `17%` A10G throughput; it is not promoted.
- [x] Implement, calibrate, and compare a per-horizon variance hinge. It can
  stabilize scale and coupling without measurable throughput cost, but every
  scale-passing point regresses DFM CE or legal mass on the forensic
  first-shard slice and is not promoted.
- [x] Replace first-shard validation selection with a seeded stateless global
  batch-slot permutation, exact epoch coverage, and recorded provenance;
  reclassify every earlier first-shard result as calibration/forensic only.
- [x] Make effective experiment overrides explicit and reject silent no-op
  configuration before enabling unattended autoresearch.
- [x] Remove always-zero legacy placeholders from experiment reports while
  preserving the exact parity oracle and failing closed on metric drift.
- [x] Implement and calibrate default-off fixed unit-RMS target/prediction
  states. They stabilize scale and coupling but do not improve the matched
  global policy result, so they are not promoted.
- [x] Audit source continuation semantics: record the TPU global-batch/local
  batch mismatch, show that the recovered checkpoint is past its best recorded
  validation point, and distinguish exact model/optimizer import from source
  cursor/PRNG continuation.
- [x] Reject source-rate exact-optimizer continuation, ten-times-lower
  exact-optimizer continuation, and policy-only continuation on matched global
  validation.
- [x] Isolate model-only initialization with a fresh constant-rate optimizer
  and zero warmup; retain it only as a provisional local initialization.
- [x] Slice globally permuted encoded batches before expansion and expose
  accelerator-only, fetch-inclusive, and whole-loop throughput. Benchmark
  batch 64 and 128 on the active A10G graph.
- [x] Decouple evaluation batch size from training batch size and rerun the
  batch-128 learning-rate comparison on the same 4,096-position,
  batch-64-partitioned validation population.
- [x] Reject linear learning-rate scaling at batch 128 and record the full
  collapse/coupling diagnostics plus an independent validation seed for the
  unscaled-rate checkpoint.
- [x] Complete the first 30-minute fresh-optimizer batch-128 run and evaluate
  every saved checkpoint read-only on seeds 10,000 and 20,000.
- [x] Tie-break updates 300 and 400 on new seeds 30,000 and 40,000. Select
  update 300 provisionally: its four-pool CE improves by `0.020745492`, all
  horizons improve, and aggregate accuracy/legal mass also improve.
- [x] Complete and scan the identical v2 repeat, quantify checkpoint-selection
  noise, and run the full latent gate on the selected repeat checkpoint.
  V2 independently selects update 300; its four-pool CE gain is within
  `0.000210253` of v1, and the recorded latent diagnostics do not show collapse.
- [x] Designate v2/update 300 as the repeat-qualified offline baseline. This
  does not append a ledger row, open unattended search, claim Elo, or promote
  the checkpoint.
- [x] Run and reject the baseline-length prediction-SIGReg `0.57` experiment.
  Its best checkpoint loses primary two-pool CE to v2/u300, and a
  checkpoint-age-matched u400 audit trades policy/legal/JEPA regressions for
  only tiny diversity gains.
- [ ] Make the per-state RMS norm term explicitly configurable, preserve exact
  compatibility behavior when enabled, and test the no-norm raw-MSE path.
- [ ] Validate the no-norm target-plus-prediction-SIGReg objective, including
  disabled-path parity, duplication/padding invariance, finite-sample metrics,
  and per-horizon collapse/coupling diagnostics.
- [ ] Profile physical batches 64, 128, 192, and 256 on the active no-norm
  target-plus-prediction-SIGReg graph; choose the A10G throughput knee before
  fixing the training batch, while retaining validation batch 64.
- [ ] Re-audit target and prediction SIGReg scalar contributions, gradient
  norms, and gradient cosines on the selected graph and physical batch. Do not
  transfer target `5.76` or prediction `0.57` blindly.
- [ ] Run matched short screens for norm-on/pred-SIGReg-off,
  norm-off/pred-SIGReg-off, and norm-off/pred-SIGReg-on. Advance only the
  corrected objective that passes policy, legality, action-coupling, and
  representation-health checks.
- [ ] Repeat-qualify a 30-minute corrected baseline and freeze its loss,
  physical batch, learning rate, validation contract, and noise envelope
  before enabling unattended architecture research.
- [x] Implement and golden-test separate legacy-absolute and board-aware LC0
  canonical 1,858 action codecs.
- [x] Implement deterministic paired-arena foundations, audit the real
  held-out FEN universe, and construct disjoint development/promotion pools.
- [x] Implement and profile cached-BT4 batched multi-pass DFM inference.
- [x] Implement the fail-closed persistent batched paired-opening arena, strict
  localized checkpoint policy adapter, and bounded GPU policy batches.
- [x] Implement the normalized-Elo promotion GSPRT and match the official
  Fishtest likelihood numerically.
- [x] Run the 16-pair source-vs-source A10G correctness tier with identical
  move traces, zero faults, and exact 50/50 paired scoring.
- [x] Add the resumable relative-strength command with immutable pair blocks,
  strict resume contracts, pair-boundary GSPRT stopping, and fail-closed model
  and codec accounting.
- [x] Run v2/u300 versus source through the 16-pair correctness tier: zero
  faults and full evaluated-action coverage, but 32/32 games are short-cap
  draws and provide no strength evidence.
- [x] Rerun static-shape development pilots at caps 64, 128, and 256 with zero
  faults. Caps 64/128 remain censored; cap 256 has 0/32 cap draws and is the
  first pilot-valid limit.
- [x] Add the sealed O(1) trusted-history arena path in commit `c2d5efb` and
  establish exact CPU gameplay-payload parity across 83 focused tests.
- [ ] Establish real-checkpoint GPU parity and timing for the sealed-history
  path, then run full fault-free 128-pair screens at cap 256 against the
  recovered DFM/JEPA checkpoint and, after adapter validation, raw BT4. The
  pre-optimization pilot timings are not evidence of optimized speed.
- [x] Regenerate and repin the promotion pool after exact replay found entry
  1,245 claim-draw terminal. Two v3 source regenerations were byte-identical,
  all 2,048 final histories pass the production replay loader, selected
  validation/test overlap is zero, and the evaluator is reopened on the
  immutable v3 pins.
- [ ] Keep SAE training and representation interventions deferred until those
  relative-Elo anchors exist; thereafter schedule all SAE GPU work
  sequentially after training and arena jobs.
