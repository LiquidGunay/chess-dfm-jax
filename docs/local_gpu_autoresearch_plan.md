# Local GPU Autoresearch Plan

Status: approved for implementation on 2026-07-18.

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

Run a paired 2×2 target-SIGReg off/on × prediction-SIGReg off/on ablation.
Prediction SIGReg is not the new default: the recovered checkpoint currently
has healthy prediction variance/rank and a strong action-shuffle gap. Accept it
only if it improves prediction stability or quality without weakening action
sensitivity, policy metrics, or relative Elo. If Gaussian marginal shape is
not the failure mode, prefer a targeted action-coupling loss. A per-horizon
VICReg-style standard-deviation hinge is the first fallback for actual
collapse.

The current future target is produced by the same trainable encoder/projector
and is not stop-gradient. That makes joint target/predictor scale contraction a
real shortcut. Preserve this behavior for the compatibility baseline, then
compare a stop-gradient target and an EMA target encoder as explicit research
ablations; do not silently change target-gradient semantics while calibrating
SIGReg.

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
- Predicted latents retain nontrivial variance and effective rank.
- JEPA beats trivial baselines at useful horizons.
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

Optimization sequence:

1. batch and microbatch size;
2. real gradient accumulation;
3. input prefetch and pinned-transfer behavior;
4. future-encoder chunking/rematerialization;
5. sampled future targets with `K=1,2,8`;
6. BF16/FP32 boundaries;
7. scan/unroll choices; and
8. only then architecture changes.

The main expected lever is avoiding nine trainable BT4 encodes per sample on
every step. Future-horizon sampling must be unbiased and compared at fixed
wall-clock and fixed-example budgets.

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

- fixed validation DFM cross-entropy;
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

Report model-pool relative logistic and normalized Elo with pair-aware 95%
uncertainty, game count, score breakdown, FEN set digest, refinement count,
candidate count, latency, and games/s. Never label it as human or Lichess Elo.

Arena tiers:

1. 16 opening pairs for correctness only;
2. 128 pairs / 256 games as a quick large-effect gate; and
3. a promotion GSPRT on the fresh pool with `H0=0`, `H1=+20 normalized Elo`,
   `alpha=beta=0.05`, checked after complete pairs and capped at 4,096 games.

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

Promotion policy:

1. smoke compile and short correctness run;
2. one 30-minute run;
3. repeat or extend ideas exceeding baseline noise without failing a gate;
4. batched relative-Elo evaluation; and
5. full-data run only for confirmed candidates.

Primary optimization signal is held-out DFM CE, with action metrics and hard
JEPA-collapse gates. The mutable weighted training loss is not by itself a
promotion metric.

Initial experiment order:

1. loss normalization and prediction collapse;
2. future-target sampling;
3. teacher-forced versus free-state rollout schedules;
4. projector width, depth, and latent dimension;
5. DFM width, depth, and refinement training;
6. recurrent versus direct multi-horizon JEPA;
7. optimizer and BT4 learning-rate/freeze schedules; and
8. closed-loop latent-conditioned DFM refinement.

## Phase 6: representation and SAE study

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
- [VICReg](https://arxiv.org/abs/2105.04906) and its
  [official implementation](https://github.com/facebookresearch/vicreg)
- [RankMe](https://arxiv.org/abs/2210.02885) for entropy-based effective rank
- [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
- [Fishtest mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
  and the [official opening books](https://github.com/official-stockfish/books)

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
- [ ] Run the explicit positive-target stop-gradient stability ablation while
  retaining target-SIGReg gradients.
- [ ] Make effective experiment overrides explicit and reject silent no-op
  configuration before enabling unattended autoresearch.
- [ ] Freeze a corrected baseline objective after a stronger target-SIGReg
  stability run.
- [x] Implement and golden-test separate legacy-absolute and board-aware LC0
  canonical 1,858 action codecs.
- [ ] Implement the persistent batched paired-opening arena.
