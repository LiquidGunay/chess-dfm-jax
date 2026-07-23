# Local GPU Autoresearch Plan

Status: implementation in progress. The plan was approved on 2026-07-18 and
revised through 2026-07-23; the frozen loop is now enabled with
`AUTORESEARCH_READY = True`. Compatibility v2/update 300 remains the
repeat-qualified control. The no-norm target-SIGReg-5.76,
prediction-SIGReg-1.0 v2/update-400 checkpoint is now the repeat-qualified
corrected baseline. Its 128-pair searchless strength anchors are complete: it
is indistinguishable from the recovered DFM/JEPA source at this resolution,
but substantially weaker in point estimate than the original raw BT4 policy.
It is not Elo-promoted. The current offline incumbent is the one-block
future-gradient-tail primary/update 2,072; two exact runs beat balanced K=1 on
the frozen offline metric, while its direct 128-pair arena against balanced K=1
is indistinguishable from a tie and does not constitute Elo promotion.

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

Host-safety update, 2026-07-23: kernel evidence confirms that two overlapping
JAX processes caused a global OOM on 2026-07-22; the older process held about
4.7 GiB anonymous RAM and the duplicate held about 4.5 GiB. Process-namespace
visibility was not a safe concurrency check. `research/run_gpu.sh` now holds a
host-visible exclusive lock for the full prepare/compile/run lifetime and
launches through a fail-closed watchdog. The fixed ceilings are two CPUs,
7 GiB process-group RSS, an 8 GiB launch MemAvailable floor, a 3 GiB runtime
floor, and a 30 GiB projected disk reserve. Periodic checkpointing is
forbidden; future runs may write at most one sparse intermediate plus terminal
and must prune to one accepted state or zero rejected states before another
model run. Safety thresholds may only be made stricter through environment
configuration.

Execution update, 2026-07-20: critical-path steps 1--5 are complete. The norm
coefficient is explicit, normalized SIGReg uses a fixed random example count,
batch 128 is the selected throughput knee, and the matched objective screens
and two 30-minute corrected repeats are complete. The corrected loss, physical
batch, learning rates, validation population, and checkpoint-selection
contract are frozen. The corrected checkpoint scored `49.61%` against the
recovered source (`-2.7` descriptive logistic Elo, pair-aware 95% interval
`[-88.0,+82.2]`) and `38.09%` against raw BT4 (`-84.4`, interval
`[-181.0,+0.6]`). The representation-study strength precondition is therefore
satisfied. Stage-0/1 read-only representation work may begin, sequentially on
the A10G. The checkpoint-retention and free-space policy is now frozen and
machine-audited. This was the last state before source parity and experiment
preregistration subsequently opened the loop on 2026-07-21.

Execution decision, 2026-07-21: keep searchless inference fixed at eight DFM
refinement passes through the first stronger-model experiments. Eight equals
the current action horizon and is the frozen strength-anchor budget. A
`1/2/4/8/16` pass-count ablation is deferred until an offline improvement is
repeat-qualified and its strength screen is less ambiguous, so test-time
compute is not confounded with architecture or training changes.

Dense-representation update, 2026-07-21: the FP32 Stage-1 core comparison is
complete on all five hooks, all 15 layers, both square-token and board-pooled
views, and all six pairs in the frozen four-model set. Raw BT4 to recovered
step 265,000 shows large final-trunk drift (`0.6521` board-cluster relative
L2, `0.6221` linear CKA). In contrast, recovered to corrected v2/update 400
is nearly identity (`0.00278` relative L2, `0.999992` CKA), and their fixed
source-head legal-policy top-1 agrees on all 128 positions. This makes the
current dense result a useful baseline but leaves little local-backbone signal
for sparse-feature interpretation. Published-artifact work remains gated on
upstream parity; the next model experiment should create a stronger,
measurably changed checkpoint before pass-count or sparse-refit sweeps.

Source-parity update, 2026-07-21: the hash-pinned PyTorch implementation at
Leela-SAEs revision `f946a573...` now matches the official FP32 JAX source at
every normative hook/layer. With epsilon `1e-3`, worst relative L2 is
`1.86e-5` and policy-logit relative L2 is `5.68e-6`. The pinned
`HookedTransformer` constructor's effective `1e-5` layer epsilon fails, with
worst relative L2 `0.0563`; it is recorded as an upstream compatibility bug,
not used as the oracle. The source-parity and first-experiment preregistration
gates are therefore complete and `AUTORESEARCH_READY` is open. Published
transcoder validation remains a separate hard gate for sparse interpretation.

First-experiment update, 2026-07-21: sampling two of eight future target
encodes increases the 30-minute run rate from `41.35` to `92.97` examples/s.
The selected update-800 checkpoint improves matched two-pool DFM CE from
`4.5102692712` to `4.5054920968`, improves accuracy and legal mass, and passes
all preregistered collapse gates. It became the first offline incumbent. Its direct
128-pair cap-256 arena against corrected v2/update 400 scores `50.586%`, or
`+4.1` descriptive logistic Elo with pair-aware 95% interval
`[-80.8,+89.4]`. This is an inconclusive-positive screen, not promotion. Only
the selected 1.85 GB state is retained. An exact v2 repeat independently
selects update 800 and clears every offline gate at CE `4.5077912323`, but it
is `0.0022991356` worse than v1, so the direction replicates more tightly than
the effect size. The pass-count and matched sparse-refit studies remain
deferred until a larger gain makes strength evidence less ambiguous.

Second-experiment update, 2026-07-21: reusing the same two sampled targets as
post-prediction recurrent anchors preserves throughput (`92.663` examples/s)
but does not improve the frozen policy metric. Its best checkpoint is terminal
update 1253 at two-pool CE `4.5064137187`, accuracy `0.1091308594`, and legal
mass `0.6438060440`; CE is `0.0009216219` worse than retained K=2 v1 and fails
the `4.5031929612` acceptance ceiling. Mean/min effective rank is
`30.8915/28.9334`, but the final-horizon feature-std p05 is `0.60744`, below
the `0.61` floor. The experiment is rejected without repeat or arena, all four
candidate states are removed, and the unanchored K=2 checkpoint remained the
offline incumbent at that point. The no-norm plus prediction-SIGReg-1.0
objective remains frozen for subsequent architecture experiments.

Current autoresearch update, 2026-07-22: Experiment 009 replaces shared
batch-level K=1 horizon sampling with one balanced assignment per example.
The primary and exact-repeat terminal checkpoints reach two-pool CE
`4.4979399741/4.4969695099`, both beat the cosine-warmdown K=2 incumbent, and
pass every frozen policy and collapse gate. The primary run processes
`202,368` examples at `117.061` examples/s. Its frozen 128-pair arena against
K=2 scores `51.172%`, descriptive logistic Elo `+8.14`, and pair-aware 95%
interval `[-76.48,+93.77]`, with zero faults. Accept primary update 1,581 as
the new offline incumbent, retain only its selected state, and keep eight-pass
inference fixed because the arena evidence remains inconclusive.

Tenth-experiment update, 2026-07-22: fully freezing BT4 raises fixed-run
throughput from `117.061` to `185.643` examples/s, cuts peak JAX HBM from
`9.57` to `3.25` GB, and processes `320,512` examples in 30 minutes. Terminal
CE `4.4965047017` clears the primary gate and every latent gate passes, but
legal mass `0.6445921361` misses the frozen floor by `0.0021314049`. Reject
the candidate without repeat or arena, delete all five states, and keep the
trainable-backbone balanced-K1 checkpoint as the offline incumbent. The
`34.27%` data-stall fraction and `55.0%` mean GPU utilization also show that a
future low-backward graph needs a separately preregistered batch/input-pipeline
retune.

Eleventh-experiment update, 2026-07-22: stopping gradients only through the
future-target BT4 encode preserves current-board action gradients and
reproducibly raises fixed-run throughput to `154.622/154.355` examples/s,
about `32%` above balanced-K1. The primary passes every frozen gate at
CE/accuracy/legal mass `4.494214/0.110535/0.649241`. The exact repeat also
beats incumbent CE at `4.496387` and passes every accuracy and latent gate,
but legal mass `0.646348` misses the `0.646724` floor by `0.000376`. Reject at
the repeat gate, run no arena or SAE work, delete all candidate states, and
keep balanced-K1/update 1,581 as the offline incumbent. Norm matching remains
off and target/prediction SIGReg stay frozen at `5.76/1.0`.

Plan adjustment after Experiment 011: separate the systems mechanism from the
gradient-routing mechanism before another strength claim. The candidate both
detached the future encoder branch and replaced the incumbent's scanned
two-board encode with explicit asymmetric calls. XLA reports `3.72%` more
static FLOPs and peak HBM rises `32.11%`, yet throughput improves `32.09%`.
The next bounded experiment should therefore test the simpler unchunked fused
current+future encode with all gradients attached, holding the now-fixed loss,
data, schedule, batch, and inference contracts constant. First run only CPU
parity plus an A10G memory smoke and 30-update profile; an OOM or failed
throughput gate ends that line without a 30-minute run. This attribution step
has priority over changing legality coefficients, pass count, or SAE work.

Twelfth-experiment update, 2026-07-22: the fused full-gradient graph fits batch
128 at `16.24` GB peak HBM and clears the useful profile gate at `133.628`
examples/s. Its fixed run reaches `134.619` examples/s and processes `229,888`
examples, confirming a `15.00%` systems gain without changing the objective.
Terminal CE/accuracy/legal mass are `4.497541/0.109970/0.646824`; every non-CE
and latent gate passes, but CE misses the repeat-noise-aware ceiling by
`0.000572`. Reject without repeat or arena, delete all five states, and return
to scanned balanced-K1.

Plan adjustment after Experiment 012: execution attribution is complete.
Fusing explains about `44%` of Experiment 011's absolute profile gain, while
removing future backward accounts for the remaining gain but destabilizes the
repeat legal-mass gate. Keep the loss fixed at `0.0/5.76/1.0`. The next
architecture candidate should expose a future-target encoder tail-gradient
boundary: keep the current-board encoder fully trainable, detach early future
layers, and train only a small final BT4 tail from future JEPA/SIGReg losses.
Preregister the exact tail depth only after CPU inspection confirms a clean
checkpoint-compatible boundary and predicts a memory footprint below the
fused graph. This has priority over legality-coefficient tuning, pass-count
sweeps, or SAE refits.

Thirteenth-experiment outcome, 2026-07-22: a three-block future tail exposes
`33,236,736` parameters (`17.02%` of the encoder) to future gradients while
the current path trains all `195,305,728`. It retains almost all of the
zero-tail systems effect: the profile reaches `150.561` examples/s and the
fixed run reaches `149.531`, `27.74%` above balanced-K1, at `13.70` GB peak
JAX HBM. Terminal update 2,024 passes accuracy and clears legal mass by only
`0.000058`, but CE `4.4977803` improves the incumbent by just `0.0001597` and
misses the repeat-noise-aware ceiling by `0.0008108`. Reject at the primary CE
gate, run no repeat/arena/collapse audit/SAE work, delete all three states, and
return the active surface to scanned balanced-K1. RMS norm matching remains
off; target and `z_pred` SIGReg remain `5.76/1.0`.

Plan adjustment after Experiment 013: the tail boundary is a good systems
primitive but three blocks do not produce an offline strength gain. One final
minimal-tail interpolation is justified before closing the line: a one-block
future tail is the closest attached variant to Experiment 011's fast zero-tail
graph and tests whether the smallest alignment gradient can repair its narrow
repeat legal-mass miss without giving up its CE behavior. Preregister it as a
single depth change with the loss fixed at `0.0/5.76/1.0`; if it fails its
primary or repeat gate, end tail-depth experiments rather than sweeping all 15
depths.

Fourteenth-experiment outcome, 2026-07-22: the one-block future tail passes
the primary and exact-repeat offline gates at selected CE
`4.4950091206/4.4971840288`, versus balanced-K1 incumbent CE `4.4979399741`.
It preserves `153.075/153.450` fixed-run examples/s, about `31%` above the
incumbent, with `12.97/12.95` GB peak JAX HBM. Both runs pass every policy,
rank, feature-tail, RMS-ratio, and trivial-control gate. The primary's frozen
128-pair arena against balanced K1 scores `49.609%`, descriptive logistic Elo
`-2.71`, and pair-aware 95% interval `[-87.96,+82.20]`. One candidate loss is
the preregistered symmetric legacy-codec failure mode: all legal moves were
black promotions, which `legacy_absolute_1858` cannot represent. Accept primary
update 2,072 as the offline incumbent, retain only its state for this
experiment, and make no Elo-promotion claim. Close the tail-depth line at one
block; do not sweep depths 2 or 4--15. Norm matching remains disabled and
target/`z_pred` SIGReg remain fixed at `5.76/1.0`.

Direct-strength update, 2026-07-22: the accepted one-block-tail checkpoint
scores `36.523%` against original raw BT4 under the same 128-pair, eight-pass,
cap-256 protocol. Descriptive logistic Elo is `-96.02`, with pair-aware 95%
interval `[-195.33,-10.24]`; the game record is 0 wins, 187 draws, and 69
losses. One candidate loss is the known legacy-codec black-promotion fault,
while raw BT4 has complete canonical coverage. Raw BT4 remains clearly
stronger. The result is 1.5625 score points below corrected v2/update 400 on
the identical roots, so the next training axis should target first-action
chess quality rather than assume another small mean eight-horizon CE gain will
improve Elo.

Plan adjustment after the direct raw-BT4 anchor: optimize the played action
explicitly before another architecture expansion. Uniform DFM CE gives each
of eight trajectory actions equal weight, but arena inference consumes only
horizon 1. Experiment 015 assigns 25% of the normalized DFM CE objective to
horizon 1 and distributes 75% evenly over horizons 2--8, while preserving the
uniform CE reporting metric as a non-regression gate. Architecture, schedule,
batch, legality, eight-pass inference, and norm/target/`z_pred` coefficients
remain fixed at `0.0/5.76/1.0`. Require repeat-qualified horizon-1 improvement
and point-score gains against both the accepted checkpoint and raw BT4 before
calling it an Elo-aligned offline incumbent.

Fifteenth-experiment outcome, 2026-07-22: assigning 25% of DFM CE to the
played action produces a large, internally consistent metric shift. Selected
terminal update 2,077 reaches horizon-1/uniform CE
`2.7167628/4.4811102`, versus accepted primary
`2.8470488/4.4950091`, while retaining `153.297` examples/s. Accuracy and all
latent gates pass, but legal mass falls to `0.6371051`, missing the frozen
floor by `0.0096185`. Reject without repeat or arena, delete all three states,
and return the active surface to the unweighted one-block tail. Because the
change doubled horizon-1 CE share while leaving the source legality
coefficient at `2.0`, it halved relative legality pressure from 16 to 8. One
coefficient-4 rescue is mechanistically justified: it restores that ratio
without changing architecture, schedule, batch, or the fixed
`0.0/5.76/1.0` loss components.

Sixteenth-experiment outcome, 2026-07-22: restoring the nominal ratio with a
25% first-action share and legality coefficient `4.0` nearly recovers legal
mass but erases the preceding CE gains. H1-first selection chooses terminal
update 2,062 at H1/uniform CE `2.8581280/4.5007452`, accuracy `0.1104431`, and
legal mass `0.6460591`. It misses the frozen H1, uniform-CE, and legal-mass
gates by `0.0178696`, `0.0035611`, and `0.0006645`, respectively. Every
latent-health and trivial-control gate passes, and the fixed run retains the
one-block tail's speed at `152.325` examples/s. Reject without repeat or
arena, delete all three candidate states, and return to the unweighted
one-block-tail incumbent. The target and `z_pred` SIGReg coefficients remain
fixed at `5.76/1.0`, with RMS norm matching disabled.

Plan adjustment after Experiment 016: the legality/action tradeoff is not
captured by the nominal coefficient ratio alone. Before introducing a new
loss form, one bounded midpoint on the same ratio-preserving line can test
whether a feasible interior exists: allocate `3/16` of DFM CE to horizon 1
and use legality coefficient `3.0`. If that midpoint cannot simultaneously
clear the existing H1, uniform-CE, and legal-mass gates, close scalar
first-action reweighting and move to an explicitly constrained or
legality-conditioned action objective rather than sweeping more shares.

Seventeenth-experiment outcome, 2026-07-22: the ratio-preserving midpoint
(`3/16` first-action share, legality coefficient `3.0`) restores legal mass
but still erases the desired CE gain. H1-first selection chooses terminal
update 2,043 at H1/uniform CE `2.8555801/4.4992252`, accuracy `0.1106110`, and
legal mass `0.6483664`. Accuracy, legal mass, latent health, and every trivial
control pass, but H1 and uniform CE miss their frozen ceilings by `0.0153216`
and `0.0020412`. The fixed run reaches `151.079` examples/s. Reject without
repeat or arena, delete all three candidate states, return to the unweighted
one-block-tail incumbent, and close scalar first-action weighting. Norm
matching stays off and target/`z_pred` SIGReg stay fixed at `5.76/1.0`.

Plan adjustment after Experiment 017: do not sweep more first-action shares
or legality coefficients. The next action-quality experiment should change a
single structural property of the objective or corruption process while
preserving the fixed loss coefficients and legal-mass gate. A high-priority
candidate is to force the played first action to be masked during training,
matching the fully masked validation/inference start and doubling its
effective per-update sample count without reweighting the loss. Preregister
that train/eval-alignment test before activation.

Eighteenth-experiment outcome, 2026-07-22: forcing the played first action to
the mask token on every training example is a strict near-miss. H1-first
selection chooses terminal update 2,069 at H1/uniform CE
`2.8417612/4.4927525`, accuracy `0.1095734`, and legal mass `0.6483661`.
Uniform CE, accuracy, legality, rank, feature-tail, RMS, and trivial-control
gates all pass. H1 improves the accepted primary by `0.0052875` but misses
the repeat-noise-aware ceiling by `0.0015028`; reject without repeat or arena,
delete all three states, and restore the accepted surface. Throughput remains
`152.678` examples/s. Norm matching stays off and target/`z_pred` SIGReg stay
fixed at `5.76/1.0`.

Plan adjustment after Experiment 018: the corruption-alignment direction is
promising enough for one causal follow-up, but the failed gate is not relaxed.
The DFM action transformer can still see ground-truth future action tokens on
partially masked examples, whereas validation and the first inference pass
start fully masked. Keep H1 forced and transform the existing uniform draw as
`t = u^2` during training, raising expected mask probability from `1/2` to
`2/3` while retaining continuous coverage of every diffusion time. Hold the
objective, legality coefficient, architecture, schedule, validation, and
`0.0/5.76/1.0` loss fixed. If that test fails, close this corruption line
rather than tuning the power.

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

Downloaded source archives are immutable staging inputs, and source
checkpoints are immutable durable assets. Recovery and conversion always write
a new copy; verified staging archives are then removable under the retention
contract below.

## Storage retention contract

The durable local set is intentionally small:

- the extracted trajectory-v3 train/validation/test shards and their
  completion manifest;
- the three raw-BT4 files needed for construction, raw-policy evaluation, and
  cross-framework representation parity;
- the recovered step-265,000 source state, metadata, run configuration, and
  original metrics history;
- compatibility v2/update 300 as the norm-on control;
- corrected v2/update 400 as the no-norm baseline; and
- compact run metrics, reports, profiles, arena evidence, and provenance
  sidecars.

Downloaded tar files and split checkpoint parts are staging objects, not
durable assets. Delete them after extraction and digest verification; their
Drive IDs, source locations, sizes, and digests remain in
`research/assets.json` and the source sidecars. During a future run, the
guarded maximum is two writes and two retained candidates: one sparse
intermediate plus terminal. Periodic `--save-every` is refused. Once
fixed-pool selection is complete, retain one state for each accepted
comparison role and delete states from rejected runs before launching another
model run. Metrics and reports remain because they are the evidence needed to
interpret later phase transitions and scaling curves.

`research/storage_retention.json` is the exact current allowlist.
`research/storage_audit.py` checks required sizes, dataset shard counts and
payload bytes, extra `state.npz` files, redundant staging archives, and a
4-GiB JAX compilation-cache ceiling. `research/env.sh` configures the same
cache limit. Run the fast audit before every GPU job and after checkpoint
selection/pruning, and run its `--verify-hashes` mode after restoring or
moving an immutable asset. The command is read-only and all declared paths
resolve below this repository in `/mountpoint/.exp`.

Initial enforcement on 2026-07-20 reduced 28 research states to the two
selected states, removed the redundant data/model/checkpoint archives and
split parts, and cleared the rebuildable JAX compilation cache. Available
filesystem space increased from 3.7 GiB to 68 GiB. The complete retained-file
hash audit passed after cleanup.

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

The measured U-statistic correction removes the diagonal self-pair term but
changes the observed batch-128/batch-256 discrepancies by only about 3--4%.
The larger difference came from evaluating a nonlinear empirical statistic on
different numbers and mixtures of examples. The training implementation now
supports a shared fixed random example count for target and prediction
SIGReg. The selected contract uses 64 physical examples regardless of training
batch, while validation remains partitioned into full batches of 64. This
keeps estimator count and cost fixed without claiming invariance across
independently sampled data.

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

The first corrected-objective screen used target and prediction coefficients
`1.0`. Across three fixed-count batch-128 gradient audits, the JEPA-group
coefficient corresponding to a 10% auxiliary gradient had medians `1.29` for
target SIGReg and `1.05` for prediction SIGReg. At coefficient one, the
unweighted target and prediction terms begin near one tenth of raw JEPA MSE.
JEPA-group cosine ranges were `0.09..0.23` for positive-JEPA/target-SIGReg,
`-0.06..0.09` for positive-JEPA/prediction-SIGReg, and `0.05..0.09` between
the two SIGReg terms. The full shared-backbone matrices show mild opposition
to policy CE, so larger coefficients are not justified before a longer matched
run.

All three matched 100-update screens used batch 128, a fixed 64-example SIGReg
sample, target coefficient `1.0`, validation seed 10,000, and the same 4,096
held-out positions. Relative to the norm-on control, the no-norm plus
prediction-SIGReg candidate is inside the previously measured `0.00394`
short-run CE dispersion, retains slightly more prediction effective rank and
low-tail feature variance, and preserves the action-shuffle gap. Absolute
target scale contracts similarly in every condition; the old 95% threshold is
therefore recorded as a mechanism diagnostic rather than used to reject only
the corrected candidate.

The checkpointed target-1.0 run then exposed the distinction that the short
screen could not: its selected update 300 is policy-competitive, but mean
target RMS is `0.868` versus `0.935` for the compatibility incumbent and its
low-tail prediction feature standard deviation is also lower. This lies
outside the incumbent representation-health envelope even though prediction
rank and action coupling remain healthy. Target coefficient `1.0` is therefore
rejected as the frozen objective.

Matched 100-update follow-ups at target coefficients `3.9` and `5.76`, with
norm off and prediction coefficient fixed at `1.0`, bracket the tradeoff.
`3.9` gives the best short CE; `5.76` gives the best target/prediction scale,
legal mass, and coupling while remaining inside the measured CE dispersion.
The scalar-balanced `5.76` point was advanced to two checkpointed 30-minute
runs.

V1 selected update 300 and v2 selected update 400 after four fixed 4,096-position
validation pools. Their aggregate CEs are `4.529851` and `4.529214`, only
`0.000637` apart; both improve on the source checkpoint. Their selected latent
audits also agree: target RMS `0.932/0.928`, prediction RMS `0.902/0.901`,
mean prediction rank `31.36/31.24`, and low-tail feature standard deviation
`0.653/0.652`, with positive prediction beating zero and action-shuffled
controls. Corrected v2/update 400 is the frozen offline baseline for Elo.

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

The active no-norm graph was measured at physical batches 64, 128, 192, and
256. After fixing SIGReg to 64 examples, the decisive cached endpoints were:
batch 128 at `44.84` device and `41.41` end-to-end examples/s with `9.71 GB`
peak JAX live memory, and batch 256 at `47.96` and `43.72` examples/s with
`15.85 GB`. Batch 256 buys only 7.0% device and 5.6% end-to-end throughput for
63% more live memory and half as many optimizer updates per example budget.
Batch 128 is therefore the selected autoresearch knee; batch 256 remains a
supported throughput mode rather than the default.

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

The first two strength anchors completed on 2026-07-19:

| Opponent | Candidate score | Logistic relative Elo | Pair-aware 95% interval | Pentanomial | Cap draws |
| --- | ---: | ---: | ---: | --- | ---: |
| Recovered step 265,000 DFM/JEPA | 49.61% | -2.7 | [-88.0, +82.2] | [0, 4, 122, 2, 0] | 3/256 |
| Original raw BT4 | 38.09% | -84.4 | [-181.0, +0.6] | [10, 42, 75, 1, 0] | 0/256 |

These are model-pool-relative descriptive estimates, not human, Lichess, or
absolute Elo. The source comparison is unresolved at 128 pairs. The raw-BT4
point estimate is large and nearly excludes equality under the deliberately
conservative bounded-pair interval, but the upper endpoint remains slightly
positive, so it is not a formal superiority claim.

The raw-BT4 run had no raw-adapter fault and complete canonical-codec
coverage. The corrected DFM had two fail-closed `no_representable_move` losses.
Both exact final positions were black to move with only four legal promotion
moves, confirming the preregistered legacy-codec limitation rather than an
inference failure. Converting both fault losses to draws changes the raw-BT4
comparison point estimate only to about `-81.5` logistic Elo. The result is
retained with the actual fail-closed scoring.

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

That precondition was satisfied on 2026-07-19. The first frozen comparison set
is now original raw BT4, recovered step 265,000, compatibility v2/update 300,
and corrected v2/update 400. Begin with the read-only Stage-0 hook/parity gate
and dense Stage-1 drift measurements. Do not download or train sparse
replacements until the source hook ABI and FP32 parity gate pass.

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
- [x] Make the per-state RMS norm term explicitly configurable, preserve exact
  compatibility behavior when enabled, and test the no-norm raw-MSE path.
- [x] Validate the no-norm target-plus-prediction-SIGReg objective, including
  disabled-path parity, duplication/padding invariance, finite-sample metrics,
  and per-horizon collapse/coupling diagnostics.
- [x] Profile physical batches 64, 128, 192, and 256 on the active no-norm
  target-plus-prediction-SIGReg graph; choose the A10G throughput knee before
  fixing the training batch, while retaining validation batch 64.
- [x] Re-audit target and prediction SIGReg scalar contributions, gradient
  norms, and gradient cosines on the selected graph and physical batch. Do not
  transfer target `5.76` or prediction `0.57` blindly.
- [x] Run matched short screens for norm-on/pred-SIGReg-off,
  norm-off/pred-SIGReg-off, and norm-off/pred-SIGReg-on. Advance only the
  corrected objective that passes policy, legality, action-coupling, and
  representation-health checks. Target `1.0` failed the longer representation
  envelope; target `5.76` with prediction `1.0` advanced.
- [x] Repeat-qualify a 30-minute corrected baseline and freeze its loss,
  physical batch, learning rate, validation contract, and noise envelope
  before strength evaluation. V1/update 300 and v2/update 400 agree within
  `0.000637` four-pool CE; v2/update 400 is the corrected offline baseline.
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
- [x] Establish real-checkpoint GPU parity and timing for the sealed-history
  path and run both 128-pair cap-256 screens. The recovered-source result is
  `-2.7` logistic Elo with interval `[-88.0,+82.2]`; the raw-BT4 result is
  `-84.4` with interval `[-181.0,+0.6]`. Raw BT4 had complete coverage and no
  faults. Three source-run cap draws and the corrected DFM's known legacy-codec
  promotion faults are reported rather than hidden.
- [x] Regenerate and repin the promotion pool after exact replay found entry
  1,245 claim-draw terminal. Two v3 source regenerations were byte-identical,
  all 2,048 final histories pass the production replay loader, selected
  validation/test overlap is zero, and the evaluator is reopened on the
  immutable v3 pins.
- [x] Keep SAE training and representation interventions deferred until both
  relative-Elo anchors exist.
- [x] Expose all five normative BT4 hook tensors and pre-`alpha` branch
  replacement boundaries in commit `7c896a8`. Real-source FP32 capture is
  bit-identical to the ordinary local tokens and matches the independent JAX
  reference within `5e-4` at every hook/layer.
- [x] Freeze and enforce the storage-retention contract. Keep only the source,
  selected norm-on control, and selected corrected states; remove verified
  staging archives and unselected weights; retain compact experimental
  evidence; and enforce the contract with `research/storage_audit.py`.
- [x] Run dense Stage-1 core drift on the frozen four-model comparison set.
  The all-pairs FP32 artifact is
  `artifacts/representations/dense-stage1-four-model-v2`; temporary captures
  were removed and the storage audit passes. Dense linear probes remain a
  separate Stage-1 follow-up. Schedule all representation GPU work
  sequentially; sparse downloads/training remain gated on source parity.
- [x] Complete upstream TransformerLens/source-weight parity. The exact pinned
  PyTorch component path passes against all local FP32 hooks and the policy
  head in `artifacts/representations/upstream-bt4-source-parity-v1`; the
  constructor-default epsilon path is explicitly rejected.
- [ ] Validate one published transcoder before interpreting fixed
  sparse-feature transfer. This remains off the stronger-model critical path.
- [x] Preregister the first post-baseline model experiment. It samples two of
  eight future BT4 targets during training while retaining full-horizon
  validation and fixed eight-pass inference; the immutable hypothesis,
  checkpoint schedule, gates, and cleanup rules are in
  `research/experiment_future_target_sampling_k2.md`.
- [x] Complete the first post-baseline model experiment. K=2 reaches `92.97`
  examples/s, selects update 800 at two-pool CE `4.5054920968`, passes every
  offline gate, and scores `50.586%` in the 128-pair incumbent arena. Retain
  only update 800 and record the arena interval as inconclusive.
- [x] Repeat the accepted K=2 run exactly. V2 again selects update 800 and
  independently passes every offline gate at CE `4.5077912323`, but the
  `0.0022991356` gap to v1 exceeds the old repeat envelope. Retain no v2 state;
  record directional replication and effect-size variability.
- [x] Test sparse post-prediction teacher forcing with the same K=2 targets.
  It preserves `92.663` examples/s, but best CE `4.5064137187` does not beat
  K=2 and final-horizon feature-std p05 `0.60744` misses its floor. Reject it,
  run no arena/repeat, and retain no candidate state. Full evidence is in
  `research/experiment_sampled_target_anchors_k2.md`.
- [x] Test one active JEPA projector block while retaining the exact two-block
  checkpoint ABI. It gains `6.70%` cached-profile throughput and ties K=2 CE
  within `0.00004`, but mean/min prediction effective rank contracts to
  `21.53/19.52` and the RMS ratio to `0.8603`. Reject it, run no arena/repeat,
  and retain no candidate state. Full evidence is in
  `research/experiment_projector_active_depth1_k2.md`.
- [x] Return model experiments to the unanchored, two-projector-block K=2
  incumbent. Keep the prediction/target RMS loss disabled and prediction
  SIGReg coefficient `1.0`; any SIGReg-weight change must be isolated from an
  architecture change.
- [x] Test the one-block projector with only prediction SIGReg increased from
  `1.0` to `4.0`. It improves CE to `4.4995188527` and partially restores
  rank, p05, and RMS ratio, but still fails every latent-diversity floor.
  Reject it without repeat/arena, retain no state, and end the
  shallow-projector line. Full evidence is in
  `research/experiment_projector_depth1_predsigreg4_k2.md`.
- [x] Move to the next original-plan model axis from the full two-block K=2
  baseline. Keep loss coefficients fixed at norm/target/prediction
  `0.0/5.76/1.0` so the next architecture result is attributable.
- [x] Test three active DFM planner blocks while retaining the exact
  four-layer checkpoint ABI. It gains `1.49%` training and `6.28%` batch-64
  inference throughput, but best CE `6.0846793652`, accuracy `0.03290`, and
  legal mass `0.36431` catastrophically fail. Reject it without repeat/arena,
  retain no state, and keep all four DFM blocks. Full evidence is in
  `research/experiment_dfm_active_depth3_k2.md`.
- [x] Continue from the full-projector, full-DFM K=2 incumbent with loss
  coefficients `0.0/5.76/1.0`. Do not revisit smaller planner depth without a
  separately controlled distillation or gradual-depth training objective.
- [x] Test a fixed optimizer warmdown from the full K=2 graph: constant peak
  rates through update 400, cosine decay to 10% over updates 400-1200, then a
  fixed floor. The goal is to remove the replicated update-1200 regression
  without changing architecture or loss. The frozen contract is in
  `research/experiment_cosine_warmdown_k2.md`. Both exact runs pass every
  offline gate at selected CE `4.5007607210/4.5022441577`; update-1200 CE
  improves by at least `0.02718` versus both constant runs. The primary state
  scores `50.391%` in an inconclusive-positive 128-pair K=2 arena and becomes
  the new offline incumbent.
- [x] Test a one-percent cosine floor from the full-projector, full-DFM
  warmdown incumbent. Keep norm/target/prediction coefficients
  `0.0/5.76/1.0`, eight-pass inference, and fixed evaluation pools; isolate
  `lr_min_ratio=0.01` as the sole change. The frozen contract is in
  `research/experiment_cosine_floor001_k2.md`. Update 1200 wins at CE
  `4.4993533697` and passes every policy/latent secondary gate, but misses the
  repeat-noise-aware CE ceiling by `0.0000760853`. Reject it without repeat or
  arena, retain no state, and restore the accepted 10% floor.
- [x] Test one sampled future BT4 target from the accepted K=2 warmdown
  incumbent. Keep the 10% cosine schedule, norm/target/prediction coefficients
  `0.0/5.76/1.0`, all eight rollout horizons, and eight-pass inference fixed;
  isolate `jepa_target_sample_count=1` as the sole change. Require at least 5%
  cached-throughput improvement and a CE gain beyond accepted repeat noise.
  The frozen contract is in
  `research/experiment_future_target_sampling_k1.md`. K=1 gains `24.88%`
  cached throughput and selects terminal CE `4.4998821113`, with every
  secondary policy/latent gate passing. Its improvement is smaller than the
  accepted repeat separation and misses the CE ceiling by `0.0006048269`, so
  reject it without repeat/arena, retain no state, and restore K=2.
- [x] Test balanced per-example K=1 target sampling. Preserve K=1's two BT4
  encodes/example, but assign one horizon to each example so batch 128 covers
  all horizons exactly 16 times per update. Require at least 95% of shared
  K=1 throughput and a CE gain beyond accepted K=2 repeat noise. The frozen
  contract is in `research/experiment_balanced_example_target_k1.md`. The
  primary/repeat terminal checkpoints reach CE
  `4.4979399741/4.4969695099`, both pass every gate, and the primary retains
  `117.061` examples/s. Its 128-pair arena scores `51.172%` with a wide
  `[-76.48,+93.77]` Elo interval. Accept primary update 1,581 as the offline
  incumbent, but do not claim Elo promotion.
- [x] Test a fully frozen BT4 backbone from the balanced-K1 configuration.
  Preserve the source-compatible model ABI and exact forward values, stop the
  encoder gradient, remove its `195,305,728` parameters from optimizer state,
  and set BT4 learning rate to zero. Keep all downstream architecture, loss,
  schedule, data, and evaluation settings fixed. The preregistered contract is
  in `research/experiment_bt4_frozen_backbone_balanced_k1.md`. It gains
  `58.59%` fixed-run throughput, cuts peak JAX HBM `66.02%`, and selects CE
  `4.4965047017` with every latent gate passing. Legal mass `0.6445921361`
  misses its floor by `0.0021314049`, so reject it without repeat/arena and
  retain no candidate state.
- [x] Stop gradients only through the future-target BT4 encode while keeping
  BT4 trainable from the current-board DFM/action and JEPA paths. Keep the
  projector attached on both sides and preserve balanced K=1, loss, schedule,
  depths, batch, and evaluation. The frozen contract is in
  `research/experiment_bt4_future_target_stopgrad_balanced_k1.md`. Its primary
  passes, but the exact repeat misses legal mass by `0.000376`; reject without
  arena or retained state.
- [x] Isolate fused execution with all future gradients attached. It gains
  `15.00%` fixed-run throughput and passes every non-CE gate, but its CE gain
  is inside repeat noise. Reject without repeat/arena or retained state; full
  evidence is in
  `research/experiment_bt4_unchunked_fused_balanced_k1.md`.
- [x] Test the preregistered three-block future-gradient tail. Keep the current
  BT4 path fully trainable, detach the future embedding and first 12 blocks,
  attach the final three blocks and shared projector, and hold loss at
  norm/target/prediction `0.0/5.76/1.0`. Run CPU routing/ABI tests, then the
  batch-128 A10G smoke/profile gates before any fixed-time run. The frozen
  contract is in `research/experiment_bt4_future_tail3_balanced_k1.md`.
  Systems gates pass and terminal accuracy/legal mass pass, but CE `4.4977803`
  is inside repeat noise. Reject without repeat/arena or retained state.
- [x] Preregister and test one final minimal one-block future-gradient tail as
  an interpolation between zero-tail's CE/system behavior and three-tail's
  attached legality signal. Keep norm/target/prediction coefficients
  `0.0/5.76/1.0`; close the tail-depth line on a primary or repeat failure.
  The frozen contract is in
  `research/experiment_bt4_future_tail1_balanced_k1.md`. Its primary/repeat
  select CE `4.4950091/4.4971840`, pass every frozen offline gate, and run at
  `153.075/153.450` examples/s. The 128-pair incumbent arena scores `49.609%`
  with interval `[-87.96,+82.20]`; accept primary update 2,072 as the offline
  incumbent, not an Elo-promoted model, and retain only its candidate state.
- [x] Measure the accepted one-block-tail checkpoint directly against original
  raw BT4 before representation work. It scores `36.523%`, descriptive
  logistic Elo `-96.02`, and pair-aware interval `[-195.33,-10.24]`, with one
  known candidate legacy-codec promotion fault and complete raw-BT4 coverage.
  Raw BT4 remains clearly stronger; lower uniform eight-horizon CE has not yet
  translated into higher measured chess strength.
- [x] Test a 25% first-action share in the normalized DFM CE objective while
  retaining uniform eight-horizon CE as a non-regression metric. Hold the
  one-block tail, schedule, batch, legality, inference, and
  norm/target/`z_pred` coefficients `0.0/5.76/1.0` fixed. The frozen contract
  is in `research/experiment_dfm_first_action_share25_tail1.md`. It improves
  horizon-1/uniform CE to `2.7167628/4.4811102` but legal mass falls to
  `0.6371051`; reject without repeat/arena and retain no candidate state.
- [x] Restore the original legality-to-horizon-1 coefficient ratio while
  retaining the 25% first-action share: change first-legality coefficient
  `2.0 -> 4.0` and hold every other model, data, schedule, inference, and loss
  setting fixed. The frozen contract is in
  `research/experiment_dfm_first_action_share25_legality4_tail1.md`. Terminal
  update 2,062 misses the H1, uniform-CE, and legal-mass gates at
  `2.8581280/4.5007452/0.6460591`; reject without repeat/arena and retain no
  candidate state.
- [x] Test one final scalar midpoint with `3/16` first-action CE share and
  first-legality coefficient `3.0`, preserving ratio 16 and every other
  contract. If it fails any primary gate, close scalar first-action
  reweighting and move to a constrained or legality-conditioned objective.
  The frozen contract is in
  `research/experiment_dfm_first_action_share1875_legality3_tail1.md`.
  Terminal update 2,043 clears accuracy/legal mass but misses H1/uniform CE
  at `2.8555801/4.4992252`; reject without repeat/arena, retain no state, and
  close scalar weighting.
- [x] Preregister a train/eval-alignment experiment that always masks the
  played first action during training while leaving the unweighted objective,
  legality coefficient, architecture, schedule, and `0.0/5.76/1.0` loss
  fixed. This tests lower-variance inference-aligned H1 supervision without
  another loss-weight sweep. The frozen contract is in
  `research/experiment_force_first_action_mask_tail1.md`. Terminal update
  2,069 clears every gate except the H1 repeat margin, missing it by
  `0.0015028`; reject without repeat/arena and retain no state.
- [x] Preregister one final corruption-alignment follow-up: keep H1 forced and
  use `t=u^2` during training so future action tokens are masked with expected
  probability `2/3` instead of `1/2`. Keep all loss coefficients and gates
  fixed; close the line on failure rather than tuning the power. The frozen
  contract is in
  `research/experiment_force_first_action_mask_timepower2_tail1.md`. Primary
  and exact-repeat terminal checkpoints pass every offline gate at H1/uniform
  CE `2.8398960/4.4907011` and `2.8451609/4.4927080`. The primary scores
  `50.781%` against incumbent update 2,072, but only `34.766%` against raw
  BT4, below the frozen `36.523%` anchor. Reject it as an Elo-aligned
  incumbent, retain no candidate state, restore power `1.0` with H1 forcing
  off, and close corruption-power tuning.
- [x] Test a checkpoint-compatible WDL auxiliary on every free-rollout
  predicted state. The shards contain played-game outcome labels but no engine
  or counterfactual action values. Add categorical WDL CE at coefficient
  `0.25`, keep norm/target/`z_pred` coefficients `0.0/5.76/1.0`, leave
  inference unchanged, require the head to beat fixed class-prior baselines,
  and retain a candidate only if repeated offline gates plus both incumbent
  and raw-BT4 arena point gates pass. The frozen contract is in
  `research/experiment_predicted_wdl_aux_tail1.md`. Primary update 2,065
  passes all gates, but repeat update 2,067 misses uniform CE and legal mass by
  `0.0014257/0.0010171`; reject without arena, retain no state, restore WDL
  coefficient zero, and do not sweep the coefficient.
- [x] Test direct parallel multi-horizon JEPA against the recurrent predictor
  while preserving the exact parameter/checkpoint ABI, fixed
  norm/target/`z_pred` coefficients `0.0/5.76/1.0`, balanced K=1 targets, the
  one-block future tail, and unchanged eight-pass DFM inference. The direct
  predictor starts every horizon from `z_0`, consumes the existing
  horizon-specific full-sequence DFM condition, and evaluates all horizons in
  one parallel transition tensor. The immutable systems, policy, latent,
  repeat, arena, two-checkpoint, and cleanup gates are in
  `research/experiment_direct_multihorizon_jepa_tail1.md`. The guarded
  batch-128 smoke aborts safely during compilation at `10.965 GB`
  process-group RSS versus the frozen `7.516 GB` ceiling, with minimum host
  MemAvailable still `6.194 GB`. Reject at the systems gate: run no profile,
  fixed-time training, repeat, checkpoint evaluation, or arena; write no
  state; remove the empty run directory; and restore recurrent mode.
- [x] Test a parameter-free closed loop without target leakage or a ninth DFM
  inference call. Use pass 7's own legal root proposal and hidden state, one
  recurrent JEPA step, and a capped adjoint of the existing hidden adapter to
  condition pass 8 only. Keep the parameter/checkpoint ABI, balanced K=1,
  one-block future tail, eight-pass count, and `0.0/5.76/1.0` loss fixed. The
  immutable contract is in
  `research/experiment_closed_loop_final_pass_adjoint_tail1.md`. One hundred
  fifty-five guarded focused CPU tests pass, but the batch-128 smoke aborts
  safely during compilation at `11.095 GB` process-group RSS versus the
  `7.516 GB` ceiling, with minimum host `MemAvailable` still `6.096 GB`.
  Reject at the systems gate: run no profile, inference benchmark, fixed-time
  training, repeat, evaluation, or arena; write no state; remove the empty run
  directory; and restore feedback mode `none`.
- [x] Test training-only frozen-source-policy-head distillation at the DFM
  root. Reuse current-board BT4 tokens, recover side to move from exact
  classical plane 108, remap canonical logits onto stored representable
  legacy legal support, and add stopped-teacher KL only when the root is
  masked. Add no encoder call, state, or inference work. Fix the sole
  coefficient once from two no-update validation pools using
  `clip(0.25 / pooled_root_KL, 0.05, 1.0)`, then retain a candidate only after
  guarded systems gates, repeated offline gates, and both incumbent/raw-BT4
  arena point gates. The immutable contract is in
  `research/experiment_bt4_policy_distillation_tail1.md`. The two no-update
  pools calibrate `K0=0.8656389851` and coefficient `0.2888039983` with full
  eligibility. The guarded batch-128 training compile then reaches
  `10.946 GB` process-group RSS versus the fixed `7.516 GB` ceiling and exits
  safely before any update or state write. Reject at the systems gate, run no
  smaller-batch rescue or later stage, and restore coefficient zero.
- [ ] Test root legal-conditional imitation without enlarging the model
  graph. Keep uniform full-vocabulary CE and `2.0 * (1 - legal_mass)`
  unchanged, and add played-action CE normalized only over stored root legal
  moves. Calibrate its single coefficient on the two no-update source pools
  to initial weighted scale `0.25`, require exact zero direct gradient on
  illegal logits, then apply the guarded smoke/profile, repeat-qualified
  offline gates, and both incumbent/raw-BT4 point-score gates in
  `research/experiment_root_legal_conditional_ce_tail1.md`. Write at most one
  intermediate plus one terminal state; do not sweep the coefficient or
  relax the resource guard.
