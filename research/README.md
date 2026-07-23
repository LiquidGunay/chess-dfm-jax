# Local GPU research path

The implementation contract is
[docs/local_gpu_autoresearch_plan.md](../docs/local_gpu_autoresearch_plan.md).

Before running local commands:

```bash
source research/env.sh
```

This creates and exports workspace-local locations for temporary files,
dependency caches, JAX compilation artifacts, data, checkpoints, profiles, and
experiment runs. The script refuses to configure itself if the repository is
outside `/mountpoint/.exp`.

The intended stable interface is:

```text
prepare.py        fixed data and asset validation
storage_audit.py  exact retained-state, dataset, archive, and cache audit
import_legacy.py  checksummed, restricted source-checkpoint boundary
train.py          single editable model/objective/training surface
resource_guard.py fail-closed host RAM/CPU/disk/checkpoint watchdog
inference.py      checked cached-BT4 multi-pass inference and profiling
arena.py          frozen pools, paired outcomes, and arena statistics
arena_history_trust.py sealed O(1) internal history attestations
local_policy.py   strict current-only localized DFM arena adapter
play_arena.py     bounded batched gameplay with exact history replay
evaluate_arena.py strict in-process relative-Elo CLI and resumable blocks
program.md        rules for automated research
results.tsv       compact append-only experiment ledger
```

The arena freezes one physical JAX inference batch shape per run and pads only
already-validated encoded rows. Shrinking game populations therefore reuse the
warmup executable instead of compiling new batch shapes.

All mutable state, including Python and JAX caches, remains below
`/mountpoint/.exp`.

Every GPU command must go through `research/run_gpu.sh`; sourcing `env.sh` and
calling a GPU Python entry point directly is not sufficient. The launcher holds
one host-visible lock across prepare, compilation, and execution, restricts the
job to two CPUs, refuses to start below 8 GiB MemAvailable, and terminates the
job if its process group exceeds 7 GiB RSS or host MemAvailable falls below
3 GiB. It also preserves at least 30 GiB projected free disk. These defaults
can only be made stricter through environment variables.

Periodic `--save-every` schedules are refused. A training run may write at
most two states: normally one sparse `--save-updates` state plus
`--save-final`, with `--max-checkpoints 2`. Scan them, retain at most one
selected state, and prune the other before starting another model run.

Check the lightweight storage contract before and after an experiment:

```bash
.venv/bin/python research/storage_audit.py
```

Use `--verify-hashes` after moving or restoring an immutable asset. The
default audit avoids rereading multi-gigabyte weights but still checks their
sizes, requires exactly the selected state files, validates the extracted
dataset, rejects redundant source archives, and enforces the JAX cache budget.

The parity oracle is run separately:

```bash
research/run_gpu.sh .venv/bin/python research/legacy_baseline.py
```

It uses strict model-state restoration and writes a deterministic loss
fingerprint under `artifacts/baselines/`.

The first compatibility trainer is:

```bash
research/run_gpu.sh .venv/bin/python research/train.py --steps 1 --batch-size 1
```

`research/train.py` now owns the checkpoint-visible model, stage-1 objective,
optimizer, training step, and validation step. The legacy implementation is
used only as a test oracle; fresh runs cross `import_legacy.py` after exact
size, digest, serialization, and ABI checks.

For a checked-in experiment, edit the small `EXPERIMENT_OVERRIDES` mapping near
the top of `train.py`. Effective configuration precedence is:

```text
source checkpoint metadata -> EXPERIMENT_OVERRIDES -> explicit CLI flags
```

The final effective configuration is written to the report and bound into the
resume contract. Unknown keys, wrong types, and non-default legacy knobs that
the local graph cannot honor fail before training.

Evaluate every retained checkpoint from one run without rebuilding or
recompiling the model for each checkpoint:

```bash
research/run_gpu.sh .venv/bin/python research/train.py \
  --eval-checkpoints research/runs/<training-run> \
  --eval-batches 64 --eval-batch-size 64 \
  --eval-checkpoint-seeds 10000 20000 \
  --run-id <training-run>-checkpoint-eval
```

This mode takes the model and objective from the checkpoint contract, reuses
the same validation positions and RNG for every checkpoint, and never restores
optimizer moments or writes beneath the evaluated run. Incremental results go
to `checkpoint_metrics.jsonl`; the aggregate and minimum-DFM-CE checkpoint go
to `checkpoint_summary.json`.

The paired evaluator has strict source/research-checkpoint loading, frozen
history validation, atomic resumable blocks, bounded model batches, a lean
action-only inference path, and an official-formulation normalized-Elo
promotion GSPRT. The repaired v3 promotion pool is open, and
`AUTORESEARCH_READY = True`. The first accepted K=2 target-sampling checkpoint
passes the offline gates and has an inconclusive-positive 128-pair incumbent
screen; an exact repeat independently passes the offline gates but shows a
larger-than-baseline between-repeat CE gap. It is not promoted. See
`docs/local_relative_arena.md`, the plan, and the baseline report for the exact
contracts and measured results. The subsequent sampled-target anchor
experiment preserved throughput but was rejected at CE `4.5064137187` and a
minimum feature-std p05 of `0.60744`; it retained no state and did not consume
arena games. A subsequent one-active-block projector experiment was also
rejected: it gained `6.70%` cached-profile throughput and tied K=2 policy CE
within `0.00004`, but prediction effective rank contracted to `21.53` mean and
`19.52` minimum. Its implementation remains default-off and no state is
retained. Normalized experiments continue with the RMS norm coefficient at
zero and prediction SIGReg at `1.0`. A coefficient-4 rescue then improved CE
to `4.4995188527` and partially recovered rank to `24.13/22.29`, but still
failed all rank/feature-tail gates and narrowly missed the RMS-ratio floor.
It also retained no state and ends the shallow-projector line.

A three-active-block DFM experiment then improves matched batch-64 eight-pass
inference throughput by `6.28%` and training throughput by `1.49%`, but its
best CE/accuracy/legal mass regress to `6.084679/0.03290/0.36431`. Direct DFM
prefix pruning is rejected, retains no state, and leaves all four planner
blocks active.

A fixed-time cosine warmdown then keeps that full K=2 graph and the
norm/target/prediction loss coefficients `0.0/5.76/1.0` unchanged. Two exact
runs independently pass every offline gate at selected CE
`4.5007607210/4.5022441577`, and both remove the replicated constant-rate
update-1200 regression. The primary update-1261 state scores `50.391%` in the
128-pair arena against K=2, with descriptive logistic Elo `+2.71` and a wide
pair-aware interval `[-82.20,+87.96]`. It became the offline incumbent, not an
Elo-promoted model; the repeat retains no state.

A one-percent cosine-floor follow-up then selects update 1200 at CE
`4.4993533697`, accuracy `0.1100006104`, and legal mass `0.6472349875`; its
latent audit passes every gate. This improves the 10%-floor incumbent point
estimate by `0.0014073513`, but misses the preregistered repeat-noise-aware CE
ceiling by `0.0000760853`. It is rejected without repeat or arena, retains no
state, and the active schedule returns to the accepted 10% floor.

A K=1 target-sampling follow-up reduces training to two BT4 encodes per
example and raises cached throughput to `116.51` examples/s, `24.88%` over
K=2. Its terminal checkpoint reaches CE `4.4998821113`, accuracy
`0.1096038818`, and legal mass `0.6480329307`, and passes every latent gate.
The CE gain is smaller than accepted repeat variability and misses the frozen
ceiling by `0.0006048269`, so it is rejected without repeat or arena, retains
no state.

Balancing that one sampled horizon across examples then preserves the
two-encode path while removing batch-level horizon sparsity. The primary and
exact-repeat runs select terminal checkpoints at CE
`4.4979399741/4.4969695099`; both beat the cosine-warmdown K=2 incumbent and
pass every policy and latent gate. The primary processes `202,368` examples
at `117.061` examples/s. Its 128-pair screen against K=2 scores `51.172%`, or
`+8.14` descriptive logistic Elo with pair-aware 95% interval
`[-76.48,+93.77]`, and zero faults. Primary update 1,581 is the current
offline incumbent, not an Elo-promoted model; only that candidate state is
retained.

Freezing the complete BT4 backbone then produces the largest systems gain so
far: `185.643` fixed-run examples/s, `320,512` examples in 30 minutes, and
3.25 GB peak JAX HBM. Its terminal checkpoint improves CE to `4.4965047` and
passes every latent gate, but legal mass falls to `0.6445921`, below the
frozen `0.6467235` floor. The experiment is rejected without repeat or arena,
all five states are deleted, and the trainable-backbone balanced-K1 checkpoint
remains the offline incumbent.

Stopping gradients only through the future-target BT4 encode then preserves
current-board action gradients and raises fixed-run throughput to
`154.622/154.355` examples/s across exact runs, about `32%` over balanced-K1.
The primary passes every gate at CE/accuracy/legal mass
`4.494214/0.110535/0.649241`; the repeat again beats incumbent CE at
`4.496387` and passes every latent gate, but legal mass is `0.646348`, missing
the frozen floor by `0.000376`. The direction is rejected at the repeat gate:
no arena, no retained state, and no incumbent change. The default-off
implementation remains as evidence that future-target encoder backward is a
large systems cost, while normalized experiments continue with norm/target/
prediction coefficients `0.0/5.76/1.0`.

Fusing the current and sampled-future boards into one full-gradient BT4 batch
then isolates the execution effect. It raises fixed-run throughput to
`134.619` examples/s (`+15.00%`) and explains about `44%` of the asymmetric
candidate's absolute profile gain, but peak JAX HBM rises to `16.34` GB. Its
terminal checkpoint passes legality, accuracy, every latent gate, and all
trivial controls at CE `4.4975412`; that is only `0.0003988` better than the
incumbent and misses the repeat-noise-aware ceiling by `0.0005717`. It is
rejected without repeat or arena, all five states are deleted, and the scanned
balanced-K1 path remains active.

A three-block future-gradient tail then keeps the current-board encoder fully
trainable, detaches the future embedding and first 12 BT4 blocks, and attaches
the final three blocks plus shared projector. It preserves almost all of the
zero-tail systems gain: `149.531` fixed-run examples/s (`+27.74%`) at `13.70`
GB peak JAX HBM. Terminal update 2,024 reaches CE/accuracy/legal mass
`4.4977803/0.1107483/0.6467812`; accuracy and legal mass pass, but the CE gain
is only `0.0001597`, inside accepted repeat noise, and misses the strict ceiling
by `0.0008108`. Reject without repeat, arena, collapse audit, or retained
state. The default path returns to scanned balanced-K1, with RMS norm loss off
and target/prediction SIGReg fixed at `5.76/1.0`.

The final one-block future-gradient tail then preserves almost all zero-tail
speed while restoring repeat-stable offline quality. Primary/repeat runs reach
`153.075/153.450` examples/s and select CE/accuracy/legal mass
`4.4950091/0.1104584/0.6488690` and
`4.4971840/0.1103058/0.6468926`; both pass every frozen policy and latent
gate. The primary's 128-pair arena against balanced K1 scores `49.609%`, or
`-2.71` descriptive logistic Elo with pair-aware 95% interval
`[-87.96,+82.20]`. One charged candidate fault occurs when every legal move
is a black promotion, a symmetric known limitation of both checkpoints'
`legacy_absolute_1858` codec. Primary update 2,072 becomes the current offline
incumbent, not an Elo-promoted model. Only its candidate state is retained for
this experiment, the tail-depth line closes, RMS norm matching remains off,
and target/`z_pred` SIGReg remain `5.76/1.0`.

Against original raw BT4, that accepted checkpoint scores `36.523%` over the
same 128 color-reversed pairs: 0 wins, 187 draws, and 69 losses, or `-96.02`
descriptive logistic Elo with pair-aware interval `[-195.33,-10.24]`. Raw BT4
is clearly stronger at this searchless budget. One candidate loss is the known
legacy-codec black-promotion fault; the canonical raw policy has complete move
coverage. This direct anchor satisfies the base-model measurement prerequisite
for later representation work, while showing that lower uniform
eight-horizon CE has not yet produced higher measured chess strength.

Weighting the played first action to 25% of DFM CE then produces a large
held-out gain without changing throughput: selected update 2,077 reaches
horizon-1/uniform CE `2.7167628/4.4811102` at `153.297` examples/s, and all
latent diagnostics pass. Legal mass regresses to `0.6371051`, however, below
the frozen `0.6467235` floor. The experiment is rejected without repeat or
arena and all three states are deleted. Since the change doubled horizon-1 CE
share while holding first-legality coefficient `2.0`, it halved their relative
weight ratio; a separately preregistered coefficient-4 rescue is the only
immediate follow-up. Norm matching remains off and target/`z_pred` SIGReg stay
fixed at `5.76/1.0`.

The coefficient-4 rescue then nearly restores legal mass but erases the CE
gain. H1-first selection chooses terminal update 2,062 at H1/uniform CE
`2.8581280/4.5007452`, accuracy `0.1104431`, and legal mass `0.6460591`.
Those values miss the frozen H1, uniform-CE, and legal-mass gates, although
all latent-health controls pass and throughput remains `152.325` examples/s.
Run no repeat or arena, retain no candidate state, and restore the unweighted
one-block-tail surface. The result rules out the simple claim that restoring
the nominal legality/H1 coefficient ratio preserves Experiment 015's CE
gain. One ratio-preserving midpoint (`3/16` first-action share, coefficient
`3.0`) is the final bounded scalar interpolation before changing the loss
form. Norm matching stays off; target and `z_pred` SIGReg stay `5.76/1.0`.

Experiment 017 is preregistered in
`research/experiment_dfm_first_action_share1875_legality3_tail1.md`. It is
the final bounded scalar interpolation: `3/16` first-action CE share and
first-legality coefficient `3.0`, preserving ratio 16 and every model, data,
schedule, inference, validation, and SIGReg contract. Failure closes this
scalar line rather than authorizing a parameter sweep.

Experiment 017 is complete and rejected at its primary gate. H1-first
selection chooses terminal update 2,043 at H1/uniform CE
`2.8555801/4.4992252`, accuracy `0.1106110`, and legal mass `0.6483664`.
Accuracy, legal mass, and every latent-health control pass, but both CE gates
fail. Run no repeat or arena, retain no candidate state, restore the
unweighted one-block-tail surface, and close scalar first-action weighting.
The next bounded action-quality test should align training with fully masked
inference by always masking the played first action, without reweighting CE or
changing the legality coefficient. Norm matching stays off; target and
`z_pred` SIGReg stay `5.76/1.0`.

Experiment 018 is preregistered in
`research/experiment_force_first_action_mask_tail1.md`. It restores the
accepted uniform CE and legality coefficient, then always masks the played
first action during training while leaving horizons 2--8 and every RNG stream
unchanged. This matches fully masked validation/inference and doubles the H1
sample count per batch without another scalar weighting change. Norm matching
remains off; target and `z_pred` SIGReg remain `5.76/1.0`.

Experiment 018 is complete and rejected only at its primary H1-margin gate.
Terminal update 2,069 reaches H1/uniform CE `2.8417612/4.4927525`, accuracy
`0.1095734`, and legal mass `0.6483661`; every non-H1 gate passes. H1 improves
the accepted primary by `0.0052875` but misses the frozen ceiling by
`0.0015028`, so run no repeat or arena, retain no candidate state, and restore
the accepted surface. One final corruption follow-up may keep H1 forced while
using `t=u^2` to reduce ground-truth future-action leakage; failure closes the
line. Norm matching remains off and target/`z_pred` SIGReg stay `5.76/1.0`.

Experiment 019 is preregistered in
`research/experiment_force_first_action_mask_timepower2_tail1.md`. It keeps
H1 forced and changes only normalized training time from `t=u` to `t=u^2`,
raising expected future-action masking from `1/2` to `2/3` without changing
RNG draws, validation, inference, objective weights, legality, or the fixed
`0.0/5.76/1.0` loss. Failure closes this corruption line rather than opening
a time-power sweep.

Experiment 019 is complete and rejected at its raw-BT4 arena gate. Primary
update 2,063 and repeat update 2,067 independently pass every offline gate at
H1/uniform CE `2.8398960/4.4907011` and `2.8451609/4.4927080`, with no latent
collapse. The primary scores `50.781%` against incumbent update 2,072 but only
`34.766%` against original raw BT4, below the frozen `36.523%` anchor. The
candidate is therefore not an Elo-aligned incumbent: delete both candidate
states, keep update 2,072, restore forced H1 masking off and training-time
power `1.0`, and close corruption-power tuning. Norm matching remains off;
target and `z_pred` SIGReg remain fixed at `5.76/1.0`.

Experiment 020 is preregistered in
`research/experiment_predicted_wdl_aux_tail1.md`. The approved shards contain
per-horizon played-game WDL labels, although they contain no search values or
counterfactual action outcomes. The experiment activates the existing
checkpoint-compatible WDL head on every free-rollout predicted state at
coefficient `0.25`, adds outcome-quality metrics, and leaves eight-pass
inference unchanged. Policy, legality, latent, repeat, and matched incumbent/
raw-BT4 arena gates remain frozen; failure restores WDL coefficient zero
rather than opening a weight sweep.

Experiment 020 is complete and rejected at its exact-repeat gate. Primary
update 2,065 passes every frozen gate at WDL CE/accuracy
`0.972390/0.511017`, H1/uniform CE `2.844706/4.496097`, action accuracy
`0.110870`, and legal mass `0.647005`. Repeat update 2,067 again learns WDL
and passes H1 plus every latent gate, but its uniform CE `4.498610` and legal
mass `0.645706` miss their frozen thresholds by `0.0014257/0.0010171`.
Therefore run no arena, retain no candidate state, restore WDL coefficient
zero, and do not open a coefficient sweep. Target and `z_pred` SIGReg remain
fixed at `5.76/1.0`, with RMS norm matching off.

Experiment 021 is complete and rejected at its guarded compilation gate. The
checkpoint-ABI-preserving direct multi-horizon JEPA implementation passes 108
focused CPU tests, but its batch-128 one-update smoke reaches `10.965 GB`
process-group RSS after `39.236` seconds, above the frozen `7.516 GB` host
ceiling. The guard terminates it before host MemAvailable falls below
`6.194 GB`; no process, report, or checkpoint survives. Per the immutable
contract, run no profile or fixed-time rescue. Direct execution remains
available default-off and recurrent JEPA is restored as the active graph.

Experiment 022 is also complete and rejected at its guarded compilation gate.
Its parameter-free closed loop uses the model's own legal proposal, one
recurrent JEPA step, and the adjoint of the existing hidden adapter before
only the eighth DFM inference call. One hundred fifty-five guarded focused CPU
tests pass, but the batch-128 one-update compile reaches `11.095 GB`
process-group RSS after `83.417` seconds versus the frozen `7.516 GB` ceiling.
The guard exits while host `MemAvailable` is still `6.096 GB`; no report,
checkpoint, state, or process survives. Run no profile, inference benchmark,
fixed-time rescue, repeat, or arena. The capability remains default-off and
the active graph restores `jepa_feedback_mode="none"`.

Experiment 023 is preregistered in
`experiment_bt4_policy_distillation_tail1.md`. The training-only candidate
feeds existing current-board tokens through the immutable source BT4 policy
head, exactly remaps canonical logits into represented legacy root support,
and minimizes stopped-teacher/legal-student KL. It adds no encoder call,
parameter, checkpoint state, or inference latency. Twelve sampled shards
prove the plane-108 side and stored-legal contracts. A frozen no-update
two-pool calibration sets the single coefficient to
`clip(0.25 / pooled_root_KL, 0.05, 1.0)` before any optimizer update.
The pools measure KL `0.8583731093/0.8729048609` with full eligibility, so
the exact frozen coefficient is `0.2888039983331077`; both runs perform zero
updates and write no checkpoint.

The guarded batch-128 training smoke is a terminal systems rejection. XLA
compilation reaches `10.946 GB` process-group RSS after `78.96` seconds versus
the frozen `7.516 GB` ceiling, while host `MemAvailable` remains `5.852 GB`.
The guard exits safely; no report, checkpoint, state, process, or nonempty run
directory remains. Run no smaller-batch rescue, profile, fixed-time training,
repeat, or arena. The implementation remains default-off and coefficient zero
is restored.

Experiment 024 is preregistered in
`experiment_root_legal_conditional_ce_tail1.md`. It adds no teacher or model
call: the played root action is ranked only against the stored legal actions,
while uniform full-vocabulary CE and the existing legality loss remain
unchanged. Its one coefficient is frozen by two source-init no-update pools to
an initial weighted scale of `0.25`. The sparse legal reduction must compile
under the existing guard before any fixed-time run, and at most update 1,600
plus terminal may be written temporarily.

The two no-update source pools measure conditional CE
`2.1329845171/2.0417164937` with full eligibility. Their pooled
`C0=2.0873505054041743` fixes the sole coefficient at
`0.11976905620438309`, making the initial weighted component exactly `0.25`.
No optimizer update or checkpoint was written.

The read-only update-2,072 control measures pooled legal-conditional CE
`2.0518385936` and legal-only top-1 `0.4296875`, again with full coverage.
Both a primary and repeat must beat that conditional-CE value.

The guarded batch-128 training smoke is a terminal systems rejection. Cold
XLA compilation reaches `10.944 GB` process-group RSS after `79.01` seconds
versus the fixed `7.516 GB` ceiling; host `MemAvailable` remains `6.316 GB`
when the guard exits safely. No update, report, compiler artifact, checkpoint,
state, process, or nonempty run directory survives. Run no smaller-batch
rescue, profile, fixed-time training, repeat, or arena. The implementation
remains default-off and coefficient zero is restored.

Experiment 025 is the bounded compiler-memory recovery in
`experiment_xla_autotune0_compile_memory.md`. The installed XLA build already
uses single-threaded LLVM-module compilation, making the initially considered
parallelism flags redundant. The one preregistered fresh-cache trial disables
stable GPU kernel autotuning on the otherwise unchanged accepted graph. It
uses the same 7 GiB guard, batch 128, a 900-second timeout, and no checkpoints.
The setting is adopted only if the compile has 256 MiB guard headroom, the
one-update metrics match the retained smoke within `1e-6`, and a cached
30-update profile retains at least 95% throughput.

The candidate is rejected at the first gate. Autotune level 0 still reaches
`11.098 GB` process-group RSS after `92.824` seconds; the unchanged guard exits
safely while host `MemAvailable` remains `6.315 GB`. No update, report,
checkpoint, or state is written. The empty run directory and 108 KiB
dedicated partial cache are removed, leaving the ordinary cache and exact
seven-state retention set unchanged. Do not adopt or sweep this flag; inspect
host initialization/import/compiler lifetimes next.

Experiment 026 is the separately guarded compile-process test in
`experiment_separate_compile_process.md`. A normal incumbent cache hit peaks
at `6.158 GB` group RSS and exactly reproduces retained metrics; aggressive
glibc arena/trim settings reduce that by only 16 MB and are not adopted. The
candidate adds a fail-closed `--compile-only` path that constructs identical
model/optimizer shapes, releases construction-only NumPy weights, populates
the JAX cache without opening the 1.8 GiB legacy state, and exits. The normal
trainer then verifies/restores the source and executes the cache hit. The two
GPU processes are sequential, use the unchanged guard, and write no
checkpoint during validation.
