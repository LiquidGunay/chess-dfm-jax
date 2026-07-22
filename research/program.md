# Autoresearch program

The goal is to improve the fixed held-out chess metrics of the local-GPU
BT4/DFM/JEPA model while preserving latent diversity and making training and
inference faster.

Current gate: `AUTORESEARCH_READY = True`. The dense source-parity gate and
first-experiment preregistration are complete, and the frozen 30-minute
autoresearch loop is active. Promotion still requires every offline and arena
gate below; readiness alone is not a strength claim.

The first accepted experiment samples two of eight future BT4 targets during
training while preserving all eight DFM/JEPA prediction horizons and
full-horizon evaluation. Its update-800 checkpoint improves two-pool DFM CE
from `4.5102692712` to `4.5054920968`, improves accuracy and legal mass, passes
every latent gate, and raises fixed-time throughput from `41.35` to `92.97`
examples/s. It was the previous offline incumbent. Its direct 128-pair arena
against corrected v2/update 400 scored `50.586%` (`+4.1` descriptive logistic
Elo, pair-aware 95% interval `[-80.8,+89.4]`), which is positive but
inconclusive. An exact v2 repeat again selects update 800 and independently
passes every gate at CE `4.5077912323`, but its selected CE is `0.0022991356`
worse than v1, beyond the old `0.000637` repeat envelope. The direction is
replicated; the effect size is not tightly repeat-stable, and the model is not
Elo-promoted.

The next experiment reused those two sampled targets as post-prediction
teacher-forcing anchors during training while keeping validation fully
free-running. It retained K=2 speed (`92.663` examples/s), but its best
two-pool checkpoint reached CE `4.5064137187`, `0.0009216219` worse than the
retained K=2 incumbent and above the preregistered `4.5031929612` ceiling.
Its final-horizon prediction feature-std p05 was `0.60744`, just below the
`0.61` floor. Sparse anchoring is rejected; no repeat or arena was run and all
candidate states were deleted. Keep the no-norm objective and prediction
SIGReg coefficient `1.0` as the active baseline.

The following one-active-block projector experiment improved the cached
profile by `6.70%` and reached `97.335` examples/s during the fixed run. Its
best two-pool CE, `4.5055253655`, was effectively tied with K=2 v1 but missed
the frozen `4.5031929612` acceptance ceiling. More importantly, its terminal
prediction effective rank fell to `21.53` mean / `19.52` minimum and its
prediction/target RMS ratio to `0.8603`, despite prediction SIGReg remaining
active. It is rejected without repeat or arena, and all candidate states were
deleted. Keep both projector blocks active. Scalar norm health is not a
substitute for rank and low-variance-tail diagnostics.

The coefficient-4 follow-up improves the one-block result to CE
`4.4995188527`, accuracy `0.1105346680`, and legal mass `0.6497509237`, so it
clears the policy gate by more than the K=2 repeat separation. Prediction
SIGReg also partially rescues mean/min rank from `21.53/19.52` to
`24.13/22.29`, mean/min feature p05 from `0.495/0.470` to `0.569/0.549`, and
the RMS ratio from `0.8603` to `0.9371`. Those values still fail every frozen
latent-diversity threshold and narrowly miss the RMS floor. The experiment is
rejected without repeat or arena, all states are deleted, and the
shallow-projector line ends. This is useful evidence that stronger prediction
SIGReg helps but does not substitute for projector capacity or explicit scale
control.

The three-active-block DFM experiment gains `1.49%` cached training throughput
and `6.28%` matched batch-64 eight-pass inference throughput, with a `0.59%`
batch-one positions/s regression. It catastrophically damages policy quality:
its best two-pool CE/accuracy/legal mass are `6.084679/0.03290/0.36431` versus
K=2 `4.505492/0.10959/0.64552`. It also misses several latent gates. Direct
planner-prefix pruning is rejected without repeat or arena, all states are
deleted, and all four DFM blocks return as the active baseline. Smaller
planners require a separately preregistered distillation or gradual LayerDrop
method rather than assuming the final source-trained block is redundant.

Fixed-time cosine warmdown became the next offline incumbent. It holds the full K=2
model and `0.0/5.76/1.0` loss fixed, stays at peak main/BT4 rates through
update 400, cosine-decays to a 10% floor at update 1200, and holds that floor.
The primary run selects terminal update 1261 at CE `4.5007607210`; an exact
repeat independently selects update 800 at CE `4.5022441577`. Both pass every
policy and latent gate, and the selected CE separation `0.0014834367` is
inside the prior K=2 repeat separation. At update 1200, both warmdown runs are
at CE `4.50150..4.50276`, eliminating the constant-rate regression to
`4.52867..4.53762`. The primary checkpoint scores `50.391%` in the frozen
128-pair arena against K=2, with descriptive logistic Elo `+2.71` and
pair-aware interval `[-82.20,+87.96]`. This is positive but inconclusive and
does not constitute Elo promotion. Retain v1/update 1261 as a prior
offline-incumbent milestone and no repeat state.

The one-percent cosine-floor follow-up selects update 1200 at CE
`4.4993533697`, accuracy `0.1100006104`, and legal mass `0.6472349875`, while
passing every latent-health gate. It improves the 10%-floor incumbent point
estimate by `0.0014073513`, and its terminal CE is essentially flat, but it
misses the frozen repeat-noise-aware ceiling by `0.0000760853`. The trial is
therefore rejected without repeat or arena, all candidate states are deleted,
and the active schedule returns to the accepted 10% floor. Loss remains
`0.0/5.76/1.0`, so prediction SIGReg—not RMS norm matching—continues to
regularize `z_pred`.

The K=1 target-sampling follow-up reduces training from three to two BT4
encodes per example and raises cached throughput by `24.88%`, processing
`40,320` more examples than K=2 v1 in 30 minutes. Its terminal checkpoint
passes every policy-secondary and latent-health gate at CE `4.4998821113`, but
the `0.0008786097` improvement over the incumbent is smaller than the accepted
repeat separation and misses the frozen ceiling by `0.0006048269`. K=1 is
rejected without repeat or arena, all candidate states are deleted, and K=2
returned as the active target sampler for the next controlled experiment.

Balanced per-example K=1 sampling keeps K=1's two-encode budget but replaces its one
batch-shared horizon with one balanced horizon assignment per example. At
batch 128, every update contains exactly 16 examples from each of the eight
future horizons. This isolates horizon-estimator variance while keeping the
accepted schedule, full-horizon prediction/evaluation, and
`0.0/5.76/1.0` loss fixed. Its primary and exact-repeat selected checkpoints
reach two-pool CE `4.4979399741/4.4969695099`, both beat the K=2 incumbent,
and differ by only `0.0009704642`, inside accepted repeat variation. The
primary run processes `202,368` examples at `117.061` examples/s and passes
every policy and latent gate. Its frozen 128-pair arena against cosine
warmdown scores `51.172%` (`+8.14` descriptive logistic Elo, pair-aware 95%
interval `[-76.48,+93.77]`) with zero faults. Accept primary update 1,581 as
the current offline incumbent; the arena remains positive but inconclusive,
so this is not Elo promotion. Retain only the primary selected state.

The full-backbone-freeze experiment retains
its exact forward values and model-state ABI. It stops gradients at BT4 token
outputs, removes the encoder's `195,305,728` parameters from optimizer state,
and fixes its learning rate to zero. Balanced K=1 sampling, the full
projector/DFM/JEPA graph, the cosine schedule, eight-pass inference, and
`0.0/5.76/1.0` loss remain fixed. It raises fixed-run throughput by `58.59%`,
cuts peak JAX HBM by `66.02%`, and selects terminal CE `4.4965047017`, which
clears the primary CE gate. Prediction rank, feature tail, RMS ratio, and
trivial controls all pass. Legal mass falls to `0.6445921361`, however,
missing its frozen floor by `0.0021314049`. Reject it without repeat or arena,
delete all candidate states, and return the active graph to the trainable BT4
balanced-K1 incumbent. The result motivates a separately controlled partial
gradient-routing experiment; it does not justify tuning the fixed loss.

The active experiment keeps BT4 trainable on the current-board action and
JEPA paths but stops gradients through the future-target BT4 token output.
The shared state projector remains attached to both branches. This isolates
whether future-side encoder gradients are worth their backward cost while
preserving the action gradients that full freeze appears to need for legal
mass. Balanced K=1, all model depths, the schedule, eight-pass inference, and
`0.0/5.76/1.0` loss remain fixed.

The repeat-qualified norm-on compatibility baseline is v2/update 300. An identical v1 run also
selected update 300, and their four-pool DFM CE gains differ by only
`0.000210253`; the selected v2 checkpoint also passes the recorded
effective-rank, low-variance-tail, prediction/target RMS-ratio, and
trivial-baseline checks. This repeat-qualified baseline is not an accepted
autoresearch experiment, an Elo result, or a promoted checkpoint. Readiness
was opened later by the separate source-parity and preregistration gates.

The baseline-length prediction-SIGReg `0.57` experiment is rejected. Its best
two-pool CE is `4.511721` versus incumbent v2/u300 `4.510334`; a matched
update-400 audit also regresses accuracy, legal mass, and JEPA MSE for only
tiny rank/variance gains. Keep prediction-SIGReg at `0.0` for the norm-on
compatibility objective. The separately qualified corrected no-norm objective
uses prediction-SIGReg `1.0`.

The corrected no-norm target-SIGReg-5.76/prediction-SIGReg-1.0 baseline is
repeat-qualified at v2/update 400. Real-checkpoint GPU parity, sealed-history
timing, and both 128-pair cap-256 strength anchors are complete. It scored
`49.61%` against the recovered step-265,000 model and `38.09%` against raw
BT4. These are descriptive model-pool-relative results, not promotion or
absolute Elo. The fixed four-model representation comparison may proceed.

Keep searchless inference fixed at eight DFM refinement passes during the
initial stronger-model experiments. Balanced K=1 now improves the frozen
offline gates in two exact runs, but its 128-pair arena interval remains
unresolved.
Pass-count ablation stays deferred until a separately preregistered compute
study has sufficiently strong chess evidence. Changing refinement compute
must not be mixed into an architecture comparison.

The promotion assets are repaired and available. The source-derived v3 pool
replaces the old claimable-threefold root before selection is frozen, has zero
selected validation/test overlap, and passes production replay for all 2,048
histories. Runtime skipping or substitution remains forbidden. This asset
repair did not by itself open readiness. Dense representation drift is complete in
`artifacts/representations/dense-stage1-four-model-v2`. Official-epsilon
PyTorch/JAX source parity passes in
`artifacts/representations/upstream-bt4-source-parity-v1`; the pinned
TransformerLens constructor-default epsilon fails and must not be used as the
source oracle. The immutable preregistrations and completed outcomes of the
first nine post-baseline experiments are in
`research/experiment_future_target_sampling_k2.md`,
`research/experiment_sampled_target_anchors_k2.md`,
`research/experiment_projector_active_depth1_k2.md`,
`research/experiment_projector_depth1_predsigreg4_k2.md`,
`research/experiment_dfm_active_depth3_k2.md`,
`research/experiment_cosine_warmdown_k2.md`,
`research/experiment_cosine_floor001_k2.md`,
`research/experiment_future_target_sampling_k1.md`, and
`research/experiment_balanced_example_target_k1.md`.

Experiment 010's preregistration and completed outcome are in
`research/experiment_bt4_frozen_backbone_balanced_k1.md`.

Experiment 011's preregistration and completed outcome are in
`research/experiment_bt4_future_target_stopgrad_balanced_k1.md`.

## Editable surface

During automated architecture research, edit only `research/train.py`.

Treat these as immutable:

- `research/prepare.py`
- `research/program.md`
- the fixed data splits and arena FENs
- the recovered baseline checkpoint
- metric definitions and promotion gates

Changes to immutable support require a separate human-reviewed commit and a new
baseline version.

## Runtime contract

- Run entirely under `/mountpoint/.exp`.
- Source `research/env.sh` or use `research/run_gpu.sh`.
- Use one NVIDIA A10G.
- Measure compilation separately.
- Give each accepted quick experiment 30 minutes of steady-state training.
- Fix seeds and data order for the first comparison.
- Fix validation batch size independently of physical training batch size.
- Default to local JSONL/TSV logging; external tracking is opt-in.

## Primary result

Optimize fixed, seeded, globally permuted validation DFM cross-entropy. Also
report:

- first-action and per-horizon accuracy;
- legal mass;
- JEPA loss against zero, identity, shuffled-target, and action-shuffled
  baselines;
- per-horizon latent RMS, feature variance, and effective rank;
- examples processed and examples/s;
- compile time and peak HBM; and
- initialization lineage.

First-shard slices are correctness and calibration instruments only. Every
quality comparison must use identical `global_permutation` validation slots
for control and candidate, with the effective seed, schedule, batch size, batch
count, and finite-sample partition recorded. A validation sampler, batch
partition, or metric change requires rerunning the matched control.

Before readiness is opened, the comparison protocol is 64 validation batches
of 64 examples. Use seed 10,000 for development and seed 20,000 for the first
independent confirmation. Training-batch changes must retain those same 4,096
positions and partitions through `--eval-batch-size 64`.
Select checkpoints on the matched mean of both pools. For a near tie, add new
predeclared matched pools to every tied checkpoint; the current tie-break uses
seeds 30,000 and 40,000. Never choose an extra seed after inspecting only one
candidate.

The weighted training loss is not by itself a promotion metric.

The strength gate consumes complete color-reversed pairs from the repinned
promotion pool. Only the normalized-Elo GSPRT may promote a checkpoint:
`H0=0`, `H1=+20`, `alpha=beta=0.05`, checked after complete pairs and capped
at 2,048 pairs. Descriptive logistic Elo never authorizes promotion.

Reject a run if:

- any metric is non-finite;
- prediction variance or effective rank crosses the collapse threshold;
- the model fails to beat the trivial JEPA baselines;
- legality regresses beyond the fixed tolerance;
- an apparent improvement does not exceed repeated-run noise;
- the run writes outside the workspace; or
- validation data order or metric code changed without a matched control rerun.

## Experiment loop

1. Inspect the current code, previous result, and baseline noise.
2. State one testable hypothesis.
3. Make one coherent change to `research/train.py`.
4. Run correctness and short GPU smoke tests.
5. Repeat the baseline/candidate when bitwise repeatability is absent.
6. Outside the unattended loop, complete a matched 30-minute baseline
   qualification and open readiness only if its repeats pass every gate.
7. Once readiness is open, run the fixed 30-minute experiment.
8. Append exactly one row to `research/results.tsv` for that accepted
   30-minute experiment; leave smoke and pre-baseline calibration runs out.
9. Keep improvements that exceed baseline noise and pass every gate.
10. Revert rejected changes without rewriting the result history.
11. Promote only confirmed candidates to relative Elo.

Prefer simple changes whose effects can be explained. Record surprises and
negative results; they are part of the research output.
