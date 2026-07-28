# Autoresearch program

The goal is to improve the fixed held-out chess metrics of the local-GPU
BT4/DFM/JEPA model while preserving latent diversity and making training and
inference faster.

Current execution gate: `PYTORCH_AUTORESEARCH_READY = False`. The completed
JAX experiments remain valid historical evidence, but exact new JAX training
graphs cannot cold-compile inside the server's resource guard. Eager PyTorch is
therefore the candidate execution path for short autoresearch runs. It must
pass the migration gates below before the 30-minute loop reopens. Keep JAX as
the numerical, checkpoint, validation, arena, and eventual hero-run oracle.

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

The initial stronger-model experiments kept searchless inference fixed at
eight DFM refinement passes. The later preregistered compute study on the
retained Torch hero checkpoint is now complete: one pass beats eight by
`0.047852` paired score over 256 frozen opening pairs, with 95% interval
`[0.028427, 0.067276]`, while using only `0.640117` times the measured
policy-call latency. Its independent 128-pair confirmation slice is also
strictly positive. Freeze one pass as the next model-search inference budget,
retain eight passes as a continuity diagnostic, and leave the eight-action
training horizon unchanged. This is checkpoint-specific evidence; a model
family that changes refinement semantics must separately report its
pass-count sensitivity.

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

Experiment 012's preregistration and completed outcome are in
`research/experiment_bt4_unchunked_fused_balanced_k1.md`.

Experiment 013's preregistration and completed outcome are in
`research/experiment_bt4_future_tail3_balanced_k1.md`. The three-block tail
retains `149.531` examples/s and passes terminal accuracy/legal mass, but CE
`4.4977803` is inside incumbent repeat noise. It is rejected without repeat,
arena, or retained state. Loss remains fixed at norm/target/prediction
coefficients `0.0/5.76/1.0`.

Experiment 014's preregistration and completed outcome are in
`research/experiment_bt4_future_tail1_balanced_k1.md`. The one-block tail
processes `265,216/266,240` examples at `153.075/153.450` examples/s in exact
runs and selects CE `4.4950091/4.4971840`; both runs clear every frozen
offline gate. Its direct 128-pair arena against balanced K1 scores `49.609%`
(`-2.71` descriptive logistic Elo, pair-aware interval
`[-87.96,+82.20]`). One candidate loss comes from the symmetric declared
legacy-codec inability to represent black promotions and is already charged
in the score. Accept primary update 2,072 as the current offline incumbent,
not an Elo-promoted model, and retain only its candidate state. The tail-depth
line ends at one block. Loss remains fixed at norm/target/prediction
coefficients `0.0/5.76/1.0`.

Its direct 128-pair anchor against original raw BT4 scores `36.523%`
(`-96.02` descriptive logistic Elo, pair-aware interval
`[-195.33,-10.24]`), with 0 wins, 187 draws, and 69 losses. The candidate has
one known black-promotion codec fault and raw BT4 has complete coverage. Raw
BT4 remains clearly stronger, and the new checkpoint's point estimate is
1.5625 percentage points below corrected v2/update 400 on the same roots.
Future experiments must therefore report first-action quality and cannot
treat a small uniform eight-horizon CE gain as chess-strength evidence.

Experiment 015 is preregistered in
`research/experiment_dfm_first_action_share25_tail1.md`. It assigns 25% of
normalized DFM CE to the played first action and distributes 75% evenly over
horizons 2--8, while retaining uniform CE as a non-regression metric. It keeps
the accepted one-block tail and loss coefficients `0.0/5.76/1.0` fixed. A
candidate must repeat its horizon-1 gain and improve the frozen point score
against both update 2,072 and raw BT4 before becoming an Elo-aligned offline
incumbent.

Experiment 015 is complete and rejected at its primary legal-mass gate. Its
selected terminal update improves horizon-1/uniform CE to
`2.7167628/4.4811102` and passes accuracy plus every latent gate at `153.297`
examples/s, but legal mass `0.6371051` misses the `0.6467235` floor by
`0.0096185`. Run no repeat or arena, retain no candidate state, and restore
the unweighted one-block-tail surface. The source legality coefficient `2.0`
had relative weight 16 against uniform horizon-1 CE and only 8 against the
25%-share objective. A coefficient-4 rescue may be preregistered to restore
the original ratio; the legal gate itself must not be weakened.

Experiment 016 is preregistered in
`research/experiment_dfm_first_action_share25_legality4_tail1.md`. It retains
the 25% first-action CE allocation and changes only first-legality coefficient
`2.0 -> 4.0`, restoring the accepted objective's relative ratio of 16. The CE,
legal-mass, latent, repeat, and two matched arena gates remain frozen; the
preceding CE gain does not authorize weakening any threshold.

Experiment 020's predicted-state WDL auxiliary is rejected at its
exact-repeat gate. Primary update 2,065 passes at WDL CE `0.972390`,
H1/uniform CE `2.844706/4.496097`, and legal mass `0.647005`. Repeat update
2,067 still passes WDL, H1, accuracy, and every latent gate, but uniform CE
`4.498610` and legal mass `0.645706` fail their frozen thresholds. Run no
arena, retain no candidate state, restore `wdl_coeff=0.0`, and do not sweep
the coefficient.

Experiment 021's direct parallel multi-horizon JEPA is rejected at guarded
compilation. Commit `a7f0f58` preserves the parameter/state ABI and passes 108
focused CPU tests, but the batch-128 one-update smoke reaches
`10,965,164,032` bytes process-group RSS after `39.236` seconds, above the
frozen `7,516,192,768`-byte ceiling. The guard exits 75 while host
MemAvailable remains at least `6,194,348,032` bytes; no process survives and
no checkpoint/report is written. Per preregistration, run no profile,
fixed-time training, repeat, evaluation, or arena. Remove the empty run
directory, restore recurrent mode, and retain direct execution default-off.

Experiment 022's proposal-derived final-pass JEPA feedback is rejected at
guarded compilation. Commit `803a3be` adds no parameter or optimizer leaves,
keeps exactly eight DFM inference calls, and passes 155 guarded focused CPU
tests. Its batch-128 one-update smoke reaches `11,095,023,616` bytes
process-group RSS after `83.417` seconds, above the frozen
`7,516,192,768`-byte ceiling. The guard exits 75 while host `MemAvailable`
remains at least `6,095,749,120` bytes; no process, report, compiler artifact,
checkpoint, or state survives. Per preregistration, run no profile, inference
benchmark, fixed-time rescue, repeat, evaluation, or arena. Remove the empty
run directory, restore `jepa_feedback_mode="none"`, and retain the capability
default-off.

Experiment 023 is preregistered in
`research/experiment_bt4_policy_distillation_tail1.md`. It reuses the already
encoded current-board tokens and immutable source BT4 policy head to teach the
DFM's legal-masked root ranking, with exact canonical-to-legacy remapping from
classical side plane 108. It adds no encoder call, trainable/checkpoint state,
or inference work. The teacher is exact raw BT4 at initialization and a
frozen source head on online encoder tokens thereafter. A no-update two-pool
calibration fixes the one allowed coefficient by
`clip(0.25 / pooled_root_KL, 0.05, 1.0)` before the guarded training smoke;
there is no weight or temperature sweep.

The two no-update calibration pools pass with root KL
`0.8583731093/0.8729048609`, full eligibility, and pooled source
teacher/student top-1 agreement `0.5622558594`. Pooled
`K0=0.8656389850657433` freezes the coefficient at
`0.2888039983331077`, for weighted source KL exactly `0.25`. No optimizer
update or checkpoint was written.

Experiment 023 is rejected at the guarded training-compilation gate. With the
frozen coefficient, the batch-128 one-update graph reaches
`10,945,781,760` bytes process-group RSS after `78.9626` seconds, above the
fixed `7,516,192,768`-byte ceiling; the guard exits while host
`MemAvailable` is still `5,851,963,392` bytes. No report, compiler artifact,
checkpoint, state, GPU process, or nonempty run directory survives. Run no
profile, fixed-time rescue, repeat, or arena. Restore coefficient zero and
retain the codec-aware distillation capability default-off.

Experiment 024 is preregistered in
`research/experiment_root_legal_conditional_ce_tail1.md`. It leaves uniform
full-vocabulary CE and the existing legality objective unchanged, then adds a
root-only played-action CE normalized over stored legal actions. This directly
optimizes the ranking used by legal-masked inference with zero direct gradient
on illegal logits and no additional model call or inference work. A two-pool,
no-update source calibration freezes its one coefficient to make the initial
weighted component `0.25`; no coefficient sweep is allowed. The fixed guard,
offline gates, repeat, and both incumbent/raw-BT4 point-score gates remain in
force, with at most one intermediate plus one terminal state.

The two 4,096-position calibration pools pass with conditional CE
`2.1329845171/2.0417164937` and full eligibility. Pooled
`C0=2.0873505054041743` freezes the coefficient at
`0.11976905620438309`, for weighted source loss exactly `0.25`. Pooled
legal-only top-1 is `0.4224853516`; no update or checkpoint was written.
The read-only accepted-update-2,072 control has conditional CE
`2.0518385936`, legal-only top-1 `0.4296875`, and full coverage. That exact CE
is the frozen candidate/repeat conditional-ranking gate.

Experiment 024 is rejected at its guarded training-compilation gate. The
batch-128 one-update graph reaches `10,943,864,832` bytes process-group RSS
after `79.0055` seconds, above the fixed `7,516,192,768`-byte ceiling; the
guard exits while host `MemAvailable` remains `6,315,798,528` bytes. No
update, report, compiler artifact, checkpoint, state, GPU process, or
nonempty run directory survives. Run no profile, smaller-batch rescue,
fixed-time run, repeat, or arena. Restore coefficient zero and retain the
sparse legal-ranking capability default-off. Four consecutive new training
graphs now identify cold XLA compilation as the immediate systems bottleneck.

Experiment 025 is a compiler-only recovery gate, preregistered in
`research/experiment_xla_autotune0_compile_memory.md`. Preflight inspection
shows that this XLA build already has LLVM-module parallel compilation
disabled, so the initially considered single-thread flags would be redundant.
The one allowed cold-cache trial instead sets stable GPU autotuning level from
`4` to `0` on the unchanged accepted one-block-tail graph. It retains batch
128, the fixed `0.0/5.76/1.0` latent objective, all data and schedule
contracts, the 7 GiB guard, and zero checkpoint writes. A successful compile
still cannot be adopted unless it matches the retained smoke numerically and
keeps at least 95% of cached profile throughput.

The one allowed Experiment-025 cold compile is rejected. Autotune level 0
reaches `11,098,423,296` bytes group RSS after `92.8240` seconds versus the
unchanged `7,516,192,768`-byte guard, while host `MemAvailable` remains
`6,314,930,176` bytes. This is effectively the same footprint as the four
default-level failures, so kernel autotuning is not the cause. No update,
report, checkpoint, state, process, or cache artifact survives; run no
numerical or profile stage and do not adopt or sweep the flag. The next
systems work must inspect host-resident initialization/import/compiler
lifetimes before another cold model graph.

Experiment 026 is preregistered in
`research/experiment_separate_compile_process.md`. The exact incumbent cache
hit peaks at `6,158,090,240` bytes group RSS and reproduces the retained
initial/final validation CE exactly. Glibc arena/trim controls save only
`16,277,504` bytes, so they are rejected without a cold trial. The new
systems hypothesis is to compile the shape-equivalent graph in a separate
guarded process before opening the 1.8 GiB legacy checkpoint, exit after
populating the cache, and then launch the ordinary guarded source-init trainer
as a cache hit. Batch, graph, loss, data, guard, and zero-checkpoint contracts
remain fixed; one fresh-cache compile is allowed only after CPU and cached
parity gates pass.

Experiment 026 stops at cached parity. Compile-only fits at
`4,531,879,936` bytes group RSS with exact compiler costs and no source-state
access, but its constructor model ABI digest differs from the restored digest.
The preregistered in-place mapping clear also changes one-update final CE from
`4.4934930801` to `4.4944372177`. Run no fresh-cache cold compile. Commit
`029f501` restores the ordinary process exactly; its cache-hit result returns
the retained CE and peaks at `6,156,472,320` bytes.

Experiment 027 is the single non-mutating correction in
`research/experiment_nonmutating_compile_process.md`. Only the disposable
compile process drops its local mapping reference; it cannot clear the mapping
or alter the ordinary trainer. It must first reproduce exact restored model
and optimizer ABI digests plus the cached systems gate. Only then may it spend
one fresh-cache guarded cold compile, followed on success by an ordinary
one-update cache-hit parity check.

Experiment 027 is rejected at that first gate. The cached process safely peaks
at `4,541,132,800` bytes group RSS and preserves exact compiler and optimizer
fields, but its model ABI remains `f9f9...b467` rather than the required
restored `b53b...d13`. Reference deletion is therefore not the cause. No
dedicated cache or cold compile is created. Before any further compiler-memory
variant, run a read-only schema diff to identify the exact path, shape, dtype,
or container distinction.

Experiment 028, preregistered in
`research/experiment_compile_abi_schema_diagnostic.md`, is that read-only
diagnostic. The legacy payload and accepted research-checkpoint manifest both
carry `f9f9...b467`; only live post-restore state carries `b53b...d13`.
Apply the restore round-trip to constructor-owned arrays, compare complete and
leaf-only schemas plus array identity, and do not lower or compile a training
graph. Any leaf-level or value-identity change blocks another compile attempt.

Experiment 028 is blocked: constructor, accepted checkpoint manifest, and
self-round-trip are identical full `f9f9...b467` schemas, share leaf-only
signature `2888...0d3`, and retain all 455 array objects. The round-trip alone
does not produce `b53b...d13`, so no compile is justified. Next compare the
live model immediately before and after the actual checksum-verified legacy
model-only restore under the unchanged guard, without data, lowering,
execution, or state writes.

Experiment 029 is preregistered in
`research/experiment_legacy_restore_abi_diagnostic.md`. It performs one
production checksum-pinned model-only import, captures complete before/after
model schemas and fresh-optimizer invariants, and writes one compact report.
Persistent cache writes, data, lowering, model execution, updates, and state
writes are forbidden. No later compile is authorized by this diagnostic
alone.

Experiment 029 completes safely and finds 404 encoder-only leaf-record changes,
all `bfloat16` to `void16` with identical paths, shapes, and bytes. Source and
fresh-optimizer contracts remain exact. This blocks compilation under its
rule. The current ABI helper converts live NNX wrappers, whereas JAX PyTree
flattening exposes raw `.value` leaves; compare those canonical abstract
signatures next before deciding whether the mismatch affects an `nnx.jit`
cache key.

Experiment 030, preregistered in
`research/experiment_jax_restore_signature_diagnostic.md`, is the final
no-compile check. Compare raw JAX leaf paths and `jax.typeof` shape, canonical
dtype, weak type, sharding, and memory-space records plus PyTree and NNX graph
definitions around the verified restore. Only exact equality can replace the
broad reporting-ABI gate in a later separately preregistered compile test.

Experiment 030 passes exactly: all 455 JAX abstract records share digest
`5fc368...0817`, PyTree definitions share `775dd6...29f7`, and equal NNX graph
definitions share `213b0a...f1a4`; only concrete JAX-versus-NumPy storage
differs. Treat the `f9f9...b467`/`b53b...d13` reporting mismatch as irrelevant
to compilation. Do not cold-compile yet; first preregister and prove a
dedicated-cache compile-only/restored-execution compatibility gate.

Experiment 031 is preregistered in
`research/experiment_minimal_cache_compatibility.md`. Seed a disposable cache
with only the pinned active train/eval executables, run compile-only, then an
ordinary restored one-update parity check. Both must finish as cache hits,
create no executable key or checkpoint, and retain exact metrics and cache
bytes under the unchanged guard. No cold compilation occurs in this step.

Experiment 031 is rejected at its first gate. The installed JAX cache key
includes the cache-directory-derived per-fusion autotune path, so an exact
executable copied to a different directory misses. The guard stops the
unintended isolated cold compile at `8.575 GB`; no executable, update, or state
is written, and the restored stage is cancelled. Remove the disposable cache
and reduce guard polling from 250 ms to 50 ms before any further compiler
trial.

Experiment 032 is preregistered in
`research/experiment_abstract_compile_arguments.md`. Only the disposable
compiler may replace concrete dynamic arguments with exact
`ShapeDtypeStruct` signatures before lowering. Require shared-cache
compiler/HLO parity and at least 1.5 GiB live-buffer release first. A single
fresh-cache batch-128 trial is then allowed only under 50 ms polling and an
enforced 6.75 GiB group-RSS ceiling; ordinary training remains unchanged.

Experiment 032 is rejected at that cached gate. Abstractification preserves
all signatures and compiler fields and releases `1,894,688,512` live JAX GPU
bytes, but cached peak group RSS is `4,906,700,800` bytes—`71.4 MiB` above
the frozen 4.5 GiB ceiling. No cold cache is created and no restored update is
run. Before another compiler proposal, use read-only/CPU evidence to separate
abstract-state lowering overhead from graph compilation, and consider a
smaller physical compile batch with fixed effective-batch semantics; do not
retry this mechanism or relax a resource gate.

Experiment 033 is preregistered in
`research/experiment_source_encoder_overlay_diagnostic.md`. A guarded
checkpoint comparison finds only `2.5e-7`--`2.7e-6` relative L2 movement in
the accepted BF16 BT4 encoder versus `4.96e-3` in the trained head; the last
block changes least. Before proposing a trainable tail, evaluate the accepted
head twice on the frozen two-pool validation set, first with its accepted
encoder and then with only the source encoder overlaid in memory. This is a
shared-cache, zero-checkpoint diagnostic. Its result chooses between a
forward-identical frozen autoresearch proxy and a split-gradient/FP32-master
systems path.

Experiment 033 is rejected and functionally inconclusive. Its accepted
control exactly reproduces the retained metrics, but the run fails before the
overlay because the evidence hasher cannot cast NumPy BF16 buffer format `E`
through `memoryview`. The guard remains healthy at `3.778 GB` peak RSS and
`8.420 GB` minimum MemAvailable; zero checkpoints are written, all 67 cache
executables remain byte-identical, and the seven retained states are
unchanged. Harden the helper with a BF16 regression test, do not rerun this
experiment, and conservatively route the next systems proposal to a
forward-identical split-gradient or FP32-master path rather than a
last-block-only tail.

Experiment 034 is preregistered in
`research/experiment_split_encoder_gradient_execution.md`. FP32 BT4 masters
alone would add `390,611,456` live model bytes without reducing the
monolithic compiler graph, so first split reverse-mode execution into exact
encoder-forward, head-VJP, encoder-VJP, and existing-optimizer-update
components. The loss, forward pass, global clip, Muon state, optimizer step,
model/checkpoint ABI, batch 128, and `0.0/5.76/1.0` objective stay fixed.
Each cold component compile is isolated under a 6.75 GiB RSS ceiling and the
ordinary guard; only a ten-update parity smoke and a cached 30-update profile
can authorize FP32 masters as a later experiment.

Experiment 034 is rejected at its second compiler gate. Encode compiles in
50.789 seconds at `4.920 GB` peak group RSS with only `151 MB` compiler
temporaries, but head VJP crosses the frozen 6.75 GiB group limit and is
terminated at `8.006 GB` after 34.724 seconds. It writes neither an executable
nor a checkpoint; encoder VJP and update are not attempted. The seven retained
states remain unchanged. Static inspection shows that NNX still presents the
entire model state to each component before dead-code elimination.

Experiment 035 is preregistered in
`research/experiment_partitioned_split_model_views.md`. Build transient,
shared-variable encoder-only and head-only model views so encode/encoder VJP
omit `315,375,896` irrelevant bytes and head VJP omits `390,611,456`
irrelevant bytes, while the canonical full model and optimizer still receive
one merged, globally clipped update. Prove path, object-identity, gradient,
two-update, restore, and concrete-argument parity on CPU before repeating the
four sequential 6.75-GiB cold compiler gates. This changes no scientific knob
or checkpoint state.

Experiment 035 passes its CPU gates. The production views contain the exact
`404/51` encoder/head leaf partition, and their concrete encode, head-VJP,
encoder-VJP, and update arguments are `427.9 MB`, `386.2 MB`, `461.4 MB`, and
`2.567 GB`. The shared cache remains at 67 executables and the retained-state
count remains seven. Proceed to the four sequential guarded GPU compiles;
stop at the first failure.

Experiment 035 is rejected at head-VJP compilation. Partitioned encode passes
at `4.918 GB` peak group RSS, but head VJP is stopped at `7.676 GB` against
the unchanged 6.75 GiB ceiling. This is `330 MB` below Experiment 034's abort
RSS but still unsafe. It creates no head executable or state; later components
are cancelled, the encode cache entry is removed after preserving its report,
and the 67-executable/seven-state baselines are restored. Do not try another
graph-view variant. Preregister a method that materially reduces
differentiated head state or activation/compiler complexity and state its
scientific tradeoff explicitly.

Experiment 036 is preregistered in
`research/experiment_split_head_projection_vjp.md`. It materially splits the
failed head graph at its existing `(z_all, z_dfm)` boundary: a
`108,385,280`-byte / 17-leaf projector VJP and a `206,990,616`-byte /
34-leaf coupled DFM+JEPA core VJP. Compose those cotangents exactly with the
existing encoder VJP, then perform one unchanged merged optimizer update. All
scientific knobs and checkpoint state remain fixed. Prove full gradient and
two-update parity plus six production concrete lowerings on CPU before six
sequential 6.75-GiB cold GPU compiler gates.

Experiment 036 is rejected at the cold core-VJP compiler gate. Encode and
project pass independently, but core VJP jumps to `11,656,556,544` bytes
process-group RSS against the unchanged `7,247,757,312`-byte ceiling. The
guard stops it, no core executable or state is written, later components are
cancelled, and the exact 67-executable/seven-state baselines are restored.
Do not retry or further split the exact core graph. Before another production
compile, preregister either a parameter-efficient scientific method that
materially reduces differentiated state/activation scope or a separately
reviewed execution-framework change. Preserve the fixed loss/data/evaluation
contract, and keep SAE work behind a trained-model Elo gate.

## PyTorch eager execution migration

The separately reviewed execution-framework change is specified in
`research/pytorch_eager_migration.md`. It is a systems change only and does
not create a model-result row.

- Start with ordinary eager PyTorch. Do not use `torch.compile` for the
  baseline. Consider regional compilation only after the eager profiler finds
  a stable region whose measured gain justifies its compilation cost.
- Restore the pinned update-265000 source weights exactly, model only, and
  initialize a fresh optimizer. Preserve the accepted data, loss, gradient,
  validation, inference, and checkpoint contracts.
- Prove source-weight mapping, CPU forward/loss parity, explicit stochastic
  choice parity, full gradients, one- and two-update parity, strict checkpoint
  round trips, and JAX evaluation/inference parity before scientific use.
- Run every CUDA check through `research/run_gpu.sh`. First perform a
  no-checkpoint smoke/profile, then select physical batch size by measured
  throughput and memory headroom rather than freezing batch 128.
- Repeat the accepted fixed-time control on the new runtime before changing
  architecture or loss. Only that matched control can establish the PyTorch
  autoresearch baseline.

Migration execution update, 2026-07-23: the one-file eager trainer, strict
source mapping, CPU FP32 forward/loss parity, stochastic choices, custom
Muon/AdamW parity, guarded smoke, component profiler, and physical-batch sweep
are complete. Freeze PyTorch training batch 512: the final profile reaches
`160.790` warm device and `150.663` sequential end-to-end examples/s at
`12.571/14.972 GB` peak allocated/reserved HBM and `1.992 GB` peak host RSS.
Reject batch 768 because its small throughput gain costs about 5.2 GB more
allocated HBM and drops 25% of each 1,024-row shard.

Deterministic depth-1 prefetch is accepted and frozen. Its ten-update replay
matches every recorded scientific scalar exactly, hides all but `13.1`
microseconds of warm data preparation, and raises warm end-to-end throughput
from `150.663` to `160.490` examples/s (`+6.52%`) without increasing HBM or
host risk.

Representative FP32 gradient parity is complete. Across the production
projector/transformer and recurrent JEPA-transition classes, all 21 tested
parameter leaves and every input/condition cotangent pass
`rtol=5e-4, atol=3e-5`; the custom SIGReg value/latent gradient passes
`rtol=2e-4, atol=2e-6`.

The tight BF16 cross-framework gate fails and must not be relabeled. CPU and
GPU traces identify gradual rounding drift amplified by the recovered final
BT4 block, whose FFN layer-norm scale peaks at `6.125`; primitive-composition
and final-block-FP32 remedies both worsen parity and are rejected. Per the
evidence amendment in `research/pytorch_eager_migration.md`, retain exact FP32
formula/gradient/optimizer parity, treat BF16 optimization as
runtime-specific, and require the matched PyTorch control plus exported JAX
validation/eight-pass inference before opening readiness. Remaining work is
the real-control checkpoint/JAX round trip and that matched fixed-time
control. The model-only save/restore and NNX pure-tree path now pass exact
FP32 and BF16-bit unit tests. A guarded full-horizon PyTorch source validation
smoke also passes at batch 64, using 3.152 GB peak allocated HBM and 1.878 GB
peak process-group RSS; the focused migration suite passes 21 tests.

The first batch-512 PyTorch fixed-time control is complete: 564 finite updates,
288,768 examples, 1,801.31 training seconds, and 160.310 end-to-end examples/s.
All per-update loss components are retained. First-to-last 64-update means
change DFM CE `3.47652 -> 3.42899`, JEPA MSE `0.29881 -> 0.21416`, accuracy
`0.25073 -> 0.25367`, and legal mass `0.87280 -> 0.87942`. One terminal
706,033,120-byte checkpoint is retained; all 455 leaves and 705,987,352 tensor
bytes pass the strict pure-tree audit. The old update-count schedule was held
fixed as declared, leaving main LR `2.73275e-5` at update 564; compare an
example-scaled schedule after qualification and before architecture changes.

The platform usage-limit gate rejected the next elevated full-pool evaluation
launch and reported access resuming 2026-07-28 17:03 UTC. Do not bypass it.
PyTorch two-pool validation and the JAX state/validation/eight-pass gates
remain queued, so `PYTORCH_AUTORESEARCH_READY` stays false.

## Editable surface

During migration, edit `research/train_torch.py` and focused parity tests.
Support-file changes require explicit review. Once
`PYTORCH_AUTORESEARCH_READY = True`, automated architecture research edits
only `research/train_torch.py`.

Treat these as immutable:

- `research/train.py` (the JAX oracle)
- `research/prepare.py`
- `research/program.md`
- the fixed data splits and arena FENs
- the recovered baseline checkpoint
- metric definitions and promotion gates

Changes to immutable support require a separate human-reviewed commit and a new
baseline version.

## Runtime contract

- Run entirely under `/mountpoint/.exp`.
- Run every GPU process through `research/run_gpu.sh`; direct GPU Python
  commands are forbidden.
- Hold the launcher's host-visible single-workload lock across prepare,
  compilation, training, evaluation, and arena work.
- Limit each GPU job to two host CPUs, require at least 8 GiB MemAvailable at
  launch, and stop it above 7 GiB process-group RSS or below 3 GiB host
  MemAvailable.
- Refuse launch unless projected free disk remains at least 30 GiB.
- Disable periodic checkpointing. Write at most one sparse intermediate plus
  the terminal state, retain no more than two during selection, and prune to
  one accepted state or zero rejected states before the next model run.
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

Every actual training experiment must nevertheless retain its full loss
record: per-update total and unclipped loss plus DFM CE, JEPA positive,
target-SIGReg, prediction-SIGReg, legality, and any enabled auxiliary
component in `metrics.jsonl`, independent of console-log frequency. Each
PyTorch run must also write `loss_summary.json` with the exact terminal update,
the mean of the first and last 64 updates (or the whole run when shorter), and
their delta for every objective component and policy/latent diagnostic.
Retain final training and initial/final validation aggregates in `report.json`,
and the primary validation DFM CE in `research/results.tsv` for accepted
30-minute experiments. Training-curve plots may use `metrics.jsonl`, and
endpoint diagnostic plots may use the fixed last-window summary; the
cross-experiment quality plot must use the matched mean validation DFM CE over
the frozen seed-10,000 and seed-20,000 pools, not a noisy terminal training
batch or the weighted multitask loss. Build consolidated comparisons from
those immutable run reports rather than hand-editing historical summaries.
Compiler-only experiments execute zero model updates and therefore record
compiler/resource metrics with no synthetic loss value.

Once the PyTorch migration gate is open, model-side autoresearch uses chess
play as its primary ranking signal. Each completed 30-minute candidate first
passes the finite-loss, legality, checkpoint, validation, and latent-health
gates, then plays the same 128 color-reversed development pairs against the
retained Torch hero incumbent under the frozen one-pass,
deterministic-greedy, cap-256 arena contract. Optimize the pentanomial pair
score against that fixed reference and report its monotonic logistic-Elo
transform; do not optimize the numerically less stable Elo transform
directly. Record held-out DFM CE as a secondary explanatory metric, not the
final ranker. Native Torch-hero-versus-Torch-hero Arena support and a guarded
same-checkpoint parity smoke now pass: every one of 16 pair scores is exactly
centered, all 32 games terminate normally, and neither role has a fault, cap
draw, or incomplete legal-action coverage. The retained count-64 checkpoint
is therefore the fixed one-pass development reference for the next
candidate.
Periodically rerun raw BT4 as an absolute anchor; its one-pass score is already
near the ceiling and must not be the sole autoresearch ranker. Report
eight-pass results only for continuity or a predeclared pass-sensitivity
check, rather than spending that compute on every rejected candidate.

Only model-side code may change inside this loop. Opening pools, histories,
action codecs, game adjudication, inference passes, batching, ply cap, and the
reference checkpoint stay frozen. A candidate that changes inference cost
must report latency and remains subject to the frozen inference budget. The
reused 128-pair development pool is a selection set and cannot authorize
promotion. Confirm winners on disjoint, predeclared pairs before the full
promotion GSPRT, and periodically rerun the raw-BT4 anchor. Any incomplete
action-codec coverage remains a charged fault and must be separated from model
strength before treating small Elo differences as scientific signal.

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
3. Make one coherent change to the active one-file trainer
   (`research/train_torch.py` for local short runs; port only selected
   candidates back to `research/train.py` for JAX hero runs).
4. Run correctness and short GPU smoke tests.
5. Repeat the baseline/candidate when bitwise repeatability is absent.
6. Outside the unattended loop, complete a matched 30-minute baseline
   qualification and open readiness only if its repeats pass every gate.
7. Once readiness is open, run the fixed 30-minute experiment.
8. If its offline health gates pass, run the frozen 128-pair development arena
   and rank it by paired score against the fixed reference.
9. Append exactly one row to `research/results.tsv` and one row to
   `research/arena_results.tsv` for that completed 30-minute experiment; the
   run ID and arena-report path bind the rows to their immutable detailed
   reports. Leave smoke and pre-baseline calibration runs out.
10. Keep improvements that exceed paired-development noise and pass every
    gate.
11. Revert rejected changes without rewriting the result history.
12. Promote only candidates confirmed on disjoint pairs to the normalized-Elo
    GSPRT.

Prefer simple changes whose effects can be explained. Record surprises and
negative results; they are part of the research output.
