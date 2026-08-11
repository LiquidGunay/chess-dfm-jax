# Hero v2 optimizer and schedule plan

Status (2026-08-10): Gate 0, precision localization, the legality-zero objective ablation, immutable-Hero1 distillation, and the matched 1,024-update full-loop architecture ablation are complete. The fixed-teacher recipe remains the strongest one-pass Phase-1 lineage. Enabling proposal-conditioned predicted-JEPA feedback made recurrent inference decisively useful: the loop candidate at eight passes scored 0.74609 against its own one-pass exit and 0.81445 against the matched control at eight passes. However, its one-pass exit scored only 0.25781 against control one-pass (95% pair-aware interval 0.13777 to 0.37785). The exact preregistered mechanism-success rule therefore fails. Retain the loop architecture, reject the current final-only loss composition for a Hero-scale run, and next test multi-exit supervision that explicitly preserves the one-pass policy.

## Decision

Hero v1 is a sealed artifact. Its `HERO_CONFIG`, legacy optimizer state ABI,
and historical checkpoint semantics must remain unchanged.

Hero v2 is a separate Torch recipe with an independently serialized optimizer
policy. Its central configuration is:

- main peak learning rate: `5e-4`;
- encoder/main peak ratio: `1/10` as the central arm, with `1/30` and `1/3`
  required controls;
- schedule: 2% linear warmup, 78% stable plateau, 20% linear cooldown to
  `1e-3` of peak;
- precision: BF16 compute parameters with FP32 master parameters and FP32
  optimizer moments;
- weight decay: ordinary decoupled decay by default, fresh matrices only;
- decay coefficient: `0.01 / 1.7784570994812559 = 0.0056228514`, preserving
  Hero v1's planned integrated fresh-matrix shrink despite WSD's larger LR
  area;
- cautious weight decay: an explicit ablation, not bundled into the baseline.

The exposed training switches are `--recipe hero_v2`,
`--main-lr-multiplier`, `--encoder-lr-ratio`, `--lr-total-examples`,
`--lr-schedule-kind`, `--wsd-decay-fraction`, `--optimizer-precision`,
`--weight-decay-multiplier`, and `--weight-decay-mode`. The total-example
override preserves the frozen 2% warmup fraction while retargeting WSD to a
new corpus horizon. Objective interventions additionally use explicit,
serialized `--legality-coeff` and `--policy-distill-coeff` overrides. Policy
distillation is training-only and never changes inference. Its teacher is now
explicitly serialized as `--policy-distill-teacher online|checkpoint`; the
checkpoint form additionally requires a trusted checkpoint directory and exact
state SHA-256. Both modes minimize legal-support `KL(teacher || DFM-root)`, but
only the checkpoint form preserves an immutable external policy target.
Main-LR multipliers inversely scale weight decay so an LR screen does not
silently become a decay-exposure screen. All controls are recorded in the run
and exact-resume contracts.

## What transfers from Modded-NanoGPT

This review is pinned to official Modded-NanoGPT master commit
`f411b3d346aa52d3504324ca93c230fd84c6c07f` (2026-08-01). The benchmark is
deliberately narrow: reach FineWeb validation loss 3.28 as quickly as possible
on 8 H100s. Its README explicitly warns that some structure-imposing tricks
may not scale. We should transfer mechanisms with a plausible optimizer or
systems rationale, not its benchmark-specific architecture wholesale.

| Idea | Local decision | Reason |
|---|---|---|
| FP32 Adam buffers, FP32 Muon momentum, precision-preserving BF16 updates | Adopt now, using conventional FP32 masters/moments | Directly addresses Hero v1's measured BF16 update starvation. A normal FP32 master is easier to audit than upstream's uint16 mantissa reconstruction. |
| Long stable phase plus terminal cooldown | Adopt now as WSD | Makes the stable checkpoint reusable and separates exploration from terminal annealing. Cooldown fraction and shape remain empirical. |
| Cautious weight decay | Implement and test, but do not make baseline | The paper is positive, but Modded-NanoGPT's own multi-seed discussion reports ordinary decay can match it after retuning. |
| Muon momentum warmup/cooldown (`0.85 -> 0.95 -> 0.85`) | Next optimizer ablation | Plausible for Hero's large early clipped gradients, but enabling it now would confound precision, LR, and schedule. |
| NorMuon row/neuron-wise second-moment normalization | Later promoted ablation | Promising sample-efficiency with much less state than Adam, but changes the optimizer geometry and must follow a stable FP32-Muon baseline. |
| Polar Express instead of Newton-Schulz | Later throughput/parity test | Primarily a speed mechanism. It is useful only if output/update parity and real GPU throughput beat the existing five-step path. |
| Per-parameter beta/LR multipliers and paired-head Muon | Telemetry-led only | Upstream settings are architecture-specific. BT4 has 32 narrow heads, so copying GPT head grouping is unjustified. |
| Batch-size and sequence/window schedules | Do not copy | Our trajectory examples and fixed chess evaluation contract are not a token stream. |
| FP8 kernels, sparse bigram embeddings, MUDD/XSA variants, prefix-token loss | Future model-v3 work only | These change architecture, hardware assumptions, or objectives and would damage Raw/Hero interpretability comparability. |

Several useful speedrun ideas already have local analogues: Muon, BF16 compute,
QK normalization, XSA, regional compilation, asynchronous preparation, and a
multi-horizon predictive objective. They should not be counted as new Hero-v2
changes.

Primary references:

- [official Modded-NanoGPT repository](https://github.com/KellerJordan/modded-nanogpt)
- [current official trainer](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)
- [MiniCPM WSD paper](https://arxiv.org/abs/2404.06395)
- [river-valley WSD analysis](https://arxiv.org/abs/2410.05192)
- [cooldown dynamics study](https://arxiv.org/abs/2508.01483)
- [Cautious Weight Decay](https://arxiv.org/abs/2510.12402)
- [NorMuon](https://arxiv.org/abs/2510.05491)
- [upstream discussion including negative CWD results](https://github.com/KellerJordan/modded-nanogpt/discussions/23)

## Exact schedule budgets

These are discrete batch-1024 sums over 27,679 updates, using the scheduler's
actual next-update example convention.

| Schedule / encoder ratio | Sum of LR ratios | Main `sum(lr)` | Encoder `sum(lr)` | Encoder budget vs Hero v1 |
|---|---:|---:|---:|---:|
| Hero v1 cosine, `1/30` | 13,853.062990 | 6.926531 | 0.230884 | 1.000x |
| Hero v2 WSD, `1/30` | 24,637.078224 | 12.318539 | 0.410618 | 1.778x |
| Hero v2 WSD, `1/10` | 24,637.078224 | 12.318539 | 1.231854 | 5.335x |
| Hero v2 WSD, `1/3` | 24,637.078224 | 12.318539 | 4.106180 | 17.785x |

Peak-matched WSD has 1.778457x Hero v1's LR area. This is why `1/3` is not
the default despite being an important arm: together with FP32 accumulation
it is a large intervention. The central `1/10` arm is already a 5.335x
integrated encoder-LR increase before accounting for the removal of BF16
rounding loss.

With ordinary decoupled decay, Hero v1's exponent is
`0.01 * 6.926531 = 0.0692653`, giving `exp(-0.0692653) = 0.93308` or 6.69%
shrink. Schedule-normalizing Hero v2's coefficient to `0.0056228514` keeps
that exponent fixed. Cautious decay will shrink less, according to its
measured active-coordinate fraction; it must not be silently compensated by
raising the coefficient.

Zero-storage meta-device accounting gives model-plus-optimizer-state totals of
`1,864,249,672` bytes for Hero v1 and `3,225,426,248` bytes for Hero v2: an
exact increase of `1,361,176,576` bytes (1.268 GiB) for persistent
model/optimizer state alone. This excludes activations, gradients, allocator
slack, and compile workspaces. The historical Hero run already reserved
`23,072,866,304` bytes (21.49 GiB), so Gate 0 remains mandatory on any 24 GiB
GPU.

## Optimization-first boundary

The immediate objective is to produce a checkpoint that beats Hero v1, not to
repeat the complete interpretability program after every optimizer change.
During phases 1-3 retain only cheap selection evidence: loss velocity, frozen
validation, legality/calibration, paired play for promoted arms, update
numerics, and a compact parameter/encoder-distance audit.

Loss is a useful early screen, but lower training loss alone cannot declare a
better chess model. The fixed-teacher checkpoint has now cleared the matched
Hero1 Phase-1 behavioral gate strongly enough to justify targeted Raw
BT4/matched-Hero1-Phase1/candidate model diffing. Keep the expensive
confirmatory LoRSA, probe, sparse-feature,
causal-patching, and attribution stack reserved for the artifact that also
passes a non-overlapping strength confirmation and the selected representation
guardrail.

## Experiment sequence

### Gate 0: engineering and numerical smoke

Before renting a long-running GPU:

1. preserve the legacy partition manifest and v1 checkpoint tests;
2. require FP32 master/moment exact-resume continuation to match bit-for-bit;
3. on a real GPU, record allocated and reserved HBM after optimizer
   construction, first forward, backward, step, validation, and recovery save;
4. fail the 24 GiB arm if reserved HBM leaves less than 1 GiB headroom; retry
   on a 40/48 GiB GPU;
5. record throughput separately for compile, warmup, steady-state training,
   validation, and checkpoint I/O;
6. confirm nonzero FP32 master updates even when the corresponding BF16 value
   does not change.

Observed Gate-0 result:

- the Hero-v2 FP32 diagnostic peaked at 22.049 GB allocated and 30.021 GB
  reserved; the exact Hero-v1 control peaked at about 20.697/28.899 GB;
  therefore 24 GB GPUs are not eligible;
- matched 10-update diagnostics measured about 2.591 seconds/useful update on
  L40S and 2.525 seconds on A100 40 GB after compilation. At the frozen
  all-in rates of $2.395584/hour and $2.543184/hour respectively, L40S is the
  cheaper useful-update worker;
- the first long control was preempted near update 540. Modal documents that
  every Function is preemptible by default and restarts the same input, so
  every Phase-1 run now saves at updates 100/200/300/400/500/554, commits each
  atomic checkpoint immediately, archives partial attempts, and resumes only
  from a verified nonterminal state. Phase 2 uses checkpoints at
  954/1354/1754/2154/2554/2768.

See [Modal preemption](https://modal.com/docs/guide/preemption) and the
[long-training checkpoint pattern](https://modal.com/docs/examples/long-training).

### Phase 1: precision isolation at 2% epoch

At 2% the old cosine and new WSD schedules share the same linear warmup. This
is the cheapest clean boundary for isolating numerical precision.

| Arm | Schedule | Precision | Encoder ratio | Decay | Purpose |
|---|---|---|---:|---|---|
| H1 | Hero v1 cosine | parameter/BF16 direct | `1/30` | Hero v1 `0.01` | Exact incumbent learning-curve control |
| A | WSD | parameter/BF16 direct | `1/30` | schedule-normalized standard | Schedule/decay control without the precision repair |
| B | WSD | FP32 master/moments | `1/30` | schedule-normalized standard | Precision-only contrast with A |
| C | WSD | FP32 master/moments | `1/10` | schedule-normalized standard | Central encoder-LR arm |
| D | WSD | FP32 master/moments | `1/3` | schedule-normalized standard | High-adaptation arm |
| E | WSD | FP32 master/moments for `encoder.*`; parameter precision elsewhere | `1/12` | schedule-normalized standard | Preregistered precision-localization diagnostic after whole-model FP32 behavior failed |

The original five arms use the same ordered examples and run to 2% after a 100-update
compile, memory, throughput, and recovery gate on the target GPU. H1 must
reproduce the incumbent curve before it is used as a comparator. Every arm
retains a model-only terminal artifact plus periodic exact-recovery states;
each atomic recovery write is committed before training continues. The
approximate Hero-v1
A10G-equivalent cost is 0.87 GPU-hours per arm, 4.35 GPU-hours total, before
hardware speed differences.
The early optimization report includes raw and 32/128-update moving-average
losses, an examples-weighted loss slope after compilation, each objective
component, examples-to-predeclared H1 loss thresholds, validation at the
common 2% boundary, and useful examples/second. Rank only within the same data
prefix. A faster loss curve is evidence for allocating more compute, not a
claim of stronger chess play.

The completed H1 control trained for 1,778.451 seconds (2,507.113 seconds
including frozen validation), with 318.98 useful examples/second. Its final
64-update mean loss was 8.139934. The 8,192-position frozen pool measured total
loss 8.057726, DFM CE 5.894656, root legal-conditional CE 2.045060, first-legal
mass 0.436630, and legal top-1 0.455078. Exact parameter diffing against Raw
BT4 found 0.036826% trunk relative-L2 movement with only 1.7660% of trunk
scalars visibly changed at stored BF16 precision; these are screening
baselines, not strength metrics.

Observed matched Phase-1 evidence is:

| Arm | Matched paired loss delta (named reference) | Frozen legal top-1 | First-legal mass | Trunk relative L2 | Paired arena vs H1 | Decision |
|---|---:|---:|---:|---:|---:|---|
| H1 exact | - | 0.45508 | 0.43663 | 0.000368 | - | incumbent |
| BF16 WSD `1/30` | -0.01199 | 0.45178 | 0.44431 | 0.000369 | not run | reject: no meaningful adaptation; frozen root-CE/WDL gate fails |
| FP32 WSD `1/30` | -0.58391 | 0.40918 | 0.78708 | 0.002167 | 0.328 (16 pairs) | reject: material policy regression |
| FP32 WSD `1/12` | -1.00137 | 0.41455 | 0.95358 | 0.004493 | 0.320 (128 pairs; 95% interval 0.200-0.440) | reject: clearly weaker play |
| FP32 WSD `1/10` | -1.05839 | 0.40515 | 0.95145 | 0.005249 | 0.297 (16 pairs) | reject: policy regression and movement ceiling |
| Encoder-only FP32 `1/12` | -1.00278 | 0.41333 | 0.94176 | 0.004546 | not run | reject: encoder-localized regression; frozen validation fails |
| Encoder-trunk-only FP32 `1/12` | -0.94541 | 0.39905 | 0.93373 | 0.004460 | not run | reject: top-1 gate fails despite conventional validation gain |
| Main-only FP32 `1/30` | +0.00305 (inconclusive) | 0.45483 | 0.44072 | 0.000367 | 0.486 (128 pairs; 95% interval 0.366-0.606) | reject as model winner: play parity, no predictive-suite gain |
| FP32 WSD `1/12`, legality 0 | -0.23692 canonical | 0.43604 | 0.42863 | not required after gate failure | 0.434 exploratory (128 pairs; 95% interval 0.314-0.554) | reject: predictive win, preregistered root top-1 floor fails |
| + stopped online policy teacher | +0.00176 canonical vs legality-zero (inconclusive) | 0.46033 | 0.39724 | 0.004013 | 0.484 DFM (128 pairs; 95% interval 0.364-0.604) | original speed gate fails; policy-only arena 0.322 (upper 0.442) falsifies Hero1 preservation |
| + immutable Hero1 checkpoint teacher | -0.01714 canonical vs legality-zero (inconclusive) | 0.50061 | 0.37380 diagnostic | 0.003782 | 0.713 DFM (128 pairs; 95% interval 0.593-0.833) | retain Phase-1 continuation candidate; policy-only 0.443, sealed JEPA gate fails |

The arena is deterministic greedy DFM policy play from a sealed paired opening
pool, with colors reversed inside each pair, complete move-codec coverage, no
faults, and one refinement pass for both checkpoints. Its Elo is relative only;
the 128-pair result corresponds to about -131 paired logistic Elo with interval
[-241, -42]. This is enough to reject the FP32 `1/12` checkpoint even though it
crosses Hero-1 loss thresholds roughly 270k examples earlier.

The causal result is now narrower than “FP32 hurts play.” FP32 accumulation
unlocks much larger encoder updates, and encoder-trunk drift is sufficient to
cause the early policy regression: excluding the policy head does not rescue
top-1, while the main-only arm reproduces Hero-1 trunk movement and arena
parity. Aggregate action accuracy is not a behavioral substitute because it
averages all eight horizons and rises with legal-mass learning; legal-masked
root top-1 matches the deterministic one-pass DFM policy used by the arena.

Legality remains a diagnostic rather than the deployed action space: complete
board-aware masking gives illegal mass no direct selection value. The
legality-zero arm answered the causal question. Its canonical predictive loss
improved by 0.23692 versus Hero 1 (95% paired interval -0.32222 to -0.14755),
but frozen root top-1 reached only 0.43604 against the preregistered 0.44508
floor. The arm is therefore rejected even though its exploratory 128-pair
arena was not clearly worse: score 0.43359, pair-aware interval 0.31355 to
0.55363, paired Elo -46.4 with interval -136.1 to +37.4, no faults or caps.

The completed online-teacher arm kept the successful legality-zero optimizer
and predictive objectives fixed, then added unit-weight legal-support
`KL(policy || DFM-root)`. Because its stopped teacher shared the current
candidate encoder, the student used the exact-forward identity
`final + stop(base) - base`; this cancelled the policy-head path in backward
while retaining DFM-residual and encoder gradients. Its canonical predictive
curve tied the legality-zero control (candidate-minus-control mean +0.00176,
95% interval -0.01790 to +0.02561), so the original strict speed gate failed.
Frozen DFM root top-1 nevertheless rose from 0.43604 to 0.46033 and root CE
from 1.72520 to 1.60411.

That recovery did not establish model strength. The matched 128-pair DFM arena
against the Hero1 Phase-1 control scored 0.48438 (95% interval 0.36433 to
0.60442), which is parity. More decisively, the matched policy-only arena
scored 0.32227 with an
upper bound of 0.44231: 24 wins, 117 draws, and 115 losses. Parameter diffing
showed 0.4013% trunk relative-L2 movement, only about 2.2% less than the
legality-zero control. Thus the online objective constrained DFM relative to a
moving candidate policy; it did not preserve the sealed Hero1 policy.

The completed fixed-teacher follow-up replaces that moving target with the exact
Hero1 Phase-1 encoder and native policy head (state
`e518083ee5476e7ff15e7c34caf9c0505f8fb7d330ea5a6882e4e9e5c1ef1912`).
The 198,458,624-parameter teacher is eval-only, has zero trainable parameters,
and is bound into the run and resume identities. Unlike the online
exact-forward cancellation, checkpoint-teacher KL sends ordinary student
gradients through the current base policy, encoder, and DFM residual while the
teacher runs under `no_grad`. There is still no inference-cost change.

Its preregistered 10-update L40S diagnostic completed without nonfinite skips,
reserved 30.203 GB, and sustained a median 334.22 examples/second over updates
3-10. Exact manifest, state, run-config, and raw-source mapping identities all
matched Hero1; the diagnostic cost USD 0.1105. The content-bound 554-update
Phase-1 run completed at 256.65 end-to-end examples/second with state
`72a51f168b6b7146360e9aeb15bb1188b73890f642556f8be8255af88b488354`
and no nonfinite or skipped optimizer updates.

Frozen validation shows the intended behavioral effect. DFM root top-1 rose to
0.50061, versus 0.43604 for legality-zero and 0.45508 for Hero1, while root
legal CE fell to 1.47956. DFM CE, aggregate accuracy, WDL accuracy, and WDL
loss passed their sealed tolerances. The canonical predictive training-curve
difference versus legality-zero was -0.01714 with a 95% block-bootstrap
interval of -0.04166 to +0.00430, hence inconclusive. Frozen JEPA positive loss
was 0.24134 against a noninferiority bound of 0.20712, so the original gate
failed. First legal mass was 0.37380 but remains diagnostic only because arena
inference applies complete board-specific legal masking. WDL Brier improved,
while ECE worsened to 0.17699 and remains a calibration warning.

Parameter diffing measured 0.37816% trunk relative-L2 movement. This is 7.80%
less than legality-zero and 5.76% less than the moving-teacher arm; embedding
movement fell 14.74% versus the moving teacher. The changed-scalar fraction
remained comparable, so the fixed target constrained movement magnitude rather
than freezing adaptation.

The user-authorized exploratory DFM arena against the matched 554-update
Hero1 Phase-1 control then found a large practical effect:
127 wins, 111 draws, and 18 losses across 256 games, for score 0.71289 and a
pair-aware 95% interval of 0.59285 to 0.83293. Paired logistic Elo was +157.99
with interval +65.28 to +279.09. All 256 games terminated normally, with zero
faults, zero cap draws, and complete legal-action coverage. This is confirmed
superiority on the sealed development tier, not an absolute-Elo or formal
promotion claim. It does not establish superiority to the terminal one-epoch
Hero1 artifact, which was not an arena participant.

The matched policy-only arena scored 0.44336 (51 wins, 125 draws, 80 losses;
95% interval 0.32332 to 0.56340), which is parity-inconclusive. It is much
healthier than the moving-teacher policy-only score of 0.32227 but does not
explain the DFM result by itself. The major strength gain therefore resides in
the trained DFM path and its interaction with the inherited policy.

The first DFM attempt failed before playing a game because the arena validator
required two newly added teacher-provenance fields on the older sealed Hero1
run config. The compatibility fix permits those fields to be absent only for
legacy config loading; the candidate still records its fixed teacher
explicitly. Ten maintained Torch arena tests and the exact Hero1 config passed
before the content-hashed retry. Total metered spend for the failed attempt,
two completed arenas, and movement audit was USD 0.18577; all workers stopped.

`compare-policy-distillation` remains the audit for the historical online arm.
`compare-fixed-policy-distillation` additionally fails closed on teacher mode,
checkpoint SHA, zero trainable teacher parameters, and teacher diagnostics. It
uses the same canonical predictive estimand with legality and distillation
omitted. Under the sealed protocol the fixed-teacher result is not Phase-2
authorized despite its behavioral win. No coefficient sweep is authorized
before the teacher-versus-representation gradient audit and a new
preregistration.

The decision boundary is implemented in `research.compare_training_runs` as a
two-step protocol. `freeze-control` hashes the completed H1 compact artifacts,
freezes 32-update moving-average loss thresholds at updates
300/400/500/554, freezes 1%--5% loss-specific validation margins (with explicit
absolute floors), and declares a 10,000-replicate, 32-update paired moving-block
bootstrap over updates 101--554. Only after that immutable JSON exists may
`compare` inspect a candidate. Comparison fails closed on source-model, ordered
training-prefix, and frozen-validation-pool drift. A candidate is *faster* only
when the paired loss interval lies wholly below zero, *slower* only when it lies
wholly above zero, and otherwise inconclusive. Passing this screen allocates
more compute; it cannot promote a model without encoder-movement and chess
behavior checks. Optional content-bound movement reports make the continuation
gate executable: a candidate must exceed H1 in both trunk relative-L2 movement
and visibly changed trunk fraction, while remaining below 0.5% trunk and 1%
global relative-L2. This detects continued starvation or a runaway; it does not
assume that larger parameter movement means better chess.

Continuation requires finite updates, exact recovery, acceptable memory
headroom, no frozen-metric regression outside the predeclared tolerance, and
healthier encoder movement than Hero v1. Eliminate a dominated arm only when
its uncertainty band is separated on loss velocity or it fails a guardrail;
do not force an arbitrary top-two cut at this cheap boundary.

### Completed full-loop architecture ablation at 1,024 updates

The user-authorized full-loop test compared two fresh, matched fixed-Hero1-teacher runs over the same ordered 1,048,576-example prefix. The control used no loop. The candidate enabled `predicted_jepa_tokens`, in which a first DFM planner proposes actions, JEPA consumes those proposals and their hidden states, and a second DFM planner consumes projected JEPA rollouts. The only fresh parameter delta was a 262,144-parameter `dfm_jepa_rollout_adapter.w`. Both arms completed all six recovery boundaries with zero skipped or nonfinite updates.

Offline metrics support the mechanism. On the frozen 8,192-example pool, candidate DFM CE improved from 5.50996 to 5.49604, WDL loss from 0.81543 to 0.77593, and JEPA positive loss from 0.15924 to 0.13615; root CE moved slightly backward from 1.46290 to 1.46809 while root top-1 was essentially unchanged. The candidate first planner had DFM CE 5.86561 and the JEPA-conditioned second planner 5.49604, a 0.36957 improvement. All eight JEPA horizons beat zero, identity, shuffled-state, and shuffled-action baselines, with prediction/target RMS ratios near one and no rank collapse. The loop adapter moved 0.50695 relative L2 with every scalar changed, while trunk movement was 2.53% lower than control. The effect is therefore a learned loop-specific computation, not runaway encoder drift.

The preregistered held-out development arenas used opening indices 256 through 383, 128 color-reversed pairs each, deterministic legal argmax, zero faults, zero cap draws, and complete codec coverage:

| Comparison | Score | Pair-aware 95% interval | Decision use |
|---|---:|---:|---|
| candidate p8 vs candidate p1 | 0.74609 | 0.62605 to 0.86613 | recurrence works |
| candidate p8 vs control p1 | 0.51172 | 0.39168 to 0.63176 | point-score recovery to parity |
| candidate p1 vs control p1 | 0.25781 | 0.13777 to 0.37785 | one-pass preservation gate fails |
| candidate p8 vs control p8 | 0.81445 | 0.69441 to 0.93449 | JEPA loop beats fair-compute control |

A separately preregistered exploratory arena found control p1 scored 0.79688 against control p8. Ordinary iterative unmasking is harmful at this horizon; JEPA feedback is what turns additional inference passes into useful computation. This exploratory comparison clarifies the mechanism but cannot alter the primary promotion rule.

The exact mechanism-success decision is **fail**, solely because the one-pass interval lies wholly below the 0.45 material-deficit boundary. This is a training-objective failure, not an architecture or representation-collapse failure. The architecture remains in the queue, but this checkpoint must not be extended as the next Hero run.

The next bounded architecture test adds deep supervision to the preliminary exit: apply the immutable-Hero1 teacher and legal-root/DFM objectives to both first-planner and final loop-conditioned logits. If one additional arm fits the cost envelope, combine this with pass dropout or a fixed mixture of one-pass and loop-conditioned batches. Screen at the same 1,024-update boundary, first requiring p1 noninferiority versus control and then p8 superiority versus its own p1. Only a recipe clearing both gates can receive a longer scaling preregistration.

The complete checksum-bound analysis is in `research/analysis/hero_v2_phase1/u1024/full_loop_ablation_decision.json`; the readable companion is `full_loop_ablation_report.md`. Metered spend for training, five arenas, and two movement audits was USD 6.00214; the final metered workspace total was USD 27.10745, billed cost was USD 0, and all workers stopped.

### Phase 2: encoder and main-LR selection on the stable plateau

Phase 2 remains locked under the current preregistration. The immutable-Hero1
teacher established confirmed superiority on the development DFM arena, but it
failed frozen JEPA noninferiority. The result is therefore retained as a
Hero2 Phase-1 continuation candidate rather than silently extended or
formally promoted.

Before any Phase-2 continuation beyond the bounded architecture screen, extend
the no-update loss-gradient audit so immutable
teacher KL is measured separately against JEPA, SIGReg, WDL, and ordinary
policy gradients for each parameter group. If conflict is broadly scalar,
preregister one lower or decaying teacher coefficient. If it is localized to
the inherited trunk, preregister one teacher-gradient routing or scaling arm.
Do not run both or sweep coefficients after seeing outcomes. In parallel, add a
content-bound arena contract for a non-overlapping opening slice; development
pairs already inspected cannot serve as independent confirmation. A new
preregistration must set the representation guardrail and strength rule before
launch. Approximate Hero-v1-equivalent runtime remains 4.35 A10G-hours per arm.
Compare using:

- frozen root legal-conditional CE, top-1, first-legal mass;
- DFM CE and JEPA MSE;
- paired Arena against the retained Hero incumbent and Raw BT4;
- encoder model diff by layer and parameter family;
- update/master-to-BF16-ULP ratios and fraction of master updates visible in
  stored BF16 weights;
- layerwise relative update norm, gradient norm, and clip scale;
- representation drift and calibration guardrails.

Only after one teacher/representation balance arm clears its new gates,
screen main-LR multipliers
`0.75`, `1.0`, and `1.25` with weight decay inversely normalized to keep the
planned integrated shrink fixed. Admit `1.5` only when the `1.25` arm is
finite, unclipped often enough to remain informative, and better on loss
velocity without a frozen-metric regression. Start these as short common-data
prefixes, then extend only non-dominated arms to 10%. This separates learning
rate from decay exposure and spends more of the search budget where loss is
actually falling faster.
`1/1` is allowed only if `1/3` remains stable, is still update-starved, and
has no behavioral or representational regression.

### Phase 3: schedule, decay, and optimizer refinements

Run the chosen standard-decay LR arm along the WSD stable branch. Save a full
recovery checkpoint immediately before the 20% cooldown. Fork that identical
checkpoint into:

- 20% linear cooldown to `1e-3` of peak (baseline);
- 20% cosine cooldown to the same floor;
- no cooldown / stable continuation control;
- optionally a 40% cooldown only if an earlier 60%-epoch recovery point was
  retained.

This common-prefix fork isolates terminal schedule shape cheaply. It does not
test CWD, because CWD changes the whole preceding trajectory.

After the LR winner is known, run a separate standard-vs-cautious pair from
the same initialization, first to 10%. Promote the cautious arm to a full run
only if its paired metrics and update telemetry improve. Report its active
coordinate fraction; do not interpret a lower effective decay amount as proof
that the sign rule itself helped.

### Phase 4: one full promoted Hero v2
If phases 1-2 expose a clearly faster stable recipe, spend the remaining
short-run search allocation in this order: cautious decay; the Muon momentum
`0.85 -> 0.95` warmup (and cooldown ramp only for terminal forks); then a
single NorMuon comparison. Polar Express is a throughput experiment and only
enters after numerical update parity. Each addition is crossed only with the
current winner, gets a same-prefix control, and must earn continuation through
loss velocity plus the frozen behavioral guardrails. Do not launch a broad
factorial sweep.

Only one configuration gets the expensive full run under the present budget.
The final choice requires a predeclared scorecard and blind evaluation. Keep
the pre-cooldown recovery state as the reusable continuation branch. The final
artifact must include the model-only checkpoint, exact recovery checkpoint,
run config, optimizer partition, complete metrics, cost record, and post-run
Raw/Hero-v1/Hero-v2 model-diff report.

Hero v1 took 156,603 wall seconds (43.50 hours) on an A10G at 184.28 examples
per second. Use that as an A10G-equivalent planning baseline, not a promise
for another GPU. Benchmark one update and at least 100 steady-state updates on
candidate GPUs before committing the full budget.

## Promotion scorecard

The primary model-selection objective remains chess behavior, not optimizer
telemetry. A promoted arm must satisfy all of:

- no nonfinite update or exact-resume failure;
- frozen validation improves on the weighted primary score;
- no material legal-masked move-selection or calibration regression; illegal
  probability mass remains diagnostic only;
- paired Arena confidence interval is not clearly worse than the incumbent;
- encoder changes are distributed enough to resolve the Hero-v1 starvation
  pattern without indiscriminate representation destruction;
- intervention and representation audits remain interpretable under the same
  Raw/Hero tooling;
- projected full-run cost fits the remaining workspace budget with a reserve
  for terminal evaluation.

## Later Modded-NanoGPT-derived ablations

After Hero v2 is stable, the next optimizer queue is:

1. Muon momentum ramp `0.85 -> 0.95` over warmup and back down during
   cooldown, crossed only with the selected LR;
2. NorMuon row/neuron normalization versus current Muon, with optimizer-state
   bytes and wall time reported;
3. Polar Express versus five-step Newton-Schulz under update-cosine, norm,
   convergence, and throughput tests;
4. telemetry-led beta or parameter-group LR changes for persistently
   over/under-updating families;
5. only then, a separately named model-v3 architecture program for residual
   routing, value embeddings, head grouping, or new auxiliary objectives.

Architecture changes must not be folded into Hero v2: the Raw/Hero-v1/Hero-v2
comparison is valuable precisely because the model graph stays comparable.
