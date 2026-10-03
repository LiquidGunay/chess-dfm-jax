# Strength-gated DFM research program

Updated from the user's priorities on 2026-09-29. The current implementation
is PyTorch; JAX files remain historical reference material.

2026-10-03 correction: use full LC0 board history in all new training,
including future-target encodes. The original trajectory-v3 Hero runs used
zeroed past-board planes, whereas PA-001 used full-history sequential data.
EV-005 tests Hero1 against a full-history raw-BT4 opponent and includes a
matched current-only opponent control. Prior raw-anchor scores are not
evidence of superiority to normally configured BT4. Historical checkpoint
input contracts stay immutable; changing their evaluation inputs requires
an explicitly labelled ablation, not a silent correction.

EV-005 closeout: on the same 128 color-reversed opening pairs, Hero1 p1
scored 18.75% against full-history raw BT4 versus 93.75% against current-only
raw BT4. Its native policy scored 41.80% against full-history raw BT4, with a
pair-aware 95% interval of 29.79%-53.80%. All 768 main games completed with
zero faults, cap draws or legal-codec coverage misses. This rejects the
interpretation that the old Hero1 p1 advantage established superiority to
normally configured BT4. No searched-strength or SOTA claim is supported.
Both smoke runs and all main matches are complete; evidence is recorded in
`research/analysis/ev005_full_history_bt4_20261003/`. The old Hero1-relative
representation-strength proposal remains unfrozen and requires recalibration
against this full-history anchor before architecture selection.

HH-002 / EV-006 closeout (2026-10-03): the 553-update full-history Hero-recipe
run and its common-pool historical validation are complete; both loss curves
are synced to the focused `chess-dfm-history` W&B project. On 128 paired
development openings per matchup, its one-pass DFM scores 65.82% versus
historical Hero1 but 30.47% versus full-history raw BT4. Its native policy
scores 48.24% versus raw BT4 (inconclusive), while eight-pass DFM scores
19.53% versus raw and 35.35% directly versus one-pass. All 1,792 main games
and 96 smoke games completed without faults, cap draws, or coverage misses.
This candidate is not promoted to searched evaluation.

The matched-initialization CPU weight audit finds 18.35% more backbone L2
movement than Hero1, concentrated in the embedding stack (+36.53%);
transformer-block movement is 0.85% smaller and native-policy-head movement
21.41% smaller. The backbone distance remains small: 0.04358% of initial
norm versus 0.03683%. This measures stored BF16 weights, not representation
quality or optimizer-master movement. The slightly contracted data and
553-versus-554 updates prevent a pure causal history-training attribution.
Frozen protocol, results, uncertainty and receipts are under
`research/analysis/ev006_hh002_play_20261003/`.

EV-007 closeout: inference-aligned legal policy CE was recomputed on the same
8,128 retained Hero positions and a separate common 8,192-position sequential
pool for raw BT4, Hero1 and HH-002 (native policy, DFM1, DFM8). Raw BT4 has
the lowest loss on both pools. HH-002 native policy improves over Hero1 but
still has worse loss than raw; its playing-strength difference from raw
remains inconclusive. DFM1 improves over Hero1 on both loss and direct play.
However, HH-002 DFM8 has worse loss than Hero1 DFM8 on both pools while
winning their direct match with 65.82% score (paired 95% interval
53.82%-77.82%). It also has better played-move top-1 accuracy. Thus CE is a
useful diagnostic, not a monotonic strength proxy or a substitute for paired
games against full-history BT4. These are deployment-policy losses, not the
teacher-forced/composite training objective. Data, results and receipts:
`research/analysis/ev007_policy_loss_play_20261003/`.

## 1. Complete the selected epoch

HH-003 schedule decision (2026-10-03): the user selected **warmup-stable-decay
(WSD)** for the upcoming history continuation so that training can be
extended. Keep the Hero peak rates (fresh modules `5e-4`, BT4 `5e-4 / 30`)
and the original absolute 566,866-example warmup; extending the horizon must
not rescale or restart warmup. HH-002 stopped at 566,272 examples, before
its first post-warmup update, making its update-553 full recovery state the
selected parent. Preserve its historical cosine contract unchanged.

Retain a full model/optimizer/counter/data/RNG recovery state on the stable
branch **before any cooldown**. Cool down a separately identified evaluation
branch; extend from the retained stable state, not the cooled evaluation
model. Evaluation cooldown work may therefore need repeating at a longer
horizon. A 20% linear cooldown to the existing 0.001 LR ratio is proposed,
not frozen; the final boundaries depend on audited data and budget. WSD
changes exposure to peak LR and is not itself evidence of stronger play.

This selection is recorded in
`research/analysis/hh003_history_wsd_20261003/plan.json`. Trainer/launcher
changes and an explicit child checkpoint contract are still required; do
not bypass strict resume checks or switch to `hero_v2`, which changes other
optimizer settings. The remaining full-history corpus must be prepared;
repeating the short HH-002 subset is not the requested original-data epoch.
No new GPU run or budget increase follows from this schedule decision.

GPU-002 systems diagnostic (2026-10-03): the full-history, fixed-weight
FP32/BF16/FP16 eager/compiled matrix and separate production-storage control
completed on one L40S. Compiled FP16/BF16 both take about 0.096s per
forward/backward at batch 64, versus about 0.18s fully eager; strict FP32 is
slower. FP16 remains substantially closer to the FP32 reference numerically.
The existing runtime already compiles fresh regions: its median is 0.118s,
so the relevant exploratory median comparison is about 1.23x speed, not
1.9x. Cold compiled first passes take 76--100 seconds. These measurements
exclude optimizer/data/checkpoint work and are not production-batch or
strength results. The main matrix promotes stored checkpoint values to FP32
parameter/gradient storage, unlike the existing mixed-storage control; no
training runtime was silently changed. Next require batch-1024 full-step
and bounded training-stability/common-validation checks under a new explicit
contract. All GPUs stopped; evidence and limitations are recorded in
`research/analysis/gpu002_precision_compile_20261003/`.

Historical PA-001 is complete with its selected Proposal A recipe. Preserve the frozen
data order, optimizer, 7,774-update schedule, validation, and checkpoints.
The update-6,219 pre-decay state is a continuation asset; the update-7,774
model is the primary final artifact. A lower validation loss alone is not
a chess-strength result.

## 2. Establish searchless strength

Evaluate the terminal checkpoint against both the frozen Hero 1 reference
and raw BT4. Report one-pass DFM as the primary endpoint and eight-pass DFM
as a separately labelled compute/quality comparison. Keep the native policy
head as an attribution ablation, not an interchangeable score.

Before games begin, freeze exact checkpoint hashes, opening/history hashes,
disjoint development and confirmation opening sets, both colors, draw rules,
ply cap, fault handling, batching, and inference settings. Audit historical
opening usage before claiming any pool is unexposed. Use a small plumbing
smoke, then a bounded paired development screen; freeze the confirmation
sample size from its variance and a practically useful improvement target
before opening confirmation results. Repeatedly looking at a fixed-sample
interval is not a stopping rule.

Report wins/draws/losses, paired scores and confidence intervals, descriptive
relative Elo, policy faults, cap draws, latency, and evaluated positions per
second. Use pair-aware inference. Search promotion requires validated
improvement over the strongest incumbent baseline, with the primary
one-pass endpoint fixed in advance and multiplicity handled if more than
one comparison is used for a claim. An inconclusive result stays
inconclusive; decide whether to collect more evidence under a new bounded
protocol or revisit training.

The user requested that the minimum strength for later representation work
be proposed **after this first evaluation**. Do not choose a numerical gate
now or derive it after looking at architecture candidates. Propose and freeze
it using the baseline spread, uncertainty, and inference budget from EV-001.

## 3. Compare search against search

Only a model that passes EV-001 proceeds to EV-002. Compare raw BT4 in native
LC0 MCTS with Proposal A adapters in the same search implementation:

- full model: legal DFM root prior plus the WDL head applied to current `z0`;
- encoder ablation: updated BT4 native policy/value heads.

Freeze the LC0 build, nodes, threads, batching, cache, temperature, openings,
color pairing, and search parameters. Keep moves-left handling identical.
Verify legal-move order, policy normalization, side-to-move Q/D semantics,
history/castling/promotion encoding, and adapter parity before strength games.
Measure both matched-node quality and a separately labelled matched-wall-time
comparison: eight denoising passes change inference cost. A searchless model
versus a searching engine is not the primary comparison.

The existing native Torch arena is available; the Proposal-A LC0 search
adapter still needs implementation and parity validation. Do not describe
that adapter as ready merely because the training checkpoint exists.

## 4. Compare architectures, then their representations

Once the baseline and minimum-strength gate are frozen, vary one architectural
question at a time. Maintain matched data, seed sets, optimizer tuning budget,
training examples and compute reports, and inference budgets. Track both
terminal and pre-decay checkpoints where scientifically justified. Qualifying
models proceed to the same held-out representation corpus and analysis:
layer geometry/alignment, information probes, targeted interventions, and
sparse features. Keep seed variability and training quality separate from
architectural effects. Record weak models and failures, while reserving
expensive representation comparisons for models that meet the strength gate.

Candidate questions include state/action conditioning, JEPA coupling and
prediction horizon, recurrent refinement, and continuous diffusion/flow
matching in action-embedding space. For the continuous-space hypothesis,
preregister the clean target embedding, corruption/path, noise or vector-field
objective, time sampling, inference integrator and step count, discrete move
decoder, and legal masking. Compare with categorical DFM at matched network
evaluations and report latency as well. Separate learning the embedding from
learning the denoiser so a collapsed or poorly decoded space is identifiable.
This is a proposed experiment, not evidence that continuous diffusion will
be stronger.

The [registry](../research/experiments/registry.json) records dependencies;
the [workspace guide](../research/experiments/README.md) owns naming, evidence,
and storage retention. Broad sweeps remain deferred until the current epoch
and strength gates provide a useful baseline.
