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

## 1. Complete the selected epoch

Run PA-001 with the already selected Proposal A recipe. Preserve the frozen
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
