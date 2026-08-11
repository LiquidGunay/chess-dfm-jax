# Curriculum specification

Version: 1.0 creator–critic freeze, 2026-08-11.

## Purpose

This course teaches mechanistic interpretability as experimental science, not
as a gallery of activation plots. A learner should finish able to:

1. turn an interpretability question into a scoped estimand and falsifier;
2. trace the chess specimen from input planes to policy/value outputs and name
   valid hook and intervention boundaries;
3. distinguish behavioral, representational, component, and circuit evidence;
4. design paired comparisons, probes, attributions, and interventions with the
   right grouping, controls, and uncertainty;
5. explain what sparse replacement models do and do not license;
6. evaluate lookahead claims without conflating decodability, refinement,
   conditional prediction, or search; and
7. defend or falsify a mechanism with a reproducible evidence record.

The target is an advanced undergraduate, graduate student, engineer, or
independent STEM researcher. Prerequisites are matrix multiplication, dot
products and norms, elementary probability, confidence intervals, train/test
splits, Python/NumPy, and the idea of a neural-network forward pass. Chess
expertise is not required.

The scope is auditable in [METHOD_COVERAGE.md](METHOD_COVERAGE.md). It records
whether each major method is derived/executable, reproduced from a frozen
specimen, introduced only as a diagnostic, or deferred behind named
prerequisites. The course does not convert a historical pilot into a result by
mentioning its method.

## The claim matrix

There is no single interpretability ladder. Every claim is located on four
axes.

| Axis | Allowed values |
| --- | --- |
| Evidence operation | observation, prediction, intervention, interchange / causal abstraction |
| Target level | behavior, activation, representation, component, path / circuit |
| Scope | example, sampled population, task family, model family |
| Endpoint | representation metric, probe objective, logit, legal action, policy distribution, value, arena result |

Examples:

- “Feature 10843 is more active on own-knight tokens in held-out Raw
  positions” is an **observation about an activation**.
- “A grouped linear probe predicts third-ply destinations” is **prediction from
  a representation**. It does not show behavioral use.
- “Magnitude-matched ablation of decoder direction 10843 changes policy TV
  more than random and semantic-placebo directions on active positions” is a
  scoped **intervention on a component with a policy endpoint**.
- “This subgraph implements pin detection” additionally requires circuit-level
  faithfulness, completeness, necessity, sufficiency, and stability tests.

Every substantive result carries this claim card:

```text
Evidence operation and target level:
Scope and sampled population:
Endpoint and estimand:
Unit and grouping:
Intervention, if any:
Controls and identification assumptions:
Uncertainty and multiplicity:
Allowed conclusion:
Excluded conclusion:
Falsifier / promotion experiment:
```

“Causes” is allowed only for a named intervention, population, dose, and
endpoint. “Represents” must name the operational test. “Understands,”
“reasons,” “searches,” and “plans” are not metric names.

## Artifact and execution contract

Every empirical notebook exposes three explicit modes:

| Mode | Purpose | Permitted conclusion |
| --- | --- | --- |
| `toy` | constructed mechanism and implementation checks | about the construction only |
| `snapshot` | content-bound real sufficient statistics | reproduction of named frozen figures only |
| `source` | re-extract from checksum-bound retained artifacts | about those artifacts and their recorded sample |

The badge is unavoidable. Toy values never populate a Raw/Hero claim card.
Snapshot and source modes share downstream estimators, but their claim scope
stays explicit. Model reruns are separate resource-gated experiments; no
slider triggers a checkpoint load, GPU call, or network request. There is no
silent `auto` fallback.

The compact snapshot records schema, source path/SHA-256/size, source schema,
repository commit, extraction version, grouping scope, and redistribution
status. Corruption, source drift, order drift, or unknown schema is an error.
Static exports show the default snapshot narrative without requiring Python.

## Course-wide experimental invariants

1. Compare Raw and Hero on identical input bytes and definitions.
2. Split by game or trajectory before feature selection or probe fitting.
3. Retain position IDs, game IDs, ordering hashes, and opening-overlap audits.
4. Use paired estimators and game-cluster uncertainty when games are the
   independent sampling units.
5. Keep discovery, hyperparameter selection, held-out replication, and frozen
   test roles distinct.
6. Label exploratory, unresolved, falsified, negative-control, and equivalence
   results differently.
7. Use simple-input, shuffled-label, untrained/independent-model,
   random-direction, identical-model, and semantic-placebo controls where they
   answer the estimand.
8. State scan multiplicity; a post-hoc best layer is not confirmatory.
9. State intervention dose, norm, sign, density/manifold diagnostics,
   legality, and policy/value endpoints.
10. Reproduce every table from a content-bound record.

## Prerequisite graph

```text
00 Evidence & design
  └─> 01 Chess & anatomy
       └─> 02 Behavior & hybrid localization
            └─> 03 Representation geometry
                 └─> 04 Probes
                      └─> 05 Attribution & intervention
                           ├─> 06 Lookahead case study
                           └─> 07 Sparse fundamentals
                                └─> 08 LoRSA & sparse diff
                      03 + 05 + 06 + 08
                           └─> 09 Circuits & abstraction
                                └─> 10 Capstone
```

Modules 00–06 are the core. Modules 07–10 are the advanced track. Lookahead
does not depend on sparse methods.

## Module specifications

### 00 — Evidence and experimental design

Mission: turn vague mechanistic stories into testable, scoped claims.

Objectives: fill the claim matrix; choose the unit and grouping variable;
distinguish paired/unpaired estimates, uncertainty/equivalence, and
exploration/confirmation; detect leakage, repeated-opening dependence,
multiplicity, and post-selection; preregister a criterion and falsifier.

Lab: repair overclaims; switch position versus game bootstrap; create leakage
deliberately; reveal a hidden confounder; classify failed, unresolved,
negative-control, and equivalence results. The unseen transfer item audits a
language-model feature claim with no chess vocabulary.

### 01 — Chess and model anatomy

Mission: trace one example from chess state to outputs and hooks.

Objectives: learn coordinates, legal moves, check, attacks, pins, material,
UCI/SAN, castling, promotion, and en passant; trace LC0 112 planes to 64 square
tokens of width 1024; trace 15 bidirectional SmolGen blocks with 32 heads of
width 32; distinguish the published LC0 head lineage from the Torch Hero
policy plus pooled value/WDL outputs; place DFM actions and JEPA states;
distinguish parameters, activations, residuals, branches, and replacements; and
identify the exact `[B,64,1024]` hook ABI.

Lab: board/legal-move explorer, softmax and legality mask, shape-tracing puzzle,
hook-boundary matcher, and tiny attention pass. The language analogy explicitly
breaks: chess uses fixed jointly encoded state, bidirectional attention, an
exact simulator, and a legality-constrained action vocabulary.

### 02 — Behavior and paired model localization

Mission: measure what changed before interpreting why.

Objectives: compute top-k agreement, NLL, calibration, total variation, and JS
divergence; use exact paired positions and game-cluster bootstrap; read the
RR/RH/HR/HH encoder-head lattice; distinguish module localization from a unique
mechanism; and compare identical-model, shuffled-pairing, head, and encoder
controls.

Lab: inspect paired positions, vary policy temperature, contrast equal aggregate
accuracy with different errors, and decompose hybrid interaction. Real result:
encoder swaps strongly change policy while native-head swaps are nearly inert.
Limit: the 128-position development pilot has one placeholder game cluster and
cannot support population-strength uncertainty.

### 03 — Representation geometry and dense model diffing

Mission: understand what similarity metrics preserve before assigning meaning.

Objectives: derive cosine, centered linear CKA, Procrustes alignment, SVCCA
intuition, and symmetric relative L2; predict behavior under rotation,
permutation, rescaling, duplication, and noise; separate parameter movement,
geometry, and functional change; use identical-model, independent-seed,
random-rotation, and shuffled-pair nulls; and interpret low-rank deltas without
equating rank with mechanism.

Lab: transform a representation and submit metric predictions before reveal;
inspect layerwise Raw/Hero curves and parameter-delta spectra.

### 04 — Probes and usable information

Mission: test decodability without confusing it with use.

Objectives: fit a linear probe from first principles; split by game before
fitting and tune only on development data; compare raw-bitboard, shuffled-label,
untrained-feature, and capacity controls; understand selectivity, capacity
sweeps, and MDL intuition; state strictly predictive conclusions. Probability
calibration is introduced with behavior metrics in module 02 rather than
misattributed to a representation probe.

Lab: toggle leaked row split versus grouped split; sweep ridge/capacity; compare
simple-input and shuffled-label baselines; inspect layer selection and
multiplicity. Real negative evidence: future-move decodability yields no
systematic Hero advantage, and only one of 30 probe-direction policy intervals
excludes zero while four random controls also do.

### 05 — Attribution and intervention

Mission: progress from sensitivity maps to controlled causal effects.

Objectives: compare gradients, input-times-gradient, integrated gradients,
SmoothGrad, SARFA-style removal, and attention maps; run completeness and
parameter-randomization checks; define clean/corrupt/patch estimands; compare
random, norm-matched, positive, and semantic-placebo controls; diagnose dose,
sign, off-manifold risk, legality, and off-target effects; and separate
localization, necessity, sufficiency, and explanation.

Lab: change saturation and watch gradients fail; compare maps; choose a patch
layer/token/dose; inspect dose response and activation density. Retained
integrated-gradients completeness failures and the chess-pin intervention
caution are first-class evidence, not appendices.

### 06 — Lookahead as an evidential case study

Mission: decide what would justify a planning-like claim.

Objectives: operationalize future-move information, lookahead, amortized
evaluation, refinement, explicit search, and planning; combine probes,
interventions, behavioral endpoints, and alternatives; distinguish
recorded-action-conditioned JEPA prediction from predicted-action rollout;
distinguish DFM p1/p8 refinement from JEPA-mediated action selection; and design
an inference-matched rollout with exact board replay and illegal-prefix masking.

Central negative case:

- JEPA recurrence consumes its own latent but is trained/evaluated with the
  recorded action line and clean bidirectional DFM hidden states;
- open-loop p1/p8 arena behavior invokes DFM refinement, not JEPA; and
- neither result currently establishes inference-time JEPA planning.

Lab: toggle teacher versus predicted actions; replay a generated line; mask the
first illegal ply and all later horizons; compare identity, teacher-action, and
predicted-action state baselines; construct non-planning explanations that fit
each evidence subset.

### 07 — Sparse representation fundamentals

Mission: understand sparse replacement objectives before interpreting features.

Objectives: explain superposition and overcomplete sparse codes; distinguish
residual SAE, transcoder, crosscoder, and branch replacement; trace
reconstruction, sparsity, dead features, fidelity, and policy replacement;
navigate the reconstruction–sparsity frontier; understand why sparse does not
imply monosemantic.

Lab: vary sparsity/top-k on a ground-truth sparse mixture; compare activation
reconstruction with downstream replacement fidelity; retain feature-splitting
and dead-feature failures.

### 08 — LoRSA and sparse model diffing

Mission: use the published branch replacement without changing its estimand.

Objectives: trace LoRSA input `hook_attn_in` and target `hook_attn_out` at layer
14; explain learned Q/K heads, sparse OV features, SmolGen, and output mapping;
distinguish a LoRSA feature from a residual SAE feature; assess Raw source
fidelity, Hero transfer, support overlap, and source asymmetry; define token
semantic opportunities before selection; evaluate matched-random controls and
design a plausible semantic-placebo control without claiming it was run.

Lab: trace a two-source learned-attention calculation; vary discovery source,
activation threshold, and cross-model drift in a known toy; apply an
opportunity gate; inspect positive semantics, supported Raw/Hero disagreements,
a null, matched-random causal controls, and the missing semantic-placebo gate.

Real result pair: Raw and Hero replacement fidelity is strong, yet mean support
Jaccard is about 0.584 and resolved activation correlation about 0.673. Three
selected decoder directions pass scoped gates in both models; others fail or
remain underpowered. This does not establish native dense features,
input-concept causality, or internal search.

### 09 — Circuits and causal abstraction

Mission: turn component results into a minimal falsifiable algorithmic claim.

Objectives: distinguish locator, mediator, component, path, and circuit; test
faithfulness, completeness, necessity, sufficiency, redundancy, and
cross-position stability; formulate interchange interventions and an abstract
causal model; compare smaller, larger, independently seeded, and random matched
circuits; recognize automated-circuit metric circularity.

Lab: assemble a known-ground-truth circuit, remove redundant paths, run
interchange tests, and see which criteria a proposed Raw/Hero story still fails.

### 10 — Capstone: defend or falsify a mechanism

Mission: produce a reviewable scientific record, not a persuasive story.

Deliverable:

1. preregister one model-comparison or mechanism hypothesis;
2. fill the claim matrix and exact estimand;
3. freeze discovery, selection, held-out, and test roles;
4. specify operation-appropriate controls: observational and predictive work
   uses relevant baselines/shuffles/independent models, interventions use
   no-op/random/semantic-placebo/positive controls, and interchange work adds
   smaller/larger circuit alternatives;
5. define promotion, rejection, and equivalence criteria;
6. run only after the gate is signed;
7. report failed, unresolved, and negative-control results correctly;
8. bind every artifact and environment;
9. give the strongest alternative explanation; and
10. propose an independent replication.

A rigorously falsified central hypothesis can receive full marks.

## Notebook contract

Every notebook contains:

1. mission and observable success criteria;
2. prerequisite diagnostic and just-in-time glossary;
3. explicit evidence-mode badge;
4. minimal derivation with dimensions and assumptions;
5. submit-before-reveal prediction;
6. an interaction that changes an estimand or assumption, not merely a filter;
7. a positive example and a genuine limitation, failure, unresolved result, or
   negative control;
8. claim cards beside substantive results;
9. automated numerical and provenance checks;
10. retrieval from earlier modules;
11. unseen transfer assessment for every objective;
12. “what this does not show” and the strongest alternative;
13. resource, device, seed, dtype, time, and provenance card;
14. primary sources labeled by publication status; and
15. a static narrative coherent without executing code.

## Assessment mapping

The module forms below are formative predict-then-reveal checks. Passing their
length validation proves only that a response was committed before the rubric
appeared. It does not prove correctness. Summative assessment requires a human
or peer to apply the named criteria to the capstone evidence record.

| Competency | Direct evidence |
| --- | --- |
| Claim calibration | unseen claim repair in 00, 04, 06, 10 |
| Model anatomy | manual shape/hook trace in 01 |
| Experimental design | leakage/grouping audit in 00/02/04 |
| Representation metrics | unseen transformation prediction in 03 |
| Probe validity | capacity/control design in 04 |
| Causal design | semantic-placebo and dose protocol in 05/08 |
| Planning claims | predicted-action rollout preregistration in 06 |
| Sparse-model ABI | architecture-boundary classification in 07/08 |
| Circuit evidence | necessity/sufficiency/interchange audit in 09 |
| Reproducible synthesis | capstone manifest and falsifier in 10 |

## Creator–critic release process

The course ships in three tranches even when implemented on one branch:

1. foundations: 00–03;
2. causal case studies: 04–06;
3. advanced sparse/circuit work: 07–10.

Each tranche requires a creator draft, scientific critic, cold-learner pedagogy
critic, runtime/accessibility critic, and a response log mapping every finding
to a change, deferral, or explicit rejection. A blocker cannot be deferred.

Hard release gates:

- every notebook executes in a clean CPU/offline snapshot environment;
- no import-time network, GPU, or model load;
- snapshot/source corruption and schema drift fail closed;
- every empirical number is content-bound;
- toy output is never presented as Raw/Hero evidence;
- no unlicensed checkpoint, LoRSA weight, or dataset is redistributed;
- JEPA and p1/p8 claims use their exact estimands;
- probe-only evidence uses no causal wording;
- causal semantic claims include random, norm-matched, and semantic-placebo
  controls;
- repeated positions are grouped by game and scans state multiplicity handling;
- `marimo check`, static export, unit tests, and interaction smoke tests pass;
- keyboard use, color-safe redundant encodings, alt text, and a no-chess path
  are checked; and
- the review log has no unresolved blocker.

Scored gate: at least 90/100 overall and at least 80% in each category.

| Category | Weight |
| --- | ---: |
| Scientific validity and calibration | 30 |
| Pedagogical progression and explanation | 25 |
| Assessment validity | 15 |
| Interaction quality | 10 |
| Reproducibility and artifact integrity | 10 |
| Accessibility and chess independence | 5 |
| Writing, citations, and source status | 5 |
