# Method coverage and deliberate boundaries

“Use every interpretability technique” is not a reproducible requirement:
methods answer different questions, several names cover incompatible
estimands, and a picture is not evidence that a method was implemented. This
matrix records what the course derives, what it executes on a known mechanism,
what it reproduces from the frozen Raw/Hero evidence, and what remains an
explicit expansion rather than an implied result.

## Status vocabulary

- **Derived + executable:** the notebook exposes the equation, a visible
  minimal implementation, a changing interaction, diagnostics, and an unseen
  transfer assessment.
- **Frozen specimen result:** compact source-bound Raw/Hero statistics appear
  with a claim card. This reproduces the named retained sample; it is not a
  fresh model run.
- **Diagnostic introduction:** enough is taught to prevent category errors,
  but the method is not claimed as a complete Raw/Hero analysis.
- **Deferred:** prerequisites or a valid experiment are absent. The missing
  work and promotion gate are named below.

## Coverage matrix

| Method family | Course status | Where | Exact boundary |
| --- | --- | --- | --- |
| Paired behavior, encoder/head swaps, distribution metrics | Derived + executable; frozen specimen result | 00, 02 | Localizes model and readout differences; does not identify an internal algorithm. |
| Direct checkpoint activation inspection | Real tensor gallery + frozen specimen result | 01, 03 | Actual 1,024-coordinate rows, square RMS, channel deltas, and depth trajectories are visible; a coordinate is not thereby a feature or mediator. |
| Dense representation similarity (cosine, CKA, Procrustes, delta spectra) | Derived + executable on constructed and real frozen activations; aggregate specimen result | 03 | Invariance is metric-specific; similarity is not semantic identity or causal use. |
| Random projection and Johnson–Lindenstrauss geometry | Derived + executable visual experiment | 03 | One sampled map is not a proof; the finite-set maximum distortion, target dimension, and seed are part of the diagnostic. The lemma is not a theory of learned features. |
| Linear/MDL-style probes with simple, shuffle, capacity, and grouped controls | Derived + executable; real trained-probe prediction gallery; aggregate specimen result | 04, 06 | Measures usable decodability under a frozen split/selection rule; not policy use. One-board maps are case studies, not new estimates. |
| Concept activation / steering directions | Diagnostic introduction; retained scoped control result | 04, 05, 08 | A probe direction is not a native feature; causal wording requires a named boundary, dose, and matched controls. |
| Gradients and input × gradient | Derived + executable on a known mechanism; real target-log-odds residual VJP and gradient×state gallery | 05, 11 | Local sensitivity; the gallery retains per-square reductions of one full endpoint-specific VJP, not an intervention or fitted J-Lens. Saturation and parameter-randomization diagnostics remain mandatory. |
| Integrated gradients | Derived + executable; frozen failure retained | 05 | Baseline/path estimand with completeness diagnostic; the retained chess pilot fails its own check. |
| SmoothGrad | Hand-derived operation and perturbation contract | 05 | Noise-averaged local sensitivity; no standalone Raw/Hero map is promoted. |
| SARFA-style chess perturbation | Derived operational contrast; frozen descriptive agreement | 05 | Requires a valid perturbed board and target-versus-alternative legal-policy contrast. |
| Attention patterns | Derived tiny attention pass; frozen pattern comparison | 01, 05 | QK weights omit values, output projection, residual mixing, and downstream use. |
| Activation/residual patching | Derived design; frozen bidirectional whole-residual result | 05, 09 | Localizes a high-bandwidth mediator; does not isolate a head, edge, or minimal circuit. |
| SAE/transcoder reconstruction and replacement | Derived + executable; frozen specimen result | 07 | Sparse faithful replacement is not uniqueness, monosemanticity, or importance. |
| Superposition geometry | Derived visual explanation plus sparse-mixture lab; real finite-gallery residual spectra | 07 | More candidate feature directions than orthogonal axes is a capacity picture. Participation ratio and entropy effective rank are variance diagnostics, not evidence of superposition, a feature count, or monosemanticity. |
| LoRSA and source-asymmetric sparse model diffing | Architecture trace; frozen transfer, semantics, and controlled interventions | 08 | LoRSA features belong to the learned replacement; they are not automatically native dense-attention features. |
| Circuit criteria and causal abstraction/interchange | Derived + executable on a known Boolean mechanism | 09 | Raw/Hero has no complete, stable, interchange-consistent circuit result. |
| DFM refinement, conditional JEPA prediction, inference-matched rollout | Derived protocol; frozen mismatch and p1>p8 checkpoint result | 06 | Any pass-count effect is not JEPA feedback; teacher-action state prediction is not predicted-action planning. |
| Identity/logit, tuned, and Jacobian lenses | Canonical definitions derived; controlled transport lab; position-level real activation/policy-lens replay and endpoint VJP; aggregate eight-position native-policy-head trace; real chess J-Lens proposed | 11 | The OLS state translator is not a Tuned Lens. Real top moves, legality, residuals, and per-square reductions of a full endpoint VJP are shown, but no corpus-average Jacobian transport is fitted and no policy-use claim follows. |
| Direct-logit attribution | **Deferred** | next 11B | Must define a fixed legal-action contrast, decompose the nonlinear policy head or name a local linearization, and reconcile contributions to the actual final logits. |
| Per-head/MLP output decomposition | **Deferred** | next 05B | Needs head/branch hook parity, additive accounting error, value/output projection, and grouped stability—not attention weights alone. |
| Path patching / edge attribution | **Deferred** | next 09A | Requires clean/corrupt pairs, sender/receiver edges, matched node controls, multiple-comparison handling, and held-out paths. |
| Causal scrubbing | **Deferred** | next 09B | Requires an explicit high-level computational graph, correspondence, resampling distribution, and equivalence margin. |
| Automated circuit discovery (for example ACDC-style search) | Diagnostic introduction only | 09 | Discovery metric and graph assumptions must be audited on held-out interventions and against smaller/random subgraphs. |
| TCAV-style concept vectors | **Deferred** | next 04A | Requires independently labeled concepts, counterexamples, seed/capacity controls, and a model endpoint before it adds evidence beyond the course’s probes. |

## Expansion sequence

The next notebooks should be added in dependency order rather than as a
catalogue of plots:

1. **11A — real chess Jacobian transport:** require final-layer native-head
   parity, fit only on frozen game/root groups, compare structured and
   preregistered low-rank estimators with a small dense reference, and report
   held-out pre-mask plus legal-policy endpoints for Raw and Hero.
2. **04A — concept vectors:** freeze labels, entities, counterexamples, and
   capacity controls; compare probe, concept-vector, and sparse-feature bases.
3. **11B — legal-action logit accounting:** derive direct-logit attribution
   for one frozen action contrast, report the unaccounted residual, and test
   whether rankings survive policy normalization.
4. **05B — head/MLP decomposition:** include QK, OV, output projection,
   normalization, and downstream endpoint; compare with matched random heads.
5. **09A — path interventions:** progress from node patches to sender–receiver
   path patches on held-out game clusters, with smaller and redundant paths.
6. **09B — causal scrubbing:** preregister a chess computation graph and test
   interchange/equivalence across positions and independent seeds.

No deferred method is needed to understand the current course’s conclusions.
It is needed before making the stronger claim named in its row.
