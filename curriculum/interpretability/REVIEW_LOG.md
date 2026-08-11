# Creator–critic review log

This is a decision record, not a marketing changelog.

## Round 0 — creator blueprint

The initial design proposed eleven notebooks, a
descriptive/correlational/causal taxonomy with a separate behavioral channel,
and `demo`/`artifact`/`auto` runtime modes. Strong elements included claim
cards, paired Raw/Hero specimens, positive and null results, group-aware splits,
and a capstone that can succeed by falsifying its hypothesis.

## Round 1 — independent curriculum/science critic

Date: 2026-08-11. Status: all blockers resolved in the frozen specification.

| Severity | Finding | Creator response |
| --- | --- | --- |
| Blocker | Missing chess/model anatomy and hook ABI | Added an executable 112-plane → token → BT4/head → DFM/JEPA anatomy module and no-chess primer. |
| Blocker | Model diff preceded geometry; sparse work preceded lookahead | Kept only behavioral hybrid localization early, moved dense diff after invariances, and moved lookahead before sparse methods. |
| Blocker | Claim taxonomy mixed inference strength and behavioral target | Replaced it with evidence-operation × target-level × scope × endpoint axes. |
| Blocker | Tiny demos could masquerade as Raw/Hero evidence | Replaced silent auto mode with explicit toy/snapshot/source modes and unavoidable badges. Toy data cannot populate Raw/Hero claim cards. |
| Blocker | JEPA/p1-p8 planning overclaim | Made the teacher-action/clean-hidden versus predicted-action mismatch the central negative case; DFM-only p1/p8 is separate. |
| Blocker | No executable creator–critic loop | Added three tranches, three critic roles, response records, and hard/scored gates. |
| Blocker | Unlicensed LoRSA/checkpoint/data redistribution | Course includes derived metrics only and requires users to supply any unlicensed weights locally. |
| Major | Experimental design was diffuse | Put grouping, leakage, multiplicity, equivalence, and preregistration in module 00 and course-wide invariants. |
| Major | Sparse module overloaded | Split sparse fundamentals from LoRSA/model diffing. |
| Major | Manufactured null requirement | Replaced “one null everywhere” with correctly typed genuine failures, unresolved results, negative controls, or equivalence evidence. |
| Major | Probe controls incomplete | Added simple-input, shuffled-label, untrained, capacity sweep, nested selection, calibration, and MDL intuition. |
| Major | Intervention/circuit criteria incomplete | Added dose/sign, norm matching, semantic placebo, density, legality, necessity, sufficiency, redundancy, faithfulness, and completeness. |
| Major | Assessments measured activity, not mastery | Added unseen transfer items mapped to every observable objective. |
| Major | Chess accessibility underspecified | Added operational chess definitions and a complete no-chess path. |
| Major | Marimo interactions might merely filter plots | Every required interaction changes an estimand, control, grouping decision, or assumption; expensive work requires a run button. |

Explicit rejection: the critic suggested a “signed” snapshot. The implementation
uses content-bound SHA-256 provenance, not a cryptographic signature, because
no course signing key or trust root exists. It is labeled accurately.

## Round 2 — scientific and code audit

Date: 2026-08-11. Initial score: 82.5/100. Final score after two
creator-response passes: **96.5/100**. Final verdict: approve, with no content
blocker or major.

| Severity | Finding | Creator response | Final verification |
| --- | --- | --- | --- |
| Major | Toy anatomy declared 16 tokens while teaching one token per 64 board squares. | Made the toy architecture a coherent 64-square reduced-width model and parameterized every shape sentence. | Toy/source/snapshot runtime and explicit 64-token assertion pass. |
| Major | Sparse derivation required nonnegative coefficients, but the toy used signed Gaussian codes and signed top-k. | Changed the generator to nonnegative lognormal sources and the approximate encoder to linear map → ReLU → top-k; exposed coefficient-minimum diagnostics. | Both ground-truth and encoded minima are tested nonnegative. |
| Major | The alleged semantic placebo was Gaussian noise and no distinct positive control ran. | Froze an orthonormal constructed concept basis: pin candidate loading 0.8, own-queen placebo loading 0, matched random loading 0, endpoint-gradient positive loading 1. | Numerical effects and all four plotted arms are regression-tested. |
| Major | Module 09 silently mixed a historical 16-position transcoder row with the current 64-position specimen. | Rebuilt the synthesis row from `published_tc_l14_transfer_dev64_v1`, added visible run identity, and used the measured 0.972× Hero/Raw NMSE ratio. | Snapshot/source equality and exact run/count/ratio assertions pass. |
| Major | Capstone export could pass with frozen decision gates, replication, or reviewer response omitted. | Centralized every stated deliverable as a hard gate, added real calendar validation, and added one blank-field adversarial test per gate. | An omitted replication plan prevents the stopped JSON export cell from executing. |
| Major | The all-mode smoke covered startup, not reactive branches. | Kept it accurately labeled as startup smoke and added live backend branches for source verification, nondefault number controls, submitted forms behind `mo.stop`, successful export, and blocked export. | The final suite executes these branches rather than inspecting source strings. |
| Major, found on re-audit | Capstone control scoring rewarded causal controls even for observation/prediction projects. | Made control families operation-aware and limited the exhaustive-control hard gate to causal operations. | Observation, prediction, intervention, and an independently checked interchange record all export above 90 using only applicable controls. |
| Minor | M05 alternative text omitted the added positive-control curve; a 2026 preprint title was shortened. | Named all four curves in alternative text and used the exact title in both the source register and Module 08. | Source/static binding and final source scan pass. |

The critic independently confirmed the Torch specimen ABI, LoRSA folding and
normalization, feature-result classifications, checkpoint-scoped p1/p8
negative, DFM/JEPA separation, grouping rules, and content hashes. No claim
was promoted merely because its code path executed.

## Round 3 — cold-learner and pedagogy audit

Date: 2026-08-11. Initial score: 90.5/100. Final score: **94.5/100**, with
every category at or above 80%. Final verdict: content approved.

| Finding | Creator response | Final verification |
| --- | --- | --- |
| Forms preserved predict-before-reveal but response length was not evidence of correctness. | Labeled module forms formative and open-book; summative use now requires capstone criteria plus independent human or peer review. | README, curriculum, publishing contract, and learner-facing labels agree. |
| Material and en passant were named more deeply than taught. | Added an operational material definition, a valid en-passant explorer state, and an explicit e5d6/d5/immediate-reply explanation. | Critic independently verified legal `exd6 / e5d6`. |
| Notebook source citations were not clickable in the blog reading path. | Replaced code-formatted relative paths with Markdown links in every module. | All eleven source notebooks and static editions expose the source-register anchor. |
| Scope could be mistaken for an exhaustive methods survey. | Positioned the release as a rigorous foundations-and-case-studies v1 and added a status/boundary matrix for implemented, diagnostic, retained-only, and deferred methods. | No deferred method is presented as a completed experiment. |

The final cold-learner review highlighted the prerequisite graph, no-chess
path, first-principles calculations, genuine negative evidence, and strict
claim language as release strengths.

## Round 4 — runtime and accessibility audit

Date: 2026-08-11. Initial score: 88/100 because marimo 0.23.3 displayed labels
without programmatically naming many nested text fields and slider thumbs.
Final verdict: **runtime/accessibility approved**, with no blocker or major.

Creator responses:

- replaced every slider thumb with a bounded, labeled number control;
- supplied explicit placeholder fallbacks for blank text/text-area controls;
- made form validation reject empty, partial, and unexpected mappings;
- added real reactive branch coverage in addition to all-mode startup smoke;
- retained redundant chart encodings, semantic tables, board descriptions,
  keyboard paths, offline CPU modes, and exact static/source binding; and
- documented that static first load depends on marimo's jsDelivr frontend.

Live Chrome audit found **zero unnamed course-owned controls** in Modules 00
and 10. All 29 capstone text fields had programmatic names, and keyboard
traversal reached the complete form. One unnamed ellipsis remains in marimo's
upstream `notebook-actions-dropdown` toolbar chrome; it is not authored or
configurable by the course. Number-control names are nonempty and
distinguishable, though marimo 0.23.3 noisily serializes Markdown markup into
some accessibility names.

## Final release gates

Date: 2026-08-11.

| Gate | Result |
| --- | --- |
| Course tests | 39 passed, including all modes and reactive allow/block branches |
| Torch + course regression matrix | 303 passed, 2 skipped; 4 non-failing warnings |
| Strict marimo graph check | passed for all 11 notebooks |
| Ruff and whitespace checks | passed |
| Source verification | 18/18 checksum-bound source records reproduced |
| Static binding | 11/11 HTML editions embed their exact Python source |
| Static payload | 2,164,882 bytes |
| Snapshot SHA-256 | `e75d545dafa1e32bf6dcf334bafa9a30c01cf7667838f3fbe44eeaad616fa68f` |
| Reviewed implementation commit | recorded in the audit-only follow-up commit after the content commit |

Publication decision: **approve for merge into this repository and for the
documented network-backed static edition**. A standalone public-course
redistribution remains conditional on adding a root license text that matches
the repository's Apache-2.0 metadata. No external checkpoint, sparse weight,
or training corpus is bundled.
