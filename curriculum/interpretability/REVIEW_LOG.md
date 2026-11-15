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
| Reviewed implementation commit | `ca5828d02ab51aee05bc1206757ac708038cab08` |

Publication decision: **approve for merge into this repository and for the
documented network-backed static edition**. A standalone public-course
redistribution remains conditional on adding a root license text that matches
the repository's Apache-2.0 metadata. No external checkpoint, sparse weight,
or training corpus is bundled.

The reviewed implementation commit above was validated from a clean worktree:
39 course tests passed and the full source/snapshot/static validator passed.
This audit-only follow-up changes only this identity record; it does not alter
a notebook, support module, evidence artifact, test, or static edition.

## Round 5 — lens and authored-visual extension

Date: 2026-08-11. Initial independent-critic verdict: block on four major
scientific/pedagogical findings. Final verdict after the creator response:
**approve, with no remaining blocker or major**.

| Critic finding | Creator response | Closure evidence |
| --- | --- | --- |
| The toy OLS state regression was named too much like the canonical Tuned Lens, whose fitting objective is decoded-distribution KL. | Renamed the executable estimator “OLS affine state translator” in controls, payload keys, prose, and tests; retained Tuned Lens only as the canonical method definition. | Regression test rejects the old estimator key; closure critic verified the distinction. |
| The eight-position policy-lens curve hid legal renormalization while pre-mask legal mass was small. | Renamed axes, threshold, table, claim card, and snapshot semantics; added a separate four-arm pre-mask legal-mass curve and exact final cards. | Snapshot binds both metrics; all four final parity checks remain exact; no population or mechanism claim was added. |
| The initial superposition scene animated vector addition without defining interference. | Rebuilt the scene around the explicit decoder cross-term d1·(d1+d2), a numeric 0.57 cross-talk error, and an orthogonal zero-cross-term control; composed a stable poster and retimed every caption cue. | Independent frame/caption review passed; the conclusion now says interference is decoder cross-talk, not addition alone. |
| J-Lens, local Jacobians, J-space, and BT4 square routing needed sharper boundaries. | Replaced “global” by fixed corpus-average transport; labeled J(h)h noncanonical; separated J-Lens from J-space and its single-token limit; stated that bidirectional BT4 has no temporal square order analogous to t'≥t. | Closure critic approved the method and chess translation with no major. |
| JL matrices used separate color scales and the on-screen distortion label was ambiguous. | Rerendered both matrices on one shared scale; displayed maximum relative pair distortion as a percentage; retained the warning that JL is not a learned-feature theory. | Shared-scale poster and late theorem-warning frame were inspected; source and output hashes pass. |

The extension remains deliberately asymmetric in evidence strength. Module 11
derives canonical identity/logit, Tuned, and Jacobian lenses; executes a known
nonlinear toy with an OLS comparison; reproduces only the retained
eight-position native-policy-head pilot; and labels the real Raw/Hero chess
Jacobian transport as proposed work. No historical curve was promoted to a
fitted J-Lens result.

Final extension gates:

| Gate | Result |
| --- | --- |
| Course tests | 42 passed, including all evidence modes, reactive branches, media integrity, and exact static binding |
| Surrounding Torch interpretability regressions | 14 passed |
| Strict marimo graph check | passed for all 12 notebooks |
| Scientific closure review | approved; no blocker or major |
| Source verification | 18/18 checksum-bound source records reproduced |
| Authored media | 2/2 source/video/poster/caption records validated; 703,169 bytes including manifest |
| Static binding | 12/12 HTML editions; 3,428,954 bytes |
| Snapshot SHA-256 | 67158d4bd43f0effb2af5f9453a406112bf9d590b35b88172287479aa48b9d8c |

Publication decision for the extension: **approve as the network-backed,
read-only blog edition and executable CPU course**. A true Raw/Hero Jacobian
lens remains an explicitly future, resource-gated experiment.

## Round 6 — real Raw/Hero activation specimen

Date: 2026-08-11. The initial independent verdict blocked merge on public
redistribution, a stale post-provenance artifact, Module 07 ordering, and
imprecise VJP language. The creator's follow-up audit also found that notebooks
handled an absent local tensor honestly while snapshot validation and one test
still assumed it existed.

| Finding | Creator response | Closure evidence |
| --- | --- | --- |
| The public repository had no established permission to redistribute checkpoint-derived tensors or rendered views. | Kept the 30.8 MB NPZ globally ignored and untracked; made all six tensor-dependent notebooks show explicit absence cards; prohibited staging/pushing both the tensor and rendered real views pending human review. | `git check-ignore` resolves to the global `*.npz` rule; critic approved the local implementation but independently retained the public-publication block. |
| The first manifest did not close over every scientific source and probe contract. | Bound 33 exact code/data/model/probe sources plus Raw/Hero/layout/corpus/split/boundary/task ABI identities and an independent prediction-level probe recomputation. | Probe recomputation maximum absolute metric difference is `4.218847493575595e-15`; every gallery row is development; test remains untouched. |
| Provenance edits made the retained artifact, snapshot, and static pages stale. | Rebuilt locally on the GTX 1660 Ti, then regenerated the snapshot and all 12 static editions. | Gallery run `ae6d4eceadc00811d51aa738c7eafd30c5a4e406ece4feedd08123403fbdea04`; data SHA-256 `49f6f35271acaaaae052191f9a758695cb545ec6e9c20d3eb2c52a8cf02a5860`; exact static/source binding passes. |
| Module 07 taught the controlled superposition mechanism before exposing the real specimen. | Moved the finite-gallery Raw/Hero residual spectrum before the controlled animation and retained the 64-position transcoder result as the aggregate return. | Cold critic verified the real → controlled → aggregate sequence. |
| “One Jacobian row” overstated what the compact artifact retained. | Described the product everywhere as per-square norm and gradient-times-state reductions of one full endpoint-specific VJP. | Builder, manifest, snapshot, methods matrix, Modules 05/11, and documentation all deny both a fitted J-Lens and a causal-intervention claim. |
| A clean checkout without the intentionally ignored NPZ could not run the complete validation path. | Made deterministic snapshot construction recover only the expected tensor identity from the integrity-bound manifest; source verification skips only that explicitly release-gated absent file; report fields distinguish materialization and source verification; tensor-heavy assertions skip honestly. | A real filesystem relocation test passed 46 tests with one tensor-only skip and the full validator with `activation_materialized=False` / `activation_sources_verified=False`; the restored NPZ reproduced its exact SHA-256. |

Final local gates:

| Gate | Materialized result | Clean-checkout result |
| --- | --- | --- |
| Course tests | 47 passed | 46 passed, 1 honest tensor-only skip |
| Focused Torch interpretability regressions | 14 passed | not tensor-pack dependent |
| Full course validator | 12 notebooks; 20 snapshot sources; 8 real positions; 30,830,817-byte gallery; 2 media; 3,955,629-byte exports | same source/snapshot/media gates; 3,635,771-byte absence-card exports |
| Snapshot SHA-256 | `1b79baf8278f730333b35bc7052f73df6b054dfd6021aec70f175cdb55ec6973` | identical |
| Ruff / strict marimo / whitespace | passed | same source |
| Independent closure review | **local implementation approved; no blocker or major** | honest absence behavior approved |

Publication decision: **do not merge or publish the current real-derived
edition yet**. Human review must clear the Raw/Hero source-model and
source-position terms, and the repository's missing root license text must be
resolved. The alternative release path is an absence-card static edition that
omits rendered real-derived views. No tensor, rendered real view, commit, or
push was published as part of this round.
