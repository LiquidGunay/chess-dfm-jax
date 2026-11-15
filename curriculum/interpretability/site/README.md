# Read-only course editions

These generated HTML files are the blog editions of the executable course.
For live controls, immutable prediction forms, and editable code, open the
corresponding source notebook with `marimo edit`.
The pages load the marimo frontend from `cdn.jsdelivr.net`; they are read-only
course editions, not self-contained offline archives. They also embed notebook
source, so the submit-before-reveal exercises are open-book rather than secure
graded assessments.

## Core

- [00 — Evidence and experimental design](00_evidence_and_experiments.html)
- [01 — Chess and model anatomy](01_chess_and_model_anatomy.html)
- [02 — Behavior and paired model localization](02_behavior_and_localization.html)
- [03 — Representation geometry](03_representation_geometry.html)
- [04 — Probes and usable information](04_probes_and_information.html)
- [05 — Attribution and intervention](05_attribution_and_intervention.html)
- [06 — Lookahead as an evidential case study](06_lookahead_case_study.html)

## Advanced

- [07 — Sparse representation fundamentals](07_sparse_fundamentals.html)
- [08 — LoRSA and sparse model diffing](08_lorsa_and_sparse_diff.html)
- [09 — Circuits and causal abstraction](09_circuits_and_abstraction.html)
- [10 — Defend or falsify a mechanism](10_capstone.html)

## Visual essay

- [11 — From the policy lens to the Jacobian lens](11_policy_and_jacobian_lenses.html)

Modules 03 and 07 embed compact, captioned course-created visual essays on
Johnson–Lindenstrauss distance preservation and superposition. Each video has
a poster and full text transcript and does not autoplay.

The default editions of Modules 01, 03, 04, 05, 07, and 11 also render views
computed from the frozen real Raw/Hero activation gallery: numeric residual
rows, board-aligned deltas and probe errors, target-policy gradients, finite
spectra, and position-level legal policy trajectories. The HTML contains the
rendered view and exact notebook source, not either checkpoint or the complete
30.8 MB tensor pack. Live selectors require the executable notebook.
Both the local tensor and these rendered real-derived views remain under the
human publication gate in the artifact register; do not push regenerated real
pages to a public branch until that gate is closed.

The HTML is generated; review and edit the Python notebooks, not these files.
See [the publishing contract](../PUBLISHING.md) and
[artifact/license register](../ARTIFACTS_AND_LICENSES.md).
