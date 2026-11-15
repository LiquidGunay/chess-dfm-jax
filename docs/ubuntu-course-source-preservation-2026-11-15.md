# Ubuntu course source preservation — 2026-11-15

This branch preserves the local interpretability course source, documentation,
tests, compact source manifests/metrics and course-authored animation assets
before Ubuntu retirement. It is an archival source branch, not a site release.

The updated real-derived static HTML and the gitignored activation NPZ were
intentionally excluded from this commit. Existing HTML inherited from the base
commit is stale relative to the new notebook source. Do not deploy it as the
updated course. Follow ARTIFACTS_AND_LICENSES.md and PUBLISHING.md before
regenerating or publishing real-derived views.

## Reproduction entry points

Start with the repository README, pyproject.toml and uv.lock. Historical research
continuation is preserved separately on research/hero-value-pilots and
research/modal-experiments; their source, experiment recipes and evidence are
independent of this course branch.

For the course, see curriculum/interpretability/REAL_ACTIVATIONS.md for exact
model identities, selection, activation extraction and source requirements.
The source artifacts are intentionally external to Git; their paths and hashes
are recorded by the compact manifests. Materialize the authorized inputs before
using source mode or attempting a fresh extraction:

```bash
.venv/bin/python -m research.interpretability.course.build_activation_gallery --device cuda --count 8
.venv/bin/python -m research.interpretability.course.build_snapshot
```

The optional local tensor can be absent for snapshot use; its manifest retains
the expected identity. Rebuild the authored media only if needed using
curriculum/interpretability/animations/README.md. Re-export notebook HTML only
after the applicable publication review, using PUBLISHING.md.

## Validation performed before archival commit

The complete local working tree, including its uncommitted generated HTML and
authorized local activation/source files, passed:

- python -m pytest -q tests/test_interpretability_course.py: 47 passed;
- python -m research.interpretability.course.validate: 12 notebooks, 20 source
  records, materialized 30,830,817-byte activation gallery and 2 media assets;
- marimo check --strict curriculum/interpretability/notebooks; and
- git diff --check.

These results do not assert that a bare clone contains the local datasets,
models or freshly regenerated HTML. Full release validation must be repeated
after reconstructing those artifacts.
