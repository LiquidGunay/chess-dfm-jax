# Mechanistic Interpretability Through Chess Transformers

This directory is a from-first-principles, executable course for a STEM
audience. Raw BT4 and Hero are the recurring paired specimens; controlled toy
models are used only when the ground-truth mechanism matters more than realism.
It is a rigorous foundations-and-case-studies v1, not a claim to implement
every method ever used in chess or language-model interpretability.

The course has a seven-module core and a four-module advanced track:

```text
Core
  00 evidence and experimental design
  01 chess and model anatomy
  02 behavior and paired model localization
  03 representation geometry
  04 probes and usable information
  05 attribution and intervention
  06 lookahead as an evidential case study

Advanced
  07 sparse representation fundamentals
  08 LoRSA and sparse model diffing
  09 circuits and causal abstraction
  10 capstone: defend or falsify a mechanism
```

Start with [the curriculum](CURRICULUM.md), even if you already know
interpretability. It defines the claim language, artifact modes, prerequisites,
and release gates used throughout. [The review log](REVIEW_LOG.md) records the
creator–critic decisions rather than hiding them. [The source register](SOURCES.md)
labels literature status and [the artifact register](ARTIFACTS_AND_LICENSES.md)
states what may and may not be redistributed.
[The method-coverage matrix](METHOD_COVERAGE.md) distinguishes techniques that
are derived and executed from those introduced diagnostically or deliberately
deferred; absence is never presented as a completed negative experiment.

## Learn in a notebook

The local environment is intentionally Torch-only and already contains marimo.
Use edit mode for the course: it keeps every derivation and minimal NumPy
implementation visible and lets you change a specimen before rerunning it.

```bash
.venv/bin/marimo edit \
  curriculum/interpretability/notebooks/00_evidence_and_experiments.py
```

Use `marimo run` only for a presentation-style view where source code is
collapsed. The committed HTML editions in [site/](site/README.md) are
read-only blogs for browsing and citation. Rendered feedback remains behind
submission controls, but these are open-book self-assessments: edit mode and
the code-bearing HTML expose the implementation and answer logic to anyone who
inspects source.

The embedded transfer forms are **formative, open-book self-checks**. Their
submission controls preserve predict-then-reveal order, but a sufficiently
long response is not automatically correct and is not summative evidence of
mastery. Summative use requires the capstone's criterion checklist plus an
independent human or peer review of the exported record.

Every notebook defaults to a visibly labeled, content-bound snapshot. Switch to
`toy` inside the notebook for constructed mechanisms. Source-verified mode
re-reads retained repository artifacts and is deliberately behind an explicit
run button. No notebook performs a network request or loads a model at import
time.

Validate the complete course with:

```bash
.venv/bin/python -m research.interpretability.course.validate
.venv/bin/marimo check --strict curriculum/interpretability/notebooks
env PYTHONPATH=. .venv/bin/pytest -q tests/test_interpretability_course.py
```

Static HTML exports are publishing products, not sources of truth. Regenerate
them from the notebooks using [the publishing contract](PUBLISHING.md); do not
hand-edit exports.

## Evidence-mode badges

- **TOY — illustration only:** a constructed mechanism with known ground truth.
  It cannot support a Raw/Hero claim.
- **SNAPSHOT — frozen figure reproduction:** compact derived metrics, bound to
  their source hashes. It reproduces only the named table or figure.
- **SOURCE — source-verified extraction:** the notebook re-reads retained
  experiment artifacts and verifies their identities. It still does not rerun a
  model unless the resource card explicitly says so.

If a snapshot or source identity is wrong, execution fails closed. There is no
environment-dependent `auto` fallback.
