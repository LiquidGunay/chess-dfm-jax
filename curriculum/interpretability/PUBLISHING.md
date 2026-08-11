# Static publishing contract

The executable marimo files in `notebooks/` are the source of truth. The
HTML files in `site/` are frozen, read-only blog editions generated from a
clean checkout. They make the narrative, code, evidence badge, source
boundaries, and resource footer available without a Python environment.

Rendered assessment feedback remains behind submitted forms, but assessment is
explicitly open-book and honor-system. Both `marimo edit` and the code-bearing
HTML expose notebook source, including feedback logic, to a reader who inspects
it. The forms enforce a pedagogical predict-then-reveal interaction; they are
not an exam-security boundary.

The generated files are not fully self-contained offline bundles. Marimo 0.23.3
loads its frontend assets from `cdn.jsdelivr.net`, so first browser use requires
network access (or a previously cached CDN response). The Python notebook and
snapshot validation path remains CPU/offline; publishing a genuinely offline
site would require vendoring and separately reviewing frontend asset licenses.

Regenerate from the repository root:

```bash
for notebook in curriculum/interpretability/notebooks/*.py; do
  stem="$(basename "$notebook" .py)"
  .venv/bin/marimo export html --include-code --force "$notebook" \
    -o "curriculum/interpretability/site/$stem.html"
done
```

Then run:

```bash
.venv/bin/python -m research.interpretability.course.validate
.venv/bin/marimo check --strict curriculum/interpretability/notebooks
.venv/bin/python -m pytest -q tests/test_interpretability_course.py
```

Do not hand-edit generated HTML. Never embed checkpoints, sparse weights,
training corpora, credentials, or unlicensed artifacts. The static editions
contain only code, constructed examples, and the content-bound derived
snapshot described in `ARTIFACTS_AND_LICENSES.md`.
The validator decodes each export's embedded `notebookCode` and requires an
exact match to the corresponding committed Python source, so stale blog
editions fail closed.
