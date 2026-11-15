# Static publishing contract

The executable marimo files in `notebooks/` are the source of truth. The
HTML files in `site/` are frozen, read-only blog editions generated from a
source checkout. They make the narrative, code, evidence badge, source
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

The two course-created animations are a different layer: their MP4 video, PNG
poster, and WebVTT captions are embedded directly into the relevant generated
pages, so they do not make a media request. They never autoplay and are paired
with a complete text transcript. Manim is an authoring dependency only; normal
course execution and HTML export consume the committed, hash-bound media. See
animations/README.md to intentionally rerender them.

Regenerate from the repository root:

The six real-result pages require the local, gitignored activation pack to
render their checkpoint views. Without it, the executable notebooks remain
readable and show an explicit unavailable-artifact card. Do not stage or push
a real-derived export until the review in `ARTIFACTS_AND_LICENSES.md` is closed.
A clean checkout can still rebuild and validate the manifest-bound snapshot:
the validator prints `activation_materialized=False` and never describes that
path as tensor-level verification. With an authorized local pack, it prints
`True` and rechecks the tensor plus every declared gallery source.

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
training corpora, credentials, or unlicensed artifacts. The release-safe
baseline contains code, constructed examples, the content-bound derived
snapshot described in `ARTIFACTS_AND_LICENSES.md`, and the two compact
course-authored animation triples recorded in their media manifest.
Selected real-activation tables, board maps, and curves are rendered outputs;
the 30.8 MB NPZ itself is never embedded into HTML. Executable local notebooks
load it through a strict manifest and degrade to an explicit absence card when
it is not materialized. Both the tensor and rendered real-derived views remain
release-gated: a public package must clear the source-model/source-position
review, or omit them and publish only separately reviewed summaries.
The validator decodes each export's embedded `notebookCode` and requires an
exact match to the corresponding committed Python source, so stale blog
editions fail closed. It also checks every animation source and media hash,
container signature, caption header, duration bound, and total size budget.
