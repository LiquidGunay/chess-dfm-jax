# May 2026 W&B lineage reconstruction

This directory turns the three disconnected joint-training runs into one
checkpoint lineage and keeps the evidence for the 100-epoch projection
reviewable.

Primary outputs:

- `report.html`: self-contained technical report.
- `wandb_last_run_reconstruction.ipynb`: executed notebook with zero cell
  errors.
- `artifact.json`: validated canonical report payload.
- `derived/summary.json`: compact machine-readable conclusion.
- `figures/`: static notebook figures.

The bounded raw W&B export remains local under `raw/` and is intentionally
ignored by Git (about 116 MiB compressed). It can be regenerated with the
credential in `/mountpoint/.exp/.env`; the exporter never prints or persists
the key.

Run from the repository root:

```bash
export XDG_CACHE_HOME=/mountpoint/.exp/chess-dfm-jax/.local/cache
export MPLCONFIGDIR=/mountpoint/.exp/chess-dfm-jax/.local/cache/matplotlib
export TMPDIR=/mountpoint/.exp/chess-dfm-jax/.local/tmp

.venv/bin/python research/analysis/wandb_last_run_20260725/fetch_wandb_lineage.py
.venv/bin/python research/analysis/wandb_last_run_20260725/analyze_lineage.py
.venv/bin/python research/analysis/wandb_last_run_20260725/make_notebook.py
.venv/bin/python research/analysis/wandb_last_run_20260725/build_artifact.py
```

The final HTML is generated from `artifact.json` with the Data Analytics
portable-artifact builder. On this server it passed canonical validation and
structural verification; browser interaction QA was unavailable because no
Chromium executable is installed.
