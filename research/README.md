# Local GPU research path

The implementation contract is
[docs/local_gpu_autoresearch_plan.md](../docs/local_gpu_autoresearch_plan.md).

Before running local commands:

```bash
source research/env.sh
```

This creates and exports workspace-local locations for temporary files,
dependency caches, JAX compilation artifacts, data, checkpoints, profiles, and
experiment runs. The script refuses to configure itself if the repository is
outside `/mountpoint/.exp`.

The intended stable interface is:

```text
prepare.py        fixed data and asset validation
import_legacy.py  checksummed, restricted source-checkpoint boundary
train.py          single editable model/objective/training surface
inference.py      checked cached-BT4 multi-pass inference and profiling
arena.py          frozen pools, paired outcomes, and arena statistics
local_policy.py   strict current-only localized DFM arena adapter
play_arena.py     bounded batched gameplay with exact history replay
evaluate_arena.py strict in-process relative-Elo CLI and resumable blocks
program.md        rules for automated research
results.tsv       compact append-only experiment ledger
```

All mutable state, including Python and JAX caches, remains below
`/mountpoint/.exp`.

The parity oracle is run separately:

```bash
research/run_gpu.sh .venv/bin/python research/legacy_baseline.py
```

It uses strict model-state restoration and writes a deterministic loss
fingerprint under `artifacts/baselines/`.

The first compatibility trainer is:

```bash
research/run_gpu.sh .venv/bin/python research/train.py --steps 1 --batch-size 1
```

`research/train.py` now owns the checkpoint-visible model, stage-1 objective,
optimizer, training step, and validation step. The legacy implementation is
used only as a test oracle; fresh runs cross `import_legacy.py` after exact
size, digest, serialization, and ABI checks.

For a checked-in experiment, edit the small `EXPERIMENT_OVERRIDES` mapping near
the top of `train.py`. Effective configuration precedence is:

```text
source checkpoint metadata -> EXPERIMENT_OVERRIDES -> explicit CLI flags
```

The final effective configuration is written to the report and bound into the
resume contract. Unknown keys, wrong types, and non-default legacy knobs that
the local graph cannot honor fail before training.

Evaluate every retained checkpoint from one run without rebuilding or
recompiling the model for each checkpoint:

```bash
research/run_gpu.sh .venv/bin/python research/train.py \
  --eval-checkpoints research/runs/<training-run> \
  --eval-batches 64 --eval-batch-size 64 \
  --eval-checkpoint-seeds 10000 20000 \
  --run-id <training-run>-checkpoint-eval
```

This mode takes the model and objective from the checkpoint contract, reuses
the same validation positions and RNG for every checkpoint, and never restores
optimizer moments or writes beneath the evaluated run. Incremental results go
to `checkpoint_metrics.jsonl`; the aggregate and minimum-DFM-CE checkpoint go
to `checkpoint_summary.json`.

The paired evaluator now has a real 16-pair source-vs-source A10G correctness
artifact, strict source/research-checkpoint loading, frozen history validation,
atomic resumable blocks, bounded model batches, a lean action-only inference
path, and an official-formulation normalized-Elo promotion GSPRT. See
`docs/local_relative_arena.md` for commands and limitations. The harness
remains intentionally marked
`autoresearch_ready=false` while the target-scale-stable objective and its
representative global-validation baseline remain unresolved. The promotion
tier is additionally fail-closed until one claim-draw-terminal frozen history
is replaced by a regenerated, repinned pool. See the plan and baseline report
for the measured results.
