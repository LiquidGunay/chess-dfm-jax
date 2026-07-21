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
storage_audit.py  exact retained-state, dataset, archive, and cache audit
import_legacy.py  checksummed, restricted source-checkpoint boundary
train.py          single editable model/objective/training surface
inference.py      checked cached-BT4 multi-pass inference and profiling
arena.py          frozen pools, paired outcomes, and arena statistics
arena_history_trust.py sealed O(1) internal history attestations
local_policy.py   strict current-only localized DFM arena adapter
play_arena.py     bounded batched gameplay with exact history replay
evaluate_arena.py strict in-process relative-Elo CLI and resumable blocks
program.md        rules for automated research
results.tsv       compact append-only experiment ledger
```

The arena freezes one physical JAX inference batch shape per run and pads only
already-validated encoded rows. Shrinking game populations therefore reuse the
warmup executable instead of compiling new batch shapes.

All mutable state, including Python and JAX caches, remains below
`/mountpoint/.exp`.

Check the lightweight storage contract before and after an experiment:

```bash
.venv/bin/python research/storage_audit.py
```

Use `--verify-hashes` after moving or restoring an immutable asset. The
default audit avoids rereading multi-gigabyte weights but still checks their
sizes, requires exactly the selected state files, validates the extracted
dataset, rejects redundant source archives, and enforces the JAX cache budget.

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

The paired evaluator has strict source/research-checkpoint loading, frozen
history validation, atomic resumable blocks, bounded model batches, a lean
action-only inference path, and an official-formulation normalized-Elo
promotion GSPRT. The repaired v3 promotion pool is open, and
`AUTORESEARCH_READY = True`. The first accepted K=2 target-sampling checkpoint
passes the offline gates and has an inconclusive-positive 128-pair incumbent
screen; an exact repeat independently passes the offline gates but shows a
larger-than-baseline between-repeat CE gap. It is not promoted. See
`docs/local_relative_arena.md`, the plan, and the baseline report for the exact
contracts and measured results. The subsequent sampled-target anchor
experiment preserved throughput but was rejected at CE `4.5064137187` and a
minimum feature-std p05 of `0.60744`; it retained no state and did not consume
arena games. A subsequent one-active-block projector experiment was also
rejected: it gained `6.70%` cached-profile throughput and tied K=2 policy CE
within `0.00004`, but prediction effective rank contracted to `21.53` mean and
`19.52` minimum. Its implementation remains default-off and no state is
retained. Normalized experiments continue with the RMS norm coefficient at
zero and prediction SIGReg at `1.0`. A coefficient-4 rescue then improved CE
to `4.4995188527` and partially recovered rank to `24.13/22.29`, but still
failed all rank/feature-tail gates and narrowly missed the RMS-ratio floor.
It also retained no state and ends the shallow-projector line.

A three-active-block DFM experiment then improves matched batch-64 eight-pass
inference throughput by `6.28%` and training throughput by `1.49%`, but its
best CE/accuracy/legal mass regress to `6.084679/0.03290/0.36431`. Direct DFM
prefix pruning is rejected, retains no state, and leaves all four planner
blocks active.

A fixed-time cosine warmdown then keeps that full K=2 graph and the
norm/target/prediction loss coefficients `0.0/5.76/1.0` unchanged. Two exact
runs independently pass every offline gate at selected CE
`4.5007607210/4.5022441577`, and both remove the replicated constant-rate
update-1200 regression. The primary update-1261 state scores `50.391%` in the
128-pair arena against K=2, with descriptive logistic Elo `+2.71` and a wide
pair-aware interval `[-82.20,+87.96]`. It is the new offline incumbent, not an
Elo-promoted model; the repeat retains no state.

A one-percent cosine-floor follow-up then selects update 1200 at CE
`4.4993533697`, accuracy `0.1100006104`, and legal mass `0.6472349875`; its
latent audit passes every gate. This improves the 10%-floor incumbent point
estimate by `0.0014073513`, but misses the preregistered repeat-noise-aware CE
ceiling by `0.0000760853`. It is rejected without repeat or arena, retains no
state, and the active schedule returns to the accepted 10% floor.

A K=1 target-sampling follow-up reduces training to two BT4 encodes per
example and raises cached throughput to `116.51` examples/s, `24.88%` over
K=2. Its terminal checkpoint reaches CE `4.4998821113`, accuracy
`0.1096038818`, and legal mass `0.6480329307`, and passes every latent gate.
The CE gain is smaller than accepted repeat variability and misses the frozen
ceiling by `0.0006048269`, so it is rejected without repeat or arena, retains
no state, and K=2 remains active.
