# Data Loading

Two data paths coexist in this repo:

- raw LC0 chunk records (`.gz`, `.zst`) for single-step inspection and horizon-1 training
- trajectory shards (`.npz`) for exact multi-step state-action supervision

The canonical training contract is now `trajectory-v2`.

## Trajectory-v2 schema

- `planes_t`: `[B, 112, 8, 8]`
- `actions`: `[B, H]`
- `planes_future`: `[B, H, 112, 8, 8]`
- optional `future_valid`: `[B, H]`
- optional `legal_masks`: `[B, H, 1858]`
- optional `value_targets`: `[B, H]`
- optional `wdl_targets`: `[B, H, 3]`
- metadata fields such as `source`, `game_id`, `ply`, `result`, `fen_t`, `input_format`

## Loader behavior

`chess_dfm_jax.data.leela.LeelaChunkDataLoader`:

- discovers `.gz`, `.zst`, and `.npz` files
- loads trajectory-v2 shards directly
- supports a `dfm_action` view that reads current planes, actions, validity, and
  legal masks without materializing dense `planes_future`
- adapts legacy `planes_target` shards into explicit terminal-only batches
- preserves first-step `legal_mask` for DFM compatibility
- exposes `future_planes`, `future_valid`, and terminal aliases such as `next_planes`
  in the full JEPA/inspection view

The current DFM trainer uses `batch_view="dfm_action"` for train and validation
loaders. This is an immediate stopgap for dense trajectory-v2 shards: it avoids
the largest unused array for DFM-only training, but dense legal masks still need
to be read when legality losses are enabled.

## Compact Latent-SASA v3 Plan

Trajectory-v2 remains the source of truth. `trajectory-v3` is the derived
compact view for throughput:

- bitpacked or `uint8` `planes_t`
- bitpacked or `uint8` `planes_future`
- `uint16` actions
- `uint8` future-valid flags
- `legal_idx_u16` plus `legal_count`, replacing dense float legal masks
- stable row metadata for replay, dedup, and debugging

Training views over the same compact source:

- `dfm_action`: `planes_t`, `actions`, legal indices/counts, validity
- `jepa_latent`: `planes_t`, `actions`, `planes_future`, `future_valid`
- `joint_latent_sasa`: all required action, future-state, legal, and optional
  value/WDL columns

Current implementation status:

- `scripts/convert_trajectory_v2_to_v3.py` converts local or GCS trajectory-v2
  shards to compact trajectory-v3 shards with status and manifest output.
- `LeelaChunkDataLoader` can read trajectory-v3 `.npz` shards through the same
  `full`, `dfm_action`, and `jepa_latent` batch views used by trajectory-v2.
- DFM-only batches use the `dfm_action` view so future boards are not decoded or
  materialized during action-only training.
- JEPA-only batches use the `jepa_latent` view so dense legal masks are not
  decoded or transferred during latent transition training.

Use a custom deterministic loader first. Grain remains a later backend option
once compact v3 throughput and sharding needs are measured. The loader boundary
should still be Grain-compatible: explicit sampler, column reader, host
transform, prefetch queue, and device transfer stages.

## Current Datasets

Trajectory-v2 datasets currently used by the DFM runs:

| dataset | train | val | test | total | status |
| --- | ---: | ---: | ---: | ---: | --- |
| LC0 test80 H8 | 942,080 | 56,320 | 50,176 | 1,048,576 | ready and validated |
| TCEC S20-S28 standard H8 | 2,067,443 | 111,168 | 104,502 | 2,283,113 | ready and validated |
| LC0 test80 H8 10M | pending | pending | pending | about 10M target | write in progress |
| TCEC+LC0 exact dedup H8 | pending | pending | pending | pending | write in progress |
| LC0 test80 H8 sets 1-3 trajectory-v3 | pending | pending | pending | about 30M target | conversion in progress |

The LC0 10M build uses `scripts/process_lc0_parallel_shards.py`, which splits
LC0 archive URLs across workers and assigns non-overlapping chunk index ranges.
Archive downloads use retry/backoff for HTTP 429/5xx responses. The current
build intentionally skips the first 64 LC0 test80 archives because those were
already used for the 1M diagnostic dataset.

LC0 validation checks already run on sampled train/val/test shards:

- schema is `trajectory-v2`
- shapes match `[B, 112, 8, 8]`, `[B, 8]`, and `[B, 8, 112, 8, 8]`
- `future_valid` is full for H8 samples
- future planes are finite and nonzero
- UCI actions replay legally from `fen_t`
- recorded action indices are present in the corresponding legal masks

## GCS Cache Policy

`scripts/train_dfm.py` supports trainer-side GCS caching through
`--gcs-train-prefix` and `--gcs-val-prefix`. Each argument may be either one
immutable shard prefix or a comma-separated list of immutable shard prefixes.
The cache hashes each source directory into the local filename, so shards with
the same basename from different prefixes do not collide.

The default startup policy is:

```bash
--gcs-startup-cache-policy all
```

This blocks step 1 until every currently visible train shard is cached locally.
It is the required policy for normal training, because beginning from a small
cache creates repeated passes over a growing subset and can make train accuracy
look better than it is.

The older behavior is still available only for explicit streaming experiments:

```bash
--gcs-startup-cache-policy minimum
```

Do not use `minimum` for model-selection runs.

For large compact datasets, replace full startup cache waits with deterministic
windowed prefetch:

1. Build the full epoch shard/row schedule before training.
2. Assign each worker a deterministic shard window.
3. Download upcoming windows to persistent SSD in the background.
4. Block if the next scheduled shard is missing.
5. Never silently train on a smaller already-cached subset.

For TPU runs, put `--gcs-cache-dir` on an attached persistent disk instead of
the boot disk, for example:

```bash
--gcs-cache-dir /mnt/chess-dfm-cache/gcs_cache
```

This keeps immutable shard downloads warm across TPU relaunches and across
curriculum stages. GCS remains the source of truth; the persistent disk is only
a reusable local cache.

When `--gcs-startup-cache-policy all` is used, the loader now tries a bulk
`gcloud storage rsync` into a prefix-specific staging directory and then
hard-links shards into the flat cache layout. That avoids the older pattern of
relisting the whole prefix after every small download batch.

## Deduplication

Use `scripts/trajectory_dedup.py` for duplicate statistics and exact
deduplicated shard writing.

Recommended statistics pass:

```bash
python scripts/trajectory_dedup.py stats \
  --input-prefix lc0=gs://bucket/data/trajectory_v2_lc0_test80_h8_1m \
  --key position_actions \
  --output-json dedup_stats.json
```

Recommended safe dedup key:

```text
position_actions = starting position + full H-step action sequence
```

This removes exact duplicate training targets while preserving repeated
positions with different continuations. Position-only dedup should be treated as
an ablation, because repeated positions can have real first-move conflicts.

Use `notebooks/trajectory_dedup_browser.py` to inspect duplicate fractions,
position repeats, and first-move conflict counts.

Legacy `planes_target` shards are marked explicitly:

- `schema_version = trajectory-v1-adapted`
- earlier horizons are unsupervised unless `future_valid[h] == 1`

## Raw chunk caveat

Raw LC0 chunk records only contain one exact move target. They are therefore limited to `horizon=1` exact rollouts. For multi-step training, preprocess into trajectory-v2 `.npz` shards first.

## Main helpers

- `discover_chunk_files()`: gather chunk or shard paths
- `iter_records()`: low-level raw record iterator
- `record_to_board()`: debug reconstruction through `python-chess`
- `load_trajectory_shard()`: schema-aware `.npz` loader
- `rollout_from_fen()`: exact rollout validation helper

## Notebook path

Use `notebooks/state_action_training_browser.py` to inspect trajectory shards, visualize per-horizon future boards, and validate that stored futures match exact rollouts.
