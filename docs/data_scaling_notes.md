# Data Scaling Notes

This note records the current Phase A data-use math, rough loss extrapolation,
data-loading tradeoffs, and a DiffuSearch dataset-size comparison.

## Phase A H8 Data Usage

Current Phase A uses H8 trajectory-v2 shards, but the DFM objective is configured
with:

```text
loss_horizon = 1
train_deterministic_t = 0.0
first_action_loss_weight = 0.0
horizon_legality_loss_weight = 0.0
```

With `loss_horizon=1`, the CE/accuracy loss only supervises `actions[:, 0]`.
The model sees an H8-shaped action tensor, but horizons 2-8 receive zero
CE/accuracy contribution in Phase A.

Important nuance: LC0/TCEC trajectory generation uses overlapping windows, one
start position per ply. So ply `t+1` from one H8 row usually appears as
`actions[:, 0]` in the next row. We are not using only one eighth of actual game
positions, but we are using only one eighth of the stored action slots in each
H8 row.

Approximate current first-ply sample counts:

| dataset | train first-ply examples | stored H8 action slots |
| --- | ---: | ---: |
| LC0 10M set1 | 9.45M | 75.6M |
| LC0 10M set2 | 9.46M | 75.7M |
| set1 + set2 | 18.91M | 151.3M |

## Rough Loss Extrapolation

I pulled W&B history for the active Phase A run and fit the post-switch
validation window from roughly step 50k to 60k. This is a short and noisy window,
so these numbers are diagnostics, not scaling laws.

Current validation around step 60k:

| metric | value |
| --- | ---: |
| `val_first_action_loss` | 5.46 |
| `val_legality_loss` | 0.557 |
| `val_loss` | 9.71 |
| `val_accuracy` | 10.6% |

Rough target estimates from that local trend:

| target | estimated step | extra examples from 60k |
| --- | ---: | ---: |
| validation CE 5.4 | ~80k | ~3.8M |
| validation CE 5.3 | ~100k | ~7.7M |
| validation CE 5.2 | ~120k-125k | ~11.5M-12.5M |
| validation CE 5.0 | ~160k-176k | ~19M-22M |
| validation legal mass 0.5 | ~120k | ~11.5M |
| validation legal mass 0.4 | ~230k-258k | ~33M-38M |

Interpretation:

- CE may keep improving with more fresh data.
- Legality mass is improving more slowly.
- If the set2 epoch ends near step 98,490 and validation CE is not near the low
  5.3s, or legality is still around 0.55, we should change the setup rather than
  only add more epochs.

## Data Loading And Shard Scaling

The current LC0 H8 shards are small:

```text
1024 samples/shard
~1 MB compressed per shard
~326 MB uncompressed per loaded shard
```

The large uncompressed size comes mostly from:

```text
planes_t        [1024,112,8,8]        ~29 MB
planes_future   [1024,8,112,8,8]      ~235 MB
legal_masks     [1024,8,1858]         ~61 MB
```

Current loader behavior matters: `.npz` shards are loaded as full arrays, not as
memory-mapped streaming. Therefore larger shards reduce GCS object overhead, but
increase peak host RAM per open shard.

Recommended shard sizes:

| samples/shard | compressed estimate | uncompressed estimate | recommendation |
| ---: | ---: | ---: | --- |
| 4096 | ~4 MB | ~1.3 GB | Good next default |
| 8192 | ~8 MB | ~2.6 GB | Use after selective loading |
| 16384 | ~16 MB | ~5.2 GB | Too large for current loader |

Initial data-loading change now implemented:

- `scripts/train_dfm.py` uses the `dfm_action` loader view, so DFM-only runs no
  longer materialize `planes_future`.

Remaining high-impact data-loading changes:

- Store legal masks as bool or bitpacked instead of float32.
- Avoid loading all horizon legal masks when only first-ply legality is used.
- Add real sample-level shuffle buffering, because larger shards reduce shuffle
  granularity.
- Use 4096 samples/shard immediately for new full-H8 data; move to 8192 only
  after selective loading is implemented.

## DiffuSearch Dataset Size

DiffuSearch used less chess data than our current LC0 pipeline. The paper reports
default 10k games and a larger 100k-game scaling setting. Table 7 reports:

| DiffuSearch setting | records |
| --- | ---: |
| 10k games | 660k records |
| 100k games | 6.6M records |

Their public Hugging Face chess10k dataset reports:

| split | examples |
| --- | ---: |
| train | 659,576 |
| test | 62,561 |

Sources:

- DiffuSearch paper PDF: <https://openreview.net/pdf/c5a3c7c5ac833a90553d9202ade6d681df36d8a2.pdf>
- DiffuSearch chess10k dataset card: <https://huggingface.co/datasets/jiacheng-ye/chess10k/blob/main/README.md>

Bottom line: our current LC0 setup is already larger in first-ply examples than
DiffuSearch's 100k-game setting. If our loss/legality remains weak, the likely
issue is objective, target/data representation, or model setup rather than raw
sample count alone.
