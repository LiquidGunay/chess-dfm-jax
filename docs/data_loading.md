# Leela Data Loading

`chess_dfm_jax.data.leela` is the standalone data entry point for LC0 self-play
chunks.

## Main pieces

- `discover_chunk_files()`: gather `.gz` or `.zst` chunk paths.
- `iter_records()`: low-level record iterator.
- `record_to_input_planes()`: fast chunk-to-dense-plane conversion.
- `record_to_board()`: debug reconstruction through `python-chess`.
- `LeelaChunkDataLoader`: batched iterator with optional prefetch.

## Quick inspection

```bash
source ../.venv/bin/activate
python scripts/inspect_leela_chunks.py --chunk-dir /path/to/chunks --batch-size 16
```

## Decode modes

- `raw`: fastest path. Uses the chunk planes directly and reconstructs the aux planes from record metadata.
- `board`: slower path. Rebuilds a board and re-runs the encoder. Use this when you want board objects or easier debugging.

## Important caveat

The chunk format available in this scaffold does not preserve enough metadata to
recover the exact en-passant auxiliary plane in all canonical formats. The raw
path therefore leaves that plane at zero.

That is acceptable for a training scaffold and for most JEPA / DFM experiments,
but it is not a perfect byte-level surrogate for the engine-side encoder.

## Batch structure

`LeelaChunkDataLoader` yields dictionaries like:

- `planes`: `[B, 112, 8, 8]`
- `played_idx`: `[B]`
- `best_idx`: `[B]`
- `side_to_move`: `[B]`
- `rule50`: `[B]`
- `invariance_info`: `[B]`
- `input_format`: Python list of per-sample strings
- optional `boards`, `played_move`, `best_move`

## JEPA / DFM usage

Keep the loader generic. Build model-specific views on top of it:

- JEPA: derive `(current_planes, action_idx, next_planes)` transitions from `planes`, `boards`, and recorded moves
- DFM: derive noised or masked targets from `planes`
- Supervised chess: use `played_idx`, `best_idx`, and later result targets if you add them

The notebook `notebooks/leela_data_pipeline.py` shows one clean place to define
those derived batch views without contaminating the raw loader.
