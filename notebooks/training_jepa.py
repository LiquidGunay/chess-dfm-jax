import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
    from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files
    from chess_dfm_jax.paths import default_bt4_paths, project_root
    from chess_dfm_jax.training.jepa import (
        JEPAConfig,
        build_synthetic_transition_batch,
        build_transition_batch,
        create_jepa_components,
        train_step,
    )

    return (
        JEPAConfig,
        LeelaChunkDataLoader,
        build_synthetic_transition_batch,
        build_transition_batch,
        create_jepa_components,
        default_bt4_paths,
        discover_chunk_files,
        load_mapped_bt4_params,
        mo,
        pd,
        project_root,
        train_step,
    )


@app.cell
def _(mo):
    mo.md(
        "# Token-Level JEPA\n\n"
        "Frozen BT4 supplies one token per square. A small action-conditioned transformer predicts the next state's 64 token latents from the current state's 64 token latents plus the recorded action."
    )
    return


@app.cell
def _(default_bt4_paths, project_root):
    MODEL_PATHS = default_bt4_paths()
    CHUNK_DIR = project_root() / "data" / "lc0-training"
    CONFIG = {
        "batch_size": 8,
        "steps": 5,
        "seed": 0,
        "encoder_dtype": "float16",
        "head_param_dtype": "float32",
        "head_compute_dtype": "float32",
        "token_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
        "mlp_dim": 1024,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "action_source": "best",
    }
    return CHUNK_DIR, CONFIG, MODEL_PATHS


@app.cell
def _(CHUNK_DIR, CONFIG, MODEL_PATHS, mo):
    mo.md(
        f"""
## Config

- pb: `{MODEL_PATHS['exported_pb']}`
- chunk dir: `{CHUNK_DIR}`
- batch size: `{CONFIG['batch_size']}`
- steps: `{CONFIG['steps']}`
- encoder dtype: `{CONFIG['encoder_dtype']}`
- head param dtype: `{CONFIG['head_param_dtype']}`
- head compute dtype: `{CONFIG['head_compute_dtype']}`
- token dim: `{CONFIG['token_dim']}`
- layers/heads: `{CONFIG['num_layers']}` / `{CONFIG['num_heads']}`
- mlp dim: `{CONFIG['mlp_dim']}`
- action source: `{CONFIG['action_source']}`
"""
    )
    return


@app.cell
def _(CHUNK_DIR, discover_chunk_files):
    chunk_paths = discover_chunk_files(chunk_dir=str(CHUNK_DIR) if CHUNK_DIR.exists() else None)
    return (chunk_paths,)


@app.cell
def _(CONFIG, LeelaChunkDataLoader, build_synthetic_transition_batch, chunk_paths):
    if chunk_paths:
        loader = LeelaChunkDataLoader(
            chunk_paths,
            batch_size=CONFIG["batch_size"],
            shuffle_files=True,
            decode_mode="raw",
            include_board=True,
            include_moves=True,
            shuffle_buffer=0,
            prefetch_batches=2,
        )
        raw_batch = next(iter(loader))
        source = "chunk"
    else:
        raw_batch = build_synthetic_transition_batch(CONFIG["batch_size"])
        source = "synthetic"
    return raw_batch, source


@app.cell
def _(CONFIG, build_transition_batch, raw_batch):
    transition_batch = build_transition_batch(raw_batch, action_source=CONFIG["action_source"])
    return (transition_batch,)


@app.cell
def _(chunk_paths, pd, source, transition_batch):
    summary = pd.DataFrame(
        [
            {"field": "source", "value": source},
            {"field": "chunk_files_found", "value": len(chunk_paths)},
            {"field": "current_planes_shape", "value": tuple(transition_batch["current_planes"].shape)},
            {"field": "next_planes_shape", "value": tuple(transition_batch["next_planes"].shape)},
            {"field": "action_idx_shape", "value": tuple(transition_batch["action_idx"].shape)},
            {"field": "valid_fraction", "value": float(transition_batch["valid"].mean())},
        ]
    )
    summary
    return


@app.cell
def _(CONFIG, JEPAConfig, create_jepa_components, load_mapped_bt4_params, MODEL_PATHS):
    params = load_mapped_bt4_params(models_dir=MODEL_PATHS["models_dir"])
    config = JEPAConfig(
        token_dim=CONFIG["token_dim"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        mlp_dim=CONFIG["mlp_dim"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        encoder_dtype=CONFIG["encoder_dtype"],
        head_param_dtype=CONFIG["head_param_dtype"],
        head_compute_dtype=CONFIG["head_compute_dtype"],
        action_source=CONFIG["action_source"],
    )
    model, optimizer = create_jepa_components(params, config, seed=CONFIG["seed"])
    return config, model, optimizer


@app.cell
def _(CONFIG, model, optimizer, pd, train_step, transition_batch):
    rows = []
    for step in range(CONFIG["steps"]):
        loss, aux = train_step(model, optimizer, transition_batch)
        row = {"step": step, "loss": float(loss)}
        row.update({key: float(value) for key, value in aux.items()})
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics
    return (metrics,)


@app.cell
def _(mo):
    mo.md(
        "## Batch Path\n\n"
        "For tracked local or TPU runs, use `python scripts/train_jepa.py --run-name <name> --resume --checkpoint-uri <path-or-gs-uri>`. The notebook stays focused on inspectability and short smoke runs."
    )
    return


if __name__ == "__main__":
    app.run()
