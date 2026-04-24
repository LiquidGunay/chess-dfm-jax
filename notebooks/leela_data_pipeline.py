import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import chess
    import chess.svg
    import jax
    import jax.numpy as jnp

    from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
    from chess_dfm_jax.data.leela import LeelaChunkDataLoader, discover_chunk_files
    from chess_dfm_jax.nnx_bt4 import jit_bt4_forward, make_bt4_model
    from chess_dfm_jax.paths import default_bt4_paths, project_root
    from chess_dfm_jax.policy import legal_move_mask, policy_index_to_move
    from chess_dfm_jax.training.jepa import build_synthetic_transition_batch, build_transition_batch

    return (
        LeelaChunkDataLoader,
        build_synthetic_transition_batch,
        build_transition_batch,
        chess,
        default_bt4_paths,
        discover_chunk_files,
        jax,
        jit_bt4_forward,
        jnp,
        legal_move_mask,
        load_mapped_bt4_params,
        make_bt4_model,
        mo,
        np,
        pd,
        plt,
        policy_index_to_move,
        project_root,
    )


@app.cell
def _(mo):
    mo.md(
        "# LC0 Data Browser\n\n"
        "Browse LC0 chunk samples with widgets, inspect the encoded planes, and compare the recorded labels against the frozen BT4 policy head."
    )
    return


@app.cell
def _(default_bt4_paths, project_root):
    CHUNK_DIR = project_root().parent / "data" / "runs" / "2026-02-02_full" / "lc0-training" / "training-run2--20251215-1017"
    BATCH_SIZE = 8
    DECODE_MODE = "raw"
    MODEL_PATHS = default_bt4_paths()
    return BATCH_SIZE, CHUNK_DIR, DECODE_MODE, MODEL_PATHS


@app.cell
def _(BATCH_SIZE, CHUNK_DIR, DECODE_MODE, MODEL_PATHS, mo):
    mo.md(
        f"""
## Config

- chunk dir: `{CHUNK_DIR}`
- batch size: `{BATCH_SIZE}`
- decode mode: `{DECODE_MODE}`
- model: `{MODEL_PATHS['policy_tune_pb']}`
"""
    )
    return


@app.cell
def _(CHUNK_DIR, discover_chunk_files):
    chunk_paths = discover_chunk_files(chunk_dir=str(CHUNK_DIR) if CHUNK_DIR.exists() else None)
    return (chunk_paths,)


@app.cell
def _(BATCH_SIZE, DECODE_MODE, LeelaChunkDataLoader, build_synthetic_transition_batch, chunk_paths):
    if chunk_paths:
        loader = LeelaChunkDataLoader(
            chunk_paths,
            batch_size=BATCH_SIZE,
            shuffle_files=True,
            decode_mode=DECODE_MODE,
            include_board=True,
            include_moves=True,
            shuffle_buffer=0,
            prefetch_batches=2,
        )
        batch = next(iter(loader))
        source = "chunk"
    else:
        batch = build_synthetic_transition_batch(BATCH_SIZE)
        source = "synthetic"
    return batch, source


@app.cell
def _(MODEL_PATHS, batch, build_transition_batch, jax, jit_bt4_forward, jnp, load_mapped_bt4_params, make_bt4_model):
    params = load_mapped_bt4_params(models_dir=MODEL_PATHS["models_dir"])
    model = make_bt4_model(params, dtype=jax.numpy.float16)
    transition_batch = build_transition_batch(batch, action_source="best")
    policy_logits, wdl, moves_left = jit_bt4_forward(model, jnp.asarray(batch["planes"], dtype=jax.numpy.float16))
    policy_logits = np.asarray(policy_logits, dtype=np.float32)
    wdl = np.asarray(wdl, dtype=np.float32)
    moves_left = np.asarray(moves_left, dtype=np.float32)
    return model, moves_left, policy_logits, transition_batch, wdl


@app.cell
def _(batch, mo):
    sample_index = mo.ui.slider(0, len(batch["planes"]) - 1, value=0, step=1, label="Sample index")
    plane_index = mo.ui.slider(0, batch["planes"].shape[1] - 1, value=0, step=1, label="Plane index")
    top_k = mo.ui.slider(3, 12, value=6, step=1, label="Top legal moves")
    mo.hstack([sample_index, plane_index, top_k], justify="start")
    return plane_index, sample_index, top_k


@app.cell
def _(batch, chunk_paths, pd, source, transition_batch, wdl, moves_left):
    batch_summary = pd.DataFrame(
        [
            {"field": "source", "value": source},
            {"field": "chunk_files_found", "value": len(chunk_paths)},
            {"field": "planes_shape", "value": tuple(batch["planes"].shape)},
            {"field": "valid_fraction", "value": float(transition_batch["valid"].mean())},
            {"field": "wdl_shape", "value": tuple(wdl.shape)},
            {"field": "moves_left_shape", "value": tuple(moves_left.shape)},
        ]
    )
    batch_summary
    return


@app.cell
def _(batch, chess, mo, sample_index, transition_batch):
    idx = sample_index.value
    board = batch.get("boards", [None] * len(batch["planes"]))[idx]
    current_svg = chess.svg.board(board=board, size=360) if board is not None else "<p>Board unavailable</p>"
    next_board = None
    if board is not None and transition_batch["valid"][idx] > 0:
        next_board = board.copy(stack=False)
        move = batch.get("best_move", [None] * len(batch["planes"]))[idx]
        if move is not None and move in next_board.legal_moves:
            next_board.push(move)
    next_svg = chess.svg.board(board=next_board, size=360) if next_board is not None else "<p>Next board unavailable</p>"
    mo.hstack(
        [
            mo.vstack([mo.md("### Current board"), mo.Html(current_svg)]),
            mo.vstack([mo.md("### Next board"), mo.Html(next_svg)]),
        ],
        justify="start",
    )
    return


@app.cell
def _(batch, pd, sample_index, transition_batch, wdl, moves_left):
    idx = sample_index.value
    played_move = batch.get("played_move", [None] * len(batch["planes"]))[idx]
    best_move = batch.get("best_move", [None] * len(batch["planes"]))[idx]
    metadata = pd.DataFrame(
        [
            {"field": "sample_index", "value": idx},
            {"field": "input_format", "value": batch["input_format"][idx]},
            {"field": "played_idx", "value": int(batch["played_idx"][idx])},
            {"field": "best_idx", "value": int(batch["best_idx"][idx])},
            {"field": "played_move", "value": None if played_move is None else played_move.uci()},
            {"field": "best_move", "value": None if best_move is None else best_move.uci()},
            {"field": "wdl", "value": tuple(float(x) for x in wdl[idx])},
            {"field": "moves_left", "value": float(moves_left[idx, 0])},
            {"field": "transition_valid", "value": float(transition_batch["valid"][idx])},
            {"field": "plane_nonzero_count", "value": int((batch["planes"][idx] != 0).sum())},
        ]
    )
    metadata
    return


@app.cell
def _(batch, np, plane_index, plt, sample_index):
    idx = sample_index.value
    plane = plane_index.value
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    image = ax.imshow(batch["planes"][idx, plane], cmap="viridis", vmin=0.0, vmax=max(float(np.max(batch["planes"][idx, plane])), 1.0))
    ax.set_title(f"Plane {plane}")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig
    return


@app.cell
def _(batch, legal_move_mask, pd, policy_index_to_move, policy_logits, sample_index, top_k):
    idx = sample_index.value
    board = batch.get("boards", [None] * len(batch["planes"]))[idx]
    if board is None:
        top_moves = pd.DataFrame()
    else:
        mask = legal_move_mask(board, "lc0_1858")
        masked = np.where(mask, policy_logits[idx], np.finfo(np.float32).min)
        top_idx = np.argsort(masked)[::-1][: top_k.value]
        played_move = batch.get("played_move", [None] * len(batch["planes"]))[idx]
        best_move = batch.get("best_move", [None] * len(batch["planes"]))[idx]
        legal_logits = masked[mask]
        legal_probs = np.exp(legal_logits - np.max(legal_logits))
        legal_probs = legal_probs / np.maximum(legal_probs.sum(), 1e-8)
        prob_lookup = {legal_index: float(prob) for legal_index, prob in zip(np.where(mask)[0], legal_probs)}
        rows = []
        for rank, move_idx in enumerate(top_idx, start=1):
            move = policy_index_to_move(int(move_idx), "lc0_1858")
            san = board.san(move) if move in board.legal_moves else "illegal"
            rows.append(
                {
                    "rank": rank,
                    "policy_idx": int(move_idx),
                    "uci": move.uci(),
                    "san": san,
                    "legal_prob": prob_lookup.get(int(move_idx), 0.0),
                    "raw_logit": float(policy_logits[idx, move_idx]),
                    "is_best": best_move is not None and move == best_move,
                    "is_played": played_move is not None and move == played_move,
                }
            )
        top_moves = pd.DataFrame(rows)
    top_moves
    return


@app.cell
def _(jax, mo):
    mo.md(f"## Runtime\n\nBackend: `{jax.default_backend()}` | Device: `{jax.devices()[0]}`")
    return


if __name__ == "__main__":
    app.run()
