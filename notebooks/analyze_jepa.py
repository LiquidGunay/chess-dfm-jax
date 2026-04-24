import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params
    from chess_dfm_jax.data.pgn import PgnTwoPlyDataLoader
    from chess_dfm_jax.paths import default_bt4_paths, project_root
    from chess_dfm_jax.training import JEPAConfig, create_jepa_components, load_training_checkpoint

    return (
        JEPAConfig,
        Path,
        PgnTwoPlyDataLoader,
        create_jepa_components,
        default_bt4_paths,
        json,
        load_mapped_bt4_params,
        load_training_checkpoint,
        np,
        pd,
        plt,
        project_root,
    )


@app.cell
def _(marimo):
    marimo.md(
        "# JEPA Analysis\n\n"
        "Inspect token-level JEPA runs: local training curves, per-square next-state similarity, and a post hoc two-ply probe fit on frozen JEPA latents."
    )
    return


@app.cell
def _(Path, project_root):
    runs_dir = project_root() / "runs" / "jepa"
    run_dirs = sorted([path for path in runs_dir.glob("*") if path.is_dir()]) if runs_dir.exists() else []
    RUN_DIR = run_dirs[-1] if run_dirs else None
    CHECKPOINT_DIR = RUN_DIR / "checkpoints" if RUN_DIR is not None else None
    PGN_PATH = project_root().parent / "data" / "runs" / "2026-02-02_full" / "lc0-training" / "lc0_100k.pgn"
    EVAL_BATCH_SIZE = 64
    return CHECKPOINT_DIR, EVAL_BATCH_SIZE, PGN_PATH, RUN_DIR, run_dirs


@app.cell
def _(CHECKPOINT_DIR, PGN_PATH, RUN_DIR, marimo, run_dirs):
    marimo.md(
        f"""
## Inputs

- run dir: `{RUN_DIR}`
- checkpoint dir: `{CHECKPOINT_DIR}`
- run dirs found: `{len(run_dirs)}`
- pgn path: `{PGN_PATH}`
"""
    )
    return


@app.cell
def _(RUN_DIR, json, pd):
    if RUN_DIR is None:
        metrics = pd.DataFrame()
        config = {}
    else:
        metrics_path = RUN_DIR / "metrics.jsonl"
        config_path = RUN_DIR / "config.json"
        rows = []
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        metrics = pd.DataFrame(rows)
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return config, metrics


@app.cell
def _(metrics, plt):
    if metrics.empty:
        fig = None
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for column in ["loss", "jepa_loss", "mean_token_cosine"]:
            if column in metrics:
                ax.plot(metrics["step"], metrics[column], label=column)
        ax.set_title("Training curves")
        ax.set_xlabel("step")
        ax.legend()
        fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(
    CHECKPOINT_DIR,
    EVAL_BATCH_SIZE,
    JEPAConfig,
    PGN_PATH,
    PgnTwoPlyDataLoader,
    RUN_DIR,
    create_jepa_components,
    default_bt4_paths,
    json,
    load_mapped_bt4_params,
    load_training_checkpoint,
    np,
    pd,
):
    if RUN_DIR is None or CHECKPOINT_DIR is None or not PGN_PATH.exists():
        eval_df = pd.DataFrame()
        token_cosine = np.zeros((0, 64), dtype=np.float32)
        token_probe = np.zeros((0,), dtype=np.float32)
        token_probe_inputs = np.zeros((0, 1), dtype=np.float32)
        token_probe_targets = np.zeros((0, 1), dtype=np.float32)
    else:
        config_json = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
        params = load_mapped_bt4_params(models_dir=default_bt4_paths()["models_dir"])
        jepa_config = JEPAConfig(
            token_dim=int(config_json.get("token_dim", 256)),
            num_layers=int(config_json.get("num_layers", 4)),
            num_heads=int(config_json.get("num_heads", 8)),
            mlp_dim=int(config_json.get("mlp_dim", 1024)),
            learning_rate=float(config_json.get("learning_rate", 3e-4)),
            weight_decay=float(config_json.get("weight_decay", 1e-4)),
            encoder_dtype=str(config_json.get("encoder_dtype", "float16")),
            head_param_dtype=str(config_json.get("head_param_dtype", "float32")),
            head_compute_dtype=str(config_json.get("head_compute_dtype", "float32")),
            action_source=str(config_json.get("action_source", "best")),
        )
        model, optimizer = create_jepa_components(params, jepa_config, seed=int(config_json.get("seed", 0)))
        load_training_checkpoint(CHECKPOINT_DIR, model=model, optimizer=optimizer)
        eval_batch = next(iter(PgnTwoPlyDataLoader([PGN_PATH], batch_size=EVAL_BATCH_SIZE)))

        pred_tokens, next_tokens = model(
            eval_batch["current_planes"],
            eval_batch["action_idx"],
            eval_batch["next_planes"],
        )
        future_tokens = model.encode_state_tokens(eval_batch["future_planes"])
        pred_tokens = np.asarray(pred_tokens, dtype=np.float32)
        next_tokens = np.asarray(next_tokens, dtype=np.float32)
        future_tokens = np.asarray(future_tokens, dtype=np.float32)

        def normalize(x):
            return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)

        pred_norm = normalize(pred_tokens)
        next_norm = normalize(next_tokens)
        future_norm = normalize(future_tokens)
        token_cosine = np.sum(pred_norm * next_norm, axis=-1).astype(np.float32)

        pooled_pred = pred_norm.mean(axis=1)
        pooled_future = future_norm.mean(axis=1)
        split = max(int(0.75 * len(pooled_pred)), 1)
        split = min(split, len(pooled_pred) - 1) if len(pooled_pred) > 1 else 1
        x_train = pooled_pred[:split]
        y_train = pooled_future[:split]
        x_eval = pooled_pred[split:] if len(pooled_pred) > split else pooled_pred
        y_eval = pooled_future[split:] if len(pooled_future) > split else pooled_future

        ridge = 1e-3
        x_mean = x_train.mean(axis=0, keepdims=True)
        y_mean = y_train.mean(axis=0, keepdims=True)
        x_centered = x_train - x_mean
        y_centered = y_train - y_mean
        gram = x_centered.T @ x_centered + ridge * np.eye(x_centered.shape[1], dtype=np.float32)
        weights = np.linalg.solve(gram, x_centered.T @ y_centered)
        probe_pred = (x_eval - x_mean) @ weights + y_mean
        token_probe = np.sum(normalize(probe_pred) * normalize(y_eval), axis=-1).astype(np.float32)
        token_probe_inputs = x_eval
        token_probe_targets = y_eval

        boards = eval_batch["boards"][split:] if len(eval_batch["boards"]) > split else eval_batch["boards"]
        action_uci = (
            eval_batch["action_uci"][split:]
            if len(eval_batch["action_uci"]) > split
            else eval_batch["action_uci"]
        )
        future_action_uci = (
            eval_batch["future_action_uci"][split:]
            if len(eval_batch["future_action_uci"]) > split
            else eval_batch["future_action_uci"]
        )
        token_eval = token_cosine[split:] if len(token_cosine) > split else token_cosine
        eval_df = pd.DataFrame(
            {
                "fen": [board.fen() for board in boards],
                "action_uci": action_uci,
                "future_action_uci": future_action_uci,
                "next_token_cosine_mean": token_eval.mean(axis=1),
                "two_ply_probe_cosine": token_probe,
            }
        )
    return eval_df, token_cosine, token_probe, token_probe_inputs, token_probe_targets


@app.cell
def _(eval_df, plt, token_cosine):
    if eval_df.empty:
        fig = None
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(eval_df["next_token_cosine_mean"], bins=20, alpha=0.8, label="next token cosine")
        axes[0].hist(eval_df["two_ply_probe_cosine"], bins=20, alpha=0.8, label="two-ply probe cosine")
        axes[0].legend()
        axes[0].set_title("Cosine histograms")
        heatmap = token_cosine.mean(axis=0).reshape(8, 8)
        image = axes[1].imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1].set_title("Mean per-square next-token cosine")
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(eval_df, plt, token_cosine):
    if eval_df.empty:
        fig = None
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sample_idx = 0
        sample_heatmap = token_cosine[sample_idx].reshape(8, 8)
        image = axes[0].imshow(sample_heatmap, cmap="magma", vmin=0.0, vmax=1.0)
        axes[0].set_title("Sample 0 per-square cosine")
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        axes[1].scatter(eval_df["next_token_cosine_mean"], eval_df["two_ply_probe_cosine"], alpha=0.7)
        axes[1].set_xlabel("next token cosine mean")
        axes[1].set_ylabel("two-ply probe cosine")
        axes[1].set_title("Next-state quality vs probe quality")
        fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(eval_df):
    if eval_df.empty:
        summary = eval_df
        hardest = eval_df
    else:
        summary = eval_df[["next_token_cosine_mean", "two_ply_probe_cosine"]].describe().T
        hardest = eval_df.sort_values("two_ply_probe_cosine").head(12)
    summary
    hardest
    return hardest, summary


if __name__ == "__main__":
    app.run()
