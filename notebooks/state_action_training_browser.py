import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import json
    import math
    import re
    import shlex
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from chess_dfm_jax.data.trajectory import (
        build_synthetic_trajectory_shard,
        load_trajectory_shard,
        rollout_from_fen,
        trajectory_shard_from_npz,
    )
    from chess_dfm_jax.paths import project_root

    def list_local_shards(local_dir: str) -> list[str]:
        if not local_dir:
            return []
        root = Path(local_dir).expanduser()
        if not root.exists():
            return []
        return [str(path) for path in sorted(root.rglob("*.npz"))[:16]]

    def list_gcs_shards(prefix: str, *, allow_remote: bool) -> list[str]:
        if not prefix or not allow_remote:
            return []
        try:
            import gcsfs
        except Exception:
            return []
        try:
            fs = gcsfs.GCSFileSystem()
            matches = sorted(fs.glob(f"{prefix.rstrip('/')}/**/*.npz"))[:16]
            return [match if match.startswith("gs://") else f"gs://{match}" for match in matches]
        except Exception:
            return []

    def load_shard(uri: str, *, allow_remote: bool):
        if uri == "synthetic":
            return build_synthetic_trajectory_shard(batch_size=2, horizon=4)
        if uri.startswith("gs://"):
            if not allow_remote:
                raise RuntimeError("Remote shard loading is disabled in script mode.")
            import gcsfs

            fs = gcsfs.GCSFileSystem()
            with fs.open(uri, "rb") as handle:
                with np.load(io.BytesIO(handle.read()), allow_pickle=False) as data:
                    return trajectory_shard_from_npz(data)
        return load_trajectory_shard(uri)

    def read_json_uri(uri: str, *, allow_remote: bool):
        if not uri:
            return None, "missing-uri"
        if uri.startswith("gs://"):
            if not allow_remote:
                return None, "remote-skipped-in-script-mode"
            try:
                import gcsfs

                fs = gcsfs.GCSFileSystem()
                with fs.open(uri, "r") as handle:
                    return json.load(handle), None
            except Exception as exc:
                return None, str(exc)
        path = Path(uri).expanduser()
        if not path.exists():
            return None, f"not-found: {path}"
        try:
            return json.loads(path.read_text(encoding="utf-8")), None
        except Exception as exc:
            return None, str(exc)

    def latest_checkpoint_step(checkpoint_uri: str, *, allow_remote: bool):
        if not checkpoint_uri:
            return None, "missing-uri"
        if checkpoint_uri.startswith("gs://"):
            if not allow_remote:
                return None, "remote-skipped-in-script-mode"
            try:
                import gcsfs

                fs = gcsfs.GCSFileSystem()
                files = fs.glob(f"{checkpoint_uri.rstrip('/')}/step*/state.npz")
            except Exception as exc:
                return None, str(exc)
        else:
            root = Path(checkpoint_uri).expanduser()
            files = [str(path) for path in root.glob("step*/state.npz")] if root.exists() else []
        steps = []
        for _file in files:
            match = re.search(r"step(\d+)", _file)
            if match:
                steps.append(int(match.group(1)))
        return (max(steps), None) if steps else (None, "no-checkpoints-found")

    def flag_value(command: str | None, flag: str) -> str | None:
        if not command:
            return None
        try:
            parts = shlex.split(command)
        except ValueError:
            return None
        for idx, part in enumerate(parts):
            if part == flag and idx + 1 < len(parts):
                return parts[idx + 1]
            if part.startswith(f"{flag}="):
                return part.split("=", 1)[1]
        return None

    def query_wandb_run(entity: str | None, project: str | None, run_id: str | None, group: str | None, *, allow_remote: bool):
        if not allow_remote:
            return None, "remote-skipped-in-script-mode"
        if not entity or not project or not run_id:
            return None, "missing-wandb-config"
        try:
            import wandb

            api = wandb.Api(timeout=10)
            filters = {"group": group} if group else {}
            for run in api.runs(f"{entity}/{project}", filters=filters):
                display_name = getattr(run, "display_name", None)
                if run.id == run_id or run.name == run_id or display_name == run_id:
                    summary = dict(run.summary)
                    return {
                        "state": run.state,
                        "url": run.url,
                        "step": summary.get("_step"),
                        "loss": summary.get("loss"),
                        "jepa_loss": summary.get("jepa_loss"),
                        "val_loss": summary.get("val_loss"),
                        "wdl_loss": summary.get("wdl_loss"),
                    }, None
            return None, "run-not-found"
        except Exception as exc:
            return None, str(exc)

    def occupancy_map(planes: np.ndarray) -> np.ndarray:
        own = planes[:6].sum(axis=0)
        opp = planes[6:12].sum(axis=0)
        return own - opp

    return (
        Path,
        build_synthetic_trajectory_shard,
        json,
        math,
        np,
        pd,
        plt,
        project_root,
        rollout_from_fen,
        latest_checkpoint_step,
        list_gcs_shards,
        list_local_shards,
        load_shard,
        occupancy_map,
        query_wandb_run,
        read_json_uri,
        flag_value,
    )


@app.cell
def _(marimo):
    import marimo as mo

    mo.md(
        "# State-Action Training Browser\n\n"
        "Inspect trajectory-v2 shards, validate exact action rollouts, and check the latest TPU/W&B training status."
    )
    return (mo,)


@app.cell
def _(mo, project_root):
    is_script_mode = mo.app_meta().mode == "script"
    local_dir = mo.ui.text(
        value=str(project_root() / "data" / "trajectory"),
        label="Local shard dir",
    )
    gcs_prefix = mo.ui.text(value="", label="GCS shard prefix")
    status_config = mo.ui.text(
        value=str(project_root() / "docs" / "tpu_spot_job_spec.local.json"),
        label="Local status config",
    )
    sample_idx = mo.ui.slider(start=0, stop=31, step=1, value=0, label="Sample index")
    mo.vstack([local_dir, gcs_prefix, status_config, sample_idx])
    return gcs_prefix, is_script_mode, local_dir, sample_idx, status_config


@app.cell
def _(list_gcs_shards, list_local_shards, gcs_prefix, is_script_mode, local_dir, pd):
    local_uris = list_local_shards(local_dir.value)
    gcs_uris = [] if local_uris else list_gcs_shards(gcs_prefix.value, allow_remote=not is_script_mode)
    shard_uris = local_uris or gcs_uris
    discovery_rows = [{"source": "local", "uri": uri} for uri in local_uris] + [
        {"source": "gcs", "uri": uri} for uri in gcs_uris
    ]
    shard_table = (
        pd.DataFrame(discovery_rows)
        if discovery_rows
        else pd.DataFrame([{"source": "synthetic", "uri": "synthetic"}])
    )
    shard_table
    return shard_table, shard_uris


@app.cell
def _(mo, shard_uris):
    options = shard_uris if shard_uris else ["synthetic"]
    shard_uri = mo.ui.dropdown(options=options, value=options[0], label="Shard")
    shard_uri
    return (shard_uri,)


@app.cell
def _(is_script_mode, load_shard, shard_uri):
    shard = load_shard(shard_uri.value, allow_remote=not is_script_mode)
    return (shard,)


@app.cell
def _(np, pd, shard, shard_uri):
    legal_mask_available = shard.legal_masks is not None
    value_available = shard.value_targets is not None
    wdl_available = shard.wdl_targets is not None
    summary = pd.DataFrame(
        [
            {"field": "uri", "value": shard_uri.value},
            {"field": "schema_version", "value": shard.schema_version},
            {"field": "batch_size", "value": shard.batch_size},
            {"field": "horizon", "value": shard.horizon},
            {"field": "planes_t_shape", "value": tuple(np.asarray(shard.planes_t).shape)},
            {"field": "planes_future_shape", "value": tuple(np.asarray(shard.planes_future).shape)},
            {"field": "legal_masks", "value": legal_mask_available},
            {"field": "value_targets", "value": value_available},
            {"field": "wdl_targets", "value": wdl_available},
        ]
    )
    summary
    return


@app.cell
def _(mo, np, pd, sample_idx, shard):
    sample_id = int(sample_idx.value) % shard.batch_size
    actions = np.asarray(shard.actions[sample_id], dtype=np.int32)
    actions_uci = (
        list(np.asarray(shard.actions_uci[sample_id]).tolist())
        if shard.actions_uci is not None
        else ["n/a"] * len(actions)
    )
    source = shard.source[sample_id] if shard.source is not None else "unknown"
    game_id = shard.game_id[sample_id] if shard.game_id is not None else "unknown"
    ply = int(shard.ply[sample_id]) if shard.ply is not None else None
    result = shard.result[sample_id] if shard.result is not None else "unknown"
    sample_summary = pd.DataFrame(
        [
            {"field": "sample_id", "value": sample_id},
            {"field": "source", "value": source},
            {"field": "game_id", "value": game_id},
            {"field": "ply", "value": ply},
            {"field": "result", "value": result},
            {"field": "fen_t", "value": shard.fen_t[sample_id] if shard.fen_t is not None else "unavailable"},
        ]
    )
    action_table = pd.DataFrame(
        {
            "horizon": np.arange(len(actions)),
            "action_idx": actions,
            "action_uci": actions_uci,
            "future_valid": np.asarray(shard.future_valid[sample_id], dtype=np.float32),
        }
    )
    mo.vstack([sample_summary, action_table])
    return action_table, sample_id


@app.cell
def _(math, np, occupancy_map, plt, sample_id, shard):
    states = [np.asarray(shard.planes_t[sample_id], dtype=np.float32)] + [
        np.asarray(planes, dtype=np.float32) for planes in shard.planes_future[sample_id]
    ]
    titles = ["t"] + [f"t+{idx}" for idx in range(1, len(states))]
    cols = min(3, len(states))
    _rows = math.ceil(len(states) / cols)
    _fig, _axes = plt.subplots(_rows, cols, figsize=(4 * cols, 3.5 * _rows))
    _axes_array = np.atleast_1d(_axes).reshape(_rows, cols)
    for idx, (_state, title) in enumerate(zip(states, titles)):
        ax = _axes_array[idx // cols, idx % cols]
        ax.imshow(occupancy_map(_state), cmap="coolwarm", vmin=-6, vmax=6)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(len(states), _rows * cols):
        _axes_array[idx // cols, idx % cols].axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(action_table, np, plt, sample_id, shard):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.5))
    if shard.legal_masks is not None:
        legal_counts = np.asarray(shard.legal_masks[sample_id], dtype=np.float32).sum(axis=1)
        action_legal = [
            float(shard.legal_masks[sample_id, idx, action_idx] > 0.0)
            for idx, action_idx in enumerate(np.asarray(shard.actions[sample_id], dtype=np.int32))
        ]
    else:
        legal_counts = np.zeros((shard.horizon,), dtype=np.float32)
        action_legal = [0.0] * shard.horizon
    _axes[0].bar(action_table["horizon"], legal_counts)
    _axes[0].set_title("Legal move count per horizon")
    _axes[0].set_xlabel("horizon")
    _axes[1].bar(action_table["horizon"], action_legal)
    _axes[1].set_title("Recorded action legal?")
    _axes[1].set_ylim(-0.05, 1.05)
    _axes[1].set_xlabel("horizon")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, pd, rollout_from_fen, sample_id, shard):
    if shard.fen_t is None:
        rollout_validation = pd.DataFrame([{"status": "unavailable", "reason": "fen_t missing"}])
    else:
        rollout = rollout_from_fen(
            str(shard.fen_t[sample_id]),
            np.asarray(shard.actions[sample_id], dtype=np.int32),
            input_format=str(shard.input_format[sample_id]) if shard.input_format is not None else "INPUT_CLASSICAL_112_PLANE",
        )
        _rows = []
        for horizon_idx in range(shard.horizon):
            stored_planes = np.asarray(shard.planes_future[sample_id, horizon_idx], dtype=np.float32)
            rollout_planes = np.asarray(rollout["planes_future"][horizon_idx], dtype=np.float32)
            plane_diff = np.abs(stored_planes - rollout_planes)
            row = {
                "horizon": horizon_idx,
                "action_idx": int(shard.actions[sample_id, horizon_idx]),
                "action_uci": rollout["actions_uci"][horizon_idx],
                "planes_match": bool(np.allclose(stored_planes, rollout_planes)),
                "max_abs_plane_diff": float(plane_diff.max()),
            }
            if shard.legal_masks is not None:
                stored_legal = np.asarray(shard.legal_masks[sample_id, horizon_idx], dtype=np.float32)
                rollout_legal = np.asarray(rollout["legal_masks"][horizon_idx], dtype=np.float32)
                row["legal_mask_match"] = bool(np.allclose(stored_legal, rollout_legal))
                row["recorded_action_is_legal"] = bool(
                    rollout_legal[int(shard.actions[sample_id, horizon_idx])] > 0.0
                )
            _rows.append(row)
        rollout_validation = pd.DataFrame(_rows)
    rollout_validation
    return


@app.cell
def _(np, pd, sample_id, shard):
    horizons = np.arange(shard.horizon)
    value_targets = (
        np.asarray(shard.value_targets[sample_id], dtype=np.float32)
        if shard.value_targets is not None
        else np.zeros((shard.horizon,), dtype=np.float32)
    )
    wdl_targets = (
        np.asarray(shard.wdl_targets[sample_id], dtype=np.float32)
        if shard.wdl_targets is not None
        else np.zeros((shard.horizon, 3), dtype=np.float32)
    )
    targets = pd.DataFrame(
        {
            "horizon": horizons,
            "value_target": value_targets,
            "wdl_win": wdl_targets[:, 0],
            "wdl_draw": wdl_targets[:, 1],
            "wdl_loss": wdl_targets[:, 2],
        }
    )
    targets
    return


@app.cell
def _(Path, flag_value, is_script_mode, json, latest_checkpoint_step, pd, query_wandb_run, read_json_uri, status_config):
    config_path = Path(status_config.value).expanduser()
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}

    status_uri = ""
    checkpoint_uri = ""
    metadata_uri = ""
    run_id = config.get("run_id")
    if config.get("run_root_uri"):
        run_root = config["run_root_uri"].rstrip("/")
    elif config.get("bucket_by_region") and run_id:
        first_region = sorted(config["bucket_by_region"])[0]
        run_family = config.get("run_family", "jepa")
        run_root = f"{config['bucket_by_region'][first_region].rstrip('/')}/runs/{run_family}/{run_id}"
    else:
        run_root = ""

    if run_root:
        status_uri = config.get("status_uri", f"{run_root}/status.json")
        checkpoint_uri = config.get("checkpoint_uri") or flag_value(config.get("entry_command"), "--checkpoint-uri") or f"{run_root}/checkpoints"
        metadata_uri = config.get("checkpoint_metadata_uri", f"{run_root}/checkpoint_state.json")

    status_json, status_error = read_json_uri(status_uri, allow_remote=not is_script_mode)
    metadata_json, metadata_error = read_json_uri(metadata_uri, allow_remote=not is_script_mode)
    latest_step, checkpoint_error = latest_checkpoint_step(
        checkpoint_uri, allow_remote=not is_script_mode
    )
    wandb_json, wandb_error = query_wandb_run(
        config.get("wandb_entity"),
        config.get("wandb_project"),
        config.get("wandb_run_id", run_id),
        config.get("wandb_group"),
        allow_remote=not is_script_mode,
    )

    last_metrics = {}
    if metadata_json is not None:
        last_metrics = metadata_json.get("extra", {}).get("last_metrics", {})

    status_row = {
        "config_path": str(config_path),
        "run_id": run_id or "unconfigured",
        "state": status_json.get("state") if status_json else "offline-fallback",
        "latest_step": latest_step if latest_step is not None else last_metrics.get("step"),
        "loss": last_metrics.get("loss", wandb_json.get("loss") if wandb_json else None),
        "jepa_loss": last_metrics.get("jepa_loss", wandb_json.get("jepa_loss") if wandb_json else None),
        "val_loss": last_metrics.get("val_loss", wandb_json.get("val_loss") if wandb_json else None),
        "wdl_loss": last_metrics.get("wdl_loss", wandb_json.get("wdl_loss") if wandb_json else None),
        "checkpoint_uri": checkpoint_uri or "unconfigured",
        "wandb_url": wandb_json.get("url") if wandb_json else "",
        "status_note": status_error or "",
        "checkpoint_note": checkpoint_error or "",
        "metadata_note": metadata_error or "",
        "wandb_note": wandb_error or "",
    }
    status_df = pd.DataFrame([status_row])
    status_df
    return


if __name__ == "__main__":
    app.run()
