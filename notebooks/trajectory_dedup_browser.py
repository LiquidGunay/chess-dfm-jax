import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import subprocess
    import textwrap
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    def synthetic_stats():
        return {
            "created_utc": "synthetic",
            "key_mode": "position_actions",
            "total_samples": 3000,
            "total_unique_keys": 2800,
            "total_duplicate_samples": 200,
            "inputs": [
                {"name": "tcec", "uri": "gs://example/tcec"},
                {"name": "lc0", "uri": "gs://example/lc0"},
            ],
            "splits": [
                {
                    "split": "train",
                    "samples": 2400,
                    "unique_keys": 2230,
                    "duplicate_samples": 170,
                    "duplicate_fraction": 170 / 2400,
                    "unique_positions": 1900,
                    "position_duplicate_samples": 500,
                    "position_conflict_count": 24,
                    "shards": 12,
                    "samples_by_dataset": {"tcec": 1500, "lc0": 900},
                    "top_duplicate_keys": [{"hash": "abc123", "count": 8}],
                },
                {
                    "split": "val",
                    "samples": 300,
                    "unique_keys": 285,
                    "duplicate_samples": 15,
                    "duplicate_fraction": 0.05,
                    "unique_positions": 260,
                    "position_duplicate_samples": 40,
                    "position_conflict_count": 2,
                    "shards": 2,
                    "samples_by_dataset": {"tcec": 180, "lc0": 120},
                    "top_duplicate_keys": [],
                },
                {
                    "split": "test",
                    "samples": 300,
                    "unique_keys": 285,
                    "duplicate_samples": 15,
                    "duplicate_fraction": 0.05,
                    "unique_positions": 260,
                    "position_duplicate_samples": 40,
                    "position_conflict_count": 2,
                    "shards": 2,
                    "samples_by_dataset": {"tcec": 180, "lc0": 120},
                    "top_duplicate_keys": [],
                },
            ],
        }

    def read_text_uri(uri: str, *, allow_remote: bool) -> str:
        if uri.startswith("gs://"):
            if not allow_remote:
                return json.dumps(synthetic_stats())
            result = subprocess.run(["gcloud", "storage", "cat", uri], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            return result.stdout
        return Path(uri).expanduser().read_text(encoding="utf-8")

    def load_stats(uri: str, *, allow_remote: bool):
        if not uri:
            return synthetic_stats()
        return json.loads(read_text_uri(uri, allow_remote=allow_remote))

    return json, load_stats, mo, pd, textwrap


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(mo):
    stats_uri = mo.ui.text(
        value="",
        label="Stats JSON URI",
        full_width=True,
    )
    stats_uri
    return (stats_uri,)


@app.cell
def _(is_script_mode, load_stats, stats_uri):
    stats = load_stats(stats_uri.value, allow_remote=not is_script_mode)
    return (stats,)


@app.cell
def _(mo, stats, textwrap):
    summary = mo.md(
        textwrap.dedent(
            f"""
        # Trajectory Deduplication Statistics

        **Created:** `{stats.get("created_utc", "unknown")}`<br>
        **Dedup key:** `{stats.get("key_mode", stats.get("dedup_key", "unknown"))}`<br>
        **Total samples:** `{stats.get("total_samples", stats.get("total_samples_read", 0)):,}`<br>
        **Total exact duplicates:** `{stats.get("total_duplicate_samples", 0):,}`
        """
        )
    )
    summary
    return


@app.cell
def _(pd, stats):
    split_rows = []
    for _split in stats.get("splits", []):
        _samples = int(_split.get("samples", _split.get("samples_read", 0)))
        _duplicates = int(_split.get("duplicate_samples", _split.get("samples_removed", 0)))
        split_rows.append(
            {
                "split": _split.get("split"),
                "samples": _samples,
                "unique_keys": int(_split.get("unique_keys", _split.get("samples_written", 0))),
                "exact_duplicate_samples": _duplicates,
                "exact_duplicate_fraction": _duplicates / _samples if _samples else 0.0,
                "unique_positions": int(_split.get("unique_positions", 0)),
                "position_duplicate_samples": int(_split.get("position_duplicate_samples", 0)),
                "position_conflict_count": int(_split.get("position_conflict_count", 0)),
                "shards": int(_split.get("shards", _split.get("chunks_written", 0))),
            }
        )
    split_df = pd.DataFrame(split_rows)
    split_df
    return (split_df,)


@app.cell
def _(pd, stats):
    dataset_rows = []
    for _split in stats.get("splits", []):
        for _name, _count in _split.get("samples_by_dataset", {}).items():
            dataset_rows.append({"split": _split.get("split"), "dataset": _name, "samples": int(_count)})
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df
    return (dataset_df,)


@app.cell
def _(pd, stats):
    top_rows = []
    for _split in stats.get("splits", []):
        for _row in _split.get("top_duplicate_keys", []):
            top_rows.append({"split": _split.get("split"), **_row})
    top_duplicate_df = pd.DataFrame(top_rows)
    top_duplicate_df
    return (top_duplicate_df,)


@app.cell
def _(mo, textwrap):
    interpretation = mo.md(
        textwrap.dedent(
            """
        ## Interpretation

        `exact_duplicate_samples` uses the selected dedup key, normally `position_actions`.
        `position_duplicate_samples` is stricter: it counts repeated starting positions even if the future action sequence differs.
        `position_conflict_count` is the number of repeated positions where the first target action disagrees; if this is high, position-level dedup would discard meaningful policy diversity.
        """
        )
    )
    interpretation
    return


if __name__ == "__main__":
    app.run()
