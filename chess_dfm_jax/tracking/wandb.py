"""Small W&B helpers for notebooks and scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chess_dfm_jax.paths import project_root


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def init_wandb_run(
    *,
    project: str,
    config: dict[str, Any],
    entity: str | None = None,
    name: str | None = None,
    run_id: str | None = None,
    resume: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    job_type: str = "train",
    mode: str | None = None,
):
    load_env_file()
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wandb is not installed in the active environment.") from exc

    settings = wandb.Settings(start_method="thread")
    return wandb.init(
        entity=entity,
        project=project,
        name=name,
        id=run_id,
        resume=resume,
        group=group,
        config=config,
        tags=tags,
        job_type=job_type,
        settings=settings,
        mode=mode,
    )
