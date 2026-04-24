"""Bulletproof raw NumPy checkpoints for JEPA experiments (no Orbax, no barriers)."""

from __future__ import annotations

import json
import os
import shutil
import time
import subprocess
import re
from pathlib import Path
from typing import Any

import jax
import numpy as np

from chess_dfm_jax.training.jepa import extract_train_state, restore_train_state

def create_checkpoint_manager(
    directory: str | Path,
    *,
    save_interval_steps: int = 100,
    max_to_keep: int = 3,
) -> Any:
    class RawManager:
        def __init__(self, directory, max_to_keep):
            self.directory = str(directory)
            self.max_to_keep = max_to_keep
        def latest_step(self):
            return latest_checkpoint_step(self.directory)
        def close(self):
            pass
    return RawManager(directory, max_to_keep)


def checkpoint_paths(run_dir: str | Path) -> dict[str, Path]:
    base = Path(run_dir)
    return {
        "run_dir": base,
        "checkpoint_dir": base / "checkpoints",
        "metadata_path": base / "checkpoint_state.json",
    }


def write_checkpoint_metadata(
    path: str | Path,
    *,
    step: int,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    if not str(out_path).startswith("gs://"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
    payload = {"latest_step": int(step)}
    if config is not None:
        payload["config"] = config
    if extra is not None:
        payload["extra"] = extra
        
    if str(out_path).startswith("gs://"):
        from etils import epath
        epath.Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def read_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        if str(p).startswith("gs://"):
            from etils import epath
            gp = epath.Path(p)
            if not gp.exists(): return {}
            return json.loads(gp.read_text(encoding="utf-8"))
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except:
        return {}


def save_training_checkpoint(
    manager: Any,
    *,
    model,
    optimizer,
    step: int,
    metrics: dict[str, Any] | None = None,
    metadata_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    force: bool = False,
) -> bool:
    if jax.process_index() != 0:
        return True

    payload = extract_train_state(model, optimizer)
    base = Path(manager.directory)
    save_dir = base / f"step{int(step):07d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(save_dir / "state.npz", **payload)
    
    try:
        steps = []
        for p in base.glob("step*"):
            if p.is_dir():
                try:
                    steps.append(int(p.name[4:]))
                except:
                    pass
        steps.sort()
        for s in steps[:-manager.max_to_keep]:
            shutil.rmtree(base / f"step{int(s):07d}", ignore_errors=True)
    except:
        pass

    if metadata_path is not None:
        write_checkpoint_metadata(metadata_path, step=step, config=config, extra=extra)
    return True


def load_training_checkpoint(
    directory: str | Path,
    *,
    model,
    optimizer=None,
    step: int | None = None,
) -> dict[str, Any]:
    base = Path(directory)
    target_step = step
    if target_step is None:
        target_step = latest_checkpoint_step(directory)
        if target_step is None:
            raise FileNotFoundError(f"No checkpoint found under {directory}.")

    load_path = base / f"step{int(target_step):07d}" / "state.npz"
    with np.load(load_path, allow_pickle=True) as data:
        if "arr_0" in data.files and len(data.files) == 1:
             payload = data["arr_0"].item()
        else:
             payload = {}
             for k in data.files:
                 val = data[k]
                 if isinstance(val, np.ndarray) and val.shape == () and val.dtype == object:
                     payload[k] = val.item()
                 else:
                     payload[k] = val
             
    restore_train_state(payload, model, optimizer)
    return payload


def latest_checkpoint_step(directory: str | Path) -> int | None:
    uri = str(directory)
    if not uri.startswith("gs://"):
        base = Path(directory)
        steps = []
        try:
            for p in base.glob("step*"):
                if p.is_dir():
                    try:
                        steps.append(int(p.name[4:]))
                    except:
                        pass
        except:
            pass
        return max(steps) if steps else None

    try:
        cmd = ["/snap/google-cloud-cli/current/bin/gcloud", "storage", "ls", f"{uri.rstrip('/')}/**/state.npz"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
            
        steps = []
        for line in result.stdout.splitlines():
            match = re.search(r'step(\d+)', line)
            if match:
                try:
                    steps.append(int(match.group(1)))
                except:
                    pass
        return max(steps) if steps else None
    except:
        return None


def wait_for_checkpoint_completion():
    pass


__all__ = [
    "checkpoint_paths",
    "create_checkpoint_manager",
    "latest_checkpoint_step",
    "load_training_checkpoint",
    "read_checkpoint_metadata",
    "save_training_checkpoint",
    "write_checkpoint_metadata",
    "wait_for_checkpoint_completion",
]
