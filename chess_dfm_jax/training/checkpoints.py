"""Training checkpoints with raw NumPy compatibility and optional Orbax mirrors."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

from chess_dfm_jax.training.jepa import extract_train_state, restore_train_state


class CheckpointUnavailableError(RuntimeError):
    """Raised when an explicitly requested checkpoint backend is unavailable."""


def _import_orbax_checkpoint():
    try:
        import orbax.checkpoint as ocp  # type: ignore[import-not-found]

        return ocp
    except ImportError as exc:
        raise CheckpointUnavailableError(
            "Orbax checkpointing was requested, but `orbax-checkpoint` is not installed."
        ) from exc


def create_checkpoint_manager(
    directory: str | Path,
    *,
    save_interval_steps: int = 100,
    max_to_keep: int = 3,
    checkpoint_format: str = "raw",
    async_orbax: bool = True,
) -> Any:
    class RawManager:
        def __init__(self, directory, max_to_keep, checkpoint_format, async_orbax):
            self.directory = str(directory)
            self.max_to_keep = int(max_to_keep)
            self.save_interval_steps = int(save_interval_steps)
            self.checkpoint_format = checkpoint_format
            self.async_orbax = bool(async_orbax)
            self.orbax_dir = Path(directory) / "orbax"
            self._orbax_checkpointer = None
            self._orbax_error = None
            if checkpoint_format not in {"raw", "raw+orbax", "orbax"}:
                raise ValueError(
                    "checkpoint_format must be one of: raw, raw+orbax, orbax."
                )
            if checkpoint_format in {"raw+orbax", "orbax"}:
                try:
                    ocp = _import_orbax_checkpoint()
                    handler = ocp.PyTreeCheckpointHandler()
                    if async_orbax and hasattr(ocp, "AsyncCheckpointer"):
                        self._orbax_checkpointer = ocp.AsyncCheckpointer(handler)
                    else:
                        self._orbax_checkpointer = ocp.Checkpointer(handler)
                except CheckpointUnavailableError as exc:
                    if checkpoint_format == "orbax":
                        raise
                    self.checkpoint_format = "raw"
                    self._orbax_error = exc

        def latest_step(self):
            return latest_checkpoint_step(self.directory)

        def wait_until_finished(self):
            checkpointer = self._orbax_checkpointer
            if checkpointer is not None and hasattr(checkpointer, "wait_until_finished"):
                checkpointer.wait_until_finished()

        def close(self):
            self.wait_until_finished()

    return RawManager(directory, max_to_keep, checkpoint_format, async_orbax)


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
    checkpoint_format = getattr(manager, "checkpoint_format", "raw")
    if checkpoint_format != "orbax":
        save_dir = base / f"step{int(step):07d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez(save_dir / "state.npz", **payload)

    if checkpoint_format in {"raw+orbax", "orbax"}:
        _save_orbax_checkpoint(manager, payload, step=int(step), force=force)

    _prune_old_checkpoints(base, max_to_keep=int(getattr(manager, "max_to_keep", 3)))
    if checkpoint_format in {"raw+orbax", "orbax"}:
        _prune_old_checkpoints(
            Path(getattr(manager, "orbax_dir", base / "orbax")),
            max_to_keep=int(getattr(manager, "max_to_keep", 3)),
        )

    if metadata_path is not None:
        write_checkpoint_metadata(metadata_path, step=step, config=config, extra=extra)
    return True


def load_training_checkpoint(
    directory: str | Path,
    *,
    model,
    optimizer=None,
    step: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    base = Path(directory)
    target_step = step
    if target_step is None:
        target_step = latest_checkpoint_step(directory)
        if target_step is None:
            raise FileNotFoundError(f"No checkpoint found under {directory}.")

    load_path = base / f"step{int(target_step):07d}" / "state.npz"
    if load_path.exists():
        payload = _load_raw_npz_payload(load_path)
    else:
        payload = _load_orbax_payload(base, int(target_step))
             
    restore_train_state(payload, model, optimizer, strict=strict)
    return payload


def _load_raw_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "arr_0" in data.files and len(data.files) == 1:
            return data["arr_0"].item()
        payload = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
                payload[key] = value.item()
            else:
                payload[key] = value
        return payload


def _orbax_step_dir(directory: str | Path, step: int) -> Path:
    return Path(directory) / "orbax" / f"step{int(step):07d}"


def _load_orbax_payload(directory: str | Path, step: int) -> dict[str, Any]:
    path = _orbax_step_dir(directory, step)
    if not path.exists():
        raise FileNotFoundError(f"No raw or Orbax checkpoint found for step {step} under {directory}.")
    ocp = _import_orbax_checkpoint()
    return ocp.PyTreeCheckpointer().restore(path)


def _save_orbax_checkpoint(manager: Any, payload: dict[str, Any], *, step: int, force: bool) -> None:
    checkpointer = getattr(manager, "_orbax_checkpointer", None)
    if checkpointer is None:
        if getattr(manager, "checkpoint_format", "raw") == "orbax":
            _import_orbax_checkpoint()
        return
    orbax_dir = Path(getattr(manager, "orbax_dir", Path(manager.directory) / "orbax"))
    orbax_dir.mkdir(parents=True, exist_ok=True)
    step_dir = orbax_dir / f"step{int(step):07d}"
    if step_dir.exists():
        if not force:
            return
        if hasattr(checkpointer, "wait_until_finished"):
            checkpointer.wait_until_finished()
        shutil.rmtree(step_dir, ignore_errors=True)
    checkpointer.save(step_dir, payload)


def _parse_step_dir(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("step"):
        return None
    try:
        return int(path.name[4:])
    except ValueError:
        return None


def _checkpoint_steps_in_directory(directory: str | Path, *, require_state_npz: bool) -> list[int]:
    base = Path(directory)
    steps: list[int] = []
    try:
        for path in base.glob("step*"):
            step = _parse_step_dir(path)
            if step is None:
                continue
            if require_state_npz and not (path / "state.npz").exists():
                continue
            steps.append(step)
    except OSError:
        pass
    return steps


def _local_checkpoint_steps(directory: str | Path) -> list[int]:
    base = Path(directory)
    steps: list[int] = []
    steps.extend(_checkpoint_steps_in_directory(base, require_state_npz=True))
    steps.extend(_checkpoint_steps_in_directory(base / "orbax", require_state_npz=False))
    return sorted(set(steps))


def _prune_old_checkpoints(directory: Path, *, max_to_keep: int) -> None:
    if max_to_keep <= 0:
        return
    steps = sorted(set(_checkpoint_steps_in_directory(directory, require_state_npz=False)))
    for step in steps[:-max_to_keep]:
        shutil.rmtree(directory / f"step{int(step):07d}", ignore_errors=True)


def latest_checkpoint_step(directory: str | Path) -> int | None:
    uri = str(directory)
    if not uri.startswith("gs://"):
        steps = _local_checkpoint_steps(directory)
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


def wait_for_checkpoint_completion(manager: Any | None = None) -> None:
    if manager is not None and hasattr(manager, "wait_until_finished"):
        manager.wait_until_finished()


__all__ = [
    "checkpoint_paths",
    "CheckpointUnavailableError",
    "create_checkpoint_manager",
    "latest_checkpoint_step",
    "load_training_checkpoint",
    "read_checkpoint_metadata",
    "save_training_checkpoint",
    "write_checkpoint_metadata",
    "wait_for_checkpoint_completion",
]
