"""Immutable support utilities for the local-GPU autoresearch path.

The autoresearch agent may import this module but must not edit it. It owns the
workspace boundary, asset verification, fixed data/evaluation metadata, and
machine-readable system information. Model and optimizer experiments belong in
``research/train.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path("/mountpoint/.exp")
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST_PATH = Path(__file__).with_name("assets.json")

PATH_ENV_VARS = (
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "PIP_CACHE_DIR",
    "HF_HOME",
    "JAX_COMPILATION_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "NUMBA_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "MPLCONFIGDIR",
    "PYTHONPYCACHEPREFIX",
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_CONFIG_DIR",
    "WANDB_DATA_DIR",
    "CHESS_DFM_DATA_ROOT",
    "CHESS_DFM_MODEL_ROOT",
    "CHESS_DFM_CHECKPOINT_ROOT",
    "CHESS_DFM_ARTIFACT_ROOT",
)


def require_within_workspace(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path or fail if it resolves outside the workspace."""

    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes {WORKSPACE_ROOT}: {resolved}") from exc
    return resolved


@dataclass(frozen=True)
class WorkspacePaths:
    repo: Path
    local: Path
    data: Path
    models: Path
    checkpoints: Path
    artifacts: Path
    runs: Path

    @classmethod
    def from_repo(cls, repo: Path = REPO_ROOT) -> "WorkspacePaths":
        repo = require_within_workspace(repo)
        return cls(
            repo=repo,
            local=repo / ".local",
            data=repo / "data",
            models=repo / "models",
            checkpoints=repo / "checkpoints",
            artifacts=repo / "artifacts",
            runs=repo / "research" / "runs",
        )

    def create(self) -> None:
        for path in (
            self.local / "tmp",
            self.data / "source",
            self.models / "source",
            self.checkpoints / "source",
            self.checkpoints / "recovered",
            self.artifacts / "baselines",
            self.artifacts / "profiles",
            self.artifacts / "arena",
            self.runs,
        ):
            require_within_workspace(path).mkdir(parents=True, exist_ok=True)


class FixedTrajectoryBatches:
    """Strict, restartable trajectory-v3 batch schedule.

    File order is a deterministic permutation for each epoch and row order
    inside a shard is unchanged. ``batch_at(step)`` is stateless with respect to
    earlier calls, so resuming at a global data step yields the same examples.
    One decoded shard is cached to avoid re-reading it for every small GPU
    batch.
    """

    def __init__(
        self,
        split_dir: Path,
        *,
        batch_size: int,
        horizon: int,
        seed: int,
        shuffle_files: bool,
        view: str = "joint_latent_sasa",
    ):
        self.split_dir = require_within_workspace(split_dir)
        self.paths = sorted(self.split_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No trajectory shards under {self.split_dir}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.shuffle_files = bool(shuffle_files)
        self.view = view

        self.samples_per_shard = self._read_sample_count(self.paths[0])
        if self.batch_size > self.samples_per_shard:
            raise ValueError(
                f"batch_size={self.batch_size} exceeds shard size {self.samples_per_shard}"
            )
        self.batches_per_shard = self.samples_per_shard // self.batch_size
        if self.batches_per_shard < 1:
            raise ValueError("No full batches fit in a shard.")
        self.steps_per_epoch = len(self.paths) * self.batches_per_shard
        self._cached_path: Path | None = None
        self._cached_batch: dict[str, Any] | None = None

    @staticmethod
    def _read_sample_count(path: Path) -> int:
        import numpy as np

        with np.load(require_within_workspace(path), allow_pickle=False) as payload:
            if "batch_size" in payload:
                return int(np.asarray(payload["batch_size"]).item())
            return int(payload["actions_u16"].shape[0])

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(len(self.paths)))
        if self.shuffle_files:
            random.Random(self.seed + int(epoch)).shuffle(order)
        return order

    def _load_shard(self, path: Path) -> dict[str, Any]:
        if self._cached_path == path and self._cached_batch is not None:
            return self._cached_batch

        import numpy as np

        from chess_dfm_jax.data.trajectory_v3 import trajectory_v3_to_batch

        try:
            with np.load(require_within_workspace(path), allow_pickle=False) as payload:
                observed_count = (
                    int(np.asarray(payload["batch_size"]).item())
                    if "batch_size" in payload
                    else int(payload["actions_u16"].shape[0])
                )
                if observed_count != self.samples_per_shard:
                    raise ValueError(
                        f"Shard size changed: expected {self.samples_per_shard}, "
                        f"found {observed_count} in {path}"
                    )
                batch = trajectory_v3_to_batch(
                    payload,
                    view=self.view,
                    horizon=self.horizon,
                    include_metadata=False,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to decode required trajectory shard {path}") from exc

        self._cached_path = path
        self._cached_batch = batch
        return batch

    def batch_at(self, step: int) -> dict[str, Any]:
        import numpy as np

        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        epoch, step_in_epoch = divmod(int(step), self.steps_per_epoch)
        file_position, batch_in_file = divmod(step_in_epoch, self.batches_per_shard)
        shard_index = self._order_for_epoch(epoch)[file_position]
        shard_batch = self._load_shard(self.paths[shard_index])
        start = batch_in_file * self.batch_size
        end = start + self.batch_size
        sliced: dict[str, Any] = {}
        for key, value in shard_batch.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != self.samples_per_shard:
                raise ValueError(
                    f"Unexpected batch leaf {key!r} with shape {array.shape} in "
                    f"{self.paths[shard_index]}"
                )
            sliced[key] = array[start:end]
        return sliced

    def provenance(self) -> dict[str, Any]:
        entries = [
            f"{path.relative_to(self.split_dir)}\t{path.stat().st_size}"
            for path in self.paths
        ]
        digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
        return {
            "split_dir": str(self.split_dir),
            "shard_count": len(self.paths),
            "samples_per_shard": self.samples_per_shard,
            "batch_size": self.batch_size,
            "batches_per_shard": self.batches_per_shard,
            "steps_per_epoch": self.steps_per_epoch,
            "seed": self.seed,
            "shuffle_files": self.shuffle_files,
            "file_manifest_sha256": digest,
        }


def validate_environment(*, require_all: bool = True) -> dict[str, str]:
    """Validate that configured mutable paths remain under the workspace."""

    checked: dict[str, str] = {}
    missing: list[str] = []
    for name in PATH_ENV_VARS:
        raw = os.environ.get(name)
        if not raw:
            missing.append(name)
            continue
        checked[name] = str(require_within_workspace(raw))
    if require_all and missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Workspace environment is incomplete; source research/env.sh. Missing: {joined}")
    return checked


def load_asset_manifest(path: Path = ASSET_MANIFEST_PATH) -> dict[str, Any]:
    path = require_within_workspace(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"Asset manifest must be an object: {path}")
    return manifest


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = require_within_workspace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, *, size_bytes: int | None, sha256: str) -> dict[str, Any]:
    path = require_within_workspace(path)
    stat = path.stat()
    if size_bytes is not None and stat.st_size != size_bytes:
        raise RuntimeError(f"Size mismatch for {path}: expected {size_bytes}, found {stat.st_size}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {sha256}, found {observed_sha256}"
        )
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": observed_sha256,
    }


def inspect_npz(path: Path) -> dict[str, Any]:
    """Return array schema without loading full array payloads eagerly."""

    import numpy as np

    path = require_within_workspace(path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: {
                "shape": list(payload[name].shape),
                "dtype": str(payload[name].dtype),
            }
            for name in payload.files
        }
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrays": arrays,
    }


def _command_output(args: Iterable[str]) -> str | None:
    try:
        result = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output or None


def system_info(*, require_gpu: bool) -> dict[str, Any]:
    validate_environment()
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": str(REPO_ROOT),
        "environment": validate_environment(),
        "nvidia_smi": _command_output(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            )
        ),
    }

    import flax
    import jax
    import jaxlib
    import numpy
    import optax

    devices = [
        {
            "id": device.id,
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
    backend = jax.default_backend()
    if require_gpu and backend != "gpu":
        raise RuntimeError(f"Expected a GPU JAX backend, found {backend!r}: {devices}")

    info["jax"] = {
        "version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": backend,
        "devices": devices,
    }
    info["packages"] = {
        "flax": flax.__version__,
        "numpy": numpy.__version__,
        "optax": optax.__version__,
    }
    return info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_within_workspace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    paths_parser = subparsers.add_parser("paths", help="Validate and print workspace paths.")
    paths_parser.add_argument("--create", action="store_true")

    inspect_parser = subparsers.add_parser("inspect-npz", help="Inspect an NPZ shard schema.")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--output", type=Path)

    verify_parser = subparsers.add_parser("verify-file", help="Verify one immutable asset.")
    verify_parser.add_argument("path", type=Path)
    verify_parser.add_argument("--size-bytes", type=int)
    verify_parser.add_argument("--sha256", required=True)

    system_parser = subparsers.add_parser("system-info", help="Report the active JAX system.")
    system_parser.add_argument("--require-gpu", action="store_true")
    system_parser.add_argument("--output", type=Path)

    subparsers.add_parser("assets", help="Print the immutable Drive asset manifest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "paths":
        paths = WorkspacePaths.from_repo()
        if args.create:
            paths.create()
        _print_json(
            {
                "paths": {key: str(value) for key, value in asdict(paths).items()},
                "environment": validate_environment(),
            }
        )
        return 0

    if args.command == "inspect-npz":
        payload = inspect_npz(args.path)
        if args.output:
            write_json(args.output, payload)
        _print_json(payload)
        return 0

    if args.command == "verify-file":
        _print_json(
            verify_file(
                args.path,
                size_bytes=args.size_bytes,
                sha256=args.sha256,
            )
        )
        return 0

    if args.command == "system-info":
        payload = system_info(require_gpu=args.require_gpu)
        if args.output:
            write_json(args.output, payload)
        _print_json(payload)
        return 0

    if args.command == "assets":
        _print_json(load_asset_manifest())
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
