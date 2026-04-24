"""Project path helpers for the standalone scaffold."""

from __future__ import annotations

import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return _PROJECT_ROOT


def candidate_models_dirs(explicit: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    env_dir = os.environ.get("LC0JAXHUMAN_MODELS_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(_PROJECT_ROOT / "models")
    candidates.append(_PROJECT_ROOT.parent / "models")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def resolve_models_dir(explicit: str | Path | None = None) -> Path:
    for candidate in candidate_models_dirs(explicit):
        if candidate.exists():
            return candidate
    tried = "\n".join(f"- {path}" for path in candidate_models_dirs(explicit))
    raise FileNotFoundError(
        "Could not find a models directory. Checked:\n"
        f"{tried}\n"
        "Set LC0JAXHUMAN_MODELS_DIR or place BT4 files in ./models or ../models."
    )


def default_bt4_paths(models_dir: str | Path | None = None) -> dict[str, Path]:
    base = resolve_models_dir(models_dir)
    return {
        "models_dir": base,
        "onnx": base / "BT4.onnx",
        "exported_pb": base / "BT4_exported.pb.gz",
        "policy_tune_pb": base / "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz",
    }
