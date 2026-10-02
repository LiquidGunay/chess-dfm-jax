"""Read-only evaluation view of the retained July model, not a Modal training run."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile


TERMINAL_LABEL = "hero-training-archive-terminal-hero-u27679"
TERMINAL_STATE_SHA256 = "665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692"
TERMINAL_MANIFEST_SHA256 = "ac4c00ed6dcee8130655dff665549e1834959cd902522be593e3e3e611062735"
TERMINAL_CONFIG_SHA256 = "5d6138706269ae4a0f87879aa91ee4298df65b70443d1c5ac663989d9b5e3b43"
TERMINAL_CONFIG_RELATIVE = "research/analysis/ev004_terminal_hero_20261002/terminal_run_config.json"
TERMINAL_SIZE_BYTES = 712339616


def retained_terminal_view(bundle_root: Path, source_root: Path, *, view_parent: Path) -> dict:
    """Bind immutable metadata; link model bytes into a temporary inference view.

    The CPU preflight and native arena loader separately hash the entire state.
    Neither the input Volume nor the retained checkpoint is modified.
    """
    checkpoint = bundle_root / "models/hero"
    manifest_path = checkpoint / "manifest.json"
    config_path = source_root / TERMINAL_CONFIG_RELATIVE
    for path, digest in ((manifest_path, TERMINAL_MANIFEST_SHA256),
                         (config_path, TERMINAL_CONFIG_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Retained terminal metadata drift: {path.name}")
    manifest = json.loads(manifest_path.read_text())
    state = manifest.get("state", {})
    state_path = checkpoint / "model.safetensors"
    if (manifest.get("format") != "chess-dfm-torch-model-v1"
            or manifest.get("model_only") is not True
            or manifest.get("optimizer_update") != 27679
            or state.get("sha256") != TERMINAL_STATE_SHA256
            or state.get("path") != "model.safetensors"
            or state.get("size_bytes") != TERMINAL_SIZE_BYTES
            or state_path.stat().st_size != TERMINAL_SIZE_BYTES):
        raise RuntimeError("Retained terminal checkpoint identity drift")
    root = Path(tempfile.mkdtemp(prefix="terminal-hero-view-", dir=view_parent))
    (root / "checkpoint").mkdir()
    for name in ("manifest.json", "model.safetensors"):
        (root / "checkpoint" / name).symlink_to((checkpoint / name).resolve())
    shutil.copyfile(config_path, root / "run_config.json")
    return {"label": TERMINAL_LABEL, "root": root,
            "state_sha256": TERMINAL_STATE_SHA256, "optimizer_update": 27679,
            "arm": "hero", "profile": "retained_non_modal_epoch"}
