import hashlib
import json
from pathlib import Path

import pytest

from research.infra import retained_terminal_hero as retained


def test_frozen_terminal_config_matches_original_identity():
    path = Path(retained.TERMINAL_CONFIG_RELATIVE)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == retained.TERMINAL_CONFIG_SHA256
    config = json.loads(path.read_text())
    assert config["data"]["steps_per_epoch"] == 27679
    assert config["config"]["lr_total_examples"] == 28343296


@pytest.mark.parametrize("corrupt", [None, "config", "state_size", "update"])
def test_terminal_view_links_immutable_state_and_rejects_drift(tmp_path, monkeypatch, corrupt):
    checkpoint = tmp_path / "bundle/models/hero"
    checkpoint.mkdir(parents=True)
    state = checkpoint / "model.safetensors"
    state.write_bytes(b"fixture")
    monkeypatch.setattr(retained, "TERMINAL_SIZE_BYTES", 7)
    config = tmp_path / retained.TERMINAL_CONFIG_RELATIVE
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    monkeypatch.setattr(retained, "TERMINAL_CONFIG_SHA256", hashlib.sha256(config.read_bytes()).hexdigest())
    manifest = {"format": "chess-dfm-torch-model-v1", "model_only": True,
                "optimizer_update": 27679, "state": {"path": "model.safetensors",
                "size_bytes": 7, "sha256": retained.TERMINAL_STATE_SHA256}}
    if corrupt == "update":
        manifest["optimizer_update"] = 554
    path = checkpoint / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(retained, "TERMINAL_MANIFEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    if corrupt == "config":
        config.write_text("changed")
    if corrupt == "state_size":
        state.write_bytes(b"short")
    if corrupt:
        with pytest.raises(RuntimeError, match="drift"):
            retained.retained_terminal_view(tmp_path / "bundle", tmp_path, view_parent=tmp_path)
    else:
        result = retained.retained_terminal_view(tmp_path / "bundle", tmp_path, view_parent=tmp_path)
        assert result["optimizer_update"] == 27679
        assert (result["root"] / "checkpoint/model.safetensors").is_symlink()
        assert (result["root"] / "checkpoint/model.safetensors").resolve() == state
        assert (result["root"] / "run_config.json").read_bytes() == config.read_bytes()
