from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.training import checkpoints  # noqa: E402


class DummyModel:
    pass


class DummyOptimizer:
    pass


def test_raw_checkpoint_roundtrip_uses_existing_npz_layout(tmp_path, monkeypatch):
    restored = {}
    payload = {
        "step": np.asarray(7, dtype=np.int64),
        "model_trainable": {"w": np.asarray([1.0, 2.0], dtype=np.float32)},
        "optimizer_state": {"count": np.asarray(1, dtype=np.int32)},
    }

    monkeypatch.setattr(checkpoints, "extract_train_state", lambda _model, _optimizer: payload)

    def fake_restore(incoming, model, optimizer, *, strict=True):
        restored["payload"] = incoming
        restored["strict"] = strict
        return int(incoming["step"])

    monkeypatch.setattr(checkpoints, "restore_train_state", fake_restore)

    manager = checkpoints.create_checkpoint_manager(tmp_path, max_to_keep=2)
    assert checkpoints.save_training_checkpoint(
        manager,
        model=DummyModel(),
        optimizer=DummyOptimizer(),
        step=7,
    )

    assert (tmp_path / "step0000007" / "state.npz").exists()
    assert checkpoints.latest_checkpoint_step(tmp_path) == 7

    loaded = checkpoints.load_training_checkpoint(
        tmp_path,
        model=DummyModel(),
        optimizer=DummyOptimizer(),
        strict=False,
    )
    assert int(loaded["step"]) == 7
    assert restored["strict"] is False
    np.testing.assert_allclose(restored["payload"]["model_trainable"]["w"], payload["model_trainable"]["w"])


def test_raw_plus_orbax_keeps_npz_compatibility(tmp_path, monkeypatch):
    payload = {
        "step": np.asarray(3, dtype=np.int64),
        "model_trainable": {"w": np.asarray([3.0], dtype=np.float32)},
        "optimizer_state": {},
    }
    monkeypatch.setattr(checkpoints, "extract_train_state", lambda _model, _optimizer: payload)

    manager = checkpoints.create_checkpoint_manager(
        tmp_path,
        max_to_keep=1,
        checkpoint_format="raw+orbax",
        async_orbax=True,
    )
    checkpoints.save_training_checkpoint(manager, model=DummyModel(), optimizer=DummyOptimizer(), step=3)
    checkpoints.wait_for_checkpoint_completion(manager)

    assert (tmp_path / "step0000003" / "state.npz").exists()
    assert checkpoints.latest_checkpoint_step(tmp_path) == 3
    if getattr(manager, "checkpoint_format", "raw") == "raw+orbax":
        assert (tmp_path / "orbax" / "step0000003").exists()
