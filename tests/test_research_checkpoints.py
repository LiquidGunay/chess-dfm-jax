from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from chess_dfm_jax.nnx_bt4 import TrainableParam
from research.prepare import REPO_ROOT
from research.train import (
    JointLatentSASAConfig,
    assert_research_state_compatible,
    checkpoint_evaluation_contract,
    completed_research_checkpoints,
    extract_research_train_state,
    latest_research_checkpoint,
    load_research_checkpoint,
    load_research_checkpoint_for_evaluation,
    research_checkpoints_for_evaluation,
    research_state_abi,
    save_research_checkpoint,
    serialized_model_config,
    strict_restore_research_payload,
    summarize_checkpoint_evaluation_records,
)


class TinyModel(nnx.Module):
    def __init__(self):
        self.w = TrainableParam(
            jnp.asarray(
                [[0.25, -0.5], [0.75, 0.125]],
                dtype=jnp.float32,
            )
        )
        self.b = TrainableParam(jnp.asarray([0.1, -0.2], dtype=jnp.float32))

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return inputs @ self.w[...] + self.b[...]


def make_components() -> tuple[TinyModel, nnx.Optimizer]:
    model = TinyModel()
    optimizer = nnx.Optimizer(model, optax.adamw(1e-2), wrt=TrainableParam)
    return model, optimizer


def train_one(
    model: TinyModel,
    optimizer: nnx.Optimizer,
    *,
    data_cursor: int,
) -> float:
    inputs = jnp.asarray(
        [[1.0 + data_cursor, -0.5], [0.25, 0.75 + data_cursor]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray([[0.4, -0.3], [0.2, 0.8]], dtype=jnp.float32)
    rng = jax.random.fold_in(jax.random.PRNGKey(1234), data_cursor)
    noise = 0.01 * jax.random.normal(rng, inputs.shape)

    def loss_fn(candidate: TinyModel) -> jax.Array:
        return jnp.mean(jnp.square(candidate(inputs + noise) - targets))

    loss, grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
    )(model)
    optimizer.update(model, grads)
    return float(loss)


def assert_payloads_equal(left: dict, right: dict) -> None:
    assert int(np.asarray(left["step"])) == int(np.asarray(right["step"]))
    for key in ("model_trainable", "optimizer_state"):
        left_leaves = jax.tree.leaves(left[key])
        right_leaves = jax.tree.leaves(right[key])
        assert len(left_leaves) == len(right_leaves)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


@pytest.fixture
def workspace_tmp() -> Path:
    root = REPO_ROOT / ".local" / "tmp" / "research-checkpoint-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def test_research_state_abi_ignores_values_but_rejects_schema_changes():
    expected = {
        "empty": {},
        "weight": np.zeros((2, 3), dtype=np.float32),
        0: {"count": np.asarray(0, dtype=np.int32)},
    }
    same_abi = {
        "empty": {},
        "weight": np.ones((2, 3), dtype=np.float32),
        0: {"count": np.asarray(19, dtype=np.int32)},
    }
    assert research_state_abi(expected)["sha256"] == research_state_abi(same_abi)["sha256"]
    assert_research_state_compatible(expected, same_abi, label="test")

    wrong_shape = dict(same_abi)
    wrong_shape["weight"] = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="test checkpoint ABI mismatch.*changed weight"):
        assert_research_state_compatible(expected, wrong_shape, label="test")

    wrong_dtype = dict(same_abi)
    wrong_dtype["weight"] = np.ones((2, 3), dtype=np.float16)
    with pytest.raises(ValueError, match="test checkpoint ABI mismatch.*changed weight"):
        assert_research_state_compatible(expected, wrong_dtype, label="test")

    missing = dict(same_abi)
    del missing["empty"]
    with pytest.raises(ValueError, match="test checkpoint ABI mismatch.*missing empty"):
        assert_research_state_compatible(expected, missing, label="test")


def test_strict_restore_preflights_both_trees_before_mutating():
    source_model, source_optimizer = make_components()
    train_one(source_model, source_optimizer, data_cursor=0)
    payload = extract_research_train_state(source_model, source_optimizer)
    payload["model_trainable"] = dict(payload["model_trainable"])
    del payload["model_trainable"]["b"]

    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)
    with pytest.raises(ValueError, match="model checkpoint ABI mismatch"):
        strict_restore_research_payload(payload, target_model, target_optimizer)
    after = extract_research_train_state(target_model, target_optimizer)
    assert_payloads_equal(before, after)


def test_checkpoint_resume_matches_uninterrupted_training_exactly(workspace_tmp: Path):
    contract = {
        "model": {"width": 2},
        "optimizer": {"kind": "adamw", "learning_rate": 1e-2},
        "data": {"seed": 1234, "batch_size": 2},
    }
    lineage = {"source": "unit-test"}

    reference_model, reference_optimizer = make_components()
    split_model, split_optimizer = make_components()
    assert train_one(reference_model, reference_optimizer, data_cursor=0) == train_one(
        split_model,
        split_optimizer,
        data_cursor=0,
    )
    checkpoint_dir = save_research_checkpoint(
        workspace_tmp / "checkpoints",
        model=split_model,
        optimizer=split_optimizer,
        research_update=1,
        next_data_cursor=1,
        resume_contract=contract,
        lineage=lineage,
    )

    reference_loss = train_one(reference_model, reference_optimizer, data_cursor=1)
    resumed_model, resumed_optimizer = make_components()
    manifest = load_research_checkpoint(
        checkpoint_dir,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_resume_contract=contract,
    )
    assert manifest["research_update"] == 1
    assert manifest["optimizer_step"] == 1
    assert manifest["next_data_cursor"] == 1
    resumed_loss = train_one(
        resumed_model,
        resumed_optimizer,
        data_cursor=int(manifest["next_data_cursor"]),
    )

    assert resumed_loss == reference_loss
    assert_payloads_equal(
        extract_research_train_state(reference_model, reference_optimizer),
        extract_research_train_state(resumed_model, resumed_optimizer),
    )


def test_latest_prune_contract_and_checksum_guards(workspace_tmp: Path):
    checkpoint_root = workspace_tmp / "checkpoints"
    contract = {"experiment": "baseline", "batch_size": 2}
    model, optimizer = make_components()
    first = save_research_checkpoint(
        checkpoint_root,
        model=model,
        optimizer=optimizer,
        research_update=0,
        next_data_cursor=0,
        resume_contract=contract,
        lineage={},
        max_to_keep=1,
    )
    assert first.is_dir()

    (checkpoint_root / ".update00000001.partial-orphan").mkdir()
    incomplete = checkpoint_root / "update00000099"
    incomplete.mkdir()
    np.savez(incomplete / "state.npz", step=np.asarray(99))

    train_one(model, optimizer, data_cursor=0)
    second = save_research_checkpoint(
        checkpoint_root,
        model=model,
        optimizer=optimizer,
        research_update=1,
        next_data_cursor=1,
        resume_contract=contract,
        lineage={},
        max_to_keep=1,
    )
    assert not first.exists()
    assert completed_research_checkpoints(checkpoint_root) == [second]
    assert latest_research_checkpoint(checkpoint_root) == second

    untouched_model, untouched_optimizer = make_components()
    before = extract_research_train_state(untouched_model, untouched_optimizer)
    with pytest.raises(ValueError, match="resume contract mismatch"):
        load_research_checkpoint(
            second,
            model=untouched_model,
            optimizer=untouched_optimizer,
            expected_resume_contract=contract | {"batch_size": 4},
        )
    assert_payloads_equal(
        before,
        extract_research_train_state(untouched_model, untouched_optimizer),
    )

    state_path = second / "state.npz"
    with state_path.open("r+b") as handle:
        handle.seek(-1, 2)
        final_byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final_byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="state checksum mismatch"):
        load_research_checkpoint(
            second,
            model=untouched_model,
            optimizer=untouched_optimizer,
            expected_resume_contract=contract,
        )


def test_checkpoint_evaluation_discovery_and_model_only_restore(
    workspace_tmp: Path,
):
    checkpoint_root = workspace_tmp / "run" / "checkpoints"
    source_model, source_optimizer = make_components()
    train_one(source_model, source_optimizer, data_cursor=0)
    checkpoint = save_research_checkpoint(
        checkpoint_root,
        model=source_model,
        optimizer=source_optimizer,
        research_update=1,
        next_data_cursor=1,
        resume_contract={"test": "checkpoint-evaluation"},
        lineage={},
        max_to_keep=0,
    )
    (checkpoint_root / ".update00000002.partial-test").mkdir()
    assert research_checkpoints_for_evaluation(checkpoint_root.parent) == (
        checkpoint_root,
        [checkpoint],
    )

    target_model, target_optimizer = make_components()
    train_one(target_model, target_optimizer, data_cursor=7)
    optimizer_before = jax.tree.map(
        np.asarray,
        nnx.state(target_optimizer.opt_state),
    )
    manifest = load_research_checkpoint_for_evaluation(
        checkpoint,
        model=target_model,
    )
    expected_model = extract_research_train_state(
        source_model,
        source_optimizer,
    )["model_trainable"]
    assert manifest["checkpoint_dir"] == str(checkpoint)
    for expected, actual in zip(
        jax.tree.leaves(expected_model),
        jax.tree.leaves(nnx.state(target_model, TrainableParam)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    for before, after in zip(
        jax.tree.leaves(optimizer_before),
        jax.tree.leaves(nnx.state(target_optimizer.opt_state)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    malformed = checkpoint_root / "update00000003"
    malformed.mkdir()
    with pytest.raises(ValueError, match="Malformed published checkpoint"):
        research_checkpoints_for_evaluation(checkpoint_root)


def test_checkpoint_evaluation_contract_and_summary(workspace_tmp: Path):
    config = JointLatentSASAConfig()
    contract = {
        "model_config": serialized_model_config(config),
        "objective": {
            "name": "normalized",
            "jepa_target_semantics": config.jepa_target_semantics,
            "jepa_target_stop_gradient": config.jepa_target_stop_gradient,
            "target_sigreg_coeff": config.jepa_sigreg_coeff,
            "pred_sigreg_coeff": config.jepa_pred_sigreg_coeff,
            "target_sigreg_reference_count": 1.0,
            "pred_sigreg_reference_count": 1.0,
        },
    }
    model, optimizer = make_components()
    checkpoints = [
        save_research_checkpoint(
            workspace_tmp / "checkpoints",
            model=model,
            optimizer=optimizer,
            research_update=update,
            next_data_cursor=update,
            resume_contract=contract,
            lineage={},
            max_to_keep=0,
        )
        for update in (1, 2)
    ]
    restored = checkpoint_evaluation_contract(checkpoints)
    assert restored[:4] == (config, "normalized", 1.0, contract)

    records = [
        {
            "research_update": update,
            "optimizer_step": update,
            "next_data_cursor": update,
            "checkpoint_dir": str(checkpoints[update - 1]),
            "state_sha256": f"state-{update}",
            "validation_seed": seed,
            "validation_seconds": 0.25,
            "validation": {"dfm_ce_loss": ce},
        }
        for update, seed, ce in (
            (1, 10_000, 4.6),
            (1, 20_000, 4.4),
            (2, 10_000, 4.3),
            (2, 20_000, 4.5),
        )
    ]
    summary = summarize_checkpoint_evaluation_records(records)
    assert summary["checkpoint_count"] == 2
    assert summary["evaluation_count"] == 4
    assert summary["best_checkpoint"]["research_update"] == 2
    assert summary["best_checkpoint"]["mean"] == pytest.approx(4.4)
    assert summary["by_checkpoint"][1][
        "dfm_ce_loss_delta_from_first"
    ] == pytest.approx(-0.1)
