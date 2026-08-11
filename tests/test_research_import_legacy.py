from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest
from flax import nnx

import research.import_legacy as legacy
from chess_dfm_jax.nnx_bt4 import TrainableParam
from research.prepare import REPO_ROOT
from research.train import extract_research_train_state, research_state_abi


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


def advance_once(model: TinyModel, optimizer: nnx.Optimizer) -> None:
    inputs = jnp.asarray([[1.0, -0.5], [0.25, 0.75]], dtype=jnp.float32)
    targets = jnp.asarray([[0.4, -0.3], [0.2, 0.8]], dtype=jnp.float32)

    def loss_fn(candidate: TinyModel) -> jax.Array:
        return jnp.mean(jnp.square(candidate(inputs) - targets))

    grads = nnx.grad(
        loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
    )(model)
    optimizer.update(model, grads)


def source_payload() -> dict[str, Any]:
    model, optimizer = make_components()
    advance_once(model, optimizer)
    return extract_research_train_state(model, optimizer)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_legacy(path: Path, payload: dict[str, Any]) -> tuple[int, str]:
    np.savez(path, **payload)
    return path.stat().st_size, sha256(path)


def assert_tree_equal(left: Any, right: Any) -> None:
    assert research_state_abi(left)["sha256"] == research_state_abi(right)["sha256"]
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def assert_train_state_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert int(np.asarray(left["step"])) == int(np.asarray(right["step"]))
    assert_tree_equal(left["model_trainable"], right["model_trainable"])
    assert_tree_equal(left["optimizer_state"], right["optimizer_state"])


def replace_first_leaf(
    value: Any,
    replacement: Callable[[Any], Any],
) -> tuple[Any, bool]:
    if type(value) is dict:
        result = dict(value)
        for key, child in value.items():
            replaced, changed = replace_first_leaf(child, replacement)
            if changed:
                result[key] = replaced
                return result, True
        return result, False
    if type(value) is list:
        result = list(value)
        for index, child in enumerate(value):
            replaced, changed = replace_first_leaf(child, replacement)
            if changed:
                result[index] = replaced
                return result, True
        return result, False
    if type(value) is tuple:
        result = list(value)
        for index, child in enumerate(value):
            replaced, changed = replace_first_leaf(child, replacement)
            if changed:
                result[index] = replaced
                return tuple(result), True
        return value, False
    return replacement(value), True


def mutate_tree(tree: Any, mutation: str) -> Any:
    changed = copy.deepcopy(tree)
    if mutation == "missing":
        first_key = next(iter(changed))
        del changed[first_key]
        return changed
    if mutation == "extra":
        changed["__unexpected__"] = np.asarray(1.0, dtype=np.float32)
        return changed

    def wrong_shape(value: Any) -> np.ndarray:
        array = np.asarray(value)
        shape = array.shape + (1,) if array.shape else (1,)
        return np.zeros(shape, dtype=array.dtype)

    def wrong_dtype(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype == np.dtype(np.float32):
            dtype = np.float16
        elif np.issubdtype(array.dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(array.dtype, np.integer):
            dtype = np.float32
        else:
            dtype = np.int8
        return array.astype(dtype)

    replacement = {"wrong-shape": wrong_shape, "wrong-dtype": wrong_dtype}[mutation]
    changed, replaced = replace_first_leaf(changed, replacement)
    assert replaced
    return changed


@pytest.fixture
def workspace_tmp() -> Path:
    root = REPO_ROOT / ".local" / "tmp" / "legacy-import-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def import_payload(
    path: Path,
    payload: dict[str, Any],
    *,
    model: TinyModel,
    optimizer: nnx.Optimizer,
    init_mode: legacy.LegacyInitMode = "exact",
    expected_step: int = 1,
) -> legacy.LegacyImportResult:
    size, digest = write_legacy(path, payload)
    return legacy.import_legacy_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_size_bytes=size,
        expected_sha256=digest,
        expected_step=expected_step,
        init_mode=init_mode,
    )


def test_exact_import_uses_allow_pickle_false_and_restores_both_trees(
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = source_payload()
    path = workspace_tmp / "state.npz"
    size, digest = write_legacy(path, payload)
    real_load = legacy.np.load
    allow_pickle_values: list[object] = []

    def load_spy(*args: Any, **kwargs: Any) -> Any:
        allow_pickle_values.append(kwargs.get("allow_pickle"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(legacy.np, "load", load_spy)
    target_model, target_optimizer = make_components()
    result = legacy.import_legacy_checkpoint(
        path,
        model=target_model,
        optimizer=target_optimizer,
        expected_size_bytes=size,
        expected_sha256=digest,
        expected_step=1,
        init_mode="exact",
    )

    assert allow_pickle_values == [False]
    assert result.source_path == path.resolve()
    assert result.source_step == 1
    assert result.optimizer_restored is True
    assert result.model_abi_sha256 == research_state_abi(payload["model_trainable"])["sha256"]
    assert (
        result.optimizer_abi_sha256
        == research_state_abi(payload["optimizer_state"])["sha256"]
    )
    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        payload,
    )


def test_verified_loader_accepts_pinned_source_bfloat16_dtype(workspace_tmp: Path):
    payload = source_payload()
    payload["model_trainable"]["b"] = np.asarray(
        payload["model_trainable"]["b"],
        dtype=ml_dtypes.bfloat16,
    )
    path = workspace_tmp / "bfloat16-state.npz"
    size, digest = write_legacy(path, payload)

    loaded = legacy.load_verified_legacy_payload(
        path,
        expected_size_bytes=size,
        expected_sha256=digest,
        expected_step=1,
    )

    assert np.asarray(loaded["model_trainable"]["b"]).dtype == np.dtype(
        ml_dtypes.bfloat16
    )


def test_verified_loader_accepts_optax_masked_node_sentinel(workspace_tmp: Path):
    payload = source_payload()
    payload["optimizer_state"]["masked"] = optax.MaskedNode()
    path = workspace_tmp / "masked-state.npz"
    size, digest = write_legacy(path, payload)

    loaded = legacy.load_verified_legacy_payload(
        path,
        expected_size_bytes=size,
        expected_sha256=digest,
        expected_step=1,
    )

    assert type(loaded["optimizer_state"]["masked"]) is optax.MaskedNode


def test_model_only_validates_but_explicitly_ignores_source_optimizer(
    workspace_tmp: Path,
):
    payload = source_payload()
    payload["optimizer_state"], replaced = replace_first_leaf(
        payload["optimizer_state"],
        lambda value: np.full_like(np.asarray(value), 7),
    )
    assert replaced

    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)
    result = import_payload(
        workspace_tmp / "state.npz",
        payload,
        model=target_model,
        optimizer=target_optimizer,
        init_mode="model-only",
    )
    after = extract_research_train_state(target_model, target_optimizer)

    assert result.source_step == 1
    assert result.optimizer_restored is False
    assert_tree_equal(after["model_trainable"], payload["model_trainable"])
    assert_tree_equal(after["optimizer_state"], before["optimizer_state"])
    assert int(after["step"]) == int(before["step"]) == 0


@pytest.mark.parametrize("tree_name", ["model_trainable", "optimizer_state"])
@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "wrong-shape", "wrong-dtype"],
)
def test_typed_abi_failures_leave_both_target_trees_unchanged(
    workspace_tmp: Path,
    tree_name: str,
    mutation: str,
):
    payload = source_payload()
    payload[tree_name] = mutate_tree(payload[tree_name], mutation)
    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)

    with pytest.raises(ValueError, match="checkpoint ABI mismatch"):
        import_payload(
            workspace_tmp / f"{tree_name}-{mutation}.npz",
            payload,
            model=target_model,
            optimizer=target_optimizer,
        )

    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        before,
    )


def test_model_only_rejects_bad_source_optimizer_before_model_mutation(
    workspace_tmp: Path,
):
    payload = source_payload()
    payload["optimizer_state"] = mutate_tree(payload["optimizer_state"], "extra")
    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)

    with pytest.raises(ValueError, match="optimizer checkpoint ABI mismatch"):
        import_payload(
            workspace_tmp / "state.npz",
            payload,
            model=target_model,
            optimizer=target_optimizer,
            init_mode="model-only",
        )

    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        before,
    )


@pytest.mark.parametrize(
    ("step", "expected_step", "error"),
    [
        (np.asarray(1.0, dtype=np.float32), 1, "integer scalar"),
        (np.asarray([1], dtype=np.int64), 1, "integer scalar"),
        (np.asarray(-1, dtype=np.int64), 1, "non-negative"),
        (np.asarray(2, dtype=np.int64), 1, "step mismatch"),
    ],
)
def test_bad_step_fails_before_mutation(
    workspace_tmp: Path,
    step: np.ndarray,
    expected_step: int,
    error: str,
):
    payload = source_payload()
    payload["step"] = step
    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)

    with pytest.raises(ValueError, match=error):
        import_payload(
            workspace_tmp / "state.npz",
            payload,
            model=target_model,
            optimizer=target_optimizer,
            expected_step=expected_step,
        )

    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        before,
    )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_top_level_key_mismatch_fails_before_mutation(
    workspace_tmp: Path,
    mutation: str,
):
    payload = source_payload()
    if mutation == "missing":
        del payload["optimizer_state"]
    else:
        payload["unexpected"] = np.asarray(0, dtype=np.int32)
    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)

    with pytest.raises(ValueError, match="payload keys differ"):
        import_payload(
            workspace_tmp / f"{mutation}.npz",
            payload,
            model=target_model,
            optimizer=target_optimizer,
        )

    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        before,
    )


def test_bad_checksum_is_rejected_before_np_load_or_mutation(
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = source_payload()
    path = workspace_tmp / "state.npz"
    size, _digest = write_legacy(path, payload)
    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)

    def forbidden_np_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("np.load must not run before checksum verification")

    monkeypatch.setattr(legacy.np, "load", forbidden_np_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        legacy.import_legacy_checkpoint(
            path,
            model=target_model,
            optimizer=target_optimizer,
            expected_size_bytes=size,
            expected_sha256="0" * 64,
            expected_step=1,
            init_mode="exact",
        )

    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        before,
    )


def test_bad_size_is_rejected_before_np_load_or_mutation(
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = source_payload()
    path = workspace_tmp / "state.npz"
    size, digest = write_legacy(path, payload)
    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)

    def forbidden_np_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("np.load must not run before size verification")

    monkeypatch.setattr(legacy.np, "load", forbidden_np_load)
    with pytest.raises(ValueError, match="size mismatch"):
        legacy.import_legacy_checkpoint(
            path,
            model=target_model,
            optimizer=target_optimizer,
            expected_size_bytes=size + 1,
            expected_sha256=digest,
            expected_step=1,
            init_mode="exact",
        )

    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        before,
    )


def test_path_outside_workspace_is_rejected_before_mutation():
    target_model, target_optimizer = make_components()
    before = extract_research_train_state(target_model, target_optimizer)

    with pytest.raises(ValueError, match="Path escapes"):
        legacy.import_legacy_checkpoint(
            "/tmp/outside-workspace-state.npz",
            model=target_model,
            optimizer=target_optimizer,
            expected_size_bytes=1,
            expected_sha256="0" * 64,
            expected_step=1,
            init_mode="exact",
        )

    assert_train_state_equal(
        extract_research_train_state(target_model, target_optimizer),
        before,
    )
