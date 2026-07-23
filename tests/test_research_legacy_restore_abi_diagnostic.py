from __future__ import annotations

import copy
import types

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chess_dfm_jax.nnx_bt4 import TrainableParam
from research import diagnose_compile_abi as schema_tools
from research import diagnose_legacy_restore_abi as diagnostic


def valid_import_result() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_abi_sha256=schema_tools.CONSTRUCTOR_MODEL_ABI,
        optimizer_abi_sha256=diagnostic.SOURCE_OPTIMIZER_ABI,
        optimizer_restored=False,
        init_mode="model-only",
    )


def runtime_optimizer_abi() -> dict[str, object]:
    return {
        "sha256": schema_tools.CONSTRUCTOR_OPTIMIZER_ABI,
        "leaf_count": schema_tools.EXPECTED_OPTIMIZER_LEAVES,
        "nbytes": schema_tools.EXPECTED_OPTIMIZER_BYTES,
    }


def test_model_only_import_invariants_accept_untouched_fresh_optimizer() -> None:
    optimizer = runtime_optimizer_abi()

    diagnostic.validate_import_invariants(
        import_result=valid_import_result(),
        optimizer_before=optimizer,
        optimizer_after=copy.deepcopy(optimizer),
        optimizer_step_before=0,
        optimizer_step_after=0,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"model_abi_sha256": "0" * 64},
            "legacy model ABI changed",
        ),
        (
            {"optimizer_abi_sha256": "0" * 64},
            "legacy optimizer ABI changed",
        ),
        (
            {"optimizer_restored": True},
            "unexpectedly restored optimizer",
        ),
        (
            {"init_mode": "exact"},
            "unexpected mode",
        ),
    ],
)
def test_model_only_import_invariants_reject_source_contract_changes(
    mutation: dict[str, object],
    message: str,
) -> None:
    result = valid_import_result()
    for key, value in mutation.items():
        setattr(result, key, value)
    optimizer = runtime_optimizer_abi()

    with pytest.raises(ValueError, match=message):
        diagnostic.validate_import_invariants(
            import_result=result,
            optimizer_before=optimizer,
            optimizer_after=optimizer,
            optimizer_step_before=0,
            optimizer_step_after=0,
        )


def test_model_only_import_invariants_reject_runtime_optimizer_changes() -> None:
    optimizer = runtime_optimizer_abi()
    changed = copy.deepcopy(optimizer)
    changed["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="changed the fresh runtime optimizer ABI"):
        diagnostic.validate_import_invariants(
            import_result=valid_import_result(),
            optimizer_before=optimizer,
            optimizer_after=changed,
            optimizer_step_before=0,
            optimizer_step_after=0,
        )

    with pytest.raises(ValueError, match="changed the fresh optimizer step"):
        diagnostic.validate_import_invariants(
            import_result=valid_import_result(),
            optimizer_before=optimizer,
            optimizer_after=optimizer,
            optimizer_step_before=0,
            optimizer_step_after=265_000,
        )


def test_jax_signature_ignores_equal_jax_and_numpy_storage_types() -> None:
    jax_state = {"weight": jnp.ones((2, 3), dtype=jnp.bfloat16)}
    numpy_state = {
        "weight": np.asarray(jax_state["weight"]),
    }

    jax_signature = diagnostic.jax_state_signature(jax_state)
    numpy_signature = diagnostic.jax_state_signature(numpy_state)

    assert jax_signature["sha256"] == numpy_signature["sha256"]
    assert jax_signature["records"] == numpy_signature["records"]
    assert jax_signature["treedef"] == numpy_signature["treedef"]
    assert jax_signature["concrete_type_counts"] == {
        "jaxlib._jax.ArrayImpl": 1,
    }
    assert numpy_signature["concrete_type_counts"] == {"numpy.ndarray": 1}


def test_jax_signature_detects_abstract_shape_or_dtype_changes() -> None:
    baseline = diagnostic.jax_state_signature(
        {"weight": jnp.ones((2, 3), dtype=jnp.bfloat16)}
    )
    shape_change = diagnostic.jax_state_signature(
        {"weight": jnp.ones((3, 2), dtype=jnp.bfloat16)}
    )
    dtype_change = diagnostic.jax_state_signature(
        {"weight": jnp.ones((2, 3), dtype=jnp.float32)}
    )

    assert baseline["sha256"] != shape_change["sha256"]
    assert baseline["sha256"] != dtype_change["sha256"]


def test_graph_definition_signature_is_stable_across_value_update() -> None:
    class Tiny(nnx.Module):
        def __init__(self) -> None:
            self.weight = TrainableParam(jnp.ones((2, 3)))

    model = Tiny()
    before, before_signature = diagnostic.graph_definition_signature(model)
    model.weight[...] = jnp.zeros((2, 3))
    after, after_signature = diagnostic.graph_definition_signature(model)

    assert before == after
    assert before_signature == after_signature
