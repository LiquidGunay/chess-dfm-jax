from __future__ import annotations

import copy
import types

import pytest

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
