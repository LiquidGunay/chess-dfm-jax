from __future__ import annotations

import copy

import numpy as np
import pytest

from research import diagnose_compile_abi as diagnostic
from research import train


def test_schema_comparison_separates_container_from_leaf_changes() -> None:
    array = np.ones((2, 3), dtype=np.float32)
    before = train.research_state_abi({"weight": [array]})
    after = train.research_state_abi({"weight": (array,)})

    differences = diagnostic.compare_schemas(before, after)

    assert differences["missing_count"] == 0
    assert differences["extra_count"] == 0
    assert differences["changed_count"] == 1
    assert differences["leaf_difference_count"] == 0
    assert diagnostic.leaf_schema_summary(before) == (
        diagnostic.leaf_schema_summary(after)
    )
    assert diagnostic.pure_leaf_identity(
        {"weight": [array]},
        {"weight": [array]},
    ) == {
        "leaf_count": 1,
        "identical_object_count": 1,
        "different_object_count": 0,
        "different_objects": [],
    }


def test_schema_comparison_detects_leaf_signature_and_identity_changes() -> None:
    before_array = np.ones((2, 3), dtype=np.float32)
    after_array = np.ones((3, 2), dtype=np.float32)
    before = train.research_state_abi({"weight": before_array})
    after = train.research_state_abi({"weight": after_array})

    differences = diagnostic.compare_schemas(before, after)
    identity = diagnostic.pure_leaf_identity(
        {"weight": before_array},
        {"weight": after_array},
    )

    assert differences["leaf_difference_count"] == 1
    assert diagnostic.leaf_schema_summary(before) != (
        diagnostic.leaf_schema_summary(after)
    )
    assert identity["identical_object_count"] == 0
    assert identity["different_object_count"] == 1


def test_reference_manifest_validation_is_fail_closed() -> None:
    config = {"architecture": "accepted"}
    model_abi = train.research_state_abi(
        {"weight": np.ones((1,), dtype=np.float32)}
    )
    model_abi.update(
        {
            "sha256": diagnostic.CONSTRUCTOR_MODEL_ABI,
            "leaf_count": diagnostic.EXPECTED_MODEL_LEAVES,
            "nbytes": diagnostic.EXPECTED_MODEL_BYTES,
        }
    )
    manifest = {
        "model_abi": model_abi,
        "resume_contract": {"model_config": config},
        "state": {"filename": "state.npz"},
    }

    assert (
        diagnostic.validate_reference_manifest(
            manifest,
            model_config=config,
        )
        is model_abi
    )

    wrong_digest = copy.deepcopy(manifest)
    wrong_digest["model_abi"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model ABI changed"):
        diagnostic.validate_reference_manifest(
            wrong_digest,
            model_config=config,
        )

    with pytest.raises(ValueError, match="model config differs"):
        diagnostic.validate_reference_manifest(
            manifest,
            model_config={"architecture": "different"},
        )
