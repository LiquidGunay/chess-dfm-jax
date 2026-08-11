from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chess_dfm_jax.nnx_bt4 import TrainableParam
from research import diagnose_source_encoder_overlay as diagnostic


class TinyEncoder(nnx.Module):
    def __init__(self) -> None:
        self.weight = TrainableParam(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))


class TinyModel(nnx.Module):
    def __init__(self) -> None:
        self.encoder = TinyEncoder()
        self.head = TrainableParam(jnp.asarray([7.0, 8.0], dtype=jnp.float32))


def tiny_source_state() -> dict[str, object]:
    return {
        "encoder": {
            "weight": np.full((2, 3), 11.0, dtype=np.float32),
        },
        "head": np.asarray([99.0, 100.0], dtype=np.float32),
    }


def test_state_value_digest_is_order_stable_and_value_sensitive() -> None:
    left = {
        "a": np.arange(4, dtype=np.float32),
        "b": np.ones((2,), dtype=np.int32),
    }
    right = {
        "b": np.ones((2,), dtype=np.int32),
        "a": np.arange(4, dtype=np.float32),
    }
    changed = copy.deepcopy(left)
    changed["a"][0] = 1.0

    assert diagnostic.state_value_digest(left) == diagnostic.state_value_digest(
        right
    )
    assert diagnostic.state_value_digest(left) != diagnostic.state_value_digest(
        changed
    )


def test_state_value_digest_supports_bfloat16_raw_bytes() -> None:
    left = {
        "weight": np.asarray(
            jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16),
        )
    }
    changed = {
        "weight": np.asarray(
            jnp.asarray([1.0, 3.0], dtype=jnp.bfloat16),
        )
    }

    digest = diagnostic.state_value_digest(left)

    assert digest["leaf_count"] == 1
    assert digest["nbytes"] == 4
    assert digest != diagnostic.state_value_digest(changed)


def test_overlay_replaces_only_encoder_and_preserves_model_abi() -> None:
    model = TinyModel()
    head_before = model.head[...]

    evidence = diagnostic.overlay_source_encoder(model, tiny_source_state())

    np.testing.assert_array_equal(
        np.asarray(model.encoder.weight[...]),
        np.full((2, 3), 11.0, dtype=np.float32),
    )
    assert model.head[...] is head_before
    np.testing.assert_array_equal(np.asarray(model.head[...]), [7.0, 8.0])
    assert evidence["encoder_only"] is True
    assert evidence["accepted_encoder"] != evidence["source_encoder"]
    assert evidence["source_encoder"] == evidence["overlaid_encoder"]
    assert evidence["non_encoder_before"] == evidence["non_encoder_after"]
    assert evidence["non_encoder_identity"]["different_object_count"] == 0
    assert evidence["model_abi_before"] == evidence["model_abi_after"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"encoder": {}}, "checkpoint ABI mismatch"),
        (
            {"encoder": {"weight": np.ones((3, 2), dtype=np.float32)}},
            "checkpoint ABI mismatch",
        ),
        (
            {"encoder": {"weight": np.ones((2, 3), dtype=np.float64)}},
            "checkpoint ABI mismatch",
        ),
        ({"head": np.ones((2,), dtype=np.float32)}, "no encoder subtree"),
    ],
)
def test_overlay_fails_closed_before_mutation(
    mutation: dict[str, object],
    message: str,
) -> None:
    model = TinyModel()
    encoder_before = np.asarray(model.encoder.weight[...]).copy()
    head_before = model.head[...]

    with pytest.raises(ValueError, match=message):
        diagnostic.overlay_source_encoder(model, mutation)

    np.testing.assert_array_equal(
        np.asarray(model.encoder.weight[...]),
        encoder_before,
    )
    assert model.head[...] is head_before


def evaluation_records(
    *,
    ce_delta: float = 0.0,
    h1_delta: float = 0.0,
    accuracy_delta: float = 0.0,
    legal_delta: float = 0.0,
) -> list[dict[str, object]]:
    records = []
    for seed, offset in zip(diagnostic.VALIDATION_SEEDS, (0.0, 0.02), strict=True):
        accepted = {
            "dfm_ce_loss": 4.5 + offset,
            "dfm_ce_loss_by_horizon_h1": 2.8 + offset,
            "accuracy": 0.11 + offset,
            "first_legal_mass": 0.65 + offset,
            "extra_metric": 1.0 + offset,
        }
        overlay = {
            **accepted,
            "dfm_ce_loss": accepted["dfm_ce_loss"] + ce_delta,
            "dfm_ce_loss_by_horizon_h1": (
                accepted["dfm_ce_loss_by_horizon_h1"] + h1_delta
            ),
            "accuracy": accepted["accuracy"] + accuracy_delta,
            "first_legal_mass": accepted["first_legal_mass"] + legal_delta,
            "extra_metric": accepted["extra_metric"] + 3.0,
        }
        records.extend(
            [
                {
                    "variant": "accepted",
                    "validation_seed": seed,
                    "validation": accepted,
                },
                {
                    "variant": "source_encoder_overlay",
                    "validation_seed": seed,
                    "validation": overlay,
                },
            ]
        )
    return records


def test_overlay_decision_checks_each_seed_and_pooled_metrics() -> None:
    records = evaluation_records(
        ce_delta=0.99 * diagnostic.CE_DELTA_LIMIT,
        h1_delta=-0.99 * diagnostic.CE_DELTA_LIMIT,
        accuracy_delta=0.99 * diagnostic.ACCURACY_DELTA_LIMIT,
        legal_delta=-0.99 * diagnostic.LEGAL_MASS_DELTA_LIMIT,
    )
    aggregate = diagnostic.aggregate_variant_records(records)
    decision = diagnostic.overlay_decision(
        records=records,
        aggregate=aggregate,
    )

    assert decision["functionally_negligible_for_autoresearch_proxy"] is True
    assert decision["failures"] == []
    assert set(decision["comparisons"]) == {
        "pooled",
        "seed_10000",
        "seed_20000",
    }
    assert (
        decision["comparisons"]["pooled"]["all_common_metric_deltas"][
            "extra_metric"
        ]
        == pytest.approx(3.0)
    )


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("dfm_ce_loss", {"ce_delta": diagnostic.CE_DELTA_LIMIT + 1e-6}),
        (
            "dfm_ce_loss_by_horizon_h1",
            {"h1_delta": diagnostic.CE_DELTA_LIMIT + 1e-6},
        ),
        (
            "accuracy",
            {"accuracy_delta": diagnostic.ACCURACY_DELTA_LIMIT + 1e-6},
        ),
        (
            "first_legal_mass",
            {"legal_delta": diagnostic.LEGAL_MASS_DELTA_LIMIT + 1e-6},
        ),
    ],
)
def test_overlay_decision_rejects_threshold_excess(
    field: str,
    kwargs: dict[str, float],
) -> None:
    records = evaluation_records(**kwargs)
    decision = diagnostic.overlay_decision(
        records=records,
        aggregate=diagnostic.aggregate_variant_records(records),
    )

    assert decision["functionally_negligible_for_autoresearch_proxy"] is False
    assert any(field in failure for failure in decision["failures"])


def test_control_validation_is_exact_and_fail_closed() -> None:
    assert diagnostic.validate_control_metrics(
        diagnostic.EXPECTED_CONTROL
    ) == {key: 0.0 for key in diagnostic.EXPECTED_CONTROL}

    changed = dict(diagnostic.EXPECTED_CONTROL)
    changed["dfm_ce_loss"] += diagnostic.CONTROL_TOLERANCE * 2
    with pytest.raises(ValueError, match="does not reproduce"):
        diagnostic.validate_control_metrics(changed)
