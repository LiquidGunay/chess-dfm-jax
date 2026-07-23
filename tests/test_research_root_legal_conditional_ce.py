from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from research import train


class DeterministicEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(
            -0.5,
            0.5,
            64,
            dtype=jnp.float32,
        )[None, :, None]
        feature = jnp.linspace(
            0.1,
            1.0,
            self.embedding_size,
            dtype=jnp.float32,
        )[None, None, :]
        return square + plane_mean[:, None, None] * feature


def _config(
    coefficient: float,
) -> train.JointLatentSASAConfig:
    return train.JointLatentSASAConfig(
        token_dim=8,
        z_dim=16,
        projector_layers=1,
        projector_num_heads=4,
        projector_mlp_dim=32,
        jepa_condition_dim=12,
        dfm_layers=1,
        jepa_layers=1,
        jepa_mlp_dim=32,
        num_heads=4,
        mlp_dim=32,
        action_vocab_size=32,
        horizon=2,
        compute_dtype="float32",
        param_dtype="float32",
        encoder_dtype="float32",
        root_legal_conditional_ce_coeff=coefficient,
        first_legality_coeff=1.0,
        legality_on_masked_only=False,
        jepa_norm_loss_coeff=0.0,
        jepa_sigreg_coeff=0.1,
        jepa_pred_sigreg_coeff=0.1,
        jepa_sigreg_proj_dim=8,
        jepa_sigreg_example_count=2,
        jepa_target_sample_count=1,
        use_qk_gain=True,
        use_qk_norm=True,
        use_xsa=True,
        use_muon=False,
        remat_blocks=False,
        scan_layers=False,
    )


def _batch() -> dict[str, jax.Array]:
    batch_size = 4
    horizon = 2
    current = jnp.linspace(
        -0.25,
        0.75,
        batch_size * 112 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((batch_size, 112, 8, 8))
    future = jnp.stack(
        [current + 0.1 * (index + 1) for index in range(horizon)],
        axis=1,
    )
    actions = jnp.asarray(
        [[1, 2], [5, 6], [9, 10], [13, 14]],
        dtype=jnp.int32,
    )
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": actions,
        "valid": jnp.ones((batch_size,), dtype=jnp.float32),
        "future_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "legal_idx": jnp.stack(
            [actions, (actions + 1) % 32, (actions + 2) % 32],
            axis=-1,
        ),
        "legal_count": jnp.full(
            (batch_size, horizon),
            3,
            dtype=jnp.int32,
        ),
        "legal_masks_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _model(coefficient: float, *, seed: int = 17):
    return train.JointLatentSASAModel(
        DeterministicEncoder(),
        _config(coefficient),
        rngs=nnx.Rngs(seed),
    )


def _assert_tree_exact(left, right) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def test_sparse_root_legal_conditional_ce_matches_manual_reduction() -> None:
    logits = jnp.asarray(
        [
            [0.0, 0.5, -1.0, 1.5, 8.0, -3.0],
            [2.0, -2.0, 1.0, 0.25, -0.5, 9.0],
        ],
        dtype=jnp.float32,
    )
    result = train.root_legal_conditional_ce_from_logits(
        logits,
        jnp.asarray([3, 2]),
        jnp.asarray([[1, 3, 2, 5], [0, 2, 4, 5]]),
        jnp.asarray([3, 3]),
        jnp.asarray([True, True]),
        jnp.asarray([True, True]),
        jnp.asarray([True, True]),
    )
    expected = np.mean(
        [
            np.log(np.exp(0.5) + np.exp(1.5) + np.exp(-1.0))
            - 1.5,
            np.log(np.exp(2.0) + np.exp(1.0) + np.exp(-0.5))
            - 1.0,
        ]
    )
    assert float(result.loss) == pytest.approx(expected, abs=2e-7)
    assert float(result.candidate_count) == 2.0
    assert float(result.eligible_count) == 2.0
    assert float(result.eligible_fraction) == 1.0
    assert float(result.played_in_legal_fraction) == 1.0
    assert float(result.finite_fraction) == 1.0
    assert float(result.mean_legal_count) == 3.0
    assert float(result.top1_accuracy) == 0.5


def test_conditional_ce_gradient_is_zero_on_illegal_logits_and_sums_to_zero() -> None:
    legal = jnp.asarray([[1, 3, 4]], dtype=jnp.int32)

    def loss_fn(logits):
        return train.root_legal_conditional_ce_from_logits(
            logits,
            jnp.asarray([3]),
            legal,
            jnp.asarray([3]),
            jnp.asarray([True]),
            jnp.asarray([True]),
        ).loss

    gradient = jax.grad(loss_fn)(
        jnp.asarray([[0.2, 0.3, -0.4, 1.1, -0.5, 4.0]])
    )[0]
    gradient_np = np.asarray(gradient)
    np.testing.assert_array_equal(gradient_np[[0, 2, 5]], 0.0)
    assert float(jnp.sum(gradient[legal[0]])) == pytest.approx(
        0.0,
        abs=2e-7,
    )
    assert float(gradient[3]) < 0.0


def test_invalid_rows_are_excluded_and_coverage_is_explicit() -> None:
    result = train.root_legal_conditional_ce_from_logits(
        jnp.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, jnp.nan, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        ),
        jnp.asarray([1, 3, 1, 1]),
        jnp.asarray(
            [
                [1, 2, -1],
                [1, 2, -1],
                [1, 2, -1],
                [1, 7, -1],
            ]
        ),
        jnp.asarray([2, 2, 2, 2]),
        jnp.asarray([True, True, True, True]),
        jnp.asarray([True, True, True, True]),
        jnp.asarray([True, True, True, True]),
    )
    assert float(result.candidate_count) == 4.0
    assert float(result.eligible_count) == 1.0
    assert float(result.eligible_fraction) == 0.25
    assert float(result.played_in_legal_fraction) == 0.75
    assert float(result.finite_fraction) == 0.75
    assert float(result.legal_count_valid_fraction) == 0.75
    assert np.isfinite(float(result.loss))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"root_logits": jnp.zeros((2, 3, 4))}, "root_logits"),
        ({"legal_idx": jnp.zeros((3, 2))}, "legal_idx"),
        ({"played_actions": jnp.zeros((3,))}, "played_actions"),
        ({"legal_count": jnp.zeros((3,))}, "legal_count"),
        ({"sample_valid": jnp.zeros((3,), dtype=bool)}, "sample_valid"),
        ({"root_masked": jnp.zeros((3,), dtype=bool)}, "root_masked"),
        (
            {"legal_metadata_valid": jnp.zeros((3,), dtype=bool)},
            "legal_metadata_valid",
        ),
    ],
)
def test_shape_validation_fails_closed(kwargs, message) -> None:
    values = {
        "root_logits": jnp.zeros((2, 4)),
        "played_actions": jnp.zeros((2,), dtype=jnp.int32),
        "legal_idx": jnp.zeros((2, 2), dtype=jnp.int32),
        "legal_count": jnp.ones((2,), dtype=jnp.int32),
        "sample_valid": jnp.ones((2,), dtype=bool),
        "root_masked": jnp.ones((2,), dtype=bool),
        "legal_metadata_valid": jnp.ones((2,), dtype=bool),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        train.root_legal_conditional_ce_from_logits(**values)


def test_active_loss_adds_only_the_weighted_conditional_component() -> None:
    control = _model(0.0)
    candidate = _model(0.25)
    _assert_tree_exact(
        nnx.to_pure_dict(
            nnx.state(control, train.TrainableParam)
        ),
        nnx.to_pure_dict(
            nnx.state(candidate, train.TrainableParam)
        ),
    )
    args = (_batch(), jax.random.PRNGKey(23), 1.0, 1.0)
    control_loss, control_aux = train.normalized_stage1_loss_fn(
        control,
        *args,
        sample_future_targets=True,
    )
    candidate_loss, candidate_aux = train.normalized_stage1_loss_fn(
        candidate,
        *args,
        sample_future_targets=True,
    )
    for name in (
        "dfm_ce_loss",
        "dfm_ce_loss_by_horizon",
        "first_legality_loss",
        "weighted_legality_loss",
        "jepa_positive_loss",
        "jepa_sigreg_loss",
        "jepa_pred_sigreg_loss",
    ):
        np.testing.assert_array_equal(
            candidate_aux[name],
            control_aux[name],
        )
    np.testing.assert_allclose(
        candidate_loss - control_loss,
        candidate_aux[
            "root_legal_conditional_ce_weighted_loss"
        ],
        rtol=0.0,
        atol=2e-6,
    )
    assert float(
        candidate_aux[
            "root_legal_conditional_ce_eligible_fraction"
        ]
    ) == 1.0


@pytest.mark.parametrize(
    "value",
    [True, -0.1, float("nan"), float("inf")],
)
def test_coefficient_validation_fails_closed(value) -> None:
    with pytest.raises(
        ValueError,
        match="root_legal_conditional_ce_coeff",
    ):
        train.validate_objective_config(
            objective="normalized",
            config=dataclasses.replace(
                _config(0.0),
                root_legal_conditional_ce_coeff=value,
            ),
        )


def test_active_coefficient_requires_normalized_objective() -> None:
    train.validate_objective_config(
        objective="normalized",
        config=_config(0.25),
    )
    with pytest.raises(
        ValueError,
        match="requires --objective normalized",
    ):
        train.validate_objective_config(
            objective="legacy",
            config=_config(0.25),
        )


def test_serialization_audit_component_and_calibration_surface() -> None:
    assert (
        "root_legal_conditional_ce_coeff"
        not in train.serialized_model_config(_config(0.0))
    )
    assert train.serialized_model_config(_config(0.25))[
        "root_legal_conditional_ce_coeff"
    ] == 0.25
    assert train.gradient_component_names(_config(0.0)) == (
        *train.GRADIENT_COMPONENT_NAMES,
    )
    assert train.gradient_component_names(_config(0.25)) == (
        *train.GRADIENT_COMPONENT_NAMES,
        train.ROOT_LEGAL_CONDITIONAL_COMPONENT,
    )
    active = train.apply_experiment_overrides(
        train.JointLatentSASAConfig()
    )
    source_statistic = 2.0873505054041743
    calibrated = min(1.0, max(0.05, 0.25 / source_statistic))
    assert calibrated == 0.11976905620438309
    assert calibrated * source_statistic == 0.25
    assert active.root_legal_conditional_ce_coeff == calibrated


def test_resume_contract_records_enabled_only_semantics(monkeypatch) -> None:
    monkeypatch.setattr(
        train,
        "require_within_workspace",
        lambda path: path,
    )
    monkeypatch.setattr(
        train,
        "load_asset_manifest",
        lambda: {
            "trajectory_v3": {
                "archive": {
                    "sha256": "trajectory",
                    "size_bytes": 123,
                }
            }
        },
    )
    monkeypatch.setattr(train, "sha256_file", lambda _path: "source")

    def contract(coefficient: float):
        return train.build_research_resume_contract(
            config=_config(coefficient),
            objective="normalized",
            sigreg_reference_count=1.0,
            batch_size=4,
            train_seed=0,
            train_provenance={"kind": "test"},
            models_dir=Path("/mountpoint/.exp/models"),
        )

    assert (
        "root_legal_conditional_ce"
        not in contract(0.0)["objective"]
    )
    semantics = contract(0.25)["objective"][
        "root_legal_conditional_ce"
    ]
    assert semantics["coefficient"] == 0.25
    assert semantics["direct_illegal_logit_gradient"] == "exact_zero"
    assert semantics["legal_logit_gradient_sum"] == "exact_zero"
    assert semantics["additional_encoder_calls"] == 0
    assert semantics["additional_planner_calls"] == 0
    assert semantics["inference_affected"] is False
    assert semantics["checkpoint_state_abi"] == "unchanged"


def test_read_only_checkpoint_diagnostic_overlay_is_explicit() -> None:
    base = _config(0.0)
    unchanged, disabled = train.checkpoint_evaluation_diagnostic_overlay(
        base,
        root_legal_conditional_ce=False,
    )
    assert unchanged is base
    assert disabled == {}

    overlaid, metadata = train.checkpoint_evaluation_diagnostic_overlay(
        base,
        root_legal_conditional_ce=True,
    )
    assert overlaid.root_legal_conditional_ce_coeff == 1.0
    assert dataclasses.replace(
        overlaid,
        root_legal_conditional_ce_coeff=0.0,
    ) == base
    semantics = metadata["root_legal_conditional_ce"]
    assert semantics["purpose"] == "read_only_metric_overlay"
    assert semantics["model_state_mutation"] is False
    assert semantics["forward_logits_affected"] is False
    assert semantics["reported_aggregate_loss_includes_overlay"] is True

    parsed = train.parse_args(
        [
            "--eval-checkpoints",
            "/mountpoint/.exp/checkpoint",
            "--eval-root-legal-conditional-ce",
        ]
    )
    assert parsed.eval_root_legal_conditional_ce is True
