from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chess_dfm_jax.nnx_bt4 import TrainableParam
from chess_dfm_jax.policy import (
    ACTION_VOCAB_SIZE,
    legacy_to_lc0_canonical_1858_index_map,
)
from research import train


class FrozenTinyPolicyHead(nnx.Module):
    def __init__(self, embedding_size: int):
        self.w = nnx.Param(
            jnp.linspace(
                -0.02,
                0.02,
                embedding_size * ACTION_VOCAB_SIZE,
                dtype=jnp.float32,
            ).reshape((embedding_size, ACTION_VOCAB_SIZE))
        )
        self.b = nnx.Param(
            jnp.linspace(
                -0.1,
                0.1,
                ACTION_VOCAB_SIZE,
                dtype=jnp.float32,
            )
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        pooled = jnp.mean(tokens, axis=1)
        return pooled @ self.w[...] + self.b[...]


class TinyEncoderWithFrozenPolicy(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size
        self.scale = TrainableParam(
            jnp.linspace(
                0.25,
                1.25,
                embedding_size,
                dtype=jnp.float32,
            )
        )
        self.policy_head = FrozenTinyPolicyHead(embedding_size)

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        batch_size = planes.shape[0]
        plane_mean = jnp.mean(planes, axis=(1, 2, 3))
        square = jnp.linspace(
            -0.5,
            0.5,
            64,
            dtype=jnp.float32,
        )[None, :, None]
        return square + plane_mean.reshape(
            (batch_size, 1, 1)
        ) * self.scale[None, None, :]


def _config(
    *,
    coefficient: float,
    **updates: Any,
) -> train.JointLatentSASAConfig:
    values: dict[str, Any] = {
        "token_dim": 8,
        "z_dim": 16,
        "projector_layers": 1,
        "projector_num_heads": 4,
        "projector_mlp_dim": 32,
        "jepa_condition_dim": 16,
        "dfm_layers": 1,
        "jepa_layers": 1,
        "jepa_mlp_dim": 32,
        "num_heads": 4,
        "mlp_dim": 32,
        "action_vocab_size": ACTION_VOCAB_SIZE,
        "horizon": 2,
        "compute_dtype": "float32",
        "param_dtype": "float32",
        "encoder_dtype": "float32",
        "dfm_ce_coeff": 0.0,
        "bt4_policy_distill_coeff": coefficient,
        "first_legality_coeff": 0.0,
        "jepa_positive_coeff": 0.0,
        "jepa_norm_loss_coeff": 0.0,
        "jepa_sigreg_coeff": 0.0,
        "jepa_pred_sigreg_coeff": 0.0,
        "jepa_sigreg_proj_dim": 8,
        "value_coeff": 0.0,
        "wdl_coeff": 0.0,
        "use_qk_gain": True,
        "use_qk_norm": True,
        "use_xsa": True,
        "use_muon": False,
        "remat_blocks": False,
        "scan_layers": False,
    }
    values.update(updates)
    return train.JointLatentSASAConfig(**values)


def _batch() -> dict[str, jax.Array]:
    current = jnp.zeros((2, 112, 8, 8), dtype=jnp.float32)
    current = current.at[1, train.CLASSICAL_SIDE_TO_MOVE_PLANE].set(1.0)
    actions = jnp.asarray(
        [[322, 159], [1498, 1755]],
        dtype=jnp.int32,
    )
    legal_idx = jnp.asarray(
        [
            [[322, 159, 0], [159, 322, 0]],
            [[1498, 1755, 0], [1755, 1498, 0]],
        ],
        dtype=jnp.int32,
    )
    return {
        "current_planes": current,
        "future_planes": jnp.stack(
            (current + 0.125, current + 0.25),
            axis=1,
        ),
        "action_indices": actions,
        "valid": jnp.ones((2,), dtype=jnp.float32),
        "future_valid": jnp.ones((2, 2), dtype=jnp.float32),
        "legal_idx": legal_idx,
        "legal_count": jnp.full((2, 2), 2, dtype=jnp.int32),
        "legal_masks_valid": jnp.ones((2, 2), dtype=jnp.float32),
        "deterministic_t": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _pure_trainable(module: nnx.Module) -> dict[str, Any]:
    return nnx.to_pure_dict(nnx.state(module, TrainableParam))


def _tree_norm(tree: Any) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf, dtype=np.float64)))
                for leaf in jax.tree.leaves(tree)
            )
        )
    )


def _distillation_inputs():
    teacher = jnp.zeros((2, ACTION_VOCAB_SIZE), dtype=jnp.float32)
    teacher = teacher.at[0, 322].set(np.log(0.8))
    teacher = teacher.at[0, 159].set(np.log(0.2))
    teacher = teacher.at[1, 322].set(np.log(0.7))
    teacher = teacher.at[1, 159].set(np.log(0.3))

    student = jnp.zeros((2, ACTION_VOCAB_SIZE), dtype=jnp.float32)
    student = student.at[0, 322].set(np.log(0.6))
    student = student.at[0, 159].set(np.log(0.4))
    student = student.at[1, 1498].set(np.log(0.3))
    student = student.at[1, 1755].set(np.log(0.7))

    planes = jnp.zeros((2, 112, 8, 8), dtype=jnp.float32)
    planes = planes.at[1, train.CLASSICAL_SIDE_TO_MOVE_PLANE].set(1.0)
    legal_idx = jnp.asarray(
        [[322, 159, 0], [1498, 1755, 0]],
        dtype=jnp.int32,
    )
    return teacher, student, planes, legal_idx


def test_exact_codec_aware_kl_and_target_label_independence() -> None:
    teacher, student, planes, legal_idx = _distillation_inputs()
    kwargs = {
        "teacher_canonical_logits": teacher,
        "student_legacy_logits": student,
        "current_planes": planes,
        "played_actions": jnp.asarray([322, 1498], dtype=jnp.int32),
        "legal_idx": legal_idx,
        "legal_count": jnp.asarray([2, 2], dtype=jnp.int32),
        "sample_valid": jnp.asarray([True, True]),
        "root_masked": jnp.asarray([True, True]),
        "legal_metadata_valid": jnp.asarray([True, True]),
    }
    result = train.bt4_policy_distillation_from_logits(**kwargs)
    expected = 0.5 * (
        0.8 * np.log(0.8 / 0.6)
        + 0.2 * np.log(0.2 / 0.4)
        + 0.7 * np.log(0.7 / 0.3)
        + 0.3 * np.log(0.3 / 0.7)
    )

    np.testing.assert_allclose(result.loss, expected, rtol=1e-6)
    assert float(result.eligible_count) == 2.0
    assert float(result.eligible_fraction) == 1.0
    assert float(result.side_plane_valid_fraction) == 1.0
    assert float(result.mapping_complete_fraction) == 1.0
    assert float(result.legal_mapping_coverage) == 1.0
    assert float(result.teacher_finite_fraction) == 1.0
    assert float(result.teacher_played_accuracy) == 1.0
    assert float(result.teacher_student_top1_agreement) == 0.5

    changed_labels = train.bt4_policy_distillation_from_logits(
        **(
            kwargs
            | {
                "played_actions": jnp.asarray(
                    [159, 1755],
                    dtype=jnp.int32,
                )
            }
        )
    )
    np.testing.assert_array_equal(result.loss, changed_labels.loss)
    assert (
        float(changed_labels.teacher_played_accuracy)
        != float(result.teacher_played_accuracy)
    )


def test_teacher_gradient_is_stopped_and_student_gradient_is_nonzero() -> None:
    teacher, student, planes, legal_idx = _distillation_inputs()

    def loss_fn(teacher_logits, student_logits):
        return train.bt4_policy_distillation_from_logits(
            teacher_logits,
            student_logits,
            planes,
            jnp.asarray([322, 1498], dtype=jnp.int32),
            legal_idx,
            jnp.asarray([2, 2], dtype=jnp.int32),
            jnp.asarray([True, True]),
            jnp.asarray([True, True]),
        ).loss

    teacher_grad, student_grad = jax.grad(
        loss_fn,
        argnums=(0, 1),
    )(teacher, student)
    np.testing.assert_array_equal(teacher_grad, jnp.zeros_like(teacher_grad))
    assert _tree_norm(student_grad) > 0.0


def test_invalid_side_plane_and_partial_black_mapping_are_gated() -> None:
    teacher, student, planes, legal_idx = _distillation_inputs()
    planes = planes.at[0, train.CLASSICAL_SIDE_TO_MOVE_PLANE].set(0.5)
    black_map = legacy_to_lc0_canonical_1858_index_map(
        black_to_move=True
    )
    invalid_black_legacy = int(np.flatnonzero(black_map < 0)[0])
    legal_idx = legal_idx.at[1, 0].set(invalid_black_legacy)

    result = train.bt4_policy_distillation_from_logits(
        teacher,
        student,
        planes,
        jnp.asarray([322, invalid_black_legacy], dtype=jnp.int32),
        legal_idx,
        jnp.asarray([2, 2], dtype=jnp.int32),
        jnp.asarray([True, True]),
        jnp.asarray([True, True]),
    )
    assert float(result.eligible_count) == 0.0
    assert float(result.loss) == 0.0
    assert float(result.side_plane_valid_fraction) == 0.5
    assert float(result.mapping_complete_fraction) == 0.5
    assert float(result.legal_mapping_coverage) == 0.75
    assert np.isfinite(float(result.teacher_played_nll))


@pytest.mark.parametrize(
    "value",
    [True, -0.25, float("nan"), float("inf")],
)
def test_distillation_coefficient_validation_fails_closed(value: Any) -> None:
    config = train.JointLatentSASAConfig(
        bt4_policy_distill_coeff=value
    )
    with pytest.raises(ValueError, match="bt4_policy_distill_coeff"):
        train.validate_objective_config(
            objective="normalized",
            config=config,
        )


def test_distillation_requires_normalized_1858_policy() -> None:
    with pytest.raises(ValueError, match="requires --objective normalized"):
        train.validate_objective_config(
            objective="legacy",
            config=train.JointLatentSASAConfig(
                bt4_policy_distill_coeff=0.25
            ),
        )
    with pytest.raises(ValueError, match="action_vocab_size=1858"):
        train.validate_objective_config(
            objective="normalized",
            config=train.JointLatentSASAConfig(
                action_vocab_size=32,
                bt4_policy_distill_coeff=0.25,
            ),
        )


def test_distillation_preserves_state_abi_and_serializes_only_when_active() -> None:
    control = train.JointLatentSASAModel(
        TinyEncoderWithFrozenPolicy(),
        _config(coefficient=0.0),
        rngs=nnx.Rngs(17),
    )
    candidate = train.JointLatentSASAModel(
        TinyEncoderWithFrozenPolicy(),
        _config(coefficient=0.25),
        rngs=nnx.Rngs(17),
    )
    assert train.research_state_abi(
        nnx.state(control, TrainableParam)
    ) == train.research_state_abi(
        nnx.state(candidate, TrainableParam)
    )
    assert (
        "bt4_policy_distill_coeff"
        not in train.serialized_model_config(control.config)
    )
    assert train.serialized_model_config(candidate.config)[
        "bt4_policy_distill_coeff"
    ] == 0.25
    policy_trainable = nnx.to_pure_dict(
        nnx.state(candidate.encoder.policy_head, TrainableParam)
    )
    assert policy_trainable == {}


def test_active_distillation_gradients_reach_student_paths_not_teacher_head() -> None:
    model = train.JointLatentSASAModel(
        TinyEncoderWithFrozenPolicy(),
        _config(coefficient=1.0),
        rngs=nnx.Rngs(23),
    )
    (loss, aux), gradients = nnx.value_and_grad(
        train.joint_stage1_loss_fn,
        argnums=nnx.DiffState(0, TrainableParam),
        has_aux=True,
    )(
        model,
        _batch(),
        jax.random.PRNGKey(29),
    )
    pure = nnx.to_pure_dict(gradients)

    np.testing.assert_allclose(
        loss,
        aux["bt4_policy_distill_loss"],
        rtol=0.0,
        atol=1e-6,
    )
    assert float(aux["bt4_policy_distill_eligible_fraction"]) == 1.0
    assert float(aux["bt4_policy_distill_legal_mapping_coverage"]) == 1.0
    for name in (
        "encoder",
        "dfm_state_projector",
        "dfm_blocks",
        "out_proj",
    ):
        assert _tree_norm(pure[name]) > 0.0, name
    assert "policy_head" not in pure["encoder"]
    assert train.gradient_component_names(model.config) == (
        train.GRADIENT_COMPONENT_NAMES
        + (train.BT4_POLICY_DISTILL_COMPONENT,)
    )
    components = train.gradient_component_vector(
        model,
        _batch(),
        jax.random.PRNGKey(29),
        1.0,
    )
    assert components.shape == (6,)
    np.testing.assert_allclose(
        components[-1],
        aux["bt4_policy_distill_loss"],
        rtol=0.0,
        atol=1e-6,
    )


def test_distillation_resume_contract_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(train, "require_within_workspace", lambda path: Path(path))
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
    config = _config(coefficient=0.25)
    contract = train.build_research_resume_contract(
        config=config,
        objective="normalized",
        sigreg_reference_count=1.0,
        batch_size=128,
        train_seed=0,
        train_provenance={"kind": "test"},
        models_dir=train.REPO_ROOT / "models",
    )
    distillation = contract["objective"]["bt4_root_policy_distillation"]
    assert distillation["coefficient"] == 0.25
    assert distillation["teacher_head"] == "frozen_source_bt4_policy_head"
    assert distillation["teacher_token_source"] == (
        "online_current_bt4_tokens"
    )
    assert distillation["additional_bt4_encoder_calls"] == 0
    assert distillation["inference_policy_head_calls"] == 0
    assert distillation["model_state_abi"] == "unchanged"
    assert any(
        path.endswith("chess_dfm_jax/policy.py")
        for path in contract["code"]["files"]
    )


def test_active_model_requires_a_policy_head() -> None:
    class EncoderWithoutPolicy(nnx.Module):
        embedding_size = 16

        def encode_tokens(self, planes):
            return jnp.zeros(
                (planes.shape[0], 64, self.embedding_size),
                dtype=jnp.float32,
            )

    with pytest.raises(ValueError, match="encoder.policy_head"):
        train.JointLatentSASAModel(
            EncoderWithoutPolicy(),
            _config(coefficient=0.25),
            rngs=nnx.Rngs(31),
        )
