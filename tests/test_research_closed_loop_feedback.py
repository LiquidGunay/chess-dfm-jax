from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chess_dfm_jax.nnx_bt4 import TrainableParam
from chess_dfm_jax.policy import (
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
)
from research import train
from research.inference import (
    _infer_dfm_actions_from_current_impl,
    _infer_dfm_from_current_impl,
    infer_dfm_from_current,
    refine_dfm_from_latents,
)


class TinyTrainableEncoder(nnx.Module):
    def __init__(self, embedding_size: int = 16):
        self.embedding_size = embedding_size
        self.scale = TrainableParam(jnp.asarray(1.0, dtype=jnp.float32))

    def encode_tokens(self, planes: jax.Array) -> jax.Array:
        batch_size = planes.shape[0]
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
        return (
            square
            + plane_mean.reshape((batch_size, 1, 1)) * feature
        ) * self.scale[...]


def _config(
    *,
    feedback_mode: str = "final_pass_adjoint",
    **updates,
) -> train.JointLatentSASAConfig:
    values = {
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
        "action_vocab_size": 32,
        "horizon": 8,
        "compute_dtype": "float32",
        "param_dtype": "float32",
        "encoder_dtype": "float32",
        "first_legality_coeff": 1.0,
        "legality_on_masked_only": False,
        "jepa_norm_loss_coeff": 0.0,
        "jepa_sigreg_coeff": 5.76,
        "jepa_pred_sigreg_coeff": 1.0,
        "jepa_sigreg_kind": "le_jepa",
        "jepa_sigreg_proj_dim": 8,
        "jepa_target_sample_count": 1,
        "jepa_target_sampling_unit": "example_balanced",
        "jepa_rollout_mode": "recurrent",
        "jepa_feedback_mode": feedback_mode,
        "use_qk_gain": True,
        "use_qk_norm": True,
        "use_xsa": True,
        "remat_blocks": False,
        "scan_layers": False,
    }
    values.update(updates)
    return train.JointLatentSASAConfig(**values)


def _model(
    *,
    feedback_mode: str = "final_pass_adjoint",
    seed: int = 17,
    **updates,
) -> train.JointLatentSASAModel:
    return train.JointLatentSASAModel(
        TinyTrainableEncoder(),
        _config(feedback_mode=feedback_mode, **updates),
        rngs=nnx.Rngs(seed),
    )


def _batch() -> dict[str, jax.Array]:
    batch_size = 4
    horizon = 8
    current = jnp.linspace(
        -0.25,
        0.75,
        batch_size * 112 * 8 * 8,
        dtype=jnp.float32,
    ).reshape((batch_size, 112, 8, 8))
    future = jnp.stack(
        [
            current
            + jnp.asarray(0.03 * (index + 1), dtype=jnp.float32)
            for index in range(horizon)
        ],
        axis=1,
    )
    actions = (
        jnp.arange(batch_size * horizon, dtype=jnp.int32)
        .reshape((batch_size, horizon))
        % 30
    ) + 1
    legal_idx = jnp.stack((actions, (actions + 1) % 32), axis=-1)
    return {
        "current_planes": current,
        "future_planes": future,
        "action_indices": actions,
        "valid": jnp.ones((batch_size,), dtype=jnp.float32),
        "future_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "legal_idx": legal_idx,
        "legal_count": jnp.full(
            (batch_size, horizon),
            2,
            dtype=jnp.int32,
        ),
        "legal_masks_valid": jnp.ones(
            (batch_size, horizon),
            dtype=jnp.float32,
        ),
        "deterministic_t": jnp.asarray(0.25, dtype=jnp.float32),
    }


def _tree_norm(tree) -> float:
    leaves = jax.tree.leaves(tree)
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf, dtype=np.float64)))
                for leaf in leaves
            )
        )
    )


def test_sparse_legal_mask_and_proposal_are_target_independent() -> None:
    legal_idx = jnp.asarray([[3, 5, 0], [7, 0, 0]], dtype=jnp.int32)
    legal_count = jnp.asarray([2, 1], dtype=jnp.int32)
    legal_mask = train.root_legal_mask_from_indices(
        legal_idx,
        legal_count,
        action_vocab_size=10,
    )
    np.testing.assert_array_equal(
        np.asarray(legal_mask),
        np.asarray(
            [
                [False, False, False, True, False, True, False, False, False, False],
                [False, False, False, False, False, False, False, True, False, False],
            ]
        ),
    )

    logits = jnp.asarray(
        [
            [9.0, 8.0, 7.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    masked = jnp.asarray([10, 10], dtype=jnp.int32)
    first = train.proposal_from_root_logits(
        logits,
        masked,
        legal_mask,
        jnp.asarray([True, True]),
        mask_token_id=10,
    )
    # No played target is an input to proposal selection. Arbitrary target
    # changes therefore cannot alter this result.
    arbitrary_targets_a = jnp.asarray([3, 7], dtype=jnp.int32)
    arbitrary_targets_b = jnp.asarray([5, 2], dtype=jnp.int32)
    del arbitrary_targets_a, arbitrary_targets_b
    second = train.proposal_from_root_logits(
        logits,
        masked,
        legal_mask,
        jnp.asarray([True, True]),
        mask_token_id=10,
    )
    np.testing.assert_array_equal(first.action, jnp.asarray([5, 7]))
    np.testing.assert_array_equal(first.action, second.action)
    np.testing.assert_array_equal(
        first.feedback_gate,
        second.feedback_gate,
    )

    revealed = train.proposal_from_root_logits(
        logits,
        jnp.asarray([3, 7], dtype=jnp.int32),
        legal_mask,
        jnp.asarray([True, False]),
        mask_token_id=10,
    )
    np.testing.assert_array_equal(revealed.action, jnp.asarray([3, 7]))
    np.testing.assert_array_equal(
        revealed.feedback_gate,
        jnp.asarray([1.0, 0.0]),
    )


def test_adjoint_feedback_matches_formula_and_caps_at_half_state_rms() -> None:
    model = _model(seed=23)
    weight = jnp.linspace(
        -0.2,
        0.2,
        model.config.token_dim * model.config.z_dim,
        dtype=jnp.float32,
    ).reshape((model.config.token_dim, model.config.z_dim))
    model.jepa_hidden_adapter.w[...] = weight
    model.jepa_hidden_adapter.b[...] = jnp.zeros_like(
        model.jepa_hidden_adapter.b[...]
    )
    z_dfm = jnp.ones((2, 64, model.config.token_dim), dtype=jnp.float32)
    z0 = jnp.zeros((2, model.config.z_dim), dtype=jnp.float32)
    z1 = jnp.stack(
        (
            jnp.linspace(-0.01, 0.01, model.config.z_dim),
            jnp.full((model.config.z_dim,), 100.0),
        )
    )
    gate = jnp.asarray([0.5, 1.0], dtype=jnp.float32)

    actual = model.dfm_latents_with_jepa_feedback(
        z_dfm,
        z0,
        z1,
        gate,
    )
    correction = np.sqrt(model.config.token_dim / model.config.z_dim)
    raw = np.asarray(z1) @ np.asarray(weight).T * correction
    raw_rms = np.sqrt(np.mean(np.square(raw), axis=-1))
    cap = np.minimum(1.0, 0.5 / np.maximum(raw_rms, 1e-6))
    expected_feedback = raw * cap[:, None] * np.asarray(gate)[:, None]
    expected = np.asarray(z_dfm) + expected_feedback[:, None, :]

    np.testing.assert_allclose(actual.latents, expected, rtol=1e-5, atol=1e-6)
    assert float(actual.applied_feedback_rms[1]) <= 0.500001
    zero = model.dfm_latents_with_jepa_feedback(
        z_dfm,
        z0,
        z1,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(zero.latents, z_dfm)
    np.testing.assert_array_equal(
        zero.applied_feedback_rms,
        jnp.zeros((2,), dtype=jnp.float32),
    )


def test_feedback_loss_is_finite_and_reports_preliminary_comparison() -> None:
    model = _model(seed=29)
    model.jepa_transition.cond_w[...] = jnp.full_like(
        model.jepa_transition.cond_w[...],
        1e-3,
    )
    loss, aux = train.normalized_stage1_loss_fn(
        model,
        _batch(),
        jax.random.PRNGKey(31),
        1.0,
        1.0,
        sample_future_targets=True,
    )

    assert jnp.isfinite(loss)
    assert aux["jepa_feedback_active"] == 1.0
    assert aux["jepa_feedback_preliminary_planner_calls"] == 1.0
    assert aux["jepa_feedback_conditioned_planner_calls"] == 1.0
    assert jnp.isfinite(aux["jepa_feedback_preliminary_dfm_ce_loss"])
    assert jnp.isfinite(aux["jepa_feedback_dfm_ce_gain"])
    assert aux["jepa_prediction_horizon_count"] == 8.0
    assignment_count = np.asarray(
        aux["jepa_target_assignment_count_by_horizon"]
    )
    assert assignment_count.sum() == 4.0
    assert assignment_count.max() - assignment_count.min() <= 1.0
    assert 0.0 < float(aux["jepa_feedback_gate_mean"]) <= 1.0
    assert float(aux["jepa_feedback_applied_to_state_rms_ratio"]) <= 0.500001


def test_feedback_gradient_reaches_every_intended_path() -> None:
    model = _model(seed=37)
    model.jepa_transition.cond_w[...] = jnp.full_like(
        model.jepa_transition.cond_w[...],
        1e-3,
    )
    batch = _batch()

    def objective(candidate):
        tokens = candidate.encode_bt4_tokens(batch["current_planes"])
        z0 = candidate.jepa_latents(tokens)
        z_dfm = candidate.dfm_latents(tokens)
        noisy = jnp.full(
            (4, 8),
            candidate.config.action_vocab_size,
            dtype=jnp.int32,
        )
        t = jnp.full((4,), 0.25, dtype=jnp.float32)
        preliminary_logits, hidden = candidate.planner_from_latents(
            z_dfm,
            noisy,
            t,
            return_hidden=True,
        )
        legal_mask = train.root_legal_mask_from_indices(
            batch["legal_idx"][:, 0, :],
            batch["legal_count"][:, 0],
            action_vocab_size=candidate.config.action_vocab_size,
        )
        proposal = train.proposal_from_root_logits(
            preliminary_logits[:, 0],
            noisy[:, 0],
            legal_mask,
            jnp.ones((4,), dtype=jnp.bool_),
            mask_token_id=candidate.config.action_vocab_size,
        )
        z1 = candidate.jepa_step_from_latents(
            z0,
            proposal.action,
            hidden["action_tokens"][:, 0],
            z0_normalized=True,
        )
        feedback = candidate.dfm_latents_with_jepa_feedback(
            z_dfm,
            z0,
            z1,
            proposal.feedback_gate,
        )
        final_logits = candidate.planner_from_latents(
            feedback.latents,
            noisy,
            t,
        )
        weights = jnp.linspace(
            -0.5,
            0.75,
            final_logits.size,
            dtype=jnp.float32,
        ).reshape(final_logits.shape)
        return jnp.mean(final_logits * weights)

    gradients = nnx.to_pure_dict(
        nnx.grad(
            objective,
            argnums=nnx.DiffState(0, TrainableParam),
        )(model)
    )
    for path in (
        ("encoder",),
        ("state_projector",),
        ("dfm_state_projector",),
        ("dfm_blocks",),
        ("action_embed",),
        ("jepa_action_embed",),
        ("jepa_hidden_adapter",),
        ("jepa_transition",),
        ("out_proj",),
    ):
        node = gradients
        for part in path:
            node = node[part]
        assert _tree_norm(node) > 0.0, path


def test_feedback_mode_preserves_parameter_state_abi() -> None:
    disabled = _model(feedback_mode="none", seed=41)
    enabled = _model(seed=41)
    assert train.research_state_abi(
        nnx.state(disabled, TrainableParam)
    ) == train.research_state_abi(
        nnx.state(enabled, TrainableParam)
    )
    assert "jepa_feedback_mode" not in train.serialized_model_config(
        disabled.config
    )
    assert train.serialized_model_config(enabled.config)[
        "jepa_feedback_mode"
    ] == "final_pass_adjoint"
    assert (
        train.apply_experiment_overrides(
            train.JointLatentSASAConfig()
        ).jepa_feedback_mode
        == "final_pass_adjoint"
    )


def test_feedback_resume_contract_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(train, "require_within_workspace", lambda path: path)
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
    contract = train.build_research_resume_contract(
        config=_config(),
        objective="normalized",
        sigreg_reference_count=64.0,
        batch_size=128,
        train_seed=0,
        train_provenance={"kind": "test"},
        models_dir=train.REPO_ROOT / "models",
    )
    feedback = contract["objective"]["jepa_closed_loop_feedback"]
    assert contract["model_config"]["jepa_feedback_mode"] == (
        "final_pass_adjoint"
    )
    assert feedback["teacher_action_visible_when_masked"] is False
    assert feedback["training_dfm_ce_source"] == (
        "feedback_conditioned_logits_only"
    )
    assert feedback["inference_dfm_planner_calls"] == 8
    assert feedback["inference_additional_bt4_encodes"] == 0
    assert feedback["model_state_abi"] == "unchanged"
    assert any(
        path.endswith("research/inference.py")
        for path in contract["code"]["files"]
    )


@pytest.mark.parametrize(
    ("updates", "objective", "message"),
    [
        (
            {"jepa_feedback_mode": "unknown"},
            "normalized",
            "jepa_feedback_mode",
        ),
        (
            {
                "jepa_feedback_mode": "final_pass_adjoint",
                "jepa_rollout_mode": "direct_sequence",
            },
            "normalized",
            "requires jepa_rollout_mode='recurrent'",
        ),
        (
            {
                "jepa_feedback_mode": "final_pass_adjoint",
                "jepa_condition_dim": 12,
            },
            "normalized",
            "effective jepa_condition_dim",
        ),
        (
            {
                "jepa_feedback_mode": "final_pass_adjoint",
                "jepa_norm_loss_coeff": 1.0,
            },
            "legacy",
            "requires --objective normalized",
        ),
        (
            {
                "jepa_feedback_mode": "final_pass_adjoint",
                "horizon": 4,
            },
            "normalized",
            "frozen for horizon=8",
        ),
    ],
)
def test_invalid_feedback_configs_fail_closed(
    updates: dict[str, object],
    objective: str,
    message: str,
) -> None:
    config = dataclasses.replace(_config(feedback_mode="none"), **updates)
    with pytest.raises(ValueError, match=message):
        train.validate_objective_config(
            objective=objective,
            config=config,
        )


@dataclasses.dataclass
class _InferenceConfig:
    horizon: int = 8
    action_vocab_size: int = ACTION_VOCAB_SIZE
    jepa_feedback_mode: str = "final_pass_adjoint"


class FeedbackInferenceModel:
    def __init__(self):
        self.config = _InferenceConfig()
        self.encode_count = 0
        self.dfm_project_count = 0
        self.jepa_project_count = 0
        self.jepa_step_count = 0
        self.feedback_count = 0
        self.planner_latent_means: list[float] = []

    def encode_bt4_tokens(self, planes):
        self.encode_count += 1
        return jnp.zeros((planes.shape[0], 64, 3), dtype=jnp.float32)

    def dfm_latents(self, bt4_tokens):
        self.dfm_project_count += 1
        return jnp.zeros((bt4_tokens.shape[0], 64, 2), dtype=jnp.float32)

    def jepa_latents(self, bt4_tokens):
        self.jepa_project_count += 1
        return jnp.zeros((bt4_tokens.shape[0], 4), dtype=jnp.float32)

    def planner_from_latents(
        self,
        z_dfm,
        action_tokens,
        t,
        *,
        return_hidden=False,
    ):
        self.planner_latent_means.append(float(jnp.mean(z_dfm)))
        batch_size = z_dfm.shape[0]
        logits = jnp.full(
            (batch_size, 8, ACTION_VOCAB_SIZE),
            -8.0,
            dtype=jnp.float32,
        )
        preferred = jnp.asarray(
            [23, 31, 37, 41, 43, 47, 53, 59],
            dtype=jnp.int32,
        )
        logits = logits.at[:, jnp.arange(8), preferred].set(8.0)
        if not return_hidden:
            return logits
        return logits, {
            "action_tokens": jnp.ones(
                (batch_size, 8, 2),
                dtype=jnp.float32,
            )
        }

    def jepa_step_from_latents(
        self,
        z0_jepa,
        action,
        action_hidden,
        *,
        z0_normalized=False,
    ):
        del action, action_hidden, z0_normalized
        self.jepa_step_count += 1
        return z0_jepa + 1.0

    def dfm_latents_with_jepa_feedback(
        self,
        z_dfm,
        z0_jepa,
        z1_jepa,
        feedback_gate,
    ):
        del z0_jepa, z1_jepa, feedback_gate
        self.feedback_count += 1
        return SimpleNamespace(latents=z_dfm + 1.0)


def _inference_mask() -> jax.Array:
    mask = np.zeros((1, ACTION_VOCAB_SIZE), dtype=np.bool_)
    mask[:, [23, 29]] = True
    return jnp.asarray(mask)


def test_inference_uses_one_feedback_step_before_only_the_eighth_pass() -> None:
    model = FeedbackInferenceModel()
    result = _infer_dfm_from_current_impl(
        model,
        jnp.zeros((1, 112, 8, 8), dtype=jnp.float32),
        _inference_mask(),
        refinement_passes=8,
        trace_top_k=2,
    )

    assert model.encode_count == 1
    assert model.dfm_project_count == 1
    assert model.jepa_project_count == 1
    assert model.jepa_step_count == 1
    assert model.feedback_count == 1
    assert len(model.planner_latent_means) == 8
    np.testing.assert_array_equal(
        model.planner_latent_means[:7],
        np.zeros((7,)),
    )
    assert model.planner_latent_means[7] == 1.0
    assert int(result.actions[0, 0]) == 23

    lean_model = FeedbackInferenceModel()
    lean = _infer_dfm_actions_from_current_impl(
        lean_model,
        jnp.zeros((1, 112, 8, 8), dtype=jnp.float32),
        _inference_mask(),
        refinement_passes=8,
    )
    np.testing.assert_array_equal(lean, result.actions)
    assert lean_model.jepa_step_count == 1
    assert lean_model.feedback_count == 1
    assert lean_model.planner_latent_means[7] == 1.0


def test_feedback_inference_pass_count_and_latent_only_api_fail_closed() -> None:
    model = FeedbackInferenceModel()
    with pytest.raises(ValueError, match="exactly 8 refinement passes"):
        infer_dfm_from_current(
            model,
            np.zeros((1, 112, 8, 8), dtype=np.float32),
            np.asarray(_inference_mask()),
            refinement_passes=7,
            trace_top_k=2,
            action_codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        )
    with pytest.raises(ValueError, match="current JEPA latents"):
        refine_dfm_from_latents(
            model,
            np.zeros((1, 64, 2), dtype=np.float32),
            np.asarray(_inference_mask()),
            refinement_passes=8,
            trace_top_k=2,
            action_codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        )
