from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.policy import (  # noqa: E402
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
)
from chess_dfm_jax.training.dfm import (  # noqa: E402
    refine_actions_from_latents as legacy_refine_actions_from_latents,
)
from research.inference import (  # noqa: E402
    _infer_dfm_from_current_impl,
    _refine_dfm_from_latents_impl,
    infer_dfm_from_current,
    root_topk_turnover,
    validate_root_legal_mask,
)


@dataclasses.dataclass
class _Config:
    horizon: int = 4
    action_vocab_size: int = ACTION_VOCAB_SIZE


class _DummyModel:
    def __init__(self):
        self.config = _Config()
        self.encode_count = 0
        self.project_count = 0
        self.planner_count = 0

    def encode_bt4_tokens(self, planes):
        self.encode_count += 1
        return jnp.zeros((planes.shape[0], 64, 3), dtype=jnp.float32)

    def dfm_latents(self, bt4_tokens):
        self.project_count += 1
        return jnp.zeros(
            (bt4_tokens.shape[0], bt4_tokens.shape[1], 2),
            dtype=jnp.float32,
        )

    def planner_from_latents(self, z_dfm, action_tokens, t):
        self.planner_count += 1
        batch_size = z_dfm.shape[0]
        logits = jnp.full(
            (
                batch_size,
                self.config.horizon,
                self.config.action_vocab_size,
            ),
            -8.0,
            dtype=jnp.float32,
        )
        preferred = jnp.asarray([7, 11, 13, 17], dtype=jnp.int32)
        logits = logits.at[:, jnp.arange(self.config.horizon), preferred].set(8.0)
        # Make the preferred root action vary by pass. The strict legal mask in
        # the test excludes both, so neither can leak into the result.
        raw_root = jnp.where(t < 0.5, 7, 11)
        logits = logits.at[:, 0, :].set(-8.0)
        return logits.at[jnp.arange(batch_size), 0, raw_root].set(8.0)


class _CompiledDummyModel(nnx.Module):
    def __init__(self):
        self.config = _Config()

    def encode_bt4_tokens(self, planes):
        return jnp.zeros((planes.shape[0], 64, 3), dtype=jnp.float32)

    def dfm_latents(self, bt4_tokens):
        return jnp.zeros(
            (bt4_tokens.shape[0], bt4_tokens.shape[1], 2),
            dtype=jnp.float32,
        )

    def planner_from_latents(self, z_dfm, action_tokens, t):
        batch_size = z_dfm.shape[0]
        logits = jnp.full(
            (
                batch_size,
                self.config.horizon,
                self.config.action_vocab_size,
            ),
            -8.0,
            dtype=jnp.float32,
        )
        preferred = jnp.asarray([7, 11, 13, 17], dtype=jnp.int32)
        return logits.at[:, jnp.arange(self.config.horizon), preferred].set(8.0)


class _NoTieOracleModel:
    def __init__(self):
        self.config = _Config()

    def planner_from_latents(self, z_dfm, action_tokens, t):
        batch_size = z_dfm.shape[0]
        logits = jnp.zeros(
            (
                batch_size,
                self.config.horizon,
                self.config.action_vocab_size,
            ),
            dtype=jnp.float32,
        )
        preferred = jnp.asarray([100, 200, 300, 400], dtype=jnp.int32)
        peak_logits = jnp.asarray([5.0, 4.0, 3.0, 2.0], dtype=jnp.float32)
        return logits.at[:, jnp.arange(self.config.horizon), preferred].set(peak_logits)


def _legal_mask(batch_size: int = 2) -> np.ndarray:
    mask = np.zeros((batch_size, ACTION_VOCAB_SIZE), dtype=np.bool_)
    mask[:, 23] = True
    mask[:, 29] = True
    return mask


def test_eager_kernel_encodes_once_and_records_strict_refinement_trace():
    model = _DummyModel()
    result = _infer_dfm_from_current_impl(
        model,
        jnp.zeros((2, 112, 8, 8), dtype=jnp.float32),
        jnp.asarray(_legal_mask()),
        refinement_passes=4,
        trace_top_k=3,
    )

    assert model.encode_count == 1
    assert model.project_count == 1
    assert model.planner_count == 4
    assert result.actions.shape == (2, 4)
    assert result.trace.actions_before.shape == (4, 2, 4)
    assert result.trace.actions_after.shape == (4, 2, 4)
    assert result.trace.root_legal_topk_indices.shape == (4, 2, 3)
    np.testing.assert_array_equal(
        np.asarray(result.trace.times),
        np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float32),
    )
    assert np.all(np.isin(np.asarray(result.actions[:, 0]), [23, 29]))
    assert np.all(np.asarray(result.trace.root_raw_legal_mass) < 1e-4)
    np.testing.assert_array_equal(
        np.asarray(result.trace.root_legal_topk_valid[..., :2]),
        np.ones((4, 2, 2), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        np.asarray(result.trace.root_legal_topk_valid[..., 2]),
        np.zeros((4, 2), dtype=np.bool_),
    )


def test_compiled_public_path_runs_with_more_passes_than_horizon():
    model = _CompiledDummyModel()
    result = infer_dfm_from_current(
        model,
        np.zeros((1, 112, 8, 8), dtype=np.float32),
        _legal_mask(batch_size=1),
        refinement_passes=6,
        trace_top_k=2,
        action_codec_id=ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    )
    result = jax.block_until_ready(result)

    assert result.actions.shape == (1, 4)
    assert not np.any(np.asarray(result.actions) == model.config.action_vocab_size)
    assert result.trace.actions_after.shape == (6, 1, 4)


def test_final_actions_match_legacy_threshold_sampler_without_ties():
    model = _NoTieOracleModel()
    latents = jnp.zeros((2, 64, 2), dtype=jnp.float32)
    legal_mask = np.zeros(
        (2, ACTION_VOCAB_SIZE),
        dtype=np.bool_,
    )
    legal_mask[:, [100, 101, 102]] = True

    local_result = _refine_dfm_from_latents_impl(
        model,
        latents,
        jnp.asarray(legal_mask),
        refinement_passes=4,
        trace_top_k=2,
    )
    legacy_actions = legacy_refine_actions_from_latents(
        model,
        latents,
        jnp.asarray(legal_mask),
        refinement_steps=4,
    )

    np.testing.assert_array_equal(
        np.asarray(local_result.actions),
        np.asarray(legacy_actions),
    )


@pytest.mark.parametrize(
    ("mask", "codec_id", "error", "match"),
    [
        (
            np.ones((1, ACTION_VOCAB_SIZE), dtype=np.float32),
            ACTION_CODEC_LEGACY_ABSOLUTE_1858,
            TypeError,
            "boolean dtype",
        ),
        (
            np.zeros((1, ACTION_VOCAB_SIZE), dtype=np.bool_),
            ACTION_CODEC_LEGACY_ABSOLUTE_1858,
            ValueError,
            "no legal actions",
        ),
        (
            np.ones((1, ACTION_VOCAB_SIZE - 1), dtype=np.bool_),
            ACTION_CODEC_LEGACY_ABSOLUTE_1858,
            ValueError,
            "must have shape",
        ),
        (
            np.ones((1, ACTION_VOCAB_SIZE), dtype=np.bool_),
            "lc0_canonical_1858",
            ValueError,
            "requires action_codec_id",
        ),
    ],
)
def test_root_legal_mask_validation_fails_closed(
    mask,
    codec_id,
    error,
    match,
):
    with pytest.raises(error, match=match):
        validate_root_legal_mask(
            mask,
            batch_size=1,
            action_vocab_size=ACTION_VOCAB_SIZE,
            action_codec_id=codec_id,
        )


def test_topk_turnover_ignores_invalid_padding_entries():
    model = _DummyModel()
    result = _infer_dfm_from_current_impl(
        model,
        jnp.zeros((1, 112, 8, 8), dtype=jnp.float32),
        jnp.asarray(_legal_mask(batch_size=1)),
        refinement_passes=3,
        trace_top_k=4,
    )

    turnover = root_topk_turnover(result.trace)
    assert turnover.shape == (2, 1)
    np.testing.assert_allclose(np.asarray(turnover), 0.0)
