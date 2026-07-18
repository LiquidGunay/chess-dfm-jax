from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from research.train import (
    latent_collapse_diagnostics,
    legal_mass_fp32,
    normalized_le_jepa_sigreg,
    should_continue,
)


def _loss(z: jax.Array) -> jax.Array:
    return normalized_le_jepa_sigreg(
        z,
        proj_dim=32,
        rng=jax.random.PRNGKey(17),
        reference_count=128.0,
    ).normalized


def test_normalized_sigreg_is_invariant_to_exact_duplication() -> None:
    z = jax.random.normal(jax.random.PRNGKey(1), (7, 11))
    base = normalized_le_jepa_sigreg(
        z,
        proj_dim=32,
        rng=jax.random.PRNGKey(17),
        reference_count=128.0,
    )
    repeated = normalized_le_jepa_sigreg(
        jnp.repeat(z, 2, axis=0),
        proj_dim=32,
        rng=jax.random.PRNGKey(17),
        reference_count=128.0,
    )

    np.testing.assert_allclose(base.normalized, repeated.normalized, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(repeated.official, 2.0 * base.official, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(repeated.valid_count, 2.0 * base.valid_count)


def test_normalized_sigreg_gradient_is_invariant_to_exact_duplication() -> None:
    z = jax.random.normal(jax.random.PRNGKey(2), (7, 11))
    base_grad = jax.grad(_loss)(z)
    repeated_grad = jax.grad(lambda value: _loss(jnp.repeat(value, 2, axis=0)))(z)
    np.testing.assert_allclose(base_grad, repeated_grad, rtol=2e-5, atol=2e-6)


def test_normalized_sigreg_ignores_zero_weight_padding() -> None:
    z = jax.random.normal(jax.random.PRNGKey(3), (5, 9))
    padding = jax.random.normal(jax.random.PRNGKey(4), (4, 9)) * 100.0
    base = normalized_le_jepa_sigreg(
        z,
        proj_dim=16,
        rng=jax.random.PRNGKey(5),
        reference_count=64.0,
    )
    padded = normalized_le_jepa_sigreg(
        jnp.concatenate([z, padding], axis=0),
        proj_dim=16,
        rng=jax.random.PRNGKey(5),
        reference_count=64.0,
        sample_weight=jnp.concatenate([jnp.ones((5,)), jnp.zeros((4,))]),
    )
    np.testing.assert_allclose(base.normalized, padded.normalized, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(base.official, padded.official, rtol=1e-6, atol=1e-6)


def test_legal_mass_is_fp32_and_bounded() -> None:
    probs = jnp.asarray([[0.6, 0.6, 0.0]], dtype=jnp.bfloat16)
    legal_idx = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    legal_count = jnp.asarray([2], dtype=jnp.int32)
    mass = legal_mass_fp32(probs, legal_idx, legal_count)
    assert mass.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(mass), np.asarray([1.0], dtype=np.float32))


def test_time_budget_runs_a_compile_step_before_deadline_is_set() -> None:
    assert should_continue(updates=0, steps=0, deadline=float("inf"))
    assert not should_continue(updates=1, steps=0, deadline=0.0)


def test_per_horizon_diagnostics_expose_one_collapsed_horizon() -> None:
    batch, horizon, dim = 32, 3, 12
    target = jax.random.normal(jax.random.PRNGKey(20), (batch, horizon, dim))
    pred = target + 0.05 * jax.random.normal(jax.random.PRNGKey(21), target.shape)
    pred = pred.at[:, 1, :].set(jnp.zeros((batch, dim)))
    current = jax.random.normal(jax.random.PRNGKey(22), (batch, dim))
    valid = jnp.ones((batch, horizon))

    metrics = latent_collapse_diagnostics(
        pred,
        target,
        current,
        valid,
        rng=jax.random.PRNGKey(23),
    )
    pred_std = np.asarray(metrics["pred_feature_std_mean_by_horizon"])
    effective_rank = np.asarray(metrics["pred_effective_rank_by_horizon"])

    assert pred_std[0] > 0.5
    assert pred_std[1] < 1e-5
    assert pred_std[2] > 0.5
    assert effective_rank[0] > 2.0
    assert effective_rank[1] < 1e-4
    assert effective_rank[2] > 2.0
