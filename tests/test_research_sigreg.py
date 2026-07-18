from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.train import (
    UNEVALUATED_LEGACY_AUX_METRICS,
    compiler_performance,
    flatten_metrics,
    gradient_group_for_path,
    latent_collapse_diagnostics,
    legal_mass_fp32,
    normalize_memory_analysis,
    normalized_le_jepa_sigreg,
    reportable_stage1_aux,
    reconstruct_polarized_grams,
    should_continue,
    summarize_gpu_samples,
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


def test_unevaluated_legacy_aux_metrics_are_explicitly_filtered() -> None:
    expected = frozenset(
        {
            "horizon_legality_loss",
            "horizon_legal_mass",
            "horizon_legality_evaluated",
            "jepa_cosine_loss",
            "jepa_normalized_mse",
            "mean_token_cosine",
            "mean_token_cosine_by_horizon",
            "pred_token_norm",
            "target_token_norm",
            "identity_jepa_loss",
            "identity_jepa_cosine_loss",
            "identity_mean_token_cosine",
            "jepa_shuffled_loss",
            "jepa_shuffled_cosine_loss",
            "jepa_shuffled_mean_token_cosine",
            "jepa_action_contrast_loss",
            "jepa_true_minus_shuffled",
            "jepa_loss_minus_identity",
        }
    )
    assert UNEVALUATED_LEGACY_AUX_METRICS == expected

    aux = {
        key: (
            jnp.zeros((2,), dtype=jnp.float32)
            if key.endswith("_by_horizon")
            else jnp.asarray(0.0, dtype=jnp.float32)
        )
        for key in expected
    }
    aux["dfm_ce_loss"] = jnp.asarray(1.25, dtype=jnp.float32)
    reportable = reportable_stage1_aux(aux)
    assert reportable == {"dfm_ce_loss": aux["dfm_ce_loss"]}
    assert flatten_metrics(reportable) == {"dfm_ce_loss": 1.25}


def test_unevaluated_aux_filter_fails_closed_if_contract_drifts() -> None:
    aux = {
        key: jnp.asarray(0.0, dtype=jnp.float32)
        for key in UNEVALUATED_LEGACY_AUX_METRICS
    }
    missing = dict(aux)
    del missing["identity_jepa_loss"]
    with pytest.raises(KeyError, match="identity_jepa_loss"):
        reportable_stage1_aux(missing)

    nonzero = dict(aux)
    nonzero["jepa_action_contrast_loss"] = jnp.asarray(
        0.5,
        dtype=jnp.float32,
    )
    with pytest.raises(
        ValueError,
        match="jepa_action_contrast_loss",
    ):
        reportable_stage1_aux(nonzero)


def test_time_budget_runs_a_compile_step_before_deadline_is_set() -> None:
    assert should_continue(updates=0, steps=0, deadline=float("inf"))
    assert not should_continue(updates=1, steps=0, deadline=0.0)


def test_slot_based_compiler_memory_stats_are_normalized() -> None:
    class MemoryStats:
        __slots__ = ("argument_size_in_bytes", "temp_size_in_bytes", "ignored")

        def __init__(self) -> None:
            self.argument_size_in_bytes = 12
            self.temp_size_in_bytes = 34
            self.ignored = 56

    assert normalize_memory_analysis(MemoryStats()) == {
        "argument_size_in_bytes": 12,
        "temp_size_in_bytes": 34,
    }


def test_compiler_rates_are_explicit_estimates() -> None:
    metrics = compiler_performance(
        {"flops": 4.0e12, "bytes accessed": 2.0e9},
        0.5,
    )
    assert metrics["compiler_estimated_achieved_tflops"] == 8.0
    assert metrics["compiler_estimated_achieved_gbps"] == 4.0
    assert metrics["compiler_estimated_arithmetic_intensity"] == 2000.0


def test_gpu_monitor_summary_ignores_malformed_rows(tmp_path) -> None:
    samples = tmp_path / "gpu_samples.csv"
    samples.write_text(
        "2026/07/18 00:00:00.000, 25, 10, 1000, 23028, 75, 1500, 6251\n"
        "malformed\n"
        "2026/07/18 00:00:00.100, 75, 20, 2000, 23028, 125, 1700, 6251\n",
        encoding="utf-8",
    )
    summary = summarize_gpu_samples(samples)
    assert summary["sample_count"] == 2
    assert summary["gpu_utilization_percent_mean"] == 50.0
    assert summary["memory_used_mib_max"] == 2000.0
    assert summary["power_watts_p50"] == 100.0


def test_gradient_parameter_groups_follow_model_roots() -> None:
    value = jax.tree_util.GetAttrKey("value")
    assert (
        gradient_group_for_path((jax.tree_util.DictKey("encoder"), value))
        == "backbone"
    )
    assert (
        gradient_group_for_path((jax.tree_util.DictKey("dfm_blocks"), value))
        == "dfm"
    )
    assert (
        gradient_group_for_path((jax.tree_util.DictKey("jepa_transition"), value))
        == "jepa"
    )
    assert (
        gradient_group_for_path((jax.tree_util.DictKey("value_wdl_head"), value))
        == "other"
    )


def test_polarization_reconstructs_group_gradient_grams() -> None:
    rng = np.random.default_rng(9)
    gradients = rng.normal(size=(3, 4, 7))
    expected_groups = np.einsum("gik,gjk->gij", gradients, gradients)
    diagonal_q = np.stack(
        [np.diag(expected_groups[group]) for group in range(3)],
        axis=1,
    )
    scales = 1.0 / np.sqrt(np.sum(diagonal_q, axis=1))
    pair_q = []
    for left in range(4):
        for right in range(left + 1, 4):
            pair_directions = []
            for sign in (1.0, -1.0):
                weights = np.zeros((4,))
                weights[left] = scales[left]
                weights[right] = sign * scales[right]
                pair_directions.append(
                    np.einsum(
                        "i,gij,j->g",
                        weights,
                        expected_groups,
                        weights,
                    )
                )
            pair_q.append(pair_directions)
    observed = reconstruct_polarized_grams(
        diagonal_q,
        np.asarray(pair_q),
        scales,
    )
    np.testing.assert_allclose(observed[:-1], expected_groups, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        observed[-1],
        np.sum(expected_groups, axis=0),
        rtol=1e-12,
        atol=1e-12,
    )


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
