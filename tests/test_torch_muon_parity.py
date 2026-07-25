import dataclasses

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from chess_dfm_jax.nnx_bt4 import muon_adamw
from research.train_torch import (
    CONFIG,
    MuonAdamW,
    _prepare_training_step,
    _summarize_training_records,
    load_model_checkpoint,
    load_model_checkpoint_numpy_tree,
    load_training_checkpoint,
    save_model_checkpoint,
    save_training_checkpoint,
)


class TinyModel(torch.nn.Module):
    def __init__(self, matrix: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.from_numpy(matrix.copy()))
        self.bias = torch.nn.Parameter(torch.from_numpy(bias.copy()))


def test_torch_muon_adamw_matches_optax_for_two_updates():
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((128, 128), dtype=np.float32) * 0.02
    bias = rng.standard_normal((128,), dtype=np.float32) * 0.02
    gradient_rows = [
        {
            "matrix": rng.standard_normal((128, 128), dtype=np.float32) * 0.01,
            "bias": rng.standard_normal((128,), dtype=np.float32) * 0.01,
        }
        for _ in range(2)
    ]
    learning_rate = 3e-3
    weight_decay = 1e-4
    config = dataclasses.replace(
        CONFIG,
        learning_rate=learning_rate,
        bt4_learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=0.0,
    )

    torch_model = TinyModel(matrix, bias)
    torch_optimizer = MuonAdamW(torch_model, config)
    jax_params = {"matrix": jnp.asarray(matrix), "bias": jnp.asarray(bias)}
    transform = muon_adamw(learning_rate, weight_decay)
    jax_state = transform.init(jax_params)

    for gradients in gradient_rows:
        torch_model.matrix.grad = torch.from_numpy(gradients["matrix"].copy())
        torch_model.bias.grad = torch.from_numpy(gradients["bias"].copy())
        torch_optimizer.step()

        jax_gradients = jax.tree.map(jnp.asarray, gradients)
        updates, jax_state = transform.update(jax_gradients, jax_state, jax_params)
        jax_params = jax.tree.map(
            lambda parameter, update: parameter + update,
            jax_params,
            updates,
        )

        np.testing.assert_allclose(
            torch_model.matrix.detach().numpy(),
            np.asarray(jax_params["matrix"]),
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            torch_model.bias.detach().numpy(),
            np.asarray(jax_params["bias"]),
            rtol=2e-5,
            atol=2e-6,
        )


def test_torch_optimizer_partition_and_schedule_contract():
    with torch.device("meta"):
        from research.train_torch import JointModel

        model = JointModel(CONFIG)
    optimizer = MuonAdamW(model, CONFIG)
    manifest = optimizer.partition_manifest()

    assert manifest["leaf_count"] == 455
    assert manifest["bt4_leaf_count"] == 404
    assert manifest["main_leaf_count"] == 51
    assert manifest["muon_leaf_count"] > 0
    assert manifest["adamw_leaf_count"] > 0
    assert optimizer.learning_rate_ratio() == 1.0
    optimizer.update = 400
    assert optimizer.learning_rate_ratio() == 1.0
    optimizer.update = 800
    assert optimizer.learning_rate_ratio() == 0.55
    optimizer.update = 1200
    assert optimizer.learning_rate_ratio() == 0.1


def test_torch_optimizer_skips_entire_nonfinite_update():
    matrix = np.zeros((128, 128), dtype=np.float32)
    bias = np.zeros((128,), dtype=np.float32)
    model = TinyModel(matrix, bias)
    optimizer = MuonAdamW(model, CONFIG)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    model.matrix.grad = torch.ones_like(model.matrix)
    model.bias.grad = torch.full_like(model.bias, float("nan"))
    metrics = optimizer.step()

    assert metrics["optimizer_skipped_nonfinite"] is True
    assert optimizer.update == 0
    assert model.matrix.grad is None
    assert model.bias.grad is None
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)
    for leaf in optimizer.leaves:
        assert torch.count_nonzero(leaf.first_moment) == 0
        if leaf.second_moment is not None:
            assert torch.count_nonzero(leaf.second_moment) == 0


def test_torch_model_checkpoint_is_strict_model_only_roundtrip(tmp_path):
    matrix = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    bias = np.arange(128, dtype=np.float32)
    model = TinyModel(matrix, bias)
    expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    manifest = save_model_checkpoint(
        output_dir=tmp_path,
        model=model,
        source_mapping_sha256="a" * 64,
        optimizer_update=7,
        data_cursor=11,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "manifest.json",
        "model.safetensors",
    ]
    assert manifest["model_only"] is True
    assert manifest["optimizer_resume_supported"] is False
    assert manifest["optimizer_update"] == 7
    assert manifest["data_cursor"] == 11
    assert manifest["state"]["leaf_count"] == 2

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored = load_model_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
    )
    tree, tree_manifest, tree_summary = load_model_checkpoint_numpy_tree(
        checkpoint_dir=checkpoint_dir,
        model=model,
    )

    assert restored == manifest
    assert tree_manifest == manifest
    assert tree_summary["leaf_count"] == 2
    assert tree_summary["nbytes"] == matrix.nbytes + bias.nbytes
    np.testing.assert_array_equal(tree["matrix"], matrix)
    np.testing.assert_array_equal(tree["bias"], bias)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            expected[name],
            rtol=0.0,
            atol=0.0,
        )


def test_torch_checkpoint_numpy_tree_preserves_bfloat16_bits(tmp_path):
    model = torch.nn.Module()
    expected = torch.tensor(
        [0.0, -1.5, 3.25, float("inf")],
        dtype=torch.bfloat16,
    )
    model.register_parameter(
        "bf16",
        torch.nn.Parameter(expected.clone()),
    )
    save_model_checkpoint(
        output_dir=tmp_path,
        model=model,
        source_mapping_sha256="b" * 64,
        optimizer_update=0,
        data_cursor=0,
    )

    tree, _, summary = load_model_checkpoint_numpy_tree(
        checkpoint_dir=tmp_path / "checkpoint",
        model=model,
    )

    assert tree["bf16"].dtype == ml_dtypes.bfloat16
    assert summary["nbytes"] == expected.numel() * expected.element_size()
    np.testing.assert_array_equal(
        tree["bf16"].view(np.uint16),
        expected.view(torch.uint16).numpy(),
    )


def test_torch_training_checkpoint_resumes_optimizer_exactly(tmp_path):
    rng = np.random.default_rng(41)
    matrix = rng.standard_normal((128, 128), dtype=np.float32) * 0.02
    bias = rng.standard_normal((128,), dtype=np.float32) * 0.02
    gradients = [
        (
            rng.standard_normal((128, 128), dtype=np.float32) * 0.01,
            rng.standard_normal((128,), dtype=np.float32) * 0.01,
        )
        for _ in range(3)
    ]
    config = dataclasses.replace(
        CONFIG,
        learning_rate=2e-3,
        bt4_learning_rate=2e-3,
        grad_clip_norm=0.0,
    )
    control_model = TinyModel(matrix, bias)
    control_optimizer = MuonAdamW(
        control_model,
        config,
        examples_per_update=8,
    )
    staged_model = TinyModel(matrix, bias)
    staged_optimizer = MuonAdamW(
        staged_model,
        config,
        examples_per_update=8,
    )

    def step(
        model: TinyModel,
        optimizer: MuonAdamW,
        gradient: tuple[np.ndarray, np.ndarray],
    ) -> None:
        model.matrix.grad = torch.from_numpy(gradient[0].copy())
        model.bias.grad = torch.from_numpy(gradient[1].copy())
        optimizer.step()

    for gradient in gradients[:2]:
        step(control_model, control_optimizer, gradient)
        step(staged_model, staged_optimizer, gradient)

    resume_contract = {
        "schema_version": "unit-test-resume-v1",
        "batch_size": 8,
    }
    source_sha256 = "c" * 64
    manifest = save_training_checkpoint(
        output_dir=tmp_path,
        model=staged_model,
        optimizer=staged_optimizer,
        source_mapping_sha256=source_sha256,
        next_data_cursor=19,
        resume_contract=resume_contract,
    )
    checkpoint_dir = tmp_path / "checkpoints" / "update00000002"
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "manifest.json",
        "state.safetensors",
    ]
    assert manifest["model_only"] is False
    assert manifest["optimizer_resume_supported"] is True
    assert manifest["optimizer_update"] == 2
    assert manifest["optimizer_examples_seen"] == 16
    assert manifest["next_data_cursor"] == 19

    resumed_model = TinyModel(
        np.zeros_like(matrix),
        np.zeros_like(bias),
    )
    resumed_optimizer = MuonAdamW(
        resumed_model,
        config,
        examples_per_update=8,
    )
    restored = load_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_source_mapping_sha256=source_sha256,
        expected_resume_contract=resume_contract,
    )
    assert restored == manifest
    assert resumed_optimizer.update == 2
    assert resumed_optimizer.examples_seen == 16
    with pytest.raises(ValueError, match="resume contract"):
        load_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=resumed_model,
            optimizer=resumed_optimizer,
            expected_source_mapping_sha256=source_sha256,
            expected_resume_contract={**resume_contract, "batch_size": 16},
        )

    step(control_model, control_optimizer, gradients[2])
    step(resumed_model, resumed_optimizer, gradients[2])
    for (control_name, control_parameter), (resumed_name, resumed_parameter) in zip(
        control_model.named_parameters(),
        resumed_model.named_parameters(),
        strict=True,
    ):
        assert control_name == resumed_name
        torch.testing.assert_close(
            control_parameter,
            resumed_parameter,
            rtol=0.0,
            atol=0.0,
        )
    for control_leaf, resumed_leaf in zip(
        control_optimizer.leaves,
        resumed_optimizer.leaves,
        strict=True,
    ):
        assert control_leaf.name == resumed_leaf.name
        torch.testing.assert_close(
            control_leaf.first_moment,
            resumed_leaf.first_moment,
            rtol=0.0,
            atol=0.0,
        )
        if control_leaf.second_moment is None:
            assert resumed_leaf.second_moment is None
        else:
            assert resumed_leaf.second_moment is not None
            torch.testing.assert_close(
                control_leaf.second_moment,
                resumed_leaf.second_moment,
                rtol=0.0,
                atol=0.0,
            )


def test_prefetch_preparation_is_schedule_keyed_and_deterministic():
    class FakeBatches:
        def __init__(self):
            self.calls: list[int] = []

        def batch_at(self, cursor: int):
            self.calls.append(cursor)
            future = np.arange(64 * 8, dtype=np.int64).reshape(64, 8, 1)
            return {
                "current_planes": np.full((64, 1), cursor, dtype=np.float32),
                "future_planes": future,
            }

    batches = FakeBatches()
    first = _prepare_training_step(
        batches,
        seed=17,
        update=3,
        data_cursor=7,
        batch_size=64,
    )
    second = _prepare_training_step(
        batches,
        seed=17,
        update=3,
        data_cursor=7,
        batch_size=64,
    )

    assert first.update == second.update == 3
    assert first.data_cursor == second.data_cursor == 7
    assert batches.calls == [7, 7]
    for left, right in zip(first.choices, second.choices, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    rows = np.arange(64)
    expected = np.arange(64 * 8, dtype=np.int64).reshape(64, 8, 1)[
        rows,
        first.choices.target_horizon.numpy(),
    ]
    np.testing.assert_array_equal(
        first.compact_batch["selected_future_planes"],
        expected,
    )
    assert first.prepare_seconds >= 0.0


def test_training_loss_summary_preserves_terminal_and_fixed_window_metrics():
    metric_names = (
        "loss",
        "unclipped_loss",
        "dfm_ce_loss",
        "accuracy",
        "first_legality_loss",
        "weighted_legality_loss",
        "first_legal_mass",
        "jepa_positive_loss",
        "jepa_sigreg_loss",
        "jepa_pred_sigreg_loss",
        "z_state_norm",
        "z_pred_norm",
        "z_target_norm",
        "learning_rate",
        "bt4_learning_rate",
        "gradient_global_norm",
        "gradient_clip_scale",
    )
    records = [
        {
            "update": update,
            "examples": update * 512,
            **{
                name: float(update + metric_index)
                for metric_index, name in enumerate(metric_names)
            },
        }
        for update in range(1, 6)
    ]

    summary = _summarize_training_records(records, window_updates=2)

    assert summary["schema_version"] == "torch-eager-loss-summary-v2"
    assert summary["window_updates"] == 2
    assert summary["terminal"]["update"] == 5
    assert summary["terminal"]["examples"] == 2560
    assert summary["terminal"]["loss"] == 5.0
    assert summary["first_window"]["updates"] == [1, 2]
    assert summary["first_window"]["loss"] == 1.5
    assert summary["last_window"]["updates"] == [4, 5]
    assert summary["last_window"]["loss"] == 4.5
    assert summary["last_minus_first"]["loss"] == 3.0
    assert (
        summary["plot_contract"]["primary_cross_experiment_metric"]
        == "matched mean validation dfm_ce_loss over frozen seeds 10000 and 20000"
    )
