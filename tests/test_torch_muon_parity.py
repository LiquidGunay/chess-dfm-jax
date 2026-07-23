import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import torch

from chess_dfm_jax.nnx_bt4 import muon_adamw
from research.train_torch import CONFIG, MuonAdamW


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
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

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
