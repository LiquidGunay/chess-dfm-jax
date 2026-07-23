import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

from research.evaluate_torch_migration_parity import _jax_sigreg
from research.train import (
    ConditionedVectorTransition,
    StateVectorProjector,
    TrainableParam,
)
from research.train_torch import (
    CONFIG,
    ConditionedTransition,
    StateProjector,
    _flatten_tree,
    _sigreg_v_stat,
)


def _name_path(name: str) -> tuple[str | int, ...]:
    return tuple(
        int(part) if part.isdigit() else part for part in name.split(".")
    )


def _copy_jax_state_to_torch(
    state: nnx.State,
    model: torch.nn.Module,
) -> None:
    flat = _flatten_tree(nnx.to_pure_dict(state))
    named = dict(model.named_parameters())
    expected = {_name_path(name) for name in named}
    assert set(flat) == expected
    with torch.no_grad():
        for name, parameter in named.items():
            value = torch.from_numpy(
                np.asarray(flat[_name_path(name)], dtype=np.float32).copy()
            )
            parameter.copy_(value)


def _jax_parameter_gradients(state: nnx.State) -> dict[str, np.ndarray]:
    flat = _flatten_tree(nnx.to_pure_dict(state))
    return {
        ".".join(map(str, path)): np.asarray(value, dtype=np.float32)
        for path, value in flat.items()
    }


def _assert_gradient_maps_close(
    jax_gradients: Mapping[str, np.ndarray],
    torch_model: torch.nn.Module,
) -> None:
    torch_gradients = {
        name: parameter.grad.detach().numpy()
        for name, parameter in torch_model.named_parameters()
    }
    assert set(torch_gradients) == set(jax_gradients)
    for name in sorted(jax_gradients):
        np.testing.assert_allclose(
            torch_gradients[name],
            jax_gradients[name],
            rtol=5e-4,
            atol=3e-5,
            err_msg=name,
        )


def _projector_pair() -> tuple[StateVectorProjector, StateProjector]:
    config = dataclasses.replace(
        CONFIG,
        z_dim=32,
        projector_layers=2,
        projector_heads=4,
        projector_mlp_dim=48,
        use_qk_norm=True,
        use_xsa=True,
        remat_blocks=False,
    )
    jax_model = StateVectorProjector(
        1024,
        config.z_dim,
        num_layers=config.projector_layers,
        active_layers=config.projector_layers,
        num_heads=config.projector_heads,
        mlp_dim=config.projector_mlp_dim,
        rngs=nnx.Rngs(11),
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        use_qk_gain=False,
        use_qk_norm=config.use_qk_norm,
        use_xsa=config.use_xsa,
        scan_layers=False,
        remat_blocks=False,
    )
    torch_model = StateProjector(config)
    _copy_jax_state_to_torch(
        nnx.state(jax_model, TrainableParam),
        torch_model,
    )
    return jax_model, torch_model


def test_projector_full_parameter_and_input_gradient_parity():
    jax_model, torch_model = _projector_pair()
    rng = np.random.default_rng(19)
    inputs = rng.standard_normal((2, 64, 1024), dtype=np.float32) * 0.1
    cotangent = rng.standard_normal((2, 32), dtype=np.float32)
    graph, state = nnx.split(jax_model, TrainableParam)

    def jax_objective(candidate_state: nnx.State, value: jax.Array):
        candidate = nnx.merge(graph, candidate_state)
        output = candidate(value)
        return jnp.sum(output * jnp.asarray(cotangent))

    jax_loss, (jax_state_grad, jax_input_grad) = jax.value_and_grad(
        jax_objective,
        argnums=(0, 1),
    )(state, jnp.asarray(inputs))

    torch_input = torch.tensor(inputs, requires_grad=True)
    torch_output = torch_model(torch_input, torch.float32)
    torch_loss = (torch_output * torch.from_numpy(cotangent)).sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        torch_output.detach().numpy(),
        np.asarray(nnx.merge(graph, state)(jnp.asarray(inputs))),
        rtol=5e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        float(torch_loss.detach()),
        float(jax_loss),
        rtol=5e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        torch_input.grad.numpy(),
        np.asarray(jax_input_grad),
        rtol=5e-4,
        atol=3e-5,
    )
    _assert_gradient_maps_close(
        _jax_parameter_gradients(jax_state_grad),
        torch_model,
    )


def _transition_pair() -> tuple[
    ConditionedVectorTransition,
    ConditionedTransition,
]:
    config = dataclasses.replace(
        CONFIG,
        z_dim=32,
        jepa_layers=2,
        jepa_mlp_dim=48,
        delta_rms_clip=0.5,
    )
    jax_model = ConditionedVectorTransition(
        config.z_dim,
        config.z_dim,
        num_layers=config.jepa_layers,
        mlp_dim=config.jepa_mlp_dim,
        rngs=nnx.Rngs(23),
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        init_scale=1e-3,
        delta_rms_clip=config.delta_rms_clip,
    )
    torch_model = ConditionedTransition(config)
    _copy_jax_state_to_torch(
        nnx.state(jax_model, TrainableParam),
        torch_model,
    )
    return jax_model, torch_model


def test_recurrent_transition_full_parameter_and_input_gradient_parity():
    jax_model, torch_model = _transition_pair()
    rng = np.random.default_rng(29)
    z = rng.standard_normal((3, 32), dtype=np.float32) * 0.2
    condition = rng.standard_normal((3, 32), dtype=np.float32) * 0.15
    cotangent = rng.standard_normal((3, 32), dtype=np.float32)
    graph, state = nnx.split(jax_model, TrainableParam)

    def jax_objective(
        candidate_state: nnx.State,
        z_value: jax.Array,
        condition_value: jax.Array,
    ):
        candidate = nnx.merge(graph, candidate_state)
        output = candidate(z_value, condition_value)
        return jnp.sum(output * jnp.asarray(cotangent))

    jax_loss, (jax_state_grad, jax_z_grad, jax_condition_grad) = (
        jax.value_and_grad(
            jax_objective,
            argnums=(0, 1, 2),
        )(state, jnp.asarray(z), jnp.asarray(condition))
    )

    torch_z = torch.tensor(z, requires_grad=True)
    torch_condition = torch.tensor(condition, requires_grad=True)
    torch_output = torch_model(
        torch_z,
        torch_condition,
        torch.float32,
    )
    torch_loss = (torch_output * torch.from_numpy(cotangent)).sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        float(torch_loss.detach()),
        float(jax_loss),
        rtol=5e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        torch_z.grad.numpy(),
        np.asarray(jax_z_grad),
        rtol=5e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        torch_condition.grad.numpy(),
        np.asarray(jax_condition_grad),
        rtol=5e-4,
        atol=3e-5,
    )
    _assert_gradient_maps_close(
        _jax_parameter_gradients(jax_state_grad),
        torch_model,
    )


def test_sigreg_value_and_latent_gradient_parity():
    rng = np.random.default_rng(31)
    z = rng.standard_normal((48, 32), dtype=np.float32)
    weights = rng.random(48, dtype=np.float32)
    weights[::7] = 0.0
    directions = rng.standard_normal((32, 24), dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)

    jax_value, jax_gradient = jax.value_and_grad(_jax_sigreg)(
        jnp.asarray(z),
        jnp.asarray(weights),
        jnp.asarray(directions),
    )
    torch_z = torch.tensor(z, requires_grad=True)
    torch_value, torch_count = _sigreg_v_stat(
        torch_z,
        torch.from_numpy(weights),
        torch.from_numpy(directions),
        reference_count=1.0,
    )
    torch_value.backward()

    np.testing.assert_allclose(
        float(torch_value.detach()),
        float(jax_value),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        float(torch_count),
        float(weights.sum()),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        torch_z.grad.numpy(),
        np.asarray(jax_gradient),
        rtol=2e-4,
        atol=2e-6,
    )
