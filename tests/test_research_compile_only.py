from __future__ import annotations

import argparse
import copy
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from research import train


def valid_compile_only_args() -> argparse.Namespace:
    return train.parse_args(
        [
            "--compile-only",
            "--compile-ahead",
            "--init",
            "model-only",
            "--objective",
            "normalized",
            "--steps",
            "0",
            "--eval-batches",
            "0",
            "--no-save-final",
        ]
    )


def test_compile_only_cli_contract_accepts_only_zero_execution_mode() -> None:
    args = valid_compile_only_args()
    train.validate_compile_only_args(args, save_updates=())

    invalid = copy.copy(args)
    invalid.compile_ahead = False
    with pytest.raises(ValueError, match="requires --compile-ahead"):
        train.validate_compile_only_args(invalid, save_updates=())

    invalid = copy.copy(args)
    invalid.steps = 1
    with pytest.raises(ValueError, match="requires --steps 0"):
        train.validate_compile_only_args(invalid, save_updates=())

    invalid = copy.copy(args)
    invalid.eval_batches = 1
    with pytest.raises(ValueError, match="requires --eval-batches 0"):
        train.validate_compile_only_args(invalid, save_updates=())

    with pytest.raises(ValueError, match="forbids every checkpoint"):
        train.validate_compile_only_args(args, save_updates=(1,))


def test_split_compile_only_requires_exactly_one_component() -> None:
    args = valid_compile_only_args()
    args.gradient_execution = "split"
    with pytest.raises(ValueError, match="requires --compile-component"):
        train.validate_compile_only_args(args, save_updates=())

    args.compile_component = "encode"
    train.validate_compile_only_args(args, save_updates=())

    args.compile_only = False
    with pytest.raises(ValueError, match="requires --compile-only"):
        train.validate_compile_only_args(args, save_updates=())

    args = valid_compile_only_args()
    args.compile_component = "encode"
    with pytest.raises(
        ValueError,
        match="requires --gradient-execution split",
    ):
        train.validate_compile_only_args(args, save_updates=())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("init", "exact", "requires --init model-only"),
        ("objective", "legacy", "requires --objective normalized"),
        ("train_seconds", 1.0, "requires --train-seconds 0"),
        ("eval_only", True, "cannot be combined with --eval-only"),
        ("gradient_audit", True, "cannot be combined with --gradient-audit"),
        ("resume_from", Path("checkpoint"), "cannot be combined with --resume-from"),
        ("save_final", True, "forbids every checkpoint"),
        ("gpu_monitor_interval_ms", 100, "requires --gpu-monitor-interval-ms 0"),
        ("donate", False, "requires the ordinary donated graph"),
    ],
)
def test_compile_only_cli_rejects_stateful_or_different_graph_modes(
    field: str,
    value: object,
    message: str,
) -> None:
    args = valid_compile_only_args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        train.validate_compile_only_args(args, save_updates=())


def test_dynamic_values_do_not_change_training_state_abi() -> None:
    zeros = {
        "model": {"weight": np.zeros((2, 3), dtype=np.float32)},
        "optimizer": [np.zeros((3,), dtype=np.float32)],
    }
    ones = {
        "model": {"weight": np.ones((2, 3), dtype=np.float32)},
        "optimizer": [np.ones((3,), dtype=np.float32)],
    }
    different_shape = {
        "model": {"weight": np.ones((3, 2), dtype=np.float32)},
        "optimizer": [np.ones((3,), dtype=np.float32)],
    }

    assert (
        train.research_state_abi(zeros)["sha256"]
        == train.research_state_abi(ones)["sha256"]
    )
    assert (
        train.research_state_abi(zeros)["sha256"]
        != train.research_state_abi(different_shape)["sha256"]
    )


def test_abstractification_preserves_raw_jax_signature_without_values() -> None:
    concrete = {
        "jax": jnp.ones((2, 3), dtype=jnp.bfloat16),
        "numpy": np.ones((4,), dtype=np.float32),
    }

    abstract, report = train.abstractify_dynamic_value(
        concrete,
        label="test",
    )

    assert report["records_equal"] is True
    assert report["before"]["sha256"] == report["after"]["sha256"]
    assert report["before"]["treedef_sha256"] == (
        report["after"]["treedef_sha256"]
    )
    assert report["before"]["nbytes"] == report["after"]["nbytes"]
    assert report["after"]["concrete_type_counts"] == {
        "jax.ShapeDtypeStruct": 2,
    }
    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaf in jax.tree.leaves(abstract)
    )


def test_donated_nnx_update_lowers_with_abstract_dynamic_arguments() -> None:
    class TinyModel(nnx.Module):
        def __init__(self) -> None:
            self.weight = nnx.Param(jnp.ones((2, 3), dtype=jnp.float32))

        def __call__(self, batch):
            return batch @ self.weight[...]

    model = TinyModel()
    optimizer = nnx.Optimizer(
        model,
        optax.sgd(1e-3),
        wrt=nnx.Param,
    )
    batch = jnp.ones((4, 2), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)

    @nnx.jit(donate_argnums=(0, 1))
    def step(model, optimizer, batch, rng):
        del rng

        def loss(candidate):
            return jnp.sum(candidate(batch))

        grads = nnx.grad(loss)(model)
        optimizer.update(model, grads)
        return loss(model)

    abstract_batch, abstract_rng, report = (
        train.abstractify_compile_only_arguments(
            model=model,
            optimizer=optimizer,
            batch={"inputs": batch},
            rng=rng,
            ema_target=None,
        )
    )

    lowered = step.lower(
        model,
        optimizer,
        abstract_batch["inputs"],
        abstract_rng,
    )

    assert lowered is not None
    assert report["all_dynamic_leaves_abstract"] is True
    assert report["all_abstract_signature_records_equal"] is True
    assert report["concrete_nbytes_replaced"] > 0


def test_compile_training_executable_lowers_without_execution(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeExecutable:
        def cost_analysis(self):
            return [{"flops": 4.0}, {"flops": 6.0, "bytes accessed": 3.0}]

        def memory_analysis(self):
            return types.SimpleNamespace(
                temp_size_in_bytes=7,
                argument_size_in_bytes=11,
            )

    class FakeLowered:
        def compile(self):
            calls.append(("compile",))
            return FakeExecutable()

    class FakeTrain:
        def lower(self, *args):
            calls.append(("lower", *args))
            return FakeLowered()

    monkeypatch.setattr(
        train,
        "training_call_args",
        lambda **_kwargs: ("dynamic-model", "dynamic-optimizer"),
    )
    times = iter((10.0, 12.5))
    monkeypatch.setattr(train.time, "perf_counter", lambda: next(times))

    result = train.compile_training_executable(
        FakeTrain(),
        objective="normalized",
        model=object(),
        optimizer=object(),
        batch={},
        rng=np.asarray([0, 1], dtype=np.uint32),
        sigreg_reference_count=1.0,
        ema_target=None,
    )

    assert calls == [
        ("lower", "dynamic-model", "dynamic-optimizer"),
        ("compile",),
    ]
    assert result.seconds == 2.5
    assert result.cost_analysis_raw == {
        "flops": 10.0,
        "bytes accessed": 3.0,
    }
    assert result.memory_analysis == {
        "argument_size_in_bytes": 11,
        "temp_size_in_bytes": 7,
    }


def test_compile_only_report_has_no_restore_update_metrics_or_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = valid_compile_only_args()
    records: dict[str, dict[str, object]] = {}

    class FakeBatches:
        def provenance(self):
            return {"batch_size": 128, "seed": 0}

        def batch_at(self, index: int):
            assert index == 0
            return {"batch": np.ones((1,), dtype=np.float32)}

    optimizer = types.SimpleNamespace(step=np.asarray(0, dtype=np.int64))
    compilation = train.TrainingCompilation(
        executable=object(),
        seconds=3.0,
        cost_analysis_raw={"flops": 5.0},
        memory_analysis={"temp_size_in_bytes": 7},
    )
    monkeypatch.setenv(
        "JAX_COMPILATION_CACHE_DIR",
        "/mountpoint/.exp/chess-dfm-jax/.local/cache/jax",
    )
    monkeypatch.setattr(
        train,
        "training_state_footprint",
        lambda _model, _optimizer: {"abi": "same"},
    )
    monkeypatch.setattr(train, "training_function", lambda **_kwargs: object())
    monkeypatch.setattr(
        train,
        "compile_training_executable",
        lambda *_args, **_kwargs: compilation,
    )
    monkeypatch.setattr(
        train,
        "gpu_memory_stats",
        lambda: {"bytes_in_use": 0, "peak_bytes_in_use": 0},
    )
    monkeypatch.setattr(
        train,
        "abstractify_compile_only_arguments",
        lambda **kwargs: (
            kwargs["batch"],
            kwargs["rng"],
            {
                "all_dynamic_leaves_abstract": True,
                "all_abstract_signature_records_equal": True,
            },
        ),
    )
    monkeypatch.setattr(
        train,
        "import_legacy_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "compile-only must not import the legacy checkpoint"
        ),
    )

    def capture(path: Path, payload: dict[str, object]) -> None:
        records[path.name] = payload

    monkeypatch.setattr(train, "write_json", capture)
    output_dir = tmp_path / "compile-only"
    result = train.run_compile_only(
        args,
        commit="abc",
        timestamp="20260723T000000Z",
        run_id="compile-only-test",
        output_dir=output_dir,
        config=train.JointLatentSASAConfig(),
        train_batches=FakeBatches(),
        resume_contract={"contract": "same"},
        model=object(),
        optimizer=optimizer,
        ema_target=None,
        checkpoint_step=265_000,
        source_checkpoint_path=Path("/unused/state.npz"),
    )

    assert result == 0
    assert set(records) == {
        "run_config.json",
        "compiler_cost_analysis.json",
        "report.json",
    }
    report = records["report.json"]
    assert report["mode"] == "compile_only_shape_equivalent"
    assert report["source_checkpoint_opened"] is False
    assert report["source_checkpoint_values_restored"] is False
    assert report["constructor_parameter_payload_released"] is True
    assert report["model_or_optimizer_executed"] is False
    assert report["updates"] == 0
    assert report["checkpoint_writes"] == 0
    assert report["optimizer_step_after_compile"] == 0
    assert report["optimizer_step_after_compile_source"] == (
        "captured_before_abstract_lowering_no_execution"
    )
    assert report["abstract_compilation_arguments"][
        "all_dynamic_leaves_abstract"
    ] is True
    assert report["checkpoint_path"] is None
    assert report["metrics_path"] is None
    assert not list(output_dir.rglob("state.npz"))


def test_split_component_compile_report_uses_concrete_args_and_zero_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = valid_compile_only_args()
    args.gradient_execution = "split"
    args.compile_component = "encode"
    records: dict[str, dict[str, object]] = {}

    class FakeBatches:
        def provenance(self):
            return {"batch_size": 128, "seed": 0}

        def batch_at(self, index: int):
            assert index == 0
            return {"batch": np.ones((1,), dtype=np.float32)}

    optimizer = types.SimpleNamespace(step=np.asarray(0, dtype=np.int64))
    compilation = train.TrainingCompilation(
        executable=object(),
        seconds=2.0,
        cost_analysis_raw={"flops": 5.0},
        memory_analysis={
            "argument_size_in_bytes": 11,
            "output_size_in_bytes": 13,
            "temp_size_in_bytes": 17,
        },
    )
    before = {
        "existing-cache": {
            "size_bytes": 1,
            "sha256": "old",
        }
    }
    after = {
        **before,
        "jit__split_encode_impl-new-cache": {
            "size_bytes": 2,
            "sha256": "new",
        },
    }
    inventories = iter((before, after))
    monkeypatch.setenv(
        "JAX_COMPILATION_CACHE_DIR",
        "/mountpoint/.exp/chess-dfm-jax/.local/cache/jax",
    )
    monkeypatch.setattr(
        train,
        "training_state_footprint",
        lambda _model, _optimizer: {"abi": "same"},
    )
    monkeypatch.setattr(
        train,
        "split_training_functions",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        train,
        "compile_split_component",
        lambda *_args, **_kwargs: compilation,
    )
    monkeypatch.setattr(
        train,
        "compilation_cache_executable_inventory",
        lambda _path: next(inventories),
    )
    monkeypatch.setattr(
        train,
        "gpu_memory_stats",
        lambda: {"bytes_in_use": 0, "peak_bytes_in_use": 0},
    )
    monkeypatch.setattr(
        train,
        "import_legacy_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "split compile-only must not import the legacy checkpoint"
        ),
    )

    def capture(path: Path, payload: dict[str, object]) -> None:
        records[path.name] = payload

    monkeypatch.setattr(train, "write_json", capture)
    output_dir = tmp_path / "split-compile-only"
    result = train.run_split_component_compile_only(
        args,
        commit="abc",
        timestamp="20260723T000000Z",
        run_id="split-compile-only-test",
        output_dir=output_dir,
        config=train.JointLatentSASAConfig(),
        train_batches=FakeBatches(),
        resume_contract={"contract": "same"},
        model=object(),
        optimizer=optimizer,
        checkpoint_step=265_000,
        source_checkpoint_path=Path("/unused/state.npz"),
    )

    assert result == 0
    assert set(records) == {
        "run_config.json",
        "compiler_cost_analysis.json",
        "report.json",
    }
    report = records["report.json"]
    assert report["mode"] == "split_component_compile_only_concrete"
    assert report["component"] == "encode"
    assert report["source_checkpoint_opened"] is False
    assert report["source_checkpoint_values_restored"] is False
    assert report["parameter_values_are_concrete_compilation_inputs"] is True
    assert report["abstract_compilation_arguments"] is False
    assert report["model_or_optimizer_executed"] is False
    assert report["optimizer_step_after_compile"] == 0
    assert report["cache_added"] == {
        "jit__split_encode_impl-new-cache": after[
            "jit__split_encode_impl-new-cache"
        ]
    }
    assert report["cache_gate_passed"] is True
    assert report["completed"] is True
    assert report["checkpoint_writes"] == 0
    assert report["checkpoint_path"] is None
    assert report["metrics_path"] is None
    assert not list(output_dir.rglob("state.npz"))
