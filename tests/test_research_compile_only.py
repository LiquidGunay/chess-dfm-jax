from __future__ import annotations

import argparse
import copy
import gc
import types
import weakref
from pathlib import Path

import numpy as np
import pytest

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


def test_construction_parameter_payload_is_actually_released() -> None:
    array = np.ones((32,), dtype=np.float32)
    reference = weakref.ref(array)
    payload = {"nested": {"weight": array}}
    del array

    train.release_construction_parameter_payload(payload)
    gc.collect()

    assert payload == {}
    assert reference() is None
    with pytest.raises(TypeError, match="plain dict"):
        train.release_construction_parameter_payload(types.MappingProxyType({}))


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
            return {"batch": "shape-only"}

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
    assert report["checkpoint_path"] is None
    assert report["metrics_path"] is None
    assert not list(output_dir.rglob("state.npz"))
