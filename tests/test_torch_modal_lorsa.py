from __future__ import annotations

import inspect
import importlib.util
import sys
import types

if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    fake_modal = types.ModuleType("modal")

    class _Image:
        @classmethod
        def debian_slim(cls, **_kwargs: object) -> "_Image":
            return cls()

        @classmethod
        def from_registry(cls, *_args: object, **_kwargs: object) -> "_Image":
            return cls()

        def __getattr__(self, _name: str):
            return lambda *_args, **_kwargs: self

    class _Volume:
        @classmethod
        def from_name(cls, *_args: object, **_kwargs: object) -> "_Volume":
            return cls()

        def with_mount_options(self, **_kwargs: object) -> "_Volume":
            return self

    class _App:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def function(self, **_kwargs: object):
            return lambda function: function

        def local_entrypoint(self):
            return lambda function: function

    fake_modal.Secret = type(
        "_Secret", (), {"from_name": classmethod(lambda cls, *args, **kwargs: cls())}
    )
    fake_modal.Volume = _Volume
    sys.modules["modal"] = fake_modal

    fake_modal.App = _App
    fake_modal.Image = _Image
from decimal import Decimal
from pathlib import Path

import pytest

from research.interpretability.modal_app import (
    LORSA_RUNNER_REVISION,
    DEFAULT_LORSA_MANIFEST_RELATIVE,
    DEFAULT_LORSA_WEIGHTS_RELATIVE,
    _bundle_file,
    _lorsa_config,
    _lorsa_result_label,
    _run_lorsa_job,
    _spec,
)
from research.interpretability.remote_cost import estimate_modal_job


def test_lorsa_t4_l4_benchmark_identity_is_gpu_specific_and_deterministic() -> None:
    base = "a" * 64
    lorsa = "b" * 64
    common = {
        "profile": "smoke",
        "manifest_relative": DEFAULT_LORSA_MANIFEST_RELATIVE,
        "weights_relative": DEFAULT_LORSA_WEIGHTS_RELATIVE,
    }
    first = _lorsa_result_label(base, lorsa, gpu="T4", **common)
    assert first == _lorsa_result_label(base, lorsa, gpu="T4", **common)
    assert first != _lorsa_result_label(base, lorsa, gpu="L4", **common)
    assert "-t4-" in first


def test_lorsa_profiles_keep_identical_smoke_workload_across_gpu_wrappers() -> None:
    smoke = _lorsa_config("smoke")
    development = _lorsa_config("development")
    assert smoke["position_count"] == 8
    assert development["position_count"] == 64
    assert LORSA_RUNNER_REVISION == "lorsa-transfer-modal-v2"
    for key in ("layer", "batch_size", "feature_summary_count", "seed", "cpu_threads"):
        assert smoke[key] == development[key]


@pytest.mark.parametrize(
    "stage",
    (
        "lorsa-smoke-t4-gpu",
        "lorsa-smoke-l4-gpu",
        "lorsa-development-t4-gpu",
        "lorsa-development-l4-gpu",
    ),
)
def test_every_lorsa_modal_profile_fits_its_strict_cap(stage: str) -> None:
    spec, cap = _spec(stage)
    estimate = estimate_modal_job(spec)
    assert Decimal(estimate["declared_attempt_upper_bound_dollars"]) <= cap
    assert spec.retries == 0


def test_t4_l4_smoke_specs_differ_only_by_gpu_and_cap() -> None:
    t4, t4_cap = _spec("lorsa-smoke-t4-gpu")
    l4, l4_cap = _spec("lorsa-smoke-l4-gpu")
    assert (
        t4.timeout_seconds,
        t4.startup_timeout_seconds,
        t4.cpu_cores,
        t4.memory_gib,
        t4.retries,
    ) == (
        l4.timeout_seconds,
        l4.startup_timeout_seconds,
        l4.cpu_cores,
        l4.memory_gib,
        l4.retries,
    )
    assert t4.gpu == "T4"
    assert l4.gpu == "L4"
    assert t4_cap < l4_cap


def test_bundle_file_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    artifact = root / "converted" / "manifest.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    assert _bundle_file(root, "converted/manifest.json") == artifact.resolve()
    with pytest.raises(ValueError, match="Unsafe"):
        _bundle_file(root, "../outside")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (root / "escape.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        _bundle_file(root, "escape.json")


def test_remote_runner_invokes_module_not_a_direct_script_path() -> None:
    source = inspect.getsource(_run_lorsa_job)
    assert "sys.executable" in source
    assert '"-m"' in source
    assert '"research.interpretability.sparse.lorsa_transfer_pilot"' in source
    assert "subprocess.run" in source
