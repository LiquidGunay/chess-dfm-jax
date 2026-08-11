from __future__ import annotations

import inspect
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from research.interpretability.artifacts import verify_checksums
from research.interpretability.sparse.lorsa import LoRSAConfig, LowRankSparseAttention
from research.interpretability.sparse.lorsa_transfer_pilot import (
    UPSTREAM_COMMIT,
    ReconstructionAccumulator,
    _module_dtype,
    _source_file_hashes,
    _git_record,
    artifact_contract_gate,
    compact_feature_summary,
    compare_attention_patterns,
    run_synthetic_smoke,
    run_transfer,
    source_compatibility_gate,
)
from research.interpretability.sparse.transcoder import reconstruction_metrics


def test_streaming_reconstruction_matches_materialized_definition() -> None:
    generator = torch.Generator().manual_seed(3)
    target = torch.randn((5, 3, 4), generator=generator)
    reconstruction = target + 0.2 * torch.randn(target.shape, generator=generator)
    accumulator = ReconstructionAccumulator()
    accumulator.update(target[:2], reconstruction[:2])
    accumulator.update(target[2:4], reconstruction[2:4])
    accumulator.update(target[4:], reconstruction[4:])
    expected = reconstruction_metrics(target, reconstruction)
    assert accumulator.finalize() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_pattern_comparison_supports_different_head_counts() -> None:
    logits = torch.randn((2, 2, 3, 3), generator=torch.Generator().manual_seed(5))
    first = torch.softmax(logits, dim=-1)
    second = first.repeat_interleave(2, dim=1)
    report = compare_attention_patterns(first, second)
    torch.testing.assert_close(report["head_mean_js"], torch.zeros(2), atol=1e-7, rtol=0)
    assert "matched_head_js" not in report
    identity = compare_attention_patterns(first, first.clone())
    torch.testing.assert_close(identity["matched_head_js"], torch.zeros(2), atol=1e-7, rtol=0)


def test_compact_feature_summary_is_bounded_and_token_grounded() -> None:
    features = torch.zeros((4, 6))
    features[1, 4] = 3.0
    features[2, 4] = 2.0
    features[3, 1] = 4.0
    summary = compact_feature_summary(features, count=2)
    assert summary is not None
    assert summary["reported_feature_count"] == 2
    assert [row["feature"] for row in summary["top_features"]] == [4, 1]
    assert summary["top_features"][0]["peak_token"] == 1
    assert summary["top_features"][0]["active_token_count"] == 2


def _reviewed_manifest(layer: int = 14) -> dict[str, object]:
    return {
        "architecture": {"d_model": 1024, "n_ctx": 64},
        "hook_contract": {
            "layer": layer,
            "input": f"blocks.{layer}.hook_attn_in",
            "target": f"blocks.{layer}.hook_attn_out",
            "sequence_shape": ["batch", 64, 1024],
        },
        "normalization": {"folded": True},
        "upstream": {"code_commit": UPSTREAM_COMMIT},
        "conversion": {
            "conversion_reviewed": True,
            "source_pickle_deserialized_in_isolated_environment": True,
            "main_environment_pickle_loaded": False,
        },
    }


def test_artifact_contract_gate_is_exact_and_fail_closed() -> None:
    assert artifact_contract_gate(_reviewed_manifest(), layer=14)["passed"]
    wrong_layer = artifact_contract_gate(_reviewed_manifest(layer=13), layer=14)
    assert not wrong_layer["passed"]
    assert not wrong_layer["criteria"]["layer_matches"]


def test_source_gate_rejects_good_metrics_when_contract_failed() -> None:
    gate = source_compatibility_gate(
        {"normalized_mse": 0.01, "cosine": 0.99},
        {
            "js_divergence": {"mean": 0.001},
            "top1_agreement": {"mean": 1.0},
        },
        {"passed": False},
        native_recompute_max_abs_error=0.0,
        position_count=8,
    )
    assert not gate["passed"]
    assert not gate["criteria"]["artifact_contract_passed"]


def test_module_dtype_rejects_mixed_checkpoint_dtypes() -> None:
    config = LoRSAConfig(
        d_model=4,
        n_ctx=3,
        n_qk_heads=2,
        d_qk_head=2,
        n_ov_heads=4,
        top_k=2,
        use_smolgen=False,
    )
    module = LowRankSparseAttention(config)
    assert _module_dtype(module) == torch.float32
    module.W_Q.data = module.W_Q.data.double()
    with pytest.raises(ValueError, match="mixed dtypes"):
        _module_dtype(module)


def test_git_record_is_explicit_when_runtime_has_no_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_git(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(
        "research.interpretability.sparse.lorsa_transfer_pilot.subprocess.run",
        missing_git,
    )
    record = _git_record()
    assert record["git_available"] is False
    assert record["repository_available"] is None
    assert record["commit"] is None
    assert record["worktree_clean"] is None
    assert record["status_sha256"] is None


def test_real_run_identity_binds_exact_source_file_hashes() -> None:
    source_files = _source_file_hashes()
    repo_root = Path(__file__).resolve().parents[1]
    expected = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "research/interpretability").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    expected.update(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "chess_dfm_jax").glob("*.py")
        if "__pycache__" not in path.parts
    )
    expected.update(
        {
            "research/__init__.py",
            "research/train_torch.py",
            "research/arena_history_trust.py",
            "research/prepare.py",
            "research/resource_guard.py",
            "research/run_torch_gpu.sh",
        }
    )
    assert set(source_files) == expected
    assert list(source_files) == sorted(source_files)
    assert {
        "research/interpretability/artifacts.py",
        "research/interpretability/attention.py",
        "research/interpretability/chessbench.py",
        "research/interpretability/models.py",
        "research/interpretability/sparse/lorsa.py",
        "research/interpretability/sparse/transcoder.py",
        "research/interpretability/sparse/transfer_pilot.py",
        "research/train_torch.py",
        "research/arena_history_trust.py",
        "research/prepare.py",
        "research/resource_guard.py",
        "research/run_torch_gpu.sh",
        "chess_dfm_jax/__init__.py",
        "chess_dfm_jax/encoding.py",
        "chess_dfm_jax/policy.py",
    } <= set(source_files)
    assert all("__pycache__" not in Path(name).parts for name in source_files)
    assert not any(name.startswith("research/analysis/") for name in source_files)
    assert not any(name.startswith("research/notebooks/") for name in source_files)
    assert not any("credential" in name.lower() for name in source_files)
    assert all(len(digest) == 64 for digest in source_files.values())
    run_source = inspect.getsource(run_transfer)
    assert "\"source_files\": content_identity(source_files)" in run_source
    assert "run_id = content_identity(payloads)" in run_source


def test_synthetic_smoke_seals_a_non_scientific_immutable_run(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    args = Namespace(output=output, seed=11, feature_summary_count=3)
    result = run_synthetic_smoke(args)
    assert result["source_gate_passed"]
    assert result["transfer_status"] == "complete"
    verify_checksums(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert manifest["experiment"]["paid_compute"] is False
    assert manifest["experiment"]["scientific_result"] is False
    assert metrics["scientific_result"] is False
    rows = (output / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    with pytest.raises(FileExistsError, match="immutable"):
        run_synthetic_smoke(args)
