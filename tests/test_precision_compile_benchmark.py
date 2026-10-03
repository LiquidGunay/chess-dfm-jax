import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from research import precision_compile_benchmark as bench


def test_matrix_separates_storage_compute_and_production_control():
    cells = bench.ARMS[:6]
    assert {(a["dtype"], a["compile"]) for a in cells} == {
        (d, c) for d in ("float32", "bfloat16", "float16") for c in ("none", "fresh-bt4")}
    assert all(a["storage"] == "float32" for a in cells)
    assert bench.ARMS[-1]["storage"] == "checkpoint"
    assert bench.ARMS[-1]["compile"] == "fresh"
    assert sum(a["timeout"] for a in bench.ARMS) < 4200


def test_comparison_and_aggregation_are_exact():
    a, b = torch.tensor([1., 2.]), torch.tensor([2., 4.])
    stats = bench.finalize_stats(bench.tensor_stats(a, b))
    assert stats["relative_l2"] == 1
    assert stats["cosine"] == pytest.approx(1)
    aggregate = bench.merge_stats(bench.tensor_stats(a[:1], b[:1]), bench.tensor_stats(a[1:], b[1:]))
    assert bench.finalize_stats(aggregate) == stats
    assert not bench.finalize_stats(bench.tensor_stats(a, b*float("nan")))["finite"]
    with pytest.raises(ValueError): bench.tensor_stats(a, b[:1])


def test_gradient_comparison_unscales_without_mutating_parameters(tmp_path):
    model = torch.nn.Linear(3, 2)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    model(torch.ones(4, 3)).sum().backward()
    result = bench.gradient_comparison(model, 1, tmp_path/"ref.safetensors", save_reference=True)
    assert result["finite"]
    for p in model.parameters(): p.grad.mul_(128)
    other = bench.gradient_comparison(model, 128, tmp_path/"ref.safetensors")
    assert other["groups"]["all"]["relative_l2"] == 0
    for n,p in model.named_parameters(): torch.testing.assert_close(p, before[n], rtol=0, atol=0)
    model.weight.grad[0, 0] = float("inf")
    assert not bench.gradient_comparison(model, 128, tmp_path/"ref.safetensors")["finite"]


def test_json_refuses_overwrite_and_nonfinite(tmp_path):
    path = tmp_path/"once.json"
    bench.write_json(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}
    with pytest.raises(FileExistsError): bench.write_json(path, {"x": 2})
    with pytest.raises(ValueError): bench.write_json(tmp_path/"bad.json", {"x": float("nan")})


def test_prepare_uses_history_and_binds_rng_and_inputs(monkeypatch):
    import research.hero_history_batches as history
    import research.input_contract as contract
    import research.train_torch as trainer
    seen = []
    class Batches:
        def __init__(self, *a, **kw): seen.append(kw)
        def batch_at(self, cursor): return {"cursor": cursor}
    monkeypatch.setattr(history, "HistoryTrajectoryBatches", Batches)
    monkeypatch.setattr(contract, "audit_full_history_batch", lambda batch, **kw: {"cursor": batch["cursor"]})
    monkeypatch.setattr(trainer, "_prepare_training_step", lambda *a, **kw: types.SimpleNamespace(
        compact_batch={"current_planes": np.array([kw["data_cursor"]], dtype=np.float32)},
        choices=types.SimpleNamespace(seed=torch.tensor([kw["seed"]]))))
    protocol = {"batch_size":64, "dataset_manifest_sha256":"a"*64, "seed":0}
    _, audit, first = bench.prepared_batch(protocol, Path("data"), 8000)
    assert audit == {"cursor":8000}
    assert seen[0]["split"] == "train" and seen[0]["batch_size"] == 64
    assert first == bench.prepared_batch(protocol, Path("data"), 8000)[2]
    assert first != bench.prepared_batch(protocol, Path("data"), 8001)[2]
    assert first != bench.prepared_batch({**protocol, "seed":1}, Path("data"), 8000)[2]


def test_config_preserves_history_recipe_runtime():
    from research.train_torch import HERO_CONFIG
    current = bench.config()
    assert current.use_bt4_policy_residual and current.horizon == 8
    assert current.remat_bt4_blocks and current.remat_projector_blocks
    assert not current.remat_dfm_blocks
    assert current.use_bt4_sdpa and current.use_head_sdpa
    assert current.root_legal_ce_coeff == HERO_CONFIG.root_legal_ce_coeff


def test_summary_rejects_training_and_inconsistent_timings():
    import copy
    from research.summarize_precision_compile_benchmark import summarize_arm
    grads = {"groups":{"all":{"cosine":1., "relative_l2":0.}}}
    batch = {"finite":True, "scalars":{"loss":3.}, "legal_top1_agreement":1.,
             "gradients_vs_fp32_eager":grads, "gradients_vs_same_dtype_eager":grads}
    result = {"status":"complete", "batches":[batch]*3, "optimizer_steps":0,
        "warm":{"repeats":[{"wall_seconds":2.}, {"wall_seconds":4.}],
                "mean_seconds":3., "peak_allocated_bytes":1024**3},
        "cold_first_forward_backward_seconds":10., "cold_minus_warm_seconds":7.,
        "parameter_bytes_by_dtype":{"torch.float32":100},
        "native_policy_legal_top1_agreement":1., "layer_trace_vs_fp32_eager":{}}
    row = summarize_arm(result, 64)
    assert row["mean_forward_backward_seconds"] == 3
    assert row["examples_per_second_forward_backward_only"] == pytest.approx(64/3)
    bad = copy.deepcopy(result); bad["optimizer_steps"] = 1
    with pytest.raises(ValueError): summarize_arm(bad, 64)
    bad = copy.deepcopy(result); bad["warm"]["mean_seconds"] = 2
    with pytest.raises(ValueError): summarize_arm(bad, 64)
    bad = copy.deepcopy(result); bad["batches"][0]["finite"] = False
    with pytest.raises(ValueError): summarize_arm(bad, 64)
    assert summarize_arm({"status":"nonfinite", "batches":[batch]}, 64)["status"] == "nonfinite"
