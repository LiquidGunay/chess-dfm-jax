import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.analysis.eval_history_loss_baseline import CurrentOnlyBatches
from research import train_torch as train


def test_current_only_control_does_not_mutate_history_dataset():
    source = {"current_planes":np.ones((2,112,8,8)),"future_planes":np.ones((2,8,112,8,8))}
    batches = CurrentOnlyBatches(SimpleNamespace(batch_size=2,batch_at=lambda step:source))
    output = batches.batch_at(0)
    for key in source:
        assert source[key].all()
        assert not output[key][...,12:104,:,:].any()
        assert output[key][...,:12,:,:].all()
        assert output[key][...,104:,:,:].all()


def test_selected_recipe_matches_original_hero_scientific_configuration():
    args = train.build_parser().parse_args(["train","--recipe","hero","--output-dir","/tmp/not-launched"])
    config, policy = train._resolve_training_recipe(args)
    config = dataclasses.replace(config, remat_bt4_blocks=True,remat_projector_blocks=True,
        remat_dfm_blocks=False,use_bt4_sdpa=True,use_head_sdpa=True)
    actual = dataclasses.asdict(config)
    path = Path(__file__).resolve().parents[1]/"research/analysis/hero_v2_phase1/hero1_exact/run_config.json"
    original = json.loads(path.read_text())["config"]
    assert {key:actual[key] for key in original} == original
    assert policy == train.LEGACY_OPTIMIZER_POLICY


def test_validation_curve_uses_common_prefix_without_mutating_terminal_pool(monkeypatch,tmp_path):
    calls = []
    monkeypatch.setattr(train,"_evaluate_validation_pool",lambda *a,**kw:(calls.append(kw["count"]) or {"loss":1.},0.))
    monkeypatch.setattr(train,"_git_commit",lambda:"test")
    monkeypatch.setattr(train,"_append_jsonl",lambda *args:None)
    resources = train._HeroValidationResources(
        fast_batches=SimpleNamespace(steps_per_epoch=127),fast_pool={
            "pool_definition":{"count":8128,"seed":2026072501},"data":{"batch_size":64},
            "intermediate_examples":1024,"terminal_update":553})
    model = SimpleNamespace(training=True,train=lambda state:None)
    for update in (100,553):
        result = train._run_frozen_validation_milestone(model,resources,output_dir=tmp_path,
            update=update,batch_size=1024,device=None)
        assert result["evaluation_examples"] == (1024 if update == 100 else 8128)
    assert calls == [16,127]
    assert resources.evaluation_examples == 8128
