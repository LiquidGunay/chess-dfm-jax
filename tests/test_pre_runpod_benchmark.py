import copy
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import pre_runpod_benchmark as bench
from research import train_torch as trainer


def tiny():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    model.head = torch.nn.Linear(4, 2)
    optimizer = trainer.MuonAdamW(model, bench.parent_config(), examples_per_update=1024)
    for leaf in optimizer.leaves:
        leaf.first_moment.fill_(0.25)
        if leaf.second_moment is not None: leaf.second_moment.fill_(0.125)
    optimizer.update = 500; optimizer.examples_seen = 512000
    return model, optimizer


def test_arms_control_storage_runtime_and_memory_independently():
    a = {r['id']: r for r in bench.ARMS}
    assert len(a) == 8
    assert sum(r['timeout'] for r in a.values()) + 500 < 5700
    for name, key in [('fp32_storage_control', 'storage'), ('fused_norm', 'norm'), ('delayed_prefetch', 'prefetch')]:
        assert {k for k in a[name] if a[name][k] != a['production_control'][k]} == {'id', key}
    assert a['bf16_compiled']['storage'] == a['fp16_compiled']['storage'] == 'float32'
    assert a['fp16_less_remat']['remat_bt4'] is False
    assert bench.parent_config() == dataclasses.replace(trainer.HERO_CONFIG,
        remat_bt4_blocks=True, remat_projector_blocks=True, remat_dfm_blocks=False,
        use_bt4_sdpa=True, use_head_sdpa=True)


def test_promotion_preserves_values_moments_and_counters():
    model, optimizer = tiny()
    before = {n:p.detach().float().clone() for n,p in model.named_parameters()}
    promoted = bench.promote_state(model, optimizer)
    assert promoted.update == 500 and promoted.examples_seen == 512000
    assert promoted.policy == trainer.LEGACY_OPTIMIZER_POLICY
    assert promoted.learning_rate_ratio() == optimizer.learning_rate_ratio()
    for n,p in model.named_parameters():
        assert p.dtype == torch.float32
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)
    for old, new in zip(optimizer.leaves, promoted.leaves):
        assert new.first_moment.dtype == torch.float32
        torch.testing.assert_close(new.first_moment, old.first_moment.float(), rtol=0, atol=0)
        torch.testing.assert_close(new.second_moment, old.second_moment.float(), rtol=0, atol=0)
        assert new.master_parameter is None


@pytest.mark.parametrize('promote', [False, True])
def test_child_checkpoint_roundtrip_reproduces_next_update_and_guards_contract(tmp_path, monkeypatch, promote):
    monkeypatch.setattr(trainer, '_WORKSPACE_ROOT', tmp_path)
    model, optimizer = tiny()
    if promote: optimizer = bench.promote_state(model, optimizer)
    contract = {'schema_version':'gpu003-test', 'precision':'float32' if promote else 'checkpoint',
                'source_sha256':'b'*64, 'seed':0}
    manifest = trainer.save_training_checkpoint(output_dir=tmp_path, model=model, optimizer=optimizer,
        source_mapping_sha256='a'*64, next_data_cursor=500, resume_contract=contract)
    checkpoint = tmp_path/'checkpoints/update00000500'
    before = bench.state_hashes(model, optimizer)
    for p in model.parameters(): p.grad = torch.full_like(p, .125)
    optimizer.step()
    expected = bench.state_hashes(model, optimizer)
    restored = trainer.load_training_checkpoint(checkpoint_dir=checkpoint, model=model, optimizer=optimizer,
        expected_source_mapping_sha256='a'*64, expected_resume_contract=contract)
    assert restored == manifest and bench.state_hashes(model, optimizer) == before
    for p in model.parameters(): p.grad = torch.full_like(p, .125)
    optimizer.step()
    assert bench.state_hashes(model, optimizer) == expected
    with pytest.raises(ValueError, match='contract'):
        trainer.load_training_checkpoint(checkpoint_dir=checkpoint, model=model, optimizer=optimizer,
            expected_source_mapping_sha256='a'*64, expected_resume_contract={**contract, 'seed':1})


def test_state_hash_binds_moments_and_rejects_nonfinite():
    model, optimizer = tiny()
    before = bench.state_hashes(model, optimizer)
    optimizer.leaves[0].first_moment.add_(1)
    after = bench.state_hashes(model, optimizer)
    assert before['model'] == after['model'] and before['optimizer'] != after['optimizer']
    optimizer.leaves[0].first_moment.fill_(float('nan'))
    with pytest.raises(FloatingPointError): bench.state_hashes(model, optimizer)


def test_digest_binds_input_and_materialized_rng():
    choices = trainer.materialize_step_choices(seed=0, update=500, batch_size=16,
        config=bench.parent_config(), device=torch.device('cpu'))
    prepared = SimpleNamespace(compact_batch={'current_planes':np.ones((2,3), dtype=np.float32)}, choices=choices)
    first = bench.prepared_digest(prepared)
    assert bench.prepared_digest(copy.deepcopy(prepared)) == first
    prepared.compact_batch['current_planes'][0,0] = 0
    assert bench.prepared_digest(prepared) != first
    prepared.compact_batch['current_planes'][0,0] = 1
    choices.training_time[0] += .1
    assert bench.prepared_digest(prepared) != first


def test_summary_excludes_warmup_checkpoint_and_validates_window():
    rows = [dict(update=i, optimizer_skipped_nonfinite=False, loss=3,
        end_to_end_seconds=2 if 511 <= i <= 540 else 100,
        forward_seconds=.8, backward_seconds=.9, optimizer_seconds=.1,
        peak_allocated_bytes=1000) for i in range(501,551)]
    result = bench.summarize(rows)
    assert result['mean_seconds'] == result['median_seconds'] == 2
    assert result['examples_per_second'] == 512
    with pytest.raises(ValueError): bench.summarize(rows[:-1])
    rows[2]['optimizer_skipped_nonfinite'] = True
    with pytest.raises(ValueError): bench.summarize(rows)


def result(seconds=2., ce=1.):
    return {'status':'complete', 'timing':{'mean_seconds':seconds},
        'validation':{'root_legal_conditional_ce':ce, 'base_policy_root_legal_conditional_ce':ce}}


def test_selection_requires_speed_and_both_common_losses():
    results = {'production_control':result(), 'fast_bad':result(1,1.03), 'small':result(1.95),
               'winner':result(1.5)}
    assert bench.choose_candidate(results) == 'winner'
    results['winner']['validation']['base_policy_root_legal_conditional_ce'] = 1.04
    assert bench.choose_candidate(results) is None
    assert bench.choose_candidate({}) is None


def test_runtime_norm_dispatch_is_real():
    layer = trainer.RawLayerNorm(8, dtype=torch.float32, implementation='eager-fused-backward')
    with torch.no_grad(): layer.scale.fill_(1); layer.bias.zero_()
    value = torch.randn(2,8,requires_grad=True)
    output = layer(value, torch.float32)
    output.square().sum().backward()
    assert bool(torch.isfinite(value.grad).all())


def test_batch_provenance_uses_canonical_volume_path(tmp_path, monkeypatch):
    import research.hero_history_batches as history
    real = tmp_path/'volume'; real.mkdir()
    alias = tmp_path/'mount'; alias.symlink_to(real, target_is_directory=True)
    seen = []
    monkeypatch.setattr(history, 'HistoryTrajectoryBatches', lambda path, **kwargs: seen.append(path))
    bench.batches({'dataset_manifest_sha256':'a'*64}, alias)
    assert seen == [real.resolve()]
