import dataclasses
import numpy as np
import pytest
import torch
from research import train_torch as t
from research.value_pilot import config, optimizer_policy


def test_future_residual_is_current_baseline_with_perspective_flip():
    native = torch.tensor([[2., 1., -3.]], requires_grad=True)
    auxiliary = torch.zeros(1, 9, 3, requires_grad=True)
    result = t.compose_value_logits("jepa_residual", native, auxiliary)
    assert torch.equal(result[:, ::2], native[:, None].expand(-1, 5, -1))
    assert torch.equal(result[:, 1::2], native.flip(-1)[:, None].expand(-1, 4, -1))
    result.sum().backward()
    assert torch.equal(native.grad, torch.full_like(native, 9.))
    assert torch.equal(auxiliary.grad, torch.ones_like(auxiliary))


def test_native_uses_native_current_and_auxiliary_future():
    native = torch.randn(2, 3, requires_grad=True)
    aux = torch.randn(2, 9, 3, requires_grad=True)
    result = t.compose_value_logits("native", native, aux)
    assert torch.equal(result[:, 0], native)
    assert torch.equal(result[:, 1:], aux[:, 1:])
    result.sum().backward()
    assert torch.count_nonzero(aux.grad[:, 0]) == 0
    assert torch.count_nonzero(native.grad) == 6


def test_native_value_head_square_flatten_parity():
    torch.manual_seed(3)
    model = t.BT4ValueHead()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, .01)
    x = torch.randn(2, 64, 1024)
    mish = lambda z: z * torch.tanh(torch.nn.functional.softplus(z))
    y = mish(x @ model.embed.w + model.embed.b).reshape(2, -1)
    y = mish(y @ model.dense.w + model.dense.b)
    torch.testing.assert_close(model(x, torch.float32), y @ model.out.w + model.out.b)


def test_actual_current_target_survives_both_batch_filters():
    current = np.array([[.8,.1,.1]], dtype=np.float32)
    future = np.array([[[.2,.3,.5]]], dtype=np.float32)
    batch = {"current_wdl_target":current, "current_value_target":current[:,0]-current[:,2],
        "wdl_targets":future,"future_planes":np.zeros((1,1,112,8,8),np.uint8)}
    compact = t._compact_training_batch(batch, np.array([0]))
    transferred = t._torch_batch(compact, torch.device("cpu"))
    torch.testing.assert_close(transferred["current_wdl_target"], torch.from_numpy(current))
    _, targets, _ = t._prepare_wdl_supervision(torch.zeros(1,1,4),torch.zeros(1,4),
        transferred["wdl_targets"],torch.ones(1),torch.ones(1,1),include_current_state=True,
        current_target=transferred["current_wdl_target"])
    torch.testing.assert_close(targets[:,0],torch.from_numpy(current))
    assert not torch.equal(targets[:,0], transferred["wdl_targets"][:,0].flip(-1))


def test_fresh_pilot_preserves_hero_hyperparameters_and_uses_stable_wsd():
    a, b = config("native"), config("jepa_residual")
    assert dataclasses.replace(a, value_supervision_mode="jepa_residual") == b
    assert a.learning_rate == t.HERO_CONFIG.learning_rate
    assert a.bt4_learning_rate == t.HERO_CONFIG.bt4_learning_rate
    assert a.lr_warmup_examples == 566866
    tiny = torch.nn.Linear(2,2)
    optimizer = t.MuonAdamW(tiny, a, policy=optimizer_policy(), examples_per_update=1024)
    assert optimizer.update == optimizer.examples_seen == 0
    optimizer.update, optimizer.examples_seen = 554, 554*1024
    assert optimizer.learning_rate_ratio() == 1
    optimizer.update, optimizer.examples_seen = 1000, 1000*1024
    assert optimizer.learning_rate_ratio() == 1


def test_bad_value_mode_is_rejected():
    with pytest.raises(ValueError):
        t.compose_value_logits("bad",torch.zeros(1,3),torch.zeros(1,9,3))


def test_historical_arena_configs_keep_default_value_mode():
    from research.evaluate_arena import _validated_torch_hero_model_config
    old = dataclasses.asdict(dataclasses.replace(config("native"), value_supervision_mode="legacy"))
    old.pop("value_supervision_mode")
    assert t.Config(**_validated_torch_hero_model_config(old)).value_supervision_mode == "legacy"
