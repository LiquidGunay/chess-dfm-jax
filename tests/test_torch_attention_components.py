from __future__ import annotations

import torch

from research.interpretability.attention import (
    attention_internals,
    attention_move_alignment,
    future_move_attention_alignment,
    qk_smolgen_balance,
    replace_attention_component_heads,
    replace_attention_heads,
)
from research.train_torch import BT4EncoderLayer, RawLayerNorm


def _layer() -> BT4EncoderLayer:
    layer = BT4EncoderLayer(use_sdpa=False, norm_impl="eager")
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0.0, 0.01, generator=generator)
        for module in layer.modules():
            if isinstance(module, RawLayerNorm):
                module.scale.fill_(1.0)
                module.bias.zero_()
    return layer


def test_attention_decomposition_reconstructs_native_branch_and_head_sum() -> None:
    layer = _layer()
    states = torch.randn((1, 64, 1024), generator=torch.Generator().manual_seed(19))
    with torch.inference_mode():
        internals = attention_internals(layer, states, compute_dtype=torch.float32)
        _output, capture = layer.forward_with_capture(states, 0.4, torch.float32)
    torch.testing.assert_close(
        internals.projected_output,
        capture.hook_attn_out,
        rtol=0.0,
        atol=0.0,
    )
    assert internals.pattern.shape == (1, 32, 64, 64)
    assert internals.projected_head_contributions.shape == (1, 32, 64, 1024)
    assert internals.contribution_reconstruction_max_abs_error < 1e-5
    same_patch = replace_attention_heads(internals, internals, (0, 7, 31))
    torch.testing.assert_close(same_patch, internals.projected_output, rtol=0.0, atol=0.0)


def test_component_head_replacement_and_pattern_validation() -> None:
    layer = _layer()
    states = torch.zeros((1, 64, 1024))
    with torch.inference_mode():
        destination = attention_internals(layer, states, compute_dtype=torch.float32)
        source = attention_internals(layer, states + 1.0, compute_dtype=torch.float32)
    replaced = replace_attention_component_heads(source, destination, "query", (3,))
    torch.testing.assert_close(replaced[:, 3], source.query[:, 3])
    torch.testing.assert_close(replaced[:, 2], destination.query[:, 2])
    with torch.inference_mode():
        replay = attention_internals(
            layer,
            states,
            compute_dtype=torch.float32,
            query_override=replaced,
        )
    assert replay.projected_output.shape == (1, 64, 1024)

    bad_pattern = destination.pattern.clone()
    bad_pattern[..., 0] += 1.0
    try:
        attention_internals(
            layer,
            states,
            compute_dtype=torch.float32,
            pattern_override=bad_pattern,
        )
    except ValueError as error:
        assert "row-normalized" in str(error)
    else:
        raise AssertionError("Invalid attention pattern was accepted")


def test_move_and_future_alignment_find_endpoint_head() -> None:
    pattern = torch.full((1, 32, 64, 64), 1.0 / 64)
    pattern[:, 5, 2] = 0.0
    pattern[:, 5, 2, 9] = 1.0
    report = attention_move_alignment(
        pattern,
        torch.tensor([2]),
        torch.tensor([9]),
    )
    assert report["origin_to_destination_top1"][0, 5]
    assert report["origin_to_destination_mass"][0, 5] == 1.0
    future = future_move_attention_alignment(
        pattern,
        torch.tensor([[2, 4]]),
        torch.tensor([[9, 7]]),
        torch.tensor([[True, False]]),
    )
    assert future["symmetric_endpoint_mass"].shape == (1, 2, 32)
    assert torch.isnan(future["symmetric_endpoint_mass"][0, 1]).all()
    balance = qk_smolgen_balance(
        attention_internals(_layer(), torch.zeros((1, 64, 1024)), compute_dtype=torch.float32)
    )
    assert balance["qk_rms"].shape == (1, 32)
