from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from research.train_torch import (
    BT4Encoder,
    BT4EncoderLayer,
    BT4EncoderLayerCapture,
    RawLayerNorm,
)


HOOK_NAMES = (
    "hook_attn_in",
    "hook_attn_out",
    "resid_mid_after_ln",
    "hook_mlp_out",
    "resid_post_after_ln",
)


def _initialized_layer() -> BT4EncoderLayer:
    generator = torch.Generator().manual_seed(7)
    layer = BT4EncoderLayer(use_sdpa=False, norm_impl="eager")
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(mean=0.0, std=0.01, generator=generator)
        for module in layer.modules():
            if isinstance(module, RawLayerNorm):
                module.scale.fill_(1.0)
                module.bias.zero_()
    return layer


def _legacy_layer_forward(
    layer: BT4EncoderLayer,
    x: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    batch, sequence, _ = x.shape
    q = x @ layer.wq.float() + layer.wq_b.float()
    k = x @ layer.wk.float() + layer.wk_b.float()
    v = x @ layer.wv.float() + layer.wv_b.float()
    q = q.reshape(batch, sequence, 32, 32).transpose(1, 2)
    k = k.reshape(batch, sequence, 32, 32).transpose(1, 2)
    v = v.reshape(batch, sequence, 32, 32).transpose(1, 2)
    logits = (q @ k.transpose(-2, -1)) / math.sqrt(32.0)
    attention = F.softmax(logits + layer.smolgen(x, torch.float32), dim=-1)
    out = attention @ v
    out = out.transpose(1, 2).reshape(batch * sequence, 1024)
    out = layer.wo(out, torch.float32).reshape(batch, sequence, 1024)
    resid_mid = layer.ln_attn(out * alpha + x, torch.float32)
    flat = resid_mid.reshape(batch * sequence, 1024)
    ffn = layer.ffn2(
        F.mish(layer.ffn1(flat, torch.float32)),
        torch.float32,
    )
    return layer.ln_ffn(
        ffn.reshape(batch, sequence, 1024) * alpha + resid_mid,
        torch.float32,
    )


def test_torch_layer_capture_is_read_only_and_uses_frozen_boundaries() -> None:
    layer = _initialized_layer()
    x = torch.randn(
        (1, 64, 1024),
        generator=torch.Generator().manual_seed(11),
        dtype=torch.float32,
    )
    alpha = 0.37

    with torch.inference_mode():
        legacy = _legacy_layer_forward(layer, x, alpha)
        ordinary = layer(x, alpha, torch.float32)
        captured, hooks = layer.forward_with_capture(x, alpha, torch.float32)

    torch.testing.assert_close(ordinary, legacy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(captured, ordinary, rtol=0.0, atol=0.0)
    assert hooks.hook_attn_in.data_ptr() == x.data_ptr()
    for name in HOOK_NAMES:
        assert getattr(hooks, name).shape == (1, 64, 1024)
    with torch.inference_mode():
        expected_mid = layer.ln_attn(
            hooks.hook_attn_out * alpha + hooks.hook_attn_in,
            torch.float32,
        )
        expected_post = layer.ln_ffn(
            hooks.hook_mlp_out * alpha + hooks.resid_mid_after_ln,
            torch.float32,
        )
    torch.testing.assert_close(
        hooks.resid_mid_after_ln,
        expected_mid,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        hooks.resid_post_after_ln,
        expected_post,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        hooks.resid_post_after_ln,
        captured,
        rtol=0.0,
        atol=0.0,
    )


def test_torch_layer_branch_overrides_are_before_alpha_residual_and_norm() -> None:
    layer = _initialized_layer()
    x = torch.randn(
        (1, 64, 1024),
        generator=torch.Generator().manual_seed(13),
        dtype=torch.float32,
    )
    zeros = torch.zeros_like(x)
    alpha = 0.41

    with torch.inference_mode():
        replaced, hooks = layer.forward_with_capture(
            x,
            alpha,
            torch.float32,
            attention_output_override=zeros,
            mlp_output_override=zeros,
        )
        expected_mid = layer.ln_attn(x, torch.float32)
        expected_post = layer.ln_ffn(expected_mid, torch.float32)

    torch.testing.assert_close(hooks.hook_attn_out, zeros, rtol=0.0, atol=0.0)
    torch.testing.assert_close(hooks.hook_mlp_out, zeros, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        hooks.resid_mid_after_ln,
        expected_mid,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(replaced, expected_post, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="attention_output_override must have shape"):
        layer.forward_with_capture(
            x,
            alpha,
            torch.float32,
            attention_output_override=torch.zeros((1, 64, 1023)),
        )
    with pytest.raises(ValueError, match="mlp_output_override must have dtype"):
        layer.forward_with_capture(
            x,
            alpha,
            torch.float32,
            mlp_output_override=torch.zeros_like(x, dtype=torch.float64),
        )


def test_torch_layer_all_five_boundaries_are_intervenable() -> None:
    layer = _initialized_layer()
    x = torch.randn(
        (1, 64, 1024),
        generator=torch.Generator().manual_seed(17),
        dtype=torch.float32,
    )
    attention_input = torch.full_like(x, 1.0)
    attention_output = torch.full_like(x, 2.0)
    resid_mid = torch.full_like(x, 3.0)
    mlp_output = torch.full_like(x, 4.0)
    resid_post = torch.full_like(x, 5.0)

    with torch.inference_mode():
        output, hooks = layer.forward_with_capture(
            x,
            0.37,
            torch.float32,
            attention_input_override=attention_input,
            attention_output_override=attention_output,
            resid_mid_override=resid_mid,
            mlp_output_override=mlp_output,
            resid_post_override=resid_post,
        )

    expected = (
        attention_input,
        attention_output,
        resid_mid,
        mlp_output,
        resid_post,
    )
    for name, value in zip(HOOK_NAMES, expected, strict=True):
        torch.testing.assert_close(getattr(hooks, name), value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output, resid_post, rtol=0.0, atol=0.0)


class _ToyEmbedding(nn.Module):
    def forward(
        self,
        planes: torch.Tensor,
        alpha: float,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        del alpha
        return planes.to(compute_dtype)


class _ToyLayer(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(
        self,
        x: torch.Tensor,
        alpha: float,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        output, _capture = self.forward_with_capture(x, alpha, compute_dtype)
        return output

    def forward_with_capture(
        self,
        x: torch.Tensor,
        alpha: float,
        compute_dtype: torch.dtype,
        *,
        attention_input_override: torch.Tensor | None = None,
        attention_output_override: torch.Tensor | None = None,
        resid_mid_override: torch.Tensor | None = None,
        mlp_output_override: torch.Tensor | None = None,
        resid_post_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BT4EncoderLayerCapture]:
        del compute_dtype
        if attention_input_override is not None:
            x = attention_input_override
        attention = torch.full_like(x, float(self.index + 1))
        if attention_output_override is not None:
            attention = attention_output_override
        mid = x + alpha * attention
        if resid_mid_override is not None:
            mid = resid_mid_override
        mlp = torch.full_like(x, float(10 * (self.index + 1)))
        if mlp_output_override is not None:
            mlp = mlp_output_override
        post = mid + alpha * mlp
        if resid_post_override is not None:
            post = resid_post_override
        return post, BT4EncoderLayerCapture(x, attention, mid, mlp, post)


def _toy_encoder() -> BT4Encoder:
    encoder = BT4Encoder.__new__(BT4Encoder)
    nn.Module.__init__(encoder)
    encoder.embedding = _ToyEmbedding()
    encoder.layers = nn.ModuleList([_ToyLayer(0), _ToyLayer(1), _ToyLayer(2)])
    encoder.policy_head = None
    encoder.future_policy_heads = nn.ModuleList()
    encoder.alpha = 0.25
    return encoder


def test_encoder_capture_selects_layers_and_intervenes_on_uncaptured_layer() -> None:
    encoder = _toy_encoder()
    planes = torch.zeros((1, 64, 1024), dtype=torch.float32)
    tokens, captures = encoder.encode_current_with_captures(
        planes,
        compute_dtype=torch.float32,
        capture_layers=(0, 2),
        mlp_output_overrides={1: torch.zeros_like(planes)},
    )

    assert captures.layer_indices == (0, 2)
    for name in HOOK_NAMES:
        assert getattr(captures, name).shape == (2, 1, 64, 1024)
    expected = torch.full_like(planes, 11.5)
    torch.testing.assert_close(tokens, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        captures.resid_post_after_ln[-1],
        expected,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="unique and increasing"):
        encoder.encode_current_with_captures(
            planes,
            compute_dtype=torch.float32,
            capture_layers=(2, 0),
        )


def test_encoder_can_patch_each_boundary_on_uncaptured_layers() -> None:
    encoder = _toy_encoder()
    planes = torch.zeros((1, 64, 1024), dtype=torch.float32)
    patch = torch.full_like(planes, 100.0)
    expected_final = {
        "attention_input_overrides": 113.75,
        "attention_output_overrides": 41.0,
        "resid_mid_overrides": 113.25,
        "mlp_output_overrides": 36.5,
        "resid_post_overrides": 108.25,
    }
    for argument, expected in expected_final.items():
        tokens, captures = encoder.encode_current_with_captures(
            planes,
            compute_dtype=torch.float32,
            capture_layers=(2,),
            **{argument: {1: patch}},
        )
        assert captures.layer_indices == (2,)
        torch.testing.assert_close(
            tokens,
            torch.full_like(tokens, expected),
            rtol=0.0,
            atol=0.0,
        )
