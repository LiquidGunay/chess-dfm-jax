"""Theory-side parameter and FLOP estimates for the token-level JEPA head."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JEPATheory:
    trainable_parameter_count: int
    trainable_parameter_bytes_bf16: int
    trainable_parameter_bytes_fp32: int
    forward_flops: int
    train_step_flops_rule_of_thumb: int


def _linear_params(in_features: int, out_features: int, *, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def _linear_flops(batch_items: int, in_features: int, out_features: int) -> int:
    return 2 * batch_items * in_features * out_features


def _attention_flops(batch: int, tokens: int, width: int, heads: int) -> int:
    head_dim = width // heads
    qkv = 3 * _linear_flops(batch * tokens, width, width)
    scores = 2 * batch * heads * tokens * tokens * head_dim
    attend = 2 * batch * heads * tokens * tokens * head_dim
    out = _linear_flops(batch * tokens, width, width)
    return qkv + scores + attend + out


def estimate_jepa_theory(
    *,
    batch_size: int,
    token_dim: int,
    num_layers: int,
    num_heads: int,
    mlp_dim: int,
    action_vocab_size: int = 1858,
    encoder_width: int = 1024,
    board_tokens: int = 64,
) -> JEPATheory:
    seq = board_tokens + 1
    jepa_width = encoder_width
    action_embed_dim = 128
    action_hidden_dim = max(token_dim * 2, jepa_width)
    swiglu_dim = max(1, int(round((2.0 / 3.0) * mlp_dim)))

    trainable_params = 0
    trainable_params += action_vocab_size * action_embed_dim  # action embedding
    trainable_params += _linear_params(action_embed_dim, action_hidden_dim)
    trainable_params += _linear_params(action_hidden_dim, jepa_width)

    # output RMSNorm
    trainable_params += jepa_width

    for _ in range(num_layers):
        # two RMSNorm scales
        trainable_params += 2 * jepa_width
        # fused qkv + out
        trainable_params += _linear_params(jepa_width, 3 * jepa_width)
        trainable_params += _linear_params(jepa_width, jepa_width)
        # fused SwiGLU gate/up + down. mlp_dim is the old dense-FFN equivalent,
        # so swiglu_dim ~= 2/3 * mlp_dim gives parameter/FLOP parity.
        trainable_params += _linear_params(jepa_width, 2 * swiglu_dim)
        trainable_params += _linear_params(swiglu_dim, jepa_width)

    forward_flops = 0
    forward_flops += _linear_flops(batch_size, action_embed_dim, action_hidden_dim)
    forward_flops += _linear_flops(batch_size, action_hidden_dim, jepa_width)
    for _ in range(num_layers):
        forward_flops += _attention_flops(batch_size, seq, jepa_width, num_heads)
        forward_flops += _linear_flops(batch_size * seq, jepa_width, 2 * swiglu_dim)
        forward_flops += _linear_flops(batch_size * seq, swiglu_dim, jepa_width)

    # rule of thumb: forward pass plus backward through trainable head is about 3x forward
    return JEPATheory(
        trainable_parameter_count=trainable_params,
        trainable_parameter_bytes_bf16=trainable_params * 2,
        trainable_parameter_bytes_fp32=trainable_params * 4,
        forward_flops=forward_flops,
        train_step_flops_rule_of_thumb=3 * forward_flops,
    )


__all__ = ["JEPATheory", "estimate_jepa_theory"]
