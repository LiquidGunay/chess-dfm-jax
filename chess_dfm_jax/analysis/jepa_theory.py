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

    trainable_params = 0
    trainable_params += _linear_params(encoder_width, token_dim)  # token projector
    trainable_params += action_vocab_size * token_dim  # action embedding
    trainable_params += token_dim  # action token bias
    trainable_params += board_tokens * token_dim  # square positions

    # output norm
    trainable_params += 2 * token_dim

    for _ in range(num_layers):
        # two layer norms
        trainable_params += 4 * token_dim
        # q, k, v, out
        trainable_params += 4 * _linear_params(token_dim, token_dim)
        # MLP up/down
        trainable_params += _linear_params(token_dim, mlp_dim)
        trainable_params += _linear_params(mlp_dim, token_dim)

    forward_flops = 0
    forward_flops += 2 * _linear_flops(batch_size * board_tokens, encoder_width, token_dim)  # current + target projector
    for _ in range(num_layers):
        forward_flops += _attention_flops(batch_size, seq, token_dim, num_heads)
        forward_flops += _linear_flops(batch_size * seq, token_dim, mlp_dim)
        forward_flops += _linear_flops(batch_size * seq, mlp_dim, token_dim)

    # rule of thumb: forward pass plus backward through trainable head is about 3x forward
    return JEPATheory(
        trainable_parameter_count=trainable_params,
        trainable_parameter_bytes_bf16=trainable_params * 2,
        trainable_parameter_bytes_fp32=trainable_params * 4,
        forward_flops=forward_flops,
        train_step_flops_rule_of_thumb=3 * forward_flops,
    )


__all__ = ["JEPATheory", "estimate_jepa_theory"]
