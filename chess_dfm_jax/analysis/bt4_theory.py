"""Theoretical BT4 parameter and FLOP estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BT4Theory:
    parameter_count: int
    parameter_bytes_fp16: int
    parameter_bytes_fp32: int
    encoder_forward_flops: int
    full_forward_flops: int
    encoder_backward_flops_rule_of_thumb: int


def _count_params(tree) -> int:
    if isinstance(tree, dict):
        return sum(_count_params(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(_count_params(value) for value in tree)
    return int(np.asarray(tree).size)


def _linear_flops(batch_items: int, in_features: int, out_features: int) -> int:
    return 2 * batch_items * in_features * out_features


def _batched_matmul_flops(batch: int, heads: int, m: int, k: int, n: int) -> int:
    return 2 * batch * heads * m * k * n


def estimate_bt4_theory(params: dict, *, batch_size: int) -> BT4Theory:
    parameter_count = _count_params(params)
    parameter_bytes_fp16 = parameter_count * 2
    parameter_bytes_fp32 = parameter_count * 4

    embed = params["embedding"]
    encoder = params["encoder"]
    policy = params["policy"]
    value = params["value"]
    moves_left = params["moves_left"]

    seq = 64
    tokens = batch_size * seq
    headcount = params["headcount"]
    embedding_size = params["embedding_size"]
    embedding_dense_size = params["embedding_dense_size"]
    pos_planes = params["pos_planes"]

    encoder_forward_flops = 0
    encoder_forward_flops += _linear_flops(batch_size, seq * pos_planes, seq * embedding_dense_size)
    encoder_forward_flops += _linear_flops(tokens, params["input_channels"], embedding_size)
    encoder_forward_flops += _linear_flops(tokens, embedding_size, embed["ffn"]["dense1_b"].shape[0])
    encoder_forward_flops += _linear_flops(tokens, embed["ffn"]["dense1_b"].shape[0], embedding_size)

    for layer in encoder:
        d_model = int(layer["d_model"])
        depth = int(layer["depth"])
        ffn_hidden = int(layer["ffn"]["dense1_b"].shape[0])
        smol_hidden_channels = int(layer["mha"]["smolgen"]["compress_w"].shape[1])
        smol_hidden = int(layer["mha"]["smolgen"]["dense1_b"].shape[0])
        smol_gen = int(layer["mha"]["smolgen"]["dense2_b"].shape[0] // headcount)

        encoder_forward_flops += _linear_flops(tokens, embedding_size, d_model) * 3
        encoder_forward_flops += _batched_matmul_flops(batch_size, headcount, seq, depth, seq)
        encoder_forward_flops += _linear_flops(tokens, embedding_size, smol_hidden_channels)
        encoder_forward_flops += _linear_flops(batch_size, seq * smol_hidden_channels, smol_hidden)
        encoder_forward_flops += _linear_flops(batch_size, smol_hidden, headcount * smol_gen)
        encoder_forward_flops += _batched_matmul_flops(batch_size, headcount, 1, smol_gen, seq * seq)
        encoder_forward_flops += _batched_matmul_flops(batch_size, headcount, seq, seq, depth)
        encoder_forward_flops += _linear_flops(tokens, d_model, embedding_size)
        encoder_forward_flops += _linear_flops(tokens, embedding_size, ffn_hidden)
        encoder_forward_flops += _linear_flops(tokens, ffn_hidden, embedding_size)

    full_forward_flops = encoder_forward_flops
    full_forward_flops += _linear_flops(tokens, embedding_size, embedding_size)
    full_forward_flops += _linear_flops(tokens, embedding_size, embedding_size) * 2
    full_forward_flops += 2 * batch_size * seq * embedding_size * seq
    full_forward_flops += 2 * batch_size * 8 * embedding_size * 4
    full_forward_flops += _linear_flops(tokens, embedding_size, value["embed_b"].shape[0])
    full_forward_flops += _linear_flops(batch_size, seq * value["embed_b"].shape[0], value["dense1_b"].shape[0])
    full_forward_flops += _linear_flops(batch_size, value["dense1_b"].shape[0], value["dense2_b"].shape[0])
    full_forward_flops += _linear_flops(tokens, embedding_size, moves_left["embed_b"].shape[0])
    full_forward_flops += _linear_flops(batch_size, seq * moves_left["embed_b"].shape[0], moves_left["dense1_b"].shape[0])
    full_forward_flops += _linear_flops(batch_size, moves_left["dense1_b"].shape[0], moves_left["dense2_b"].shape[0])

    return BT4Theory(
        parameter_count=parameter_count,
        parameter_bytes_fp16=parameter_bytes_fp16,
        parameter_bytes_fp32=parameter_bytes_fp32,
        encoder_forward_flops=encoder_forward_flops,
        full_forward_flops=full_forward_flops,
        encoder_backward_flops_rule_of_thumb=3 * encoder_forward_flops,
    )


__all__ = ["BT4Theory", "estimate_bt4_theory"]
