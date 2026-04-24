"""Reference BT4 forward pass copied into the standalone scaffold.

This module is the ground-truth math path. Use it for parity checks while the
manual notebook version is still under construction.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(jax.nn.softplus(x))


def swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def bt4_forward(params: dict, planes: jnp.ndarray, *, capture: bool = False):
    """Run BT4 forward pass using mapped protobuf weights."""
    activations = {} if capture else None

    def save(name: str, value: jnp.ndarray) -> None:
        if activations is not None:
            activations[name] = value

    planes = jnp.asarray(planes, dtype=jnp.float32)
    if planes.ndim == 3:
        planes = planes[None, ...]
    batch = planes.shape[0]

    emb = params["embedding"]
    headcount = params["headcount"]
    embedding_size = params["embedding_size"]
    input_channels = params["input_channels"]
    pos_planes = params["pos_planes"]
    mapping_table = params.get("mapping_table")

    alpha = math.pow(2.0 * len(params["encoder"]), -0.25)
    eps = 1e-3

    x = jnp.transpose(planes, (0, 2, 3, 1))
    base_channels = x.shape[-1]
    x = x.reshape((batch, 64, base_channels))
    pos = x[:, :, :pos_planes]
    pos = pos.reshape((batch, 64 * pos_planes))
    pos = pos @ emb["preproc_w"] + emb["preproc_b"]
    pos = pos.reshape((batch, 64, params["embedding_dense_size"]))
    x = jnp.concatenate([x, pos], axis=2)
    x = x.reshape((-1, input_channels))
    x = x @ emb["w"] + emb["b"]
    x = mish(x)
    x = layer_norm(x, emb["ln_scale"], emb["ln_bias"], eps)
    x = x.reshape((batch, 64, embedding_size))
    x = x * emb["mul_gate"] + emb["add_gate"]
    x = x.reshape((-1, embedding_size))
    ffn = mish(x @ emb["ffn"]["dense1_w"] + emb["ffn"]["dense1_b"])
    ffn = ffn @ emb["ffn"]["dense2_w"] + emb["ffn"]["dense2_b"]
    ffn = ffn * alpha
    x = ffn + x
    x = layer_norm(x, emb["ffn_ln_scale"], emb["ffn_ln_bias"], eps)
    save("attn_body", x)

    for idx, layer in enumerate(params["encoder"]):
        x_in = x
        d_model = layer["d_model"]
        depth = layer["depth"]

        q = x @ layer["mha"]["q_w"] + layer["mha"]["q_b"]
        q = q.reshape((batch, 64, headcount, depth)).transpose(0, 2, 1, 3)
        k = x @ layer["mha"]["k_w"] + layer["mha"]["k_b"]
        k = k.reshape((batch, 64, headcount, depth)).transpose(0, 2, 3, 1)
        v = x @ layer["mha"]["v_w"] + layer["mha"]["v_b"]
        v = v.reshape((batch, 64, headcount, depth)).transpose(0, 2, 1, 3)

        attn = jnp.matmul(q, k) * (1.0 / math.sqrt(depth))

        smol_hidden_channels = layer["mha"]["smolgen"]["compress_w"].shape[1]
        smol_gen_sz = layer["mha"]["smolgen"]["dense2_b"].shape[0] // headcount
        smol = x @ layer["mha"]["smolgen"]["compress_w"]
        smol = smol.reshape((batch, 64 * smol_hidden_channels))
        smol = smol @ layer["mha"]["smolgen"]["dense1_w"] + layer["mha"]["smolgen"]["dense1_b"]
        smol = swish(smol)
        smol = layer_norm(
            smol,
            layer["mha"]["smolgen"]["ln1_scale"],
            layer["mha"]["smolgen"]["ln1_bias"],
            eps,
        )
        smol = smol @ layer["mha"]["smolgen"]["dense2_w"] + layer["mha"]["smolgen"]["dense2_b"]
        smol = swish(smol)
        smol = layer_norm(
            smol,
            layer["mha"]["smolgen"]["ln2_scale"],
            layer["mha"]["smolgen"]["ln2_bias"],
            eps,
        )
        smol = smol.reshape((batch, headcount, smol_gen_sz))
        smol = jnp.matmul(smol, params["smolgen_w"])
        smol = smol.reshape((batch, headcount, 64, 64))

        attn = jax.nn.softmax(attn + smol, axis=-1)
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape((-1, d_model))
        out = out @ layer["mha"]["dense_w"] + layer["mha"]["dense_b"]
        out = out * alpha
        x = layer_norm(out + x_in, layer["ln1"]["scale"], layer["ln1"]["bias"], eps)

        ffn = mish(x @ layer["ffn"]["dense1_w"] + layer["ffn"]["dense1_b"])
        ffn = ffn @ layer["ffn"]["dense2_w"] + layer["ffn"]["dense2_b"]
        ffn = ffn * alpha
        x = layer_norm(ffn + x, layer["ln2"]["scale"], layer["ln2"]["bias"], eps)
        save(f"encoder_{idx}", x)

    save("trunk", x)

    pol = params["policy"]
    policy = mish(x @ pol["dense1_w"] + pol["dense1_b"])
    q = policy @ pol["q_w"] + pol["q_b"]
    k = policy @ pol["k_w"] + pol["k_b"]
    q = q.reshape((batch, 64, -1))
    k = k.reshape((batch, 64, -1))
    attn = jnp.matmul(q, k.transpose(0, 2, 1)) * (1.0 / math.sqrt(k.shape[-1]))

    prom = k[:, 56:64, :] @ pol["prom_w"]
    prom = prom.transpose(0, 2, 1)
    prom = prom[:, :3, :] + prom[:, 3:4, :]
    prom = prom.transpose(0, 2, 1).reshape((batch, 1, 24))

    sl = attn[:, 48:56, 56:64].reshape((batch, 64, 1))
    sl = jnp.concatenate([sl, sl, sl], axis=2).reshape((batch, 8, 24))
    prom = (sl + prom).reshape((batch, 3, 64))

    policy = jnp.concatenate([attn, prom], axis=1).reshape((batch, 67 * 64))
    if mapping_table is not None:
        policy = policy[:, mapping_table]

    val = params["value"]
    value = mish(x @ val["embed_w"] + val["embed_b"])
    value = value.reshape((batch, 64 * val["embed_b"].shape[0]))
    value = mish(value @ val["dense1_w"] + val["dense1_b"])
    value = value @ val["dense2_w"] + val["dense2_b"]
    wdl = jax.nn.softmax(value, axis=-1)

    mlh = params["moves_left"]
    moves_left = mish(x @ mlh["embed_w"] + mlh["embed_b"])
    moves_left = moves_left.reshape((batch, 64 * mlh["embed_b"].shape[0]))
    moves_left = mish(moves_left @ mlh["dense1_w"] + mlh["dense1_b"])
    moves_left = moves_left @ mlh["dense2_w"] + mlh["dense2_b"]
    moves_left = jax.nn.relu(moves_left)

    if activations is None:
        return policy, wdl, moves_left
    return policy, wdl, moves_left, activations


def forward(params: dict, planes: jnp.ndarray, *, mask=None, capture: bool = False):
    outputs = bt4_forward(params, planes, capture=capture)
    if capture:
        policy, wdl, moves_left, activations = outputs
    else:
        policy, wdl, moves_left = outputs
        activations = None

    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        if mask_arr.ndim == 1:
            mask_arr = mask_arr[None, :]
        policy = jnp.where(mask_arr, policy, -1e9)

    if capture:
        return policy, wdl, moves_left, activations
    return policy, wdl, moves_left
