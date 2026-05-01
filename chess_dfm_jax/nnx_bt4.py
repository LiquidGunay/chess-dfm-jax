"""Flax NNX BT4 encoder and full-model forward path."""

from __future__ import annotations

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def _stats_dtype(dtype) -> jnp.dtype:
    return jnp.float32 if jnp.dtype(dtype) in (jnp.float16, jnp.bfloat16) else jnp.dtype(dtype)


def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(jax.nn.softplus(x))


def swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


class TrainableParam(nnx.Param):
    """Marker class for JEPA trainable parameters."""
    pass


class FixedLinear(nnx.Module):
    def __init__(self, w, b=None, *, dtype=jnp.float32):
        self.dtype = jnp.dtype(dtype)
        self.w = nnx.Param(jnp.asarray(w, dtype=self.dtype))
        self.b = None if b is None else nnx.Param(jnp.asarray(b, dtype=self.dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(x, dtype=self.dtype) @ self.w[...]
        if self.b is not None:
            y = y + self.b[...]
        return y


class TrainableLayerNorm(nnx.Module):
    def __init__(self, width: int, *, param_dtype=jnp.float32, compute_dtype=jnp.float32, eps: float = 1e-6):
        self.width = width
        self.eps = eps
        self.param_dtype = jnp.dtype(param_dtype)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.scale = TrainableParam(jnp.ones((width,), dtype=self.param_dtype))
        self.bias = TrainableParam(jnp.zeros((width,), dtype=self.param_dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=self.compute_dtype)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x_hat = (x - mean) / jnp.sqrt(var + self.eps)
        out = x_hat * jnp.asarray(self.scale[...], dtype=self.compute_dtype) + jnp.asarray(self.bias[...], dtype=self.compute_dtype)
        return out


class TrainableEmbedding(nnx.Module):
    def __init__(self, num_embeddings: int, features: int, *, rngs: nnx.Rngs, param_dtype=jnp.float32, compute_dtype=jnp.float32):
        self.num_embeddings = num_embeddings
        self.features = features
        self.param_dtype = jnp.dtype(param_dtype)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.embedding = TrainableParam(
            jax.random.normal(rngs.params(), (num_embeddings, features), dtype=self.param_dtype)
            / np.sqrt(max(features, 1))
        )

    def __call__(self, indices: jnp.ndarray) -> jnp.ndarray:
        embedding = jnp.asarray(self.embedding[...], dtype=self.compute_dtype)
        return embedding[indices]


class FixedLayerNorm(nnx.Module):
    def __init__(self, scale, bias, eps: float = 1e-3, *, dtype=jnp.float32):
        self.dtype = jnp.dtype(dtype)
        self.scale = nnx.Param(jnp.asarray(scale, dtype=self.dtype))
        self.bias = nnx.Param(jnp.asarray(bias, dtype=self.dtype))
        self.eps = float(eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        stats_x = jnp.asarray(x, dtype=_stats_dtype(self.dtype))
        mean = jnp.mean(stats_x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(stats_x - mean), axis=-1, keepdims=True)
        x_hat = (stats_x - mean) / jnp.sqrt(var + self.eps)
        out = x_hat * jnp.asarray(self.scale[...], dtype=x_hat.dtype) + jnp.asarray(self.bias[...], dtype=x_hat.dtype)
        return jnp.asarray(out, dtype=self.dtype)


class InputEmbedding(nnx.Module):
    def __init__(
        self, emb_params: dict, *, embedding_size: int, embedding_dense_size: int, pos_planes: int, dtype=jnp.float32
    ):
        self.dtype = jnp.dtype(dtype)
        self.preproc = FixedLinear(emb_params["preproc_w"], emb_params["preproc_b"], dtype=self.dtype)
        self.proj = FixedLinear(emb_params["w"], emb_params["b"], dtype=self.dtype)
        self.ln = FixedLayerNorm(emb_params["ln_scale"], emb_params["ln_bias"], dtype=self.dtype)
        self.mul_gate = nnx.Param(jnp.asarray(emb_params["mul_gate"], dtype=self.dtype))
        self.add_gate = nnx.Param(jnp.asarray(emb_params["add_gate"], dtype=self.dtype))
        self.ffn1 = FixedLinear(emb_params["ffn"]["dense1_w"], emb_params["ffn"]["dense1_b"], dtype=self.dtype)
        self.ffn2 = FixedLinear(emb_params["ffn"]["dense2_w"], emb_params["ffn"]["dense2_b"], dtype=self.dtype)
        self.ffn_ln = FixedLayerNorm(emb_params["ffn_ln_scale"], emb_params["ffn_ln_bias"], dtype=self.dtype)
        self.embedding_size = int(embedding_size)
        self.embedding_dense_size = int(embedding_dense_size)
        self.pos_planes = int(pos_planes)

    def __call__(self, planes: jnp.ndarray, alpha: float) -> tuple[jnp.ndarray, int]:
        x = jnp.asarray(planes, dtype=self.dtype)
        if x.ndim == 3:
            x = x[None, ...]
        batch = x.shape[0]

        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.reshape((batch, 64, x.shape[-1]))
        pos = x[:, :, : self.pos_planes]
        pos = pos.reshape((batch, 64 * self.pos_planes))
        pos = self.preproc(pos)
        pos = pos.reshape((batch, 64, self.embedding_dense_size))
        x = jnp.concatenate([x, pos], axis=2)
        x = x.reshape((-1, x.shape[-1]))
        x = mish(self.proj(x))
        x = self.ln(x)
        x = x.reshape((batch, 64, self.embedding_size))
        x = x * self.mul_gate[...] + self.add_gate[...]
        x = x.reshape((-1, self.embedding_size))
        ffn = mish(self.ffn1(x))
        ffn = self.ffn2(ffn)
        x = self.ffn_ln(ffn * alpha + x)
        return x, batch


class Smolgen(nnx.Module):
    def __init__(self, params: dict, shared_w: np.ndarray, *, headcount: int, dtype=jnp.float32):
        self.dtype = jnp.dtype(dtype)
        self.headcount = headcount
        self.compress = FixedLinear(params["compress_w"], dtype=self.dtype)
        self.dense1 = FixedLinear(params["dense1_w"], params["dense1_b"], dtype=self.dtype)
        self.ln1 = FixedLayerNorm(params["ln1_scale"], params["ln1_bias"], dtype=self.dtype)
        self.dense2 = FixedLinear(params["dense2_w"], params["dense2_b"], dtype=self.dtype)
        self.ln2 = FixedLayerNorm(params["ln2_scale"], params["ln2_bias"], dtype=self.dtype)
        self.shared_w = nnx.Param(jnp.asarray(shared_w, dtype=self.dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [Batch, 64, D]
        batch = x.shape[0]
        s = self.compress(x) # [Batch, 64, HiddenChannels]
        s = s.reshape((batch, -1)) # [Batch, 64 * HiddenChannels]
        s = swish(self.dense1(s))
        s = self.ln1(s)
        s = swish(self.dense2(s))
        s = self.ln2(s)
        # s: [Batch, Headcount * SmolGenSz]
        s = s.reshape((batch, self.headcount, -1))
        s = s @ self.shared_w[...] # [Batch, Headcount, 64*64]
        return s.reshape((batch, self.headcount, 64, 64))


class EncoderLayer(nnx.Module):
    def __init__(
        self,
        *,
        width: int,
        num_heads: int,
        mlp_dim: int,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        use_qk_gain: bool = False,
        use_xsa: bool = False,
        attention_impl: str = "manual",
        layer_params: dict | None = None,
        shared_smolgen_w: np.ndarray | None = None,
    ):
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.param_dtype = jnp.dtype(param_dtype)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.use_qk_gain = use_qk_gain
        self.use_xsa = use_xsa
        if attention_impl not in {"sdpa", "manual"}:
            raise ValueError(f"Unsupported attention_impl: {attention_impl!r}")
        self.attention_impl = attention_impl

        if layer_params:
            self.wq = nnx.Param(jnp.asarray(layer_params["mha"]["q_w"], dtype=self.param_dtype))
            self.wq_b = nnx.Param(jnp.asarray(layer_params["mha"]["q_b"], dtype=self.param_dtype))
            self.wk = nnx.Param(jnp.asarray(layer_params["mha"]["k_w"], dtype=self.param_dtype))
            self.wk_b = nnx.Param(jnp.asarray(layer_params["mha"]["k_b"], dtype=self.param_dtype))
            self.wv = nnx.Param(jnp.asarray(layer_params["mha"]["v_w"], dtype=self.param_dtype))
            self.wv_b = nnx.Param(jnp.asarray(layer_params["mha"]["v_b"], dtype=self.param_dtype))
            self.wo = FixedLinear(layer_params["mha"]["dense_w"], layer_params["mha"]["dense_b"], dtype=self.param_dtype)
            self.ln_attn = FixedLayerNorm(layer_params["ln1"]["scale"], layer_params["ln1"]["bias"], dtype=self.param_dtype)
            self.ffn1 = FixedLinear(layer_params["ffn"]["dense1_w"], layer_params["ffn"]["dense1_b"], dtype=self.param_dtype)
            self.ffn2 = FixedLinear(layer_params["ffn"]["dense2_w"], layer_params["ffn"]["dense2_b"], dtype=self.param_dtype)
            self.ln_ffn = FixedLayerNorm(layer_params["ln2"]["scale"], layer_params["ln2"]["bias"], dtype=self.param_dtype)
            self.smolgen = Smolgen(layer_params["mha"]["smolgen"], shared_smolgen_w, headcount=num_heads, dtype=self.param_dtype) if shared_smolgen_w is not None else None
        else:
            self.wq = TrainableParam(jax.random.normal(rngs.params(), (width, width), dtype=self.param_dtype)/np.sqrt(width))
            self.wq_b = TrainableParam(jnp.zeros((width,), dtype=self.param_dtype))
            self.wk = TrainableParam(jax.random.normal(rngs.params(), (width, width), dtype=self.param_dtype)/np.sqrt(width))
            self.wk_b = TrainableParam(jnp.zeros((width,), dtype=self.param_dtype))
            self.wv = TrainableParam(jax.random.normal(rngs.params(), (width, width), dtype=self.param_dtype)/np.sqrt(width))
            self.wv_b = TrainableParam(jnp.zeros((width,), dtype=self.param_dtype))
            self.wo = TrainableParam(jax.random.normal(rngs.params(), (width, width), dtype=self.param_dtype)/np.sqrt(width))
            self.wo_b = TrainableParam(jnp.zeros((width,), dtype=self.param_dtype))
            self.ln_attn = TrainableLayerNorm(width, param_dtype=param_dtype, compute_dtype=compute_dtype)
            self.ffn1 = TrainableParam(jax.random.normal(rngs.params(), (width, mlp_dim), dtype=self.param_dtype)/np.sqrt(width))
            self.ffn1_b = TrainableParam(jnp.zeros((mlp_dim,), dtype=self.param_dtype))
            self.ffn2 = TrainableParam(jax.random.normal(rngs.params(), (mlp_dim, width), dtype=self.param_dtype)/np.sqrt(mlp_dim))
            self.ffn2_b = TrainableParam(jnp.zeros((width,), dtype=self.param_dtype))
            self.ln_ffn = TrainableLayerNorm(width, param_dtype=param_dtype, compute_dtype=compute_dtype)
            self.smolgen = None

        if use_qk_gain:
            self.qk_gain = TrainableParam(jnp.array(1.0 / np.sqrt(self.head_dim), dtype=self.param_dtype))

    def __call__(self, x: jnp.ndarray, alpha: float | None = None) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=self.compute_dtype)
        res = x
        batch, seq_len, _ = x.shape
        alpha_val = alpha if alpha is not None else 1.0
        
        # Attention
        q = x @ jnp.asarray(self.wq[...], dtype=self.compute_dtype)
        if hasattr(self, "wq_b"):
            q = q + jnp.asarray(self.wq_b[...], dtype=self.compute_dtype)
        k = x @ jnp.asarray(self.wk[...], dtype=self.compute_dtype)
        if hasattr(self, "wk_b"):
            k = k + jnp.asarray(self.wk_b[...], dtype=self.compute_dtype)
        v = x @ jnp.asarray(self.wv[...], dtype=self.compute_dtype)
        if hasattr(self, "wv_b"):
            v = v + jnp.asarray(self.wv_b[...], dtype=self.compute_dtype)

        q = q.reshape((batch, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = k.reshape((batch, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((batch, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)

        bias = self.smolgen(x) if self.smolgen else None
        if self.attention_impl == "sdpa" and not self.use_qk_gain:
            out = jax.nn.dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                bias=bias,
                implementation="xla",
            )
            out = out.transpose(0, 2, 1, 3)
        else:
            logits = jnp.matmul(q, k.transpose(0, 1, 3, 2))
            if self.use_qk_gain:
                logits = logits * self.qk_gain[...]
            else:
                logits = logits / np.sqrt(self.head_dim)
            if bias is not None:
                logits = logits + bias
            attn = jax.nn.softmax(logits, axis=-1)
            out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape((batch * seq_len, self.width))
        
        if hasattr(self, "wo_b"): # Trainable version
             out = (
                 out @ jnp.asarray(self.wo[...], dtype=self.compute_dtype)
                 + jnp.asarray(self.wo_b[...], dtype=self.compute_dtype)
             )
        else: # Fixed version
             out = self.wo(out)
        
        out = out.reshape((batch, seq_len, self.width)) * alpha_val
        x = self.ln_attn(out + res)
        
        # FFN
        res = x
        x_flat = x.reshape((batch * seq_len, self.width))
        if hasattr(self, "ffn1_b"): # Trainable JEPA version
            h = mish(
                x_flat @ jnp.asarray(self.ffn1[...], dtype=self.compute_dtype)
                + jnp.asarray(self.ffn1_b[...], dtype=self.compute_dtype)
            )
            out_flat = (
                h @ jnp.asarray(self.ffn2[...], dtype=self.compute_dtype)
                + jnp.asarray(self.ffn2_b[...], dtype=self.compute_dtype)
            )
        else: # Fixed BT4 version
            out_flat = self.ffn2(mish(self.ffn1(x_flat)))
            
        out = out_flat.reshape((batch, seq_len, self.width)) * alpha_val
        return self.ln_ffn(out + res)


class PolicyHead(nnx.Module):
    def __init__(self, params: dict, mapping_table: np.ndarray | None = None, *, dtype=jnp.float32):
        self.dtype = jnp.dtype(dtype)
        self.dense1 = FixedLinear(params["dense1_w"], params["dense1_b"], dtype=self.dtype)
        self.q = FixedLinear(params["q_w"], params["q_b"], dtype=self.dtype)
        self.k = FixedLinear(params["k_w"], params["k_b"], dtype=self.dtype)
        self.prom_w = nnx.Param(jnp.asarray(params["prom_w"], dtype=self.dtype))
        self.mapping_table = nnx.Param(jnp.asarray(mapping_table)) if mapping_table is not None else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [Batch, 64, Width]
        batch = x.shape[0]
        policy = mish(self.dense1(x))
        q = self.q(policy).reshape((batch, 64, -1))
        k = self.k(policy).reshape((batch, 64, -1))
        attn = jnp.matmul(q, k.transpose(0, 2, 1)) * (1.0 / math.sqrt(k.shape[-1]))

        prom = k[:, 56:64, :] @ self.prom_w[...]
        prom = prom.transpose(0, 2, 1)
        prom = prom[:, :3, :] + prom[:, 3:4, :]
        prom = prom.transpose(0, 2, 1).reshape((batch, 1, 24))

        sl = attn[:, 48:56, 56:64].reshape((batch, 64, 1))
        sl = jnp.concatenate([sl, sl, sl], axis=2).reshape((batch, 8, 24))
        prom = (sl + prom).reshape((batch, 3, 64))

        policy = jnp.concatenate([attn, prom], axis=1).reshape((batch, 67 * 64))
        if self.mapping_table is not None:
            policy = policy[:, self.mapping_table[...]]
        return policy


class ValueHead(nnx.Module):
    def __init__(self, params: dict, *, dtype=jnp.float32):
        self.dtype = jnp.dtype(dtype)
        self.dense1 = FixedLinear(params["embed_w"], params["embed_b"], dtype=self.dtype)
        self.dense2 = FixedLinear(params["dense1_w"], params["dense1_b"], dtype=self.dtype)
        self.out = FixedLinear(params["dense2_w"], params["dense2_b"], dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [Batch, 64, Width]
        batch = x.shape[0]
        x = mish(self.dense1(x))
        x = x.reshape((batch, -1))
        x = mish(self.dense2(x))
        return jax.nn.softmax(self.out(x), axis=-1)


class MovesLeftHead(nnx.Module):
    def __init__(self, params: dict, *, dtype=jnp.float32):
        self.dtype = jnp.dtype(dtype)
        self.dense1 = FixedLinear(params["embed_w"], params["embed_b"], dtype=self.dtype)
        self.dense2 = FixedLinear(params["dense1_w"], params["dense1_b"], dtype=self.dtype)
        self.out = FixedLinear(params["dense2_w"], params["dense2_b"], dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [Batch, 64, Width]
        batch = x.shape[0]
        x = mish(self.dense1(x))
        x = x.reshape((batch, -1))
        x = mish(self.dense2(x))
        return jax.nn.relu(self.out(x))


class BT4Model(nnx.Module):
    def __init__(self, params: dict, *, dtype=jnp.float32, attention_impl: str = "manual"):
        self.dtype = jnp.dtype(dtype)
        p = params
        self.embedding = InputEmbedding(
            p["embedding"],
            embedding_size=p["embedding_size"],
            embedding_dense_size=p["embedding_dense_size"],
            pos_planes=p["pos_planes"],
            dtype=self.dtype,
        )
        self.layers = nnx.List(
            [
                EncoderLayer(
                    width=p["embedding_size"],
                    num_heads=p["headcount"],
                    mlp_dim=p["embedding_size"] * 4,
                    rngs=nnx.Rngs(0),
                    param_dtype=self.dtype,
                    compute_dtype=self.dtype,
                    attention_impl=attention_impl,
                    layer_params=lp,
                    shared_smolgen_w=p["smolgen_w"],
                )
                for lp in p["encoder"]
            ]
        )
        self.policy_head = PolicyHead(p["policy"], p.get("mapping_table"), dtype=self.dtype)
        self.value_head = ValueHead(p["value"], dtype=self.dtype)
        self.moves_left_head = MovesLeftHead(p["moves_left"], dtype=self.dtype)
        self.embedding_size = p["embedding_size"]

    def encode_tokens(self, planes: jnp.ndarray, alpha: float | None = None) -> jnp.ndarray:
        if alpha is None:
            alpha = float(math.pow(2.0 * len(self.layers), -0.25)) if len(self.layers) > 0 else 1.0
        x, batch = self.embedding(planes, alpha)
        # Reshape for layers: [Batch, 64, Width]
        x = x.reshape((batch, 64, self.embedding_size))
        for layer in self.layers:
            x = layer(x, alpha)
        return x

    def __call__(self, planes: jnp.ndarray, alpha: float | None = None):
        x = self.encode_tokens(planes, alpha)
        # Heads receive full state or pooled state
        p = self.policy_head(x)
        v = self.value_head(x)
        ml = self.moves_left_head(x)
        return p, v, ml


def make_bt4_model(params: dict, *, dtype=jnp.float32, attention_impl: str = "manual") -> BT4Model:
    return BT4Model(params, dtype=dtype, attention_impl=attention_impl)


@nnx.jit
def jit_bt4_forward(model: BT4Model, planes: jnp.ndarray):
    return model(planes)


@nnx.jit
def jit_encode_tokens(model: BT4Model, planes: jnp.ndarray):
    return model.encode_tokens(planes)


def _encoder_surrogate_loss(model: BT4Model, planes: jnp.ndarray) -> jnp.ndarray:
    tokens = model.encode_tokens(planes)
    return jnp.mean(jnp.square(tokens))


_encoder_loss_and_grad = nnx.value_and_grad(_encoder_surrogate_loss, argnums=nnx.DiffState(0, nnx.Param))


@nnx.jit
def jit_encoder_loss_and_grad(model: BT4Model, planes: jnp.ndarray):
    return _encoder_loss_and_grad(model, planes)


def bt4_forward(params: dict, planes: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = make_bt4_model(params, dtype=jnp.float32)
    return jit_bt4_forward(model, jnp.asarray(planes, dtype=jnp.float32))


def bt4_forward_fp16(params: dict, planes: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = make_bt4_model(params, dtype=jnp.float16)
    return jit_bt4_forward(model, jnp.asarray(planes, dtype=jnp.float16))


def bt4_forward_fp32(params: dict, planes: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return bt4_forward(params, planes)


def muon_adamw(learning_rate, weight_decay):
    import optax.contrib
    return optax.contrib.muon(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_weight_decay=weight_decay
    )


__all__ = [
    "BT4Model",
    "EncoderLayer",
    "TrainableParam",
    "TrainableLayerNorm",
    "TrainableEmbedding",
    "bt4_forward",
    "bt4_forward_fp16",
    "bt4_forward_fp32",
    "jit_bt4_forward",
    "jit_encode_tokens",
    "jit_encoder_loss_and_grad",
    "make_bt4_model",
    "mish",
    "swish",
    "muon_adamw",
]
