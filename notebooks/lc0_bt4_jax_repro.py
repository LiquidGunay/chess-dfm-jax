import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():

    import numpy as np
    import pandas as pd
    import jax.numpy as jnp
    import onnxruntime as ort
    import chess

    from chess_dfm_jax.encoding import encode_board
    from chess_dfm_jax.paths import default_bt4_paths
    from chess_dfm_jax.policy import attention_policy_map
    from chess_dfm_jax.reference_bt4 import (
        bt4_forward as reference_bt4_forward,
        layer_norm as reference_layer_norm,
        mish as reference_mish,
        swish as reference_swish,
    )
    from chess_dfm_jax.weights import load_pb_gz, map_bt4_weights

    return (
        attention_policy_map,
        chess,
        default_bt4_paths,
        encode_board,
        jnp,
        load_pb_gz,
        map_bt4_weights,
        np,
        ort,
        pd,
        reference_bt4_forward,
        reference_layer_norm,
        reference_mish,
        reference_swish,
    )


@app.cell
def _(marimo):
    marimo.md(
        "# LC0 BT4 Manual Reproduction\n\n"
        "This notebook is deliberately structured as a sequence of editable stages. "
        "Every exercise function currently delegates to the trusted reference math, "
        "so the whole notebook runs before you start rewriting anything.\n\n"
        "Recommended order:\n\n"
        "1. Replace `exercise_mish`, `exercise_swish`, and `exercise_layer_norm`.\n"
        "2. Replace `exercise_input_embedding`.\n"
        "3. Replace `exercise_encoder_stack`.\n"
        "4. Replace the three head functions.\n"
        "5. Run the parity tables again.\n"
    )
    return


@app.cell
def _(default_bt4_paths):
    MODEL_PATHS = default_bt4_paths()
    INPUT_FORMAT = "INPUT_CLASSICAL_112_PLANE"
    FENS = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/pp1nbppp/2p1pn2/2Pp4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 1 8",
        "8/5pk1/3p2p1/2pPp2p/2P1P2P/5PK1/8/8 w - - 0 40",
    ]
    return FENS, INPUT_FORMAT, MODEL_PATHS


@app.cell
def _(FENS, INPUT_FORMAT, MODEL_PATHS, marimo):
    marimo.md(
        f"""
    ## Working config

    - ONNX: `{MODEL_PATHS['onnx']}`
    - PB: `{MODEL_PATHS['exported_pb']}`
    - Input format: `{INPUT_FORMAT}`
    - Positions: `{len(FENS)}`
    """
    )
    return


@app.cell
def _(FENS, INPUT_FORMAT, chess, encode_board, np):
    boards = [chess.Board(fen) for fen in FENS]
    planes = np.stack([encode_board(board, [], input_format=INPUT_FORMAT) for board in boards], axis=0).astype(np.float32)
    return (planes,)


@app.cell
def _(MODEL_PATHS, attention_policy_map, load_pb_gz, map_bt4_weights):
    bundle = load_pb_gz(str(MODEL_PATHS["exported_pb"]))
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
    return (params,)


@app.cell
def _(reference_layer_norm, reference_mish, reference_swish):
    def exercise_mish(x):
        # TODO: replace this fallback with your own Mish implementation.
        return reference_mish(x)

    def exercise_swish(x):
        # TODO: replace this fallback with your own Swish implementation.
        return reference_swish(x)

    def exercise_layer_norm(x, scale, bias, eps):
        # TODO: replace this fallback with your own manual layer norm.
        return reference_layer_norm(x, scale, bias, eps)

    return exercise_layer_norm, exercise_mish, exercise_swish


@app.cell
def _(exercise_layer_norm, exercise_mish, np):
    def exercise_input_embedding(params, planes):
        # TODO: rewrite this cell from scratch once you want to own the math.
        batch = planes.shape[0]
        emb = params["embedding"]
        embedding_size = params["embedding_size"]
        input_channels = params["input_channels"]
        pos_planes = params["pos_planes"]
        eps = 1e-3

        x = np.transpose(planes, (0, 2, 3, 1))
        base_channels = x.shape[-1]
        x = x.reshape((batch, 64, base_channels))
        pos = x[:, :, :pos_planes]
        pos = pos.reshape((batch, 64 * pos_planes))
        pos = pos @ emb["preproc_w"] + emb["preproc_b"]
        pos = pos.reshape((batch, 64, params["embedding_dense_size"]))
        x = np.concatenate([x, pos], axis=2)
        x = x.reshape((-1, input_channels))
        x = x @ emb["w"] + emb["b"]
        x = exercise_mish(x)
        x = exercise_layer_norm(x, emb["ln_scale"], emb["ln_bias"], eps)
        x = x.reshape((batch, 64, embedding_size))
        x = x * emb["mul_gate"] + emb["add_gate"]
        return x

    return (exercise_input_embedding,)


@app.cell
def _(
    exercise_input_embedding,
    exercise_layer_norm,
    exercise_mish,
    exercise_swish,
    np,
):
    def exercise_encoder_stack(params, planes):
        # TODO: split this into input FFN, encoder blocks, and trunk capture if you
        # want a cleaner manual implementation.
        emb = params["embedding"]
        headcount = params["headcount"]
        alpha = float((2.0 * len(params["encoder"])) ** -0.25)
        eps = 1e-3
        batch = planes.shape[0]

        x = exercise_input_embedding(params, planes)
        x = x.reshape((-1, params["embedding_size"]))
        ffn = exercise_mish(x @ emb["ffn"]["dense1_w"] + emb["ffn"]["dense1_b"])
        ffn = ffn @ emb["ffn"]["dense2_w"] + emb["ffn"]["dense2_b"]
        x = exercise_layer_norm(ffn * alpha + x, emb["ffn_ln_scale"], emb["ffn_ln_bias"], eps)

        for layer in params["encoder"]:
            x_in = x
            depth = layer["depth"]
            d_model = layer["d_model"]

            q = x @ layer["mha"]["q_w"] + layer["mha"]["q_b"]
            q = q.reshape((batch, 64, headcount, depth)).transpose(0, 2, 1, 3)
            k = x @ layer["mha"]["k_w"] + layer["mha"]["k_b"]
            k = k.reshape((batch, 64, headcount, depth)).transpose(0, 2, 3, 1)
            v = x @ layer["mha"]["v_w"] + layer["mha"]["v_b"]
            v = v.reshape((batch, 64, headcount, depth)).transpose(0, 2, 1, 3)
            attn = np.matmul(q, k) * (1.0 / np.sqrt(depth))

            smol_hidden_channels = layer["mha"]["smolgen"]["compress_w"].shape[1]
            smol_gen_sz = layer["mha"]["smolgen"]["dense2_b"].shape[0] // headcount
            smol = x @ layer["mha"]["smolgen"]["compress_w"]
            smol = smol.reshape((batch, 64 * smol_hidden_channels))
            smol = smol @ layer["mha"]["smolgen"]["dense1_w"] + layer["mha"]["smolgen"]["dense1_b"]
            smol = exercise_swish(smol)
            smol = exercise_layer_norm(
                smol,
                layer["mha"]["smolgen"]["ln1_scale"],
                layer["mha"]["smolgen"]["ln1_bias"],
                eps,
            )
            smol = smol @ layer["mha"]["smolgen"]["dense2_w"] + layer["mha"]["smolgen"]["dense2_b"]
            smol = exercise_swish(smol)
            smol = exercise_layer_norm(
                smol,
                layer["mha"]["smolgen"]["ln2_scale"],
                layer["mha"]["smolgen"]["ln2_bias"],
                eps,
            )
            smol = smol.reshape((batch, headcount, smol_gen_sz))
            smol = np.matmul(smol, params["smolgen_w"])
            smol = smol.reshape((batch, headcount, 64, 64))

            attn = np.exp(attn + smol - np.max(attn + smol, axis=-1, keepdims=True))
            attn = attn / np.sum(attn, axis=-1, keepdims=True)
            out = np.matmul(attn, v)
            out = out.transpose(0, 2, 1, 3).reshape((-1, d_model))
            out = out @ layer["mha"]["dense_w"] + layer["mha"]["dense_b"]
            x = exercise_layer_norm(out * alpha + x_in, layer["ln1"]["scale"], layer["ln1"]["bias"], eps)

            ffn = exercise_mish(x @ layer["ffn"]["dense1_w"] + layer["ffn"]["dense1_b"])
            ffn = ffn @ layer["ffn"]["dense2_w"] + layer["ffn"]["dense2_b"]
            x = exercise_layer_norm(ffn * alpha + x, layer["ln2"]["scale"], layer["ln2"]["bias"], eps)

        return x, batch

    return (exercise_encoder_stack,)


@app.cell
def _(exercise_mish, np):
    def exercise_policy_head(params, trunk, batch):
        # TODO: rewrite the policy head manually once the trunk matches reference.
        pol = params["policy"]
        policy = exercise_mish(trunk @ pol["dense1_w"] + pol["dense1_b"])
        q = policy @ pol["q_w"] + pol["q_b"]
        k = policy @ pol["k_w"] + pol["k_b"]
        q = q.reshape((batch, 64, -1))
        k = k.reshape((batch, 64, -1))
        attn = np.matmul(q, k.transpose(0, 2, 1)) * (1.0 / np.sqrt(k.shape[-1]))

        prom = k[:, 56:64, :] @ pol["prom_w"]
        prom = prom.transpose(0, 2, 1)
        prom = prom[:, :3, :] + prom[:, 3:4, :]
        prom = prom.transpose(0, 2, 1).reshape((batch, 1, 24))

        sl = attn[:, 48:56, 56:64].reshape((batch, 64, 1))
        sl = np.concatenate([sl, sl, sl], axis=2).reshape((batch, 8, 24))
        prom = (sl + prom).reshape((batch, 3, 64))

        policy = np.concatenate([attn, prom], axis=1).reshape((batch, 67 * 64))
        mapping_table = params.get("mapping_table")
        if mapping_table is not None:
            policy = policy[:, mapping_table]
        return policy

    return (exercise_policy_head,)


@app.cell
def _(exercise_mish, np):
    def exercise_value_head(params, trunk, batch):
        val = params["value"]
        value = exercise_mish(trunk @ val["embed_w"] + val["embed_b"])
        value = value.reshape((batch, 64 * val["embed_b"].shape[0]))
        value = exercise_mish(value @ val["dense1_w"] + val["dense1_b"])
        value = value @ val["dense2_w"] + val["dense2_b"]
        shifted = value - value.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    return (exercise_value_head,)


@app.cell
def _(exercise_mish, np):
    def exercise_moves_left_head(params, trunk, batch):
        mlh = params["moves_left"]
        moves_left = exercise_mish(trunk @ mlh["embed_w"] + mlh["embed_b"])
        moves_left = moves_left.reshape((batch, 64 * mlh["embed_b"].shape[0]))
        moves_left = exercise_mish(moves_left @ mlh["dense1_w"] + mlh["dense1_b"])
        moves_left = moves_left @ mlh["dense2_w"] + mlh["dense2_b"]
        return np.maximum(moves_left, 0.0)

    return (exercise_moves_left_head,)


@app.cell
def _(
    exercise_encoder_stack,
    exercise_moves_left_head,
    exercise_policy_head,
    exercise_value_head,
):
    def manual_bt4_forward(params, planes):
        trunk, batch = exercise_encoder_stack(params, planes)
        policy = exercise_policy_head(params, trunk, batch)
        wdl = exercise_value_head(params, trunk, batch)
        moves_left = exercise_moves_left_head(params, trunk, batch)
        return policy, wdl, moves_left

    return (manual_bt4_forward,)


@app.cell
def _(jnp, manual_bt4_forward, np, params, planes, reference_bt4_forward):
    manual_policy, manual_wdl, manual_mlh = manual_bt4_forward(params, planes)
    reference_policy, reference_wdl, reference_mlh = reference_bt4_forward(params, jnp.asarray(planes))
    reference_policy = np.asarray(reference_policy)
    reference_wdl = np.asarray(reference_wdl)
    reference_mlh = np.asarray(reference_mlh)
    return (
        manual_mlh,
        manual_policy,
        manual_wdl,
        reference_mlh,
        reference_policy,
        reference_wdl,
    )


@app.cell
def _(
    manual_mlh,
    manual_policy,
    manual_wdl,
    np,
    pd,
    reference_mlh,
    reference_policy,
    reference_wdl,
):
    summary = pd.DataFrame(
        [
            {
                "tensor": "policy",
                "shape": tuple(manual_policy.shape),
                "max_abs_diff": float(np.abs(manual_policy - reference_policy).max()),
                "mean_abs_diff": float(np.abs(manual_policy - reference_policy).mean()),
            },
            {
                "tensor": "wdl",
                "shape": tuple(manual_wdl.shape),
                "max_abs_diff": float(np.abs(manual_wdl - reference_wdl).max()),
                "mean_abs_diff": float(np.abs(manual_wdl - reference_wdl).mean()),
            },
            {
                "tensor": "moves_left",
                "shape": tuple(manual_mlh.shape),
                "max_abs_diff": float(np.abs(manual_mlh - reference_mlh).max()),
                "mean_abs_diff": float(np.abs(manual_mlh - reference_mlh).mean()),
            },
        ]
    )
    summary
    return


@app.cell
def _(MODEL_PATHS, np, ort, planes):
    def pick(outputs, *names):
        for name in names:
            value = outputs.get(name)
            if value is not None:
                return value
        return None

    session = ort.InferenceSession(str(MODEL_PATHS["onnx"]), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    onnx_outputs = dict(zip(output_names, session.run(None, {input_name: planes.astype(np.float32)})))
    onnx_policy = pick(onnx_outputs, "/output/policy", "output/policy", "policy")
    onnx_wdl = pick(onnx_outputs, "/output/wdl", "output/wdl", "wdl")
    onnx_mlh = pick(onnx_outputs, "/output/mlh", "output/mlh", "mlh")
    return onnx_mlh, onnx_policy, onnx_wdl


@app.cell
def _(
    manual_mlh,
    manual_policy,
    manual_wdl,
    np,
    onnx_mlh,
    onnx_policy,
    onnx_wdl,
    pd,
):
    oracle_summary = pd.DataFrame(
        [
            {
                "tensor": "policy_vs_onnx",
                "shape": tuple(manual_policy.shape),
                "max_abs_diff": float(np.abs(manual_policy - onnx_policy).max()),
                "mean_abs_diff": float(np.abs(manual_policy - onnx_policy).mean()),
            },
            {
                "tensor": "wdl_vs_onnx",
                "shape": tuple(manual_wdl.shape),
                "max_abs_diff": float(np.abs(manual_wdl - onnx_wdl).max()),
                "mean_abs_diff": float(np.abs(manual_wdl - onnx_wdl).mean()),
            },
            {
                "tensor": "moves_left_vs_onnx",
                "shape": tuple(manual_mlh.shape),
                "max_abs_diff": float(np.abs(manual_mlh - onnx_mlh).max()),
                "mean_abs_diff": float(np.abs(manual_mlh - onnx_mlh).mean()),
            },
        ]
    )
    oracle_summary
    return


@app.cell(hide_code=True)
def _():
    import marimo

    marimo.md(
        "## Flax NNX GPU parity\n\n"
        "This section rebuilds BT4 as an `flax.nnx.Module`, runs it with `nnx.jit` on the visible GPU, and compares its outputs against both the current reference path and the ONNX oracle."
    )

    return (marimo,)


@app.cell
def _():
    import time
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx


    return jax, jnp, nnx, time


@app.cell
def _(jax, jnp, nnx):
    def nnx_mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))


    def nnx_swish(x):
        return x * jax.nn.sigmoid(x)


    class FixedLinear(nnx.Module):
        def __init__(self, w, b=None):
            self.w = nnx.Param(jnp.asarray(w, dtype=jnp.float32))
            self.b = None if b is None else nnx.Param(jnp.asarray(b, dtype=jnp.float32))

        def __call__(self, x):
            y = x @ self.w[...]
            if self.b is not None:
                y = y + self.b[...]
            return y


    class FixedLayerNorm(nnx.Module):
        def __init__(self, scale, bias, eps=1e-3):
            self.scale = nnx.Param(jnp.asarray(scale, dtype=jnp.float32))
            self.bias = nnx.Param(jnp.asarray(bias, dtype=jnp.float32))
            self.eps = float(eps)

        def __call__(self, x):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            x_hat = (x - mean) / jnp.sqrt(var + self.eps)
            return x_hat * self.scale[...] + self.bias[...]


    class InputEmbedding(nnx.Module):
        def __init__(self, emb_params, *, embedding_size, embedding_dense_size, pos_planes):
            self.preproc = FixedLinear(emb_params["preproc_w"], emb_params["preproc_b"])
            self.proj = FixedLinear(emb_params["w"], emb_params["b"])
            self.ln = FixedLayerNorm(emb_params["ln_scale"], emb_params["ln_bias"])
            self.mul_gate = nnx.Param(jnp.asarray(emb_params["mul_gate"], dtype=jnp.float32))
            self.add_gate = nnx.Param(jnp.asarray(emb_params["add_gate"], dtype=jnp.float32))
            self.ffn1 = FixedLinear(emb_params["ffn"]["dense1_w"], emb_params["ffn"]["dense1_b"])
            self.ffn2 = FixedLinear(emb_params["ffn"]["dense2_w"], emb_params["ffn"]["dense2_b"])
            self.ffn_ln = FixedLayerNorm(emb_params["ffn_ln_scale"], emb_params["ffn_ln_bias"])
            self.embedding_size = int(embedding_size)
            self.embedding_dense_size = int(embedding_dense_size)
            self.pos_planes = int(pos_planes)

        def __call__(self, planes, alpha):
            x = jnp.asarray(planes, dtype=jnp.float32)
            if x.ndim == 3:
                x = x[None, ...]
            batch = x.shape[0]

            x = jnp.transpose(x, (0, 2, 3, 1))
            base_channels = x.shape[-1]
            x = x.reshape((batch, 64, base_channels))
            pos = x[:, :, : self.pos_planes]
            pos = pos.reshape((batch, 64 * self.pos_planes))
            pos = self.preproc(pos)
            pos = pos.reshape((batch, 64, self.embedding_dense_size))
            x = jnp.concatenate([x, pos], axis=2)
            x = x.reshape((-1, x.shape[-1]))
            x = nnx_mish(self.proj(x))
            x = self.ln(x)
            x = x.reshape((batch, 64, self.embedding_size))
            x = x * self.mul_gate[...] + self.add_gate[...]
            x = x.reshape((-1, self.embedding_size))
            ffn = nnx_mish(self.ffn1(x))
            ffn = self.ffn2(ffn)
            x = self.ffn_ln(ffn * alpha + x)
            return x, batch


    class Smolgen(nnx.Module):
        def __init__(self, smol_params):
            self.compress = FixedLinear(smol_params["compress_w"])
            self.dense1 = FixedLinear(smol_params["dense1_w"], smol_params["dense1_b"])
            self.ln1 = FixedLayerNorm(smol_params["ln1_scale"], smol_params["ln1_bias"])
            self.dense2 = FixedLinear(smol_params["dense2_w"], smol_params["dense2_b"])
            self.ln2 = FixedLayerNorm(smol_params["ln2_scale"], smol_params["ln2_bias"])
            self.smol_hidden_channels = int(smol_params["compress_w"].shape[1])
            self.output_size = int(smol_params["dense2_b"].shape[0])

        def __call__(self, x, batch, headcount, smolgen_w):
            smol = self.compress(x)
            smol = smol.reshape((batch, 64 * self.smol_hidden_channels))
            smol = nnx_swish(self.dense1(smol))
            smol = self.ln1(smol)
            smol = nnx_swish(self.dense2(smol))
            smol = self.ln2(smol)
            smol_gen_sz = self.output_size // headcount
            smol = smol.reshape((batch, headcount, smol_gen_sz))
            smol = jnp.matmul(smol, smolgen_w)
            return smol.reshape((batch, headcount, 64, 64))


    class EncoderLayer(nnx.Module):
        def __init__(self, layer_params, headcount):
            self.q = FixedLinear(layer_params["mha"]["q_w"], layer_params["mha"]["q_b"])
            self.k = FixedLinear(layer_params["mha"]["k_w"], layer_params["mha"]["k_b"])
            self.v = FixedLinear(layer_params["mha"]["v_w"], layer_params["mha"]["v_b"])
            self.out = FixedLinear(layer_params["mha"]["dense_w"], layer_params["mha"]["dense_b"])
            self.smolgen = Smolgen(layer_params["mha"]["smolgen"])
            self.ln1 = FixedLayerNorm(layer_params["ln1"]["scale"], layer_params["ln1"]["bias"])
            self.ffn1 = FixedLinear(layer_params["ffn"]["dense1_w"], layer_params["ffn"]["dense1_b"])
            self.ffn2 = FixedLinear(layer_params["ffn"]["dense2_w"], layer_params["ffn"]["dense2_b"])
            self.ln2 = FixedLayerNorm(layer_params["ln2"]["scale"], layer_params["ln2"]["bias"])
            self.headcount = int(headcount)
            self.depth = int(layer_params["depth"])
            self.d_model = int(layer_params["d_model"])

        def __call__(self, x, batch, smolgen_w, alpha):
            x_in = x
            q = self.q(x).reshape((batch, 64, self.headcount, self.depth)).transpose(0, 2, 1, 3)
            k = self.k(x).reshape((batch, 64, self.headcount, self.depth)).transpose(0, 2, 3, 1)
            v = self.v(x).reshape((batch, 64, self.headcount, self.depth)).transpose(0, 2, 1, 3)

            attn = jnp.matmul(q, k) * (1.0 / jnp.sqrt(self.depth))
            smol = self.smolgen(x, batch, self.headcount, smolgen_w)
            attn = jax.nn.softmax(attn + smol, axis=-1)

            out = jnp.matmul(attn, v)
            out = out.transpose(0, 2, 1, 3).reshape((-1, self.d_model))
            out = self.out(out)
            x = self.ln1(out * alpha + x_in)

            ffn = nnx_mish(self.ffn1(x))
            ffn = self.ffn2(ffn)
            return self.ln2(ffn * alpha + x)


    class PolicyHead(nnx.Module):
        def __init__(self, pol_params, mapping_table):
            self.dense1 = FixedLinear(pol_params["dense1_w"], pol_params["dense1_b"])
            self.q = FixedLinear(pol_params["q_w"], pol_params["q_b"])
            self.k = FixedLinear(pol_params["k_w"], pol_params["k_b"])
            self.prom_w = nnx.Param(jnp.asarray(pol_params["prom_w"], dtype=jnp.float32))
            self.mapping_table = None if mapping_table is None else tuple(int(i) for i in mapping_table.tolist())

        def __call__(self, trunk, batch):
            policy = nnx_mish(self.dense1(trunk))
            q = self.q(policy).reshape((batch, 64, -1))
            k = self.k(policy).reshape((batch, 64, -1))
            attn = jnp.matmul(q, k.transpose(0, 2, 1)) * (1.0 / jnp.sqrt(k.shape[-1]))

            prom = k[:, 56:64, :] @ self.prom_w[...]
            prom = prom.transpose(0, 2, 1)
            prom = prom[:, :3, :] + prom[:, 3:4, :]
            prom = prom.transpose(0, 2, 1).reshape((batch, 1, 24))

            sl = attn[:, 48:56, 56:64].reshape((batch, 64, 1))
            sl = jnp.concatenate([sl, sl, sl], axis=2).reshape((batch, 8, 24))
            prom = (sl + prom).reshape((batch, 3, 64))

            policy = jnp.concatenate([attn, prom], axis=1).reshape((batch, 67 * 64))
            if self.mapping_table is not None:
                policy = policy[:, jnp.asarray(self.mapping_table, dtype=jnp.int32)]
            return policy


    class ValueHead(nnx.Module):
        def __init__(self, value_params):
            self.embed = FixedLinear(value_params["embed_w"], value_params["embed_b"])
            self.dense1 = FixedLinear(value_params["dense1_w"], value_params["dense1_b"])
            self.dense2 = FixedLinear(value_params["dense2_w"], value_params["dense2_b"])
            self.embed_size = int(value_params["embed_b"].shape[0])

        def __call__(self, trunk, batch):
            value = nnx_mish(self.embed(trunk))
            value = value.reshape((batch, 64 * self.embed_size))
            value = nnx_mish(self.dense1(value))
            return jax.nn.softmax(self.dense2(value), axis=-1)


    class MovesLeftHead(nnx.Module):
        def __init__(self, mlh_params):
            self.embed = FixedLinear(mlh_params["embed_w"], mlh_params["embed_b"])
            self.dense1 = FixedLinear(mlh_params["dense1_w"], mlh_params["dense1_b"])
            self.dense2 = FixedLinear(mlh_params["dense2_w"], mlh_params["dense2_b"])
            self.embed_size = int(mlh_params["embed_b"].shape[0])

        def __call__(self, trunk, batch):
            moves_left = nnx_mish(self.embed(trunk))
            moves_left = moves_left.reshape((batch, 64 * self.embed_size))
            moves_left = nnx_mish(self.dense1(moves_left))
            return jax.nn.relu(self.dense2(moves_left))


    class BT4NNX(nnx.Module):
        def __init__(self, params):
            self.embedding = InputEmbedding(
                params["embedding"],
                embedding_size=params["embedding_size"],
                embedding_dense_size=params["embedding_dense_size"],
                pos_planes=params["pos_planes"],
            )
            self.encoder = nnx.List([EncoderLayer(layer, params["headcount"]) for layer in params["encoder"]])
            self.policy_head = PolicyHead(params["policy"], params.get("mapping_table"))
            self.value_head = ValueHead(params["value"])
            self.moves_left_head = MovesLeftHead(params["moves_left"])
            self.smolgen_w = nnx.Param(jnp.asarray(params["smolgen_w"], dtype=jnp.float32))
            self.alpha = float((2.0 * len(params["encoder"])) ** -0.25)

        def encode(self, planes):
            x, batch = self.embedding(planes, self.alpha)
            smolgen_w = self.smolgen_w[...]
            for layer in self.encoder:
                x = layer(x, batch, smolgen_w, self.alpha)
            return x, batch

        def __call__(self, planes):
            trunk, batch = self.encode(planes)
            policy = self.policy_head(trunk, batch)
            wdl = self.value_head(trunk, batch)
            moves_left = self.moves_left_head(trunk, batch)
            return policy, wdl, moves_left


    @nnx.jit
    def nnx_bt4_forward(model, planes):
        return model(planes)


    @nnx.jit
    def nnx_bt4_forward_with_trunk(model, planes):
        trunk, batch = model.encode(planes)
        policy = model.policy_head(trunk, batch)
        wdl = model.value_head(trunk, batch)
        moves_left = model.moves_left_head(trunk, batch)
        return trunk, policy, wdl, moves_left


    return BT4NNX, nnx_bt4_forward_with_trunk


@app.cell
def _(BT4NNX, jax, jnp, nnx_bt4_forward_with_trunk, time):
    import marimo as mo
    import sys
    from pathlib import Path

    ROOT = Path("/home/ubuntu/chess-dfm-jax")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import chess
    import numpy as np
    import onnxruntime as ort
    import pandas as pd

    from chess_dfm_jax.encoding import encode_board
    from chess_dfm_jax.paths import default_bt4_paths
    from chess_dfm_jax.policy import attention_policy_map
    from chess_dfm_jax.reference_bt4 import bt4_forward as reference_bt4_forward
    from chess_dfm_jax.weights import load_pb_gz, map_bt4_weights


    def summarize_diff(name, lhs, rhs):
        diff = np.abs(lhs - rhs)
        return {
            "tensor": name,
            "shape": tuple(lhs.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        }


    def pick(outputs, *names):
        for name in names:
            value = outputs.get(name)
            if value is not None:
                return value
        return None


    MODEL_PATHS = default_bt4_paths()
    FENS = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/pp1nbppp/2p1pn2/2Pp4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 1 8",
        "8/5pk1/3p2p1/2pPp2p/2P1P2P/5PK1/8/8 w - - 0 40",
    ]

    boards = [chess.Board(fen) for fen in FENS]
    planes = np.stack([encode_board(board, [], input_format="INPUT_CLASSICAL_112_PLANE") for board in boards], axis=0).astype(np.float32)

    bundle = load_pb_gz(str(MODEL_PATHS["exported_pb"]))
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError("No GPU device visible to JAX.")

    gpu = gpu_devices[0]
    planes_jax = jax.device_put(jnp.asarray(planes, dtype=jnp.float32), gpu)
    nnx_model = BT4NNX(params)

    start = time.perf_counter()
    nnx_trunk, nnx_policy, nnx_wdl, nnx_mlh = nnx_bt4_forward_with_trunk(nnx_model, planes_jax)
    for arr in (nnx_trunk, nnx_policy, nnx_wdl, nnx_mlh):
        arr.block_until_ready()
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    nnx_trunk, nnx_policy, nnx_wdl, nnx_mlh = nnx_bt4_forward_with_trunk(nnx_model, planes_jax)
    for arr in (nnx_trunk, nnx_policy, nnx_wdl, nnx_mlh):
        arr.block_until_ready()
    steady_seconds = time.perf_counter() - start

    reference_policy, reference_wdl, reference_mlh, reference_activations = reference_bt4_forward(params, planes_jax, capture=True)
    reference_trunk = np.asarray(reference_activations["trunk"])
    reference_policy = np.asarray(reference_policy)
    reference_wdl = np.asarray(reference_wdl)
    reference_mlh = np.asarray(reference_mlh)

    session = ort.InferenceSession(str(MODEL_PATHS["onnx"]), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    onnx_outputs = dict(zip(output_names, session.run(None, {input_name: planes.astype(np.float32)})))
    onnx_policy = pick(onnx_outputs, "/output/policy", "output/policy", "policy")
    onnx_wdl = pick(onnx_outputs, "/output/wdl", "output/wdl", "wdl")
    onnx_mlh = pick(onnx_outputs, "/output/mlh", "output/mlh", "mlh")

    nnx_trunk_np = np.asarray(nnx_trunk)
    nnx_policy_np = np.asarray(nnx_policy)
    nnx_wdl_np = np.asarray(nnx_wdl)
    nnx_mlh_np = np.asarray(nnx_mlh)

    nnx_runtime = pd.DataFrame(
        [
            {"metric": "backend", "value": jax.default_backend()},
            {"metric": "device", "value": str(gpu)},
            {"metric": "compile_plus_first_run_seconds", "value": compile_seconds},
            {"metric": "steady_state_seconds", "value": steady_seconds},
        ]
    )

    nnx_summary = pd.DataFrame(
        [
            summarize_diff("trunk_vs_reference", nnx_trunk_np, reference_trunk),
            summarize_diff("policy_vs_reference", nnx_policy_np, reference_policy),
            summarize_diff("wdl_vs_reference", nnx_wdl_np, reference_wdl),
            summarize_diff("moves_left_vs_reference", nnx_mlh_np, reference_mlh),
            summarize_diff("policy_vs_onnx", nnx_policy_np, onnx_policy),
            summarize_diff("wdl_vs_onnx", nnx_wdl_np, onnx_wdl),
            summarize_diff("moves_left_vs_onnx", nnx_mlh_np, onnx_mlh),
        ]
    )

    mo.vstack([
        mo.md(
            f"""
    ### NNX parity report

    - Backend: `{jax.default_backend()}`
    - Device: `{gpu}`
    - Compile + first run: `{compile_seconds:.3f}s`
    - Steady-state run: `{steady_seconds:.3f}s`
    """
        ),
        nnx_runtime,
        nnx_summary,
    ])

    return (
        FENS,
        MODEL_PATHS,
        attention_policy_map,
        chess,
        default_bt4_paths,
        encode_board,
        load_pb_gz,
        map_bt4_weights,
        np,
        onnx_mlh,
        onnx_policy,
        onnx_wdl,
        ort,
        params,
        pd,
        planes,
        reference_bt4_forward,
        reference_mlh,
        reference_policy,
        reference_wdl,
    )


@app.cell
def _(marimo):
    marimo.md(
        "## Next steps\n\n"
        "- Keep replacing one helper cell at a time.\n"
        "- Once this notebook still matches ONNX after your edits, move the stable code into a reusable module.\n"
        "- Then use `notebooks/leela_data_pipeline.py` and `notebooks/training_roofline.py` to shape the training stack around the trunk.\n"
    )
    return


if __name__ == "__main__":
    app.run()
