"""LC0 .pb.gz weight loading and mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
import gzip

import numpy as np
from google.protobuf.descriptor import FieldDescriptor


@dataclass
class WeightsBundle:
    tensors: dict
    metadata: dict


def _decode_layer(layer, default_encoding: int) -> np.ndarray:
    encoding = layer.encoding if layer.HasField("encoding") else default_encoding
    params = layer.params
    if encoding == layer.LINEAR16:
        data = np.frombuffer(params, dtype=np.uint16).astype(np.float32)
        data = data / 65535.0
        data = data * (layer.max_val - layer.min_val) + layer.min_val
    elif encoding == layer.FLOAT16:
        data = np.frombuffer(params, dtype=np.float16).astype(np.float32)
    elif encoding == layer.BFLOAT16:
        raw = np.frombuffer(params, dtype=np.uint16).astype(np.uint32)
        data = (raw << 16).view(np.float32)
    else:
        raise ValueError(f"Unsupported layer encoding: {encoding}")

    dims = list(layer.dims)
    if dims:
        data = data.reshape(dims)
    return data


def _extract_layers(message, prefix: str, default_encoding: int, out: dict) -> None:
    if message.DESCRIPTOR.full_name == "pblczero.Weights.Layer":
        out[prefix] = _decode_layer(message, default_encoding)
        return

    for field, value in message.ListFields():
        name = f"{prefix}.{field.name}" if prefix else field.name
        if field.type != FieldDescriptor.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for idx, item in enumerate(value):
                _extract_layers(item, f"{name}[{idx}]", default_encoding, out)
        else:
            _extract_layers(value, name, default_encoding, out)


def load_pb_gz(path: str) -> WeightsBundle:
    """Parse .pb.gz into named tensors and metadata."""
    try:
        from chess_dfm_jax.proto import net_pb2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing protobuf bindings. The standalone scaffold ships with vendored protobuf bindings."
        ) from exc

    net = net_pb2.Net()
    with gzip.open(path, "rb") as f:
        net.ParseFromString(f.read())

    default_encoding = net.format.weights_encoding if net.HasField("format") else net_pb2.Format.LINEAR16

    tensors: dict[str, np.ndarray] = {}
    if net.HasField("weights"):
        _extract_layers(net.weights, "weights", default_encoding, tensors)

    network_format = net.format.network_format if net.HasField("format") else None
    if network_format is not None:
        input_embedding = network_format.input_embedding
        if not network_format.HasField("input_embedding") and net.HasField("weights"):
            if net.weights.HasField("policy_heads") and net.weights.HasField("value_heads"):
                input_embedding = net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
        nf = {
            "input": network_format.input,
            "output": network_format.output,
            "network": network_format.network,
            "policy": network_format.policy,
            "value": network_format.value,
            "moves_left": network_format.moves_left,
            "default_activation": network_format.default_activation,
            "smolgen_activation": network_format.smolgen_activation,
            "ffn_activation": network_format.ffn_activation,
            "input_embedding": input_embedding,
        }
    else:
        nf = None

    metadata = {
        "min_version": {
            "major": net.min_version.major,
            "minor": net.min_version.minor,
            "patch": net.min_version.patch,
        }
        if net.HasField("min_version")
        else None,
        "network_format": nf,
        "weights_encoding": default_encoding,
        "has_onnx": net.HasField("onnx_model"),
        "has_weights": net.HasField("weights"),
        "encoder_count": len(net.weights.encoder) if net.HasField("weights") else 0,
        "headcount": net.weights.headcount if net.HasField("weights") and net.weights.HasField("headcount") else None,
        "policy_headcount": (
            net.weights.policy_heads.vanilla.pol_headcount
            if net.HasField("weights")
            and net.weights.HasField("policy_heads")
            and net.weights.policy_heads.HasField("vanilla")
            else None
        ),
        "has_smolgen": net.weights.HasField("smolgen_w") if net.HasField("weights") else False,
    }
    return WeightsBundle(tensors=tensors, metadata=metadata)


def _reshape_mat(flat: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reshape and transpose LC0 matmul weights into [rows, cols]."""
    return flat.reshape(cols, rows).T.astype(np.float32, copy=False)


def _reshape_vec(flat: np.ndarray, size: int) -> np.ndarray:
    if flat.size != size:
        raise ValueError(f"Expected vector of size {size}, got {flat.size}")
    return flat.astype(np.float32, copy=False).reshape(size)


def map_bt4_weights(bundle: WeightsBundle, *, mapping_table: np.ndarray | None = None) -> dict:
    """Map BT4 weights into a structured dict for JAX inference."""
    t = bundle.tensors
    if "weights.ip_emb_b" not in t:
        raise ValueError("Missing BT4 input embedding weights.")

    embedding_size = t["weights.ip_emb_b"].size
    input_channels = t["weights.ip_emb_w"].size // embedding_size
    embedding_dense_size = t["weights.ip_emb_preproc_b"].size // 64
    preproc_total = t["weights.ip_emb_preproc_w"].size
    denom = 64 * embedding_dense_size * 64
    if preproc_total % denom != 0:
        raise ValueError("Unexpected ip_emb_preproc_w shape.")
    pos_planes = preproc_total // denom
    preproc_in = 64 * pos_planes
    preproc_out = 64 * embedding_dense_size

    headcount = bundle.metadata.get("headcount") or 32

    def mat(name: str, rows: int, cols: int) -> np.ndarray:
        if name not in t:
            raise KeyError(f"Missing weight: {name}")
        return _reshape_mat(t[name], rows, cols)

    def vec(name: str, size: int) -> np.ndarray:
        if name not in t:
            raise KeyError(f"Missing weight: {name}")
        return _reshape_vec(t[name], size)

    embedding = {
        "preproc_w": mat("weights.ip_emb_preproc_w", preproc_in, preproc_out),
        "preproc_b": vec("weights.ip_emb_preproc_b", preproc_out),
        "w": mat("weights.ip_emb_w", input_channels, embedding_size),
        "b": vec("weights.ip_emb_b", embedding_size),
        "ln_scale": vec("weights.ip_emb_ln_gammas", embedding_size),
        "ln_bias": vec("weights.ip_emb_ln_betas", embedding_size),
        "mul_gate": mat("weights.ip_mult_gate", 64, embedding_size),
        "add_gate": mat("weights.ip_add_gate", 64, embedding_size),
        "ffn": {
            "dense1_w": mat("weights.ip_emb_ffn.dense1_w", embedding_size, t["weights.ip_emb_ffn.dense1_b"].size),
            "dense1_b": vec("weights.ip_emb_ffn.dense1_b", t["weights.ip_emb_ffn.dense1_b"].size),
            "dense2_w": mat("weights.ip_emb_ffn.dense2_w", t["weights.ip_emb_ffn.dense1_b"].size, embedding_size),
            "dense2_b": vec("weights.ip_emb_ffn.dense2_b", embedding_size),
        },
        "ffn_ln_scale": vec("weights.ip_emb_ffn_ln_gammas", embedding_size),
        "ffn_ln_bias": vec("weights.ip_emb_ffn_ln_betas", embedding_size),
    }

    encoder_layers = []
    num_layers = bundle.metadata.get("encoder_count", 0)
    for idx in range(num_layers):
        prefix = f"weights.encoder[{idx}]"
        d_model = t[f"{prefix}.mha.q_b"].size
        depth = d_model // headcount
        smol_hidden_channels = t[f"{prefix}.mha.smolgen.compress"].size // embedding_size
        smol_hidden_sz = t[f"{prefix}.mha.smolgen.dense1_b"].size
        smol_gen_sz = t[f"{prefix}.mha.smolgen.dense2_b"].size // headcount
        ffn_hidden = t[f"{prefix}.ffn.dense1_b"].size
        layer = {
            "mha": {
                "q_w": mat(f"{prefix}.mha.q_w", embedding_size, d_model),
                "q_b": vec(f"{prefix}.mha.q_b", d_model),
                "k_w": mat(f"{prefix}.mha.k_w", embedding_size, d_model),
                "k_b": vec(f"{prefix}.mha.k_b", d_model),
                "v_w": mat(f"{prefix}.mha.v_w", embedding_size, d_model),
                "v_b": vec(f"{prefix}.mha.v_b", d_model),
                "dense_w": mat(f"{prefix}.mha.dense_w", d_model, embedding_size),
                "dense_b": vec(f"{prefix}.mha.dense_b", embedding_size),
                "smolgen": {
                    "compress_w": mat(
                        f"{prefix}.mha.smolgen.compress", embedding_size, smol_hidden_channels
                    ),
                    "dense1_w": mat(
                        f"{prefix}.mha.smolgen.dense1_w", 64 * smol_hidden_channels, smol_hidden_sz
                    ),
                    "dense1_b": vec(f"{prefix}.mha.smolgen.dense1_b", smol_hidden_sz),
                    "ln1_scale": vec(f"{prefix}.mha.smolgen.ln1_gammas", smol_hidden_sz),
                    "ln1_bias": vec(f"{prefix}.mha.smolgen.ln1_betas", smol_hidden_sz),
                    "dense2_w": mat(
                        f"{prefix}.mha.smolgen.dense2_w", smol_hidden_sz, smol_gen_sz * headcount
                    ),
                    "dense2_b": vec(f"{prefix}.mha.smolgen.dense2_b", smol_gen_sz * headcount),
                    "ln2_scale": vec(f"{prefix}.mha.smolgen.ln2_gammas", smol_gen_sz * headcount),
                    "ln2_bias": vec(f"{prefix}.mha.smolgen.ln2_betas", smol_gen_sz * headcount),
                },
            },
            "ln1": {
                "scale": vec(f"{prefix}.ln1_gammas", embedding_size),
                "bias": vec(f"{prefix}.ln1_betas", embedding_size),
            },
            "ffn": {
                "dense1_w": mat(f"{prefix}.ffn.dense1_w", embedding_size, ffn_hidden),
                "dense1_b": vec(f"{prefix}.ffn.dense1_b", ffn_hidden),
                "dense2_w": mat(f"{prefix}.ffn.dense2_w", ffn_hidden, embedding_size),
                "dense2_b": vec(f"{prefix}.ffn.dense2_b", embedding_size),
            },
            "ln2": {
                "scale": vec(f"{prefix}.ln2_gammas", embedding_size),
                "bias": vec(f"{prefix}.ln2_betas", embedding_size),
            },
            "d_model": d_model,
            "depth": depth,
        }
        encoder_layers.append(layer)

    policy = {
        "dense1_w": mat("weights.policy_heads.ip_pol_w", embedding_size, embedding_size),
        "dense1_b": vec("weights.policy_heads.ip_pol_b", embedding_size),
        "q_w": mat("weights.policy_heads.vanilla.ip2_pol_w", embedding_size, embedding_size),
        "q_b": vec("weights.policy_heads.vanilla.ip2_pol_b", embedding_size),
        "k_w": mat("weights.policy_heads.vanilla.ip3_pol_w", embedding_size, embedding_size),
        "k_b": vec("weights.policy_heads.vanilla.ip3_pol_b", embedding_size),
        "prom_w": mat("weights.policy_heads.vanilla.ip4_pol_w", embedding_size, 4),
    }

    value = {
        "embed_w": mat("weights.value_heads.winner.ip_val_w", embedding_size, 128),
        "embed_b": vec("weights.value_heads.winner.ip_val_b", 128),
        "dense1_w": mat("weights.value_heads.winner.ip1_val_w", 128 * 64, 128),
        "dense1_b": vec("weights.value_heads.winner.ip1_val_b", 128),
        "dense2_w": mat("weights.value_heads.winner.ip2_val_w", 128, 3),
        "dense2_b": vec("weights.value_heads.winner.ip2_val_b", 3),
    }

    moves_left = {
        "embed_w": mat("weights.ip_mov_w", embedding_size, 32),
        "embed_b": vec("weights.ip_mov_b", 32),
        "dense1_w": mat("weights.ip1_mov_w", 32 * 64, 128),
        "dense1_b": vec("weights.ip1_mov_b", 128),
        "dense2_w": mat("weights.ip2_mov_w", 128, 1),
        "dense2_b": vec("weights.ip2_mov_b", 1),
    }

    smolgen_rows = t["weights.smolgen_w"].size // (64 * 64)
    smolgen_w = mat("weights.smolgen_w", smolgen_rows, 64 * 64)

    params = {
        "embedding": embedding,
        "encoder": encoder_layers,
        "policy": policy,
        "value": value,
        "moves_left": moves_left,
        "smolgen_w": smolgen_w,
        "headcount": headcount,
        "embedding_size": embedding_size,
        "input_channels": input_channels,
        "embedding_dense_size": embedding_dense_size,
        "pos_planes": pos_planes,
        "mapping_table": mapping_table,
    }
    return params
