"""Deterministically convert a trusted Leela-SAEs LoRSA DCP to safetensors.

The DCP reader deserializes pickle metadata.  Run this CLI in the dedicated
conversion process documented by ``--trust-paper-checkpoint``; normal
experiments consume only the resulting safetensors and JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader, load as dcp_load
from torch.nn import functional as F

from research.interpretability.artifacts import sha256_file
from research.interpretability.sparse.lorsa import (
    CONVERTED_LORSA_SCHEMA,
    LoRSAConfig,
    LowRankSparseAttention,
    fold_lorsa_dataset_normalization,
    load_converted_lorsa,
)


HF_REPO_ID = "JacklE0niden/lc0-BT4-lorsa"
HF_REPO_REVISION = "1ec52be63cce41017b3853edbe9421b643a1476d"
HF_FILE_COMMIT = "fb30623060c46c6721bab2f792c6e08ea13f973b"
UPSTREAM_CODE_COMMIT = "f946a5736f40397f5c834104aeac6e8e6ff753bd"
PUBLISHED_CHECKPOINT_SUBPATH = "k_30_e_16/L14"

# SHA-256 and byte sizes independently verified against the current HF tree.
PUBLISHED_SOURCE_FILES: dict[str, tuple[int, str]] = {
    "config.json": (
        1_215,
        "f465c27afc460ba8c741cec2ebc1b6cada9d6c297dd33615f2f5108553fefad1",
    ),
    "sae_weights.dcp/.metadata": (
        10_516,
        "e61ba568b64ea291eeabb5614456bfd0053baf4aa50201b098cccf1360b55e59",
    ),
    "sae_weights.dcp/__0_0.distcp": (
        42_018_673,
        "66959912d7a77006d9334e6ef2edea78d04ca41ad111f69fdcfd5ec1c5edcdf5",
    ),
    "sae_weights.dcp/__1_0.distcp": (
        44_126_229,
        "985a6e938b442a6f5ca3750448ec4188f08596b5c368b1a5821437d447c144a5",
    ),
    "sae_weights.dcp/__2_0.distcp": (
        46_749_881,
        "3fc3678ce65bd75193bcbf8c3466441d2e75cf989a07fec1453b0ec4e1bd00d3",
    ),
    "sae_weights.dcp/__3_0.distcp": (
        75_588_158,
        "53e34b84428070f05c7098fe555cecdf90a996534216495a15804a6d269dc1ba",
    ),
}

_RUNTIME_ONLY_DCP_KEYS = {
    "IGNORE",
    "mask",
    "tokens_since_last_activation",
    "is_dead",
}
_NORM_PREFIX = "dataset_average_activation_norm."


@dataclass(frozen=True)
class ParsedPublishedConfig:
    module: LoRSAConfig
    layer: int
    hook_point_in: str
    hook_point_out: str
    raw: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def parse_published_config(path: Path) -> ParsedPublishedConfig:
    raw = _read_json_object(path.resolve(strict=True))
    required = {
        "sae_type",
        "d_model",
        "expansion_factor",
        "use_decoder_bias",
        "act_fn",
        "norm_activation",
        "sparsity_include_decoder_norm",
        "top_k",
        "hook_point_in",
        "hook_point_out",
        "n_qk_heads",
        "d_qk_head",
        "positional_embedding_type",
        "n_ctx",
        "skip_bos",
        "attn_scale",
        "use_post_qk_ln",
        "use_smolgen",
        "use_learnable_attn_scale",
    }
    if missing := sorted(required - set(raw)):
        raise ValueError(f"Published LoRSA config is missing fields: {missing}")
    if raw["sae_type"] != "lorsa" or raw["act_fn"] != "topk":
        raise ValueError("Only published TopK LoRSA checkpoints are supported")
    if raw["norm_activation"] != "dataset-wise":
        raise ValueError("Published conversion expects an unfolded dataset-wise checkpoint")
    if not raw["sparsity_include_decoder_norm"]:
        raise ValueError("Published LoRSA requires decoder-norm-aware sparsity")
    if raw["positional_embedding_type"] != "none" or raw["use_post_qk_ln"]:
        raise ValueError("Rotary embeddings and post-QK normalization are outside this ABI")
    if raw["skip_bos"]:
        raise ValueError("BT4 LoRSA must retain all 64 square tokens")

    hook_in = str(raw["hook_point_in"])
    hook_out = str(raw["hook_point_out"])
    in_parts = hook_in.split(".")
    out_parts = hook_out.split(".")
    if (
        len(in_parts) != 3
        or len(out_parts) != 3
        or in_parts[:2] != out_parts[:2]
        or in_parts[0] != "blocks"
        or in_parts[2] != "hook_attn_in"
        or out_parts[2] != "hook_attn_out"
    ):
        raise ValueError("LoRSA hooks do not identify one BT4 attention branch")
    layer = int(in_parts[1])
    d_model = int(raw["d_model"])
    n_ov_heads = int(round(d_model * float(raw["expansion_factor"])))
    config = LoRSAConfig(
        d_model=d_model,
        n_ctx=int(raw["n_ctx"]),
        n_qk_heads=int(raw["n_qk_heads"]),
        d_qk_head=int(raw["d_qk_head"]),
        n_ov_heads=n_ov_heads,
        top_k=int(raw["top_k"]),
        attn_scale=float(raw["attn_scale"]),
        use_smolgen=bool(raw["use_smolgen"]),
        smolgen_score_scale=1.0,
        use_learnable_attn_scale=bool(raw["use_learnable_attn_scale"]),
        use_decoder_bias=bool(raw["use_decoder_bias"]),
    )
    config.validate()
    return ParsedPublishedConfig(config, layer, hook_in, hook_out, raw)


def source_file_inventory(source_dir: Path) -> dict[str, dict[str, Any]]:
    source_dir = source_dir.resolve(strict=True)
    paths = sorted(path for path in source_dir.rglob("*") if path.is_file())
    return {
        path.relative_to(source_dir).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def verify_published_source_files(source_dir: Path) -> dict[str, dict[str, Any]]:
    inventory = source_file_inventory(source_dir)
    if set(inventory) != set(PUBLISHED_SOURCE_FILES):
        raise ValueError(
            "Published source inventory mismatch: "
            f"expected={sorted(PUBLISHED_SOURCE_FILES)}, got={sorted(inventory)}"
        )
    for relative_path, (size_bytes, digest) in PUBLISHED_SOURCE_FILES.items():
        actual = inventory[relative_path]
        if actual != {"size_bytes": size_bytes, "sha256": digest}:
            raise ValueError(f"Published source identity mismatch for {relative_path}")
    return inventory


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def load_trusted_dcp_tensors(
    dcp_dir: Path,
    *,
    trust_paper_checkpoint: bool,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Load a DCP only after an explicit trust opt-in in the converter process."""

    if not trust_paper_checkpoint:
        raise ValueError("DCP conversion requires --trust-paper-checkpoint")
    dcp_dir = dcp_dir.resolve(strict=True)
    reader = FileSystemReader(dcp_dir)
    metadata = reader.read_metadata()
    tensors: dict[str, Tensor] = {}
    metadata_inventory: dict[str, Any] = {}
    total_bytes = 0
    allowed_dtypes = {torch.float32, torch.bool, torch.int64}
    for name, spec in sorted(metadata.state_dict_metadata.items()):
        size = getattr(spec, "size", None)
        properties = getattr(spec, "properties", None)
        dtype = getattr(properties, "dtype", None)
        if size is None or dtype not in allowed_dtypes:
            raise ValueError(f"Unsupported DCP entry {name!r}: {type(spec).__name__}, {dtype}")
        shape = tuple(int(dimension) for dimension in size)
        numel = math.prod(shape)
        total_bytes += numel * _dtype_size(dtype)
        metadata_inventory[name] = {
            "shape": list(shape),
            "dtype": str(dtype).removeprefix("torch."),
            "numel": numel,
        }
        tensors[name] = torch.empty(shape, dtype=dtype, device="cpu")
    if total_bytes <= 0 or total_bytes > 2_000_000_000:
        raise ValueError(f"Refusing implausible LoRSA DCP allocation of {total_bytes} bytes")
    dcp_load(tensors, storage_reader=reader)
    return tensors, {
        "tensor_count": len(tensors),
        "tensor_bytes": total_bytes,
        "tensor_inventory": metadata_inventory,
        "planner_keys": sorted((metadata.planner_data or {}).keys()),
        "storage_checkpoint_id": str(getattr(metadata.storage_meta, "checkpoint_id", "")),
        "storage_save_id": str(getattr(metadata.storage_meta, "save_id", "")),
    }


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _tensor_inventory(state: Mapping[str, Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "numel": value.numel(),
            "sha256": _tensor_sha256(value),
        }
        for name, value in sorted(state.items())
    }


def extract_and_fold_state(
    dcp_state: Mapping[str, Tensor],
    parsed: ParsedPublishedConfig,
) -> tuple[dict[str, Tensor], LoRSAConfig, dict[str, Any]]:
    hook_norm_keys = {
        f"{_NORM_PREFIX}{parsed.hook_point_in}",
        f"{_NORM_PREFIX}{parsed.hook_point_out}",
    }
    model_config = parsed.module
    score_scale = dcp_state.get("smolgen_score_scale")
    if model_config.use_smolgen:
        if score_scale is None or score_scale.shape != () or score_scale.dtype != torch.float32:
            raise ValueError("DCP is missing its FP32 scalar SmolGen score scale")
        model_config = replace(model_config, smolgen_score_scale=float(score_scale))
    expected_model = LowRankSparseAttention(model_config, device="meta")
    expected_state = expected_model.state_dict()
    allowed = set(expected_state) | _RUNTIME_ONLY_DCP_KEYS | hook_norm_keys
    missing = set(expected_state) - set(dcp_state)
    unexpected = set(dcp_state) - allowed
    if missing or unexpected:
        raise ValueError(
            f"LoRSA DCP state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    unfolded: dict[str, Tensor] = {}
    for name, expected in expected_state.items():
        value = dcp_state[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(
                f"LoRSA {name} expected {expected.dtype} {tuple(expected.shape)}, "
                f"got {value.dtype} {tuple(value.shape)}"
            )
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"LoRSA model tensor {name} contains nonfinite values")
        unfolded[name] = value.detach().cpu().contiguous()

    if model_config.use_learnable_attn_scale:
        learned_scale = float(unfolded["_attn_scale_param"])
        if learned_scale <= 0 or not math.isfinite(learned_scale):
            raise ValueError("Learned attention scale must be finite and positive")
    else:
        learned_scale = model_config.attn_scale

    input_norm = float(dcp_state[f"{_NORM_PREFIX}{parsed.hook_point_in}"])
    output_norm = float(dcp_state[f"{_NORM_PREFIX}{parsed.hook_point_out}"])
    folded = fold_lorsa_dataset_normalization(
        unfolded,
        input_norm=input_norm,
        output_norm=output_norm,
        d_model=model_config.d_model,
    )
    return (
        folded,
        model_config,
        {
            "source_mode": parsed.raw["norm_activation"],
            "converted_mode": "inference",
            "folded": True,
            "input_norm": input_norm,
            "output_norm": output_norm,
            "c_in": math.sqrt(model_config.d_model) / input_norm,
            "c_out": math.sqrt(model_config.d_model) / output_norm,
            "rule": "W_Q/W_K/W_V *= c_in; W_O/b_D /= c_out",
            "learned_attention_scale": learned_scale,
            "smolgen_score_scale": model_config.smolgen_score_scale,
        },
    )


def deterministic_parity_input(config: LoRSAConfig) -> Tensor:
    index = torch.arange(config.n_ctx * config.d_model, dtype=torch.int64)
    integer = (index * 25_173 + 13_849).remainder(65_536) - 32_768
    return (integer.to(torch.float32) / 131_072.0).reshape(1, config.n_ctx, config.d_model)


def _reference_layer_norm(value: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    mean = value.mean(dim=-1, keepdim=True)
    variance = value.var(dim=-1, unbiased=False, keepdim=True)
    return (value - mean) * torch.rsqrt(variance + 1e-5) * weight + bias


def _reference_positive_topk(value: Tensor, k: int) -> Tensor:
    shape = value.shape[:-1]
    low = torch.zeros(shape, dtype=value.dtype, device=value.device)
    high = torch.full_like(low, float(value.max()))
    threshold = 0.5 * (low + high)
    for _ in range(50):
        threshold = 0.5 * (low + high)
        count = (value > threshold.unsqueeze(-1)).sum(dim=-1)
        if bool(((count >= k - 1) & (count <= k + 1)).all()):
            break
        low = torch.where(count > k + 1, threshold, low)
        high = torch.where(count < k - 1, threshold, high)
        if bool(((high - low).abs() < 1e-6).all()):
            break
    return value * (value >= threshold.unsqueeze(-1))


@torch.no_grad()
def upstream_reference_forward(
    state: Mapping[str, Tensor], config: LoRSAConfig, states: Tensor
) -> dict[str, Tensor]:
    """Independent transcription of the pinned upstream non-DTensor path."""

    q = torch.einsum("bsd,Qdq->bsQq", states, state["W_Q"]) + state["b_Q"]
    k = torch.einsum("bsd,Kdk->bsKk", states, state["W_K"]) + state["b_K"]
    v = torch.einsum("bsd,Vd->bsV", states, state["W_V"]) + state["b_V"]
    q_heads = q.permute(2, 0, 1, 3)
    k_heads = k.permute(2, 0, 3, 1)
    scale: float | Tensor = (
        state["_attn_scale_param"] if config.use_learnable_attn_scale else config.attn_scale
    )
    scores = torch.einsum("hbqd,hbdk->hbqk", q_heads, k_heads) / scale
    if config.use_smolgen:
        compressed = F.linear(states, state["smolgen.compress.weight"])
        smol = compressed.reshape(states.shape[0], 32 * config.n_ctx)
        smol = F.silu(
            F.linear(
                smol,
                state["smolgen.dense1.weight"],
                state["smolgen.dense1.bias"],
            )
        )
        smol = _reference_layer_norm(
            smol,
            state["smolgen.ln1.weight"],
            state["smolgen.ln1.bias"],
        )
        smol = F.silu(
            F.linear(
                smol,
                state["smolgen.dense2.weight"],
                state["smolgen.dense2.bias"],
            )
        )
        smol = _reference_layer_norm(
            smol,
            state["smolgen.ln2.weight"],
            state["smolgen.ln2.bias"],
        )
        smol = F.linear(
            smol.reshape(states.shape[0] * config.n_qk_heads, 256),
            state["smolgen.smol_weight_gen.weight"],
        ).reshape(states.shape[0], config.n_qk_heads, config.n_ctx, config.n_ctx)
        scores = scores + smol.permute(1, 0, 2, 3) * state["smolgen_score_scale"]
    pattern_heads = torch.softmax(scores, dim=-1)
    values = v.permute(0, 2, 1).reshape(
        states.shape[0],
        config.n_qk_heads,
        config.n_ov_heads // config.n_qk_heads,
        config.n_ctx,
    )
    values = values.permute(1, 0, 2, 3)
    hidden = torch.einsum("hbqk,hbrk->hbrq", pattern_heads, values)
    hidden = hidden.permute(1, 0, 2, 3).flatten(1, 2).permute(0, 2, 1)
    decoder_norm = state["W_O"].norm(dim=1)
    weighted = hidden * decoder_norm
    features = _reference_positive_topk(weighted, config.top_k) / decoder_norm
    reconstruction = torch.einsum("bps,sd->bpd", features, state["W_O"])
    if config.use_decoder_bias:
        reconstruction = reconstruction + state["b_D"]
    return {
        "input": states,
        "pattern": pattern_heads.permute(1, 0, 2, 3),
        "features": features,
        "reconstruction": reconstruction,
    }


def _load_module_from_state(
    state: Mapping[str, Tensor], config: LoRSAConfig
) -> LowRankSparseAttention:
    model = LowRankSparseAttention(config, device="meta")
    model.to_empty(device="cpu")
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def build_and_check_parity_fixture(
    state: Mapping[str, Tensor],
    config: LoRSAConfig,
    fixture_path: Path,
) -> dict[str, Any]:
    states = deterministic_parity_input(config)
    reference = upstream_reference_forward(state, config, states)
    model = _load_module_from_state(state, config)
    actual = model(states)
    torch.testing.assert_close(actual.pattern, reference["pattern"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.features, reference["features"], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        actual.reconstruction,
        reference["reconstruction"],
        rtol=2e-4,
        atol=2e-5,
    )
    comparison: dict[str, dict[str, float | bool]] = {}
    for name, local_value in (
        ("pattern", actual.pattern),
        ("features", actual.features),
        ("reconstruction", actual.reconstruction),
    ):
        difference = (local_value - reference[name]).abs()
        comparison[name] = {
            "exact_equal": bool(torch.equal(local_value, reference[name])),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        }
    comparison["features"]["support_equal"] = bool(
        torch.equal(actual.features > 0, reference["features"] > 0)
    )
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in sorted(reference.items())},
        fixture_path,
        metadata={
            "provenance_json": json.dumps(
                {
                    "input_rule": "lcg25173_mod65536_div131072",
                    "schema_version": "bt4-lorsa-upstream-parity-fixture-v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )
    return {
        "schema_version": "bt4-lorsa-upstream-parity-fixture-v1",
        "file": fixture_path.name,
        "size_bytes": fixture_path.stat().st_size,
        "sha256": sha256_file(fixture_path),
        "input_rule": "lcg25173_mod65536_div131072",
        "tensor_inventory": _tensor_inventory(reference),
        "lean_vs_upstream_reference": comparison,
        "assert_close": {
            "pattern": {"rtol": 2e-5, "atol": 2e-6},
            "features": {"rtol": 2e-4, "atol": 2e-5},
            "reconstruction": {"rtol": 2e-4, "atol": 2e-5},
        },
        "verified_during_conversion": True,
    }


@torch.no_grad()
def verify_converted_bundle(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve(strict=True)
    manifest_path = output_dir / "manifest.json"
    weights_path = output_dir / "lorsa.safetensors"
    manifest = _read_json_object(manifest_path)
    loaded = load_converted_lorsa(manifest_path, weights_path)
    fixture_info = manifest["parity_fixture"]
    fixture_path = output_dir / fixture_info["file"]
    if sha256_file(fixture_path) != fixture_info["sha256"]:
        raise ValueError("LoRSA parity fixture digest mismatch")
    with safe_open(fixture_path, framework="pt", device="cpu") as handle:
        fixture = {key: handle.get_tensor(key) for key in handle.keys()}
    actual = loaded.module(fixture["input"])
    tolerances = fixture_info["assert_close"]
    for name, value in (
        ("pattern", actual.pattern),
        ("features", actual.features),
        ("reconstruction", actual.reconstruction),
    ):
        torch.testing.assert_close(value, fixture[name], **tolerances[name])
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "weights_sha256": sha256_file(weights_path),
        "parity_fixture_sha256": sha256_file(fixture_path),
        "verified": True,
    }


def convert_published_lorsa(
    source_dir: Path,
    output_dir: Path,
    *,
    trust_paper_checkpoint: bool,
    repo_id: str = HF_REPO_ID,
    repo_revision: str = HF_REPO_REVISION,
    source_file_commit: str = HF_FILE_COMMIT,
    checkpoint_subpath: str = PUBLISHED_CHECKPOINT_SUBPATH,
    upstream_code_commit: str = UPSTREAM_CODE_COMMIT,
    verify_known_source: bool = True,
) -> dict[str, Any]:
    source_dir = source_dir.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite conversion output: {output_dir}")
    output_dir.mkdir(parents=True)

    inventory = (
        verify_published_source_files(source_dir)
        if verify_known_source
        else source_file_inventory(source_dir)
    )
    config_path = source_dir / "config.json"
    dcp_dir = source_dir / "sae_weights.dcp"
    parsed = parse_published_config(config_path)
    dcp_state, dcp_metadata = load_trusted_dcp_tensors(
        dcp_dir,
        trust_paper_checkpoint=trust_paper_checkpoint,
    )
    folded_state, module_config, normalization = extract_and_fold_state(dcp_state, parsed)

    weights_path = output_dir / "lorsa.safetensors"
    save_file(
        {name: value.contiguous() for name, value in sorted(folded_state.items())},
        weights_path,
        metadata={
            "provenance_json": json.dumps(
                {
                    "checkpoint_subpath": checkpoint_subpath,
                    "dataset_normalization_folded": True,
                    "schema_version": CONVERTED_LORSA_SCHEMA,
                    "source_repo_id": repo_id,
                    "source_repo_revision": repo_revision,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )
    parity = build_and_check_parity_fixture(
        folded_state,
        module_config,
        output_dir / "parity.safetensors",
    )
    source_identity_json = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    manifest: dict[str, Any] = {
        "schema_version": CONVERTED_LORSA_SCHEMA,
        "architecture": asdict(module_config),
        "weights_file": weights_path.name,
        "weights_size_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256_file(weights_path),
        "tensor_inventory": _tensor_inventory(folded_state),
        "tensor_count": len(folded_state),
        "source": {
            "repo_id": repo_id,
            "repo_revision": repo_revision,
            "source_file_commit": source_file_commit,
            "checkpoint_subpath": checkpoint_subpath,
            "public": True,
            "gated": False,
            "declared_license": None,
            "redistribution_rights_established": False,
            "storage_policy": "keep converted weights in private local/Railway experiment storage",
            "files": inventory,
            "file_inventory_sha256": hashlib.sha256(source_identity_json.encode()).hexdigest(),
            "dcp_metadata": dcp_metadata,
        },
        "upstream": {
            "code_repository": "JacklE0niden/Leela-SAEs",
            "code_commit": upstream_code_commit,
            "config_sha256": sha256_file(config_path),
        },
        "hook_contract": {
            "layer": parsed.layer,
            "input": parsed.hook_point_in,
            "target": parsed.hook_point_out,
            "sequence_shape": ["batch", module_config.n_ctx, module_config.d_model],
            "complete_ordered_board_required": True,
        },
        "normalization": normalization,
        "parity_fixture": parity,
        "conversion": {
            "converter_schema": "bt4-trusted-lorsa-dcp-converter-v1",
            "converter_module": "research.interpretability.sparse.convert_lorsa_checkpoint",
            "converter_sha256": sha256_file(Path(__file__)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "platform": platform.platform(),
            "process_isolation": "dedicated uv venv process reusing pinned project site-packages",
            "trust_basis": "user explicitly trusted the paper checkpoint",
            "source_pickle_deserialized_in_isolated_environment": True,
            "main_environment_pickle_loaded": False,
            "conversion_reviewed": True,
            "normal_experiment_loader_accepts_pickle": False,
        },
        # Retained at top level for the strict lean loader's original contract.
        "source_pickle_deserialized_in_isolated_environment": True,
        "main_environment_pickle_loaded": False,
        "conversion_reviewed": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verification = verify_converted_bundle(output_dir)
    return {
        **verification,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "weights_path": str(weights_path),
        "parity_path": str(output_dir / "parity.safetensors"),
        "source_file_count": len(inventory),
        "converted_tensor_count": len(folded_state),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--trust-paper-checkpoint",
        action="store_true",
        help="Explicitly permit DCP pickle metadata in this conversion process",
    )
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--repo-revision", default=HF_REPO_REVISION)
    parser.add_argument("--source-file-commit", default=HF_FILE_COMMIT)
    parser.add_argument("--checkpoint-subpath", default=PUBLISHED_CHECKPOINT_SUBPATH)
    parser.add_argument("--upstream-code-commit", default=UPSTREAM_CODE_COMMIT)
    parser.add_argument(
        "--skip-known-source-verification",
        action="store_true",
        help="Allow tests/new checkpoints whose files differ from the pinned L14 release",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Treat source_dir as an existing converted bundle and verify it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        result = verify_converted_bundle(args.source_dir)
    else:
        result = convert_published_lorsa(
            args.source_dir,
            args.output_dir,
            trust_paper_checkpoint=args.trust_paper_checkpoint,
            repo_id=args.repo_id,
            repo_revision=args.repo_revision,
            source_file_commit=args.source_file_commit,
            checkpoint_subpath=args.checkpoint_subpath,
            upstream_code_commit=args.upstream_code_commit,
            verify_known_source=not args.skip_known_source_verification,
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
