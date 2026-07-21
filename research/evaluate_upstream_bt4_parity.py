#!/usr/bin/env python3
"""Cross-framework FP32 parity for local BT4 and pinned Leela-SAEs code.

The harness imports the exact upstream BT4 component files with inert hook
points, binds the already-audited local dense weights explicitly, and compares
PyTorch activations to the local JAX implementation.  Both the official LC0
epsilon and the pinned constructor's effective default epsilon are measured.
No external checkpoint object is deserialized.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chess_dfm_jax.analysis.profile_targets import load_mapped_bt4_params  # noqa: E402
from chess_dfm_jax.nnx_bt4 import make_bt4_model  # noqa: E402
from research.evaluate_dense_representations import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MODELS_DIR,
    DEFAULT_POOL,
    HOOK_NAMES,
    _capture_fp32_hooks,
    encoder_payload_sha256,
    load_frozen_corpus,
)
from research.prepare import (  # noqa: E402
    REPO_ROOT,
    require_within_workspace,
    sha256_file,
    write_json,
)


UPSTREAM_REVISION = "f946a5736f40397f5c834104aeac6e8e6ff753bd"
UPSTREAM_FILE_SHA256 = {
    "TransformerLens/transformer_lens/components/leela_embed.py": (
        "b6987dd52bb4e9342f6d46b3fce80387bd8f0ff644b254060034f3a323c4a5a7"
    ),
    "TransformerLens/transformer_lens/components/leela_encoder.py": (
        "06b6893539128726e464ca49522e244a23d312af02af29da51e4c5e323616052"
    ),
    "TransformerLens/transformer_lens/components/leela_policyhead.py": (
        "0355ba2f2bbfd927458c2779db25183be8d53341b68563864ef60c3c8be81c92"
    ),
    "TransformerLens/transformer_lens/pretrained/weight_conversions/leela.py": (
        "88203f566c2d31ab41e9c72f145d5facb24547f7452308f88aa20beeb7d792cd"
    ),
    "TransformerLens/transformer_lens/HookedTransformer.py": (
        "cba8eb5f09ccbe64e86052da0253b834b71f4c0c03e9fad9d7d81a8e4feea317"
    ),
    "TransformerLens/transformer_lens/loading_from_pretrained.py": (
        "348215cfabe8d1d5653354ad5f486c0ba0ebaeaed7842a8929046f5385cadff6"
    ),
}
DEFAULT_UPSTREAM_ROOT = Path("/mountpoint/.exp/upstream-leela-saes")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "representations"
    / "upstream-bt4-source-parity-v1"
)
RAW_BT4_SHA256 = "61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651"
OFFICIAL_EPSILON = 1e-3
UPSTREAM_DEFAULT_EPSILON = 1e-5
RELATIVE_L2_TOLERANCE = 5e-4
MAX_ABSOLUTE_TOLERANCE = 1e-2
COSINE_TOLERANCE = 0.999999


class _IdentityHook(torch.nn.Module):
    """Small stand-in for TransformerLens HookPoint during audited import."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


@dataclass(frozen=True)
class UpstreamClasses:
    embed: type[torch.nn.Module]
    encoder_layer: type[torch.nn.Module]
    policy_head: type[torch.nn.Module]
    source: dict[str, Any]


def _within_experiment_workspace(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    workspace = Path("/mountpoint/.exp").resolve(strict=True)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Upstream source escapes {workspace}: {resolved}") from exc
    return resolved


def _load_module(path: Path, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot construct an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pinned_upstream_classes(upstream_root: Path) -> UpstreamClasses:
    """Import only the pinned BT4 component files, without package install."""

    root = _within_experiment_workspace(upstream_root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(
            f"Leela-SAEs revision drift: {revision} != {UPSTREAM_REVISION}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("Pinned Leela-SAEs checkout has uncommitted changes")

    file_records: dict[str, dict[str, Any]] = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"Pinned upstream file drift for {relative}: {observed} != {expected}"
            )
        file_records[relative] = {
            "sha256": observed,
            "size_bytes": path.stat().st_size,
        }

    fake_root = types.ModuleType("transformer_lens")
    fake_hooks = types.ModuleType("transformer_lens.hook_points")
    fake_components = types.ModuleType("transformer_lens.components")
    fake_hooks.HookPoint = _IdentityHook
    # The custom BT4 encoder imports this symbol but uses Lc0LayerNorm.
    fake_components.LayerNorm = torch.nn.LayerNorm
    names = (
        "transformer_lens",
        "transformer_lens.hook_points",
        "transformer_lens.components",
    )
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules[names[0]] = fake_root
    sys.modules[names[1]] = fake_hooks
    sys.modules[names[2]] = fake_components
    try:
        embed_module = _load_module(
            root
            / "TransformerLens"
            / "transformer_lens"
            / "components"
            / "leela_embed.py",
            "_pinned_leela_embed",
        )
        encoder_module = _load_module(
            root
            / "TransformerLens"
            / "transformer_lens"
            / "components"
            / "leela_encoder.py",
            "_pinned_leela_encoder",
        )
        policy_module = _load_module(
            root
            / "TransformerLens"
            / "transformer_lens"
            / "components"
            / "leela_policyhead.py",
            "_pinned_leela_policyhead",
        )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    return UpstreamClasses(
        embed=embed_module.BT4LeelaEmbed,
        encoder_layer=encoder_module.EncoderLayer,
        policy_head=policy_module.PolicyHead,
        source={
            "root": str(root),
            "revision": revision,
            "files": file_records,
            "import_boundary": (
                "exact pinned component files with HookPoint replaced by identity; "
                "no external checkpoint deserialization"
            ),
        },
    )


def _copy_parameter(
    parameter: torch.nn.Parameter,
    value: Any,
    *,
    name: str,
) -> None:
    array = np.asarray(value, dtype=np.float32)
    if tuple(parameter.shape) != tuple(array.shape):
        raise ValueError(
            f"Shape mismatch for {name}: {tuple(parameter.shape)} != {array.shape}"
        )
    with torch.no_grad():
        parameter.copy_(torch.from_numpy(np.ascontiguousarray(array)))


def _copy_linear(
    module: torch.nn.Linear,
    weight: Any,
    bias: Any | None,
    *,
    name: str,
) -> None:
    _copy_parameter(module.weight, np.asarray(weight).T, name=f"{name}.weight")
    if bias is None:
        if module.bias is not None:
            raise ValueError(f"Expected bias-free upstream linear at {name}")
    else:
        if module.bias is None:
            raise ValueError(f"Expected an upstream bias at {name}")
        _copy_parameter(module.bias, bias, name=f"{name}.bias")


def bind_embed_weights(
    module: torch.nn.Module,
    params: Mapping[str, Any],
    *,
    alpha: float,
) -> None:
    _copy_linear(
        module.embedding_preprocess,
        params["preproc_w"],
        params["preproc_b"],
        name="embed.embedding_preprocess",
    )
    _copy_linear(
        module.main_linear,
        params["w"],
        params["b"],
        name="embed.main_linear",
    )
    _copy_parameter(module.ma_gating_mul, params["mul_gate"], name="embed.ma_gating_mul")
    _copy_parameter(module.ma_gating_add, params["add_gate"], name="embed.ma_gating_add")
    _copy_linear(
        module.ffn_dense1,
        params["ffn"]["dense1_w"],
        params["ffn"]["dense1_b"],
        name="embed.ffn_dense1",
    )
    _copy_linear(
        module.ffn_dense2,
        params["ffn"]["dense2_w"],
        params["ffn"]["dense2_b"],
        name="embed.ffn_dense2",
    )
    _copy_parameter(module.ffn_alpha, [alpha], name="embed.ffn_alpha")
    _copy_parameter(module.ln.weight, params["ln_scale"], name="embed.ln.weight")
    _copy_parameter(module.ln.bias, params["ln_bias"], name="embed.ln.bias")
    _copy_parameter(
        module.ln2.weight,
        params["ffn_ln_scale"],
        name="embed.ln2.weight",
    )
    _copy_parameter(
        module.ln2.bias,
        params["ffn_ln_bias"],
        name="embed.ln2.bias",
    )


def bind_encoder_layer_weights(
    module: torch.nn.Module,
    params: Mapping[str, Any],
    shared_smolgen_weight: Any,
    *,
    alpha: float,
) -> None:
    mha = params["mha"]
    _copy_linear(module.mha.q_proj, mha["q_w"], mha["q_b"], name="mha.q_proj")
    _copy_linear(module.mha.k_proj, mha["k_w"], mha["k_b"], name="mha.k_proj")
    _copy_linear(module.mha.v_proj, mha["v_w"], mha["v_b"], name="mha.v_proj")
    _copy_linear(
        module.mha.out_proj,
        mha["dense_w"],
        mha["dense_b"],
        name="mha.out_proj",
    )
    _copy_parameter(
        module.mha.qk_scale,
        [1.0 / math.sqrt(32.0)],
        name="mha.qk_scale",
    )
    smolgen = mha["smolgen"]
    _copy_linear(
        module.mha.smolgen.compress,
        smolgen["compress_w"],
        None,
        name="mha.smolgen.compress",
    )
    _copy_linear(
        module.mha.smolgen.dense1,
        smolgen["dense1_w"],
        smolgen["dense1_b"],
        name="mha.smolgen.dense1",
    )
    _copy_parameter(
        module.mha.smolgen.ln1.weight,
        smolgen["ln1_scale"],
        name="mha.smolgen.ln1.weight",
    )
    _copy_parameter(
        module.mha.smolgen.ln1.bias,
        smolgen["ln1_bias"],
        name="mha.smolgen.ln1.bias",
    )
    _copy_linear(
        module.mha.smolgen.dense2,
        smolgen["dense2_w"],
        smolgen["dense2_b"],
        name="mha.smolgen.dense2",
    )
    _copy_parameter(
        module.mha.smolgen.ln2.weight,
        smolgen["ln2_scale"],
        name="mha.smolgen.ln2.weight",
    )
    _copy_parameter(
        module.mha.smolgen.ln2.bias,
        smolgen["ln2_bias"],
        name="mha.smolgen.ln2.bias",
    )
    _copy_linear(
        module.mha.smolgen.smol_weight_gen,
        shared_smolgen_weight,
        None,
        name="mha.smolgen.smol_weight_gen",
    )
    _copy_parameter(module.ln1.w, params["ln1"]["scale"], name="ln1.w")
    _copy_parameter(module.ln1.b, params["ln1"]["bias"], name="ln1.b")
    _copy_parameter(module.ln2.w, params["ln2"]["scale"], name="ln2.w")
    _copy_parameter(module.ln2.b, params["ln2"]["bias"], name="ln2.b")
    _copy_linear(
        module.mlp.dense1,
        params["ffn"]["dense1_w"],
        params["ffn"]["dense1_b"],
        name="mlp.dense1",
    )
    _copy_linear(
        module.mlp.dense2,
        params["ffn"]["dense2_w"],
        params["ffn"]["dense2_b"],
        name="mlp.dense2",
    )
    _copy_parameter(module.alpha_input, [alpha], name="alpha_input")
    _copy_parameter(module.alpha_out1, [alpha], name="alpha_out1")


def bind_policy_weights(
    module: torch.nn.Module,
    params: Mapping[str, Any],
    mapping_table: Any,
) -> None:
    _copy_linear(module.dense1, params["dense1_w"], params["dense1_b"], name="policy.dense1")
    _copy_linear(module.q_proj, params["q_w"], params["q_b"], name="policy.q_proj")
    _copy_linear(module.k_proj, params["k_w"], params["k_b"], name="policy.k_proj")
    _copy_parameter(module.scale, [1.0 / math.sqrt(1024.0)], name="policy.scale")
    _copy_linear(module.promotion, params["prom_w"], None, name="policy.promotion")
    _copy_parameter(module.indices, mapping_table, name="policy.indices")
    # Pinned upstream declares this unused legacy parameter at a fixed width of
    # 768. It is absent from forward and intentionally receives deterministic
    # zeros rather than source weights.
    _copy_parameter(
        module.promotion_weight,
        np.zeros(tuple(module.promotion_weight.shape), dtype=np.float32),
        name="policy.unused_promotion_weight",
    )


def representation_error_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    source = np.asarray(reference, dtype=np.float64)
    other = np.asarray(candidate, dtype=np.float64)
    if source.shape != other.shape:
        raise ValueError(f"Representation shape mismatch: {source.shape} != {other.shape}")
    delta = other - source
    source_norm = float(np.linalg.norm(source.reshape(-1)))
    other_norm = float(np.linalg.norm(other.reshape(-1)))
    relative = np.abs(delta) / np.maximum(np.abs(source), 1e-6)
    cosine = float(
        np.vdot(source.reshape(-1), other.reshape(-1))
        / max(source_norm * other_norm, 1e-30)
    )
    return {
        "max_absolute_error": float(np.max(np.abs(delta))),
        "max_relative_error_at_floor_1e-6": float(np.max(relative)),
        "relative_l2_error": float(np.linalg.norm(delta.reshape(-1)) / max(source_norm, 1e-30)),
        "cosine_similarity": cosine,
        "reference_rms": float(np.sqrt(np.mean(np.square(source)))),
        "candidate_rms": float(np.sqrt(np.mean(np.square(other)))),
    }


def metric_passes(metric: Mapping[str, float]) -> bool:
    return bool(
        metric["relative_l2_error"] <= RELATIVE_L2_TOLERANCE
        and metric["max_absolute_error"] <= MAX_ABSOLUTE_TOLERANCE
        and metric["cosine_similarity"] >= COSINE_TOLERANCE
    )


def _capture_upstream_layer(
    layer: torch.nn.Module,
    value: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    captures: dict[str, np.ndarray] = {}
    handles: list[Any] = []
    for hook_name in HOOK_NAMES:
        hook = getattr(layer, hook_name)

        def save_output(_module, _inputs, output, *, name=hook_name):
            captures[name] = output.detach().cpu().numpy().copy()

        handles.append(hook.register_forward_hook(save_output))
    try:
        output = layer(value)
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(HOOK_NAMES):
        raise RuntimeError(f"Incomplete upstream hook capture: {sorted(captures)}")
    return output, captures


def _variant_summary(
    *,
    name: str,
    epsilon: float,
    rows: list[dict[str, Any]],
    embedding_metric: dict[str, float],
    policy_metric: dict[str, float],
) -> dict[str, Any]:
    all_metrics = [embedding_metric, policy_metric, *(row["metrics"] for row in rows)]
    return {
        "name": name,
        "encoder_layer_epsilon": epsilon,
        "embedding": embedding_metric,
        "layers": rows,
        "policy": policy_metric,
        "worst_relative_l2_error": max(item["relative_l2_error"] for item in all_metrics),
        "worst_max_absolute_error": max(item["max_absolute_error"] for item in all_metrics),
        "minimum_cosine_similarity": min(item["cosine_similarity"] for item in all_metrics),
        "gate_pass": all(metric_passes(item) for item in all_metrics),
    }


def run(args: argparse.Namespace) -> Path:
    if jax.default_backend() != "cpu" and not args.allow_non_cpu_jax:
        raise RuntimeError(
            "Cross-framework source parity defaults to CPU JAX so PyTorch and JAX "
            f"do not contend for the A10G; found {jax.default_backend()!r}"
        )
    if args.examples < 2:
        raise ValueError("--examples must be at least two")
    output_dir = require_within_workspace(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Parity output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-",
            dir=output_dir.parent,
        )
    )
    work_dir = require_within_workspace(work_dir)

    try:
        upstream = load_pinned_upstream_classes(args.upstream_root)
        corpus = load_frozen_corpus(
            args.pool,
            args.data_root,
            example_count=args.examples,
        )
        mapped = load_mapped_bt4_params(models_dir=str(args.models_dir))
        raw_model_path = require_within_workspace(
            args.models_dir / "BT4_exported.pb.gz"
        )
        if sha256_file(raw_model_path) != RAW_BT4_SHA256:
            raise ValueError(f"Raw BT4 source asset drift: {raw_model_path}")
        alpha = float((2.0 * len(mapped["encoder"])) ** -0.25)

        local_model = make_bt4_model(
            mapped,
            dtype=jnp.float32,
            attention_impl="manual",
        )
        local_hooks, local_policy = _capture_fp32_hooks(
            local_model,
            jnp.asarray(corpus.planes, dtype=jnp.float32),
        )
        local_hooks, local_policy = jax.device_get(
            jax.block_until_ready((local_hooks, local_policy))
        )
        local_hooks = np.asarray(local_hooks, dtype=np.float32)
        local_policy = np.asarray(local_policy, dtype=np.float32)
        del local_model
        gc.collect()

        torch.set_grad_enabled(False)
        torch.set_num_threads(max(1, int(args.torch_threads)))
        planes = torch.from_numpy(np.ascontiguousarray(corpus.planes))
        embed = upstream.embed(1024).to(dtype=torch.float32, device="cpu")
        bind_embed_weights(embed, mapped["embedding"], alpha=alpha)
        embed.eval()
        with torch.inference_mode():
            upstream_embed = embed(planes)
        embedding_metric = representation_error_metrics(
            local_hooks[0, 0],
            upstream_embed.detach().cpu().numpy(),
        )
        del embed, planes
        gc.collect()

        variants: dict[str, dict[str, Any]] = {}
        variant_specs = (
            ("official_lc0_epsilon", OFFICIAL_EPSILON),
            ("pinned_constructor_default_epsilon", UPSTREAM_DEFAULT_EPSILON),
        )
        for variant_name, epsilon in variant_specs:
            value = upstream_embed.clone()
            rows: list[dict[str, Any]] = []
            for layer_index, layer_params in enumerate(mapped["encoder"]):
                layer = upstream.encoder_layer(
                    1024,
                    32,
                    1536,
                    "mish",
                    "post",
                    eps=epsilon,
                ).to(dtype=torch.float32, device="cpu")
                bind_encoder_layer_weights(
                    layer,
                    layer_params,
                    mapped["smolgen_w"],
                    alpha=alpha,
                )
                layer.eval()
                with torch.inference_mode():
                    value, captures = _capture_upstream_layer(layer, value)
                for hook_index, hook_name in enumerate(HOOK_NAMES):
                    metric = representation_error_metrics(
                        local_hooks[hook_index, layer_index],
                        captures[hook_name],
                    )
                    rows.append(
                        {
                            "layer": layer_index,
                            "hook": hook_name,
                            "metrics": metric,
                            "gate_pass": metric_passes(metric),
                        }
                    )
                del layer, captures
                gc.collect()

            policy = upstream.policy_head(1024, 1858).to(
                dtype=torch.float32,
                device="cpu",
            )
            bind_policy_weights(policy, mapped["policy"], mapped["mapping_table"])
            policy.eval()
            with torch.inference_mode():
                upstream_policy = policy(value)
            policy_metric = representation_error_metrics(
                local_policy,
                upstream_policy.detach().cpu().numpy(),
            )
            variants[variant_name] = _variant_summary(
                name=variant_name,
                epsilon=epsilon,
                rows=rows,
                embedding_metric=embedding_metric,
                policy_metric=policy_metric,
            )
            del value, policy, upstream_policy
            gc.collect()

        result = {
            "schema_version": "bt4-upstream-cross-framework-parity-v1",
            "gate": {
                "official_epsilon_pass": variants["official_lc0_epsilon"]["gate_pass"],
                "upstream_default_epsilon_pass": variants[
                    "pinned_constructor_default_epsilon"
                ]["gate_pass"],
                "interpretation": (
                    "The official-epsilon path is the source-parity gate. The pinned "
                    "constructor-default path is a measured compatibility warning."
                ),
                "tolerances": {
                    "relative_l2_error_max": RELATIVE_L2_TOLERANCE,
                    "max_absolute_error_max": MAX_ABSOLUTE_TOLERANCE,
                    "cosine_similarity_min": COSINE_TOLERANCE,
                },
            },
            "variants": variants,
        }
        write_json(work_dir / "results.json", result)
        write_json(
            work_dir / "corpus.json",
            {
                "contract": corpus.contract,
                "entries": list(corpus.entries),
            },
        )
        source_encoder_digest = encoder_payload_sha256(
            {
                "embedding": mapped["embedding"],
                "encoder": {
                    str(index): layer
                    for index, layer in enumerate(mapped["encoder"])
                },
                "smolgen_w": mapped["smolgen_w"],
            }
        )
        files = []
        for name in ("results.json", "corpus.json"):
            path = work_dir / name
            files.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": "bt4-upstream-cross-framework-parity-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "upstream": upstream.source,
            "local_git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "jax_backend": jax.default_backend(),
            "torch_version": torch.__version__,
            "torch_device": "cpu",
            "torch_threads": int(args.torch_threads),
            "raw_bt4_path": str(raw_model_path),
            "raw_bt4_sha256": RAW_BT4_SHA256,
            "source_encoder_payload_sha256": source_encoder_digest,
            "alpha": alpha,
            "hook_names": list(HOOK_NAMES),
            "example_count": args.examples,
            "stored_planes_primary": True,
            "external_checkpoint_deserialization": False,
            "temporary_activations_retained": False,
            "files": files,
        }
        write_json(work_dir / "manifest.json", manifest)
        os.replace(work_dir, output_dir)
        if not result["gate"]["official_epsilon_pass"]:
            raise RuntimeError(
                f"Official-epsilon cross-framework parity failed; see {output_dir}"
            )
        return output_dir
    except BaseException:
        if work_dir.exists():
            import shutil

            shutil.rmtree(work_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--allow-non-cpu-jax", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run(args)
    print(json.dumps({"output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
