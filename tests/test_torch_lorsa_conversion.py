from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.distributed.checkpoint import save as dcp_save

from research.interpretability.artifacts import sha256_file
from research.interpretability.sparse.convert_lorsa_checkpoint import (
    convert_published_lorsa,
    verify_converted_bundle,
)
from research.interpretability.sparse.lorsa import (
    LoRSAConfig,
    LowRankSparseAttention,
    initialize_lorsa,
    load_converted_lorsa,
)


def _published_tiny_config() -> LoRSAConfig:
    return LoRSAConfig(
        d_model=4,
        n_ctx=3,
        n_qk_heads=2,
        d_qk_head=2,
        n_ov_heads=4,
        top_k=2,
        attn_scale=2.0,
        use_smolgen=True,
        smolgen_score_scale=1.0,
        use_learnable_attn_scale=True,
        use_decoder_bias=True,
    )


def _write_source(root: Path) -> Path:
    root.mkdir()
    config = {
        "sae_type": "lorsa",
        "d_model": 4,
        "expansion_factor": 1,
        "use_decoder_bias": True,
        "act_fn": "topk",
        "norm_activation": "dataset-wise",
        "sparsity_include_decoder_norm": True,
        "top_k": 2,
        "hook_point_in": "blocks.14.hook_attn_in",
        "hook_point_out": "blocks.14.hook_attn_out",
        "n_qk_heads": 2,
        "d_qk_head": 2,
        "positional_embedding_type": "none",
        "n_ctx": 3,
        "skip_bos": False,
        "attn_scale": 2.0,
        "use_post_qk_ln": False,
        "use_smolgen": True,
        "use_learnable_attn_scale": True,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = LowRankSparseAttention(_published_tiny_config())
    initialize_lorsa(model, seed=17)
    with torch.no_grad():
        assert model._attn_scale_param is not None
        model._attn_scale_param.fill_(1.75)
        model.smolgen_score_scale.fill_(0.875)
    state = dict(model.state_dict())
    state.update(
        {
            "IGNORE": torch.tensor(float("-inf")),
            "mask": torch.tril(torch.ones((3, 3), dtype=torch.bool)),
            "tokens_since_last_activation": torch.zeros(4, dtype=torch.int64),
            "is_dead": torch.zeros(4, dtype=torch.bool),
            "dataset_average_activation_norm.blocks.14.hook_attn_in": torch.tensor(1.0),
            "dataset_average_activation_norm.blocks.14.hook_attn_out": torch.tensor(4.0),
        }
    )
    dcp_save(state, checkpoint_id=root / "sae_weights.dcp")
    return root


def test_published_lorsa_state_inventory_preserves_both_trained_scales() -> None:
    keys = set(LowRankSparseAttention(_published_tiny_config(), device="meta").state_dict())
    assert len(keys) == 20
    assert "_attn_scale_param" in keys
    assert "smolgen_score_scale" in keys
    assert "smolgen.smol_weight_gen.weight" in keys
    assert "smolgen.weight_generator.weight" not in keys


def test_trusted_dcp_conversion_is_deterministic_and_strict(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    first = tmp_path / "converted_first"
    second = tmp_path / "converted_second"
    convert_published_lorsa(
        source,
        first,
        trust_paper_checkpoint=True,
        verify_known_source=False,
    )
    convert_published_lorsa(
        source,
        second,
        trust_paper_checkpoint=True,
        verify_known_source=False,
    )
    for name in ("lorsa.safetensors", "parity.safetensors", "manifest.json"):
        assert sha256_file(first / name) == sha256_file(second / name)

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tensor_count"] == 20
    assert manifest["conversion"]["conversion_reviewed"] is True
    assert manifest["conversion"]["main_environment_pickle_loaded"] is False
    assert manifest["normalization"]["folded"] is True
    assert manifest["normalization"]["learned_attention_scale"] == 1.75
    assert manifest["normalization"]["smolgen_score_scale"] == 0.875
    assert manifest["parity_fixture"]["verified_during_conversion"] is True
    loaded = load_converted_lorsa(first / "manifest.json", first / "lorsa.safetensors")
    assert loaded.module._attn_scale_param is not None
    torch.testing.assert_close(loaded.module._attn_scale_param, torch.tensor(1.75))
    torch.testing.assert_close(loaded.module.smolgen_score_scale, torch.tensor(0.875))
    assert verify_converted_bundle(first)["verified"] is True


def test_converter_refuses_pickle_without_explicit_trust(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    try:
        convert_published_lorsa(
            source,
            tmp_path / "converted",
            trust_paper_checkpoint=False,
            verify_known_source=False,
        )
    except ValueError as error:
        assert "--trust-paper-checkpoint" in str(error)
    else:
        raise AssertionError("Converter unexpectedly loaded DCP without explicit trust")
