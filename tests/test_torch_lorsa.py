from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from research.interpretability.artifacts import sha256_file
from research.interpretability.sparse.lorsa import (
    CONVERTED_LORSA_SCHEMA,
    LoRSAConfig,
    LowRankSparseAttention,
    fold_lorsa_dataset_normalization,
    initialize_lorsa,
    inspect_untrusted_lorsa_dcp,
    load_converted_lorsa,
)


def _config() -> LoRSAConfig:
    return LoRSAConfig(
        d_model=4,
        n_ctx=3,
        n_qk_heads=2,
        d_qk_head=2,
        n_ov_heads=4,
        top_k=2,
        attn_scale=2.0,
        use_smolgen=False,
    )


def test_lorsa_preserves_sequence_and_exposes_components() -> None:
    model = LowRankSparseAttention(_config())
    initialize_lorsa(model, seed=5)
    states = torch.randn((2, 3, 4), generator=torch.Generator().manual_seed(6))
    output = model(states)
    assert output.query.shape == (2, 3, 2, 2)
    assert output.pattern.shape == (2, 2, 3, 3)
    assert output.features.shape == (2, 3, 4)
    assert output.reconstruction.shape == states.shape
    torch.testing.assert_close(output.pattern.sum(dim=-1), torch.ones((2, 2, 3)))
    replay = model(states, pattern_override=output.pattern, feature_override=output.features)
    torch.testing.assert_close(replay.reconstruction, output.reconstruction)


def test_lorsa_normalization_fold_matches_upstream_parameter_rule() -> None:
    state = {
        "W_Q": torch.ones((2, 4, 2)),
        "W_K": torch.ones((2, 4, 2)),
        "W_V": torch.ones((4, 4)),
        "W_O": torch.ones((4, 4)),
        "b_D": torch.ones(4),
    }
    folded = fold_lorsa_dataset_normalization(
        state,
        input_norm=1.0,
        output_norm=4.0,
        d_model=4,
    )
    torch.testing.assert_close(folded["W_Q"], state["W_Q"] * 2)
    torch.testing.assert_close(folded["W_K"], state["W_K"] * 2)
    torch.testing.assert_close(folded["W_V"], state["W_V"] * 2)
    torch.testing.assert_close(folded["W_O"], state["W_O"] * 2)
    torch.testing.assert_close(folded["b_D"], state["b_D"] * 2)


def test_untrusted_dcp_inventory_never_opens_metadata(tmp_path: Path) -> None:
    dcp = tmp_path / "sae_weights.dcp"
    dcp.mkdir()
    (dcp / ".metadata").write_bytes(b"not actually a pickle")
    (dcp / "__0_0.distcp").write_bytes(b"opaque shard")
    report = inspect_untrusted_lorsa_dcp(dcp)
    assert report["pickle_metadata_present"]
    assert report["pickle_loaded"] is False
    assert report["direct_load_allowed"] is False


def test_converted_safetensors_loader_is_strict(tmp_path: Path) -> None:
    model = LowRankSparseAttention(_config())
    initialize_lorsa(model, seed=9)
    weights = tmp_path / "lorsa.safetensors"
    save_file(model.state_dict(), weights)
    manifest = {
        "schema_version": CONVERTED_LORSA_SCHEMA,
        "architecture": {
            "d_model": 4,
            "n_ctx": 3,
            "n_qk_heads": 2,
            "d_qk_head": 2,
            "n_ov_heads": 4,
            "top_k": 2,
            "attn_scale": 2.0,
            "use_smolgen": False,
            "smolgen_score_scale": 1.0,
            "use_decoder_bias": True,
        },
        "weights_sha256": sha256_file(weights),
        "main_environment_pickle_loaded": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_converted_lorsa(manifest_path, weights)
    assert loaded.manifest["main_environment_pickle_loaded"] is False
    states = torch.zeros((1, 3, 4))
    torch.testing.assert_close(
        loaded.module(states).reconstruction,
        model(states).reconstruction,
    )
