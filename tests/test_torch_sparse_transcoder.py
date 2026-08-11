from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from research.interpretability.sparse.transcoder import (
    SparseSupportAccumulator,
    TopKTranscoder,
    fit_transcoder,
    fold_dataset_normalization,
    load_published_transcoder,
    positive_topk,
    reconstruction_metrics,
    sparse_support_metrics,
)


def test_positive_topk_and_decoder_norm_support_selection() -> None:
    value = torch.tensor([[3.0, 2.0, 1.0, -4.0]])
    exact = positive_topk(value, 2, mode="exact")
    assert exact.tolist() == [[3.0, 2.0, 0.0, 0.0]]
    upstream = positive_topk(value, 2, mode="upstream_threshold", tolerance=0)
    assert torch.count_nonzero(upstream) == 2

    module = TopKTranscoder(2, 3, k=1, topk_mode="exact")
    with torch.no_grad():
        module.W_E.copy_(torch.tensor([[1.0, 0.75, 0.0], [0.0, 0.0, 0.0]]))
        module.b_E.zero_()
        module.W_D.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]))
        module.b_D.zero_()
    output = module(torch.tensor([[1.0, 0.0]]))
    # Raw preactivation 0 wins only after multiplying feature 1 by decoder norm 2.
    assert output.features.tolist() == [[0.0, 0.75, 0.0]]
    torch.testing.assert_close(output.reconstruction, torch.tensor([[1.5, 0.0]]))


def test_dataset_norm_fold_preserves_normalized_function() -> None:
    state = {
        "W_E": torch.tensor([[2.0, 1.0], [0.0, 1.0]]),
        "b_E": torch.tensor([0.5, -0.25]),
        "W_D": torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        "b_D": torch.tensor([0.25, -0.5]),
    }
    folded = fold_dataset_normalization(state, input_norm=2.0, output_norm=4.0, d_model=4)
    # c_in=1 and c_out=0.5.
    torch.testing.assert_close(folded["b_E"], state["b_E"])
    torch.testing.assert_close(folded["W_D"], state["W_D"] * 2)
    torch.testing.assert_close(folded["b_D"], state["b_D"] * 2)


def test_strict_safetensors_loader_folds_without_pickle(tmp_path: Path) -> None:
    config = {
        "sae_type": "sae",
        "d_model": 1024,
        "expansion_factor": 0.00390625,
        "act_fn": "topk",
        "top_k": 2,
        "hook_point_in": "blocks.14.resid_mid_after_ln",
        "hook_point_out": "blocks.14.hook_mlp_out",
        "sparsity_include_decoder_norm": True,
        "norm_activation": "dataset-wise",
        "use_decoder_bias": True,
        "use_glu_encoder": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(
        {
            "W_E": torch.zeros((1024, 4)),
            "b_E": torch.ones(4),
            "W_D": torch.ones((4, 1024)),
            "b_D": torch.zeros(1024),
            "dataset_average_activation_norm.blocks.14.resid_mid_after_ln": torch.tensor(16.0),
            "dataset_average_activation_norm.blocks.14.hook_mlp_out": torch.tensor(8.0),
        },
        weights_path,
        metadata={"version": "unit"},
    )
    loaded = load_published_transcoder(config_path, weights_path)
    assert loaded.module.d_sae == 4
    assert loaded.manifest["pickle_loaded"] is False
    assert loaded.manifest["dataset_normalization_folded"] is True
    # c_in=2, c_out=4.
    torch.testing.assert_close(loaded.module.b_E, torch.full((4,), 0.5))
    torch.testing.assert_close(loaded.module.W_D, torch.full((4, 1024), 0.5))


def test_reconstruction_support_metrics_and_bounded_fit() -> None:
    generator = torch.Generator().manual_seed(9)
    inputs = torch.randn((2, 4, 3), generator=generator)
    targets = 0.5 * inputs
    fitted = fit_transcoder(
        inputs,
        targets,
        d_sae=6,
        k=2,
        epochs=3,
        learning_rate=0.03,
        batch_size=4,
        seed=5,
    )
    with torch.inference_mode():
        result = fitted.module(inputs)
    metrics = reconstruction_metrics(targets, result.reconstruction)
    assert metrics["relative_l2"] >= 0.0
    support = sparse_support_metrics(result.features, result.features)
    assert support["mean_support_jaccard"] == 1.0
    accumulator = SparseSupportAccumulator(result.features.shape[-1])
    accumulator.update(result.features[:1], result.features[:1])
    accumulator.update(result.features[1:], result.features[1:])
    streamed = accumulator.finalize()
    assert streamed["mean_support_jaccard"] == support["mean_support_jaccard"]
    assert streamed["mean_first_l0"] == support["mean_first_l0"]
    assert (
        streamed["resolved_activation_correlations"] == support["resolved_activation_correlations"]
    )
    assert fitted.metadata["scope"].startswith("bounded pilot")
