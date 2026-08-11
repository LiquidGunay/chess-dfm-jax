from __future__ import annotations

import torch

from research.interpretability.sparse.delta_crosscoder import (
    PairedDeltaCrosscoder,
    delta_crosscoder_metrics,
    fit_delta_crosscoder,
    initialize_delta_crosscoder,
    remove_sparse_features,
    shared_feature_taxonomy,
)


def test_delta_crosscoder_shapes_metrics_and_feature_removal() -> None:
    model = PairedDeltaCrosscoder(
        3,
        shared_features=4,
        specific_features=3,
        shared_k=2,
        specific_k=1,
    )
    initialize_delta_crosscoder(model, seed=3)
    raw = torch.randn((2, 5, 3), generator=torch.Generator().manual_seed(4))
    hero = raw + 0.1
    output = model(raw, hero)
    assert output.shared_features.shape == (2, 5, 4)
    assert output.raw_reconstruction.shape == raw.shape
    metrics = delta_crosscoder_metrics(raw, hero, output)
    assert metrics["sparsity"]["shared_mean_l0"] <= 2
    taxonomy = shared_feature_taxonomy(model)
    assert taxonomy["shared_count"] == 4
    raw_removed, hero_removed = remove_sparse_features(
        output,
        model,
        shared_indices=torch.tensor([0]),
    )
    assert raw_removed.shape == raw.shape
    assert hero_removed.shape == hero.shape
    assert not torch.equal(raw_removed, output.raw_reconstruction)


def test_bounded_delta_crosscoder_fit_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(8)
    raw = torch.randn((4, 3, 4), generator=generator)
    hero = raw + 0.2 * torch.roll(raw, 1, dims=-1)
    kwargs = dict(
        shared_features=6,
        specific_features=4,
        shared_k=2,
        specific_k=1,
        epochs=2,
        learning_rate=0.02,
        batch_size=4,
        seed=7,
    )
    first = fit_delta_crosscoder(raw, hero, **kwargs)
    second = fit_delta_crosscoder(raw, hero, **kwargs)
    with torch.inference_mode():
        first_output = first.module(raw, hero)
        second_output = second.module(raw, hero)
    torch.testing.assert_close(first_output.raw_reconstruction, second_output.raw_reconstruction)
    torch.testing.assert_close(first_output.hero_reconstruction, second_output.hero_reconstruction)
    assert first.metadata["scope"].startswith("bounded development pilot")
