from __future__ import annotations

import torch

from research.interpretability.sparse.lorsa_causal_validation import (
    _deterministic_injection_mask,
)


def test_injection_token_selection_is_deterministic_and_at_most_one_per_position() -> None:
    concepts = torch.tensor(
        [[False, True, True, False], [False, False, False, False]],
        dtype=torch.bool,
    )
    first = _deterministic_injection_mask(concepts, ["p0", "p1"], "pair", 17)
    second = _deterministic_injection_mask(concepts, ["p0", "p1"], "pair", 17)

    assert torch.equal(first, second)
    assert first.sum(dim=1).tolist() == [1, 0]
    assert not bool((first & ~concepts).any())
