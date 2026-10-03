from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.evaluate_policy_loss_play import metric_vectors, refinement_logits, validate_arrays
from research.evaluate_common_policy import legal_metrics


def test_metric_vectors_match_existing_reduction_and_ignore_illegal_logits():
    logits = torch.tensor([[1., 100., 2.], [3., 1., 200.]])
    legal = torch.tensor([[True, False, True], [True, True, False]])
    targets = torch.tensor([2, 1])
    ce, correct = metric_vectors(logits, legal, targets)
    expected_ce, expected_correct = legal_metrics(logits, legal, targets)
    assert ce.sum() == pytest.approx(expected_ce)
    assert correct.sum() == expected_correct == 1
    with pytest.raises(ValueError):
        metric_vectors(logits, legal, torch.tensor([1, 2]))
    with pytest.raises(ValueError):
        metric_vectors(logits * float('inf'), legal, targets)


def test_arrays_require_history_and_legal_targets():
    data = {'planes': np.zeros((2, 112, 8, 8), dtype=np.uint8),
            'actions': np.array([0, 1]), 'legal': np.ones((2, 1858), dtype=bool)}
    with pytest.raises(ValueError, match='no history'):
        validate_arrays(data, 2)
    data['planes'][:, 13, 0, 0] = 1
    assert validate_arrays(data, 2)['history_nonzero_positions'] == 2
    data['legal'][1, 1] = False
    with pytest.raises(ValueError, match='Illegal target'):
        validate_arrays(data, 2)


def test_capture_uses_actual_eight_pass_schedule_and_restores_planner():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(horizon=8, jepa_feedback_mode='none', dfm_closed_loop_mode='none')
            self.calls = []

        def planner(self, z, tokens, t, dtype, **kwargs):
            self.calls.append((t.clone(), tokens.clone()))
            logits = torch.zeros((2, 8, 1858))
            logits[:, :, 10] = 100  # illegal at the root
            logits[:, 0, 0] = 3 - 5*t
            logits[:, 0, 1] = 5*t
            return logits

    model = Model()
    original = model.planner
    legal = torch.zeros((2, 1858), dtype=torch.bool)
    legal[:, :2] = True
    result = refinement_logits(model, torch.zeros((2, 4)), torch.zeros((2, 1858)), legal, check_one_pass=True)
    assert len(model.calls) == 9
    assert [float(call[0][0]) for call in model.calls[:8]] == [i/8 for i in range(8)]
    assert (model.calls[0][1] == 1858).all()
    assert (model.calls[-1][1] == 1858).all()
    assert (model.calls[7][1] != 1858).sum(1).tolist() == [7, 7]
    assert model.planner == original
    first = result['dfm_one_pass'].masked_fill(~legal, -torch.inf).argmax(-1)
    last = result['dfm_eight_pass'].masked_fill(~legal, -torch.inf).argmax(-1)
    assert first.tolist() == [0, 0]
    assert last.tolist() == [1, 1]
