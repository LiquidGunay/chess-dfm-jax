import chess
import numpy as np
import pytest
import torch

from chess_dfm_jax.encoding import encode_board
from research.evaluate_common_policy import input_planes, legal_metrics


def test_current_only_transform_exactly_matches_original_preprocessing():
    board = chess.Board()
    history = [board.copy(stack=False)]
    for move in "g1f3 g8f6 f3g1 f6g8 e2e4 d7d5 e4d5 d8d5 b1c3 d5d6".split():
        board.push_uci(move)
        history.append(board.copy(stack=False))
        stored = encode_board(board, history)[None].astype(np.uint8)
        transformed = input_planes(stored, "current_only_as_preprocessed")
        np.testing.assert_array_equal(transformed[0], encode_board(board, []))
        np.testing.assert_array_equal(input_planes(stored, "lc0_full_history_as_preprocessed"), stored)


def test_root_metrics_use_same_targets_and_only_legal_moves():
    logits = torch.tensor([[1., 100., 2.], [3., 1., 200.]])
    legal = torch.tensor([[True, False, True], [True, True, False]])
    targets = torch.tensor([2, 1])
    ce, correct = legal_metrics(logits, legal, targets)
    assert correct == 1
    expected = torch.nn.functional.cross_entropy(torch.tensor([[1., 2.], [3., 1.]]), torch.tensor([1, 1]), reduction="sum")
    assert ce == pytest.approx(float(expected))
    with pytest.raises(ValueError):
        legal_metrics(logits, legal, torch.tensor([1, 2]))
