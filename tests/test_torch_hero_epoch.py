from __future__ import annotations

import dataclasses

import chess
import numpy as np
import torch

from chess_dfm_jax.policy import (
    LC0_CANONICAL_1858_INPUT_FORMAT,
    encode_lc0_canonical_1858,
)
from research.train_torch import (
    HERO_CONFIG,
    MuonAdamW,
    _sigreg_v_stat,
    canonicalize_trajectory_batch,
)


def _canonical_batch(boards_and_moves: list[tuple[chess.Board, chess.Move]]):
    batch_size = len(boards_and_moves)
    return {
        "fen_t": np.asarray([board.fen() for board, _ in boards_and_moves]),
        "input_format": np.asarray(
            [LC0_CANONICAL_1858_INPUT_FORMAT] * batch_size
        ),
        "actions_uci": np.asarray(
            [[move.uci()] for _, move in boards_and_moves]
        ),
        "future_valid": np.ones((batch_size, 1), dtype=np.float32),
        "action_indices": np.zeros((batch_size, 1), dtype=np.int32),
        "action_idx": np.zeros((batch_size,), dtype=np.int32),
        "legal_idx": np.full((batch_size, 1, 128), 65535, dtype=np.int32),
        "legal_count": np.zeros((batch_size, 1), dtype=np.int32),
    }


def test_canonical_batch_conversion_covers_black_moves_and_knight_promotions():
    black = chess.Board()
    black.push_uci("e2e4")
    black_move = chess.Move.from_uci("e7e5")
    promotion = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    promotion_move = chess.Move.from_uci("a7a8n")
    converted = canonicalize_trajectory_batch(
        _canonical_batch(
            [
                (black, black_move),
                (promotion, promotion_move),
            ]
        )
    )

    expected = np.asarray(
        [
            encode_lc0_canonical_1858(black, black_move),
            encode_lc0_canonical_1858(promotion, promotion_move),
        ]
    )
    np.testing.assert_array_equal(converted["action_indices"][:, 0], expected)
    assert converted["legal_masks_valid"].tolist() == [[1.0], [1.0]]
    for row, action in enumerate(expected):
        count = converted["legal_count"][row, 0]
        assert action in converted["legal_idx"][row, 0, :count]


def test_normalized_sigreg_does_not_change_when_samples_are_duplicated():
    rng = np.random.default_rng(23)
    z = rng.standard_normal((17, 12), dtype=np.float32)
    weight = rng.random(17, dtype=np.float32)
    directions = rng.standard_normal((12, 9), dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)

    value, count = _sigreg_v_stat(
        torch.from_numpy(z),
        torch.from_numpy(weight),
        torch.from_numpy(directions),
        reference_count=1.0,
    )
    duplicated_value, duplicated_count = _sigreg_v_stat(
        torch.from_numpy(np.concatenate((z, z), axis=0)),
        torch.from_numpy(np.concatenate((weight, weight), axis=0)),
        torch.from_numpy(directions),
        reference_count=1.0,
    )

    torch.testing.assert_close(duplicated_value, value, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(duplicated_count, 2.0 * count, rtol=0.0, atol=1e-6)


class _ScheduleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.raw = torch.nn.Parameter(torch.ones(4, 4))
        self.fresh_matrix = torch.nn.Parameter(torch.ones(4, 4))
        self.fresh_bias = torch.nn.Parameter(torch.ones(4))
        self.fresh_norm_scale = torch.nn.Parameter(torch.ones(4))
        self.fresh_embedding = torch.nn.Parameter(torch.ones(8, 4))


def test_hero_schedule_is_example_based_and_decay_is_fresh_matrix_only():
    config = dataclasses.replace(
        HERO_CONFIG,
        lr_warmup_examples=100,
        lr_total_examples=1000,
    )
    small_batch = MuonAdamW(
        _ScheduleModel(),
        config,
        examples_per_update=10,
    )
    large_batch = MuonAdamW(
        _ScheduleModel(),
        config,
        examples_per_update=20,
    )

    small_batch.examples_seen = 190
    large_batch.examples_seen = 180
    assert small_batch.learning_rate_ratio() == large_batch.learning_rate_ratio()
    decay = {leaf.name: leaf.apply_weight_decay for leaf in small_batch.leaves}
    assert decay == {
        "fresh_matrix": True,
        "fresh_bias": False,
        "fresh_norm_scale": False,
        "fresh_embedding": False,
        "encoder.raw": False,
    }
