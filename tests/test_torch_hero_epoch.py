from __future__ import annotations

import dataclasses

import chess
import numpy as np
import pytest
import torch

from chess_dfm_jax.policy import (
    LC0_CANONICAL_1858_INPUT_FORMAT,
    encode_lc0_canonical_1858,
    legal_move_mask,
)
from research.train_torch import (
    HERO_CONFIG,
    MuonAdamW,
    StateProjector,
    _polarized_gradient_cosines,
    _shared_sigreg_gradient_projection,
    _sigreg_v_stat,
    canonicalize_trajectory_batch,
)


def _canonical_batch(boards_and_moves: list[tuple[chess.Board, chess.Move]]):
    batch_size = len(boards_and_moves)
    legacy_legal = np.full((batch_size, 1, 128), 65535, dtype=np.int32)
    legacy_count = np.zeros((batch_size, 1), dtype=np.int32)
    for row, (board, _) in enumerate(boards_and_moves):
        legal = np.flatnonzero(legal_move_mask(board, "lc0_1858"))
        legacy_legal[row, 0, : legal.size] = legal
        legacy_count[row, 0] = legal.size
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
        "legal_idx": legacy_legal,
        "legal_count": legacy_count,
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


def test_canonical_batch_conversion_recovers_chess960_castling_semantics():
    board = chess.Board(
        "rk2r3/p1pp1n1p/1p4n1/3PqN1Q/8/1P6/P1P1PN1P/1RK1R3 w KQ - 5 18",
        chess960=True,
    )
    castle = chess.Move.from_uci("c1b1")
    assert board.is_valid()
    assert board.is_castling(castle)
    converted = canonicalize_trajectory_batch(
        _canonical_batch([(board, castle)])
    )

    expected = encode_lc0_canonical_1858(board, castle)
    assert converted["action_indices"][0, 0] == expected
    assert expected in converted["legal_idx"][0, 0, : converted["legal_count"][0, 0]]


def test_canonical_conversion_uses_stored_legality_for_ambiguous_chess960_fen():
    fen = "rb2k3/p2pn2p/1pp1n1p1/3q4/2P5/1Q3r1P/PP1P1B2/2KBR1N1 w q - 0 23"
    moves = [
        chess.Move.from_uci(uci)
        for uci in (
            "g1f3",
            "d5f5",
            "b3c3",
            "b8d6",
            "c3h8",
            "f5f8",
            "h8h7",
            "e8a8",
        )
    ]
    board = chess.Board(fen, chess960=True)
    assert board.is_valid()
    assert chess.Board(fen).is_valid()
    legacy_legal = np.full((1, 8, 128), 65535, dtype=np.int32)
    legacy_count = np.zeros((1, 8), dtype=np.int32)
    expected_actions: list[int] = []
    for offset, move in enumerate(moves):
        legal = np.flatnonzero(legal_move_mask(board, "lc0_1858"))
        legacy_legal[0, offset, : legal.size] = legal
        legacy_count[0, offset] = legal.size
        expected_actions.append(encode_lc0_canonical_1858(board, move))
        board.push(move)

    converted = canonicalize_trajectory_batch(
        {
            "fen_t": np.asarray([fen]),
            "input_format": np.asarray([LC0_CANONICAL_1858_INPUT_FORMAT]),
            "actions_uci": np.asarray([[move.uci() for move in moves]]),
            "future_valid": np.ones((1, 8), dtype=np.float32),
            "action_indices": np.zeros((1, 8), dtype=np.int32),
            "action_idx": np.zeros((1,), dtype=np.int32),
            "legal_idx": legacy_legal,
            "legal_count": legacy_count,
        }
    )

    np.testing.assert_array_equal(
        converted["action_indices"][0],
        np.asarray(expected_actions),
    )
    assert expected_actions[-1] in converted["legal_idx"][
        0,
        7,
        : converted["legal_count"][0, 7],
    ]


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


def test_hero_loss_contract_is_frozen_after_gradient_audit():
    assert HERO_CONFIG.root_legal_ce_coeff == pytest.approx(
        0.08707768618513193
    )
    assert HERO_CONFIG.target_sigreg_coeff == HERO_CONFIG.pred_sigreg_coeff == 2.0
    assert HERO_CONFIG.sigreg_example_count == 64
    assert HERO_CONFIG.loss_clip_value == 0.0


def test_gradient_polarization_recovers_component_cosine():
    groups = (
        "raw_bt4",
        "state_projector",
        "dfm_state_projector",
        "jepa",
        "value_wdl",
        "dfm_policy",
        "all",
    )
    left = {group: {"squared_norm": 9.0} for group in groups}
    right = {group: {"squared_norm": 16.0} for group in groups}
    orthogonal_sum = {group: {"squared_norm": 25.0} for group in groups}

    result = _polarized_gradient_cosines(left, right, orthogonal_sum)

    for group in groups:
        assert result[group]["dot"] == 0.0
        assert result[group]["cosine"] == 0.0


def test_shared_sigreg_projection_recovers_vector_rescaling():
    groups = (
        "raw_bt4",
        "state_projector",
        "dfm_state_projector",
        "jepa",
        "value_wdl",
        "dfm_policy",
        "all",
    )

    def record(squared_norm: float):
        return {
            "gradient_groups": {
                group: {"squared_norm": squared_norm} for group in groups
            }
        }

    records = {
        "policy_total": record(1.0),
        "non_sigreg_representation": record(4.0),
        "sigreg_total": record(9.0),
    }
    interactions = {
        "policy_vs_representation": {
            group: {"dot": 0.0} for group in groups
        },
        "policy_vs_sigreg": {
            group: {"dot": 0.0} for group in groups
        },
        "non_sigreg_vs_sigreg": {
            group: {"dot": 0.0} for group in groups
        },
    }

    projection = _shared_sigreg_gradient_projection(
        records,
        interactions,
        baseline_coefficient=2.0,
        candidate_coefficients=(0.0, 1.0, 2.0),
    )

    assert projection["candidates"]["0"]["gradient_groups"]["all"][
        "representation_norm"
    ] == 2.0
    assert projection["candidates"]["1"]["gradient_groups"]["all"][
        "representation_norm"
    ] == 2.5
    assert projection["candidates"]["2"]["gradient_groups"]["all"][
        "total_norm"
    ] == pytest.approx(14.0**0.5)


def test_head_sdpa_preserves_manual_attention_semantics():
    manual_config = dataclasses.replace(
        HERO_CONFIG,
        z_dim=32,
        projector_layers=2,
        projector_heads=4,
        projector_mlp_dim=48,
        remat_blocks=False,
        remat_projector_blocks=False,
        use_head_sdpa=False,
    )
    sdpa_config = dataclasses.replace(manual_config, use_head_sdpa=True)
    torch.manual_seed(41)
    manual = StateProjector(manual_config)
    with torch.no_grad():
        for parameter in manual.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    sdpa = StateProjector(sdpa_config)
    sdpa.load_state_dict(manual.state_dict())
    inputs = torch.randn(3, 64, 1024) * 0.1

    expected = manual(inputs, torch.float32)
    actual = sdpa(inputs, torch.float32)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


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
