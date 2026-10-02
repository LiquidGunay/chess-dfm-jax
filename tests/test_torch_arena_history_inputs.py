from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch

from chess_dfm_jax.encoding import encode_board
from research.arena_history_trust import _mint_trusted_arena_history_endpoint
from research.evaluate_arena import TrackingPolicy, PolicyTracker
from research.input_contract import CURRENT_ONLY, LC0_FULL_HISTORY, torch_hero_plane_history_mode
from research.train_torch import HERO_CONFIG, TorchHeroArenaPolicy


@pytest.mark.parametrize("record, expected", [
    ({}, CURRENT_ONLY),
    ({"args": {"data_format": "trajectory_v3"}}, CURRENT_ONLY),
    ({"args": {"data_format": "lc0_sequential"}}, LC0_FULL_HISTORY),
    ({"data": {"schema_version": "lc0-sequential-v1"}}, LC0_FULL_HISTORY),
])
def test_history_mode_comes_from_training_provenance(record, expected):
    assert torch_hero_plane_history_mode(record) == expected


@pytest.mark.parametrize("record", [
    {"args": {"data_format": "unknown"}},
    {"args": {"data_format": "trajectory_v3"}, "data": {"schema_version": "lc0-sequential-v1"}},
])
def test_history_provenance_fails_closed(record):
    with pytest.raises(ValueError):
        torch_hero_plane_history_mode(record)


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = HERO_CONFIG
        self.encoder = SimpleNamespace(policy_head=True, encode_current=self.capture)
        self.planes = None

    def capture(self, planes, **kwargs):
        self.planes = planes.clone()
        return planes

    def policy_outputs(self, tokens, dtype):
        return torch.zeros((tokens.shape[0], 1858)), None


def position():
    board = chess.Board()
    history = [board.copy(stack=False)]
    # Includes a repeated position, both colors, and subsequent irreversible moves.
    for move in "g1f3 g8f6 f3g1 f6g8 e2e4 d7d5".split():
        board.push_uci(move)
        history.append(board.copy(stack=False))
    return board, history


def endpoint(board, history, *, include=True):
    return _mint_trusted_arena_history_endpoint(
        pair_id="fixture", opening_index=0, game_in_pair=0,
        current_fen=board.fen(en_passant="legal"), position_count=len(history),
        authoritative_move_stack_length=board.ply(), root_ply=board.ply(),
        additional_plies=0, played_suffix_count=0, ply_cap=256,
        plane_history_fens=tuple(b.fen(en_passant="legal") for b in history) if include else None,
    )


@pytest.mark.parametrize("mode", [CURRENT_ONLY, LC0_FULL_HISTORY])
@pytest.mark.parametrize("trusted", [False, True])
def test_native_adapter_receives_exact_checkpoint_inputs(mode, trusted):
    model = CaptureModel()
    policy = TorchHeroArenaPolicy(model, "fixture", "policy_only", inference_batch_size=2, plane_history_mode=mode)
    board, history = position()
    if trusted:
        policy.select_actions_from_trusted_arena([board.copy(stack=False)], [endpoint(board, history)])
    else:
        policy.select_actions([board], [history])
    expected = encode_board(board, history if mode == LC0_FULL_HISTORY else [])
    np.testing.assert_array_equal(model.planes[0].numpy(), expected)
    np.testing.assert_array_equal(model.planes[1].numpy(), expected)  # static padding
    assert bool(np.any(expected[13:104])) == (mode == LC0_FULL_HISTORY)


def test_full_history_adapter_rejects_endpoint_without_history():
    model = CaptureModel()
    policy = TorchHeroArenaPolicy(model, "fixture", "policy_only", plane_history_mode=LC0_FULL_HISTORY)
    board, history = position()
    with pytest.raises(ValueError, match="requires attested"):
        policy.select_actions_from_trusted_arena([board.copy(stack=False)], [endpoint(board, history, include=False)])
    assert model.planes is None


def test_tracking_wrapper_preserves_input_contract():
    model = CaptureModel()
    delegate = TorchHeroArenaPolicy(model, "fixture", "policy_only", plane_history_mode=LC0_FULL_HISTORY)
    policy = TrackingPolicy(delegate, PolicyTracker())
    assert policy.plane_history_mode == LC0_FULL_HISTORY
