from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import chess
import numpy as np
import pytest

from chess_dfm_jax.policy import (
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_VOCAB_SIZE,
    encode_action,
)
from research.arena import make_color_reversed_pairs
from research.arena_history_trust import (
    HISTORY_VALIDATION_SCHEMA,
    HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    verify_trusted_arena_history_endpoint,
)
from research.play_arena import (
    FAULT_EXCEPTION,
    FAULT_ILLEGAL_ACTION,
    FAULT_NONFINITE_OUTPUT,
    FAULT_NO_REPRESENTABLE_MOVE,
    FAULT_TIMEOUT,
    ArenaOpeningHistory,
    _assert_game_state_history_invariant,
    _GameState,
    play_arena_pairs,
)


@dataclasses.dataclass(frozen=True)
class _Selection:
    action_indices: Any


class _DeterministicPolicy:
    action_codec_id = ACTION_CODEC_LC0_CANONICAL_1858

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.batch_sizes: list[int] = []
        self.history_lengths: list[tuple[int, ...]] = []

    def select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Sequence[chess.Board]],
    ) -> _Selection:
        self.batch_sizes.append(len(boards))
        self.history_lengths.append(tuple(len(history) for history in histories))
        indices = []
        for board, history in zip(boards, histories, strict=True):
            assert history
            assert history[-1].fen(en_passant="legal") == board.fen(en_passant="legal")
            move = min(board.legal_moves, key=lambda candidate: candidate.uci())
            indices.append(
                encode_action(
                    move,
                    codec_id=self.action_codec_id,
                    board=board,
                )
            )
        return _Selection(np.asarray(indices, dtype=np.int32))


class _ForcedMovePolicy(_DeterministicPolicy):
    def __init__(self, model_id: str, forced_uci: str):
        super().__init__(model_id)
        self.forced_uci = forced_uci

    def select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Sequence[chess.Board]],
    ) -> _Selection:
        self.batch_sizes.append(len(boards))
        self.history_lengths.append(tuple(len(history) for history in histories))
        indices = []
        for board in boards:
            move = chess.Move.from_uci(self.forced_uci)
            assert board.is_legal(move)
            indices.append(
                encode_action(
                    move,
                    codec_id=self.action_codec_id,
                    board=board,
                )
            )
        return _Selection(np.asarray(indices, dtype=np.int32))


class _FaultPolicy(_DeterministicPolicy):
    def __init__(self, model_id: str, fault_kind: str):
        super().__init__(model_id)
        self.fault_kind = fault_kind

    def select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Sequence[chess.Board]],
    ) -> _Selection:
        self.batch_sizes.append(len(boards))
        self.history_lengths.append(tuple(len(history) for history in histories))
        if self.fault_kind == FAULT_EXCEPTION:
            raise RuntimeError("synthetic adapter failure")
        if self.fault_kind == FAULT_TIMEOUT:
            raise TimeoutError("synthetic adapter timeout")
        if self.fault_kind == FAULT_NONFINITE_OUTPUT:
            return _Selection(np.full((len(boards),), np.nan))
        if self.fault_kind == FAULT_ILLEGAL_ACTION:
            return _Selection(np.full((len(boards),), -1, dtype=np.int32))
        raise AssertionError(f"Unsupported synthetic fault: {self.fault_kind}")


class _FailOnCallPolicy(_DeterministicPolicy):
    def __init__(self, model_id: str, *, fail_call: int):
        super().__init__(model_id)
        self.fail_call = fail_call

    def select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Sequence[chess.Board]],
    ) -> _Selection:
        call_number = len(self.batch_sizes) + 1
        if call_number == self.fail_call:
            self.batch_sizes.append(len(boards))
            self.history_lengths.append(tuple(len(history) for history in histories))
            raise RuntimeError("synthetic chunk-local failure")
        return super().select_actions(boards, histories)


class _TrustedEndpointPolicy(_DeterministicPolicy):
    history_validation_mode = HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT
    trusted_arena_history_schema = HISTORY_VALIDATION_SCHEMA

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.endpoint_additional_plies: list[tuple[int, ...]] = []
        self.endpoint_stack_lengths: list[tuple[int, ...]] = []

    def select_actions(self, boards, histories):
        raise AssertionError("trusted endpoint policy used the full replay boundary")

    def select_actions_from_trusted_arena(
        self,
        boards: Sequence[chess.Board],
        endpoints: Sequence[Any],
    ) -> _Selection:
        self.batch_sizes.append(len(boards))
        additional_plies: list[int] = []
        stack_lengths: list[int] = []
        indices: list[int] = []
        for board, raw_endpoint in zip(boards, endpoints, strict=True):
            endpoint = verify_trusted_arena_history_endpoint(raw_endpoint)
            assert board.move_stack == []
            assert endpoint.current_fen == board.fen(en_passant="legal")
            assert endpoint.authoritative_move_stack_length == board.ply()
            assert endpoint.position_count == board.ply() + 1
            move = min(board.legal_moves, key=lambda candidate: candidate.uci())
            indices.append(
                encode_action(
                    move,
                    codec_id=self.action_codec_id,
                    board=board,
                )
            )
            additional_plies.append(endpoint.additional_plies)
            stack_lengths.append(len(board.move_stack))
        self.endpoint_additional_plies.append(tuple(additional_plies))
        self.endpoint_stack_lengths.append(tuple(stack_lengths))
        return _Selection(np.asarray(indices, dtype=np.int32))


def _history_from_moves(moves_uci: Sequence[str]) -> tuple[str, ...]:
    board = chess.Board()
    fens = [board.fen(en_passant="legal")]
    for move_uci in moves_uci:
        board.push_uci(move_uci)
        fens.append(board.fen(en_passant="legal"))
    return tuple(fens)


def _pairs_and_histories(
    moves_uci: Sequence[str],
    *,
    count: int = 1,
) -> tuple[
    tuple[tuple[Any, Any], ...],
    dict[str, ArenaOpeningHistory],
]:
    history = _history_from_moves(moves_uci)
    pairs = make_color_reversed_pairs(
        [history[-1]] * count,
        model_a="candidate",
        model_b="reference",
    )
    histories = {
        pair[0].pair_id: ArenaOpeningHistory(
            pair_id=pair[0].pair_id,
            fens=history,
        )
        for pair in pairs
    }
    return pairs, histories


def test_identical_policies_reverse_colors_and_emit_deterministic_payload():
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5", "g1f3", "b8c6"))
    first_candidate = _DeterministicPolicy("candidate")
    first_reference = _DeterministicPolicy("reference")
    first = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={
            "candidate": first_candidate,
            "reference": first_reference,
        },
        additional_ply_cap=6,
    )
    second = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={
            "candidate": _DeterministicPolicy("candidate"),
            "reference": _DeterministicPolicy("reference"),
        },
        additional_ply_cap=6,
    )

    assert first.as_dict() == second.as_dict()
    assert first.payload_sha256 == second.payload_sha256
    assert first.as_dict()["opening_histories"][0]["fens"] == list(
        histories[pairs[0][0].pair_id].fens
    )
    assert [record.outcome.game_in_pair for record in first.records] == [0, 1]
    assert all(record.outcome.termination == "ply_cap" for record in first.records)
    assert first.records[0].moves_uci == first.records[1].moves_uci
    assert first.records[0].outcome.white_model == "candidate"
    assert first.records[1].outcome.black_model == "candidate"
    assert {stats.model_id: stats.points for stats in first.model_stats} == {
        "candidate": 1.0,
        "reference": 1.0,
    }
    assert first_candidate.history_lengths[0] == (5,)
    assert first_reference.history_lengths[0] == (5,)


def test_active_positions_are_grouped_by_model_in_stable_batches():
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5"), count=2)
    candidate = _DeterministicPolicy("candidate")
    reference = _DeterministicPolicy("reference")
    result = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={"candidate": candidate, "reference": reference},
        additional_ply_cap=1,
    )

    assert candidate.batch_sizes == [2]
    assert reference.batch_sizes == [2]
    assert result.total_policy_calls == 2
    assert result.max_policy_batch_size == 2
    assert [record.outcome.pair_id for record in result.records] == [
        pairs[0][0].pair_id,
        pairs[0][0].pair_id,
        pairs[1][0].pair_id,
        pairs[1][0].pair_id,
    ]


def test_trusted_endpoint_runner_exactly_matches_full_replay_gameplay():
    pairs, histories = _pairs_and_histories(
        ("e2e4", "e7e5", "g1f3", "b8c6"),
        count=2,
    )
    full = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={
            "candidate": _DeterministicPolicy("candidate"),
            "reference": _DeterministicPolicy("reference"),
        },
        additional_ply_cap=6,
    )
    candidate = _TrustedEndpointPolicy("candidate")
    reference = _TrustedEndpointPolicy("reference")
    endpoint = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={
            "candidate": candidate,
            "reference": reference,
        },
        additional_ply_cap=6,
    )

    assert endpoint.as_dict() == full.as_dict()
    assert endpoint.payload_sha256 == full.payload_sha256
    assert all(
        stack_length == 0
        for policy in (candidate, reference)
        for call in policy.endpoint_stack_lengths
        for stack_length in call
    )
    assert sorted(
        {
            additional_ply
            for policy in (candidate, reference)
            for call in policy.endpoint_additional_plies
            for additional_ply in call
        }
    ) == list(range(6))


def test_policy_batches_are_stably_bounded_and_chunk_faults_are_isolated():
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5"), count=3)
    candidate = _FailOnCallPolicy("candidate", fail_call=2)
    reference = _DeterministicPolicy("reference")
    result = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={"candidate": candidate, "reference": reference},
        additional_ply_cap=1,
        policy_batch_size_cap=2,
    )

    assert candidate.batch_sizes == [2, 1]
    assert reference.batch_sizes == [2, 1]
    assert result.policy_batch_size_cap == 2
    assert result.max_policy_batch_size == 2
    assert result.total_policy_calls == 4
    assert result.as_dict()["config"]["policy_batch_size_cap"] == 2
    faults = [record for record in result.records if record.fault_kind is not None]
    assert len(faults) == 1
    assert faults[0].outcome.pair_id == pairs[2][0].pair_id
    assert faults[0].fault_model == "candidate"
    by_model = {stats.model_id: stats for stats in result.model_stats}
    assert by_model["candidate"].fault_losses == 1
    assert by_model["candidate"].points == 2.5
    assert by_model["reference"].points == 3.5


@pytest.mark.parametrize(
    ("batch_size_cap", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_policy_batch_size_cap_must_be_a_positive_integer(
    batch_size_cap: Any,
    error_type: type[Exception],
):
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5"))
    policies = {
        "candidate": _DeterministicPolicy("candidate"),
        "reference": _DeterministicPolicy("reference"),
    }
    with pytest.raises(error_type, match="policy_batch_size_cap"):
        play_arena_pairs(
            pairs,
            opening_histories=histories,
            policies=policies,
            additional_ply_cap=1,
            policy_batch_size_cap=batch_size_cap,
        )
    assert policies["candidate"].batch_sizes == []
    assert policies["reference"].batch_sizes == []


def test_stats_filter_games_for_each_model_in_a_multi_opponent_pool():
    history = _history_from_moves(("e2e4", "e7e5"))
    first_pair = make_color_reversed_pairs(
        [history[-1]],
        model_a="candidate",
        model_b="reference-a",
    )[0]
    second_pair = make_color_reversed_pairs(
        [history[-1]],
        model_a="candidate",
        model_b="reference-b",
        start_index=1,
    )[0]
    pairs = (first_pair, second_pair)
    histories = {
        pair[0].pair_id: ArenaOpeningHistory(
            pair_id=pair[0].pair_id,
            fens=history,
        )
        for pair in pairs
    }
    result = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={
            model_id: _DeterministicPolicy(model_id)
            for model_id in ("candidate", "reference-a", "reference-b")
        },
        additional_ply_cap=1,
    )
    by_model = {stats.model_id: stats for stats in result.model_stats}
    assert by_model["candidate"].games == 4
    assert by_model["candidate"].points == 2.0
    assert by_model["reference-a"].games == 2
    assert by_model["reference-a"].points == 1.0
    assert by_model["reference-b"].games == 2
    assert by_model["reference-b"].points == 1.0


def test_normal_terminal_result_precedes_draw_at_exact_additional_ply_cap():
    pairs, histories = _pairs_and_histories(("f2f3", "e7e5", "g2g4"))
    candidate = _ForcedMovePolicy("candidate", "d8h4")
    reference = _ForcedMovePolicy("reference", "d8h4")
    result = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={"candidate": candidate, "reference": reference},
        additional_ply_cap=1,
    )

    assert [record.outcome.result for record in result.records] == ["0-1", "0-1"]
    assert all(record.outcome.termination == "normal" for record in result.records)
    assert all(record.normal_termination == "checkmate" for record in result.records)
    assert all(record.outcome.ply_count == 1 for record in result.records)


@pytest.mark.parametrize(
    "fault_kind",
    [
        FAULT_ILLEGAL_ACTION,
        FAULT_NONFINITE_OUTPUT,
        FAULT_EXCEPTION,
        FAULT_TIMEOUT,
    ],
)
def test_policy_faults_are_acting_model_losses_without_fallback(fault_kind: str):
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5"))
    result = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={
            "candidate": _FaultPolicy("candidate", fault_kind),
            "reference": _DeterministicPolicy("reference"),
        },
        additional_ply_cap=4,
    )

    assert [record.fault_kind for record in result.records] == [
        fault_kind,
        fault_kind,
    ]
    assert all(record.fault_model == "candidate" for record in result.records)
    assert all(record.outcome.loser_model == "candidate" for record in result.records)
    assert all(record.outcome.score_for("candidate") == 0.0 for record in result.records)
    assert result.records[0].moves_uci == ()
    assert len(result.records[1].moves_uci) == 1


def test_empty_representable_mask_is_loss_and_never_calls_policy(monkeypatch):
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5"))
    candidate = _DeterministicPolicy("candidate")
    reference = _DeterministicPolicy("reference")
    monkeypatch.setattr(
        "research.play_arena.legal_action_mask",
        lambda board, *, codec_id: np.zeros(ACTION_VOCAB_SIZE, dtype=bool),
    )
    result = play_arena_pairs(
        pairs,
        opening_histories=histories,
        policies={"candidate": candidate, "reference": reference},
        additional_ply_cap=4,
    )

    assert all(record.fault_kind == FAULT_NO_REPRESENTABLE_MOVE for record in result.records)
    assert {record.fault_model for record in result.records} == {
        "candidate",
        "reference",
    }
    assert candidate.batch_sizes == []
    assert reference.batch_sizes == []
    assert result.total_policy_calls == 0


def test_missing_or_truncated_opening_history_fails_before_policy_call():
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5", "g1f3", "b8c6"))
    policies = {
        "candidate": _DeterministicPolicy("candidate"),
        "reference": _DeterministicPolicy("reference"),
    }
    with pytest.raises(ValueError, match="exactly match pair IDs"):
        play_arena_pairs(
            pairs,
            opening_histories={},
            policies=policies,
            additional_ply_cap=2,
        )

    pair_id = pairs[0][0].pair_id
    truncated = {
        pair_id: ArenaOpeningHistory(
            pair_id=pair_id,
            fens=(histories[pair_id].fens[-1],),
        )
    }
    with pytest.raises(ValueError, match="standard initial position"):
        play_arena_pairs(
            pairs,
            opening_histories=truncated,
            policies=policies,
            additional_ply_cap=2,
        )
    assert policies["candidate"].batch_sizes == []
    assert policies["reference"].batch_sizes == []


def test_corrupt_opening_transition_fails_before_trusted_endpoint_call():
    pairs, histories = _pairs_and_histories(("e2e4", "e7e5", "g1f3"))
    pair_id = pairs[0][0].pair_id
    corrupted_fens = list(histories[pair_id].fens)
    corrupted_fens[1] = _history_from_moves(("d2d4",))[-1]
    policies = {
        "candidate": _TrustedEndpointPolicy("candidate"),
        "reference": _TrustedEndpointPolicy("reference"),
    }

    with pytest.raises(ValueError, match="legal reconstructions"):
        play_arena_pairs(
            pairs,
            opening_histories={
                pair_id: ArenaOpeningHistory(
                    pair_id=pair_id,
                    fens=tuple(corrupted_fens),
                )
            },
            policies=policies,
            additional_ply_cap=2,
        )
    assert policies["candidate"].batch_sizes == []
    assert policies["reference"].batch_sizes == []


def test_threefold_terminal_opening_never_mints_trusted_endpoint():
    repetition = (
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    )
    pairs, histories = _pairs_and_histories(repetition)
    policies = {
        "candidate": _TrustedEndpointPolicy("candidate"),
        "reference": _TrustedEndpointPolicy("reference"),
    }

    with pytest.raises(ValueError, match="terminal"):
        play_arena_pairs(
            pairs,
            opening_histories=histories,
            policies=policies,
            additional_ply_cap=2,
        )
    assert policies["candidate"].batch_sizes == []
    assert policies["reference"].batch_sizes == []


def test_authoritative_state_invariant_rejects_corrupted_endpoint_and_suffix():
    moves = ("e2e4", "e7e5")
    pairs, _histories = _pairs_and_histories(moves)
    board = chess.Board()
    history_boards = [board.copy(stack=False)]
    for move in moves:
        board.push_uci(move)
        history_boards.append(board.copy(stack=False))
    state = _GameState(
        spec=pairs[0][0],
        board=board,
        history=history_boards,
        moves_uci=[],
    )
    _assert_game_state_history_invariant(state)

    state.history[-1] = chess.Board()
    with pytest.raises(RuntimeError, match="endpoint diverged"):
        _assert_game_state_history_invariant(state)
    state.history[-1] = board.copy(stack=False)
    state.moves_uci.append("a2a3")
    with pytest.raises(RuntimeError, match="played suffix"):
        _assert_game_state_history_invariant(state)
