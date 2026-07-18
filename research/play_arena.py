"""Fail-closed, in-process batched gameplay for the paired research arena.

This module owns chess state, batching, legality, and adjudication.  Model
adapters remain checkpoint- and framework-specific: they receive defensive
copies of boards and their complete histories, then return one action index per
position.

A FEN by itself cannot recover repetition and claim state.  Therefore every
color-reversed pair must provide an :class:`ArenaOpeningHistory` containing the
exact position sequence from the standard initial position through the opening
FEN.  Missing, truncated, or discontinuous histories fail before play.

The recovered local model is a separate concern: its source preprocessing used
``encode_board(board, [])``, so its adapter must encode the current board only.
The histories supplied at this boundary preserve chess state and make the game
auditable; they must not silently change that model's input distribution.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import operator
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import chess
import numpy as np

from chess_dfm_jax.policy import (
    ACTION_CODEC_LC0_CANONICAL_1858,
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
    ActionCodecError,
    decode_action,
    encode_action,
    legal_action_mask,
)
from research.arena import (
    ArenaGameSpec,
    GameOutcome,
    classify_game_outcome,
)


GAMEPLAY_SCHEMA = "chess-dfm-batched-arena-gameplay-v1"

FAULT_NO_REPRESENTABLE_MOVE = "no_representable_move"
FAULT_ILLEGAL_ACTION = "illegal_action"
FAULT_NONFINITE_OUTPUT = "nonfinite_output"
FAULT_EXCEPTION = "exception"
FAULT_TIMEOUT = "timeout"
FAULT_KINDS = frozenset(
    {
        FAULT_NO_REPRESENTABLE_MOVE,
        FAULT_ILLEGAL_ACTION,
        FAULT_NONFINITE_OUTPUT,
        FAULT_EXCEPTION,
        FAULT_TIMEOUT,
    }
)
SUPPORTED_ACTION_CODECS = frozenset(
    {
        ACTION_CODEC_LEGACY_ABSOLUTE_1858,
        ACTION_CODEC_LC0_CANONICAL_1858,
    }
)


class BatchedActionSelection(Protocol):
    """Structural result required from an arena policy adapter."""

    action_indices: Any


class BatchedArenaPolicy(Protocol):
    """Minimal checkpoint-free boundary used by :func:`play_arena_pairs`."""

    model_id: str
    action_codec_id: str

    def select_actions(
        self,
        boards: Sequence[chess.Board],
        histories: Sequence[Sequence[chess.Board]],
    ) -> BatchedActionSelection:
        """Return one action index per board in the original batch order."""


def _canonical_position_fen(board: chess.Board) -> str:
    return board.fen(en_passant="legal")


def _parse_canonical_history_fen(fen: str, *, history_index: int) -> chess.Board:
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError(f"Opening history FEN {history_index} must be non-empty.")
    try:
        board = chess.Board(" ".join(fen.split()))
    except ValueError as exc:
        raise ValueError(f"Opening history FEN {history_index} is invalid: {fen!r}") from exc
    if not board.is_valid():
        raise ValueError(
            f"Opening history FEN {history_index} is not valid standard chess: {fen!r}"
        )
    canonical_fen = _canonical_position_fen(board)
    if fen != canonical_fen:
        raise ValueError(
            f"Opening history FEN {history_index} is not canonical: expected {canonical_fen!r}"
        )
    return board


@dataclasses.dataclass(frozen=True)
class ArenaOpeningHistory:
    """Exact standard-game position history shared by one reversed pair.

    ``fens`` is chronological, includes the standard initial position, and
    includes the pair's current opening position as its final entry.
    """

    pair_id: str
    fens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("ArenaOpeningHistory pair_id must be non-empty.")
        fens = tuple(self.fens)
        if not fens:
            raise ValueError("ArenaOpeningHistory fens must be non-empty.")
        if not all(isinstance(fen, str) for fen in fens):
            raise TypeError("ArenaOpeningHistory fens must contain strings.")
        object.__setattr__(self, "fens", fens)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "pair_id": self.pair_id,
            "fens": list(self.fens),
        }
        payload["history_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        return payload


@dataclasses.dataclass
class _GameState:
    spec: ArenaGameSpec
    board: chess.Board
    history: list[chess.Board]
    moves_uci: list[str]
    additional_plies: int = 0


@dataclasses.dataclass(frozen=True)
class ArenaGameRecord:
    """Auditable result for one game, including its exact played move suffix."""

    outcome: GameOutcome
    final_fen: str
    moves_uci: tuple[str, ...]
    normal_termination: str | None = None
    fault_kind: str | None = None
    fault_model: str | None = None

    def __post_init__(self) -> None:
        try:
            board = chess.Board(self.final_fen)
        except ValueError as exc:
            raise ValueError(f"Invalid final FEN: {self.final_fen!r}") from exc
        if not board.is_valid() or _canonical_position_fen(board) != self.final_fen:
            raise ValueError("ArenaGameRecord final_fen must be canonical standard chess.")
        if self.fault_kind is None:
            if self.fault_model is not None:
                raise ValueError("A non-fault game cannot identify a fault_model.")
            if self.outcome.termination == "normal" and not self.normal_termination:
                raise ValueError("Normal games must record a chess termination.")
            if self.outcome.termination != "normal" and self.normal_termination is not None:
                raise ValueError("Only normal games can record a chess termination.")
        else:
            if self.fault_kind not in FAULT_KINDS:
                raise ValueError(f"Unsupported arena fault kind: {self.fault_kind!r}")
            if self.normal_termination is not None:
                raise ValueError("Faulted games cannot record a normal termination.")
            if self.fault_model != self.outcome.loser_model:
                raise ValueError("The acting model at fault must lose the game.")

    def as_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self.outcome),
            "final_fen": self.final_fen,
            "moves_uci": list(self.moves_uci),
            "normal_termination": self.normal_termination,
            "fault_kind": self.fault_kind,
            "fault_model": self.fault_model,
        }


@dataclasses.dataclass(frozen=True)
class ModelGameplayStats:
    model_id: str
    action_codec_id: str
    games: int
    wins: int
    draws: int
    losses: int
    points: float
    fault_losses: int
    policy_calls: int
    positions_evaluated: int

    def __post_init__(self) -> None:
        if not self.model_id or self.action_codec_id not in SUPPORTED_ACTION_CODECS:
            raise ValueError("Invalid model gameplay identity.")
        if (
            min(
                self.games,
                self.wins,
                self.draws,
                self.losses,
                self.fault_losses,
                self.policy_calls,
                self.positions_evaluated,
            )
            < 0
        ):
            raise ValueError("Model gameplay counters must be non-negative.")
        if self.wins + self.draws + self.losses != self.games:
            raise ValueError("Model W/D/L counters do not sum to games.")
        if not math.isclose(
            self.points,
            self.wins + 0.5 * self.draws,
            abs_tol=1e-12,
        ):
            raise ValueError("Model points do not match W/D/L counters.")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclasses.dataclass(frozen=True)
class ArenaGameplayResult:
    """Deterministic completed arena payload; wall-clock timing is excluded."""

    additional_ply_cap: int
    policy_timeout_seconds: float
    opening_histories: tuple[ArenaOpeningHistory, ...]
    records: tuple[ArenaGameRecord, ...]
    model_stats: tuple[ModelGameplayStats, ...]
    total_policy_calls: int
    max_policy_batch_size: int

    def __post_init__(self) -> None:
        if self.additional_ply_cap < 1:
            raise ValueError("additional_ply_cap must be positive.")
        if not math.isfinite(self.policy_timeout_seconds) or self.policy_timeout_seconds <= 0.0:
            raise ValueError("policy_timeout_seconds must be positive and finite.")
        if len(self.records) == 0 or len(self.records) % 2:
            raise ValueError("A gameplay result must contain complete pairs.")
        if self.total_policy_calls < 0 or self.max_policy_batch_size < 0:
            raise ValueError("Policy-call counters must be non-negative.")
        history_pair_ids = [history.pair_id for history in self.opening_histories]
        if len(history_pair_ids) != len(set(history_pair_ids)):
            raise ValueError("Gameplay opening histories contain duplicate pair IDs.")
        record_pair_ids: list[str] = []
        for offset in range(0, len(self.records), 2):
            first, second = self.records[offset : offset + 2]
            if first.outcome.pair_id != second.outcome.pair_id or (
                first.outcome.game_in_pair,
                second.outcome.game_in_pair,
            ) != (0, 1):
                raise ValueError("Gameplay records must contain ordered complete pairs.")
            record_pair_ids.append(first.outcome.pair_id)
        if history_pair_ids != record_pair_ids:
            raise ValueError("Gameplay opening histories must align exactly with record pairs.")
        if self.total_policy_calls != sum(stats.policy_calls for stats in self.model_stats):
            raise ValueError("Gameplay total_policy_calls does not match model stats.")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": GAMEPLAY_SCHEMA,
            "config": {
                "additional_ply_cap": self.additional_ply_cap,
                "policy_timeout_seconds": self.policy_timeout_seconds,
                "terminal_precedes_exact_cap": True,
                "fault_policy": "acting_model_loses_no_fallback",
                "opening_history": "full_standard_game_fen_sequence_required",
            },
            "opening_histories": [history.as_dict() for history in self.opening_histories],
            "games": [record.as_dict() for record in self.records],
            "stats": {
                "game_count": len(self.records),
                "pair_count": len(self.records) // 2,
                "fault_counts": dict(
                    sorted(
                        Counter(
                            record.fault_kind
                            for record in self.records
                            if record.fault_kind is not None
                        ).items()
                    )
                ),
                "termination_counts": dict(
                    sorted(Counter(record.outcome.termination for record in self.records).items())
                ),
                "total_policy_calls": self.total_policy_calls,
                "max_policy_batch_size": self.max_policy_batch_size,
                "models": [dataclasses.asdict(stats) for stats in self.model_stats],
            },
        }
        payload["payload_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        return payload

    @property
    def payload_sha256(self) -> str:
        return str(self.as_dict()["payload_sha256"])


def _validate_pair_specs(
    pairs: Sequence[tuple[ArenaGameSpec, ArenaGameSpec]],
) -> tuple[tuple[ArenaGameSpec, ArenaGameSpec], ...]:
    normalized = tuple(tuple(pair) for pair in pairs)
    if not normalized:
        raise ValueError("At least one color-reversed pair is required.")
    seen_pair_ids: set[str] = set()
    seen_game_ids: set[tuple[str, int]] = set()
    for pair_index, pair in enumerate(normalized):
        if len(pair) != 2:
            raise ValueError(f"Arena pair {pair_index} must contain exactly two games.")
        first, second = pair
        if not isinstance(first, ArenaGameSpec) or not isinstance(second, ArenaGameSpec):
            raise TypeError("Arena pairs must contain ArenaGameSpec values.")
        if (first.game_in_pair, second.game_in_pair) != (0, 1):
            raise ValueError("Arena pairs must be ordered as game_in_pair 0 then 1.")
        if first.pair_id != second.pair_id or first.fen != second.fen:
            raise ValueError("Paired game specs must share pair_id and opening FEN.")
        if first.opening_index != second.opening_index:
            raise ValueError("Paired game specs must share opening_index.")
        if not (
            first.white_model == second.black_model and first.black_model == second.white_model
        ):
            raise ValueError("Arena game specs must be exact color reversals.")
        if first.pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate arena pair_id: {first.pair_id}")
        seen_pair_ids.add(first.pair_id)
        for spec in pair:
            game_id = (spec.pair_id, spec.game_in_pair)
            if game_id in seen_game_ids:
                raise ValueError(f"Duplicate arena game identity: {game_id}")
            seen_game_ids.add(game_id)
    return normalized


def _replay_opening_history(
    spec: ArenaGameSpec,
    opening_history: ArenaOpeningHistory,
) -> tuple[chess.Board, list[chess.Board]]:
    if opening_history.pair_id != spec.pair_id:
        raise ValueError(
            f"Opening history {opening_history.pair_id!r} does not match pair {spec.pair_id!r}."
        )

    parsed = [
        _parse_canonical_history_fen(fen, history_index=index)
        for index, fen in enumerate(opening_history.fens)
    ]
    initial = chess.Board()
    initial_fen = _canonical_position_fen(initial)
    if opening_history.fens[0] != initial_fen:
        raise ValueError(
            "Arena opening history must begin at the standard initial position; "
            "a root FEN alone cannot reconstruct exact repetition state."
        )

    board = initial
    replayed = [board.copy(stack=False)]
    for history_index, target in enumerate(parsed[1:], start=1):
        target_fen = _canonical_position_fen(target)
        matches: list[chess.Move] = []
        for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci()):
            candidate = board.copy(stack=True)
            candidate.push(move)
            if _canonical_position_fen(candidate) == target_fen:
                matches.append(move)
        if len(matches) != 1:
            raise ValueError(
                f"Opening history transition {history_index - 1}->{history_index} "
                f"has {len(matches)} legal reconstructions; expected exactly one."
            )
        board.push(matches[0])
        if _canonical_position_fen(board) != target_fen:
            raise RuntimeError("Validated opening-history replay diverged.")
        if board.is_game_over(claim_draw=True) and history_index != len(parsed) - 1:
            raise ValueError("Arena opening history continues after a terminal position.")
        replayed.append(board.copy(stack=False))

    if _canonical_position_fen(board) != spec.fen:
        raise ValueError(
            f"Opening history for {spec.pair_id!r} ends at "
            f"{_canonical_position_fen(board)!r}, not {spec.fen!r}."
        )
    if len(replayed) != board.ply() + 1:
        raise ValueError("Arena opening history is truncated or has inconsistent move counters.")
    if board.is_game_over(claim_draw=True):
        raise ValueError("Arena opening history ends at a terminal position.")
    return board, replayed


def _validate_opening_histories(
    pairs: Sequence[tuple[ArenaGameSpec, ArenaGameSpec]],
    opening_histories: Mapping[str, ArenaOpeningHistory],
) -> dict[str, tuple[chess.Board, list[chess.Board]]]:
    pair_ids = {pair[0].pair_id for pair in pairs}
    if set(opening_histories) != pair_ids:
        missing = sorted(pair_ids - set(opening_histories))
        extra = sorted(set(opening_histories) - pair_ids)
        raise ValueError(
            f"Opening histories must exactly match pair IDs; missing={missing}, extra={extra}."
        )
    validated: dict[str, tuple[chess.Board, list[chess.Board]]] = {}
    for first, _second in pairs:
        history = opening_histories[first.pair_id]
        if not isinstance(history, ArenaOpeningHistory):
            raise TypeError("opening_histories values must be ArenaOpeningHistory.")
        validated[first.pair_id] = _replay_opening_history(first, history)
    return validated


def _validate_policies(
    pairs: Sequence[tuple[ArenaGameSpec, ArenaGameSpec]],
    policies: Mapping[str, BatchedArenaPolicy],
) -> dict[str, BatchedArenaPolicy]:
    required_model_ids = {
        model_id
        for pair in pairs
        for spec in pair
        for model_id in (spec.white_model, spec.black_model)
    }
    if set(policies) != required_model_ids:
        missing = sorted(required_model_ids - set(policies))
        extra = sorted(set(policies) - required_model_ids)
        raise ValueError(
            f"Policies must exactly match arena models; missing={missing}, extra={extra}."
        )
    validated: dict[str, BatchedArenaPolicy] = {}
    for model_id in sorted(required_model_ids):
        policy = policies[model_id]
        declared_model_id = getattr(policy, "model_id", None)
        codec_id = getattr(policy, "action_codec_id", None)
        if declared_model_id != model_id:
            raise ValueError(
                f"Policy mapping key {model_id!r} does not match declared "
                f"model_id {declared_model_id!r}."
            )
        if codec_id not in SUPPORTED_ACTION_CODECS:
            raise ValueError(
                f"Policy {model_id!r} must declare a supported action_codec_id; got {codec_id!r}."
            )
        if not callable(getattr(policy, "select_actions", None)):
            raise TypeError(f"Policy {model_id!r} has no callable select_actions.")
        validated[model_id] = policy
    return validated


def _failure_termination(fault_kind: str) -> str:
    if fault_kind == FAULT_TIMEOUT:
        return "timeout"
    if fault_kind in {FAULT_EXCEPTION, FAULT_NONFINITE_OUTPUT}:
        return "exception"
    return "illegal_move"


def _fault_record(
    state: _GameState,
    *,
    fault_kind: str,
    fault_model: str,
    ply_cap: int,
) -> ArenaGameRecord:
    outcome = classify_game_outcome(
        state.spec,
        termination=_failure_termination(fault_kind),
        ply_count=state.additional_plies,
        ply_cap=ply_cap,
        fault_model=fault_model,
    )
    return ArenaGameRecord(
        outcome=outcome,
        final_fen=_canonical_position_fen(state.board),
        moves_uci=tuple(state.moves_uci),
        fault_kind=fault_kind,
        fault_model=fault_model,
    )


def _normal_or_cap_record(
    state: _GameState,
    *,
    ply_cap: int,
) -> ArenaGameRecord | None:
    # Normal chess termination intentionally wins at the exact cap boundary.
    board_outcome = state.board.outcome(claim_draw=True)
    if board_outcome is not None:
        outcome = classify_game_outcome(
            state.spec,
            termination="normal",
            reported_result=board_outcome.result(),
            ply_count=state.additional_plies,
            ply_cap=ply_cap,
        )
        return ArenaGameRecord(
            outcome=outcome,
            final_fen=_canonical_position_fen(state.board),
            moves_uci=tuple(state.moves_uci),
            normal_termination=board_outcome.termination.name.lower(),
        )
    if state.additional_plies == ply_cap:
        outcome = classify_game_outcome(
            state.spec,
            termination="ply_cap",
            ply_count=state.additional_plies,
            ply_cap=ply_cap,
        )
        return ArenaGameRecord(
            outcome=outcome,
            final_fen=_canonical_position_fen(state.board),
            moves_uci=tuple(state.moves_uci),
        )
    if state.additional_plies > ply_cap:
        raise RuntimeError("Arena game advanced beyond its additional-ply cap.")
    return None


def _acting_model(state: _GameState) -> str:
    return state.spec.white_model if state.board.turn == chess.WHITE else state.spec.black_model


def _policy_inputs(
    states: Sequence[_GameState],
) -> tuple[list[chess.Board], list[tuple[chess.Board, ...]]]:
    boards = [state.board.copy(stack=True) for state in states]
    histories = [
        tuple(position.copy(stack=False) for position in state.history) for state in states
    ]
    return boards, histories


def _materialize_action_indices(
    selection: BatchedActionSelection,
    *,
    batch_size: int,
) -> np.ndarray:
    action_indices = np.asarray(selection.action_indices)
    if action_indices.shape != (batch_size,):
        raise ValueError(
            f"Policy action_indices must have shape {(batch_size,)}, got {action_indices.shape}."
        )
    return action_indices


def _selected_index_or_fault(value: Any) -> tuple[int | None, str | None]:
    array_value = np.asarray(value)
    if array_value.shape != ():
        return None, FAULT_ILLEGAL_ACTION
    if np.issubdtype(array_value.dtype, np.inexact):
        try:
            if not bool(np.isfinite(array_value)):
                return None, FAULT_NONFINITE_OUTPUT
        except TypeError:
            return None, FAULT_ILLEGAL_ACTION
        return None, FAULT_ILLEGAL_ACTION
    if np.issubdtype(array_value.dtype, np.bool_):
        return None, FAULT_ILLEGAL_ACTION
    try:
        index = operator.index(array_value.item())
    except (TypeError, ValueError, OverflowError):
        return None, FAULT_ILLEGAL_ACTION
    if index < 0 or index >= ACTION_VOCAB_SIZE:
        return None, FAULT_ILLEGAL_ACTION
    return int(index), None


def _decode_unique_legal_action(
    board: chess.Board,
    *,
    index: int,
    codec_id: str,
    legal_mask: np.ndarray,
) -> chess.Move:
    if legal_mask.shape != (ACTION_VOCAB_SIZE,) or legal_mask.dtype != np.bool_:
        raise RuntimeError("Internal arena legal-mask invariant failed.")
    if not bool(legal_mask[index]):
        raise ActionCodecError(f"Selected action index {index} is not legal.")

    matches: list[chess.Move] = []
    for move in board.legal_moves:
        try:
            encoded = encode_action(move, codec_id=codec_id, board=board)
        except (ActionCodecError, KeyError, IndexError, ValueError):
            continue
        if encoded == index:
            matches.append(move)
    if len(matches) != 1:
        raise ActionCodecError(f"Selected action index {index} has {len(matches)} legal decodes.")
    decoded = decode_action(index, codec_id=codec_id, board=board)
    if decoded != matches[0] or not board.is_legal(decoded):
        raise ActionCodecError(
            f"Selected action index {index} did not decode to its unique legal move."
        )
    return decoded


def _model_gameplay_stats(
    records: Sequence[ArenaGameRecord],
    *,
    policies: Mapping[str, BatchedArenaPolicy],
    policy_calls: Mapping[str, int],
    positions_evaluated: Mapping[str, int],
) -> tuple[ModelGameplayStats, ...]:
    stats: list[ModelGameplayStats] = []
    for model_id in sorted(policies):
        model_records = [
            record
            for record in records
            if model_id
            in {
                record.outcome.white_model,
                record.outcome.black_model,
            }
        ]
        scores = [record.outcome.score_for(model_id) for record in model_records]
        wins = sum(score == 1.0 for score in scores)
        draws = sum(score == 0.5 for score in scores)
        losses = sum(score == 0.0 for score in scores)
        stats.append(
            ModelGameplayStats(
                model_id=model_id,
                action_codec_id=str(policies[model_id].action_codec_id),
                games=len(scores),
                wins=wins,
                draws=draws,
                losses=losses,
                points=float(sum(scores)),
                fault_losses=sum(record.fault_model == model_id for record in model_records),
                policy_calls=int(policy_calls.get(model_id, 0)),
                positions_evaluated=int(positions_evaluated.get(model_id, 0)),
            )
        )
    return tuple(stats)


def play_arena_pairs(
    pairs: Sequence[tuple[ArenaGameSpec, ArenaGameSpec]],
    *,
    opening_histories: Mapping[str, ArenaOpeningHistory],
    policies: Mapping[str, BatchedArenaPolicy],
    additional_ply_cap: int,
    policy_timeout_seconds: float = 30.0,
    _clock: Callable[[], float] = time.monotonic,
) -> ArenaGameplayResult:
    """Play complete color-reversed pairs in deterministic in-process batches.

    Active positions are grouped by acting model ID once per arena round.  A
    failed batch call charges every position in that batch to the acting model;
    per-row invalid actions charge only their own game.  There is no random or
    legal-move fallback.

    The timeout covers adapter execution plus host materialization of returned
    action indices.  The runner detects an overrun after control returns; an
    adapter that needs preemptive cancellation must enforce it internally and
    raise :class:`TimeoutError`.
    """
    if isinstance(additional_ply_cap, bool) or not isinstance(additional_ply_cap, int):
        raise TypeError("additional_ply_cap must be an integer.")
    if additional_ply_cap < 1:
        raise ValueError("additional_ply_cap must be positive.")
    try:
        policy_timeout_seconds = float(policy_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise TypeError("policy_timeout_seconds must be numeric.") from exc
    if not math.isfinite(policy_timeout_seconds) or policy_timeout_seconds <= 0.0:
        raise ValueError("policy_timeout_seconds must be positive and finite.")

    validated_pairs = _validate_pair_specs(pairs)
    validated_histories = _validate_opening_histories(
        validated_pairs,
        opening_histories,
    )
    validated_policies = _validate_policies(validated_pairs, policies)

    states: list[_GameState] = []
    for pair in validated_pairs:
        board, history = validated_histories[pair[0].pair_id]
        for spec in pair:
            states.append(
                _GameState(
                    spec=spec,
                    board=board.copy(stack=True),
                    history=[position.copy(stack=False) for position in history],
                    moves_uci=[],
                )
            )

    completed: dict[tuple[str, int], ArenaGameRecord] = {}
    policy_calls: Counter[str] = Counter()
    positions_evaluated: Counter[str] = Counter()
    max_policy_batch_size = 0

    while len(completed) < len(states):
        grouped: dict[
            str,
            list[tuple[_GameState, np.ndarray]],
        ] = defaultdict(list)
        progress = False

        for state in states:
            game_key = (state.spec.pair_id, state.spec.game_in_pair)
            if game_key in completed:
                continue
            boundary_record = _normal_or_cap_record(
                state,
                ply_cap=additional_ply_cap,
            )
            if boundary_record is not None:
                completed[game_key] = boundary_record
                progress = True
                continue

            model_id = _acting_model(state)
            codec_id = str(validated_policies[model_id].action_codec_id)
            try:
                mask = np.asarray(
                    legal_action_mask(
                        state.board,
                        codec_id=codec_id,
                    )
                )
            except (ActionCodecError, KeyError, IndexError, ValueError):
                completed[game_key] = _fault_record(
                    state,
                    fault_kind=FAULT_NO_REPRESENTABLE_MOVE,
                    fault_model=model_id,
                    ply_cap=additional_ply_cap,
                )
                progress = True
                continue
            except Exception:
                completed[game_key] = _fault_record(
                    state,
                    fault_kind=FAULT_EXCEPTION,
                    fault_model=model_id,
                    ply_cap=additional_ply_cap,
                )
                progress = True
                continue
            if mask.shape != (ACTION_VOCAB_SIZE,) or mask.dtype != np.bool_:
                completed[game_key] = _fault_record(
                    state,
                    fault_kind=FAULT_EXCEPTION,
                    fault_model=model_id,
                    ply_cap=additional_ply_cap,
                )
                progress = True
                continue
            if not bool(np.any(mask)):
                completed[game_key] = _fault_record(
                    state,
                    fault_kind=FAULT_NO_REPRESENTABLE_MOVE,
                    fault_model=model_id,
                    ply_cap=additional_ply_cap,
                )
                progress = True
                continue
            grouped[model_id].append((state, mask))

        for model_id in sorted(grouped):
            pending = grouped[model_id]
            batch_states = [state for state, _mask in pending]
            batch_size = len(batch_states)
            max_policy_batch_size = max(max_policy_batch_size, batch_size)
            policy_calls[model_id] += 1
            positions_evaluated[model_id] += batch_size
            boards, histories = _policy_inputs(batch_states)

            started = _clock()
            call_error: Exception | None = None
            actions: np.ndarray | None = None
            try:
                selection = validated_policies[model_id].select_actions(
                    boards,
                    histories,
                )
                actions = _materialize_action_indices(
                    selection,
                    batch_size=batch_size,
                )
            except Exception as exc:  # Model/adaptor faults are match results.
                call_error = exc
            finished = _clock()
            if not (math.isfinite(started) and math.isfinite(finished) and finished >= started):
                raise RuntimeError("Arena monotonic clock returned invalid values.")

            if isinstance(call_error, TimeoutError) or finished - started > policy_timeout_seconds:
                batch_fault = FAULT_TIMEOUT
            elif call_error is not None:
                batch_fault = FAULT_EXCEPTION
            else:
                batch_fault = None

            if batch_fault is not None:
                for state in batch_states:
                    game_key = (state.spec.pair_id, state.spec.game_in_pair)
                    completed[game_key] = _fault_record(
                        state,
                        fault_kind=batch_fault,
                        fault_model=model_id,
                        ply_cap=additional_ply_cap,
                    )
                progress = True
                continue

            if actions is None:
                raise RuntimeError("Successful policy call did not return actions.")
            codec_id = str(validated_policies[model_id].action_codec_id)
            for row, (state, mask) in enumerate(pending):
                game_key = (state.spec.pair_id, state.spec.game_in_pair)
                selected_index, row_fault = _selected_index_or_fault(actions[row])
                if row_fault is not None:
                    completed[game_key] = _fault_record(
                        state,
                        fault_kind=row_fault,
                        fault_model=model_id,
                        ply_cap=additional_ply_cap,
                    )
                    progress = True
                    continue
                if selected_index is None:
                    raise RuntimeError("Validated action index is unexpectedly missing.")
                try:
                    move = _decode_unique_legal_action(
                        state.board,
                        index=selected_index,
                        codec_id=codec_id,
                        legal_mask=mask,
                    )
                except (ActionCodecError, KeyError, IndexError, ValueError):
                    completed[game_key] = _fault_record(
                        state,
                        fault_kind=FAULT_ILLEGAL_ACTION,
                        fault_model=model_id,
                        ply_cap=additional_ply_cap,
                    )
                    progress = True
                    continue
                except Exception:
                    completed[game_key] = _fault_record(
                        state,
                        fault_kind=FAULT_EXCEPTION,
                        fault_model=model_id,
                        ply_cap=additional_ply_cap,
                    )
                    progress = True
                    continue

                state.board.push(move)
                state.history.append(state.board.copy(stack=False))
                state.moves_uci.append(move.uci())
                state.additional_plies += 1
                progress = True
                boundary_record = _normal_or_cap_record(
                    state,
                    ply_cap=additional_ply_cap,
                )
                if boundary_record is not None:
                    completed[game_key] = boundary_record

        if not progress:
            raise RuntimeError("Arena gameplay made no progress.")

    ordered_records = tuple(
        completed[(state.spec.pair_id, state.spec.game_in_pair)] for state in states
    )
    model_stats = _model_gameplay_stats(
        ordered_records,
        policies=validated_policies,
        policy_calls=policy_calls,
        positions_evaluated=positions_evaluated,
    )
    return ArenaGameplayResult(
        additional_ply_cap=additional_ply_cap,
        policy_timeout_seconds=policy_timeout_seconds,
        opening_histories=tuple(opening_histories[pair[0].pair_id] for pair in validated_pairs),
        records=ordered_records,
        model_stats=model_stats,
        total_policy_calls=sum(policy_calls.values()),
        max_policy_batch_size=max_policy_batch_size,
    )


__all__ = [
    "FAULT_EXCEPTION",
    "FAULT_ILLEGAL_ACTION",
    "FAULT_KINDS",
    "FAULT_NONFINITE_OUTPUT",
    "FAULT_NO_REPRESENTABLE_MOVE",
    "FAULT_TIMEOUT",
    "GAMEPLAY_SCHEMA",
    "ArenaGameRecord",
    "ArenaGameplayResult",
    "ArenaOpeningHistory",
    "BatchedActionSelection",
    "BatchedArenaPolicy",
    "ModelGameplayStats",
    "play_arena_pairs",
]
