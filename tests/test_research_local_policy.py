from __future__ import annotations

import dataclasses
from collections.abc import Callable

import chess
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import research.local_policy as local_policy
from research.arena_history_trust import (
    HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    TrustedArenaHistoryEndpoint,
    _mint_trusted_arena_history_endpoint,
)
from chess_dfm_jax.encoding import encode_board
from chess_dfm_jax.policy import (
    ACTION_CODEC_LEGACY_ABSOLUTE_1858,
    ACTION_VOCAB_SIZE,
    move_to_policy_index,
)
from research.inference import DFMInferenceResult, DFMRefinementTrace
from research.local_policy import (
    PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED,
    LocalDFMPolicy,
    LocalPolicyError,
    select_local_dfm_actions,
)


@dataclasses.dataclass
class _Config:
    horizon: int = 4
    action_vocab_size: int = ACTION_VOCAB_SIZE


class _Model:
    def __init__(self) -> None:
        self.config = _Config()


class _CompiledModel(nnx.Module):
    def __init__(self) -> None:
        self.config = _Config()

    def encode_bt4_tokens(self, planes):
        return jnp.zeros((planes.shape[0], 64, 3), dtype=jnp.float32)

    def dfm_latents(self, bt4_tokens):
        return jnp.zeros(
            (bt4_tokens.shape[0], bt4_tokens.shape[1], 2),
            dtype=jnp.float32,
        )

    def planner_from_latents(self, z_dfm, action_tokens, t):
        preferred = move_to_policy_index("e2e4", "lc0_1858")
        logits = jnp.full(
            (
                z_dfm.shape[0],
                self.config.horizon,
                self.config.action_vocab_size,
            ),
            -8.0,
            dtype=jnp.float32,
        )
        return logits.at[:, :, preferred].set(8.0)


def _push_history(*moves: str) -> tuple[chess.Board, ...]:
    board = chess.Board()
    history = [board.copy(stack=False)]
    for move in moves:
        board.push_uci(move)
        history.append(board.copy(stack=False))
    return tuple(history)


def _trusted_endpoint(
    history: tuple[chess.Board, ...],
    *,
    root_ply: int | None = None,
    additional_plies: int = 0,
    current_fen: str | None = None,
    ply_cap: int = 64,
) -> TrustedArenaHistoryEndpoint:
    current = history[-1]
    resolved_root_ply = current.ply() if root_ply is None else root_ply
    return _mint_trusted_arena_history_endpoint(
        pair_id="fixture-pair",
        opening_index=0,
        game_in_pair=0,
        current_fen=(current.fen(en_passant="legal") if current_fen is None else current_fen),
        position_count=current.ply() + 1,
        authoritative_move_stack_length=current.ply(),
        root_ply=resolved_root_ply,
        additional_plies=additional_plies,
        played_suffix_count=additional_plies,
        ply_cap=ply_cap,
    )


def _valid_result(
    root_legal_mask: np.ndarray,
    *,
    horizon: int,
    refinement_passes: int,
    trace_top_k: int,
) -> DFMInferenceResult:
    batch_size = root_legal_mask.shape[0]
    legal_indices = [
        np.flatnonzero(root_legal_mask[row])[:trace_top_k] for row in range(batch_size)
    ]
    selected = np.asarray([indices[0] for indices in legal_indices], dtype=np.int32)
    actions = np.broadcast_to(selected[:, None], (batch_size, horizon)).copy()

    actions_before = np.empty(
        (refinement_passes, batch_size, horizon),
        dtype=np.int32,
    )
    actions_after = np.broadcast_to(
        actions[None, :, :],
        (refinement_passes, batch_size, horizon),
    ).copy()
    actions_before[0] = ACTION_VOCAB_SIZE
    actions_before[1:] = actions_after[:-1]

    topk_indices = np.zeros(
        (refinement_passes, batch_size, trace_top_k),
        dtype=np.int32,
    )
    topk_probabilities = np.zeros_like(topk_indices, dtype=np.float32)
    topk_valid = np.zeros_like(topk_indices, dtype=np.bool_)
    for row, indices in enumerate(legal_indices):
        count = len(indices)
        topk_indices[:, row, :count] = indices
        topk_probabilities[:, row, :count] = 1.0 / count
        topk_valid[:, row, :count] = True

    pass_batch_shape = (refinement_passes, batch_size)
    return DFMInferenceResult(
        actions=jnp.asarray(actions),
        trace=DFMRefinementTrace(
            times=jnp.arange(refinement_passes, dtype=jnp.float32) / refinement_passes,
            actions_before=jnp.asarray(actions_before),
            actions_after=jnp.asarray(actions_after),
            root_raw_entropy=jnp.full(pass_batch_shape, 4.0, dtype=jnp.float32),
            root_raw_legal_mass=jnp.full(
                pass_batch_shape,
                0.25,
                dtype=jnp.float32,
            ),
            root_legal_entropy=jnp.full(
                pass_batch_shape,
                2.0,
                dtype=jnp.float32,
            ),
            root_legal_topk_indices=jnp.asarray(topk_indices),
            root_legal_topk_probabilities=jnp.asarray(topk_probabilities),
            root_legal_topk_valid=jnp.asarray(topk_valid),
        ),
    )


def _install_fake_inference(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[[DFMInferenceResult], DFMInferenceResult] | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_infer(
        model,
        current_planes,
        root_legal_mask,
        *,
        refinement_passes,
        trace_top_k,
        action_codec_id,
    ):
        calls.append(
            {
                "model": model,
                "current_planes": np.asarray(current_planes),
                "root_legal_mask": np.asarray(root_legal_mask),
                "refinement_passes": refinement_passes,
                "trace_top_k": trace_top_k,
                "action_codec_id": action_codec_id,
            }
        )
        result = _valid_result(
            np.asarray(root_legal_mask),
            horizon=model.config.horizon,
            refinement_passes=refinement_passes,
            trace_top_k=trace_top_k,
        )
        return result if mutate is None else mutate(result)

    monkeypatch.setattr(local_policy, "infer_dfm_from_current", fake_infer)
    return calls


def _install_fake_lean_inference(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_infer(
        model,
        current_planes,
        root_legal_mask,
        *,
        refinement_passes,
        action_codec_id,
    ):
        mask = np.asarray(root_legal_mask)
        calls.append(
            {
                "model": model,
                "current_planes": np.asarray(current_planes),
                "root_legal_mask": mask,
                "refinement_passes": refinement_passes,
                "action_codec_id": action_codec_id,
            }
        )
        selected = np.asarray(
            [np.flatnonzero(row)[0] for row in mask],
            dtype=np.int32,
        )
        actions = np.broadcast_to(
            selected[:, None],
            (mask.shape[0], model.config.horizon),
        ).copy()
        return jnp.asarray(actions if mutate is None else mutate(actions))

    monkeypatch.setattr(local_policy, "infer_dfm_actions_from_current", fake_infer)
    return calls


def _replace_trace(
    result: DFMInferenceResult,
    **updates,
) -> DFMInferenceResult:
    return DFMInferenceResult(
        actions=result.actions,
        trace=result.trace._replace(**updates),
    )


def test_adapter_validates_full_histories_but_encodes_current_only(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _Model()
    histories = (
        _push_history(),
        _push_history("e2e4"),
    )
    boards = tuple(history[-1] for history in histories)
    calls = _install_fake_inference(monkeypatch)

    encode_calls: list[tuple[str, tuple[chess.Board, ...], str, str]] = []
    real_encode_board = encode_board

    def recording_encode(board, history, *, planes_layout, input_format):
        encode_calls.append(
            (
                board.fen(en_passant="fen"),
                tuple(history),
                planes_layout,
                input_format,
            )
        )
        return real_encode_board(
            board,
            history,
            planes_layout=planes_layout,
            input_format=input_format,
        )

    monkeypatch.setattr(local_policy, "encode_board", recording_encode)
    adapter = LocalDFMPolicy(
        model=model,
        model_id="candidate-step-265000",
        refinement_passes=3,
        trace_top_k=2,
    )
    result = adapter.select_actions(boards, histories)

    assert adapter.action_codec_id == ACTION_CODEC_LEGACY_ABSOLUTE_1858
    assert adapter.plane_history_mode == PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED
    assert len(encode_calls) == 2
    assert all(history == () for _, history, _, _ in encode_calls)
    assert all(layout == "nchw" for _, _, layout, _ in encode_calls)
    assert all(
        input_format == "INPUT_CLASSICAL_112_PLANE" for _, _, _, input_format in encode_calls
    )

    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_array_equal(
        call["current_planes"],
        np.stack(
            [
                real_encode_board(board, [], input_format="INPUT_CLASSICAL_112_PLANE")
                for board in boards
            ]
        ),
    )
    assert call["current_planes"].shape == (2, 112, 8, 8)
    assert call["action_codec_id"] == ACTION_CODEC_LEGACY_ABSOLUTE_1858
    assert call["refinement_passes"] == 3
    assert call["trace_top_k"] == 2

    assert result.action_indices.shape == (2,)
    assert np.issubdtype(result.action_indices.dtype, np.integer)
    assert [move.uci() for move in result.moves] == [
        "b1a3",
        "a7a5",
    ]
    assert result.action_indices.flags.writeable is False
    assert result.diagnostics is not None
    assert result.diagnostics.encoded_planes_shape == (2, 112, 8, 8)
    assert result.diagnostics.plane_history_mode == PLANE_HISTORY_MODE_CURRENT_ONLY_AS_PREPROCESSED
    np.testing.assert_array_equal(result.diagnostics.legal_move_counts, [20, 20])
    np.testing.assert_array_equal(
        result.diagnostics.representable_legal_action_counts,
        [20, 20],
    )
    assert result.diagnostics.root_raw_entropy.flags.writeable is False


def test_lean_adapter_returns_checked_actions_without_trace_diagnostics():
    history = _push_history()
    adapter = LocalDFMPolicy(
        model=_CompiledModel(),
        model_id="lean-candidate",
        refinement_passes=4,
        trace_top_k=2,
        collect_diagnostics=False,
    )

    result = adapter.select_actions(
        [history[-1]],
        [history],
    )

    assert result.diagnostics is None
    assert result.action_indices.shape == (1,)
    assert result.moves == (chess.Move.from_uci("e2e4"),)
    assert result.action_indices.flags.writeable is False


@pytest.mark.parametrize("active_batch_size", [1, 3])
def test_frozen_lean_batch_matches_unpadded_actions_for_odd_and_singleton_tails(
    active_batch_size: int,
):
    histories = (
        _push_history(),
        _push_history("e2e4"),
        _push_history("d2d4", "d7d5"),
    )[:active_batch_size]
    boards = tuple(history[-1] for history in histories)
    dynamic = LocalDFMPolicy(
        model=_CompiledModel(),
        model_id="dynamic",
        refinement_passes=2,
        collect_diagnostics=False,
    ).select_actions(boards, histories)
    frozen = LocalDFMPolicy(
        model=_CompiledModel(),
        model_id="frozen",
        refinement_passes=2,
        collect_diagnostics=False,
        inference_batch_size=4,
    ).select_actions(boards, histories)

    np.testing.assert_array_equal(frozen.action_indices, dynamic.action_indices)
    assert frozen.moves == dynamic.moves
    assert frozen.action_indices.shape == (active_batch_size,)
    assert frozen.diagnostics is None


def test_frozen_lean_batch_uses_one_physical_shape_as_population_shrinks(
    monkeypatch: pytest.MonkeyPatch,
):
    histories = (
        _push_history(),
        _push_history("e2e4"),
        _push_history("d2d4", "d7d5"),
        _push_history("c2c4", "g8f6", "b1c3"),
    )
    calls = _install_fake_lean_inference(monkeypatch)
    adapter = LocalDFMPolicy(
        model=_Model(),
        model_id="static-shape",
        refinement_passes=3,
        collect_diagnostics=False,
        inference_batch_size=4,
    )

    for active_batch_size in (4, 3, 1):
        active_histories = histories[:active_batch_size]
        result = adapter.select_actions(
            [history[-1] for history in active_histories],
            active_histories,
        )
        assert result.action_indices.shape == (active_batch_size,)

    assert [call["current_planes"].shape[0] for call in calls] == [4, 4, 4]
    assert [call["root_legal_mask"].shape[0] for call in calls] == [4, 4, 4]
    for active_batch_size, call in zip((4, 3, 1), calls, strict=True):
        if active_batch_size == 4:
            continue
        np.testing.assert_array_equal(
            call["current_planes"][active_batch_size:],
            np.repeat(
                call["current_planes"][:1],
                4 - active_batch_size,
                axis=0,
            ),
        )
        np.testing.assert_array_equal(
            call["root_legal_mask"][active_batch_size:],
            np.repeat(
                call["root_legal_mask"][:1],
                4 - active_batch_size,
                axis=0,
            ),
        )


@pytest.mark.parametrize("inference_batch_size", [0, -1, True, 1.5])
def test_frozen_inference_batch_size_must_be_a_positive_integer(
    inference_batch_size,
):
    with pytest.raises(LocalPolicyError, match="inference_batch_size"):
        LocalDFMPolicy(
            model=_Model(),
            model_id="bad-static-shape",
            inference_batch_size=inference_batch_size,
        )


def test_active_rows_cannot_exceed_frozen_inference_batch_size(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = _install_fake_lean_inference(monkeypatch)
    histories = (_push_history(), _push_history("e2e4"))
    adapter = LocalDFMPolicy(
        model=_Model(),
        model_id="too-small-static-shape",
        collect_diagnostics=False,
        inference_batch_size=1,
    )
    with pytest.raises(LocalPolicyError, match="exceeds frozen"):
        adapter.select_actions(
            [history[-1] for history in histories],
            histories,
        )
    assert calls == []


def test_frozen_lean_padding_outputs_are_inert_but_real_outputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    history = _push_history()

    def invalidate_padding(actions: np.ndarray) -> np.ndarray:
        actions[1:] = -1
        return actions

    _install_fake_lean_inference(monkeypatch, mutate=invalidate_padding)
    adapter = LocalDFMPolicy(
        model=_Model(),
        model_id="inert-padding",
        collect_diagnostics=False,
        inference_batch_size=4,
    )
    result = adapter.select_actions([history[-1]], [history])
    assert result.action_indices.shape == (1,)
    assert len(result.moves) == 1

    def invalidate_real_row(actions: np.ndarray) -> np.ndarray:
        actions[0] = -1
        return actions

    _install_fake_lean_inference(monkeypatch, mutate=invalidate_real_row)
    with pytest.raises(LocalPolicyError, match="out-of-range"):
        adapter.select_actions([history[-1]], [history])


def test_frozen_diagnostic_batch_slices_every_trace_to_real_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    histories = (_push_history(), _push_history("e2e4"), _push_history("d2d4"))

    def poison_padding(result: DFMInferenceResult) -> DFMInferenceResult:
        return _replace_trace(
            result,
            root_raw_entropy=result.trace.root_raw_entropy.at[:, 3].set(jnp.nan),
        )

    calls = _install_fake_inference(monkeypatch, mutate=poison_padding)
    result = LocalDFMPolicy(
        model=_Model(),
        model_id="diagnostic-static-shape",
        refinement_passes=3,
        trace_top_k=2,
        collect_diagnostics=True,
        inference_batch_size=4,
    ).select_actions(
        [history[-1] for history in histories],
        histories,
    )

    assert calls[0]["current_planes"].shape == (4, 112, 8, 8)
    assert result.action_indices.shape == (3,)
    assert result.diagnostics is not None
    assert result.diagnostics.encoded_planes_shape == (3, 112, 8, 8)
    assert result.diagnostics.actions_before.shape == (3, 3, 4)
    assert result.diagnostics.actions_after.shape == (3, 3, 4)
    assert result.diagnostics.root_raw_entropy.shape == (3, 3)
    assert result.diagnostics.root_legal_topk_indices.shape == (3, 3, 2)


def test_adapter_runs_through_real_compiled_inference_boundary():
    result = select_local_dfm_actions(
        _CompiledModel(),
        (chess.Board(),),
        (_push_history(),),
        refinement_passes=2,
        trace_top_k=2,
    )

    assert result.moves == (chess.Move.from_uci("e2e4"),)
    np.testing.assert_array_equal(
        result.action_indices,
        [move_to_policy_index("e2e4", "lc0_1858")],
    )
    assert result.diagnostics.actions_after.shape == (2, 1, 4)
    assert np.all(np.isfinite(result.diagnostics.root_raw_entropy))


def test_trusted_endpoint_matches_full_replay_with_static_padding(
    monkeypatch: pytest.MonkeyPatch,
):
    history = _push_history("e2e4", "e7e5", "g1f3")
    board = history[-1].copy(stack=False)
    calls = _install_fake_inference(monkeypatch)
    full_policy = LocalDFMPolicy(
        model=_Model(),
        model_id="full",
        refinement_passes=3,
        trace_top_k=2,
        inference_batch_size=4,
    )
    endpoint_policy = LocalDFMPolicy(
        model=_Model(),
        model_id="endpoint",
        refinement_passes=3,
        trace_top_k=2,
        inference_batch_size=4,
        history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    )

    full = full_policy.select_actions([board], [history])
    endpoint = endpoint_policy.select_actions_from_trusted_arena(
        [board],
        [_trusted_endpoint(history)],
    )

    np.testing.assert_array_equal(endpoint.action_indices, full.action_indices)
    assert endpoint.moves == full.moves
    assert endpoint.diagnostics is not None
    assert full.diagnostics is not None
    np.testing.assert_array_equal(
        endpoint.diagnostics.actions_after,
        full.diagnostics.actions_after,
    )
    assert [call["current_planes"].shape[0] for call in calls] == [4, 4]
    np.testing.assert_array_equal(
        calls[1]["current_planes"],
        calls[0]["current_planes"],
    )
    np.testing.assert_array_equal(
        calls[1]["root_legal_mask"],
        calls[0]["root_legal_mask"],
    )


def test_trusted_endpoint_rejects_mismatch_before_inference(
    monkeypatch: pytest.MonkeyPatch,
):
    history = _push_history("e2e4", "e7e5")
    board = history[-1].copy(stack=False)
    calls = _install_fake_lean_inference(monkeypatch)
    policy = LocalDFMPolicy(
        model=_Model(),
        model_id="endpoint",
        collect_diagnostics=False,
        inference_batch_size=4,
        history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    )
    corrupted = _trusted_endpoint(
        history,
        current_fen=chess.Board().fen(en_passant="legal"),
    )

    with pytest.raises(LocalPolicyError, match="must match"):
        policy.select_actions_from_trusted_arena([board], [corrupted])
    assert calls == []


def test_trusted_endpoint_hot_path_never_replays_transitions(
    monkeypatch: pytest.MonkeyPatch,
):
    histories = (
        _push_history("e2e4", "e7e5", "g1f3", "b8c6"),
        _push_history(
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1b5",
            "a7a6",
        ),
    )
    calls = _install_fake_lean_inference(monkeypatch)
    replay_calls = 0

    def reject_replay(*args, **kwargs):
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("trusted endpoint attempted transition replay")

    monkeypatch.setattr(local_policy, "_matching_legal_move", reject_replay)
    policy = LocalDFMPolicy(
        model=_Model(),
        model_id="endpoint",
        collect_diagnostics=False,
        inference_batch_size=4,
        history_validation_mode=HISTORY_VALIDATION_TRUSTED_ARENA_ENDPOINT,
    )
    root_ply = 2
    for history in histories:
        additional_plies = history[-1].ply() - root_ply
        result = policy.select_actions_from_trusted_arena(
            [history[-1].copy(stack=False)],
            [
                _trusted_endpoint(
                    history,
                    root_ply=root_ply,
                    additional_plies=additional_plies,
                )
            ],
        )
        assert result.action_indices.shape == (1,)

    assert replay_calls == 0
    assert [call["current_planes"].shape[0] for call in calls] == [4, 4]


def test_trusted_endpoint_is_sealed_and_default_policy_cannot_use_it():
    history = _push_history("e2e4")
    with pytest.raises(TypeError, match="only be minted"):
        TrustedArenaHistoryEndpoint(
            pair_id="forged",
            opening_index=0,
            game_in_pair=0,
            current_fen=history[-1].fen(en_passant="legal"),
            position_count=2,
            authoritative_move_stack_length=1,
            root_ply=1,
            additional_plies=0,
            played_suffix_count=0,
            ply_cap=4,
            claim_draw_checked_nonterminal=True,
            below_ply_cap=True,
            _seal=object(),
        )

    sealed = _trusted_endpoint(history)
    with pytest.raises(AttributeError, match="immutable"):
        sealed.current_fen = chess.Board().fen(en_passant="legal")

    default_policy = LocalDFMPolicy(
        model=_Model(),
        model_id="strict-default",
    )
    with pytest.raises(LocalPolicyError, match="disabled"):
        default_policy.select_actions_from_trusted_arena(
            [history[-1].copy(stack=False)],
            [sealed],
        )


def test_history_accepts_legal_fen_round_trip_that_drops_irrelevant_ep_square(
    monkeypatch: pytest.MonkeyPatch,
):
    raw_history = _push_history("e2e4")
    assert raw_history[-1].ep_square == chess.E3
    sidecar_history = tuple(chess.Board(board.fen(en_passant="legal")) for board in raw_history)
    assert sidecar_history[-1].ep_square is None
    current = chess.Board(raw_history[-1].fen(en_passant="legal"))
    calls = _install_fake_inference(monkeypatch)

    result = select_local_dfm_actions(
        _Model(),
        (current,),
        (sidecar_history,),
        refinement_passes=2,
        trace_top_k=2,
    )

    assert result.action_indices.shape == (1,)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("boards_factory", "histories_factory", "match"),
    (
        (
            lambda: (chess.Board(),),
            lambda: (),
            "must not be empty",
        ),
        (
            lambda: (chess.Board(), chess.Board()),
            lambda: (_push_history(),),
            "exactly one",
        ),
        (
            lambda: (chess.Board(),),
            lambda: ((),),
            "must not be empty",
        ),
        (
            lambda: (_push_history("e2e4")[-1],),
            lambda: ((_push_history("e2e4")[-1],),),
            "beginning at the standard initial",
        ),
        (
            lambda: (chess.Board(),),
            lambda: ((chess.Board(), chess.Board()),),
            "exactly one legal ply",
        ),
        (
            lambda: (chess.Board(),),
            lambda: (_push_history("e2e4"),),
            "must end at",
        ),
        (
            lambda: (_push_history("e2e4", "e7e5")[-1],),
            lambda: ((chess.Board(), _push_history("e2e4", "e7e5")[-1]),),
            "exactly one legal ply",
        ),
    ),
)
def test_adapter_fails_closed_on_missing_or_inconsistent_full_history(
    monkeypatch: pytest.MonkeyPatch,
    boards_factory,
    histories_factory,
    match: str,
):
    calls = _install_fake_inference(monkeypatch)
    with pytest.raises(LocalPolicyError, match=match):
        select_local_dfm_actions(
            _Model(),
            boards_factory(),
            histories_factory(),
            refinement_passes=2,
            trace_top_k=2,
        )
    assert calls == []


def test_adapter_rejects_terminal_board_before_inference(
    monkeypatch: pytest.MonkeyPatch,
):
    history = _push_history("f2f3", "e7e5", "g2g4", "d8h4")
    calls = _install_fake_inference(monkeypatch)
    with pytest.raises(LocalPolicyError, match="terminal"):
        select_local_dfm_actions(
            _Model(),
            (history[-1],),
            (history,),
            refinement_passes=2,
            trace_top_k=2,
        )
    assert calls == []


def test_adapter_fails_when_no_legacy_action_is_representable(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = _install_fake_inference(monkeypatch)
    monkeypatch.setattr(
        local_policy,
        "legal_action_mask",
        lambda board, *, codec_id: np.zeros(
            (ACTION_VOCAB_SIZE,),
            dtype=np.bool_,
        ),
    )
    with pytest.raises(LocalPolicyError, match="no representable legal"):
        select_local_dfm_actions(
            _Model(),
            (chess.Board(),),
            (_push_history(),),
            refinement_passes=2,
            trace_top_k=2,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda result: _replace_trace(
                result,
                root_raw_entropy=result.trace.root_raw_entropy.at[0, 0].set(jnp.nan),
            ),
            "nonfinite",
        ),
        (
            lambda result: _replace_trace(
                result,
                root_legal_topk_probabilities=(
                    result.trace.root_legal_topk_probabilities.at[0, 0, 0].set(jnp.inf)
                ),
            ),
            "nonfinite",
        ),
        (
            lambda result: _replace_trace(
                result,
                root_legal_entropy=result.trace.root_legal_entropy[:, :0],
            ),
            "must have shape",
        ),
    ),
)
def test_adapter_rejects_nonfinite_or_misshapen_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    match: str,
):
    calls = _install_fake_inference(monkeypatch, mutate=mutate)
    with pytest.raises(LocalPolicyError, match=match):
        select_local_dfm_actions(
            _Model(),
            (chess.Board(),),
            (_push_history(),),
            refinement_passes=2,
            trace_top_k=2,
        )
    assert len(calls) == 1


def test_adapter_rejects_illegal_or_undecodable_selection_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    illegal_index = move_to_policy_index("e7e5", "lc0_1858")

    def select_illegal(result: DFMInferenceResult) -> DFMInferenceResult:
        actions = result.actions.at[0, 0].set(illegal_index)
        actions_after = result.trace.actions_after.at[-1, 0, 0].set(illegal_index)
        return DFMInferenceResult(
            actions=actions,
            trace=result.trace._replace(actions_after=actions_after),
        )

    calls = _install_fake_inference(monkeypatch, mutate=select_illegal)
    with pytest.raises(LocalPolicyError, match="nonrepresentable or illegal"):
        select_local_dfm_actions(
            _Model(),
            (chess.Board(),),
            (_push_history(),),
            refinement_passes=2,
            trace_top_k=2,
        )
    assert len(calls) == 1

    calls = _install_fake_inference(monkeypatch)

    def fail_decode(index, policy_format):
        raise KeyError(index)

    monkeypatch.setattr(local_policy, "policy_index_to_move", fail_decode)
    with pytest.raises(LocalPolicyError, match="undecodable"):
        select_local_dfm_actions(
            _Model(),
            (chess.Board(),),
            (_push_history(),),
            refinement_passes=2,
            trace_top_k=2,
        )
    assert len(calls) == 1


def test_adapter_preserves_timeout_classification(
    monkeypatch: pytest.MonkeyPatch,
):
    def time_out(*args, **kwargs):
        raise TimeoutError("arena move deadline")

    monkeypatch.setattr(local_policy, "infer_dfm_from_current", time_out)
    with pytest.raises(TimeoutError, match="arena move deadline"):
        select_local_dfm_actions(
            _Model(),
            (chess.Board(),),
            (_push_history(),),
            refinement_passes=2,
            trace_top_k=2,
        )


def test_frozen_lean_adapter_preserves_timeout_without_padded_row_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    def time_out(*args, **kwargs):
        raise TimeoutError("static arena move deadline")

    monkeypatch.setattr(local_policy, "infer_dfm_actions_from_current", time_out)
    history = _push_history()
    adapter = LocalDFMPolicy(
        model=_Model(),
        model_id="static-timeout",
        collect_diagnostics=False,
        inference_batch_size=4,
    )
    with pytest.raises(TimeoutError, match="static arena move deadline"):
        adapter.select_actions(
            [history[-1]],
            [history],
        )


def test_adapter_rejects_bad_encoded_planes_and_does_not_call_inference(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = _install_fake_inference(monkeypatch)
    monkeypatch.setattr(
        local_policy,
        "encode_board",
        lambda *args, **kwargs: np.full(
            (112, 8, 8),
            np.nan,
            dtype=np.float32,
        ),
    )
    with pytest.raises(LocalPolicyError, match="nonfinite"):
        select_local_dfm_actions(
            _Model(),
            (chess.Board(),),
            (_push_history(),),
            refinement_passes=2,
            trace_top_k=2,
        )
    assert calls == []
