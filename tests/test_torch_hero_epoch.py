from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

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
    JepaFeedbackResult,
    JointModel,
    MuonAdamW,
    RawLinear,
    RawLayerNorm,
    StateProjector,
    TorchHeroArenaPolicy,
    _analyze_lr_range_records,
    _apply_hero_architecture_overrides,
    _apply_hero_dfm_jepa_conditioning_override,
    _apply_hero_feedback_override,
    _apply_hero_wdl_override,
    _apply_sigreg_sample_override,
    _canonicalize_trajectory_batch_reference,
    _closed_loop_proposal_actions,
    _hero_milestone_update,
    _latent_spectrum_metrics,
    _normalize_frozen_indices,
    _polarized_gradient_cosines,
    _prepare_wdl_supervision,
    _proposal_from_root_logits,
    _root_legal_mask_from_indices,
    _shared_sigreg_gradient_projection,
    _sigreg_v_stat,
    _torch_refine_dfm_actions,
    _validate_hero_architecture_config,
    build_parser,
    canonicalize_trajectory_batch,
    initialize_fresh_modules,
)


def test_eager_fused_backward_layernorm_preserves_forward_bits() -> None:
    generator = torch.Generator().manual_seed(91)
    eager = RawLayerNorm(64, dtype=torch.bfloat16)
    candidate = RawLayerNorm(
        64,
        dtype=torch.bfloat16,
        implementation="eager-fused-backward",
    )
    with torch.no_grad():
        eager.scale.normal_(generator=generator)
        eager.bias.normal_(generator=generator)
        candidate.load_state_dict(eager.state_dict())
    eager_input = torch.randn(
        (16, 64),
        generator=generator,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    candidate_input = eager_input.detach().clone().requires_grad_()
    eager_output = eager(eager_input, torch.bfloat16)
    candidate_output = candidate(candidate_input, torch.bfloat16)
    torch.testing.assert_close(candidate_output, eager_output, rtol=0.0, atol=0.0)

    cotangent = torch.randn(
        eager_output.shape,
        generator=generator,
        dtype=torch.bfloat16,
    )
    eager_output.backward(cotangent)
    candidate_output.backward(cotangent)
    torch.testing.assert_close(
        candidate_input.grad,
        eager_input.grad,
        rtol=0.0,
        atol=2 ** -22,
    )
    torch.testing.assert_close(
        candidate.scale.grad,
        eager.scale.grad,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate.bias.grad,
        eager.bias.grad,
        rtol=0.0,
        atol=0.0,
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


def _assert_fast_canonical_conversion_matches_reference(batch):
    converted = canonicalize_trajectory_batch(batch)
    reference = _canonicalize_trajectory_batch_reference(batch)
    assert converted.keys() == reference.keys()
    for key in converted:
        np.testing.assert_array_equal(converted[key], reference[key])
    return converted


def test_canonical_batch_conversion_covers_black_moves_and_knight_promotions():
    black = chess.Board()
    black.push_uci("e2e4")
    black_move = chess.Move.from_uci("e7e5")
    promotion = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    promotion_move = chess.Move.from_uci("a7a8n")
    converted = _assert_fast_canonical_conversion_matches_reference(
        _canonical_batch(
            [(black, black_move), (promotion, promotion_move)]
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
    converted = _assert_fast_canonical_conversion_matches_reference(
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

    converted = _assert_fast_canonical_conversion_matches_reference(
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


def test_canonical_batch_conversion_recovers_all_black_promotions():
    board = chess.Board("7k/8/8/8/8/8/p7/7K b - - 0 1")
    promotion = chess.Move.from_uci("a2a1n")
    assert board.is_legal(promotion)
    converted = _assert_fast_canonical_conversion_matches_reference(
        _canonical_batch([(board, promotion)])
    )

    expected = encode_lc0_canonical_1858(board, promotion)
    count = int(converted["legal_count"][0, 0])
    legal = converted["legal_idx"][0, 0, :count]
    assert expected in legal
    assert count == board.legal_moves.count()


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
    assert HERO_CONFIG.dfm_condition_on_current_jepa_state is False


def test_sigreg_sample_override_is_explicit_and_hero_only():
    candidate = _apply_sigreg_sample_override(
        HERO_CONFIG,
        recipe="hero",
        sigreg_example_count=256,
    )
    assert candidate.sigreg_example_count == 256
    assert HERO_CONFIG.sigreg_example_count == 64
    assert candidate.target_sigreg_coeff == candidate.pred_sigreg_coeff == 2.0

    with pytest.raises(ValueError, match="hero-only"):
        _apply_sigreg_sample_override(
            HERO_CONFIG,
            recipe="continuation",
            sigreg_example_count=256,
        )
    with pytest.raises(ValueError, match="positive integer"):
        _apply_sigreg_sample_override(
            HERO_CONFIG,
            recipe="hero",
            sigreg_example_count=0,
        )


def test_wdl_override_is_explicit_and_hero_only():
    unchanged = _apply_hero_wdl_override(
        HERO_CONFIG,
        recipe="hero",
        wdl_coeff=None,
    )
    candidate = _apply_hero_wdl_override(
        HERO_CONFIG,
        recipe="hero",
        wdl_coeff=0.0,
    )
    assert unchanged == HERO_CONFIG
    assert candidate.wdl_coeff == 0.0
    assert HERO_CONFIG.wdl_coeff == 0.25
    assert candidate.sigreg_example_count == 64
    assert candidate.target_sigreg_coeff == candidate.pred_sigreg_coeff == 2.0

    with pytest.raises(ValueError, match="hero-only"):
        _apply_hero_wdl_override(
            HERO_CONFIG,
            recipe="continuation",
            wdl_coeff=0.0,
        )
    for value in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            _apply_hero_wdl_override(
                HERO_CONFIG,
                recipe="hero",
                wdl_coeff=value,
            )


def test_feedback_override_is_explicit_default_off_and_hero_only():
    unchanged = _apply_hero_feedback_override(
        HERO_CONFIG,
        recipe="hero",
        jepa_feedback_mode=None,
    )
    candidate = _apply_hero_feedback_override(
        HERO_CONFIG,
        recipe="hero",
        jepa_feedback_mode="final_pass_adjoint",
    )
    assert unchanged == HERO_CONFIG
    assert HERO_CONFIG.jepa_feedback_mode == "none"
    assert candidate.jepa_feedback_mode == "final_pass_adjoint"
    assert dataclasses.replace(
        candidate,
        jepa_feedback_mode="none",
    ) == HERO_CONFIG

    with pytest.raises(ValueError, match="hero-only"):
        _apply_hero_feedback_override(
            HERO_CONFIG,
            recipe="continuation",
            jepa_feedback_mode="final_pass_adjoint",
        )
    with pytest.raises(ValueError, match="must be"):
        _apply_hero_feedback_override(
            HERO_CONFIG,
            recipe="hero",
            jepa_feedback_mode="unknown",
        )


def test_dfm_jepa_conditioning_override_is_default_off_and_hero_only():
    unchanged = _apply_hero_dfm_jepa_conditioning_override(
        HERO_CONFIG,
        recipe="hero",
        enabled=None,
    )
    candidate = _apply_hero_dfm_jepa_conditioning_override(
        HERO_CONFIG,
        recipe="hero",
        enabled=True,
    )

    assert unchanged == HERO_CONFIG
    assert candidate.dfm_condition_on_current_jepa_state is True
    assert dataclasses.replace(
        candidate,
        dfm_condition_on_current_jepa_state=False,
    ) == HERO_CONFIG
    with pytest.raises(ValueError, match="hero-only"):
        _apply_hero_dfm_jepa_conditioning_override(
            HERO_CONFIG,
            recipe="continuation",
            enabled=True,
        )
    with pytest.raises(ValueError, match="must be boolean"):
        _apply_hero_dfm_jepa_conditioning_override(
            HERO_CONFIG,
            recipe="hero",
            enabled=1,
        )


def test_dfm_jepa_conditioning_cli_is_explicit_for_train_and_evaluation():
    parser = build_parser()
    train_args = parser.parse_args(
        [
            "train",
            "--output-dir",
            "candidate",
            "--dfm-condition-on-current-jepa-state",
        ]
    )
    eval_args = parser.parse_args(
        [
            "hero-evaluate",
            "--pool",
            "fast",
            "--output-dir",
            "evaluation",
            "--dfm-condition-on-current-jepa-state",
        ]
    )

    assert train_args.dfm_condition_on_current_jepa_state is True
    assert eval_args.dfm_condition_on_current_jepa_state is True


def test_post_hero_architecture_overrides_are_explicit_and_validated():
    candidate = _apply_hero_architecture_overrides(
        HERO_CONFIG,
        recipe="hero",
        policy_passthrough_mode="all_horizon_heads",
        dfm_state_source="policy_prelogit",
        dfm_jepa_fusion_mode=None,
        wdl_include_current_state=True,
        dfm_closed_loop_mode=None,
    )
    assert candidate.policy_passthrough_mode == "all_horizon_heads"
    assert candidate.dfm_state_source == "policy_prelogit"
    assert candidate.wdl_include_current_state is True
    assert HERO_CONFIG.policy_passthrough_mode == "root_only"

    with pytest.raises(ValueError, match="recipe='hero'"):
        _apply_hero_architecture_overrides(
            HERO_CONFIG,
            recipe="continuation",
            policy_passthrough_mode="all_horizon_heads",
            dfm_state_source=None,
            dfm_jepa_fusion_mode=None,
            wdl_include_current_state=None,
            dfm_closed_loop_mode=None,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_hero_architecture_config(
            dataclasses.replace(
                HERO_CONFIG,
                jepa_feedback_mode="final_pass_adjoint",
                dfm_closed_loop_mode="predicted_jepa_tokens",
            )
        )


class _ConstantPolicyHead(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward_with_features(self, current, compute_dtype):
        batch_size = current.shape[0]
        logits = torch.full(
            (batch_size, 1858),
            self.value,
            dtype=compute_dtype,
        )
        return logits, current + 7.0

    def forward(self, current, compute_dtype):
        return torch.full(
            (current.shape[0], 1858),
            self.value,
            dtype=compute_dtype,
        )


def test_all_horizon_policy_passthrough_materializes_independent_slots():
    harness = SimpleNamespace(
        config=SimpleNamespace(
            horizon=3,
            policy_passthrough_mode="all_horizon_heads",
        ),
        encoder=SimpleNamespace(
            policy_head=_ConstantPolicyHead(1.0),
            future_policy_heads=torch.nn.ModuleList(
                (_ConstantPolicyHead(2.0), _ConstantPolicyHead(3.0))
            ),
        ),
    )
    current = torch.zeros((2, 64, 1024))
    logits, features = JointModel.policy_outputs(
        harness,
        current,
        torch.float32,
    )

    assert logits is not None
    assert logits.shape == (2, 3, 1858)
    torch.testing.assert_close(logits[:, :, 0], torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1))
    torch.testing.assert_close(features, current + 7.0)


class _EncodeSelectedEncoder:
    def __init__(self, current: torch.Tensor, future: torch.Tensor):
        self.current = current
        self.future = future

    def encode_current(self, planes, *, compute_dtype, remat):
        del planes, compute_dtype, remat
        return self.current

    def encode_future_tail(
        self,
        planes,
        *,
        compute_dtype,
        trainable_tail_layers,
    ):
        del planes, compute_dtype, trainable_tail_layers
        return self.future


class _ZeroStateProjector(torch.nn.Module):
    def forward(self, tokens, compute_dtype):
        del compute_dtype
        return torch.zeros((tokens.shape[0], 4))


class _CaptureDFMProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.source = None

    def forward(self, source, compute_dtype):
        del compute_dtype
        self.source = source
        return torch.zeros((source.shape[0], 64, 2))


def test_policy_prelogit_precursor_is_the_dfm_passthrough_source():
    current = torch.zeros((2, 64, 1024))
    future = torch.ones_like(current)
    policy_features = torch.full_like(current, 11.0)
    dfm_projector = _CaptureDFMProjector()
    harness = SimpleNamespace(
        config=SimpleNamespace(
            remat_bt4_blocks=False,
            remat_blocks=False,
            future_trainable_tail_layers=1,
            z_dim=4,
            dfm_state_source="policy_prelogit",
        ),
        encoder=_EncodeSelectedEncoder(current, future),
        state_projector=_ZeroStateProjector(),
        dfm_state_projector=dfm_projector,
        policy_outputs=lambda tokens, dtype: (
            torch.zeros((tokens.shape[0], 1858), dtype=dtype),
            policy_features,
        ),
        condition_dfm_latents=lambda z_dfm, z_jepa, dtype: SimpleNamespace(
            latents=z_dfm,
            residual_rms=torch.zeros(z_dfm.shape[0]),
            state_rms=torch.zeros(z_dfm.shape[0]),
            weight_rms=torch.tensor(0.0),
        ),
    )
    JointModel.encode_selected(
        harness,
        torch.zeros((2, 112, 8, 8)),
        torch.zeros((2, 112, 8, 8)),
        torch.float32,
    )
    assert dfm_projector.source is policy_features


class _DFMJepaConditioningHarness(torch.nn.Module):
    class _Config:
        token_dim = 2
        z_dim = 4

    def __init__(self, *, enabled: bool):
        super().__init__()
        self.config = self._Config()
        self.config.dfm_condition_on_current_jepa_state = enabled
        self.dfm_jepa_state_adapter = (
            RawLinear(4, 2, dtype=torch.float32, bias=False)
            if enabled
            else None
        )
        if self.dfm_jepa_state_adapter is not None:
            with torch.no_grad():
                self.dfm_jepa_state_adapter.w.zero_()


def test_current_jepa_conditioning_is_zero_residual_then_learns():
    disabled = _DFMJepaConditioningHarness(enabled=False)
    original = torch.randn(3, 64, 2)
    disabled_result = JointModel.condition_dfm_latents(
        disabled,
        original,
        None,
        torch.float32,
    )
    assert disabled_result.latents is original
    torch.testing.assert_close(
        disabled_result.residual_rms,
        torch.zeros(3),
    )

    active = _DFMJepaConditioningHarness(enabled=True)
    z_dfm = original.detach().clone().requires_grad_()
    z_jepa = torch.randn(3, 4, requires_grad=True)
    zero_result = JointModel.condition_dfm_latents(
        active,
        z_dfm,
        z_jepa,
        torch.float32,
    )
    torch.testing.assert_close(
        zero_result.latents,
        z_dfm,
        rtol=0.0,
        atol=0.0,
    )
    assert float(zero_result.weight_rms.detach()) == 0.0
    zero_result.latents.sum().backward()
    assert active.dfm_jepa_state_adapter is not None
    assert active.dfm_jepa_state_adapter.w.grad is not None
    assert torch.count_nonzero(
        active.dfm_jepa_state_adapter.w.grad
    )
    assert z_jepa.grad is not None
    assert torch.count_nonzero(z_jepa.grad) == 0

    active.zero_grad(set_to_none=True)
    with torch.no_grad():
        active.dfm_jepa_state_adapter.w.fill_(0.25)
    z_jepa = z_jepa.detach().clone().requires_grad_()
    learned_result = JointModel.condition_dfm_latents(
        active,
        z_dfm.detach(),
        z_jepa,
        torch.float32,
    )
    assert torch.all(learned_result.residual_rms > 0.0)
    assert float(learned_result.weight_rms.detach()) == pytest.approx(
        0.25
    )
    learned_result.latents.sum().backward()
    assert z_jepa.grad is not None
    assert torch.count_nonzero(z_jepa.grad)


class _NormalizedDFMJepaHarness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            token_dim=2,
            z_dim=4,
            dfm_condition_on_current_jepa_state=False,
            dfm_jepa_fusion_mode="normalized_add",
        )
        self.dfm_jepa_state_adapter = RawLinear(
            4,
            2,
            dtype=torch.float32,
            bias=False,
        )


def test_normalized_dfm_jepa_fusion_has_equal_scale_branches():
    model = _NormalizedDFMJepaHarness()
    with torch.no_grad():
        model.dfm_jepa_state_adapter.w.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ]
            )
        )
    z_dfm = torch.randn(3, 64, 2, requires_grad=True)
    z_jepa = torch.randn(3, 4, requires_grad=True)
    result = JointModel.condition_dfm_latents(
        model,
        z_dfm,
        z_jepa,
        torch.float32,
    )
    residual = model.dfm_jepa_state_adapter(z_jepa, torch.float32)
    normalized_residual = residual * torch.rsqrt(
        residual.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = (
        z_dfm
        * torch.rsqrt(z_dfm.square().mean(dim=-1, keepdim=True) + 1e-6)
        + normalized_residual.unsqueeze(1)
    ) / math.sqrt(2.0)
    torch.testing.assert_close(result.latents, expected)
    result.latents.square().mean().backward()
    assert z_dfm.grad is not None and torch.count_nonzero(z_dfm.grad)
    assert z_jepa.grad is not None and torch.count_nonzero(z_jepa.grad)


class _ConditioningInitializationHarness(torch.nn.Module):
    def __init__(self, *, enabled: bool):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.pre = RawLinear(3, 3, dtype=torch.float32, bias=False)
        self.dfm_jepa_state_adapter = (
            RawLinear(4, 2, dtype=torch.float32, bias=False)
            if enabled
            else None
        )
        self.post = RawLinear(3, 3, dtype=torch.float32, bias=False)


def test_zero_bridge_does_not_perturb_shared_initialization_rng():
    control = _ConditioningInitializationHarness(enabled=False)
    candidate = _ConditioningInitializationHarness(enabled=True)

    initialize_fresh_modules(control, seed=17)
    initialize_fresh_modules(candidate, seed=17)

    torch.testing.assert_close(candidate.pre.w, control.pre.w)
    torch.testing.assert_close(candidate.post.w, control.post.w)
    assert candidate.dfm_jepa_state_adapter is not None
    assert torch.count_nonzero(candidate.dfm_jepa_state_adapter.w) == 0


class _OptionalAdapterInitializationHarness(torch.nn.Module):
    def __init__(self, *, adapters: bool):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.config = SimpleNamespace(
            dfm_condition_on_current_jepa_state=False,
        )
        self.pre = RawLinear(3, 3, dtype=torch.float32, bias=False)
        self.dfm_jepa_state_adapter = (
            RawLinear(4, 2, dtype=torch.float32, bias=False)
            if adapters
            else None
        )
        self.dfm_jepa_rollout_adapter = (
            RawLinear(4, 2, dtype=torch.float32, bias=False)
            if adapters
            else None
        )
        self.post = RawLinear(3, 3, dtype=torch.float32, bias=False)


def test_optional_adapters_preserve_shared_hero_initialization_stream():
    control = _OptionalAdapterInitializationHarness(adapters=False)
    candidate = _OptionalAdapterInitializationHarness(adapters=True)

    initialize_fresh_modules(control, seed=23)
    initialize_fresh_modules(candidate, seed=23)

    torch.testing.assert_close(candidate.pre.w, control.pre.w)
    torch.testing.assert_close(candidate.post.w, control.post.w)
    assert candidate.dfm_jepa_state_adapter is not None
    assert candidate.dfm_jepa_rollout_adapter is not None
    assert torch.count_nonzero(candidate.dfm_jepa_state_adapter.w)
    assert torch.count_nonzero(candidate.dfm_jepa_rollout_adapter.w)


def test_current_wdl_supervision_swaps_win_loss_and_keeps_scale():
    predicted = torch.randn(2, 2, 4)
    current = torch.randn(2, 4)
    raw = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    valid = torch.ones(2)
    future_valid = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    states, targets, weight = _prepare_wdl_supervision(
        predicted,
        current,
        raw,
        valid,
        future_valid,
        include_current_state=True,
    )
    assert states.shape == (2, 3, 4)
    torch.testing.assert_close(states[:, 0], current)
    torch.testing.assert_close(
        targets[:, 0],
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        weight,
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
    )


def test_root_proposal_is_legal_target_independent_and_fail_closed():
    legal_idx = torch.tensor(
        [
            [3, 5, 65535, 65535],
            [7, 7, 65535, 65535],
        ]
    )
    legal_count = torch.tensor([2, 2])
    legal_mask = _root_legal_mask_from_indices(legal_idx, legal_count)
    assert torch.equal(
        legal_mask[0].nonzero().flatten(),
        torch.tensor([3, 5]),
    )
    assert torch.equal(
        legal_mask[1].nonzero().flatten(),
        torch.tensor([7]),
    )

    logits = torch.full((2, 1858), -10.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 3] = 1.0
        logits[0, 5] = 2.0
        logits[1, 7] = 4.0
    proposal = _proposal_from_root_logits(
        logits,
        torch.tensor([1858, 3]),
        legal_mask,
        torch.tensor([True, True]),
    )
    assert proposal.action.tolist() == [5, 3]
    assert 0.5 < proposal.feedback_gate[0] < 1.0
    assert proposal.feedback_gate[1] == 1.0
    assert not proposal.action.requires_grad
    assert not proposal.feedback_gate.requires_grad

    invalid = _proposal_from_root_logits(
        logits,
        torch.tensor([1858, 1858]),
        legal_mask,
        torch.tensor([False, False]),
    )
    assert invalid.feedback_gate.tolist() == [0.0, 0.0]


def test_closed_loop_proposal_is_legal_and_preserves_visible_targets():
    logits = torch.full((1, 3, 1858), -10.0)
    logits[0, 0, 100] = 50.0
    logits[0, 0, 3] = 2.0
    logits[0, 0, 5] = 3.0
    logits[0, 1, 99] = 9.0
    logits[0, 2, 77] = 8.0
    legal = torch.zeros((1, 1858), dtype=torch.bool)
    legal[:, 3] = True
    legal[:, 5] = True
    proposal = _closed_loop_proposal_actions(
        logits,
        torch.tensor([[1858, 42, 1858]]),
        torch.tensor([[True, False, True]]),
        legal,
    )
    torch.testing.assert_close(
        proposal,
        torch.tensor([[5, 42, 77]]),
    )
    assert not proposal.requires_grad


class _FeedbackHarness(torch.nn.Module):
    class _Config:
        token_dim = 2
        z_dim = 4
        jepa_feedback_mode = "final_pass_adjoint"

    def __init__(self):
        super().__init__()
        self.config = self._Config()
        self.jepa_hidden_adapter = RawLinear(2, 4, dtype=torch.float32)


def test_adjoint_feedback_math_cap_and_gradients():
    model = _FeedbackHarness()
    with torch.no_grad():
        model.jepa_hidden_adapter.w.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                ]
            )
        )
    z_dfm = torch.full(
        (2, 3, 2),
        4.0,
        requires_grad=True,
    )
    z0 = torch.zeros((2, 4), requires_grad=True)
    z1 = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [4.0, 4.0, 0.0, 0.0]],
        requires_grad=True,
    )
    result = JointModel.dfm_latents_with_jepa_feedback(
        model,
        z_dfm,
        z0,
        z1,
        torch.tensor([1.0, 0.0]),
        torch.float32,
    )

    expected_raw_first = torch.tensor(
        [math.sqrt(0.5), math.sqrt(2.0)]
    )
    torch.testing.assert_close(
        result.latents[0, 0] - z_dfm[0, 0],
        expected_raw_first,
    )
    torch.testing.assert_close(
        result.latents[1],
        z_dfm[1],
    )
    assert result.cap_fraction == 0.5
    result.latents.sum().backward()
    assert z_dfm.grad is not None and torch.count_nonzero(z_dfm.grad)
    assert z0.grad is not None and torch.count_nonzero(z0.grad)
    assert z1.grad is not None and torch.count_nonzero(z1.grad)
    assert model.jepa_hidden_adapter.w.grad is not None
    assert torch.count_nonzero(model.jepa_hidden_adapter.w.grad)


def test_hero_optimizer_hyperparameters_are_frozen_after_lr_range():
    assert HERO_CONFIG.learning_rate == 5e-4
    assert HERO_CONFIG.bt4_learning_rate == pytest.approx(5e-4 / 30.0)
    assert HERO_CONFIG.weight_decay == 1e-2


def test_hero_milestones_round_up_by_examples():
    expected = {
        10: 2768,
        20: 5536,
        25: 6920,
        50: 13840,
        75: 20760,
        100: 27679,
    }
    assert {
        percentage: _hero_milestone_update(
            percentage,
            total_examples=28_343_296,
            batch_size=1024,
        )
        for percentage in expected
    } == expected


def test_latent_spectrum_metrics_expose_rank_and_centered_scale():
    generator = torch.Generator().manual_seed(20260727)
    diverse = torch.randn(32, 12, generator=generator)
    collapsed = torch.full((32, 12), 3.0)
    values = torch.stack((diverse, collapsed), dim=1)
    weights = torch.ones((32, 2))

    metrics = _latent_spectrum_metrics(values, weights)

    assert metrics["effective_rank"][0] > 5.0
    assert metrics["stable_rank"][0] > 2.0
    assert metrics["centered_rms"][0] > 0.5
    assert metrics["explained_variance_top1"][0] < 1.0
    torch.testing.assert_close(
        metrics["explained_variance_top32"][0],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["effective_rank"][1],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["stable_rank"][1],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["centered_rms"][1],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["explained_variance_top32"][1],
        torch.tensor(0.0),
    )


class _FixedArenaPlanner:
    class _Config:
        horizon = 8
        action_codec = "lc0_canonical_1858"
        jepa_feedback_mode = "none"

    config = _Config()

    def __init__(self):
        self.planner_calls = 0

    def planner(
        self,
        z_dfm,
        action_tokens,
        t,
        compute_dtype,
        *,
        base_root_logits,
        return_hidden=False,
    ):
        del action_tokens, t, compute_dtype, base_root_logits
        assert not return_hidden
        self.planner_calls += 1
        batch_size = z_dfm.shape[0]
        logits = torch.full(
            (batch_size, 8, 1858),
            -8.0,
            dtype=torch.float32,
        )
        logits[:, 0, 100] = 20.0
        logits[:, 0, 3] = 1.0
        logits[:, 0, 5] = 2.0
        for horizon in range(1, 8):
            logits[:, horizon, 10 + horizon] = 3.0 + horizon
        return logits


@pytest.mark.parametrize("refinement_passes", [1, 2, 4, 8, 16])
def test_torch_dfm_refinement_masks_root_and_unmasks_all_positions(
    refinement_passes,
):
    model = _FixedArenaPlanner()
    root_legal_mask = torch.zeros((2, 1858), dtype=torch.bool)
    root_legal_mask[:, 3] = True
    root_legal_mask[:, 5] = True
    selected = _torch_refine_dfm_actions(
        model,
        torch.zeros((2, 64, 256)),
        torch.zeros((2, 1858)),
        root_legal_mask,
        refinement_passes=refinement_passes,
        compute_dtype=torch.float32,
    )

    assert model.planner_calls == refinement_passes
    torch.testing.assert_close(selected, torch.full((2,), 5))


@pytest.mark.parametrize("refinement_passes", [1, 2, 4, 8, 16])
def test_torch_hero_arena_policy_accepts_feedback_off_pass_sweep(
    refinement_passes,
):
    policy = TorchHeroArenaPolicy(
        model=_FixedArenaPlanner(),
        model_id=f"pass-{refinement_passes}",
        policy_mode="dfm",
        inference_batch_size=16,
        refinement_passes=refinement_passes,
    )
    assert policy.refinement_passes == refinement_passes


def test_torch_hero_arena_policy_accepts_checkpoint_policy_only_mode():
    policy = TorchHeroArenaPolicy(
        model=_FixedArenaPlanner(),
        model_id="hero-policy-only",
        policy_mode="policy_only",
        inference_batch_size=16,
        refinement_passes=16,
    )

    assert policy.policy_mode == "policy_only"


@pytest.mark.parametrize("refinement_passes", [1, 2, 4, 16])
def test_torch_hero_arena_policy_keeps_feedback_at_eight(
    refinement_passes,
):
    with pytest.raises(ValueError, match="horizon/pass count 8"):
        TorchHeroArenaPolicy(
            model=_FeedbackArenaPlanner(),
            model_id=f"feedback-pass-{refinement_passes}",
            policy_mode="dfm",
            inference_batch_size=16,
            refinement_passes=refinement_passes,
        )


class _FeedbackArenaPlanner:
    class _Config:
        horizon = 8
        token_dim = 256
        z_dim = 4
        action_codec = "lc0_canonical_1858"
        jepa_feedback_mode = "final_pass_adjoint"

    def __init__(self):
        self.config = self._Config()
        self.planner_latents = []
        self.feedback_calls = 0

    def planner(
        self,
        z_dfm,
        action_tokens,
        t,
        compute_dtype,
        *,
        base_root_logits,
        return_hidden=False,
    ):
        del action_tokens, t, compute_dtype, base_root_logits
        self.planner_latents.append(z_dfm.detach().clone())
        batch_size = z_dfm.shape[0]
        logits = torch.full(
            (batch_size, 8, 1858),
            -8.0,
            dtype=torch.float32,
        )
        logits[:, 0, 5] = 4.0
        for horizon in range(1, 8):
            logits[:, horizon, 10 + horizon] = 3.0 + horizon
        if return_hidden:
            return logits, torch.zeros((batch_size, 8, 256))
        return logits

    def jepa_step(
        self,
        z0,
        action,
        action_hidden,
        compute_dtype,
    ):
        del action, action_hidden, compute_dtype
        return z0 + 1.0

    def dfm_latents_with_jepa_feedback(
        self,
        z_dfm,
        z0,
        z1,
        gate,
        compute_dtype,
    ):
        del z0, z1, gate, compute_dtype
        self.feedback_calls += 1
        zeros = torch.zeros((z_dfm.shape[0],))
        return JepaFeedbackResult(
            latents=z_dfm + 2.0,
            delta_rms=zeros,
            raw_feedback_rms=zeros,
            applied_feedback_rms=zeros,
            state_rms=zeros,
            cap_fraction=torch.tensor(0.0),
        )


def test_torch_dfm_feedback_changes_only_pass_eight():
    model = _FeedbackArenaPlanner()
    z_dfm = torch.zeros((2, 64, 256))
    root_legal_mask = torch.zeros((2, 1858), dtype=torch.bool)
    root_legal_mask[:, 3] = True
    root_legal_mask[:, 5] = True
    selected = _torch_refine_dfm_actions(
        model,
        z_dfm,
        torch.zeros((2, 1858)),
        root_legal_mask,
        refinement_passes=8,
        compute_dtype=torch.float32,
        z_jepa=torch.zeros((2, 4)),
    )

    assert model.feedback_calls == 1
    assert len(model.planner_latents) == 8
    for latents in model.planner_latents[:7]:
        torch.testing.assert_close(latents, z_dfm)
    torch.testing.assert_close(
        model.planner_latents[7],
        z_dfm + 2.0,
    )
    torch.testing.assert_close(selected, torch.full((2,), 5))


class _ClosedLoopArenaPlanner:
    class _Config:
        horizon = 8
        token_dim = 256
        z_dim = 4
        action_codec = "lc0_canonical_1858"
        jepa_feedback_mode = "none"
        dfm_closed_loop_mode = "predicted_jepa_tokens"

    def __init__(self):
        self.config = self._Config()
        self.contexts = []
        self.rollout_calls = 0

    def planner(
        self,
        z_dfm,
        action_tokens,
        t,
        compute_dtype,
        *,
        base_root_logits,
        jepa_action_context=None,
        return_hidden=False,
    ):
        del action_tokens, t, compute_dtype, base_root_logits
        self.contexts.append(
            None
            if jepa_action_context is None
            else jepa_action_context.detach().clone()
        )
        batch_size = z_dfm.shape[0]
        logits = torch.full((batch_size, 8, 1858), -8.0)
        logits[:, 0, 5] = 4.0
        for horizon in range(1, 8):
            logits[:, horizon, 10 + horizon] = 3.0 + horizon
        if return_hidden:
            hidden = torch.ones((batch_size, 8, 256))
            return logits, hidden
        return logits

    def jepa_rollout(
        self,
        z0,
        actions,
        action_hidden,
        compute_dtype,
    ):
        del actions, action_hidden, compute_dtype
        self.rollout_calls += 1
        return z0.unsqueeze(1).expand(-1, 8, -1) + 2.0

    def jepa_rollout_action_context(
        self,
        predicted_states,
        compute_dtype,
    ):
        del compute_dtype
        return predicted_states[..., :1].expand(-1, -1, 256)


def test_torch_dfm_closed_loop_feeds_predicted_states_into_next_pass():
    model = _ClosedLoopArenaPlanner()
    root_legal_mask = torch.zeros((2, 1858), dtype=torch.bool)
    root_legal_mask[:, 3] = True
    root_legal_mask[:, 5] = True
    selected = _torch_refine_dfm_actions(
        model,
        torch.zeros((2, 64, 256)),
        torch.zeros((2, 1858)),
        root_legal_mask,
        refinement_passes=2,
        compute_dtype=torch.float32,
        z_jepa=torch.zeros((2, 4)),
    )

    assert model.rollout_calls == 1
    assert model.contexts[0] is None
    assert model.contexts[1] is not None
    torch.testing.assert_close(
        model.contexts[1],
        torch.full((2, 8, 256), 2.0),
    )
    torch.testing.assert_close(selected, torch.full((2,), 5))


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
    small_batch.set_learning_rates(main=7e-4, bt4=2e-5)
    assert small_batch._learning_rate("main") == 7e-4
    assert small_batch._learning_rate("bt4") == 2e-5


def test_lr_range_analysis_is_plot_ready_and_finds_descent_candidates():
    learning_rates = np.geomspace(1e-5, 1e-2, num=64)
    records = [
        {
            "update": index + 1,
            "learning_rate": learning_rate,
            "bt4_learning_rate": learning_rate / 30.0,
            "loss": 10.0 + (np.log10(learning_rate) + 3.2) ** 2,
        }
        for index, learning_rate in enumerate(learning_rates)
    ]

    analysis = _analyze_lr_range_records(
        records,
        smoothing_beta=0.0,
        regression_radius=3,
    )

    assert len(analysis["curve"]) == len(records)
    assert analysis["curve"][0]["local_loss_slope_per_log_lr"] is None
    assert (
        analysis["candidates"]["steepest_smoothed_descent"]["learning_rate"]
        < analysis["candidates"]["minimum_smoothed_loss"]["learning_rate"]
    )
    assert 1e-5 <= analysis["candidates"]["one_decade_below_minimum_loss"][
        "learning_rate"
    ] <= 1e-2


def test_frozen_evaluation_indices_are_sorted_and_fail_closed():
    normalized = _normalize_frozen_indices(
        np.asarray([7, 1, 4, 2]),
        total_examples=8,
        batch_size=2,
    )
    np.testing.assert_array_equal(normalized, np.asarray([1, 2, 4, 7]))

    with pytest.raises(ValueError, match="duplicates"):
        _normalize_frozen_indices(
            np.asarray([1, 1]),
            total_examples=8,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="not divisible"):
        _normalize_frozen_indices(
            np.asarray([1, 2, 3]),
            total_examples=8,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        _normalize_frozen_indices(
            np.asarray([1, 8]),
            total_examples=8,
            batch_size=2,
        )
    _proposal_from_root_logits,
    _root_legal_mask_from_indices,
