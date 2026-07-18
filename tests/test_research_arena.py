from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tempfile
from pathlib import Path

import chess
import numpy as np
import pytest

from research.arena import (
    DRAW_RESULT,
    GSPRTConfig,
    GSPRTState,
    LEGACY_ACTION_CODEC_CAPABILITY,
    NORMALIZED_ELO_PER_T_VALUE,
    NORMALIZED_PROMOTION_MAX_PAIRS,
    OPENING_SELECTION_ALGORITHM,
    PentanomialStats,
    arena_foundation_contract,
    build_opening_pool,
    canonicalize_fen,
    classify_game_outcome,
    load_opening_pool,
    make_color_reversed_pairs,
    opening_selection_hash,
    pair_aware_score_elo_interval,
    pair_score_for_model,
    pentanomial_gsprt_llr,
    pentanomial_normalized_elo_diagnostics,
    pentanomial_stats,
    save_opening_pool,
)
from research.prepare import REPO_ROOT


@pytest.fixture
def workspace_tmp() -> Path:
    root = REPO_ROOT / ".local" / "tmp" / "arena-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def _fen_after_deterministic_line(seed: int, *, plies: int = 12) -> str:
    board = chess.Board()
    for ply in range(plies):
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        board.push(moves[(seed * 11 + ply * 7) % len(moves)])
    assert board.is_valid()
    assert not board.is_game_over(claim_draw=True)
    return board.fen(en_passant="legal")


def _terminal_fen() -> str:
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(move)
    assert board.is_game_over()
    return board.fen(en_passant="legal")


def _write_metadata_shard(
    path: Path,
    *,
    fens: list[str],
    plies: list[int],
    schema: str = "trajectory-v3",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=np.asarray(schema),
        batch_size=np.asarray(len(fens), dtype=np.int32),
        fen_t=np.asarray(fens),
        ply=np.asarray(plies, dtype=np.int32),
        source_uri=np.asarray(f"fixture://{path.name}"),
    )


def test_opening_pool_is_deduplicated_hash_ranked_persisted_and_excludable(
    workspace_tmp: Path,
):
    val_dir = workspace_tmp / "val"
    fens = [_fen_after_deterministic_line(seed) for seed in range(8)]
    invalid_castling = fens[0].replace(" Kkq ", " KQkq ")
    inconsistent_ply_fen = _fen_after_deterministic_line(17, plies=10)
    first = val_dir / "chunk_000001.npz"
    second = val_dir / "chunk_000000.npz"
    _write_metadata_shard(
        first,
        fens=[
            fens[0],
            fens[1],
            fens[2],
            invalid_castling,
            _terminal_fen(),
            inconsistent_ply_fen,
            fens[7],
        ],
        plies=[12, 12, 12, 12, 12, 12, 11],
    )
    _write_metadata_shard(
        second,
        fens=[fens[0], fens[3], fens[4], fens[5], fens[6]],
        plies=[12, 12, 12, 12, 12],
    )

    pool = build_opening_pool(
        [first, second],
        seed=0x1234,
        count=5,
    )
    reordered = build_opening_pool(
        [second, first],
        seed=0x1234,
        count=5,
    )
    assert pool == reordered
    assert pool["selection"]["opening_ply"] == 12
    assert pool["selection"]["deduplicated_candidate_count"] == 7
    assert pool["selection"]["invalid_standard_fen_count"] == 1
    assert pool["selection"]["terminal_fen_count"] == 1
    assert pool["selection"]["inconsistent_fen_ply_count"] == 1
    assert pool["selection"]["duplicate_fen_count"] == 1
    assert len({entry["fen"] for entry in pool["openings"]}) == 5
    rank_keys = [
        (entry["selection_hash"], entry["fen"])
        for entry in pool["openings"]
    ]
    assert rank_keys == sorted(rank_keys)

    golden_fen = canonicalize_fen(fens[0])
    expected_hash = hashlib.sha256(
        OPENING_SELECTION_ALGORITHM.encode("utf-8")
        + b"\0"
        + (0x1234).to_bytes(8, "big")
        + b"\0"
        + golden_fen.encode("utf-8")
    ).hexdigest()
    assert opening_selection_hash(golden_fen, seed=0x1234) == expected_hash

    different_seed = build_opening_pool(
        [first, second],
        seed=0x5678,
        count=5,
    )
    assert different_seed["pool_sha256"] != pool["pool_sha256"]
    assert (
        different_seed["ordered_fens_sha256"]
        != pool["ordered_fens_sha256"]
    )

    excluded = [entry["fen"] for entry in pool["openings"]]
    fresh = build_opening_pool(
        [first, second],
        seed=0x9ABC,
        count=2,
        exclude_fens=excluded,
    )
    assert {entry["fen"] for entry in fresh["openings"]}.isdisjoint(
        excluded
    )
    assert fresh["excluded_fens"]["count"] == len(excluded)

    destination = workspace_tmp / "pools" / "dev-v1.json"
    assert save_opening_pool(pool, destination) == destination
    assert load_opening_pool(destination) == pool
    assert load_opening_pool(
        destination,
        expected_pool_sha256=pool["pool_sha256"],
    ) == pool
    with pytest.raises(ValueError, match="pinned digest"):
        load_opening_pool(
            destination,
            expected_pool_sha256="0" * 64,
        )
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    persisted["selection"]["seed"] += 1
    destination.write_text(
        json.dumps(persisted, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        load_opening_pool(destination)


def test_opening_pool_fails_closed_on_nonheldout_bad_or_insufficient_sources(
    workspace_tmp: Path,
):
    fen = _fen_after_deterministic_line(0)
    train_shard = workspace_tmp / "train" / "chunk_000000.npz"
    _write_metadata_shard(
        train_shard,
        fens=[fen],
        plies=[12],
    )
    with pytest.raises(ValueError, match="held-out split"):
        build_opening_pool([train_shard], seed=1, count=1)

    bad_schema = workspace_tmp / "val" / "chunk_000000.npz"
    _write_metadata_shard(
        bad_schema,
        fens=[fen],
        plies=[12],
        schema="trajectory-v2",
    )
    with pytest.raises(ValueError, match="Expected trajectory-v3"):
        build_opening_pool([bad_schema], seed=1, count=1)

    valid = workspace_tmp / "test" / "chunk_000000.npz"
    _write_metadata_shard(valid, fens=[fen], plies=[12])
    with pytest.raises(ValueError, match="Only 1 unique"):
        build_opening_pool([valid], seed=1, count=2)


def test_color_reversed_pairs_and_fail_closed_outcomes_are_symmetric():
    fens = [
        _fen_after_deterministic_line(0),
        _fen_after_deterministic_line(1),
    ]
    pairs = make_color_reversed_pairs(
        fens,
        model_a="candidate",
        model_b="reference",
    )
    assert len(pairs) == 2
    for first, second in pairs:
        assert first.pair_id == second.pair_id
        assert first.fen == second.fen
        assert first.white_model == second.black_model == "candidate"
        assert first.black_model == second.white_model == "reference"

    first, second = pairs[0]
    for termination in ("illegal_move", "timeout", "exception"):
        candidate_fault = classify_game_outcome(
            first,
            termination=termination,
            fault_model="candidate",
            ply_count=7,
            ply_cap=80,
        )
        assert candidate_fault.score_for("candidate") == 0.0
        assert candidate_fault.loser_model == "candidate"
        reference_fault = classify_game_outcome(
            second,
            termination=termination,
            fault_model="reference",
            ply_count=8,
            ply_cap=80,
        )
        assert reference_fault.score_for("candidate") == 1.0

    cap_first = classify_game_outcome(
        first,
        termination="ply_cap",
        ply_count=80,
        ply_cap=80,
    )
    cap_second = classify_game_outcome(
        second,
        termination="ply_cap",
        ply_count=80,
        ply_cap=80,
    )
    assert cap_first.result == cap_second.result == DRAW_RESULT
    assert pair_score_for_model(
        [cap_first, cap_second],
        model_id="candidate",
    ) == 1.0

    with pytest.raises(ValueError, match="model at fault"):
        classify_game_outcome(
            first,
            termination="exception",
            ply_count=1,
            ply_cap=80,
        )
    with pytest.raises(ValueError, match="requires a final result"):
        classify_game_outcome(
            first,
            termination="normal",
            ply_count=1,
            ply_cap=80,
            reported_result="*",
        )
    for invalid_ply_count in (79, 81):
        with pytest.raises(ValueError, match="exactly at 80"):
            classify_game_outcome(
                first,
                termination="ply_cap",
                ply_count=invalid_ply_count,
                ply_cap=80,
            )
    with pytest.raises(ValueError, match="game_in_pair 0 and 1"):
        pair_score_for_model(
            [cap_first, dataclasses.replace(cap_second, game_in_pair=0)],
            model_id="candidate",
        )


def test_pentanomial_score_and_pair_aware_uncertainty_are_non_degenerate():
    stats = pentanomial_stats([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])
    assert stats.counts == (1, 1, 2, 1, 1)
    assert stats.pair_count == 6
    assert stats.game_count == 12
    assert stats.points == 6.0
    assert stats.score == 0.5

    interval = pair_aware_score_elo_interval(stats)
    assert interval.confidence == 0.95
    assert interval.score_lower < interval.score < interval.score_upper
    assert interval.elo == 0.0
    assert interval.elo_lower < 0.0 < interval.elo_upper
    assert interval.elo_model == "logistic"
    assert interval.promotion_eligible is False

    all_wins = pair_aware_score_elo_interval(
        pentanomial_stats([2.0] * 8)
    )
    assert all_wins.score == 1.0
    assert all_wins.score_lower < all_wins.score_upper == 1.0

    with pytest.raises(ValueError, match="Pair score must be one of"):
        pentanomial_stats([0.25])


def test_pentanomial_gsprt_matches_reference_and_stops_only_at_pair_cap():
    reference_counts = (10789, 19328, 33806, 19402, 10543)
    llr = pentanomial_gsprt_llr(
        reference_counts,
        elo0=-3.0,
        elo1=1.0,
    )
    assert llr == pytest.approx(2.1310678117940807, abs=2e-11)

    concentrated = pentanomial_gsprt_llr(
        (0, 0, 10, 0, 0),
        elo0=0.0,
        elo1=20.0,
    )
    dispersed = pentanomial_gsprt_llr(
        (5, 0, 0, 0, 5),
        elo0=0.0,
        elo1=20.0,
    )
    assert concentrated != pytest.approx(dispersed)

    config = GSPRTConfig(
        elo0=-100.0,
        elo1=100.0,
        max_pairs=3,
    )
    state = GSPRTState(config)
    for expected_pairs in (1, 2):
        state = state.update(1.0)
        assert state.pair_count == expected_pairs
        assert state.decision == "continue"
    state = state.update(1.0)
    assert state.pair_count == 3
    assert state.decision == "max_pairs"
    assert state.terminal
    assert state.as_dict()["update_unit"] == (
        "completed_color_reversed_pair"
    )
    assert state.as_dict()["promotion_eligible"] is False
    assert GSPRTState.from_dict(state.as_dict()) == state
    with pytest.raises(RuntimeError, match="terminal GSPRT"):
        state.update(2.0)

    with pytest.raises(ValueError, match="elo_model"):
        GSPRTConfig(
            elo0=0.0,
            elo1=20.0,
            elo_model="unsupported",
        )


def test_normalized_elo_gsprt_matches_official_fishtest_reference():
    # Values independently evaluated with official-stockfish/fishtest
    # server/fishtest/stats/LLRcalc.py::LLR_normalized.
    assert pentanomial_gsprt_llr(
        (10789, 19328, 33806, 19402, 10543),
        elo0=0.0,
        elo1=20.0,
        elo_model="normalized",
    ) == pytest.approx(-340.2492043700217, abs=1e-10)
    assert pentanomial_gsprt_llr(
        (100, 200, 400, 250, 150),
        elo0=0.0,
        elo1=20.0,
        elo_model="normalized",
    ) == pytest.approx(7.004863094500454, abs=1e-11)
    assert pentanomial_gsprt_llr(
        (39, 2226, 31451, 2412, 40),
        elo0=0.764,
        elo1=3.439,
        elo_model="normalized",
    ) == pytest.approx(2.162940869584568, abs=1e-11)

    diagnostics = pentanomial_normalized_elo_diagnostics(
        (100, 200, 400, 250, 150)
    )
    assert diagnostics.pair_count == 1100
    assert diagnostics.regularized_sample_count == 1100.0
    assert diagnostics.pair_mean == pytest.approx(0.5340909090909091)
    assert diagnostics.pair_variance == pytest.approx(
        0.08122417355371901
    )
    assert diagnostics.normalized_t_value == pytest.approx(
        0.08458258116519006
    )
    assert diagnostics.normalized_elo == pytest.approx(
        29.386998612140772
    )
    assert diagnostics.normalized_elo == pytest.approx(
        diagnostics.normalized_t_value * NORMALIZED_ELO_PER_T_VALUE
    )


def test_normalized_promotion_gate_is_auditable_and_passes_at_pair_boundaries():
    config = GSPRTConfig.normalized_promotion()
    assert config.elo0 == 0.0
    assert config.elo1 == 20.0
    assert config.max_pairs == NORMALIZED_PROMOTION_MAX_PAIRS
    assert config.elo_model == "normalized"
    assert config.promotion_eligible

    state = GSPRTState(config)
    while not state.terminal:
        state = state.update(2.0)
    assert state.pair_count == 38
    assert state.decision == "accept_h1"
    payload = state.as_dict()
    assert payload["schema_version"] == (
        "chess-dfm-arena-normalized-elo-gsprt-v1"
    )
    assert payload["update_unit"] == "completed_color_reversed_pair"
    assert payload["llr_method"] == (
        "pentanomial_constrained_mle_normalized_t_value"
    )
    assert payload["promotion_eligible"] is True
    assert payload["promotion_gate"]["candidate_passed"] is True
    assert payload["promotion_gate"]["hard_pair_cap"] == 2048
    assert payload["promotion_gate"]["hypothesis_pair_t_values"] == (
        pytest.approx([0.0, 0.08140867667575735])
    )
    audit = payload["normalized_elo_diagnostics"]
    assert audit["pair_count"] == state.pair_count
    assert audit["regularization_pseudocount"] == 1e-3
    assert math.isfinite(audit["normalized_elo"])
    assert GSPRTState.from_dict(payload) == state


def test_normalized_promotion_gate_fails_closed_on_bad_or_incomplete_pairs():
    with pytest.raises(ValueError, match="H0=0 and H1=20"):
        GSPRTConfig(
            elo0=-1.0,
            elo1=20.0,
            elo_model="normalized",
        )
    with pytest.raises(ValueError, match="hard cap of 2048"):
        GSPRTConfig.normalized_promotion(max_pairs=2049)

    invalid_counts = (
        (1.0, 0, 0, 0, 0),
        (True, 0, 0, 0, 0),
        (-1, 0, 0, 0, 0),
        ("1", 0, 0, 0, 0),
    )
    for counts in invalid_counts:
        with pytest.raises(ValueError, match="non-negative integers"):
            PentanomialStats(counts)
        with pytest.raises(ValueError, match="non-negative integers"):
            pentanomial_gsprt_llr(
                counts,
                elo0=0.0,
                elo1=20.0,
                elo_model="normalized",
            )

    first, second = make_color_reversed_pairs(
        [_fen_after_deterministic_line(41)],
        model_a="candidate",
        model_b="reference",
    )[0]
    first_outcome = classify_game_outcome(
        first,
        termination="ply_cap",
        ply_count=8,
        ply_cap=8,
    )
    second_outcome = classify_game_outcome(
        second,
        termination="ply_cap",
        ply_count=8,
        ply_cap=8,
    )
    state = GSPRTState(GSPRTConfig.normalized_promotion(max_pairs=2))
    with pytest.raises(ValueError, match="requires two games"):
        state.update_from_pair_outcomes(
            [first_outcome],
            model_id="candidate",
        )
    state = state.update_from_pair_outcomes(
        [first_outcome, second_outcome],
        model_id="candidate",
    )
    assert state.pair_count == 1
    assert state.counts == (0, 0, 1, 0, 0)

    tampered = state.as_dict()
    tampered["counts"][2] = 1.5
    with pytest.raises(ValueError, match="non-negative integers"):
        GSPRTState.from_dict(tampered)


def test_foundation_contract_records_missing_adapters_and_codec_handicap():
    contract = arena_foundation_contract()
    assert contract["engine_adapter"] == "absent"
    assert contract["gpu_model_loading"] == "absent"
    assert contract["uci_adapter"] == "absent"
    promotion = contract["promotion_stopping"]
    assert promotion["status"] == "supported"
    assert promotion["elo_model"] == "normalized"
    assert promotion["elo0"] == 0.0
    assert promotion["elo1"] == 20.0
    assert promotion["max_games"] == 4096
    assert promotion["max_pairs"] == 2048
    assert promotion["promotion_eligible"] is True
    assert promotion["llr_method"] == (
        "pentanomial_constrained_mle_normalized_t_value"
    )
    codec = contract["legacy_action_codec"]
    assert codec == LEGACY_ACTION_CODEC_CAPABILITY
    assert codec["complete_legal_move_coverage"] is False
    assert "white_knight_promotion" in codec[
        "unrepresentable_move_classes"
    ]
    assert "black_knight_promotion" in codec[
        "unrepresentable_move_classes"
    ]
