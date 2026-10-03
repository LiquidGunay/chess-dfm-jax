import copy

import chess
import numpy as np
import pytest

from chess_dfm_jax.encoding import encode_board
from research.analysis.audit_hero_history_recovery import audit_payloads, profile_payload


def payload():
    moves = "g1f3 g8f6 f3g1 f6g8 e2e4 d7d5 e4d5 d8d5 b1c3 d5a5".split()
    board = chess.Board()
    fens, planes = [board.fen()], [encode_board(board, []).astype(np.uint8)]
    for uci in moves:
        board.push_uci(uci)
        fens.append(board.fen())
        planes.append(encode_board(board, []).astype(np.uint8))
    h, n = 3, len(moves) - 2
    return {"schema_version": np.array("trajectory-v3"), "batch_size": np.array(n),
            "ply": np.arange(n), "fen_t": np.array(fens[:n]),
            "game_id": np.array(["placeholder"] * n), "result": np.array(["1-0"] * n),
            "input_format": np.array(["INPUT_CLASSICAL_112_PLANE"] * n),
            "actions_uci": np.array([moves[i:i+h] for i in range(n)]),
            "future_valid_u8": np.ones((n,h), dtype=np.uint8),
            "planes_t_u8": np.stack(planes[:n]),
            "planes_future_u8": np.stack([planes[i+1:i+h+1] for i in range(n)])}


def sliced(data, start, end):
    result = {k: v[start:end] if v.ndim else v for k,v in data.items()}
    result["batch_size"] = np.array(end-start)
    return result


def test_recover_repetition_and_future_without_changing_source():
    data = payload()
    original = copy.deepcopy(data)
    result = audit_payloads([data])
    assert result["target_rows"] == 8
    assert result["future_targets"] == 24
    assert result["target_rows_with_current_repetition"] == 1
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


def test_shard_boundary_requires_prefix():
    data = payload()
    prefix, target = sliced(data, 0, 4), sliced(data, 4, 8)
    assert audit_payloads([prefix, target], context_count=1)["target_rows"] == 4
    with pytest.raises(ValueError, match="Missing initial"):
        audit_payloads([target])


@pytest.mark.parametrize("mutation, match", [
    (lambda d: d["ply"].__setitem__(3, 5), "nonconsecutive"),
    (lambda d: d["actions_uci"].__setitem__((1,0), "e7e5"), "Conflicting"),
    (lambda d: d["fen_t"].__setitem__(1, chess.STARTING_FEN), "FEN mismatch"),
    (lambda d: d["result"].__setitem__(3, "0-1"), "Result changed"),
    (lambda d: d["future_valid_u8"].__setitem__((1,0), 0), "complete Hero"),
])
def test_bad_source_fails_closed(mutation, match):
    data = payload()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        audit_payloads([data])


def test_does_not_use_placeholder_headers_as_game_keys():
    data = payload()
    result = audit_payloads([data, data])
    assert result["reconstructed_games_touching_targets"] == 2
    assert result["distinct_header_game_ids"] == 1


def test_unrooted_context_is_ignored_but_target_is_not():
    data = payload()
    result = audit_payloads([sliced(data, 4, 8), data], context_count=1)
    assert result["ignored_unrooted_context_rows"] == 4
    assert result["target_rows"] == 8


def test_changed_nonhistory_planes_fail():
    data = payload()
    data["planes_future_u8"][0,0,0,0,0] ^= 1
    with pytest.raises(AssertionError, match="Non-history future"):
        audit_payloads([data])


def test_profile_identifies_missing_windows_and_nonstandard_roots():
    data = payload()
    data["ply"][3] = 5
    data["fen_t"][0] = "rnkbrqbn/pppppppp/8/8/8/8/PPPPPPPP/RNKBRQBN w KQkq - 0 1"
    result = profile_payload(data)
    assert result["nonstandard_root_rows"][0]["row"] == 0
    assert result["within_shard_ply_gaps"][0] == {"row": 3, "previous_ply": 2, "ply": 5}
    with pytest.raises(ValueError, match="nonconsecutive"):
        audit_payloads([data])


def test_unrelated_context_game_cannot_block_target_game():
    bad = payload()
    bad["ply"][3] = 5
    assert audit_payloads([bad, payload()], context_count=1)["passed"]
