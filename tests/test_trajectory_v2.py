import io
import tempfile
from pathlib import Path

import chess
import chess.pgn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_dfm_jax.data.leela import LeelaChunkDataLoader  # noqa: E402
from chess_dfm_jax.data.gcs_cache import local_name_for_uri  # noqa: E402
from chess_dfm_jax.data.trajectory import (  # noqa: E402
    TrajectoryShard,
    build_synthetic_trajectory_shard,
    load_trajectory_shard,
    rollout_from_fen,
    validate_trajectory_shard,
)  # noqa: E402
from chess_dfm_jax.encoding import encode_board  # noqa: E402
from chess_dfm_jax.policy import policy_index_to_move  # noqa: E402
from scripts.process_tcec_to_gcs import slice_game  # noqa: E402


PGN_TEXT = """[Event "FIDE World Cup 2017"]
[Site "Tbilisi GEO"]
[Date "2017.09.09"]
[Round "3.1"]
[White "Carlsen,M"]
[Black "Bu Xiangzhi"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 h6 5. O-O d6 6. c3 g6 7. Re1 Bg7 8. h3 O-O
9. Nbd2 Re8 10. b4 a6 11. a4 Be6 12. Bxe6 Rxe6 13. Bb2 d5 14. b5 Ne7 15. c4 axb5
16. axb5 d4 17. Rxa8 Qxa8 18. Qb3 Nd7 19. Ra1 Qb8 20. Ba3 Nc8 21. c5 c6 22. Nc4
Na7 23. b6 Nxb6 24. cxb6 Nb5 25. Bb2 Re8 26. Ra5 Qd8 27. Rxb5 cxb5 28. Qxb5 Re6
29. Ba3 Kh7 30. Nfd2 h5 31. Nb3 Qg5 32. Bc1 Qe7 33. Nc5 Rc6 34. Ba3 Qg5 35. g3 h4
36. Kg2 hxg3 37. fxg3 Qh5 38. g4 Qg5 39. Nxb7 Rf6 40. Qb2 Qh4 41. Qe2 Bh6 42. Be7
Rf2+ 43. Qxf2 Qxe7 44. Nbd6 Bf4 45. b7 Qd8 46. Qb2 Qh4 47. b8=Q Qg3+ 48. Kf1 Qxd3+
49. Qe2 Qxh3+ 50. Qg2 Qd3+ 51. Kg1 Qd1+ 52. Qf1 Qxg4+ 53. Kf2 Qg3+ 54. Ke2 d3+
55. Kd1 Qg4+ 0-1
"""


def _load_samples(horizon: int = 4) -> list[dict[str, np.ndarray]]:
    game = chess.pgn.read_game(io.StringIO(PGN_TEXT))
    assert game is not None
    samples = list(slice_game(game, horizon=horizon))
    assert samples
    return samples


def test_tcec_slice_game_rollout_contract():
    sample = _load_samples(horizon=4)[0]
    assert sample["planes_future"].shape == (4, 112, 8, 8)
    assert sample["legal_masks"].shape == (4, 1858)
    assert sample["value_targets"].shape == (4,)
    assert sample["wdl_targets"].shape == (4, 3)
    assert sample["future_valid"].tolist() == [1.0, 1.0, 1.0, 1.0]

    rollout = rollout_from_fen(sample["fen_t"].item(), sample["actions"])
    np.testing.assert_allclose(sample["planes_future"], rollout["planes_future"])
    np.testing.assert_allclose(sample["legal_masks"], rollout["legal_masks"])

    board = chess.Board(sample["fen_t"].item())
    result = sample["result"].item()
    for horizon_idx, action_idx in enumerate(sample["actions"]):
        move = policy_index_to_move(int(action_idx), "lc0_1858")
        assert move in board.legal_moves
        assert sample["legal_masks"][horizon_idx, action_idx] == 1.0
        board.push(move)
        expected_planes = encode_board(board, []).astype(np.float32)
        np.testing.assert_allclose(sample["planes_future"][horizon_idx], expected_planes)
        expected_wdl = np.asarray((0.0, 0.0, 1.0) if board.turn == chess.WHITE else (1.0, 0.0, 0.0), dtype=np.float32)
        if result == "0-1":
            expected_wdl = np.asarray((0.0, 0.0, 1.0) if board.turn == chess.WHITE else (1.0, 0.0, 0.0), dtype=np.float32)
        elif result == "1-0":
            expected_wdl = np.asarray((1.0, 0.0, 0.0) if board.turn == chess.WHITE else (0.0, 0.0, 1.0), dtype=np.float32)
        else:
            expected_wdl = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        np.testing.assert_allclose(sample["wdl_targets"][horizon_idx], expected_wdl)
        np.testing.assert_allclose(
            sample["value_targets"][horizon_idx],
            expected_wdl[0] - expected_wdl[2],
)


def test_legacy_planes_target_adapter_marks_terminal_only():
    samples = _load_samples(horizon=3)
    sample = samples[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = Path(tmpdir) / "legacy_chunk.npz"
        np.savez_compressed(
            chunk_path,
            planes_t=np.stack([sample["planes_t"]], axis=0),
            legal_mask=np.stack([sample["legal_masks"][0]], axis=0),
            actions=np.stack([sample["actions"]], axis=0),
            planes_target=np.stack([sample["planes_future"][-1]], axis=0),
            value_target=np.asarray([sample["value_targets"][-1]], dtype=np.float32),
            wdl_target=np.stack([sample["wdl_targets"][-1]], axis=0),
        )

        shard = load_trajectory_shard(chunk_path)
        assert shard.schema_version == "trajectory-v1-adapted"
        assert shard.future_valid.tolist() == [[0.0, 0.0, 1.0]]
        np.testing.assert_allclose(shard.planes_future[0, -1], sample["planes_future"][-1])

        loader = LeelaChunkDataLoader([str(chunk_path)], batch_size=1, horizon=3)
        batch = next(iter(loader))
        assert batch["future_valid"].tolist() == [[0.0, 0.0, 1.0]]
        np.testing.assert_allclose(batch["next_planes"][0], sample["planes_future"][-1])


def test_trajectory_v2_validator_rejects_zero_filled_valid_horizon():
    shard = build_synthetic_trajectory_shard(batch_size=1, horizon=4)
    broken = TrajectoryShard(
        **{
            **shard.__dict__,
            "planes_future": shard.planes_future.copy(),
        }
    )
    broken.planes_future[0, 2] = 0.0

    with np.testing.assert_raises_regex(ValueError, "zero-filled valid horizon"):
        validate_trajectory_shard(broken)


def test_trajectory_v2_validator_rejects_illegal_recorded_action():
    shard = build_synthetic_trajectory_shard(batch_size=1, horizon=4)
    broken_masks = shard.legal_masks.copy()
    broken_masks[0, 0, shard.actions[0, 0]] = 0.0
    broken = TrajectoryShard(
        **{
            **shard.__dict__,
            "legal_masks": broken_masks,
        }
    )

    with np.testing.assert_raises_regex(ValueError, "Recorded action is illegal"):
        validate_trajectory_shard(broken)


def test_gcs_cache_local_name_avoids_split_collisions():
    train_name = local_name_for_uri("gs://bucket/data/train/chunk_000000.npz")
    val_name = local_name_for_uri("gs://bucket/data/val/chunk_000000.npz")
    assert train_name.endswith("chunk_000000.npz")
    assert val_name.endswith("chunk_000000.npz")
    assert train_name != val_name
