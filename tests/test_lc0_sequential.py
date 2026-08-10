from __future__ import annotations

from pathlib import Path

import numpy as np

from chess_dfm_jax.data.lc0_sequential import (
    SequentialBatches,
    SequentialShard,
    records_to_sequential_game,
    save_sequential_shard,
)
from chess_dfm_jax.data.leela import TrainingRecord


def _record(
    marker: int,
    *,
    played: int,
    root_q: float,
    root_d: float,
    kingside_castling: bool = False,
) -> TrainingRecord:
    policy = np.full((1858,), -1.0, dtype=np.float32)
    policy[played] = 0.75
    policy[159] = 0.25
    planes = np.zeros((104,), dtype="<u8")
    planes[0] = np.uint64(1 << marker)
    return TrainingRecord(
        version=6,
        input_format=1,
        planes=planes,
        castling=(0, int(kingside_castling), 0, 0),
        side_to_move=marker % 2,
        rule50=marker,
        invariance_info=0,
        played_idx=played,
        best_idx=played,
        probabilities=policy,
        side_to_move_or_enpassant=marker % 2,
        root_q=root_q,
        best_q=root_q,
        root_d=root_d,
        best_d=root_d,
        root_m=10.0,
        best_m=9.0,
        plies_left=20.0,
        result_q=1.0,
        result_d=0.0,
        played_q=root_q,
        played_d=root_d,
        played_m=10.0,
        orig_q=root_q,
        orig_d=root_d,
        orig_m=10.0,
        visits=128,
        policy_kld=0.1,
        reserved=0,
    )


def test_sequential_shard_materializes_windows_without_crossing_games(
    tmp_path: Path,
) -> None:
    first = records_to_sequential_game(
        (
            _record(
                0,
                played=103,
                root_q=0.0,
                root_d=0.5,
                kingside_castling=True,
            ),
            _record(1, played=17, root_q=0.2, root_d=0.4),
            _record(2, played=18, root_q=-0.3, root_d=0.3),
        ),
        name="game-a.gz",
    )
    second = records_to_sequential_game(
        (
            _record(3, played=19, root_q=0.1, root_d=0.6),
            _record(4, played=20, root_q=-0.2, root_d=0.2),
        ),
        name="game-b.gz",
    )
    shard_path = tmp_path / "shard-00000"
    manifest = save_sequential_shard(
        (first, second),
        shard_path,
        split="train",
        shard_index=0,
    )
    assert manifest["position_count"] == 5
    assert manifest["trainable_start_count"] == 3

    shard = SequentialShard(shard_path)
    batch = shard.materialize(np.asarray((0, 1, 3)), horizon=2)

    assert batch["current_planes"].shape == (3, 112, 8, 8)
    assert batch["future_planes"].shape == (3, 2, 112, 8, 8)
    assert batch["future_valid"].tolist() == [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    assert batch["action_indices"].tolist() == [[102, 17], [17, 0], [19, 0]]
    assert batch["legal_count"][0, 0] == 2
    first_legal = batch["legal_idx"][0, 0, :2].tolist()
    assert first_legal == [102, 159]
    assert np.allclose(batch["wdl_targets"][0, 0], (0.4, 0.4, 0.2))
    assert np.allclose(batch["wdl_targets"][0, 1], (0.2, 0.3, 0.5))
    assert np.allclose(
        batch["current_wdl_target"],
        (
            (0.25, 0.5, 0.25),
            (0.4, 0.4, 0.2),
            (0.25, 0.6, 0.15),
        ),
    )
    assert np.allclose(batch["current_value_target"], (0.0, 0.2, 0.1))
    assert batch["value_targets"].tolist() == [
        [np.float32(0.2), np.float32(-0.3)],
        [np.float32(-0.3), np.float32(0.0)],
        [np.float32(-0.2), np.float32(0.0)],
    ]


def test_saved_policy_values_retain_probability_mass(tmp_path: Path) -> None:
    game = records_to_sequential_game(
        (
            _record(0, played=17, root_q=0.0, root_d=0.5),
            _record(1, played=18, root_q=0.0, root_d=0.5),
        ),
        name="game.gz",
    )
    shard_path = tmp_path / "shard"
    save_sequential_shard((game,), shard_path, split="validation", shard_index=0)
    shard = SequentialShard(shard_path)

    for position in range(shard.position_count):
        begin = int(shard.policy_offsets_i64[position])
        end = int(shard.policy_offsets_i64[position + 1])
        assert np.isclose(
            np.asarray(shard.policy_values_f16[begin:end], dtype=np.float32).sum(),
            1.0,
            atol=1e-3,
        )


def test_sequential_batches_are_stateless_and_skip_terminal_starts(
    tmp_path: Path,
) -> None:
    game = records_to_sequential_game(
        tuple(
            _record(
                marker,
                played=17 + marker,
                root_q=0.1 * marker,
                root_d=0.5,
            )
            for marker in range(5)
        ),
        name="long-game.gz",
    )
    shard_path = tmp_path / "chunks" / "chunk-00000" / "train"
    save_sequential_shard((game,), shard_path, split="train", shard_index=0)
    batches = SequentialBatches(
        tmp_path,
        split="train",
        batch_size=2,
        horizon=3,
        seed=7,
        shuffle_batches=False,
    )

    assert batches.steps_per_epoch == 2
    first = batches.batch_at(0)
    repeated = batches.batch_at(0)
    second = batches.batch_at(1)
    assert np.array_equal(first["action_indices"], repeated["action_indices"])
    assert first["action_indices"][:, 0].tolist() == [17, 18]
    assert second["action_indices"][:, 0].tolist() == [19, 20]
    assert second["future_valid"].tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
