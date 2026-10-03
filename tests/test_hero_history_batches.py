import json

import chess
import numpy as np
import pytest

from research.hero_history_batches import HistoryTrajectoryBatches
from research.hero_history_data import SCHEMA, sha256, write_npz_once, write_json_once
from research.input_contract import audit_full_history_batch, torch_hero_plane_history_mode, LC0_FULL_HISTORY
from chess_dfm_jax.encoding import encode_board


def make_dataset(root):
    records = []
    for index, values in enumerate(([1], [2,3,4])):
        n = len(values)
        current = np.zeros((n,112,8,8), dtype=np.uint8)
        current[:,0,0,0] = values
        payload = {"schema_version": np.array("trajectory-v3"), "plane_codec": np.array("uint8"),
                   "batch_size": np.array(n), "horizon": np.array(8),
                   "actions_u16": np.zeros((n,8),dtype=np.uint16),
                   "future_valid_u8": np.ones((n,8),dtype=np.uint8),
                   "legal_idx_u16": np.zeros((n,8,1),dtype=np.uint16),
                   "legal_count_u16": np.ones((n,8),dtype=np.uint16),
                   "planes_t_u8": current, "planes_future_u8": np.repeat(current[:,None],8,axis=1)}
        path = root/"validation"/f"chunk_{index:06d}.npz"
        write_npz_once(path,payload)
        records.append({"file": path.name, "examples": n, "size_bytes": path.stat().st_size,"sha256": sha256(path)})
    write_json_once(root/"manifest.json", {"schema_version": SCHEMA,"passed":True,
                                         "validation":records,"validation_examples":4,
                                         "original_training_inventory_sha256":"fixture"})
    return sha256(root/"manifest.json")


def test_original_order_across_repacked_validation_shards(tmp_path):
    digest = make_dataset(tmp_path)
    batches = HistoryTrajectoryBatches(tmp_path,split="validation",batch_size=2,horizon=8,
                                       manifest_sha256=digest,canonicalize=lambda value:value)
    assert batches.batch_at(0)["current_planes"][:,0,0,0].tolist() == [1,2]
    assert batches.batch_at(1)["current_planes"][:,0,0,0].tolist() == [3,4]
    with pytest.raises(IndexError):batches.batch_at(2)
    assert batches.provenance()["manifest_sha256"] == digest


def test_content_mismatch_fails_closed(tmp_path):
    digest = make_dataset(tmp_path)
    with pytest.raises(ValueError,match="manifest checksum"):
        HistoryTrajectoryBatches(tmp_path,split="validation",batch_size=2,horizon=8,
                                 manifest_sha256="0"*64,canonicalize=lambda value:value)
    batches = HistoryTrajectoryBatches(tmp_path,split="validation",batch_size=2,horizon=8,
                                       manifest_sha256=digest,canonicalize=lambda value:value)
    with (tmp_path/"validation/chunk_000000.npz").open("ab") as stream:stream.write(b"drift")
    with pytest.raises(ValueError,match="shard checksum"):batches.batch_at(0)


def test_chess960_initial_exception_requires_opt_in():
    board = chess.Board.from_chess960_pos(123)
    current = encode_board(board,[])
    history = [board.copy(stack=False)]
    board.push(next(iter(board.legal_moves)));history.append(board.copy(stack=False))
    future = encode_board(board,history)
    batch = {"current_planes":current[None],"future_planes":future[None,None],"future_valid":np.ones((1,1))}
    with pytest.raises(ValueError,match="noninitial"):audit_full_history_batch(batch)
    assert audit_full_history_batch(batch,allow_chess960_initial=True)["current_with_history"] == 0


def test_reconstructed_checkpoint_needs_complete_input_contract():
    record = {"args":{"data_format":"trajectory_v3_history"},
              "data":{"schema_version":SCHEMA,"manifest_sha256":"f"*64},
              "input_contract":{"plane_history_mode":LC0_FULL_HISTORY}}
    assert torch_hero_plane_history_mode(record) == LC0_FULL_HISTORY
    del record["data"]["manifest_sha256"]
    with pytest.raises(ValueError):torch_hero_plane_history_mode(record)


def test_local_training_without_preflight_fails_before_cuda():
    from research.train_torch import build_parser,train
    args=build_parser().parse_args(["train","--recipe","hero","--data-format","trajectory_v3_history","--output-dir","/tmp/not-started"])
    with pytest.raises(ValueError,match="immutable manifest"):train(args)
