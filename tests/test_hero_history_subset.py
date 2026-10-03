import numpy as np
import pytest

from research.analysis.audit_hero_history_subset import verified_subset
from research.hero_history_data import reconstruct_shard, UnresolvedHistory, write_npz_once
from tests.test_hero_history_recovery import payload


def source_with_gap(tmp_path):
    source = payload()
    # Three-ply windows: row 0 supplies plies 0..2, the next row starts at 5.
    # The preceding window is safe; the later windows need missing moves 3..4.
    rows = [0, 5, 6, 7]
    data = {k: np.concatenate((v[rows], v)) if v.ndim else v.copy()
            for k, v in source.items()}
    data.update(batch_size=np.array(12), plane_codec=np.array("uint8"),
                source_uri=np.array("gs://b/train/chunk_000000.npz"))
    path = tmp_path/"chunk_000000.npz"
    write_npz_once(path, data)
    return path, data


def test_only_unverified_suffix_is_dropped(tmp_path):
    path, source = source_with_gap(tmp_path)
    with pytest.raises(UnresolvedHistory):
        reconstruct_shard(path)
    rebuilt, audit = verified_subset(path)
    assert audit["dropped_rows"] == [1, 2, 3]
    assert audit["rows"] == [0, 4, 5, 6, 7, 8, 9, 10, 11]
    assert audit["examples"] == 9
    np.testing.assert_array_equal(rebuilt["actions_uci"], source["actions_uci"][audit["rows"]])
    assert np.any(rebuilt["planes_future_u8"][:, :, 13:104])


def test_all_requested_rows_can_be_unrecoverable(tmp_path):
    path, _ = source_with_gap(tmp_path)
    rebuilt, audit = verified_subset(path, rows=[1, 3])
    assert rebuilt is None
    assert audit["dropped_rows"] == [1, 3]


def test_bad_records_are_not_silently_filtered(tmp_path):
    path, data = source_with_gap(tmp_path)
    data["planes_t_u8"][4, 0, 0, 0] ^= 1
    other = tmp_path/"chunk_000001.npz"
    write_npz_once(other, data)
    with pytest.raises(AssertionError, match="Non-history input changed"):
        verified_subset(other)
