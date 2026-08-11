from __future__ import annotations

import io
import struct

import numpy as np

from chess_dfm_jax.data.leela import (
    V6_RECORD_SIZE,
    iter_records,
    record_to_input_planes,
)


def _v6_record_bytes() -> bytes:
    probabilities = np.full((1858,), -1.0, dtype="<f4")
    probabilities[17] = 0.6
    probabilities[42] = 0.4
    planes = np.zeros((104,), dtype="<u8")
    planes[0] = 1
    values = (
        0.2,
        0.3,
        0.5,
        0.4,
        12.0,
        11.0,
        20.0,
        -1.0,
        0.0,
        0.25,
        0.5,
        13.0,
        0.1,
        0.6,
        14.0,
    )
    payload = b"".join(
        (
            struct.pack("<II", 6, 1),
            probabilities.tobytes(),
            planes.tobytes(),
            bytes((1, 0, 1, 0, 1, 23, 0x20, 0)),
            struct.pack(
                "<15fIHHfI",
                *values,
                321,
                17,
                42,
                0.125,
                7,
            ),
        )
    )
    assert len(payload) == V6_RECORD_SIZE
    return payload


def test_iter_records_uses_official_v6_field_semantics() -> None:
    (record,) = tuple(
        iter_records(
            io.BytesIO(_v6_record_bytes()),
            include_probabilities=True,
        )
    )

    assert record.version == 6
    assert record.input_format == 1
    assert record.side_to_move == 1
    assert record.castling == (1, 0, 1, 0)
    assert record.played_idx == 17
    assert record.best_idx == 42
    assert record.visits == 321
    assert record.reserved == 7
    assert np.isclose(record.root_q, 0.2)
    assert np.isclose(record.best_q, 0.3)
    assert np.isclose(record.root_d, 0.5)
    assert np.isclose(record.played_q, 0.25)
    assert np.allclose(record.wdl, (0.35, 0.5, 0.15))
    assert record.probabilities is not None
    assert np.flatnonzero(record.probabilities >= 0).tolist() == [17, 42]


def test_record_to_input_planes_materializes_classical_aux_planes() -> None:
    record = next(iter_records(io.BytesIO(_v6_record_bytes())))
    planes = record_to_input_planes(record)

    assert planes.shape == (112, 8, 8)
    assert planes.dtype == np.float32
    assert planes[0, 0, 7] == 1.0
    assert planes[0].sum() == 1.0
    assert np.all(planes[104] == 1.0)
    assert np.all(planes[105] == 0.0)
    assert np.all(planes[106] == 1.0)
    assert np.all(planes[107] == 0.0)
    assert np.all(planes[108] == 1.0)
    assert np.all(planes[109] == 23.0)
    assert np.all(planes[110] == 0.0)
    assert np.all(planes[111] == 1.0)
