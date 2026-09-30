import chess
import numpy as np

from chess_dfm_jax.policy import encode_lc0_canonical_1858
from research.analysis.audit_epoch_opening_overlap import opening_prefix, training_prefixes


def test_exact_history_prefix_matches_canonical_training_actions(tmp_path):
    board = chess.Board()
    fens = [board.fen(en_passant='legal')]
    actions = []
    for san in ('e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'd3', 'd6', 'Nc3', 'Nf6', 'Be3', 'O-O'):
        move = board.parse_san(san)
        actions.append(encode_lc0_canonical_1858(board, move))
        board.push(move)
        fens.append(board.fen(en_passant='legal'))
    entry = {'opening_index': 0, 'fen': fens[-1], 'history_fens': fens}
    path = tmp_path / 'chunks/chunk-00000/train'
    path.mkdir(parents=True)
    np.save(path / 'played_actions_u16.npy', np.array(actions + [65535], dtype=np.uint16))
    np.save(path / 'game_offsets_i64.npy', np.array([0, 13], dtype=np.int64))
    prefixes, counts, fingerprint = training_prefixes(tmp_path)
    assert opening_prefix(entry) in prefixes
    assert counts['valid_game_prefixes'] == 1
    assert len(fingerprint) == 64
