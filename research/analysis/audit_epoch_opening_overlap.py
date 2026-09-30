"""Audit exact ply-12 action-prefix overlap, not whole-game identity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from chess_dfm_jax.policy import encode_lc0_canonical_1858
from research.arena import ArenaGameSpec
from research.play_arena import ArenaOpeningHistory, _replay_opening_history


def opening_prefix(entry: dict) -> bytes:
    index = entry['opening_index']
    spec = ArenaGameSpec(str(index), index, 0, entry['fen'], 'candidate', 'baseline')
    board, _ = _replay_opening_history(spec, ArenaOpeningHistory(str(index), tuple(entry['history_fens'])))
    initial = chess.Board()
    actions = []
    for move in board.move_stack:
        actions.append(encode_lc0_canonical_1858(initial, move))
        initial.push(move)
    if len(actions) != 12:
        raise ValueError('Expected a complete ply-12 opening')
    return np.asarray(actions, dtype='<u2').tobytes()


def training_prefixes(root: Path) -> tuple[set[bytes], dict, str]:
    prefixes: set[bytes] = set()
    counts: Counter = Counter()
    fingerprint = hashlib.sha256()
    for path in sorted(root.glob('chunks/chunk-*/train/played_actions_u16.npy')):
        offsets_path = path.with_name('game_offsets_i64.npy')
        for source in (path, offsets_path):
            fingerprint.update(source.relative_to(root).as_posix().encode())
            fingerprint.update(hashlib.sha256(source.read_bytes()).digest())
        actions = np.load(path, mmap_mode='r', allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode='r', allow_pickle=False)
        counts['shards'] += 1
        for start, end in zip(offsets[:-1], offsets[1:]):
            counts['games'] += 1
            if end - start < 13:
                counts['too_short_for_nonterminal_ply12_root'] += 1
                continue
            prefix = np.asarray(actions[int(start):int(start) + 12], dtype='<u2')
            if np.any(prefix >= 1858):
                counts['incomplete_action_prefix'] += 1
                continue
            prefixes.add(prefix.tobytes())
            counts['valid_game_prefixes'] += 1
    counts['unique_prefixes'] = len(prefixes)
    return prefixes, dict(counts), fingerprint.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    history_path = repo / 'research/assets/arena/promotion-ply12-n2048-histories-v3.json'
    raw = history_path.read_bytes()
    protocol = json.loads(args.protocol.read_text())
    if hashlib.sha256(raw).hexdigest() != protocol['openings']['history_file_sha256']:
        raise RuntimeError('Frozen histories changed')
    prefixes, counts, fingerprint = training_prefixes(args.data_root)
    if counts['games'] != 73908 or counts['shards'] != 254:
        raise RuntimeError('Training-game inventory differs from the frozen corpus')
    entries = json.loads(raw)['entries']
    matches = [entry['opening_index'] for entry in entries if opening_prefix(entry) in prefixes]
    selected = set(matches)
    slices = {}
    for name in ('smoke', 'development', 'confirmation_reserve'):
        allocation = protocol['openings'][name]
        start = allocation['start']
        count = allocation.get('pairs', allocation.get('maximum_pairs'))
        overlap = sorted(selected.intersection(range(start, start + count)))
        slices[name] = {'start': start, 'count': count, 'matching_prefixes': len(overlap),
                        'matching_fraction': len(overlap) / count, 'matching_indices': overlap}
    result = {
        'schema_version': 'chess-dfm-opening-prefix-overlap-v1',
        'training': counts, 'training_action_offsets_fingerprint': fingerprint,
        'history_file_sha256': hashlib.sha256(raw).hexdigest(),
        'pool_openings': len(entries), 'matching_prefixes': len(matches),
        'matching_indices': matches, 'slices': slices,
        'limitations': [
            'A matching common opening prefix does not establish that the full source game was used in training.',
            'No matching prefix does not exclude a transposition or another occurrence of the same position in training.',
            'This check alone cannot certify source-game disjointness; keep development results descriptive.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'matching_indices'}, sort_keys=True))


if __name__ == '__main__':
    main()
