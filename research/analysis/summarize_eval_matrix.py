"""Verify frozen arena receipts and summarize descriptive development results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize(protocol_path: Path) -> dict:
    root = protocol_path.parent
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    receipt = json.loads((root / 'runs.json').read_text())
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    require(receipt['protocol_sha256'] == protocol_sha, 'Frozen protocol changed')
    rows = []
    for phase in ('smoke', 'development'):
        allocation = protocol['openings'][phase]
        for edge in protocol[f'{phase}_edges']:
            edge_id = edge['id']
            run = receipt['runs'].get(edge_id, {})
            require(run.get('status') == 'complete', f'{edge_id}: incomplete evaluation')
            result = run['result']
            state_path = root / 'results' / edge_id / 'state.json'
            raw = state_path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == result['state_sha256'],
                    f'{edge_id}: state checksum differs')
            require(json.loads(state_path.with_name('summary.json').read_text()) == result,
                    f'{edge_id}: summary differs from receipt')
            state = json.loads(raw)
            require(state['status'] == 'complete', f'{edge_id}: nonterminal state')
            contract = state['contract']
            aggregate = state['aggregate']
            require(aggregate['pair_count'] == allocation['pairs'], f'{edge_id}: pair count')
            require(aggregate['game_count'] == 2 * allocation['pairs'], f'{edge_id}: game count')
            require(contract['run']['opening_start_index'] == allocation['start'],
                    f'{edge_id}: opening allocation')
            for key in ('pool_sha256', 'pool_file_sha256', 'history_manifest_sha256',
                        'history_file_sha256'):
                require(contract['tier'][key] == protocol['openings'][key],
                        f'{edge_id}: opening identity {key}')
            for key in ('additional_ply_cap', 'block_pairs', 'policy_batch_size_cap',
                        'policy_timeout_seconds'):
                require(contract['run'][key] == protocol['game_contract'][key],
                        f'{edge_id}: game contract {key}')
            sides = {}
            for side in ('candidate', 'opponent'):
                identity = protocol[edge[side]]
                descriptor = contract['models'][side]
                if 'plane_history_mode' in identity:
                    require(descriptor['plane_history_mode'] == identity['plane_history_mode'],
                            f'{edge_id}: {side} input-history contract')
                source = descriptor.get('state', descriptor.get('bt4_checkpoint'))
                require(source['sha256'] == identity['state_sha256'],
                        f'{edge_id}: {side} model identity')
                require(result[f'{side}_state_sha256'] == identity['state_sha256'],
                        f'{edge_id}: {side} summary identity')
                require(result[f'{side}_policy_mode'] == edge[f'{side}_mode'],
                        f'{edge_id}: {side} mode')
                require(contract['run'][f'{side}_refinement_passes'] == edge[f'{side}_passes'],
                        f'{edge_id}: {side} passes')
                stats = aggregate['models'][descriptor['model_id']]
                require(stats['games'] == aggregate['game_count'], f'{edge_id}: {side} games')
                require(stats['wins'] + stats['draws'] + stats['losses'] == stats['games'],
                        f'{edge_id}: {side} outcome total')
                require(math.isclose(stats['points'], stats['wins'] + stats['draws'] / 2),
                        f'{edge_id}: {side} points')
                coverage = stats['coverage']
                sides[side] = {
                    'model': edge[side], 'mode': edge[f'{side}_mode'],
                    'passes': edge[f'{side}_passes'],
                    'plane_history_mode': descriptor.get('plane_history_mode'),
                    'wins': stats['wins'], 'draws': stats['draws'], 'losses': stats['losses'],
                    'positions': coverage['positions'], 'policy_calls': coverage['calls'],
                    'mean_policy_call_seconds': coverage['mean_call_seconds'],
                    'policy_positions_per_second': coverage['positions'] / coverage['call_seconds'],
                    'incomplete_coverage_positions': coverage['incomplete_coverage_positions'],
                }
            candidate = sides['candidate']
            score = (candidate['wins'] + candidate['draws'] / 2) / aggregate['game_count']
            require(math.isclose(score, result['pair_score']), f'{edge_id}: score differs')
            require(aggregate['pentanomial'] == result['pentanomial'], f'{edge_id}: pair counts')
            require(aggregate['pair_aware_logistic_interval'] == result['pair_aware_logistic_interval'],
                    f'{edge_id}: interval differs')
            rows.append({
                'id': edge_id, 'phase': phase, 'role': edge.get('role', 'plumbing_only'),
                **sides, 'games': aggregate['game_count'], 'pairs': aggregate['pair_count'],
                'score': score, 'pentanomial': aggregate['pentanomial']['counts'],
                'pair_aware_interval': aggregate['pair_aware_logistic_interval'],
                'fault_counts': aggregate['fault_counts'],
                'cap_draw_count': aggregate['cap_draw_count'],
                'cap_draw_rate': aggregate['cap_draw_rate'],
                'gameplay_wall_seconds': aggregate['gameplay_wall_seconds'],
                'state_sha256': result['state_sha256'],
                'state_path': str(state_path), 'app_id': run['app_id'],
            })
    primary = [row for row in rows if row['role'] == 'primary']
    require(len(primary) == 2, 'Expected the two frozen primary comparisons')
    failed = []
    for row in primary:
        if row['score'] <= 0.5:
            failed.append(f"{row['id']}: primary development score does not exceed 0.5")
        if any(row['fault_counts'].values()):
            failed.append(f"{row['id']}: policy faults")
        if any(row[side]['incomplete_coverage_positions'] for side in ('candidate', 'opponent')):
            failed.append(f"{row['id']}: incomplete legal coverage")
        if row['cap_draw_rate'] > 0.05:
            failed.append(f"{row['id']}: excessive cap draws")
    return {
        'schema_version': 'chess-dfm-searchless-development-summary-v1',
        'experiment_id': protocol['experiment_id'], 'protocol_sha256': protocol_sha,
        'status': 'smoke_and_development_complete',
        'development_games': sum(row['games'] for row in rows if row['phase'] == 'development'),
        'smoke_games': sum(row['games'] for row in rows if row['phase'] == 'smoke'),
        'comparisons': rows,
        'confirmation_entry_passed': not failed,
        'confirmation_entry_failures': failed,
        'candidate_promoted': False,
        'decision': ('confirmation_design_required_before_more_games' if not failed
                     else 'do_not_run_confirmation_or_matched_search_for_this_candidate'),
        'limitations': [
            'Development-only opening allocation; no held-out confirmation or absolute Elo claim.',
            'Intervals are marginal 95% bounded-pair Hoeffding intervals, not simultaneous across the matrix.',
            'Logistic Elo is descriptive and relative to the opponent and this opening pool; infinite endpoints are clipped.',
            'Policy timings use padded batches of 16 on L4, exclude loading and compilation, and are not single-position latency.',
            'The opening-prefix overlap audit does not certify full source-game disjointness.',
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.protocol)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'comparisons'}))


if __name__ == '__main__':
    main()
