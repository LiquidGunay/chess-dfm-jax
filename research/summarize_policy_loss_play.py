"""Join EV-007 validation evidence to already frozen EV-005/006 game scores."""
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(root):
    root = Path(root)
    result = json.loads((root / 'result.json').read_text())
    evidence = root / 'position_metrics.npz'
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == result['position_metrics_sha256']
    lookup = {(r['pool'], r['model'], r['mode']): r for r in result['rows']}
    assert len(lookup) == len(result['rows']) == 14
    with np.load(evidence, allow_pickle=False) as vectors:
        for (pool, model, mode), row in lookup.items():
            key = f'{pool}__{model}__{mode}'
            ce, correct = vectors[key+'__ce'], vectors[key+'__correct']
            assert len(ce) == len(correct) == row['positions']
            assert np.isfinite(ce).all()
            assert float(ce.astype(np.float64).mean()) == row['legal_conditional_ce']
            assert float(correct.mean()) == row['legal_top1_accuracy']
    previous = json.loads((root.parent/'ev003_history_corrected_20261002/common_validation_result.json').read_text())
    old = next(r for r in previous['rows'] if r['model'] == 'hero1' and r['plane_history_mode'] == 'current_only_as_preprocessed')
    parity = []
    for mode in ('native_policy', 'dfm_one_pass'):
        current = lookup['sequential_validation', 'hero1', mode]
        prior = old['metrics'][mode]
        delta = current['legal_conditional_ce']-prior['legal_conditional_ce']
        assert abs(delta) < 1e-5
        assert current['legal_top1_accuracy'] == prior['legal_top1_accuracy']
        parity.append({'mode': mode, 'ce_difference_vs_ev003': delta, 'top1_identical': True})
    old_games = json.loads((root.parent/'ev005_full_history_bt4_20261003/development_summary.json').read_text())
    new_games = json.loads((root.parent/'ev006_hh002_play_20261003/development_summary.json').read_text())
    games = {}
    for mode, id_ in [('native_policy','policy_full'), ('dfm_one_pass','p1_full')]:
        row = next(r for r in old_games['rows'] if r['id'] == id_)
        games['hero1', mode] = {'score': row['score'], 'games': row['wins']+row['draws']+row['losses'],
                               'interval': row['interval'], 'source_experiment': 'EV-005', 'edge': id_}
    for mode, id_ in [('native_policy','policy_raw'), ('dfm_one_pass','p1_raw'), ('dfm_eight_pass','p8_raw')]:
        row = next(r for r in new_games['comparisons'] if r['id'] == id_)
        games['hh002', mode] = {'score': row['score'], 'games': row['games'],
                               'interval': row['pair_aware_interval'], 'source_experiment': 'EV-006', 'edge': id_}
    rows = []
    for model in ('raw_bt4', 'hero1', 'hh002'):
        for mode in (('native_policy',) if model == 'raw_bt4' else ('native_policy','dfm_one_pass','dfm_eight_pass')):
            rows.append({'model': model, 'mode': mode,
                         'validation': {pool: {k: lookup[pool,model,mode][k] for k in ('positions','legal_conditional_ce','legal_top1_accuracy')}
                                        for pool in ('hero_validation','sequential_validation')},
                         'versus_full_history_bt4': games.get((model,mode)),
                         'missing_play_reason': ('reference, not a self-play result' if model == 'raw_bt4' else
                                                 'no corresponding full-history BT4 match' if (model,mode) not in games else None)})
    rankings = {}
    for pool in ('hero_validation', 'sequential_validation'):
        observed = sorted(games, key=lambda key: games[key]['score'], reverse=True)
        loss_order = sorted(games, key=lambda key: lookup[pool,*key]['legal_conditional_ce'])
        rankings[pool] = {'observed_score_descending': observed, 'ce_ascending': loss_order,
                          'ordering_agrees': observed == loss_order,
                          'scope': 'Descriptive only, five learned policies with observed BT4 matches; no significance or causal claim'}
    direct = []
    for mode, id_ in [('native_policy','policy_hero1'), ('dfm_one_pass','p1_hero1'), ('dfm_eight_pass','p8_hero1')]:
        game = next(r for r in new_games['comparisons'] if r['id'] == id_)
        direct.append({'edge': id_, 'mode': mode, 'candidate': 'hh002', 'opponent': 'hero1',
                       'candidate_score': game['score'], 'interval': game['pair_aware_interval'],
                       'validation_differences_candidate_minus_opponent': {
                           pool: {metric: lookup[pool,'hh002',mode][metric]-lookup[pool,'hero1',mode][metric]
                                  for metric in ('legal_conditional_ce','legal_top1_accuracy')}
                           for pool in ('hero_validation','sequential_validation')}})
    return {'experiment_id': 'EV-007', 'rows': rows, 'ev003_reproduction': parity,
            'descriptive_rankings': rankings,
            'direct_history_versus_hero1_comparisons': direct,
            'hh002_eight_versus_one_play': next(r for r in new_games['comparisons'] if r['id'] == 'p8_vs_p1')['pair_aware_interval'],
            'validated_per_position_aggregations': 14,
            'limitations': result['protocol']['analysis']['limitations']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    with (args.root/'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write('\n')
    for row in summary['rows']:
        print(row['model'], row['mode'], row['validation'], row['versus_full_history_bt4'])
    print('REPRODUCTION', summary['ev003_reproduction'])
    print('RANKINGS', summary['descriptive_rankings'])
