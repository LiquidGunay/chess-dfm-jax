"""Descriptive post-hoc kernel mix and historical sensitivity for GPU-001."""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def kernel_class(name):
    name = name.lower()
    # Mutually exclusive name-based groups, not hardware bandwidth counters.
    if 'fmha' in name or 'flash' in name:
        return 'attention'
    if 'gemm' in name or 'gemv' in name:
        return 'matmul'
    if 'copy' in name:
        return 'copy'
    if 'elementwise' in name:
        return 'elementwise'
    if 'reduce_kernel' in name:
        return 'reduction'
    if name.startswith('triton'):
        return 'triton_unclassified'
    return 'other'


def kernel_mix(path):
    with gzip.open(path, 'rt') as stream:
        events = json.load(stream)['traceEvents']
    counts, seconds = Counter(), defaultdict(float)
    for event in events:
        if event.get('ph') == 'X' and event.get('cat') == 'kernel':
            group = kernel_class(event['name'])
            counts[group] += 1
            seconds[group] += event['dur']/1e6
    total = sum(seconds.values())
    return {group:{'kernels':counts[group], 'seconds':seconds[group],
                   'fraction_of_summed_kernel_duration':seconds[group]/total}
            for group in sorted(counts)}


def diagnose(root, historical_metrics):
    summary = json.loads((root/'summary.json').read_text())
    current = {row['gpu']:row for row in summary['rows']}
    raw = historical_metrics.read_bytes()
    historical = {row['update']:row for row in map(json.loads, raw.decode().splitlines())}
    old_seconds = (historical[550]['elapsed_seconds']-historical[510]['elapsed_seconds'])/40
    old_rate = current['L40S']['declared_resource_dollars_per_hour']/3600
    h100_rate = current['H100']['declared_resource_dollars_per_hour']/3600
    return {
        'experiment_id':'GPU-001', 'analysis_type':'post_hoc_descriptive_diagnostics',
        'kernel_mix':{gpu:kernel_mix(root/gpu/'profile_trace.json.gz') for gpu in current},
        'historical_sensitivity':{
            'historical_metrics_sha256':hashlib.sha256(raw).hexdigest(),
            'historical_remote_path':'runs/hh002-hero-history-0fe540075e5361da6c25/metrics.jsonl',
            'volume':'chess-dfm-training-results', 'updates':[511,550],
            'historical_seconds_per_update':old_seconds,
            'mean_timings':{key:statistics.mean(historical[i][key] for i in range(511,551))
                            for key in current['L40S']['mean_timings']},
            'h100_speedup_over_historical':old_seconds/current['H100']['seconds_per_update'],
            'h100_cost_ratio_to_historical':current['H100']['seconds_per_update']*h100_rate/(old_seconds*old_rate),
            'caveat':'Historical, not a simultaneous randomized hardware control. Same update window but allocation/startup history differs; this is sensitivity, not a replacement for the frozen primary comparison.'},
        'limits':[
            'Kernel classes are descriptive name matches, not measured roofline/occupancy or a unique causal bottleneck attribution.',
            'Kernel duration shares exclude inter-kernel idle time and do not equal wall-clock time shares.',
            'One separate instrumented final update per GPU; unlike the warm training window it has no following-batch prefetch.',
            'Near-zero exposed data wait does not rule out CPU preprocessing contention with kernel dispatch.',
            'No additional GPU experiments or model/runtime changes were made for these diagnostics.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('historical_metrics', type=Path)
    args = parser.parse_args()
    result = diagnose(args.root, args.historical_metrics)
    with (args.root/'diagnostics.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2))
