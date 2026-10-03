"""Recompute GPU-001 timing/cost and describe its separate instrumented traces."""
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from research.gpu_training_benchmark import summarize, TIMINGS
from research.interpretability.remote_cost import GPU_DOLLARS_PER_SECOND, CPU_DOLLARS_PER_CORE_SECOND, MEMORY_DOLLARS_PER_GIB_SECOND


def merged_duration(intervals):
    total = 0.0; left = right = None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start > right:
            total += right-left; left, right = start, end
        else:
            right = max(right, end)
    return total + (right-left if left is not None else 0)


def trace_summary(path):
    with gzip.open(path, 'rt') as stream:
        events = json.load(stream)['traceEvents']
    device = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    kernels = [e for e in device if e['cat'] == 'kernel']
    if not kernels:
        raise ValueError('Profiler captured no device kernels')
    span = max(e['ts']+e['dur'] for e in device)-min(e['ts'] for e in device)
    busy = merged_duration([(e['ts'], e['ts']+e['dur']) for e in device])
    counts, durations = Counter(), defaultdict(float)
    for e in kernels:
        counts[e['name']] += 1; durations[e['name']] += e['dur']
    return {'kernel_count':len(kernels), 'device_span_seconds':span/1e6,
            'device_active_union_seconds':busy/1e6, 'device_idle_fraction_within_span':1-busy/span,
            'median_kernel_us':statistics.median(e['dur'] for e in kernels),
            'kernel_duration_sum_seconds':sum(e['dur'] for e in kernels)/1e6,
            'top_kernels': [{'name':name,'count':counts[name],'seconds':durations[name]/1e6}
                            for name in sorted(durations,key=durations.get,reverse=True)[:25]],
            'caveat':'Instrumented single update only; profiler perturbs CPU launch timing. Device idle intervals do not uniquely distinguish CPU overhead, synchronization or other scheduling causes.'}


def analyze(root):
    root = Path(root); reports = {}; rows = {}; rate = {}; traces = {}
    for gpu in ('L40S','H100'):
        folder=root/gpu
        reports[gpu]=json.loads((folder/'result.json').read_text())
        for name, artifact in reports[gpu]['artifacts'].items():
            p=folder/name
            if p.exists():
                assert p.stat().st_size == artifact['size_bytes']
                assert hashlib.sha256(p.read_bytes()).hexdigest() == artifact['sha256']
        rows[gpu]=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        assert summarize(rows[gpu]) == reports[gpu]['measurement']
        rate[gpu]=float(GPU_DOLLARS_PER_SECOND[gpu]+4*CPU_DOLLARS_PER_CORE_SECOND+32*MEMORY_DOLLARS_PER_GIB_SECOND)
        traces[gpu]=trace_summary(folder/'profile_trace.json.gz')
    assert reports['L40S']['protocol'] == reports['H100']['protocol']
    n=reports['L40S']['protocol']['analysis']['remaining_original_epoch_updates']
    result_rows=[]
    for gpu in ('L40S','H100'):
        measure=reports[gpu]['measurement']; seconds=measure['seconds_per_update']
        result_rows.append({'gpu':gpu, 'device_name':reports[gpu]['device_name'], **measure,
            'declared_resource_dollars_per_hour':rate[gpu]*3600,
            'projected_remaining_train_loop_hours':n*seconds/3600,
            'projected_remaining_train_loop_dollars':n*seconds*rate[gpu],
            'dollars_per_million_examples':1e6/1024*seconds*rate[gpu],
            'trace_summary':traces[gpu]})
    slow,fast=result_rows
    timing_deltas={k:slow['mean_timings'][k]-fast['mean_timings'][k] for k in TIMINGS}
    return {'experiment_id':'GPU-001','rows':result_rows,
            'h100_speedup':slow['seconds_per_update']/fast['seconds_per_update'],
            'h100_cost_ratio':fast['projected_remaining_train_loop_dollars']/slow['projected_remaining_train_loop_dollars'],
            'break_even_speedup':rate['H100']/rate['L40S'],
            'component_seconds_saved_per_update':timing_deltas,
            'first_update_diagnostics':{g:{k:rows[g][0][k] for k in ('loss','dfm_ce_loss','jepa_positive_loss','gradient_global_norm')}
                                        for g in rows},
            'last_unprofiled_update_diagnostics':{g:{k:rows[g][-2][k] for k in ('loss','dfm_ce_loss','jepa_positive_loss','gradient_global_norm')}
                                                  for g in rows},
            'limits':['One container per GPU, 40 warmed updates each, not a fleet-level confidence estimate.',
                      'Identical checkpoint, code, batches, recipe and seeds; BF16 kernels may differ across hardware.',
                      'Projection excludes data reconstruction, cold start/compilation, validation, checkpointing and interruptions.',
                      'Profiled update is excluded from throughput. CUDA event intervals include inter-launch device idle time; profiler GPU activity is not a direct bandwidth/occupancy measurement.']}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=analyze(args.root)
    with (args.root/'summary.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(result,indent=2))
