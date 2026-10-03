"""GPU-001: matched disposable HH-002 resume and warm-window measurements."""
import statistics
import sys


def command(data, manifest_sha, raw, checkpoint, output):
    return [sys.executable, '-m', 'research.train_torch', 'train', '--recipe', 'hero',
            '--data-format', 'trajectory_v3_history', '--history-manifest-sha256', manifest_sha,
            '--data-root', str(data), '--raw-bt4-path', str(raw), '--output-dir', str(output),
            '--resume-checkpoint', str(checkpoint), '--batch-size', '1024', '--steps', '553',
            '--seed', '0', '--data-start', '0', '--threads', '2', '--remat-mode', 'bt4-projector',
            '--attention-impl', 'sdpa-all', '--compile-regions', 'fresh', '--prefetch-depth', '1',
            '--train-seconds', '0', '--log-every', '10', '--gpu-monitor-interval-ms', '100',
            '--profile-update', '553', '--max-checkpoints', '0']


TIMINGS = ('step_seconds', 'data_wait_seconds', 'data_prepare_seconds', 'transfer_seconds',
           'forward_cuda_seconds', 'backward_cuda_seconds', 'optimizer_cuda_seconds',
           'host_overhead_in_step_seconds')


def summarize(records):
    if [r['update'] for r in records] != list(range(501, 554)):
        raise ValueError('Expected the fixed 53-update resume window')
    if any(r['optimizer_skipped_nonfinite'] for r in records):
        raise ValueError('Nonfinite benchmark update')
    by_update = {r['update']: r for r in records}
    selected = [by_update[i] for i in range(511, 551)]
    elapsed = by_update[550]['elapsed_seconds']-by_update[510]['elapsed_seconds']
    return {'warmup_updates': list(range(501, 511)), 'measured_updates': list(range(511, 551)),
            'profiled_update': 553, 'measured_examples': 40*1024,
            'elapsed_seconds': elapsed, 'seconds_per_update': elapsed/40,
            'examples_per_second': 40*1024/elapsed,
            'mean_timings': {k: statistics.mean(r[k] for r in selected) for k in TIMINGS},
            'median_step_seconds': statistics.median(r['step_seconds'] for r in selected),
            'ten_update_blocks_seconds_per_update': [
                (by_update[end]['elapsed_seconds']-by_update[end-10]['elapsed_seconds'])/10
                for end in (520, 530, 540, 550)],
            'peak_allocated_bytes': max(r['gpu_peak_memory_allocated_bytes'] for r in selected),
            'peak_reserved_bytes': max(r['gpu_peak_memory_reserved_bytes'] for r in selected),
            'measurement': 'Elapsed-time difference includes input wait, transfer, training, metrics and logging; excludes compile warmup, profiler and checkpoint/validation I/O.'}
