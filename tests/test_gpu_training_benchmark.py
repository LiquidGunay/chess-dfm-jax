import pytest
from research.gpu_training_benchmark import command, summarize, TIMINGS


def test_benchmark_is_disposable_exact_resume_with_excluded_profiler():
    args=command('data','a'*64,'raw','checkpoint','output')
    assert args[args.index('--resume-checkpoint')+1] == 'checkpoint'
    assert args[args.index('--steps')+1] == '553'
    assert args[args.index('--profile-update')+1] == '553'
    assert args[args.index('--max-checkpoints')+1] == '0'
    assert '--save-final' not in args and '--save-updates' not in args
    assert '--validation-updates' not in args
    assert args[args.index('--batch-size')+1] == '1024'


def records():
    return [dict(update=u,elapsed_seconds=1000+2*(u-500),optimizer_skipped_nonfinite=False,
                 gpu_peak_memory_allocated_bytes=20,gpu_peak_memory_reserved_bytes=30,
                 **{k:2.0 for k in TIMINGS}) for u in range(501,554)]


def test_elapsed_delta_excludes_cold_start_and_profile():
    rows=records()
    rows[-1]['elapsed_seconds']+=1000
    result=summarize(rows)
    assert result['seconds_per_update'] == 2
    assert result['examples_per_second'] == 512
    assert result['measured_examples'] == 40960
    assert result['ten_update_blocks_seconds_per_update'] == [2]*4


def test_rejects_missing_or_nonfinite_updates():
    rows=records()
    with pytest.raises(ValueError): summarize(rows[1:])
    rows[20]['optimizer_skipped_nonfinite']=True
    with pytest.raises(ValueError): summarize(rows)
