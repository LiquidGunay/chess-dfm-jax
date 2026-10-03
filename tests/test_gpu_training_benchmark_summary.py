import gzip
import json

import pytest
from research.summarize_gpu_training_benchmark import merged_duration, trace_summary


def test_device_activity_is_union_not_sum():
    assert merged_duration([]) == 0
    assert merged_duration([(5,10),(0,6),(12,14)]) == 12


def test_trace_uses_device_events_and_discloses_instrumentation(tmp_path):
    path=tmp_path/'trace.json.gz'
    events=[{'ph':'X','cat':'kernel','name':'matmul','ts':0,'dur':6},
            {'ph':'X','cat':'kernel','name':'matmul','ts':5,'dur':5},
            {'ph':'X','cat':'kernel','name':'elementwise','ts':12,'dur':2},
            {'ph':'X','cat':'cpu_op','name':'aten::mm','ts':0,'dur':100}]
    with gzip.open(path,'wt') as stream: json.dump({'traceEvents':events},stream)
    result=trace_summary(path)
    assert result['kernel_count'] == 3
    assert result['device_active_union_seconds'] == 12/1e6
    assert result['device_span_seconds'] == 14/1e6
    assert result['device_idle_fraction_within_span'] == pytest.approx(1/7)
    assert result['top_kernels'][0]['count'] == 2
