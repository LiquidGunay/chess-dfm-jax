import gzip
import json

import pytest

from research.diagnose_gpu_training_benchmark import kernel_class, kernel_mix


def test_kernel_classes_do_not_double_count_copy_or_attention():
    assert kernel_class('unrolled_elementwise_direct_copy') == 'copy'
    assert kernel_class('fmha_gemm') == 'attention'
    assert kernel_class('sm90_xmma_gemm_bf16') == 'matmul'
    assert kernel_class('triton_') == 'triton_unclassified'


def test_mix_uses_kernel_durations_not_cpu_or_memcpy_events(tmp_path):
    path = tmp_path/'trace.gz'
    events = [{'ph':'X','cat':cat,'name':name,'dur':duration}
              for cat,name,duration in [('kernel','gemm',2),('kernel','elementwise',3),
                                        ('cpu_op','aten::mm',100),('gpu_memcpy','copy',100)]]
    with gzip.open(path, 'wt') as stream:
        json.dump({'traceEvents':events}, stream)
    result = kernel_mix(path)
    assert sum(row['kernels'] for row in result.values()) == 2
    assert result['matmul']['fraction_of_summed_kernel_duration'] == pytest.approx(.4)
