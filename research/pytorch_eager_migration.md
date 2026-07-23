# PyTorch eager migration contract

## Decision

Use ordinary eager PyTorch for local A10G autoresearch. The purpose is to
recover fast, reliable architecture iteration under the existing 6.75 GiB
host-RSS compiler guard. This is an execution-framework change, not a
scientific experiment. Do not use `torch.compile` in the initial baseline.
Keep the JAX implementation frozen as the numerical and artifact oracle, and
return to it for long hero runs after an architecture is selected.

## Immutable scientific contract

The first PyTorch control must preserve:

- the checksum-pinned update-265000 model-only initialization and a fresh
  optimizer;
- the accepted K=2-of-8 target sampling, all eight recurrent JEPA horizons,
  projector depth two, DFM depth four, one-block future-gradient tail, and
  eight-pass inference;
- objective coefficients `norm=0`, `target_sigreg=5.76`,
  `pred_sigreg=1.0`, SIGReg V-statistic size 64, and legality coefficient 2;
- main and BT4 learning rates `3e-5` and `1e-6`, the 400-to-1200 update cosine
  schedule with floor 0.1, merged global clipping, and the accepted
  Muon/AdamW parameter partition;
- fixed global-permutation train/validation schedules, fixed validation batch
  size 64, metric definitions, promotion gates, and arena FENs; and
- sparse checkpoint policy, lineage metadata, and every artifact/cache path
  under `/mountpoint/.exp`.

Physical training batch size is deliberately not immutable. Select it only
after the guarded A10G sweep, then freeze it for matched control/candidate
comparisons. Effective examples and stochastic samples must be reported
explicitly.

## One-file boundary

`research/train_torch.py` is the readable experimental surface. It owns the
production PyTorch model, objective, optimizer construction, update, trainer,
metrics, and CLI. It may import audited immutable data/checkpoint utilities,
but it must not depend on an external TransformerLens or upstream repository
at runtime. Migration-only parity tests may use the frozen JAX path as an
oracle.

## Correctness gates

All gates fail closed:

1. Map every source leaf exactly and emit path, shape, dtype, byte-count, and
   checksum manifests. Reject missing, duplicate, or unused leaves.
2. On CPU FP32, match deterministic encoder, projector, DFM logits, recurrent
   JEPA predictions, and all loss components on fixed tiny inputs.
3. At production BF16 boundaries, document tolerances and compare the same
   intermediates. Do not hide differences inside a total-loss tolerance.
4. Materialize stochastic choices outside either framework: data slots,
   K-of-8 target indices, recurrent noise/timesteps, and any masking. Feed the
   same choices to both implementations.
5. Match representative full parameter gradients, global norm/clipping,
   optimizer partition membership, learning-rate schedule, and one- and
   two-update states. Separately identify unavoidable optimizer arithmetic
   differences before accepting them.
6. Save a sparse PyTorch state, restore it exactly, and export a strict JAX
   model-state round trip with no silent casts or dropped leaves.
7. Run the exported model through the frozen JAX validation and eight-pass
   inference paths and require metric/logit/move parity at declared
   tolerances.

## Guarded A10G gates

Run CUDA only via `research/run_gpu.sh`, with its exclusive lock, two-CPU
limit, 50 ms polling, 8/3 GiB host-memory thresholds, 6.75 GiB process-group
RSS ceiling, and 30 GiB disk reserve.

1. One forward/backward/update smoke, no checkpoint.
2. Profile a representative steady-state window: examples/s, step-time
   breakdown, peak HBM, host RSS, data wait, and synchronization.
3. Sweep physical batch sizes until throughput plateaus or safe HBM/RSS
   headroom is lost. Include gradient accumulation only if it improves the
   relevant fixed-time objective.
4. Run and repeat the accepted fixed-time control with the frozen validation
   pools. Record every loss component per evaluation and enough step-level
   statistics to diagnose scaling.

Set `PYTORCH_AUTORESEARCH_READY = True` only after these gates and the JAX
evaluation/inference cross-check pass. Later, profile before considering
regional `torch.compile`; a compiled variant is acceptable only if warm
throughput gain pays back compile time within a typical 30-minute experiment
and remains inside the same guard.
