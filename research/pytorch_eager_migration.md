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
- the accepted balanced per-example K=1-of-8 target sampling, all eight
  recurrent JEPA horizons, projector depth two, DFM depth four, one-block
  future-gradient tail, and eight-pass inference;
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

## Execution evidence, 2026-07-23

The one-file eager implementation, strict 455-leaf source map, custom
Muon/AdamW update, stochastic-choice materialization, model-only checkpoint
format, guarded trainer, and focused tests are implemented. CPU FP32 batch-2
parity passes for all named intermediates and every loss component. The worst
tensor relative-L2 error is `2.72e-5`; the total-loss error is `1.48e-4`
absolute and `2.00e-5` relative. Two consecutive optimizer updates match
Optax, including the accepted partition and schedule. A NaN in any gradient
skips the entire update.

Representative FP32 reverse-mode parity also passes. The production JAX and
PyTorch classes use identical weights for a two-layer, QK-normalized/XSA
projector and a two-layer clipped recurrent JEPA transition. All 21 parameter
leaves and every input/condition cotangent pass `rtol=5e-4, atol=3e-5`; the
custom SIGReg value and latent gradient pass `rtol=2e-4, atol=2e-6`. The
combined focused suite passes 19 tests in 16.74 seconds.

The guarded A10G sweep measured physical batches 64, 128, 256, 512, and 768.
Batch 512 is selected:

- warm device throughput is `160.790` examples/s and sequential
  fetch-inclusive throughput is `150.663` examples/s;
- warm step time is `3.1843` seconds: `1.0644` forward, `1.9103` backward,
  `0.2096` optimizer, and less than `0.0001` measured in-step host overhead;
- peak HBM is `12,570,979,328` bytes allocated and `14,971,568,128` bytes
  reserved; peak process-group RSS is `1,992,486,912` bytes;
- 100 ms monitoring reports 100% median/p95 GPU utilization, 86.96% mean
  utilization including startup and input gaps, and 205.72/215.04 W
  median/p95 power; and
- batch 768 adds only a few percent throughput, consumes about 5.2 GB more
  allocated HBM, leaves unsafe headroom, and—because 768 does not divide the
  1,024-row shards—would silently omit 25% of every shard. Batch 1024 is
  projected beyond safe memory and was deliberately not attempted.

Deterministic depth-1 prefetch preserves all 10 profiled updates bit-for-bit
across total/component losses, policy metrics, gradient norm/clip scale, and
latent norms. It hides `0.24157` of `0.24158` warm preparation seconds, reduces
warm data wait to `13.1` microseconds, and raises warm end-to-end throughput
from `150.663` to `160.490` examples/s (`+6.52%`) while device throughput
remains `160.681` examples/s. Mean utilization including startup rises from
86.96% to 91.39%; peak process-group RSS remains `1,959,333,888` bytes. Freeze
prefetch depth 1 for the matched control.

The preregistered production-BF16 intermediate gate does **not** pass. In two
sequential guarded GPU processes, source-weight PyTorch versus JAX batch-2
comparison reaches `0.1142/0.0965` relative-L2 error for current/future BT4
tokens, `5.82%` DFM-CE error, and `0.0522` absolute legality-loss error. CPU
BF16 reproduces the same effect, ruling out a GPU-only defect. A 16-stage
trace starts at `0.00757` after the embedding, drifts to `0.03322` after layer
13, then jumps to `0.10626` after layer 14. The recovered final block has an
FFN layer-normalization scale maximum of `6.125` versus at most about `1.2`
for the preceding blocks, amplifying normal cross-framework BF16 rounding.
Matching JAX's primitive activation structure worsens the endpoint to
`0.12594`; computing only the final block in FP32 also worsens it to
`0.11068`. Both changes are rejected.

The model-only checkpoint ABI now has an exact write/restore/pure-tree test,
including bit-preserving BF16 conversion for NNX. The focused migration suite
passes 21 tests. The frozen PyTorch validator encodes all eight future
horizons even though training samples balanced K=1, and reports the historical
per-horizon policy, legality, feature-variance/effective-rank, trivial-baseline,
and action-shuffle diagnostics. A guarded source smoke at batch 64 completes
in 2.367 seconds with 3.152 GB peak allocated HBM and 1.878 GB peak
process-group RSS. Every scalar is finite; the prediction beats zero,
identity, target-shuffled, and action-shuffled controls at every horizon.
This smoke writes no checkpoint.

This evidence amends, rather than silently relaxes, correctness gate 3:

1. FP32 remains the exact formula, source-map, representative-gradient, and
   optimizer oracle.
2. BF16 is a documented runtime-specific numerical trajectory. Its failed
   tight-intermediate result remains recorded as a known difference and is
   not relabeled a pass.
3. Before scientific use, the unchanged PyTorch fixed-time control must pass
   the frozen validation/latent gates in PyTorch, then its exported checkpoint
   must pass the same frozen validation and eight-pass inference gates in JAX.
   Candidate promotion must survive that JAX checkpoint cross-check as well.
4. A selected architecture may return to JAX for a fresh hero run; exact
   continuation of a BF16 PyTorch optimization trajectory in JAX is not
   claimed.

`PYTORCH_AUTORESEARCH_READY` remains false. The remaining gates are
the real-control checkpoint's strict JAX round trip and the matched fixed-time
control plus full two-pool JAX validation/inference cross-check.
