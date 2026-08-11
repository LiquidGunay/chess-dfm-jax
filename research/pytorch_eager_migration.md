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

The first guarded batch-512 fixed-time control is complete. It performs 564
finite, unskipped updates over 288,768 examples in 1,801.31 training seconds,
or 160.310 examples/s end to end. Mean GPU utilization including startup is
98.57% with 100% p50 utilization and 215.23 W p50 power. Peak allocated and
reserved HBM are 12,570,979,328 and 14,971,568,128 bytes; guarded peak
process-group RSS is 2,498,887,680 bytes. All 564 per-update total/component
loss rows are retained. Comparing the first and last 64-update windows,
training DFM CE changes `3.47652 -> 3.42899`, JEPA positive loss
`0.29881 -> 0.21416`, accuracy `0.25073 -> 0.25367`, and legal mass
`0.87280 -> 0.87942`. Target/prediction norms settle at `29.90/28.97`;
there is no training-stream collapse claim because the frozen full-pool rank
gate is still authoritative.

Every subsequent PyTorch run writes every update to `metrics.jsonl` regardless
of console-log frequency and automatically emits `loss_summary.json`. The
summary stores the exact terminal metrics plus fixed first/last-64-update
means and deltas. These are plot-ready training diagnostics; matched two-pool
validation DFM CE remains the cross-experiment quality endpoint, and Elo
remains the strength endpoint.

Exactly one terminal checkpoint was written: 455 leaves, 705,987,352 tensor
bytes in a 706,033,120-byte safetensors file, SHA-256
`cd45d2ecb17d35439ec1ee54b3ad3ce02bcb5d2cdc4571e37bddcda15e34c49c`.
The strict CPU pure-tree audit passes with combined leaf checksum
`a3a99de695eb73b28a35eda75323a075713d0032debdaf2e3c8919a429486b3d`.
The old update-count schedule was intentionally preserved for this control;
at update 564 the main LR is still `2.73275e-5`. After baseline qualification,
an example-scaled schedule/LR comparison should precede architecture changes.

The immediately following elevated two-pool validation launch was rejected by
the platform usage-limit gate, which reported availability resuming
2026-07-28 17:03 UTC. Do not bypass that gate. The checkpoint is retained as
the sole candidate state, and the matched PyTorch pools plus frozen JAX
round-trip/validation/eight-pass inference remain queued.

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
the control checkpoint's matched full-pool PyTorch validation, exact JAX round
trip, and full two-pool JAX validation/eight-pass inference cross-check.

## First Torch-checkpoint Elo timing, 2026-07-24

The frozen arena now accepts the strict model-only eager-Torch checkpoint
directly. It constructs the matching JAX inference model and requires an exact
455-leaf, 705,987,352-byte shape/dtype/value round trip before warm-up; it
does not write a duplicate checkpoint. The current batch-512/update-564
control passed with combined state checksum
`a3a99de695eb73b28a35eda75323a075713d0032debdaf2e3c8919a429486b3d`.

The 128-pair, 256-game development arena against original raw BT4 completed
through `research/run_gpu.sh` in 176.583 seconds cold end to end. Exact
checkpoint load/audit took 23.778 seconds, candidate/raw warm-up took
2.855/1.242 seconds, and gameplay took 131.606 seconds. The candidate/raw
mean batched policy-call times were 41.463/34.226 ms at physical batch 16.
The guard measured 6,588,268,544 bytes peak process-group RSS and
7,485,341,696 bytes minimum host `MemAvailable`.

The control scored `35.7422%`: 2 wins, 179 draws, and 75 losses, for
descriptive logistic Elo `-101.90` with the deliberately conservative
pair-aware interval `[-202.74,-15.67]`. Four losses were charged
`no_representable_move` faults from incomplete legacy action-codec coverage;
raw BT4 had complete canonical coverage. This result is a valid frozen arena
outcome but is not a clean estimate of small model-strength differences until
the codec asymmetry is removed or the fixed-reference comparison makes it
symmetrical.

At this throughput, a cold 128-pair screen adds under three minutes to a
30-minute training experiment. That is suitable for rejecting large chess
regressions and ranking large gains, but not for resolving roughly 10--20 Elo:
the development interval is far wider. Use paired score—not the transformed
point Elo—as the numerical autoresearch objective, retain offline
legality/collapse checks as hard gates, and confirm apparent winners on
disjoint pairs.

All local controls and architecture experiments restore every one of the 455
model leaves from the joint step-265,000 checkpoint and create a fresh
optimizer. The restored leaves include the fine-tuned BT4 encoder, state
projector, DFM planner, JEPA embeddings/adapters/transition/norm, and disabled
value/WDL head. Constructor initializations are overwritten. Runs do not chain
from the preceding local winner, but they are continuations from the pretrained
joint model rather than raw BT4 plus fresh JEPA/DFM modules.
