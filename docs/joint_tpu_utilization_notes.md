# Joint Latent-SASA TPU Utilization Notes

Date: 2026-05-01

This note summarizes the TPU profiling discussion around joint Latent-SASA
training: BT4 frozen encoder, DFM action denoising, and JEPA latent future
prediction on trajectory-v2/v3-style chess rollouts.

## Current Training Shape

The active joint Stage 1 path is:

```text
planes_t
  -> frozen BT4 encoder
  -> raw BT4 current tokens z_t [B, 64, 1024]

z_t
  -> compact DFM projector/adapters
  -> DFM action denoiser over [64 board tokens + H action tokens]
  -> action-token hidden states [B, H, D]

z_t + actions + DFM action hidden states
  -> recurrent JEPA transition
  -> predicted future BT4 tokens [B, H, 64, 1024]

planes_future[:, h]
  -> frozen BT4 encoder
  -> raw BT4 target tokens [B, target_count, 64, 1024]
```

The hot JEPA loss is raw MSE against raw frozen BT4 target tokens. Cosine and
normalized MSE are diagnostics only.

## Measured Single-Chip Profiles

Hardware:

```text
TPU: single v5litepod-1
peak used for MFU estimate: 197 TFLOP/s bf16
HBM capacity observed by XLA: ~15.75 GiB
```

Steady-state medians from W&B history, excluding compile, trace, and final
checkpoint/upload windows:

```text
config                                  median step   median MFU   notes
H2, all future targets, B64              ~0.0758 s     ~24.0%      baseline full target
H2, one future target, B64               ~0.0628 s     ~22.3%      valid sampled target
H4, one future target, B64               ~0.0736 s     ~23.4%      deeper horizon
H8, one future target, B64               ~0.0968 s     ~24.4%      current useful path
H8, one future target, B128              ~0.1854 s     ~25.4%      batch-size check
H2, no future target diagnostic, B64     ~0.0517 s     ~17.0%      invalid objective
H8, D640 DFM8 JEPA4, B64                 ~0.1678 s     ~24.7%      wider/deeper neutral
H8, D1024 DFM8 JEPA4, B64                ~0.2100 s     ~23.8%      wider DFM worse
H8, D640 DFM12 JEPA8, B64                OOM           n/a         compile-time HBM OOM
```

Representative W&B runs:

```text
joint-s1-h8-b64-target-sample1-profile-20260501a
joint-s1-h8-b128-target-sample1-profile-20260501a
joint-wide-h8-b64-d640-dfm8-jepa4-profile-20260501a
joint-wide-h8-b64-d1024-dfm8-jepa4-profile-20260501a
joint-wide-h8-b64-d640-dfm12-jepa8-profile-20260501a
```

The JEPA8 run failed during compilation:

```text
RESOURCE_EXHAUSTED: CompileTimeHbmOom
required HBM: ~19.12 GiB program
available HBM: ~15.75 GiB
largest temporaries: repeated bf16[8,4096,4096] allocations
```

## Post-Fix Profiling Results

The first RMSNorm/SwiGLU/layer-scan profile exposed a host-side synthetic data
generation bottleneck, not a model bottleneck:

```text
run: joint-profile-rmsswiglu-b64-h8-20260501c
config: synthetic H8 B64, DFM4, JEPA2, Muon, quantile SigReg

post-profile step time:      ~0.146 s
estimated step MFU:          ~16.1%
estimated iteration MFU:     ~5.7%
data fetch time:             ~0.26 s
host overhead fraction:      ~64%
```

Perfetto attributed the host time to:

```text
build_synthetic_joint_batch
  -> build_synthetic_trajectory_shard
  -> rollout_from_fen
  -> encode_board
```

The synthetic generator now precomputes a few canned legal opening rollouts and
the trainer caches synthetic batches by `(batch_size, horizon, legal_lmax)`.
After this fix, the same synthetic profiling path has effectively zero host
fetch time:

```text
data fetch time:             ~0.000002 s
host overhead fraction:      ~3%
```

The second profiling suite compared SigReg and optimizer variants:

```text
profile artifact root:
  artifacts/profiles/profile-compare-20260501e

run                                           step time   MFU     conclusion
joint-profile-moments-muon-b64-h8-20260501e   ~0.0991 s  23.8%  current best profile
joint-profile-quantile-muon-b64-h8-20260501e  ~0.1429 s  16.5%  sort/scatter bottleneck
joint-profile-moments-adamw-b64-h8-20260501e  ~0.0993 s  23.7%  same speed as selective Muon
```

The old quantile SigReg path creates large sort/scatter work:

```text
scatter_custom_fusion.3: ~256 ms aggregate in profile window
sort.15:                 ~27 ms aggregate
sort.14:                 ~23 ms aggregate
```

Switching JEPA SigReg to the moment regularizer removes that bottleneck and
improves throughput by roughly 44% versus quantile SigReg for the profiled
H8/B64 configuration. The current default is:

```text
--jepa-sigreg-kind moments
```

Selective Muon and AdamW-only were effectively identical for this small
D256/DFM4/JEPA2 profile once quantile SigReg was removed. That means Muon is
not the current dominant single-chip bottleneck for this configuration,
although the conservative Muon routing should remain in place for larger
rectangular FFN-heavy configs.

The remaining profile bottlenecks are device-side:

```text
XLA while regions:
  dominant remaining category; likely recurrent JEPA horizon scan, layer scans,
  and optimizer/update loops. Needs named-scope/HLO attribution.

convolution/dot fusion:
  large but productive matmul work; not obviously pathological.

data formatting:
  still visible in XProf but no longer a Python synthetic-generation issue.
  Likely layout conversion / host-to-device / buffer linearization overhead.
```

The practical improvement so far is:

```text
before synthetic + SigReg fixes:
  iteration MFU ~= 5.7%

after synthetic cache + moment SigReg:
  iteration MFU ~= 23.0%
  step MFU      ~= 23.8%
```

The 20260501f scoped profile isolates the remaining costs:

```text
run                                           median step   median MFU   notes
future_bt4 B64 moment SigReg                  ~0.0996 s     ~23.7%      real sampled-target path
current_repeat B64 moment SigReg              ~0.0811 s     ~22.6%      removes future BT4 target encode
future_bt4 B128 moment SigReg                 ~0.2035 s     ~23.2%      batch scaling does not improve MFU
```

Future target encoding therefore costs about:

```text
0.0996 s - 0.0811 s ~= 0.0185 s/step
```

or roughly 19% of the real B64 step. The remaining 81% is the current-state
BT4 encode, two DFM passes, recurrent JEPA rollout, loss, and optimizer update.

Named-scope attribution from the B64 future-target trace:

```text
joint_jepa_recurrent_rollout        ~356 ms aggregate, ~44.3 TF, ~357 GB
joint_encode_current_bt4             ~94 ms aggregate,  ~8.3 TF,  ~70 GB
joint_encode_future_bt4_targets      ~90 ms aggregate,  ~8.2 TF,  ~63 GB
joint_dfm_noisy_planner              ~16 ms aggregate,  ~0.9 TF,  ~17 GB
joint_dfm_clean_planner_hidden       ~16 ms aggregate,  ~0.8 TF,  ~18 GB
joint_stage1_optimizer_update         ~8 ms aggregate,  ~0.0 TF,   ~6 GB
joint_jepa_sigreg                     ~3 ms aggregate,  ~0.0 TF,   ~3 GB
```

XLA category attribution from the same trace:

```text
while                    ~711 ms aggregate, ~25.3 TF, ~186 GB
convolution fusion       ~384 ms aggregate, ~62.5 TF, ~193 GB
data formatting           ~85 ms aggregate,  0.0 TF,  ~74 GB
loop fusion               ~85 ms aggregate, ~0.1 TF, ~146 GB
```

The scoped trace confirms that the largest remaining model-side bottleneck is
the recurrent JEPA rollout. Importantly, `--jepa-target-sample-count 1` samples
only the target horizon for BT4 target encoding and loss. The implementation
still unrolls JEPA through all H action steps and applies moment SigReg over
all predicted horizons. That is mathematically valid, but it means target
sampling saves target-encoder compute only; it does not reduce recurrent JEPA
rollout compute.

The 20260501g remat/layer-scan ablation on the same H8/B64 synthetic profile:

```text
config                          median step   median MFU   conclusion
scan + remat                    ~0.0996 s     ~23.7%      current default profile
scan + no remat                 ~0.1327 s     ~17.8%      bad; remat helps lowering
no scan + remat                 ~0.0963 s     ~24.5%      fastest shallow setting
no scan + no remat              ~0.0965 s     ~24.4%      similar to no-scan/remat
```

For shallow JEPA2/DFM4 single-chip runs, `--no-scan-layers` is a small speed
win. For larger JEPA depth, scanned layers are still useful as a memory and
compile-structure tool; do not remove the scan path until JEPA6/JEPA8 compile
and HBM behavior are measured.

## Target-Horizon Sampling

`--jepa-target-sample-count 1` samples one target horizon per step, encodes only
that future board, and computes raw BT4 MSE for that horizon. This is a valid
stochastic training objective because the target remains the true BT4 encoding
of a real future state.

It is different from `--jepa-target-mode current_repeat`, which is profiling-only:

```text
current_repeat target = repeated z_t
```

That diagnostic removes future BT4 target encoding from the graph, but it is
not model-selection evidence and should not be used for real training.

Target-horizon sampling improved wall time:

```text
H2 all targets B64:    ~0.0758 s
H2 sampled target B64: ~0.0628 s
```

It is especially useful for H4/H8 because it avoids encoding every future board
on every step while still training against real future BT4 latents over time.

## Arithmetic Intensity

Using v5e numbers:

```text
peak compute ~= 197 TFLOP/s bf16
HBM bandwidth ~= 819 GB/s
machine balance ~= 197e12 / 819e9 ~= 240 FLOP/byte
```

Lower-bound step-level arithmetic intensity estimates are far above the memory
roofline balance point:

```text
H8 B64 sampled target:
  total FLOPs/step ~= 4.64 TF
  lower-bound bytes/step ~= 0.89 GB
  lower-bound AI ~= 5240 FLOP/byte

H8 B128 sampled target:
  total FLOPs/step ~= 9.28 TF
  lower-bound bytes/step ~= 0.97 GB
  lower-bound AI ~= 9600 FLOP/byte
```

These byte estimates are lower bounds. Real HBM traffic is higher because XLA
materializes intermediates, layouts, optimizer states, activations, and
temporary buffers. Still, the GEMM-level arithmetic intensity suggests the
large BT4/JEPA matmuls are not inherently memory-bandwidth limited:

```text
BT4 q/k/v GEMMs, B64        ~= 455 FLOP/byte
BT4 FFN GEMMs, B64          ~= 534 FLOP/byte
JEPA q/k/v GEMMs, B64       ~= 455 FLOP/byte
JEPA FFN GEMMs, B64         ~= 683 FLOP/byte
DFM 256 q/k/v GEMMs, B64    ~= 125 FLOP/byte
DFM 256 FFN GEMMs, B64      ~= 196 FLOP/byte
```

The small DFM blocks are closer to or below the machine balance point, but DFM
is a relatively small part of total FLOPs at the current widths.

## Width, Depth, And FLOP Estimates

For H8, B64, one sampled future target, JEPA width fixed at raw BT4 width
1024:

```text
D=256  DFM4  JEPA2: total ~= 4.64 TF, BT4 ~= 44%, DFM ~= 0.09 TF, JEPA ~= 2.50 TF
D=640  DFM8  JEPA4: total ~= 8.16 TF, BT4 ~= 25%, DFM ~= 1.11 TF, JEPA ~= 5.00 TF
D=1024 DFM8  JEPA4: total ~= 9.87 TF, BT4 ~= 21%, DFM ~= 2.82 TF, JEPA ~= 5.00 TF
D=640  DFM12 JEPA8: total ~= 13.71 TF, BT4 ~= 15%, DFM ~= 1.66 TF, JEPA ~= 10.00 TF
```

The measured profiles show:

```text
D640 DFM8 JEPA4: roughly neutral MFU
D1024 DFM8 JEPA4: worse MFU
D640 DFM12 JEPA8: OOM without remat/sharding
```

This means making DFM much wider/deeper does not automatically improve MFU. The
more promising way to add useful high-arithmetic-intensity compute is JEPA
depth, but JEPA8 needs memory work first.

## What "Kernel Limitation" Means Here

The workload has high theoretical arithmetic intensity, but TPU utilization can
still be low if XLA lowers the graph into many kernels that do not saturate the
systolic arrays.

The current graph has several unfavorable properties:

```text
BT4:
  64-token sequence
  15 encoder layers
  smolgen attention bias
  many small ops around attention and FFNs

JEPA:
  recurrent lax.scan over H action steps
  64-token transformer blocks
  raw width 1024
  repeated small transformer calls instead of one long standard sequence

DFM:
  sequence length 64 + H
  small action horizon H <= 8
  D=256-640 unless explicitly widened

Training stack:
  NNX module graph
  Muon/AdamW optimizer
  single-chip execution
  many small parameter groups and state updates
```

This differs from high-MFU transformer training setups, which usually have long
sequences, large batches, fused standard attention/MLP kernels, and explicit
multi-device sharding.

## Recurrent JEPA Versus Direct Multi-Horizon JEPA

The current JEPA is recurrent:

```text
z1_hat = f(z0, a0)
z2_hat = f(z1_hat, a1)
z3_hat = f(z2_hat, a2)
...
```

This requires sequential unrolling because each future latent depends on the
previous predicted latent.

A parallel/direct alternative would be:

```text
z_h_hat = F(z0, a0:h)
```

This can process horizons more like one large sequence or batch. It may be
faster and easier to fuse, but it is not mathematically equivalent to the
recurrent latent dynamics model. It changes the model class from compositional
one-step transition to direct multi-horizon prediction.

Both styles exist in sequence/future-prediction literature:

```text
recurrent/unrolled transition:
  better match for learned dynamics and compositional rollout

direct multi-horizon prediction:
  easier to parallelize, may optimize finite-horizon prediction better
```

Recommendation: keep recurrent JEPA as the default for now. Consider direct
multi-horizon JEPA as an ablation if recurrent JEPA remains too slow or too
hard to optimize.

## Fused Kernel Feasibility

Custom TPU kernels are not realistic for this repo right now. Practical
"fusion" means changing model structure or JAX lowering:

```text
possible:
  stack layer parameters and use lax.scan over layers
  add remat/checkpointing around JEPA blocks
  flatten candidate chunks for contrastive training
  batch target horizon encodes where it helps
  reduce separate projections/adapters in hot paths
  use direct multi-horizon JEPA as a separate model ablation

not practical now:
  hand-written custom TPU fused attention/MLP kernels
```

The previous JAX SDPA attempt for BT4 was slower than the manual attention path
for this 64-token smolgen-biased shape, so manual BT4 attention remains the
default.

## Contrastive Loss Implications

Stage 2 contrastive/ranking uses candidate chunks:

```text
[B, K, H, ...]
```

The current Stage 2 implementation flattens candidates to:

```text
[B * K, ...]
```

This is the right direction for utilization because it increases effective
batch size and GEMM sizes. It also increases HBM pressure.

Optimizations likely to survive contrastive training:

```text
target-horizon sampling
BT4 target batching or precompute
transparent hugepage startup fix
remat/checkpointing
layer-scan implementation cleanup
candidate flattening
```

Optimizations that need rethinking with contrastive/ranking:

```text
direct multi-horizon JEPA
candidate memory layout
DFM hidden-state reuse across positive and negative chunks
precomputed latents if target/projector choices change
```

Contrastive may naturally improve utilization by increasing candidate batch
size, but it may also require remat or multi-device sharding because memory
scales with K.

## Current Conclusion

After fixing synthetic generation and quantile SigReg, the current single-chip
implementation still appears to hit a practical MFU ceiling around 24-25%.
This is not because data loading is slow and not simply because batch size is
too small:

```text
B128 H8 sampled target only reached ~25.4% MFU.
D640/DFM8/JEPA4 only reached ~24.7% MFU.
D1024/DFM8/JEPA4 was worse.
JEPA8 OOMed before running.
```

This is not necessarily a universal ceiling for the project. It is the current
single-chip ceiling for the existing recurrent JEPA/DFM/BT4 implementation.

## Potential Next Steps

Highest-value next steps:

```text
1. Attribute the remaining XLA while regions with named scopes and optimized HLO.
2. Run current_repeat versus future_bt4 target ablations to isolate BT4 target encode cost.
3. Retry B128/B256 with moment SigReg to check whether the fixed path scales better.
4. Add remat/checkpointing to frozen BT4 only if target-encode attribution justifies it.
5. Run Stage 2 contrastive with candidate flattening and measure whether B*K improves MFU.
6. Implement explicit multi-device sharding before using v5litepod-8+ seriously.
7. Consider a direct multi-horizon JEPA ablation if recurrent scan remains a bottleneck.
8. Explore precomputed frozen BT4 tokens for fixed-encoder experiments only.
```

Lower priority:

```text
1. More batch-size sweeps on single chip.
2. More DFM width-only sweeps.
3. More cache/data-loading work for this MFU issue.
```

When BT4 is eventually unfrozen, encoder efficiency becomes more important
again. At that stage, remat, sharding, attention kernel/layout choices, and
optimizer-state memory become first-class blockers rather than profiling
curiosities.

## Follow-Up Optimizer/Shape Audit

A later audit pointed at two plausible sources for the JEPA8 compile-time HBM
OOM and the single-chip MFU ceiling: bad attention flattening or optimizer
temporaries.

The current `EncoderLayer` attention path does not appear to flatten the batch
into the attention sequence. Q/K/V are shaped as:

```text
[B, L, heads, head_dim] -> [B, heads, L, head_dim]
logits = q @ k^T -> [B, heads, L, L]
```

So the specific bad shape:

```text
[heads, B*L, B*L]
```

is not present in the main manual attention path.

The installed Optax Muon implementation also already transposes tall matrices
before Newton-Schulz when `rows > cols`, so the naive "FFN [4096,1024] creates a
[4096,4096] Gram" explanation is probably not the exact source under Optax
0.2.8. However, applying Muon to every 2D trainable leaf still spends optimizer
compute on FFN expansion/contraction matrices, embeddings, and output heads.
The repo now routes Muon conservatively:

```text
Muon: square-ish 2D matrices with aspect <= 2 and min dim >= 128
AdamW: FFN expansion/contraction, embeddings, output heads, norms, biases, tiny matrices
```

This should be tested in the next profiling run against:

```text
current conservative Muon
AdamW-only diagnostic
previous all-2D Muon behavior, if needed for comparison
```

The next profile should still dump or inspect optimized HLO for the JEPA8
configuration to identify the actual `bf16[8,4096,4096]` source before any
custom kernel work.

## Optimization Checklist Status

The follow-up suggestions are being handled in this order:

```text
QKV projection fusion:
  Implemented in EncoderLayer forward pass. Parameters remain checkpoint-
  compatible as wq/wk/wv, but execution now concatenates them into one QKV
  projection matmul before splitting q/k/v.

MLP input fusion:
  Implemented for trainable DFM/JEPA blocks via TrainableTransformerStack.
  The frozen BT4 EncoderLayer remains unchanged. The trainable stack uses
  gate_up = x @ w_gate_up, splits gate/up, applies silu(gate) * up, then
  projects down.

Attention layout:
  Manual attention already keeps logits at [B, heads, L, L]. The bad
  [heads, B*L, B*L] pattern was not found. SDPA remains optional because it
  was slower on the 64-token smolgen-biased BT4 shape.

Muon routing:
  Implemented conservatively. Muon is routed only to square-ish 2D matrices and
  stacked square-ish 3D matrices with leading layer axes. FFN expansion/
  contraction, embeddings, heads, norms, biases, and tiny leaves use AdamW.

Remat/checkpointing:
  Implemented for trainable DFM/JEPA blocks via --remat-blocks (default on).
  The implementation no longer remats NNX modules inside the JEPA horizon scan;
  it remats a pure array-parameter block function. Frozen BT4 encoder remat is
  not enabled yet.

Layer scan:
  Implemented for trainable DFM/JEPA blocks via --scan-layers. The joint
  trainer default is now off because JEPA2/DFM4 profiling was faster without
  the layer scan. The scan path remains available for larger-depth memory/
  compile experiments. Both paths use the same stacked TrainableParam layout,
  so toggling --scan-layers does not change the new trainable block checkpoint
  schema.

Buffer donation:
  Implemented for joint Stage 1/2 train steps via --donate-train-state
  (default on). Model and optimizer buffers are donated; batches are not.

HLO/XProf attribution:
  Profile capture flags already exist in train_joint_latent_sasa.py. The next
  profiling job should use them plus an optimized-HLO dump to attribute the
  remaining large temporaries.

Custom kernels:
  Deferred. Full custom attention/MLP kernels are not the right first move.
  A Pallas RMSNorm/LayerNorm kernel is only worth testing if XProf shows norm
  fusions are a material bottleneck.

Contrastive Stage 2:
  Candidate flattening is the intended path and should survive the QKV/Muon
  changes. Memory pressure will need separate profiling once K candidates are
  enabled.

Multi-device sharding:
  Not implemented. This is the main path beyond the current single-chip MFU
  ceiling once the single-chip shape/optimizer issues are understood.
```

## Stage 2 Contrastive Smoke Profile

Stage 2 contrastive training has been smoke-tested on one v5e/v5lite-style TPU
chip with synthetic H4 batches and `candidate_count=3`.

```text
Run ids:
  joint-stage2-h4-k3-b32-profile-20260502a
  joint-stage2-h4-k3-b64-profile-20260502a

Config:
  stage = stage2
  horizon = 4
  loss_horizon = 4
  candidate_count = 3
  contrastive_coeff = 0.5
  contrastive_temperature = 0.1
  DFM = d256, 4 layers, 4 heads, SwiGLU
  JEPA = raw BT4 token target, 2 layers, 8 heads, MLP 4096
  jepa_target_sample_count = 1
  jepa_sigreg_coeff = 0.05
  train t = random uniform
```

The smoke run completed without OOM for both B32 and B64. The old run config
under-counted Stage 2 MFU because it only counted the Stage 1 DFM/JEPA pass and
two BT4 encoder batches. The corrected Stage 2 estimator counts:

```text
Stage 1:
  noisy DFM CE pass
  clean DFM hidden-state pass
  JEPA rollout on the true chunk
  current BT4 encode
  sampled future BT4 target encode

Stage 2 contrastive:
  flattened candidate DFM pass over B * K chunks
  flattened candidate JEPA rollout over B * K chunks
  current + all-H future BT4 target encodes for contrastive targets
```

For H4/K3/B64, the corrected estimate is about `12.4 TFLOP/step`. The observed
steady step time was about `0.24 s/step`, implying roughly `26%` model-FLOP MFU
on the single-chip nominal `197 TFLOP/s` denominator. This is materially higher
than the stale logged `~7%` number, but it is still the same achieved-throughput
ceiling seen in Stage 1 profiling.

Interpretation:

```text
candidate flattening works:
  Stage 2 compiles and runs without immediate memory failure at K=3/H4/B64.

single-chip ceiling remains:
  B64 roughly doubles B32 step time. The path is still linear-throughput bound
  on one chip rather than unlocking better matmul utilization.

next useful scaling step:
  implement explicit data-parallel sharding before provisioning a 4-chip run.
  Without sharded batch/state placement, a multi-chip TPU is likely to behave
  like an expensive single-chip run for this script.
```

## Stage 2 Real-Data Smoke

The first real-data Stage 2 smoke used the same H4/K3/B64 shape on compact v3
LC0 shards:

```text
run:
  joint-stage2-h4-k3-b64-v3-smoke-20260502d

initialization:
  --init-checkpoint-uri points at the Stage 1 long-run checkpoint root
  --init-checkpoint-step 387072
  non-strict model migration, fresh optimizer

data:
  gs://gunay-chess-experiments-us-central1/data/trajectory_v3_lc0_test80_h8_sets1_3_20260430
  minimum-cache smoke: 32 train shards and 32 val shards visible

runtime:
  first train step compile/startup: about 96 s
  steady step time: about 0.24 s
  steady model-FLOP MFU: about 26%
```

The checkpoint migration was required because the Stage 1 checkpoint predates
the fused trainable stack. The migration maps old per-layer Q/K/V attention
weights into fused QKV, copies output projection and norm scales, seeds the
compatible FFN subspace into the SwiGLU up/down path, and intentionally starts
a fresh optimizer.

Validation metrics from the 200-step smoke:

```text
step 100:
  val_contrastive_loss       1.2755
  val_contrastive_accuracy   0.3359
  val_jepa_raw_mse           1.8177
  val_dfm_ce_loss            5.7810
  val_first_legality_loss    0.8545

step 200:
  val_contrastive_loss       0.4978
  val_contrastive_accuracy   0.7148
  val_jepa_raw_mse           1.5724
  val_dfm_ce_loss            5.6433
  val_first_legality_loss    0.8115
```

Interpretation:

```text
Stage 2 contrastive signal is live:
  positive-vs-legal-negative separation improves quickly in the smoke.

Action model quality is not established:
  DFM CE, first legality, and action accuracy remain rough over 200 steps.

Data/cache caveat:
  this was a minimum-cache validation of the code path, not a full-dataset run.
  Serious Stage 2 should run from a persistent-disk dataset view or a full
  deterministic cache/window, not a small growing cache.
```

## Activation And Normalization Details

Frozen BT4 still uses the checkpoint-compatible `EncoderLayer`.
Trainable DFM/JEPA blocks now use `TrainableTransformerStack`.

```text
Trainable DFM/JEPA transformer FFN:
  pre-RMSNorm residual block
  fused SwiGLU gate/up projection
  hidden = silu(gate) * up
  down projection back to residual width

Trainable DFM/JEPA attention:
  pre-RMSNorm
  fused QKV projection
  logits shaped [B, heads, L, L]
  residual add without post-norm

DFM time embedding activation:
  ReLU between the first and second time-embedding projections

Standalone JEPA auxiliary MLPs:
  ActionMLP and ValuePredictionHead use Swish

Frozen BT4 FFNs and heads:
  Mish, matching the BT4 checkpoint
```

The trainable blocks use RMSNorm on branch inputs:

```text
TrainableRMSNorm:
  rms over feature dim, no mean subtraction
  eps = 1e-6
  trainable scale, no bias
  stats in fp32
  applied before attention and before SwiGLU MLP
  no post-residual norm, preserving a clean residual identity path

Fixed BT4 LayerNorm:
  eps = 1e-3
  fixed scale and bias from BT4
  stats promoted to fp32 for fp16/bf16 encoder dtype
```
