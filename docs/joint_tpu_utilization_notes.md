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

The current single-chip implementation appears to hit a practical MFU ceiling
around 24-25%. This is not because data loading is slow and not simply because
batch size is too small:

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
1. Add remat/checkpointing to JEPA blocks and retry JEPA6/JEPA8.
2. Run Stage 2 contrastive with candidate flattening and measure whether B*K improves MFU.
3. Implement explicit multi-device sharding before using v5litepod-8+ seriously.
4. Consider a direct multi-horizon JEPA ablation if recurrent scan remains a bottleneck.
5. Explore precomputed frozen BT4 tokens for fixed-encoder experiments only.
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
