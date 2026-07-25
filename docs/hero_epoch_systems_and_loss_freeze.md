# Hero Epoch Systems and Loss Freeze

Status: systems runtime and loss coefficients frozen on 2026-07-25. Learning
rate and weight decay remain intentionally unfrozen pending the resettable
LR-range calibration.

This is the execution record for steps 7 and the loss-coefficient portion of
step 8 in `docs/hero_epoch_plan.md`. All measurements used the A10G through
`research/run_gpu.sh`, with two-CPU affinity, the 7 GiB process-group RSS
ceiling, host-memory and disk floors, an exclusive GPU lock, and zero
checkpoint writes.

## Frozen runtime

The hero epoch will use:

- physical batch size `1024`;
- normalized SIGReg sample size `64`;
- BF16 PyTorch training with native scaled-dot-product attention in BT4 and
  the fresh attention stacks;
- rematerialization in BT4 and the state projector, but not in DFM;
- regional `torch.compile` for state-projector blocks, DFM blocks, and the
  JEPA transition; and
- eager BT4 encoder layers and the existing Muon/Nesterov-AdamW optimizer.

The paired batch sweep measured `174.77` examples/s at batch 512 and `186.49`
examples/s at batch 1024. A separate monitored batch-1024 confirmation
measured `190.93` examples/s over updates 11--20, or a projected
`41.24` hours per 28,343,296-example epoch. Use `186--191` examples/s and
`41.2--42.2` hours as the honest pre-launch range rather than the compile-cold
end-to-end rate.

The monitored run reached:

- `20.696 GB` peak allocated and `23.073 GB` peak reserved HBM;
- `22,330 MiB` peak device memory reported by NVML out of `23,028 MiB`;
- approximately `82.7%` post-first-update and `83.9%` late-window GPU
  utilization;
- approximately `191.9 W` late-window power at a fixed `1710 MHz` SM clock;
  and
- no data wait after prefetch warmup.

Batch 1024 is therefore the HBM-edge throughput point. Batch 1152/1280 was
not attempted: less than 700 MiB remained at the NVML peak, so an OOM would
be more likely than a useful sustained-throughput gain.

## Systems matrix

| Candidate | Batch | Sustained result | Decision |
|---|---:|---:|---|
| Eager/manual-attention control | 512 | about `152` examples/s | Baseline |
| Native SDPA in all attention | 512 | about `164` examples/s | Accept |
| SDPA plus no DFM rematerialization | 512 | about `167` examples/s | Accept |
| Compile DFM and JEPA | 512 | `174.77` examples/s | Accept |
| Compile projector, DFM, and JEPA | 512 | up to `193.89` examples/s in the earlier clock regime | Accept |
| Paired compiled-fresh control | 512 | `174.77` examples/s | Batch comparison control |
| Paired compiled-fresh candidate | 1024 | `186.49` examples/s | Select |
| Monitored compiled-fresh confirmation | 1024 | `190.93` examples/s | Confirm |
| Batched Muon matrices | 512 | optimizer `197.3` to `190.8 ms`; no end-to-end gain | Reject |
| Inductor max-autotune | 512 | stopped at `7,523,008,512` bytes RSS | Reject safely |

The single-update profiler counted approximately `61.986 TFLOP` at batch 512
and identified GEMMs as the dominant device work, with substantial
pointwise, copy, and reduction traffic and roughly 20,736 launches. Doubling
that approximate FLOP count at the monitored batch-1024 step time gives about
`23 TFLOP/s`, or roughly `18.4%` of the A10G's headline dense-BF16 rate. This
is an approximate diagnostic because profiler FLOP accounting is incomplete;
the direct utilization and throughput measurements are the primary systems
metrics.

The immediate `200` examples/s target was missed by roughly 5%, but the work
raised the prior rough `14%` MFU estimate, selected native fused attention,
removed unnecessary DFM recomputation, compiled every fresh repeated region
that passed correctness, and raised steady utilization into the mid-80s.
Further BT4 fusion or full-step CUDA graphs remain later systems experiments,
not launch blockers for the first clean epoch.

## Rejected BT4 compilation

Compiling the raw BT4 path was faster but failed the preregistered
update-zero forward/backward parity gate. Representative results were:

| BT4 compile region | Legal top-1 agreement | Global gradient cosine | Raw-BT4 gradient cosine |
|---|---:|---:|---:|
| Full layer, default numerics | `95.12%` | `0.9063` | `0.8318` |
| Full layer, emulated eager casts | `97.85%` | `0.9664` | `0.9520` |
| Smolgen only, default numerics | `96.29%` | `0.8930` | `0.7750` |
| Smolgen only, emulated eager casts | `96.48%` | `0.9280` | `0.8537` |

Disabling epilogue fusion did not improve the strict full-layer result.
Because update zero must reproduce raw BT4 and these deviations reach the
policy-bearing backbone gradients, BT4 remains eager.

Inductor max-autotune was also rejected. The resource guard stopped its
single compile thread after about eight minutes when group RSS crossed the
fixed `7,516,192,768`-byte ceiling. It did not checkpoint or affect host
state. The ordinary regional compiler remains below the ceiling.

The host NVML userspace package did not match the loaded `580.159.03` kernel
driver. An exact `580.159.03` NVML library was extracted under
`/mountpoint/.exp` and is preloaded only by `research/run_gpu.sh`. No host
driver/library was installed or modified. This restores PyTorch allocator
low-memory handling and guarded GPU telemetry while leaving `libcuda` and all
training kernels unchanged.

## Frozen loss coefficients

The final loss is:

```text
L =
    1.0 * L_DFM_CE_H1_to_H8
  + 0.08707768618513193 * L_root_legal_conditional_CE
  + 2.0 * L_root_illegal_mass
  + 1.0 * L_JEPA_raw_MSE
  + 2.0 * SIGReg(z_target)
  + 2.0 * SIGReg(z_pred)
  + 0.25 * L_WDL(z_pred)
```

The root coefficient comes from deterministic update-zero batch-1024 root CE
`2.870999574661255`, making its weighted contribution exactly `0.25`.

The no-update batch-512 gradient audit replayed that frozen initialization and
measured:

| Component | Weighted scalar | Global gradient norm |
|---|---:|---:|
| Eight-horizon DFM CE | `7.4484` | `18.96` |
| Root legal-conditional CE | `0.2536` | `12.07` |
| Root illegal mass | `1.9552` | `1.32` |
| JEPA raw MSE | `0.9377` | `98.97` |
| Target SIGReg | `1.0331` | `48.49` |
| Prediction SIGReg | `1.1009` | `62.32` |
| WDL CE | `0.2946` | `35.82` |

The target/prediction SIGReg gradients are strongly aligned
(`cosine = 0.835`), non-SIGReg representation and SIGReg gradients are
positively aligned (`0.267`), and policy and total representation gradients
are positively aligned (`0.406`). Shared-coefficient projections were:

| Shared SIGReg coefficient | Representation norm | Total norm | Policy/representation cosine |
|---:|---:|---:|---:|
| `0.0` | `95.56` | `112.54` | `0.435` |
| `0.5` | `105.79` | `122.86` | `0.449` |
| `1.0` | `121.09` | `137.59` | `0.441` |
| `2.0` | `160.72` | `175.61` | `0.406` |
| `4.0` | `255.13` | `267.53` | `0.349` |

Freeze the shared coefficient at `2.0`: its two scalar contributions are
close to the JEPA-MSE scale, it provides the intended replacement for the
removed prediction-norm loss, and the audit finds alignment rather than a
gradient conflict. The projected coefficient-1 and coefficient-2 total
gradients have cosine `0.972`, so coefficient 2 does not rotate the update
into a qualitatively different direction. Monitor the component gradients and
representation statistics through the warmup; do not independently tune the
target and prediction coefficients.

Physical batch size does not rescale either SIGReg coefficient. Each
estimator uses a deterministic 64-example sample and a normalized
V-statistic, so moving from batch 512 to 1024 changes neither the estimator's
expectation nor its scalar convention. A future change to the 64-example
sample is a scientific change and requires another gradient audit, not an
automatic batch-ratio correction.

## Evidence and remaining calibration

Primary retained evidence:

- monitored batch-1024 report:
  `research/runs/torch_hero_compile_fresh_monitored_b1024_v4/report.json`,
  SHA-256
  `c4f1062d405db60a8938b28292c94ce4806eab3b8bba077fef7ed61ea8555df4`;
- matched update-zero metrics used for root calibration:
  `research/runs/torch_hero_compile_fresh_monitored_b1024_v4/metrics.jsonl`,
  SHA-256
  `749daad67ad808a4ec5dd3684f77b03f67a2a10ed5aeca27687d634e80b2ef6d`;
- profiler report:
  `research/runs/torch_hero_compile_dfmjepa_b512_profile_v1/report.json`,
  SHA-256
  `e6cb9d2eb452abbe5286cb21d97aa06d7395efc481a8960a724000d66c899acd`;
  and
- loss-gradient audit:
  `artifacts/profiles/hero_loss_gradient_audit_b512_rootcal_b1024_v2.json`,
  SHA-256
  `1a46e941880a86d4cc986a17af2b0f2c1d14a49f2766b21b0d5c78c76b0647b8`.

A guarded, no-checkpoint three-update smoke then passed at batch 1024. Update
zero reproduced root CE `2.870999574661255` and weighted root CE `0.25`
exactly; all loss/gradient values were finite, prefetch hid data preparation
after the first update, and the guard recorded `2.432 GB` peak group RSS with
at least `8.930 GB` host memory available.

Next run the resettable LR-range calibration at batch 1024. Weight decay must
be recomputed after selecting the peak LR. At the provisional `3e-4` peak, the
exact discrete one-epoch schedule has `sum(lr) = 4.1559`; decay `0.01`
produces only `4.07%` integrated shrink. The coefficients for 5%, 7.5%, and
10% shrink would be approximately `0.01234`, `0.01876`, and `0.02535`,
respectively.
