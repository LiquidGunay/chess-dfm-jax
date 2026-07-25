# Hero Epoch Systems and Loss Freeze

Status: systems runtime, loss coefficients, learning rates, and weight decay
frozen on 2026-07-25.

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

## Frozen learning rates and weight decay

Two disposable, no-checkpoint LR-range runs started from the identical raw
BT4 plus fresh-head initialization, used batch 1024 and zero weight decay, and
preserved the 30:1 fresh/BT4 rate ratio.

The planned `1e-5` to `1e-3` sweep completed 128 finite updates without
bracketing failure. Across consecutive 16-update bins:

- mean DFM CE improved monotonically from `7.430` to `6.429`;
- mean legal mass improved from `0.0234` to `0.1587`;
- target and prediction norms remained close; and
- the last `5.6e-4` to `1e-3` region still had the best policy loss.

This establishes that the provisional `3e-4` peak is safe, but not that it is
near the useful limit. A second `1e-5` to `1e-2` sweep located that limit:

- smoothed total loss reached its minimum at main LR `1.411e-3`;
- representation stress was visible by `1.754e-3`, where JEPA MSE rose to
  `1.174` and prediction norm separated to `39.97` versus target `33.76`;
- the automatic divergence criterion fired at `4.188e-3`, where total loss
  reached `48.85` and latent norms jumped to about `500`; and
- by `1e-2`, total loss was `1180.7` and latent norms exceeded `23,000`.

Freeze:

```text
fresh peak LR = 5e-4
raw-BT4 peak LR = 1.6666666666666667e-5
fresh matrix weight decay = 1e-2
```

The fresh peak is approximately one third of the observed minimum-loss LR,
more than threefold below the first representation-stress point, and over
eightfold below detected divergence. The one-decade-below-minimum heuristic
would select `1.41e-4`, but it is rejected here because the early steep loss
drop is dominated by SIGReg equilibration and both sweeps directly show
healthy policy improvement well above that rate.

At batch 1024, the exact discrete warmup/cosine schedule with a `5e-4` peak
has `sum(lr) = 6.92653`. Weight decay `0.01` therefore produces integrated
shrink `exp(-0.01 * 6.92653) = 0.93308`, or `6.69%`, inside the planned
5--10% interval. No further decay adjustment is needed.

## Evidence and remaining launch work

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
  `1a46e941880a86d4cc986a17af2b0f2c1d14a49f2766b21b0d5c78c76b0647b8`;
- planned-range metrics:
  `research/runs/torch_hero_lr_range_b1024_v1/metrics.jsonl`, SHA-256
  `39106dca5988ab7cccf9557ef07984d21e8c798ce2dc351ea4abae64928824bd`;
  and
- extended-range metrics:
  `research/runs/torch_hero_lr_range_b1024_to1e2_v1/metrics.jsonl`, SHA-256
  `40082e253134b06ba87d307ddb0ef596abb6d9a11f2fe32902dda53d2f428079`.

A guarded, no-checkpoint three-update smoke then passed at batch 1024. Update
zero reproduced root CE `2.870999574661255` and weighted root CE `0.25`
exactly; all loss/gradient values were finite, prefetch hid data preparation
after the first update, and the guard recorded `2.432 GB` peak group RSS with
at least `8.930 GB` host memory available.

The frozen 8,192-position initialization validation then passed. Its update-zero
anchors are:

| Metric | Value |
|---|---:|
| Total weighted loss | `13.432807` |
| All-horizon DFM CE | `7.441815` |
| Root legal-conditional CE | `2.903243` |
| Root legal top-1 | `19.9585%` |
| Root legal mass | `2.3251%` |
| JEPA raw MSE | `0.786685` |
| Target / prediction SIGReg | `0.681952 / 0.661630` |
| WDL CE / Brier | `1.243358 / 0.758591` |

The report is
`research/runs/torch_hero_init_fast_eval_v2/report.json`, SHA-256
`4e0664534c42e350ed30d2f5de3ba618b4fa011df6ed6441c65daee08fd6d49d`.
The evaluation batch remains frozen at 64. A requested batch-256 invocation
failed closed before model loading because changing that batch would change
the per-batch SIGReg/evaluation protocol.

Commit `83f1fef` adds one deterministic CPU prefetch slot without changing
batch order, stochastic choices, or the batch-64 metric contract. A repeated
fast-pool run reproduced every metric exactly, reduced evaluation time from
`345.241` to `268.400` seconds (`1.286x` speedup), and raised mean GPU
utilization from `22.24%` to `28.52%`. Its report is
`research/runs/torch_hero_init_fast_eval_prefetch_v1/report.json`, SHA-256
`7ffcee6a1e519396d3063f5e8a89fb00e56b5fd75a35e8f2a9b6e3ab448d7bb5`.

The disjoint 65,536-position primary initialization gate then passed in
`655.139` seconds. Its anchors are:

| Metric | Value |
|---|---:|
| Total weighted loss | `13.472500` |
| All-horizon DFM CE | `7.443514` |
| Root legal-conditional CE | `2.914379` |
| Root legal top-1 | `20.0485%` |
| Root legal mass | `2.3147%` |
| JEPA raw MSE | `0.786217` |
| Target / prediction SIGReg | `0.693541 / 0.668344` |
| WDL CE / Brier | `1.246055 / 0.759601` |

The primary report is
`research/runs/torch_hero_init_primary_eval_v1/report.json`, SHA-256
`9155673862b578fc1bcf3f9cfe20f496eea7e5ac6da170952a05affce6463dde`.

Exact recovery is also complete. Commit `bfc2106` adds an atomic 1.864 GB
model-plus-optimizer safetensors state with all Muon/AdamW moments, optimizer
and example counters, the next data cursor, checksums, and a strict resume
contract. A guarded batch-1024 regional-compile audit saved after update 1,
resumed update 2, and compared against an uninterrupted two-update control:

- all 462 terminal model tensors and 712,293,144 tensor bytes were bitwise
  identical;
- the canonical terminal state digest was
  `ee59c9be0fa6e847bc628029c5c36e887bfa90507fa7b04f54f45de2152c5b75`;
- every non-timing update-2 metric was identical;
- the recovery write took `13.674` seconds; and
- checkpoint-stage peak process-group RSS was `3.955 GB`, below the fixed
  `7 GiB` ceiling.

The complete compact audit is
`artifacts/pytorch/hero_recovery_audit_v1.json`, SHA-256
`71acc893def5b0ab33c345543dfc78a8fffa23395144b555216f6a20e42ea48c`.
The raw safetensors file hashes differ because the serializer emitted the same
metadata map in a different key order; named tensor content is exactly equal.

Update-zero policy and Arena gates are complete:

- guarded JAX-GPU versus Torch-GPU raw-BT4 inference selected the same legal
  action on all 8 checked canonical positions, while confirming that both DFM
  residual-output tensors are exactly zero; the report is
  `artifacts/pytorch/hero_init_policy_parity_v1.json`, SHA-256
  `d52246dedba06205b82ebf7165e98c138f6c564325cd6acc832acdffd50cbd3f`;
- commit `a5f9b55` adds a raw-BT4 candidate alias and a 16-pair correctness
  tier backed by the exact 2,048-opening hero pool; and
- the color-reversed self-match produced sixteen pair scores of exactly
  `1.0`, candidate score `0.5`, and zero faults over 512 evaluated positions
  per model. The immutable block is
  `artifacts/arena/hero-update-zero-canonical-selfmatch-16pairs-v1/blocks/`
  `block-00000000-pairs-0016.json`, SHA-256
  `25bf5a5b48084330b1ecc64f229ff5e61ef4e939041e7bb3cc1440bffb0f8d43`.

The live milestone instrumentation is now implemented in the one-file Torch
trainer. Fast validation runs from the live model at the first update crossing
each 10% example boundary. The native Torch Arena adapter uses current-only
canonical LC0 planes, a complete root legality mask, fixed batch-16 padding,
and the frozen stable-rank eight-pass decoder. It compares the live DFM model
to a separately restored raw-BT4 policy at 25%, 50%, and 75%, then unloads the
opponent before training resumes. Neither path serializes an intermediate
model.

At batch 1024 the exact milestone updates are:

| Fraction | Update | Measurement |
|---:|---:|---|
| 10% | `2,768` | fast validation |
| 20% | `5,536` | fast validation |
| 25% | `6,920` | 16-pair Arena |
| 30% | `8,304` | fast validation |
| 40% | `11,072` | fast validation |
| 50% | `13,840` | fast validation, 16-pair Arena, recovery state |
| 60% | `16,608` | fast validation |
| 70% | `19,376` | fast validation |
| 75% | `20,760` | 16-pair Arena |
| 80% | `22,144` | fast validation |
| 90% | `24,912` | fast validation |
| 100% | `27,679` | fast validation, then terminal checkpoint |

The training utilization monitor pauses during live evaluation, so evaluation
does not contaminate training-only utilization or throughput. Evaluation wall
time is recorded separately. A resume exactly at a milestone deliberately
repeats that milestone, making a completed halfway recovery state sufficient
even if the original process failed during the subsequent evaluation.

The remaining launch gate is the guarded compiled-shape smoke for one
batch-64 validation step and one native Torch Arena pair. After that passes,
freeze the terminal command and hashes and start the one-epoch job. The epoch
will request only update `13,840` as the sparse recovery state plus one
terminal model checkpoint; the sparse state is deleted only after the
terminal cross-framework audit passes.
