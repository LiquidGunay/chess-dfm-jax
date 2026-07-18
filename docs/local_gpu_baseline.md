# Local A10G baseline

Date: 2026-07-18

This records the first verified local-GPU baseline before any intentional model
or objective change.

## Provenance

- Legacy branch: `legacy/tpu-joint-latent-sasa`
- Research branch: `research/local-gpu-autoresearch`
- Legacy source commit: `15ed0a068b90909ff84c8151eceade0fca74a18c`
- GPU: NVIDIA A10G, compute capability 8.6, 23,028 MiB visible
- Driver: 580.159.03
- Python: 3.11.4
- JAX/JAXLIB: 0.10.1
- Flax: 0.12.7
- Optax: 0.2.8
- NumPy: 2.4.6

The full machine-readable environment is
`artifacts/baselines/a10g_system.json`.

## Verified source assets

### Trajectory data

- Archive size: 11,531,386,880 bytes
- SHA-256:
  `d8feffa259580563c8097fa4a604ca415469dd80da66dabda170ac9ce929b968`
- Extracted shards: 30,726
- Samples: 31,463,424

| Split | Shards | Samples |
|---|---:|---:|
| Train | 27,679 | 28,343,296 |
| Validation | 1,524 | 1,560,576 |
| Test | 1,523 | 1,559,552 |

### Joint checkpoint

- Step: 265,000
- Split parts: 22/22 verified
- Reassembled tar size: 2,124,789,760 bytes
- Reassembled tar SHA-256:
  `525f8bca369921c85ed10a14eee3bc23027b26013c1aba075ad63c61c9dcf39b`
- `state.npz` size: 1,851,704,172 bytes
- `state.npz` SHA-256:
  `16a3c7e77e411a8a7577ff04dac1ca5173ce24ecb343b5fa4938e2d2b5fb8906`

The tar contains the original `checkpoint_state.json`, `run_config.json`, and
metrics history. No metadata was synthesized.

The effective checkpoint config has:

- horizon 8;
- token dimension 256;
- projected state dimension 1024;
- 2 projector layers;
- 4 DFM layers;
- 4 recurrent vector-JEPA layers;
- BF16 compute with FP32 parameters;
- trainable BT4 encoder at learning rate `1e-5`;
- main learning rate `3e-4`;
- target LeJEPA SIGReg coefficient `0.01`;
- prediction SIGReg coefficient `0.0`; and
- value/WDL coefficients `0.0`.

The `sig005` run name is not authoritative: the checkpoint metadata and final
loss arithmetic both show target SIGReg coefficient `0.01` at step 265,000.

### Base BT4

- Models archive size: 1,459,712,000 bytes
- Archive SHA-256:
  `6ad63c183d766818774c9c5ec7dc3523a4473b95fea5b4ae06e92e3f8d7fa231`
- `BT4_exported.pb.gz` SHA-256:
  `61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651`

The joint checkpoint contains the fine-tuned BT4 embedding and encoder but
omits the frozen policy/value/moves-left heads, so the base protobuf remains
required for construction and the raw-BT4 anchor.

## Strict joint-loss fingerprint

Command:

```bash
research/run_gpu.sh .venv/bin/python research/legacy_baseline.py \
  --batch-size 4 \
  --seed 0 \
  --deterministic-t 0.0
```

Input:

- validation shard `val/chunk_000000.npz`;
- shard SHA-256
  `71555a5c59419cc0a5038c02cb2b9cc898b3d5b4372541f2c77b2afda158ffae`;
- batch size 4;
- every DFM token masked (`t=0`);
- strict model-state restoration.

Selected results:

| Metric | Value |
|---|---:|
| Total legacy loss | 4.0333166 |
| DFM CE | 3.2872314 |
| Masked-action accuracy | 0.09375 |
| JEPA positive loss | 0.6962885 |
| Raw JEPA MSE | 0.6048825 |
| Target SIGReg statistic | 5.2958865 |
| Prediction latent norm | 31.8479 |
| Target latent norm | 33.3781 |
| Checkpoint restore | 3.31 s |
| First compile/eval, cold | 64.75 s |
| First compile/eval, persistent cache | 2.16 s |
| Steady evaluation | 0.083 s |

The complete fingerprint is
`artifacts/baselines/legacy_step265000_loss.json`.

## DFM refinement fingerprint

The first eight held-out samples were evaluated at 1, 2, 4, and 8 refinement
passes. First moves were legal in all cases because the sampler masks first-move
logits. No generated eight-ply sequence was fully legal, because later moves are
not masked against the board reached by the generated prefix.

The first pass included compilation, so its timing is not a steady-state
latency measurement. The complete result is
`artifacts/baselines/legacy_refinement_step265000.json`.

## Exact-optimizer training smoke

`research/train.py` strictly restored both model and optimizer state at step
265,000 and completed real A10G updates.

Batch-one observations:

- cold backward compile and first update: 221.18 s;
- persistent-cache startup and first update in a new process: 18.10 s;
- steady update mean over the following two steps: 0.568 s;
- JAX peak live-buffer report: about 4.5 GB;
- CUDA runtime reservation observed by `nvidia-smi`: about 17.2 GB.

This is only a compatibility smoke measurement. Batch one severely
underutilizes the GPU, and the legacy step FLOP/MFU estimator is invalid for the
active vector JEPA.

## Issues confirmed by the baseline

1. Prediction SIGReg exists but is disabled.
2. Target SIGReg scale changes with the number of valid target vectors.
3. Several JEPA baseline/cosine metrics are hard-coded zero placeholders.
4. BF16 legal-probability summation produced `first_legal_mass=1.001581`, making
   the legality loss negative. The clean objective must accumulate in FP32 and
   bound legal mass to `[0, 1]`.
5. The sampler constrains only the first move to be legal in the generated
   position sequence.
6. JEPA predictions do not feed back into DFM action logits at inference.
7. Cold compilation is material; persistent compilation caching is effective
   but still has nontrivial executable-load/startup cost.
8. The legacy loader silently catches corrupt NPZ errors. The research loader
   now fails closed and maps `batch_at(global_step)` deterministically.

## First per-horizon collapse audit

A batch of 16 held-out examples was evaluated with free JEPA rollout and real
zero, identity, target-shuffle, and action-shuffle baselines.

Across horizons 1–8:

- prediction RMS was approximately `0.976–1.005`;
- target RMS was approximately `1.037–1.074`;
- mean prediction feature standard deviation was `0.888–0.916`;
- fifth-percentile prediction feature standard deviation was `0.558–0.603`;
- effective rank was `11.56–12.96`, with a batch-limited maximum of 15;
- prediction-target cosine was `0.740–0.800`;
- prediction MSE was `0.415–0.537`;
- zero MSE was `1.076–1.153`;
- identity MSE was `1.726–2.193`;
- shuffled-target MSE was `1.787–2.000`; and
- action-shuffled MSE was `1.426–2.024`.

This slice does not show current prediction collapse. The strong
action-shuffled gap also confirms that the JEPA transition uses the
action/DFM-hidden conditioning. Prediction SIGReg should therefore be evaluated
as a controlled stability/quality ablation, not assumed to be an unconditional
improvement.

The complete report is
`research/runs/normalized-collapse-eval-step265k-b16/report.json`.

## Corrected objective fingerprint

The research objective now trains against the unscaled ECF discrepancy while
retaining the official count-scaled Epps-Pulley statistic as a diagnostic. The
first-token policy softmax, compact legal gather, and probability sum are
recomputed from the original logits in FP32, so the correction is
gradient-connected and does not require another model forward.

On the same deterministic batch-four validation slice:

| Metric | Value |
|---|---:|
| DFM CE | 3.2872314 |
| FP32 first legal mass | 0.9995466 |
| FP32 legality loss | 0.0004534 |
| Target unscaled discrepancy | 0.1506310 |
| Target official statistic | 5.4227142 |
| Prediction unscaled discrepancy | 0.1407625 |
| Prediction official statistic | 4.5043998 |
| Corrected total loss | 3.9873407 |

The previous BF16 aggregate on this slice exceeded one and was clipped only
after its gradient had already been formed. The new value is computed correctly
inside the forward graph. The complete report is
`research/runs/normalized-fp32-eval-step265k-b4/report.json`.

The hook is inert in legacy mode. A strict rerun retained the exact original
total loss `4.0333166122` and every original component; its report is
`artifacts/baselines/legacy_step265000_loss_after_fp32_hook.json`.

## First active-graph batch sweep

The compatibility trainer strictly restored the step-265,000 model and
optimizer, explicitly compiled the real backward/update executable, and sampled
the GPU every 100 ms after its first update. All rows use the normalized target
plus prediction SIGReg graph with reference count one; these coefficients are
for profiling only and are not the frozen research objective.

Each example encodes nine boards: the current state plus eight future states.

| Physical batch | Mean update (s) | Examples/s | Encoded boards/s | Mean GPU util. | Mean power | Peak live HBM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.316 | 3.16 | 28.5 | 73.9% | 128 W | 4.49 GB |
| 4 | 0.381 | 10.49 | 94.4 | 76.3% | 142 W | 4.50 GB |
| 8 | 0.453 | 17.65 | 158.9 | 73.3% | 163 W | 4.51 GB |
| 16 | 0.660 | 24.25 | 218.2 | 77.6% | 186 W | 4.50 GB |
| 32 | 1.003 | 31.90 | 287.1 | 85.2% | 221 W | 4.51 GB |
| 64 | 1.641 | 39.01 | 351.1 | 90.3% | 248 W | 5.93 GB |

Batch 64 is 12.3 times faster than batch one in examples/s. Its update-time
median/p95 were 1.587/1.867 s and its power p95 was 277 W. Cold explicit
compilation grew from 115 s at batch one to 278 s at batch 64.

XLA reports aggregate FLOPs, traffic, compiled memory, and transcendental
counts. Those static FLOP estimates barely scale with batch because constant
optimizer work and loop accounting dominate, so they are retained as compiler
diagnostics and are not presented as a reliable MFU numerator.

JAX/CUPTI tracing failed on this driver combination with
`CUDA_ERROR_ILLEGAL_ADDRESS`. The trainer therefore uses safe `nvidia-smi`
sampling plus XLA analysis until that compatibility issue is resolved.

## Objective-gradient calibration

A fixed stochastic batch of 64 training examples at the recovered checkpoint
was audited with one forward and 25 sequential reverse sweeps. Symmetric
plus/minus polarization reconstructed exact per-group component-gradient Gram
matrices while keeping only one gradient tree resident. Every Gram was positive
semidefinite to numerical tolerance.

The model has 274,149,702 trainable parameters:

- 195,305,728 in the BT4 backbone;
- 4,705,346 in the DFM/planner;
- 72,031,232 in the projector/JEPA path; and
- 2,107,396 in the currently inactive value/WDL head.

Unweighted scalar values and global gradient norms were:

| Component | Scalar | Global gradient norm | JEPA-group norm |
|---|---:|---:|---:|
| DFM CE | 3.39674 | 74.735 | 0 |
| JEPA positive | 0.45812 | 12.356 | 8.472 |
| Target unscaled SIGReg | 0.04390 | 0.987 | 0.642 |
| Prediction unscaled SIGReg | 0.05213 | 1.541 | 0.450 |
| FP32 legality | 0.12016 | 14.275 | 0 |

With reference count one, coefficient `0.01` is effectively inert. On the JEPA
parameter group, coefficients `0.132/0.396/1.320` make target SIGReg
approximately `1%/3%/10%` of the primary gradient. Prediction SIGReg uses
`0.188/0.565/1.885` for the same ratios.

The first controlled grid therefore rounds the 3% values to:

```text
target_sigreg_coeff in {0.00, 0.40}
pred_sigreg_coeff   in {0.00, 0.57}
sigreg_reference_count = 1
```

These remain a 2×2 ablation, not an assumption that prediction SIGReg helps.
On the JEPA group, target and prediction SIGReg have gradient cosine `-0.460`
and `-0.194` with the positive prediction loss, respectively. Their modest
starting scale is intentional.

The complete audit is
`research/runs/gradient-audit-step265k-b64-v3/gradient_audit.json`.

## First normalized-objective stability run

The first 100-update acceptance run used physical batch 64, target SIGReg
coefficient `0.40`, prediction SIGReg coefficient `0.0`, and fixed reference
count one. It strictly resumed the step-265,000 model and optimizer and trained
on 6,400 examples. Initial and final validation used the same two deterministic
held-out batches (128 examples), so the before/after delta is exact for that
slice but is not yet a population-quality estimate.

The run was numerically and operationally stable:

- every reported scalar was finite, all 100 loss clip scales were one, and the
  optimizer step advanced contiguously from 265,001 through 265,100 (the
  current logger does not expose the apply-if-finite skip counter);
- steady updates averaged `1.624 s` (`39.4` examples/s and `354.8` encoded
  boards/s);
- mean GPU utilization was `88.9%`, mean power was `250.7 W`, and JAX reported
  `5.94 GB` peak live buffers; and
- the 100-update training loop took `169.4 s`, excluding initial evaluation and
  compilation.

It was not a quality win:

| Fixed validation metric | Initial | Final |
|---|---:|---:|
| DFM CE | 4.8472 | 4.9547 |
| Masked-action accuracy | 0.0586 | 0.0693 |
| JEPA positive loss | 0.4897 | 0.4051 |
| Target normalized SIGReg | 0.0475 | 0.0573 |
| FP32 first legal mass | 0.4548 | 0.4700 |

More importantly, target RMS contracted from approximately `1.008–1.016` to
`0.846–0.860`, and prediction RMS contracted from `0.936–0.971` to
`0.881–0.894`. Prediction effective rank remained about `26.4–32.1`, so this is
early scale contraction rather than rank collapse. Because the future target
encoder is jointly trainable (the raw JEPA target is not stop-gradient), lower
positive loss alone can reward shrinking both sides. The target SIGReg
discrepancy worsening at the same time confirms that coefficient `0.40` is too
weak to counter that pressure. The positive-MSE/zero-baseline-MSE ratio also
worsened from `0.418` to `0.472`, and mean prediction-target cosine slipped
from `0.778` to `0.773`; the lower raw MSE is therefore not evidence of better
scale-independent prediction.

This point is rejected rather than promoted. Before the 2×2 prediction-SIGReg
ablation, the target regularizer must be re-anchored at a stronger gradient
fraction (the calibration audit gives `1.32` at 10% and `3.96` at 30% on the
JEPA parameter group), with latent scale and rank as acceptance metrics.

The complete report is
`research/runs/baseline-target040-smoke-b64-100/report.json`.
This compatibility run did not save final weights, so strict final checkpoint
save/resume is a gate before any 30-minute experiment.

## Strict local checkpoint/resume

Commit `f6b9c40` adds a research checkpoint format with:

- atomic publication from a hidden temporary directory under the run;
- state size and SHA-256 verification;
- a semantic resume-contract digest covering model/objective configuration,
  deterministic data schedule, source assets, code, software, and GPU kind;
- complete typed path/container/shape/dtype ABIs for both trainable model and
  optimizer state, checked before either tree is mutated;
- separate optimizer step, absolute research update, and next data cursor; and
- parent-checkpoint lineage plus completed-checkpoint-only discovery/pruning.

Fifteen focused CPU tests pass, including a bit-for-bit comparison of
uninterrupted training against save/reconstruct/resume with cursor-derived RNG.
A real 274M-parameter GPU smoke then saved after one update:

| Item | Result |
|---|---:|
| State payload | 1,851,704,172 bytes |
| Save + hash + atomic publish | 15.82 s |
| Model ABI | 455 leaves / 705,987,352 bytes |
| Optimizer ABI | 830 leaves / 1,145,636,465 bytes |
| Research update / cursor / optimizer | 1 / 1 / 265,001 |

The next process strictly restored the full payload in `6.32 s`, consumed
`data_step=1`, and advanced to research update `2`, next cursor `2`, and
optimizer step `265,002`. The first resumed update took `18.73 s`, primarily
persistent executable load/startup; this is still excluded from steady-state
timing. The source and resumed reports are:

- `research/runs/real-checkpoint-save-b1-step1/report.json`
- `research/runs/real-checkpoint-resume-b1-step2/report.json`

The initial import of the known legacy step-265,000 checkpoint still uses its
older loader. Its exact fingerprint and ABI are verified, but a future cleanup
should route that one-time import through the same preflight machinery.

## Action-codec audit

The official 1,858-move and attention-map tables are correct, but the current
boardless helpers use them with the wrong orientation. Plane encoding mirrors
black-to-move boards into side-to-move coordinates, whereas preprocessing
stored raw absolute UCI indices and legal masks. In 163,840 sampled held-out
moves, every stored label matched the absolute adapter and no black move
matched the canonical adapter; for example, black `c7c5` is stored as index
`1440`, while canonical `c2c4` is `264`.

The recovered DFM action space must therefore be named and preserved as
`legacy_absolute_1858`. Native BT4 policy inference needs a separate,
board-aware `lc0_canonical_1858` adapter that mirrors black moves before lookup.
The two may coexist in an arena, but their logits and legal masks must never be
interchanged.

Promotion handling is also incomplete. LC0 encodes knight promotion with the
ordinary from-to slot and appends suffixes only for queen, rook, and bishop.
The boardless helper rejects white knight promotion and every black promotion.
A scan of 500 shards in each split found only white rank-7-to-8 queen/rook/bishop
promotions, confirming preprocessing selection bias. The canonical adapter
must enumerate board-legal moves, encode all four promotion types, require a
unique match on decode, and fail closed. The legacy adapter cannot fully
represent black underpromotions without changing checkpoint semantics, so the
arena must record that limitation rather than invent a silent mapping.

Commit `ea62aa8` implements both explicit codec IDs without changing any
existing preprocessing call. Its golden tests cover mirrored pawn/knight moves,
castling, en passant, straight and capture promotions for all four pieces,
full legal-mask round trips, invalid indices, unsupported input formats, and
whole-board mirror invariance. The combined codec/trajectory compatibility
suite passes.

## One-file model localization

Commit `4a981ca` moves the complete checkpoint-visible model configuration,
adapters, projector, recurrent JEPA transition, DFM planner, value/WDL head,
optimizer construction, and legacy step wrappers into `research/train.py`.
Only the legacy stage-1 loss remains as a temporary parity oracle.

Three CPU parity tests compare the local and legacy paths with the same seed:

- every configuration field/default;
- every trainable-state path, dtype, shape, and initialized value;
- every Muon/warmup optimizer-state path, dtype, shape, and value;
- encoded tokens/vectors, noisy planner logits and hidden state, and free JEPA
  rollout; and
- the complete legacy loss/auxiliary tree.

All comparisons are exact. A real step-265,000 B4 GPU evaluation then compared
114 validation metrics against the pre-refactor corrected-objective report and
found zero differences. It exactly retained total loss `3.9873406887`, DFM CE
`3.2872314453`, FP32 legal mass `0.9995466471`, target SIGReg
`0.1506309509`, and prediction SIGReg `0.1407624930`. The report is
`research/runs/local-model-parity-step265k-b4/report.json`.

## Next acceptance point

The compatibility harness is not yet open to autoresearch. The next milestone
is a clean loss definition in `research/train.py` that:

- matches this fingerprint in compatibility mode;
- computes legal mass correctly;
- replaces batch-scaled training SIGReg with the normalized definition;
- computes real collapse/baseline diagnostics; and
- activates a controlled prediction-SIGReg ablation.
