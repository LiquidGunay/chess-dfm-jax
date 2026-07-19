# Local A10G baseline

Started: 2026-07-18

Last updated: 2026-07-19

This records the first verified local-GPU baseline before any intentional model
or objective change.

Current disposition: v2/update 300 is the repeat-qualified offline baseline.
It is not an accepted autoresearch result or promoted checkpoint, and
`AUTORESEARCH_READY` remains false.

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

## Validation-sampling correction

All objective studies below that predate commit `8c2dcb6` used a shard-major
validation iterator with `shuffle_files=false`. The nominal validation seed was
therefore inert: `eval_batches=2` at batch 64 always selected the first 128
rows of `val/chunk_000000.npz`, while the later `eval_batches=16` expansion
covered all 1,024 rows of that same shard. The batch-16 collapse audit also
started from this shard.

These results remain useful as paired calibration and forensic evidence:
they found scale contraction, tested gradient semantics, checked collapse
instrumentation, and measured hardware cost on identical inputs. They are not
representative validation estimates and must not be used to freeze an
objective, claim a quality improvement, or authorize promotion. The smaller
loss fingerprints above remain valid parity oracles, not quality estimates.
Accordingly, terms such as “best,” “rejected,” and “improved” in the
pre-correction sections below describe only that controlled first-shard slice.

Commit `8c2dcb6` adds a deterministic `global_permutation` schedule over every
`(shard, batch_in_shard)` slot and makes it the validation default. The
schedule is seeded, stateless across calls and resume, covers every slot
exactly once per epoch, records its semantics in provenance/resume contracts,
and fails closed if global permutation is requested without shuffling.
Training retains its cache-friendly shard-major schedule. The first corrected
comparison uses 32 globally permuted batch-64 slots (2,048 examples) from the
1,524-shard validation split; it is reported under
[Fixed unit-RMS and corrected global validation](#fixed-unit-rms-and-corrected-global-validation).

## First normalized-objective calibration run

The first 100-update calibration run used physical batch 64, target SIGReg
coefficient `0.40`, prediction SIGReg coefficient `0.0`, and fixed reference
count one. It strictly resumed the step-265,000 model and optimizer and trained
on 6,400 examples. Initial and final validation used the same two deterministic
held-out batches (128 examples), so the before/after delta is exact for that
first-shard slice but is not a population-quality estimate.

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

### Target-SIGReg 1.32 calibration point

The calibrated 10%-gradient point also fails the scale gate. A second
batch-64, 100-update run used target coefficient `1.32`, prediction coefficient
zero, reference count one, and the same two fixed validation batches:

| Fixed validation metric | Initial | Final |
|---|---:|---:|
| DFM CE | 4.8472 | 5.0349 |
| Masked-action accuracy | 0.0586 | 0.0605 |
| JEPA positive loss | 0.4897 | 0.4041 |
| Target normalized SIGReg | 0.0475 | 0.0556 |
| FP32 first legal mass | 0.4548 | 0.4149 |

Mean target RMS contracted from `1.0105` to `0.8599`; mean prediction RMS
contracted from `0.9611` to `0.8985`. Mean prediction effective rank remained
healthy (`29.44` to `29.14`) and mean prediction-target cosine improved
slightly (`0.7782` to `0.7808`), but positive-MSE/zero-MSE worsened from
`0.4180` to `0.4562`. The regularizer discrepancy itself again increased.
This is still a target-scale shortcut, not prediction-rank collapse, and
coefficient `1.32` is rejected.

The local loss completed 6,400 examples at `39.1` examples/s and `351.5`
encoded boards/s. Mean GPU utilization was `89.1%`, mean power was `246.4 W`,
peak JAX live memory was `5.94 GB`, and scalar loss clipping was never active.
The next justified calibration point is `3.96`, whose initial target-SIGReg
gradient is approximately 30% of the JEPA-positive gradient. The report is
`research/runs/target132-smoke-b64-100/report.json`.

### Target-SIGReg 3.96 and 5.76 calibration points

The same paired run at coefficient `3.96` slowed target contraction but did not
stop it. Mean target RMS ended at `0.8943`, mean prediction RMS at `0.9149`,
and normalized target SIGReg still worsened from `0.0475` to `0.0503`.
Prediction rank stayed healthy, cosine improved from `0.7782` to `0.7858`,
accuracy improved to `0.0713`, and legal mass improved to `0.5246`, but DFM CE
worsened to `5.1425`. This point also fails the scale gate. Its report is
`research/runs/target396-smoke-b64-100/report.json`.

Coefficient `5.76` is the normalized equivalent of the original coefficient
`0.01` for a fully valid local batch of 64 (`64 * 9 * 0.01`). It produced the
best policy-side result on this first-shard calibration slice:

| Fixed validation metric | Initial | Final |
|---|---:|---:|
| DFM CE | 4.8472 | 4.8106 |
| Masked-action accuracy | 0.0586 | 0.0732 |
| FP32 first legal mass | 0.4548 | 0.5039 |
| Mean prediction-target cosine | 0.7782 | 0.7854 |
| Mean prediction effective rank | 29.44 | 30.01 |

It still did not remove the shortcut. Mean target RMS fell from `1.0105` to
`0.9078`, target feature standard deviation fell from `0.9705` to `0.8719`,
and target SIGReg worsened from `0.0475` to `0.0491`. Prediction RMS fell to
`0.9186`; positive-MSE/zero-MSE worsened slightly from `0.4180` to `0.4285`.
The action-shuffled/positive-MSE ratio improved from `3.90` to `4.00`, so
action dependence and rank are not collapsing.

This is the strongest compatibility point, but it is not yet the frozen clean
objective. The next controlled experiment changes target-gradient semantics:
stop the JEPA positive-loss gradient through future target vectors while
retaining target-SIGReg gradients into the encoder/projector. This directly
removes the self-shrinking target shortcut without conflating it with
prediction-SIGReg.

The `5.76` run processed 6,400 examples at `38.9` examples/s. Mean GPU
utilization was `86.4%`, mean power was `242.5 W`, peak JAX live memory was
`6.04 GB`, and scalar loss clipping was never active. Its report is
`research/runs/target576-smoke-b64-100/report.json`.

### Target detachment and prediction-SIGReg ablations

Commit `c30a69e` adds a default-off, explicitly recorded positive-target
stop-gradient option. It detaches only the target used by JEPA positive
MSE/norm loss; target SIGReg remains attached to `z_all`. Tests prove exact
legacy parity when disabled, unchanged forward values when enabled, a changed
full trainable gradient, and nonzero target-SIGReg projector gradients under
detachment.

Two paired 100-update detach runs show a clear tradeoff:

| Endpoint | Attached 5.76 | Detached 5.76 | Detached 1.32 |
|---|---:|---:|---:|
| Mean target RMS | 0.9078 | 0.9531 | 0.9667 |
| Mean prediction RMS | 0.9186 | 0.9370 | 0.9520 |
| DFM CE | 4.8106 | 5.0034 | 4.9922 |
| Accuracy | 0.0732 | 0.0674 | 0.0605 |
| Mean cosine | 0.7854 | 0.7538 | 0.7656 |
| Positive-MSE / zero-MSE | 0.4285 | 0.4786 | 0.4554 |
| Action-shuffled / positive-MSE | 4.00 | 3.54 | 3.71 |

Detachment substantially preserves scale, and `5.76` detached improves target
SIGReg itself to `0.0449`, but both points weaken prediction quality, action
sensitivity, and policy loss over 100 updates. Simple detachment is therefore
not promoted. A slower EMA target is the next target-semantics candidate.
Reports:

- `research/runs/target576-stopgrad-smoke-b64-100/report.json`
- `research/runs/target132-stopgrad-smoke-b64-100/report.json`

The requested prediction-SIGReg test kept the target attached, used target
coefficient `5.76`, and added the calibrated 10%-gradient prediction
coefficient `1.885`. It modestly moved mean prediction RMS from the attached
endpoint's `0.9186` to `0.9266` and feature standard deviation from `0.8744`
to `0.8825`. Rank (`30.08`) and action-shuffle separation (`4.00`) remained
healthy, but DFM CE worsened to `4.9898`, accuracy fell to `0.0654`, cosine
slipped to `0.7847`, and positive-MSE/zero-MSE worsened to `0.4321`.
Throughput was unchanged within run noise.

This point is also rejected. The current checkpoint does not exhibit
prediction rank/variance collapse, and marginal prediction Gaussianity is not
the main bottleneck. The report is
`research/runs/target576-pred1885-smoke-b64-100/report.json`.

### Controlled EMA-target ablation (first-shard calibration)

Commits `2667829`, `4ca012d`, and `fce9783` add an explicit EMA target
semantics without changing the default online path. The positive JEPA target is
always detached and comes from the EMA teacher, while normalized target SIGReg
remains attached to the online current-and-future `z_all`. With decay `0.99`,
the authoritative teacher is updated after each online optimizer step:

```text
teacher = 0.99 * teacher + 0.01 * online_updated
```

The teacher keeps a headless FP32 encoder master plus FP32 projector and state
normalization state. A physically independent native-BF16 encoder mirror is
used for the forward pass and refreshed after source initialization, every EMA
update, and checkpoint restore. This split is required for exact A10G compute
parity: the two rejected forensic prototypes
`ema-target099-a10g-b1-smoke-v1` and
`ema-target099-a10g-b1-parity-v2` produced initial online/teacher MSE
`0.02834` and `0.03129`, respectively, because FP32 teacher buffers changed
the GPU compute path even when cast at their read sites. The final
master-plus-mirror implementation has exactly zero initial MSE at every
horizon and identical initial RMS and norms. Its batch-one report is
`research/runs/ema-target099-a10g-b1-master-mirror-v3/report.json`
(SHA-256
`3b0dbcc9bffb578a25cd0bd057c2a7b9ff528196def5b8def16fccef60ffc58d`).

Only the authoritative master/projector/normalization state is checkpointed:
`888,558,592` bytes. The derived BF16 compute mirror is excluded and rebuilt,
giving `1,279,170,048` bytes of total runtime teacher state. The encoder master
is `781,222,912` bytes, its mirror is `390,611,456` bytes, and refreshing it
reads/writes `1,171,834,368` bytes per update. The extra teacher forward
encodes the eight future boards, so the active graph processes 17 boards per
example rather than nine.

On the A10G batch-64 profile, EMA sustained `32.20` examples/s versus `38.94`
for attached-online target `5.76`, a `17.3%` throughput cost. JAX peak live
memory rose from `6,036,286,464` to `7,053,931,776` bytes, about `1.02 GB`.
The EMA profile averaged `88.1%` GPU utilization. Its report is
`research/runs/ema-target099-a10g-b64-profile-v1/report.json` (SHA-256
`6e3d2bd88e10051ed179866a94af2e045bf0fd3fab80aa657398de92834014ed`).

The paired 100-update run retained target SIGReg `5.76`, prediction SIGReg
zero, the same 6,400 training examples, and the same two fixed validation
batches as the attached-online run:

| Fixed-slice result | Attached online 5.76 | EMA 0.99 + 5.76 |
|---|---:|---:|
| Mean positive-target RMS, final / initial | 0.9078 / 1.0105 | 0.9865 / 1.0105 |
| Mean prediction RMS, final / initial | 0.9186 / 0.9611 | 0.9487 / 0.9611 |
| Mean prediction effective rank, final / initial | 30.01 / 29.44 | 30.01 / 29.44 |
| DFM CE, final (delta) | 4.8106 (-0.0366) | 4.9219 (+0.0747) |
| Accuracy, final (delta) | 0.0732 (+0.0146) | 0.0498 (-0.0088) |
| First legal mass, final (delta) | 0.5039 (+0.0491) | 0.4132 (-0.0416) |
| Mean prediction-target cosine, final (delta) | 0.7854 (+0.0072) | 0.7704 (-0.0078) |
| Action-shuffled / positive MSE, final (initial 3.90) | 4.00 | 3.79 |

EMA passes the provisional scale/rank gates: target RMS retains `97.62%`,
prediction RMS `98.71%`, and prediction effective rank `101.94%` of their
initial means. It nevertheless regresses policy metrics, cosine alignment, and
action-shuffle separation on this slice; the online/EMA target MSE also grows
from zero to `0.04584` as intended. The 100-update run sustained `33.28`
examples/s with `91.1%` mean GPU utilization and `7,116,167,680` bytes peak
JAX memory. EMA is therefore a valid target-semantics experiment, but it is not
promoted or frozen on its own. This motivated the per-horizon variance-hinge
comparison below. The complete report is
`research/runs/ema-target099-target576-b64-100-v1/report.json` (SHA-256
`e0020ccd1395a8d6c5b4faa8db30f6323eb1bddabcc3d833951be56d80872fd3`).

### Per-horizon target variance hinge (first-shard calibration)

Commit `134b074` adds a default-off variance hinge on attached online projected
future targets. For each horizon independently, it computes weighted FP32
population variance across the batch with `valid * future_valid`, then applies

```text
std[h,d] = sqrt(max(var[h,d], 0) + 1e-4)
hinge[h] = mean_d relu(gamma - std[h,d])
```

Only horizons with at least two valid examples are eligible, and eligible
horizons receive equal weight. The population denominator makes the loss
exactly invariant to duplicating a batch. Enabled-only diagnostics record
counts, eligibility, mean/fifth-percentile/median feature standard deviation,
active-feature fraction, and hinge per horizon. The gradient remains attached
to both the BT4 backbone and state projector. Focused tests cover the formula,
NaN padding and ineligible horizons, duplication invariance, masked gradients,
backbone/projector connectivity, metrics and gradient-audit ABI, fail-closed
configuration, and disabled serialization compatibility.

A zero-update batch-64 screen selected `gamma=0.85`: it initially activates
`20.5–24.9%` of features at each horizon, within the predefined 25% ceiling.
`gamma=0.90` activates `31.3–34.9%` and fails. Active fraction is monotone in
gamma, so `0.95` and `1.00` are ruled out without redundant runs. The reports
are:

- `research/runs/variance-hinge-gamma085-b64-eval-v1/report.json`
  (SHA-256
  `735ad5d190527f2c7ccf3b5f760f1301678f5a94c71a2e9a68988a56cf02638d`);
- `research/runs/variance-hinge-gamma090-b64-eval-v1/report.json`
  (SHA-256
  `9d4213fed558975539059cd827f3af5ff559733b0db193627b634e804801d081`).

The batch-64 gradient audit calibrated the hinge against the configured
JEPA-positive plus `5.76` target-SIGReg gradient. On the JEPA group, 3%, 10%,
and 30% fractions suggest coefficients `0.373`, `1.243`, and `3.728`;
all-parameter and backbone suggestions are nearly identical at
`0.385/1.284/3.851`. The run grid rounded these to `0.38`, `1.25`, and `3.75`,
then added `1.75` to locate the scale-gate boundary. The audit is
`research/runs/gradient-audit-variance-hinge-gamma085-target576-b64-v1/gradient_audit.json`
(SHA-256
`36f2cb0d6425a84ff8c72b8d9981a4144f6c166b67da61b16c57cf8cf4be5f3c`).

Each hinge run used online targets, `gamma=0.85`, target SIGReg `5.76`,
prediction SIGReg zero, batch 64, the same 6,400 training examples, and the
same two fixed validation batches. RMS and rank columns are final/initial
ratios. The two coupling columns are final positive-MSE/zero-MSE (lower is
better) and action-shuffled-MSE/positive-MSE (higher is better); their common
initial values are `0.418` and `3.90`.

| Objective | Target RMS | Pred RMS | Pred rank | DFM CE delta | Accuracy delta | Legal-mass delta | Pos/zero | Action/pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Online 5.76 | 89.84% | 95.58% | 101.95% | -0.0366 | +0.0146 | +0.0491 | 0.428 | 4.00 |
| EMA 0.99 + 5.76 | 97.62% | 98.71% | 101.94% | +0.0747 | -0.0088 | -0.0416 | 0.437 | 3.79 |
| Hinge 0.38 + 5.76 | 91.18% | 96.76% | 99.24% | +0.2347 | +0.0039 | -0.0125 | 0.414 | 4.13 |
| Hinge 1.25 + 5.76 | 94.11% | 98.11% | 99.59% | +0.0805 | +0.0088 | +0.0543 | 0.395 | 4.28 |
| Hinge 1.75 + 5.76 | 95.52% | 98.65% | 99.24% | +0.2351 | +0.0049 | -0.0196 | 0.392 | 4.25 |
| Hinge 3.75 + 5.76 | 100.05% | 100.33% | 100.26% | +0.0511 | +0.0068 | -0.0518 | 0.384 | 4.26 |

| Objective | Examples/s | Peak JAX bytes |
|---|---:|---:|
| Online 5.76 | 38.94 | 6,036,286,464 |
| EMA 0.99 + 5.76 | 33.28 | 7,116,167,680 |
| Hinge 0.38 + 5.76 | 39.40 | 5,948,592,896 |
| Hinge 1.25 + 5.76 | 39.12 | 5,948,717,056 |
| Hinge 1.75 + 5.76 | 39.58 | 5,979,598,592 |
| Hinge 3.75 + 5.76 | 39.53 | 5,945,558,272 |

Coefficient `0.38` is too weak. `1.25` improves both coupling ratios and policy
accuracy/legal mass, but just misses the 95% target-scale gate. `1.75` is the
smallest measured coefficient that passes the scale gate, yet its DFM CE and
legal mass regress. `3.75` fully stabilizes target/prediction scale and rank
and retains strong action coupling, but also loses DFM CE and legal mass
relative to the attached-online point. The hinge has no measurable training
cost, but no coefficient passes scale, coupling, and policy/legal gates
together. None is promoted or frozen, and the harness remains
`autoresearch_ready=false`.

The four reports and SHA-256 digests are:

- `research/runs/variance-hinge-g085-c038-target576-b64-100-v1/report.json`:
  `68c5713962e96ccfd20bb42d578c6d33f382ae98f3314fc9780478642a02fb22`;
- `research/runs/variance-hinge-g085-c125-target576-b64-100-v1/report.json`:
  `584f859379b0b2e9424567e1e45a996a863ff1a2f567d78cc459fb9cd40f0343`;
- `research/runs/variance-hinge-g085-c175-target576-b64-100-v1/report.json`:
  `1bfd6798756320a2493abd316db886959a735291cddecb5222cb78abc7aa7b8b`;
- `research/runs/variance-hinge-g085-c375-target576-b64-100-v1/report.json`:
  `90a4b4329f9e409b4b93517ba5e2c1f967c2f5308a460793102bca4f57f58b3c`.

All of those objectives have `jepa_state_rmsnorm=false`. The reported inactive
state-scale value remains `0.986831` while actual target and prediction RMS can
contract. This motivated the fixed-unit-RMS ablation below.

### Fixed unit-RMS and corrected global validation

Commit `8257881` adds a default-off, non-trainable JEPA state manifold:

```text
unit_rms(z) = z / sqrt(mean_d(z^2) + 1e-6)
```

Statistics are computed in FP32 and the result returns to the model compute
dtype. Normalization is applied to the projected current state, every projected
future target, and every recurrent JEPA prediction. The historical scale
parameter remains in the checkpoint ABI but is not read and has exactly zero
loss gradient. Initial experiments require online attached targets and reject
combinations with EMA, target detachment, the variance hinge, or the old
trainable RMSNorm. Tests cover exact disabled-path compatibility, all three
application sites in BF16, scale invariance with attached gradients, zero scale
gradient with nonzero model gradients, fail-closed combinations, CLI
serialization, and the resume contract.

The source checkpoint then has target/prediction RMS within `0.00004` of one at
every horizon and latent norms approximately `32`, as expected for 1,024
features. The zero-update report is
`research/runs/fixed-unit-rms-target576-b64-eval-v1/report.json` (SHA-256
`bd9ce52e891d02f456537e6e903ffbf16b2b4172c2ddf9c3011c57d87972f201`).

On the same batch-64 gradient audit, target-SIGReg coefficients
`0.150/0.450/1.499/4.497` correspond to approximately 1%/3%/10%/30% of the
JEPA-positive gradient on the JEPA parameter group. The calibration rounded
these to a target-SIGReg grid of `0`, `1.50`, `4.50`, and the compatibility
point `5.76`. Its audit is
`research/runs/gradient-audit-fixed-unit-rms-target576-b64-v1/gradient_audit.json`
(SHA-256
`7ac290852141deb7fc689c5848a67ad887dd9f9817c478507740dee4b0bfd1bd`).

The original two-batch first-shard screen was:

| Target SIGReg | Target RMS ratio | Pred RMS ratio | DFM CE delta | Accuracy delta | Legal-mass delta | Pos/zero final | Action/pos final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 100.00% | 100.00% | +0.2650 | +0.0107 | +0.0420 | 0.406 | 4.38 |
| 1.50 | 100.00% | 100.00% | +0.0358 | +0.0049 | +0.0695 | 0.427 | 4.17 |
| 4.50 | 100.00% | 100.00% | +0.0828 | +0.0059 | +0.0275 | 0.428 | 4.17 |
| 5.76 | 100.00% | 100.00% | +0.1353 | +0.0195 | +0.0790 | 0.406 | 4.40 |

This selected `1.50` only as the least policy-disruptive calibration point.
Expanding it to 16 batches changed DFM CE by `+0.6554`, accuracy by `-0.0272`,
and legal mass by `-0.1442`, revealing strong within-shard heterogeneity but
still sampling only `chunk_000000.npz`. Neither screen is promotion evidence.
Their report hashes are:

- target `0`: `research/runs/fixed-unit-rms-target000-b64-100-v1/report.json`,
  SHA-256
  `16359fc943ea8991af4e681b0fb3211eed498c89e261daf6af365d796216e32c`;
- target `1.50`: `research/runs/fixed-unit-rms-target150-b64-100-v1/report.json`,
  SHA-256
  `23c36a7d04f92aadc932a89ec04995b9eb41071bf34983bf83d73def873f1018`;
- target `4.50`: `research/runs/fixed-unit-rms-target450-b64-100-v1/report.json`,
  SHA-256
  `dd0b8ea57e7fd8bee84557cc4726f56a71c64172abd7cf1652a68b747df86633`;
- target `5.76`: `research/runs/fixed-unit-rms-target576-b64-100-v1/report.json`,
  SHA-256
  `2ef43adfb61041050a5098853e0757a1e561816a7f36b94e28bba8540de76905`;
- target `1.50`, 16 batches:
  `research/runs/fixed-unit-rms-target150-b64-100-val16-v1/report.json`,
  SHA-256
  `9cbf05fd53978604b1750aa32ec6b0be4d4cdfc7cf698e827d090e503aef1fee`.

After the schedule correction in `8c2dcb6`, online target-SIGReg `5.76` and
fixed-unit-RMS target-SIGReg `1.50` were rerun from the same source checkpoint
through the same 100 training updates. Both use the same seed and 32 globally
permuted validation batches before and after training. Policy metrics therefore
share an exact initial value:

| Global validation metric | Common initial | Online 5.76 final (delta) | Unit RMS + 1.50 final (delta) |
|---|---:|---:|---:|
| DFM CE | 4.655153 | 5.151332 (+0.496179) | 5.159355 (+0.504202) |
| Accuracy | 0.101868 | 0.065613 (-0.036255) | 0.065857 (-0.036011) |
| First legal mass | 0.618943 | 0.515743 (-0.103201) | 0.514597 (-0.104347) |

The latent comparison is:

| Global validation metric | Online 5.76 initial → final | Unit RMS + 1.50 initial → final |
|---|---:|---:|
| JEPA positive loss | 0.405311 → 0.369331 | 0.386153 → 0.371684 |
| Target SIGReg | 0.044931 → 0.048551 | 0.045329 → 0.048077 |
| Mean target RMS | 0.993857 → 0.903679 | 1.000011 → 0.999991 |
| Mean prediction RMS | 0.951802 → 0.909740 | 0.999999 → 0.999995 |
| Mean prediction effective rank | 30.702 → 31.350 | 30.679 → 30.939 |
| Mean prediction-target cosine | 0.814093 → 0.803727 | 0.807066 → 0.814280 |
| Positive MSE / zero MSE | 0.351044 → 0.386967 | 0.385864 → 0.371439 |
| Action-shuffled / positive MSE | 4.74477 → 4.48086 | 4.61511 → 4.76239 |

Fixed unit RMS removes uniform scale contraction and ends with better cosine
and coupling ratios than the online control. It does not improve the matched
global policy result: final DFM CE is `0.0080` worse, legal mass is `0.00115`
worse, and accuracy is only `0.00024` higher. Its final JEPA positive loss is
`0.00235` higher, although that raw value is scale-confounded across the two
manifolds. Both configurations substantially regress global DFM CE, accuracy,
and legal mass over these 100 updates, overturning the apparent policy
improvement on the old two-batch slice.

Throughput is effectively equal (`39.70` versus `39.55` examples/s), and peak
JAX memory is `6,089,154,304` versus `6,043,895,552` bytes. The corrected
reports are:

- `research/runs/online-target576-b64-100-globalval32-v1/report.json`
  (SHA-256
  `50f99c2c60241f154d46f4d9391f83584cae0a4cb6e1e271cffd85eceeda3fdf`);
- `research/runs/fixed-unit-rms-target150-b64-100-globalval32-v1/report.json`
  (SHA-256
  `c5ee7c4234fabbf6757675a8c7fd249c923e436ad38db3eef33d1dff00a0fa82`).

The 32-batch result is a corrected deterministic calibration sample, not a
promotion tier. It rejects fixed unit RMS `+1.50` as the new baseline, but it
also shows that online `5.76` cannot be frozen from the available evidence.
No normalized objective is frozen, `autoresearch_ready=false`, and the next
objective comparison must use matched global validation from its first run.

## Continuation audit and provisional local optimizer baseline

The first corrected global-validation result exposed a second confound: the
source optimizer was built for a TPU global batch of `8,192`, while the first
local continuation used physical/global batch `64` with no accumulation. The
local batch is therefore 128 times smaller. The source configuration used main
and BT4 learning rates `3e-4` and `1e-5`; applying them unchanged to batch 64
is not a batch-matched continuation.

The word “exact” is also deliberately narrow at the legacy import boundary.
`--init exact` restores the trainable model, optimizer state, and optimizer
step `265,000`. The legacy payload has no input cursor or PRNG stream, so the
local run starts at local cursor zero and derives randomness from the local
seed and cursor. This is an exact state import, not a continuation of the TPU
stochastic process. In contrast, a subsequent local research checkpoint
records `next_data_cursor`; its RNG is derived from the resume-bound seed and
that cursor.

The source itself had already moved past its best recorded validation point.
All nine validation records present in the source history span steps
225,000–265,000:

| Source step | Validation total | DFM CE | Accuracy | First legal mass |
|---:|---:|---:|---:|---:|
| 250,000 | **5.925633** | 4.508655 | 0.108962 | 0.640421 |
| 255,000 | 5.938301 | **4.503008** | **0.109890** | **0.644553** |
| 260,000 | 5.949284 | 4.509257 | 0.109631 | 0.644534 |
| 265,000 | 6.106098 | 4.541201 | 0.107515 | 0.643071 |

Thus step 265,000 is `+0.038193` CE beyond the recorded CE minimum, and
accuracy/legal mass also peaked at step 255,000. Absolute source values are not
compared to the corrected local validation protocol; the within-run history is
used only to show that inheriting the last optimizer moments is not a neutral
baseline choice. The source `run_config.json` and `metrics.jsonl` SHA-256
digests are
`f4f44c673322fbb04367db7b28fcc82c30f89fc12340e66322607d53bbf99555`
and
`bf62337ae6c74281fe790fbc743fe030dd8f0a28813f7590aa9e73716b8c67fc`.

### Global training order and optimizer-isolation experiments

Commit `42bf65c` exposed a seeded `global_permutation` training schedule and
explicit main/BT4 learning-rate overrides. The following runs all trained on
the same 6,400 examples and used the same 2,048-position, seed-10,000 global
validation slice before and after. Deltas are final minus initial:

| Initialization/objective | Main / BT4 LR | Total loss | DFM CE | Accuracy | Legal mass | Disposition |
|---|---:|---:|---:|---:|---:|---|
| Exact optimizer, full objective | `3e-4` / `1e-5` | +0.725222 | +0.472953 | -0.029663 | -0.139341 | reject |
| Exact optimizer, full objective | `3e-5` / `1e-6` | -0.016040 | +0.014015 | -0.002930 | -0.007042 | reject |
| Exact optimizer, policy only | `3e-5` / `1e-6` | +0.050695 | +0.025827 | -0.004822 | -0.012434 | reject |
| Fresh optimizer, full objective | `3e-5` / `1e-6` | -0.080489 | -0.005833 | -0.000671 | +0.004513 | provisional |

The policy-only total uses a different objective and is comparable only
within that row, not to the full-objective totals.

The source-rate run shows that merely correcting training order does not
remove the policy collapse. Reducing both rates by ten is close to square-root
batch scaling for the 128-fold batch reduction, but the inherited optimizer
still makes policy CE, accuracy, and legal mass worse even while the mutable
weighted loss improves. This is direct evidence that training loss alone
cannot select the baseline.

Commit `1fca0f0` then disabled both JEPA positive loss and target SIGReg to
isolate policy-only continuation. It regressed every policy-side metric more
than the matched lower-rate full objective and was reverted by `5e8e7b1`.
Policy-only continuation is therefore rejected, not retained as the clean
baseline.

Commits `d8ab42c` and `0803f06` isolate local optimization instead. Model-only
initialization still validates the complete source optimizer ABI but starts a
new optimizer at step zero. `lr_warmup_steps=0` uses constant stateful
schedules, preserving optimizer-tree compatibility without spending the first
local updates near zero learning rate. At batch 64, the fresh-optimizer
candidate was then evaluated on 4,096 positions at two validation seeds:

| Validation seed | DFM CE delta | Accuracy delta | Legal-mass delta |
|---:|---:|---:|---:|
| 10,000 | -0.000766 | -0.000610 | +0.001746 |
| 20,000 | -0.000672 | +0.001221 | +0.004401 |

The CE direction repeats, but its magnitude is tiny. The same run retains
`94.30%` of mean target RMS after 100 updates, below the provisional 95% scale
gate, although prediction rank, cosine, and action coupling remain healthy.
Fresh local optimizer state is the best initialization candidate found here;
the batch-64 endpoint is not a frozen objective or promoted checkpoint.

### Row-sliced random loading and honest throughput

Commit `bf18a5b` slices encoded trajectory rows before plane expansion and
legal-mask construction for globally permuted batches. It also separates
accelerator-only update throughput from fetch-inclusive and whole-loop
throughput:

```text
device examples/s = batch / update time
end-to-end examples/s = batch / (fetch + update time)
training-wall examples/s = all examples / measured loop wall time
```

An isolated B64-from-B1024 loader benchmark measured about `245 MiB` less peak
host RSS and roughly 17% lower decode time than expanding the full shard.
This is not compressed random access: NumPy still inflates an accessed NPZ
member before slicing, but the expensive plane conversion and legal-mask
expansion now operate only on the selected rows.

The old `steady_examples_per_second` excluded host input time and could hide a
large stall. Reconstructing the new definitions from old JSONL records and
comparing them with post-change reports gives:

| Loader/run | Batch | Device ex/s | End-to-end ex/s | Input stall | Fetch p95 |
|---|---:|---:|---:|---:|---:|
| Pre-slice, source LR | 64 | 39.69 | 32.58 | 17.9% | 0.429 s |
| Pre-slice, lower LR | 64 | 39.17 | 20.50 | 47.7% | 3.137 s |
| Row-sliced, three runs | 64 | 39.52–39.75 | 32.88–33.07 | 16.7–16.9% | 0.348–0.355 s |
| Row-sliced, three runs | 128 | 44.68–44.77 | 41.30–41.42 | 7.48–7.57% | 0.240–0.248 s |

The pre-slice lower-rate run demonstrates why update-only throughput is not a
hardware-utilization result: it reported `39.17` examples/s while delivering
only `20.50` examples/s across fetch plus update. Post-change batch-64 results
are stable near `33.0` end-to-end examples/s. Batch 128 raises that to about
`41.4` examples/s, a roughly 25% gain, raises mean sampled GPU utilization
from about 77.6% to 89.0%, and reduces the measured input-stall fraction by
more than half. Its cost is roughly `9.70 GB` peak JAX live memory versus
`6.01 GB` at batch 64.

### Fixed validation population and batch-128 learning rate

Commit `455a806` adds `--eval-batch-size`, decoupling validation partitioning
from physical training batch. Every batch-128 comparison below therefore uses
exactly 64 validation batches of 64 examples: the same 4,096 seed-10,000
positions and the same finite-batch partition as the batch-64 control. This is
required because normalized SIGReg is duplication-invariant but remains a
non-additive finite-sample statistic; changing evaluation batch partition
would change the reported statistic even at fixed positions.

All batch-128 runs consume the same 6,400-example budget as 100 updates at
batch 64. Because changing batch size changes the global slot partition, this
is a fixed-example-count comparison, not a claim that batch 64 and batch 128
read identical training rows:

| Batch-128 run | Main / BT4 LR | DFM CE delta | Accuracy delta | Legal-mass delta | Decision |
|---|---:|---:|---:|---:|---|
| Linear-scaled LR | `6e-5` / `2e-6` | +0.021454 | -0.003479 | -0.016390 | reject |
| Unscaled LR, repeat A | `3e-5` / `1e-6` | -0.005654 | -0.000793 | -0.003962 | provisional |
| Unscaled LR, repeat B + diagnostics | `3e-5` / `1e-6` | -0.001716 | -0.000885 | -0.005235 | provisional |

Doubling the learning rates with batch size is decisively rejected. The two
unscaled runs have the same model initialization, optimizer, seed, training
slots, example count, and fixed validation positions. Their first update
metrics are identical, but their training metrics diverge at update two and
their final DFM CE differs by `0.003938`. The only intentional computational
difference was whether read-only collapse diagnostics were computed during
validation. This proves that the current comparison procedure is not
bit-for-bit repeatable under that perturbation, but it does not isolate
accelerator nondeterminism from a hidden state, compilation/cache effect, or
another cause. The two metrics-log hashes are respectively
`01c674f2e0c84084ebb0b40321b9fee26c697151484e5570260c27886750f582`
and
`580225d322108231ee16dbc18bb01968737d63dbd13573882803962d6a913dac`.
The final CE spread is comparable to the apparent CE improvement, so it is the
current empirical noise floor rather than evidence for ranking the repeats.

The diagnostic repeat gives the complete latent gate:

| Gate metric | Initial | Final | Result |
|---|---:|---:|---|
| Mean target RMS | 0.995286 | 0.943101 (94.76%) | misses 95% retention |
| Mean prediction RMS | 0.955479 | 0.932835 (97.63%) | pass |
| Mean prediction effective rank | 31.429 | 31.559 (100.41%) | pass |
| Mean prediction feature std. | 0.909275 | 0.889949 (97.87%) | pass |
| Minimum horizon prediction std. p05 | 0.679048 | 0.662265 | pass |
| Mean prediction-target cosine | 0.814534 | 0.829621 | improves |
| Positive MSE / zero MSE | 0.350225 | 0.329258 | improves |
| Action-shuffled MSE / positive MSE | 4.79731 | 5.21775 | improves |

Final raw positive MSE is `0.292969`, versus zero `0.889786`, identity
`1.580654`, shuffled-target `1.668545`, and action-shuffled `1.528640`.
Prediction collapse and loss of action conditioning are not the failure modes.
Target scale narrowly fails, while policy accuracy and legal mass regress.

An independent seed-20,000 evaluation of the saved batch-128 checkpoint finds
DFM CE `-0.001443`, accuracy `+0.000397`, and legal mass `-0.001741` relative
to the source on the same 4,096 positions. Legal mass therefore regresses at
both seeds, and the CE gain remains small relative to repeated-run spread.
Batch 128 with fresh optimizer and unscaled lower learning rates is only the
provisional efficiency candidate for a longer matched baseline; it is not an
accepted quality baseline.

### Evidence hashes and disposition

The report SHA-256 digests for this continuation study are:

| Run | `report.json` SHA-256 |
|---|---|
| `online-target576-b64-100-globaltrain-globalval32-v1` | `28e719d5914a250102a53b375478e9db40fd92bed60b3836eef977bca45609a2` |
| `online-target576-b64-lr3e5-bt4lr1e6-100-globaltrain-globalval32-v1` | `8e5a6a85acce21e06b1fb1a8f435fcb1f5e9a4ebe3521bd5f075937cc92487f1` |
| `policy-only-b64-lr3e5-bt4lr1e6-100-globaltrain-globalval32-v1` | `6c624e007db07dbd14dcb624657985ff37959f11a15dcba33114265cde98a64f` |
| `freshopt-online-target576-b64-lr3e5-bt4lr1e6-100-globaltrain-globalval32-v1` | `4afb2fa947a823a08c67e837d9d0aca6b42b1f5f27987d2e635bcb820191cfa5` |
| `freshopt-online-target576-b64-lr3e5-bt4lr1e6-100-globaltrain-globalval64-collapse-v1` | `2155946e408a8afc951063dce5c9e3627d3fa9466e4aa6e23ee0719a6d00308b` |
| `source-eval-b64-globalval64-seed20000-v1` | `2b6b6a32871857c017bcc60ba811a2c039f742af15debd553837389f188e4420` |
| `candidate100-eval-b64-globalval64-seed20000-v1` | `73f0510f0af0a0f3953757efda9938cde01a32d9886fdb2310aa37eb9d70396c` |
| `freshopt-online-target576-b128-lr6e5-bt4lr2e6-50-globaltrain-fixedval4096-v1` | `ec575ae6a7b49344a259f4c77d098b9eef0c244e1da6f5ab3050331ae76767ef` |
| `freshopt-online-target576-b128-lr3e5-bt4lr1e6-50-globaltrain-fixedval4096-v1` | `0b366e7610cd0ecb50af27c814d2d782f466fe4844818983fea3adaa56b77727` |
| `freshopt-online-target576-b128-lr3e5-bt4lr1e6-50-globaltrain-fixedval4096-collapse-v1` | `a7c39bd33fc07d9ebc45bb09d7188764cdb947be7abab5372e7ec24a0142980f` |
| `candidate-b128-u50-eval-fixedval4096-seed20000-v1` | `c39d68c93bd65984a86445481c60c51529f6f902102d785f1b8051aaf94d7a20` |

The saved batch-64/update-100 and batch-128/update-50 state payloads have
SHA-256
`760272ff8230a1c8039c3c6a2933158108b82b621ac18f52a06e0da701fb4694`
and
`5af4e4246f9151b67000bff8b6ef7491a3821f4617ec43d16ba6a1c288aa12dc`,
respectively.

### Completed 30-minute v1

The first full baseline-qualification run used the provisional configuration:
model-only source import, fresh optimizer, zero warmup, batch 128, main/BT4
rates `3e-5`/`1e-6`, online target SIGReg `5.76`, globally permuted training,
and fixed validation batches of 64. It ran for 1,800 configured training
seconds and saved every 100 updates plus the final update.

| Run property | Value |
|---|---:|
| Updates / examples | 557 / 71,296 |
| Measured training-loop wall time | 1,819.51 s |
| Device / fetch-inclusive examples/s | 44.69 / 41.31 |
| Whole-loop examples/s | 39.18 |
| Input-stall fraction | 7.58% |
| Explicit compile | 17.35 s |
| Synchronous checkpoint saves | 85.10 s |
| Mean sampled GPU utilization / power | 85.41% / 251.20 W |
| Peak JAX live memory | 9,703,864,832 bytes |

The update-557 endpoint is not the selected checkpoint. On the seed-10,000
validation pool, its DFM CE changes from `4.557536` to `4.549606`, while
accuracy changes from `0.105713` to `0.104950` and legal mass from `0.641367`
to `0.632197`. The endpoint improves scale-independent JEPA diagnostics but
still misses the target-scale gate:

| Diagnostic | Initial | Update 557 |
|---|---:|---:|
| Mean target RMS | 0.995286 | 0.937633 (94.21%) |
| Mean prediction RMS | 0.955479 | 0.925837 (96.90%) |
| Mean prediction effective rank | 31.429 | 30.657 (97.54%) |
| Mean prediction-target cosine | 0.814534 | 0.860854 |
| Positive MSE / zero MSE | 0.350225 | 0.265803 |
| Action-shuffled MSE / positive MSE | 4.79731 | 6.39821 |

This is target-scale contraction without prediction rank collapse or loss of
action conditioning. The complete report is
`research/runs/baseline-freshopt-online-target576-b128-lr3e5-bt4lr1e6-30m-v1/report.json`
(SHA-256
`c52b4ef8368ed73d6d9ba194647acce275d3d8a9a19dff757eaec87a444ee6d4`).

### Read-only checkpoint scan

Commit `ec58c63` adds one-process, read-only evaluation of a checkpoint series.
It validates the authoritative resume contract and state digest, restores only
evaluation state, never restores or mutates optimizer state, and reuses one
model object/JIT cache. Every saved v1 checkpoint was evaluated on the same
two independently seeded pools, each 64 batches × 64 examples. The source row
is the mean of the matched source evaluations at seeds 10,000 and 20,000:

| Update | Mean DFM CE | Accuracy | Legal mass | CE delta from source |
|---:|---:|---:|---:|---:|
| Source | 4.530784944 | 0.106887817 | 0.638989463 | — |
| 100 | 4.519870488 | 0.107345581 | 0.639392403 | -0.010914456 |
| 200 | 4.516547283 | 0.108047485 | 0.640690137 | -0.014237661 |
| 300 | **4.512040799** | 0.108215332 | 0.642176097 | **-0.018744145** |
| 400 | 4.512268856 | **0.108352661** | **0.644024267** | -0.018516088 |
| 500 | 4.516230294 | 0.107864380 | 0.634826829 | -0.014554650 |
| 557 | 4.516885983 | 0.107650757 | 0.635045255 | -0.013898961 |

The scan finds a real intermediate optimum: update 300 has the lowest primary
CE, update 400 is only `0.000228057` worse and has better secondary
accuracy/legal mass, and updates 500 and 557 both lose CE and legal mass. The
scan artifacts under
`research/runs/baseline-freshopt-online-target576-b128-30m-checkpoint-scan-seeds10000-20000-v1/`
are:

- `checkpoint_metrics.jsonl`, SHA-256
  `235bd4626417d1f2d352586e16b61b15da0e46c933cbdf8f17bacdbac4abb8d2`;
- `checkpoint_summary.json`, SHA-256
  `305e72f927eff1deb4ab75959682a56eb89d95eaad4e5fdc0236ddf5816fc06f`.

### Extra-pool tie-break and provisional update 300

Updates 300 and 400 were then evaluated on two new global-permutation pools,
seeds 30,000 and 40,000, again with 4,096 examples per seed and the same
batch-64 partition. New matched source reports have SHA-256:

- `research/runs/source-eval-b64-globalval64-seed30000-v1/report.json`:
  `4f395663955e55e4a6c423748a6ddecf82b7e087011800a864c75f8a3619c692`;
- `research/runs/source-eval-b64-globalval64-seed40000-v1/report.json`:
  `949350447b9fe108da2cc095cfc82efe9d6d8f7982375e48eb45cd85b66f1fd2`.

| Seed | Source CE | Update-300 CE (delta) | Update-400 CE (delta) |
|---:|---:|---:|---:|
| 30,000 | 4.541255090 | 4.520179298 (-0.021075793) | 4.522099853 (-0.019155238) |
| 40,000 | 4.597208142 | 4.572790258 (-0.024417885) | 4.573618725 (-0.023589417) |

Update 300 wins CE on both tie-break pools. Across all four independent pools
(16,384 validation examples for each model), the aggregate is:

| Model | DFM CE | Accuracy | First legal mass |
|---|---:|---:|---:|
| Source | 4.550008280 | 0.106750488 | 0.642926642 |
| Update 300 | **4.529262789** | 0.107345581 | 0.644614464 |
| Delta | **-0.020745492** | +0.000595093 | +0.001687822 |
| Update 400 | 4.530064072 | **0.107704163** | **0.645898934** |
| Delta | -0.019944208 | +0.000953674 | +0.002972292 |

Update 300 is `0.000801284` better in aggregate CE; update 400 remains better
on aggregate accuracy and legal mass. Update 300 improves DFM CE at every
horizon, not only in the aggregate:

| Horizon | Source CE | Update-300 CE | Delta |
|---:|---:|---:|---:|
| 1 | 2.931054348 | 2.889833486 | -0.041220862 |
| 2 | 3.361853272 | 3.334500283 | -0.027352989 |
| 3 | 4.238529742 | 4.222269058 | -0.016260684 |
| 4 | 4.677275896 | 4.653810143 | -0.023465753 |
| 5 | 4.985557675 | 4.973970413 | -0.011587262 |
| 6 | 5.220019460 | 5.200613737 | -0.019405723 |
| 7 | 5.407092810 | 5.394062519 | -0.013030291 |
| 8 | 5.578682899 | 5.565042496 | -0.013640404 |

The action-metric improvement is not unanimous per pool: update 300 loses
legal mass on seeds 10,000 and 30,000 and loses a small amount of accuracy on
seed 40,000. It is therefore selected only provisionally by the declared
primary metric. Its state payload is `1,851,704,172` bytes with SHA-256
`0cd9e538381baf154fa9491aa67239228ccc550fb2b256b55435fb9bd3274b06`;
the manifest SHA-256 is
`12e77e1f7a0ffc2a5eac2747f1b822a77d06879bd244f1501d14a8f1a1e3680c`.

The extra-pool evidence digests under the corresponding
`research/runs/baseline-b128-u{300,400}-checkpoint-scan-seeds30000-40000-v1/`
directories are:

| Evaluation | `checkpoint_metrics.jsonl` | `checkpoint_summary.json` |
|---|---|---|
| Update 300, seeds 30k/40k | `0522512d16cba9ccb1e14ca80a0372d00d9aeb775abd335d2ae45b4d996daa8d` | `2bfba7366fe0e6bfc72c21c56459f9635a56876dfed04c0dc9ce83596d64a804` |
| Update 400, seeds 30k/40k | `3f4540825d070f4dc81765be358987630c07518e98d5270f464e63c8e239aa2b` | `5a98e926aa7870f94e836b7809d3a41379c6fffcafb783b80f23e9cc09d289cf` |

### Identical v2 repeat and repeat-qualified offline baseline

The identical v2 run completed from the same source model, fresh optimizer,
seed, data schedule, rates, batch sizes, checkpoint cadence, and objective. It
again produced 557 updates and 71,296 examples. The run report is
`research/runs/baseline-freshopt-online-target576-b128-lr3e5-bt4lr1e6-30m-v2/report.json`
with SHA-256
`9bf71b98f9c6196ca67805c523e356ab538fbf440613a242c0e78681706b4d3e`.
The unselected final update-557 state has SHA-256
`f21698d52c80bca85e84966a116f342ec44460df9c2ea29fbffeccfb69e9a4ce`.

The full seed-10,000/20,000 checkpoint scan independently selects update 300:

| Update | Mean DFM CE | Accuracy | Legal mass | CE delta from source |
|---:|---:|---:|---:|---:|
| Source | 4.530784944 | 0.106887817 | 0.638989463 | — |
| 100 | 4.517196121 | 0.108261108 | 0.640673155 | -0.013588823 |
| 200 | 4.516854411 | 0.108337402 | 0.640184840 | -0.013930533 |
| 300 | **4.510334061** | **0.109420776** | 0.643805286 | **-0.020450883** |
| 400 | 4.510955336 | 0.108886719 | **0.644191438** | -0.019829608 |
| 500 | 4.516683474 | 0.107666016 | 0.634432756 | -0.014101470 |
| 557 | 4.513480280 | 0.107772827 | 0.634521960 | -0.017304664 |

Update 300 wins the primary CE by `0.000621274` over update 400. The scan
artifacts under
`research/runs/baseline-freshopt-online-target576-b128-30m-v2-checkpoint-scan-seeds10000-20000-v1/`
have SHA-256:

- `checkpoint_metrics.jsonl`:
  `1078126b0ef0db1e0222fdebe4eea21e8bd9a708e8cec1abeebbb3f6cee67eab`;
- `checkpoint_summary.json`:
  `61ca6e7c407438ad3fcdf229aaf61139932ff6636c417b739476109715c74c84`.

The selected update-300 state has SHA-256
`491da32259be4669e8824a7a06dc06401483f1d98febdb860e65bbae2f63c1ba`;
its manifest has SHA-256
`6872ab05f240cef5dbc3c0f87e5150ccea20cd5b1f46690a2dd80e585e365cb2`.
The additional seed-30,000/40,000 evidence under
`research/runs/baseline-b128-v2-u300-checkpoint-scan-seeds30000-40000-v1/`
has SHA-256:

- `checkpoint_metrics.jsonl`:
  `76447f4f906ac3dd30da488c677717d45d50b71fdeb9af85c26b93801d334895`;
- `checkpoint_summary.json`:
  `f836424758e99bcf14b0b0f69dca9a6fa043d099e559f542273c349680a6395d`.

Across all four matched 4,096-position pools, the v2 aggregate is:

| Model | DFM CE | Accuracy | First legal mass |
|---|---:|---:|---:|
| Source | 4.550008280 | 0.106750488 | 0.642926642 |
| V2/update 300 | **4.529473042** | **0.108810425** | **0.646496401** |
| Delta | **-0.020535238** | +0.002059937 | +0.003569760 |

Every horizon improves:

| Horizon | Source CE | V2/update-300 CE | Delta |
|---:|---:|---:|---:|
| 1 | 2.931054348 | 2.896588316 | -0.034466032 |
| 2 | 3.361853272 | 3.328875899 | -0.032977372 |
| 3 | 4.238529742 | 4.222782284 | -0.015747458 |
| 4 | 4.677275896 | 4.652556300 | -0.024719596 |
| 5 | 4.985557675 | 4.972647548 | -0.012910128 |
| 6 | 5.220019460 | 5.200192332 | -0.019827127 |
| 7 | 5.407092810 | 5.395286083 | -0.011806726 |
| 8 | 5.578682899 | 5.566855431 | -0.011827469 |

The v1/u300 four-pool CE gain was `-0.020745492`; the independently trained
v2/u300 gain is `-0.020535238`, an absolute difference of only
`0.000210253`. The repeated checkpoint-selection decision and near-identical
held-out gain qualify v2/u300 as the offline baseline.

The selected checkpoint also received a full seed-10,000 latent audit:

| Collapse/coupling diagnostic | Source | V2/update 300 |
|---|---:|---:|
| Mean prediction effective rank | 31.4294 | 31.0318 |
| Minimum horizon effective rank | 29.5291 | 29.1529 |
| Mean prediction feature-std p05 | 0.6876 | 0.6644 |
| Minimum horizon feature-std p05 | 0.6790 | 0.6515 |
| Mean prediction/target RMS ratio | 0.9600 | 0.9853 |

At every horizon, positive-prediction MSE is lower than both zero-prediction
and action-shuffled MSE. The audit therefore finds no prediction-rank collapse
or loss of action conditioning. Absolute target scale still contracts:
mean target RMS changes `0.995286 → 0.935079` (`93.95%` retention), while mean
prediction RMS changes `0.955479 → 0.921361`. This remaining scale behavior is
why “repeat-qualified offline baseline” does not mean “frozen objective.”

The collapse artifacts under
`research/runs/baseline-b128-v2-u300-collapse-globalval64-seed10000-v1/`
have SHA-256:

- `checkpoint_metrics.jsonl`:
  `6938ccbca92ef820efdc1c5a828750a040ed29214874c1c1a05c679ca1b3e3bf`;
- `checkpoint_summary.json`:
  `511fd6e04fa1f76260d83958d6af6c371863db043c9abecc5ce9646f6032ef60`.

### Prediction-SIGReg 0.57 30-minute rejection

The first baseline-length prediction-SIGReg experiment changed only the
prediction coefficient from `0.0` to `0.57`. It completed 547 updates
(70,016 examples) in the fixed 30-minute window at `40.9593` steady
fetch-inclusive examples/s. Its report is
`research/runs/experiment-predsigreg057-target576-b128-lr3e5-bt4lr1e6-30m-v1/report.json`
with SHA-256
`c6f3da8c68ff1a1298624ec2c473ea2e9f6535e5225b4335339f87224a4edead`.

The run's pruned update-200 checkpoint was retained separately and evaluated
with the same two-pool protocol. The complete checkpoint selection is:

| Update | Mean DFM CE | Accuracy | Legal mass |
|---:|---:|---:|---:|
| Retained 200 | 4.520289022 | 0.108291626 | 0.638778948 |
| 300 | 4.512970001 | 0.107666016 | 0.640452380 |
| 400 | **4.511721076** | **0.108612061** | **0.642275913** |
| 500 | 4.514939129 | 0.107498169 | 0.634509238 |
| 547 | 4.518924110 | 0.107955933 | 0.634585755 |

Update 400 is the experiment's best checkpoint, but its two-seed CE
`4.511721076` is worse than the incumbent v2/u300 CE `4.510334061`. Its state
has SHA-256
`fa712e6d641aab2df74f8993c8833fa4fde3e0dda6c590c972e51b335e32f581`;
its manifest has SHA-256
`481c0ea5a4ffe3e55de236ca2ec9ec6c53eef1a4990c27493ea775d80bea48e6`.

The selection artifacts have SHA-256:

| Evaluation | `checkpoint_metrics.jsonl` | `checkpoint_summary.json` |
|---|---|---|
| Updates 300/400/500/547, seeds 10k/20k | `1ad738c04e76318251b62f5b44435ab4bfb27e4b99ec93de3e01073bd0dc06fc` | `c44a1e3ee72bc12259b806fd73a2d15485874d920a537fdf29acea4a00986dcb` |
| Retained update 200, seeds 10k/20k | `05d48d2a1709c3c35325ffa73df115e296f085790ba716ab201d24429ecc2b22` | `a98f6209681e5e8ad031fe8bd613204004e11950718a25a901ca0a6dbefdd6d7` |

A matched seed-10,000 collapse audit compares prediction-SIGReg/update 400
with baseline-v2/update 400, avoiding checkpoint-age confounding:

| Metric | Baseline v2/u400 | Prediction-SIGReg/u400 | Delta |
|---|---:|---:|---:|
| DFM CE | 4.540353823 | 4.542889878 | +0.002536055 |
| Accuracy | 0.107238770 | 0.106201172 | -0.001037598 |
| First legal mass | 0.643751895 | 0.641704401 | -0.002047494 |
| JEPA raw MSE | 0.240220358 | 0.241087545 | +0.000867187 |
| Mean prediction effective rank | 30.860577 | 30.980221 | +0.119643 |
| Mean prediction feature-std p05 | 0.664535 | 0.665982 | +0.001447 |
| Mean prediction RMS | 0.922548 | 0.923893 | +0.001345 |

The small diversity/scale movements do not compensate for worse policy CE,
accuracy, legal mass, and JEPA MSE. Prediction-SIGReg `0.57` is rejected; it
does not replace the incumbent and receives no arena or promotion run.

The prediction-SIGReg collapse artifacts have SHA-256:

- `research/runs/predsigreg057-target576-b128-u400-collapse-globalval64-seed10000-v1/checkpoint_metrics.jsonl`:
  `c511cb8e5d1922e31650579eb89e4a1cb0c50b56be70eb5efd7056b51660c042`;
- its `checkpoint_summary.json`:
  `5674d5308cea9c38af2dc6f81433a0b99c169cc28856a4764d05bdcf27798fb6`.

The matched baseline-v2/u400 collapse artifacts have SHA-256:

- `research/runs/baseline-b128-v2-u400-collapse-globalval64-seed10000-v1/checkpoint_metrics.jsonl`:
  `076ce2eca889ad41fd0768cfccd840c7eb4c3c8bcc25c62b79558094685621fe`;
- its `checkpoint_summary.json`:
  `d77a7c4c1bb28b544536893ed15211b25e9fd8e2b5d769cebfdf82924045e014`.

### Resumable arena foundation and promotion-pool repair

Commit `a2ea5ed` adds the checked local relative-strength command described in
`docs/local_relative_arena.md`. It strictly loads source or research
checkpoints, runs deterministic cached-BT4 multi-pass DFM inference, accounts
for codec coverage and fail-closed faults, plays exact color-reversed pairs,
records pair-aware descriptive statistics, and maintains the official
normalized-Elo GSPRT. Results are written as immutable checksummed pair blocks
plus an atomically replaced state file. Resume binds checkpoints, code, pool,
history, configuration, state, and every block digest.

The new exact-history audit also found that v2 promotion entry 1,245 was
already claim-draw terminal by threefold repetition at its root, although the
root FEN alone appeared nonterminal. Promotion correctly failed before model
loading rather than silently changing the frozen ordered pool.

Commit `5a49df9` then rebuilt the promotion assets from the immutable shards.
Across two history-filter passes it rejected eight selected candidates: four
histories that did not begin at the standard initial position, three invalid
standard-history FENs, and the threefold-terminal root. The final 2,048-entry
v3 pool differs from v2 by exactly one selected removal and one source-derived
replacement. Two complete regenerations were byte-identical, selected
validation/test overlap is zero, and the production loader exactly replays
every history. Commit `12147e8` pins those assets and reopens the promotion
tier. This repairs the pool; a strength-valid full screen and promotion result
remain pending.

The v2/u300-versus-source correctness run completed 16 pairs with zero faults,
full representable coverage on every evaluated position, and no incomplete
coverage positions. All 32 games reached the 16-ply additional cap, so this is
plumbing evidence rather than strength evidence. Its state file is
`artifacts/arena/baseline-b128-v2-u300-vs-source-correctness-v1/state.json`
with SHA-256
`57904c1131b99db41c22204d9bfa44c348bc238a4e344670072a40c59d9d1308`.

The first 16-pair development pilot at cap 64 is invalid strength evidence:
29 of 32 games ended in timeout faults. The shrinking live-game population
caused JAX to compile new batch shapes during nominal policy calls. Its
fail-closed state is retained at
`artifacts/arena/baseline-b128-v2-u300-vs-source-development-pilot16-cap64-v1/state.json`
with SHA-256
`f06ecfa2cc6a7882d59826abf69701ff460f9a76de2ac66c794152d5b98de9a1`;
its score and Elo diagnostics must not be interpreted.

Commit `72af918` fixes that defect by pinning one physical inference batch
shape for the entire run, padding only already validated rows, slicing outputs
back to real rows before semantic checks, and binding those rules in arena
schema v2. Three post-fix, 16-pair static-batch pilots then completed with zero
faults:

| Additional-ply cap | Normal / cap draws | Gameplay wall | Disposition |
|---:|---:|---:|---|
| 64 | 5 / 27 | 156.943 s | cap-censored |
| 128 | 19 / 13 | 396.576 s | cap-censored |
| 256 | 32 / 0 | 592.284 s | first pilot-valid cap |

All 32 cap-256 games were normal draws. The paired score is `0.5`, descriptive
logistic Elo is `0.0`, and its 95% pair-aware interval is
`[-287.451, +287.451]`; 16 pairs are calibration evidence, not a strength
decision. At the measured pre-optimization rate, a strict 128-pair screen
projects to about 79 minutes.

The immutable state and block digests under the corresponding
`artifacts/arena/baseline-b128-v2-u300-vs-source-development-pilot16-cap{64,128,256}-staticb16-v2/`
directories are:

| Cap | `state.json` SHA-256 | Pair-block SHA-256 |
|---:|---|---|
| 64 | `884d0c0bbe07c49e675797492e53918ade98d5e1213529a36f985c4bac7835e2` | `6dc1b7258b87144509aa2c99ae476f790e9943909a4c46b4cc43d8085ccde719` |
| 128 | `8e9d4744230f2888a3dc496565a3282ccdcd3a87222ded774b34515f1566e9eb` | `142bb65837a18f22b9a1f2c89bdced5c2836cb258a7f89f50cd6f8a59a6c0cd0` |
| 256 | `0c4c16937f4682c9e09d7ea699ebab78525813620915a54ae8a09fd4e776d438` | `78066de67f88c1107387b81f44d9eb9ffc82be6e8845256a42d6d7c14c47e721` |

Commit `c2d5efb` adds an arena-only sealed, constant-size state attestation so
the policy adapter can validate each trusted history endpoint in O(1). Full
exact replay remains at the external opening boundary, and the authoritative
stackful runner retains repetition and adjudication authority. The focused CPU
suite passes 83 tests, including exact gameplay-payload parity between
full-history replay and the sealed endpoint path.

The three GPU pilots above predate that optimization and do not establish
real-checkpoint GPU parity or post-optimization timing. An attempted elevated
GPU benchmark/full 128-pair screen was blocked by the account usage limit
reported through 2026-07-25. Do not project a speedup or reuse the
pre-optimization timing as optimized-path performance; both the GPU benchmark
and full screen remain pending.

No checkpoint has been promoted. The v2/u300 result is a
baseline-qualification result, not an accepted autoresearch experiment, so
`research/results.tsv` remains header-only. The code-level readiness flag
remains `AUTORESEARCH_READY = False` pending real-checkpoint GPU validation of
the sealed path and a fault-free full strength screen.

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

The original legacy loader's `strict=True` mode was found by failure injection
to accept missing and wrong-shaped leaves, accept an empty optimizer tree and a
non-integral step, use pickle-enabled NPZ loading, and mutate the model before
later optimizer validation failed.

Commit `7532f3a` removes it from the clean trainer. The new pinned-source import
boundary:

- verifies the pinned `1,851,704,172`-byte size and SHA-256 on the same open
  file descriptor before decoding;
- opens the NPZ envelope with `allow_pickle=False`, then decodes the historical
  scalar object members through a restricted NumPy/BT4-dtype/Optax-sentinel
  unpickler;
- validates exact top-level keys and integer step;
- validates complete typed model and optimizer ABIs before mutating either;
  and
- makes model-only initialization explicit while still validating the ignored
  source optimizer.

Twenty-two focused importer tests cover exact and model-only success, BF16 and
the Optax masking sentinel, bad size/hash/step/top-level keys, and
missing/extra/wrong-shape/wrong-dtype leaves in both trees. Every failure test
asserts that model and optimizer remain unchanged. The actual source imported
in `6.97 s`; all 114 B4 GPU metrics then matched the prior local-loss report
exactly. The report is
`research/runs/strict-import-parity-step265k-b4/report.json`.

The same commit pins JAX, JAXlib, CUDA plugin/PJRT, Flax, NumPy, Optax,
protobuf, and python-chess versions. Every `research/run_gpu.sh` entry now
checks those versions before executing. This was added after a concurrent
development sync changed JAX/JAXlib to `0.10.2` while the CUDA plugin remained
`0.10.1`; the mismatch was detected during startup, the run was aborted before
an update, and no result was accepted.

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
represent white knight promotion or any black promotion without changing
checkpoint semantics, so the arena must record those incomplete move classes,
strict-mask coverage, and any resulting losses rather than invent a remapping
or fallback.

Commit `ea62aa8` implements both explicit codec IDs without changing any
existing preprocessing call. Its golden tests cover mirrored pawn/knight moves,
castling, en passant, straight and capture promotions for all four pieces,
full legal-mask round trips, invalid indices, unsupported input formats, and
whole-board mirror invariance. The combined codec/trajectory compatibility
suite passes.

## One-file model localization

Commit `4a981ca` moves the complete checkpoint-visible model configuration,
adapters, projector, recurrent JEPA transition, DFM planner, value/WDL head,
optimizer construction, and step wrappers into `research/train.py`.
Commit `45bd269` then moves the full stage-1 objective and its closed helper
set into the same file. Production has no import from the legacy joint
model/loss module; that module is now used only by tests as a parity oracle.

Three CPU parity tests compare the local and legacy paths with the same seed:

- every configuration field/default;
- every trainable-state path, dtype, shape, and initialized value;
- every Muon/warmup optimizer-state path, dtype, shape, and value;
- encoded tokens/vectors, noisy planner logits and hidden state, and free JEPA
  rollout; and
- the complete legacy loss/auxiliary tree and every trainable gradient leaf.

All comparisons are exact. Real step-265,000 B4 GPU evaluations after both the
model and loss moves compared 114 validation metrics against the pre-refactor
corrected-objective report and found zero differences. They exactly retained
total loss `3.9873406887`, DFM CE `3.2872314453`, FP32 legal mass
`0.9995466471`, target SIGReg `0.1506309509`, and prediction SIGReg
`0.1407624930`. The final report is
`research/runs/local-loss-parity-step265k-b4/report.json`.

## Checked research edit surface

Commit `d12c45b` makes the intended Karpathy-style edit point explicit:
`EXPERIMENT_OVERRIDES` is a small checked-in mapping near the top of
`research/train.py`. The effective configuration is constructed in this order:

1. restore the pinned source-checkpoint metadata;
2. apply the checked-in experiment mapping; and
3. apply only explicit CLI overrides.

The final configuration is recorded in the run report and included in the
strict resume contract. Tests cover precedence and resume binding. Unknown
keys and wrong types fail before model construction. Non-default fields for
legacy features absent from the local stage-1 graph—such as action contrast,
candidate sampling, horizon legality, and scheduled teacher forcing—also fail
closed rather than creating silent no-op trials.

Commit `f5f6b27` also prevents 18 always-zero compatibility placeholders from
appearing as measured experiment results. The raw loss auxiliary tree is left
unchanged for exact legacy parity. At the reporting boundary, the placeholders
must still exist and be exactly zero, are recorded by name in run metadata,
and are then omitted. A missing or newly nonzero placeholder fails closed so a
future implementation cannot silently inherit the old reporting semantics.
Authoritative free-rollout collapse and action-dependence diagnostics remain
separately computed and reported.

## Arena and initial promotion foundation

Commit `c65f77f` adds the engine-agnostic persistent arena layer:

- exact-ply, strict-standard-chess, hash-ranked opening pools with source,
  selection, ordered-FEN, and whole-pool digests;
- exact same-FEN color-reversed game pairs;
- fail-closed result classification in which illegal moves, timeouts, and
  exceptions are losses and a ply-cap draw must occur exactly at the cap;
- pentanomial pair statistics and a conservative complete-pair Hoeffding
  interval; and
- count-based constrained-multinomial logistic GSPRT bookkeeping checked only
  after complete pairs.

The logistic likelihood matches the official Fishtest construction, but is
explicitly descriptive and `promotion_eligible=false`.

Commits `8d6f805`, `4dff39a`, `5a14994`, and `b60f176` complete the in-process
path:

- fail-closed batched gameplay with exact history replay and color reversal;
- a strict localized DFM policy adapter with no fallback or codec remapping;
- stable policy chunks capped at 64 positions by default, so promotion-scale
  pools cannot turn into one unbounded GPU allocation; and
- the official constrained-multinomial normalized-Elo GSPRT with fixed
  `H0=0`, `H1=20`, `alpha=beta=0.05`, and a hard cap of 2,048 complete pairs.

The normalized implementation includes Fishtest's pentanomial `sqrt(2)`
conversion and matched the official implementation across 100 random count
vectors with maximum absolute LLR error `1.11e-12`. This state, unlike the
logistic diagnostic, can make a promotion decision only with a valid frozen
pool.

The pinned artifacts constructed in the first audit are:

- development: 128 validation FENs, pool digest
  `451784d106bc25b06a3891e910213220275d93eae434badddfe1a38b2e58138a`,
  ordered-FEN digest
  `f86f0fa9de93a8753d2d3b508af4f2479bf9d36f007cd45a378607603b9fb52b`;
  and
- initial promotion: 2,048 test FENs, pool digest
  `8653033334e79c57f321dcdb0b5fd965ed10e4e4586b2826ec670876530ca80f`,
  ordered-FEN digest
  `81d8e0e158ded335899934d3f385b0b065702f469f520175e2fdb0f94e7aa60b`.

The promotion pool excludes all 12,297 valid unique validation candidates plus
36 test roots whose root FEN looked standard but whose reconstructed history
was nonstandard. The combined 12,333-FEN exclusion digest is
`f897aec77466188c29e1dba9734e61481093eebced4777c6c3f320c047c4e450`,
leaving 10,787 candidates; selected development/promotion overlap is zero.

Full standard histories are pinned by manifest digest
`496eace967e101fea28c9c26d6f9517c311d9e7f82a127510d9852397578a36c`
for development and
`35e9cb69c1ed8c263bc05e61faf99ec95319112b91872f5a2063f0f9c27dd168`
for promotion. The corresponding sidecar file SHA-256 values are
`aaa038fc169b686ce1027e9e8773bf859767609acc875b2d6830423a4b5b9717`
and
`e72e62d85571476fccd69e2381ad96043b43ec52234547fec1b48404a78dc0d7`.
The local JSON artifacts are under `artifacts/arena/`.

The later exact-replay audit in commit `a2ea5ed` invalidated that initial
promotion artifact: entry 1,245 was already claim-draw terminal by root
threefold repetition. Commits `5a49df9` and `12147e8` replace and repin it as
v3:

- pool SHA-256 `e750e87643c482d28b4201668bd355234fe1fb27f2a690bf784706b1ddd37459`,
  ordered-FEN SHA-256
  `013778d027d919da9e6a25957e1439c8351081b9cbadf28e29df1c5c7b06b0f2`,
  and file SHA-256
  `5632cc726112c2ab700e0f4d935fa83d58418b406fc100262f3aa19ce3befe2e`;
- history manifest SHA-256
  `d8781efb5d76066bcf2ce2e9ab2897f3261b20f7c4488c90992a8eadcc918a4b`
  and file SHA-256
  `bac58cc7380d9a4173e7618a765917aa0990355e1f8e460196affbe28580ed37`;
  and
- canonical contract SHA-256
  `b3b282d454bfa332af5634c0eb735814335f84528d6fc0c63cd099e2c10fd3c0`.

The tracked v3 assets live under `research/assets/arena/`; regeneration verifies
the source archive, shard identities, split names, output containment, exact
history replay, terminality, and validation/test disjointness.

### Plane-history compatibility audit

The trajectory source explicitly calls `encode_board(board, [])` for both
current and future planes. A direct audit confirmed that current-only encoding
exactly matches every stored root plane among all 128 development and 2,048
initial promotion positions. Conversely, adding reconstructed history
mismatched all 224 rows in the history-aware comparison slice.

Arena histories are still mandatory for legal replay, repetition, and claim
state. The local policy adapter validates them but records and uses
`plane_history_mode=current_only_as_preprocessed`. Feeding history planes to
this checkpoint would be a new input-distribution experiment.

### Real source selfcheck

The first end-to-end A10G run played the source step-265,000 checkpoint against
itself on the 16-pair correctness tier with eight DFM passes, a 16-ply
additional cap, and batch size 16. Compilation was warmed before match timing.

- all 32 games completed without a fault;
- each color-reversed pair had identical move traces and split exactly 1–1;
- pentanomial counts were `[0, 0, 16, 0, 0]`;
- normalized Elo was numerically zero (`-2.18e-12`);
- all games reached the symmetric cap;
- match time was `20.24 s`, or `1.58 games/s` and `25.30 plies/s`;
- compile plus warmup was `28.06 s`; and
- peak JAX live memory was `1,894,688,768` bytes.

The complete artifact is
`artifacts/arena/source-selfcheck-16pairs-pass8-cap16-v1.json`, file SHA-256
`5f54887ec99661af5339994670062a9d0ca746229f649453fc75fe64c7d11cf4`.
This is an evaluator correctness result, not a strength estimate.

## Local inference profile

Commit `c79d94b` adds checked batched inference for the localized model. It
encodes BT4 once, caches the DFM latents across any fixed number of refinement
passes, requires a nonempty boolean `legacy_absolute_1858` root mask, and never
remaps or falls back. The per-pass trace records raw entropy, raw legal mass,
legal-conditioned entropy/top-k, actions before/after refinement, and top-k
turnover. A no-tie oracle matches the historical sampler; deterministic stable
rank handling removes its threshold over-unmask ambiguity on confidence ties.

The source step-265,000 A10G profile used 100 measured calls after ten warmups:

| Batch | Passes | p50 latency | p95 latency | Positions/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 23.47 ms | 30.82 ms | 42.6 |
| 1 | 8 | 25.86 ms | 26.96 ms | 38.7 |
| 64 | 1 | 48.82 ms | 50.33 ms | 1,311.0 |
| 64 | 8 | 62.56 ms | 64.67 ms | 1,023.0 |

Peak JAX memory was 1.765 GiB. Thus seven additional lightweight DFM passes
cost about 2.4 ms at batch one and 13.7 ms at batch 64; the cached BT4 encode
dominates inference. The cold compile/first-call sweep is
`artifacts/profiles/inference-step265k-a10g-v1.json` (file SHA-256
`a6fac32e1b9698655484e78870686cd12965cd2fb1061d7ee6335ab5ff77b926`);
the warmed profile is
`artifacts/profiles/inference-step265k-a10g-steady-v2.json` (file SHA-256
`e457d95d7be97bbb0a264cd8a8a1a944ebd884337c30519485e35991f7cbe237`).

## Next acceptance point

The compatibility harness is not yet open to unattended autoresearch. A
checked-in `EXPERIMENT_OVERRIDES` block now applies after checkpoint metadata
and before explicit CLI flags; the final effective configuration is
resume-bound and recorded. Unknown keys, wrong types, and non-default legacy
knobs that the local graph cannot honor fail closed.

The continuation study explains a large part of the earlier policy regression:
the exact source optimizer and learning rates came from global batch `8,192`,
the recovered source checkpoint was already past its best recorded validation
point, and local batch 64 restarted the data/RNG stream. A fresh lower-rate
optimizer removes the large short-run regression. Both completed 30-minute
runs independently select update 300. V2/u300 is the repeat-qualified offline
baseline: across four matched validation pools it improves aggregate DFM CE,
accuracy, legal mass, and every per-horizon CE relative to source, and its CE
gain is within `0.000210253` of v1/u300.

The remaining acceptance work is to:

- retain prediction-SIGReg `0.0`; the `0.57` baseline-length experiment is
  rejected on primary CE and matched policy/JEPA metrics;
- run real-checkpoint GPU payload parity and timing for commit `c2d5efb` once
  elevated execution is available;
- complete the fault-free 128-pair strength screen at the pilot-valid cap 256;
- decide how absolute target-scale contraction enters the frozen objective
  without undoing the repeat-qualified policy improvement;
- calibrate a strength-valid long-game cap before any Elo result.

Until then `AUTORESEARCH_READY = False`, `research/results.tsv` remains
header-only, and neither the objective nor the optimizer/batch configuration
is frozen.
