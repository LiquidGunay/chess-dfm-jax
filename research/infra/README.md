# Interpretability compute and artifact runbook

This is the operational contract for raw-BT4 versus Hero experiments. The
scientific order and acceptance gates live in
[`docs/hero_bt4_interpretability_plan.md`](../../docs/hero_bt4_interpretability_plan.md).

## Default routing

| Workload | Default | Why | Promote elsewhere when |
| --- | --- | --- | --- |
| Tests, manifests, parameter summaries, individual-position attribution, small patch/probe pilots | GTX 1660 Ti local machine | zero marginal cost and fastest edit loop | RAM, wall time, or corpus size makes streaming impractical |
| Immutable model/corpus/result exchange | Railway private S3-compatible bucket | already funded, cheap, and shared by local/Modal | never use it for hot scratch or raw activation dumps |
| Independent batch maps, larger attribution/patching/probe sweeps, bounded sparse pilots | Modal | scale-to-zero and no idle-pod cost | a workload needs a warm process, large local NVMe, or many contiguous GPU-hours |
| Long stateful sparse/crosscoder training or interactive remote debugging | Runpod, only after a fresh cost comparison | warm process and local NVMe can beat repeated startup/download overhead | the work can be partitioned into short idempotent maps |
| Source backups and completed promoted bundles | Google Drive | human-controlled cold archive | never read the 11.5 GB backup during routine jobs |

The default decision is therefore local plus Modal, with Railway as the
exchange plane. Runpod is a deliberate exception, not a second always-on
backend.

## Price and utilization snapshot

Rates are frozen on 2026-08-10 for planning and must be refreshed before a
material launch. Modal lists T4 at $0.000164/GPU-second and L4 at
$0.000222/GPU-second, plus $0.0000131/core-second and
$0.00000222/GiB-second. Training rates are $0.000542/GPU-second for L40S and
$0.000583/GPU-second for A100 40 GB. With one CPU core and 2 GiB RAM, that is about
$0.654/hour on T4 or $0.862/hour on L4. The $25.50 planned-work allocation buys
about 39 T4-hours or 29.6 L4-hours before credits. See
[Modal pricing](https://modal.com/pricing).

Runpod currently lists Community Pod rates of $0.99/hour for L40S and
$1.39/hour for A100 PCIe 80 GB. Its June 2026 Secure Cloud price update lists
$1.19/hour and $1.69/hour respectively. Standard network storage is
$0.07/GB-month, but transferable network volumes can attach only to Secure
Cloud Pods; temporary container disk is the fastest storage. See
[Pod pricing](https://www.runpod.io/pricing), the
[Secure price update](https://www.runpod.io/blog/runpod-slashes-gpu-prices-more-power-less-cost-for-ai-builders),
and [network-volume constraints](https://docs.runpod.io/storage/network-volumes).
For the Proposal A container (four cores and 32 GiB), Modal L40S costs
$2.395584/hour all-in. The matched steady timings are 2.591 seconds/update on
L40S and 2.525 on A100 40 GB, making the A100 about 2.5% faster but 3.46% more
expensive per update on Modal. At Runpod Secure rates, the same measured A100
speedup would make A100 roughly 38.5% more expensive per update than L40S.

A $1.19/hour Secure Runpod L40S beats that Modal configuration when useful-GPU
utilization exceeds about 49.7%: `useful_hours / rented_hours >
1.19 / 2.395584`. The 7,774-update epoch projects to 5.595 useful L40S-hours,
or about $6.66 on a Secure Pod versus $13.40 on Modal before staging and
storage. Community at $0.99/hour lowers compute to about $5.54, but requires
volume-disk plus Railway checkpoint recovery rather than a transferable
network volume. This makes Modal the default for short fail-closed screens and
a scripted, auto-terminating Runpod L40S the default for the promoted
contiguous epoch. Credits, startup/download time, availability, and measured
throughput can move the boundary, so the Pod must still pass the stateful-pod
gate below.

An FP32-master recovery checkpoint was measured on the training Volume at
about 3.0 GiB (`state.safetensors`, plus a roughly 198 KiB manifest). The full
converted sequential corpus is 8.64 GB and its immutable source tar is 1.77
GB. Provision a 60 GB temporary Secure network volume for the promoted epoch:
this accommodates six sparse recovery states, the final model, corpus, source,
environment, and scratch with reserve. At the standard rate, 60 GB is
$4.20/month if retained for a full month. Publish the pre-decay and terminal
artifacts to Railway, verify them remotely, then delete the temporary volume;
do not keep a Pod or its storage alive for convenience.

## Promoted Proposal A epoch

The 2026-08-10 frozen screen selected an equal encoder/main peak learning rate
of `5e-5` and ordinary decoupled decay. All arms used batch 1,024, so the
matched L40S/A100 result is not a small-batch comparison:

| Arm | Validation total | Root legal top-1 | Current WDL CE | Frozen decision |
| --- | ---: | ---: | ---: | --- |
| `5e-5`, no decay | 7.6351 | 0.5191 | 0.8944 | LR winner/control |
| `5e-5`, decoupled decay | **7.5942** | 0.5165 | 0.8967 | **promoted** |
| `5e-5`, cautious decay | 7.6089 | **0.5243** | **0.8802** | secondary; missed the preregistered total-loss improvement threshold |

The higher rates `1e-4`, `2e-4`, and `4e-4` failed the frozen policy/value
guardrails even where DFM or total loss improved. The retained selector outputs
are
`research/analysis/proposal_a_tuning/lr_analysis.json` and
`research/analysis/proposal_a_tuning/decay_analysis.json`. Final Modal usage
after all seven screens was `$31.74240029` metered and `$1.08389398` billed
after credits; do not launch another tuning arm against the internal
`$34.00` stop line.

The one-epoch contract is
`research/analysis/proposal_a_one_epoch_preregister_20260810.json`:

- open-loop `z0 -> DFM(a0..a7) -> JEPA(z1..z8)`, current and future WDL;
- no teacher KL, legality auxiliary, closed loop, JEPA inference feedback, or
  future-state teacher forcing;
- 7,774 updates / 7,960,576 examples, 2% warmup, 78% stable, 20% linear decay;
- exact recovery at updates 1,000, 2,500, 4,000, 5,500, 6,219, and 7,000;
- deterministic 8,192-position validation at 6,219 and 7,774; and
- one Secure L40S, a 60 GB network volume, 40 GB container disk, stale
  heartbeat deletion after ten minutes, and an absolute nine-hour deletion.

At `$1.19/hour`, the 6.25-hour central estimate is `$7.44`; the absolute
nine-hour compute cap is `$10.71`. Six measured checkpoint writes add only
about one minute. The remaining overhead reserve covers image/bootstrap,
roughly fifteen minutes of corpus conversion, an 8.64 GB copy to container
NVMe, two bounded validations, and provider variance.

The launch tooling is
`research/infra/runpod_proposal_a_epoch.py`,
`runpod_epoch_bootstrap.py`, and `runpod_proposal_a_worker.py`. It uploads
only the 1.77 GB source tar, 336 MB raw encoder, bootstrap, and content-bound
manifest through the Runpod S3 API before a GPU exists. The Pod gets no Railway
credential. It converts once on the network volume, audits the result, copies
the hot corpus to container NVMe, and trains from the exact pushed Git commit.
A terminal or failed worker syncs its status and deletes its own Pod with the
provider-supplied Pod-scoped key. The local monitor independently deletes a
wrong-GPU, over-price, stale, or terminal-but-still-running Pod.

No Runpod resource has been created yet. Authentication requires two separate
items:

1. a normal Runpod API key for `runpodctl doctor` (Pod/volume lifecycle); and
2. a Runpod S3 access-key ID and secret exposed only in the current shell as
   `RUNPOD_S3_ACCESS_KEY_ID` and `RUNPOD_S3_SECRET_ACCESS_KEY`.

Choose a datacenter that has both Secure L40S capacity and a documented S3 API
endpoint. Then use:

```bash
.local/bin/runpodctl doctor
.local/bin/runpodctl datacenter list
.local/bin/runpodctl gpu list

.local/bin/runpodctl network-volume create \
  --name chess-dfm-proposal-a-epoch-v1 \
  --size 60 \
  --data-center-id S3_COMPATIBLE_DATACENTER

.venv/bin/python -m research.infra.runpod_proposal_a_epoch stage \
  --volume-id NETWORK_VOLUME_ID \
  --s3-endpoint https://s3api-DATACENTER.runpod.io/ \
  --output .local/runpod/proposal-a-stage.json

# Dry-run: inspect the absolute deletion time and every launch argument.
.venv/bin/python -m research.infra.runpod_proposal_a_epoch launch \
  --stage-receipt .local/runpod/proposal-a-stage.json \
  --output .local/runpod/proposal-a-launch-dry-run.json

# Creating the Pod requires the explicit --execute switch.
.venv/bin/python -m research.infra.runpod_proposal_a_epoch launch \
  --stage-receipt .local/runpod/proposal-a-stage.json \
  --output .local/runpod/proposal-a-launch.json \
  --execute

.venv/bin/python -m research.infra.runpod_proposal_a_epoch monitor \
  --launch-receipt .local/runpod/proposal-a-launch.json \
  --s3-endpoint https://s3api-DATACENTER.runpod.io/ \
  --output .local/runpod/proposal-a-monitor.jsonl
```

Staging refuses a dirty or unpushed checkout, verifies local and remote sizes
plus SHA-256 metadata, and never recursively lists the volume. Launch refuses
an unverified stage receipt, then immediately deletes a Pod if its returned GPU
is not L40S or its quoted rate exceeds `$1.25/hour`.

## Local safety envelope

Treat WSL as 30 GB total even if the mount reports more. Keep the durable
working set below 20 GB and preserve a 5 GiB operational reserve. Run every
real local GPU command through:

```bash
research/run_torch_gpu.sh .venv/bin/python -m MODULE [ARGS...]
```

The launcher holds a single-GPU lock, requires 4 GiB `MemAvailable` before
launch, aborts below 1.5 GiB, caps process-group RSS at 5 GiB, binds two CPUs,
and applies the logical 30 GiB workspace budget. It redirects caches under the
ignored `.local/` tree. Raw activation dumps are forbidden by default; stream
exact moments, bounded projections, position summaries, and a small audited
example set.

Measured routing refinement: the final source-bound L14 LoRSA transfer over
512 development roots completed locally in 430.024 seconds internally
(435.045 guarded), peaked at 1.230/1.260 GB CUDA and 2.443 GB process RSS,
retained about 2.7 MB, and cost `$0`. The exact-token audit took 112.818
seconds internally (116.542 guarded), peaked at 1.113/1.141 GB CUDA and 2.636
GB RSS, and retained 1.045 MB. Keep these runs and CPU bootstrap local; route
larger independent maps to Modal T4 when isolation, parallelism, corpus scale,
or richer retention is worth startup. GPU model loading alone does not justify
remote execution.

The definitive 256-root causal LoRSA run is also a local workload: both model
arms, six targets, three direction-matched controls per target, dose curves,
and 20,000 paired resamples completed in 522.716 seconds, peaked at
1.168/1.248 GB CUDA, retained 11,132,463 bytes, and cost `$0`. This makes the
whole current LoRSA intervention tier local by default. In contrast, the
historical Hero training report records 20.696/23.073 GB peak
allocated/reserved CUDA and 153,808 seconds (42.72 hours), so optimizer/LR
training sweeps require remote compute. Gate 0 now rules out every 24 GB
worker: the exact Hero-v1 control reached about 20.697/28.899 GB
allocated/reserved, while Hero-v2 FP32 reached 22.049/30.021 GB. Matched
diagnostics measured about 2.591 steady seconds/update on L40S versus 2.525 on
A100 40 GB; L40S has the lower cost per useful update at the frozen rates.
Use Modal L40S for short, restartable successive-halving arms; consider a
Runpod pod only for the selected long arm after measuring useful-GPU
utilization and testing external checkpoint recovery.

The proposed FP32 repair adds a lower-bound 1,361,176,576 bytes (1.268 GiB):
793,834,496 bytes for FP32 masters over all 198,458,624 encoder parameters,
226,492,416 bytes to upcast the single Muon moment, and 340,849,664 bytes to
upcast both AdamW moments. That lower-bound projection was directionally
correct but understated allocator reservation; the measured 30.021 GB
Hero-v2 peak is the routing authority. Every Phase-1 arm writes exact recovery
states at updates 100/200/300/400/500/554 and commits each atomic directory to
the Modal Volume immediately. A preempted partial run is archived and resumes
from its newest verified nonterminal checkpoint. This is required because
Modal preempted the first long control near update 540 and automatically
restarted the same input.

After the exact Hero-1 Phase-1 control completes, freeze its statistical gates
before launching or inspecting candidates:

```bash
.venv/bin/python -m research.compare_training_runs freeze-control \
  /path/to/compact/hero1-control \
  --output research/analysis/hero_v2_phase1/control_contract.json

modal run --detach --env chess-dfm-research research/infra/modal_train_app.py \
  --mode movement --result-label REMOTE_RESULT_LABEL \
  --recorded-spend CURRENT_OR_PROJECTED_SPEND

.venv/bin/python -m research.compare_training_runs compare \
  --control-contract research/analysis/hero_v2_phase1/control_contract.json \
  --control-dir /path/to/compact/hero1-control \
  --candidate-dir /path/to/compact/candidate \
  --control-movement /path/to/hero1/parameter_diff.json \
  --candidate-movement /path/to/candidate/parameter_diff.json \
  --output research/analysis/hero_v2_phase1/candidate-comparison.json
```

The compact directories need `metrics.jsonl`, `run_config.json`, and
`report.json`; no checkpoint download is required. The comparator verifies
their hashes and matched scientific identities, reports raw and 32/128-update
curves and loss slopes, evaluates predeclared examples-to-threshold crossings,
and applies frozen validation guardrails. The `movement` mode uses a CPU-only
worker to diff Raw BT4 against a remote final checkpoint in place, retaining a
small content-bound JSON instead of downloading model state. When both movement
reports are supplied, continuation also requires more visible trunk movement
than H1 without crossing the predeclared runaway ceiling.
`eligible_for_more_compute` remains a successive-halving signal, not a
chess-strength promotion.

Use `--detach` on every nontrivial `modal run`. Two attached training clients
lost their local DNS/heartbeat connection and stopped otherwise healthy remote
functions. Detached mode keeps the last triggered function alive if the local
client disconnects. It does not make source staging or the local entrypoint
itself resumable, so continue to use immutable identities and exact recovery
checkpoints. Before each launch, refresh `modal billing summary --json`; when
another job is active, pass the larger of current metered spend and the active
job's conservative projected-spend floor.

Download completed training evidence with
`research/infra/pull_modal_training_compact.py`. The puller accepts only the
seven bounded report/config/metric files, verifies result identity and a
contiguous terminal metric series, writes atomically, and never downloads a
checkpoint. Arena and movement bundles likewise retain only their compact JSON
state locally; model and exact-resume states remain on the Modal Volume.

The Torch venv intentionally contains no JAX runtime. Install control-plane
dependencies separately:

```bash
uv pip install --python .venv/bin/python \
  --requirement research/requirements_infra.txt
```

## Railway exchange plane

Verified resource identities:

- project `chess-dfm-research`, ID
  `05efeaf8-614b-4944-ad1f-a7c317dda351`;
- environment `production`, ID
  `efc03f33-0b4f-4ad2-8608-83f82cd807b2`;
- private bucket `chess-dfm-artifacts`, ID
  `f456f511-c716-4032-98fb-4fc41c1b0e40`, region `iad`.

Credentials are stored only in the ignored, mode-600 file
`.local/config/railway-bucket-modal.json` and in the Modal secret
`chess-dfm-railway-s3`. Never print, log, upload, or commit either value set.
The repository-local Railway 5.30.4 CLI is `.local/bin/railway`; it coexists
with the older system CLI because current bucket commands require the newer
client.

The store is content-addressed under
`chess-dfm/v1/bundles/sha256/<bundle-sha256>/`. Every object carries SHA-256
metadata. Existing keys must match size and digest or the operation refuses an
immutable collision. Uploads use 16 MiB multipart chunks, two workers,
adaptive SDK retries, and three bounded whole-file attempts; interrupted
uploads resume by verifying and skipping completed objects. Pulls use a
temporary directory, verify every byte, and atomically promote it.

Verified bundles:

| Kind | Bundle SHA-256 | Payload |
| --- | --- | --- |
| Pilot inputs | `691d35245494c806e2f82d1204009f0e39a636cde8140389e4f7f8760df3e895` | raw archive, Hero checkpoint/manifest, pilot corpus; 5 files, 1,048,473,285 bytes |
| Pilot result | `467ef03d322bef1f16b680679817c3d84b552ff07a7f80193ac24b46444e4c87` | complete checksummed model-diff pilot; 6 files, 1,286,276 bytes |
| Current interpretation inputs | `8be17957ad779e3756092306daa95ad9c1c7f5d482b6409bda7a16bc6d6c7a44` | raw/Hero models plus public-v2 corpus; 7 files, 1,053,370,877 bytes |
| Modal probe smoke | `53ade980bbc2edd0747d4f7932c6626ceafed8636c7d1d783a720276d193a1e9` | complete checksummed smoke; 7 files, 1,330,753 bytes |
| Modal all-layer result | `226c215e2fe63eddd7c14a36046235497172eaa9e4f06053d75794704ba1b096` | complete checksummed 15-layer development result; 7 files, 74,753,320 bytes |
| Reviewed LoRSA input | `fd4f42431ea6b7b3b3ee158c4131993d8b61bbd1ee648afdc6dc622104590362` | safetensors, parity fixture, and manifest; 3 files, 215,081,137 bytes |
| Final LoRSA transfer | `dee7536374cd85c4573b8399f1d8696947bd63f98784c8b39729b161d4e6a55f` | source-bound transfer v3; 5 files, 2,774,129 bytes |
| Final paired bootstrap | `86e373d08e3dd464c3171fdce8adb970c178b4e94457d574e51d765e61ee4062` | 20,000-replicate root bootstrap v3; 5 files, 23,762 bytes |
| Final compact screen | `5ad94b63df56bba029424f6b127d05e95501f49e926ce19af818f4f282bb795e` | censored peak-semantics v4; 3 files, 331,169 bytes |
| Final exact-token audit | `a3f1f50b2ccabd7794368a2b3a9e30d8201fcd67aba92f1e5b898535b462100f` | group-disjoint semantic audit v3; 6 files, 1,044,759 bytes |
| Definitive causal LoRSA development run | `12e42f17402bfa1a05ee9e32e47f9e4afebab446cfc5403e90daf19a0abcf1c8` | exact scalar/token/norm-matched causal v2; 7 files, 11,132,463 bytes |
| Exact source snapshot | `c53390977c6dad107ca20614dcb18bcfa44ea3e477987b9d58bc357f35ccdfa8` | all 51 source files bound by transfer v3; 1,427,467 bytes |

The canonical base input, LoRSA input, all-layer result, and the earlier five
final LoRSA bundles total 1,348,806,620 payload bytes (about 1.35 GB decimal);
causal v2 adds 11,132,463 bytes. A read-only inventory taken before that upload
on 2026-08-08 measured the full historical store at 16 bundles, 147 objects,
and 2,403,901,056 bytes. At the documented rate and whole-GB rounding, that
snapshot was about `$0.045/month`; re-query before forecasting future charges.
Each of the final result/source bundles was pulled into a new directory and
compared byte-for-byte locally; the source set reproduced all 51 paths and
hashes. The checkpoint and derived
LoRSA artifacts stay private because the upstream project declares no license.

Railway documents private S3 compatibility and free API operations/egress;
bucket storage is $0.015 per aggregate GB-month with fractional monthly GB
rounded upward. This remains inexpensive at the current scale, but it is not
suitable for raw activation dumps. Recheck:
[bucket behavior](https://docs.railway.com/storage-buckets) and
[billing](https://docs.railway.com/storage-buckets/billing).

```bash
.venv/bin/python -m research.interpretability.artifact_store \
  push-run research/analysis/published_lorsa_l14_transfer_dev512_v3

.venv/bin/python -m research.interpretability.artifact_store \
  push-run research/analysis/published_lorsa_l14_token_semantics_dev512_v3

.venv/bin/python -m research.interpretability.artifact_store \
  pull BUNDLE_SHA256 /new/nonexistent/target

research/run_torch_gpu.sh .venv/bin/python -m \
  research.interpretability.sparse.lorsa_causal_validation \
  --output /new/nonexistent/causal-run

.venv/bin/python -m research.interpretability.hero_encoder_lr_audit \
  --output /new/nonexistent/hero-encoder-lr-audit.json
```

Never point `pull` at an existing directory. Remote workers download only the
required input bundle and upload compact results; they do not create additional
checkpoint identities.

## Modal execution plane

Verified surface:

- workspace/profile `liquidgunay`;
- environment `chess-dfm-research`;
- app `chess-dfm-interpretability`;
- Modal client 1.5.3;
- GPU image `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`;
- Torch `2.5.1+cu121`, CUDA 12.1, Tesla T4.

The production experiment path is a three-stage pipeline:

```text
Railway immutable bundle
        |
        v
CPU-only download + full SHA-256 -> immutable Modal input Volume
        |
        v  (verified cache/path preflight)
T4 model load + experiment + checksummed result Volume
        |
        v
CPU-only Railway publisher -> immutable result bundle
```

The CPU input stage verified 7 files and 1,053,370,877 bytes. The T4 stage
recorded a Volume cache hit, a workspace-path guard, and
`network_download_bytes: 0`; the CPU publisher uploaded 74,753,320 result bytes.
Therefore Railway download and result-upload time do not accrue GPU charges.
The T4 is billable while its container loads cached files/models and performs
the experiment.

Verified experiment artifacts:

| Run | Local result | Railway bundle or status |
| --- | --- | --- |
| Probe smoke | `research/analysis/modal_probe_smoke_v1`; run `sha256:7c0099ed94d1121f8d9cbd35fa6beb2aa491d323574f659cb8c51f4271dcff4c` | `53ade980bbc2edd0747d4f7932c6626ceafed8636c7d1d783a720276d193a1e9`, 1,330,753 bytes |
| All-layer development | `research/analysis/modal_probe_all_layers_dev_v1`; run `sha256:8c9ce9ee97af6f6753688a36a1997994f8803fe358578edf07be7b356b22d963` | `226c215e2fe63eddd7c14a36046235497172eaa9e4f06053d75794704ba1b096`, 74,753,320 bytes |
| LoRSA T4 smoke | `research/analysis/modal_lorsa_smoke_t4_v2`; run `sha256:b73e15c11a6ac1f1788e80b0d970bcdef4f8dcc0683c2e13990a1e7723ac4e9a` | local checksum-verified result; 8-position identical-GPU comparison workload |
| LoRSA L4 smoke | `research/analysis/modal_lorsa_smoke_l4_v2`; run `sha256:0f2e4160cb375a295331be5a17474927070600a298de7aabbb48493ce9d690b0` | local checksum-verified result; 8-position identical-GPU comparison workload |

The powered all-layer T4 stage took 1,629.42 seconds and peaked at 1.406 GB
allocated / 1.474 GB reserved. After billing collection, all activity settled
at `$0.46430113` metered and `$0` billed after credits; Volumes remain `$0`.
The powered CPU→GPU→CPU pipeline added `$0.44357680`. This is about 1.04% of
the `$42.50` budget for that pipeline and 1.09% total. The durable evidence is
`research/analysis/raw_hero_interpretability_report_v1/cost_audit.json`.

For the identical 8-position LoRSA smoke, retained internal workload time was
25.6692 seconds on T4 and 22.8506 seconds on L4; both peaked at 1.230 GB
allocated / 1.260 GB reserved. The frozen combined rates for these actual
4-core/24-GiB functions are `$0.00026968/s` on T4 and `$0.00032768/s` on L4.
Multiplying by internal workload time gives planning estimates of about
`$0.00692248` and `$0.00748767`, respectively. They are not billed
attributions: exact successful-attempt metering was unavailable. The exact
observed Modal session increment was `$0.03017528`, comprising control-plane
work, CPU staging/publishing, one failed v1 T4 attempt (exact increment
`$0.00889735`), and the successful T4 and L4 smokes.

The smoke result changes routing, not the safety model. Tiny sequential LoRSA
inference is faster operationally and free locally once artifacts are present;
its remote timing is dominated by startup/model loading, and the 8-position
smokes are not directly comparable with the 512-position local run. Keep this
inference local. Use T4 for heavier restartable remote maps unless a measured
job-specific L4 speedup clears L4's higher combined rate. Attempt-level cost
claims must remain estimates until Modal exposes attributable billing.

The reliability audit records three short scientific-path failures: two
incorrect raw-model logical paths (under one second of model work each) and a
21.7-second successful smoke whose optional Git provenance step failed because
the lean image has no Git binary. An unsupported ephemeral-disk override was
rejected locally before a remote function. The corrected pipeline treats Git
as optional provenance, uses `models/raw/BT4_exported.pb.gz`, and passed smoke
plus the full run.

The user-confirmed provider-native budget is `$42.50`; the repository stops new
launches at 80% (`$34.00`). Before **every** CPU or GPU stage, the local
entrypoint reads current metered spend with `modal billing summary`, carries a
conservative projected-spend floor across the pipeline, applies the frozen
price snapshot, and fails closed on unknown prices, unreadable billing, or cap
breaches. Conservative declared-attempt bounds are:

| Stage | Declared upper bound | Per-stage cap |
| --- | ---: | ---: |
| CPU input stage | `$0.073668` | `$0.10` |
| T4 probe smoke | `$0.453456` | `$0.50` |
| T4 all-layer development | `$2.184408` | `$2.25` |
| T4 LoRSA smoke | `$0.647232` | `$0.70` |
| L4 LoRSA smoke | `$0.786432` | `$0.85` |
| T4 LoRSA development fallback | `$1.698984` | `$1.80` |
| L4 LoRSA development fallback | `$2.064384` | `$2.20` |
| CPU result publisher | `$0.021048` | `$0.03` |

Each remote function uses one container, single-use lifecycle, explicit startup
and execution timeouts, bounded/retry-free GPU stages, and immutable input and
output identities. Modal can reschedule a crashed container independently of
function `retries`, so declared-attempt estimates are not global provider caps;
the native hard budget and 20% reserve cover that risk.

Verified commands:

```bash
modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode control-plane

modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode stage-input

modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode probe-smoke

modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode probe-all-layers

modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode lorsa-smoke-t4

modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode lorsa-smoke-l4
```

The two probe modes automatically run CPU stage → GPU compute → CPU publish.
The standalone `stage-input` mode is useful for inspecting transfer/cache cost
without launching a GPU. Repeated staging is idempotent but still fully hashes
the cached files before returning a cache hit.

Before adding another real Modal job:

1. consume an immutable bundle SHA rather than a mutable path;
2. make each map input idempotent and restartable;
3. retain compact sufficient statistics, not raw activation dumps;
4. return plain JSON primitives and content identities;
5. set one container and a short first-smoke timeout;
6. add a `ModalJobSpec`, explicit per-stage cap, and native budget check;
7. smoke the smallest scientific prefix before scaling; and
8. query billing after collection lag and retain the audit JSON.

The frozen policy is [`modal_budget_policy.json`](modal_budget_policy.json);
the implementations are `research/interpretability/remote_cost.py` and
`research/interpretability/modal_app.py`.

## Stateful-pod gate

Do not rent a pod merely because a job is “large.” First estimate useful GPU
seconds, startup/download time, checkpoint interval, expected retries, and the
cost of idle debugging time under both providers. Use Runpod only when a warm
process or local NVMe materially changes the failure/cost model. A pod launch
must have an auto-termination condition, an external heartbeat, a Railway
checkpoint target, and a tested bootstrap script. A network-volume design must
price Secure Cloud explicitly; a Community design must push recovery state to
Railway before depending on termination-prone volume disk. No Runpod
credential is committed to the repository or written into a launch manifest.

## Recovery rules

- A local memory/disk guard refusal is not overridden; reduce batch/sketch
  width or route the task remotely.
- A Railway transfer failure is resumed with the same bundle identity; never
  rename or overwrite the partial logical result.
- A Modal crash-loop is aborted, its app ID and metered cost are recorded, and
  the import/startup path is fixed on a minimal smoke before retrying model
  work.
- A completed result is trusted only after local checksum verification and an
  idempotent remote re-upload.
- Provider dashboards are operational evidence, not the sole provenance;
  manifests, hashes, costs, and exclusions belong in retained run artifacts.
