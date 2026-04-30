# Roadmap

## Current Status

The first implementation wave is substantially in place:

1. Public hygiene/docs pass: project naming, vision/roadmap docs, example TPU
   specs, and ignored local operational config paths are in place.
2. Trajectory-v2 data path: loaders, shard validation, legacy terminal-only
   adaptation, TCEC processing, LC0 PGN processing, and exact rollout helpers
   are implemented.
3. DFM diagnostics and curriculum training: `scripts/train_dfm.py` supports
   fixed train/eval `t`, `loss_horizon`, first-action CE, first-step legality
   regularization, horizon-wise metrics, W&B logging, and MFU estimates.
4. GCS data cache guardrail: GCS training now defaults to caching all visible
   train shards before step 1. This removes the earlier growing-cache replay
   artifact.
5. Monitoring notebooks: `state_action_training_browser.py` inspects shards and
   run status; `trajectory_dedup_browser.py` displays duplicate statistics.
6. Dataset QA: `scripts/trajectory_dedup.py` computes duplicate statistics and
   writes exact-deduplicated trajectory-v2 shards.
7. Persistent experiment queues: `scripts/run_experiment_queue.py` can execute
   multiple experiments sequentially on one already-provisioned worker, with
   per-experiment status files and checkpoint namespace validation.

## Current Status Snapshot

As of 2026-04-29:

- The LC0-only Phase A run on the 1M sample dataset completed 30k steps. It
  covered about 6.1 effective epochs, which is too much reuse for model
  selection.
- That run showed useful loss/legality progress, but validation CE/accuracy
  stagnated or worsened. Treat it as a system and curriculum diagnostic, not a
  final checkpoint.
- A larger LC0 test80 H8 trajectory-v2 build is running on a CPU data VM. It
  skips the first 64 archives already used, processes the next 640 archives in
  parallel, and targets about 10M samples before deduplication.
- Two short 15k-step TCEC legality ablations are running on Spot TPU. They use
  the existing validated TCEC H8 split, fixed `t=0`, `loss_horizon=1`, and
  first-step legality weights 10 and 20.
- The combined TCEC+LC0 exact-deduplicated H8 dataset is still a later training
  input. Do not use it until its manifest exists and rollout validation passes.

## Training Policy

- BT4 stays frozen through DFM baselines, JEPA baselines, and the first joint
  Latent-SASA runs. Unfreeze only after the heads and data path are proven on
  held-out action, legality, and latent-rollout metrics. The first unfreeze
  should be adapters/LoRA or last BT4 blocks with a smaller learning rate; full
  BT4 fine-tuning is a late joint-refinement step.
- The joint direction is not fully separate DFM/JEPA projectors. Use a shared
  online BT4-to-planning-latent base, small DFM and JEPA adapters, shared action
  embeddings, and a stop-gradient or EMA target projector for JEPA targets. The
  existing standalone DFM and JEPA scripts remain baselines.
- Normal training runs must use immutable dataset prefixes. The CPU data worker
  can keep producing additional LC0 trajectory-v2 chunks while TPU training
  runs, but it should write to a new prefix that is not consumed until manifest,
  rollout validation, and dedup stats pass.
- Sweep infrastructure should provision TPU workers for queues, not one TPU per
  experiment. A worker should run its assigned queue until completion,
  preemption, or a code/data failure. Code/data failures stop the queue;
  preemptions are resumed by the controller from the unfinished experiment.
- Grain is not the first loader backend. Build a custom deterministic,
  column-selective loader boundary first; consider Grain later if host input
  throughput or distributed sharding remains a bottleneck after compact v3.
- Checkpoints remain raw NumPy for compatibility. The next checkpointing step is
  async GCS upload of completed local checkpoints with `latest_local_step` and
  `latest_uploaded_step` status reporting; Orbax is a later migration, not a
  blocker for the next experiments.

## Experiment Defaults

Use these defaults for the first queued experiments unless a run explicitly says
otherwise:

| run family | horizon | learning rate | model | loss |
| --- | ---: | ---: | --- | --- |
| DFM first-ply baseline | 1 | `6e-4` | `D640/L8/H10/MLP2560` | `CE(a0) + 7.64 * illegal_mass(a0)` |
| DFM H4 action baseline | 4 | `6e-4` | `D640/L8/H10/MLP2560` | `CE(a0:a3) + 7.64 * illegal_mass(a0) + 7.64 * teacher_forced_illegal_mass(a1:a3)` |
| JEPA H2 | 2 | `1e-4` | `D256/L4/H4/MLP1024` | `latent_cosine + 0.01 * sigreg` |
| JEPA H4 | 4 | `1e-4` | `D256/L4/H4/MLP1024` | `latent_cosine + 0.01 * sigreg` |
| Joint Latent-SASA H2 | 2 | `3e-4` | `D256/L4/H4/MLP1024` | `DFM CE + legal + latent_jepa` |
| Joint Latent-SASA H4 | 4 | `3e-4` | `D256/L4/H4/MLP1024` | `DFM CE + legal + latent_jepa` |
| Joint H4 rank | 4 | `3e-4` | `D256/L4/H4/MLP1024` | previous loss plus `0.2 * chunk_rank` |

The `7.64` legality coefficient is the current random-policy-balanced default:
random CE is `log(1858) ~= 7.53`, while random illegal mass is close to `1`.
Recompute it from the dataset's average legal-move count when launching a new
major dataset, but do not run a broad Phase A legality sweep before the first
clean baseline. Batch size should be auto-probed per TPU shape and then kept as
large as fits.

## Audit Reconciliation

The latest repo audit is useful, but some findings are already addressed:

- `JEPAConfig.use_xsa` now exists and is passed through model construction.
- Raw LC0 chunks now hard-fail for `horizon != 1` instead of silently creating
  fake zero-action horizons.
- Trajectory-v2 is the canonical multi-step contract, and DFM evaluation has a
  first sampler/refinement path.
- GCS-backed DFM training defaults to cache-all startup to avoid repeated
  partial-cache artifacts.
- Dtype parsing now fails closed and accepts explicit aliases such as `bf16`,
  `bfloat16`, `fp16`, and `fp32`.
- JEPA defaults now keep value/WDL losses off and use a small SigReg coefficient
  for dynamics-first training.
- Raw LC0 `.zst` chunks are decompressed through `zstandard`.
- Raw LC0 `action_source` is wired for `best` and `played` targets.
- DFM curriculum branches can initialize model weights from a prior checkpoint
  with `--init-checkpoint-uri` while writing to a new run ID.
- DFM evaluation restores the training loss horizon and loss weights from
  checkpoint metadata.

Remaining blockers before serious new sweeps:

- Add policy-sampled raw LC0 targets if we decide to train DFM against the LC0
  visit distribution instead of hard best/played targets.
- Add JEPA metrics that defeat the identity shortcut: changed-square cosine,
  identity-baseline cosine, and true-action-vs-random-action delta.
- Add DFM model-selection metrics beyond CE: legal argmax rate, legal top-k,
  illegal probability mass, accuracy by `t` bin, and full exact-rollout sequence
  legality from the sampler.
- Document and then implement real multi-device sharding before claiming
  multi-host TPU scaling; current training should be treated as single-process
  unless a run explicitly uses sharded arrays.

## Near-Term Plan

1. Finish the loader and queue-runner work:
   - DFM action batches should avoid materializing unused future boards.
   - TPU-side queues should run several experiments on one provisioned VM.
   - Controller-side queue splitting should request multiple workers only when
     we intentionally want parallel capacity.
2. Let the LC0 10M build finish, then validate sampled train/val/test shards:
   schema, shape, legal replay from `fen_t`, future board equality, and legal
   mask consistency.
3. Compute the LC0 10M average legal-move count and set the Phase A legality
   coefficient to match random-policy CE scale:
   `lambda = log(1858) / (1 - avg_legal_moves / 1858)`.
4. Run duplicate statistics on the LC0 10M dataset and write an exact
   `position_actions` deduplicated version if duplicate rates are meaningful.
5. Start one clean LC0 Phase A run on the full validated LC0 10M dataset:
   `learning_rate=6e-4`, `loss_horizon=1`, fixed `t=0`,
   `first_action_loss_weight=0`, `horizon_legality_loss_weight=0`, cache-all
   startup, and the measured random-policy-balanced legality weight. Do not
   sweep this Phase A launch.
6. Finish and validate the combined TCEC+LC0 exact-deduplicated H8 dataset.
   Do not train from that prefix until `manifest.json` exists and a sample shard
   validation passes.
7. Keep the LC0 CPU data worker running on a new immutable prefix for future
   Phase B/C data while the Phase A TPU run trains.
8. Run short queued baselines, then move quickly to joint Latent-SASA:
   - DFM H1 and H4 action-only baselines.
   - JEPA H2 and H4 teacher-forced latent transition baselines.
   - Joint H2 with shared latent base and clean teacher-forced actions.
   - Joint H4 with DFM action-state conditioning.
   - Joint H4 with chunk-ranking negatives.
9. Run comparable LC0-only, TCEC-only, and deduplicated TCEC+LC0 validation
   curves before scaling depth/width.
10. Add sampler/reranking evaluation before treating a checkpoint as
    planner-ready.

## Trajectory-v2 Contract

- `planes_t`: `[B, 112, 8, 8]`
- `actions`: `[B, H]`
- `planes_future`: `[B, H, 112, 8, 8]`
- optional `legal_masks`: `[B, H, 1858]`
- optional `value_targets`: `[B, H]`
- optional `wdl_targets`: `[B, H, 3]`
- metadata: source, game, ply, result, plus optional helpers such as `fen_t`

## Acceptance Focus

- TCEC shards encode exact per-step future boards.
- LC0 shards encode exact per-step future boards and pass legal rollout checks.
- Legacy `planes_target` shards still load through an explicit terminal-only adapter.
- JEPA returns sequence outputs shaped `[B, H, 64, D]`.
- The marimo notebook runs in script mode without cloud credentials.
- Tracked docs/configs only ship sanitized examples; local operational details live in ignored `.local.json` files.
- GCS-backed training must not start from a small partial cache unless an
  experiment explicitly opts into `--gcs-startup-cache-policy minimum`.
- Queued sweeps must not launch one TPU per experiment. One worker should run a
  queue until preempted or failed, and each queue entry must own a unique
  checkpoint URI.
- DFM-only loaders should not materialize `planes_future`; JEPA and joint views
  may load future states.

## Evaluation Gaps

- DFM model selection must use held-out trajectory shards, not only training loss.
- Training-time `legality_loss` is a one-step first-move illegal-probability regularizer; it is not final sampler legality.
- LC0 1M results are vulnerable to overfitting because the 30k-step run saw the
  training split about 6.1 times.
- Fast train accuracy during first-ply curriculum is not enough; cache-all
  startup and held-out validation are required to avoid memorization artifacts.
- Add sampler/refinement evals before treating a checkpoint as planner-ready:
  first-move legal rate, full exact-rollout legal sequence rate, first-move top-k accuracy,
  action-chunk accuracy by horizon, and loss/legality curves on a fixed validation split.
- JEPA selection should report horizon-wise latent error, value/WDL accuracy, and predicted-vs-exact rollout agreement.
