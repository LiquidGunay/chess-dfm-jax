# Hero / Proposal A continuation handoff

Status: paused deliberately on 2026-08-11, before the promoted one-epoch run.

This is the restart document for the Hero model-improvement thread. The
implementation, tuning evidence, data receipts, cloud safety controls, and
one-epoch preregistration are complete. No promoted epoch has been launched,
no Runpod Pod or network volume has been created, and no GPU is currently
running for this work. The next active project is the interpretability course;
do not mix course-authoring changes into the frozen experiment contract below.

## Where the thread stopped

The selected model is **Proposal A**, the open-loop state-action-state model:

```text
board x0 -> BT4 encoder -> z0
z0 -> one DFM call -> proposed actions a0...a7
(z0, a0...a7, DFM hidden states) -> action-conditioned JEPA -> z1...z8
z0...z8 -> shared WDL value head
```

The model predicts an eight-ply action trajectory and the corresponding future
state embeddings. At inference, the standard searchless path makes one DFM
trajectory proposal and one JEPA rollout. There is no optimization through the
JEPA at inference and no recurrent refinement loop.

The frozen scientific choices are:

- fresh initialization from the raw BT4 artifact;
- horizon eight, batch 1,024, BF16 forward/backward compute with FP32 master
  weights and optimizer state;
- identical encoder and new-module peak learning rates of `5e-5`;
- WSD schedule with 2% linear warmup, 78% stable plateau, and 20% linear
  decay, clocked by examples;
- ordinary decoupled weight decay `0.05622851404690516` on the selected fresh
  matrices;
- current-state and future-state WDL supervision from LC0 sequential roots;
- no Hero teacher checkpoint or teacher KL;
- no legality auxiliary (`legality_coeff = 0`);
- no JEPA-to-DFM inference feedback or closed loop;
- no future JEPA teacher forcing; and
- no value-based action selection during DFM denoising.

Do not reinterpret an earlier fixed-Hero-teacher or closed-loop diagnostic as
the selected recipe. Those runs answered bounded questions and were rejected
for this promoted epoch.

## Frozen evidence and identities

The authoritative one-epoch contract is
[`research/analysis/proposal_a_one_epoch_preregister_20260810.json`](../research/analysis/proposal_a_one_epoch_preregister_20260810.json).
Validate its raw bytes before launch; its frozen SHA-256 is
`ed9d5c0d05e995085686442f1530d18b67d59599a7b1ce870d9782c52ead2cf2`.

The selection evidence is:

- learning-rate selector:
  [`research/analysis/proposal_a_tuning/lr_analysis.json`](../research/analysis/proposal_a_tuning/lr_analysis.json),
  winner `proposal_a_equal_lr_5em5_wd0`;
- decay selector:
  [`research/analysis/proposal_a_tuning/decay_analysis.json`](../research/analysis/proposal_a_tuning/decay_analysis.json),
  winner `proposal_a_equal_lr_5em5_wd1`; and
- full interpretation and LC0 adapter contract:
  [`proposal_a_training_and_search_plan.md`](proposal_a_training_and_search_plan.md).

The cautious-decay arm had the best short-prefix root top-1 and current-WDL
metrics, but missed the preregistered total-loss improvement threshold. It is a
useful secondary result, not permission to replace the selected ordinary
decoupled-decay recipe after inspection.

The immutable data inputs are:

| Input | Identity |
| --- | --- |
| LC0 source tar | 1,773,731,840 bytes; SHA-256 `1c5e5d0d1d335bfeca9693700a1ad1415abfb190772bd051d1f00cb193eb3c2f`; Railway bundle `cf83360c179fa939176bda91f9e8ffa7cb0e115e463dbdb5c06cf1a3b502d3aa` |
| Raw BT4 | 335,916,563 bytes; SHA-256 `61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651`; Railway bundle `c8b25fa7d151aa82ab1c91f395a1a2f6c238d0a2f88e06ed1c04aa873080fe7a` |
| Converted corpus | 8,635,794,032 bytes; audit SHA-256 `247a1c44766a273aa479fb825f065f874508cbdfdc54ca39ad2870ab7e120ebe` |
| Train order | 254 shards; manifest SHA-256 `63602198610284b1cbe710e0e112a9149384502579353693f827be019b4063b5`; seed 0 |
| Validation pool | 243 shards; manifest SHA-256 `9240e014a2d9126e9a937cb2c908af68bf1aab28f51e692bc2a645111d859629`; seed 20,000 |

One epoch is 7,774 complete updates and 7,960,576 consumed examples. Retain
exact recovery states at updates 1,000, 2,500, 4,000, 5,500, 6,219, and 7,000.
Update 6,219 is the reusable pre-decay state. Evaluate the same deterministic
8,192-position validation subset at updates 6,219 and 7,774, then retain the
terminal model-only artifact.

## Compute decision

Use one **Secure Runpod L40S**, subject to a fresh price and availability check.
The matched Proposal A benchmark measured 2.591 seconds/update on L40S and
2.525 seconds/update on A100 40 GB. The A100 was only 2.5% faster and was more
expensive per update on both provider snapshots. The workload peaked near
22.05 GB allocated and 31.24 GB reserved, so an L40S's 48 GB is sufficient.

The 2026-08-10 planning estimate was 6.25 rented hours and `$7.44` at
`$1.19/hour`, with an absolute nine-hour / `$10.71` compute ceiling and a
maximum accepted quoted rate of `$1.25/hour`. These are stale planning inputs
after any material delay: refresh provider prices before creating a resource,
but do not switch GPU types without a matched workload benchmark.

Provision a 60 GB temporary network volume and 40 GB container disk. Stage the
1.77 GB source tar, raw BT4 artifact, standalone bootstrap, exact clean Git
commit, and content-bound manifest before a GPU exists. Convert on the network
volume, copy the 8.64 GB hot corpus to container NVMe, and train there. Never
put Railway credentials into the Pod.

## What is needed from the operator

Two Runpod credential classes are intentionally absent from Git and must be
provided in the current shell only:

1. a normal Runpod API key recognized by `.local/bin/runpodctl doctor`, for
   Pod and network-volume lifecycle; and
2. `RUNPOD_S3_ACCESS_KEY_ID` plus `RUNPOD_S3_SECRET_ACCESS_KEY`, for staging to
   the selected datacenter's S3-compatible network volume.

Select a datacenter only if it has both Secure L40S capacity and a documented
S3 endpoint. Do not print either credential, store it in a launch manifest, or
forward it to Railway.

## Restart checklist

Start from a clean branch based on the commit that merged this handoff. Read
this file, the preregistration, and
[`research/infra/README.md`](../research/infra/README.md) before changing code.
Then:

1. Confirm that no previous Pod or volume exists and refresh the L40S price.
2. Run the JAX-free research tests and preregistration validator.
3. Verify the raw BT4, source-tar, converted-corpus, split, and ordering hashes.
4. Confirm `git status` is clean and the current commit is pushed. The staging
   receipt must bind that exact commit.
5. Authenticate both Runpod control-plane and S3 interfaces.
6. Create the 60 GB network volume in the chosen S3-compatible datacenter.
7. Run `stage`; inspect its receipt and remote checksums.
8. Run `launch` without `--execute` and inspect the resolved GPU, price,
   absolute deletion time, input identities, and command.
9. Only then repeat `launch --execute`, immediately followed by the external
   monitor.

The prepared commands are:

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

.venv/bin/python -m research.infra.runpod_proposal_a_epoch launch \
  --stage-receipt .local/runpod/proposal-a-stage.json \
  --output .local/runpod/proposal-a-launch-dry-run.json

.venv/bin/python -m research.infra.runpod_proposal_a_epoch launch \
  --stage-receipt .local/runpod/proposal-a-stage.json \
  --output .local/runpod/proposal-a-launch.json \
  --execute

.venv/bin/python -m research.infra.runpod_proposal_a_epoch monitor \
  --launch-receipt .local/runpod/proposal-a-launch.json \
  --s3-endpoint https://s3api-DATACENTER.runpod.io/ \
  --output .local/runpod/proposal-a-monitor.jsonl
```

The stage command must refuse a dirty or unpushed checkout. The launch command
must delete a newly created Pod if the returned GPU is not `NVIDIA L40S`, if
the quote exceeds `$1.25/hour`, or if post-creation verification fails. The
worker publishes a heartbeat every 30 seconds and self-deletes its Pod after a
terminal receipt; the independent monitor deletes a wrong, over-price, stale
for ten minutes, or terminal-but-still-running Pod. The provider-level absolute
termination is nine hours.

If an attempt is interrupted, do not restart from weights alone. The worker
must resume from the highest checksum-valid exact recovery state while keeping
the original Git commit and resume contract. If no recovery state verifies,
start a new attempt directory from update zero and retain the failed evidence.

## After the epoch: evidence before promotion

A completed training process is not a promoted Hero. In order:

1. Verify the terminal receipt, contiguous metrics, both deterministic
   validation records, checkpoint hashes, run configuration, optimizer
   partition, and exact data cursor.
2. Copy the update-6,219 recovery checkpoint, update-7,774 model-only artifact,
   and compact evidence to Railway. Verify the remote bytes before deleting
   the Runpod volume.
3. Run a non-overlapping paired searchless arena against both terminal Hero 1
   and raw BT4. Report Proposal A p1 and p8 as distinct agents; do not select
   the better pass count after seeing the same games.
4. Implement and validate the LC0 backend contract: legal-order policy priors,
   `q = P(win) - P(loss)`, `d = P(draw)`, side-to-move orientation, and no
   moves-left head initially.
5. Compare matched LC0 MCTS using raw BT4, Proposal A's root DFM policy plus
   `z0` WDL value, and the updated encoder's native heads. Freeze nodes,
   threads, cache, batching, openings, colors, temperature, MLH, and all search
   parameters.

Only call the new artifact stronger if the preregistered behavioral evidence
supports that claim. A loss improvement alone is insufficient.

## Interpretability restart after promotion

If Proposal A is promoted, rerun the same first-principles stack used for raw
BT4 versus Hero 1, now as a three-model comparison: raw BT4, sealed Hero 1, and
the promoted artifact. Begin with parameter/model diffing, encoder movement,
behavioral decomposition, and layerwise probes. Then run causal patching,
lookahead/trajectory interventions, published transcoder and LoRSA transfer,
exact-token semantics, and matched refits where transfer fails. Keep the test
split sealed until all feature choices, directions, layers, doses, and null
controls are frozen.

The main scientific question is whether the higher encoder update budget plus
action-conditioned predictive objective creates interpretable state and plan
representations that Hero 1's nearly static encoder did not. Preserve negative
results and distinguish association, decodability, causal effect, behavioral
usefulness, and chess strength throughout.

## Conditions that justify reopening the frozen recipe

Do not retune because time has passed or because a newer GPU is available.
Reopen the recipe only if a pre-launch invariant fails, the exact environment
cannot execute the preregistered contract, provider economics change the GPU
choice after a matched benchmark, or the one-epoch run fails for a diagnosed
scientific rather than incidental systems reason. Record any amendment before
observing the affected result, preserve the original preregistration, and make
the deviation explicit in every comparison.
