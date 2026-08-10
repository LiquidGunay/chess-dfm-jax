# Proposal A training and LC0-search plan

Status: preregistered 2026-08-10 before the first Proposal A GPU run.

## Model contract

Proposal A is the open-loop state-action-state model, not the recurrent
feedback arm. Starting from the current board, the BT4 encoder produces
`z0`; one DFM call predicts the eight-action chunk; the action-conditioned
JEPA predicts `z1..z8`; and a shared WDL head predicts values from `z0..z8`.
There is no Hero teacher KL, no teacher checkpoint, no future-state teacher
forcing, no JEPA-to-DFM inference feedback, and no value-based action
selection. The JEPA loss still updates the encoder through `z0`, so raising
the encoder rate is a direct test of the hypothesis that Hero 1's encoder was
too static for the auxiliary dynamics objective to reshape it.

The LC0 sequential records provide the actual current record's root Q/D for
`z0` and successor root Q/D for predicted future states. Proposal A therefore
learns a direct leaf value suitable for search rather than approximating the
current value by negating the first successor.

## What LC0 search consumes

An LC0 backend evaluation must return:

- `p`: a finite normalized prior in exactly the engine-provided legal-move
  order;
- `q = P(win) - P(loss)`, from the side-to-move perspective;
- `d = P(draw)`; and
- optionally `m`, predicted remaining plies. We will initially advertise no
  moves-left head and disable MLH in matched comparisons.

LC0 generates successors and performs PUCT/backups itself. It does not need
our eight-ply trajectory, predicted JEPA states, per-action Q values, or a
model-generated tree. The primary full-model adapter will legal-mask and
softmax the DFM root-slot logits from one p1 call, and obtain Q/D by applying
the shared WDL head directly to `z0`. A second adapter will expose the updated
encoder's native policy/value heads to isolate how much useful chess
evaluation moved into BT4 itself.

## Immutable tuning data

The full converted archive contains 7,774 complete batch-1024 updates, or
7,960,576 examples actually consumed by one full-batch epoch. The local pilot
selects archive chunks `0,32,64,96,128,160,192,224,253`, including both
endpoints. It contains:

| Split | Positions | Trainable starts | Full batches |
|---|---:|---:|---:|
| train at 1024 | 264,105 | 261,701 | 252 |
| validation at 64 | 1,473 | 1,458 | 18 |
| test at 64 | 2,201 | 2,181 | 31 |

The pilot occupies 278,048,204 bytes and has inventory SHA-256
`4a3a640970aeac08fd152790e4b5dfe7b19ac600e6bec08d086eaf3510fe012e`.
It is hardlinked locally, so it consumes negligible extra WSL blocks, and is
uploaded once to a read-only Modal Volume. The test split remains sealed
during tuning.

## FP32 equal-rate screen

“FP32” means FP32 optimizer moments and FP32 master parameters with the
existing BF16 forward/backward compute. A full-FP32 activation run would be a
different memory and throughput experiment. Every arm uses:

- fresh initialization from raw BT4 plus the Proposal A heads;
- batch 1024, identical seed and shuffled batch order;
- encoder/main LR ratio exactly `1.0`;
- 2% linear warmup, 78% stable plateau, and 20% linear cooldown, clocked
  against the full 7,960,576-example epoch rather than the 252-step pilot;
- no illegal-mass auxiliary (`legality_coeff=0`), because legal support is
  exactly available at decoding and this screen should not let it hide policy
  movement;
- no weight decay in the LR stage;
- current-state WDL supervision enabled; and
- one held-out sequential validation pass after update 252, with no model or
  optimizer checkpoint retained.

The fixed LR candidates are `5e-5`, `1e-4`, `2e-4`, and `4e-4` for both the
encoder and all new modules. They span 3x to 24x Hero 1's encoder peak while
reducing the fresh-module peak to 0.1x to 0.8x Hero 1. This is a bounded sanity
screen, not a claim that 3.2% of an epoch establishes final strength.

Reject any arm with non-finite/skipped updates. Among the rest, the provisional
LR winner minimizes held-out total loss while remaining within 0.02 absolute
of the best root legal top-1 accuracy and within 0.05 CE of the best current
WDL loss. Call an LR improvement substantial only if it beats the `5e-5`
control by at least 0.10 total validation loss and 0.05 current-WDL CE without
losing more than 0.02 root top-1. Otherwise report a Pareto result and choose
the lower rate.

## Weight-decay follow-up

At the selected LR, compare the already-run no-decay result with:

1. schedule-normalized ordinary decoupled decay; and
2. the same coefficient with cautious coordinate masking.

Selective decay applies only to fresh matrix parameters, not BT4 encoder
weights, biases, norms, or embeddings. The LR multiplier first rescales decay
inversely to preserve planned integrated shrink; `weight_decay_multiplier=1`
then retains that matched shrink and `0` removes it. Prefer no decay when an
alternative improves total validation loss by less than 0.03 and current WDL
CE by less than 0.02, or loses more than 0.01 root top-1. This short prefix can
detect a badly mis-scaled coefficient but cannot estimate terminal one-epoch
regularization precisely.

## One-epoch preparation and later comparisons

The promoted recipe will run 7,774 updates. WSD warmup ends near update 156;
cooldown begins after update 6,219, so update 6,219 must be retained as the
reusable pre-decay checkpoint requested for continuation experiments. The
terminal update is 7,774. Exact recovery checkpoints, the model-only artifact,
and compact metrics live remotely; only compact reports return to the 30 GB
local workspace.

Promotion after the epoch requires a non-overlapping searchless arena against
Hero 1 and raw BT4, reporting Proposal A p1 and p8 separately. Search is a
second experiment: compare raw BT4 in native LC0 MCTS against each Proposal A
adapter with identical LC0 build, nodes, threads, batching, cache, openings,
colors, temperature, MLH setting, and search parameters. Only after a stronger
artifact passes these gates should the full interpretability stack be rerun.
