# Experiment 034: forward-identical split encoder gradients

Status: rejected at the guarded compiler gate on 2026-07-23. This is a
training-execution and compiler experiment, not a loss, architecture, data,
optimizer, or model-strength experiment.

## Evidence and question

The accepted batch-128 one-block-tail executable combines BT4 forward and
backward, the DFM/JEPA head backward, global clipping, and Muon updates in one
XLA graph. Its compiler estimate is:

- `14,257,494,163,456` FLOPs;
- `133,447,442,432` bytes accessed;
- `1,894,455,276` argument bytes; and
- `10,957,716,808` temporary bytes.

The original cold compile took `173.230` seconds. Later cold architecture
trials exceeded the protected host-memory budget, including after concrete
dynamic arguments were replaced by abstract signatures. This monolithic
compiler path is therefore not a safe autoresearch substrate.

The accepted model stores `195,305,728` BT4 encoder parameters in BF16. A
future FP32-master representation would add exactly `390,611,456` model bytes
before optimizer or compiler effects, increasing model trainable state from
`705,987,352` to `1,096,598,808` bytes. It would preserve sub-BF16 updates but
would not simplify the monolithic graph. First test whether reverse-mode
composition can be split without changing the training objective or update.

## Candidate execution

Add a default-off research execution mode that performs one logical optimizer
update in four sequential JIT executables:

1. **Encode:** reproduce the accepted current-board and balanced per-example
   K=1 future-board selection from the same batch and RNG, then return the
   exact BF16 BT4 tokens used by the monolithic loss.
2. **Head VJP:** run the unchanged normalized loss from those tokens,
   differentiating all non-BT4 `TrainableParam` leaves and the token input.
   Return the ordinary loss/aux metrics, head gradients, and token cotangent.
3. **Encoder VJP:** recompute the same BT4 tokens and contract them with the
   stopped token cotangent, differentiating only `BT4TrainableParam` leaves.
   Preserve the accepted full current-board gradient and the `14/1`
   detached/attached future-target boundary.
4. **Update:** merge the disjoint head and BT4 gradient states and call the
   existing full optimizer update once. The existing global clip,
   `apply_if_finite`, Muon/Adam partitioning, schedules, weight decay, and
   single optimizer step remain authoritative.

The candidate must not use separate per-partition clipping, separate optimizer
step counters, asynchronous execution, gradient accumulation, host-offloaded
gradients, new precision, or a new checkpoint representation. Materialized
tokens and cotangents are transient device arrays only. Batch size remains
128.

The model forward, inference path, legality mask, parameter tree, optimizer
tree, resume contract, and checkpoint ABI must remain unchanged. Keep the
accepted objective fixed:

```text
RMS norm coefficient       0.0
target SIGReg coefficient  5.76
z_pred SIGReg coefficient  1.0
target sampling            balanced per-example K=1
future BT4 gradient tail   1 of 15 blocks
main / BT4 peak LR         3e-5 / 1e-6
schedule                   accepted 400--1200 cosine warmdown
```

## CPU correctness gates

Before any GPU compile:

1. Prove that the head and BT4 filters are disjoint and their merge contains
   every and only ordinary `TrainableParam` path.
2. Prove bitwise equality between monolithic and split token selection and
   encoding for the active balanced-K1 and `14/1` future-tail semantics.
3. On deterministic small models, compare monolithic gradients with the
   merged split gradients path by path. Require identical shapes/dtypes,
   finite values, gradient cosine at least `0.999999`, and relative L2 error at
   most `1e-5` for both the BT4 and non-BT4 partitions.
4. Starting from identical model and optimizer states, require the loss and
   all scalar aux metrics to agree within `1e-6`, then require model and
   optimizer states after one and two updates to have identical ABIs and
   relative L2 error at most `1e-5`.
5. Exercise global clipping, nonfinite-update suppression, target sampling,
   and the future-tail boundary explicitly. Fail closed if a gradient path is
   missing, duplicated, or assigned to the wrong partition.
6. Keep monolithic mode the default and run the full focused checkpoint,
   model-parity, future-tail, SIGReg, and resource-guard suites plus lint and
   static checks.

No CPU test may initialize CUDA. All caches and temporary files remain below
`/mountpoint/.exp`.

## Guarded compiler gate

Compile the four components in order, one process and one GPU workload at a
time. Each component uses concrete arguments and the ordinary shared cache;
do not reuse the rejected abstract-argument mechanism or create a
path-distinct cache. Stop after the first failure.

Every process must run through `research/run_gpu.sh` with:

- the exclusive GPU lock;
- two CPUs and library thread pools capped at two;
- 50 ms guard polling;
- at least 8 GiB MemAvailable at launch and 3 GiB while running;
- an enforced 6.75 GiB process-group RSS ceiling;
- at least 30 GiB free disk;
- a 300-second hard timeout; and
- checkpoint writes capped at zero.

For each component, record cold compile wall time, guard peak RSS/minimum
MemAvailable, executable name/hash/bytes, compiler cost analysis, and compiler
memory analysis. Require:

- no guard stop, timeout, OOM, source/checkpoint mutation, or state write;
- at most one new top-level executable cache entry for that component and at
  most four across the experiment;
- compiler temporary bytes below `8.5 GiB` for every individual component,
  versus `10,957,716,808` monolithic bytes; and
- encode, head-VJP, and encoder-VJP arguments and every component output at or
  below the monolithic values; and
- update-component arguments at or below `2.5 GiB`.

The update exception is fixed before implementation reaches a GPU: its
interface necessarily contains the unchanged `1,851,623,817`-byte
model/optimizer state plus the `705,987,352`-byte merged gradient state, or
about `2.558 GB` before negligible scalar metadata. The `2.5 GiB` bound is
therefore an ABI-derived ceiling, not a post-result relaxation. Donation must
keep its output at or below the monolithic output size, and the temporary/RSS
gates remain the actual compiler-safety criteria.

Do not compile the next component if the current one fails. Do not relax a
resource gate or retry a failed component with another batch size, abstract
arguments, a different cache, or more host resources.

## Real-checkpoint parity and speed gates

Only after all four components compile, run a zero-checkpoint split smoke from
the immutable source checkpoint with seed 0, global-permutation data, and
batch 128. Compare its first ten update records with the retained accepted
one-block-tail run on the identical data/RNG sequence.

Require:

- exact optimizer steps, data cursors, target assignments, target masks,
  SIGReg sample counts, and unclipped/finite status;
- update-1 loss and every common scalar forward metric within `2e-5`;
- common scalar metric deltas through update 10 within `5e-4`;
- no missing or extra reportable metric;
- no checkpoint or hybrid state write; and
- exactly the original seven retained state files after the smoke.

Then run one cached 30-update profile with the accepted batch/data contract.
Relative to the matched one-block-tail profile, require:

- at least `145.0` end-to-end examples/s;
- at least `205.0` device examples/s;
- peak JAX HBM below `15 GiB`;
- identical encoded-board and gradient-boundary accounting; and
- no NaN, Inf, clipped loss, skipped update, or unexpected cache entry.

Report per-component and total update time so launch/materialization overhead
is visible. A passing result establishes a safe, forward-identical compiler
substrate; it is not a strength promotion and writes no model state. The next
separately preregistered experiment may add FP32 BT4 masters on this split
path. A failure does not authorize a frozen backbone or a last-block-only
current tail; it sends the systems plan to a separately justified
state-efficient update method such as an adapter or stochastic rounding.

## Outcome

All preregistered CPU correctness gates passed. The focused split-execution
suite passed 34 tests in 92.57 seconds, including bitwise token parity,
path-complete and disjoint gradient partitioning, monolithic-versus-split
gradient and two-update parity, global clipping, nonfinite suppression,
future-tail semantics, concrete lowering of all four components, and
fail-closed configuration checks. The broader guarded suite passed 158 tests
in 211.22 seconds. Its parent guard observed a 4,975,194,112-byte peak
process-group RSS and 7,269,494,784-byte minimum MemAvailable. Lint,
`compileall`, and diff checks were clean.

CPU JIT validation had written eight CPU-backend executables and their eight
access-time markers into the ordinary cache. Before the GPU gate, those exact
validation byproducts were removed. This restored the previously recorded
baseline of 67 executable entries and 157,684,302 bytes. The storage audit
then passed with exactly seven retained state files.

The first cold GPU component, `encode`, passed:

- explicit compile time: 50.789 seconds;
- process wall time: 71.834 seconds;
- peak process-group RSS: 4,919,771,136 bytes;
- minimum MemAvailable: 8,196,894,720 bytes;
- compiler arguments / outputs / temporaries:
  748,281,712 / 748,809,800 / 151,022,560 bytes;
- compiler FLOPs / bytes accessed:
  4,096,336,855,040 / 29,928,880,128;
- peak JAX device allocation: 1,928,431,104 bytes; and
- one new 1,085,760-byte executable with SHA-256
  `f88d4f4bd39cc761969142dd129b23a4a6b68891ccf5ccf0f1dc49f909fe770c`.

It wrote no checkpoint and reported no compiler-gate failure.

The second cold component, `head_vjp`, failed the fixed host-RSS guard before
compilation completed. At 34.724 seconds the process group reached
8,006,049,792 bytes, exceeding the preregistered 7,247,757,312-byte ceiling.
The guard terminated the process while system MemAvailable was still
7,964,499,968 bytes. No head executable was written; its output directory
contains only the 30,621-byte `run_config.json`. Per protocol, neither
`encoder_vjp` nor `update` was compiled, and the failed component was not
retried.

The closing audit finds 68 executable entries—the original 67 plus the valid
encode executable—and exactly the original seven retained state files. No
checkpoint was written. Exp034 therefore rejects this four-component
implementation as the safe compiler substrate. A new experiment must first
test whether the exact head graph can omit encoder model/optimizer payload
from its compilation ABI; if that cannot materially reduce host compilation
state, the next candidate must be a separately justified state-efficient
update method.
