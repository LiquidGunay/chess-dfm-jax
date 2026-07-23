# Experiment 036: split head projection VJP

Status: preregistered on 2026-07-23. This is an exact reverse-mode
composition and compiler-memory experiment. It changes no model parameter,
loss term, coefficient, data path, optimizer rule, precision, batch, or
inference function.

## Evidence and hypothesis

Experiments 034 and 035 both fail while compiling the complete head VJP.
Removing the encoder and every fixed legacy BT4 head from its arguments lowers
the observed abort RSS from 8,006,049,792 to 7,676,387,328 bytes, but the
compiler still crosses the fixed 7,247,757,312-byte process-group ceiling.
This rules out another argument-only graph-view variant.

The accepted non-BT4 trainable state is not one indivisible graph. Its
production root-level ABI is:

```text
latent projection
  state_projector       107,331,584 bytes   14 leaves
  jepa_state_norm             4,096 bytes    1 leaf
  dfm_state_projector     1,049,600 bytes    2 leaves
  subtotal              108,385,280 bytes   17 leaves

coupled DFM+JEPA core
  jepa_transition       172,122,112 bytes    7 leaves
  dfm_blocks             13,684,736 bytes   10 leaves
  remaining roots        21,183,768 bytes   17 leaves
  subtotal              206,990,616 bytes   34 leaves

complete head           315,375,896 bytes   51 leaves
```

The state-projector forward/backward is also a large activation graph. Cutting
reverse mode at its existing latent outputs therefore removes both
108,385,280 differentiated parameter bytes and the entire projector
forward/backward from the coupled loss executable. This is a material
activation/compiler split, not a relaxation of Experiment 035.

Hypothesis: separately compiling the latent projector and the coupled
DFM+JEPA core keeps both VJPs below the protected host-RAM ceiling while
preserving the exact full gradient.

The read-only production state audit completed under a 4-GiB RSS guard in
15.254 seconds, with 3,376,738,304 bytes peak group RSS and 8,432,365,568
bytes minimum MemAvailable. Its temporary launcher was deleted.

## Candidate execution

For the accepted precomputed BT4 tokens, define:

```text
(z_all, z_dfm) = P(tokens; theta_projection)
loss, aux      = L(z_all, z_dfm; theta_core)
```

Here `P` is exactly the existing `state_projector`,
`normalize_jepa_state`, and `dfm_state_projector` computation. `L` is the
entire unchanged remainder of the normalized loss. It retains both noisy and
clean DFM planner calls, DFM CE and legality, the action-token conditioning of
the JEPA transition, JEPA positive loss, target SIGReg, z_pred SIGReg, and all
metrics. Thus the DFM-to-JEPA coupling remains inside one core graph.

Compose one logical update from six sequential JIT executables:

1. **Encode:** unchanged encoder-view current/K=1-future BT4 tokens.
2. **Project:** compute the exact `z_all` and `z_dfm` from those tokens using a
   projection-only shared-variable view.
3. **Core VJP:** run the unchanged normalized loss from those latent
   overrides. Differentiate the 34 core leaves plus `z_all` and `z_dfm`,
   returning loss, every auxiliary metric, core gradients, and both latent
   cotangents.
4. **Projection VJP:** recompute `P`, contract its outputs with the stopped
   latent cotangents, and differentiate the 17 projection leaves plus the BT4
   token input.
5. **Encoder VJP:** unchanged token-cotangent contraction through the full
   current encoder and one-block future tail.
6. **Update:** merge the disjoint core, projection, and encoder states, then
   call the existing full optimizer once.

Build transient encoder, projection, and core graph views with
`nnx.clone(..., variables=False)`. Every retained leaf must be the canonical
`Variable` object. Only the canonical full model and optimizer are updated or
checkpointed. Views own no copied arrays and add no persistent state.

Do not split loss terms, change stop-gradient placement, approximate a
cotangent, separately clip partitions, increment more than one optimizer
step, accumulate across batches, freeze a parameter, change precision, add an
adapter, or alter the checkpoint ABI. Preserve the existing single global
clip, full-tree finite check, Muon/Adam partitioning, schedules, weight decay,
and data cursor.

Keep the accepted contract fixed:

```text
physical batch              128
RMS norm coefficient        0.0
target SIGReg coefficient   5.76
z_pred SIGReg coefficient   1.0
SIGReg estimator/count      V-stat / 64 examples / 1 reference
target sampling             balanced per-example K=1
future BT4 gradient tail    1 of 15 blocks
main / BT4 peak LR          3e-5 / 1e-6
schedule                    400--1200 cosine warmdown
data order / seed           global permutation / 0
```

The accepted static branch contract additionally requires no closed-loop
feedback, BT4 policy distillation, root-conditional CE, value/WDL objective,
target-variance hinge, or EMA target. Fail closed if any becomes active.

## CPU correctness and ABI gates

Before any GPU compile:

1. Require the encoder/projection/core view trainable states to be disjoint
   and to partition every canonical trainable path exactly:

   ```text
   encoder      404 leaves   390,611,456 bytes
   projection    17 leaves   108,385,280 bytes
   core          34 leaves   206,990,616 bytes
   full         455 leaves   705,987,352 bytes
   ```

   Require path, shape, dtype, and `Variable` object identity equality for
   every retained leaf. Reject copied arrays, extra variables, or fixed legacy
   heads in a view.
2. Require bitwise equality of BT4 tokens, `z_all`, and `z_dfm` between the
   monolithic, Experiment 035, and six-component paths.
3. Require loss and every scalar auxiliary metric within `1e-6`; require
   latent and token cotangent shape/dtype equality.
4. Compare the merged three-part gradients with both the monolithic gradient
   and Experiment 035's two-part gradient. Require identical state ABI, finite
   values, cosine at least `0.999999`, and relative L2 error at most `1e-5`
   for full, encoder, projection, and core partitions.
5. Starting from identical state, require one- and two-update model and
   optimizer relative L2 error at most `1e-5`. Exercise global clipping,
   full-tree nonfinite suppression, balanced target assignment, and the
   `14/1` future encoder boundary explicitly.
6. Exercise donated six-component CPU execution and prove the three views
   still share canonical variables after update and after rebuild following a
   restore.
7. Lower all six concrete production-shaped components without execution and
   independently count dynamic array arguments. Require:

   - encode at or below `432,905,816` bytes;
   - project at or below `150 MiB`;
   - core VJP at or below `260 MiB`;
   - projection VJP at or below `150 MiB`;
   - encoder VJP at or below `450 MiB`; and
   - update at or below `2.5 GiB`.

8. Run the focused split, checkpoint, model-parity, future-tail, SIGReg,
   target-sampling, compile-only, resource-guard, import, and storage suites
   under a two-CPU / 6.75-GiB parent guard. Disable persistent CPU compilation
   writes. Run lint, `compileall`, and diff checks.

No CPU process may initialize CUDA. All temporary files and caches remain
under `/mountpoint/.exp`.

## Guarded compiler gate

Start from the exact retained manifest of 67 executable cache files. Compile
encode, project, core VJP, projection VJP, encoder VJP, and update in that
order, one process and one GPU workload at a time. Stop after the first
failure.

Every process must run through `research/run_gpu.sh` with:

- the exclusive GPU lock;
- two CPUs and all library thread pools capped at two;
- 50 ms guard polling;
- at least 8 GiB MemAvailable at launch and 3 GiB while running;
- an enforced 6.75 GiB process-group RSS ceiling;
- at least 30 GiB free disk;
- a 300-second hard timeout; and
- checkpoint writes capped at zero.

Each component uses concrete arguments and the ordinary shared cache. Require:

- no guard stop, timeout, OOM, source/checkpoint mutation, or state write;
- at most one new top-level executable per component and six total;
- compiler temporary bytes below `8.5 GiB` for every component;
- compiler-reported argument bytes within the CPU ceilings above;
- every component output at or below the monolithic output size;
- optimizer step zero after every compile-only process; and
- exact before/after executable names, sizes, and hashes.

Do not retry a failed component, relax a resource/ABI gate, change batch size,
use abstract arguments, create a path-distinct cache, or compile a later
component after failure.

## Real-checkpoint parity and speed gates

Only after all six cold compiles pass, run a zero-checkpoint ten-update smoke
from the immutable source checkpoint with the fixed seed/data contract.
Compare it with the first ten records of
`future-tail1-balanced-k1-cosine-b128-30m-v1`. Require exact optimizer steps,
data cursors, target assignments/masks, SIGReg counts, unclipped/finite
status, and metric keys. Require update-1 common scalar deltas at or below
`2e-5` and common scalar deltas through update 10 at or below `5e-4`.

Then run one cached 30-update profile with 100 ms GPU monitoring and no state
write. Require:

- at least `145.0` end-to-end examples/s;
- at least `205.0` device examples/s;
- peak JAX HBM below `15 GiB`;
- exact encoded-board and gradient-boundary accounting;
- no NaN, Inf, clipped loss, skipped update, or unexpected cache entry; and
- exactly the original seven retained state files.

A full pass establishes the safe, forward-identical execution substrate and
permits a later separately preregistered FP32-BT4-master experiment. Failure
of core or projection VJP ends exact head-graph splitting and routes the next
experiment to a parameter-efficient scientific method. Failure only at update
routes to an exact optimizer-state/update partition. A throughput failure
rejects this path for autoresearch even if it is mathematically exact.
