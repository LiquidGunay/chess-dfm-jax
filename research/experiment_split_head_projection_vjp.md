# Experiment 036: split head projection VJP

Status: rejected at the guarded core-VJP GPU compiler gate on 2026-07-23.
The CPU implementation and every preregistered CPU gate passed, but the core
compiler exceeded the fixed host-RAM ceiling. This is an exact reverse-mode
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
(z_all, z_dfm, s_rms) = P(tokens; theta_projection)
loss, aux             = L(z_all, z_dfm, s_rms; theta_core)
```

Here `P` is exactly the existing `state_projector`,
`normalize_jepa_state`, and `dfm_state_projector` computation. `L` is the
entire unchanged remainder of the normalized loss. It retains both noisy and
clean DFM planner calls, DFM CE and legality, the action-token conditioning of
the JEPA transition, JEPA positive loss, target SIGReg, z_pred SIGReg, and all
metrics. Thus the DFM-to-JEPA coupling remains inside one core graph.
`s_rms` is the projector-owned 4,096-byte FP32 RMS-scale vector needed to
preserve four existing auxiliary diagnostics. It is a forward-only boundary
argument excluded from the core VJP `argnums`; its trainable gradient still
flows exactly through `z_all`. It adds no parameter partition, cotangent, or
optimizer state.

Compose one logical update from six sequential JIT executables:

1. **Encode:** unchanged encoder-view current/K=1-future BT4 tokens.
2. **Project:** compute the exact `z_all`, `z_dfm`, and diagnostic `s_rms`
   from those tokens using a projection-only shared-variable view.
3. **Core VJP:** run the unchanged normalized loss from those latent
   overrides and the nondifferentiated diagnostic scale. Differentiate the
   34 core leaves plus `z_all` and `z_dfm`, returning loss, every auxiliary
   metric, core gradients, and both latent cotangents.
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

## Implementation and CPU-gate result

The implementation preserves the legacy two-part gradient helper as a CPU
reference and adds the exact three-part production path. The encoder,
projector, and core views contain only their respective trainable Variables,
share every retained canonical `Variable` object, and contain no copied array
or fixed legacy head. The accepted production ABI passed exactly:

```text
encoder      404 leaves   390,611,456 bytes
projection    17 leaves   108,385,280 bytes
core          34 leaves   206,990,616 bytes
full         455 leaves   705,987,352 bytes
```

The small-model exactness suite passed bitwise BT4-token and latent equality,
loss and full auxiliary-metric parity, merged monolithic/head/six-stage
gradient parity, latent and token cotangent checks, one- and two-update model
and optimizer parity, global clipping, nonfinite suppression, donated
execution, restored-view identity, and all six concrete lowerings. Ten focused
split tests and 17 compile-only tests passed. The broader gate covered 162
tests: 161 passed under the outer exclusive guard, while the test that must
itself acquire and challenge that lock passed separately under the same
two-CPU/RAM guard without an outer lock.

The production batch-128 CPU lowering audit passed every frozen argument
ceiling without executing a component or changing optimizer step:

```text
encode             427,868,168 bytes
project            141,939,712 bytes
core VJP           248,970,016 bytes
projection VJP     146,658,304 bytes
encoder VJP        461,422,600 bytes
update           2,566,875,381 bytes
```

The audit completed in 28.237 seconds with 3,592,171,520 bytes peak
process-group RSS and 8,035,336,192 bytes minimum MemAvailable. Persistent
CPU cache writes were disabled, optimizer step remained zero, and the
temporary launcher was deleted. Lint, `compileall`, and diff checks pass.
The shared cache remains exactly 67 executable files / 157,686,153 apparent
bytes, and the closing storage audit passes with exactly the seven retained
state files.

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

## GPU outcome

The first cold component, `encode`, passed:

- explicit compile time: 52.310 seconds;
- guarded process wall time: 74.936 seconds;
- peak process-group RSS: 4,010,414,080 bytes;
- minimum MemAvailable: 7,842,000,896 bytes;
- dynamic / compiler arguments: 427,868,168 / 423,641,608 bytes;
- compiler outputs / temporaries: 424,169,128 / 302,016,936 bytes;
- peak JAX device allocation: 1,893,640,704 bytes; and
- one new 1,076,350-byte executable with SHA-256
  `f462541eedde3e273a95432972a7b7496230e146df5cb64d5c394f369926da77`.

The second cold component, `project`, also passed:

- explicit compile time: 10.483 seconds;
- guarded process wall time: 32.201 seconds;
- peak process-group RSS: 3,791,073,280 bytes;
- minimum MemAvailable: 8,100,466,688 bytes;
- dynamic / compiler arguments: 141,939,712 / 141,939,712 bytes;
- compiler outputs / temporaries: 113,108,128 / 491,390,624 bytes;
- peak JAX device allocation: 2,199,305,472 bytes; and
- one new 311,510-byte executable with SHA-256
  `b8ebda4f2c550070be671600b55ede92ee93a7cded8a426c6292974e2a503c3e`.

The third component, `core_vjp`, failed the fixed host-RSS guard before
compilation completed. At 33.752 seconds the process group jumped to
11,656,556,544 bytes, exceeding the unchanged 7,247,757,312-byte ceiling.
The guard terminated the process while system MemAvailable remained
7,948,034,048 bytes. The server remained responsive. The component wrote no
executable, compiler report, model state, or checkpoint; its run directory
contains only the 31,229-byte `run_config.json`.

Per protocol, `projection_vjp`, `encoder_vjp`, and `update` were not compiled,
and the failed core was not retried. After preserving the two successful
reports and executable hashes above, their cache entries and access markers
were deleted. The current 67 executable names, sizes, and hashes exactly match
the pre-experiment manifest. Directory inode metadata is nonshrinking, so the
cache's apparent size is now 157,691,751 bytes, 5,598 bytes above the prior
apparent baseline despite identical executable contents. The closing storage
audit passes with exactly the original seven retained state files.

Experiment 036 therefore rejects exact head-graph splitting as the safe local
A10G compiler substrate. Its preregistered decision rule forbids another
exact core/projection split or a relaxed retry. The next experiment must use a
scientifically explicit parameter-efficient training method, or change the
execution framework in a separately reviewed systems plan, before any new
production compile.

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
