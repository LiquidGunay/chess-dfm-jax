# Experiment 026: separate shape-equivalent compilation process

Status: completed and rejected at cached state/update parity on 2026-07-23.
No fresh-cache cold compile was run. This was a training-systems experiment;
it was not allowed to change model values, gradients, updates, data, loss,
batch size, checkpoint ABI, or inference.

## Evidence and hypothesis

Five distinct batch-128 cold training compiles now reach
`10.944--11.098 GB` process-group RSS and are stopped by the immutable
`7,516,192,768`-byte guard. Disabling stable GPU autotuning leaves the peak at
`11,098,423,296` bytes, so candidate-kernel search is not the cause.

The exact accepted graph still executes from the ordinary persistent cache.
A guarded source-initialized one-update cache hit records:

```text
peak group RSS                6,158,090,240 bytes
minimum host MemAvailable     6,194,888,704 bytes
explicit cache load/compile      14.325261 s
initial validation DFM CE        4.504147529602051
final validation DFM CE          4.493493080139160
```

Repeating it with `MALLOC_ARENA_MAX=2` and `MALLOC_TRIM_THRESHOLD_=0` changes
peak group RSS by only `-16,277,504` bytes (`-0.264%`) and preserves the
reported metrics exactly. Allocator retention is therefore not a plausible
source of the roughly 3.6 GB deficit, and those environment changes must not
be adopted or tested in a cold compile.

Every ordinary training process currently expands the 321 MiB raw BT4 export,
constructs the model and optimizer, then verifies and decodes the 1.8 GiB
legacy state—including optimizer values that model-only initialization
deliberately ignores—before asking XLA to compile. JIT compilation depends on
the graph definition and array shapes/dtypes, not the dynamic parameter
values. The hypothesis is that a separate process can compile the identical
graph before legacy decoding, populate the persistent cache, and exit below
the guard. A subsequent ordinary source-initialized process can then execute
as the already-demonstrated safe cache hit.

## Implementation contract

Add `--compile-only` to `research/train.py`. It must:

1. resolve the same metadata, checked experiment overrides, explicit CLI
   objective settings, fixed data provenance, model structure, optimizer
   transform, donation mode, and training function as an ordinary run;
2. require `--compile-ahead`, `--init model-only`, `--steps 0`, zero training
   seconds, zero evaluation batches, no resume/eval/gradient mode, and no
   checkpoint request;
3. load the checksum-pinned raw BT4 export only to construct the exact dynamic
   model/optimizer shapes, then delete the mapped NumPy parameter tree and run
   garbage collection before lowering;
4. not open, hash, decode, restore, or otherwise read the 1.8 GiB legacy
   `state.npz`;
5. lower and compile exactly the ordinary donated normalized training call for
   data cursor 0 and seed 0;
6. execute no validation or optimizer update and write no checkpoint,
   `state.npz`, metrics row, or model value;
7. write only compact run configuration, compiler cost/memory analysis, and a
   terminal compile-only report; and
8. fail closed if any dynamic state ABI differs from the ordinary
   source-initialized graph.

The ordinary trainer remains authoritative for checkpoint verification,
source restore, initial validation, updates, evaluation, and reports. The two
processes are sequential and each must independently run through
`research/run_gpu.sh`; they never share live Python or GPU state.

Deleting `model_params` immediately after `create_joint_components` is also
required on every path. The mapped NumPy tree is construction-only and must
not remain host-resident during import, compilation, training, or evaluation.
CPU tests must prove that deletion does not alter the model/state ABI or
values.

## Fixed graph and guard

Keep the accepted one-block-tail graph and all contracts fixed:

- source step-265,000 model-only initialization in the execution process;
- balanced per-example K=1 targets and one trainable future BT4 tail block;
- full projector/DFM/recurrent-JEPA depths;
- uniform CE, legality `2.0`, and latent loss coefficients
  `0.0/5.76/1.0`;
- fixed 64-example V-statistic SIGReg;
- train/eval batches `128/64`, rates `3e-5/1e-6`, cosine
  `400/800/0.1`, seed 0, and global train permutation; and
- unchanged eight-pass legal-masked inference.

Both processes retain the exclusive GPU lock, two CPUs, 8 GiB launch floor,
3 GiB runtime floor, 7 GiB group-RSS ceiling, 30 GiB disk reserve, and zero
checkpoint writes. The compile-only cold stage has a 900-second wall timeout.
No threshold may be relaxed.

## Staged gates

1. Focused CPU tests must cover CLI fail-closed combinations, ordinary-path
   parity, shape/state ABI identity, construction-payload release,
   compile-only report semantics, zero update/checkpoint behavior, and
   compiler-cache configuration.
2. Run compile-only first against the ordinary incumbent cache. It must be a
   successful cache hit, write no state, and peak below `4.5 GiB` group RSS.
   Then run the ordinary cached one-update path and require the exact retained
   initial/final metrics above.
3. Use a previously nonexistent dedicated cache at
   `.local/cache/jax/experiment026-separate-compile-v1`. Run one cold
   compile-only process. It passes only if it completes in 900 seconds, peaks
   at no more than `7,247,757,312` bytes group RSS, keeps every existing floor,
   writes no state, and leaves a valid bounded cache.
4. Only on a cold-compile pass, launch the ordinary source-initialized
   one-update command against that same dedicated cache. Require a cache hit,
   zero checkpoints, exact discrete/configuration parity, and at most `1e-6`
   absolute difference in every shared floating loss/policy metric versus the
   retained accepted smoke.
5. Compare compiler cost/memory analysis with the retained accepted report.
   Static FLOPs, bytes, argument/output/temp sizes, and generated-code size
   must match within the existing serialization precision. Because the
   executable is identical and the execution process is ordinary, no
   30-update performance rerun is required unless a compiler field or
   one-update timing differs by more than 5%.

A failure ends the experiment. Do not rescue it with a different batch,
compiler flag, allocator, guard, timeout, checkpoint, model, or loss. Remove
the dedicated cache and partial/empty run directories after retaining compact
evidence. On success, keep the `--compile-only` path and use the ordinary
bounded cache for future new graphs; remove the dedicated validation cache.
This systems experiment never appends a model result row or changes the
offline incumbent.

## Outcome

Commit `56f96ad` implemented the preregistered compile-only path after 152
focused CPU tests passed. The cached compile-only process completed with zero
updates, checkpoint writes, validation batches, or legacy-state access. It
reproduced the accepted compiler FLOPs, bytes, and temp/output sizes, took
`14.9691` seconds, peaked at `4,531,879,936` bytes group RSS and
`1,894,689,280` bytes JAX HBM, and kept host `MemAvailable` at or above
`7,734,702,080` bytes. It passed the cached systems threshold.

Two parity failures stop the experiment before the fresh-cache stage:

1. the compile-only constructor model ABI digest is
   `f9f9bde96785c9b9bb24a3b12ac10bde16ec5761f501f3783fc45733c0f1b467`,
   while the restored authoritative model ABI is
   `b53b21a8113b74655bb897e8171315503d31f4434d33eb2b579d83bb38614d13`;
   leaf count and byte count match, but the immutable shape/dtype/path schema
   digest does not; and
2. clearing the mapped construction tree on the ordinary path preserves
   initial validation CE `4.5041475296` and pre-update loss
   `4.3671379089`, but post-update validation CE becomes `4.4944372177`
   instead of `4.4934930801`, an absolute failure of `0.0009441376` versus
   the frozen `1e-6` tolerance.

The guard records `5,418,979,328` bytes peak group RSS and
`6,971,273,216` bytes minimum `MemAvailable` for that failed execution gate.
No state or checkpoint is written.

Per the immutable contract, run no fresh-cache cold compile and do not use the
shared payload-clearing behavior. Commit `029f501` restores the ordinary
trainer and evaluation lifetime exactly. Its guarded one-update cache hit
returns initial/final validation CE `4.5041475296/4.4934930801`, peak group
RSS `6,156,472,320` bytes, and the accepted compiler/GPU-memory fields. Keep
`--compile-only` default-off, but do not call its mutating payload-clear helper
without a new preregistration.

The closing storage audit still finds exactly seven allowlisted states and
`157,684,302` bytes in the ordinary JAX cache. The next bounded variant may
drop only the compile process's local Python reference without mutating the
mapping and must prove the exact restored ABI digest before any cold compile.
