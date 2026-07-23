# Experiment 027: non-mutating isolated compile process

Status: preregistered on 2026-07-23. This narrows Experiment 026; it is a
systems experiment and cannot change the authoritative training process.

## Correction and hypothesis

Experiment 026 proved that a compile-only cache hit fits at `4.532 GB` group
RSS, but it also cleared the mapped BT4 dictionary in place. That violated the
constructor model ABI and changed the ordinary one-update result. Restoring
the ordinary lifetime returns the retained metrics exactly.

This follow-up changes one property: the disposable compile process may drop
its own local reference to the mapped NumPy tree after model construction, but
must not mutate or clear that tree. The ordinary source-init, evaluation,
training, checkpoint, and inference paths remain byte-for-byte unchanged.

The hypothesis is that ordinary Python reference release is sufficient to
avoid retaining unreferenced construction arrays while preserving the exact
model structure. If the model legitimately owns an array or container, its
reference remains live. The compile process still skips the 1.8 GiB legacy
checkpoint; parameter values remain dynamic JIT inputs.

## Candidate and hard parity

Replace the compile-only call to the mutating
`release_construction_parameter_payload` helper with:

```python
del model_params
gc.collect()
```

Remove the mutating helper and its weak-reference test. Do not alter the
ordinary or read-only checkpoint-evaluation paths. Keep the rest of
`--compile-only` fail-closed behavior and reporting unchanged.

Before any cold compile, the cached compile-only report must match the
authoritative restored state:

```text
model ABI     b53b21a8113b74655bb897e8171315503d31f4434d33eb2b579d83bb38614d13
model leaves  455
model bytes   705,987,352
optimizer ABI 6d45c99b53bf22d638ecdfe9f9a3cfae5f1a50d8ecfa4c54a41bb8c8b84b707e
opt leaves    2,743
opt bytes     1,145,636,465
```

It must also reproduce exact static compiler FLOPs, bytes, argument/output/
temp sizes, write zero state, avoid the legacy checkpoint, and remain below
`4.5 GiB` cached peak group RSS. The already restored ordinary cache-hit
control at commit `029f501` is authoritative: initial/final validation CE
`4.504147529602051/4.493493080139160`, zero checkpoints, and
`6,156,472,320` bytes peak group RSS.

## One cold-cache trial

Only after CPU tests and the cached ABI gate pass, use a previously nonexistent
cache:

```text
.local/cache/jax/experiment027-nonmutating-compile-v1
```

Run exactly one batch-128 compile-only process with the accepted one-block-tail
graph, normalized objective, balanced K=1 targets, fixed loss
`0.0/5.76/1.0`, rates `3e-5/1e-6`, schedule `400/800/0.1`, seed 0, cursor 0,
donation enabled, zero evaluation, and zero checkpoints. Use no XLA or glibc
override. Apply a 900-second timeout.

The cold compile passes only if it completes normally, writes a valid bounded
cache, retains the exact ABI and compiler contract, peaks at no more than
`7,247,757,312` bytes group RSS, and keeps every existing CPU, host-memory,
disk, and exclusive-lock guard unchanged.

Only on a pass, launch the completely ordinary one-update trainer against that
same dedicated cache. It must be a cache hit, reproduce the exact authoritative
initial/final validation CE and shared training metrics within `1e-6`, retain
the accepted compiler/GPU fields, and write no checkpoint. A cache miss,
guard stop, ABI mismatch, metric mismatch, timeout, or extra artifact rejects
the experiment immediately.

Do not rescue failure with another release mechanism, graph, batch, compiler
flag, allocator, timeout, or guard. Remove the dedicated cache and empty/
partial run directories after preserving compact evidence. On success, keep
the default-off compile-only capability as the required preflight for future
new graphs, remove the dedicated validation cache, and leave the offline
incumbent unchanged.
