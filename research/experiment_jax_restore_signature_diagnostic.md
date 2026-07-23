# Experiment 030: JAX restore-signature diagnostic

Status: preregistered on 2026-07-23. This is the final read-only signature
diagnostic before deciding whether isolated compilation is structurally valid.

## Evidence and hypothesis

Experiment 029 reproduces live model ABI `f9f9...b467` to `b53b...d13` and
finds 404 changed records. Every change is an encoder-only `bfloat16` to
`void16` transition; paths, shapes, and byte counts are identical. The
verified incoming payload itself has full ABI `f9f9...b467`.

`research_state_abi` recursively walks the NNX state mapping and applies
`np.asarray` to each terminal variable object. JAX instead flattens an NNX
variable through its `.value` child when forming JIT inputs. A small CPU check
shows that `jax.typeof` assigns the same `bfloat16[shape]` abstract value to a
JAX bfloat16 array and a NumPy bfloat16 array. The `void16` string may
therefore describe wrapper conversion in the checkpoint-reporting ABI without
changing the actual JAX compilation signature.

The hypothesis is exact: raw JAX PyTree leaf paths and abstract values, the
PyTree definition, and the NNX graph definition are identical before and
after the production model-only restore even though `research_state_abi`
changes.

## Diagnostic contract

Extend only the standalone legacy-restore diagnostic; do not change
`research/train.py`. Before and after the same single checksum-verified
model-only import used by Experiment 029, record:

1. `jax.tree_util.tree_flatten_with_path` paths for
   `nnx.state(model, TrainableParam)`;
2. for every raw leaf, `jax.typeof` shape, canonical dtype, weak type,
   sharding, memory space, and abstract-value text;
3. a canonical digest over those records;
4. the PyTree definition text and digest;
5. concrete raw-leaf Python type counts as a diagnostic field that is not part
   of the abstract-signature digest; and
6. `nnx.split(model, TrainableParam, ...)` graph-definition equality and
   stable representation digests.

Retain all Experiment 029 source identity, model schema, optimizer ABI/step,
artifact, and fail-closed checks. Unit-test the signature helper on JAX and
NumPy arrays with equal and unequal abstract values. Write one JSON report and
no state or metrics stream.

Run once with persistent compilation-cache writes disabled through the
unchanged exclusive guard: two CPUs, 8 GiB launch `MemAvailable`, 3 GiB
runtime floor, 7 GiB process-group RSS ceiling, 30 GiB disk reserve, and zero
checkpoint writes. Do not create a data loader or batch, lower or compile a
training graph, run inference, or execute an optimizer update. Do not relax
the guard or retry a completed measurement.

## Decision rule

Pass only if:

- Experiment 029's exact source, constructor, post-restore, and optimizer
  fields repeat;
- before/after raw JAX signature digests and complete records are identical;
- PyTree definitions are identical;
- NNX graph definitions compare equal and have identical representation
  digests; and
- the only expected diagnostic difference is concrete raw storage type
  (`jax.Array` constructor leaves versus NumPy payload leaves).

On a pass, classify the `f9f9...b467`/`b53b...d13` mismatch as irrelevant to
the JIT input signature, but do not cold-compile yet. Preregister a separate
compile-only candidate that replaces the broad reporting-ABI gate with this
exact JAX abstract-signature gate and first proves a dedicated-cache hit by an
ordinary restored one-update process.

Any path, abstract shape/dtype/weak-type/sharding/memory-space, PyTree, graph
definition, optimizer, source, guard, or artifact difference blocks
compilation. Keep exactly seven state files and leave the offline incumbent
unchanged.
