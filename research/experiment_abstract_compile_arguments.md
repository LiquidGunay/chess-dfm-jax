# Experiment 032: abstract-only compilation arguments

Status: preregistered on 2026-07-23. This is a training-systems memory
experiment; it cannot change the ordinary trainer, model, loss, data, update,
checkpoint, or inference semantics.

## Evidence and hypothesis

The accidental isolated cold compile in Experiment 031 crosses the 7 GiB
guard at `8.575 GB`, even without reading the legacy source state. Its cached
compile-only control still holds about `1.895 GB` of live JAX GPU buffers and
peaks near `4.54 GB` host RSS because it constructs concrete model and
optimizer values.

Those values are dynamic compilation inputs. Flax NNX accepts
`jax.ShapeDtypeStruct` as variable values, and a CPU prototype can lower an
NNX model/optimizer update after replacing concrete variable leaves with
shape/dtype/sharding/weak-type structs. The hypothesis is that converting only
the disposable compile process to abstract leaves after reporting its state
footprint preserves the exact HLO/cache key while releasing enough concrete
storage to fit cold backend compilation.

## Candidate

Change only the default-off `--compile-only` path in `research/train.py`.
After constructing the model/optimizer and computing their existing footprint:

1. map every raw model and optimizer variable leaf to
   `jax.ShapeDtypeStruct` using `jax.typeof` shape, dtype, sharding, and weak
   type;
2. map every compile-batch and RNG leaf the same way;
3. update only the disposable NNX model/optimizer objects with those abstract
   states;
4. delete all replaced concrete argument trees and run `gc.collect()` before
   calling `.lower(...).compile()`; and
5. capture the initial optimizer step before abstractification rather than
   reading an abstract step after compilation.

Do not touch the ordinary source-init, evaluation, training, checkpoint, or
inference paths. Do not read the legacy checkpoint in compile-only. Keep all
existing fail-closed CLI validation and zero-state behavior.

Add compact compile-only evidence:

- before/after canonical raw-JAX abstract-signature digests and leaf counts for
  model and optimizer;
- counts of concrete versus `ShapeDtypeStruct` leaves;
- live JAX GPU bytes immediately before and after abstractification; and
- current process RSS and host `MemAvailable` immediately before and after
  abstractification.

The before/after abstract records must be exactly identical; after conversion
every dynamic model, optimizer, batch, and RNG array leaf must be a
`ShapeDtypeStruct`.

## CPU and cached gates

Unit-test:

- conversion preserves path, shape, canonical dtype, weak type, sharding,
  memory space, and PyTree definition;
- concrete leaves are not retained in the converted state;
- a small donated NNX optimizer step lowers with abstract model, optimizer,
  batch, and RNG arguments; and
- compile-only reports the captured concrete optimizer step without executing
  abstract state.

Run the focused CPU suite and static checks under the resource guard. Then run
one compile-only process against the ordinary shared cache, whose cache path
matches the retained executable key. It must:

- finish within 300 seconds and use less than 30 seconds explicit
  compile/cache-load time;
- reproduce exact compiler FLOPs, bytes, transcendentals, argument/output/temp/
  generated-code sizes;
- reproduce exact pre/post abstract-signature records;
- contain only abstract dynamic leaves at lower time;
- reduce live JAX GPU bytes by at least 1.5 GiB before lowering;
- stay below the existing 4.5 GiB cached group-RSS gate; and
- write only three compact JSON files, zero state, and avoid the source
  checkpoint.

A cache miss, signature mismatch, insufficient release, compiler-field
change, timeout, guard stop, or artifact rejects the candidate before any
fresh-cache trial.

## One stricter cold trial

Only after the cached result and acceptance checks are committed, create the
previously nonexistent cache:

```text
.local/cache/jax/experiment032-abstract-args-v1
```

Run exactly one compile-only batch-128 cold process with no XLA, allocator, or
glibc override and a 900-second timeout. Use the now-default 50 ms polling and
strengthen the process-group RSS stop to `7,247,757,312` bytes (6.75 GiB),
which makes the success threshold itself the enforced ceiling and retains
256 MiB below the original 7 GiB guard. Keep the 8/3 GiB host-memory floors,
two CPUs, 30 GiB disk reserve, and zero checkpoints.

The cold stage passes only if it completes normally, writes the exact expected
training executable and bounded per-fusion cache, preserves all compiler and
abstract-signature fields, and never crosses the stricter RSS ceiling.

Only on a cold pass, run one completely ordinary restored model-only update
against the same cache with `--eval-batches 0`, compile-ahead, and zero
checkpoints. It must hit in under 30 explicit compile seconds, create no
second training executable, retain the exact compiler fields, and reproduce
within `1e-6` training loss `4.367137908935547` and DFM CE
`3.6751866340637207`. This execution gate uses the ordinary trainer and
unchanged 7 GiB ceiling.

Any failure stops the experiment. Do not rescue it with another trial, batch,
flag, timeout, allocator, graph, or resource relaxation. Remove disposable
partial caches after preserving compact evidence. Keep exactly seven state
files and leave the offline incumbent unchanged.
