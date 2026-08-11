# Experiment 028: compile-state ABI schema diagnostic

Status: completed and blocked on 2026-07-23. This was a read-only systems
diagnostic, not a model or compiler experiment.

## Evidence and question

Experiments 026 and 027 stopped because the compile-only constructor reports
model ABI `f9f9bde96785c9b9bb24a3b12ac10bde16ec5761f501f3783fc45733c0f1b467`
while a live model after legacy model-only restore reports
`b53b21a8113b74655bb897e8171315503d31f4434d33eb2b579d83bb38614d13`.
Both have 455 leaves and 705,987,352 bytes.

The retained evidence changes the interpretation of that mismatch. The
checksum-pinned legacy payload lineage reports model ABI `f9f9...b467`, and
the accepted update-2,072 research checkpoint manifest also stores the full
`f9f9...b467` model schema. Its live post-restore training report instead
stores compact ABI `b53b...d13`. The legacy restore path converts an NNX state
to a pure dictionary, calls `nnx.replace_by_pure_dict`, and then
`nnx.update`. The gate may therefore compare pre- and post-round-trip
container representations even when the dynamic array signature is
unchanged.

The question is exact: does applying that same round-trip to the constructor's
own values change only state-container records, or does it change any leaf
path, shape, dtype, or byte count relevant to a JIT call?

## Diagnostic contract

Add a small standalone diagnostic under `research/` without changing
`research/train.py`, the model, optimizer, data, loss, training function, or
ordinary trainer. It must:

1. construct the exact accepted one-block-tail batch-128 model and optimizer
   from the mapped BT4 parameters;
2. record the full constructor model ABI and compact optimizer ABI;
3. derive a pure dictionary from the constructor trainable state, apply the
   same `nnx.replace_by_pure_dict` plus `nnx.update` round-trip using those
   same constructor arrays, and record the resulting full model ABI;
4. report all missing, extra, and changed schema records, plus a separate
   digest over leaf path/shape/dtype/byte records only;
5. report how many pure-dictionary leaves retain identical Python array
   objects across the round-trip;
6. compare both digests with the fixed `f9f9...b467` and `b53b...d13`
   references and the accepted checkpoint manifest; and
7. write one compact JSON report and no model state, checkpoint, cache,
   metrics stream, or training artifact.

The diagnostic must not open `state.npz`, create a training batch, lower or
compile the training graph, execute a model or optimizer step, or touch the
GPU concurrently with another workload. Run it once through the unchanged
exclusive resource guard with two CPUs, the 8 GiB launch floor, 3 GiB runtime
floor, 7 GiB process-group RSS ceiling, 30 GiB disk reserve, and zero planned
checkpoint writes. Unit-test the schema comparison and fail-closed paths on
small in-memory trees before the guarded run.

## Decision rule

If the constructor is `f9f9...b467`, the self-round-tripped model is
`b53b...d13`, all leaf records and array objects are unchanged, and only
container schema records differ, then the earlier exact-runtime-ABI gate was
overly broad. Preregister a separate candidate that canonicalizes only the
disposable compile process with this value-preserving round-trip, then repeat
the cached compiler gate before considering one cold compile.

If any path, shape, dtype, byte count, array object, optimizer ABI, reference
digest, or checkpoint-manifest comparison differs, stop. Do not compile and
inspect the exact discrepancy first. In either case keep exactly seven state
files and leave the offline incumbent unchanged.

## Outcome

Commit `ec04ebd` added the standalone diagnostic after 106 guarded focused
CPU tests passed at `3,153,174,528` bytes peak process-group RSS and
`9,226,035,200` bytes minimum `MemAvailable`; Ruff, byte-compilation, and diff
checks also passed. Commit `0b6210d` added the same direct-script repo-root
bootstrap used by `research/train.py` after the first invocation exited
before model construction. That failed launcher attempt peaked at only
`202,199,040` bytes RSS and wrote nothing.

The corrected guarded diagnostic completed its measurements in `19.3314`
seconds and deliberately returned the blocked decision:

- peak process-group RSS was `3,559,596,032` bytes;
- minimum host `MemAvailable` was `8,838,500,352` bytes;
- peak JAX GPU memory was `1,894,688,768` bytes;
- persistent compilation-cache writes were disabled;
- the constructor model ABI and accepted update-2,072 manifest were exactly
  the same full `f9f9...b467` schema;
- the constructor, self-round-trip, and manifest leaf-only signature was
  `2888cc219245647e4f645ee13ac7cdafa628f737d95334a9a9da0bf4728eb0d3`;
- the self-round-trip remained full ABI `f9f9...b467`, with zero missing,
  extra, changed, or leaf-different schema records;
- all 455 pure-dictionary leaves retained identical Python array objects; and
- optimizer ABI remained exact `6d45...707e`.

Thus the NNX pure-dictionary round-trip with constructor arrays is not the
cause. A one-leaf CPU check also preserved its schema after replacing a JAX
array with a host NumPy array, but that does not substitute for the actual
restore. The required `b53b...d13` state cannot be synthesized or justified
from this evidence. Do not compile. The compact report is `2,327` bytes, the
shared cache remains `157,684,302` bytes, and exactly seven state files
remain. The next bounded diagnostic must compare the live model immediately
before and after the actual verified legacy model-only restore.
