# Experiment 028: compile-state ABI schema diagnostic

Status: preregistered on 2026-07-23. This is a read-only systems diagnostic,
not a model or compiler experiment.

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
