# Experiment 029: actual legacy-restore ABI diagnostic

Status: completed and blocked at the leaf-signature rule on 2026-07-23. This
was a read-only checkpoint/schema diagnostic, not a model, training, or
compiler experiment.

## Evidence and question

Experiment 028 proved that constructor state, the accepted update-2,072
checkpoint manifest, and a value-preserving NNX self-round-trip are exactly
the same full model ABI `f9f9...b467`. Their leaf-only signature is also
identical, and the round-trip retains all 455 array objects. It does not
produce the live post-restore ABI `b53b...d13`.

The remaining controlled difference is the actual checksum-verified legacy
payload and the complete model-only restore operation. The existing ordinary
cache-hit control performs that restore safely at `6,156,472,320` bytes peak
group RSS, but retained reports store only its compact post-restore digest.
We need the exact schema-record diff before another compile-only proposal.

## Diagnostic contract

Add a second small standalone diagnostic under `research/`, reusing the
schema utilities from Experiment 028 and leaving `research/train.py`
unchanged. It must:

1. construct the exact accepted one-block-tail batch-128 model and fresh
   optimizer, with current model config byte-equal to the accepted update-2,072
   manifest;
2. record the full constructor model ABI plus compact optimizer ABI and step;
3. release only the diagnostic's local mapped-parameter reference;
4. call the production `import_legacy_checkpoint` once, against only the
   checksum/size/step-pinned
   `checkpoints/source/step0265000/checkpoints/step0265000/state.npz`, in
   `model-only` mode with the production asset manifest;
5. record the import-reported source model and optimizer ABIs, then the full
   live model ABI and compact fresh-optimizer ABI/step after restore;
6. report every missing, extra, and changed model schema record and separate
   leaf-only path/shape/dtype/byte digests before and after; and
7. write one compact JSON report and no state, checkpoint, metrics stream,
   training artifact, or persistent compilation-cache entry.

Fail closed unless the constructor and accepted checkpoint manifest are exact
full ABI `f9f9...b467`, the verified payload reports source model ABI
`f9f9...b467` and source optimizer ABI `3e794fe647c35f5b2d5e1f78c5625178c783cd19ec0a3fb1e9aa8f3d9e790b65`,
and the fresh optimizer remains step zero with runtime ABI `6d45...707e`.

The diagnostic must not instantiate a data loader or batch, lower or compile
the training graph, call model inference, execute an optimizer update, or
write a checkpoint. Disable persistent compilation-cache writes. Run it once
through the unchanged exclusive resource guard with two CPUs, the 8 GiB launch
floor, 3 GiB runtime floor, 7 GiB process-group RSS ceiling, 30 GiB disk
reserve, and zero planned checkpoint writes. A guard stop is a valid bounded
diagnostic failure and cannot be rescued by relaxed limits.

## Decision rule

If the post-restore live model is exact `b53b...d13`, preserve the complete
before/after record diff and classify whether any leaf path, shape, dtype, or
byte record changes. Do not infer cache compatibility from equal aggregate
leaf counts/bytes alone.

If only non-leaf container records change, the next separately preregistered
candidate may reproduce that structure without reading the source state and
must pass a dedicated-cache compatibility test before a cold graph. If any
leaf record changes, first determine whether a shape/dtype-only template can
be constructed without source values; do not compile until that mechanism is
explicit and tested. Any unexpected digest, optimizer mutation, guard stop,
or artifact also blocks compilation. Keep exactly seven state files and leave
the offline incumbent unchanged.

## Outcome

Commit `0f107f0` added the diagnostic after 112 guarded focused CPU tests
passed at `3,154,632,704` bytes peak process-group RSS and
`9,217,970,176` bytes minimum `MemAvailable`; direct invocation, Ruff,
byte-compilation, and diff checks also passed.

The single guarded production import completed in `24.3518` seconds with:

- `5,426,556,928` bytes peak process-group RSS;
- `6,862,241,792` bytes minimum host `MemAvailable`;
- `1,894,688,768` bytes peak JAX GPU memory;
- exact source size `1,851,704,172` bytes, SHA-256
  `16a3c7e77e411a8a7577ff04dac1ca5173ce24ecb343b5fa4938e2d2b5fb8906`,
  and step 265,000;
- exact source model/optimizer ABIs `f9f9...b467`/`3e79...0b65`; and
- unchanged fresh runtime optimizer ABI `6d45...707e` and step zero.

The live model changes from full ABI `f9f9...b467` to the expected
`b53b...d13`, so the diagnostic itself is authoritative. Its exact schema
diff has no missing or extra path and changes 404 leaf records:

- all 404 paths are below `encoder`;
- 248 are rank-one and 156 are rank-two;
- all 404 shapes and byte counts are identical; and
- every recorded dtype transition is `bfloat16` to `void16`.

The other 51 model leaves are unchanged. Consequently the aggregate
455-leaf/705,987,352-byte equality hid a uniform live-wrapper dtype-name
distinction, and Experiment 029 must remain blocked under its leaf-difference
rule. Code inspection gives a narrower next question: `research_state_abi`
walks the NNX mapping manually and applies `np.asarray` to its variable
objects, while JAX flattening exposes each variable's raw `.value` leaf. The
`void16` record may therefore be a reporting artifact rather than the JAX
abstract input dtype used by `nnx.jit`, but that has not yet been measured.

Do not compile. The complete report is `1,489,929` bytes, persistent cache
writes were disabled and the shared cache remains `157,684,302` bytes, and
exactly seven state files remain. The next bounded diagnostic must compare
raw JAX PyTree paths, shapes, canonical dtypes, weak types, and tree
definitions before and after the same verified restore.
