# Experiment 029: actual legacy-restore ABI diagnostic

Status: preregistered on 2026-07-23. This is a read-only checkpoint/schema
diagnostic, not a model, training, or compiler experiment.

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
