# Experiment 030: JAX restore-signature diagnostic

Status: completed and passed on 2026-07-23. This was the final read-only
signature diagnostic before deciding whether isolated compilation is
structurally valid.

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

## Outcome

Commit `8642ccc` extended the standalone diagnostic after 115 guarded focused
CPU tests passed at `3,164,430,336` bytes peak process-group RSS and
`9,137,074,176` bytes minimum `MemAvailable`; direct invocation, Ruff,
byte-compilation, and diff checks also passed.

The single guarded measurement completed in `24.3526` seconds with:

- `5,428,547,584` bytes peak process-group RSS;
- `6,875,848,704` bytes minimum host `MemAvailable`;
- `1,894,685,184` bytes peak JAX GPU memory;
- all Experiment 029 source, constructor, restored reporting ABI, and
  optimizer invariants repeated exactly; and
- zero batches, graph lowering/compilation, model/optimizer execution, or
  checkpoint writes, with persistent cache writes disabled.

The hypothesis passes completely:

- all 455 before/after raw abstract records are byte-equal with digest
  `5fc36815eb505214856f36262f6e8e9bee91a5c7ff14d27a70ce4513bd640817`;
- both PyTree definitions are byte-equal with digest
  `775dd6145d61118d79b99ed993dd2a958f5b8fc705a02d450b870c7ad51c29f7`;
- NNX graph definitions compare equal and their 217,265-character stable
  representations share digest
  `213b0a116c760788410995a1534cf11541ef687089b06b690e073e68ebd7f1a4`;
  and
- the only measured difference is concrete storage type: all 455 constructor
  leaves are `jaxlib._jax.ArrayImpl`, while all 455 restored leaves are
  `numpy.ndarray`.

Therefore the `f9f9...b467` to `b53b...d13` reporting-ABI change is irrelevant
to the JAX abstract input and NNX graph signatures used for compilation. It
must not block isolated cache population. This does not itself authorize a
cold compile: next preregister a dedicated-cache compatibility experiment
that uses the exact JAX signature as its gate and proves an ordinary restored
one-update cache hit before spending a fresh cold-compile trial.

The complete report is `1,932,135` bytes, the shared cache remains
`157,684,302` bytes, and exactly seven state files remain. The offline
incumbent is unchanged.
