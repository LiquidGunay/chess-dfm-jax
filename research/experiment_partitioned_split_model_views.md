# Experiment 035: partitioned model views for split reverse mode

Status: rejected at the guarded head-VJP compiler gate on 2026-07-23. This is
an exact execution-ABI and compiler-memory experiment. It does not change the
model, loss, data, optimizer, precision, batch, or inference function.

## Evidence and question

Experiment 034 proved the four-component split mathematically correct on CPU,
then rejected its implementation at the second cold GPU compile. The encode
component passed at 4,919,771,136 bytes peak process-group RSS, but head VJP
crossed the fixed 7,247,757,312-byte ceiling after 34.724 seconds and was
terminated at 8,006,049,792 bytes. No head executable or checkpoint was
written.

The failed head call received the complete `JointLatentSASAModel`, even though
precomputed BT4 tokens make every encoder variable dead in that graph. The
inverse is true for encode and encoder VJP: they received every JEPA/DFM head
variable even though they use only BT4. NNX graph arguments preserve those
unused variable states before XLA dead-code elimination.

The accepted model has:

- `705,987,352` bytes of trainable model state;
- `390,611,456` bytes in `195,305,728` BF16 BT4 parameters; and
- `315,375,896` bytes in all non-BT4 trainable parameters.

A CPU design probe verified that `nnx.clone(model, variables=False)` creates a
new module graph while sharing the exact `Variable` objects. Removing the
cloned `encoder` attribute leaves the same 53 non-BT4 trainable paths and
does not mutate the canonical model. Thus the head compiler ABI can omit
390,611,456 bytes without copying or changing one parameter value. A
symmetric encoder-only view can omit 315,375,896 bytes.

Question: are exact, shared-variable model views sufficient to keep every
split component below the protected host-RAM ceiling?

## Candidate execution

Build two transient views once, after model initialization or restore:

1. **Encoder view:** a shared-variable clone containing the encoder and the
   static configuration/scalars needed by the accepted encoding functions,
   but no non-BT4 variable root.
2. **Head view:** a shared-variable clone containing every non-BT4 variable
   and the same static configuration/scalars, but no encoder.

Use the encoder view for split encode and encoder VJP. Use the head view for
head VJP. Continue to use the canonical full model and canonical full
optimizer for the single merged-gradient update. The views are not optimizer
targets, are not checkpointed, own no copied arrays, and must continue to
observe canonical variable updates through object identity.

The component math remains exactly Experiment 034:

1. encode the accepted current board and balanced per-example K=1 future
   target;
2. compute unchanged head gradients and the BT4-token cotangent;
3. recompute the same BT4 tokens and compute encoder gradients from that
   cotangent; and
4. merge the disjoint states and call the existing optimizer once, preserving
   its single global clip, all-tree finite check, Muon/Adam partitions,
   schedules, weight decay, and step counter.

Do not copy parameters between views, update a view separately, alter a state
path, add a persistent state, or change the checkpoint representation. Do not
introduce adapters, frozen-backbone training, stochastic rounding, gradient
accumulation, per-partition clipping, or separate optimizer steps in this
experiment.

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

## CPU correctness and ABI gates

Before any GPU compile:

1. Prove the encoder and head views contain disjoint trainable paths whose
   union exactly equals the canonical model. Require identical path, shape,
   dtype, and variable object identity for every retained leaf.
2. Require the view ABIs to be exactly `390,611,456` and `315,375,896` bytes,
   with no copied array storage and no change to the canonical
   `705,987,352`-byte model ABI.
3. Prove a head view observes an optimizer update applied through the
   canonical model, and that rebuilding views after restore preserves the
   same state schema and shared identities.
4. Against both the current full-model split implementation and the
   monolithic objective, require bitwise token equality, loss and every scalar
   auxiliary metric within `1e-6`, gradient cosine at least `0.999999`,
   relative L2 error at most `1e-5` by full/head/encoder partition, and
   one- and two-update model/optimizer relative L2 error at most `1e-5`.
5. Exercise the accepted future-tail boundary, global clipping, nonfinite
   suppression, donated four-component execution, and fail-closed view/path
   validation.
6. Lower all four components from concrete production-shaped arguments
   without execution. Record array argument bytes independently of compiler
   reporting and require:
   - encode arguments at or below `432,905,816` bytes;
   - head-VJP arguments at or below `400 MiB`; and
   - encoder-VJP arguments at or below `450 MiB`.
7. Run the focused split, checkpoint, model-parity, future-tail, SIGReg,
   target-sampling, compile-only, resource-guard, import, and storage suites
   under a two-CPU / 6.75-GiB parent guard. Keep persistent CPU compilation
   disabled so validation cannot alter the shared GPU cache. Run lint,
   `compileall`, and diff checks.

No CPU process may initialize CUDA. All temporary files and caches remain
under `/mountpoint/.exp`.

## Implementation and CPU-gate result

The initial 53-leaf design probe used the small correctness model. The
production model has 455 trainable leaves: 404 encoder leaves and 51 head
leaves. The byte accounting in the preregistration was already the production
accounting and remains unchanged.

The first production argument audit also exposed `4,226,560` bytes of fixed
legacy policy, value, and moves-left head variables nested inside `BT4Model`.
Those heads are not called by `encode_tokens` or by the token-cotangent encoder
VJP. The transient encoder view now removes those three fixed heads as well as
all non-BT4 trainable roots. The canonical model retains them for inference
and checkpoint compatibility. This change satisfies the original argument
ceiling; no ceiling was raised.

The guarded production CPU probe passed with the exact trainable ABI:

```text
full model       455 leaves   705,987,352 bytes
encoder view     404 leaves   390,611,456 bytes
head view         51 leaves   315,375,896 bytes
```

All retained view leaves share the canonical `Variable` objects. Neither view
contains a copied array or an extra non-trainable variable. Concrete
production arguments lowered without execution at:

```text
encode          427,868,168 bytes
head VJP        386,187,040 bytes
encoder VJP     461,422,600 bytes
update        2,566,875,381 bytes
```

All are below their frozen ceilings, and optimizer step remained zero. The
probe completed in 28.738 seconds with 3,686,354,944 bytes peak process-group
RSS and 8,310,370,304 bytes minimum MemAvailable. Its temporary launcher was
deleted.

The focused view/split/compile-only suite passed 38 tests. The broader
checkpoint, model-parity, future-tail, SIGReg, target-sampling, compile-only,
resource-guard, import, and storage suite passed 162 tests in 242.39 seconds.
The parent guard observed 4,938,330,112 bytes peak process-group RSS and
7,066,775,552 bytes minimum MemAvailable. Lint, `compileall`, and diff checks
passed. Persistent CPU cache writes were disabled; the shared GPU cache
remains at exactly 67 executable files, and the storage audit finds exactly
the original seven retained states.

## Guarded compiler gate

Start from an exact manifest of 67 retained executable cache files. The
deleted Experiment 034 encode entry is preserved by its committed compiler
report and SHA-256, not retained as an executable. The cache directory's
apparent size is not a gate because its directory inode retained 1,851 bytes
of filename metadata after deletion; executable names, sizes, hashes, and
additions are authoritative.

Compile encoder-view encode, head-view head VJP, encoder-view encoder VJP, and
the unchanged full update in that order. Use one process and one GPU workload
at a time. Stop after the first failure. Every process must run through
`research/run_gpu.sh` with:

- the exclusive GPU lock;
- two CPUs and library thread pools capped at two;
- 50 ms guard polling;
- at least 8 GiB MemAvailable at launch and 3 GiB while running;
- an enforced 6.75 GiB process-group RSS ceiling;
- at least 30 GiB free disk;
- a 300-second hard timeout; and
- checkpoint writes capped at zero.

Each component uses concrete arguments and the ordinary shared cache. Require:

- no guard stop, timeout, OOM, source/checkpoint mutation, or state write;
- at most one new top-level executable per component and four total;
- compiler temporary bytes below `8.5 GiB` for every component;
- compiler-reported encode, head-VJP, and encoder-VJP argument bytes within
  the CPU ABI ceilings above;
- every component output at or below the monolithic output size;
- update arguments at or below the preregistered `2.5 GiB` Exp034 exception;
  and
- optimizer step zero after every compile-only process.

Record cold compile wall time, guard peak RSS/minimum MemAvailable, executable
name/hash/bytes, compiler cost analysis, compiler memory analysis, and the
exact before/after executable manifests. Do not retry a failed component,
relax a gate, change batch size, create a path-distinct cache, or compile the
next component after failure.

## Real-checkpoint parity and speed gates

Only after all four cold compiles pass, run a zero-checkpoint ten-update smoke
from the immutable source checkpoint with the fixed seed/data contract.
Compare against the first ten records of
`future-tail1-balanced-k1-cosine-b128-30m-v1`. Require exact optimizer steps,
data cursors, target assignments/masks, SIGReg sample counts, unclipped/finite
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
of head VJP despite its exact reduced ABI rejects further graph-view retries.
Failure only at the full update routes the next systems experiment to an
exact optimizer-state/update partition. If exact partitioning cannot satisfy
the guard, only then consider a separately justified state-efficient
scientific method such as adapters or stochastic rounding.

## GPU outcome

The partitioned encode component passed:

- explicit compile time: 50.440 seconds;
- process wall time: 73.501 seconds;
- peak process-group RSS: 4,918,267,904 bytes;
- minimum MemAvailable: 7,988,039,680 bytes;
- independent dynamic arguments: 427,868,168 bytes;
- compiler arguments / outputs / temporaries:
  423,641,608 / 424,169,128 / 302,016,936 bytes;
- compiler FLOPs / bytes accessed:
  4,096,336,855,040 / 29,279,604,736;
- peak JAX device allocation: 1,929,288,192 bytes; and
- one 1,076,336-byte executable with SHA-256
  `469c0ac8bdf82b92268bf2160c675a78b8c2914a1a76b57ca05be9dae9c3be09`.

All component gates passed and no checkpoint was written. Relative to
Experiment 034 encode, compiler argument bytes fell by 324,640,104 and output
bytes fell by 324,640,672.

The partitioned head VJP still failed the fixed host-RSS gate before
compilation completed. The guard terminated it after 30.040 seconds at
7,676,387,328 bytes process-group RSS, above the fixed 7,247,757,312-byte
ceiling. Minimum system MemAvailable remained 8,149,041,152 bytes. This is a
329,662,464-byte reduction from Experiment 034's observed abort RSS, but still
428,630,016 bytes over the limit. The head run wrote only its 30,929-byte
`run_config.json`; it produced no executable or checkpoint.

Per protocol, encoder VJP and update were not compiled, and head VJP was not
retried. The valid encode executable and access marker were deleted after
their compact report and hash were preserved, returning the shared cache to
the exact 67-file executable manifest. The cache directory's apparent size is
157,686,153 bytes because of the previously documented nonshrinking directory
metadata. The closing storage audit finds exactly the original seven retained
state files.

Exp035 therefore rejects model-graph partitioning as a safe batch-128
compiler substrate. Its forward-identical implementation remains useful
evidence but is not authorized for training. The preregistered decision rule
forbids another head graph-view retry. The next proposal must materially
reduce differentiated head state or activation/compiler complexity, with a
scientific tradeoff stated explicitly—for example, a parameter-efficient
adapter/low-rank update—or use a separately justified stochastic update
method. It must preserve the fixed loss, data, evaluation, legality, and
zero-excess-checkpoint contracts.
