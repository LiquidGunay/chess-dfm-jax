# Experiment 033: accepted-head source-encoder overlay

Status: rejected at the GPU implementation gate on 2026-07-23. This was a
read-only representation and systems diagnostic, not a training, compiler,
loss, or model experiment.

## Evidence and question

The accepted one-block-tail update 2,072 pays for gradients through all 15
current-board BT4 blocks and carries optimizer state for all
`195,305,728` encoder parameters. A guarded checkpoint comparison against
source step 265,000 finds that this produces very little stored movement:

- the embedding has relative L2 delta `2.7075e-6`, with `0.2688%` of BF16
  elements changed;
- individual transformer blocks have relative L2 deltas from `2.5460e-7` to
  `6.2125e-7`, with only `0.0071%` to `0.0250%` of elements changed;
- block 14 changes least by relative L2, so a one-block current-tail proposal
  is not supported by the observed state; and
- the non-encoder trainable state has relative L2 delta `4.9632e-3`, with
  `96.7657%` of elements changed.

The encoder is stored in BF16 and its peak learning rate is `1e-6`. This is
consistent with most adaptive updates rounding back to their old BF16 values,
but small weight deltas can still matter after 15 nonlinear blocks. The
decisive test is therefore functional: keep the accepted trained head and
exact evaluation graph, replace only its encoder values with source values in
memory, and measure the resulting held-out change.

## Read-only candidate

Add a small diagnostic entry point that:

1. reads the accepted update-2,072 checkpoint and restores its model only;
2. evaluates it on validation seeds 10,000 and 20,000, each with 64 fixed
   batches of 64 positions and deterministic `t=0`;
3. reads only `model_trainable` from the immutable source checkpoint, verifies
   exact encoder paths, shapes, dtypes, and bytes, and overlays only the
   `encoder` subtree in memory;
4. proves every non-encoder leaf remains object- and value-identical across
   the overlay;
5. reevaluates the hybrid on the identical batches and RNG keys; and
6. writes compact configuration, per-seed metrics, and aggregate report JSON
   only.

Do not write a hybrid checkpoint, optimizer, cache copy, dataset shard, or
temporary state. Do not change `research/train.py`, the accepted checkpoint,
source checkpoint, model configuration, loss, inference, or legality mask.
The overlay must preserve the exact model tree and abstract signature.

## Correctness and decision gates

Focused CPU tests must cover strict subtree compatibility, encoder-only
replacement, non-encoder identity, aggregate delta arithmetic, and fail-closed
path/shape/dtype mismatches. Static checks and the focused suite run under the
resource guard.

The one GPU process must use `research/run_gpu.sh`, the ordinary shared cache,
the exclusive lock, two CPUs, 50 ms guard polling, the 8/3 GiB host-memory
floors, the 7 GiB group-RSS ceiling, the 30 GiB disk reserve, a 300-second wall
timeout, and zero checkpoint writes. The retained evaluation executable must
be a cache hit: no new top-level `*-cache` entry may appear.

Before interpreting the overlay, the restored accepted control must reproduce
the retained two-pool means within `1e-6`:

```text
uniform DFM CE  4.495009120553732
horizon-1 CE    2.8470487520098686
accuracy        0.1104583740234375
legal mass      0.6488690404221416
```

Treat the stored encoder movement as functionally negligible for the
autoresearch proxy only if, both per seed and pooled:

- absolute uniform and horizon-1 CE deltas are at most `2e-4`;
- absolute legal-mass delta is at most `2e-4`; and
- absolute aggregate accuracy delta is at most `2 / 8192`.

Also report all other common scalar deltas without using them to weaken these
gates. Passing does not prove that frozen-backbone *training* has identical
dynamics; it authorizes a separately preregistered, forward-identical frozen
proxy baseline whose strength must be measured. Failure sends the systems
plan to split encoder/head gradient executables or FP32 encoder masters rather
than a last-block-only tail.

Any baseline mismatch, cache miss, guard stop, timeout, tree mismatch,
non-encoder mutation, unexpected artifact, or state write rejects the
diagnostic. Do not retry with fewer batches, a different threshold, another
checkpoint, or relaxed resource limit. Keep the seven retained states and the
offline incumbent unchanged.

## Outcome

Run `source-encoder-overlay-tail1-u2072-eval-v1` used implementation commit
`fefe5c8`. The accepted-control stage completed both preregistered validation
pools and exactly reproduced all four retained pooled means. The run then
failed before mutating the encoder: the evidence hasher attempted
`memoryview(...).cast("B")` on NumPy's BF16 buffer format `E`, which raises
`ValueError`. No overlay result exists, so this experiment is inconclusive
about the functional importance of the stored encoder movement and does not
authorize a frozen-backbone proxy.

The resource guard stopped normally after `89.445` seconds with zero
checkpoint writes, `3,778,281,472` bytes peak group RSS, and
`8,419,737,600` bytes minimum MemAvailable. The output contains only a
`27,093`-byte run configuration and `7,800` bytes of accepted metrics. All 67
shared-cache executable names, sizes, and SHA-256 hashes remain identical;
the cache remains `157,684,302` bytes. Storage audit still finds exactly the
seven allowed state files and no extra or missing state.

The raw-byte helper is hardened after the run by viewing a flattened
contiguous array as `uint8`, with an explicit BF16 regression test. This
postmortem fix is not grounds to rerun Experiment 033. Follow the conservative
branch of the preregistration: investigate a forward-identical split-gradient
or FP32-master training path, and do not infer that a last-block-only tail is
useful.
