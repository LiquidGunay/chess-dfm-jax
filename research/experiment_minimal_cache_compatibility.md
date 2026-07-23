# Experiment 031: minimal seeded-cache compatibility

Status: preregistered on 2026-07-23. This tests cache-key compatibility only;
it does not compile a fresh graph or change training code.

## Evidence and hypothesis

Experiment 030 proves exact equality of all 455 raw JAX abstract state records,
the PyTree definition, and the NNX graph definition before and after legacy
model-only restore. The earlier `research_state_abi` mismatch is irrelevant to
the JIT input signature.

The shared cache access markers identify the retained active executables:

```text
training key
jit_train_normalized_stage1_step_donated-bbf0f77836127e6cbe49c84d11012d562695d44bf2e597ff3ee76fe22f30cbd9
cache bytes 6,799,026
cache sha256 49af04f544865cb553c2ec1136fd8d0aac4b5c226826b45f57d9eaef85c83db4
last marker write 2026-07-23 13:08:28 UTC (Experiment 027 compile-only)

evaluation key
jit_eval_normalized_stage1_step-7ae5beae657a5b3311203fac6cc946aa9316c1192e5252f8d806c6330ba10692
cache bytes 1,935,619
cache sha256 d312609806ddd7a4545bb8d03f19a83ebd9a06b5c373d56ab756a7e016dba580
last marker write 2026-07-23 12:59:50 UTC (restored one-update control)
```

The hypothesis is that a compile-only constructor and an ordinary restored
trainer request the same training key. A cache directory containing only
these two pinned executables should support both without a miss.

## Fixed setup

Create the previously nonexistent directory:

```text
.local/cache/jax/experiment031-minimal-compat-v1
```

Copy only each pinned `-cache` file and its eight-byte `-atime` partner from
the shared cache. Verify both cache-file sizes and SHA-256 digests before and
after copying. Do not copy the shared autotune directory or any other
executable. Record the initial four-file inventory; marker contents may change
when JAX records an access, but executable bytes may not.

Use the exact accepted one-block-tail batch-128 graph, normalized objective,
balanced K=1 target sampling, losses `0.0/5.76/1.0`, rates `3e-5/1e-6`,
schedule `400/800/0.1`, seed 0, validation seed 10,000, global-permutation
data, donation enabled, and no checkpoint writes.

## Sequential gates

Run these as two separate, sequential GPU processes through the unchanged
exclusive resource guard.

1. Run compile-only with zero steps and zero validation against the minimal
   cache. It must finish within a 300-second timeout, report the exact retained
   compiler cost/memory fields, take less than 30 seconds in its explicit
   compile/cache-load section, stay below 4.5 GiB group RSS, avoid the source
   checkpoint, and write only its three compact JSON files.
2. Only on that pass, run the completely ordinary model-only source restore
   for one update and one validation batch against the same minimal cache. It
   must finish within 300 seconds, stay below the unchanged 7 GiB RSS guard,
   and reproduce within `1e-6`:

```text
initial validation CE 4.504147529602051
initial accuracy      0.10546875
training loss         4.367137908935547
training DFM CE       3.6751866340637207
final validation CE   4.493493080139160
final accuracy        0.10546875
```

The ordinary run must write zero checkpoints. Timings and throughput are
diagnostic, not parity fields.

After each process, inventory the dedicated cache. It must contain no new
`*-cache` executable. The two pinned executable files must retain exact size
and SHA-256; only access-marker bytes/timestamps and an empty implementation
directory may change. This exact inventory, sub-30-second explicit compile,
and normal completion jointly prove cache hits. A key miss is expected to
enter the much larger cold compiler working set and must be stopped by the
unchanged guard; do not retry it.

## Decision rule

Pass only if both gates and both inventories pass. Then the default-off
compile-only process is proven compatible with ordinary restored execution.
Remove the disposable minimal cache after preserving compact evidence, leave
the shared cache unchanged, and separately preregister one fresh-cache
compile-only trial with the existing 7 GiB guard and 256 MiB headroom
threshold.

Any timeout, guard stop, new cache key, executable-byte change, compiler-field
change, metric mismatch, source access in compile-only, checkpoint, or extra
state rejects the experiment. Do not rescue it with another cache seed, flag,
batch, graph, timeout, or resource limit. Keep exactly seven state files and
leave the offline incumbent unchanged.
