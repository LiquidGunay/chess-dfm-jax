# Experiment: native Torch hero incumbent Arena

Status: completed and accepted on 2026-07-28. This is an
evaluator-enablement and same-checkpoint parity test, not a model experiment.

## Motivation

The retained feedback-off/count-64 hero checkpoint scores `0.958984` against
raw BT4 over 256 frozen pairs at its confirmed one-pass inference setting.
That absolute anchor is useful, but its ceiling leaves little resolution for
ranking future autoresearch candidates.

Add an independently loaded native Torch hero as the opponent so candidates
can play the retained checkpoint directly. This produces a score centered
near `0.5` while preserving the same canonical codec, root legality mask,
static batching, and deterministic game runner.

## Implementation contract

Add a mutually exclusive `--opponent-torch-hero PATH` option to
`research/evaluate_arena.py`. It is valid only with
`--candidate-torch-hero`; both policies must stay in the single native Torch
runtime and run sequentially on the one guarded GPU.

The candidate and opponent must:

- resolve their checkpoint directory, manifest, run config, model config,
  state path, size, and SHA-256 independently;
- receive distinct role-prefixed model IDs even when the state SHA-256 is
  identical;
- instantiate separate model objects from their own recorded configs;
- verify the raw-BT4 initialization mapping and exact checkpoint restore
  independently;
- compile the same bounded regions recorded by each run;
- use the canonical LC0 1,858-action codec, strict root legality masking,
  trusted frozen histories, and static physical batches; and
- record whether either model uses JEPA feedback at inference.

Reject mixed JAX/native-Torch execution, diagnostics-on execution, invalid
refinement counts, incompatible codec or horizon contracts, and unrecognized
configuration keys before gameplay. Do not weaken state checksum or
checkpoint ABI checks.

CPU tests must cover CLI mutual exclusion, native-runtime mode validation,
distinct role IDs for an identical checkpoint digest, independent descriptor
selection, and opponent loading dispatch. Existing raw-BT4 and pass-count
paths must remain unchanged.

## Guarded parity smoke

After the CPU suite passes, run exactly:

```text
candidate:
  research/runs/torch_autoresearch_sigreg64_u1024_v1

opponent:
  research/runs/torch_autoresearch_sigreg64_u1024_v1

checkpoint SHA-256:
  05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9

tier:
  hero_development

pairs / block pairs / seed:
  16 / 16 / 0

refinement passes / policy batch cap / additional-ply cap:
  1 / 16 / 256
```

Use `research/run_gpu.sh` with the existing CPU/RAM/disk/GPU guard. Do not
overlap another GPU workload, write model state, or retain compiler debris
outside `/mountpoint/.exp`.

The smoke passes only if:

- the immutable contract records two `torch_hero` descriptors with the exact
  retained state SHA-256 and distinct candidate/opponent IDs;
- all 16 pair scores equal exactly `1.0` point out of `2.0`, so aggregate
  score is exactly `0.5` and the pentanomial is center-only;
- all 32 games terminate normally, with zero policy faults and zero cap
  draws;
- both policies report finite physical-call timing and complete canonical
  action coverage; and
- no model checkpoint is written.

If the smoke fails, keep raw BT4 as the only native Torch opponent and debug
the evaluator before training another model. If it passes, freeze the
retained checkpoint as the one-pass development reference for the next
model-side candidate. Raw BT4 remains a periodic absolute anchor, and
eight-pass evaluation remains a continuity diagnostic.

## Outcome

The guarded smoke completed in `107.37` seconds including two independent
model restores, lazy compilation, warmup, and gameplay. The Arena itself used
`14.97` seconds. The guard held the process to two CPUs, observed
`3,063,353,344` bytes peak process-group RSS, and never let host available
memory fall below `8,115,777,536` bytes. It planned and wrote zero
checkpoints.

Every preregistered gate passes:

- both contract roles are `torch_hero`, independently restore checkpoint
  SHA-256
  `05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9`,
  and have distinct role-prefixed IDs;
- every one of the 16 pair scores is exactly `1.0` point out of `2.0`, giving
  score `0.5` and pentanomial counts `[0, 0, 16, 0, 0]`;
- all 32 games terminate normally, with zero faults and zero cap draws;
- both roles have symmetric `7/18/7` W/D/L, evaluate exactly 1,646 positions
  in 173 physical calls, and cover every legal action;
- candidate/opponent mean physical calls are `28.085/28.903 ms`; and
- the Arena directory contains only its 20-KiB state and 104-KiB immutable
  block, with no model state.

Open one-pass candidate-versus-retained-incumbent Arena evaluation for the
next model-side experiment. The reproducible sealed summary is
`research/analysis/torch_hero_incumbent_arena_20260728.json`; its source Arena
state is
`artifacts/arena/torch-hero-selfmatch-p1-16pairs-v1/state.json`, SHA-256
`b793698a5d3b614b6963b8e7abd9f25e59e8658740b153d73d1fa89ef85308e0`.
