# Experiment: post-Hero inference-pass and horizon-bootstrap audit

Status: running on 2026-07-28. This is a measurement and design audit. It
does not promote a checkpoint or authorize a new training loss.

## Questions

1. Which checkpoint was the opponent in the `0.509765625` direct Arena for
   current-JEPA conditioning?
2. Does the retained one-epoch Hero still become weaker with additional DFM
   refinement passes, as the early 1,024-update model did?
3. Does the 1,024-update current-JEPA conditioning candidate have a different
   pass curve from its matched control?
4. How much of weak early H2--H8 behavior follows from initialization, and
   what target-safe bootstrap could improve it?

## Model identities

Keep these three models distinct:

| Role | Updates | Checkpoint SHA-256 |
|---|---:|---|
| Matched 1,024-update control | `1,024` | `05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9` |
| Current-JEPA broadcast candidate | `1,024` | `eb20c698fea1eabd4ebf71c13a1c195204736183f7b173079bd34f2a6e9a995f` |
| One-epoch Hero | `27,679` | `665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692` |

The `0.509765625` score was candidate versus the matched 1,024-update
control, not versus the one-epoch Hero. The prior use of “retained Hero” for
the short control was ambiguous and must not be repeated.

## Frozen pass-count measurement

For the one-epoch Hero, measure passes `1, 2, 4, 8, 16` against immutable raw
BT4 on the first 128 color-reversed `hero_development` opening pairs. Reuse
the first 128 pairs of the existing 1,024-pair eight-pass Arena only after
checking checkpoint, opening pool, history sidecar, seed, batch cap, ply cap,
codec, and inference settings. Run the other four counts in new directories.

For the current-JEPA candidate, the tensor was deleted after the original
offline-gate rejection. Reproduce the exact 1,024-update training contract
from raw BT4 with the same seed, data order, batch, schedule, losses, systems
settings, and sole active conditioning flag. Compare the resulting checkpoint
hash and scalar trace with the sealed original:

- if the checkpoint hash is exact, call it a restored deterministic replay;
- if the hash differs but the trace and frozen metrics reproduce within
  ordinary numerical tolerance, call it an independent matched replicate;
- otherwise stop and do not attribute its pass curve to the deleted model.

After successful reproduction, measure the same `1, 2, 4, 8, 16` pass curve
against raw BT4 on the same first 128 pairs. Freeze:

```text
policy batch cap / additional-ply cap / seed:
  16 / 256 / 0
```

Every run must have zero faults, zero cap draws, complete legal-action
coverage, and normal termination. These openings have already been used for
model development, so all pass curves are descriptive rather than promotion
tests. Do not infer an absolute Elo.

Retain the reproduced candidate tensor until this audit and its interpretation
are explicitly closed. Do not delete it solely because an offline surrogate
gate failed.

## Initialization audit

The current Hero recipe has two different bootstraps:

1. DFM horizon one permanently adds the raw BT4 current-board policy logits
   as a residual. The learned DFM output projection starts at zero, so H1 is
   exactly BT4 at initialization.
2. H2--H8 have no analogous future-board policy residual. Their learned output
   projection starts at zero, hence their initial distribution is uniform.

The recurrent JEPA transition is separately initialized close to identity:
conditioning modulation starts at zero and the residual down projection is
tiny. Thus every early predicted future state is approximately the current
state. Quantify JEPA MSE relative to the identity baseline, not by raw MSE
alone, at initialization, 1,024 updates, and one epoch.

The preferred future-policy bootstrap candidate is training-only
distillation, not inference leakage:

```text
actual sampled future board
    -> stopped-gradient BT4 policy head
    -> legal teacher distribution for sampled horizon h
    -> KL target for DFM slot h
```

The sampled future BT4 tokens already exist for the K=1 JEPA target, so this
requires a policy-head call but no additional encoder call. The teacher must
not be added as a future residual at inference because the actual future board
is unavailable. If pursued, preregister a warm-start coefficient and decay it
to zero so the stronger LC0-MCTS action labels remain the terminal objective.

For the JEPA dynamics themselves, separately consider a sampled
teacher-forced adjacent-state auxiliary:

```text
T(z_target[h-1], action[h]) -> z_target[h]
```

followed by a transition to the existing free recurrent rollout. This directly
bootstraps one-step compositional dynamics but requires one additional
previous-state target encoding for most sampled horizons, so it needs a
systems gate before scientific training.

Do not combine policy distillation and teacher-forced JEPA in one first
experiment. The pass curves and the one-epoch horizon audit determine which
failure is actually limiting before choosing either intervention.
