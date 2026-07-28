# Experiment: post-Hero inference-pass and horizon-bootstrap audit

Status: completed on 2026-07-28. This is a measurement and design audit. It
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
    -> legal teacher distribution for its next action
    -> KL target for the aligned future DFM slot
```

The sampled future BT4 tokens already exist for the K=1 JEPA target, so this
requires a policy-head call but no additional encoder call. The teacher must
not be added as a future residual at inference because the actual future board
is unavailable. If pursued, preregister a warm-start coefficient and decay it
to zero so the stronger LC0-MCTS action labels remain the terminal objective.

The index alignment is exact: `future_planes[s]` is the board after action
`s + 1`, so for sampled indices `s = 0..6` its policy supervises DFM slot
`s + 1`, namely H2--H8. The sampled board after H8 has no next-action slot
inside the stored chunk and is ineligible for this auxiliary. Do not
accidentally teach slot H1 from the board after its action.

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

## Deterministic replay result

The candidate replay completed all `1,024` updates with zero optimizer skips
under the guarded `2 CPU / 7 GiB RSS` contract. Its terminal checkpoint is
bit-exact to the deleted original:

```text
eb20c698fea1eabd4ebf71c13a1c195204736183f7b173079bd34f2a6e9a995f
```

All `52` retained scientific scalars match exactly at every update
(`53,248/53,248` comparisons); timing and resource counters were deliberately
excluded. This is a restored deterministic replay, not an approximate
replicate. Keep its `713,388,280`-byte tensor while the follow-up decision
remains open.

## Pass-scaling result

Every run completed 128 pairs / 256 games with zero faults, zero cap draws,
complete action coverage, and normal termination:

| Passes | Matched 1,024 control | 1,024 JEPA broadcast | One-epoch Hero |
|---:|---:|---:|---:|
| 1 | `0.960938` | `0.986328` | `0.664062` |
| 2 | `0.937500` | `0.943359` | `0.707031` |
| 4 | `0.925781` | `0.958984` | `0.824219` |
| 8 | `0.908203` | `0.945312` | `0.904297` |
| 16 | `0.910156` | `0.933594` | `0.935547` |

The first 128 pairs of the existing 1,024-pair one-epoch eight-pass Arena were
reused only after its frozen contract and pair-score prefix matched.

The early control and broadcast candidate both remain best at one pass. Their
16-pass minus one-pass paired deltas are respectively `-0.050781` and
`-0.052734`, with descriptive 95% intervals
`[-0.079483,-0.022080]` and `[-0.075126,-0.030342]`. The broadcast therefore
does not solve early refinement exposure by update 1,024.

The one-epoch model has undergone the desired qualitative transition. Its
score rises at every measured count; 16 passes beats one by `+0.271484`, with
descriptive interval `[0.233855,0.309114]`. The finished artifact does not
inherit the early model's pass-degradation behavior.

Against raw BT4, the broadcast candidate exceeds the matched control at every
count by `+0.025391/+0.005859/+0.033203/+0.037109/+0.023438`. The paired
descriptive intervals exclude zero at passes 1, 4, and 8. This is favorable,
but it is a post-hoc comparison on a reused, ceiling-prone development pool.
The more direct candidate-versus-control Arena remains a near tie:
`0.509766`, W/D/L `66/129/61`, interval `[0.467119,0.552412]`. These results
can reflect style or non-transitivity and do not authorize promotion.

## Horizon and initialization result

At initialization, JEPA MSE is numerically equal to the identity baseline at
every horizon, while DFM H1 CE is `6.829365` and H2--H8 are the uniform
`7.53125`. The initialization asymmetry is therefore real.

After one epoch:

| Horizon | JEPA MSE | Identity MSE | MSE / identity | DFM CE |
|---:|---:|---:|---:|---:|
| H1 | `0.034062` | `0.462440` | `0.0737` | `1.959745` |
| H2 | `0.027633` | `0.499418` | `0.0553` | `3.860296` |
| H4 | `0.040566` | `1.284965` | `0.0316` | `4.910862` |
| H8 | `0.066352` | `2.447475` | `0.0271` | `5.611434` |

H8 raw MSE remains about twice H1, but the future target has moved about
`5.3x` farther from the current-state identity. Relative to that required
change, H8 prediction is better than H1. The remaining large horizon gap is
clearer in action CE than in JEPA dynamics, so future-policy distillation is
the better first bootstrap. Teacher-forced JEPA remains a separate second
candidate if a later causal audit points to recurrent dynamics.

## Policy-only isolation result

The direct bypass test uses the same terminal one-epoch Hero state on both
sides. The candidate selects the legal argmax of the checkpoint-updated BT4
policy logits `B(s)`. The opponent uses 16-pass DFM root logits
`B(s) + R_16(s, a_1..a_8)`. Everything else is matched: 1,024 color-reversed
`hero_development` opening pairs, deterministic greedy play, canonical codec,
batch cap 16, additional-ply cap 256, and seed 0.

The policy-only path won decisively:

| Candidate | Score | W/D/L | Pair-aware logistic Elo | Conservative 95% interval |
|---|---:|---:|---:|---:|
| Hero BT4 policy-only vs Hero DFM-16 | `0.726318` | `1077/821/150` | `+169.55` | `[+134.05,+208.69]` |

All 2,048 games terminated normally with zero faults, zero cap draws, and
complete legal-action coverage. Mean physical batch-call latency was
`26.17 ms` for policy-only and `64.46 ms` for DFM-16.

This resolves two superficially conflicting observations. Additional passes
are useful *inside the DFM path*: against raw BT4, the one-epoch Hero rose
from `0.664062` at one pass to `0.935547` at 16 passes. But the best current
DFM path is still substantially weaker than bypassing DFM and using its own
jointly trained BT4 policy path. Thus iterative refinement has learned useful
behavior relative to DFM-1, while its net root residual still subtracts chess
strength from `B(s)`.

Policy-only includes the jointly trained Hero BT4 trunk as well as its policy
head. This result therefore does not show that the DFM/JEPA training losses
failed to improve the trunk. The next clean decomposition is Hero policy-only
versus immutable raw BT4; a direct policy-only versus DFM-1 match would then
measure how much damage exists before iterative refinement recovers it.

Future DFM artifacts must report a direct same-checkpoint policy-only bypass
gate. Beating raw BT4 is insufficient if the DFM root is weaker than `B(s)`.
The immutable run is:

```text
artifacts/arena/hero-epoch-v1-policy-only-vs-dfm-p16-1024pairs-v1/state.json
sha256 6cd279d23591d49d723ac052d677eee26c66e506ae6701c9f8a0d5537a447783
```

## Revised decision

Do not describe the broadcast architecture as chess-rejected. It failed the
preregistered held-out DFM-CE, accuracy, legal-mass, and rank promotion gates,
so it was correctly **not promoted**. Its centered chess result is
inconclusive and its post-hoc raw-BT4 pass curve is favorable. Retain the
checkpoint and, before a longer training extension, require a disjoint
centered confirmation that can distinguish strength from development-pool
style effects.

The immutable audit record and portable technical report are:

- `research/analysis/posthero_pass_bootstrap_audit_20260728.json`
- `research/analysis/posthero_pass_bootstrap_20260728/report.html`
