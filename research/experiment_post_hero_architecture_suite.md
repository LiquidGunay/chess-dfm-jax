# Experiment: post-Hero passthrough and JEPA/DFM architecture suite

Status: completed on 2026-07-28. All five matched candidates, frozen
validations, the 15-edge primary round robin, five refinement diagnostics,
and one focused closed-loop follow-up are complete. All six terminal
checkpoints are retained.

## Goal

The retained 1,024-update Hero learned a much stronger BT4 policy path than
DFM path, and additional DFM refinement made it weaker. Test five isolated
ways to give DFM a better starting point or a more direct state/outcome
coupling. Keep every terminal model checkpoint until the Arena is complete.

The immutable external baseline is:

```text
research/runs/torch_autoresearch_sigreg64_u1024_v1/checkpoint
SHA-256 05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9
```

Every new arm starts from the same raw BT4 weights and seed-0 fresh Hero
initialization. No arm continues from the Hero checkpoint or another
candidate.

## Five isolated arms

### A. All-horizon policy-logit passthrough

Use eight independent native BT4 attention-policy heads:

```text
combined_logits[h] = BT4_policy_head_h(current_tokens) + DFM_residual[h]
```

H1 is the original head. H2--H8 are exact parameter clones at update zero,
then learn independently through the existing DFM action CE for their
horizon. There is no extra head-only loss, future-board input, or teacher.
The DFM output remains zero-initialized, so all eight horizons initially
materialize the pretrained BT4 next-move policy.

### B. Equal-scale current-JEPA fusion

Fuse the current JEPA state into every DFM board token:

```text
u = RMSNorm(DFM_board_token)
v = RMSNorm(project(current_JEPA_state))
fused = (u + v) / sqrt(2)
```

The projection is freshly initialized, not zero-gated. Optional projection
initialization uses an isolated name-derived RNG stream so parameters shared
with the control retain their exact Hero initialization.

### C. Current-state WDL supervision

Apply the existing shared WDL head to the current JEPA state as well as all
eight predicted future states. Stored labels describe the side to move after
each action, so the current-state target is H1 with win and loss swapped.
Average uniformly over the nine active states under the existing coefficient
`0.25`; do not add a second `0.25` term. This keeps the WDL contribution near
its existing scale while directly training outcome information in the
current state.

### D. Predicted-JEPA closed loop

On each noisy training example:

1. run the ordinary DFM planner;
2. preserve visible corruption tokens, fill masked H1 with legal argmax and
   masked H2--H8 with deterministic argmax;
3. recurrently predict all eight JEPA states from that target-safe proposal;
4. project those states into per-action DFM context tokens; and
5. rerun DFM, applying policy losses only to the second logits.

Proposal indices are stopped-gradient, while gradients from the final policy
may flow through the state-context projection, JEPA rollout, and first DFM
hidden states. At inference, pass `p + 1` consumes the rollout predicted from
pass `p`; no future board, played action, WDL label, or training target is
available.

### E. BT4 policy-prelogit feature passthrough

Use the pretrained policy head's exact feature precursor

```text
Mish(policy_dense1(current_tokens))  # [batch, 64, 1024]
```

as the input to the DFM state projector instead of the raw final BT4 trunk
tokens. This is before policy q/k, promotion handling, and 1,858-way action
logit materialization. The ordinary H1 materialized-policy residual remains
unchanged; this arm tests whether the policy-specialized representation is a
better DFM base without duplicating horizon heads.

## Matched training contract

Freeze all non-arm settings to the retained Hero recipe:

- canonical LC0 action codec and legal root masking;
- horizon eight; uniform DFM CE; root legal-conditional CE and legality
  coefficients unchanged;
- sampled positive JEPA loss, target/prediction SIGReg coefficients `2/2`,
  fixed SIGReg sample count `64`, and WDL coefficient `0.25`;
- raw BT4 plus freshly initialized JEPA/DFM, seed `0`, and global-permutation
  data cursor `0`;
- example-based Hero LR schedule, optimizer partition, selective decay,
  rematerialization, attention kernels, and compilation settings unchanged;
- exactly `1,048,576` training examples per arm; and
- one terminal model-only checkpoint, with no intermediate model state.

First run checkpoint-free guarded smokes at batch `1,024`. If every arm fits,
use `1,024` batches and `1,024` updates. If any arm does not fit, select the
largest common exact divisor in `512, 256`, train every new arm for the
corresponding number of updates, and add a root-only matched replay control at
that batch. The retained Hero-1024 remains in the Arena in either case.
Scheduling is by examples, and SIGReg remains a fixed 64-example estimator,
so neither SIGReg coefficient changes with physical batch size.

## Required gates and measurements

Before training:

- CPU tests cover config fail-closed behavior, all-horizon tensor semantics,
  policy-prelogit routing, equal-scale fusion, optional-module RNG isolation,
  WDL parity conversion, target-safe closed-loop proposals, inference
  feedback, and unchanged default behavior.
- Each arm completes a guarded real-data smoke with finite loss and gradients,
  no resource warning, recorded peak HBM, examples/s, and arm-specific
  nonzero diagnostics.

For each terminal checkpoint, run the same frozen fast validation and retain
the complete loss/action/legality/WDL/latent-health record. Record the final
and trailing-64-update training losses in the experiment ledger.

## Arena

Use frozen `hero_development` opening histories, the first 128 color-reversed
pairs, greedy legal action selection, inference batch cap 16, and additional
ply cap 256.

1. Primary: a one-pass all-vs-all round robin over the five candidates and
   retained Hero-1024. This is the confirmed fast/strength setting for the
   retained Hero and yields one connected relative-Elo fit.
2. Refinement diagnostic: for each candidate, play its eight-pass DFM policy
   against its own one-pass DFM policy on the same pairs. This directly tests
   whether the new representation makes iteration useful. The closed-loop
   arm is inactive on the first pass by construction and active from pass two
   onward.

Report direct paired scores, intervals, W/D/L, cap/fault counts, latency, the
connected Bradley--Terry fit for the primary round robin, and one-versus-eight
paired deltas. These are pool-relative Arena Elo estimates, not external
human or engine ratings. Keep all terminal checkpoints through analysis;
remove none without a later explicit decision.

## Execution

All five guarded smokes fit at the preregistered common batch size `1,024`,
so every arm trained for `1,024` updates and exactly `1,048,576` examples.
Approximate steady-state throughput and peak PyTorch allocation were:

| Arm | examples/s | peak allocated HBM |
| --- | ---: | ---: |
| all-horizon heads | 240 | 21.97 GB |
| normalized JEPA fusion | 256 | 18.48 GB |
| current-state WDL | 257 | 18.48 GB |
| predicted-JEPA closed loop | 247 | 23.05 GB |
| policy-prelogit | 256 | 18.48 GB |

One current-WDL launch failed before update one because a shared
TorchInductor/Triton cache artifact raised an internal AST error. A diagnostic
with the projector eager then OOMed, confirming that the compiled projector
is required at batch `1,024`. Rebuilding the identical bounded regions in an
isolated workspace-local cache restored the already-smoked behavior. The
successful run used no changed numerical setting. The guard held the failed
compiler process below 2.5 GB host RSS and no server-pressure event occurred.
The remaining large runs used isolated workspace-local compiler caches.

Every successful arm wrote only its terminal model-only checkpoint. Training
wall time ranged from 68.6 to 72.6 minutes. The complete hashes, loss streams,
and source paths are in
`research/analysis/posthero_architecture_suite_20260728.json`.

## Loss and frozen-validation results

The table reports the trailing 64 training updates and the identical frozen
8,192-example fast validation:

| Model | train DFM CE | train total | val DFM CE | val action acc. | val legal mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hero-1024 | 5.7025 | 7.4496 | 5.7063 | 7.243% | 55.42% |
| all-horizon heads | 6.0994 | 8.0428 | 6.0689 | 5.576% | 51.85% |
| normalized JEPA fusion | 5.7385 | 7.5146 | 5.7386 | 7.098% | 53.86% |
| current-state WDL | 5.7049 | 7.4729 | 5.7086 | 7.306% | 55.37% |
| predicted-JEPA closed loop | 5.7218 | 7.4712 | 5.7121 | 7.286% | 54.77% |
| policy-prelogit | 5.7259 | 7.5541 | 5.7579 | 7.002% | 51.59% |

Current-state WDL matches Hero's validation CE to within `0.0023` and has
slightly higher action accuracy. During the final 64 training updates, the
closed loop lowers second-call DFM CE by `0.1993` on average relative to its
own first proposal. All-horizon cloned heads are already clearly worse by the
offline metrics.

## One-pass Arena

All 15 primary edges completed on the same first 128 color-reversed opening
pairs, for 3,840 games total. There were no policy faults, incomplete legal
coverage positions, cap draws, or abnormal terminations. Direct candidate
scores against Hero-1024 were:

| Candidate | score vs Hero | points / games |
| --- | ---: | ---: |
| current-state WDL | 51.17% | 131.0 / 256 |
| normalized JEPA fusion | 49.80% | 127.5 / 256 |
| policy-prelogit | 49.02% | 125.5 / 256 |
| predicted-JEPA closed loop, one pass | 45.51% | 116.5 / 256 |
| all-horizon heads | 14.45% | 37.0 / 256 |

The connected Bradley--Terry fit, anchored at Hero `0`, is:

| Model | relative Elo | matched-opening bootstrap 95% |
| --- | ---: | ---: |
| policy-prelogit | +4.4 | [-13.7, +22.6] |
| current-state WDL | +3.2 | [-14.1, +19.1] |
| Hero-1024 | 0.0 | fixed anchor |
| normalized JEPA fusion | -3.5 | [-23.4, +15.5] |
| predicted-JEPA closed loop, one pass | -41.2 | [-57.2, -26.1] |
| all-horizon heads | -297.9 | [-320.5, -276.3] |

The first three new arms are statistically unresolved from Hero at this
sample size. Policy-prelogit's connected estimate is marginally positive
despite its worse validation CE, so small loss differences are not a reliable
ordering of chess strength. The all-horizon prior is decisively harmful at
this training budget.

## Refinement result

Eight-pass candidate scores against their own one-pass policy were:

| Candidate | 8-pass score vs 1-pass |
| --- | ---: |
| predicted-JEPA closed loop | 55.08% |
| all-horizon heads | 46.68% |
| normalized JEPA fusion | 40.23% |
| current-state WDL | 39.26% |
| policy-prelogit | 33.01% |

Predicted-JEPA closed loop is the only arm for which recurrent inference does
positive chess work. A focused follow-up put its 8-pass policy directly
against Hero-1024 at one pass: `130.5/256 = 50.98%`, direct logistic
`+6.8` relative Elo with a deliberately conservative Hoeffding interval
`[-77.9, +92.3]`. The closed-loop side averaged `9.08 ms` per evaluated
position in Arena versus `2.83 ms` for Hero one-pass, about `3.2x` slower.

This is a mechanism success rather than a promotion: closed-loop iteration
recovers its weaker one-pass base and reaches approximately Hero strength,
but it does not yet establish an Elo improvement. The next architecture
iteration should preserve this predicted-state feedback while bootstrapping
the one-pass policy from the current-WDL or policy-prelogit-strength base.

## Reproducible artifacts

- `research/analysis/posthero_architecture_suite_20260728.json`: validated
  source ledger, all direct matches, connected fit, checkpoint hashes, and
  latency.
- `research/analysis/posthero_architecture_suite_20260728.csv`: compact
  per-model comparison table.
- `research/analysis/posthero_architecture_training_curves_20260728.png`:
  matched training-loss curves.
- `research/analysis/posthero_architecture_arena_20260728.png`: connected Elo
  and refinement summary.
- `research/analysis/build_posthero_architecture_suite.py`: deterministic
  validator and artifact builder.
