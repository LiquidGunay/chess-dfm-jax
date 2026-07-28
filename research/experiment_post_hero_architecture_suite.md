# Experiment: post-Hero passthrough and JEPA/DFM architecture suite

Status: preregistered on 2026-07-28; implementation and CPU tests are in
progress. No GPU candidate has started.

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
