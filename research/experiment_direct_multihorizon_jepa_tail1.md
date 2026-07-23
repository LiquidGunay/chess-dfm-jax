# Experiment 021: direct parallel multi-horizon JEPA

Status: preregistered on 2026-07-23; implementation and measurements have not
started.

## Question and hypothesis

The accepted one-block-tail model predicts all eight future JEPA states by
recurrently applying one four-layer vector transition. This preserves a
compositional one-step dynamics interpretation, but it serializes the largest
remaining JEPA computation across the horizon. The original research plan
therefore placed recurrent versus direct multi-horizon prediction before
closed-loop latent-conditioned refinement.

This experiment replaces only that recurrent execution with a direct
multi-output predictor using the same stored parameters. For each horizon
`h`, the clean DFM planner supplies its horizon-specific action hidden state
after jointly encoding the complete played eight-action sequence. The JEPA
condition remains

```text
c_h = jepa_action_embed(a_h) + jepa_hidden_adapter(dfm_hidden_h)
```

but every future prediction starts from the same current state:

```text
z_pred[h] = normalize(jepa_transition(z_0, c_h))
```

The eight `(z_0, c_h)` pairs are evaluated as one parallel tensor rather than
as a horizon scan whose carry is the preceding prediction. This is a direct
sequence predictor: because DFM attention is bidirectional, `c_h` may encode
the complete proposed action sequence. It must not be described as a
composable one-step dynamics model or silently used for recurrent planning.

The systems hypothesis is that a larger parallel transition improves A10G
utilization and fixed-time examples without changing parameter count,
checkpoint state, target-encoder work, DFM inference, or the frozen loss. The
learning hypothesis is that removing recurrent error propagation and giving
each horizon a direct gradient to `z_0` preserves or improves held-out action
quality and latent health. Direct multi-step prediction can reduce compounding
error under model misspecification, but fully observed deterministic chess
does not guarantee that it will beat a recurrent one-step model; this is a
bounded empirical ablation, not a general claim
([Somalwar et al., 2025](https://arxiv.org/abs/2504.01766)).

The incumbent already beats action-shuffled prediction by roughly sixfold.
Therefore this experiment does not reactivate the legacy shuffled-action
margin loss: that would target an absent failure mode and add a second JEPA
rollout per update.

## Candidate contract

Add a default-recurrent string setting and activate exactly:

```python
"jepa_rollout_mode": "direct_sequence",
```

Candidate semantics:

- `recurrent` preserves the existing free rollout exactly.
- `direct_sequence` broadcasts normalized `z_0` over all eight horizons,
  forms the existing per-horizon action/DFM conditions, applies the existing
  four-layer `jepa_transition` once to the `[batch,horizon,z_dim]` tensor, and
  normalizes every output.
- Direct mode uses the same clean full-sequence DFM hidden states as the
  recurrent control. It introduces no new horizon embedding, pooling rule,
  parameter, optimizer leaf, random draw, target, label, or loss coefficient.
- Training loss, full-horizon validation, collapse diagnostics, and
  action-shuffled diagnostics must all dispatch to the configured rollout.
  A diagnostic must never evaluate a recurrent model under the direct graph
  or vice versa.
- Sampled target anchors and teacher-forced recurrent carries are incompatible
  with direct mode and must fail closed rather than silently changing
  semantics.
- The configured model/optimizer state ABI and every restored array remain
  exact. The resume contract records rollout mode and rejects a mismatched
  continuation.
- Current searchless inference does not execute JEPA. Cached BT4 encoding,
  four-block DFM refinement, legality masking, eight passes, codecs, and arena
  behavior remain unchanged.

Keep balanced per-example K=1 future-target sampling, the accepted one-block
future-gradient tail, both projector blocks, all four DFM and JEPA blocks,
uniform eight-horizon DFM CE, source first-legality coefficient `2.0`,
horizon eight, train/eval batch `128/64`, peak main/BT4 learning rates
`3e-5/1e-6`, the `400/800/0.1` cosine schedule, global data permutation, and
eight-pass inference.

The loss is fixed: RMS norm matching `0.0`, target SIGReg `5.76`, and
`z_pred` SIGReg `1.0`, with the normalized V-statistic and 64 sampled
examples. Expected effective target/prediction counts remain `576/512`.
Initialize from recovered step 265,000 model-only with a fresh optimizer and
seed 0. Do not initialize from the accepted update 2,072.

## Staged gates

Before fixed-time training:

1. Focused CPU tests must prove exact default-recurrent parity; direct output
   equivalence to independently applying the same transition at every
   horizon; identical horizon-one mathematics for an identical condition;
   finite nonzero gradients through the transition, JEPA action embedding,
   DFM hidden adapter, clean DFM path, and current BT4 path; unchanged target
   assignments, loss coefficients, RNG streams, state ABI, and inference;
   mode-aware collapse diagnostics and report/resume serialization; and
   fail-closed invalid or incompatible modes.
2. A guarded one-update real-checkpoint A10G smoke at batch 128 must compile
   and finish finite and unclipped, report direct mode, eight predictions,
   balanced 16-per-horizon positive targets, SIGReg counts `576/512`, future
   routing `14/1`, and no checkpoint write.
3. A guarded checkpoint-free cached 30-update profile must reach at least
   `145.0` end-to-end examples/s, remain below `20,000,000,000` peak JAX HBM
   bytes, and satisfy every host RAM/CPU/disk guard. Record compiler
   FLOPs/bytes, end-to-end and device throughput, utilization, power, and data
   stalls. The speed hypothesis succeeds only at or above `161.213`
   examples/s, a 5% gain over the matched recurrent profile
   `153.5362`; the lower `145.0` threshold is only the continuation floor.

Any smoke/profile OOM, guard abort, non-finite value, checkpoint write, or
systems-floor failure ends the experiment before fixed-time training.

## Fixed-time decision

Train for 1,800 steady-state seconds after compilation. The guarded launcher
may write only update 1,200 and terminal, with `max_checkpoints=2`. Evaluate
both states on the unchanged seed-10,000 and seed-20,000 pools, each 64
batches of 64 examples. Among states passing every latent gate and the uniform
CE ceiling below, select lower pooled horizon-1 CE; break an exact tie with
lower uniform CE. If neither state qualifies, select the lower uniform CE
only for diagnosis and reject.

Primary acceptance requires:

- pooled horizon-1 CE no greater than `2.8538390882`;
- uniform DFM CE no greater than `4.4971840288`;
- aggregate action accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94,1.02]`.

A passing primary requires one exact repeat satisfying the same systems,
policy, and latent gates. Only then run frozen 128-pair cap-256 arenas against
accepted update 2,072 and original raw BT4. An Elo-aligned direct-model
incumbent requires point score above `50%` against update 2,072 and above
`36.5234375%` against raw BT4. These development screens cannot authorize an
absolute-Elo or promotion claim.

Failure at any earlier gate gets no later repeat or arena. Retain a candidate
state only after all repeated offline and arena point gates; otherwise delete
both state files after preserving compact hashes, reports, telemetry, and
decisions. Restore recurrent mode on rejection. Keep every mutable file,
cache, compiler artifact, checkpoint, and temporary file under
`/mountpoint/.exp`, and never overlap GPU workloads.

## Outcome

Pending.
