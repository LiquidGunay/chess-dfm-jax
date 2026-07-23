# Experiment 022: proposal-derived JEPA feedback before the final DFM pass

Status: preregistered on 2026-07-23. No GPU smoke, profile, fixed-time run,
checkpoint evaluation, repeat, or arena has been run.

## Question and hypothesis

The accepted one-block-tail model is coupled only as an auxiliary-objective
system:

```text
current state -> DFM action hidden -> JEPA predicted future latent
```

Its predicted latent never changes an action logit at inference. Experiment
021 tested the preceding recurrent-versus-direct item in the original plan and
ended safely at the host-memory guard. The final original architecture item is
therefore a genuinely closed-loop refinement:

```text
DFM proposal -> JEPA predicted next latent -> later DFM refinement
```

A teacher-forced loop would be invalid because embedding the played move in
the feedback path would expose the answer to the policy loss whenever that
move is masked. This experiment instead uses only the model's own legal-masked
root proposal. At inference, DFM pass 7 of 8 supplies that proposal and its
root hidden state. One recurrent JEPA step predicts the proposal-conditioned
next latent, and an adjoint of the already learned
`jepa_hidden_adapter` maps the predicted latent change back to DFM token
width. The mapped global residual conditions pass 8 only.

The learning hypothesis is that a consequence-aware final refinement improves
played-action CE and chess strength because the last policy decision can use
the model's predicted state change. The systems hypothesis is deliberately
bounded: the implementation adds no ninth DFM pass, no parameter, and no
optimizer state, but it does add the two-block JEPA state projector and one
four-layer vector transition to inference. It is acceptable only if the
measured latency and training throughput remain useful.

The incumbent action-shuffled/positive JEPA MSE ratio is already about six, so
this experiment does not add an action-contrastive loss. Predicted-state WDL
was rejected at repeat in Experiment 020 and remains disabled.

## Frozen feedback mathematics

Add the default-off mode:

```python
"jepa_feedback_mode": "final_pass_adjoint",
```

For a current normalized JEPA latent `z0`, proposed root action `a_hat`, and
the proposing DFM root hidden state `h`, compute exactly:

```text
c       = jepa_action_embed(a_hat) + jepa_hidden_adapter(h)
z1      = normalize(jepa_transition(z0, c))
delta_z = z1 - z0

raw_feedback =
    delta_z @ transpose(jepa_hidden_adapter.w)
    * sqrt(token_dim / jepa_condition_dim)
```

The square-root factor is the variance-preserving correction for applying the
adjoint of the `token_dim -> condition_dim` matrix. The adapter bias has no
adjoint and is not used. This mode requires
`jepa_condition_dim == z_dim` and recurrent JEPA rollout; incompatible
configurations fail closed.

For each example, clip `raw_feedback` to at most `0.5` times the RMS of its
current 64 DFM state tokens. Reuse no trainable gate and expose no scale
hyperparameter. If the current root token is already unmasked, the feedback
gate is one. If it is masked, multiply by the stopped-gradient legal proposal
confidence. Missing/invalid training legality metadata sets the gate to zero;
production inference already rejects an empty root legal mask.

Broadcast the resulting width-`token_dim` residual over all 64 DFM state
tokens. Do not change action tokens, positional embeddings, time embeddings,
or the planner transformer.

## Training graph and target-safety contract

The active training/evaluation graph performs:

1. the existing noisy DFM planner call, now returning action hidden states;
2. a sparse-root-legality-masked proposal from its logits, using the existing
   noisy root token if that token was already revealed;
3. one recurrent JEPA step from current `z0` using that proposal and noisy
   root hidden state;
4. the fixed adjoint feedback described above; and
5. one feedback-conditioned DFM planner call on the same noisy action tokens
   and time.

The existing single DFM CE and legality terms consume the final
feedback-conditioned logits. There is no second CE coefficient, averaged
loss, teacher action, future target, WDL target, extra random draw, or
distillation target. The preliminary CE is detached and reported only as a
diagnostic. Gradient from the final CE may flow through the second planner,
the adjoint bridge, JEPA transition, action/hidden condition, preliminary DFM
hidden state, current projector, and current BT4 encoder. Proposal indices and
proposal-confidence gates are stopped-gradient.

The positive JEPA rollout remains the same clean, played-action, recurrent
eight-step objective. Balanced per-example K=1 future targets, one trainable
future BT4 tail block, target SIGReg `5.76`, prediction SIGReg `1.0`, RMS norm
loss `0.0`, and all existing collapse diagnostics remain fixed.

Focused tests must prove that changing the played target while holding noisy
inputs, logits, legality metadata, and model state fixed cannot change a
masked proposal or its feedback. This is the explicit no-target-leak gate.

## Inference contract

Searchless inference remains exactly eight DFM planner transformer calls.
Passes 1--7 use the ordinary current-state DFM latents. Pass 7 additionally
returns its root hidden state. After its scheduled action-token update:

- use its resulting legal root token as `a_hat`; if the root slot is still
  masked, use that pass's legal-masked root argmax;
- compute current `z0` from the already cached BT4 tokens, without a second
  BT4 encode;
- execute exactly one recurrent JEPA transition and the frozen adjoint
  feedback; and
- use the feedback-conditioned DFM state tokens for pass 8 only.

Both diagnostic and lean action-only kernels must choose identical final
actions. Root legality masking remains active on every pass. Active feedback
requires exactly eight refinement passes; other counts fail closed rather
than silently changing the trained computation. The latent-only public
inference entry point fails closed in this mode because it does not own the
required current JEPA latent.

## Fixed controls

Keep:

- recovered step 265,000 model-only initialization and fresh optimizer;
- seed 0 and global data permutation;
- horizon 8 and exactly eight inference passes;
- train/eval batch `128/64`;
- main/BT4 peak learning rates `3e-5/1e-6`;
- cosine schedule `400/800/0.1`;
- one-block future-target BT4 gradient tail;
- balanced one-of-eight future target per physical example;
- both projector blocks, all four DFM blocks, and all four JEPA transition
  layers;
- uniform eight-horizon DFM CE and source first-legality coefficient `2.0`;
- normalized V-statistic SIGReg on a fixed 64 examples; and
- loss coefficients norm/target/prediction `0.0/5.76/1.0`.

The accepted offline incumbent is
`future-tail1-balanced-k1-cosine-b128-30m-v1/update 2,072`. Its primary/repeat
two-pool uniform CE is `4.4950091/4.4971840`, primary horizon-one CE is no
greater than the frozen `2.8538390882` ceiling, and legal mass is
`0.6488690/0.6468926`. Its direct raw-BT4 point score is `36.5234375%`.
Matched recurrent training profile throughput is `153.5362` examples/s.
The source step-265,000 eight-pass inference profile is `25.86/26.96 ms`
batch-one p50/p95 and `62.56/64.67 ms` batch-64 p50/p95; arena policy-call
time for the incumbent is about `44.14 ms` at physical batch 16.

## Staged gates

Before any fixed-time training:

1. Focused CPU tests must prove exact default-off training and inference
   parity; legal proposal selection; revealed-token preservation; zero
   feedback for invalid legality metadata; the no-target-leak property; the
   exact adjoint and RMS-cap mathematics; finite nonzero feedback gradients
   through every intended path; exactly eight inference planner calls with
   feedback only before call eight; diagnostic/action-only agreement;
   unchanged model and optimizer state ABI; explicit report/resume semantics;
   and fail-closed invalid modes, pass counts, and latent-only inference.
2. A guarded real-checkpoint batch-128 one-update smoke must compile and finish
   finite and unclipped, report the active feedback mode, one preliminary and
   one feedback-conditioned noisy planner call, eight positive JEPA
   predictions, balanced 16-per-horizon targets, SIGReg counts `576/512`,
   future routing `14/1`, nonzero finite feedback/gate diagnostics, no state
   write, and peak JAX HBM below `20,000,000,000` bytes.
3. The same launcher must enforce two host CPUs, at least 8 GiB
   `MemAvailable` at launch, a 7 GiB process-group RSS ceiling, a 3 GiB
   runtime-available-memory floor, at least 30 GiB free disk, and zero
   overlapping GPU work. Any guard abort ends this experiment; do not rescue
   it with a smaller batch, altered graph, or looser guard.
4. A guarded checkpoint-free cached 30-update profile must reach at least
   `110.0` end-to-end examples/s and stay below `20,000,000,000` peak JAX HBM
   bytes. Record compiler FLOPs/bytes, device/end-to-end throughput,
   utilization, power, data stalls, and feedback diagnostics.
5. Before training, benchmark the initialized candidate and matched
   default-off graph at batch 1 and batch 64 with eight passes. Candidate
   batch-one p50 must be at most `60 ms`, batch-64 p50 at most `120 ms`, and
   candidate peak inference HBM below `4 GiB`. These are viability ceilings,
   not speed-improvement claims.

Failure at any earlier gate gets no later profile, fixed run, checkpoint,
repeat, or arena.

## Fixed-time and strength decision

If every pre-run gate passes, train for 1,800 steady-state seconds after
compilation. Disable periodic saves and write at most update 1,200 plus the
terminal state. Evaluate both on the unchanged seed-10,000 and seed-20,000
pools, each 64 batches of 64 examples. Select lower pooled horizon-one CE
among states passing all gates; break an exact tie with lower uniform CE.

Primary acceptance requires:

- pooled horizon-one CE no greater than `2.8538390882`;
- uniform DFM CE no greater than `4.4971840288`;
- aggregate action accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- feedback-conditioned horizon-one CE below the detached preliminary
  horizon-one CE on the same batches;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94,1.02]`.

A passing primary requires one exact repeat satisfying the same systems,
policy, latent, and feedback-usefulness gates. Only then run the frozen
128-pair cap-256 arenas against update 2,072 and raw BT4. An Elo-aligned
closed-loop incumbent requires point score above `50%` against update 2,072
and above `36.5234375%` against raw BT4. These screens cannot authorize an
absolute-Elo or formal promotion claim.

Retain a state only after all repeated offline and arena point gates.
Otherwise delete all candidate states after preserving compact reports,
hashes, telemetry, and the decision. Keep every mutable file, cache,
compiler artifact, temporary file, and checkpoint under `/mountpoint/.exp`,
and never overlap GPU workloads.

## Outcome

Pending.
