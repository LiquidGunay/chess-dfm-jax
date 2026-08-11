# Experiment: proposal-derived JEPA feedback before the final DFM pass

Status: completed and rejected on 2026-07-28. Keep
`jepa_feedback_mode = "none"` active; the rejected candidate tensor has been
removed after its hash and compact evidence were sealed.

## Question

Can the learned state predictor improve searchless action choice when its
predicted state change directly conditions the final DFM refinement, rather
than acting only through shared training gradients?

The accepted hero/count-64 model is coupled during training:

```text
current state -> DFM action hidden -> JEPA predicted future state
```

but `z_pred` is not consumed by DFM inference. This experiment closes that
loop without adding a ninth planner pass, a parameter, a teacher action, or a
new loss coefficient.

## Frozen mechanism

Add the default-off model mode:

```text
jepa_feedback_mode = "final_pass_adjoint"
```

Given current JEPA state `z0`, the model's legal root proposal `a_hat`, and
its proposing DFM root hidden state `h`, compute:

```text
condition = jepa_action_embed(a_hat) + jepa_hidden_adapter(h)
z1        = jepa_transition(z0, condition)
delta_z   = z1 - z0

raw_feedback =
    delta_z @ transpose(jepa_hidden_adapter.weight)
    * sqrt(token_dim / z_dim)
```

The adapter bias has no adjoint. Cap each example's feedback RMS at `0.5`
times the RMS of its current 64 DFM state tokens, multiply it by a
stopped-gradient proposal gate in `[0,1]`, and broadcast the resulting
width-256 residual over those state tokens.

The current hero transition does not apply an explicit JEPA-state RMSNorm;
the feedback step uses the exact same recurrent transition semantics as its
eight-horizon positive loss. Do not introduce a normalization change in this
experiment.

## Target-safety and training graph

On the ordinary noisy action sequence and sampled time:

1. run the existing DFM planner and retain preliminary logits/root hidden;
2. if the root token is revealed, use that token; otherwise take the
   legal-masked root argmax from the preliminary logits;
3. use stopped-gradient legal-softmax confidence for a masked proposal,
   one for a revealed proposal, and zero when training legality metadata is
   invalid;
4. run one proposal-conditioned recurrent JEPA step and the fixed adjoint
   bridge; and
5. rerun the DFM planner on the feedback-conditioned state tokens.

Only the final logits receive DFM CE, root ranking, and legality losses.
Preliminary CE is a detached diagnostic. Proposal indices and confidence
gates are detached, but final policy gradients may flow through the second
planner, adjoint bridge, JEPA transition, preliminary root hidden state,
current state projector, and current BT4 encoder.

Changing the played target while holding model state, noisy input, logits,
and legality metadata fixed must not change the proposal or feedback. The
existing clean played-action JEPA rollout, sampled future target, WDL
auxiliary, both SIGReg terms, and every coefficient remain unchanged.

## Inference

Run exactly eight DFM planner calls. Passes 1--7 use the ordinary current DFM
state. After pass 7's scheduled token update, form a legal proposal from the
resulting root token or pass-7 logits, compute one JEPA transition and the
adjoint feedback, and use the feedback-conditioned state for pass 8 only.

Root legality masking remains active on every pass. Active feedback requires
exactly eight passes and the current JEPA state; incompatible calls fail
closed. Record policy-call time because this mode adds the current
state-projector and one JEPA transition to inference even though the planner
call count is unchanged.

## Matched training comparison

Reuse `torch_autoresearch_sigreg64_u1024_v1` as the feedback-off control.
Train one candidate from the same raw BT4 weights and identically seeded
fresh modules:

| Setting | Control | Candidate |
|---|---:|---:|
| JEPA feedback | `none` | `final_pass_adjoint` |
| WDL coefficient | `0.25` | `0.25` |
| SIGReg examples | `64` | `64` |
| Target / prediction SIGReg | `2.0 / 2.0` | `2.0 / 2.0` |
| Batch / updates | `1,024 / 1,024` | `1,024 / 1,024` |
| Examples | `1,048,576` | `1,048,576` |
| Seed / data start | `0 / 0` | `0 / 0` |

Every other architecture, loss, optimizer, example-based schedule, data
order, canonical codec, stochastic choice, and systems setting is fixed.
The mode must preserve the exact parameter/state ABI.

Before the candidate run, pass focused tests for default-off parity,
proposal legality, revealed-token preservation, invalid-metadata gating,
target independence, adjoint/cap mathematics, nonzero intended gradients,
exactly eight inference planner calls with feedback only on pass 8,
descriptor/config compatibility, and fail-closed invalid modes. Then run a
guarded checkpoint-free batch-1,024 smoke/profile. It must finish finite,
apply nonzero feedback, keep peak allocated HBM below the physical A10G
limit, and sustain at least 200 examples/s on its warm steps. Failure ends
this candidate rather than silently changing batch, rematerialization, or
the scientific graph.

Write every loss/feedback/system scalar and exactly one terminal model-only
checkpoint.

### Systems-gate result

The guarded checkpoint-free batch-1,024 smoke completed 20/20 finite updates
with zero skips and no resource warning. Updates 2--20 averaged
`251.329` examples/s (`4.07435` seconds/update), versus `256.002`
examples/s for the selected feedback-off runtime. Peak allocated HBM was
`21,781,476,864` bytes and guard peak process-group RSS was
`2,528,571,392` bytes. No checkpoint was written.

The stopped proposal gate averaged `0.61591`; applied feedback RMS was
nonzero and reached `0.03870` by update 20. Its preliminary-minus-final CE
effect remained around `-1e-5` during this very early warmup because the DFM
residual head begins at zero. The smoke gate requires a real, finite bridge,
not an early strength benefit; the frozen 1,024-update usefulness gate below
must resolve whether feedback eventually improves the logits.

The compact smoke report is
`research/runs/torch_autoresearch_feedback_smoke_b1024_v1/report.json`,
SHA-256
`8e11fded40a7cab730cc8d1741104ac885a0ca7a88c8d8e399a0141332d3d4e3`.

## Frozen evaluation and decision

Evaluate the candidate on the same 8,192-example fast pool with feedback
active. Report final and detached-preliminary DFM/root metrics, all ordinary
action/legality metrics, JEPA controls, norms/ranks/feature tails, WDL
diagnostics, throughput, HBM, and non-finite skips.

Then run the identical first 128 color-reversed opening pairs against raw BT4
with eight passes, policy batch cap 16, and additional-ply cap 256. Compare
the candidate's per-opening pair scores directly with the retained
feedback-off control.

Advance only if:

1. paired Arena score is directionally non-worse than the control, with zero
   policy faults and zero cap draws;
2. active-feedback frozen DFM CE is no more than `0.005` worse, root legal CE
   no more than `0.01` worse, accuracy loses less than `0.1` percentage point,
   and legal mass loses less than `0.2` percentage point;
3. final frozen DFM CE is lower than the candidate's own detached preliminary
   CE, proving that feedback helps its logits rather than merely being
   present;
4. horizon-8 prediction and target effective/stable ranks retain at least
   95% of control and feature tails do not collapse;
5. training sustains at least 200 examples/s and mean Arena policy-call time
   is no more than 1.5 times the control; and
6. every numerical and execution gate passes.

A positive screen requires a fresh repeat and at least 10% of an epoch before
promotion. A negative result closes only this parameter-free final-pass
adjoint design; it does not reject other ways of coupling state prediction to
action choice.

Retain only the model needed for the active recipe after hashes and compact
evidence are sealed. Never overlap training, validation, Arena, or SAE work
on the single A10G.

## Result

The candidate completed all 1,024 updates / 1,048,576 examples with zero
non-finite skips in 4,196.70 seconds. End-to-end throughput was `249.857`
examples/s, mean sampled GPU utilization was `98.729%`, and peak allocated
HBM was `21,781,476,864` bytes. Exactly one terminal model-only checkpoint
was written:

```text
a206e3b74c1ace48dfe97edae03d66f31fee916816eb8822436bb5c0070c7f1f
```

On the identical 8,192-example frozen fast pool:

| Metric | Feedback off | Final-pass feedback | Candidate - control |
|---|---:|---:|---:|
| DFM CE | 5.706299 | 5.712420 | +0.006121 |
| Accuracy | 0.072433 | 0.073868 | +0.001434 |
| Legal mass | 0.554204 | 0.547545 | -0.006660 |
| Root legal CE | 2.119581 | 2.103285 | -0.016296 |
| JEPA positive MSE | 0.133190 | 0.137335 | +0.004145 |
| WDL CE | 0.771990 | 0.824971 | +0.052980 |

Feedback was real and useful to the candidate's own logits: detached
preliminary DFM CE was `5.712801`, final DFM CE was `5.712420`, and the
improvement was `0.000382`. Mean applied feedback RMS was `0.087471` versus
state RMS `1.250025`; the cap activated on `1.404%` of examples. This was not
enough to offset the candidate's worse learned model. It missed the
preregistered DFM-CE and legal-mass guardrails.

Horizon-8 feature tails remained healthy, but both representations lost
rank relative to the control:

| Latent | Effective-rank retention | Stable-rank retention | p05 feature-std retention |
|---|---:|---:|---:|
| Prediction | 93.52% | 93.14% | 103.10% |
| Target | 94.02% | 92.44% | 101.68% |

The identical 128 color-reversed opening pairs produced:

| Metric | Feedback off | Final-pass feedback |
|---|---:|---:|
| Score versus raw BT4 | 0.908203 | 0.916016 |
| Points / games | 232.5 / 256 | 234.5 / 256 |
| Mean physical policy call | 44.586 ms | 49.106 ms |
| Faults / cap draws | 0 / 0 | 0 / 0 |

The paired score delta was `+0.0078125`, with
better/equal/worse opening-pair counts `33/67/28`, a descriptive paired-t
95% interval `[-0.025985, 0.041610]`, and two-sided `p = 0.648`. The Arena
point estimate passes the directional and timing gates, but is too noisy to
rescue the failed validation and rank gates.

During closeout, the Arena execution path and recorded model descriptor
confirmed that feedback was active, but a legacy run-level
`jepa_used_at_inference` field was hardcoded false. The metadata was fixed,
15 focused Arena tests passed, and the deterministic Arena rerun reproduced
the exact score and pentanomial counts. The final contracts record false for
the control and true for the candidate.

Reject this parameter-free final-pass adjoint mechanism. Do not repeat it or
extend it to 10% of an epoch. This result does not reject learned feedback,
multi-pass state/action coupling, or other ways of consuming predicted state.
Keep the accepted count-64, WDL-0.25, feedback-off checkpoint as the active
baseline. The rejected tensor was deleted after sealing its expected hash and
size; all loss rows, validation metrics, Arena blocks, manifests, and compact
evidence remain in:

- `research/analysis/hero_closed_loop_feedback_20260728.json`
- `research/analysis/hero_closed_loop_feedback_20260728.png`
