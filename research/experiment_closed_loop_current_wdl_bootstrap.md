# Experiment: current-WDL bootstrap for predicted-JEPA closed loop

Status: preregistered on 2026-07-29; execution in progress.

## Motivation

The completed post-Hero architecture suite found two complementary effects:

- current-state WDL retained the strongest one-pass policy (`51.17%` direct
  score against Hero-1024, statistically unresolved); and
- predicted-JEPA closed loop was the only arm improved by recurrent
  inference (`55.08%` for eight passes against its own one-pass policy).

The closed-loop arm's eight-pass policy recovered its weak one-pass start and
scored `50.98%` against Hero-1024, but did not establish an improvement. This
experiment asks whether current-state outcome supervision can preserve a
strong one-pass base while predicted-state feedback makes later passes useful.

The all-horizon policy-head arm is not reused. Although H1 has the same
architecture and update-zero weights as the root Hero policy head, H2--H8 are
copies of a *next-move* head attached to the same trainable current BT4 trunk.
Their seven later-horizon losses therefore backpropagate through the shared
trunk from update zero. At 1,024 updates, frozen H1 DFM CE degraded from
`3.1440` to `4.0184`, root legal-conditional CE from `2.1196` to `2.8152`,
and legal mass from `55.42%` to `51.85%`. Architectural equality did not make
the horizon tasks or their gradients equal.

## Candidate

Enable both accepted mechanisms in one fresh Hero initialization:

```text
wdl_include_current_state = true
dfm_closed_loop_mode = predicted_jepa_tokens
```

All other model, objective, optimizer, data, and systems settings remain the
same as the completed 1,024-update suite. In particular:

- raw BT4 plus seed-0 fresh JEPA/DFM initialization; no checkpoint
  continuation;
- canonical LC0 action codec, horizon eight, and batch `1,024`;
- `1,024` updates / `1,048,576` examples from data cursor zero;
- target and prediction SIGReg coefficients `2/2`, estimated on the same
  fixed 64 examples per update;
- shared WDL coefficient `0.25`, averaged uniformly over the current state
  and eight predicted states;
- example-based Hero learning-rate schedule and selective weight decay;
- SDPA-all, fused BT4 LayerNorm backward, compiled fresh regions, and the
  existing CPU/RAM/GPU guard; and
- one terminal model-only checkpoint, no intermediate checkpoints.

The combination changes neither the physical batch size nor the fixed
SIGReg estimator size, so no SIGReg coefficient adjustment is made.

## Gates and evaluation

1. Verify the combined configuration and objective on CPU.
2. Run one guarded real-data batch-1,024 smoke. Require finite losses and
   gradients, nonzero current-WDL and closed-loop diagnostics, no guard fault,
   and no out-of-memory failure.
3. Train the matched 1,024-update candidate.
4. Run the frozen 8,192-example fast validation and record complete
   per-horizon policy, WDL, legality, and latent-health metrics.
5. On the frozen first 128 color-reversed Arena pairs, measure:
   - candidate one pass versus Hero-1024 one pass;
   - candidate eight passes versus its own one-pass policy; and
   - candidate eight passes versus Hero-1024 one pass.

Use greedy legal action selection, inference batch cap 16, and additional ply
cap 256. Report all W/D/L counts, direct score and interval, cap/fault counts,
and per-position latency. Retain the terminal checkpoint until analysis is
complete.

## Decision rule

This is a mechanism-combination test, not a promotion test. It succeeds
mechanistically if:

- its one-pass strength is statistically compatible with Hero/current-WDL;
- eight passes improve on its own one-pass policy; and
- offline policy/WDL/latent metrics show no new collapse.

Only a later larger Arena can establish a real Elo gain.
