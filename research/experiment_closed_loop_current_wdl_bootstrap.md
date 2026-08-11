# Experiment: current-WDL bootstrap for predicted-JEPA closed loop

Status: completed on 2026-07-29; rejected as a refinement architecture.

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

## Results

The guarded batch-1,024 smoke passed, followed by all 1,024 registered
updates. Training consumed 4,268.43 seconds (71.14 minutes) at 245.66
examples/second end to end. Peak allocated HBM was 23,065,796,096 bytes;
there were no guard faults, skipped updates, or intermediate checkpoints.
The retained terminal state has SHA-256
`845a4a6a08eb71f16a5e493e25be99fba9443b45c3f88818898bc7d9e85ef08e`.

On the frozen 8,192-example validation pool, the combined candidate had:

- DFM CE `5.71280`, action top-1 `7.240%`, and legal mass `54.829%`;
- H1 CE `3.13721` and root legal-conditional CE `2.10079`;
- current-WDL loss `0.79190` and accuracy `62.891%`;
- closed-loop second-call CE improvement `0.04804`;
- target/prediction SIGReg `0.06642/0.06660`; and
- prediction effective rank rising from `6.50` at H1 to `10.32` at H8,
  with prediction/target RMS ratios from `1.008` to `1.020`.

The latent and offline gates passed, but offline policy quality was effectively
tied with the isolated closed-loop arm (`5.71280` versus `5.71207` DFM CE)
and slightly below Hero/current-WDL.

All Arena games used the registered frozen first 128 color-reversed opening
pairs, greedy legal selection, inference batch cap 16, and additional ply cap
256. There were no faults, cap draws, or incomplete policy coverage.

| Candidate | Reference | Score | W/D/L | Direct Elo (95% Hoeffding) | Candidate latency |
| --- | --- | ---: | ---: | ---: | ---: |
| combined p1 | Hero p1 | `47.85%` | 53/139/64 | `-14.9` (`-101.1`, `+69.4`) | 2.95 ms/position |
| combined p8 | combined p1 | `49.02%` | 49/153/54 | `-6.8` (`-92.3`, `+77.9`) | 8.77 ms/position |
| combined p8 | Hero p1 | `46.29%` | 50/137/69 | `-25.8` (`-113.0`, `+58.2`) | 8.65 ms/position |

Current-WDL improved the combined arm's one-pass point estimate by 2.34
percentage points relative to isolated closed-loop versus Hero, although the
matched-opening bootstrap interval includes zero. It did not preserve
iterative improvement: combined p8 versus its own p1 was 6.05 points below
the isolated closed-loop arm's corresponding score, with a 50,000-sample
matched-opening bootstrap interval of `[-10.94, -1.17]` points.

The candidate therefore fails the registered mechanism test. The one-step
closed-loop training objective does not establish that feedback can be
composed seven times at inference, and adding current-state WDL interfered
with the previously observed iterative gain. The checkpoint is retained for
diagnostics rather than promoted.

## H1 isolation diagnostic

The all-horizon arm's original H1 BT4 policy was also evaluated with DFM
bypassed against Hero's original H1 BT4 policy. It scored `24.80%`
(22/83/151) over the same 256 games, corresponding to direct Elo `-192.7`
with a 95% Hoeffding interval of `[-333.3, -93.9]`; there were no faults or
cap draws.

This rules out the idea that only the all-horizon DFM combination was poor.
The H1 module had the same architecture and exact initialization as Hero, but
did not remain the same learned function. H2--H8 used different future-action
targets, and all seven losses flowed through the shared, trainable current
BT4 trunk from update zero. Their aggregate optimization pressure damaged
the representation used by H1. The direct all-heads DFM score (`14.45%`
against Hero) was worse still, but the policy-only isolation already shows
that shared-trunk interference is sufficient to explain a large part of the
failure.

The sealed machine-readable analysis is in
`research/analysis/closed_loop_current_wdl_bootstrap_20260729.json`, with a
compact comparison table in the adjacent CSV.
