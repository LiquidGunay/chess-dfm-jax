# Experiment: refinement-pass compute/strength sweep

Status: preregistered on 2026-07-28. This is a no-training diagnostic on the
retained feedback-off hero/count-64 checkpoint.

Execution note: the first guarded one-pass launch stopped safely before
model materialization or gameplay because the Torch Arena loader retained an
obsolete `refinement_passes == horizon` check. The underlying refiner already
supports every positive count. Replace that loader check with a
default-neutral guard that permits arbitrary positive counts only when
feedback is off and continues to require exactly eight passes for
`final_pass_adjoint`; pass loader-level CPU tests before retrying.

## Question

How much searchless chess strength does iterative DFM refinement gain from
additional passes, and is eight passes near the latency/strength knee?

The pass-count study was deliberately deferred until a stronger model
existed. That condition is now met: the retained 1,024-update checkpoint
scores `0.908203` against raw BT4 over the frozen first 128 color-reversed
opening pairs at eight passes, with zero faults and zero cap draws.

This experiment changes test-time compute only. It does not train a model,
change a loss, select on validation data, or alter the eight-pass strength
anchor used by model-side autoresearch.

## Frozen model and Arena

Use only:

```text
run:
  research/runs/torch_autoresearch_sigreg64_u1024_v1

checkpoint:
  research/runs/torch_autoresearch_sigreg64_u1024_v1/checkpoint/model.safetensors

SHA-256:
  05068c96b2bac8f10a3f3b853363bdb2ce49f935b026566705b3bdcd4d9060c9
```

The model has SIGReg sample count `64`, target/prediction SIGReg coefficients
`2.0/2.0`, WDL coefficient `0.25`, and
`jepa_feedback_mode = "none"`. No other checkpoint may enter this sweep.

Run refinement counts:

```text
1, 2, 4, 8, 16
```

For every count, freeze:

- raw BT4 as the opponent;
- `hero_development` opening tier;
- first 128 color-reversed opening pairs and seed `0`;
- policy batch-size cap `16`;
- additional-ply cap `256`;
- deterministic greedy actions;
- canonical LC0 action codec and root legality masking;
- no diagnostic-mode execution inside timed policy calls; and
- no JEPA feedback at inference.

Reuse the existing eight-pass Arena only after verifying its checkpoint,
opening-pool, history, seed, batching, cap, codec, and model-inference
contracts. Run the other four counts sequentially through
`research/run_gpu.sh`; never overlap them.

The refiner's schedule remains the production implementation. For pass index
`i` of `P`, it targets `floor(8 * (i + 1) / P)` revealed action slots. This
means fewer than eight passes reveal multiple slots per call, while more than
eight passes provide extra iterative revisions; no separate schedule is
tuned for a count.

## CPU and execution gates

Before GPU work, a focused test must exercise all five pass counts, prove
that planner-call count equals the requested count, preserve strict root
legality, and finish with all eight action positions revealed. Existing
checkpoint descriptor and Arena contract tests must still pass.

Every Arena must finish with:

- exactly 128 pairs / 256 games;
- zero policy faults and zero cap draws;
- all terminations normal;
- the requested pass count recorded in the immutable run contract; and
- finite mean physical-call timing.

No model checkpoint is written.

## Analysis and decision

Report for every pass count:

- score, W/D/L, pentanomial counts, and descriptive logistic Elo versus raw
  BT4;
- mean physical policy-call time and speedup relative to eight passes;
- per-opening paired score delta versus eight passes, with better/equal/worse
  counts and a descriptive paired-t 95% interval; and
- the score-versus-latency Pareto frontier.

This 128-pair reused development pool is descriptive and cannot promote a new
inference setting. Keep eight passes as the strength anchor unless a
predeclared confirmation is triggered:

- if 16 passes beats eight by at least `0.01` score, or its paired interval
  excludes zero on the positive side, confirm on additional frozen pairs
  before considering a higher-compute strength setting;
- if one or four passes is within `0.01` score of eight while reducing mean
  policy-call time by at least `20%`, confirm on additional frozen pairs
  before considering it as a fast screening tier; and
- otherwise retain eight passes and treat the curve as the answer.

Do not tune a count between these five values from the same openings. Keep
all artifacts below `/mountpoint/.exp`, publish one compact JSON record and
one score/latency plot, and store no additional model state.
