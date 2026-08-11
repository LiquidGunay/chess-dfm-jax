# Experiment: refinement-pass compute/strength sweep

Status: completed and accepted on 2026-07-28. This is a no-training
inference result on the retained feedback-off hero/count-64 checkpoint.

Execution note: the first guarded one-pass launch stopped safely before
model materialization or gameplay because the Torch Arena loader retained an
obsolete `refinement_passes == horizon` check. The underlying refiner already
supports every positive count. Replace that loader check with a
default-neutral guard that permits arbitrary positive counts only when
feedback is off and continues to require exactly eight passes for
`final_pass_adjoint`; pass loader-level CPU tests before retrying.

The retry then stopped safely after model restoration but before policy
adapter construction or gameplay because the native Torch adapter duplicated
the same obsolete invariant. Centralize the feedback-aware pass validation in
the Torch refiner/adapter, test adapter construction for all five counts and
feedback-mode rejection away from eight, and rerun only after the complete
CPU suite passes.

## Question

How much searchless chess strength does iterative DFM refinement gain from
additional passes, and is eight passes near the latency/strength knee?

The pass-count study was deliberately deferred until a stronger model
existed. That condition is now met: the retained 1,024-update checkpoint
scores `0.908203` against raw BT4 over the frozen first 128 color-reversed
opening pairs at eight passes, with zero faults and zero cap draws.

This experiment changes test-time compute only. It does not train a model,
change a loss, or select on validation data. The development sweep alone did
not authorize an inference-anchor change; the separately frozen confirmation
below did.

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

## Development-screen result and confirmation

All five 128-pair runs completed with zero faults, zero cap draws, and normal
termination for all 256 games:

| Passes | Score vs raw BT4 | W/D/L | Mean physical call |
|---:|---:|---:|---:|
| 1 | 0.960938 | 236/20/0 | 28.955 ms |
| 2 | 0.937500 | 224/32/0 | 31.059 ms |
| 4 | 0.925781 | 218/38/0 | 36.415 ms |
| 8 | 0.908203 | 210/45/1 | 44.586 ms |
| 16 | 0.910156 | 210/46/0 | 65.198 ms |

Relative to eight passes, the one-pass paired score delta is `+0.052734`,
with better/equal/worse opening-pair counts `40/74/14`, descriptive paired-t
95% interval `[0.024942, 0.080526]`, and two-sided `p = 0.000263`. One pass
also reduces mean physical-call time by `35.06%`. This exceeds the
predeclared fast-setting trigger and is positive rather than merely within
the `0.01` score tolerance.

Freeze one confirmation before changing any inference anchor:

1. Run only one and eight passes over the first 256 frozen opening pairs in
   new output directories. The first 128 must reproduce the existing pair
   scores exactly; the primary confirmation slice is the newly added pair
   indices 128--255.
2. Keep every model, opponent, seed, codec, history, batching, and cap field
   identical to the development screen.
3. Confirm one pass only if the new 128-pair slice has positive paired score
   delta, the combined 256-pair paired-t 95% interval has a strictly positive
   lower bound, one-pass mean policy-call time is at most `0.8` times the
   eight-pass time, and both runs have zero faults and zero cap draws.
4. If any condition fails, retain eight passes as the model-search anchor and
   treat the first result as development-pool selection. If all pass,
   designate one pass as the confirmed fast/strength setting for this
   checkpoint while retaining eight-pass results for continuity with earlier
   experiments. Do not change training horizon or DFM loss horizon.

## Confirmation outcome

The new 256-pair one- and eight-pass runs completed with zero faults, zero
cap draws, and normal termination for every game. Their first 128 pair-score
arrays reproduce the development runs exactly.

| Passes | Score vs raw BT4 | Points | W/D/L | Mean physical call |
|---:|---:|---:|---:|---:|
| 1 | 0.958984 | 491.0/512 | 470/42/0 | 28.902 ms |
| 8 | 0.911133 | 466.5/512 | 422/89/1 | 45.152 ms |

On the independently added pair indices 128--255, one pass beats eight by
`+0.042969` score, with better/equal/worse pair counts `32/83/13`,
descriptive paired-t 95% interval `[0.015473, 0.070465]`, and two-sided
`p = 0.002442`. Across all 256 pairs, the delta is `+0.047852`, with counts
`72/157/27`, interval `[0.028427, 0.067276]`, and
`p = 0.000002137`.

The one/eight latency ratio is `0.640117`, a `35.99%` reduction. Every
predeclared confirmation gate passes. One pass is therefore the confirmed
fast/strength inference setting for this exact checkpoint, and the
model-side search budget moves to one pass. Eight-pass results remain a
continuity diagnostic; the DFM training horizon remains eight actions.

This does not establish a universal property of DFM architectures. With the
current feedback-off implementation, one planner call exposes all eight
action slots, while later calls condition on the model's own intermediate
trajectory. The monotonic strength decline from one through eight passes is
consistent with refinement exposure mismatch or error propagation, but this
experiment does not identify the cause. Architectures that change the
refinement mechanism must report their own pass sensitivity before claiming
the same optimum.

The immutable scalar record and Pareto plot are
`research/analysis/hero_refinement_pass_sweep_20260728.json` and `.png`. No
model checkpoint was written.
