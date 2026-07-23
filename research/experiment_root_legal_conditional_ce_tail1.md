# Experiment 024: root legal-conditional imitation

Status: preregistered on 2026-07-23. No implementation, calibration, or
accelerator measurement has started.

## Question and hypothesis

The accepted update 2,072 chooses moves with a legality mask, but its training
CE normalizes over all 1,858 legacy actions. Experiments 015--019 show that
more first-action supervision can improve held-out H1 CE, yet changing the
share of the ordinary full-vocabulary CE can weaken legal mass or regress
against raw BT4. Experiment 023 attempted a stronger legal-ranking target
from the frozen BT4 head, but that larger backward graph exceeded the fixed
host-memory guard during compilation.

This experiment uses only labels already present in each trajectory row. It
adds played-action CE at H1 after normalizing logits over the stored,
representable root legal set. The existing uniform eight-horizon
full-vocabulary CE and the existing `2.0 * (1 - legal_mass)` objective remain
bitwise unchanged. For fixed logits, the new auxiliary has exactly zero
gradient on every illegal action and a zero-sum gradient over legal logits.
It therefore trains the ranking used by legal-masked inference without
directly changing the legal-versus-illegal pressure.

The hypothesis is that this small, label-only auxiliary recovers a
repeat-stable H1 improvement while preserving uniform CE, legal mass, latent
health, and the raw-BT4 point-score anchor. It adds no model parameters,
encoder or planner call, action target, data dependency, RNG draw, or
inference work.

## Candidate contract

Add one default-off scalar and activate:

```python
"root_legal_conditional_ce_coeff": CALIBRATED_COEFFICIENT,
```

For each row, gather the root logits at `legal_idx[:, 0]`, mask padding using
`legal_count[:, 0]`, and compute:

```text
root_legal_conditional_ce =
    logsumexp(root_logits[stored_legal_actions])
    - root_logits[played_action]
```

Use FP32 logits for this reduction. A row is eligible only when the example
is valid, H1 is masked, root legal metadata is valid, the count is positive
and within the physical legal-index width, the played action occurs in an
unmasked legal slot, and all gathered/played logits used by the reduction are
finite. Normalize by eligible rows, not physical batch size. Report the raw
and weighted loss, eligible count/fraction, played-in-legal fraction, finite
fraction, mean legal count, and legal-conditional top-1 accuracy.

The active path must fail closed on an unsupported objective or invalid
coefficient. Data-contract violations are visible through coverage metrics;
the calibration and smoke gates require full eligibility. The default-zero
path must preserve the incumbent loss, auxiliaries, gradients, RNG,
serialization, compilation graph, and resume contract exactly.

Keep all of the following fixed:

- source step-265,000 model-only initialization with a fresh optimizer;
- balanced per-example K=1 future-target sampling;
- stopped future-target prefix with only the final BT4 block trainable;
- both projector blocks and all four DFM and recurrent JEPA blocks;
- uniform eight-horizon full-vocabulary DFM CE;
- first-legality coefficient `2.0`;
- RMS norm coefficient `0.0`, target SIGReg `5.76`, and `z_pred` SIGReg
  `1.0`;
- a fixed 64-example V-statistic SIGReg sample;
- train/eval batches `128/64`, main/BT4 peak rates `3e-5/1e-6`, the
  `400/800/0.1` cosine schedule, seed 0, and global data permutation; and
- deterministic eight-pass, legal-masked searchless inference.

Forced H1 masking, nonuniform training time, WDL, BT4 policy distillation,
closed-loop JEPA feedback, direct JEPA rollout, and RMS norm matching remain
off. The model, optimizer, and checkpoint-state ABIs must not change.

## One-shot coefficient calibration

Before any optimizer update, evaluate the recovered source initialization on
the exact validation pools with seeds 10,000 and 20,000, each 64 batches of
64 positions, with coefficient `1.0`. Let `C0` be the example-count-pooled
root legal-conditional CE. Freeze:

```text
coefficient = clip(0.25 / C0, 0.05, 1.0)
```

The `0.25` target makes the initial weighted auxiliary comparable to the
roughly quarter-scale JEPA/SIGReg auxiliaries while its gradient role remains
restricted to within-legal ranking. Both calibration runs perform zero
updates and write no checkpoint. Require eligibility, played-in-legal, and
finite coverage all equal to `1.0`; otherwise reject the experiment as a
data-contract failure. Commit the exact pooled statistic and coefficient
before the training smoke. Do not retune the coefficient from candidate
results.

Also evaluate accepted update 2,072 read-only on the same two pools to record
its new conditional-CE and legal-conditional-accuracy controls. This
measurement describes the incumbent and does not alter the coefficient or
acceptance thresholds.

## Guarded staged gates

Every accelerator process must run through `research/run_gpu.sh`, hold the
exclusive host-visible lock, use at most two CPUs, start with at least 8 GiB
host `MemAvailable`, stop below 3 GiB, stop above 7 GiB process-group RSS,
keep at least 30 GiB disk free, and write only below `/mountpoint/.exp`.

1. Focused CPU tests must prove the exact sparse legal reduction, padding and
   eligibility behavior, zero illegal-logit gradient, zero-sum legal-logit
   gradient, coefficient-zero parity, active gradient routing, unchanged
   model/checkpoint ABI and inference, enabled-only serialization/resume
   semantics, calibration arithmetic, and fail-closed validation.
2. Run the two source no-update calibration pools and the read-only incumbent
   control sequentially. Write compact JSON/JSONL evidence only.
3. A real-checkpoint batch-128 one-update smoke must compile and finish
   finite and unclipped with full auxiliary eligibility, fixed
   `0.0/5.76/1.0` latent-loss coefficients, uniform CE, legality coefficient
   `2.0`, balanced 16-per-horizon targets, SIGReg counts `576/512`, future
   routing `14/1`, and zero checkpoint writes.
4. A checkpoint-free cached 30-update profile must reach at least `145.0`
   end-to-end examples/s and stay below `13,100,000,000` peak JAX HBM bytes.
   Record compiler work, utilization, power, data stalls, host peak RSS, and
   minimum `MemAvailable`.

A guard failure, nonfinite result, incomplete eligibility, state-ABI change,
or systems-gate failure rejects the experiment immediately. Do not rescue it
with a smaller batch, relaxed guard, coefficient change, or alternate legal
loss.

## Fixed-time decision

Only after all systems gates pass, train for 1,800 steady-state seconds. Write
at most two temporary checkpoints: update 1,600 and terminal. Evaluate both on
the unchanged seed-10,000 and seed-20,000 pools, each 64 batches of 64
positions. Select minimum pooled H1 CE, breaking an exact tie with lower root
legal-conditional CE.

The frozen accepted controls are primary/repeat H1 CE
`2.8470487520/2.8538390882`, uniform CE
`4.4950091206/4.4971840288`, accuracy
`0.1104583740/0.1103057861`, and legal mass
`0.6488690404/0.6468926072`. Primary acceptance requires:

- H1 CE below `2.8402584158`;
- root legal-conditional CE below the read-only update-2,072 control;
- uniform DFM CE no greater than `4.4971840288`;
- aggregate accuracy at least `0.1080921631`;
- legal mass at least `0.6467235410`;
- every metric and gradient finite;
- positive JEPA prediction beating zero and action-shuffled predictions at
  every horizon;
- mean/minimum prediction effective rank at least `30.0/28.5`;
- mean/minimum prediction feature-std p05 at least `0.63/0.61`;
- mean target RMS at least `0.90`; and
- prediction/target RMS ratio in `[0.94, 1.02]`.

A passing primary requires one exact repeat. Its selected state must beat
accepted-primary H1 CE `2.8470487520`, beat the incumbent conditional CE, and
pass every remaining gate. Only then run the frozen 128-pair cap-256 arenas
against update 2,072 and original raw BT4. An Elo-aligned offline incumbent
requires point score above `50%` against update 2,072 and above
`36.5234375%` against raw BT4. These development screens are not an absolute
Elo promotion.

On any failure, run no later stage, delete all candidate states after
preserving compact evidence, restore coefficient zero, and retain update
2,072. On success, retain only the primary selected state. Never overlap GPU
workloads, and do no SAE or pass-count work during this experiment.
