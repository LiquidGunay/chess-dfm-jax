# Experiment: terminal Hero policy/DFM round robin

Status: completed on 2026-07-28. This is a no-training inference audit of the
terminal one-epoch Hero checkpoint.

## Question

The same terminal Hero checkpoint is much stronger when its jointly trained
BT4 policy path is used directly than when its 16-pass DFM root output is
used. Does substantially more iterative compute repair that gap, and how do
DFM-128, DFM-16, Hero policy-only, and immutable raw BT4 rank on one matched
Arena?

## Frozen contract

All six edges of the four-player round robin use:

- terminal Hero state SHA-256
  `665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692`;
- immutable raw-BT4 SHA-256
  `61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651`;
- the first 128 color-reversed `hero_development` opening pairs;
- 256 games per edge / 1,536 games total;
- seed 0, physical batch 16, and additional-ply cap 256;
- the canonical 1,858-action codec and strict root legality mask; and
- one guarded GPU workload at a time, with no checkpoint writes.

The Arena is fully deterministic. Policy-only and raw BT4 mask illegal root
moves and take `argmax`. DFM takes per-pass `argmax` predictions, uses stable
confidence-ordered progressive unmasking over its eight action slots, masks
the root logits for legality, and returns the final H1 action. There is no
temperature, categorical sampling, Gumbel noise, exploration noise, or
stochastic tie breaking.

At 128 passes, the production unmask schedule is unchanged:

```text
target_unmasked = floor(8 * (pass_index + 1) / 128)
```

Thus each newly revealed slot receives about 16 refinement evaluations. This
is deliberately a test-time compute intervention, not a new decoding
schedule.

## Direct results

Every game terminated normally. There were zero policy faults, cap draws, or
incomplete legal-action coverage.

| Candidate | Opponent | Score | W/D/L | Direct logistic Elo |
|---|---|---:|---:|---:|
| DFM-128 | DFM-16 | `0.501953` | `49/159/48` | `+1.36` |
| DFM-128 | policy-only | `0.277344` | `21/100/135` | `-166.37` |
| DFM-128 | raw BT4 | `0.927734` | `219/37/0` | `+443.40` |
| DFM-16 | policy-only | `0.255859` | `12/107/137` | `-185.46` |
| DFM-16 | raw BT4 | `0.935547` | `223/33/0` | `+464.73` |
| policy-only | raw BT4 | `0.994141` | `253/3/0` | `+891.84` |

The direct Elo values are transforms of each edge's observed score. The
policy/raw edge is almost saturated, so its direct number is especially
unstable as an estimate of an additive common Elo scale.

Fit one Bradley-Terry model to all six edges, anchor raw BT4 at zero, and
resample the same opening-pair index synchronously across every edge for
5,000 bootstrap replicates:

| Player | Joint relative Elo vs raw BT4 | Matched-opening bootstrap 95% |
|---|---:|---:|
| raw BT4 | `0.0` | fixed anchor |
| DFM-16 | `+474.92` | `[+433.24,+521.45]` |
| DFM-128 | `+479.00` | `[+439.53,+524.08]` |
| policy-only | `+659.96` | `[+618.90,+706.39]` |

The key joint contrasts are:

- DFM-128 minus DFM-16: `+4.08` Elo,
  95% `[-20.03,+27.39]`;
- policy-only minus DFM-128: `+180.95` Elo,
  95% `[+150.16,+212.67]`; and
- policy-only minus DFM-16: `+185.03` Elo,
  95% `[+158.65,+213.19]`.

The joint model's maximum edge-score residual is only `0.01652`, so the
four-player ordering is reasonably represented by one additive local scale.
These are pool-relative Arena differences, not human, Stockfish, or globally
calibrated engine Elo.

## Compute result

On the direct DFM-128/DFM-16 edge, mean physical batch-call latency was:

| Policy | Mean call |
|---|---:|
| Hero policy-only | about `27 ms` on its direct edges |
| DFM-16 | `64.73 ms` |
| DFM-128 | `331.15 ms` |

DFM-128 is about `5.12x` slower than DFM-16 and `12.2x` slower than
policy-only per physical call. It produces no measurable strength gain over
DFM-16.

## Interpretation

The current DFM inference path is doing negative work **relative to its own
jointly trained base policy path**. This does not mean the DFM/JEPA training
objectives were globally harmful: policy-only includes the shared BT4 trunk
after that joint training and may have benefited from those auxiliary
gradients. It means the learned DFM root residual and decoding path take a
much stronger `B(s)` decision and make it about 181--185 local Elo worse.

Additional passes are not uniformly harmful. Earlier, DFM improved markedly
from one to sixteen passes against raw BT4. This round robin shows that the
improvement has plateaued by sixteen: 128 passes are statistically tied with
sixteen and remain far below bypassing DFM. The problem is therefore not
simply insufficient test-time iteration.

## Fresh-pretrain architecture implication

The earlier 10%-of-DFM-RMS cap was a retrofit guard for injecting a new JEPA
branch into a trained DFM. It should not be the default for a fresh joint
pretrain. For a fresh model, give the DFM and JEPA branches comparable
opportunity through a variance-preserving fusion such as:

```text
u = RMSNorm(z_dfm)
v = RMSNorm(P(z_jepa))
z_fused = (u + v) / sqrt(2)
```

or concatenate the two normalized branches and use a variance-preserving
learned projection. Log both branch RMS values, their cosine, and gradient
norms. Do not impose an arbitrary `0.1` multiplier unless a measured
instability later motivates it.

The Hero already has the relevant fresh JEPA WDL head: it applies one shared
head to every free-rollout `z_pred[h]` and optimizes per-horizon categorical
WDL CE with coefficient `0.25`. The WDL-off ablation was strongly negative.
Keep this head. When DFM consumes the current JEPA state, these WDL gradients
can become inference-relevant through JEPA-to-DFM fusion; no pretrained BT4
value head is required for the first test.

## Next passthrough experiment

The next isolated model experiment should remove the H1/H2--H8 base-logit
asymmetry without adding a second loss:

```text
combined_logits[h] = B_h(current_bt4_tokens) + R_h(dfm_inputs)
```

- `B_1` is the native pretrained BT4 policy head.
- `B_2..B_8` are independent exact clones of `B_1` at update zero.
- The shared DFM output projection is zero-initialized, so every horizon
  initially passes through its cloned BT4 head.
- The existing single DFM CE on ground-truth `a_h` is applied only to
  `combined_logits[h]`. Do **not** add a separate head-only CE; both the
  cloned base head and DFM residual learn through the same supervised
  objective.
- No future-board encoding, teacher policy, or target action is consumed at
  inference. Every `B_h` sees only the current board representation.
- Log detached base-head-only CE/accuracy by horizon to measure division of
  labor, but do not optimize those diagnostics separately.

One BT4 attention-policy head has `3,152,896` BF16 parameters. Seven full
clones add `22,070,272` parameters (about 44 MB of model weights before
optimizer state) and nontrivial compute, but they are evaluated once per
board and reused across all DFM refinement passes. Profile the vectorized
eight-head implementation before training. A shared-logit broadcast is the
cheap fallback, but it cannot learn horizon-specific current-state priors and
is a weaker test of the proposed bootstrap.

For scientific attribution, compare this all-horizon passthrough against the
current H1-only passthrough under the same fresh initialization, data order,
loss, JEPA/WDL recipe, update budget, and inference count. Test normalized
JEPA-to-DFM fusion as the next orthogonal change or include it in both arms;
do not change it in only the passthrough candidate.

## Evidence

The validated scalar record is:

```text
research/analysis/hero_policy_round_robin_20260728.json
```

Each source Arena state and its SHA-256 are sealed inside that record. The
states live under:

```text
artifacts/arena/hero-epoch-v1-round-robin-first128-*/
```
