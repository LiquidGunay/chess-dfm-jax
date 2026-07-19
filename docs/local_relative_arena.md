# Local relative-strength arena

`research/evaluate_arena.py` is the checked, in-process evaluator for the
localized JointLatentSASA/DFM policy and the original raw BT4 policy head. It
reports model-pool-relative strength only. Its logistic and normalized Elo
values are not human, Lichess, or absolute Elo.

The evaluator does not use Stockfish, an opening book, or a puzzle dataset. It
uses the pinned ply-12 validation/test pools and exact standard-game history
sidecars under `artifacts/arena/`. The sidecars are required for repetition and
draw-claim state; the model still receives current-only planes to match its
training preprocessing.

## What is evaluated

- deterministic greedy DFM action refinement;
- a fixed, recorded refinement-pass count;
- one newly encoded BT4 state per played position, with BT4 latents cached
  across refinement passes;
- one run-frozen physical inference batch shape, including shrinking and final
  partial gameplay batches;
- root legality in the recovered `legacy_absolute_1858` codec; and
- normal `python-chess` outcomes or a symmetric additional-ply-cap draw.

With `--opponent-raw-bt4`, the opponent instead uses the immutable original
BF16 BT4 policy head and the complete board-aware
`lc0_canonical_1858` codec. Both models still receive current-only classical
planes under the frozen preprocessing contract. The evaluator records a
separate codec capability for each side; logits and masks are never remapped
between codecs.

JEPA prediction is a training auxiliary and is not executed or fed back into
actions in this baseline. Illegal actions, timeouts, exceptions, nonfinite
outputs, and positions with no representable action fail closed as losses.

The legacy codec cannot represent white knight promotions or any black
promotion. Every result records this capability contract, fault counts, total
legal/representable actions, and positions with incomplete coverage.

## Static inference batches

JAX compiles a different executable when the batch dimension changes. Arena
populations shrink as games terminate, so passing each live population
directly to the model can turn normal moves into compilation-length timeouts.
The evaluator freezes the physical inference shape for the whole run at:

```text
min(block_pairs, policy_batch_size_cap)
```

For the default 16-pair correctness/development block and a cap of 64, both
models warm and run only physical batch 16.

The local policy first validates full histories, encodes boards, and builds
legacy legal masks for real rows only. A short batch then repeats the first
already-validated encoded plane and its legal mask to the frozen shape. No
fabricated all-legal or empty mask is used, and no additional active-row mask
is needed: the model has no cross-batch operation, and every output and
diagnostic is sliced back to the ordered real-row prefix before semantic
validation. A bad padded action cannot become a move or fault; a bad real
action still fails closed.

Inference is deterministic greedy computation and contains no runtime random
draw, dropout, or batch-stat update. Padding therefore neither consumes nor
reorders RNG state. The construction seed and explicit no-runtime-RNG contract
remain recorded for provenance. A future stochastic policy requires stateless
per-game keys before it can use this batching protocol.

Coverage and `positions_evaluated` count real rows only. Call timing includes
the actual padded device work. A timeout or exception still loses only the
real games in the affected policy call; padded rows are never decoded,
adjudicated, or used as fallback moves.

## History validation trust boundary

`LocalDFMPolicy.select_actions()` remains the public and default boundary. It
fully replays every supplied history from standard chess and rejects a missing,
truncated, or corrupt transition.

Arena gameplay uses a separate capability after two independent checks:

1. `load_opening_history_sidecar()` verifies the pinned file and manifest,
   aligns every entry to the pool, and exactly replays every opening.
2. `play_arena_pairs()` exactly replays the selected openings into
   authoritative stackful boards. Before each policy call it checks normal
   termination with draw claims, the ply cap, standard-start/root/current
   history invariants, move-stack and position counts, and the played suffix.
   It advances those boards only through uniquely decoded legal actions.

Only then does the runner mint an immutable process-local sealed attestation.
The model receives a stackless current-board copy and that constant-size
packet—not the growing history. The adapter verifies the seal, schema,
standard-start marker, exact current FEN, ply/count/cap arithmetic, and the
runner's stackful nonterminal attestation before encoding. A plain endpoint
tuple, forged packet, mismatched FEN, or policy without the exact capability
fails closed.

The stackless copy is intentional: repetition claims cannot be re-established
from a FEN. The authoritative stackful runner remains the sole repetition and
adjudication authority. This makes per-call adapter history checking O(1)
without weakening the external full-replay boundary.

## Tiers

- `correctness`: first 16 of the 128 frozen validation openings; cap 16 by
  default. This checks plumbing and is not a strength result.
- `development`: up to all 128 validation openings. There is deliberately no
  default ply cap: pass one explicitly and interpret the reported cap-draw
  rate.
- `promotion`: the disjoint, history-sidecar-backed 2,048-opening test pool. The
  official normalized-Elo GSPRT uses `H0=0`, `H1=+20`,
  `alpha=beta=0.05`, and at most 2,048 complete color-reversed pairs.

Promotion enforces one-pair atomic blocks so the GSPRT can stop at the exact
completed-pair boundary. `accept_h1` is the only promoted outcome;
`accept_h0`, `max_pairs`, and an exhausted shorter requested slice are not
promotion.

Exact replay found that the old v2 promotion entry 1,245 was claim-draw
terminal by threefold repetition, although its root FEN alone appeared
nonterminal. Commit `5a49df9` regenerates the pool from the source shards
instead of skipping that position at runtime; commit `12147e8` repins and
reopens the evaluator on the v3 assets. The loader verifies the pinned pool
and history digests and exactly replays all 2,048 histories before model
loading. This repairs asset validity; it does not by itself authorize a
promotion run or an Elo claim.

The evaluator correctness path has been exercised on the A10G, but a
strength-valid long-game cap has not been frozen. Calibrate it with a pilot and
report cap rate before treating development or promotion output as chess
strength.

## Current empirical status

Corrected no-norm target-SIGReg-5.76/prediction-SIGReg-1.0 v2/update 400 has
completed both 128-pair development anchors at cap 256:

- versus recovered step 265,000: score `0.49609375`, descriptive logistic Elo
  `-2.714`, pair-aware 95% interval `[-87.959,+82.198]`, three cap draws, and
  one source-side legacy-codec fault;
- versus original raw BT4: score `0.380859375`, descriptive logistic Elo
  `-84.410`, pair-aware 95% interval `[-180.965,+0.625]`, no cap draws, no
  raw-BT4 fault, and two candidate-side legacy-codec promotion faults.

The two raw-anchor faults occurred only after the game reached a black pawn
promotion position with no non-promotion legal move. Treating both as draws
would change the point estimate only to about `-81.5` Elo; immutable scoring
retains the preregistered fail-closed losses. These screens establish the
initial strength anchors, not promotion or absolute strength.

The state files are
`artifacts/arena/corrected-nonorm-target5p76-pred1-v2-u400-vs-source-development-128pairs-cap256-v1/state.json`
(SHA-256
`5ba9efa4593806d538a038be86669e9a439b7d3ae8380d78a8ef2b4c9b183e92`)
and
`artifacts/arena/corrected-nonorm-target5p76-pred1-v2-u400-vs-raw-bt4-development-128pairs-cap256-v1/state.json`
(SHA-256
`0bb0ef97628364da76970d7247f66d58d9cfef25d94fdd58b648f10b0334fc13`).

V2/update 300 versus source completed the 16-pair correctness tier with zero
faults, a representable fraction of `1.0` for both models, and no incomplete
coverage position. All 32 games reached the 16-ply additional cap, so the run
checks wiring and color reversal but contains no strength information. Its
state is
`artifacts/arena/baseline-b128-v2-u300-vs-source-correctness-v1/state.json`
(SHA-256
`57904c1131b99db41c22204d9bfa44c348bc238a4e344670072a40c59d9d1308`).

The first 16-pair development pilot with an additional-ply cap of 64 is
invalid: 29 of 32 games ended in timeout faults while shrinking live
populations triggered compilation of new JAX batch shapes. The retained
fail-closed state is
`artifacts/arena/baseline-b128-v2-u300-vs-source-development-pilot16-cap64-v1/state.json`
(SHA-256
`f06ecfa2cc6a7882d59826abf69701ff460f9a76de2ac66c794152d5b98de9a1`).
Do not interpret its score or Elo diagnostics.

Commit `72af918` introduced the static inference-batch contract above. The
pre-fix pilot uses arena schema v1 and cannot resume into the fixed path.
Post-static-batch pilots eliminated the recompilation timeout faults. The
strict full-history cap-256 parity target completed all 32 games by normal
chess termination with zero faults and no cap draws: 3,913 real positions in
592.28 gameplay seconds (`6.61` positions/second). The sealed endpoint path
must reproduce those games and aggregate statistics under arena schema v3
before its speedup is accepted.

## Commands

Use the lean diagnostics-off path for gameplay:

```bash
research/run_gpu.sh .venv/bin/python research/evaluate_arena.py \
  --candidate research/runs/EXPERIMENT/checkpoints/update00000500 \
  --opponent-source \
  --tier correctness \
  --output-dir artifacts/arena/EXPERIMENT-correctness
```

A development comparison requires an explicit cap:

```bash
research/run_gpu.sh .venv/bin/python research/evaluate_arena.py \
  --candidate research/runs/EXPERIMENT/checkpoints/update00000500 \
  --opponent-checkpoint research/runs/INCUMBENT/checkpoints/update00000500 \
  --tier development \
  --additional-ply-cap 128 \
  --output-dir artifacts/arena/EXPERIMENT-vs-INCUMBENT-development
```

Use the original BT4 policy anchor without supplying a DFM checkpoint:

```bash
research/run_gpu.sh .venv/bin/python research/evaluate_arena.py \
  --candidate research/runs/EXPERIMENT/checkpoints/update00000500 \
  --opponent-raw-bt4 \
  --tier development \
  --additional-ply-cap 256 \
  --output-dir artifacts/arena/EXPERIMENT-vs-raw-bt4-development
```

Resume the exact same contract after interruption:

```bash
research/run_gpu.sh .venv/bin/python research/evaluate_arena.py \
  --candidate research/runs/EXPERIMENT/checkpoints/update00000500 \
  --opponent-checkpoint research/runs/INCUMBENT/checkpoints/update00000500 \
  --tier development \
  --additional-ply-cap 128 \
  --output-dir artifacts/arena/EXPERIMENT-vs-INCUMBENT-development \
  --resume
```

Once the promotion pool has been regenerated and the tier reopened, promotion
is reserved for repeated candidates that already pass every offline gate:

```bash
research/run_gpu.sh .venv/bin/python research/evaluate_arena.py \
  --candidate research/runs/EXPERIMENT/checkpoints/update00000500 \
  --opponent-checkpoint research/runs/INCUMBENT/checkpoints/update00000500 \
  --tier promotion \
  --additional-ply-cap 128 \
  --output-dir artifacts/arena/EXPERIMENT-vs-INCUMBENT-promotion
```

The run writes `state.json` plus immutable per-block JSON files below
`blocks/`. Writes use temporary files in the destination directory, `fsync`,
and atomic replacement; no system `/tmp` path is used. Resume checks the
checkpoint, code, pool, history, configuration, state, and every block digest.
An orphaned, missing, modified, reordered, or cross-run block fails closed.
The run and block schemas are version 3, and the resume contract pins the
physical batch size, padding rule, real-row metric basis, output slicing, and
no-runtime-RNG behavior. It also pins the sealed endpoint schema, public
full-replay default, stackless policy copy, authoritative runner state, and
repetition authority. Earlier pilot state cannot resume into this path.

Full refinement traces remain available with `--collect-diagnostics`, but the
default action-only kernel avoids constructing and transferring entropy/top-k
trace arrays on every move. Compilation/warmup and model loading are recorded
separately from gameplay.

## Current performance caveat

The hot arena path no longer replays or copies complete histories at every
policy call. The authoritative runner still performs chess outcome and
draw-claim adjudication on its full move stack, and promotion uses exact
pair-boundary stopping rather than throughput-oriented multi-pair blocks.
Benchmark a long-cap pilot before budgeting a full 2,048-pair test.

Run the CPU support tests with:

```bash
source research/env.sh
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  tests/test_research_arena.py \
  tests/test_play_arena.py \
  tests/test_research_arena_cli.py \
  tests/test_research_inference.py \
  tests/test_research_local_policy.py
```
