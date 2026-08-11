# Discrete DFM refinement, JEPA feedback, and LC0 value data

Status: implementation contract, 2026-08-10.

This document separates three mechanisms that must not be conflated:

1. iterative discrete action completion inside DFM;
2. auxiliary state/value prediction that shapes training representations; and
3. online use of predicted states or values to change an action at inference.

The primary model remains searchless. It uses no Hero-teacher KL. Online
state feedback and value-based candidate selection are separately named
ablations rather than hidden changes to the main inference algorithm.

## Primary state-action-state architecture

For a current board s0:

    z0 = encoder(s0)
    a0:7 = DFM(z0)
    predicted_z[h + 1] = JEPA(predicted_z[h], a[h], action_hidden[h])
    predicted_value[h + 1] = value_head(predicted_z[h + 1])

The recurrent JEPA starts from z0 and consumes its own previous predicted
state. Future state embeddings are not teacher-forced. During ordinary
supervised training, however, the actions are the observed eight-ply game
trajectory.

This is a coherent amortized trajectory policy plus an auxiliary latent world
model. It does not require another DFM call after JEPA. Another DFM call is
required only for the distinct claim that an imagined consequence revises
the proposed action online.

## Exact discrete refinement used at inference

Let H be the eight action slots and P be the number of DFM passes. The decoder
starts with every slot masked. On pass k:

1. Set diffusion time to k divided by P.
2. Predict a categorical distribution for every action slot, conditioned on
   the current board and all actions already committed.
3. Mask illegal root logits.
4. Take the argmax and confidence of every still-masked slot.
5. Commit the highest-confidence slots until floor(H times (k + 1) divided by
   P) slots are unmasked.
6. Keep every committed token fixed.

For H equals P equals 8, exactly one additional slot is committed per pass.
The reveal order is confidence-based and need not be left-to-right. For
example, the model may commit ply four first, use it to revise predictions for
the other slots, and commit the root move later.

Thus current refinement is order-adaptive progressive completion. It is not
continuous optimization, and it is not a rewrite loop over a completed
trajectory. A one-pass decode predicts all slots simultaneously; an
eight-pass decode gives later predictions seven opportunities to condition on
an increasingly complete action line. This alone can make p8 better than p1.

The implementation is in research/train_torch.py in
_torch_refine_dfm_actions.

## Baseline train and inference paths

### Training

The baseline samples one continuous noise level and masks each action with
probability one minus that level. DFM cross-entropy recovers the observed
actions. A second clean DFM call on the observed action line supplies hidden
states to JEPA. JEPA predicts future latents recurrently, and the WDL head
predicts values from those predicted latents.

The JEPA and WDL gradients reach the shared encoder and the clean DFM action
hidden states. Therefore these losses can improve the policy representation
even when JEPA is absent at inference.

There is one important causal caveat: the current DFM action transformer is
bidirectional across the clean eight-action line. Consequently the hidden
state used for the first JEPA transition may contain information about later
actions. This is valid for a plan-conditioned trajectory model, but it is not
a strict Markov state-action-state transition. The required ablation is:

- action embedding only or causally masked action hidden for strict dynamics;
- current full-line DFM hidden for plan-conditioned dynamics.

### Inference

With both feedback switches disabled, inference runs only the progressive DFM
decoder. JEPA and the value head provide no signal between denoising passes.
They may be run after the trajectory is complete for diagnostics and
interpretability, but they do not alter the selected root move.

Models can work in this regime because training can compile predictive
structure into the shared representation and policy. This is analogous to
auxiliary self-predictive representation objectives: the training task
improves features, while the auxiliary predictor need not execute in the
deployed policy. The appropriate claim is amortized or internalized planning,
not causal online planning.

## Closed-loop predicted-state feedback

The implemented predicted_jepa_tokens ablation changes inference to:

    partial actions
      -> DFM provisional complete line
      -> JEPA predicted z1:8
      -> per-slot JEPA context adapter
      -> next DFM pass

After every non-final DFM pass:

1. Already committed actions remain fixed.
2. Every remaining mask is provisionally filled by argmax.
3. JEPA rolls through that full provisional line.
4. An adapter maps predicted z1 through z8 to DFM action-slot contexts.
5. The next DFM pass consumes the partial committed line and those contexts.
6. JEPA is recomputed after proposals change.

The context is one pass old when consumed, but is refreshed after every pass.
No WDL/value-head call occurs in this inference loop. The feedback is the rich
predicted state, not a scalar value.

Current closed-loop training performs a preliminary noisy DFM call, one JEPA
proposal rollout, and a second noisy DFM call at the same sampled mask/time.
The hard proposal actions are detached. Cross-entropy gradients can still
flow through JEPA context and the preliminary continuous hidden states, but
not through the argmax token identity. The ordinary clean DFM and clean JEPA
objectives are also retained.

Total calls per example are therefore:

- baseline training: two DFM calls and one JEPA rollout;
- current closed-loop training: three DFM calls and two JEPA rollouts;
- baseline p8 inference: eight DFM calls;
- full closed-loop p8 inference: eight DFM calls and seven recurrent
  eight-step JEPA rollouts.

The current closed-loop train path sees one correction, while inference can
see seven. That is a larger mismatch than ordinary random-time discrete
diffusion. If this mechanism is promoted, training should sample a loop depth
and unroll shared-weight adjacent refinement steps, with hard commits and
explicit truncated-gradient semantics. A cheaper middle-feedback variant is
also worth measuring.

## What the value loss does and does not do

Supervised WDL loss teaches value prediction. It does not maximize predicted
value and does not by itself provide a policy-improvement gradient.

With one provisional trajectory, a scalar V says whether that trajectory is
good but does not identify which alternative action would be better. Explicit
value-guided action choice requires one of:

- several candidate lines and value reranking, which is test-time search;
- action-specific Q estimates over alternatives;
- differentiable or straight-through optimization of soft action tokens,
  which risks world-model exploitation; or
- offline policy-improvement targets derived from search values.

Merely computing V during a denoising pass cannot change the DFM output. State
feedback is richer: the next DFM can learn a correction from the predicted
consequence, while the value head shapes which distinctions the latent state
must preserve.

The clean experimental separation is:

1. Primary searchless baseline: p8 DFM, JEPA/value training auxiliaries only.
2. Causal online-planning ablation: predicted JEPA state context between
   passes, without scalar-value search.
3. Search comparator: sample or branch candidate lines and rerank by predicted
   value, labeled explicitly as test-time search.

## Official LC0 value supervision

The downloaded V6 Test80 archive contains, per position:

- root Q, D, and moves-left estimate;
- best-move Q, D, and moves-left estimate;
- played-move Q, D, and moves-left estimate;
- final result Q and D;
- original pre-repair Q, D, and moves-left estimate;
- visits, played and best policy indices, policy KL, and the full sparse legal
  visit distribution.

For predicted successor state z at i plus h plus 1, the default V target is
root Q/D from record i plus h plus 1, expressed from that successor state's
side-to-move perspective. It converts to WDL as:

    win  = (1 - D + Q) / 2
    draw = D
    loss = (1 - D - Q) / 2

Played Q/D at record i plus h is an action-value target for the transition
before the side-to-move flip. A useful audit is played_Q at s,a versus
negative root_Q at the recorded successor, with draw unchanged. Best Q/D is
retained for later Q and policy-improvement experiments, not substituted
silently for V.

The earlier parser incorrectly interpreted root_q, best_q, root_d, best_d as
Q, win, draw, loss. The parser and regression tests now follow the official
V6 structure.

## Storage and data-loader contract

The source tar is 1,773,731,840 bytes and contains 78,608 gzip game members.
The archive mixes standard chess and Chess960. A 1,000-game audit found 951
standard games and 49 Chess960/other starting layouts. Chess960 is filtered
by exact initial-position bitboards because Hero's persisted standard-chess
action codec cannot represent its castling semantics without collisions.

The lc0-sequential-v1 format stores every position once:

- 104 packed uint64 history planes;
- eight bytes of auxiliary-plane metadata;
- all fifteen V6 value floats;
- played and best actions;
- visits and policy metadata;
- sparse legal policy indices/probabilities in CSR form; and
- game offsets and immutable source member names.

Eight-ply windows are gathered at load time and never cross a game boundary.
The final position of each game is not used as a start because its successor
state is not present. Shards are uncompressed NPY arrays so Modal or rented
workers can memory-map them after one staged download; no gzip decompression
occurs in the GPU batch path.

The 64-game pilot produced 6,966 positions at about 1,044 bytes per position.
It materialized a 1,024-example, eight-ply batch at about 3,000 examples per
second on local CPU, substantially ahead of observed GPU training throughput.
Every valid action was present in its sparse legal set and WDL mass normalized
within 1.2e-7.

The completed conversion processed all 78,608 members, retained 75,424
standard-chess games, filtered 3,184 Chess960 games, and wrote 8,315,254
positions. The processed dataset occupies 8,635,794,032 bytes; together with
the 1,773,731,840-byte source archive, the whole workspace occupies about
19.30 GB and remains below the assumed 30 GB ceiling. The source SHA-256 is:

    1c5e5d0d1d335bfeca9693700a1ad1415abfb190772bd051d1f00cb193eb3c2f

The whole-dataset gate passed. Across 8,239,830 stored transitions, every
played and best action is in its sparse legal set, all required value fields
are finite, and every root Q/D pair defines a valid WDL distribution. The
largest legal set is 74 moves. Float16 sparse-policy storage changes total
mass by at most 0.0004445. Played Q versus negative successor root Q has
correlation 0.99704 and mean absolute error 0.03221; played D versus successor
root D has correlation 0.99665 and mean absolute error 0.02171. This strongly
supports successor root Q/D as the default state-value target while retaining
played Q/D for an eventual action-value head.

Conversion is chunk-atomic and resumable. Each committed chunk has an
immutable receipt, and progress advances only after its directory rename.
The split is deterministic at game level using a seeded BLAKE2b hash:
98 percent train, 1 percent validation, and 1 percent test.

## Frozen model comparison matrix

The incumbent and proposed model names are:

| Arm | Training/inference contract | Purpose |
|---|---|---|
| H1 | frozen Hero 1; one-way DFM to JEPA graph; no teacher KL or JEPA inference feedback; report p1 and p8 | incumbent artifact |
| A / Hero2-open-loop | LC0 successor Q/D and direct current-state Q/D supervision; one-way DFM to JEPA graph; no teacher KL, JEPA inference feedback, or value-based action selection; report p1 and p8 | primary Proposal A model |
| B / Hero2-feedback | exactly A except predicted JEPA state context is consumed between DFM passes | causal online state-feedback test |
| C / causal-SAS | exactly the selected A/B inference mode except JEPA receives an action embedding or causally masked action hidden instead of the bidirectional clean full-line hidden | future-action leakage control |
| D / searched | A supplies leaf policy and value to an explicitly budgeted LC0 search | search comparator, never labeled searchless |

For A versus B, keep encoder initialization, sequential data order, optimizer,
schedule, loss coefficients, and DFM inference-pass count matched. H1 remains
sealed and is evaluated, not retrained. Proposal A is deliberately an improved
open-loop Hero control rather than a new feedback architecture: if it used the
old data and optimizer, its only behavioral difference from H1 would be the
chosen p1/p8 inference setting.

The p1/p8 setting is selected separately for every retained checkpoint. Hero
1's confirmed optimum was p1; Proposal A must report both and cannot call p8
the production setting unless its own matched Arena supports that choice.

Report p1 and p8 policy metrics, paired Arena strength, value calibration,
JEPA rollout error by horizon, state-context intervention effects, inference
latency, and DFM/JEPA call counts. A p8 gain in A is evidence for iterative
action completion, not by itself evidence for JEPA-mediated online planning.
A causal online-planning claim requires B to beat its matched A control and
must survive context ablation, context shuffling, and predicted-state
interventions.

## LC0 MCTS backend output contract

LC0 search does not consume the eight-ply trajectory directly. For every
expanded leaf, its current backend API supplies the position history and the
legal moves, and requires:

- `p`: one finite prior probability per supplied legal move, in exactly that
  order, each in `[0, 1]` and normalized across the legal set;
- `q`: expected outcome `win - loss` in `[-1, 1]`, from the leaf side-to-move
  perspective;
- `d`: draw probability in `[0, 1]`, with `win` and `loss` recoverable from
  `q,d`; and
- `m`: predicted remaining plies, which is optional when the backend declares
  that it has no moves-left head.

The engine creates legal successors, batches leaf evaluations, combines `p`
with backed-up `q,d` using PUCT, flips `q` at every ply, and chooses the root
move primarily by visits. It therefore does not require per-action Q values,
JEPA states, or a serialized search tree from the model.

Proposal A can meet this interface as follows:

1. Run one DFM call with all eight action slots masked, select the root-slot
   logits at the 1,858-action canonical LC0 indices, mask to the engine's legal
   moves, and apply a legal-set softmax to produce `p`. Search-leaf evaluation
   should use this p1 marginal even if standalone Proposal A is also evaluated
   with p8 trajectory completion; eight DFM calls per MCTS leaf would conflate
   network quality and a large hidden inference multiplier.
2. Apply the shared WDL head directly to current `z0` and return
   `q = P(win) - P(loss)` and `d = P(draw)`. The value-rich sequential loader
   must supply the actual current record's root Q/D target; negating the first
   successor target is only an approximation and is not the training contract.
3. Initially declare `has_mlh = false`. A moves-left head can later use stored
   root M targets, but it is not required for valid LC0 search. Disable MLH for
   both sides in the first BT4-versus-Proposal-A search comparison if strict
   search-mechanism parity is desired.

The first search suite should distinguish two Proposal A adapters:

- updated encoder-native policy/value heads, testing whether the adapted BT4
  encoder itself became a better LC0 evaluator; and
- DFM root policy plus the direct shared WDL head, testing the complete
  Proposal A evaluator.

Compare those against raw BT4 under identical LC0 version, node count, thread
count, batching, cache, openings, colors, temperature, MLH setting, and search
parameters. This is a search comparison, separate from the searchless p1/p8
Arena.

Primary implementation references are LC0's current
[`EvalResult`](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/backend.h),
the legacy-network wrapper's legal-set
[`SoftmaxPolicy`](https://github.com/LeelaChessZero/lc0/blob/master/src/neural/wrapper.cc),
and classic search's
[`FetchSingleNodeResult`](https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/search.cc).

## Literature relationship

- MaskGIT provides the closest precedent for confidence-ordered iterative
  masked-token generation: https://arxiv.org/abs/2202.04200
- SPR is precedent for action-conditioned multi-step latent prediction as a
  training auxiliary that need not execute in the deployed policy:
  https://arxiv.org/abs/2007.05929
- Diffusion Policy is precedent for observation-conditioned action chunks
  without explicit future-state inference: https://arxiv.org/abs/2303.04137
- V-JEPA 2 and DINO-WM use action-conditioned latent rollouts for downstream
  planning: https://arxiv.org/abs/2506.09985 and
  https://arxiv.org/abs/2411.04983
- MuZero is the clearest reference for learned latent dynamics plus policy and
  value heads, while using explicit planning for policy improvement:
  https://arxiv.org/abs/1911.08265

Our one-way model is closest to Diffusion Policy plus SPR-style latent
dynamics and a MuZero-like value head. The closed-loop state-context variant
is a new local mechanism and must earn its complexity empirically.
