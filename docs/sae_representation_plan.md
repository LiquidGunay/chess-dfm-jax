# BT4 Sparse-Replacement and Representation Study

Status: implementation may begin at the read-only Stage-0/1 gates. The
corrected checkpoint now has 128-pair strength anchors against both the
recovered DFM/JEPA source and original raw BT4. No published sparse artifact
has yet passed the source-model parity gate described below.

Local Stage-0 progress, 2026-07-19: commit `7c896a8` exposes the five normative
pre/post-branch tensors without adding checkpoint state. On the real
15-layer FP32 source, the captured final tokens are bit-identical to the
ordinary local path and every hook at every layer agrees with the independent
`reference_bt4.py` path within `5e-4`. Unit tests also prove that attention and
MLP overrides are applied before `alpha`, residual addition, and layer norm.
This passes the local JAX dense-hook ABI gate. Cross-framework
TransformerLens parity, source sparse reconstruction, and published-artifact
support parity remain pending.

Local Stage-1 core result, 2026-07-21: the immutable FP32 all-pairs run in
`artifacts/representations/dense-stage1-four-model-v2` evaluated 128 pinned
source-game roots using their exact stored plane bytes. It contains coordinate
moments, full covariance spectra, effective/stable/participation ranks,
corresponding-layer CKA, SVCCA, PWCCA, orthogonal Procrustes, principal angles,
complete `15 x 15` layer correspondence, game/board bootstrap intervals, and
fixed-source-head policy controls. The superseded raw-only v1 artifact was
removed after v2 passed every recorded digest and the workspace storage audit.

The dominant representation change predates local autoresearch. At the final
`resid_post_after_ln` square-token hook:

| Pair | Relative L2 | Cosine | Linear CKA | SVCCA | Fixed-head policy JS | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw BT4 -> recovered 265k | 0.652109 | 0.730787 | 0.622069 | 0.557865 | 0.129090 | 56.25% |
| recovered 265k -> compatibility v2/u300 | 0.002882 | 0.999993 | 0.999990 | 0.999970 | 5.11e-7 | 100% |
| recovered 265k -> corrected v2/u400 | 0.002784 | 0.999994 | 0.999992 | 0.999973 | 4.77e-7 | 100% |
| compatibility v2/u300 -> corrected v2/u400 | 0.002822 | 0.999993 | 0.999991 | 0.999972 | 4.49e-7 | 100% |

Across every square-token hook/layer, recovered-to-corrected linear CKA stays
above `0.9999919`. This does not prove semantic identity, but it shows that the
current 30-minute local runs provide almost no dense-backbone movement for a
high-power feature-change comparison. Keep this artifact as the longitudinal
zero point; prioritize a stronger checkpoint before spending the single GPU
on matched sparse refits. Cross-framework parity and one published-artifact
source reconstruction remain hard gates for any fixed-feature interpretation.

This document defines how to compare the original BT4 backbone with the
recovered step-265,000 backbone, promoted intermediate checkpoints, and a final
locally trained BT4 + DFM + JEPA model. It is the detailed Phase 6 companion to
the [local GPU autoresearch plan](local_gpu_autoresearch_plan.md).

Two facts are non-negotiable:

1. The published Leela-SAEs artifacts are **MLP transcoders and LoRSA attention
   branch replacements**, not sparse autoencoders over BT4's final trunk or
   ordinary post-layer residual stream.
2. The current model does **not** feed `z_pred` into DFM action inference.
   JEPA influences the policy through shared training gradients; local
   inference executes BT4 and DFM only.

Confusing either point would change the scientific question being measured.

## Questions

The study should answer four progressively stronger questions:

1. How far do dense BT4 activations move as joint DFM + JEPA training improves
   validation loss and relative Elo?
2. Do the fixed published sparse replacements still reconstruct and causally
   substitute for the same branches after backbone training?
3. Which fixed published features preserve their activation and causal
   behavior, and which are gained, lost, split, merged, or repurposed?
4. After controlling for dictionary training data and compute, do separately
   fitted sparse replacements expose systematically different representations?

The first three questions use the published dictionaries unchanged. The fourth
requires matched refits and cannot use raw feature indices as identities.

## Models under comparison

The minimum comparison set is:

- the original local self-play/policy-tuned BT4 source;
- the recovered step-265,000 joint checkpoint;
- compatibility v2/update 300 as the norm-on local control;
- corrected no-norm v2/update 400 as the first selected BT4 + DFM + JEPA
  checkpoint;
- preregistered log-spaced checkpoints from later promoted local runs; and
- scratch or alternative-pretraining controls when those experiments begin.

The corrected checkpoint state SHA-256 is
`cbeb1bafa74d0983ae4c0cad2a33c405c3507b2428c5bcebbe4fb7f8df44aa61`.
At 128 paired openings it scored `49.61%` against the recovered joint source
and `38.09%` against raw BT4. Those results select the comparison set; they do
not claim promotion, absolute Elo, or that lower validation loss improved
chess strength.

For every model, record the repository commit, complete effective training
configuration, parameter digest, checkpoint step, parameter dtype, compute
dtype, source-run lineage, and whether the fixed LC0 heads are source or
checkpoint-specific.

## Dense model and tensor map

The local implementation is
[`chess_dfm_jax/nnx_bt4.py`](../chess_dfm_jax/nnx_bt4.py), with an independent
mathematical path in
[`chess_dfm_jax/reference_bt4.py`](../chess_dfm_jax/reference_bt4.py).

### BT4 input and trunk

The input is an LC0 `INPUT_CLASSICAL_112_PLANE` tensor with shape
`[B, 112, 8, 8]`. The embedding:

1. transposes it to `[B, 8, 8, 112]`;
2. reshapes the board to 64 ordered square tokens;
3. also flattens the first 12 planes to 768 values and projects them to
   `64 x 512`;
4. concatenates the 112 local plane values with that projection, giving 624
   values per square;
5. projects, applies Mish and layer normalization, then learned multiplicative
   and additive gates; and
6. applies the embedding FFN, scaled residual, and another layer
   normalization.

The result is `[B, 64, 1024]`. BT4 then applies 15 bidirectional SmolGen
transformer layers with 32 heads of width 32. For layer `l`, define:

```text
x_l       = layer input                              [B, 64, 1024]
a_l       = projected raw attention branch           [B, 64, 1024]
m_l       = LN1(x_l + alpha * a_l)                   [B, 64, 1024]
f_l       = raw MLP branch evaluated at m_l           [B, 64, 1024]
x_(l+1)   = LN2(m_l + alpha * f_l)                   [B, 64, 1024]

alpha = (2 * 15)^(-1/4) = 30^(-1/4)
BT4 and SmolGen layer-normalization epsilon = 1e-3
```

The attention is non-causal. SmolGen adds a learned
`[B, 32, 64, 64]` bias before the attention softmax.

The current reference capture names `attn_body`, each post-layer
`encoder_l`, and `trunk`. Those captures are insufficient for the published
sparse replacements because they omit `a_l`, `m_l`, and `f_l`.

### Normative hook ABI

Published names are zero-indexed from `L0` through `L14`.

| Published hook | Exact local tensor | Shape | Replacement boundary |
| --- | --- | --- | --- |
| `blocks.l.hook_attn_in` | `x_l`, before Q/K/V and SmolGen | `[B,64,1024]` | LoRSA input |
| `blocks.l.hook_attn_out` | `a_l`, after attention output projection and before `alpha` | `[B,64,1024]` | LoRSA target/output |
| `blocks.l.resid_mid_after_ln` | `m_l = LN1(x_l + alpha*a_l)` | `[B,64,1024]` | transcoder input |
| `blocks.l.hook_mlp_out` | `f_l`, after the second MLP projection and before `alpha` | `[B,64,1024]` | transcoder target/output |
| `blocks.l.resid_post_after_ln` | `x_(l+1) = LN2(m_l + alpha*f_l)` | `[B,64,1024]` | next layer input/final tokens |

In `EncoderLayer.__call__`, `hook_attn_in` is the value at function entry.
The implementation currently multiplies the attention output by `alpha` on
the same expression that reshapes it, so a capture adapter must expose the raw
post-output-projection tensor immediately before that multiplication.
`resid_mid_after_ln` is the output of `ln_attn`. Similarly,
`hook_mlp_out` is `out_flat.reshape(...)` before multiplication by `alpha`,
and `resid_post_after_ln` is the result of `ln_ffn`.

A replacement must decode the raw branch and then let the native BT4 code
apply `alpha`, the residual addition, and the corresponding layer norm:

```text
LoRSA:       x_l -> a_l_hat -> LN1(x_l + alpha * a_l_hat)
transcoder:  m_l -> f_l_hat -> LN2(m_l + alpha * f_l_hat)
```

Feeding `encoder_l` or final `trunk` into either published artifact is an ABI
error, even though all tensors have width 1024.

## DFM and JEPA coupling

The checkpoint-visible implementation lives in the one-file trainer
[`research/train.py`](../research/train.py). The recovered step-265,000
configuration uses horizon 8:

- BT4 width 1024, trainable embedding and encoder at learning rate `1e-5`;
- DFM state projection `1024 -> 256`, four transformer layers, four heads, and
  MLP width 1024;
- JEPA state width 1024;
- a two-layer, eight-head, MLP-4096 state projector;
- a four-layer recurrent vector transition with MLP width 4096;
- main learning rate `3e-4`;
- BF16 BT4/compute and FP32 main parameters;
- target SIGReg coefficient `0.01` and prediction SIGReg coefficient `0`;
- an online projected target without stop-gradient in the recovered
  configuration; and
- zero value and WDL coefficients.

The forward graph is:

```text
current planes ---------------------> shared trainable BT4
                                              |
                         +--------------------+-------------------+
                         |                                        |
                         v                                        v
              square tokens [B,64,1024]               state projector + CLS
                         |                                        |
                  adapter 1024->256                               z_0
                         |
       masked/clean actions + position + time embeddings
                         |
                  four-layer DFM
                    |          |
                    |          +---- clean true-action hidden [B,H,256]
                    v                                |
            action logits [B,H,1858]                 v
                                           hidden adapter 256->1024
                                                     +
                                           true action embedding
                                                     |
                                                     v
                                          recurrent JEPA transition
                                                     |
                                                z_pred [B,H,1024]

current + future planes -> same online BT4 + state projector
                         -> z_all [B,H+1,1024] -> future JEPA targets
```

The gradient paths in the recovered objective are:

| Loss | Parameters and activations reached |
| --- | --- |
| masked DFM cross-entropy and legal-mass loss | current BT4, DFM state adapter, DFM blocks, action/time/position embeddings, output norm and projection |
| JEPA positive loss | recurrent JEPA, JEPA action embedding and hidden adapter, clean DFM hidden path, current BT4/projector through `z_0`, and future BT4/projector through the undetached online targets |
| target SIGReg | shared BT4 and state projector over current and future boards |
| prediction SIGReg, when enabled | predicted-state branch and, through its current-state and clean-DFM inputs, JEPA, DFM, and current BT4 |
| value/WDL losses | inactive in the recovered configuration because both coefficients are zero |

The fixed LC0 policy, value, and moves-left heads are not trained by these
losses.

### Inference boundary

Local inference is defined by
[`research/inference.py`](../research/inference.py). Its model interface is:

```text
encode_bt4_tokens -> dfm_latents -> planner_from_latents
```

Each permitted refinement pass reuses the current BT4/DFM state tokens and
updates action tokens. It does not call the state projector, JEPA transition,
or `z_pred`. Therefore:

> The current system is a DFM policy trained with a JEPA auxiliary objective,
> not a policy that evaluates or conditions on predicted future latents.

JEPA can improve action logits by changing shared parameters during training,
but there is no forward `z_pred -> action logits` edge. A later closed-loop
planner is a distinct experiment and must not be conflated with this baseline.

## Published sparse artifacts

The primary upstream release is
[Leela-SAEs](https://github.com/JacklE0niden/Leela-SAEs), accompanying
[Tracing the Thought of a Grandmaster-level Chess-Playing
Transformer](https://arxiv.org/abs/2604.10158). The audit pins upstream source
revision
[`f946a5736f40397f5c834104aeac6e8e6ff753bd`](https://github.com/JacklE0niden/Leela-SAEs/tree/f946a5736f40397f5c834104aeac6e8e6ff753bd).

### MLP transcoder

The
[published transcoder repository](https://huggingface.co/JacklE0niden/lc0-BT4-tc)
contains every layer `L0` through `L14` for:

- `k_30_e_16`, `k_30_e_32`;
- `k_60_e_16`, `k_60_e_32`; and
- `k_90_e_16`, `k_90_e_32`.

Here `e=16` and `e=32` give 16,384 and 32,768 sparse features at
`d_model=1024`. The encoder input is
`blocks.l.resid_mid_after_ln`, while its reconstruction target is
`blocks.l.hook_mlp_out`. It approximates the MLP function
`m_l -> f_l`; it is not an autoencoder for either tensor and is not a final
trunk SAE.

The pinned upstream
[activation generator](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/examples/gen_tc_BT4.py)
uses float32 BT4 activations, context length 64, both hook families, and a
nominal 801 million activation tokens. The
[training configuration](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/examples/train_tc_BT4.py)
uses top-k, dataset-wise activation normalization, decoder-norm-aware sparsity,
decoder bias, AuxK, seed 42, and projected input/output data.

For a standard transcoder checkpoint:

```text
pre       = m_l @ W_E + b_E
scores    = pre * row_norm(W_D)
support   = top_k_indices(scores, k)
features  = top_k_values(scores, k) / selected(row_norm(W_D))
f_l_hat   = features @ W_D + b_D
```

For vanilla top-k, the final nonzero feature magnitudes are therefore the
selected original preactivations, while decoder norms determine which
preactivations win the support. Omitting either multiplication or division is
not a numerically equivalent implementation.

### LoRSA attention replacement

The
[published LoRSA repository](https://huggingface.co/JacklE0niden/lc0-BT4-lorsa)
contains every layer `L0` through `L14` for the six transcoder-style
combinations above and also:

- `k_64_e_32`;
- `k_128_e_64`, `k_128_e_128`; and
- `k_256_e_128`.

Its input is `blocks.l.hook_attn_in` and its target is
`blocks.l.hook_attn_out`. Published configurations use context length 64,
128 learned Q/K heads of width 32, SmolGen, a learned attention scale, no
causal or rotary mask, and top-k sparse OV heads. The OV-head count equals
`d_sae`. A LoRSA feature is a dynamic sparse attention/value head whose output
is mapped back to width 1024. It is not an independent residual-stream feature.

LoRSA must receive the complete ordered `[B, 64, 1024]` board sequence. It is
invalid to flatten the 64 squares into unrelated activation rows before
running its Q/K/SmolGen computation. The pinned implementation is
[`src/lm_saes/lorsa.py`](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/src/lm_saes/lorsa.py).

### Dataset-normalization fold

Upstream training normalizes each hook family so that its average activation
L2 norm becomes `sqrt(1024) = 32`. Let:

```text
c_in  = sqrt(d_model) / average_input_L2
c_out = sqrt(d_model) / average_output_L2
```

On loading a transcoder for inference, upstream folds this normalization into
the parameters: `b_E /= c_in`, `W_D *= c_in / c_out`, and
`b_D /= c_out`. LoRSA instead multiplies `W_Q`, `W_K`, and `W_V` by `c_in`,
and divides `W_O` and `b_D` by `c_out`. The exact upstream sparse encoder and
fold are in
[`src/lm_saes/sae.py`](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/src/lm_saes/sae.py).

A JAX implementation must either reproduce those folds exactly or normalize
both input and target activations exactly as training did. It must not do both.

## Can the published artifacts be applied to trained backbones?

They are dimensionally compatible as long as every compared backbone retains
15 layers, width 1024, 64 square tokens, and the exact branch boundaries.
That does not establish scientific compatibility.

The correct order is:

1. validate the published artifact against the original dense source model;
2. freeze the artifact weights and normalization;
3. apply that identical artifact at the same hook in every trained backbone;
4. measure reconstruction, support, and causal degradation against each
   backbone's own native branch; and
5. only later fit matched new artifacts to each backbone.

Failure on the original source is an ABI, dense-weight, dtype, normalization,
or upstream-version problem. It is not evidence that fine-tuning changed the
representation.

## Source identity and compatibility risks

### Local dense assets

The immutable local containers audited for this plan have SHA-256 digests:

| Asset | SHA-256 |
| --- | --- |
| `BT4_exported.pb.gz` | `61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651` |
| `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` | `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618` |
| `BT4.onnx` | `4206710c9adf00137f6592f7d5de8991ba1603be72716612caa43977a930bcb0` |

A CPU audit decoded both protobuf containers and found all 444 numeric tensors
bit-identical, including embedding, all encoder layers, and all heads. The
different compressed protobuf hashes therefore identify different containers
for the same decoded dense model. The local exported source is numerically the
named `policytune-332` model.

### Published dense-source ambiguity

The Leela-SAEs model cards identify the sparse checkpoints only as trained on
`lc0/BT4-1024x15x32h`. Their README illustrates obtaining
`BT4-1024x15x32h-swa-6147500` and does not publish the dense-model hash used
for the uploaded artifacts. It does not name the local `policytune-332`
container. Exact equality between the sparse artifacts' training backbone and
the local dense source remains unproven.

### Epsilon and upstream-version risk

At pinned revision `f946a573...`, the
[LC0 config converter](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/TransformerLens/transformer_lens/loading_from_pretrained.py)
declares BT4 epsilon `1e-3`. However, the
[HookedTransformer constructor](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/TransformerLens/transformer_lens/HookedTransformer.py)
constructs the custom `EncoderLayer` without passing the config epsilon. The
[custom layer](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/TransformerLens/transformer_lens/components/leela_encoder.py)
defaults its layer norms and SmolGen norms to `1e-5`. The official local JAX
path uses `1e-3`.

The published checkpoint metadata records the producing code version as
`unknown`, and the repository does not prove that its current `main` is the
revision that produced every uploaded artifact. Weight conversion, including
embedding layer-normalization affine parameters, also needs empirical parity
checking.

Consequently, source-model FP32 branch parity and source reconstruction are
hard gates. Do not “fix” upstream epsilon or conversion behavior until both
the upstream-compatible path and the official JAX path have been measured and
the selected interpretation is recorded.

### Checkpoint format risk

The transcoder files are safetensors and can be read without executing Python
objects. Current LoRSA weights use PyTorch distributed-checkpoint directories
with a pickle-based `.metadata` file and multiple `distcp` shards. Convert
LoRSA in an isolated, version-pinned environment after reviewing the loader;
do not blindly unpickle an external checkpoint in the training environment.

No transcoder or LoRSA weights, loader dependency, branch-capture adapter, or
fixed representation harness are currently part of this repository. The
approved Drive asset set contains trajectory data, the dense base models, and
the recovered joint checkpoint, but no sparse artifacts.

## Frozen representation ABI

Every representation run must write an immutable manifest covering the
following invariants.

### Dense model

- source protobuf container hash and decoded parameter digest;
- compared checkpoint parameter digest and repository commit;
- layer count 15, width 1024, 32 attention heads, and MLP width 1536;
- `alpha = 30^(-1/4)`;
- layer and SmolGen epsilon;
- exact manual attention, SmolGen, Mish, affine layer norm, and residual math;
- fixed-head identity and action-codec identity; and
- parameter and compute dtype.

### Input and tokenization

- raw stored plane bytes, not only FEN;
- `INPUT_CLASSICAL_112_PLANE`;
- game, shard, and sample IDs;
- eight-ply history, repetition, castling, side-to-move, and rule-50 planes;
- LC0 orientation/perspective exactly as stored;
- row-major JAX square token index `8 * row + file`; and
- all 64 positions retained in order.

Reconstructing from FEN can silently discard history and repetition state, so
stored planes are the primary comparison input.

### Hook and sparse artifact

- zero-based layer ID and one of the five normative hook names;
- raw branch before `alpha` versus post-residual/post-layer-normalization
  semantics;
- tensor shape and square ordering;
- sparse file hashes, configuration, library revision, and loader revision;
- `d_model`, `d_sae`, expansion, top-k, decoder bias, AuxK, and SmolGen flags;
- dataset-average input/output activation norms and whether they were folded;
- decoder-norm-aware support selection;
- top-k tie behavior, positivity/nonzero convention, and numerical dtype; and
- explicit TC versus LoRSA artifact type.

Published activations were generated in float32. Primary compatibility and
transfer results must therefore use FP32. BF16 or FP16 sensitivity is a
separate paired ablation, not an undocumented performance optimization.

### Corpus and statistics

- one immutable game-disjoint heldout corpus shared by every model;
- exact sample order and board IDs;
- separate development/selection and final-test partitions;
- aggregation unit declared as board, square token, trajectory, or game;
- confidence intervals bootstrapped at game level;
- identical probe train/test rows and hyperparameters;
- no data-dependent layer, hook, or metric selection on the final test set;
  and
- log-spaced checkpoint schedule fixed before inspecting feature results.

## Staged experiment plan

The stages are ordered. Later stages cannot repair a failed earlier ABI gate.
The strength precondition is complete; Stage 0 is the current implementation
target. The GPU remains single-tenant, so capture/parity jobs must not overlap
training or arena jobs.

### Stage 0: capture and source-parity gate

Add a read-only hook/capture path for all five normative tensors and a
replacement path for the two raw branches. On a fixed FP32 source batch:

1. show that post-layer tensors from the hooked path match the unmodified JAX
   forward; **complete locally in commit `7c896a8`**;
2. export identical stored planes to the pinned upstream-compatible model;
3. compare every layer and branch with max absolute/relative error, cosine,
   and RMS;
4. load one transcoder layer and reproduce upstream features and
   reconstruction in JAX;
5. substitute the full reconstruction at the raw MLP boundary and measure
   downstream policy change; and
6. repeat the parity exercise for one LoRSA layer after safe conversion.

The gate fails closed if source reconstruction is poor, support indices differ,
or an unexplained cross-framework drift accumulates through layers. Report both
the official-epsilon and upstream-default-epsilon result when that ambiguity is
material.

### Stage 1: dense representation drift

Before invoking sparse artifacts, compare every source/checkpoint pair at all
five hooks. Report:

- coordinate mean, RMS, and centered variance;
- covariance eigenspectrum;
- participation ratio, entropy-based effective rank, and stable rank;
- linear CKA;
- SVCCA and PWCCA;
- orthogonal Procrustes `R^2`;
- principal angles;
- a complete `15 x 15` source-layer/checkpoint-layer correspondence matrix;
  and
- both per-square and board-pooled views.

Dense functional controls use the same fixed source LC0 policy head where
shape-compatible and report policy cross-entropy, KL/Jensen-Shannon divergence,
top-1/top-k agreement, and legal probability mass. A checkpoint-native head,
if one exists, is reported separately.

Use identical game-disjoint linear probes for piece occupancy/type/color,
attacks and defenders, pins, check, material, legal moves, and available
tactical labels. Include random-label controls and a raw linear ceiling. Probe
performance is evidence about accessible information, not by itself a claim
that a sparse feature represents the concept.

### Stage 2: frozen published-artifact transfer

For each selected layer and artifact:

1. establish source-native reconstruction;
2. keep weights, normalization, top-k, and feature IDs frozen;
3. reconstruct each checkpoint's own native raw branch from that checkpoint's
   artifact input; and
4. compare the paired degradation from the source.

Report normalized MSE, relative L2, cosine, and explained variance
`1 - SSE/SST`. Also report nominal `k`, positive count, nonzero count, firing
frequency, dead/rare feature rates, feature magnitude distribution,
same-position support Jaccard, activation correlation, and top-feature rank
stability.

Only a frozen dictionary gives published feature IDs a stable meaning across
backbones. Reconstruction degradation is itself a transfer result, provided
Stage 0 passed.

### Stage 3: causal replacement and interventions

Substitute the decoded raw branch before `alpha`, then measure downstream:

- fixed-head policy KL and Jensen-Shannon divergence;
- action-policy logit change;
- top-1/top-k agreement;
- legal probability mass;
- DFM action-logit and hidden-state change; and
- relative-arena strength only after cheap functional checks pass.

For individual features, ablate, patch between matched positions, or clamp the
feature before decoding. Every intervention needs:

- the unchanged dense branch;
- the complete sparse reconstruction;
- an activation- and norm-matched random decoder direction;
- a shuffled-feature control; and
- identical positions and legality masks.

For LoRSA, additionally compare attention-pattern KL and head/group behavior.
Interpret a LoRSA feature as a sparse dynamic OV head, not as a static residual
direction.

### Stage 4: matched refit controls

Published-artifact transfer and refitting answer different questions. Preserve
the frozen-transfer result, then train or fine-tune matched TC/LoRSA instances
for:

- the source backbone;
- each selected trained backbone; and
- any scratch/alternative-pretraining control.

Hold fixed the activation corpus, token budget, optimizer, expansion, top-k,
normalization, initialization family, checkpoint selection, and at least three
seeds. Compare the reconstruction/sparsity Pareto frontier rather than one
unmatched checkpoint.

If fine-tuning a published artifact on a trained backbone, fine-tune an
identical clone on source activations for the same updates as an equal-update
control.

Independently fitted dictionaries do not share feature IDs. Align them using
decoder cosine, encoder similarity, and activation correlation with
one-to-one assignment, then add a many-to-many split/merge graph. Report
unmatched mass and alignment uncertainty rather than forcing every feature
into a pair.

### Stage 5: longitudinal representation dynamics

Apply the frozen dense and sparse metrics to the preregistered log-spaced
checkpoint series. Align representation changes with:

- train and heldout DFM cross-entropy;
- JEPA positive loss and per-horizon metrics;
- target and prediction variance/effective-rank diagnostics;
- relative Elo with confidence intervals;
- collapse and recovery events; and
- feature firing, transfer reconstruction, and causal effects.

Claims of phase transitions, grokking, double descent, feature birth/death, or
causal reorganization require repeated seeds and a predetermined sampling
schedule. A visually striking single run is a hypothesis generator.

## Required outputs

Each run should emit:

- an immutable manifest implementing the ABI checklist;
- per-model/per-layer dense metrics;
- per-artifact reconstruction and support metrics;
- causal intervention definitions and paired outputs;
- game-level bootstrap intervals;
- compact figures generated from machine-readable tables; and
- an explicit gate status explaining whether downstream interpretation is
  allowed.

Do not append these experiments to `research/results.tsv` until the main
autoresearch contract admits representation studies. Keep representation
artifacts in their own immutable result area and link promoted conclusions back
to the main experiment ledger later.

## Open risks and decisions

- The exact dense checkpoint used to train the uploaded sparse artifacts is
  not identified by a published hash.
- The producing sparse-code revision is recorded as `unknown`.
- Upstream config epsilon and constructed-layer default epsilon disagree.
- Exact cross-framework input-plane and embedding parity is not established.
- LoRSA conversion requires a reviewed external-checkpoint boundary.
- The local code does not yet expose or replace the normative raw branches.
- The first layer/config subset for source parity should be chosen before
  examining tuned-model results.
- Concept labels beyond board mechanics need a provenance and leakage audit.

These are gates or preregistration decisions, not reasons to skip the dense
baseline.

## Primary sources

- [Tracing the Thought of a Grandmaster-level Chess-Playing
  Transformer](https://arxiv.org/abs/2604.10158)
- [Leela-SAEs repository](https://github.com/JacklE0niden/Leela-SAEs)
- [Pinned audited Leela-SAEs
  revision](https://github.com/JacklE0niden/Leela-SAEs/tree/f946a5736f40397f5c834104aeac6e8e6ff753bd)
- [Published BT4 transcoder
  checkpoints](https://huggingface.co/JacklE0niden/lc0-BT4-tc)
- [Published BT4 LoRSA
  checkpoints](https://huggingface.co/JacklE0niden/lc0-BT4-lorsa)
- [Pinned BT4 hook
  implementation](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/TransformerLens/transformer_lens/components/leela_encoder.py)
- [Pinned transcoder implementation and normalization
  fold](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/src/lm_saes/sae.py)
- [Pinned LoRSA
  implementation](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/src/lm_saes/lorsa.py)
- [Pinned activation-generation
  configuration](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/examples/gen_tc_BT4.py)
- [Pinned transcoder-training
  configuration](https://github.com/JacklE0niden/Leela-SAEs/blob/f946a5736f40397f5c834104aeac6e8e6ff753bd/examples/train_tc_BT4.py)
