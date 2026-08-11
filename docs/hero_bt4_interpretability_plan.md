# Raw BT4 vs One-Epoch Hero: Torch Interpretability Program

Status: Torch tooling, local literature replications, published sparse
transfer, the exact-token semantic audit, strict LoRSA feature interventions,
the Hero encoder update-budget audit, and the cost-gated all-layer development
probe/causal run are complete. The Hero-v2 FP32/WSD Phase-1 screen and its
fixed-Hero1-teacher follow-up are also complete as of 2026-08-09. That
checkpoint won a development arena against the matched 554-update Hero1
Phase-1 control, but failed JEPA noninferiority and has not faced terminal
one-epoch Hero1. The frozen test split, multi-seed confirmatory analysis,
engine-PV benchmark, component/path causal localization, longer candidate
training, terminal comparison, and paired sparse-model training remain future
work.

This is the primary execution plan for comparing the original LC0 BT4 model
with the terminal one-epoch Hero model in PyTorch. It rebases the older
[sparse-replacement plan](sae_representation_plan.md) onto the retained Hero
artifact and makes model diffing the first experiment. The older document
remains the detailed source for published transcoder and LoRSA compatibility
risks; this document owns experiment order, controls, compute, and acceptance
gates.

The program has two products:

1. a reusable, framework-native interpretability toolkit for BT4-family chess
   models; and
2. a defensible account of what changed between raw BT4 and Hero, whether the
   change is causally used, and whether it improved chess-relevant computation.

The second product is not allowed to outrun the first. A visually interesting
feature, probe, saliency map, or sparse direction is not a result until it
survives the controls defined here.

## Current state and fixed artifacts

The repository is pinned to branch `research/local-gpu-autoresearch`, source
commit `86ba0b3`. The current artifact ledger is:

| Artifact | Canonical identity | Local status |
| --- | --- | --- |
| Raw BT4 protobuf | `BT4_exported.pb.gz`, SHA-256 `61e43e98d2c4cb747498c6bc13a5bbb3b07e985d3f5d87f0351a4db855fa0651` | restored and verified; keep immutable |
| Terminal Hero model | `research/runs/torch_hero_epoch_v1/checkpoint/model.safetensors`, 712,339,616 bytes, SHA-256 `665ac91ad37e14a6c8810068700dba021e39e9c0fe145c8bf620882bb5a71692` | restored and verified |
| Hero manifest | format `chess-dfm-torch-model-v1`, 462 leaves, update/data cursor 27,679 | restored and verified |
| Hero lineage | 28,343,296 examples; PyTorch `2.5.1+cu121`; source mapping digest `c7a22e25a74e959357f294dfdda4a8f47372d79c96ada49131967bec75b2d93c` | retained in `run_config.json` |
| Published LoRSA L14 | `JacklE0niden/lc0-BT4-lorsa`, snapshot `1ec52be63cce41017b3853edbe9421b643a1476d`, upstream code `f946a5736f40397f5c834104aeac6e8e6ff753bd`; converted weights SHA-256 `2dc7c7cd90550bc3489e4940bb39746e8a3cdcfdc613ac73e9bb2e4bdfe5b8fd`; manifest SHA-256 `2fe2d2cf6a132e4a205b25373c50b74e4462caf44057ccb98c0e7e7a3f736b61` | reviewed, converted without pickle, parity-verified to about `4e-6`, and kept private because the upstream artifact declares no license |
| Definitive LoRSA causal development run | `published_lorsa_l14_causal_dev256_v2`; run `sha256:d40ef11d19cc250412825309e81df6b8dfcc4c83d61072f7987d32da871e9d20`; metrics SHA-256 `bcc84d4ea3d604d1fcffc8d301e5d74dc581620a23b7d7684446a55ad64e4427` | 256 group-disjoint development roots, Raw and Hero, strict matched-direction controls, 20,000 resamples; Railway bundle `12e42f17402bfa1a05ee9e32e47f9e4afebab446cfc5403e90daf19a0abcf1c8` byte-exactly round-tripped |
| Hero encoder LR audit | `research/analysis/hero_encoder_lr_audit_20260808.json`; file SHA-256 `cac39ddc221b290a9f6b0107b34b9ef712bb212dd723098648a5744f8d43f534`; content identity `fed1772df9581ee4a5f9205cd8ee326dc703feb139becf46039f165a3acda8b2` | joins the sealed run config, validation, optimizer partition, and parameter diff; diagnoses a severely constrained but confounded encoder update budget |
| Hero2 fixed-teacher Phase-1 candidate | result `hero-training-phase1-v2_fp32_1_12_legality0_hero1distill1-l40s-1712d50db9d0a488031b`; state SHA-256 `72a51f168b6b7146360e9aeb15bb1188b73890f642556f8be8255af88b488354`; 554 updates | remote checkpoint retained with compact local evidence; DFM development score 0.71289 against matched Hero1 Phase 1, policy-only score 0.44336, sealed JEPA gate failed; not a terminal-model winner |

The Hero checkpoint is model-only. It supports inference, representation
analysis, head swaps, interventions, and continued training from weights. It
does not support an exact optimizer/scheduler/RNG resume. We must never label a
weight-only continuation as an exact continuation of the Hero run.

Raw weights and Hero weights are immutable inputs. Derived encoder-only files
would duplicate roughly the same bytes and create another identity to manage,
so the toolkit extracts encoder subtrees in memory from the full checkpoint.

## Foundation completion snapshot

The first implementation slice is complete and reproducible:

| Object | Identity and result |
| --- | --- |
| Frozen development corpus | 128 positions; canonical manifest SHA-256 `9b38b1cb42d187555e2fff2bf83754213bfe0f57420cb037b6d3080fede64c2e`; plane payload SHA-256 `cce0247810039fbe9ebfec863febe1f1b3c3e8d2c92b896c51714afca8df1121` |
| Parameter diff | 462 strictly aligned leaves; retained in `research/analysis/raw_hero_parameter_diff_20260807.json` |
| Matrix spectra | all 160 aligned matrices; 93.6986% of global update energy; report SHA-256 `6655885089cacf9cb4515d17d715067d6c92a4607fa4a667a9b8f5bf8d00973c` |
| Local pilot | run ID `sha256:8e96bdc849cf50e681af9e0c6e64a2a73063716f279071911a2566bb9f662f13`; 128 positions in 35.93 seconds; current checksum-ledger metrics SHA-256 `671cbb68699881e31de68c95119a518f01c811939d637391b1776578c3b72880` |
| Railway result bundle | `467ef03d322bef1f16b680679817c3d84b552ff07a7f80193ac24b46444e4c87`; 6 run files plus immutable bundle manifest; verified idempotent re-upload and byte-for-byte pull |
| Current Railway input bundle | `8be17957ad779e3756092306daa95ad9c1c7f5d482b6409bda7a16bc6d6c7a44`; 7 files and 1,053,370,877 bytes; includes raw/Hero models plus public corpus v2; fully SHA-256 verified by a CPU-only Modal stage |

The run passed BF16 and FP32 bit-exact repeatability and capture-parity
preflights. It captured all five hooks at all 15 layers without retaining raw
activation dumps, computed exact scalar/per-position statistics, bounded
projected CKA/SVCCA/PWCCA/Procrustes/principal-angle summaries, and the full
15 x 15 residual-post depth map. The frozen expensive-layer rule selected
layers 0, 7, and 14; layer 14 was the maximum-change layer.

The exploratory functional result is unambiguous enough to set priorities:

- RR and RH target NLL were 2.8776 and 2.8816 with top-1 accuracy 0.25.
- HR and HH target NLL were 1.4288 and 1.4284 with top-1 accuracy 0.53125.
- Encoder-only swaps had JS divergence about 0.256 and top-1 agreement
  0.367–0.375; head-only swaps had JS divergence `5.84e-6` to `1.40e-5`
  and top-1 agreement 0.984–1.0.
- Corresponding residual-post CKA fell from 0.8186 at layer 0 to 0.2357 at
  layer 14; symmetric relative L2 rose from 0.5306 to 1.1444.

The development conclusion is therefore “the Hero encoder, not the small
native policy-head update, mediates nearly all resolved policy change.” This
is a layer-selection and experiment-routing result, not a strength or causal
claim. The source exposes one placeholder `game_id`, and the 128 rows contain
125 unique plane digests, so game-cluster bootstrap intervals are degenerate
and three repeated boards retain source-row weight. The 64-wide projection is
also not converged: the 32-to-64 CKA sensitivity has mean absolute difference
0.0583 and maximum 0.1034. Confirmatory work must use deduplicated positions,
many true game-disjoint clusters, and either a wider validated sketch or exact
selected-layer statistics.

## Completed exploratory evidence snapshot

The first ambitious development pass now spans the full evidence ladder. All
claims below remain exploratory until the frozen test partition is opened once
under a preregistered, multi-seed, multiplicity-corrected protocol.

| Family | Retained run | Audited outcome |
| --- | --- | --- |
| Literature-method pilot | `raw_hero_literature_pilot_v2`; run `sha256:e0fd1090b49c6fa95f949442dff96b236dfac6eb182d5a49256b1de3ad6f2632`; metrics SHA-256 `7e4752acf4b1815bd27e4125441948a3d892449fa72757eb365ffe8d856a8f31` | All parity/identity gates pass; whole-residual patching reaches essentially full fixed-head delta projection at layer 14; SARFA agrees across models on two examples; integrated gradients fails completeness and is excluded from interpretation. |
| Published MLP-transcoder transfer | `published_tc_l14_transfer_pilot_v1`; run `sha256:4f33f7d5a5f72a00dcb902edafb4039d52f44ae3065f311dbb305ae5cc810fe2`; metrics SHA-256 `80ae71de820d525c4d5e9c498566858dba78e26a30eae3272ea9b37f4c9726f7` | Source-compatibility gate passes; Hero normalized reconstruction MSE is 0.9796 times Raw and all 16 policy top-1 decisions survive replacement. This licenses feature auditing, not semantic equivalence. |
| Published LoRSA transfer | `published_lorsa_l14_transfer_dev512_v3`; run `sha256:e222c1cce5a2824c48168b5cdba3b2018211eb4c7817be95e354f9e5388e91f6`; metrics SHA-256 `63da2ac0237dc95b86a55921d151a9b5f6cc56714ecec318f0bcffd700a362d4` | All 51 source files and every artifact dependency are bound into the run. Hero reconstruction NMSE is 0.99536 times Raw, while policy JS degradation is larger by `0.00061246`; both models retain 502/512 native top-1 moves under replacement. |
| Paired LoRSA uncertainty | `published_lorsa_l14_transfer_dev512_paired_bootstrap_v3`; run `sha256:133adf9ce415c2ee5170eb4e41c33e8ddbfa19beb2448e7009e09f596201b5c2`; metrics SHA-256 `5bcfa32108c6710cfbc5f2a7ee61c7c67e3524a57de9d4ec4755ba1b8ccc82ce` | With 20,000 paired root bootstraps, reconstruction-NMSE and cosine deltas are unresolved; policy-JS degradation is small but resolved (`+0.000612`, 95% interval `[+0.000261,+0.000999]`). Raw-only and Hero-only top-1 failures are 4 versus 4 (exact McNemar `p=1`). |
| Exact-token LoRSA audit | `published_lorsa_l14_token_semantics_dev512_v3`; run `sha256:a5c2d7bd75b4371a8d9778538a5bd0dc1f9be84e261e441eed22ee00fd150834`; metrics SHA-256 `5f61d5aff5d6865bf977a3d86131e9747398f9a0d57093f7f2984ac083276b80` | Group-disjoint Raw fit and Raw/Hero audit halves produce 430 balanced concept-feature pairs. Raw replication is 410/412 and Hero transfer 406/408 among supported pairs; two Raw nonreplications, two Hero nontransfers, and two cross-model disagreements are retained. These are descriptive activation associations, not causal or confirmatory semantics. |
| Strict LoRSA causal validation | `published_lorsa_l14_causal_dev256_v2`; run `sha256:d40ef11d19cc250412825309e81df6b8dfcc4c83d61072f7987d32da871e9d20`; metrics SHA-256 `bcc84d4ea3d604d1fcffc8d301e5d74dc581620a23b7d7684446a55ad64e4427` | Target scalar, token support, and decoder norm are identical across each target and three random decoder directions. Features 10843 (own knight), 5429 (own queen), and 10784 (attacked/undefended ours) pass the Holm-corrected specificity and ablation-dose gates in both models. Feature 6661 and the null 5851 fail; 4516 has only two active positions. |
| Hero encoder update-budget audit | `hero_encoder_lr_audit_20260808.json`; content identity `fed1772df9581ee4a5f9205cd8ee326dc703feb139becf46039f165a3acda8b2` | Encoder peak LR was 30x below the main LR and all 411 encoder leaves were BF16 direct-update leaves. Muon-managed parameters were 99.707% bit-identical and carried only 0.251% of total encoder delta energy. This supports a precision-plus-LR intervention, not the causal claim that LR alone was wrong. |
| Fixed-teacher Hero2 Phase-1 screen | candidate state `72a51f168b6b7146360e9aeb15bb1188b73890f642556f8be8255af88b488354`; decision `v2_fp32_1_12_legality0_hero1distill1-decision.json` | Frozen legal root top-1 is 0.50061. The development DFM arena scored 0.71289 against the matched 554-update Hero1 control; policy-only play was inconclusive at 0.44336. JEPA noninferiority failed, so this is a continuation candidate, not a promoted or terminal winner. |
| Pedagogical notebook | `research/notebooks/raw_hero_interpretability.py`; reactive marimo source | Strict marimo check, widget states, evidence ledgers, final-chain contracts, fresh no-code export, and visual QA pass. Positive and negative sparse-feature examples, strict causal v2 results, the encoder update audit, and all claim boundaries are shown together. |
| Powered all-layer probes | `modal_probe_all_layers_dev_v1`; run `sha256:8c9ce9ee97af6f6753688a36a1997994f8803fe358578edf07be7b356b22d963`; metrics SHA-256 `c9823c56098d62767e2f9b79506dc98ff57bfc0b76d85602969752671e332c28` | 1,024 fit, 256 selection, and 512 development-evaluation roots; all 15 layers; third-ply public puzzle labels; frozen test untouched. |
| Independent validation | `raw_hero_interpretability_report_v1/validation.json`, SHA-256 `390e8b8053cecd8972614ae08ef2d34a85cc6b8476aee208513e68c22f9ac134` | Reconstructed splits/IDs and corpus integrity match; 120 metric groups recompute from saved predictions with maximum absolute error `4.22e-15`; all paired points and intervals match; safetensors-only weights; assessment `share with caveats`. |
| Technical synthesis | `raw_hero_interpretability_report_v1/report.html`, SHA-256 `9f90cb63ae3fa7284f5b4455a78aee16014f62c12492d5b7e8153d1cd15bf302` | Self-contained, source-linked report; package/schema and structural verification pass. Browser interaction QA is explicitly unverified because both installed headless engines failed the packaged reader contract. |

The powered probe result is not “Hero is better everywhere.” Legal-destination
paired accuracy favors Hero at 13 of 15 layers, Raw at layer 0, and is
unresolved at layer 3. Opponent-attack decoding is mixed: nine Hero-favoring,
three Raw-favoring, and three unresolved layers. Piece-code differences resolve
at only four layers and favor Raw at layers 10, 12, and 14. Both models encode
third-ply destination/joint-move information late. The only resolved destination
difference is layer 6 in Raw's favor; Hero has a resolved `+0.0547` joint-move
advantage at layer 14, but no systematic depth-wide lookahead advantage.

All 30 learned lookahead directions causally move their probe objective, but
only Raw layer 2 has a current-policy target-log-odds derivative interval that
excludes zero (mean `0.00252`, 95% interval `[0.000307, 0.00558]`). Random
probe-objective intervals contain zero in 26 of 30 controls. The defensible
interpretation is therefore **decodable, steerable representation change with
weak evidence of current-policy use**. It is not evidence that either model
implements internal search.

## Hero encoder update-budget verdict

The user concern about the 30x lower encoder learning rate is supported, with
one important correction: **learning rate and numerical precision are
confounded in Hero v1**. The peak main/encoder rates were `5e-4` and
`1.6667e-5`, and the integrated one-epoch schedule budgets were `6.9265` and
`0.23088`. All 411 encoder leaves were BF16; their first moments, Adam second
moments, and parameters used the parameter dtype, and updates were applied
directly in place without a recorded FP32 master-weight accumulator.

The observed movement is consistent with update quantization. Across
198,458,624 encoder parameters, 96.995% are bit-identical to Raw. The 93
Muon-managed encoder leaves contain 113,246,208 parameters, of which 99.707%
are unchanged; they contribute only 0.251% of total encoder delta energy. In
contrast, AdamW-managed leaves carry 99.749% of delta energy, concentrated in
SmolGen (68.55%), the embedding (26.51%), layer norms, and biases. Attention
and MLP matrices contribute only 2.50% and 0.62%. This is too lopsided to treat
Hero as a clean test of broad encoder adaptation.

Hero v1 is still a valid artifact: from the retained 10%-to-100% validation
trajectory, root legal conditional CE improves by `0.3081`, top-1 by `0.0398`,
first legal mass by `0.2138`, DFM CE by `0.7920`, and JEPA MSE by `0.0459`.
The trajectory begins at 10%, however, and is not a Raw step-zero control. The
correct conclusion is therefore “Hero learned useful behavior while its dense
encoder matrices were severely update-constrained,” not “the run failed” and
not “30x lower LR alone caused the constraint.”

The audited encoder inventory made the memory tradeoff concrete. Keeping BF16
compute weights while adding FP32 masters costs 793,834,496 bytes; upcasting
one Muon moment and two AdamW moments adds another 567,342,080 bytes, for a
1,361,176,576-byte (1.268 GiB) persistent-state increment before activations,
gradients, allocator slack, and compile workspaces. The subsequent real-GPU
Gate 0 measured 22.049 GB allocated and 30.021 GB reserved for Hero v2.
Consequently 24-GiB workers are not eligible for this recipe; the retained
training route uses an L40S-class 48-GiB worker with explicit memory and cost
guards.

The precision repair and WSD scheduler are now implemented, exact-resume tested,
and screened remotely at the matched 554-update Phase-1 boundary. FP32 master
weights and moments substantially accelerated predictive-loss learning, but
the unconstrained whole-encoder arms also moved the trunk roughly an order of
magnitude more than the exact Hero1 control and regressed legal-masked root
ranking. Encoder-only and encoder-trunk-only controls localized that regression
to broad trunk adaptation; the main-only FP32 arm retained arena parity but did
not improve the predictive suite.

A legality-zero arm confirmed that illegal probability mass was not the
deployed action-space problem: legal-masked root play still missed its frozen
gate. Adding online stopped-policy distillation restored only DFM parity and
failed policy-only preservation. Replacing that moving target with the exact
sealed Hero1 Phase-1 checkpoint produced the retained candidate: frozen root
top-1 0.50061, DFM development-arena score 0.71289 with pair-aware 95% interval
0.59285 to 0.83293, and policy-only score 0.44336 with interval 0.32332 to
0.56340. The same candidate failed the sealed JEPA noninferiority bound
(0.24134 versus 0.20712), so it is not authorized for Phase 2 under that
preregistration and has not been shown better than terminal Hero1.

Before another training arm, decompose immutable-teacher KL gradients against
JEPA, SIGReg, WDL, and policy gradients by parameter group without taking an
optimizer step. Use that audit to preregister exactly one balance intervention:
a lower or decaying teacher coefficient if conflict is broadly scalar, or
teacher-gradient scaling/routing if conflict is concentrated in the inherited
trunk. The detailed implementation, complete Phase-1 evidence, arena scope,
cost audit, and continuation gate live in
[`docs/hero_v2_optimizer_plan.md`](hero_v2_optimizer_plan.md).

## Exact comparison objects

“Raw versus Hero” is ambiguous because both the encoder and policy head can
change. The primary policy-only comparison is therefore the following 2 x 2
lattice:

| Arm | Encoder and embedding | LC0 attention-policy head | Question |
| --- | --- | --- | --- |
| RR | raw | raw | original baseline |
| HR | Hero | raw | encoder-mediated change under a fixed readout |
| RH | raw | Hero | head-mediated change under a fixed representation |
| HH | Hero | Hero | terminal Hero policy-only model |

“Encoder” means the input embedding plus all 15 BT4 transformer layers.
“Head” means the native LC0 attention-policy head. The four arms must share the
same Torch implementation, action codec, legal mask, input planes, dtype, and
batch order. They are assembled as state-dict subtree overlays, not stored as
four checkpoints.

The primary scientific comparison is RR versus HH, decomposed by HR and RH.
The native DFM/JEPA system is secondary because it changes the forward
algorithm as well as the weights. After the encoder result is understood, add
Hero DFM policy results at fixed preregistered refinement counts 1 and 16.
JEPA is studied as an auxiliary representation and training influence unless a
future model explicitly feeds predicted JEPA states back into action logits.

## Questions and falsifiable outcomes

The study is organized from weak evidence to strong evidence:

1. **Parameter delta:** where did the weights move, and is the movement larger
   than serialization, dtype, and mapping noise?
2. **Activation delta:** on identical stored positions, where do internal
   representations diverge?
3. **Functional delta:** which logit, ranking, calibration, legality, and value
   behaviors change?
4. **Semantic delta:** which human- or engine-defined chess variables become
   more or less decodable?
5. **Causal delta:** which changed components or directions mediate the changed
   move distribution?
6. **Algorithmic delta:** is there stronger evidence of iterative lookahead,
   future-move representation, or dynamic tactical computation?

The null outcome is legitimate: Hero may be behaviorally different while its
encoder is almost unchanged, or neither difference may be detectable with our
power. The toolkit must be able to report “no resolved encoder delta” without
turning noisy exploratory features into a story.

Before each experiment family, write its sample set, metric, aggregation unit,
seed, exclusions, and stopping rule into the run manifest. Confirmatory tests
use the frozen test partition once. Exploratory work uses development data and
is labeled exploratory.

## First-principles tensor contract

The LC0 input is `[B,112,8,8]`. The embedding yields 64 ordered square tokens
of width 1024. There are 15 non-causal SmolGen transformer layers, 32 attention
heads of width 32, MLP width 1536, residual scale
`alpha = (2 * 15)^(-1/4)`, and layer-normalization epsilon `1e-3`.

For layer `l`:

```text
x_l       = layer input
a_l       = attention branch after output projection, before alpha
m_l       = LN1(x_l + alpha * a_l)
f_l       = MLP branch output, before alpha
x_(l+1)   = LN2(m_l + alpha * f_l)
```

The frozen Torch hook ABI is:

| Canonical name | Tensor | Shape | Intervention meaning |
| --- | --- | --- | --- |
| `blocks.l.hook_attn_in` | `x_l` | `[B,64,1024]` | input to Q/K/V and SmolGen |
| `blocks.l.hook_attn_out` | `a_l` | `[B,64,1024]` | replace before alpha/residual/LN1 |
| `blocks.l.resid_mid_after_ln` | `m_l` | `[B,64,1024]` | MLP/transcoder input |
| `blocks.l.hook_mlp_out` | `f_l` | `[B,64,1024]` | replace before alpha/residual/LN2 |
| `blocks.l.resid_post_after_ln` | `x_(l+1)` | `[B,64,1024]` | next layer/final trunk tokens |

Captures must not add parameters or buffers to checkpoints. Capture-only mode
must be bit-identical to the ordinary Torch forward. Override shape, dtype, and
device must match the native branch. The first implementation slice exposes
all five tensors, selected-layer capture, and raw attention/MLP branch
replacement. Residual-stream patching, Q/K/V/head capture, policy-head
intermediates, gradients, and Jacobian-vector products build on this ABI.

## Corpora and labels

No single corpus can answer all questions. We use five disjoint families with
stable position IDs and stored LC0 plane bytes.

### A. Natural heldout positions

Sample game-disjoint roots from the retained trajectory-v3 test split. This is
the primary distribution for parameter-to-activation and behavior comparisons.
Store shard, example, game, ply, history, and raw plane digest. FEN is metadata,
not the source of truth: reconstructing from FEN can lose repetition and
history planes.

Use a deterministic nested schedule so cheap pilots are prefixes of expensive
runs: 128, 512, 2,048, then 10,000 positions. The confirmatory test set is not
used for layer selection, feature naming, or hyperparameter tuning.

### B. Tactical and strategic concepts

Build engine-verified sets for material, square occupancy, legal moves, check,
pins, skewers, forks, discovered attacks, hanging pieces, passed pawns, king
safety, promotion threats, mating nets, and exchange sacrifices. Each label
records the engine/version, analysis settings, rule, confidence, and whether it
is state-derived or engine-inferred.

Positive examples require matched negatives. Match side to move, ply, material
range, phase, legal-move count, evaluation range, and source family where
possible. Otherwise a probe can solve dataset artifacts instead of the chess
concept.

### C. Model-disagreement positions

Select positions by RR/HH legal-policy JS divergence, top-1 disagreement,
rank displacement, or evaluation swing, using development data only. Stratify
by tactical/quiet, game phase, confidence, and which model agrees with a fixed
engine budget. Keep a matched low-divergence control set.

These positions have high diagnostic yield but are not an unbiased performance
estimate. Report them separately from natural heldout results.

### D. Minimal counterfactuals

Create legal paired positions that change one fact while preserving as much
context as possible: add/remove or relocate a piece, toggle a blocker, alter a
pin, change a defender, change side to move, or cross a tactical threshold.
Reject pairs with unintended large engine or legality changes. Preserve both
FEN and encoded-plane digests.

Minimal pairs support finite-difference activation analysis and causal tests;
they are stronger than correlational probes because nuisance variation is
reduced.

### E. Short trajectories and variations

For lookahead work, retain roots with engine principal variations, alternative
branches, played continuations, and future board states for up to seven plies.
Split by source game and root so future positions from one line cannot leak
across train/test.

## Stage 0 — artifact, environment, and parity gates

This stage blocks every scientific claim.

1. Verify raw archive/container hashes and Hero bundle/checkpoint hashes.
2. Record git commit, Python, Torch, CUDA runtime, driver, GPU, dtype, and model
   configuration in every run.
3. Strict-load all Hero leaves and reject missing, extra, shape-mismatched, or
   dtype-mismatched values.
4. Load raw BT4 through the existing strict source mapper and preserve its
   decoded mapping digest.
5. Re-run raw Torch policy parity against the retained parity oracle on a small
   fixed corpus. The local experimental venv contains no JAX; a retained
   parity artifact or isolated legacy environment is the oracle boundary.
6. Prove capture-only output equals ordinary output exactly in FP32 and in the
   deployed BF16 path.
7. Prove attention/MLP replacement happens before alpha, residual addition,
   and layer normalization.
8. Run repeatability twice and compare output/artifact digests.

Gate: no model diff begins until both models produce finite logits, identical
legal masks, deterministic position order, and a complete provenance manifest.

## Stage 1 — model diffing before interpretation

Model diffing answers whether a meaningful delta exists and where to spend the
rest of the budget.

### 1A. Parameter diff

For every aligned parameter, report:

- absolute and relative L2/Frobenius norm;
- cosine similarity and update-to-weight norm ratio;
- mean, RMS, max, quantiles, and fraction unchanged at stored precision;
- singular-value spectrum and effective rank of matrix deltas;
- layer/module totals split into embedding, Q, K, V, O, SmolGen, MLP, layer
  norms, and policy head; and
- update alignment with the raw matrix's leading singular directions.

Aggregate only after leaf-level checks. A large layer can dominate global L2,
so include normalized and per-parameter views. Cluster delta matrices and test
whether they are well approximated by low rank; that informs later causal
subspace and compression experiments.

### 1B. Dense activation diff

On identical positions, capture the five hooks at all 15 layers for each model
sequentially. Report corresponding-layer relative L2, cosine, mean/variance,
effective rank, linear CKA, SVCCA, PWCCA, orthogonal Procrustes residual, and
principal angles. Also compute the full 15 x 15 layer-correspondence map to
detect shifts in depth rather than assuming layer identity.

Use game-level bootstrap intervals. For exact corresponding-layer metrics,
stream means and covariance/cross-covariance sufficient statistics in FP64 on
CPU. For all-pairs correspondence, use a frozen random projection/sketch and
validate sketch error on the 128/512-position prefixes.

### 1C. Functional and readout diff

For RR, HR, RH, and HH report legal-policy KL/JS, top-k overlap, rank
correlation, top-1 agreement, entropy, legal mass before masking, target
cross-entropy, calibration, and engine agreement at fixed engine settings.
Decompose RR-to-HH change by the two mixed arms; include an encoder-head
interaction term rather than assuming additive effects.

### 1D. Delta localization decision

Use development data to choose at most four primary layers for expensive
analysis: an early, middle, late, and maximum-change layer. Freeze that choice
before final evaluation. If the encoder delta is below the preregistered
resolution threshold across parameters, activations, and fixed-head behavior,
stop encoder-specific refits and redirect effort to head/DFM analysis.

## Stage 2 — attribution and concept replication

Replicate established chess-model techniques on both RR and HH with identical
inputs and evaluation.

### Input and action attribution

- SARFA-style action-specific saliency, which contrasts the selected move
  against alternatives rather than measuring only generic sensitivity;
- integrated gradients with multiple legal baselines and convergence checks;
- gradient x input and SmoothGrad as diagnostics;
- Grad-CAM only as a historical baseline, with its limitations stated; and
- information-interaction maps for pairwise square/piece interactions.

Evaluate attribution with deletion/insertion curves, legal counterfactuals,
stability across seeds/baselines, and localization on engine-derived tactical
facts. A prettier heat map is not evidence of a better explanation.

### Linear, nonlinear, and sparse concept probes

Replicate board-state and chess-concept probing at every layer. Start with
regularized linear probes; add small nonlinear probes only to measure
nonlinearly available information. Use group-disjoint splits, fixed
regularization grids, class balancing, selectivity controls, label
randomization, and matched random directions.

Targets include piece type/color/square, attacks and defense, legal moves,
check, material, phase, evaluation bins, tactics, and future principal-
variation moves. Report AUROC/AUPRC or accuracy as appropriate, calibration,
sample count, and confidence intervals. Compare probe deltas, not only each
model's absolute score.

### Post-layer logit lens

Apply each model's own head and the fixed raw/Hero heads to intermediate
residual states where the interface is valid. Record entropy, target rank,
legal mass, and move emergence by depth. Because BT4 is post-LN and the policy
head is attention-based, do not import decoder-transformer logit-lens
assumptions without validation.

## Stage 3 — causal tracing and algorithmic analysis

Correlation establishes candidates; intervention tests use.

### Bidirectional activation patching

Patch raw activations into Hero and Hero activations into raw at matched
positions. Start with whole hook/layer tensors, then squares, attention heads,
MLP channels, and learned subspaces. Use both restoration and corruption:

```text
raw run -> inject Hero component -> measure movement toward Hero logits
Hero run -> inject raw component -> measure movement toward raw logits
```

Primary outcomes are change in legal-policy JS, target-move log-odds, engine
preferred-move rank, and task-specific behavior. Normalize mediation by the
full RR-to-HH effect and report sign reversals and overshoot.

Controls include same-model patching, shuffled positions, mean/noise patches,
norm-matched random subspaces, non-causal layers, and patches selected on a
separate development set.

### Path, head, and edge analysis

Decompose attention into Q/K similarity, SmolGen bias, attention pattern, V/OV
content, and output projection. Patch these separately to avoid calling a
whole attention branch a mechanism. For candidate circuits, use path patching
or edge attribution with heldout validation and test sufficiency/necessity by
keeping or ablating the proposed path.

### Iterative inference and future moves

Reproduce prior tests for future-move and lookahead representations:

- probe whether layer depth increasingly decodes moves or states at plies 1–7;
- compare played continuation, engine principal variation, and alternative
  branches;
- patch a future-move direction and measure current-policy consequences;
- test whether information appears, disappears, or is transformed across
  layers; and
- compare quiet versus forcing positions and correct versus incorrect model
  decisions.

The analysis must distinguish “future move is decodable” from “the model
causally used an internal search-like computation.” The latter requires
interventions and branch-specific predictions.

## Stage 4 — published sparse replacements

Use the detailed compatibility gates in
[the sparse-replacement plan](sae_representation_plan.md).

Published BT4 transcoders approximate the MLP function
`resid_mid_after_ln -> hook_mlp_out`. They are not residual-stream SAEs.

LoRSA is the published sparse attention-branch replacement. It consumes the
complete ordered `hook_attn_in` board sequence, activates a sparse set of
learned dynamic Q/K/OV attention features, and reconstructs
`hook_attn_out`. It is not LoRA, not a static feature dictionary over flattened
squares, and not a final-residual SAE.

The order is mandatory:

1. safely convert/load one artifact in an isolated pinned environment;
2. reproduce upstream feature support and reconstruction on its source path;
3. pass raw-BT4 reconstruction and causal replacement gates;
4. freeze weights and normalization;
5. transfer the identical artifact to Hero; and
6. compare reconstruction, support stability, feature semantics, and causal
   degradation.

### Completed LoRSA development evidence

The published `k_30_e_16/L14` checkpoint has completed that frozen transfer
order. Conversion parity errors are `2.6e-6` to `3.8e-6`; the immutable
safetensors hash is unchanged before and after every run. The final transfer
binds the exact 51-file implementation snapshot under source identity
`806e65aeeb3d7e00b41287a918d3e032e4725f14998f2972f5125099eca90920`.
The checkpoint and derived outputs remain private because neither the model
repository nor its upstream code declares a redistribution license.

On 512 development positions, the final transfer run is
`sha256:e222c1cce5a2824c48168b5cdba3b2018211eb4c7817be95e354f9e5388e91f6`.
Raw/Hero global reconstruction NMSE is `0.0647438/0.0644431`, cosine is
`0.967099/0.967336`, and native-versus-replaced policy JS is
`0.00302918/0.00364164`. Both replacements retain 502/512 native top-1 moves.
The Raw/Hero feature-support Jaccard is `0.583606`; matched supported-feature
magnitudes correlate at `0.6730` over 7,372 resolved entries. Transfer is
therefore numerically viable, but support is not interchangeable and Hero's
small extra policy degradation is real at this development scale.

The 20,000-replicate paired-root bootstrap run
`sha256:133adf9ce415c2ee5170eb4e41c33e8ddbfa19beb2448e7009e09f596201b5c2`
resolves the policy-JS delta (`+0.000612`, 95% interval
`[+0.000261,+0.000999]`) but not the per-position NMSE delta
(`-0.0000777`, `[-0.000508,+0.000348]`) or cosine delta (`+0.000104`,
`[-0.000106,+0.000322]`). The top-1 contingency is 498 both preserved,
4 Raw-only, 4 Hero-only, and 6 neither; exact McNemar `p=1`.

The exact-token audit uses 256 Raw fit roots and a group-disjoint 256-root
Raw/Hero audit half. Selection requires an expected joint opportunity before
forming 215 positive and 215 negative concept-feature pairs. Among supported
tests, 410/412 Raw associations replicate, 406/408 transfer to Hero, and
403/405 agree in sign across models. The retained failures are substantive:
two Raw nonreplications, two Hero nontransfers, and two cross-model sign
disagreements. For example, feature 6661 on `pinned_ours` changes from
`-0.389` Raw lift to `+0.311` Hero lift, while feature 4516 on
`attack_count_theirs=two` changes from `-1.079` to `+0.430`.

The earlier v1 selector is diagnostic only: 164/215 negative pairs had no
expected-joint opportunity gate. The compact peak screen is likewise
censored discovery evidence: it covers only about 3.6% of candidate
concept-feature pairs, and three apparent Hero-positive replications have
joint peak support of one. Neither artifact supports a population-rate or
semantic-equivalence claim.

All final outputs were byte-exactly pulled back from private Railway bundles:
transfer `dee7536374cd85c4573b8399f1d8696947bd63f98784c8b39729b161d4e6a55f`,
bootstrap `86e373d08e3dd464c3171fdce8adb970c178b4e94457d574e51d765e61ee4062`,
compact screen `5ad94b63df56bba029424f6b127d05e95501f49e926ce19af818f4f282bb795e`,
exact-token audit `a3f1f50b2ccabd7794368a2b3a9e30d8201fcd67aba92f1e5b898535b462100f`,
and the exact source snapshot
`c53390977c6dad107ca20614dcb18bcfa44ea3e477987b9d58bc357f35ccdfa8`.
The definitive causal v2 run closes the development intervention gap. It uses
the same 256 held-out development roots and never opens test data. For every
target, three seeded random control directions receive the exact target
activation scalar, target token support, and target decoder norm; only decoder
direction changes. The in-batch no-op numerical floor is bounded at
`6.78e-7` legal-policy TV, exact suffix parity error is zero, LoRSA remains
frozen, and 20,000 paired bootstrap/sign-flip replicates plus Holm correction
are retained. The run took 522.72 local seconds, peaked at 1.168 GB allocated,
retained 11.13 MB, and cost `$0`.

Six of twelve strict Raw/Hero feature gates pass: 10843 (own knight), 5429
(own queen), and 10784 (attacked/undefended ours) pass in both models. Their
Raw target-minus-control ablation-TV effects are `+0.00329`, `+0.00128`, and
`+0.00191`; Hero effects are `+0.00350`, `+0.00140`, and `+0.00204`, with all
95% intervals above zero and Holm-adjusted `p=0.000600`. Feature 6661 fails
specifically: matched random directions move policy more than the target in
both models. The intended null 5851 also fails. Feature 4516 is not estimable
with only two naturally active held-out positions.

The cross-model causal deltas sharpen model diffing. For the three passing
features, median Raw/Hero policy-delta cosine is `0.993`, `0.993`, and `0.996`.
Hero has a resolved larger active-position effect for features 10843
(`+0.000863`, 95% `[+0.000162,+0.001710]`) and 10784 (`+0.000990`,
`[+0.000469,+0.001622]`); the queen-feature magnitude difference is unresolved.
Dose response is directional for every natural-ablation target with adequate
support. Dose-1 insertion is target-direction-specific for the three passing
features in both models, while the queen insertion curve is thresholded rather
than smoothly monotone.

This establishes causal use **inside the frozen LoRSA replacement path**. It
does not establish a native dense-attention feature, input-concept causality,
semantic monosemanticity, or a confirmatory test result. The earlier causal v1
run is retained only as a superseded audit artifact because its controls did
not match target token/scalar support and its baseline had a batch-kernel
numerical floor. Confirmatory test data and matched sparse training remain
undone.

Do not interpret transfer failure as representation change until the artifact
passes against raw BT4. Current LoRSA distributed-checkpoint metadata includes
pickle; review and convert outside the main experiment environment.

After frozen transfer, fit matched transcoders/LoRSA models only at the frozen
selected layers, with identical corpus, activation count, architecture,
sparsity target, optimizer, steps, seed set, and evaluation.

## Stage 5 — new paired-difference methods

The main novel direction is to model the delta directly instead of interpreting
two models independently and matching stories afterward.

### Paired residual delta atlas

For paired activations `h_raw` and `h_hero`, analyze `delta = h_hero - h_raw`
after optional orthogonal Procrustes alignment. Compute delta covariance,
effective rank, position/square concentration, dependence on model
disagreement, and projection onto parameter-delta and concept directions.
Test delta directions causally by adding/subtracting scaled directions in both
models with norm-matched random controls.

### Cross-model sparse decomposition

Train a paired crosscoder or Delta-Crosscoder on matched activations. Separate:

- shared features reconstructed in both models;
- raw-specific and Hero-specific features;
- shared features with changed activation frequency or decoder effect; and
- low-rank continuous drift not captured by sparse features.

Feature identity is established by joint training, decoder geometry, activation
matching, examples, and causal effects—not by coincident feature indices from
independent SAEs. Train on development data and evaluate reconstruction,
sparsity, feature stability, and causal substitution on heldout games.

### Behavior-conditioned model diff

Fit a small, preregistered mediator from internal deltas to RR/HH logit deltas.
Ask whether a compact set of layer/square/features predicts the changed move
distribution out of sample, then intervene on that set. Compare against equal-
dimension PCA, random, and high-variance baselines.

### Counterfactual delta consistency

For minimal position pairs, compare the finite-difference response in each
model and the cross-model delta. A candidate Hero-specific concept should
respond consistently to the controlled board change and causally affect the
relevant move—not merely correlate with a dataset label.

## Stage 6 — causal falsification and synthesis

For every claimed mechanism, run:

- necessity: ablation reduces the claimed behavior;
- sufficiency: insertion/steering increases it in the predicted context;
- specificity: unrelated moves/concepts change less;
- sign and dose response: effects follow preregistered direction and scale;
- cross-position generalization: heldout games and concept templates;
- cross-method agreement: probes, sparse features, and patching converge; and
- negative controls: random, shuffled, norm-matched, and same-model patches.

Report counterexamples. If a pin/tactic feature is decodable but patching fails,
the conclusion is “represented but not shown causally used,” not “the model
reasons with the feature.”

The final synthesis ranks claims by evidence:

1. descriptive weight/activation change;
2. decodable information;
3. behavior-correlated feature;
4. causally mediating component;
5. validated multi-component circuit; and
6. algorithmic account that predicts new interventions.

## Statistics and anti-overfitting rules

- Split by game/root, never by individual square or future ply.
- Bootstrap games or roots, not the 64 correlated square tokens.
- Use paired estimators wherever positions are shared across models.
- Report effect sizes and intervals, not only p-values.
- Correct the frozen confirmatory family for multiple comparisons; exploratory
  heat maps are clearly labeled and do not receive confirmatory language.
- Probe hyperparameters, layers, concepts, and feature names are selected on
  development data.
- Run at least three seeds for learned probes/sparse models; more when seed
  variance is material.
- Keep engine labels and settings fixed across models.
- Publish failures, exclusions, nonfinite counts, and stopping reasons.
- A model-selected disagreement corpus never substitutes for unbiased heldout
  performance.

## Compute and storage design

The local machine has 16 GB system RAM and a 6 GB GTX 1660 Ti. Treat the WSL
workspace as a 30 GB device even when the mount reports more.

### Local machine

Use local compute for artifact checks, parameter diffing, dataset construction,
small corpus capture, attribution for individual positions, probe inference,
patching pilots, visualization, tests, and result inspection. Load models
sequentially for paired capture and keep batches at 1–4 unless measured memory
permits more. Use FP32 for compatibility claims and BF16 only as a recorded
sensitivity/performance mode.

The setup observed on 2026-08-07 exposes 7.7 GiB RAM and 2 GiB swap to WSL.
That is enough for batch-one tooling because the two-encoder lattice uses about
0.79 GB on the GPU, but local runs must use the guarded launcher: require 4 GiB
MemAvailable at launch, abort below 1.5 GiB, cap process-group RSS at 5 GiB,
and enforce 25 GiB usable inside the nominal 30 GiB workspace budget. The
workspace measured 7.59 GB after both source models and the Torch environment
were installed.

The final source-bound L14 LoRSA transfer belongs on the local machine: 512
sequential paired positions finished in 430.024 seconds internally (435.045
seconds through the guard), peaked at 1.230/1.260 GB CUDA allocated/reserved
and 2.443 GB process-group RSS, retained 2.7 MB, and incurred zero provider
cost. The exact-token audit finished in 112.818 seconds internally (116.542
guarded), peaked at 1.113/1.141 GB CUDA and 2.636 GB RSS, and retained 1.045
MB. The canonical 430-second transfer is slower than earlier diagnostic runs
because it shared the constrained host; use it as the reproducible wall-time
record, not as a device-throughput benchmark. Keep source-gated replacement,
CPU bootstraps, token semantics, and similarly bounded inference local.
Promote to Modal T4 when independent maps, larger corpora, richer retained
per-position summaries, or RAM/runtime isolation outweigh container startup.

A future WSL ceiling of 10–11 GB would leave the Windows host headroom and make
CPU-side analysis less brittle, but it is not a prerequisite for the first
pilot. GPU memory does not replace host memory: checkpoint loading, batches,
labels, streaming accumulators, and Python still consume RAM. Use swap only as
an OOM safety valve, not as an execution strategy.

All five FP32 hooks at all 15 layers consume 18.75 MiB per position per model.
Ten thousand positions would therefore be about 183 GiB for one model and
366 GiB for a pair. Raw activation dumps are forbidden as a default. Stream
batches into sufficient statistics, bounded sketches, top examples, and small
audited samples; then delete ephemeral tensors.

### Modal

Modal is the default remote backend for bursty, restartable maps. The verified
pipeline deliberately separates network transfer from GPU work:

1. a CPU-only function downloads the immutable Railway bundle, verifies all
   seven files and 1,053,370,877 bytes with SHA-256, commits them to the
   `chess-dfm-interpretability-inputs` Volume, and records a completion marker;
2. the T4 function starts only after the CPU cache/path preflight succeeds,
   reads the cached model/corpus files during ordinary model loading, and logs
   `network_download_bytes: 0`;
3. the GPU commits its checksummed result to
   `chess-dfm-interpretability-results`; and
4. a separate CPU-only function publishes that result to Railway.

Thus Railway download time does **not** accrue GPU charges. GPU billing begins
when the T4 container starts and includes cached-file reads, model loading,
inference, probe fitting, validation, and result commit. Modal Volumes are a
write-once/read-many cache, not a sole artifact copy; promoted results also live
in Railway and locally.

Verified identities and outputs:

| Object | Identity/result |
| --- | --- |
| Workspace/environment/app | `liquidgunay`; `chess-dfm-research`; `chess-dfm-interpretability` |
| Input bundle | `8be17957ad779e3756092306daa95ad9c1c7f5d482b6409bda7a16bc6d6c7a44`; full CPU hash verification; Volume cache hit before GPU |
| Smoke result | run `sha256:7c0099ed94d1121f8d9cbd35fa6beb2aa491d323574f659cb8c51f4271dcff4c`; Railway bundle `53ade980bbc2edd0747d4f7932c6626ceafed8636c7d1d783a720276d193a1e9`; 1,330,753 bytes |
| All-layer result | run `sha256:8c9ce9ee97af6f6753688a36a1997994f8803fe358578edf07be7b356b22d963`; Railway bundle `226c215e2fe63eddd7c14a36046235497172eaa9e4f06053d75794704ba1b096`; 74,753,320 bytes |
| Powered GPU envelope | Tesla T4; 1,629.42 seconds; peak allocated/reserved 1.406/1.474 GB of 15.637 GB; Torch `2.5.1+cu121`, CUDA 12.1 |
| Billing after collection | `$0.46430113` metered, `$0` billed after credits, `$0` Volume charge; powered pipeline increment `$0.44357680` |

The LoRSA GPU comparison used an identical 8-position source-gated smoke on
both devices. Retained internal workload time was 25.6692 seconds on T4 (run
`sha256:b73e15c11a6ac1f1788e80b0d970bcdef4f8dcc0683c2e13990a1e7723ac4e9a`)
and 22.8506 seconds on L4 (run
`sha256:0f2e4160cb375a295331be5a17474927070600a298de7aabbb48493ce9d690b0`);
both peaked at 1.230/1.260 GB allocated/reserved. At the frozen combined
4-core/24-GiB rates, multiplying only those internal times gives estimated
workload costs of about `$0.00692248` on T4 and `$0.00748767` on L4. These are
planning estimates, not provider-attributed attempt bills. Exact attempt-level
attribution was unavailable. The observed Modal session increment was
`$0.03017528`; it includes control-plane work, CPU staging/publishing, a failed
v1 T4 attempt (`$0.00889735` exact increment), and both successful smokes.

This tiny workload is dominated by fixed startup/model-loading overhead, and
the remote smokes contain 8 positions versus 512 locally, so elapsed times are
not a throughput comparison. Use local compute for this bounded inference
path. Prefer T4 over L4 for heavier remote maps unless a measured job-specific
L4 speedup exceeds its higher combined rate; use Modal for isolation,
parallelism, and scale-to-zero rather than merely because a GPU is remote.

The powered pipeline used about 1.04% of the `$42.50` workspace budget; all
metered activity so far uses about 1.09%. The retained cost audit is
`research/analysis/raw_hero_interpretability_report_v1/cost_audit.json`.
Conservative declared-attempt bounds were `$0.073668` for CPU input staging,
`$2.184408` for the all-layer T4 stage, and `$0.021048` for CPU publishing.
Actual cost is lower because declared timeouts are safety ceilings.

The reliability audit retains failures instead of silently discarding them:

- an unsupported `ephemeral_disk` override was rejected locally, before remote
  GPU work;
- two wrong logical raw-model paths ended in under one second of model work
  each and led to a corrected `models/raw/BT4_exported.pb.gz` contract;
- one 21.7-second scientific smoke finished computation but lacked the Git
  binary for optional provenance; and
- the subsequent smoke and powered run passed, were pulled locally, checksum
  verified, independently recomputed, and published.

Each function declares resources, timeouts, retries, `max_containers=1`, and
single-use containers. Immediately before every stage, the local entrypoint
reads `modal billing summary`, applies the frozen price snapshot, and refuses
an unknown price, unreadable bill, per-stage cap breach, or projected monthly
spend above `$34.00`. Modal may reschedule a crashed container independently of
application retries, so the provider-native `$42.50` budget remains the hard
outer cap and the final 20% remains reserved.

Verified commands:

```bash
modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode stage-input

modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode probe-smoke

modal run --env chess-dfm-research \
  research/interpretability/modal_app.py --mode probe-all-layers
```

The full modes run CPU stage → GPU compute → CPU publish automatically. Reusing
an already verified bundle is idempotent: the CPU stage re-hashes the cached
files and returns a cache hit rather than downloading again.

### Runpod

Use a rented Runpod instance when a job needs a warm process, local NVMe, or
many contiguous GPU-hours: full sparse-model training, large crosscoders, or
interactive debugging after a Modal task repeatedly fails. Provision from one
script, sync only required inputs, write heartbeats/checkpoints, and terminate
automatically on completion or stall.

Choose Runpod only when estimated useful GPU time plus setup is cheaper than
the equivalent serverless run, or when statefulness materially reduces risk.

### Storage tiers and budget

- Local disk: current code, the raw source artifacts, Hero checkpoint, compact
  corpora, active results, and at most one learned-artifact working set.
- Railway S3-compatible storage: active remote manifests, selected checkpoints,
  compact statistics, and results that Modal/Runpod must exchange.
- Google Drive: cold, human-controlled source backups and completed bundles.
- Provider-local volumes: caches and resumable scratch only; never the sole
  copy of a promoted artifact.

Railway resources are isolated in project `chess-dfm-research`
(`05efeaf8-614b-4944-ad1f-a7c317dda351`), production environment
`efc03f33-0b4f-4ad2-8608-83f82cd807b2`, and private bucket
`chess-dfm-artifacts` (`f456f511-c716-4032-98fb-4fc41c1b0e40`, region `iad`).
The immutable store retains the earlier pilot bundles plus the current
1,053,370,877-byte input bundle, 1,330,753-byte smoke result, and
74,753,320-byte all-layer result. Query a fresh inventory before forecasting
aggregate storage because content-addressed historical bundles remain billable.
Railway is an exchange and cold-result tier, not a scratch filesystem or
database.

The current monthly compute ceiling is $42.50. Allocate $25.50 to planned GPU
work, $8.50 to the highest-value follow-up, and leave $8.50 unallocated for
provider reschedules, failed starts, storage, billing lag, and price drift.
The local guard refuses a new launch when recorded spend plus its
declared-attempt estimate would exceed $34.00. Modal's native $42.50 workspace
budget remains the hard outer cap. Every remote job also has a per-job cap,
wall-clock and startup timeouts, bounded retries, a kill condition, and a
content-addressed output contract. The machine-readable policy is
`research/infra/modal_budget_policy.json`.

Keep the local durable working set below 20 GB so temporary downloads and
package operations cannot breach 30 GB. Prune package/download caches after
environment verification, delete reconstructed archives after hash-verified
extraction, and never duplicate encoder weights.

## Reproducible artifact contract

New code lives under `research/interpretability/` as small composable modules:

```text
research/interpretability/
  artifacts.py                 atomic files, identities, checksum ledgers
  artifact_store.py            immutable Railway S3 bundle push/pull
  models.py                    RR/HR/RH/HH assembly and strict loading
  accumulators.py              bounded moments, sketches, bootstrap
  parameter_diff.py
  spectral_diff.py
  activation_diff.py
  behavior_diff.py
  logit_lens.py
  attention.py
  attribution.py
  patching.py
  paired_diff.py
  chess_concepts.py
  chessbench.py                 public Searchless/ChessBench download ABI
  pilot_corpus.py               frozen and public corpus builders
  probes.py
  lookahead.py
  probe_pilot.py               nested-split probes plus causal steering
  literature_pilot.py          local literature-method integration
  validate_probe_result.py     independent retained-result recomputation
  build_interpretability_report.py
  remote_cost.py               fail-closed Modal launch accounting
  modal_app.py                 CPU stage → GPU run → CPU publish
  sparse/
    transcoder.py              published transcoder loader/runner
    transfer_pilot.py
    lorsa.py                   LoRSA-compatible sparse replacement module
    delta_crosscoder.py        paired model-difference sparse module
```

Every run directory contains:

```text
manifest.json        inputs, hashes, code, environment, hypotheses, seeds
metrics.json         final machine-readable metrics
examples.jsonl       bounded qualitative examples with stable position IDs
run.log              progress, warnings, exclusions, timings
checksums.sha256     every retained payload
```

Large arrays use safetensors, NPZ, or Parquet according to structure, never
pickle. Outputs are written to a temporary name, validated, hashed, then
atomically promoted. A run is complete only when the manifest can reproduce
the metric from retained inputs without an undocumented local path.

## Execution order and gates

1. **Foundation — complete:** restored artifacts, Torch-only environment,
   strict loaders, capture/intervention tests, and policy parity.
2. **Diff pilot — complete:** aligned parameter/spectral diff plus the
   128-position RR/HR/RH/HH activation/behavior run.
3. **Diff confirmatory — planned:** preregister metrics, use true game-disjoint
   clusters, and run the frozen heldout family with a converged sketch or exact
   selected-layer statistics.
4. **Literature replication — exploratory pass complete:** logit lens,
   attention/SmolGen decomposition, SARFA, multiple gradient attributions,
   probes, future-move decoding, and controls. IG remains a recorded negative
   diagnostic.
5. **Causal localization — residual/probe-direction pilots complete:**
   bidirectional whole-residual patching and probe steering work; component,
   path, and matched-counterfactual localization remain.
6. **Sparse transfer, semantics, and causal validation — development pass complete:**
   the transcoder and trusted LoRSA checkpoints pass Raw source gates and
   frozen Hero transfer evaluation. The source-bound 512-root transfer, paired
   uncertainty analysis, compact censored screen, exact-token audit, and strict
   256-root matched-direction causal v2 run are retained. Matched sparse
   training and confirmatory evaluation remain.
7. **Novel paired methods — interfaces complete:** paired-delta covariance,
   LoRSA, and delta-crosscoder modules/tests exist; selected-layer training has
   not been run.
8. **Synthesis — development report and notebook complete:** independent
   recomputation, evidence ladder, cost audit, portable technical report, and
   pedagogical marimo notebook are retained. Confirmatory reporting waits for
   the one-time test opening.

Each later launch is gated by the evidence before it. A negative result cancels
or redirects dependent work rather than being converted into a story.

## Immediate implementation checklist

- [x] Restore and hash-verify raw BT4 and terminal Hero artifacts.
- [x] Create the project-local Torch 2.5.1/CUDA 12.1 environment with no JAX
  runtime.
- [x] Add strict RR/HR/RH/HH assembly, captures, interventions, and parity
  tests without on-disk model duplication.
- [x] Build immutable pilot and public-v2 corpora with stable IDs, stored plane
  bytes, nested splits, and integrity digests.
- [x] Run aligned parameter/spectral and 128-position activation/behavior
  model diffing; freeze expensive layers.
- [x] Implement logit lens, attention decomposition, SARFA/gradient
  attribution, paired delta analysis, and bidirectional residual patching.
- [x] Implement concept probes, third-ply lookahead probes, controls, causal
  steering, and saved-prediction artifacts.
- [x] Load and gate the published L14 transcoder on both models.
- [x] Convert, strict-load, source-gate, and frozen-transfer the trusted
  published L14 LoRSA checkpoint on 512 unique development groups; retain the
  20,000-replicate paired analysis and T4/L4 smoke comparison.
- [x] Run the group-disjoint exact-token semantic audit with an opportunity
  gate; retain positive associations, Raw nonreplications, Hero nontransfers,
  cross-model sign disagreements, and the invalid-v1 selector diagnostic.
- [x] Run strict LoRSA ablation/insertion dose curves with scalar/token/norm-
  matched random decoder directions on Raw and Hero; retain three bilateral
  positives, the 6661 and 5851 negatives, and the underpowered 4516 result.
- [x] Audit the Hero encoder schedule, implement FP32-master/WSD training,
  complete the matched Phase-1 screen, and retain the fixed-teacher candidate
  with its failed JEPA gate and development-only arena scope.
- [x] Build the compact peak-report adapter and label its selected support as
  censored discovery evidence rather than an inferential replication rate.
- [x] Add tested LoRSA and delta-crosscoder building blocks.
- [x] Verify Railway content-addressed exchange and Modal cost guards.
- [x] Split the Modal pipeline into CPU input staging, T4 compute, and CPU
  result publishing; verify zero network-download bytes in the GPU stage.
- [x] Run the all-15-layer development experiment and publish/pull its
  immutable Railway bundle.
- [x] Publish and byte-exactly round-trip the final LoRSA transfer, bootstrap,
  compact, exact-token, and 51-file source-snapshot bundles through Railway.
- [x] Independently reconstruct splits and recompute all 120 saved metric
  groups; verify safetensors-only probe weights and zero reserved test rows
  selected or evaluated.
- [x] Generate the source-linked synthesis, cost audit, and portable report.
- [x] Build and visually QA the reactive Raw/Hero interpretability notebook,
  including widget states, final ledgers, exact examples, and claim gates.
- [ ] Run the compact Raw BT4/matched-Hero1-Phase1/candidate model diff.
- [ ] Extend the no-update objective-gradient audit for fixed-teacher KL and
  preregister exactly one representation-balance arm.
- [ ] Freeze a content-bound, non-overlapping Phase-1 arena slice before
  inspecting it; the first 128 development pairs are spent.
- [ ] Continue only a balance arm that passes both representation and behavior
  gates, then compare equivalent-boundary and terminal artifacts.
- [ ] Freeze a three-seed, multiplicity-corrected confirmatory protocol and
  open the test partition once.
- [ ] Build a licensed, versioned Stockfish-PV benchmark with stored engine
  settings and matched tactical/quiet roots.
- [ ] Run component/path patching and matched counterfactual interventions at
  validated high-change layers.
- [ ] Decide whether matched LoRSA or paired delta-crosscoder training is worth
  remote spend after component/path localization of the three causal features.

## Broader next steps after causal v2

The fixed-teacher result earns a carefully gated continuation, not an immediate
full-epoch run. The next sequence is:

1. **Run a compact three-way Phase-1 model diff.** Compare Raw BT4, the matched
   554-update Hero1 control, and the fixed-teacher candidate at identical hooks
   and frozen positions. Attribute parameter movement by family, repeat the
   inexpensive RR/HR/RH/HH swaps, and measure representation change at the
   already selected layers. Stream aggregates and discard activations. This
   answers what changed at the fair Phase-1 boundary without pretending that
   the candidate has already replaced terminal Hero1.
2. **Measure objective conflict before changing training.** Extend the
   no-update gradient audit to report norm, cosine, and groupwise contribution
   for immutable-teacher KL versus JEPA, SIGReg, WDL, ordinary policy, and DFM
   losses. If conflict is broadly scalar, preregister one lower or decaying
   teacher coefficient. If it is concentrated in the inherited trunk,
   preregister one teacher-gradient scaling or routing arm. Do not run both and
   do not convert this into a coefficient sweep.
3. **Confirm Phase-1 behavior independently.** Freeze a content-bound,
   non-overlapping paired-opening slice before inspection. Train the single
   selected balance arm to the same 554-update boundary with the same ordered
   data and compare it with the exact Hero1 Phase-1 control and the retained
   candidate. Require the new representation guardrail, legal-masked root
   metrics, zero arena faults, complete move-codec coverage, and a
   preregistered pair-aware strength rule.
4. **Earn the longer WSD run.** Only a balance arm that passes both
   representation and behavior gates proceeds through the stable phase and
   cooldown. Preserve restartable recovery checkpoints, validate at
   predeclared boundaries, and compare like-for-like example counts against
   Hero1 before the final direct arena against terminal one-epoch Hero1. A
   Phase-1 win alone cannot name a terminal Hero2.
5. **Repeat the full interpretability stack only on the winner.** Keep Raw and
   terminal Hero1 immutable. Compare Raw/terminal-Hero1/Hero2 with model diffing
   first, RR/HR/RH/HH swaps, representation similarity, probes, published LoRSA
   transfer, exact-token semantics, sparse causal interventions, attribution,
   and component/path patching. The compact Phase-1 diff from step 1 is
   exploratory context, not a substitute for this matched terminal analysis.
6. **Tie mechanisms to chess quality.** Deepen the validated LoRSA features
   10843, 5429, and 10784 through attention/MLP/policy paths and matched board
   counterfactuals. Add the pinned Stockfish-PV benchmark and comparisons with
   public searchless-chess baselines only after model selection, so benchmark
   work does not steer the training choice.
7. **Keep compute lifecycle-specific.** The local 1660 Ti owns tests, compact
   inference, gradient-audit development, and lightweight causal work. Modal
   owns restartable L40S training, validation, and arenas; CPU functions stage
   inputs and publish outputs so GPU billing starts only for GPU work. Railway
   stores content-addressed compact bundles and manifests, not an unbounded
   checkpoint history. Consider Runpod only for a measured, contiguous long
   winner run when its utilization advantage exceeds orchestration overhead.
   Preserve the $8.50 reserve and keep the repository new-launch guard at
   $34.00 against the $42.50 workspace budget.
8. **Spend the test split once.** After the terminal winner, analysis code, and
   hypotheses are frozen, run the three-seed, game-clustered,
   multiplicity-corrected confirmation and the licensed engine-PV benchmark.

The practical priority is therefore: compact Phase-1 diff -> fixed-teacher
gradient audit -> one preregistered balance arm -> non-overlapping Phase-1
confirmation -> longer WSD continuation -> terminal Hero1 arena -> full
Raw/Hero1/Hero2 interpretation -> one-time test.

## Inputs needed later, not blockers now

No additional authentication is needed for local, Railway, or Modal work.
Runpod credentials are needed only if a stateful sparse job wins a fresh cost
gate. Before confirmatory work, the scientific choices that require approval
are the exact frozen test protocol and whether to derive the engine-PV corpus
from already retained/public Searchless/ChessBench roots or another licensed
source. Stockfish itself can be pinned and run remotely; it need not live on
the constrained local machine. The corpus manifest must retain engine version,
nodes/time, threads, hash, MultiPV, line depth, adjudication/confidence, source
license, and root/game grouping.

## Primary literature and implementation references

- [McGrath et al., concept discovery and causal analysis in AlphaZero](https://pmc.ncbi.nlm.nih.gov/articles/PMC9704706/)
- [SARFA action-specific saliency](https://arxiv.org/abs/1912.12191)
- [Integrated-gradient analysis of chess representations](https://www.aiml.informatik.tu-darmstadt.de/papers/czech24representation.pdf)
- [Information-interaction maps for chess models](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364554/)
- [Karvonen, board-state representations and interventions](https://arxiv.org/abs/2403.15498)
- [Iterative inference in chess-playing transformers](https://arxiv.org/abs/2508.21380)
- [Jenner et al., evidence of lookahead in chess models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/37d9f19150fce07bced2a81fc87d47a6-Abstract-Conference.html)
- [Cruz et al., dynamic chess concepts](https://openreview.net/forum?id=np4Bg2zIxL)
- [Contrastive sparse autoencoders](https://arxiv.org/abs/2406.04028)
- [Sparse feature analysis in board-game models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9736acf007760cc2b47948ae3cf06274-Abstract-Conference.html)
- [Tracing the Thought of a Grandmaster-level Chess-Playing Transformer](https://arxiv.org/html/2604.10158)
- [Sparse Crosscoders for model comparison](https://transformer-circuits.pub/2024/crosscoders/index.html)
- [Delta-Crosscoder](https://arxiv.org/abs/2603.04426)
- [Causal caveats for chess-feature interpretation](https://openreview.net/forum?id=sbt3pBy9Rx)
