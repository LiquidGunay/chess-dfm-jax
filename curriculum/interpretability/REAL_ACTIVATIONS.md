# Real activation gallery

The course uses controlled toy systems when a known mechanism is required,
but the default learning path now begins with checkpoint computations. The
gallery is a compact bridge between a roughly one-gigabyte pair of source
models and a blog that can expose actual internal values without loading a
model on every interaction.

## Exact specimen

The frozen gallery contains the eight lowest position IDs in the immutable
public-v2 development split. For each position it stores:

- Raw BT4 and Hero `resid_post_after_ln` activations at all 15 layers, with
  shape `[8 positions, 15 layers, 64 square tokens, 1024 channels]` per model;
- RR, HR, RH, and HH native nonlinear policy-head logits at every layer;
- legal action IDs, UCI moves, recorded target actions, FENs, themes, and
  rule-derived square labels;
- predictions from the retained Raw/Hero all-layer probes for legal
  destination, opponent attack count, and 13-way piece code;
- native-policy target-log-odds vector–Jacobian-product summaries at layers
  0, 7, and 14; and
- exact finite-gallery centered singular values at layers 0, 7, and 14.

Residuals and lens logits were computed in float32 and stored in float16.
Gradient and singular-value summaries are stored in float32. The deterministic
NPZ is 30,830,817 bytes and has SHA-256
`49f6f35271acaaaae052191f9a758695cb545ec6e9c20d3eb2c52a8cf02a5860`.
The current run ID is
`sha256:ae6d4eceadc00811d51aa738c7eafd30c5a4e406ece4feedd08123403fbdea04`.

The manifest records array shape/dtype ABIs, source paths and hashes, model
identities, selection, endpoint definitions, final policy-lens tolerance,
runtime environment, and claim boundaries. Loading fails closed on manifest,
data, array, or optional source drift.

## Where learners touch the tensors

- Module 01 selects one position, layer, model, and absolute square; it prints
  the largest coordinates of the actual 1,024-value residual row.
- Module 03 compares Raw and Hero channel values, whole-vector square RMS,
  token cosine, and layerwise delta trajectories.
- Module 04 applies the retained trained probes to the selected activation and
  displays predictions, rule-derived targets, and errors on the board.
- Module 05 displays one real endpoint-specific residual VJP and
  gradient-times-state map before teaching attribution failure modes.
- Module 07 connects the superposition essay to real centered activation
  spectra while refusing to treat effective rank as a feature count.
- Module 11 shows position-level four-arm policy trajectories, legal mass,
  actual top UCI moves, and per-square reductions of one full real endpoint VJP.

Each lesson follows the same evidence rhythm: inspect a real specimen, use a
toy only when ground truth matters, then return to the aggregate retained
experiment and its claim card.

## Rebuild locally

The extraction is deliberately suitable for the local GTX 1660 Ti. The
recorded eight-position build took roughly 19 seconds. It does not use Modal,
RunPod, or a rented GPU.

```bash
.venv/bin/python -m research.interpretability.course.build_activation_gallery \
  --device cuda --count 8
.venv/bin/python -m research.interpretability.course.build_snapshot
```

The source checkpoints and trained probe weights are read locally and never
copied into the gallery. Rebuilding after a code, checkpoint, probe, corpus,
or selection change produces a new identity. Do not edit the NPZ or manifest
by hand.

The NPZ is deliberately gitignored because this repository is public and the
retained record does not yet establish permission to redistribute derived
checkpoint tensors. Do not force-add, stage, push, or attach it to a release.
The tracked manifest records the expected identity; snapshot notebooks return
an explicit absence card when the local pack is missing. Source mode remains
strict about every materialized source. In a clean checkout, deterministic
snapshot construction recovers the expected tensor SHA-256 and byte count from
the integrity-bound manifest, and the validator reports that activation bytes
and gallery sources were not inspected. When the pack is local, the same path
performs full tensor ABI, checksum, and all-source verification.

## Scientific boundaries

- Eight development positions support inspection, not a population estimate.
- A large coordinate or Raw/Hero delta is not a demonstrated chess feature.
- A trained-probe prediction is decodability, not policy use.
- A target-action VJP is local sensitivity, not an intervention.
- The native-head depth trace is a policy/logit-lens diagnostic, not a fitted
  corpus-average J-Lens.
- Participation ratio or entropy effective rank is not evidence of
  superposition, a feature count, or monosemanticity.

## Redistribution boundary

The tensor pack contains no checkpoint or bulk corpus, but it is derived from
source model and position artifacts. Its manifest therefore marks it
`internal_derived_tensor_release_review_required`. A standalone public course
must complete a human review of the source-model and source-position terms,
in addition to resolving the repository's missing root license text. The
generated HTML includes selected rendered views, not the 30.8 MB NPZ itself.
Those real-derived rendered views are also release-gated and must not be pushed
to a public branch until the same human review explicitly clears them.
