# Published BT4 LoRSA L14 checkpoint

This note records the private local conversion of the smallest published L14
LoRSA artifact selected for the Raw-versus-Hero transfer study.

## Frozen identity

- Hugging Face repository: `JacklE0niden/lc0-BT4-lorsa`
- repository snapshot: `1ec52be63cce41017b3853edbe9421b643a1476d`
- last commit touching these files: `fb30623060c46c6721bab2f792c6e08ea13f973b`
- checkpoint: `k_30_e_16/L14`
- upstream code: `JacklE0niden/Leela-SAEs` at
  `f946a5736f40397f5c834104aeac6e8e6ff753bd`
- hooks: `blocks.14.hook_attn_in -> blocks.14.hook_attn_out`
- architecture: 64 tokens, width 1,024, 128 Q/K heads of width 32,
  16,384 OV features, TopK 30, SmolGen, and a learned attention scale

The repository was public and ungated when inspected, but its model card did
not declare a license. Redistribution rights are therefore not established.
Keep the source and converted weights in private local or Railway experiment
storage; do not publish them as project artifacts.

The source directory is ignored:

```text
.local/artifacts/lorsa/upstream/JacklE0niden_lc0-BT4-lorsa/
  fb30623060c46c6721bab2f792c6e08ea13f973b/k_30_e_16/L14/
```

| Relative file | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,215 | `f465c27afc460ba8c741cec2ebc1b6cada9d6c297dd33615f2f5108553fefad1` |
| `sae_weights.dcp/.metadata` | 10,516 | `e61ba568b64ea291eeabb5614456bfd0053baf4aa50201b098cccf1360b55e59` |
| `sae_weights.dcp/__0_0.distcp` | 42,018,673 | `66959912d7a77006d9334e6ef2edea78d04ca41ad111f69fdcfd5ec1c5edcdf5` |
| `sae_weights.dcp/__1_0.distcp` | 44,126,229 | `985a6e938b442a6f5ca3750448ec4188f08596b5c368b1a5821437d447c144a5` |
| `sae_weights.dcp/__2_0.distcp` | 46,749,881 | `3fc3678ce65bd75193bcbf8c3466441d2e75cf989a07fec1453b0ec4e1bd00d3` |
| `sae_weights.dcp/__3_0.distcp` | 75,588,158 | `53e34b84428070f05c7098fe555cecdf90a996534216495a15804a6d269dc1ba` |

The four shard hashes equal the LFS object IDs advertised by the frozen HF
tree. Future acquisition should resolve URLs at the repository snapshot above,
not at floating `main`.

## Conversion boundary

The DCP contains pickle metadata. The user explicitly trusted the checkpoint,
so conversion is permitted only in a dedicated subprocess with the explicit
flag below. Normal experiment code loads only safetensors.

The conversion validates all 26 DCP entries. It retains the exact 20-tensor
inference state, including `_attn_scale_param` and `smolgen_score_scale`, and
rejects missing or unknown entries. The other six entries are validated
dataset-norm or runtime/training buffers.

The inference fold exactly matches pinned upstream
`standardize_parameters_of_dataset_norm`:

```text
c_in  = sqrt(1024) / input_norm
c_out = sqrt(1024) / output_norm
W_Q, W_K, W_V *= c_in
W_O, b_D       /= c_out
```

SmolGen weights are not folded. For this checkpoint, `input_norm` is
`3.4936718940734863`, `output_norm` is `84.97552490234375`, `c_in` is
`9.159417647170422`, and `c_out` is `0.37657902127436454`. The learned
attention scale is `5.656854152679443`; the SmolGen score scale is `1.0`.

```bash
uv venv --clear --system-site-packages \
  --python .venv/bin/python .local/venvs/lorsa-convert

PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ubuntu/chess-dfm-jax:/home/ubuntu/chess-dfm-jax/.venv/lib/python3.12/site-packages \
.local/venvs/lorsa-convert/bin/python -m \
  research.interpretability.sparse.convert_lorsa_checkpoint \
  .local/artifacts/lorsa/upstream/JacklE0niden_lc0-BT4-lorsa/fb30623060c46c6721bab2f792c6e08ea13f973b/k_30_e_16/L14 \
  .local/artifacts/lorsa/converted/JacklE0niden_lc0-BT4-lorsa/1ec52be63cce41017b3853edbe9421b643a1476d/k_30_e_16/L14_reviewed_v1 \
  --trust-paper-checkpoint
```

This command is CPU-only and launches no paid compute.

## Reviewed output

The private reviewed output is:

```text
.local/artifacts/lorsa/converted/JacklE0niden_lc0-BT4-lorsa/
  1ec52be63cce41017b3853edbe9421b643a1476d/k_30_e_16/L14_reviewed_v1/
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `lorsa.safetensors` | 208,249,832 | `2dc7c7cd90550bc3489e4940bb39746e8a3cdcfdc613ac73e9bb2e4bdfe5b8fd` |
| `parity.safetensors` | 6,816,224 | `1eb55a5f18493f0cebdb15075af72893a20aba661f37d35b7ce0c32e37117cb1` |
| `manifest.json` | 15,081 | `2fe2d2cf6a132e4a205b25373c50b74e4462caf44057ccb98c0e7e7a3f736b61` |

The parity fixture uses a deterministic complete `[1, 64, 1024]` sequence and
an independent transcription of the pinned upstream, normalization-folded
inference path. Against the lean module, sparse support is exactly identical;
maximum absolute errors are `2.88e-6` for patterns, `2.62e-6` for features,
and `3.81e-6` for reconstruction. The canonical experiment artifact gate
passes all criteria.

For an immutable exchange bundle, place these files at:

```text
converted/manifest.json
converted/lorsa.safetensors
converted/parity.safetensors
```

Verify an existing conversion without touching DCP:

```bash
.venv/bin/python -m research.interpretability.sparse.convert_lorsa_checkpoint \
  PATH/TO/converted unused --verify-only
```

Focused validation:

```bash
.venv/bin/ruff check \
  research/interpretability/sparse/lorsa.py \
  research/interpretability/sparse/convert_lorsa_checkpoint.py \
  tests/test_torch_lorsa.py tests/test_torch_lorsa_conversion.py

.venv/bin/python -m pytest \
  tests/test_torch_lorsa.py tests/test_torch_lorsa_conversion.py -q
```
