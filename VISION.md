# Vision

`chess-dfm-jax` is moving toward Latent-SASA planning on top of a frozen LC0
BT4 encoder.

Core direction:

- BT4 encodes chess states into square-token latents.
- DFM denoises explicit action chunks.
- JEPA predicts BT4-derived latent future trajectories for those chunks.
- JEPA consumes DFM action-token hidden states so future-consequence losses
  shape the planner, not just a separate action-ID-conditioned world model.
- Value, WDL, ranking, and later concept heads score predicted futures.
- Exact rollout comparison is the research loop: propose, predict, score, and
  compare against real future boards.

Near-term intent:

- keep trajectory-v2 shards as the source-of-truth contract
- add compact, column-selective loader views for DFM, JEPA, and joint training
- keep action-only DFM and teacher-forced JEPA as baselines
- train joint Latent-SASA with a shared online latent basis, small DFM/JEPA
  adapters, shared action embeddings, and stop-gradient/EMA target latents
- follow `docs/implementation_plan_gold.md` for the research design and
  `docs/latent_sasa_coupling.md` for repo-grounded work items and guardrails
- run experiment queues on already-provisioned TPU workers instead of
  provisioning a new TPU for every small ablation
- preserve public-repo hygiene by keeping operational cloud details in ignored
  local config

Default training stance:

- BT4 remains frozen until joint heads beat action-only baselines on held-out
  action, legality, and latent-rollout metrics.
- Grain is a later loader backend option. The first implementation is a custom
  deterministic loader boundary because the compact schema and training views
  are still evolving.
- Checkpoints remain raw NumPy locally for now. The next checkpointing upgrade is
  asynchronous GCS upload of completed local checkpoints, not an immediate Orbax
  migration.
