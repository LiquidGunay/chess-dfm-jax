# Vision

`chess-dfm-jax` is moving toward latent state-action sequence planning on top of a frozen LC0 BT4 encoder.

Core direction:

- BT4 encodes the current chess state into square tokens.
- DFM proposes action chunks as the fast action prior.
- JEPA predicts latent future trajectories for those action chunks.
- Value, WDL, and later concept heads score and interpret predicted futures.
- Exact rollout comparison is the research loop: propose, predict, score, and compare against real future boards.

Near-term intent:

- standardize trajectory-v2 shards as the canonical training contract
- keep DFM as the action-only baseline
- upgrade JEPA from terminal-only prediction to sequence prediction
- add notebook-first inspection so data quality and training status stay visible
- preserve public-repo hygiene by keeping operational cloud details in ignored local config
