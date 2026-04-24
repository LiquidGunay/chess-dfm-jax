# Phase 2: JEPA Deep Scaling Suite Progress Report

## 1. Current Status vs. Initial Plan

**Plan Objective:** 
Deploy a token-level JEPA head on top of a frozen LC0 BT4 encoder, training on Spot TPU pods via a resilient GCP orchestrator. Test deep scaling laws (layers) and multi-step prediction horizons (H=8).

**Current Status:**
- **TPU Orchestration:** The `sweep_manager.py` orchestrator has been thoroughly stress-tested against GCP quota limits, preemption events, and metadata caching. It successfully managed the multi-zone queuing of `v5litepod-16` and `v4-16` resources.
- **Architectural Fidelity:** We performed a rigorous audit of the NNX BT4 model to match the official `reference_bt4.py`. The `Smolgen` block was refactored into a per-head multi-stage generator inside the `EncoderLayer`, and the `Value` / `MovesLeft` heads were aligned with the proper nested projections.
- **Action-Conditioning Fixed:** Replaced a bug where the action token was appended to the sequence (causing a 65-token shape mismatch) with a proper latent conditioning approach (adding the action embedding to each of the 64 board tokens).
- **First Full Run Completed:** The V6 scaling run (`jepa-modern-l4-h8-v6`) successfully completed a 5000-step training loop on a `v4-16` pod with a batch size of 32, proving the multi-host JAX compilation and Orbax checkpoint-bypassing raw numpy persistence.

## 2. Analysis of the V6 Run (`jepa-modern-l4-h8-v6`)

The run successfully unrolled 8 steps into the future using the fixed `EncoderLayer` and `Smolgen` blocks.

**Metrics:**
- **Step 1 Loss:** `1.951`
- **Step 5000 Loss:** `0.000269`
- **Token Cosine Similarity:** `0.9999`
- **Step Time:** `~30ms` per step

**Takeaways:**
The model reached near-perfect token cosine similarity (`0.9999`) and a near-zero loss. However, this perfectly saturated convergence is characteristic of an easy task. The architecture scales and compiles efficiently across multiple TPU hosts, but the extreme stability indicates we need to look closely at the data pipeline.

## 3. Training Data Used

Initially, the models trained on synthetic (all-zero) data due to a missing sync directory bug. To correct this, we implemented a data download pipeline.
Because the official `storage.lczero.org` server was down (Cloudflare 522 Timeout), we briefly generated high-quality `.npz` chunks directly from the TCEC games archive. 

However, since the `storage.lczero.org` server came back online, we have completely removed the TCEC dataset and replaced it with **the official Lc0 self-play data chunks (`.gz` / `.zst` format)** from `training-run1`. 
The `LeelaChunkDataLoader` now automatically discovers and natively parses these binary records without conversion, successfully unpacking the 15 target floats (Q-value and WDL probabilities) encoded in each sample.

## 4. Next Steps & Mitigations

To prevent wasting TRC trial quota on failed compilations or synthetic fallbacks, we implemented the following mitigations:

1. **Pre-flight Assertions:** Added a strict `tests/test_training_step.py` script that initializes the exact configuration, constructs the new multi-task heads, the ActionMLP, the SigReg loss, and the `muon_adamw` optimizer, running a local forward/backward pass. The orchestrator's `start_sweep.sh` now executes this test and *aborts* the sweep if it fails.
2. **Cache-Busting Versioning:** Solved persistent GCP `startup-script` caching by enforcing immutable code snapshots (`verified_source_v12.tar.gz`) and distinct experiment IDs (e.g., `-v12`).
3. **Data Pipeline Hardening:** The training script now loudly crashes if no `.zst`, `.gz`, or `.npz` data is found, completely removing the silent synthetic fallback. 

The V12 suite is now actively provisioning on the TPU pods, training on genuine Lc0 self-play data with the verified architecture and multi-task loss.
