# Full-loop 1,024-update ablation

## Decision

**Retain the architecture, reject this exact training composition for a Hero run.** The predicted-JEPA loop makes recurrent inference genuinely useful, but the final-only objective materially damages the one-pass policy. The exact preregistered mechanism-success rule therefore fails.

The two arms used the same Raw BT4 initialization, fixed Hero1 teacher, ordered 1,048,576-example prefix, WSD schedule, frozen validation pool, and optimizer recipe. The only scientific changes were enabling `predicted_jepa_tokens` and adding its 262,144-parameter rollout adapter.

## Arena evidence

All results use 128 color-reversed opening pairs (256 games), opening indices 256-383, deterministic greedy legal argmax, zero faults, zero cap draws, and complete move-codec coverage.

| Comparison | Score | Pair-aware 95% interval | Relative Elo | Pentanomial |
|---|---:|---:|---:|---:|
| Candidate p8 vs candidate p1 | 0.7461 | [0.6261, 0.8661] | 187.2 | `[0, 6, 25, 62, 35]` |
| Candidate p8 vs control p1 | 0.5117 | [0.3917, 0.6318] | 8.1 | `[6, 30, 50, 36, 6]` |
| Candidate p1 vs control p1 | 0.2578 | [0.1378, 0.3779] | -183.7 | `[34, 62, 26, 6, 0]` |
| Candidate p8 vs control p8 | 0.8145 | [0.6944, 0.9345] | 257.0 | `[0, 0, 20, 55, 53]` |
| Control p1 vs control p8 (exploratory) | 0.7969 | [0.6768, 0.9169] | 237.5 | `[0, 0, 22, 60, 46]` |

The recurrence effect is large: candidate p8 scores 0.7461 against its own p1 exit and 0.8145 against control p8. Candidate p8 recovers to point-score parity with control p1 (0.5117). In contrast, candidate p1 scores 0.2578 against control p1, with its entire interval below the preregistered 0.45 material-deficit boundary. The exploratory control comparison shows that plain iterative unmasking hurts: control p1 scores 0.7969 against control p8.

## Frozen offline evidence

Terminal validation used the same 8,192 positions for both arms.

| Metric | Control | Candidate | Candidate - control | Direction |
|---|---:|---:|---:|---|
| `dfm_ce_loss` | 5.509959 | 5.496045 | -0.013915 | lower |
| `root_legal_conditional_ce` | 1.462899 | 1.468088 | 0.005189 | lower |
| `root_legal_top1_accuracy` | 0.503174 | 0.503784 | 0.000610 | higher |
| `wdl_loss` | 0.815435 | 0.775929 | -0.039505 | lower |
| `wdl_accuracy` | 0.617935 | 0.623962 | 0.006027 | higher |
| `wdl_brier_score` | 0.484850 | 0.461911 | -0.022938 | lower |
| `wdl_ece_15` | 0.104832 | 0.099988 | -0.004844 | lower |
| `jepa_positive_loss` | 0.159244 | 0.136146 | -0.023098 | lower |

The candidate's first planner has DFM CE 5.865612; the JEPA-conditioned second planner reaches 5.496045, an improvement of 0.369568. JEPA itself is healthy: all eight horizons beat zero, identity, shuffled-state, and shuffled-action baselines; RMS ratios stay inside [0.5, 2.0], and no effective-rank collapse is present.

## Preregistered gates

| Gate | Result |
|---|---|
| Candidate p8 beats candidate p1 | PASS |
| Candidate p8 recovers to control p1 | PASS |
| Candidate p1 is not materially damaged | FAIL |
| Offline latent/systems/arena health | PASS |

Mechanism success is **FAIL** solely because the one-pass preservation gate fails decisively. This is not a representation-collapse result.

## Parameter movement and systems

The loop adapter changed in every scalar and moved 0.5070 relative L2 (cosine 0.8897). Candidate trunk movement was 0.006643 versus 0.006815 for control, so the behavioral result is not explained by runaway encoder drift.

Both 1,024-update runs completed all planned recoveries with zero nonfinite or skipped updates. The preregistered diagnostic passed immutable-teacher, loop activity, throughput, memory-headroom, and no-checkpoint checks.

## Cost

Metered spend rose from $21.11 to $27.11: $6.00 for the full experiment. Arenas plus movement audits cost $0.62. The final billed value was $0.00; workers are stopped. $6.89 remains before the repository's conservative new-launch stop.

## Next bounded experiment

Keep the loop, but train the first exit explicitly. The primary follow-up is deep supervision: apply the fixed-Hero1 teacher and legal-root/DFM losses to both the preliminary one-pass logits and the final JEPA-conditioned logits. If budget permits one additional arm, combine that with pass dropout or a fixed mixture of one-pass and loop-conditioned batches.

At 1,024 updates, first require candidate p1 noninferiority to control p1 and candidate p8 superiority to its own p1. Only a recipe satisfying both should receive a longer scaling preregistration. This targets the observed failure directly while preserving the mechanism that worked.

## Scope and limitations

- All play is deterministic greedy policy play, not search-based Elo.
- The arena tier is development-only and relative Elo is checkpoint-pool relative.
- The four primary comparisons reuse the same held-out opening slice; they are matched contrasts, not independent replications.
- The ablation covers 1,048,576 examples, far short of the full Hero horizon.
- The exploratory fifth arena was preregistered separately and cannot rescue the primary promotion rule.

The machine-readable companion contains every input SHA-256, curve bootstrap, latent horizon, arena block checksum, exact gate evaluation, and cost ledger.
