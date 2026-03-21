# Parameter Golf — Task Tracking

## Current Status: Ready for H100. Awaiting compute credits.

- PR #328 submitted (non-record, Mac MLX)
- Compute grant application submitted
- Both scripts (MLX + CUDA) have all techniques implemented
- CUDA script bug-fixed and verified (1339 lines)
- MLX script verified (1297 lines)

## Best MLX Results

| Config | Pre-quant BPB | Int8 BPB | Notes |
|--------|---------------|----------|-------|
| SG+BH+WD=0.02+LR=0.02 | **2.3164** | **2.3189** | Best 200-step recipe |
| 14L×416d KV2 (750 steps, full val) | 1.9578 | 1.9588 | Best absolute Mac result |

## Competition Leaderboard (as of 2026-03-21)

| Rank | BPB | Author | Key Techniques |
|---|---|---|---|
| 1 | **1.1254** | alertcat | TTT + XSA + EMA + SmearGate + BigramHash + Int6 QAT |
| 2 | 1.1320 | saml212 | Gradient-guided adaptive quant + 12L + LN Scale |
| 3 | 1.1428 | thwu1 | 10L + Int5 MLP + BigramHash(10240) + SWA(0.4) |

## Implemented Techniques (both scripts)

| Technique | MLX | CUDA | Validated on Mac |
|---|---|---|---|
| SmearGate | yes | yes | yes (-0.006 BPB) |
| BigramHash | yes | yes | yes (-0.006 BPB) |
| Sliding window eval | yes | yes | yes (marginal at seq1024) |
| Muon weight decay | yes | yes | yes (WD=0.02 best) |
| Overtone init | yes | yes | yes (hurts at 200 steps with WD+LR) |
| Phase resid_mix | yes | yes | yes (hurts at 200 steps with WD+LR) |
| OrthoInit | — | yes | — |
| FP16 embed | yes | yes | yes (near-zero quant gap) |
| Int5/Int6 quant | yes | yes | yes (works on baseline, breaks recurrence) |
| QAT with STE | — | yes | — |
| zstd-22 | — | yes | — |
| SWA | — | yes | — |
| Multi-eval | yes | — | yes |

## H100 Execution Plan

See `docs/H100_PLAN.md` for full 4-phase plan targeting sub-1.12 BPB.

## Key Lessons Learned

See `tasks/lessons.md` for full documentation. Highlights:
- Depth recurrence + int6 = catastrophic quant gap (0.61 BPB) — dead end
- FP16 embed + Muon WD = near-zero quant gap (0.001 BPB)
- WD=0.02 + LR=0.02 is best at short training
- OvertoneInit + PhaseResidMix conflict with WD+LR tuning at 200 steps
- SmearGate + BigramHash = -0.006 BPB (cheap, effective)
- Sliding window needs seq2048+ training to show big gains
