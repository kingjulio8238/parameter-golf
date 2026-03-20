# Parameter Golf — Task Tracking

## Current Status: MLX prototyping complete. Need H100 for remaining gains.

### Best MLX Result
**9×512 MLP3, int6, fp16_embed**: pre-quant **2.3087 BPB**, int6 **2.3460 BPB** (200 steps, 2M val)

### Competition Leaders (H100 validated)
- PR #135: **1.1539 BPB** (OrthoInit + Int6 MLP3x + BigramHash + SmearGate + seq2048 + sliding window)
- PR #128: **1.1594 BPB** (Int6 MLP3x + STE QAT + sliding window stride-64 + seq4096)
- PR #136: **1.2101 BPB** (just TRAIN_SEQ_LEN=2048 — single line change!)

## What We've Validated on MLX

### Proven Techniques (confirmed working)
- [x] MLP_MULT=3: -0.013 BPB improvement (2.3396 → 2.3262)
- [x] Int6 quantization: works on baseline (0.023-0.034 BPB gap), compresses 21.8M params to 3.9MB
- [x] FP16 embed passthrough: implemented and working
- [x] Compiled NTK forward: fixed so seq4096 eval is feasible on Mac

### Validated Negative Results
- [x] Depth recurrence + int6: catastrophic quant gap (0.61 BPB) — DEAD
- [x] DWA (DenseFormer weighted avg): slight regression at this scale
- [x] Loop embeddings: no benefit
- [x] Extra eval loops: catastrophic regression
- [x] NTK-RoPE extrapolation (train 1024 → eval 4096): degrades quality (+0.06 BPB)
- [x] Sliding window at native seq_len: negligible benefit (-0.0005)

### Cannot Validate on Mac (need H100)
- Train at seq_len=2048+ (batch too small on Mac: 2 seqs/step)
- Sliding window eval at seq2048+ (only works when model is trained at that length)
- QAT (needs longer training to be meaningful)
- Optimizer tuning (muon_momentum=0.99, lower LRs, warmdown=3000)

## Full Experiment Results

### Session 2: Architecture Experiments (10% val, 200 steps)

| Config | Params | Compressed | Pre-quant BPB | Post-quant BPB |
|--------|--------|-----------|---------------|----------------|
| E0: Baseline 9×512 | 17.1M | 9.8MB (int8) | 2.3396 | 2.3413 |
| E2: Recurrence 3×3 512 | 6.0M | 1.7MB (int8) | 2.3368 | 2.3399 |
| D2: Recurrence 4×3 768 | 17.3M | 7.4MB (int8) | 2.3117 | 2.3170 |

### Session 2: Proven Recipe Experiments (2M val, 200 steps)

| Config | Params | Compressed | Pre-quant BPB | Post-quant BPB |
|--------|--------|-----------|---------------|----------------|
| Baseline MLP3 int8 | 21.8M | 11.7MB | 2.3262 | 2.3281 |
| Baseline MLP3 int6+fp16e | 21.8M | 6.7MB | 2.3087 | 2.3460 |
| + stride-256 (seq1024) | — | — | — | 2.3455 |
| + NTK seq2048 stride-256 | — | — | — | 2.3666 (worse) |
| + NTK seq4096 stride-256 | — | — | — | 2.4069 (much worse) |

## Next Steps (require H100)

### Phase 2: H100 Validation
1. **Apply for compute grant** — we have validated results + clear technical plan
2. **Baseline + seq2048**: single-line change, PR #136 shows -0.014 BPB
3. **Full proven recipe**: MLP3 + int6 + fp16_embed + seq2048 + sliding window stride-64
4. **Novel additions**: SmearGate, Bigram Hash, Orthogonal init (from PR #135)
5. **QAT**: STE or Muon-Aware QAT (from PRs #128, #130)
6. **Optimizer tuning**: matrix_lr=0.02, muon_momentum=0.99, warmdown=3000
7. **zstd-22** compression (better than zlib-9 for int6)

### Target: sub-1.15 BPB
Current leader: 1.1539. Our stack (proven recipe + novel techniques) should be competitive.
