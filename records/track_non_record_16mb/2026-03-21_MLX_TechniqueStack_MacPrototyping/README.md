# MLX Technique Stack — Mac Prototyping Harness

**val_bpb: 1.9588** (Mac, 14L×416d, 750 steps, 10 shards, int8+zlib, full FineWeb val)

## Status

Non-record submission — Mac MLX prototyping only, pending H100 validation.
Submitting to document systematic technique exploration and support compute grant application.

## Approach

Systematic validation of leaderboard techniques through 25+ MLX experiments,
identifying what works, what doesn't, and why.

### Implemented & Validated
- 10-14 layer architectures with KV2 (GQA)
- MLP 3x expansion (-0.013 BPB vs MLP 2x)
- Int6 per-row quantization + FP16 tied embedding passthrough
- Sliding window eval (stride-64, compiled forward)
- Muon decoupled weight decay (0.02)
- Overtone spectral embedding init (SVD power-law S_k ~ k^{-0.5})
- Phase-transition resid_mix initialization
- Multi-eval mode (EVAL_CONFIGS for multiple eval passes per run)
- VAL_MAX_TOKENS for fast directional experiments
- Compiled NTK-RoPE eval for sequence length extension

### Key Finding
FP16 embed + Muon WD achieves near-zero quantization gap (0.001 BPB).
Post-quant ≈ pre-quant, critical for maximizing quality within 16MB.

### Explored & Rejected (negative results)
| Technique | Result | Why |
|-----------|--------|-----|
| Depth recurrence + int6 | +0.61 BPB quant gap | Shared weights amplify quantization error through loops |
| DenseFormer DWA | +0.003 BPB regression | No benefit at this scale |
| Eval-time loop scaling | +1.15 BPB regression | Model calibrated to exact training loop count |
| NTK-RoPE extrapolation (1024->4096) | +0.06 BPB regression | Must train at target seq_len |

### Results (750 steps, 10 shards, full val)

| Config | Params | Compressed | Val BPB | Int8 BPB | Quant Gap |
|--------|--------|-----------|---------|----------|-----------|
| 14L×416d KV2 | 16.2M | 12.3MB | 1.9578 | 1.9588 | 0.0010 |
| 10L×512d + all tricks | 19.0M | 10.8MB | 1.9800 | 1.9808 | 0.0008 |

## Command

```bash
ITERATIONS=750 TRAIN_BATCH_TOKENS=8192 GRAD_ACCUM_STEPS=1 \
NUM_LAYERS=14 MODEL_DIM=416 NUM_HEADS=8 NUM_KV_HEADS=2 \
MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=32768 \
python3 train_gpt_mlx.py
```

## Next Steps (require H100)
- Port to train_gpt.py, train at seq2048
- Stack MLP3x + int6 + QAT + BigramHash + SmearGate + SWA
- Optimizer sweep + multi-seed validation
- Target: sub-1.14 BPB
