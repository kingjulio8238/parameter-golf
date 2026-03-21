# Parameter Golf — H100 Execution Plan

## Target: sub-1.12 BPB

Current SOTA: 1.1254 (alertcat, TTT + XSA + EMA + SmearGate + BigramHash(2048) + Int6 QAT + zstd-22)

## Top 5 Breakdown

| Rank | BPB | User | Key Differentiators |
|---|---|---|---|
| 1 | 1.1254 | alertcat | TTT + XSA + EMA + SmearGate + BigramHash(2048) + Int6 QAT + zstd-22 |
| 2 | 1.1320 | saml212 | Gradient-guided adaptive quant + 12L + LN Scale + Partial RoPE |
| 3 | 1.1428 | thwu1 | 10L, Int5 MLP / int6 attn, BigramHash(10240), SWA(0.4), WD=0.04 |
| 4 | 1.1458 | Raahil | 9L, SmearGate, BigramHash(4096), SWA(0.5), WD=0.04 |
| 5 | 1.1502 | aruniyer | 11L, int6 QAT, WD=0.04, zstd-22 |

## Consensus Techniques (all top 5 use)

- Sliding window eval stride=64
- FP16 tied embedding passthrough
- Muon momentum=0.99, warmup from 0.92 over 1500 steps
- Lower LRs: matrix=0.02, scalar=0.02, embed=0.03-0.04
- Warmdown=3000 iters
- Grad clip=0.3
- zstd-22 compression
- Int6 (minimum) quantization

## What Separates #1 from the Pack

1. **TTT (Test-Time Training)** — 3 epochs SGD on val tokens at eval time, ~+0.002 BPB free
2. **XSA (Cross-Sequence Attention)** — last 4 layers attend across sequences in the batch
3. **EMA** — exponential moving average of weights (decay=0.997) replaces SWA
4. **SmearGate + BigramHash(2048)** — smaller hash table than old #1 but combined with TTT/XSA/EMA

## What Separates #2 from #3–5

1. **Gradient-guided adaptive quantization** — sensitivity-aware mixed precision, saves ~1MB
2. **12 layers** — deepest submission on the board, funded by quant savings
3. **LN Scale** — zero-param RMSNorm scaling that stabilizes deep networks
4. **Partial RoPE** — rotary on only 25% of head dims, frees capacity for content

## Execution Phases

### Phase 1: Reproduce #4's recipe + new techniques (~3 runs, baseline)

```bash
NUM_LAYERS=9 MLP_MULT=3 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288
MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.04 WARMDOWN_ITERS=3000
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
GRAD_CLIP_NORM=0.3 FP16_EMBED=1 EVAL_STRIDE=64
USE_SMEARGATE=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128
EMA_DECAY=0.997
# + int6 + zstd-22 + OrthoInit
```

Target: ~1.146 with EMA instead of SWA, batch 524K for more steps.

### Phase 2: Push toward #2's recipe (~5 runs)

- Add gradient-guided adaptive quantization (int7/int6/int5 mixed)
- Use quant savings (~1MB) to fund 12th layer
- Add LN Scale (`1/sqrt(layer_idx+1)`) for 12L stability
- Add Partial RoPE (25% of head dims)
- Keep batch=524K, EMA=0.997

### Phase 3: Push toward #1 (~10 runs)

- Add XSA on last 4 layers
- Add TTT (3 epochs SGD, lr=0.002, freeze first 2 blocks)
- BigramHash sweep: 2048 vs 4096 vs 10240 (alertcat uses 2048 — smaller may be better with XSA)
- Muon-aware QAT (Gaussian noise mode from PR #130) — but NOT late QAT at 12L (known negative result)
- Explore WD sweep (0.02-0.06)
- Try 11L vs 12L tradeoff (12L needs LN Scale + adaptive quant)

### Phase 4: Submission (~5 runs)

- 3+ seeds on best config, p<0.01
- Target: sub-1.12 BPB

## Implementation Status

### Already implemented in train_gpt.py (CUDA):
- SmearGate, BigramHash, SWA, OrthoInit, OvertoneInit, PhaseResidMix
- Int5/Int6 quantization, QAT with STE, zstd-22, FP16 embed
- Sliding window eval, Muon WD
- Bug fixes: SWA float guard, rank-guarded quantization

### Still needed for H100 (implement on pod):
1. **EMA** — shadow weight copy with decay=0.997 (~15 lines, replaces SWA in Phase 1)
2. **TTT** — SGD fine-tuning on val tokens at eval time (~30 lines, Phase 3)
3. **XSA** — cross-sequence attention on last N layers (~25 lines, Phase 3)
4. **Gradient-guided adaptive quant** — per-tensor sensitivity ranking (~20 lines, Phase 2)
5. **LN Scale** — `1/sqrt(layer_idx+1)` in RMSNorm (~5 lines, Phase 2)
6. **Partial RoPE** — rotary on 25% of head dims (~10 lines, Phase 2)

## New Techniques from Latest Leaderboard

### TTT (Test-Time Training)
3 epochs SGD on val tokens during eval, lr=0.002, freeze first 2 blocks. Runs in ~47s on 8xH100.
Effectively free BPB improvement (~+0.002 BPB) since it runs at eval time, not training time.
The model fine-tunes itself on the validation distribution before being scored.

### Gradient-Guided Adaptive Quantization
Accumulate squared gradients during the last 10% of warmdown. Rank tensors by sensitivity:
- Top 10% (highest grad magnitude) → int7
- Middle 70% → int6
- Bottom 20% (least sensitive) → int5

Saves ~1MB vs uniform int6 → enough budget to fund a 12th transformer layer.

### LN Scale
RMSNorm output scaled by `1/sqrt(layer_idx + 1)`. Damps activations in deeper layers.
Zero additional parameters. Stabilizes training for 12L models that otherwise diverge.

### Partial RoPE
Apply rotary position embeddings on only 25% of head dimensions (16 of 64).
Remaining 75% are position-free, acting as content-only dimensions.
Reduces RoPE's interference with learned attention patterns.

### EMA (Exponential Moving Average)
Maintain a shadow copy of weights with decay=0.997, updated every step.
Use the EMA weights for final export. Replaces SWA in alertcat's submission.

### XSA (Cross-Sequence Attention)
Last 4 layers attend across all sequences in the batch (not just within-sequence).
Allows the model to leverage inter-sequence context at eval time.

### Batch Size 524K (vs 786K)
Smaller batch = 22% more gradient steps at the same wallclock budget.
More updates can matter more than larger batches when training is step-limited.

## Negative Results

- **Late QAT at 12 layers**: Harmful at -0.004 BPB. The step overhead of fake-quantize ops in the forward pass costs too many training steps when the model is already deep. Better to use post-training quantization with gradient-guided sensitivity ranking.

## Key Architecture Details from Top Submissions

### SmearGate (~512 params)
```python
gate = sigmoid(self.gate)  # shape [dim], init via sigmoid(3.0) ≈ 0.95
output = gate * current_emb + (1 - gate) * prev_token_emb
```
Applied after embedding lookup + bigram hash, before RMS norm.

### BigramHash (4096-10240 buckets, dim=128)
```python
hash_idx = (prev_token * 31 + curr_token) % num_buckets  # or * 92821
bigram_emb = self.bigram_table[hash_idx]  # (B, T, 128)
bigram_proj = bigram_emb @ self.bigram_proj  # (B, T, model_dim)
x = tok_emb + bigram_proj  # added before SmearGate
```
~524K params at 4096 buckets, ~1.3M at 10240.

### SWA (Stochastic Weight Averaging)
```python
# During warmdown, every swa_every steps:
if step >= swa_start and step % swa_every == 0:
    swa_state = ema_update(swa_state, model.state, n_averaged)
# At end of training, load swa_state into model before export
```
start_frac=0.4-0.5, every=50 steps.

### OrthoInit
```python
for module in model.modules():
    if hasattr(module, 'weight') and module.weight.ndim == 2:
        nn.init.orthogonal_(module.weight)
# Output projections scaled by 1/sqrt(2*num_layers)
```

### Int5 Quantization (MLP weights only)
```python
# Same as int6 but clip to [-16, 15], scale = clip_abs / 16.0
# Stored in int8 container — top 3 bits are zero/sign, compresses extremely well
```

### QAT with STE
```python
def fake_quantize_int6(w):
    scale = w.abs().amax(dim=-1, keepdim=True) / 31.0
    w_q = (w / scale).round().clamp(-31, 31)
    return w + (w_q * scale - w).detach()  # STE: forward uses quantized, backward uses original
```

## Compute Budget

- Phase 1: 3 runs × 10 min = 30 min 8×H100
- Phase 2: 5 runs × 10 min = 50 min 8×H100
- Phase 3: 10 runs × 10 min = 100 min 8×H100
- Phase 4: 5 runs × 10 min = 50 min 8×H100
- Iteration on 1×H100: ~20 hours
- **Total: ~4 hours 8×H100 + ~20 hours 1×H100**

## MLX Validation (completed)

- 25+ experiments on Mac (Apple M2)
- Best Mac result: 1.9588 BPB (14L×416d, 750 steps, 10 shards, full val)
- Near-zero quant gap validated (0.001 BPB with FP16 embed + Muon WD)
- Dead ends eliminated: depth recurrence + int6, DWA, eval-time loops, NTK extrapolation
- PR #328 submitted as non-record
