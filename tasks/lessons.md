# Parameter Golf — Lessons Learned

## Session 1: Initial Setup & MLX Prototyping

### MLX Performance
- Baseline 200 steps with batch=8192 takes ~228 seconds on Apple Silicon (~1.14s/step)
- Validation on full val set (62M tokens) with batch=8192 is EXTREMELY slow on Mac (~7,500 batches). Use VAL_BATCH_SIZE=131072 or higher for acceptable speed (~473 batches)
- Warmup (20 steps) is compilation-only in MLX, doesn't update weights
- Don't use `&` inside background bash commands — it orphans processes and loses output. Use `run_in_background: true` directly on the tool call

### Architecture Notes
- Baseline has 17.1M params, compresses to ~15.8MB with int8+zlib
- Encoder/decoder U-Net split (4 enc + 5 dec) with skip connections is the default
- Block has: attn (Q/K/V/proj) + MLP (fc/proj) + resid_mix + attn_scale + mlp_scale + q_gain
- resid_mix blends current residual with original x0 — important for depth recurrence
- Output projections (attn.proj, mlp.proj) are zero-initialized (cold start)

### Implementation Decisions
- Recurrence mode drops U-Net skips entirely (simplest first)
- No per-loop gating in v1 — relying on resid_mix in shared blocks to adapt across loops
- Will add LoRA adapters in v2 if plain recurrence shows promise

## Session 2: Eval Improvements + Performance

### Critical: mx.compile is mandatory for eval forward pass
- Removing `compiled_loss` and using uncompiled `model(x)` for eval caused ~10-50x slowdown
- Uncompiled MLX forward pass incurs full Python overhead per operation per batch
- With 475 batches on full val, uncompiled eval takes 20+ minutes vs ~1 min compiled
- **Fix**: create `compiled_forward = mx.compile(lambda x: model(x), ...)` for default eval path
- Fall back to uncompiled ONLY when eval requires non-default loop count (EVAL_NUM_LOOPS>0) or NTK-RoPE scaling (EVAL_SEQ_LEN != train_seq_len), since compiled functions don't pick up replaced RoPE modules
- Sliding window (EVAL_STRIDE) works fine with compiled forward — only the scoring window changes, not the forward pass

### Eval batch size
- Original divided val_batch_size by grad_accum_steps (artifact from training) — unnecessarily small batches
- Removing the division entirely (128 seqs/batch) may work but is aggressive for Mac memory with DWA (9 history tensors)
- Safe approach: remove grad_accum_steps division but keep val_batch_size as the memory knob users can tune

### Full val eval is too slow for Mac iteration
- 62M token val set with compiled forward on 9-layer model takes ~18 min per eval pass
- With 2 eval passes (step 0 + final) + int8 roundtrip + multi-eval = 72+ min of eval per run
- **Fix**: use VAL_MAX_TOKENS=6000000 (~10% val) for directional experiments, full val only for final numbers
- **Fix**: use VAL_LOSS_EVERY=0 to skip step 0 eval (untrained baseline not needed)
- Background bash commands have 10-min timeout by default — use `timeout: 600000` for long runs

### Empty array serialization
- Recurrence mode sets skip_weights to (0, dim) — MLX's mx.savez can't serialize empty arrays
- Fixed by filtering `v.size > 0` in flat_state before save

## Session 2: Experiment Results

### DWA doesn't improve recurrence at this scale
- DenseFormer DWA (weighted average of prior layer outputs) slightly hurt BPB vs plain recurrence (2.3396 vs 2.3368)
- The near-identity init (logit 5.0 on current state) means DWA starts as almost no-op, but the training overhead and additional parameters don't pay off in 200 steps
- At larger scale / more training steps, DWA might still help — but not worth the complexity for the competition

### Extra eval loops are broken with cyclic DWA
- Running 6 loops at eval (trained with 3) gave catastrophic BPB regression (3.49 vs 2.34)
- The model's internal representations are deeply calibrated to exactly `num_loops` passes — you can't just add more loops
- The cyclic DWA reuse (layer_idx % effective_depth) doesn't preserve the original DWA semantics when history is longer than training
- **Lesson**: eval-time depth scaling requires training-time support (e.g., training with variable loop counts, or progressive depth)

### Plain recurrence is the winner
- 3×3 512-dim (6M params) slightly outperformed 9×512 baseline (17M params) at 200 steps
- This means wider recurrence (3×3 768) could significantly beat baseline while compressing much smaller
- Simple recurrence without bells and whistles is the right path for the competition

### Width sweep: 768 is sweet spot at 200 steps
- 512 → 768 → 896 → 1024: BPB goes 2.337 → **2.316** → 2.325 → 2.372
- Wider models underfit at 200 steps (higher train loss) but have more capacity
- On H100 with 20K steps, 896 or 1024 may overtake 768 — worth testing
- All configs fit easily in 16MB: 768=6.2MB, 896=7.5MB, 1024=8.5MB compressed

### More unique blocks > more loops
- 4×3 768 (2.3117) beats 3×3 768 (2.3155) which beats 3×4 768 (2.3199)
- Adding unique blocks (more params, same effective depth) helps more than adding loops (same params, more depth)
- This makes sense: each unique block learns a distinct transformation, while extra loops just repeat
- **Best config: 4×3 768-dim = 17.3M params, 7.4MB compressed, val_bpb 2.3117**

### Int6 quantization is incompatible with depth recurrence
- Int6 on baseline (no recurrence): quant gap = 0.023 BPB — works fine
- Int6 on 4×3 768 recurrence: quant gap = **0.61 BPB** — catastrophic
- Root cause: each shared weight is used N_LOOPS times, amplifying quantization error
- Confirmed by competition PR #76 which also rejected recurrence due to quant gap
- **Implication**: recurrence models MUST use int8 (or QAT), which limits how much the budget headroom can be exploited
- With int8, 4×3 768 compresses to 7.4MB — leaving 8.6MB unused. Int6 could fill that space but only for non-recurrence models

### MLP_MULT=3 is a confirmed improvement
- Baseline 9×512 MLP2: val_bpb 2.3396 (10% val, 200 steps)
- Baseline 9×512 MLP3: val_bpb 2.3262 (10% val, 200 steps)
- Delta: -0.013 BPB. Consistent with competition PRs reporting ~-0.019 at full training

### Leaderboard #1 recipe validated on Mac
- 10L + MuonWD(0.02) + OvertoneInit + PhaseResidMix + FP16Embed + TIED_EMBED_LR=0.10
- **Near-zero quantization gap (0.0002 BPB)** with int8 — the key insight
- FP16 embed + Muon WD makes quantization virtually invisible
- 10 layers underfits at 200 steps but wins at full training (proven on H100: 1.1748 BPB)
- Sliding window stride-64 at native seq1024: small gain (-0.0015 BPB at 200 steps, -0.032 at full training)
- Adding MLP3 + int6 increases quant gap to 0.030 — still usable, needs QAT to close
- **This recipe is the path forward for H100 submission**

### WD + LR sweep results
- WD=0.02 + matrix_lr=0.02 is the best combo at 200 steps (-0.005 BPB over default)
- WD=0.04 slightly worse than WD=0.02 at short training (may reverse at 10K+ steps on H100)
- OvertoneInit + PhaseResidMix HURT when combined with WD+LR tuning — they conflict at short training
- Best Mac recipe: SmearGate + BigramHash(4096) + WD=0.02 + LR=0.02 (no init tricks)

### Train at seq4096 needs H100 batch sizes
- On Mac with batch=8192: only 2 seqs/step → severe underfitting (val_bpb 2.61 vs 2.34)
- Top submissions use 393K+ tokens/step giving 96+ seqs at seq4096
- Cannot validate on MLX, must wait for H100
