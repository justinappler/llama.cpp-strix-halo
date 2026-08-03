# Qwen3-Coder-Next 80B-A3B — baseline on gfx1151

Companion to [qwen3.6-baseline.md](qwen3.6-baseline.md). The two share a kernel
(TILE FA, D=256) but very different KV-cache shapes, so the [kv-cache.md](kv-cache.md)
finding needs re-validation here before we trust the production config.

## Why this bench

[Hugging Face model card](https://huggingface.co/Qwen/Qwen3-Coder-Next) confirms
the architecture:

| Param                 | Value                                                   |
| --------------------- | ------------------------------------------------------- |
| Total / active params | 80 B / 3 B                                              |
| Layers                | 48                                                      |
| Layout                | 12 × (3 × GatedDeltaNet → 1 × GatedAttention)           |
| Full-attention layers | **12 of 48** (the rest are linear-attn recurrent state) |
| Heads (Q / KV)        | 16 / 2 (GQA 8:1)                                        |
| Head dim              | **256**                                                 |
| Hidden                | 2048                                                    |
| Native context        | 262 144                                                 |

Two consequences for this hardware:

1. **D=256 → TILE FA kernel** (same as Qwen 3.6, see [fa-dispatcher.md](fa-dispatcher.md)).
   We do not unlock rocWMMA-tuned or any D≤128 path by switching to this model.
2. **KV is small.** Per-token KV at f16/f16 is `2 × 12 × 2 × 256 × 2 B ≈ 24 KB`.
   At 262k ctx that's ~6.3 GB. At 128k, ~3.1 GB. The original `q8_0/q4_0` KV
   config in the deploy repo's model config
   was inherited from Qwen 3.6 without measurement; there is no memory reason
   to pay the V-quant penalty kv-cache.md identified.

## Hypotheses

**H1 — KV-quant penalty replicates here.** [kv-cache.md](kv-cache.md) showed V-quant
on TILE FA / D=256 collapses pp at depth on Qwen 3.6 (V=q4_0 alone: 736 → 45 t/s
@ d=16k). Coder-Next's 12 attention layers run the same TILE kernel at the same
head dim, so the per-attention-layer cost ratio should be similar. f16/f16 KV is
predicted to dominate at depth.

**H2 — Depth scaling beats Qwen 3.6 even on the same kernel.** Only 12 of 48
layers are quadratic-attention; the rest are GatedDeltaNet recurrent (cost
constant in depth). pp@d=16k vs pp@d=0 should fall off less steeply than Qwen 3.6
(where Run 3 f16/f16 was 1029 → 731, a 29% fall). If H2 is wrong — i.e. depth
scaling is no better than Qwen 3.6 — Coder-Next at 73 GB has no compute-side win
to justify the 3.3× weight footprint over Qwen 3.6's 22 GB.

**H3 — f16/f16 KV fits at 262k.** Memory budget at full context: 73 GB weights

- ~6.3 GB KV + 16 GB cache-ram ≈ **95 GB** of ~100 GB usable. Predicted to load
  without OOM. If it doesn't fit, fall back to 131 072.

## Bench plan

| Component   | Value                                                                             |
| ----------- | --------------------------------------------------------------------------------- |
| Host        | Same as Qwen 3.6 baseline (Ryzen AI Max 395+, 8060S, 128 GB)                      |
| Container   | `llamacpp-server:local` (TheRock ROCm nightly per Dockerfile pin)                 |
| llama.cpp   | Same SHA the box is currently running (record `git rev-parse HEAD` at bench time) |
| Model       | `Qwen3-Coder-Next-UD-Q6_K_XL.gguf` (~73 GB)                                       |
| Bench flags | `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3`                             |

### Run 1 — KV quant matrix at depth (the load-bearing measurement)

Same shape as the [kv-cache.md](kv-cache.md) isolation matrix. Depths chosen to
straddle the agentic-coding range (most turns < 32k; long-context refactors hit
128k+).

Build: `a237ea1aa` (fork master), TheRock ROCm `7.13.0a20260504`, container env
includes the production `ROCBLAS_USE_HIPBLASLT_BATCHED=0` etc.

**Run aborted early.** V=q4_0 results were so collapsed by d=8k (and the kernel
visibly fell off the GPU path — `radeontop` showed GPU idle / CPU pinned during
the V=q4_0 cells, while f16/f16 pinned the GPU) that completing the q-V rows
would have burned hours for no marginal information. H1 is decisively
confirmed; q8_0/f16 (K-quant only) is the only untested cell worth revisiting,
and only if a memory-pressure scenario ever forces it.

#### pp512 (t/s)

| K cache | V cache | d=0          | d=8 192      | d=32 768     |
| ------: | ------: | -----------: | -----------: | -----------: |
|     f16 |     f16 |   **793.55** |   **633.29** |   **439.80** |
|     f16 |    q4_0 |       465.75 |        57.84 |          n/a |
|    q8_0 |     f16 |          n/a |          n/a |          n/a |
|    q8_0 |    q4_0 |          n/a |          n/a |          n/a |

#### tg128 (t/s)

| K cache | V cache | d=0          | d=8 192      | d=32 768     |
| ------: | ------: | -----------: | -----------: | -----------: |
|     f16 |     f16 |    **37.50** |    **36.17** |    **32.96** |
|     f16 |    q4_0 |        36.20 |        19.07 |          n/a |
|    q8_0 |     f16 |          n/a |          n/a |          n/a |
|    q8_0 |    q4_0 |          n/a |          n/a |          n/a |

#### What this says

- **H1 confirmed and amplified.** V=q4_0 collapses pp by **10.9× at d=8k**
  (633 → 58 t/s). For comparison, Qwen 3.6's V=q4_0 isolation in
  [kv-cache.md](kv-cache.md) showed ~16× at d=16k. Coder-Next's collapse is at
  least as severe per unit depth — likely worse because GQA to 2 KV heads makes
  the V-dequant a higher fraction of total attention cost.
- **V-quant hurts even at d=0.** pp512@d=0 drops 41% (793 → 466 t/s) with
  V=q4_0 alone. A 512-token prompt builds up V cache as it goes and the
  attention pass over it pays the dequant cost in-window. tg @ d=0 is barely
  affected (37.5 → 36.2) because tg128 is bandwidth-bound on weights, not
  KV traffic, when KV is small.
- **Production config is now correct.** `cache-type-k = f16`, `cache-type-v = f16`
  — locked in. The original q8_0/q4_0 setting in models.ini was a real footgun
  inherited from Qwen 3.6.
- **q8_0/f16 untested but uninteresting.** [kv-cache.md](kv-cache.md) showed K
  alone at q8_0 costs ~7× at depth on Qwen 3.6, on top of which we'd save only
  ~25% of an already-tiny KV (~6 GB at 262k). There's no scenario on this box
  where that trade pays.

### Run 2 — depth comparison vs Qwen 3.6

Direct compare against [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 3 (f16/f16,
FA on) at matched depths. This is the H2 test.

|  depth | Qwen 3.6 pp512 (Run 3) | Coder-Next pp512 (f16/f16) | Qwen 3.6 tg128 | Coder-Next tg128 |
| -----: | ---------------------: | -------------------------: | -------------: | ---------------: |
|      0 |                  1 029 |                    **794** |           46.5 |             37.5 |
|  8 192 |                      — |                    **633** |              — |             36.2 |
| 16 384 |                    731 |                  ~550 (interp.) |           43.3 |        ~34 (interp.) |
| 32 768 |                      — |                    **440** |              — |             33.0 |

**H2 falsified at f16/f16.** Same TILE-FA / D=256 kernel, Coder-Next pp@d≈16k
is *lower* than Qwen 3.6's by ~25% (interpolated 550 vs measured 731). The "12
of 48 layers do quadratic attention" structural advantage doesn't show up in
pp512 — most of the cost at this depth is the depth-linear attention cost in
those 12 layers, which is the same per-attention-layer-token regardless of how
many other layers are recurrent.

What Coder-Next *does* win on:
- **Quality at the same active-param count.** 80B / 3B vs 35B / 3B — more
  total experts, MoE routing can specialize harder. Not measured here.
- **tg flatness with depth.** Coder-Next: 37.5 → 33.0 (−12% at d=32k).
  Qwen 3.6 Run 3: 46.5 → 43.3 (−7% at d=16k). Roughly comparable per-depth slope,
  but Coder-Next runs ~25% slower in absolute tg because Q6_K_XL has more
  bytes/active-param than Q4_K_XL (~6.5 vs ~4.5 bits/weight).

**Net at f16/f16, on raw t/s alone, Qwen 3.6 is the faster model on this box.**
Coder-Next's case has to be made on coding quality, not throughput.

### Run 3 — wall-clock cold prefill at the working contexts

Mirrors the qwen3.6-baseline.md "wall-clock for a cold 10 k prefill" table.
Computed by integrating the f16/f16 pp curve from Run 1 (linear interpolation
between measured depths):

| Depth target | Cold prefill (estimate) |
| -----------: | ----------------------: |
|          8 k |                  ~11 s  |
|         32 k |                  ~58 s  |
|        128 k |       **TBD** (H3 fit-test not yet run) |

H3 (does f16/f16 KV at 262k fit?) remains untested — the 131k-depth row was
not run because the q-V early-termination ate the bench window. Re-run
single-config (`-ctk f16 -ctv f16 -d 131072`) when convenient; estimated wall
clock at d=131k is dominated by the 131k-token prefill setup before each rep
(~3-4 min × 3 reps = ~12 min for the row alone). Memory budget at 262k ctx
should be 73 + 6 + 16 ≈ 95 GB of ~104 GiB usable.

## Theoretical reference

- 8060S peak: ~59 TFLOPS FP16 (WMMA), 256 GB/s LPDDR5x-8000.
- Active-param tg ceiling at 256 GB/s, Q6_K_XL ≈ 6.5 bits/weight: 3 B × 6.5/8 ≈
  2.4 GB/token → **~105 t/s tg ceiling**. (Compare Qwen 3.6 A3B at Q4_K_XL: 3 B
  × 4.5/8 ≈ 1.7 GB/token → ~150 t/s ceiling. Coder-Next's heavier quant trades
  ~30% of tg ceiling for higher fidelity.)
- pp ceiling is GEMM-bound, not KV-bound, on this kernel — direct compare to
  Qwen 3.6 numbers is the meaningful reference.

## What we are _not_ measuring in this pass

Held back so the KV variable isolates cleanly:

- `parallel = 1 → 2` + `slot-prompt-similarity` — same agentic motivation as
  Qwen 3.6 (subagent slot eviction), but a workload-pattern choice not directly
  measured by `llama-bench`. Phase 2.
- `ctx-checkpoints` / `cache-ram` / `checkpoint-every-n-tokens` — Qwen 3.6 went
  from defaults to 1024/384/32768 after measurement. Coder-Next's GDN-recurrent
  state changes the checkpoint cost; needs its own pass once KV is settled.
- `ctx-size` floor — does 131 072 give 90% of the value at lower memory? Worth
  knowing if we ever want to free headroom for cache-ram or a second model.
- `GGML_HIP_ROCWMMA_FATTN=ON` — D=256, so [Finding #6](../README.md) keeps it OFF.
  Not in scope here.

## Related findings

- [kv-cache.md](kv-cache.md) — the V-quant collapse this bench re-validates.
- [qwen3.6-baseline.md](qwen3.6-baseline.md) — the comparison anchor.
- [fa-dispatcher.md](fa-dispatcher.md) — why D=256 means TILE FA on this box.
