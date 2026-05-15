# Quantized KV cache collapses throughput at depth (TILE FA kernel)

## Finding

On gfx1151 with the TILE flash-attention kernel (the fallback kernel RDNA 3.5 lands on — see [fa-dispatcher.md](fa-dispatcher.md)), setting `cache-type-k = q8_0` and `cache-type-v = q4_0` collapses prompt-processing throughput by an order of magnitude at long context compared to `f16/f16` KV.

## Evidence

Qwen 3.6 35B-A3B Q4_K_XL, `b=4096 ub=2048`, FA on, `-mmp 0`, llama.cpp build `45cac7c`:

| K cache | V cache | pp512 @ d=0 | pp512 @ d=16,384 | tg128 @ d=0 | tg128 @ d=16,384 |
|--------:|--------:|------------:|-----------------:|------------:|-----------------:|
|    q8_0 |    q4_0 |         767 |           **43** |        45.4 |             19.1 |
|     f16 |     f16 |       1,029 |          **731** |        46.5 |             43.3 |

**17× pp at depth 16k. 2.3× tg at depth 16k. 1.34× pp even at depth 0.**

An attempted `fa=0, q8_0/q4_0` run fails with `failed to create context` — the TILE kernel with FA off does not support quantized KV at all. FA-on is also using TILE (per dispatcher trace) but tolerates quant KV at a massive performance cost.

## Isolation: K-quant vs V-quant

Re-ran the matrix with K and V varied independently (same build/flags):

| K cache | V cache | pp512 @ d=0 | pp512 @ d=16,384 | tg128 @ d=0 | tg128 @ d=16,384 |
|--------:|--------:|------------:|-----------------:|------------:|-----------------:|
|     f16 |     f16 |       1,063 |          **736** |        48.6 |             45.1 |
|    q8_0 |     f16 |         862 |          **106** |        47.0 |             25.8 |
|     f16 |    q4_0 |         742 |           **45** |        46.4 |             19.4 |
|    q8_0 |    q4_0 |         737 |           **47** |        45.9 |             21.0 |

**V-quant is the dominant cost.** `f16/q4_0` alone already matches `q8_0/q4_0` at depth (45 vs 47 t/s pp @ 16k). Adding K quantization on top contributes almost nothing further. In isolation K=q8_0 costs ~7× at depth; V=q4_0 alone costs ~16×.

## Interpretation

Hypothesis: the TILE kernel dequantizes K/V on every attention step rather than operating on quantized values directly. Cost per token scales with attended context, so the penalty compounds at depth. The asymmetry (V hurts ~2× more than K) probably reflects that V participates in the post-softmax `attn @ V` matmul across every attended position, while K is consumed by `Q @ K^T` where the TILE kernel may have a less expensive dequant path.

This is separate from the rocWMMA FA issue noted in the parent Dockerfile (`GGML_HIP_ROCWMMA_FATTN=OFF` due to a 100× slowdown on gfx1151 — see [NOTES.md](NOTES.md) item #4 and [ROCm/ROCm#6042](https://github.com/ROCm/ROCm/issues/6042)).

## Recommendation

**While on the TILE kernel**, use `cache-type-v = f16` unconditionally — this is the load-bearing change. `cache-type-k` can be `q8_0` if RAM is tight (saves ~25% of KV RAM for a ~15% pp hit at d=16k), but on a 128 GB Strix Halo there is no reason to: use `f16/f16`.

RAM cost is tractable on 128 GB systems:

| ctx-size | KV @ f16/f16 | + 22 GB model + 16 GB cache-ram |
|---------:|-------------:|--------------------------------:|
|     32 k |       ~8 GB  |                         ~46 GB  |
|     65 k |      ~16 GB  |                         ~54 GB  |
|    128 k |      ~32 GB  |                         ~70 GB  |

Well within the ~88-100 GB of available memory after system reservations on a 128 GB Strix Halo.

## Re-evaluate after the FA dispatcher patch

Once the [fa-dispatcher.md](fa-dispatcher.md) patch routes RDNA 3.5 through MMA_F16 instead of TILE, re-benchmark quantized KV. The MMA kernel has genuine quantized-KV support and may make q8_0/q4_0 viable again with a much smaller depth penalty.
