# ROCm config flags — LLVM unroll + HIPBLASLT_BATCHED — null on Qwen 3.6

**Status: bench null, kept on anyway.** Two community-recommended ROCm config flags for Strix Halo; no measurable change on our Qwen 3.6 Q4_K_XL config. They stay enabled in server-configs as AMD-recommended safety nets for other models / future ROCm versions, not as Strix Halo pp wins for this workload.

## Background

[ggml-org/llama.cpp#17917](https://github.com/ggml-org/llama.cpp/issues/17917) is an active, documented pp regression on Strix Halo. Root causes per the thread:

1. **LLVM unroll-threshold regression** in ROCm 7.2+ codegen. Reverted in a later rocm-llvm commit but still present in the TheRock nightly tarballs our Dockerfile pulls from. Reports of ~2× pp recovery on gpt-oss-120b with `-mllvm --amdgpu-unroll-threshold-local=600` added to `CMAKE_HIP_FLAGS`.
2. **rocBLAS batched-GEMM routing through hipBLASLt** introduced in ROCm 7.0. hipBLASLt doesn't implement general batched GEMMs, so some shapes fall off a fast path. AMD's own engineer (`@slojosic-amd`) called `ROCBLAS_USE_HIPBLASLT_BATCHED=0` *mandatory* when building with `GGML_HIP_ROCWMMA_FATTN=OFF` (our config).

Both reports were primarily on **gpt-oss 120B MXFP4** at `-ub 2048`. Worth checking whether Qwen 3.6 Q4_K_XL hits the same pessimized paths.

## Evidence

Qwen 3.6 35B-A3B Q4_K_XL, `b=4096 ub=2048 ngl=999 mmp=0 fa=1`, f16/f16 KV. Baseline from [qwen3.6-baseline.md](qwen3.6-baseline.md) run 3, same build `309b410e2`, same ROCm nightly `7.13.0a20260411`:

| test | baseline | +unroll +batched=0 | delta |
|---|---:|---:|---:|
| pp512 @ d=0      | 1,029 | 1,077 | +4.7% |
| pp512 @ d=16,384 |   731 |   737 | +0.8% |
| tg128 @ d=0      |  46.5 | 46.75 | +0.5% |
| tg128 @ d=16,384 |  43.3 |  43.6 | +0.7% |

Within run-to-run noise on the baseline (the 1,029 baseline itself was the best of three runs that spanned 1,025-1,029).

## Interpretation

Two candidate explanations, not exclusive:

- **Workload mismatch.** The reported 2× recoveries were on gpt-oss 120B MXFP4 — a quant format and a model size that routes through different GEMM shapes than our Q4_K_XL MoE with 3B active params. Our path may not touch the kernels the unroll regression pessimized.
- **Flag didn't propagate.** `CMAKE_HIP_FLAGS` *should* feed the HIP device compiler, but we didn't verify the resulting `.hsaco` dump. Possible it's only affecting host-side HIP runtime code, not the compute kernels.

We didn't chase explanation 2 — even if the flag took effect, the gpt-oss-120b reports don't promise anything for Qwen 3.6, and the baseline is already close to the theoretical compute ceiling for MLP-only pp ([qwen3.6-baseline.md](qwen3.6-baseline.md) notes ~10% of the 9,800 t/s MLP ceiling; depth-0 pp is already a reasonable fraction given attention + MoE routing overhead).

## Why keep them on anyway

Both are zero-risk for our config:

- `-DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600"` — the unroll-threshold override is a compiler hint, not a correctness change. Confirmed recovery on other models, no reported regressions on Q4_K.
- `ROCBLAS_USE_HIPBLASLT_BATCHED=0` — AMD-recommended when `GGML_HIP_ROCWMMA_FATTN=OFF`. Our config hits that condition; following the recommendation is cheap insurance for other models we might load.

Both flags live in server-configs, not this repo. Not part of the llama.cpp build tree.

## Recommendation

Don't count this as a Strix Halo pp win. But also don't remove the flags — they're correct by AMD's own guidance for our build, and the null delta here doesn't disprove their value on other models.

If a future model load shows unexpectedly slow pp vs community reports, flipping either off for A/B is the first thing to try.
