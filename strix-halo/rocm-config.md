# ROCm config flags — LLVM unroll + HIPBLASLT_BATCHED — null on Qwen 3.6

## Update (2026-09-25): ROCm 10.0.0, installed from apt

The deploy build moved from the ROCm 7.14.0 TheRock tarball to **ROCm 10.0.0** (LLVM 23 -> 24, HIP 7.14 -> 10.0), installed from AMD's signed stable apt repo as per-arch packages: `amdrocm-core-dev10.0-gfx1151` to build, `amdrocm10.0-gfx1151` at runtime. Two things to know:

- **Layout changed.** Packages install under `/opt/rocm/core-10.0/`, not a flat `/opt/rocm`. `ROCM_PATH`, `HIP_PATH`, `PATH` and the loader path point there. The series is in every package name, so going to 10.1 renames packages, not just the version pin.
- **The tarball path moved too.** From 10.0 the stable tarballs live at `stable.repo.amd.com/rocm/core/tarball/`; the old `repo.amd.com/rocm/tarball/` index stops at 7.13 (7.14.0 is there but unlisted). ROCm 7.14.1 (2026-08-31) only fixes an MI300X RCCL regression and an amdflang build break, so it was skipped.

**Benched 2026-09-25, same llama.cpp build `60d0850`, 3 interleaved cycles, ROCm the only variable:**

| test | 7.14.0 | 10.0.0 | delta |
| --- | ---: | ---: | ---: |
| pp512 d=0 | 1347.0 (1336-1364) | 1355.3 (1353-1357) | +0.6% |
| pp512 d=2,048 | 1292.7 (1285-1298) | 1282.0 (1268-1293) | -0.8% |
| pp512 d=8,192 | 1136.7 (1099-1159) | 1154.5 (1145-1169) | +1.6% |
| pp512 d=16,384 | 1030.9 (1013-1040) | 1004.8 (990-1015) | -2.5% |
| tg128 d=0 / 2k / 8k / 16k | 52.12 / 51.76 / 50.42 / 48.53 | 51.85 / 52.04 / 50.50 / 48.64 | -0.5% to +0.5% |

**Adopted: a wash, with one soft spot.** Decode is flat and prefill is within noise at three depths. d=16k is -2.5% with ranges that just touch (1013 vs 1015), not beyond noise but worth watching on the next re-bench. The HIP runtime library reports `7.15.26333` despite the 10.0 package version.

## Status (2026-08-02): unroll flag retired, the underlying bug is fixed

**The `-mllvm --amdgpu-unroll-threshold-local=600` workaround was dropped from the build.** The LLVM regression it worked around is fixed in our toolchain.

The revert landed in rocm-llvm ([#1348](https://github.com/ROCm/llvm-project/pull/1348), [#1349](https://github.com/ROCm/llvm-project/pull/1349)) and reached TheRock nightlies around 2026-02-13. On the [ROCm/rocm-systems#2865](https://github.com/ROCm/rocm-systems/issues/2865) thread, AMD's `@fjankovi` confirmed on 2026-03-02 that "current ROCm 7.12 nightly builds already have this fix", and a third party verified it on a **gfx1151** 7.12 tarball: "the performance test is normal, and the BUG has indeed been fixed." This fork moved to ROCm 7.14.0 on 2026-07-16, so the fix has been in the toolchain since then and the flag was forcing a non-default unroll threshold on a compiler that no longer needs it.

The issue is still open upstream, but that is triage hygiene, not an open bug. **The old note in the deploy Dockerfile ("still open... so it stays on through 7.14") was written in the TheRock 7.11 era and went stale at the 7.14.0 switch.**

`ROCBLAS_USE_HIPBLASLT_BATCHED=0` is unaffected - it is a runtime env var, not a build flag, and it stays.

**Benched clear (2026-08-02, build `b73cfa4`).** Prefill was flat versus the previous build (+0.3% to +1.9%, mostly inside noise) with the flag gone, so removing it cost nothing measurable on Qwen 3.6 - consistent with the null A/B above and with the compiler bug being fixed. Upstream's `-ffast-math` removal rode along in the same build and is likewise clear. See [qwen3.6-baseline.md](qwen3.6-baseline.md#2026-08-02--post-rebase-re-bench-build-b73cfa4). Restoring the flag is one line in the deploy Dockerfile if a future model ever wants it.

---

**Original status: bench null, kept on anyway.** Two community-recommended ROCm config flags for Strix Halo; no measurable change on our Qwen 3.6 Q4_K_XL config. They stayed enabled in the deploy config as AMD-recommended safety nets for other models / future ROCm versions, not as Strix Halo pp wins for this workload.

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

Both flags live in the deploy config, not this repo. Not part of the llama.cpp build tree.

> [!NOTE]
> **2026-08-02:** upstream [PR #26046](https://github.com/ggml-org/llama.cpp/pull/26046) deleted rocWMMA FlashAttention, so `GGML_HIP_ROCWMMA_FATTN` no longer exists. The condition attached to `ROCBLAS_USE_HIPBLASLT_BATCHED=0` above is now vacuously satisfied. Keep the flag on its own merits (bench-null here, AMD-recommended, cheap insurance) rather than on that reasoning.
>
> Separately, upstream [PR #25495](https://github.com/ggml-org/llama.cpp/pull/25495) (merged 2026-07-27) **removed `-ffast-math -fno-finite-math-only` from the HIP build**. That is a global codegen change on our exact backend that we did not make and did not measure. It is the most likely confounder in the next re-bench, and it sits next to the still-pending unroll-threshold bisect below.

## Recommendation

Don't count this as a Strix Halo pp win. But also don't remove the flags — they're correct by AMD's own guidance for our build, and the null delta here doesn't disprove their value on other models.

If a future model load shows unexpectedly slow pp vs community reports, flipping either off for A/B is the first thing to try.

> Superseded for the unroll flag by the [2026-08-02 status](#status-2026-08-02-unroll-flag-retired-the-underlying-bug-is-fixed) above: "don't remove it" was correct while the compiler bug was live. It isn't any more. `ROCBLAS_USE_HIPBLASLT_BATCHED=0` still stands as written.
