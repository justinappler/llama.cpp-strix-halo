# Strix Halo Optimization Notes

Exploration of potential improvements in llama.cpp for AMD Strix Halo (Ryzen AI Max, `gfx1151`): Zen 5 CPU + RDNA 3.5 iGPU + unified LPDDR5X memory.

Findings are ordered by likely impact. None have been benchmarked — each is a pointer worth investigating, not a verified win.

## HIP / ROCm backend

### 1. Zero-copy for integrated GPUs is globally disabled — **researched, deprioritized**

[ggml/src/ggml-cuda/ggml-cuda.cu:243](ggml/src/ggml-cuda/ggml-cuda.cu#L243)

`integrated = false` is hard-coded with a comment referencing issue #15034. Originally flagged here as the likely single biggest Strix Halo win. **Research in [uma-integrated.md](uma-integrated.md) concluded otherwise**: the flag only gates the `cuda_host` pinned-buffer path (logits, small scratch), not weight or KV traffic. Weights go through `hipMalloc` on HIP regardless. PR #16308 author reported no Jetson perf impact from toggling this, which is a strong prior-against.

Not worth pursuing as a pp optimization on gfx1151. See the doc for the full consumer-site analysis and the separate `GGML_CUDA_ENABLE_UNIFIED_MEMORY` env var that provides a real UMA lever if one is wanted.

### 2. MMVQ routes RDNA3.5 to the RDNA2 tuning table

[ggml/src/ggml-cuda/mmvq.cu:77](ggml/src/ggml-cuda/mmvq.cu#L77), [mmvq.cu:93](ggml/src/ggml-cuda/mmvq.cu#L93)

RDNA3.5 has dual-issue VALU and WMMA like RDNA3_0, but MMVQ sends it down `MMVQ_PARAMETERS_RDNA2` (nwarps=1). The RDNA3_0 table at [mmvq.cu:348](ggml/src/ggml-cuda/mmvq.cu#L348) uses `nwarps=8` on a whitelist of types.

Worth benchmarking whether RDNA3.5 wants its own table or should join RDNA3_0.

### 3. mmf gates `src1_ncols > 8` on RDNA3_0 only

[ggml/src/ggml-cuda/mmf.cu:169](ggml/src/ggml-cuda/mmf.cu#L169)

RDNA3.5 falls through to the generic `src1_ncols > 16` cap. The asymmetry may be a missed tuning decision either way, and is worth a benchmark.

## Vulkan backend

### 4. No `AMD_RDNA3_5` architecture classification

[ggml/src/ggml-vulkan/ggml-vulkan.cpp:270-279](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L270-L279)

Strix Halo is bucketed as `AMD_RDNA3`, identical tuning to a 7900 XTX — despite very different cache hierarchy, 40 CUs vs 96, and 256 vs 960 GB/s bandwidth. A dedicated class would enable different warptile selection at [line 3326-3332](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L3326-L3332) and different FA occupancy caps.

### 5. FA `limit_occupancy_shmem` workaround tuned on discrete RDNA2

[ggml/src/ggml-vulkan/ggml-vulkan.cpp:2990-2994](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2990-L2994)

Comment says "guessed, tested on RDNA2". On Strix Halo, LLC is shared with CPU — cache thrashing characteristics differ. Separately, `row_split_max_hsk` at [line 2948](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2948) is kept low *because* of UMA. Both deserve re-measurement.

### 6. UMA device allocation doesn't prefer host memory by default

[ggml/src/ggml-vulkan/ggml-vulkan.cpp:2799-2808](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2799-L2808)

On UMA, the code asks for `DeviceLocal` first and falls back to `HostVisible+HostCoherent`. On Strix Halo, DeviceLocal is a small GART window (~512 MB unless BIOS-reconfigured) — large models always hit the fallback.

Auto-setting `prefer_host_memory` when `device->uma` is true would avoid the failed first-try and the driver thrash it causes.

### 7. UMA DeviceLocal buffer doesn't request HostCached

Same allocation site as #6. Write-combined memory is fine for weight upload (write-only), but any CPU-side readback paths pay a penalty. [line 6465](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6465) already uses Cached for pinned host memory — the main device buffer could at least prefer it on UMA.

## CPU backend

### 8. No `zen5` CPU backend variant

[ggml/src/CMakeLists.txt:379](ggml/src/CMakeLists.txt#L379)

Only `zen4` exists. Zen 5's ISA bits are the same (AVX-512 + VBMI + VNNI + BF16), but Zen 5 executes 512-bit ops on a native 512-bit datapath vs Zen 4's double-pumped 256-bit — different loop unrolling / tile sizes may win. Easy to add; needs benchmarking to justify.

### 9. mmap path has no huge-page hint

[src/llama-mmap.cpp:437-467](src/llama-mmap.cpp#L437-L467)

With 64-128 GB of LPDDR5X and 60B+ parameter models, TLB pressure during inference is real. A `madvise(MADV_HUGEPAGE)` after mmap (when `numa == false`) is cheap; `MAP_HUGETLB` would need flag opt-in.

## Cross-cutting

### 10. No Strix Halo bench reference

[benches/](benches/) has `dgx-spark` and `mac-m2-ultra` but no Strix Halo. A committed baseline would make tuning claims defensible and catch regressions in RDNA3.5 paths.

---

## Prioritized next experiments

T-shirt sizes: **S** = hours, **M** = a day or two, **L** = a week, **XL** = multi-week. Benefit is measured or plausible pp/tg delta on the Qwen 3.6 workload unless noted. "Shipped" items are listed in the [README findings table](README.md#findings), not here.

### Highest ROI first

| # | Item | Cost | Benefit | Why this position |
|--:|---|:--:|:--:|---|
| 1 | **#2 MMVQ RDNA3.5 tuning table** — RDNA3.5 is routed through the RDNA2 `nwarps=1` path despite having RDNA3.0-like dual-issue VALU + WMMA. [mmvq.cu:77](../ggml/src/ggml-cuda/mmvq.cu#L77), [:93](../ggml/src/ggml-cuda/mmvq.cu#L93), [:348](../ggml/src/ggml-cuda/mmvq.cu#L348). Tuning constants only; MoE tg on Qwen 3.6 is MMVQ-heavy. | S | M-L | Single-line constant change, analogous in shape to the shipped MMQ win (Finding #5). Highest info-per-hour candidate. |
| 2 | **MoE prefill / `MUL_MAT_ID`** — [issue #17014](https://github.com/ggml-org/llama.cpp/issues/17014) tied a 15-20 % HIP PP regression on Strix Halo to a MoE kernel change; [issue #21948](https://github.com/ggml-org/llama.cpp/issues/21948) reports `MUL_MAT_ID` dominates large-MoE prefill. Biggest unexplored pp lever for our actual workload (Qwen/GPT-OSS MoE). | M-L | L-XL | Most promising pp win after MMQ; investigation, not a patch — scope grows fast. Do after #1 so we have a MMVQ baseline to compare against. |
| 3 | **#3 mmf `src1_ncols` gate asymmetry** — RDNA3.0 caps at `>8`, RDNA3.5 inherits generic `>16`. [mmf.cu:169](../ggml/src/ggml-cuda/mmf.cu#L169). | S | S-M | Pair with #1 — same kind of single-line threshold bench. |
| 4 | **#8 Zen 5 CPU backend variant** — [CMakeLists.txt:379](../ggml/src/CMakeLists.txt#L379) only has `zen4`. Zen 5 has a native 512-bit datapath vs Zen 4's double-pumped 256-bit. | S | S-M | CPU-side only, off the GPU hot path, so ROI is bounded. Cheap enough to try anyway. |
| 5 | **#9 `madvise(MADV_HUGEPAGE)` after mmap** — [llama-mmap.cpp:437-467](../src/llama-mmap.cpp#L437-L467). TLB pressure is real at 60B+ params in 128 GB LPDDR5X. | S | S (load-time; small ongoing tg at best) | One-line hint, low risk, low reward. |
| 6 | **#10 Commit a Strix Halo bench to `benches/`** — gives us a defensible regression signal and an upstream talking point. | S | infra | Not a perf win; worth doing once we have one more patch landed to anchor the baseline. |

### Vulkan-only — only if we switch backends

| # | Item | Cost | Benefit | Note |
|--:|---|:--:|:--:|---|
| V1 | **#4 `AMD_RDNA3_5` architecture class** — gfx1151 bucketed as `AMD_RDNA3` despite 40 vs 96 CUs, 256 vs 960 GB/s BW. [ggml-vulkan.cpp:270-279](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L270-L279). | M | unknown, plausibly L | Unlocks dedicated warptile + FA occupancy paths. |
| V2 | **#5 FA `limit_occupancy_shmem` re-tune** — [ggml-vulkan.cpp:2990-2994](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2990-L2994). Heuristic was "guessed, tested on RDNA2"; Strix Halo shares LLC with CPU. | S | S-M | Bundle with V1. |
| V3 | **#6/#7 UMA allocation (`prefer_host_memory` + `HostCached`)** — [ggml-vulkan.cpp:2799-2808](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2799-L2808). `DeviceLocal` is a ~512 MB GART window unless BIOS-reconfigured; large models always fall back. | S-M | S (mostly load-time) | |

### Dead ends — documented, not pursued

- **FA MMA_F16 on gfx1151** ([fa-dispatcher.md](fa-dispatcher.md)): blocked on an RDNA3 unpacked-WMMA register-layout bug that killed upstream [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063). Revisit only if someone lands a correctness fix upstream.
- **#1 UMA / `integrated = false`** ([uma-integrated.md](uma-integrated.md)): PR #16308 author reported no perf impact; the flag gates only small scratch buffers, not weight/KV traffic.
- **ROCm config flags** ([rocm-config.md](rocm-config.md)): `ROCBLAS_USE_HIPBLASLT_BATCHED=0` + LLVM unroll-threshold. Community reports 2× pp on gpt-oss-120b; null on Qwen 3.6. Kept on as AMD-recommended safety nets.

### Watching upstream

- **[PR #22051](https://github.com/ggml-org/llama.cpp/pull/22051)** (JohannesGaessler, open 2026-04-17) — refactors AMD mma data loading in `mma.cuh`. Doesn't unblock FA on gfx1151 and doesn't fix the RDNA3 WMMA layout bug, but reshapes MMQ code we touched in Finding #5. **Action:** on next upstream sync after this merges, re-bench Qwen 3.6 Q4_K_XL with + without our MMQ commit to confirm the port still wins. If it does, keep; if it no-ops, drop and annotate.
