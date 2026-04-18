# Strix Halo Optimization Notes

Exploration of potential improvements in llama.cpp for AMD Strix Halo (Ryzen AI Max, `gfx1151`): Zen 5 CPU + RDNA 3.5 iGPU + unified LPDDR5X memory.

Findings are ordered by likely impact. None have been benchmarked — each is a pointer worth investigating, not a verified win.

## HIP / ROCm backend

### 1. Zero-copy for integrated GPUs is globally disabled

[ggml/src/ggml-cuda/ggml-cuda.cu:243](ggml/src/ggml-cuda/ggml-cuda.cu#L243)

`integrated = false` is hard-coded with a comment referencing issue #15034. That flag gates a real UMA fast-path — GPU kernels being allowed to operate directly on CUDA-host (pinned/UMA) buffers instead of requiring a DeviceLocal copy ([line 4085](ggml/src/ggml-cuda/ggml-cuda.cu#L4085), [line 5125](ggml/src/ggml-cuda/ggml-cuda.cu#L5125)).

On Strix Halo, "VRAM" is a GART carve-out of the same DDR, so the copy is pure waste. Fixing #15034 and conditionally re-enabling would likely be the single biggest win.

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

## Suggested starting point

**#1 (integrated flag for HIP)** — it's a documented, deliberately-disabled optimization, and the linked issue (#15034) would reveal exactly what broke. The fix may be narrower than it appears.
