# Flash-attention dispatcher gates RDNA 3.5 out of the MMA kernel

**Status: attempted, abandoned.** The top-level dispatcher change is necessary but not sufficient — the MMA_F16 kernel's device code is not compiled for `gfx1151`. See [Outcome](#outcome) below. Unblocks require cmake changes to `template-instances/`, not a dispatcher patch.

## Symptom

On `gfx1151` with default build flags (`GGML_HIP_ROCWMMA_FATTN=OFF`), flash-attention runs on the generic **TILE** fallback kernel. TILE has no efficient quantized-KV path, which makes its throughput collapse at depth (see [kv-cache.md](kv-cache.md)).

The root cause appeared to be dispatcher logic in [fattn.cu](../ggml/src/ggml-cuda/fattn.cu), not a missing kernel. That turned out to be incomplete — the dispatcher gate was defensive, guarding against a real problem (no compiled device code) rather than an arbitrary restriction.

## Dispatch trace for gfx1151

Reading [ggml/src/ggml-cuda/fattn.cu:307-505](../ggml/src/ggml-cuda/fattn.cu#L307-L505) for RDNA 3.5:

1. `turing_mma_available(cc)` — NVIDIA Turing+, skipped.
2. `volta_mma_available(cc)` — NVIDIA Volta, skipped.
3. `ggml_cuda_should_use_wmma_fattn(cc)` — skipped because the build has `GGML_HIP_ROCWMMA_FATTN=OFF`. See [fattn-wmma-f16.cuh:26-49](../ggml/src/ggml-cuda/fattn-wmma-f16.cuh#L26-L49).
4. `amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc)` — **false** for gfx1151; RDNA4-only gate. See [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454).
5. `amd_mfma_available(cc)` — CDNA only, skipped.
6. **Fallthrough: TILE kernel** at [fattn.cu:505](../ggml/src/ggml-cuda/fattn.cu#L505).

## The MMA_F16 kernel's host dispatcher says it supports RDNA 3

The kernel's own config dispatcher at [fattn-mma-f16.cuh:171-186](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L171-L186):

```cpp
if (amd_wmma_available(cc)) {
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
}
```

and `amd_wmma_available()` in [common.cuh:318-320](../ggml/src/ggml-cuda/common.cuh#L318-L320) returns true for both RDNA 3 and RDNA 4:

```cpp
static bool amd_wmma_available(const int cc) {
    return (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
}
```

So the RDNA path **in the host-side code** was there. That turned out to be misleading.

## What we tried

Two incremental changes, both on a dedicated branch at the time:

1. **Dispatcher gate relaxation** — widen the condition at [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454) to include RDNA 3.5:

   ```diff
   -if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && ...) {
   +if (amd_wmma_available(cc) && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) && ...) {
   ```

   Bench result: identical to baseline TILE (within noise). A dispatcher trace revealed why — all attention calls have `Q->ne[0] = 256` (head dim 256) on Qwen 3.6 and Qwen3-Coder-Next, and the block at line 454 has an additional `Q->ne[0] <= 128` guard. The patch put RDNA 3.5 inside the gate, but the inner head-dim check still excluded us. **Patch never fired.**

2. **Head-dim relaxation** — widen the inner guard from `<= 128` to `<= 256`, matching the MMA config table's explicit `(256, 256, ...)` RDNA entries in [fattn-mma-f16.cuh](../ggml/src/ggml-cuda/fattn-mma-f16.cuh).

   Bench result: **GPU hang with no device code**:

   ```
   ERROR: HIP kernel flash_attn_ext_f16 has no device code compatible with HIP arch 1300
   HW Exception by GPU node-1 ... reason: GPU Hang
   ```

## Outcome

The MMA_F16 kernel's *host-side* dispatcher advertises RDNA support, but the *device-side* template instantiations under [ggml/src/ggml-cuda/template-instances/](../ggml/src/ggml-cuda/template-instances/) are only compiled for the GPU targets upstream explicitly enumerates — RDNA 4 among AMD, plus NVIDIA archs, plus CDNA. `gfx1151` was never added to that list, so the .hsaco binaries shipped in the build have no entry for compute capability `1300`. Routing RDNA 3.5 into the MMA kernel at runtime therefore launches a kernel with no matching device code → GPU hang.

The RDNA4-only gate at line 454 was effectively defending against this reality, not expressing a perf choice.

## What an actual fix would look like

A real fix is a template-instance cmake change, not a dispatcher patch:

1. Identify the cmake list that drives which GPU archs get MMA_F16 device code emitted (likely under [ggml/src/ggml-cuda/template-instances/](../ggml/src/ggml-cuda/template-instances/) or the surrounding CMakeLists).
2. Add `gfx1151` to that list — possibly behind a guard that requires a compatible HIP toolchain (TheRock nightly compiles cleanly for it; official ROCm apt packages don't).
3. Rebuild. If compilation succeeds and the kernel runs, *then* a dispatcher patch becomes relevant — and only then can we bench whether MMA_F16 actually beats TILE on RDNA 3.5 (still unknown).

Adjacent unknowns worth flagging for whoever picks this up:

- The `Q->ne[1] * gqa_ratio_eff <= 8` carveout at [fattn.cu:473-474](../ggml/src/ggml-cuda/fattn.cu#L473-L474) routes short-batch cases back to TILE. For tg (single token), this always fires on Qwen 3.x regardless of head dim, so tg would stay on TILE even after the MMA path is unblocked.
- Quantized-KV viability under MMA (the original motivation from [kv-cache.md](kv-cache.md)) is still untested. The hypothesis that MMA has a dequant-free path is unverified.

## Upstreamability

A template-instance cmake addition for `gfx1151` is a plausible upstream contribution if it builds cleanly on stock ROCm. The dispatcher gate relaxation by itself is not — it's actively unsafe without the device code present.
