# Flash-attention dispatcher gates RDNA 3.5 out of the MMA kernel

## Symptom

On `gfx1151` with default build flags (`GGML_HIP_ROCWMMA_FATTN=OFF`), flash-attention runs on the generic **TILE** fallback kernel. TILE has no efficient quantized-KV path, which makes its throughput collapse at depth (see [kv-cache.md](kv-cache.md)).

The root cause is dispatcher logic in [fattn.cu](../ggml/src/ggml-cuda/fattn.cu), not a missing kernel.

## Dispatch trace for gfx1151

Reading [ggml/src/ggml-cuda/fattn.cu:307-505](../ggml/src/ggml-cuda/fattn.cu#L307-L505) for RDNA 3.5:

1. `turing_mma_available(cc)` — NVIDIA Turing+, skipped.
2. `volta_mma_available(cc)` — NVIDIA Volta, skipped.
3. `ggml_cuda_should_use_wmma_fattn(cc)` — skipped because the build has `GGML_HIP_ROCWMMA_FATTN=OFF`. See [fattn-wmma-f16.cuh:26-49](../ggml/src/ggml-cuda/fattn-wmma-f16.cuh#L26-L49).
4. `amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc)` — **false** for gfx1151; RDNA4-only gate. See [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454).
5. `amd_mfma_available(cc)` — CDNA only, skipped.
6. **Fallthrough: TILE kernel** at [fattn.cu:505](../ggml/src/ggml-cuda/fattn.cu#L505).

## The MMA_F16 kernel already supports RDNA 3

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

So the RDNA path inside the MMA kernel was already there — only the top-level dispatcher refused to route RDNA 3.5 into it.

## Patch

Branch: [`strix-halo/fa-mma-rdna35`](../../../tree/strix-halo/fa-mma-rdna35). One-line change to [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454):

```diff
-if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && ...) {
+if (amd_wmma_available(cc) && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) && gqa_opt_applies && ...) {
```

## Open questions

- Does the MMA_F16 path tolerate quantized K/V on RDNA 3.5, or does it also fall back to dequantized compute? If it works, [kv-cache.md](kv-cache.md)'s f16 recommendation may be revisitable.
- The `Q->ne[1] * gqa_ratio_eff <= 8` carveout at [fattn.cu:473-474](../ggml/src/ggml-cuda/fattn.cu#L473-L474) routes short-batch cases back to TILE. On RDNA 3.5 this may or may not be the right crossover — worth benchmarking independently.

## Status

- Patch committed on branch.
- **Not yet benchmarked.** Requires rebuilding the llamacpp container with `LLAMACPP_VERSION` pointed at this branch's HEAD.
- Needs A/B against baseline across `{f16, q8_0/q4_0}` KV × `{0, 2k, 8k, 16k}` depth to confirm no regression and measure the long-context gain.

## Upstreamability

Probably not PR-ready as-is — upstream will want benchmarks across multiple RDNA 3.5 systems and multiple models/head-dims. Fine as a fork-local patch; reconsider upstreaming after we have coverage.
