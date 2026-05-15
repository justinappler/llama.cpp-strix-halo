# UMA / `integrated = false` — researched, deprioritized

**Status: researched, no patch.** Was flagged in [NOTES.md #1](NOTES.md#1-zero-copy-for-integrated-gpus-is-globally-disabled) as the likely-biggest Strix Halo win. After digging into what the flag actually gates, the Jetson precedent, and the HIP allocator behavior, the expected upside on gfx1151 looks narrow. Moving it below the other NOTES.md items.

## What `integrated = true` actually gates

Two consumers in [ggml-cuda.cu](../ggml/src/ggml-cuda/ggml-cuda.cu):

1. **[line 4085](../ggml/src/ggml-cuda/ggml-cuda.cu#L4085)** — a debug-only `assert` widening the set of acceptable input buffer types. Zero runtime effect in release builds (`GGML_UNUSED(integrated)` on line 4089).

2. **[line 5125](../ggml/src/ggml-cuda/ggml-cuda.cu#L5125)** — `device_supports_buft` returns true for the CUDA **host** buffer type (allocated with `hipHostMalloc` on HIP) in addition to the device buffer type. That's the one real effect: it lets the backend scheduler place tensors in pinned host memory and feed them directly to kernels, avoiding a host→device stage copy.

It does **not** change where model weights or KV live. Those use the device buffer type (`hipMalloc`) regardless of this flag.

## Why upstream disabled it — #16308

The hardcoded `false` was introduced in [PR #16308](https://github.com/ggml-org/llama.cpp/pull/16308), fixing [issue #15034](https://github.com/ggml-org/llama.cpp/issues/15034): Gemma 3n produced garbled output on a Jetson Orin Nano when `integrated=true`. Root cause was never identified; the flag is gated off as a "temporary" workaround.

Critical data point from the PR description: *"disabling the `integrated` flag seems to neither affect performance nor memory usage on Jetson."* Another UMA system saw zero pp impact from flipping this — that's strong prior-against the "single biggest win" framing.

## Why it's narrow on Strix Halo specifically

In our workload (`-ngl 999`, Qwen 3.6 Q4_K_XL, 128 GB LPDDR5X, no CPU offload):

- **Weights** go into the CUDA device buffer type at load. On HIP, `hipMalloc` allocates from the GART carve-out — physically the same LPDDR5X as host memory, but via device-mapped pages. Flipping the `integrated` flag doesn't change this allocation path.
- **KV cache** same story — device buffer type.
- **What the `cuda_host` buffer type holds** on Strix Halo at runtime (from baseline bench log): `CUDA_Host output buffer ~1 MB` (logits), `CUDA_Host compute buffer ~31 MB` (scratch). Small, not on the per-layer hot path.

So the flag's effect is gating a ~32 MB staging path, not a weight-traffic path. The doc's original framing ("pure waste of bandwidth of weights") assumed a dGPU VRAM≠RAM model that doesn't match the HIP APU allocator.

## Alternative UMA lever already exists

HIP already has a separate UMA env var: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` at [ggml-cuda.cu:122](../ggml/src/ggml-cuda/ggml-cuda.cu#L122), which routes allocations through `hipMallocManaged` instead of `hipMalloc`. Distinct mechanism from the `integrated` flag, independent opt-in. Not pursued here; worth a standalone bench if UMA becomes the focus again.

## Prior attempts in git log

- [`eb3949938`](https://github.com/ggml-org/llama.cpp/commit/eb3949938) added the flag, reading it from `prop.integrated`.
- [`9d0882840`](https://github.com/ggml-org/llama.cpp/commit/9d0882840) (PR #16308) hardcoded `false` as a #15034 workaround.
- [`909072abc`](https://github.com/ggml-org/llama.cpp/commit/909072abc) later fixed an unrelated UMA-detection path (different site at [ggml-cuda.cu:4675](../ggml/src/ggml-cuda/ggml-cuda.cu#L4675), not the one this doc is about).

No one has attempted a conditional re-enable gated on arch or runtime detection.

## Recommendation

Deprioritize. The hypothesis that re-enabling `integrated` is a pp win on Strix Halo rests on assumptions that don't hold on the HIP path. Revisit only if the remaining NOTES.md items null out.

If revisited: the narrow-fix framing (gate on `hipDeviceGetAttribute(integrated)==1`, or an env var) is still a valid one-line patch — but expect a small delta, and bench `CUDA_Host compute buffer` sizes in the log first to confirm there's anything meaningful flowing through that path at all.
