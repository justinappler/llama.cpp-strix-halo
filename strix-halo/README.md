# llama.cpp — Strix Halo fork

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with experimental changes targeting **AMD Strix Halo** (`gfx1151`, RDNA 3.5 iGPU + Zen 5 + 128 GB LPDDR5x unified memory), found in the Framework Desktop and similar systems.

The goal is a reproducible, benchmarked set of patches that meaningfully improve inference performance on this specific chip — primarily for agentic coding workloads, where long-context prompt processing and TTFT dominate the UX.

## Findings

| # | Finding | Impact | Status |
|--:|---|---|---|
| 1 | [Quantized KV cache collapses throughput at depth](kv-cache.md) | **17× pp @ d=16k** on Qwen 3.6 | Config fix only; no patch needed |
| 2 | [FA dispatcher gates RDNA3.5 out of MMA_F16 kernel](fa-dispatcher.md) | 1-line patch; bench pending | Branch `strix-halo/fa-mma-rdna35` |

See [NOTES.md](NOTES.md) for the initial survey of other possible optimization sites in the llama.cpp source (most not yet pursued).

## Benchmarks

- [Qwen 3.6 35B-A3B baseline on gfx1151](qwen3.6-baseline.md) — establishes production-config numbers and the f16-KV recovery.

## Branch organization

- `master` — tracks upstream `ggml-org/llama.cpp` plus this folder and a README banner.
- `strix-halo/<name>` — one branch per discrete patch, cleanly forkable.
- (Planned) `strix-halo-main` — integration branch composing validated patches.

## Build

See [server-configs Dockerfile](https://github.com/justinappler/server-configs) for a working multi-stage build against TheRock ROCm nightly that the upstream llama.cpp Dockerfile does not yet support on gfx1151. Official ROCm apt packages ship broken `gfx1151` `.hsaco` kernels — see [ROCm/ROCm#6042](https://github.com/ROCm/ROCm/issues/6042).

To use this fork's patches, point the Dockerfile's `LLAMACPP_VERSION` arg at a commit on the relevant `strix-halo/*` branch and rebuild.
