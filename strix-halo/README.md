# llama.cpp — Strix Halo fork

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with experimental changes targeting **AMD Strix Halo** (`gfx1151`, RDNA 3.5 iGPU + Zen 5 + 128 GB LPDDR5x unified memory), found in the Framework Desktop and similar systems.

The goal is a reproducible, benchmarked set of patches that meaningfully improve inference performance on this specific chip — primarily for agentic coding workloads, where long-context prompt processing and TTFT dominate the UX.

## Findings

| # | Finding | Impact | Status |
|--:|---|---|---|
| 1 | [Quantized KV cache collapses throughput at depth](kv-cache.md) | **17× pp @ d=16k** on Qwen 3.6; V-quant is the dominant cost | Config fix only; no patch needed |
| 2 | [FA dispatcher gates RDNA3.5 out of MMA_F16 kernel](fa-dispatcher.md) | Attempted 1-line patch; **abandoned** | See doc — blocked on MMA device code not compiled for gfx1151 |

See [NOTES.md](NOTES.md) for the initial survey of other possible optimization sites in the llama.cpp source (most not yet pursued).

## Benchmarks

- [Qwen 3.6 35B-A3B baseline on gfx1151](qwen3.6-baseline.md) — establishes production-config numbers and the f16-KV recovery.

## Approach: one branch, one commit at a time

Single working branch: `master` on the fork (this repo). It tracks `ggml-org/llama.cpp` upstream plus this `strix-halo/` docs folder plus any accumulated, validated patches.

Rather than maintain a separate branch per patch + an integration branch, each optimization attempt is a single commit landed on master. The workflow per attempt:

1. **Write up the hypothesis** in a new markdown doc under `strix-halo/` before touching code. Link the source lines you plan to change, state what you expect to measure.
2. **Make the change** as one commit on master.
3. **Build & benchmark** via the server-configs Dockerfile (`LLAMACPP_VERSION=<sha>`). Always pin to a SHA, never a branch name — Docker caches the git-clone layer by command string.
4. **Decide**: if the bench shows a real gain (outside noise, across the matrix of depths/quants we care about), the commit stays. If it's null or negative, revert the commit and leave the doc as a record of the dead end, with the reason annotated at the bottom.

This keeps the history as a log of "tried this, here's what happened" rather than a pile of speculative branches. Docs accumulate even when patches don't — we've already gotten two concrete findings (KV quant, FA dispatcher dead-end) out of exactly this rhythm.

## Keeping up with upstream

Upstream `ggml-org/llama.cpp` moves daily. Resync cadence is roughly every few days.

```bash
# Fetch upstream (origin = ggml-org/llama.cpp)
git fetch origin

# Rebase fork master onto upstream. Docs-only changes shouldn't conflict;
# if a patch commit conflicts with an upstream change that supersedes it,
# drop the commit during rebase and annotate the doc accordingly.
git checkout master
git rebase origin/master
git push -f strix-halo master          # strix-halo = justinappler/llama.cpp-strix-halo
```

## Build

See [server-configs Dockerfile](https://github.com/justinappler/server-configs) for a working multi-stage build against TheRock ROCm nightly that the upstream llama.cpp Dockerfile does not yet support on gfx1151. Official ROCm apt packages ship broken `gfx1151` `.hsaco` kernels — see [ROCm/ROCm#6042](https://github.com/ROCm/ROCm/issues/6042).

Point `LLAMACPP_VERSION` at a specific commit SHA on this fork's master (or on upstream) to rebuild. Branch names work but Docker caches the clone layer by command string — always use SHA when iterating.
