# llama.cpp - Strix Halo fork

> [!IMPORTANT]
> **Our work here is done.** This fork was retired on 2026-10-19 and no longer takes changes. **For AMD Strix Halo (`gfx1151`), use [upstream llama.cpp](https://github.com/ggml-org/llama.cpp).**
>
> Upstream has now absorbed or beaten everything this fork carried. The last patch, an RDNA3.5 MMQ tile retune, landed upstream as [PR #28195](https://github.com/ggml-org/llama.cpp/pull/28195) (`4dd6dd4b9`, 2026-10-14). What is left here was worth about 2-3% prefill at depth, which is not worth keeping a fork in sync for. See [What is left](#what-is-left).
>
> The repo stays up as a lab notebook. [`strix-halo/`](strix-halo/README.md) has every finding, every postmortem and the bench method, and most of it still applies to upstream on this chip. The rest of this README is frozen as of the retirement and describes the fork as it stood.

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that tunes inference for **AMD Strix Halo** - the `gfx1151` chip in machines like the Framework Desktop and other Ryzen AI Max boxes. It pairs an RDNA 3.5 integrated GPU with Zen 5 cores and one shared pool of LPDDR5x memory.

The target workload is **agentic coding**: long prompts, deep context, and time-to-first-token that you actually feel. That means prefill speed at depth matters more here than raw decode speed on a short prompt.

Everything in this fork is meant to be measured. A change stays only if a benchmark on real hardware says it should.

## Where things stand

Latest benchmark - Qwen 3.6 35B-A3B Q4_K_XL, ROCm 7.14.0 (production has since moved to 10.0.0, within noise - see [rocm-config.md](strix-halo/rocm-config.md#update-2026-09-25-rocm-1000-installed-from-apt)), f16/f16 KV cache, FlashAttention on, mean of 3 runs of build `000887b` (2026-09-25, same kernels as current master):

| context depth | prefill (tok/s) | decode (tok/s) |
| ------------: | --------------: | -------------: |
|             0 | 1341 | 52.7 |
|         2,048 | 1297 | 52.4 |
|         8,192 | 1163 | 51.0 |
|        16,384 | 1028 | 49.1 |

**Over 1000 tok/s of prefill at 16k of context, and decode above 49 tok/s at every depth.** Best measured at 8k and 16k of context and on decode (the 2026-08-02 build still holds d=0 and 2k). Full flags and history: [strix-halo/qwen3.6-baseline.md](strix-halo/qwen3.6-baseline.md).

**Upstream caught up with our biggest patch, and passed it.** For months our FlashAttention tile patch was the fork's largest single win, because upstream's faster "MMA" attention kernel lost on this chip at our head size (-11.6% at 16k when we last checked, on 2026-08-29). Upstream has since reworked that kernel and switched our chip onto it. We benched our patch against theirs head to head: **upstream's kernel is +11.2% faster at 16k**, and the gap grows with depth. So we dropped our patch. Details in [fa-mma-d256-26419.md](strix-halo/fa-mma-d256-26419.md#outcome-2026-09-25-upstream-mma-wins-guard-and-tile-patch-dropped).

Earlier in the year we also learned that the MMQ table - the patch this fork always credited with its ~27% prefill win - is worth only +2% to +4% on its own. Most of that 27% was the attention patch that upstream has now overtaken. Details in [findings.md](strix-halo/findings.md#two-caveats-worth-carrying-forward).

## What is actually patched

Small on purpose, and smaller than it used to be. The whole fork is **one change to one file**, plus documentation:

| What | Where | Why |
|------|-------|-----|
| **MMQ tile shape for RDNA3.5** | [mmq-config-rdna3-5.cuh](ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh) | Upstream ships a table for our chip but filled it with values copied from RDNA4 - a much larger, much higher-bandwidth GPU. We halve the tile (`nthreads` 256 -> 128, `I` 128 -> 64) so it fits gfx1151's register budget. Worth +2% to +4% prefill, last measured 2026-08-29. **Mostly upstream since 2026-10-14** ([#28195](https://github.com/ggml-org/llama.cpp/pull/28195)); see [What is left](#what-is-left). |

Two patches retired on 2026-09-25, both because upstream solved the same problem: the MoE tile-width cap (upstream now sizes MoE tiles from the average tokens per expert) and the FlashAttention tile config (upstream's MMA kernel is now faster).

Everything else here is upstream, plus the `strix-halo/` notebook and a deleted GitHub Actions directory (this fork does not run upstream's CI).

## The story so far

Roughly chronological, in plain terms. The full register with numbers and links is [strix-halo/findings.md](strix-halo/findings.md).

**We started by looking for configuration mistakes, and found a big one.** Quantizing the KV cache - normally a sensible memory saving - turned out to cost **17x** on prefill at 16k context on this chip. Quantizing the V cache is the expensive half. No code change needed; just do not do it. This remains the single largest effect anyone here has measured.

**Then we went after the matmul kernels, and this became the fork's long-running thread.** llama.cpp picks matmul tile sizes per GPU architecture, and gfx1151 kept inheriting settings meant for far larger AMD GPUs. We long credited that with about +27% prefill; the control later showed the table alone is worth +2% to +4%. That patch has now been rewritten three times as upstream restructured the code underneath it - twice because upstream deleted the functions it edited, and most recently because upstream added the exact per-architecture table we had been maintaining privately, then filled it with the wrong numbers. Each rewrite kept the same idea: smaller tiles, because this chip runs out of registers before it runs out of work.

**We chased FlashAttention down a dead end for about a month.** The theory was that gfx1151 was being locked out of a faster attention kernel. Three separate attempts - widening a dispatcher check, cherry-picking an upstream developer's work-in-progress branch, and hand-porting a register-layout trick - produced one abandoned patch, one measured 22% regression, and one debugging spiral that ended without a root cause. Upstream then settled the question in the opposite direction from where we were pushing: for our head size, the simpler "tile" kernel is genuinely the faster one. The only thing that survived was a single tuned constant in that tile kernel - until September 2026, when upstream reworked the MMA kernel and it overtook the tile kernel by 11% at 16k. The argument ended in the direction we had first pushed, just not by our hand.

**One patch quietly turned into a regression and we did not notice for five weeks.** A FlashAttention tuning port measured "flat" when it landed, so we stopped checking. A later change made it actively harmful - 3.5x worse prefill at depth - and it sat there. This is the origin of the [re-bench checklist](strix-halo/upstream.md#re-bench-checklist), and it is the main reason this fork writes down what was *measured* rather than what was *expected*.

**Most recently we stopped guessing and profiled the thing.** A kernel-level trace ([kernel-time-breakdown.md](strix-halo/kernel-time-breakdown.md)) finally answered where the time goes, and it re-priced the entire backlog:

- **Prefill at depth is becoming an attention problem, not a matmul problem.** Attention grows from 2% to 32% of prefill time as context deepens, while matmul shrinks from 58% to 38%. They cross around 13k of context. More matmul tuning has a shrinking ceiling.
- **Decode is essentially one kernel.** A single quantized matrix-vector kernel is 51% of decode time, called 161 times per token.
- **A promising-looking 16% GPU idle gap during decode is not free money.** It looked like launch overhead that graph capture would fix - but tracing showed graph capture is *already* on and working. That win was already banked.

## What is left

Nothing that justifies a fork. Diffing this fork's table against upstream after #28195 leaves:

- **J=96 for Q4_0, Q4_K, Q5_K, Q6_K and Q8_0.** Upstream left these on the wide tile on purpose, because they measured no gain on gfx1151.
- **A few edges of the same types:** J=128 for Q5_K and Q8_0, J=112 for Q6_K, and the fallback rows at J=48.
- **Every other quant type** (the IQ types, Q4_1, Q5_0/1, Q2_K, Q3_K, MXFP4, NVFP4, Q1_0/Q2_0). Upstream has not touched them, and this fork has never benched them on their own. If you run MXFP4 or IQ quants on gfx1151, that is the one place a small upstream PR might still pay, measured with `test-backend-ops perf` the way #28195 was.

The last head-to-head (2026-09-26, three interleaved cycles) put this fork +2.0% ahead at 8k of context and +2.8% at 16k, against upstream `9f70b2cec` with #28195 applied. The unfinished ideas in [strix-halo/backlog.md](strix-halo/backlog.md), such as an RDNA3.5 matrix-vector table for decode, are upstream work now, not fork work.

## How to work on this fork

Kept for the record; the fork no longer takes changes. One branch, `master`, tracking upstream plus our patches and the `strix-halo/` notebook. Each attempt is ideally **one commit**:

1. **Write the hypothesis first** as a new markdown file in `strix-halo/`. Link the lines you intend to change and state what you will measure before you measure it.
2. **Land the code change** as a single commit.
3. **Build and benchmark on real gfx1151 hardware.** Pin a full commit SHA, not a branch name, if Docker layer caching is involved.
4. **Keep it or revert it**, and either way annotate the doc with what the numbers said.

The point of step 4 is that this repo's history reads as *tried / measured / kept or reverted*. Docs accumulate even when patches do not - about half the `strix-halo/` folder is postmortems of things that did not work, and that is the half that saves the most time.

Two hard rules from [AGENTS.md](AGENTS.md) that apply to any agent working here: **never** push, open a pull request, or write a comment or reviewer reply on the user's behalf. Commits on this local fork are fine when asked.

## Keeping up with upstream

Upstream moves fast, and it has retired several of this fork's patches by solving the same problem better. That is the expected outcome, not a failure. [strix-halo/upstream.md](strix-halo/upstream.md) has the running record, the resync procedure, and the re-bench checklist to run afterward.

## Building for gfx1151

Official ROCm packages have shipped broken `gfx1151` kernel artifacts in some releases - see [ROCm/ROCm#6042](https://github.com/ROCm/ROCm/issues/6042). The fork's last builds used **ROCm 10.0.0** from AMD's apt repository; before that, the ROCm 7.14.0 release, after moving off TheRock nightlies in July 2026.

The multi-stage Docker build, the deployment playbooks, and the profiling harness are maintained separately from this repo, in a private deploy repo. Benchmarks quoted here were run through that harness; the bench flags themselves are always written out in full in the topic docs so the numbers can be reproduced without it.

## Map of `strix-halo/`

| File | What it is |
|------|------------|
| [findings.md](strix-halo/findings.md) | Every attempt, what it was worth, and its current status |
| [backlog.md](strix-halo/backlog.md) | What to try next, priced against the kernel profile |
| [upstream.md](strix-halo/upstream.md) | Upstream changes that affected us, resync procedure, re-bench checklist |
| [NOTES.md](strix-halo/NOTES.md) | Survey of tunable sites across the HIP / Vulkan / CPU backends |
| [README.md](strix-halo/README.md) | Index of every topic doc and postmortem |

Individual experiments each have their own file; start from the index.
