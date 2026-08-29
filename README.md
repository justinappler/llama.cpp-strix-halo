# llama.cpp - Strix Halo fork

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that tunes inference for **AMD Strix Halo** - the `gfx1151` chip in machines like the Framework Desktop and other Ryzen AI Max boxes. It pairs an RDNA 3.5 integrated GPU with Zen 5 cores and one shared pool of LPDDR5x memory.

The target workload is **agentic coding**: long prompts, deep context, and time-to-first-token that you actually feel. That means prefill speed at depth matters more here than raw decode speed on a short prompt.

Everything in this fork is meant to be measured. A change stays only if a benchmark on real hardware says it should.

## Where things stand

Latest production benchmark - Qwen 3.6 35B-A3B Q4_K_XL, ROCm 7.14.0, f16/f16 KV cache, FlashAttention on, build `b73cfa4` (2026-08-02):

| context depth | prefill (tok/s) | decode (tok/s) |
| ------------: | --------------: | -------------: |
|             0 | 1455 | 51.4 |
|         2,048 | 1304 | 51.0 |
|         8,192 | 1138 | 49.7 |
|        16,384 |  986 | 47.9 |

**About 986 tok/s of prefill at 16k of context, and decode that barely sags across the whole depth range.** Best measured on every axis so far. Full flags and history: [strix-halo/qwen3.6-baseline.md](strix-halo/qwen3.6-baseline.md).

Worth knowing how to read that: versus the previous build, **prefill is flat and decode is up about 3-4%**. The decode gain is almost certainly upstream's, not ours - nothing this fork patches can move decode. The flat prefill is the good news for our side: this build rewrote the matmul patch onto a new upstream file, and losing that tuning would have cost 27-37%, so flat means it landed intact.

**We finally ran the control, and it overturned our headline claim.** For four months this section carried a caveat that we had never built with our patches removed. On 2026-08-29 we built clean upstream and our tree from the same commit, same host, same session: **the three patches together are worth +26.6% to +28.8% of prefill, at every depth, with decode flat to within 0.2%.** Then we built a third time with *only* the MMQ table change - the patch this fork has always credited with the win - and it came back at **+2.1% to +4.2%**, about a tenth of the total.

So the fork is worth roughly what we said, but not for the reason we said. Every previous "+27% from the MMQ retune" was a bundle delta wearing one patch's name. The `J_max` cap measured flat back in Finding #10, which points at the FlashAttention tile patch for most of the remainder - not yet isolated. Details in [findings.md](strix-halo/findings.md#two-caveats-worth-carrying-forward).

## What is actually patched

Small on purpose. The whole fork is **three changes to three files**, plus documentation:

| What | Where | Why |
|------|-------|-----|
| **MMQ tile shape for RDNA3.5** | [mmq-config-rdna3-5.cuh](ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh) | Upstream ships a table for our chip but filled it with values copied from RDNA4 - a much larger, much higher-bandwidth GPU. We halve the tile (`nthreads` 256 -> 128, `I` 128 -> 64) so it fits gfx1151's register budget. |
| **Smaller tiles for MoE experts** | [mmq.cuh](ggml/src/ggml-cuda/mmq.cuh#L1478) | Wide tiles help dense matmuls but waste work on mixture-of-experts routing, where each expert only covers a slice of the rows. Five lines that cap the tile width for expert dispatch. |
| **FlashAttention tile config at D=256** | [fattn-tile.cuh](ggml/src/ggml-cuda/fattn-tile.cuh#L315) | One constant (`nbatch_K` 128 -> 64) for the attention kernel our production model uses. Currently the fork's highest-leverage patch: attention is 32% of prefill time at 16k depth. |

Everything else here is upstream, plus the `strix-halo/` notebook and a deleted GitHub Actions directory (this fork does not run upstream's CI).

## The story so far

Roughly chronological, in plain terms. The full register with numbers and links is [strix-halo/findings.md](strix-halo/findings.md).

**We started by looking for configuration mistakes, and found a big one.** Quantizing the KV cache - normally a sensible memory saving - turned out to cost **17x** on prefill at 16k context on this chip. Quantizing the V cache is the expensive half. No code change needed; just do not do it. This remains the single largest effect anyone here has measured.

**Then we went after the matmul kernels, and this became the fork's long-running thread.** llama.cpp picks matmul tile sizes per GPU architecture, and gfx1151 kept inheriting settings meant for far larger AMD GPUs. Correcting that was worth about **+27% prefill**. That patch has now been rewritten three times as upstream restructured the code underneath it - twice because upstream deleted the functions it edited, and most recently because upstream added the exact per-architecture table we had been maintaining privately, then filled it with the wrong numbers. Each rewrite kept the same idea: smaller tiles, because this chip runs out of registers before it runs out of work.

**We chased FlashAttention down a dead end for about a month.** The theory was that gfx1151 was being locked out of a faster attention kernel. Three separate attempts - widening a dispatcher check, cherry-picking an upstream developer's work-in-progress branch, and hand-porting a register-layout trick - produced one abandoned patch, one measured 22% regression, and one debugging spiral that ended without a root cause. Upstream then settled the question in the opposite direction from where we were pushing: for our head size, the simpler "tile" kernel is genuinely the faster one. The only thing that survived is a single tuned constant in that tile kernel.

**One patch quietly turned into a regression and we did not notice for five weeks.** A FlashAttention tuning port measured "flat" when it landed, so we stopped checking. A later change made it actively harmful - 3.5x worse prefill at depth - and it sat there. This is the origin of the [re-bench checklist](strix-halo/upstream.md#re-bench-checklist), and it is the main reason this fork writes down what was *measured* rather than what was *expected*.

**Most recently we stopped guessing and profiled the thing.** A kernel-level trace ([kernel-time-breakdown.md](strix-halo/kernel-time-breakdown.md)) finally answered where the time goes, and it re-priced the entire backlog:

- **Prefill at depth is becoming an attention problem, not a matmul problem.** Attention grows from 2% to 32% of prefill time as context deepens, while matmul shrinks from 58% to 38%. They cross around 13k of context. More matmul tuning has a shrinking ceiling.
- **Decode is essentially one kernel.** A single quantized matrix-vector kernel is 51% of decode time, called 161 times per token.
- **A promising-looking 16% GPU idle gap during decode is not free money.** It looked like launch overhead that graph capture would fix - but tracing showed graph capture is *already* on and working. That win was already banked.

## What is next

Full list with cost and rationale: [strix-halo/backlog.md](strix-halo/backlog.md). The top three:

1. **Tune the FlashAttention tile kernel at depth.** We ship one hand-picked constant and never swept the others. This is where the measured headroom is.
2. **Run the control we skipped.** Three builds settles whether our MMQ retune is actually earning its keep, and unblocks proposing it upstream.
3. **A dedicated matrix-vector table for this chip**, starting with the one quantization type that dominates decode. A naive version of this failed before; the corrected approach is narrower.

## How to work on this fork

One branch, `master`, tracking upstream plus our patches and the `strix-halo/` notebook. Each attempt is ideally **one commit**:

1. **Write the hypothesis first** as a new markdown file in `strix-halo/`. Link the lines you intend to change and state what you will measure before you measure it.
2. **Land the code change** as a single commit.
3. **Build and benchmark on real gfx1151 hardware.** Pin a full commit SHA, not a branch name, if Docker layer caching is involved.
4. **Keep it or revert it**, and either way annotate the doc with what the numbers said.

The point of step 4 is that this repo's history reads as *tried / measured / kept or reverted*. Docs accumulate even when patches do not - about half the `strix-halo/` folder is postmortems of things that did not work, and that is the half that saves the most time.

Two hard rules from [AGENTS.md](AGENTS.md) that apply to any agent working here: **never** push, open a pull request, or write a comment or reviewer reply on the user's behalf. Commits on this local fork are fine when asked.

## Keeping up with upstream

Upstream moves fast, and it has retired several of this fork's patches by solving the same problem better. That is the expected outcome, not a failure. [strix-halo/upstream.md](strix-halo/upstream.md) has the running record, the resync procedure, and the re-bench checklist to run afterward.

## Building for gfx1151

Official ROCm packages have shipped broken `gfx1151` kernel artifacts in some releases - see [ROCm/ROCm#6042](https://github.com/ROCm/ROCm/issues/6042). This fork currently builds against the **ROCm 7.14.0 release**, having moved off TheRock nightlies in July 2026.

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
