# Backlog - what to try next

Ordered by expected return, **re-priced against the 2026-07-18 kernel profile** ([kernel-time-breakdown.md](kernel-time-breakdown.md)). Before that profile the ordering was guesswork about where time went; now it is not.

Cost is calendar effort: **S** = hours, **M** = a day or two, **L** = a week, **XL** = multi-week. Benefit is a measured or plausible pp/tg delta on the Qwen 3.6 35B-A3B workload unless noted.

`#n` tags refer to numbered sections in [NOTES.md](NOTES.md), a survey of tunable sites across the HIP / Vulkan / CPU backends.

## Where the time actually goes

This is the table that should drive every decision below:

| phase | MMQ | FA (TILE) | MMVQ | gdn/ssm | other |
|---|---:|---:|---:|---:|---:|
| pp512 @ d=0 | **58%** | 2% | - | 12% | 28% |
| pp512 @ d=16,384 | 38% | **32%** | - | 9% | 21% |
| tg @ d=0 | - | 0.7% | **77%** | 1.8% | 20% |
| tg128 @ d=16,384 | - | 8.5% | **71%** | 1.7% | 19% |

Read it this way: **prefill at depth is becoming an attention problem, not a matmul problem**, and **decode is essentially one kernel**. FA's share grows with depth while MMQ's shrinks, and they cross around d~13k at `ub=2048`.

## Ranked

| # | Item | Cost | Benefit | Why here |
|--:|------|:----:|:-------:|----------|
| 1 | **FA TILE D=256 tuning at depth.** The fork already overrides one knob for RDNA3.5 (`nbatch_K` 128 -> 64, [fattn-tile.cuh:315](../ggml/src/ggml-cuda/fattn-tile.cuh#L315)); `nthreads`, `occupancy` and `nbatch_fa` were never swept. `flash_attn_tile<256,256,4,8>` burns 159ms of the 501ms pp512@16k budget across just 10 dispatches. | M | **M-L** | The profile's clearest signal. FA is 32% of prefill at depth and rising, the override we ship is one hand-picked constant, and nothing else on this list has that much measured headroom. |
| 2 | **MMQ port-off A/B + `occupancy` sweep.** Build once with upstream's table restored (`git checkout upstream/master -- ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`), once with ours, once with ours at `occupancy=4`. | S | attribution + S | Finding #9's gain has never been attributed and `occupancy=2` shipped as a guess. This is three builds and it unblocks upstreaming. Do it before any further MMQ tuning - a broad table sweep is *not* worth it at a 38% and shrinking share. |
| 3 | **#2 MMVQ RDNA3.5 table, `Q8_0` first.** `mul_mat_vec_q<Q8_0>` alone is **51% of decode busy time** at 161 calls/token - the dense and gdn projections, not the experts. Start with a `Q8_0`-only whitelist and `nwarps` in {2, 4}. [mmvq.cu:77](../ggml/src/ggml-cuda/mmvq.cu#L77), [:93](../ggml/src/ggml-cuda/mmvq.cu#L93), [:348](../ggml/src/ggml-cuda/mmvq.cu#L348). | M | M | Decode is one kernel, and this is the kernel. The naive version of this already failed once ([mmvq-rdna3_5.md](mmvq-rdna3_5.md)) by joining RDNA3_0's table wholesale; a narrow whitelist is the corrected approach, not a retry. |
| 4 | **#3 mmf `src1_ncols` gate asymmetry.** RDNA3.0 caps at `>8`, RDNA3.5 inherits the generic `>16`. [mmf.cu:169](../ggml/src/ggml-cuda/mmf.cu#L169). | S | S-M | A single-line threshold with a cheap bench. Low ceiling, but it costs almost nothing to settle. |
| 5 | **#8 Zen 5 CPU backend variant.** [CMakeLists.txt:379](../ggml/src/CMakeLists.txt#L379) only has `zen4`. Zen 5 has a native 512-bit datapath against Zen 4's double-pumped 256-bit. | S | S-M | Off the GPU hot path, so ROI is bounded by definition. Cheap enough to try anyway. |
| 6 | **#9 `madvise(MADV_HUGEPAGE)` after mmap.** [llama-mmap.cpp:437-467](../src/llama-mmap.cpp#L437-L467). TLB pressure is real at 60B+ params in 128 GB of LPDDR5X. | S | S (load-time; small tg at best) | One-line hint, low risk, low reward. |
| 7 | **#10 Commit a Strix Halo bench to `benches/`.** | S | infra | Not a perf win. Gives a defensible regression signal and an upstream talking point. Worth doing once one more patch has landed to anchor the baseline. |

## Mostly closed, one loose end

- **HIP graphs at decode** ([hip-graphs.md](hip-graphs.md)). The GPU is idle 16% of decode wall time across ~1,595 dispatches per token, which looked like an obvious launch-overhead win. Traced 2026-07-18: **graphs are already engaged in production** - after the final `hipGraphInstantiate` there are zero further `hipLaunchKernel` calls, so steady-state decode is 100% graph replay. There is no "flip the flag, get +15% tg" win sitting on the table; that fix already shipped and the 16% was measured with it on.

  **Still open:** whether 16% is the floor of hipGraph replay on ROCm 7.14, or whether graphs are already buying something and simply not closing the gap. Settling it needs a throwaway `-DGGML_HIP_GRAPHS=OFF` build, which no existing build-arg passthrough covers. Low priority - the answer changes nothing we would ship, it only tells us whether to keep looking here.

## Vulkan-only - relevant only if we switch backends

| # | Item | Cost | Benefit | Note |
|--:|------|:----:|:-------:|------|
| V1 | **#4 `AMD_RDNA3_5` architecture class.** gfx1151 is bucketed as `AMD_RDNA3` despite 40 vs 96 CUs and 256 vs 960 GB/s of bandwidth. [ggml-vulkan.cpp:270-279](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L270-L279). | M | unknown, plausibly L | Unlocks dedicated warptile and FA occupancy paths. |
| V2 | **#5 FA `limit_occupancy_shmem` re-tune.** [ggml-vulkan.cpp:2990-2994](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2990-L2994). The heuristic was guessed and tested on RDNA2; Strix Halo shares LLC with the CPU. | S | S-M | Bundle with V1. |
| V3 | **#6/#7 UMA allocation (`prefer_host_memory` + `HostCached`).** [ggml-vulkan.cpp:2799-2808](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2799-L2808). `DeviceLocal` is a ~512 MB GART window unless the BIOS is reconfigured, so large models always fall back. | S-M | S (mostly load-time) | |

## Dead ends - documented, not pursued

Do not re-attempt these without reading the linked postmortem first. Each one cost real time.

- **MMVQ "join RDNA3_0's table"** ([mmvq-rdna3_5.md](mmvq-rdna3_5.md)). Routing RDNA3.5 to RDNA3_0's `nwarps=8` table regressed tg128 by 18% at d=0 and 38% at d=16k - register-pressure spill on RDNA3.5's smaller LLC. Reverted. Backlog item 3 above is the corrected version, not a retry.
- **FA MMA_F16 on gfx1151** ([fa-dispatcher.md](fa-dispatcher.md)). Blocked on an RDNA3 unpacked-WMMA register-layout bug that also killed upstream [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063). **Closed 2026-05-14**: upstream [PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880) shipped JG's chain and deliberately kept TILE, not MMA, for RDNA3 at D>128. Settled. D<=128 gets the new mma FA kernel for free.
- **OPSEL-paired half2 accumulator port** ([fa-rdna3-opsel-pair.md](fa-rdna3-opsel-pair.md)). The structural fix for Finding #7's f32 accumulator ceiling. Prototyped 2026-05-04, abandoned the same day: `test-backend-ops` failed en masse on D=256 numerics and debugging turned into in-kernel printf archaeology without isolating the cause. The postmortem lists four candidate failures and the unit-test scaffolding a retry would need before anyone touches `fattn-mma-f16.cuh` again.
- **UMA / `integrated = false`** ([uma-integrated.md](uma-integrated.md)). Gates only small scratch buffers, not weight or KV traffic.
- **ROCm config flags** ([rocm-config.md](rocm-config.md)). Community reported 2x pp on gpt-oss-120b; null on Qwen 3.6. Kept on anyway as cheap insurance.
