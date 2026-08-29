# Findings register

Every optimization attempt this fork has made on gfx1151, in the order they were tried, with what happened to each. This is the detailed version of the summary in the [root README](../README.md).

"Status" is the current state, not the state at the time. Several findings have been retired by upstream landing something equivalent or better; those rows say so and the linked doc keeps the measurement history.

|   # | Finding | What it was worth | Status |
| --: | ------- | ----------------- | ------ |
| 1 | [Quantized KV cache collapses throughput at depth](kv-cache.md) | **17x pp @ d=16k** on Qwen 3.6; V-quant is the dominant cost | **Live.** Config fix only, no patch needed. Run f16/f16 KV. |
| 2 | [FA dispatcher gates RDNA3.5 out of the MMA_F16 kernel](fa-dispatcher.md) | Attempted 1-line patch | **Abandoned.** MMA device code was never compiled for gfx1151. Superseded by #7, then closed by upstream PR #22880. |
| 3 | [UMA / `integrated = false`](uma-integrated.md) | Originally flagged as the likely biggest win | **Deprioritized after research.** The flag gates only small scratch buffers, not weight or KV traffic. PR #16308's author reported no perf impact. |
| 4 | [ROCm config flags: unroll-threshold + `HIPBLASLT_BATCHED=0`](rocm-config.md) | Community reported 2x pp on other models; **null** on Qwen 3.6 | **Split 2026-08-02.** The unroll-threshold flag is **retired** - the LLVM bug it worked around is fixed as of ROCm 7.12 and we run 7.14.0. `HIPBLASLT_BATCHED=0` stays. Both live in the deploy config, not this repo. |
| 5 | [MMQ tile/nwarp tuning (port of PR #21344)](mmq-rdna3_5.md) | **+27% pp @ d=0, +17% pp @ d=16k**; tg flat (measured 2026-05-14) | **Code dropped 2026-07-16.** Upstream [PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127) deleted all six functions it patched. Re-ported as #9. Doc kept for the measurement history - it is the only real A/B this tuning has ever had. |
| 6 | [rocWMMA FA tuning (port of PR #16827)](rocwmma-tuned.md) | Flat at landing, then **actively harmful** at D=256 (pp512@d=16k 244 vs 853 t/s with the flag off) | **Closed for good 2026-08-02.** Upstream [PR #26046](https://github.com/ggml-org/llama.cpp/pull/26046) deleted rocWMMA FlashAttention entirely - there is no flag left to flip. Doc kept as postmortem. |
| 7 | [FA MMA_F16 D=256 on RDNA3 (JG cherry-pick + guard widen)](jg-cuda-fa-rdna3-4.md) | **Regression** in production: pp512@d=16k 851 -> 660 t/s (-22.5%) at f16/f16 KV | **Superseded by upstream [PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880).** Upstream went the opposite way at D>128 - TILE, not MMA. Never promoted. |
| 8 | [Dense-aware MMQ + TILE FA D=256 follow-up](pp-rdna3_5-tile-mmq.md) | **+6.3% pp @ d=16k** vs the 2026-05-14 build; d=0 flat, tg in noise | **Split.** The TILE FA half is **live on master** and is now the fork's highest-leverage patch (FA is 32% of prefill time at 16k depth). The MMQ half folded into #9 as the `J_max` cap. |
| 9 | [RDNA3.5 MMQ config table](mmq-rdna3_5-config-table.md) | **+27.6% to +29.2% prefill**, decode unchanged (0.0% to +0.2%) - measured against a matched control, 2026-08-29 | **Live on master, and finally attributed.** The port-off A/B was run at last as arms A/C of [fa-mma-d256-26419.md](fa-mma-d256-26419.md#outcome): same upstream base `c841aeeb8`, only our three patched files differing. Gain is uniform across all four depths, and the decode control came back flat to within 0.2%, so the comparison is single-variable. Confirms the long-carried "+27%" claim that had only ever been inferred from before/after builds. |
| 10 | [Routed-MoE ncols picker (port of PR #24546)](mmq-moe-ncols-picker.md) | Flat - `J=48` and `J=64` are equivalent at `ub=2048` | **Reverted 2026-07-17.** The static `J=48` cap from #8 does the same job with less code. |

## Two caveats worth carrying forward

**Finding #9 has never had its control run.** Every number attached to it is a bundle delta - builds that moved upstream commits, ROCm versions and the port all at once. The 2026-08-02 re-bench bounds the *downside* (prefill flat after the port was rewritten onto upstream's file, where losing it would have cost 27-37%), but nothing has ever measured what the port is worth on top of current upstream. Until someone builds with `git checkout upstream/master -- ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`, "our patch is worth X%" is not a claim these numbers support. See [mmq-rdna3_5-config-table.md § Outcome](mmq-rdna3_5-config-table.md#outcome).

**Upstream now ships its own RDNA3.5 MMQ table, and it is untuned.** [PR #26199](https://github.com/ggml-org/llama.cpp/pull/26199) (merged 2026-07-28) added `mmq-config-rdna3-5.cuh` with values copied verbatim from rdna4. This fork's patch is now a retune of that file rather than a new file of its own. If the patch is ever dropped in a future sync, gfx1151 silently goes back to rdna4's wide tiles - the exact shape the 2026-04 A/Bs beat by +27%/+37%.

## How #6 got away from us

Finding #6 was recorded as "flat at landing" on 2026-04-19 and not re-validated. By 2026-04-27 it was costing 3.5x on prefill at depth, and that went unnoticed for about five weeks. That single episode is why the [re-bench checklist](upstream.md#re-bench-checklist) exists and why the findings above lead with what was *measured* rather than what was *expected*.
