# llama.cpp — Strix Halo fork

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with experimental changes for **AMD Strix Halo** (`gfx1151`: RDNA 3.5 iGPU, Zen 5, unified LPDDR5x), targeting machines like the Framework Desktop with Ryzen AI Max.

## Strix Halo: goals

The aim is a reproducible, benchmarked set of changes that improve inference on this chip—especially **agentic coding** workloads where long-context prompt processing and time-to-first-token dominate.

## Where this fork stands

> [!WARNING]
> **These numbers predate the 2026-07-16 upstream rebase and no longer describe a build that exists.** That sync absorbed 234 upstream commits (including [PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127), which deleted the MMQ patch this table's prefill numbers depend on), switched to ROCm 7.14.0, and re-ported the MMQ tuning as a config table ([Finding #9](#strix-halo-findings)). The re-bench is **pending**; treat the table below as the last-known-good target to beat, not as current.

Last production bench, Qwen 3.6 35B-A3B Q4_K_XL on gfx1151 (commit `3511e7d`, TheRock `7.13.0a20260514`, f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3`):

| depth   | pp512 (t/s)      | tg128 (t/s)   |
| ------: | ---------------: | ------------: |
|       0 | 1350.31 ± 7.27   | 47.25 ± 0.06  |
|   2,048 | 1261.93 ± 4.56   | 46.96 ± 0.15  |
|   8,192 | 1085.56 ± 16.49  | 45.79 ± 0.15  |
|  16,384 |  916.76 ± 4.62   | 44.25 ± 0.15  |

Headline: **~917 t/s prefill at 16k depth**, **~44 t/s decode through the depth axis**. Full bench config and the recovery story from earlier (pre-MMQ-port) baselines are in [strix-halo/qwen3.6-baseline.md](strix-halo/qwen3.6-baseline.md); the A/B that validated the MMQ base is in [strix-halo/mmq-rdna3_5.md § Post-rebase re-bench](strix-halo/mmq-rdna3_5.md#post-rebase-re-bench-2026-05-14-build-e4184dbb), and the dense-MMQ/TILE FA follow-up is in [strix-halo/pp-rdna3_5-tile-mmq.md](strix-halo/pp-rdna3_5-tile-mmq.md).

The bench plan that replaces these numbers is in [strix-halo/mmq-rdna3_5-config-table.md § Bench plan](strix-halo/mmq-rdna3_5-config-table.md#bench-plan) — two runs, port off then on, so the re-port is attributable separately from the ROCm and upstream bumps.

## Strix Halo findings

|   # | Finding                                                                                  | Impact                                                                                                                 | Status                                                                                                                                                                                 |
| --: | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1 | [Quantized KV cache collapses throughput at depth](strix-halo/kv-cache.md)               | **17× pp @ d=16k** on Qwen 3.6; V-quant is the dominant cost                                                           | Config fix only; no patch needed                                                                                                                                                       |
|   2 | [FA dispatcher gates RDNA3.5 out of MMA_F16 kernel](strix-halo/fa-dispatcher.md)         | Attempted 1-line patch; **abandoned**                                                                                  | See doc — blocked on MMA device code not compiled for gfx1151                                                                                                                          |
|   3 | [UMA / `integrated = false`](strix-halo/uma-integrated.md)                               | Originally flagged as likely biggest win; research says otherwise                                                      | **Researched, deprioritized** — narrow on HIP APUs                                                                                                                                     |
|   4 | [ROCm config flags: unroll-threshold + `HIPBLASLT_BATCHED=0`](strix-halo/rocm-config.md) | Community reports 2× pp on other models; null on Qwen 3.6                                                              | **Bench null, kept on** as AMD-recommended safety nets                                                                                                                                 |
|   5 | [MMQ tile/nwarp tuning for gfx1151 (port of PR #21344)](strix-halo/mmq-rdna3_5.md)       | **+27% pp @ d=0, +17% pp @ d=16k** on Qwen 3.6 Q4_K_XL vs no-port; tg128 flat (last measured 2026-05-14)               | **Code dropped on the 2026-07-16 upstream rebase.** Upstream [PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127) deleted all six functions this patch edited. Re-ported as a config table — see Finding #9. Doc retained for the measurement history. |
|   6 | [rocWMMA FA tuning for gfx1151 (port of PR #16827)](strix-halo/rocwmma-tuned.md)         | Flat at landing (2026-04-19) on Qwen 3.6 D=256; actively harmful at D=256 by 2026-04-27 (pp512@d=16k 244 vs 853 t/s with flag off). lhl's +35-65% D≤128 numbers were untested on this fork. | **Retired on 2026-05-14 upstream rebase.** Flag was already `GGML_HIP_ROCWMMA_FATTN=OFF` in production; upstream [PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880) routes RDNA3 D>128 to the TILE kernel (not WMMA), so the rocWMMA path has no production effect on Qwen 3.6 A3B. Doc kept as postmortem. |
|   7 | [FA MMA_F16 D=256 on RDNA3 (JG `cuda-fa-rdna3-4` cherry-pick + 1-line guard widen)](strix-halo/jg-cuda-fa-rdna3-4.md) | Regression at f16/f16 KV in production (pp512@d=16k 851 → 660 t/s, −22.5%) when held in 2026-05. Held branch was a cherry-pick of JG's WIP. | **Superseded by upstream [PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880) (merged 2026-05-14).** JG's commit message: "For RDNA3/4 I was not able to get better performance than the tile kernel for head sizes > 128." Upstream chose the opposite direction from the held branch at D>128 — TILE, not MMA. Held branches `experiment/jg-fa-rdna3{,-tune}` are now archival; delete after this rebase pushes. |
|   8 | [Dense-aware MMQ + TILE FA D=256 follow-up](strix-halo/pp-rdna3_5-tile-mmq.md)           | **+6.3% pp @ d=16k** vs the 2026-05-14 shipped build; d=0 flat; tg128 within noise (last measured 2026-05-22)           | **Split on the 2026-07-16 upstream rebase.** The TILE FA D=256/ncols=32 override rebased clean and is **kept** on master. The dense/MoE MMQ half was dropped with Finding #5 and re-ported into Finding #9 (it survives as the `J_max` cap). |
|   9 | [RDNA3.5 MMQ config table (re-port onto PR #24127)](strix-halo/mmq-rdna3_5-config-table.md) | **Unmeasured.** Carries Findings #5 + #8's MMQ semantics onto upstream's new per-arch table (`I=64`, `nthreads=128`, MoE `J`≤48) | **Ported 2026-07-16, bench pending.** Table is verified well-formed and dispatching correctly, but has never run on gfx1151. `occupancy=2` is a guess — the knob did not exist before `__launch_bounds__` became mandatory. Gated on the two-run A/B in the doc. |

Topic docs, code pointers, and dead-end postmortems live under [`strix-halo/`](strix-halo/). A longer survey of optimization sites in the tree (HIP / Vulkan / CPU), numbered §1–10, is in [`strix-halo/NOTES.md`](strix-halo/NOTES.md); the **#n** tags in the next-experiments tables below refer to those sections.

## Strix Halo: next experiments

T-shirt sizes: **S** = hours, **M** = a day or two, **L** = a week, **XL** = multi-week. Benefit is measured or plausible pp/tg delta on the Qwen 3.6 workload unless noted. Shipped items are in the [findings table](#strix-halo-findings) above. In the tables below, **#n** in an item name (e.g. **#2** MMVQ) refers to the matching numbered section in [`strix-halo/NOTES.md`](strix-halo/NOTES.md).

### Highest ROI first

|   # | Item                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Cost |                 Benefit                 | Why this position                                                                                               |
| --: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--: | :-------------------------------------: | --------------------------------------------------------------------------------------------------------------- |
|   1 | **#2 MMVQ RDNA3.5 dedicated table + `nwarps` sweep** — the cheap "join RDNA3_0's table" attempt regressed tg128 −18 % @ d=0, −38 % @ d=16k ([mmvq-rdna3_5.md](strix-halo/mmvq-rdna3_5.md)). Real follow-up is a dedicated RDNA3.5 entry with a tighter quant whitelist (e.g. Q4_0/Q8_0 only) and `nwarps ∈ {2, 4}` swept against the bench matrix. [mmvq.cu:77](ggml/src/ggml-cuda/mmvq.cu#L77), [:93](ggml/src/ggml-cuda/mmvq.cu#L93), [:348](ggml/src/ggml-cuda/mmvq.cu#L348). |  M   |                    M                    | Multi-knob register-pressure search, not a single constant flip. Park unless tg becomes the binding constraint. |
|   2 | **#3 mmf `src1_ncols` gate asymmetry** — RDNA3.0 caps at `>8`, RDNA3.5 inherits generic `>16`. [mmf.cu:169](ggml/src/ggml-cuda/mmf.cu#L169).                                                                                                                                                                                                                                                                                                                                     |  S   |                   S-M                   | Single-line threshold bench.                                                                                    |
|   3 | **#8 Zen 5 CPU backend variant** — [CMakeLists.txt:379](ggml/src/CMakeLists.txt#L379) only has `zen4`. Zen 5 has a native 512-bit datapath vs Zen 4's double-pumped 256-bit.                                                                                                                                                                                                                                                                                                     |  S   |                   S-M                   | CPU-side only, off the GPU hot path, so ROI is bounded. Cheap enough to try anyway.                             |
|   4 | **#9 `madvise(MADV_HUGEPAGE)` after mmap** — [llama-mmap.cpp:437-467](src/llama-mmap.cpp#L437-L467). TLB pressure is real at 60B+ params in 128 GB LPDDR5X.                                                                                                                                                                                                                                                                                                                      |  S   | S (load-time; small ongoing tg at best) | One-line hint, low risk, low reward.                                                                            |
|   5 | **#10 Commit a Strix Halo bench to `benches/`** — gives us a defensible regression signal and an upstream talking point.                                                                                                                                                                                                                                                                                                                                                         |  S   |                  infra                  | Not a perf win; worth doing once we have one more patch landed to anchor the baseline.                          |

### Vulkan-only — only if we switch backends

|   # | Item                                                                                                                                                                                                                                            | Cost |       Benefit        | Note                                             |
| --: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--: | :------------------: | ------------------------------------------------ |
|  V1 | **#4 `AMD_RDNA3_5` architecture class** — gfx1151 bucketed as `AMD_RDNA3` despite 40 vs 96 CUs, 256 vs 960 GB/s BW. [ggml-vulkan.cpp:270-279](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L270-L279).                                                  |  M   | unknown, plausibly L | Unlocks dedicated warptile + FA occupancy paths. |
|  V2 | **#5 FA `limit_occupancy_shmem` re-tune** — [ggml-vulkan.cpp:2990-2994](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2990-L2994). Heuristic was "guessed, tested on RDNA2"; Strix Halo shares LLC with CPU.                                            |  S   |         S-M          | Bundle with V1.                                  |
|  V3 | **#6/#7 UMA allocation (`prefer_host_memory` + `HostCached`)** — [ggml-vulkan.cpp:2799-2808](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2799-L2808). `DeviceLocal` is a ~512 MB GART window unless BIOS-reconfigured; large models always fall back. | S-M  | S (mostly load-time) |                                                  |

### Dead ends — documented, not pursued

- **MMVQ join RDNA3_0 table** ([strix-halo/mmvq-rdna3_5.md](strix-halo/mmvq-rdna3_5.md)): the obvious "route RDNA3.5 to RDNA3_0's `nwarps=8` table" patch regressed tg128 −18 % @ d=0, −38 % @ d=16k on Qwen 3.6 Q4_K_XL — register-pressure spill on RDNA3.5's smaller LLC. Reverted. The real next experiment is a dedicated RDNA3.5 table (backlog #2), not joining RDNA3_0.
- **FA MMA_F16 on gfx1151** ([strix-halo/fa-dispatcher.md](strix-halo/fa-dispatcher.md)): blocked on an RDNA3 unpacked-WMMA register-layout bug that killed upstream [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063). The original 1-line dispatcher widening is abandoned. The 2026-04-29 attempt to resolve via Finding #7 (cherry-pick of JG's `cuda-fa-rdna3-4` branch + line-1672 widen) was a held regression at f16/f16 KV in production. **Closed 2026-05-14** — upstream [PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880) shipped JG's chain and explicitly kept TILE (not MMA) for RDNA3 D>128, so the original line of investigation is settled. The D≤128 path now uses the new mma FA kernel for free.
- **OPSEL-paired half2 accumulator port for RDNA3 FA-MMA** ([strix-halo/fa-rdna3-opsel-pair.md](strix-halo/fa-rdna3-opsel-pair.md)): proper structural fix for Finding #7's f32 `T_C_VKQ` accumulator ceiling. **Attempted 2026-05-04, abandoned** — prototype landed (`tile_pair_16x8_half2_rdna3` + paired `wmma_f16_..._tied_w32` calls with opposite OPSEL), `test-backend-ops` failed en masse on D=256 numerics, debugging escalated into in-kernel printf instrumentation without bisecting the root cause. Static-VGPR Phase 0 gate was never reached. Postmortem in the doc lists the four candidate failures and the unit-test scaffolding a retry would need before touching `fattn-mma-f16.cuh` again.
- **#1 UMA / `integrated = false`** ([strix-halo/uma-integrated.md](strix-halo/uma-integrated.md)): [PR #16308](https://github.com/ggml-org/llama.cpp/pull/16308) author reported no perf impact; the flag gates only small scratch buffers, not weight/KV traffic.
- **ROCm config flags** ([strix-halo/rocm-config.md](strix-halo/rocm-config.md)): `ROCBLAS_USE_HIPBLASLT_BATCHED=0` + LLVM unroll-threshold. Community reports 2× pp on gpt-oss-120b; null on Qwen 3.6. Kept on as AMD-recommended safety nets.

### Watching upstream

- **[PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127)** (JohannesGaessler, merged 2026-07-13) — refactored MMQ kernel configuration into per-arch tables (`mmq-config-*.cuh`), renamed `mmq_x`/`mmq_y` to `J`/`I`, and made `__launch_bounds__` mandatory. **Deleted all six functions Finding #5 patched**, so that patch was dropped rather than rebased on 2026-07-16. **Resolved:** re-ported as [Finding #9](#strix-halo-findings), a dedicated `mmq-config-rdna3_5.cuh`. Net effect is favourable — the refactor is the extension point this fork was faking with prepended ternaries, and it makes the tuning plausibly upstreamable. Note `amd_wmma_available()` still covers all of RDNA3, so **without our dispatch branch gfx1151 silently inherits the rdna4 table** (`nthreads=256, I=128`) — watch for that if the branch is ever dropped in a future sync.
- **[PR #22051](https://github.com/ggml-org/llama.cpp/pull/22051)** (JohannesGaessler, merged 2026-04-17) — refactored AMD mma data loading in `mma.cuh` and the MMQ host/device helpers we touch in Finding #5. **Resolved 2026-04-19:** rebased our patch on top, re-benched with + without. Port still wins by +37% pp @ d=0 / +14% pp @ d=16k on Qwen 3.6; see [mmq-rdna3_5.md § post-upstream sync](strix-halo/mmq-rdna3_5.md#post-upstream-sync-re-bench-2026-04-19). Separately surfaced an upstream tg-at-depth regression unrelated to our patch, tracked in [tg-at-depth-regression.md](strix-halo/tg-at-depth-regression.md).
- **[PR #22298](https://github.com/ggml-org/llama.cpp/pull/22298)** (CUDA: reduce MMQ stream-k overhead, merged 2026-04-26) — **Resolved the tg-at-depth regression** that was tracked in [tg-at-depth-regression.md](strix-halo/tg-at-depth-regression.md) (tg128@d=16k 31.47 → 44.90 t/s). Touches the same `mma.cuh` hot path #22051 reshaped. No fork-side action.
- **[PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880)** (JohannesGaessler, merged 2026-05-14) — landed the `cuda-fa-rdna3-*` chain: RDNA3 mma FA for D≤128, faster AMD transpose, AMD kernel tuning. **D>128 still routes to the TILE kernel on RDNA3/4** (per JG: "I was not able to get better performance than the tile kernel for head sizes > 128"). Effect on this fork: retires Finding #6 (rocWMMA tuning moot when D>128 goes to TILE) and supersedes Finding #7 (held branch went the other direction at D=256). `fattn-wmma-f16.cu` still exists in tree; the AMD-WMMA + AMD-MFMA paths now live alongside it in `mma.cuh`. **Re-bench gate: run the full Qwen 3.6 matrix at `{0, 2048, 8192, 16384}` after this rebase to confirm D=256 is no worse than the pre-rebase baseline.** Also benches a D≤128 model to characterise the free win from the new mma FA path.
- **[PR #16827](https://github.com/ggml-org/llama.cpp/pull/16827)** (lhl, rejected upstream 2025-10-29) — rocWMMA FA tuning for gfx1151. Carried as Finding #6 from 2026-04-19 until 2026-05-14; dropped when #22880 made the WMMA-FA dispatcher path inactive for our production D=256 workload. See [rocwmma-tuned.md](strix-halo/rocwmma-tuned.md) for the postmortem.

### Re-bench checklist after upstream sync or ROCm bump

The 2026-04-27 rocWMMA regression went undetected for ~5 weeks because the doc said "flat" at landing and we didn't re-validate. After every upstream sync OR TheRock nightly bump, run the full Qwen 3.6 bench at `{0, 2048, 8192, 16384}` and compare against the prior baseline. If pp@d=16k is significantly different, **bisect the build flag** (`GGML_HIP_ROCWMMA_FATTN`) before assuming the upstream/ROCm change is the cause — that's the cheapest possible test and historically the highest-yield.

## How this fork is developed

Single branch: **`master`** here, tracking upstream plus the `strix-halo/` docs folder and validated patches. Each optimization attempt is ideally **one commit** on `master`:

1. Write the hypothesis in a new markdown file under `strix-halo/` (link the lines you plan to change, state what you will measure).
2. Land the code change as one commit.
3. Build and benchmark on real gfx1151 hardware (pinned SHA, not a floating branch name, if you use Docker layer caching).
4. Keep the commit if the bench shows a clear win across the depth/quant matrix you care about; otherwise revert and annotate the doc with why it failed.

That keeps history readable: tried / measured / kept or reverted. Docs accumulate even when patches do not.

## Keeping up with upstream

Upstream moves quickly. A typical resync:

```bash
git fetch upstream   # upstream = ggml-org/llama.cpp
git checkout master
git rebase upstream/master
# resolve conflicts; drop superseded patch commits if upstream replaced them
git push origin master   # origin = this fork
```

## ROCm / gfx1151 build

Official ROCm packages have shipped broken `gfx1151` kernel artifacts for some releases; see [ROCm/ROCm#6042](https://github.com/ROCm/ROCm/issues/6042). A working multi-stage Docker build for this fork uses **TheRock** ROCm nightlies and is maintained alongside deployment playbooks in a companion repo—see [Profiling & lab workflow](https://github.com/justinappler/server-configs/blob/main/services/llamacpp/profiling/README.md) in [`server-configs`](https://github.com/justinappler/server-configs) (`services/llamacpp/`).
