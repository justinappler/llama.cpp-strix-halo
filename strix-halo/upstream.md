# Tracking upstream

Upstream moves fast and repeatedly lands things that make this fork's patches redundant, obsolete, or wrong. This file is the running record of which upstream changes mattered and what we did about each.

**The pattern to expect:** most of this fork's findings have eventually been retired by upstream, usually because someone solved the same problem better. That is a good outcome. The cost of not tracking it is carrying a patch that has silently become a regression - which has happened here once already, see [findings.md](findings.md#how-6-got-away-from-us).

## Re-bench checklist

Run this after **every** upstream sync or ROCm bump. No exceptions - the one time it was skipped, a 3.5x prefill regression hid for five weeks.

1. Run the full Qwen 3.6 matrix at depths `{0, 2048, 8192, 16384}` and compare against the previous baseline in [qwen3.6-baseline.md](qwen3.6-baseline.md). Noise floor on this host is about +/-1.5%.
2. If anything moved more than that, **do not assume it was the upstream bundle.** Bisect the cheapest suspects first, in this order:
   - **Our own patches.** There are only three, and two are one file each. `git checkout upstream/master -- ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh` and `-- ggml/src/ggml-cuda/fattn-tile.cuh` are both single-command reverts that produce a clean A/B.
   - **HIP compiler flags.** Two changed under us in this window and neither has been measured: upstream [PR #25495](https://github.com/ggml-org/llama.cpp/pull/25495) removed `-ffast-math` from the whole HIP build, and we dropped `--amdgpu-unroll-threshold-local=600` from the deploy build on 2026-08-02 ([rocm-config.md](rocm-config.md)). Both are one-line restores.
   - **The ROCm version**, last.
3. Record the result in the relevant topic doc even if nothing moved. "Re-benched, flat" is a useful entry; silence is not.

> The old version of this checklist said to bisect `GGML_HIP_ROCWMMA_FATTN` first. That flag no longer exists - upstream deleted rocWMMA FlashAttention in [PR #26046](https://github.com/ggml-org/llama.cpp/pull/26046).

## 2026-08-02 sync (136 commits, `1a064ab09` -> `221f0f635`)

- **[PR #26199](https://github.com/ggml-org/llama.cpp/pull/26199)** (Geramy Loveless, merged 2026-07-28) - added `mmq-config-rdna3-5.cuh` and `mmq-config-rdna3.cuh` and replaced the `amd_wmma_available` dispatch with explicit `RDNA4 / RDNA3_5 / RDNA3` branches. **This is structurally Finding #9, landed independently.** The values are a verbatim copy of rdna4, so the structure landed upstream but the tuning did not. **Action taken:** deleted our standalone table, wrote our values into upstream's file, kept the MoE `J` cap. The fork's `mmq.cuh` diff shrank from +11/-1 to +5/-1. Full detail in [mmq-rdna3_5-config-table.md § Collision with PR #26199](mmq-rdna3_5-config-table.md#collision-with-pr-26199).
- **[PR #26046](https://github.com/ggml-org/llama.cpp/pull/26046)** (JohannesGaessler, merged 2026-07-24) - removed rocWMMA FlashAttention entirely: `fattn-wmma-f16.{cu,cuh}`, the `GGML_HIP_ROCWMMA_FATTN` option, and the docs. **Closes Finding #6 permanently** and invalidates the old re-bench advice above. No code action needed; our `fattn-tile.cuh` patch is in a different hunk and rebased clean.
- **[PR #25495](https://github.com/ggml-org/llama.cpp/pull/25495)** (merged 2026-07-27) - **removed `-ffast-math -fno-finite-math-only` from the HIP build.** A global codegen change on our exact backend that we did not make and have not measured. Treat it as the prime suspect if the next re-bench moves. Noted in [rocm-config.md](rocm-config.md).
- **[PR #26171](https://github.com/ggml-org/llama.cpp/pull/26171)** (merged 2026-07-30) - transpose-free gemmv. Decode-side, and the 2026-07-18 profile puts MMVQ at 71-77% of decode time, so this is the one non-MMQ commit in this sync that could move tg. Watch for it in the re-bench.
- **[PR #26141](https://github.com/ggml-org/llama.cpp/pull/26141)** (merged 2026-07-29) - disables MMQ on devices reporting under 48 KiB of shared memory. **No effect on us** (gfx1151 reports 64 KiB) but it sits in `ggml_cuda_should_use_mmq`, so it is worth knowing it is benign rather than rediscovering it mid-bisect.
- **[PR #25707](https://github.com/ggml-org/llama.cpp/pull/25707)** (merged 2026-07-30) - adds the `Q2_0` type with MMQ support. Our retuned table picked up `Q2_0` coverage for free as part of the merge onto upstream's file.
- **[PR #26233](https://github.com/ggml-org/llama.cpp/pull/26233)** (merged 2026-07-28) - adds the `Laguna-S-2.1` LLM_TYPE, following #25165 which this fork already rebased onto. Needed for the pending Laguna coder bench.
- **[PR #26012](https://github.com/ggml-org/llama.cpp/pull/26012)** (merged 2026-07-23) - **upstream reversed its AI policy.** AI-generated code is now allowed provided a human understands it and will maintain it; the old "majority human-authored" rule is gone. This removes a stated blocker from the fork's upstreaming plans. What has **not** changed: an agent must never write a PR description, a commit message, or a reply to a reviewer. See [AGENTS.md](../AGENTS.md).

## Earlier

- **[PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127)** (JohannesGaessler, merged 2026-07-13) - refactored MMQ configuration into per-arch tables, renamed `mmq_x`/`mmq_y` to `J`/`I`, made `__launch_bounds__` mandatory. **Deleted all six functions Finding #5 patched**, so that patch was dropped rather than rebased. Re-ported as Finding #9. Net effect was favourable: the refactor is the extension point this fork had been faking with prepended ternaries.
- **[PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880)** (JohannesGaessler, merged 2026-05-14) - landed the `cuda-fa-rdna3-*` chain: RDNA3 mma FA for D<=128, faster AMD transpose, AMD kernel tuning. **D>128 still routes to TILE on RDNA3/4** (JG: "I was not able to get better performance than the tile kernel for head sizes > 128"). Retired Finding #6 and superseded Finding #7 - the held branch had gone the opposite direction at D=256. D<=128 models get the new mma FA path for free.
- **[PR #22298](https://github.com/ggml-org/llama.cpp/pull/22298)** (merged 2026-04-26) - reduced MMQ stream-k overhead and **resolved the tg-at-depth regression** tracked in [tg-at-depth-regression.md](tg-at-depth-regression.md) (tg128@d=16k 31.47 -> 44.90 t/s). No fork-side action.
- **[PR #22051](https://github.com/ggml-org/llama.cpp/pull/22051)** (JohannesGaessler, merged 2026-04-17) - refactored AMD mma data loading and the MMQ helpers Finding #5 touched. Rebased our patch on top and re-benched with and without: the port still won by +37% pp @ d=0. See [mmq-rdna3_5.md § post-upstream sync](mmq-rdna3_5.md#post-upstream-sync-re-bench-2026-04-19).
- **[PR #16827](https://github.com/ggml-org/llama.cpp/pull/16827)** (lhl, rejected upstream 2025-10-29) - rocWMMA FA tuning for gfx1151. Carried as Finding #6 from 2026-04-19 until 2026-05-14. See [rocwmma-tuned.md](rocwmma-tuned.md).

## How to resync

```bash
git fetch upstream                  # upstream = ggml-org/llama.cpp
git branch backup/master-pre-rebase-$(date +%Y%m%d) master
git checkout master
git rebase upstream/master
# resolve conflicts; drop superseded patch commits if upstream replaced them
git push origin master              # origin = this fork
```

Take the backup branch every time. Two of the last three syncs required dropping or reshaping a patch commit mid-rebase, and having the pre-rebase tree to diff against is what made those safe.
