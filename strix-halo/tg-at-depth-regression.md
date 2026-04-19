# Upstream tg-at-depth regression surfaced by the 2026-04-17 sync

## What we saw

Re-benching Qwen 3.6 35B-A3B Q4_K_XL at [mmq-rdna3_5.md#post-upstream-sync-re-bench-2026-04-19](mmq-rdna3_5.md#post-upstream-sync-re-bench-2026-04-19) on the rebased fork turned up an unexpected regression at depth that is **not** caused by our MMQ port. The isolation A/B (same post-sync upstream, with and without our patch) showed our port contributes nothing to tg — so comparing the pre-sync bench (`d8ad713`) against the post-sync no-port build (`b078e4b`) isolates what upstream changed:

| test | d8ad713 (pre-sync) | b078e4b (post-#22051, no port) | delta (upstream only) |
|---|---:|---:|---:|
| pp512 @ d=16,384 | 855 ± 6 | 651 ± 2 | **−23.9%** |
| tg128 @ d=16,384 | 43.46   | 31.47 ± 0.12 | **−27.6%** |
| tg128 @ d=8,192  | — (not in that run) | 37.19 ± 0.01 | (new) |
| tg128 @ d=2,048  | — | 43.25 ± 0.08 | |
| tg128 @ d=0      | 46.66 | 46.17 ± 0.03 | flat |

tg at d=0 is untouched; the hit scales with depth. That's the signature of an attention-kernel slowdown, not a GEMM/MMVQ slowdown — FA cost grows with KV length, the GEMM cost of the output projection does not.

Our MMQ port recovers roughly +14% of the pp-at-depth gap but zero of the tg gap (as expected — MMQ tuning doesn't touch the MMVQ or FA paths that tg uses). So the upstream regression is attention-bound and sitting on top of whatever our port can fix.

## The bisect window

The rebase moved our base from upstream `23b8cc499` to `4eac5b450`. Only five upstream commits touched `ggml/` in that window, and only two touched the CUDA/HIP backend:

```
4eac5b450 CUDA: refactor mma data loading for AMD (#22051)
471540ae8 HIP: Remove unesscary NCCL_CHECK (#21914)
```

`#21914` is a trivial guard removal and shouldn't affect perf. That leaves **#22051 as the overwhelming prior**. The PR refactored `mma.cuh` data-loading helpers that are shared between MMQ *and* the FA kernels — the FA MMA paths on AMD go through the same `tile<>::load` machinery that #22051 reshaped. A regression in that shared load path would hit FA (and thus tg-at-depth) without touching MMQ-with-our-tuning outcomes, which matches what we see.

Caveat: the Docker build pulls TheRock ROCm nightly, which is not pinned. Some of the delta could be a ROCm-side shift across the interval, not a llama.cpp-side shift. Any bisect has to hold ROCm constant.

## Next steps

1. **Confirm the window.** Rebuild with `LLAMACPP_VERSION=471540ae8` (the commit immediately before #22051) and re-bench. If tg @ d=16k comes back to ~43, the regression is pinned to #22051; if it's still depressed, look at the other three non-CUDA commits or at a ROCm-nightly shift.
2. **If it's #22051**, read the PR diff for FA-path call sites. JohannesGaessler's PR description focuses on MMQ but the refactor is in `mma.cuh`, which is shared. Likely suspects: any register-layout or `load_ldmatrix_trans` change that affects the FA kernel's q/k/v tile loads on gfx1151.
3. **If the next upstream sync resolves it on its own** (the refactor gets a follow-up), mark this doc resolved and move on. If it persists, either file an upstream issue with a minimal repro or add a local revert of the offending portion on the fork.

## Why this isn't urgent

We still ship +14-37% pp gains from our MMQ port post-rebase, which is what matters for TTFT on the agentic-coding workload this fork targets. tg-at-depth was never in our findings table and the 27% drop puts us at ~31.5 t/s @ d=16k — still comfortably above "usable" for interactive generation. This is worth chasing next, but after the higher-ROI MMVQ and MoE investigations queued in [NOTES.md](NOTES.md).
