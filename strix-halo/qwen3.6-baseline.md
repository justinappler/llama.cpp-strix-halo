# Qwen 3.6 35B-A3B — baseline on gfx1151

## Setup

| Component | Value |
|---|---|
| Host | AMD Ryzen AI Max 395+, Radeon 8060S (gfx1151), 128 GB LPDDR5x-8000 |
| Kernel / driver | Linux 7.0 OEM (24.04 HWE/OEM track), TTM pages_limit raised, `amdgpu.cwsr_enable=0` |
| Container | `llamacpp-server:local` (TheRock ROCm nightly `7.13.0a20260411`) |
| llama.cpp | build `45cac7c` |
| Model | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (20.81 GiB, 34.66 B params) |
| Bench flags | `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3` |

## Run 1 — production config (q8_0 / q4_0 KV, FA on)

| depth | pp512 (t/s) | tg128 (t/s) |
|------:|------------:|------------:|
|     0 |         767 |        45.4 |
| 2,048 |         209 |        38.9 |
| 8,192 |          76 |        26.9 |
| 16,384 |         43 |        19.1 |

## Run 2 — f16 / f16 KV, FA off

| depth | pp512 (t/s) | tg128 (t/s) |
|------:|------------:|------------:|
|     0 |       1,025 |        48.0 |
| 16,384 |        651 |        31.0 |

## Run 3 — f16 / f16 KV, FA on

| depth | pp512 (t/s) | tg128 (t/s) |
|------:|------------:|------------:|
|     0 |       1,029 |        46.5 |
| 16,384 |        731 |        43.3 |

## Wall-clock for a cold 10 k prefill

Integrating the pp512 curves (linear interpolation between measured depths):

| Config | Estimated cold 10 k prefill |
|---|---:|
| q8_0 / q4_0 KV, FA on (production) | **~75 s** |
| f16 / f16 KV, FA on                |  **~12 s** |

## Theoretical reference

- Radeon 8060S peak: ~59 TFLOPS FP16 (WMMA), 256 GB/s LPDDR5x-8000.
- Qwen 3.6 A3B compute/token (MLP portion, ~3 B active): ~6 GFLOPs.
- Short-context pp ceiling (MLP-only): ~9,800 t/s. We're at ~10 % of that — MoE routing, attention, and non-matmul overhead all contribute.

## 2026-04-27 — pp-at-depth regression observed and resolved

Initial re-bench with the same model and flags showed pp512@d=16k collapsed sharply versus Run 3 above:

| depth | pp512 (t/s) | tg128 (t/s) | vs Run 3 baseline pp |
|------:|------------:|------------:|---------------------:|
|     0 |       1,185 |        45.9 |               +15 % |
| 2,048 |         785 |        45.5 |                   — |
| 8,192 |         398 |        44.4 |                   — |
| 16,384 |         238 |        42.9 |               −67 % |

### Initial (incorrect) attribution

The first guess was something in the ROCm 7.13 nightly progression — TheRock had been bumped from `0411` → `0426` and host packages had churned (`amdrocm7.12-gfx1151` swapped for Ubuntu's distro `rocm` 7.1.x on 2026-04-22, then reinstalled on 2026-04-27). That guess was wrong.

### Actual cause: `GGML_HIP_ROCWMMA_FATTN=ON` regressed silently on Qwen 3.6

Systematic bisection over the day eliminated, with evidence: container ROCm version (rebuild against `0411` would have been the same), llama.cpp upstream delta (rebased onto pre-#22298 to confirm), source-level patch drift (verified intact), host ROCm package set (full reinstall + reboot), modprobe.d state, KFD userptr eviction (zero firings), memory pressure (49 GiB free), GPU clocks (boosting cleanly to 2895 MHz at 100% busy, no throttling), GPU/system firmware (identical across boots), bench methodology (`docker exec` vs one-off, both shapes give the same number).

The remaining variable was the `GGML_HIP_ROCWMMA_FATTN` build flag. Flipping it from `ON` to `OFF` with everything else held constant:

| depth | rocWMMA ON | rocWMMA OFF | Δ |
|------:|-----------:|------------:|---:|
|     0 |   1,210.89 |  **1,367.46** | +12.9% |
| 2,048 |     809.63 |  **1,234.25** | +52.4% |
| 8,192 |     406.96 |  **1,043.46** | +156.4% |
| 16,384|     241.69 |    **852.79** | **+252.9%** |

`OFF` recovered the Run 3 baseline and exceeded it by 14.6% at d=16k. The patched rocWMMA FA path (commit `1be00ab87` / today's `030e29029`, [rocwmma-tuned.md](rocwmma-tuned.md)) had silently regressed at D=256 between landing on 2026-04-19 (where the doc's outcome bench called it "flat ±1.5%") and now. See [rocwmma-tuned.md "Re-bench 2026-04-27 — flag back OFF (regression)"](rocwmma-tuned.md#re-bench-2026-04-27--flag-back-off-regression) for the full investigation and the candidate mechanisms.

### What the d=0 improvement *was*

The +15% pp512@d=0 in the regression-state numbers above (1,185 vs 1,029 in Run 3) is real — it tracks the orthogonal #22298 MMQ stream-k overhead reduction, plus our MMQ port (PR #21344) being now well-rebased onto current upstream. Even with the rocWMMA FA path costing 12.9% at d=0, the MMQ improvements showed through at shallow context.

### Disregard the original `userptr_restore_worker` hypothesis

The regression note initially flagged "escalating `amdgpu_amdkfd_restore_userptr_worker` activity" as consistent with userptr eviction stalls. Direct check during this investigation: `dmesg` shows zero firings of `amdgpu_amdkfd_restore_userptr_worker` since boot. The hypothesis is dead. The depth-proportional shape of the original regression was real but came from the rocWMMA FA path scaling worse with KV cache size, not from KV being paged out.

## 2026-08-29 — post-rebase re-bench (build `df16ecb`) — **prefill regression, not yet attributed**

Arm C of [fa-mma-d256-26419.md](fa-mma-d256-26419.md). Build `df16ecb` (fork master on upstream
`c841aeeb8`), ROCm 7.14.0, canonical bench with `-lm none`. **Run twice on a confirmed-idle host**
because run 1's error bars were anomalous; run 2 reproduced them.

| depth | run 1 pp512 | run 2 pp512 | mean | vs `b73cfa4` | run 1 tg128 | run 2 tg128 | vs `b73cfa4` |
| ------: | ----------------: | ----------------: | ------: | ------: | -----------: | -----------: | ------: |
|       0 | 1388.67 ± 97.25 | 1366.92 ± 101.85 | 1377.80 | **-5.3%** | 51.20 ± 0.20 | 51.11 ± 0.19 | -0.4% |
|   2,048 | 1264.92 ± 47.60 | 1253.11 ± 46.20 | 1259.02 | **-3.5%** | 50.83 ± 0.19 | 50.75 ± 0.26 | -0.5% |
|   8,192 | 1115.28 ± 52.99 | 1103.68 ± 58.99 | 1109.48 | **-2.5%** | 49.43 ± 0.18 | 49.38 ± 0.18 | -0.5% |
|  16,384 |  952.16 ± 30.31 |  952.07 ± 26.95 |  952.12 | **-3.4%** | 47.62 ± 0.23 | 47.55 ± 0.23 | -0.5% |

The d=0 comparator is ambiguous by mode: `b73cfa4` measured 1454.79 with `-mmp 0` and 1444.97 with
`-lm none`. Against the like-for-like `-lm none` figure the d=0 delta is **-4.6%**, not -5.3%.

### Two separate anomalies, and only one of them is the mean

**1. Prefill is down 2.5-5.3% at every depth.** Reproducible across two runs. The d=16k figure is the
solid one: 952.16 and 952.07 across independent invocations, agreeing to 0.01%, against a baseline of
986.05 ± 13.84. Six reps at ~952 versus three at ~986 is a real difference, not a sampling artifact.

**2. Per-rep variance exploded, and it is not the host.** This is the more interesting finding:

| depth | `b73cfa4` rel. sd | `df16ecb` run 1 | `df16ecb` run 2 |
| ------: | ------: | ------: | ------: |
|       0 | 0.24% | 7.00% | 7.45% |
|   2,048 | 0.88% | 3.76% | 3.69% |
|   8,192 | 1.78% | 4.75% | 5.34% |
|  16,384 | 1.40% | 3.18% | 2.83% |

Run 1 was initially discounted as host interference. Run 2 on a confirmed-idle box reproduced both
the means *and* the spread, so **the variance is a property of this build**. Note the shape: spread is
worst at d=0 (7%), where prefill is MMQ-dominated (58% per
[kernel-time-breakdown.md](kernel-time-breakdown.md)) and FA is only 2% - so this does not point at
the FA path. Decode is unaffected on both axes: tg error bars stayed at ±0.2 and the means moved
-0.4% to -0.5%, at or just inside noise.

The combination - stable aggregate mean, high rep-to-rep spread, worst where MMQ dominates - is
consistent with a codegen change altering scheduling or occupancy rather than with a tile-config
change, which would move the mean without touching variance.

### Arm A (upstream clean, `c841aee`) settles two of the three questions

Benched same day, same host, same command, image tagged separately so production was untouched.

| depth | arm A pp512 | arm C pp512 | **C - A** | arm A tg128 | tg delta |
| ------: | ----------------: | ------: | ------: | -----------: | ------: |
|       0 | 1077.79 ± 60.59 | 1377.80 | **+27.8%** | 51.06 ± 0.22 | +0.2% |
|   2,048 |  974.19 ± 27.28 | 1259.02 | **+29.2%** | 50.84 ± 0.19 | -0.1% |
|   8,192 |  864.05 ± 36.55 | 1109.48 | **+28.4%** | 49.44 ± 0.18 | -0.1% |
|  16,384 |  746.32 ± 18.31 |  952.12 | **+27.6%** | 47.59 ± 0.21 |  0.0% |

**The fork's patches are worth +27.6% to +29.2% prefill with decode flat to within 0.2%** - the
port-off control that had never been run. See
[fa-mma-d256-26419.md](fa-mma-d256-26419.md#this-closes-the-port-off-attribution-backlog-item-2).

**And arm A reproduces the variance blow-up with zero fork code in it** (rel. sd 2.45-5.62% against
`b73cfa4`'s 0.24-1.78%). That eliminates our patches as the cause and points the remaining regression
at the upstream window.

### Prime suspect: the second math-flag removal

Per the [re-bench checklist](upstream.md#re-bench-checklist), cheapest suspects first. Our three
patches rebased **byte-identical** and upstream touched none of the three files, which makes them a
weaker candidate than usual. The flag layer did change:

[PR #26696](https://github.com/ggml-org/llama.cpp/pull/26696) (merged 2026-08-12) removed
`-funsafe-math-optimizations` from `ggml/src/ggml-hip/CMakeLists.txt`. That was the **remaining**
fast-math flag for HIP - the deliberate replacement for `-ffast-math`, whose own removal
([PR #25495](https://github.com/ggml-org/llama.cpp/pull/25495)) was cleared in the 2026-08-02 bench
below. The deleted comment said so explicitly: *"Fast math for HIP, like CUDA's `-use_fast_math`."*
So HIP builds went from fast-math to IEEE-conformant in this window, across every kernel. Upstream's
motivation was correctness, not speed: `-fassociative-math` reassociates FP reductions and can flip
greedy argmax on RDNA3.5.

**A global codegen change on our exact backend, landed in the exact window that produced a broad
prefill regression, is the hypothesis to test first** - and it is a one-line restore
(`-DCMAKE_HIP_FLAGS=-funsafe-math-optimizations`). If it is the cause, the honest framing is that we
were banking a few percent of prefill on non-IEEE FP and upstream took it back deliberately; the
question then becomes whether we want it locally, not whether it is a bug.

Not yet excluded: our own patches interacting with 27 days of upstream. Arm A (upstream clean,
`c841aeeb8`) separates the two - if A shows the same regression and spread against its own history,
nothing here is ours.

**Do not treat `df16ecb` as the new headline baseline.** `b73cfa4` remains the reference until this
is attributed.

## 2026-08-02 — post-rebase re-bench (build `b73cfa4`)

Current headline numbers. Build `b73cfa4`, ROCm 7.14.0, canonical bench (f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3 -d 0,2048,8192,16384`), no compose env vars — same shape as the `05e837f` run it replaces.

| depth | pp512 (t/s) | tg128 (t/s) | pp vs `05e837f` | tg vs `05e837f` |
| ------: | ---------------: | ------------: | ------: | ------: |
|       0 | 1454.79 ± 3.52  | 51.37 ± 0.19 | +1.9% | +3.1% |
|   2,048 | 1304.31 ± 11.52 | 51.02 ± 0.20 | +0.4% | +4.2% |
|   8,192 | 1138.47 ± 20.30 | 49.68 ± 0.18 | +0.3% | +3.2% |
|  16,384 |  986.05 ± 13.84 | 47.85 ± 0.19 | +1.5% | +3.1% |

Nothing regressed. New bests at every depth on both axes; ~986 t/s prefill at d=16k and 51.4 t/s decode at d=0.

### What this run actually tells us

**Prefill is flat; decode moved.** Three of four pp deltas are inside the ±1.5-2% noise floor, and the fourth is at its edge. All four tg deltas are +3.1% to +4.2% with error bars of ±0.4%, so decode is unambiguously outside noise.

That decoupling is the useful part. The previous bench moved pp and tg together (+5.8% / +5.4%), which is precisely why it could not be attributed. This one separates them:

- **tg up ~3-4% is upstream's, not ours.** No patch this fork carries can move decode - MMQ tile tuning is prompt-side and decode goes through MMVQ, established by the 2026-04-19 A/B. The candidate flagged *before* this bench ran was [PR #26171](https://github.com/ggml-org/llama.cpp/pull/26171) (transpose-free gemmv, merged 2026-07-30), the one commit in the 136-commit window touching the gemv path that owns 71-77% of decode time per [kernel-time-breakdown.md](kernel-time-breakdown.md). Consistent, but not proven - no A/B was run.
- **pp flat is the positive result for the MMQ port.** The port was rewritten onto upstream's own table in this window ([mmq-rdna3_5-config-table.md](mmq-rdna3_5-config-table.md#collision-with-pr-26199)). Had the retune been lost in that merge, gfx1151 would have fallen back to rdna4's wide tiles - historically worth **-27% to -37%** on prefill. Flat prefill across 136 upstream commits, a table reshape, and two build-flag removals means the tuning landed intact. This is weaker than a real A/B but stronger evidence than the last bench produced.
- **Both flag removals are clear.** Dropping `-ffast-math` (upstream [PR #25495](https://github.com/ggml-org/llama.cpp/pull/25495)) and `--amdgpu-unroll-threshold-local=600` (ours, [rocm-config.md](rocm-config.md)) cost nothing measurable on prefill. Two of the three confounders flagged going into this bench are retired.

Still not an A/B: the port-off control remains unrun, so the port's own share is still unmeasured. See [mmq-rdna3_5-config-table.md § Outcome](mmq-rdna3_5-config-table.md#outcome).

### Flag deprecation - resolved same day

llama-bench warns that `-mmp` is deprecated in favour of `--load-mode`. The canonical bench has been migrated to **`-lm none`**, which is the exact equivalent (`-mmp 0` parses to `LLAMA_LOAD_MODE_NONE`).

Two traps worth recording:

- **The deprecation message is wrong.** It says "Please use --load-mode mmap instead" no matter which value you passed. Following it after `-mmp 0` would turn mmap **on**, silently changing TLB behaviour on the unified pool - a confounder that would present as a mysterious regression. The valid values are `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`; `dio` is direct I/O and is *not* a synonym for `none`.
- **`mmap` is llama-bench's default**, so if `-mmp 0` ever becomes a true no-op rather than a warning, the bench flips to mmap silently.

Verified equivalent before switching, same build and host: `-mmp 0` gave pp512 1454.79 ± 3.52 / tg128 51.37 ± 0.19, `-lm none` gave 1444.97 ± 13.45 / 51.35 ± 0.23. Inside noise on both axes, so **every prior baseline taken with `-mmp 0` stays directly comparable** - no re-baseline needed. The `lm` column in the results table reports the mode actually used; read it rather than trusting the flag.

Historical bench commands elsewhere in `strix-halo/` still show `-mmp 0` and are left as written - they record what was actually run.

## Related findings

- [kv-cache.md](kv-cache.md) — why q8_0/q4_0 collapses at depth.
- [fa-dispatcher.md](fa-dispatcher.md) — why we're stuck on the TILE kernel.
