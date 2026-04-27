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

## Related findings

- [kv-cache.md](kv-cache.md) — why q8_0/q4_0 collapses at depth.
- [fa-dispatcher.md](fa-dispatcher.md) — why we're stuck on the TILE kernel.
