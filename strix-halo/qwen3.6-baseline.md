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

## 2026-04-27 — pp-at-depth regression observed (cause unclear)

Re-benched with the same model and flags. d=0 improved; d=16k regressed sharply versus Run 3 above.

| depth | pp512 (t/s) | tg128 (t/s) | vs Run 3 baseline pp |
|------:|------------:|------------:|---------------------:|
|     0 |       1,185 |        45.9 |               +15 % |
| 2,048 |         785 |        45.5 |                   — |
| 8,192 |         398 |        44.4 |                   — |
| 16,384 |         238 |        42.9 |               −67 % |

The d=0 improvement comes from hipBLASLt loading clean gfx1151 `.hsaco` kernels again (the official ROCm apt path was missing them; the container build switched back to TheRock nightly, which ships them). The d=16k regression is not yet explained — reverting llama.cpp SHA didn't recover the numbers, so it isn't a userspace SHA we can blame. The host has since drifted off the 24.04 HWE/OEM kernel that this baseline was taken on (now on Ubuntu 26.04 stock `7.0.0-14-generic`; no OEM kernel exists for 26.04 yet), but the upgrade predates the regression by a meaningful amount, so it isn't a clean attribution. `dmesg` does show escalating `amdgpu_amdkfd_restore_userptr_worker` activity, which is consistent with userptr eviction stalls but not proven to fire during the bench window. Best current guess is something in the ROCm 7.13 nightly progression, but it's unconfirmed.

## Related findings

- [kv-cache.md](kv-cache.md) — why q8_0/q4_0 collapses at depth.
- [fa-dispatcher.md](fa-dispatcher.md) — why we're stuck on the TILE kernel.
