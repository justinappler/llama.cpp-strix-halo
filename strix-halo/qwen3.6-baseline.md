# Qwen 3.6 35B-A3B — baseline on gfx1151

## Setup

| Component | Value |
|---|---|
| Host | AMD Ryzen AI Max 395+, Radeon 8060S (gfx1151), 128 GB LPDDR5x-8000 |
| Kernel / driver | Linux 7.0 OEM, TTM pages_limit raised, `amdgpu.cwsr_enable=0` |
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

## Related findings

- [kv-cache.md](kv-cache.md) — why q8_0/q4_0 collapses at depth.
- [fa-dispatcher.md](fa-dispatcher.md) — why we're stuck on the TILE kernel.
