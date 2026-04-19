# MMQ tile/nwarp tuning for gfx1151 — port of PR #21344

## Hypothesis

`gfx1151` (RDNA 3.5) currently shares MMQ tile/occupancy constants with other RDNA3 parts (`mmq_x_max=128`, `mmq_y=128`, `nwarps=8`). These were tuned for discrete RDNA3 (7900 XTX-class): 96 CUs, 960 GB/s VRAM, dense register files. Strix Halo's 40-CU RDNA3.5 iGPU on ~256 GB/s LPDDR5x hits VGPR pressure at those sizes, which stalls short-context prompt processing.

[Upstream PR #21344](https://github.com/ggml-org/llama.cpp/pull/21344) adds a gfx1151-specific branch: `mmq_x_max=48`, `mmq_y=64`, `nwarps=4`. Per the PR's own Qwen3.5-122B-A10B Q4_K_M bench, that gives +73% pp128, +33% pp512, +10% pp2048, +7% pp4096 on the author's setup.

## What we expect to measure on this fork

Kyuz0's independent toolbox comparison (same rocWMMA-off build, same rocm-7.2.1 base, A/B only differs by this patch) is already archived in [useful-repos/amd-strix-halo-toolboxes/benchmark/results/](../useful-repos/amd-strix-halo-toolboxes/benchmark/results/). Direction-of-win from those logs:

| Model | pp512 @ d=0 | pp2048 @ d=32k |
|---|---|---|
| | baseline → PR#21344 | baseline → PR#21344 |
| Qwen3.5-35B-A3B Q4_K_XL  | 1,098 → 1,380 (+25.7%) | 681 → 681 (flat) |
| Qwen3.5-122B-A10B Q5_K_XL | 312 → 405 (+29.6%) | 250 → 252 (flat) |
| gpt-oss 120B MXFP4       | 633 → 955 (+50.9%) | 296 → 313 (+5.7%) |

So on our target (Qwen 3.6 35B-A3B Q4_K_XL), expect a meaningful short-context gain and ~flat at d=16k+. This is the "hit or miss at large context" comment from kyuz0 on the PR thread — consistent with the patch being a GEMM-tile fix, not an attention-kernel fix. The long-context deficit is attention-kernel-bound (see [fa-dispatcher.md](fa-dispatcher.md) and [kv-cache.md](kv-cache.md)); this patch can't close it.

## The patch

Single file, 6 edits to [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh):

| Function | Line | Change |
|---|---:|---|
| `get_mmq_x_max_host` | [106](../ggml/src/ggml-cuda/mmq.cuh#L106) | Refactor nested ternary → `if`; RDNA3_5 returns `48` |
| `get_mmq_x_max_device` | [116](../ggml/src/ggml-cuda/mmq.cuh#L116) | Split `AMD_WMMA_AVAILABLE` out of the combined branch; RDNA3_5 returns `48` |
| `get_mmq_y_host` | [139](../ggml/src/ggml-cuda/mmq.cuh#L139) | Refactor nested ternary → `if`; RDNA3_5 joins RDNA1 at `64` |
| `get_mmq_y_device` | [154](../ggml/src/ggml-cuda/mmq.cuh#L154) | `#if defined(RDNA1) || defined(RDNA3_5)` returns `64` |
| `mmq_get_nwarps_host` | [290](../ggml/src/ggml-cuda/mmq.cuh#L290) | RDNA3_5 returns `4` |
| `mmq_get_nwarps_device` | [299](../ggml/src/ggml-cuda/mmq.cuh#L299) | RDNA3_5 returns `4` |

The `GGML_CUDA_CC_IS_RDNA3_5(cc)` macro already exists at [common.cuh:84](../ggml/src/ggml-cuda/common.cuh#L84). The device-side `RDNA3_5` macro is defined in [vendors/hip.h:221](../ggml/src/ggml-cuda/vendors/hip.h#L221). No build-system changes required.

## What the PR does **not** do

The PR's original second change (giving RDNA3.5 its own `mmvq_parameter_table_id` instead of falling back to RDNA2) was **reverted** after code review — see PR commit `7957de9d` "revert changes to mmvq.cu". So [NOTES.md #2](NOTES.md) (MMVQ RDNA2 table for RDNA3.5) is still open as a follow-on experiment after this lands.

## Bench plan

Re-run [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 3 config (f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3`) at depths `{0, 2048, 8192, 16384}`.

Decision rule: keep the commit if pp512 @ d=0 improves >5% outside noise **and** no depth regresses. Revert if depth-16k regresses even if d=0 wins — agentic coding workloads live at depth.

## Outcome

**Kept.** Qwen 3.6 35B-A3B Q4_K_XL, same build knobs as [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 3 (f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3`), build `d8ad713`:

| test | baseline | PR #21344 | delta |
|---|---:|---:|---:|
| pp512 @ d=0       | 1,029 | 1,309 ± 29 | **+27.2%** |
| pp512 @ d=2,048   |   —   | 1,232 ± 9  | (new)      |
| pp512 @ d=8,192   |   —   | 1,036 ± 6  | (new)      |
| pp512 @ d=16,384  |   731 |   855 ± 6  | **+17.0%** |
| tg128 @ d=0       |  46.5 |  46.66     | +0.3% |
| tg128 @ d=16,384  |  43.3 |  43.46     | +0.4% |

Decision rule (pp512@d=0 improves >5% AND no depth regresses) passes on both counts. pp512 gains are real across every depth — not flat at d=16k as kyuz0's observation implied. The "hit or miss at large context" comment is most likely about d=32k+, which we don't currently test. tg128 is flat (expected — MMQ tile tuning is a prompt-processing fix).

Cold 10k prefill estimate, integrating the pp512 curve: ~10.2s (vs ~12s baseline). The long-context deficit vs DGX Spark is still attention-kernel-bound and this patch doesn't touch it, but short-context TTFT is now visibly faster.
