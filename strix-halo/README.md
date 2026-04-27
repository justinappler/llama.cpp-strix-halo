# Strix Halo topic docs (`strix-halo/`)

Markdown in this directory is the **lab notebook** for AMD Strix Halo (`gfx1151`) work: hypotheses, source pointers, benchmark tables, and dead ends.

**Overview and findings table:** start at the repository root [`README.md`](../README.md) (Strix Halo section at the top).

**Code-level survey** (HIP / Vulkan / CPU, numbered §1–10): [`NOTES.md`](NOTES.md) — **ranked next experiments** are in the root [`README.md`](../README.md#strix-halo-next-experiments) only.

| Document | Topic |
|----------|--------|
| [`kv-cache.md`](kv-cache.md) | Quantized KV vs throughput at depth |
| [`fa-dispatcher.md`](fa-dispatcher.md) | Flash-attention MMA path / gfx1151 gate |
| [`uma-integrated.md`](uma-integrated.md) | `integrated = false` / UMA research |
| [`rocm-config.md`](rocm-config.md) | ROCm env flags (hipBLASLt batching, unroll) |
| [`mmq-rdna3_5.md`](mmq-rdna3_5.md) | MMQ tile tuning (PR #21344 port) |
| [`rocwmma-tuned.md`](rocwmma-tuned.md) | rocWMMA FA tuning (PR #16827 port) — **flag flipped back OFF 2026-04-27** after regression on Qwen 3.6 |
| [`mmvq-rdna3_5.md`](mmvq-rdna3_5.md) | MMVQ routing notes |
| [`tg-at-depth-regression.md`](tg-at-depth-regression.md) | TG-at-depth regression — **resolved upstream by #22298, 2026-04-27** |
| [`qwen3.6-baseline.md`](qwen3.6-baseline.md) | Qwen 3.6 35B-A3B baseline numbers |
| [`codex-insights.md`](codex-insights.md) | Consolidated assistant read of the above |

Deploy, Docker, and profiling automation live in [**server-configs** `services/llamacpp/profiling/`](https://github.com/justinappler/server-configs/blob/main/services/llamacpp/profiling/README.md).
