# Strix Halo topic docs (`strix-halo/`)

Markdown in this directory is the **lab notebook** for AMD Strix Halo (`gfx1151`) work: hypotheses, source pointers, benchmark tables, and dead ends.

**New here?** Read the root [`README.md`](../README.md) first - it is the plain-language orientation. Then pick up whichever of the three registers you need:

| Register | What it answers |
|----------|-----------------|
| [`findings.md`](findings.md) | What has been tried, what it was worth, what its status is now |
| [`backlog.md`](backlog.md) | What to try next and why, priced against the kernel profile |
| [`upstream.md`](upstream.md) | Which upstream changes hit us, how to resync, what to re-bench afterward |

**Code-level survey** (HIP / Vulkan / CPU, numbered §1-10): [`NOTES.md`](NOTES.md).

The rest of this directory is one file per experiment - hypotheses, source pointers, benchmark tables, and postmortems:

| Document | Topic |
|----------|--------|
| [`kv-cache.md`](kv-cache.md) | Quantized KV vs throughput at depth |
| [`fa-dispatcher.md`](fa-dispatcher.md) | Flash-attention MMA path / gfx1151 gate |
| [`uma-integrated.md`](uma-integrated.md) | `integrated = false` / UMA research |
| [`rocm-config.md`](rocm-config.md) | ROCm env flags (hipBLASLt batching, unroll) |
| [`mmq-rdna3_5.md`](mmq-rdna3_5.md) | MMQ tile tuning (PR #21344 port) — **code dropped 2026-07-16**, superseded by the config-table re-port |
| [`mmq-rdna3_5-config-table.md`](mmq-rdna3_5-config-table.md) | RDNA3.5 MMQ config table — **live**; reshaped 2026-08-02 onto upstream's own `mmq-config-rdna3-5.cuh` (PR #26199), port's own share still unmeasured |
| [`pp-rdna3_5-tile-mmq.md`](pp-rdna3_5-tile-mmq.md) | Dense MMQ + TILE FA D=256 follow-up — TILE FA half still live; MMQ half folded into the config-table re-port |
| [`mmq-moe-ncols-picker.md`](mmq-moe-ncols-picker.md) | Routed-MoE tile sizing: static `J=48` cap vs re-port of upstream PR #24546 — **reverted 2026-07-17**, flat (J=48 ≡ J=64 at ub=2048) |
| [`rocwmma-tuned.md`](rocwmma-tuned.md) | rocWMMA FA tuning (PR #16827 port) — **closed for good 2026-08-02**: upstream PR #26046 deleted the kernel and the flag |
| [`mmvq-rdna3_5.md`](mmvq-rdna3_5.md) | MMVQ routing notes |
| [`tg-at-depth-regression.md`](tg-at-depth-regression.md) | TG-at-depth regression — **resolved upstream by #22298, 2026-04-27** |
| [`jg-cuda-fa-rdna3-4.md`](jg-cuda-fa-rdna3-4.md) | FA MMA_F16 D=256 enablement (JG cherry-pick + 1-line guard widen) — **held, not promoted 2026-05-01** after honest f16/f16-KV A/B showed pp regression at depth |
| [`fa-rdna3-opsel-pair.md`](fa-rdna3-opsel-pair.md) | OPSEL-paired half2 accumulator port — **attempted, abandoned 2026-05-04** in test-backend-ops debugging spiral; its f32-accumulator premise is **stale**, see `fa-mma-d256-26419.md` |
| [`fa-mma-d256-26419.md`](fa-mma-d256-26419.md) | FA MMA_F16 at D=256 via upstream PR #26419 — **evaluated and declined 2026-08-29**: -4.7% pp@d=16k on stock upstream, -11.6% on our tree. Also carries the four-arm matrix that finally attributed the fork's patches at **+27.6-29.2% prefill** |
| [`kernel-time-breakdown.md`](kernel-time-breakdown.md) | rocprofv3 kernel-time shares (pp/tg × shallow/deep) — **measured 2026-07-18**, reprices the backlog |
| [`hip-graphs.md`](hip-graphs.md) | HIP graphs at decode — **traced 2026-07-18**: graphs already engaged in production; the 16% idle is replay-internal, no free lunch |
| [`qwen3.6-baseline.md`](qwen3.6-baseline.md) | Qwen 3.6 35B-A3B baseline numbers |
| [`qwen3.6-mtp.md`](qwen3.6-mtp.md) | Qwen 3.6 MTP runtime check — decode win, prompt-side cost |
| [`codex-insights.md`](codex-insights.md) | Consolidated assistant read of the above |
| [`mmq-table-check.cpp`](mmq-table-check.cpp) | Host-side check that the RDNA3.5 MMQ table is well-formed and dispatches right — no ROCm needed |

Deploy, Docker, and profiling automation are maintained in a separate private repo. Every doc here writes out its full bench command so results can be reproduced without it.
