# MMVQ parameter table: route RDNA3.5 to RDNA3_0

## Hypothesis

`get_device_table_id` in [mmvq.cu](../ggml/src/ggml-cuda/mmvq.cu) assigns every AMD arch to one of five tuning tables (`GENERIC`, `GCN`, `RDNA2`, `RDNA3_0`, `RDNA4`). Today gfx1151 (RDNA 3.5) is grouped with RDNA2 on both the device constexpr ([line 77](../ggml/src/ggml-cuda/mmvq.cu#L77)) and the host path ([line 93](../ggml/src/ggml-cuda/mmvq.cu#L93)).

RDNA2 lacks dual-issue VALU and WMMA; RDNA 3.5 has both, just like RDNA3_0. Sending RDNA3.5 through the RDNA2 table forces `nwarps=1` for every mmvq dispatch — only `GENERIC` and `GCN` override, and `RDNA2` is not among them. The RDNA3_0 branch in `calc_nwarps` ([mmvq.cu:348](../ggml/src/ggml-cuda/mmvq.cu#L348)) returns `nwarps=8` when `ncols_dst==1` for a whitelist of quants (Q4_0/1, Q5_0/1, Q8_0, Q4_K, Q6_K, IQ4_NL). Q4_K dominates Q4_K_XL tensor mass, so RDNA3.5 should benefit from the same parallelism the W7900 gets.

This is NOTES.md item #2, the highest-ROI experiment after the MMQ win (commit `d8ad713`). The companion host/device max-batch path (`get_mmvq_mmid_max_batch`) already treats RDNA3.5 via `GGML_CUDA_CC_IS_RDNA3(cc)` — only the parameter-table selector is out of step.

Upstream [PR #21344](https://github.com/ggml-org/llama.cpp/pull/21344) originally proposed giving RDNA3.5 its own mmvq table but reverted that half after review (commit `7957de9d` "revert changes to mmvq.cu"). The reviewer feedback there is about process, not correctness — so the question "should RDNA3.5 share RDNA3_0's table?" is still worth benchmarking on this specific chip.

## What we expect to measure

tg128 is the natural target — MMVQ is the bs=1 decode path. The MMQ port (Finding #5) moved pp hard and tg barely (+0.3 % across depths). If this patch is a pure MMVQ tuning flip, the mirror prediction is: tg128 moves, pp512 barely moves. Possible shapes:

- **Win:** tg128 at d=0 goes up a few percent, holds at depth. That's the RDNA3_0 nwarps=8 paying off for Q4_K.
- **Null:** no movement. Means RDNA3.5's register file / LLC / BW profile doesn't reward RDNA3_0-scale parallelism the way W7900's does.
- **Regress:** tg128 drops, most likely at depth. Means register pressure at `nwarps=8` costs more than the extra warps win. This is the scenario that got the upstream revert nominated in the first place.

Any tg128 regression at any depth triggers revert — decode t/s is the UX-visible number during generation.

## The patch

Two-line edit to [ggml/src/ggml-cuda/mmvq.cu](../ggml/src/ggml-cuda/mmvq.cu). Device and host must stay in sync (the kernel's `__launch_bounds__` is computed from `get_device_table_id()` at compile time, and host launch math uses `get_device_table_id(cc)` — upstream commit `88d5f8ffc` exists specifically to enforce that invariant).

**Device constexpr ([mmvq.cu:75-78](../ggml/src/ggml-cuda/mmvq.cu#L75-L78)):** move `RDNA3_5` from the `RDNA2 || RDNA3_5` arm to the `RDNA3_0` arm.

**Host function ([mmvq.cu:90-95](../ggml/src/ggml-cuda/mmvq.cu#L90-L95)):** analogous move of `GGML_CUDA_CC_IS_RDNA3_5(cc)` from the RDNA2 branch to the RDNA3_0 branch.

## Bench plan

Same shape as [mmq-rdna3_5.md](mmq-rdna3_5.md): Qwen3.6 35B-A3B Q4_K_XL, Run 3 knobs (f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3`), depths `{0, 2048, 8192, 16384}`.

Decision rule: keep if tg128 @ d=0 improves >3 % outside noise **and** no depth regresses tg128 or pp512. Revert on any tg regression — agentic coding workloads feel decode speed directly.

## Outcome

**Reverted.** tg128 collapsed at every depth; pp512 held at d=0 but regressed sharply from d=8k onward. Qwen3.6 35B-A3B Q4_K_XL, same knobs as [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 3, build `35291ec`, compared against MMQ baseline (`d8ad713`):

| test | baseline (d8ad713) | RDNA3_0 table (35291ec) | delta |
|---|---:|---:|---:|
| pp512 @ d=0       | 1,309        | 1,316.18 ± 18.26 | +0.5 % (noise) |
| pp512 @ d=2,048   | 1,232        | 1,213.36 ± 18.32 | −1.5 % (noise) |
| pp512 @ d=8,192   | 1,036        |   960.00 ± 5.02  | **−7.3 %**  |
| pp512 @ d=16,384  |   855        |   740.31 ± 4.70  | **−13.4 %** |
| tg128 @ d=0       |  46.66       |    38.40 ± 0.12  | **−17.7 %** |
| tg128 @ d=2,048   |  — (unmeasured) |  36.62 ± 0.03  | n/a        |
| tg128 @ d=8,192   |  — (unmeasured) |  31.90 ± 0.01  | n/a        |
| tg128 @ d=16,384  |  43.46       |    26.99 ± 0.04  | **−37.9 %** |

Every decision criterion fails: tg128 doesn't improve at d=0, tg regresses at every depth, pp512 regresses at depth ≥ 8k. Reverted on next commit.

### What the numbers actually say

The hypothesis "regress via register pressure at `nwarps=8`" is the simplest fit. `calc_nwarps` at [mmvq.cu:348](../ggml/src/ggml-cuda/mmvq.cu#L348) returns 8 for Q4_K at `ncols_dst=1` only (i.e., the bs=1 decode path). That maps exactly to what regressed: tg128 is bs=1 and collapses everywhere, while pp512 at d=0 (where `ncols_dst > 1` for the GEMV column-grouping) survives. The depth-dependent pp512 fall-off is attention-side, not MMVQ — KV grows, attention cost grows, and whatever headroom the change cost elsewhere compounds.

That's consistent with RDNA3.5's smaller register file vs W7900. W7900 has ~192 KB VGPR/CU and 40 MB L3; RDNA 3.5 has the same nominal VGPR but no discrete L3, and LLC is shared with the CPU. At `nwarps=8` on Q4_K the kernel's live-register footprint spills or forces low-occupancy scheduling — exactly the kind of win-in-FLOPS, lose-in-bandwidth tradeoff the RDNA3.5 profile punishes.

### Follow-ups this suggests

- The "give RDNA3.5 its own table" variant (PR #21344's original plan, reverted after review) is more promising than "join RDNA3_0" — a tighter whitelist (say, only Q4_0/Q8_0 where vec_dot is simple) plus `nwarps ∈ {2, 4}` is the natural next sweep. Parking this until there's a concrete reason to revisit — register-pressure tuning is a many-knob search and the cheap version already failed.
- NOTES.md #2 should be demoted or rewritten: "join RDNA3_0" is the wrong question. The real question is whether any non-RDNA2 table suits RDNA 3.5, and at what nwarps.

### Upstream context

This patch implements one half of what [PR #21344](https://github.com/ggml-org/llama.cpp/pull/21344) originally proposed and then withdrew (commit `7957de9d` "revert changes to mmvq.cu"). The upstream reviewer's skepticism was correct for this chip: on this workload it is a real regression. Left as a dead end doc rather than deleted, because the next person who sees NOTES.md item #2 will otherwise try the same thing.
