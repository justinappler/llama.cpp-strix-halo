# RDNA3 FA-MMA via OPSEL-paired half2 accumulator

## Status (2026-05-04 — attempted, abandoned)

Prototype landed on `experiment/jg-fa-rdna3-half2` then `codex/rdna3-opsel-pair` (commit `86d0d38fe`) and **failed `test-backend-ops` en masse** as soon as it hit the FA sweep. Three follow-up commits walked the failure into a printf-instrumentation spiral (`ab0c92ec1` adds debug dumps gated by `GGML_CUDA_FA_RDNA3_DEBUG` for the small `DKQ=DV=64, ncols1=4, ncols2=1` probe shape; `cc9f9c132` hard-enables the macro; `b3cc76ad9` strips the guards entirely so it always fires) without isolating the root cause. **Abandoned without reaching the static-VGPR verification gate** the plan below set as Phase 0.

The structural premise was right — JG's f32 `T_C_VKQ` accumulator is the binding constraint at D=256 on RDNA3, and OPSEL pairing of `wmma_f16_16x16x16_f16_w32` is the canonical hardware usage for RDNA4-equivalent register cost. What we couldn't get past was the implementation: the paired `mma()` overload using `_tied_w32` (with non-tied fallback) plus the `tile_pair_16x8_half2_rdna3` lo/hi struct produced incorrect VKQ values at the smallest probe shape. Every probe path (lo/hi prints right after MMA, after VKQ-scale, into the combine path) showed values that didn't match the expected accumulation, but instrumentation alone couldn't disambiguate between (a) wrong OPSEL semantics for the `_tied` builtin, (b) wrong layout assumption in the `extract_lo`/`extract_hi` accessors, (c) wrong A/B pairing in the doubled `load_ldmatrix_trans`, or (d) wrong stride math in the half2-typed combine writeback. With backend-ops failures cascading across nearly every D=256 FA shape, narrowing further by printf was not converging.

The original analysis still holds: the perf ceiling at D=256 on RDNA3 is structural to JG's `tile<16, 16, float>` choice for `T_C_VKQ`, and the high-leverage path is upstream — when JG iterates `cuda-fa-rdna3-5+` or files a real PR, watch specifically for changes to the `mma_tile_sizes` block at [fattn-mma-f16.cuh:1026-1034](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1026-L1034). Doing this port locally is plausible in principle but not cost-effective without a working numerical-correctness harness that bisects which of (a)-(d) is wrong; printf debugging on a kernel hot loop wasn't it.

Branches `experiment/jg-fa-rdna3-half2` and `codex/rdna3-opsel-pair` retained as a postmortem reference. **Production stays on Finding #6 (TILE FA + rocWMMA OFF for D=256).** Finding #7 (`82736929a`) remains held; this document is its postmortem-by-association.

## Why we cared

`82736929a` is a measured regression at production f16/f16 KV: pp512@d=16k 851→660 t/s (−22.5%), tg128 flat. Static binary inspection localized the cost to `T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>` ([fattn-mma-f16.cuh:1026-1034](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1026-L1034)). Per-tile footprint sits at 8 VGPRs/lane vs RDNA4/CDNA's 4 — pins VGPR/wave at 256 and occupancy at 6 waves/SIMD. K/V-tile halving (`79bfad7f3`) moved spills but not VGPRs, confirming the accumulator type is the binding constraint.

Master remained on Finding #6 (TILE + rocWMMA OFF) throughout. This experiment, had it landed, would have obsoleted both Finding #6 and the held `82736929a`, bringing D=256 FA to RDNA3 at RDNA4-equivalent register cost. JG was not iterating his branch (last activity 2026-04-26 at `b4ef403e`); the upstream watch was paused, so the fork pursued the proper port directly.

## Mechanism — OPSEL pairing is the canonical RDNA3 WMMA usage

RDNA3's `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, opsel)` writes a `halfx16_t` accumulator across 8 VGPRs/lane. The 4th OPSEL argument selects whether the result lands in the lo or hi half of each 32-bit slot. A single call with OPSEL=0 fills only the lo halves; the hi halves preserve the input value. Two calls with opposite OPSEL write into the same 8 VGPRs without interfering — together holding two independent 16x16 outputs.

Used this way, two logical `tile<16, 8, half2>` outputs share one 8-VGPR register group → effective **4 VGPRs per logical tile**, matching RDNA4's `halfx8_t` footprint exactly. JG's `floatx8_t` accumulator (one tile per 8-VGPR group, OPSEL trick unused) is the workaround. OPSEL pairing is what RDNA3's hardware was designed for.

Reference points already in tree:

- **RDNA4 packed path (target footprint):** [mma.cuh:947-957](../ggml/src/ggml-cuda/mma.cuh#L947-L957) — `wmma_f16_16x16x16_f16_w32_gfx12` directly into `halfx8_t`.
- **RDNA3 f32 path (current, the workaround):** [mma.cuh:1143-1149](../ggml/src/ggml-cuda/mma.cuh#L1143-L1149) — `wmma_f32_16x16x16_f16_w32` into `floatx8_t`.
- **RDNA3 f16 path (target, currently `NO_DEVICE_CODE`):** [mma.cuh:954-957](../ggml/src/ggml-cuda/mma.cuh#L954-L957) for the `tile<16,8,half2>` mma() overload. [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063) attempted a naive port and produced incorrect output — the bug was treating the unpacked WMMA writeback as if it were packed. The `tile_pair` abstraction below explicitly models the unpacked layout. **The prototype on this branch failed for what looks like a related class of bug.**

## Plan that was attempted

Branch `experiment/jg-fa-rdna3-half2` off `82736929a`. Sequence:

### Phase 1 — `tile_pair` abstraction in mma.cuh (landed)

New `tile_pair_16x8_half2_rdna3` wrapping 8 VGPRs of accumulator state representing two logical `tile<16, 8, half2>` outputs (lo and hi halves of the pair). Provided:

- `mma()` overload taking the pair, two A/B pairs; issued two `wmma_f16_16x16x16_f16_tied_w32` calls with OPSEL=false and OPSEL=true (with non-tied fallback if the `_tied` builtin wasn't available).
- `extract_lo()` / `extract_hi()` accessors returning standard `tile<16, 8, half2>` for downstream consumption.
- Default-init zero (both halves cleared before first use, since each OPSEL only touches its half).

Code: [mma.cuh:680-694](../ggml/src/ggml-cuda/mma.cuh#L680-L694) and [:1007-1029](../ggml/src/ggml-cuda/mma.cuh#L1007-L1029) in commit `86d0d38fe`.

### Phase 2 — restructure VKQ tile loop (landed)

[fattn-mma-f16.cuh:861-998](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L861-L998): `VKQ_C` became a `tile_pair_16x8_half2_rdna3` array of length `(DV/T_A_VKQ::I)/2`. The K-loop body issued the doubled `load_ldmatrix_trans` (one for the lo i_VKQ_0, one for the hi i_VKQ_0+T_A_VKQ::I) and called the paired `mma()`. VKQ-scale and combine paths got dedicated RDNA3 branches that walked the lo/hi halves explicitly.

### Phase 3 — `mma_tile_sizes` flip (landed)

```diff
 #elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
 template<int ncols> struct mma_tile_sizes {
     using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
     using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
     using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>; // KQ stays f32: softmax range, single tile not 16
     using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
     using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
-    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;
+    using T_C_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>; // paired; physical storage 4 VGPRs/tile
 };
```

### Phase 0 / verification gate (NEVER REACHED)

The plan called for static binary measurement *before any bench* — `llvm-objdump` recipe in [jg-cuda-fa-rdna3-4.md § Mechanism](jg-cuda-fa-rdna3-4.md#mechanism-2026-05-03--register-pressure--scratch-spills) — to confirm VGPRs/wave dropped from 256 to ≤192 across all 16 D=256 MMA configs. This was Phase 0 specifically because the load-bearing risk was "compiler doesn't allocate the paired accumulator into a single 8-VGPR group" (see Risks below). **We never got there.** `test-backend-ops` was meant to be Phase 1 of *runtime* verification after the static phase passed; it was the first thing run instead, and it failed before anything else got measured.

## What actually happened

`test-backend-ops -o FLASH_ATTN_EXT -b ROCm0` produced cascading NMSE failures across the D=256 FA cases — not the four narrow `max_bias=0, nb=32, type_KV=f16` failures the held `82736929a` already produces (those are a known precision quirk in JG's `__shfl_xor_sync` repack), but a much broader pattern hitting the `nb > 1` shapes the OPSEL kernel was supposed to handle. The kernel ran (no traps, no `NO_DEVICE_CODE`), it just produced numerically wrong output.

Three things tried, none diagnostic enough:

1. **`ab0c92ec1`** added printf instrumentation gated by `GGML_CUDA_FA_RDNA3_DEBUG`, dumping `VKQ_C[0].lo.x[0]` / `.hi.x[0]` at four checkpoints (after MMA, before VKQ-scale, after VKQ-scale, in the combine writeback) for the smallest sweep shape: `DKQ=DV=64, ncols1=4, ncols2=1, blockIdx=(0,0), threadIdx=(0,0)`.
2. **`cc9f9c132`** flipped the macro on by default since the gate was being missed at build time.
3. **`b3cc76ad9`** stripped the `#if GGML_CUDA_FA_RDNA3_DEBUG` guards entirely so the prints fired unconditionally on the rdna3 path.

Output never narrowed the bug to a single layer. The dumps showed values that disagreed with hand-computed expectations, but printf alone couldn't tell us *which* of the candidate failures was responsible:

- **(a) `_tied_w32` builtin OPSEL semantics.** The `__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32` form was chosen because the documentation reads as "preserve the other-half on this call"; if the actual semantics differ from the non-tied form (e.g. tied requires a specific A/B alignment or input layout), both calls would silently corrupt each other. No isolated unit test was written.
- **(b) `extract_lo` / `extract_hi` layout assumption.** The accessors returned the lo and hi `tile<16,8,half2>` as if the unpacked `halfx16_t` accumulator stored them as adjacent half2 lanes. PR #19063's failure mode was exactly this kind of unpacked-layout misunderstanding. The `tile_pair` abstraction was *supposed to* model the unpacked layout explicitly — the implementation may have only modeled it nominally.
- **(c) Doubled `load_ldmatrix_trans`.** The K-loop pair-load reused the same `B[k00/...]` for both A_lo and A_hi but loaded different V-tile rows (`i_VKQ_0` and `i_VKQ_0 + T_A_VKQ::I`). If the load stride or the mirrored-layout offsetting was off-by-one for the second load, A_hi would be wrong but A_lo right; the printf showed only `[0].lo.x[0]` and `[0].hi.x[0]`, which is the *first* output tile pair's accumulator state, not directly the input loads.
- **(d) Combine-writeback stride math.** The half2-typed combine path indexed `tile_Q[j*tile_stride + k_lo]` and `[k_hi]` with `k_hi = k_lo + T_C_VKQ::J`. If the OPSEL-paired layout requires interleaving differently from a standard packed half2 tile, the output would be scrambled even with correct accumulator values.

To bisect properly we'd have needed:

- A **standalone test** of `tile_pair_16x8_half2_rdna3 + paired mma()` against a CPU reference, with no FA kernel context. That isolates (a) and (b).
- An **AMDGCN ISA dump** showing whether the compiler actually allocated `acc_lo` and `acc_hi` into the same 8-VGPR group (the Phase 0 gate). If it didn't, the OPSEL pairing isn't doing what we think regardless of (a)-(d), and the perf premise dies anyway.
- A **test for (c)** by feeding known A_lo/A_hi inputs that are constants (e.g. all-ones, identity-shaped), independent of the V-tile load math.
- A **test for (d)** by writing a known accumulator state directly into VKQ_C and verifying the combine output.

None of those got built. The escalation pattern (prototype → instrument → harden macro → strip guards) is exactly the smell of a debugging session that shouldn't have started in-kernel — the right move was to back out, build the unit test for Phase 0, and re-attempt only after the compiler's VGPR allocation was confirmed.

## Decision rules (with hindsight)

| Result | Action that was specified | What actually triggered |
| --- | --- | --- |
| Static phase fails (VGPRs unchanged) | Annotate, abandon before bench. | **Static phase was never run** — went straight to backend-ops, which failed on numerics. |
| pp512@d=16k beats `5d34ca3b` AND tg128 within ±2% across depths | Promote — replaces Finding #6. | N/A |
| pp512@d=16k matches `5d34ca3b` (±2%) | Keep on branch, don't promote. | N/A |
| pp512@d=16k worse than `5d34ca3b` but better than `82736929a` | Dead end — OPSEL idea right but something else dominates. Annotate, walk back. | N/A |
| Worse than `82736929a` | Numerical bug or compiler losing the pairing. Annotate, abandon. | This row, modulo "never benched". |

## Risks (which mostly fired)

- **Compiler may not allocate the paired accumulator into a single 8-VGPR group.** LLVM's AMDGPU backend has to recognize that two `wmma_f16_*_w32` calls writing opposite OPSEL halves can share VGPR allocation. If it allocates separately (16 VGPRs/pair), we get the same VGPR cost as JG's f32 path with worse precision. Mitigation was *supposed* to be ISA inspection after Phase 1 — this never happened, so we don't even know if the perf premise holds, never mind the correctness premise.
- **PR #19063's "incorrect output" wall.** Their failure was ignoring the unpacked output layout. The `tile_pair` abstraction was meant to model it explicitly, but the test-backend-ops cascade suggests the modeling wasn't actually correct, or wasn't preserved through the `extract_lo`/`extract_hi` path. **This is the most likely root cause, given the symptom.**
- **JG resumes work.** If `cuda-fa-rdna3-5+` appears, drop ours and re-bench upstream. Long-term value of this branch is the postmortem and the upstream bug report, not the code.

## What to try if this is revived

1. **Phase 0 first.** Build the prototype, then immediately run `llvm-objdump` on the gfx1151 ELF for one of the D=256 MMA kernel symbols. If `acc_lo` and `acc_hi` aren't in the same 8-VGPR group, abandon — the perf premise is dead and there's no point fixing the numerics.
2. **CPU-vs-GPU unit test for `tile_pair_16x8_half2_rdna3 + mma()` in isolation.** Outside the FA kernel. Feed constant `A_lo, B_lo, A_hi, B_hi`, compare against a CPU-side `__half`-precision matmul reference. This isolates (a) and (b) cleanly. Until this passes, do not touch fattn-mma-f16.cuh.
3. **Try `wmma_f16_16x16x16_f16_w32` (non-tied) as the primary**, not the fallback. The `_tied` form's OPSEL semantics are what the prototype assumed but never independently verified; the non-tied form's behavior at OPSEL=true is documented more clearly in the LLVM AMDGPU intrinsics reference.
4. **Validate (c) and (d) only after (1)-(3) are clean.** Plug the verified `tile_pair` into the FA kernel, run `test-backend-ops` immediately at the smallest D=256 shape, and only escalate to the full sweep once that passes.

## Cross-references

- [jg-cuda-fa-rdna3-4.md](jg-cuda-fa-rdna3-4.md) — Finding #7, the held cherry-pick this branched from; mechanism + measurement that motivated this attempt.
- [fa-dispatcher.md](fa-dispatcher.md) — Finding #2, original dispatcher dead-end and Option A/B/C framing (this was a refined Option A: tile-layout rework, not a repack-on-write hack).
- [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063) — closed-by-author RDNA3 WMMA naive port. Same failure class as this attempt: "treat unpacked layout as packed → wrong output". Worth re-reading before any retry.
- [PR #16827 comment](https://github.com/ggml-org/llama.cpp/pull/16827#issuecomment-3454830174) — JG's roadmap; FA-MMA-on-AMD-WMMA hasn't merged, branch hadn't moved since 2026-04-26.
- Branches retained: `experiment/jg-fa-rdna3-half2`, `codex/rdna3-opsel-pair` (the four commits `86d0d38fe`, `ab0c92ec1`, `cc9f9c132`, `b3cc76ad9` — prototype + instrumentation escalation).
