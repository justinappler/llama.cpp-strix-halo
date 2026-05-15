# JohannesGaessler `cuda-fa-rdna3-4` — what's done, what's not, what it means for us

## Status (2026-05-14 — superseded by upstream PR #22880)

The `cuda-fa-rdna3-*` chain landed upstream as [PR #22880](https://github.com/ggml-org/llama.cpp/pull/22880) (merged 2026-05-14). Upstream went the **opposite** direction from this held cherry-pick at D>128 — kept the TILE kernel rather than enabling MMA. From JG's commit message:

> For RDNA3/4 I was not able to get better performance than the tile kernel for head sizes > 128.

So the question "does the JG chain unlock D=256 FA on gfx1151?" is now answered: no — at least not in a way that beats TILE. The D≤128 path does get the new mma FA kernel for free post-rebase. Held branches `experiment/jg-fa-rdna3` and `experiment/jg-fa-rdna3-tune` are archival; can be deleted. Static-trace and structural-ceiling analyses below are preserved as historical record.

## Status (2026-05-04 — held, structural ceiling identified)

JG's branch (cherry-pick `0cf15294b` on `experiment/jg-fa-rdna3`) plus a **one-line widening** of the line-1672 device guard (commit `82736929a`) unlocks D=256 FA on gfx1151 cleanly at the kernel level — the static trace below was the probe and a near-clean test-backend-ops sweep (2844 / 2848) confirmed it. **But the controlled A/B against actual production KV config (f16/f16) is a clear regression at depth: pp512@d=16k 851 → 660 t/s (−22.5%), tg128 flat across all depths.** See [§ Outcome](#outcome) below.

**Update 2026-05-04**: static binary inspection ([§ Mechanism](#mechanism-2026-05-03--register-pressure--scratch-spills)) plus a controlled K/V-tile reduction experiment ([§ Followup](#followup-experiment--halve-nbatch_k2--nbatch_v2-for-rdna-d256-2026-05-04)) localize the regression to **JG's f32 `T_C_VKQ` accumulator** ([:1026-1034](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1026-L1034)) — the workaround for RDNA3's unpacked-half2 WMMA write-back. That choice doubles the accumulator's per-lane register footprint vs RDNA4/CDNA's `tile<16, 8, half2>` and pins VGPR/wave at 256 regardless of tile-size knobs (confirmed by direct measurement). The 6-waves/SIMD occupancy ceiling at D=256 is structural; tile-size sweeps in the existing config table can't move it. Path forward is upstream: re-bench when JG iterates the `mma_tile_sizes` block.

The original "+15.2× pp / +2.3× tg" framing on 2026-04-29 was real-but-misframed — those numbers compared the patched f16/f16 run against [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 1 (q8_0/q4_0 KV, the abandoned pre-Finding-#1 production config), not against current production state. The headline reversed when re-benched against the right baseline on 2026-05-01.

**Branch retained, not promoted to `master`.** The cherry-pick remains useful as: (1) a real win for any q8_0/q4_0 KV workload (still ~15× at d=16k for that config), (2) a base for re-bench when JG iterates the branch (`-5` and beyond) or lands a real PR. Production stays on Finding #6 (rocWMMA OFF + TILE).

**Earlier on 2026-04-29**, the un-widened cherry-pick was tested via `test-backend-ops -o FLASH_ATTN_EXT -b ROCm0` on gfx1151 and got **2,674 FA cases passed, then GPU hang at the first D=256 prefill case** — the load-bearing line-1672 `DKQ > 128` clause that the static trace then identified.

## Why we care

Our hot path is Qwen 3.6 35B-A3B at D=256. The current state of the world on master:

- FA dispatcher routes D=256 to **TILE** ([fa-dispatcher.md](fa-dispatcher.md)). TILE has no tensor-core path on RDNA 3.5 — biggest dead silicon on the chip.
- The rocWMMA FA path is `#if`-gated OFF after the 2026-04-27 regression ([rocwmma-tuned.md § Re-bench 2026-04-27](rocwmma-tuned.md#re-bench-2026-04-27--flag-back-off-regression)).
- Kernel slated for upstream removal once FA-MMA-on-AMD-WMMA lands ([PR #16827 thread](https://github.com/ggml-org/llama.cpp/pull/16827#issuecomment-3454830174)).

JG's `cuda-fa-rdna3-4` branch is the concrete WIP behind that roadmap — the branch that, when finished, unblocks D=256 FA on RDNA 3.5 and obsoletes both Finding #2 and Finding #6.

## What JG's branch does (as of `b4ef403e`)

Three files, all under [`ggml/src/ggml-cuda/`](../ggml/src/ggml-cuda/). Diff against his merge base (`9d34231b`, several weeks behind upstream): +145/−52 in `fattn-mma-f16.cuh`, +4/−15 in `fattn.cu`, +19/−2 in `mma.cuh`.

### The layout-mismatch fix that PR #19063 lacked

The blocker that killed [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063) was that RDNA3's `wmma_f16_16x16x16_f16_w32` writes an **unpacked** `halfx16_t` (one half per 32-bit lane), while RDNA4's packed format is what the surrounding `mma.cuh` tiles assume. JG's answer ([mma.cuh:660-679](../ggml/src/ggml-cuda/mma.cuh#L660-L679)) is to use the **f32 accumulator** intrinsic and convert to packed `half2` via `__shfl_xor_sync`:

```cpp
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
static __device__ __forceinline__ tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> get_half2(
        const tile<16, 16, float, DATA_LAYOUT_I_MAJOR> & tile_float) {
    tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> ret;
#pragma unroll
    for (int l = 0; l < tile_float.ne; ++l) {
        float tmp[2];
        int i = threadIdx.x / 16;
        tmp[i] = tile_float.x[l];
        i ^= 1;
        tmp[i] = __shfl_xor_sync(0xFFFFFFFF, tile_float.x[l], 16, WARP_SIZE);
        ret.x[l] = make_half2(tmp[0], tmp[1]);
    }
    return ret;
}
```

This is **Option C** from [fa-dispatcher.md § What an actual fix would look like](fa-dispatcher.md#what-an-actual-fix-would-look-like) ("f32 accumulator + convert"), combined with **Option B** (rework tile layouts: the new `DATA_LAYOUT_I_MAJOR_MIRRORED` carries the unpacked-but-rearranged form).

### Host dispatcher widened

[fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454) — the RDNA4-only + `Q->ne[0] <= 128` gate that [fa-dispatcher.md](fa-dispatcher.md) flagged is gone:

```diff
-if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
+if (((amd_wmma_available(cc) && gqa_opt_applies) || amd_mfma_available(cc)) && Q->ne[0] != 40 && Q->ne[0] != 72) {
```

Any RDNA + `gqa_opt_applies` + non-degenerate head-dim now routes to MMA_F16. Includes the D=256 case Qwen 3.6 exercises.

### Forward-looking RDNA config table

[fattn-mma-f16.cuh:122-160](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L122-L160) replaces the old "fall through to Ampere" table with a full RDNA-specific one for D ∈ **{64, 80, 96, 112, 128, 256, 512, 576}**, each at four `ncols` values. The `(256, 256, ...)` entries explicitly target our shape:

```cpp
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 2,  32, 128, 128, 128, 1, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 2,  32, 128, 128, 128, 1, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 1, true);
GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 1, true);
```

The intent to support D=256 is unambiguous.

### Other RDNA3 plumbing

- [fattn-mma-f16.cuh:1014-1023](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1014-L1023) — dedicated `mma_tile_sizes` for `AMD_WMMA && RDNA3`, all using the new `DATA_LAYOUT_I_MAJOR_MIRRORED` tile layout.
- RDNA3-specific mask-load path [fattn-mma-f16.cuh:735-744](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L735-L744): the upstream `half2`-consecutive load assumption breaks on RDNA3's mask layout, so JG falls back to per-half loads with `__half2float`.
- VKQ scaling specialized for f32 vs `half2` accumulator types [fattn-mma-f16.cuh:861-879](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L861-L879) — required because his RDNA3 path uses f32 accumulators while RDNA4/CUDA use `half2`.
- [mma.cuh:820-834](../ggml/src/ggml-cuda/mma.cuh#L820-L834) — `load_ldmatrix_trans` templated over `data_layout`, with `static_assert` on which layouts each backend supports.
- Two `(AMD_WMMA_AVAILABLE && RDNA4)` → `AMD_WMMA_AVAILABLE` device-side guards widened ([fattn-mma-f16.cuh:541](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L541) and [:1006](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1006)).

## What's NOT done

### The kernel entry-point still gates D > 128 to NO_DEVICE_CODE

[fattn-mma-f16.cuh:1671-1676](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1671-L1676) — inside the `flash_attn_ext_f16` global kernel:

```cpp
#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 128) {
        NO_DEVICE_CODE;
        return;
    }
#endif
```

The host-side dispatcher widening + the forward-looking config table both _advertise_ D=256 support, but this device-side guard says "no". On gfx1151 with D=256 + `Q->ne[1] > 1`, the dispatcher routes the call to this kernel, the kernel hits `NO_DEVICE_CODE` (which is a `__trap`), GPU hangs.

**This is the single load-bearing guard for D=256 enablement on RDNA3.** When someone widens or removes this `DKQ > 128` clause, the branch becomes useful for our workload.

### Possible additional inner guards (unverified)

We didn't exhaustively review every `NO_DEVICE_CODE` site. Counts on our cherry-pick: 11 NO_DEVICE_CODE call sites in `fattn-mma-f16.cuh`, of which the line-1672 entry guard is the only one we've directly hit. Others may also have stale RDNA3 exclusions that surface once the entry guard is opened. Won't know without re-testing.

## Static trace (2026-04-29) — what's actually wired up

Before risking a build-and-bench cycle on hardware, traced the device-side code path for WMMA+RDNA3+DKQ=256 statically. **Result: line 1672 is the only static obstruction.** Every other piece of the path is in place. Details by checkpoint:

| # | Checkpoint | Verdict | Evidence |
|---|---|---|---|
| 1 | Config table macro instantiation | Wired up | [fattn-mma-f16.cuh:151-154](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L151-L154); RDNA `(256, 256, ncols)` rows hit the macro static_asserts (`nbatch_fa ∈ {32,64} ≤ 256`, `nbatch_K2/V2 = 128 ≤ 512/256`, `nbatch_combine ∈ {64,128} ≤ 128`) cleanly |
| 2 | `mma_tile_sizes` for WMMA+RDNA3 | Not parameterized over DKQ | [fattn-mma-f16.cuh:1026-1034](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1026-L1034) — fixed `tile<16,8,half2,DATA_LAYOUT_I_MAJOR_MIRRORED>` and `tile<16,16,float,DATA_LAYOUT_I_MAJOR>`; same shapes used at any DKQ |
| 3 | `get_half2` shape match | Single signature, exact match | [mma.cuh:666-679](../ggml/src/ggml-cuda/mma.cuh#L666-L679) accepts T_C_KQ from #2 and returns T_A_VKQ from #2 |
| 4 | VKQ scaling f32 path | Explicit branch | [fattn-mma-f16.cuh:880-888](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L880-L888) — `if constexpr` on `T_C_VKQ::x` type, with `static_assert` catching unknown types. Loop `i < DV/T_C_VKQ::J` works at DV=256 (16 iterations) |
| 5 | Mask-load | DKQ-independent | [fattn-mma-f16.cuh:746-753](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L746-L753) — buffer is `nbatch_fa × ncols1`-shaped, doesn't reference DKQ |
| 6 | Other `NO_DEVICE_CODE` sites | None block our path | Audited 9 sites total. 1005/1622/1801 are backend selectors (we satisfy `AMD_WMMA_AVAILABLE`); 1103 is runtime dispatch-validity (`cols_per_warp > ncols` = 16 > 8 only for ncols=8, which the line-1672 guard already excludes); 1659/1666/1680 are Volta/Turing/MFMA-only. **Site 1654 explicitly whitelists DKQ=256 for `use_logit_softcap`** — strong signal the codebase already anticipates DKQ=256 in the WMMA path |
| 7 | `static_assert`s | All pass for our configs | 36 asserts checked. Top-level (28-34, 1107, 1370, 1489) all pass arithmetically for `(DKQ=256, DV=256, ncols ∈ {16,32,64})`. Risky-looking ones (1199 `nbatch_K2 == DKQ/2`, 1508 VKQ type) sit inside `if constexpr` blocks (`nstages > 1`, `cols_per_warp == 8`) that never instantiate — our configs have `nstages_target=1` and `cols_per_warp=T_B_KQ::I=16` |

### What this changes about the plan

Phase 1 is just a one-line widening of the line-1672 guard, no probing required. The static trace is the probe.

```diff
 #if defined(AMD_WMMA_AVAILABLE)
-    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 128) {
+    if (ncols1*ncols2 < 16 || ncols2 == 1) {
         NO_DEVICE_CODE;
         return;
     }
 #endif // defined(AMD_WMMA_AVAILABLE)
```

Phase 2 (compile-error / additional-guard triage) is **not expected to fire**. If it does, the static trace was wrong somewhere — bug-hunt that, don't widen further by guess.

### What the trace did NOT verify (runtime-only concerns)

- **Numerical correctness** of the f32-accumulator + `__shfl_xor_sync` packed-half2 conversion at [mma.cuh:666-679](../ggml/src/ggml-cuda/mma.cuh#L666-L679). Static analysis confirms shape and call-site fit; only a CPU-vs-GPU NMSE comparison verifies math.
- **Wave size**. The trace assumes `ggml_cuda_get_physical_warp_size() == 32` on gfx1151 (wave32). The shfl masks are `0xFFFFFFFF` and shfl offsets up to 16 — consistent with wave32, would silently miscompute on wave64. Inherited assumption, not re-verified.
- **Performance**. Static trace says nothing about whether MMA_F16 at D=256 actually beats TILE on Qwen 3.6 — that's the Phase 3 question.

## What we ran

```bash
docker exec llamacpp /app/test-backend-ops test -o FLASH_ATTN_EXT 2>&1 | tee /tmp/fa-sweep.log
```

(`test-backend-ops` from `/app/test-backend-ops` after flipping `LLAMA_BUILD_TESTS=ON` in [Dockerfile](https://github.com/justinappler/server-configs/blob/main/services/llamacpp/files/Dockerfile) for the lab build. See [test-backend-ops.cpp:8598-8649](../tests/test-backend-ops.cpp#L8598-L8649) for the FA sweep matrix.)

### Result

|                                    |                           count |
| ---------------------------------- | ------------------------------: |
| FA cases reached the backend       |                           4,242 |
| Passed (CPU vs ROCm0, NMSE < 5e-4) |                       **2,674** |
| Skipped ("not supported")          |                           1,568 |
| Failed                             | 0 (process killed before tally) |

Last passing case before the trap:

```
FLASH_ATTN_EXT(hsk=256,hsv=256,nh=4,nr23=[4,1],kv=512,nb=1,
  mask=1,sinks=1,max_bias=0,logit_softcap=0,prec=f32,
  type_KV=f16,permute=[0,2,1,3]): OK
```

`nb=1` (decode) routes to the VEC kernel, not MMA, so it didn't hit the guard. The next case in the sweep (likely `nb=3` at the same shape) routed to MMA → trap → GPU hang → process killed → rest of the sweep didn't run.

Trap output:

```
fattn-mma-f16.cuh:1673: ERROR: HIP kernel flash_attn_ext_f16
  has no device code compatible with HIP arch 1300.
HW Exception by GPU node-1 (Agent handle: 0x57a994e50680) reason: GPU Hang
```

(Line 1673 = the `NO_DEVICE_CODE;` itself, line 1672 = the guard.)

### What this tells us

- **D ≤ 128 paths**: 2,674 OKs is consistent with most RDNA3 + D ≤ 128 configurations working. We didn't isolate by head-dim, but the sweep iterates `hsk ∈ {40, 64, 72, 80, 96, 128, 192, 256, 320, 512, 576}` in order, and the trap fires inside the D=256 block — so all earlier head-dims passed the cases that reached the backend. **A D ≤ 128 model (Llama 3.x, Qwen 2.5/3 dense, gpt-oss 20B, Phi) would likely run on this branch.** Not validated end-to-end though.
- **D=256 + decode (`Q->ne[1] == 1`)**: passes, because dispatcher routes to VEC.
- **D=256 + prefill (`Q->ne[1] > 1`)**: traps. That's all of Qwen 3.6's prompt-processing path.
- **D ∈ {320, 512, 576} (Mistral4 MLA, DeepSeek MLA, GLM)**: not reached; same guard would block them too.

## Re-test recipe

When the signal fires:

```bash
git fetch jg                            # JG remote = JohannesGaessler/llama.cpp
git checkout -b experiment/jg-fa-rdna3-N upstream/master
git diff $(git merge-base upstream/master jg/<branch>)..jg/<branch> \
    -- ggml/src/ggml-cuda/fattn-mma-f16.cuh \
       ggml/src/ggml-cuda/mma.cuh \
       ggml/src/ggml-cuda/fattn.cu \
  > /tmp/jg.patch
git apply --3way /tmp/jg.patch
# resolve any conflicts in the RDNA config table
# push to origin, bump server-configs llamacpp_version to the SHA, rebuild
ssh lab.28r.net "docker exec llamacpp /app/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0"
```

Pass criteria for our use case: full FA sweep finishes (all `hsk ∈ {40..576}` × `nb ∈ {1, 3, 32, 75}` cases either OK or "not supported", no traps, no FAILs above NMSE 5e-4).

If pass: bench Qwen 3.6 at the standard depth matrix `{0, 2048, 8192, 16384}` against the 2026-04-27 baseline (rocWMMA OFF, MMQ ON). MMA_F16 should beat TILE meaningfully at depth, otherwise something else is wrong even if correctness holds.

## Outcome

Patch landed as `82736929a` on `experiment/jg-fa-rdna3` — one line, exactly the diff above. Built and tested on gfx1151 in the lab container.

### test-backend-ops sweep — 2844 / 2848 pass (2026-04-29)

The four failures are a single shape: `DKQ=DV=256, nh=4, GQA=4, kv=512, nb=32, mask=1, max_bias=0, prec=f32, type_KV=f16`, both permutes (`[0,1,2,3]` and `[0,2,1,3]`), with sinks ∈ {0, 1}. ERR ranges 0.0035–0.020 vs threshold 5e-4. **No traps, no HIP errors, no NO_DEVICE_CODE** — the kernel runs cleanly, just produces slightly imprecise output on this narrow shape.

Smoking gun for a precision (not logic) bug:
- Same shape with `max_bias=8` (ALiBi): **OK**
- Same shape with `nb=3`: **OK**, `nb=32`: **FAIL**

Pattern fits accumulator-precision loss: when softmax is flat (`max_bias=0`) and many cols are processed (nb=32 → ncols1·ncols2 = 32·4 = 128), small errors in the `__shfl_xor_sync` half2 conversion ([mma.cuh:666-679](../ggml/src/ggml-cuda/mma.cuh#L666-L679)) or VKQ scaling ([fattn-mma-f16.cuh:880-888](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L880-L888)) accumulate to 1–2%. Exactly the runtime caveat called out in the trace.

### Perplexity A/B (2026-04-29) — `llama-perplexity` on wikitext-2 raw, 32 chunks × 512 ctx

Same build, same model; only `-fa` toggled. This isolates the patched FA path against the no-FA reference within one binary, so a real precision regression would show as ppl divergence.

| Run            |    PPL |    ±σ |
| -------------- | -----: | ----: |
| FA=1 (patched) | 5.4742 | 0.143 |
| FA=0 (reference) | 5.4608 | 0.143 |
| **Δ** | **+0.0134** | (~10 % of one σ) |

Statistically indistinguishable. Real Qwen 3.6 text doesn't exercise the failing test envelope (Qwen uses RoPE not ALiBi, but the failures only fire at `max_bias=0` *with* `nb=32` — apparently the model's actual attention distribution stays peaky enough that the precision drift never accumulates).

### Honest llama-bench A/B (2026-05-01) — production KV config

The 2026-04-29 llama-bench in this doc compared patched output against [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 1 (`767/209/76/43`, q8_0/q4_0 KV) — the pre-Finding-#1 config that's no longer production. At actual production state ([models.ini](https://github.com/justinappler/server-configs/blob/main/services/llamacpp/files/models.ini) qwen3.6 → `cache-type-k=f16`, `cache-type-v=f16`), a controlled A/B on the same host with the same TheRock nightly tells a different story.

Bench config: [server-configs canonical bench](https://github.com/justinappler/server-configs/blob/main/services/llamacpp/profiling/README.md#canonical-bench-qwen-36-35b-a3b-on-gfx1151) — Qwen 3.6 35B-A3B Q4_K_XL, f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3 -d 0,2048,8192,16384`.

| test           | A: `5d34ca3b` (pre-JG) | B: `82736929a` (JG cherry-pick + widen) |          Δ |
| -------------- | ---------------------: | --------------------------------------: | ---------: |
| pp512 @ d=0    |        1352.63 ± 7.83  |                         1330.51 ± 14.39 |     −1.6 % |
| pp512 @ d=2k   |        1237.42 ± 4.78  |                          1183.38 ± 5.70 |     −4.4 % |
| pp512 @ d=8k   |       1033.59 ± 11.68  |                           942.91 ± 9.40 | **−8.8 %** |
| pp512 @ d=16k  |         851.37 ± 8.82  |                           660.13 ± 6.49 | **−22.5 %** |
| tg128 @ d=0    |          49.06 ± 0.31  |                            49.26 ± 0.07 |       flat |
| tg128 @ d=2k   |          48.64 ± 0.16  |                            48.76 ± 0.19 |       flat |
| tg128 @ d=8k   |          47.60 ± 0.21  |                            47.56 ± 0.18 |       flat |
| tg128 @ d=16k  |          45.91 ± 0.17  |                            45.84 ± 0.16 |       flat |

Clear pp regression scaling with depth (i.e. with KV cache size). tg128 flat across the matrix — the patched MMA_F16 path doesn't help token generation on this shape.

The "+2.3× tg128" claim from 2026-04-29 was an artifact of comparing patched-with-f16-KV against Run-1-baseline-with-q4_0-V (whose tg-at-depth is dominated by V-quant cost — see [Finding #1](kv-cache.md), the actual driver of that regression). Likewise the "+15.2× pp" claim — same baseline-mismatch trick. Both numbers were real arithmetic on the wrong comparison.

### Verdict

**Held, not promoted to `master`.** Branch `experiment/jg-fa-rdna3` retained at `82736929a`; production-equivalent build pins (`lab/vars.yaml`) should sit on a pre-cherry-pick SHA (e.g. `5d34ca3bd`) until further notice.

The patch fails its own decision rule (any pp regression at depth → revert) when measured against the right baseline. We're holding it instead of reverting because:

1. **Real win on q8_0/q4_0 KV configs** (660 vs 43 ≈ 15× at d=16k for that config). Useful base if a future production swap to quantized KV is forced by memory pressure (longer context, larger batches, multiple slots).
2. **Re-bench candidate** when JG iterates (`cuda-fa-rdna3-5` and beyond) or turns the branch into a real PR. Kernel-level enablement is correct; what's missing is the f16-KV-at-depth performance work JG hasn't done yet for D=256.
3. **The 4 test-backend-ops precision failures** are still worth reporting upstream regardless of whether we run the patch — that signal helps JG's own iteration, not just us.

### What changes for the rest of the fork

- **Finding #2** ([fa-dispatcher.md](fa-dispatcher.md)): stays as documented (abandoned 1-line dispatcher patch, real fix is structural). The 2026-04-29 claim that it was "superseded" by this cherry-pick is reversed — the cherry-pick doesn't deliver a production win on f16/f16, so Finding #2's dead-end status holds until JG iterates.
- **Finding #6** ([rocwmma-tuned.md](rocwmma-tuned.md)): stays active. rocWMMA path remains flag-OFF; TILE FA at D=256 is still the production state for f16/f16 KV. The 2026-04-29 "obsoleted by Finding #7" annotation is reversed.
- **Re-bench checklist on JG iteration**: when JG pushes `-5` (or a real PR), repeat the canonical A/B at f16/f16 KV — not just test-backend-ops. The kernel-level enablement is solid; the open question is whether subsequent iterations close the f16-KV-at-depth pp gap or whether the structural choice (f32 accumulator + shfl-pack vs upstream's packed-half2) inherently leaves performance on the table at this shape.

### Lesson — baseline cherry-picking

The 2026-04-29 keep decision presented "+15.2× pp512 @ d=16k" by lining up patched f16-KV bench output against Run 1's q8/q4-KV numbers, which collapse for orthogonal reasons (Finding #1, V-quant-dominated). That's valid arithmetic but the wrong framing — A/B comparisons must be against the current production state, not whichever baseline column makes the headline biggest.

Structural fix: canonical bench config is now pinned in [server-configs profiling/README.md](https://github.com/justinappler/server-configs/blob/main/services/llamacpp/profiling/README.md#canonical-bench-qwen-36-35b-a3b-on-gfx1151) with KV flags explicit (`-ctk f16 -ctv f16`) and a flag-rationale table. Cultural fix: any A/B that doesn't show its full bench command should be treated as suspect on this fork.

### Mechanism (2026-05-03) — register pressure + scratch spills

Static binary inspection of the deployed `82736929a` build (`llvm-objdump` on `libggml-hip.so` inside the lab container) pinpoints why the patched MMA path loses to TILE at depth on this hardware. Per-kernel resource counts at D=256 in the same binary:

| kernel | VGPRs/wave | private_seg/thread | occupancy ceiling (RDNA3 wave32) |
| --- | ---: | ---: | ---: |
| `flash_attn_tile<256, 256, *, *, *>` (32 configs) | 102–254 (most ≤200) | **0** | up to 8 waves/SIMD |
| `flash_attn_ext_f16<256, 256, *, *, *, 0>` MMA (16 configs) | **256 (all)** | **1,620–2,568 bytes** | 6 waves/SIMD |

Why the MMA path lands here on RDNA3 specifically: the [AMD\_WMMA + RDNA3 `mma_tile_sizes`](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1026-L1034) carries `tile<16, 16, float, DATA_LAYOUT_I_MAJOR>` for both `T_C_KQ` *and* `T_C_VKQ`. Every other backend ([:1009-1042](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1009-L1042)) keeps `T_C_VKQ` as `tile<16, 8, half2>` — half the per-tile footprint. The f32 VKQ accumulator is JG's workaround for RDNA3's unpacked-`halfx16_t` write-back (the bug that killed [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063)); the cost is exactly what this measurement shows — accumulator state in registers ~doubled vs RDNA4/CDNA, multiplied across however many tiles the loop body instantiates.

Mapping to the bench shape:
- **Lower occupancy → less latency-hiding.** 256 VGPRs caps at 6 waves/SIMD; TILE at ~150 hits 8+. The K-loop has to wait through more memory latency uncovered, and the gap widens with K size (depth).
- **Spills land in scratch (VRAM-backed).** At d=16k with f16/f16 KV (~512 MiB cache footprint) the spill traffic competes for the same L2 lines KV loads need. d=0 has tiny KV footprint, no contention; d=16k saturates. That's the −1.6% → −22.5% scaling.

How to repro the static inspection (~5 min, no rebuild):

```bash
ssh lab "docker exec llamacpp sh -c '/opt/rocm/lib/llvm/bin/llvm-objcopy --dump-section .hip_fatbin=/tmp/fb.bin /app/libggml-hip.so'"
# .hip_fatbin holds 128 concatenated CLANG_OFFLOAD_BUNDLE__ chunks (one per .cu TU).
# Slice the bundle containing the symbol of interest, unbundle the gfx1151 ELF,
# and read .num_vgpr / .private_seg_size / .num_sgpr from the symbol table.
```

### Followup experiment — halve `nbatch_K2` / `nbatch_V2` for RDNA D=256 (2026-05-04)

Hypothesis: the four [`ggml_cuda_fattn_mma_get_config_rdna` D=256 entries](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L151-L154) carry `nbatch_K2 = nbatch_V2 = 128` from JG's forward-looking table. Halving to 64 should reduce K/V tile residency paired with the f32 VKQ accumulator, predicted to drop VGPR below the 192-wave/SIMD threshold for 8 waves and shrink scratch spills toward zero.

Landed as `79bfad7f3` on `experiment/jg-fa-rdna3-tune` (branched from `82736929a`, four-line edit + comment). Risk surface checked pre-build: `nbatch_K2 == DKQ/2` assert at [:1199](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1199) is `if constexpr (nstages > 1)`-gated and dead for our `nstages=1` configs; macro asserts (`% 4 == 0`, `≤ 256/512`) all pass at 64.

#### Static measurement (post-build)

| metric | `82736929a` (held) | `79bfad7f3` (tune) | Δ |
| --- | ---: | ---: | ---: |
| VGPRs/wave (all 16 D=256 MMA configs) | **256** | **256** | **unchanged** |
| `private_seg_size` per thread | 1,620–2,568 B | 1,188–2,184 B | −15 to −25 % |
| Occupancy ceiling | 6 waves/SIMD | 6 waves/SIMD | unchanged |

The K/V halving moved spills but not VGPRs. **This is the load-bearing finding for the whole investigation:** register pressure at D=256 is set by the f32 accumulator tile residency ([:1026-1034](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1026-L1034) — `tile<16, 16, float, DATA_LAYOUT_I_MAJOR>` for both `T_C_KQ` and `T_C_VKQ`), not by K/V loop state. The 256-VGPR ceiling tracks the accumulator footprint regardless of what we do with K2/V2.

#### Bench (Qwen 3.6 35B-A3B Q4_K_XL, f16/f16 KV, canonical config)

| test | A: `5d34ca3b` (pre-JG prod) | B: `82736929a` (held) | C: `79bfad7f3` (tune) | Δ vs B | Δ vs A |
| --- | ---: | ---: | ---: | ---: | ---: |
| pp512 @ d=0 | 1352.63 ± 7.83 | 1330.51 ± 14.39 | 1363.16 ± 7.55 | +2.5 % | +0.8 % |
| pp512 @ d=2k | 1237.42 ± 4.78 | 1183.38 ± 5.70 | 1173.88 ± 6.71 | −0.8 % | −5.1 % |
| pp512 @ d=8k | 1033.59 ± 11.68 | 942.91 ± 9.40 | 943.00 ± 7.28 | flat | −8.8 % |
| pp512 @ d=16k | 851.37 ± 8.82 | 660.13 ± 6.49 | 643.25 ± 2.29 | **−2.6 %** | **−24.4 %** |
| tg128 @ d=0 | 49.06 ± 0.31 | 49.26 ± 0.07 | 48.01 ± 0.19 | −2.5 % | −2.1 % |
| tg128 @ d=2k | 48.64 ± 0.16 | 48.76 ± 0.19 | 47.67 ± 0.15 | −2.2 % | −2.0 % |
| tg128 @ d=8k | 47.60 ± 0.21 | 47.56 ± 0.18 | 46.45 ± 0.16 | −2.3 % | −2.4 % |
| tg128 @ d=16k | 45.91 ± 0.17 | 45.84 ± 0.16 | 44.86 ± 0.16 | −2.1 % | −2.3 % |

Fails the decision rule ("revert on any pp regression vs `82736929a`") on pp512@d=16k (−2.6 %, outside ~1 % noise floor). tg128 dropped ~2 % across all depths — borderline noise but consistent and unidirectional.

The pp@d=0 +2.5 % is the only positive signal: at zero depth there's no KV pressure for spills to compete with, and the smaller K/V tile is just a tighter loop. Doesn't carry through to where the regression lives.

The tg regression is unexplained — tg routes to VEC, not MMA, so the patch shouldn't touch it directly. Most plausible explanations: (1) thermal/scheduling noise at the host's ~2 % floor, (2) some indirect effect via prefill warmup state. Not investigated further given the pp result already disqualifies the patch.

#### Verdict

Branch `experiment/jg-fa-rdna3-tune` retained at `79bfad7f3` as a documented dead-end (cheap, future-grep-able). `vars.yaml` stays on whatever the prior production pin is until the next experiment is cued up.

#### What this teaches

The "weird packing shape" — JG's f32 T_C_VKQ accumulator workaround for RDNA3's unpacked-half2 WMMA write-back — is **structurally** the binding constraint on this hardware at D=256, not a tuning miss. Direct evidence: VGPRs didn't budge when we eliminated the K/V tile contribution, which says register pressure is dominated by accumulator state. The 6-waves/SIMD occupancy ceiling that follows from 256 VGPRs is a downstream consequence of choosing `tile<16, 16, float>` for `T_C_VKQ` instead of RDNA4/CDNA's `tile<16, 8, half2>`. Halving the per-tile lane footprint would require replacing the f32 accumulator with a packed-half2 path — most likely via `wmma_f16_16x16x16_f16_w32` with `OPSEL` to write high/low halves separately and reassemble, which is exactly the work-in-progress in JG's branch sequence.

Implication: tile-size knobs in the existing config table can't move the perf ceiling at D=256 on RDNA3. The fork can stop sweeping `nbatch_*` for this shape. Lower-priority alternatives that *could* move VGPRs (still speculative, not currently planned):
- **`nthreads` 64 → 128** in the small-ncols D=256 entries (CDNA's same shape uses 128). Halves per-thread accumulator state by spreading work across more lanes; introduces 4-warp coordination overhead.
- **Reducing instantiated tile count in the loop body** (less ILP, less concurrent accumulator residency).

The high-leverage path is upstream: when JG pushes `cuda-fa-rdna3-5+` or files a PR, watch specifically for changes to the `mma_tile_sizes` block at [:1026-1034](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1026-L1034). If `T_C_VKQ` changes type, re-bench. If it stays `tile<16, 16, float>`, the perf gap to TILE at f16/f16-KV-D=256 is structural and won't close without a different approach to the unpacked-half2 problem.

## Cross-references

- [fa-dispatcher.md](fa-dispatcher.md) — Finding #2; the original dispatcher dead-end and roadmap framing.
- [rocwmma-tuned.md](rocwmma-tuned.md) — Finding #6; rocWMMA path that this branch was expected to obsolete but doesn't, at f16/f16 KV. Production state for D=256 stays on TILE + flag-OFF.
- [PR #16827 thread](https://github.com/ggml-org/llama.cpp/pull/16827#issuecomment-3454830174) — JG's roadmap comment from 2025-10-29.
- [PR #22051](https://github.com/ggml-org/llama.cpp/pull/22051), [PR #22298](https://github.com/ggml-org/llama.cpp/pull/22298) — early steps in JG's chain that already merged.
