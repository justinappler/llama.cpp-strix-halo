# Flash-attention dispatcher gates RDNA 3.5 out of the MMA kernel

**Status: attempted, abandoned. Upstream attempt ([PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063)) also abandoned for a deeper reason.** The naive port (widen guards, add RDNA3 `make_identity_mat`, add 4-arg RDNA3 WMMA intrinsic) compiles and runs — but produces incorrect output because RDNA3's WMMA f16→f16 accumulator is unpacked (1 half per 32-bit lane) while RDNA4's is packed (2 halves per lane) and the surrounding tile/FA math assumes packed. An actual fix requires either a repack step, a tile-layout redesign, or an f32 accumulator — none of which is a small change. Upstream is additionally refactoring this code right now ([PR #22051](https://github.com/ggml-org/llama.cpp/pull/22051)). See [Upstreamability](#upstreamability).

## Symptom

On `gfx1151` with default build flags (`GGML_HIP_ROCWMMA_FATTN=OFF`), flash-attention runs on the generic **TILE** fallback kernel. TILE has no efficient quantized-KV path, which makes its throughput collapse at depth (see [kv-cache.md](kv-cache.md)).

The root cause appeared to be dispatcher logic in [fattn.cu](../ggml/src/ggml-cuda/fattn.cu), not a missing kernel. That turned out to be incomplete — the dispatcher gate was defensive, guarding against a real problem (no compiled device code) rather than an arbitrary restriction.

## Dispatch trace for gfx1151

Reading [ggml/src/ggml-cuda/fattn.cu:307-505](../ggml/src/ggml-cuda/fattn.cu#L307-L505) for RDNA 3.5:

1. `turing_mma_available(cc)` — NVIDIA Turing+, skipped.
2. `volta_mma_available(cc)` — NVIDIA Volta, skipped.
3. `ggml_cuda_should_use_wmma_fattn(cc)` — skipped because the build has `GGML_HIP_ROCWMMA_FATTN=OFF`. See [fattn-wmma-f16.cuh:26-49](../ggml/src/ggml-cuda/fattn-wmma-f16.cuh#L26-L49).
4. `amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc)` — **false** for gfx1151; RDNA4-only gate. See [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454).
5. `amd_mfma_available(cc)` — CDNA only, skipped.
6. **Fallthrough: TILE kernel** at [fattn.cu:505](../ggml/src/ggml-cuda/fattn.cu#L505).

## The MMA_F16 kernel's host dispatcher says it supports RDNA 3

The kernel's own config dispatcher at [fattn-mma-f16.cuh:171-186](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L171-L186):

```cpp
if (amd_wmma_available(cc)) {
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
}
```

and `amd_wmma_available()` in [common.cuh:318-320](../ggml/src/ggml-cuda/common.cuh#L318-L320) returns true for both RDNA 3 and RDNA 4:

```cpp
static bool amd_wmma_available(const int cc) {
    return (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
}
```

So the RDNA path **in the host-side code** was there. That turned out to be misleading.

## What we tried

Two incremental changes, both on a dedicated branch at the time:

1. **Dispatcher gate relaxation** — widen the condition at [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454) to include RDNA 3.5:

   ```diff
   -if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && ...) {
   +if (amd_wmma_available(cc) && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) && ...) {
   ```

   Bench result: identical to baseline TILE (within noise). A dispatcher trace revealed why — all attention calls have `Q->ne[0] = 256` (head dim 256) on Qwen 3.6 and Qwen3-Coder-Next, and the block at line 454 has an additional `Q->ne[0] <= 128` guard. The patch put RDNA 3.5 inside the gate, but the inner head-dim check still excluded us. **Patch never fired.**

2. **Head-dim relaxation** — widen the inner guard from `<= 128` to `<= 256`, matching the MMA config table's explicit `(256, 256, ...)` RDNA entries in [fattn-mma-f16.cuh](../ggml/src/ggml-cuda/fattn-mma-f16.cuh).

   Bench result: **GPU hang with no device code**:

   ```
   ERROR: HIP kernel flash_attn_ext_f16 has no device code compatible with HIP arch 1300
   HW Exception by GPU node-1 ... reason: GPU Hang
   ```

## Outcome

The MMA_F16 kernel's *host-side* dispatcher advertises RDNA support. The *device-side* template-instance `.cu` files at [ggml/src/ggml-cuda/template-instances/](../ggml/src/ggml-cuda/template-instances/) **are** compiled for `gfx1151` — the HIP cmake at [ggml/src/ggml-hip/CMakeLists.txt:66-67](../ggml/src/ggml-hip/CMakeLists.txt#L66-L67) globs every `fattn-mma*.cu` unconditionally and applies `CMAKE_HIP_ARCHITECTURES` (set from `GPU_TARGETS=gfx1151`) to all of them. There is no per-arch enumeration to extend.

What's missing isn't the compiled symbol — it's the kernel *body*. Six sites in [fattn-mma-f16.cuh](../ggml/src/ggml-cuda/fattn-mma-f16.cuh) (lines 495, 965, 1030, 1544, 1571, 1723) wrap the kernel in:

```cpp
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) \
    || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)) \
    || defined(AMD_MFMA_AVAILABLE)
    // ... kernel ...
#else
    NO_DEVICE_CODE;
#endif
```

On `gfx1151`, `AMD_WMMA_AVAILABLE` is already defined (via `__GFX11__` → `RDNA3` at [hip.h:216-222](../ggml/src/ggml-cuda/vendors/hip.h#L216-L222) and [common.cuh:243-245](../ggml/src/ggml-cuda/common.cuh#L243-L245)). But `RDNA4` is not, so the inner body compiles to `NO_DEVICE_CODE` — a `__trap()` with a runtime log. That's the "no device code compatible with HIP arch 1300" message: not a missing symbol, but a symbol whose body is the trap. The function launches, hits `__trap`, GPU hangs.

The RDNA4-only gate at [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454) was defending against this reality, not expressing a perf choice.

## What an actual fix would look like

Not a cmake change. The `.cu` files are already compiled for gfx1151 — the fix is device-side: widen the preprocessor guards so RDNA 3 / 3.5 takes a real kernel path, not the trap stub. This is a non-trivial port because the existing RDNA4 WMMA path relies on helpers in [mma.cuh](../ggml/src/ggml-cuda/mma.cuh) that are currently RDNA4-only or trap on AMD_WMMA.

What's needed, roughly:

1. **Widen the six `(AMD_WMMA_AVAILABLE && RDNA4)` guards** in [fattn-mma-f16.cuh](../ggml/src/ggml-cuda/fattn-mma-f16.cuh) at lines 495, 965, 1030, 1544, 1571, 1723 to include RDNA 3 (either `(AMD_WMMA_AVAILABLE && (RDNA3 || RDNA4))` or just `AMD_WMMA_AVAILABLE`, depending on whether any other RDNA4-only assumptions remain).

2. **Implement `make_identity_mat` for RDNA3** at [mma.cuh:704-715](../ggml/src/ggml-cuda/mma.cuh#L704-L715). Currently RDNA4-only; `NO_DEVICE_CODE` on RDNA3. Called unconditionally on the AMD_WMMA fattn path at [fattn-mma-f16.cuh:867](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L867) whenever `!LDMATRIX_TRANS_AVAILABLE` (true for all AMD).

3. **Audit `get_transposed(tile<16,4,half2>)`** at [mma.cuh:682-685](../ggml/src/ggml-cuda/mma.cuh#L682-L685), which is `NO_DEVICE_CODE` for all `AMD_WMMA_AVAILABLE`. Called at [fattn-mma-f16.cuh:840](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L840) and [:1448](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L1448). Either these call sites are unreachable under the RDNA config (`cols_per_warp` routing should rule out the `cols_per_warp == 8` branch for head-dim 256), or they also need an RDNA implementation. Verify before widening guards.

4. **Widen the dispatcher** at [fattn.cu:454](../ggml/src/ggml-cuda/fattn.cu#L454) to include `GGML_CUDA_CC_IS_RDNA3_5(cc)` and relax the inner `Q->ne[0] <= 128` to `<= 256`, matching the RDNA entries in the MMA config table at [fattn-mma-f16.cuh:116-131](../ggml/src/ggml-cuda/fattn-mma-f16.cuh#L116-L131). This is the step that hung last time — safe once (1)–(3) are in place because real device code now exists.

5. **Rebuild and bench** on TheRock nightly. Only after this is the MMA_F16 vs TILE question actually answerable on RDNA 3.5.

The commit that added the RDNA4 path ([PR #18481](https://github.com/ggml-org/llama.cpp/pull/18481), commit `ea4a321f2`) has a squash-commit note **`disable fattn-mma-f16 on rdna3`** — a deliberate exclusion. Reading the PR review resolves why:

- The author (zhang-hui-yulo) explicitly scoped RDNA3 out to a follow-up PR: *"RDNA3 support as I might find a good way to handle RDNA3 fused gemm without smem, of course this needs gmem to smem transportation."* Not correctness, not perf — scope. Quote: *"I would suggest to keep this PR simple and make more changes in the future."*
- The specific RDNA3 hurdle they named is register-level transposition. RDNA4 has a weak global-memory transpose load (*"RDNA4 global transpose loading doesn't help much"*); RDNA3 has none. PR #18481 works around this on RDNA4 by using an identity matrix and MMA to transpose in registers — that's exactly why `make_identity_mat` and `get_transposed` landed as RDNA4-only in `mma.cuh`. The author intended to solve RDNA3 differently (*"fused gemm without smem"*), not by porting the same path.
- The author never followed up. After PR #18481 merged (2026-01-13) they pivoted to CDNA work (#18896 mmf CDNA, #20123 fattn CDNA). No RDNA3 FA PR exists from them or anyone else as of this writing.
- Adjacent confirming signal: PR #17879 enabled MMF for RDNA3 in 2025-12, so `mma.cuh`'s RDNA3 WMMA intrinsic paths (`__builtin_amdgcn_wmma_*_w32` without the `_gfx12` suffix) are proven functional for matmul. FA reuses these same primitives; the blocker is specifically the two RDNA4-only helpers above, not the underlying matrix ops.

Practical read (revised): someone already tried the naive port and found it produces wrong output.

**[PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063) (linus-amg, opened 2026-01-24, closed by author)** attempted exactly the 17-line port we sketched above — widen the six `#if` guards, add an `RDNA3` branch to `make_identity_mat`, add the 4-arg `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` intrinsic, widen the dispatcher. Scoped narrowly to head-size 576 (GLM-4.7 MLA on gfx1100). Bench showed it was functional enough to improve tg (~27 → ~83 t/s). Then the author closed it with:

> *"Closing this PR — the RDNA3 f16→f16 WMMA implementation produces incorrect output due to unpacked output format incompatibility with the tile structure. RDNA3 works correctly with tile-based flash attention instead of MMA. May revisit with a proper fix in the future."*

The root cause is a concrete RDNA3↔RDNA4 register-layout difference. RDNA4's `wmma_f16_16x16x16_f16_w32_gfx12` writes a packed `halfx8_t` (two halves per 32-bit lane). RDNA3's `wmma_f16_16x16x16_f16_w32` (no `_gfx12`) writes an unpacked `halfx16_t` — each half occupies its own 32-bit slot, with the 4th `OPSEL` arg controlling hi/lo. [mma.cuh](../ggml/src/ggml-cuda/mma.cuh) tiles and all downstream FA math assume the packed layout. Swapping the intrinsic without repacking silently produces garbage in the accumulator.

Fixing this isn't a preprocessor change — it's a tile-layout problem:

- Option A: a compact/repack step after every WMMA to convert unpacked → packed. Cheap in ops, expensive in register pressure, and adds a per-mma latency.
- Option B: rework `tile<>` layouts in [mma.cuh](../ggml/src/ggml-cuda/mma.cuh) to carry unpacked form through, with RDNA3-specific access patterns everywhere tiles are read. Invasive; would need maintainer buy-in before attempting.
- Option C: use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (f32 accumulator — RDNA3 has this natively in packed-ish form) and convert to f16 at the end. Different perf profile; may defeat the point of using MMA on a bandwidth-limited iGPU.

None of these is a weekend change. Also relevant: **[PR #22051](https://github.com/ggml-org/llama.cpp/pull/22051) (JohannesGaessler, opened 2026-04-18 — today)** is a 112+/395− refactor of AMD mma data loading, already approved by am17an, in review by IMbackK. It reshapes the exact `load_ldmatrix` / transpose code we'd be patching. Any port attempt should wait for this to land.

Adjacent unknowns worth flagging:

- The `Q->ne[1] * gqa_ratio_eff <= 8` carveout at [fattn.cu:473-474](../ggml/src/ggml-cuda/fattn.cu#L473-L474) routes short-batch cases back to TILE. For tg (single token), this always fires on Qwen 3.x regardless of head dim, so tg would stay on TILE even after the MMA path is unblocked.
- Quantized-KV viability under MMA (the original motivation from [kv-cache.md](kv-cache.md)) is still untested. The hypothesis that MMA has a dequant-free path is unverified.

## Upstreamability

Downgraded from earlier read. A port is not a simple gate-widening:

1. **A 17-line version was already tried and closed for producing wrong output.** [PR #19063](https://github.com/ggml-org/llama.cpp/pull/19063) made the exact changes this doc's "What an actual fix would look like" section described. The author found the RDNA3 unpacked WMMA output doesn't match the tile layout and closed the PR. Any future attempt has to address that register-layout mismatch, not just the guards.
2. **Upstream is actively refactoring this code.** [PR #22051](https://github.com/ggml-org/llama.cpp/pull/22051) (JohannesGaessler, open as of 2026-04-18) restructures AMD mma data loading in [mma.cuh](../ggml/src/ggml-cuda/mma.cuh) and simplifies usage in `fattn-mma-f16.cuh`. Starting a port against current master means rebasing onto a moving target. Wait for #22051 to merge.
3. **Original author's view on RDNA3 (from PR #18481 review): not the same approach.** zhang-hui-yulo's TODO was *"find a good way to handle RDNA3 fused gemm without smem"* — they already anticipated that the RDNA4 path wouldn't port directly. #19063's failure confirms that intuition.

If we want to pursue this despite the above, the minimum-viable path is: wait for #22051, then attempt Option C (f32 accumulator + convert) as a separate branch and bench against TILE honestly. That's a multi-week investigation, not a patch. For the strix-halo fork's near-term purposes, this is the wrong place to spend effort — TILE with f16 KV (Finding #1) and the MMQ tuning (Finding #5) are already delivering real wins; the dispatcher is a dead end until someone lands the RDNA3 MMA port upstream.
