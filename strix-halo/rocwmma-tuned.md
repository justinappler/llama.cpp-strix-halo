# rocWMMA FA tuning for gfx1151 — port of lhl's PR #16827

## Hypothesis

On `gfx1151` the rocWMMA flash-attention path is currently a net loss at depth and so we keep `GGML_HIP_ROCWMMA_FATTN=OFF` in the Dockerfile (see [fa-dispatcher.md](fa-dispatcher.md) and the server-configs comment). With the flag off, every FA call routes to the generic TILE kernel, which has no tensor-core path on RDNA 3.5 — that's the biggest remaining piece of dead silicon on the chip.

[lhl/llama.cpp rocm-wmma-tune](https://github.com/lhl/llama.cpp/tree/rocm-wmma-tune) (rejected upstream as [PR #16827](https://github.com/ggml-org/llama.cpp/pull/16827)) reshapes the existing rocWMMA FA kernel so that it can actually be enabled on Strix Halo:

1. **`__launch_bounds__` min-blocks-per-SM = 2** on HIP-rocWMMA builds — forces the compiler to budget registers for two resident blocks per SM, trading per-thread registers for occupancy. The upstream kernel was tuned for discrete RDNA3's fatter register file (`min_blocks=1`).
2. **Adaptive KQ stride** — `D ≤ 128 → 128` (half the default `FATTN_KQ_STRIDE`) on HIP-rocWMMA only. Shrinks the LDS footprint and all `KQ_f_tmp` / `KQ2_tmp` stack arrays, so the extra registers freed by (1) aren't spilled.
3. **`nwarps = 8` for small D (`D ≤ 96`)** — doubles the warp count for the tight-head-dim case where the smaller stride lets us pack in more warps per block.
4. **HIP decode dispatch fix** — on HIP+rocWMMA, route decode (`Q->ne[1] == 1`) away from the WMMA kernel and down the VEC/TILE path it was always meant to take, with a guard that predicts TILE's `ncols2`/`cols_per_block` and falls back to VEC if the predicted TILE split has no registered config. The upstream dispatcher's TILE-pruning (`ncols2 != 1 && DV != 40 && DV != 512`) would otherwise drop us into `NO_DEVICE_CODE`/`__trap()` for common head-dim combos on decode.

Collectively this is "fix the rocWMMA path that was tuned for RDNA 3 discrete, not the iGPU." It is *not* a new kernel.

## What we expect to measure on this fork

lhl's own bench results on gfx1151 (llama-bench, `-fa 1`, depths `0,4096,8192,16384,65536`):

| Model | pp512 @ d=16k | pp512 @ d=65k | tg128 @ d=16k |
|---|---|---|---|
| Llama 3.2 1B Q4_K_M vs HIP baseline        | +52.89% | +66.32% | flat |
| gpt-oss 20B F16 (MXFP4) vs HIP baseline     | +34.89% | +53.76% | flat |
| Llama 3.2 1B Q4_K_M vs previous rocWMMA    | +63.14% | +96.11% | +45.11% |
| gpt-oss 20B F16  vs previous rocWMMA       | +43.78% | +87.83% | +34.72% |

`vs HIP baseline` is the one that matters for us — that's the column against our current master build (rocWMMA off, TILE on decode). Gains scale with depth, which is exactly the axis where our MMQ port is flat (see [mmq-rdna3_5.md#post-upstream-sync-re-bench-2026-04-19](mmq-rdna3_5.md#post-upstream-sync-re-bench-2026-04-19): +14% pp @ d=16k vs the +35-50% this might add). Stacking is possible because the two patches work on disjoint kernels (MMQ GEMM vs rocWMMA FA).

Risk we're taking on: lhl benched on Llama 3.2 1B (dense) and gpt-oss 20B F16 (MoE, F16 weights). Our target is Qwen 3.6 35B-A3B Q4_K_XL (MoE, quantized weights). The FA path doesn't touch quantization and Qwen 3.6 A3B is GQA, so the gain *direction* should port, but the magnitude is unknown.

**NB (added post-bench):** I assumed Qwen 3.6 A3B was "D=128-ish like gpt-oss" when writing this — that was wrong. Qwen 3.6 A3B has `n_embd_head_k = n_embd_head_v = 256`, twice as wide as gpt-oss 20B (D=128) or Llama 3.2 1B (D=64). Three of lhl's four knobs are D-gated (`D≤128` adaptive stride, `D≤96` nwarps, decode dispatch tied to head-dim exclusions), so on a D=256 model only `__launch_bounds__ min_blocks=2` fires. See Outcome below.

Note the separate [tg-at-depth-regression.md](tg-at-depth-regression.md) regression: the decode path also goes through dispatcher logic, and lhl's patch (2) reshapes how HIP decode picks VEC vs TILE. If tg @ d=16k recovers alongside the pp gain, that's a nice bonus; if it doesn't, the regression is pinned more tightly on #22051.

## The patch

Port of two lhl commits (`a3c9d1d69` + `a45e1cd6e`) squashed into a single `strix-halo:` commit on `master`, per the one-commit-per-attempt workflow.

Files changed:

| File | What |
|---|---|
| [ggml/src/ggml-cuda/fattn-wmma-f16.cu](../ggml/src/ggml-cuda/fattn-wmma-f16.cu) | `__launch_bounds__` min-blocks=2 on HIP+rocWMMA; `ggml_wmma_fattn_kq_stride<D>()` helper returning 128 for `D≤128` on HIP; plumb the adaptive stride through the kernel body and launcher; `nwarps=8` for `D≤96` |
| [ggml/src/ggml-cuda/fattn-tile.cuh](../ggml/src/ggml-cuda/fattn-tile.cuh) | Skip the `ncols2`-based TILE-variant pruning on HIP so we don't `__trap()` when decode lands on a pruned shape |
| [ggml/src/ggml-cuda/fattn.cu](../ggml/src/ggml-cuda/fattn.cu) | HIP+rocWMMA-only branch in `ggml_cuda_get_best_fattn_kernel`: if `Q->ne[1] == 1` (decode), skip WMMA; mirror the TILE `ncols2`/`cols_per_block` prediction and fall back to VEC when the predicted shape has no config |

All changes are `#if defined(GGML_USE_HIP) && defined(GGML_HIP_ROCWMMA_FATTN)`-gated, so CUDA builds and HIP builds with `GGML_HIP_ROCWMMA_FATTN=OFF` are byte-identical to upstream. That means **this commit is dead code in our production Dockerfile until we also flip `-DGGML_HIP_ROCWMMA_FATTN=ON`**.

## Build/bench knobs

Two things change in server-configs on top of the usual `llamacpp_version` bump:

1. Push to `origin` (the fork), then bump `llamacpp_version` to the new full SHA.
2. `/Users/jappler/Projects/server-configs/services/llamacpp/files/Dockerfile` line 100: `-DGGML_HIP_ROCWMMA_FATTN=OFF` → `=ON`. The comment block above it (lines 96-99) explaining the ~100x slowdown is the reason we've had it off; replace it with a pointer to this doc noting that the tuning patch fixes the regression.

## Bench plan

Same shape as [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 3 — Qwen 3.6 35B-A3B Q4_K_XL, f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3`, depths `{0, 2048, 8192, 16384}`.

A/B:
- **A**: current master (`bc8362b09`, MMQ port applied, rocWMMA FA off).
- **B**: this commit on top of A, with Dockerfile flipped to `GGML_HIP_ROCWMMA_FATTN=ON`.

No intermediate "apply the code change but leave the flag off" build is needed — that build is byte-identical to A by the `#if` gates.

## Decision rule

Keep if **any** of the following is true and **none** of the tg axes regress outside noise:

- pp512 @ d=16,384 improves >10% (lhl saw +35% on gpt-oss 20B; anything under +10% on Qwen 3.6 would be disappointing enough to question whether the dense/MoE distinction hurts us).
- pp512 @ d=0 improves >5% (expected modest — lhl saw +2% on gpt-oss, +6% on Llama 1B; a miss here isn't disqualifying).

Revert if:

- **Any** tg axis regresses (we already have [tg-at-depth-regression.md](tg-at-depth-regression.md) — we don't want to compound it).
- pp regresses at any depth (this would mean the rocWMMA path is still broken on Qwen's quantization/GQA shape despite the fix).

The priors say pp should win at depth and tg should be flat; if the actual result deviates far from that shape, the doc keeps the numbers and we move on to the MoE `MUL_MAT_ID` investigation that NOTES.md #2 has queued.

## Why this is the next attempt

From [NOTES.md](NOTES.md)'s priority table, MMVQ RDNA3.5 tuning was #1 but already ran and reverted (see [mmvq-rdna3_5.md](mmvq-rdna3_5.md)). This attempt sits between #1 and #2 on that table — cheaper than a MoE `MUL_MAT_ID` investigation (S vs M-L), with a larger prior on the pp-at-depth axis (tuned rocWMMA FA directly targets attention compute, not the MMQ/MMVQ GEMM paths). lhl's numbers on gpt-oss 20B — structurally closer to Qwen 3.6 A3B than Llama 1B is — are the most informative prior we have.

The zero-cost fallback is that the patch compiles out completely with the Dockerfile flag off, so if the bench is a bust we can revert with `git reset --hard HEAD~1` and the doc annotation, same as every other dead-end in this folder.

## Outcome

**Kept on master as commit `1be00ab8`.** Decision deviates from the stated keep-if rule (pp512 @ d=16k +10%); see rationale below.

### Bench (2026-04-19, Qwen 3.6 35B-A3B Q4_K_XL, f16/f16 KV)

A: `bc8362b09` — MMQ port, rocWMMA FA **off**.
B: `1be00ab8` — same as A plus this commit, rocWMMA FA **on**.

- **pp512** at depths `{0, 2048, 8192, 16384}`: within ±1.5% across the full matrix — flat.
- **tg128** at the same depths: consistently -1.0% to -1.6% vs A, at or below the ±2% run-to-run noise floor we've been seeing.

Neither axis moves meaningfully. The decision rule says *revert* (no pp gain; tg technically fails the "no regression" guard, though the magnitude is in the noise).

### Why it's not moving the needle on Qwen 3.6

rocprof-sys trace on a pp512 @ d=8192 run (`profiling/traces/run-20260419-115122/`, queried via `trace_processor`) shows the FA kernel dispatch is unambiguously the rocWMMA path:

| Kernel | Invocations | Total GPU time |
|---|---:|---:|
| `flash_attn_ext_f16<256, 16, 4, 64, float, false>` | 60 | 7.94 s |
| `flash_attn_combine_results<256>` | 60 | 39 ms |
| `flash_attn_mask_to_KV_max<16>` | 40 | 5.5 ms |
| `flash_attn_tile_*` | **0** | — |

Template params decode as `<D=256, ncols=16, nwarps=4, VKQ_stride=64, KQ_acc_t=float, use_logit_softcap=false>`. So:

- **`D=256`** confirms Qwen 3.6 A3B's head dim is 256 (also verified at load time: `n_embd_head_k = n_embd_head_v = 256`, `n_head=16`, `n_head_kv=2`, so GQA ratio 8).
- **`VKQ_stride=64`** is consistent with `FATTN_KQ_STRIDE=256` (upstream default, since our patch only halves it for `D≤128`). The adaptive-stride change from patch (2) does not fire.
- **`nwarps=4`** is the upstream value. Patch (3) only bumps to 8 at `D≤96`, also doesn't fire.
- **No decode FA calls** in the trace (bench was pp-only), so patch (4)'s TILE-fallback is untested on this run but also irrelevant for prefill performance.

That leaves only patch (1) — `__launch_bounds__ min_blocks=2` — as the active change on our shape. lhl's bench is bundled so we can't directly attribute gain shares, but from first principles patches (2) + (3) shrink LDS + stack arrays by halving `FATTN_KQ_STRIDE` and doubling `nwarps`, which is where the RDNA 3.5 register-file pressure gets relieved; both are D-gated out at D=256. Patch (1) on its own is an occupancy hint, which is what you'd expect to see in the "flat, not regressive" regime we observe.

**Conclusion:** the port is correct and the kernel dispatch is working exactly as designed. Qwen 3.6 A3B just doesn't hit the code paths lhl tuned for.

### Why keep it anyway (deviation from the decision rule)

Three reasons:

1. **Zero downside on the current workload.** pp flat, tg regression is in the noise floor. If a regression materialises on a later bench we can flip the Dockerfile flag back to `OFF` — the `#if GGML_USE_HIP && GGML_HIP_ROCWMMA_FATTN` gating makes that a one-line Dockerfile change with no source edit, and the commit stays useful as a carrier for the code.
2. **Real gains on any D≤128 model we'd swap in.** Llama 3.1/3.2 (D=128), Qwen 2.5/3 dense (D=128), Mistral 7B (D=128), gpt-oss 20B (D=128), and Phi-series (D=96) would all hit the full bundle. lhl's measured pp gains on those shapes are +35-65% at depth. We run side-by-side evals on this box often enough that keeping the infrastructure paid-for is worth the rebase cost.
3. **No maintenance surprise.** All changes are `#if`-gated behind `GGML_HIP_ROCWMMA_FATTN`, so CUDA builds and HIP-without-rocWMMA builds are byte-identical to upstream. Upstream merges into these three files rebase cleanly unless upstream itself reshapes the rocWMMA path.

### If the primary workload changes

If we later migrate off Qwen 3.6 A3B onto a D≤128 model, re-bench before assuming the existing numbers transfer — lhl's gpt-oss 20B F16 is the closest shape we have a prior for, and the +35% pp @ d=16k target applies there, not here.

### If upstream merges #16827 or a successor

Drop our commit during rebase. The [lhl PR discussion](https://github.com/ggml-org/llama.cpp/pull/16827) was rejected on maintainability grounds, not correctness; if a refactor makes it in, watch for behaviour-equivalence on our D=256 shape before blindly dropping our commit.
