# MMQ RDNA3.5 config table — re-port of Finding #5/#8 onto upstream PR #24127

## Status (2026-07-16 — shipped and benched)

Third incarnation of the gfx1151 MMQ tuning. The previous two forms (the six-edit ternary patch from [mmq-rdna3_5.md](mmq-rdna3_5.md), and the dense/MoE split from [pp-rdna3_5-tile-mmq.md](pp-rdna3_5-tile-mmq.md)) were **dropped, not rebased**, on the 2026-07-16 upstream sync. Upstream [PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127) (JohannesGaessler, merged 2026-07-13, `6eddde06`) deleted every function they edited. This doc covers the re-port against the replacement architecture.

**Shipped and running in production.** Correctness verified on real gfx1151 and the full bench matrix is in [Outcome](#outcome) below: every depth improved, nothing regressed, and ~971 t/s prefill at d=16k is the best number this fork has produced. The port-off A/B was **not** run, so the port's own share of that gain is unmeasured — see the caveat in [Outcome](#outcome) before quoting these numbers as validation of the port.

Pre-rebase history is preserved on `backup/master-pre-rebase-20260716` (dropped commits `0adc0a1b5`, and the `mmq.cuh` half of `9a318552e`).

## What upstream changed

PR #24127 replaced the per-arch macro/ternary helpers with a **table of `ggml_cuda_mmq_config` entries per architecture**, keyed by `(type, J, fallback)`:

- `mmq_x` / `mmq_y` were renamed `J` / `I` (matching the FA kernels).
- The helpers we patched (`get_mmq_x_max_host`/`_device`, `get_mmq_y_host`/`_device`, `mmq_get_nwarps_host`/`_device`) are **gone**. Tile shape now comes from the table; `nwarps` is derived as `nthreads / warp_size`.
- `__launch_bounds__` became **mandatory** — every entry must declare an `occupancy`.
- Tables live in `mmq-config-{pascal,ampere,blackwell,cdna,rdna2,rdna4}.cuh`.

`amd_wmma_available(cc)` still returns true for all of RDNA3 ([common.cuh:340](../ggml/src/ggml-cuda/common.cuh#L340)), so **gfx1151 currently free-rides on the rdna4 table**, which is uniformly `nthreads=256, occupancy=2, I=128` for every type. That is functionally the same tile shape as the pre-patch default this fork has been beating since 2026-04. Upstream added no RDNA3.5 entries anywhere.

The important consequence: the refactor is an **extension point**, not an obstacle. The old patch was hand-rolling a per-arch config table with prepended ternaries; upstream now provides the real thing.

## Semantics mapping

The port preserves the old knobs exactly. Nothing here is a new tuning decision except `occupancy`:

| Old knob (deleted) | Old value | New expression | New value |
|---|---:|---|---:|
| `get_mmq_y_*` (`mmq_y`) | 64 | table `I` | 64 |
| `mmq_get_nwarps_*` | 4 | table `nthreads` (`nwarps = nthreads/32`) | 128 |
| `get_mmq_x_max_*` (`mmq_x_max`), dense | 128 | table `J` (uncapped) | 128 |
| `mmq_x_max`, MoE (`expert_bounds != nullptr`) | 48 | `J_max` cap in `mul_mat_q_switch_J` | 48 |
| *(did not exist — `__launch_bounds__` was optional)* | — | table `occupancy` | **2 (unvalidated)** |

`I` and `nthreads` are **not independent**. The MMA write-back path computes `i0 = (threadIdx.y / ntx) * (ntx * tile_C::I)` ([mmq.cuh:458](../ggml/src/ggml-cuda/mmq.cuh#L458)), and on AMD WMMA `tile_C::I == 16` and `rows_per_warp == 16`, so `ntx == 1` and coverage requires:

```
I == nwarps * 16 == (nthreads / 32) * 16      (wave32)
```

rdna4 satisfies this as `128 == 8 * 16`; our table as `64 == 4 * 16`. The old patch's `(mmq_y=64, nwarps=4)` pair was obeying this same constraint implicitly. **Changing one without the other silently corrupts output** — worth remembering if the table is swept later.

## The port

> [!NOTE]
> **Reshaped on the 2026-08-02 rebase.** Upstream [PR #26199](https://github.com/ggml-org/llama.cpp/pull/26199) (merged 2026-07-28) added `mmq-config-rdna3-5.cuh` and the `GGML_CUDA_CC_IS_RDNA3_5` dispatch itself, so pieces 1 and 2 below are now **upstream's code, not ours**. See [Collision with PR #26199](#collision-with-pr-26199) for what the fork still owns.

Three pieces, +11/-1 in `mmq.cuh` plus one new file:

1. **`mmq-config-rdna3_5.cuh`** (new, 303 lines) — generated mechanically from `mmq-config-rdna4.cuh` with `nthreads: 256 -> 128`, `I: 128 -> 64`. `sram_layout`, `K_vram`, and `stream_k` are carried over unchanged for all 21 types; only the tile shape is retuned.
2. **Dispatch** in [mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh) — host-side `GGML_CUDA_CC_IS_RDNA3_5(cc)` branch and device-side `#elif defined(RDNA3_5)`, both placed **before** the `amd_wmma_available` / `AMD_WMMA_AVAILABLE` branch that would otherwise route us to rdna4.
3. **MoE J cap** in `mul_mat_q_switch_J` — `args.expert_bounds != nullptr` still reaches the J-selection loop, so the old dense/MoE split survives as a one-line bound on the search:

```cpp
const int J_max = GGML_CUDA_CC_IS_RDNA3_5(cc) && args.expert_bounds != nullptr ? 48 : 128;
```

### The J=48 fallback rows

rdna4's `fallback=true` rows only cover `J ∈ {16, 32, 64, 128}` (PR #24127 deliberately thinned the fallback specializations to powers of 2 to cut compile time). With a MoE cap of 48 and no `J=48` fallback entry, the selection loop would silently settle on **32** whenever `ne01 % 128 != 0` — a behaviour change from the pre-rebase build that would confound the re-bench.

Our table therefore adds a `J=48, fallback=true` row per type (21 rows). Verified: MoE resolves to `J=48` for both `fallback` values, dense stays at `J=128`.

## Collision with PR #26199

**2026-08-02.** Upstream [PR #26199](https://github.com/ggml-org/llama.cpp/pull/26199) (Geramy Loveless, merged 2026-07-28, `60bccc376`) added `mmq-config-rdna3-5.cuh` and `mmq-config-rdna3.cuh`, and replaced the `amd_wmma_available` dispatch with explicit `RDNA4 / RDNA3_5 / RDNA3` branches on both the host and device sides. That is structurally pieces 1 and 2 of this port, landed independently and with the same function name, `ggml_cuda_mmq_get_config_rdna3_5`.

**The structure landed upstream; the tuning did not.** Upstream's rdna3_5 table is a verbatim copy of rdna4:

| J | upstream rdna3_5 (`nthreads`, `occupancy`, `I`) | this fork |
|---|---|---|
| `<= 32` | 128, 2, 64 | same |
| `>= 48` | **256, 2, 128** | **128, 2, 64** |

So gfx1151 on stock upstream now gets its own table but still runs rdna4's wide tiles at every `J >= 48` - the same tile shape the 2026-04 A/Bs beat by +27%/+37%. The warning that used to live in the root README ("without our dispatch branch gfx1151 silently inherits the rdna4 table") still holds in substance; only the mechanism changed.

**Resolution.** Our `mmq-config-rdna3_5.cuh` was deleted and its values written into upstream's `mmq-config-rdna3-5.cuh`. The fork's remaining patch to that file:

- 164 shared entries retuned `256, 2, 128` -> `128, 2, 64` (`J >= 48`, all types)
- 26 rows added: the 22 `J=48, fallback=true` rows the MoE cap needs (one per type), plus 4 `Q2_K` rows at `J >= 96` that only fit in LDS at `I=64`
- nothing dropped, and no entry diverges from upstream on `sram_layout`, `K_vram`, or `stream_k`

`Q2_0` support came free with the merge - it is a new type from upstream [PR #25707](https://github.com/ggml-org/llama.cpp/pull/25707) that our old standalone table did not cover.

The `mmq.cuh` diff shrank from +11/-1 to **+5/-1**: the dispatch is upstream's now, and the MoE `J` cap is all that is left.

### Post-merge bench (2026-08-02, build `b73cfa4`)

Prefill came back **flat** against `05e837f` (+0.3% to +1.9%, three of four depths inside the noise floor) while decode rose 3-4%. Full table in [qwen3.6-baseline.md](qwen3.6-baseline.md#2026-08-02--post-rebase-re-bench-build-b73cfa4).

Flat prefill is the result this merge needed. The failure mode was losing the retune - if the values had not carried across to upstream's file, gfx1151 would have fallen back to rdna4's wide tiles at `J >= 48`, which the 2026-04 A/Bs measured at **-27% to -37%**. Nothing remotely that size appeared, across a window that also absorbed 136 upstream commits and two build-flag removals.

This still is not the port-off A/B. It bounds the downside, not the upside: it says the tuning survived, not what the tuning is worth.

## Verification so far

gfx1151 can't be compiled or benched from the dev host, so the table is checked by [`mmq-table-check.cpp`](mmq-table-check.cpp) — a standalone harness that stubs just enough of `mmq.cuh` to include the real config tables and reuse the real `CASE` macro under plain `clang++`:

```bash
cd ggml/src/ggml-cuda
clang++ -std=c++17 -I. -Wall -o /tmp/mmq-table-check ../../../strix-halo/mmq-table-check.cpp && /tmp/mmq-table-check
```


- The table compiles, so the real `static_assert`s (`nthreads % 32 == 0 && <= 512`, `occupancy <= 8`, `I % 32 == 0`, `J % 8 == 0`, `K_vram % 256 == 0`) all hold.
- Every `(type, J, fallback)` entry reachable in rdna4 exists in rdna3_5 with `nthreads=128, I=64, occupancy=2`, and identical `sram_layout` / `K_vram` / `stream_k`.
- `I == nwarps*16` holds for every type.
- `J=48` present for both `fallback` values; MoE cap resolves to 48, dense to 128.
- No `(type, fallback, ncols_max)` combination drives `J_best` to 0 (which would `GGML_ABORT`).

This proves the table is *well-formed and dispatches as intended*. It says nothing about whether it is *fast*. That needs hardware.

## Open knob: `occupancy`

`occupancy` is the second argument to `__launch_bounds__` (min blocks per CU), and it did **not exist** when this tuning was originally validated — the old build let the compiler choose. It is the one value in the table that is a guess.

The table ships `occupancy = 2`, copied from rdna4. Note what that means relative to rdna4:

| | nthreads | occupancy | threads resident/CU | register budget/thread |
|---|---:|---:|---:|---|
| rdna4 (what gfx1151 inherits without this port) | 256 | 2 | 512 | baseline |
| **rdna3_5 (this port)** | 128 | 2 | 256 | **2x baseline** |
| rdna3_5, `occupancy=4` | 128 | 4 | 512 | baseline |

`occupancy=2` is the thesis-consistent choice: this whole finding exists because RDNA3.5 hits VGPR pressure at rdna4's tile sizes, and halving the block while holding the occupancy target gives each thread twice the register headroom. It is also the conservative choice against spills — the failure mode that killed the MMVQ "join RDNA3_0" experiment ([mmvq-rdna3_5.md](mmvq-rdna3_5.md)).

The risk is the other direction: 256 threads/CU may under-occupy a 40-CU part and lose latency hiding on ~256 GB/s LPDDR5x. **`occupancy ∈ {2, 4}` is the first sweep** whenever this table is next revisited — it shipped unvalidated at 2 and the bench in [Outcome](#outcome) does not probe it. Sweep it against a port-off baseline, one variable at a time.

## Outcome

**Kept.** Build `05e837f`, ROCm 7.14.0, gfx1151, [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 3 config (Qwen 3.6 35B-A3B Q4_K_XL, f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3 -d 0,2048,8192,16384`), compared against the last shipped build `3511e7d` (TheRock `7.13.0a20260514`):

| test | 3511e7d (shipped) | 05e837f (this) | delta |
|---|---:|---:|---:|
| pp512 @ d=0       | 1350.31 ± 7.27  | 1428.13 ± 19.35 | **+5.8%** |
| pp512 @ d=2,048   | 1261.93 ± 4.56  | 1299.39 ± 8.82  | +3.0% |
| pp512 @ d=8,192   | 1085.56 ± 16.49 | 1135.42 ± 21.21 | +4.6% |
| pp512 @ d=16,384  |  916.76 ± 4.62  |  971.25 ± 9.54  | **+5.9%** |
| tg128 @ d=0       |   47.25 ± 0.06  |   49.81 ± 0.11  | +5.4% |
| tg128 @ d=2,048   |   46.96 ± 0.15  |   48.98 ± 0.94  | +4.3% |
| tg128 @ d=8,192   |   45.79 ± 0.15  |   48.13 ± 0.14  | +5.1% |
| tg128 @ d=16,384  |   44.25 ± 0.15  |   46.43 ± 0.14  | +4.9% |

Nothing regressed at any depth. ~971 t/s prefill at d=16k is the best result this fork has produced.

Correctness was verified on real hardware before the bench, since the `I=64` / `nthreads=128` tile shape changes the MMA write-back geometry: `test-backend-ops test -b ROCm0` passed **790/790 on `MUL_MAT_ID`** (the MoE path the `J_max` cap touches) and **1134/1134 on `MUL_MAT`**.

### Caveat: this is a bundle delta, not a port A/B

**The port-off run was not done, so the port's own contribution is unmeasured.** The table above compares two builds that differ by three things at once: 234 upstream commits, ROCm 7.13 -> 7.14.0, and this re-port.

The tg128 column is the tell. It rose ~5% at every depth, and **this port cannot move tg** — MMQ tile tuning is a prompt-side fix, decode goes through MMVQ. The 2026-04-19 A/B in [mmq-rdna3_5.md](mmq-rdna3_5.md#post-upstream-sync-re-bench-2026-04-19) measured tg flat while pp moved +37%, confirming that. So roughly +5% of this is the bundle, and the pp gains (+3.0% to +5.9%) are the same magnitude. The port's share could be most of it or close to none; these numbers cannot tell.

Do **not** read the +5.8% at d=0 as the decision rule passing. That rule ("keep if pp512@d=0 improves >5% and no depth regresses") is written to compare port vs no-port. Applying it to a bundle delta is the same class of error that let the rocWMMA regression hide for five weeks: a number that looked fine against the wrong reference.

**Why it was kept anyway** (2026-07-16 call): the rdna4 table gfx1151 would otherwise inherit is `nthreads=256, occupancy=2, I=128` uniformly for every type — **numerically identical** to the generic WMMA constants gfx1151 got from upstream before #24127 (`get_mmq_x_max_host` -> 128 via `amd_wmma_available`, `get_mmq_y_host` -> 128, `mmq_get_nwarps_host` -> `256/32` = 8 warps). Those are exactly the values the 2026-04 A/Bs beat by +27% and +37%. The port is therefore very likely still winning; it just wasn't re-proven here.

The residual doubt worth holding: those A/Bs are 234 commits old, the MMQ kernel has been reworked since (stream-k in #22298, the data-layout and config refactors), and `occupancy=2` did not exist when they were run. Upstream's own work may have narrowed the gap. If this ever needs settling, the A/B is one throwaway branch: `git checkout upstream/master -- ggml/src/ggml-cuda/mmq.cuh`, build with `--build-arg LLAMACPP_VERSION=<sha>`, run the matrix above.

## Upstreamability

Worth a look, but not on these numbers. PR #24127's stated purpose is exactly this — per-arch tuning tables without cross-arch side effects — and gfx1151 silently inheriting the generic 96-CU-era constants is the gap the refactor was built to close.

**PR #26199 made this materially easier.** The file exists upstream, was added explicitly "so they can be tuned independently", and shipped untuned — a copy of rdna4. The contribution is no longer "add a new per-arch table"; it is "fill in the one that is already there, with numbers". That is a much smaller ask of a reviewer.

Blockers, in order:

1. **A real port-off A/B.** The bundle delta in [Outcome](#outcome) will not persuade a maintainer, and shouldn't. Since #26199 this is easier too — the baseline is now literally `git checkout upstream/master -- ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`, no dispatch surgery.
2. **An `occupancy` sweep** so the value isn't a guess.
3. **A decision on the MoE `J` cap** — whether it belongs in the table rather than as a special case in `mul_mat_q_switch_J`. A reviewer will ask, and "it's a per-arch table, put it in the table" is the likely answer.

Upstream's AI policy changed on 2026-07-23 ([PR #26012](https://github.com/ggml-org/llama.cpp/pull/26012)): AI-generated code is now allowed, provided a human understands it and will maintain it. The old "majority human-authored" rule is gone. What has **not** changed is that an agent must never write the PR description or reply to reviewers — see [AGENTS.md](../AGENTS.md).
