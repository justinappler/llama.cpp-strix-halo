# MMQ RDNA3.5 config table — re-port of Finding #5/#8 onto upstream PR #24127

## Status (2026-07-16 — ported, **not yet benched**)

Third incarnation of the gfx1151 MMQ tuning. The previous two forms (the six-edit ternary patch from [mmq-rdna3_5.md](mmq-rdna3_5.md), and the dense/MoE split from [pp-rdna3_5-tile-mmq.md](pp-rdna3_5-tile-mmq.md)) were **dropped, not rebased**, on the 2026-07-16 upstream sync. Upstream [PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127) (JohannesGaessler, merged 2026-07-13, `6eddde06`) deleted every function they edited. This doc covers the re-port against the replacement architecture.

**No numbers here yet.** All prior deltas (+27% pp@d=0, +6.3% pp@d=16k) were measured against a code path that no longer exists. Treat them as direction-of-win only until the matrix below is run.

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

Three pieces, +11/-1 in `mmq.cuh` plus one new file:

1. **[ggml/src/ggml-cuda/mmq-config-rdna3_5.cuh](../ggml/src/ggml-cuda/mmq-config-rdna3_5.cuh)** (new, 303 lines) — generated mechanically from `mmq-config-rdna4.cuh` with `nthreads: 256 -> 128`, `I: 128 -> 64`. `sram_layout`, `K_vram`, and `stream_k` are carried over unchanged for all 21 types; only the tile shape is retuned.
2. **Dispatch** in [mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh) — host-side `GGML_CUDA_CC_IS_RDNA3_5(cc)` branch and device-side `#elif defined(RDNA3_5)`, both placed **before** the `amd_wmma_available` / `AMD_WMMA_AVAILABLE` branch that would otherwise route us to rdna4.
3. **MoE J cap** in `mul_mat_q_switch_J` — `args.expert_bounds != nullptr` still reaches the J-selection loop, so the old dense/MoE split survives as a one-line bound on the search:

```cpp
const int J_max = GGML_CUDA_CC_IS_RDNA3_5(cc) && args.expert_bounds != nullptr ? 48 : 128;
```

### The J=48 fallback rows

rdna4's `fallback=true` rows only cover `J ∈ {16, 32, 64, 128}` (PR #24127 deliberately thinned the fallback specializations to powers of 2 to cut compile time). With a MoE cap of 48 and no `J=48` fallback entry, the selection loop would silently settle on **32** whenever `ne01 % 128 != 0` — a behaviour change from the pre-rebase build that would confound the re-bench.

Our table therefore adds a `J=48, fallback=true` row per type (21 rows). Verified: MoE resolves to `J=48` for both `fallback` values, dense stays at `J=128`.

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
| rdna4 (what gfx1151 gets today) | 256 | 2 | 512 | baseline |
| **rdna3_5 (this port)** | 128 | 2 | 256 | **2x baseline** |
| rdna3_5, `occupancy=4` | 128 | 4 | 512 | baseline |

`occupancy=2` is the thesis-consistent choice: this whole finding exists because RDNA3.5 hits VGPR pressure at rdna4's tile sizes, and halving the block while holding the occupancy target gives each thread twice the register headroom. It is also the conservative choice against spills — the failure mode that killed the MMVQ "join RDNA3_0" experiment ([mmvq-rdna3_5.md](mmvq-rdna3_5.md)).

The risk is the other direction: 256 threads/CU may under-occupy a 40-CU part and lose latency hiding on ~256 GB/s LPDDR5x. **`occupancy ∈ {2, 4}` is the first sweep** if the headline bench comes in flat or down. Do not sweep it before establishing the baseline below — one variable at a time.

## Bench plan

This rebase moves three things at once (234 upstream commits including PR #24127, the ROCm 7.14.0 switch per [rocm-config.md](rocm-config.md), and this re-port), so the port cannot be attributed without an explicit A/B.

**Run 1 — new baseline (port OFF).** Build at the rebase commit with the `mmq-config-rdna3_5.cuh` dispatch reverted (gfx1151 falls through to rdna4). This is the honest "what does upstream give us today" number and it is what every future delta gets measured against. Required because both the ROCm bump and 234 commits landed in the same window.

**Run 2 — port ON.** Same build, dispatch restored.

Both at the [qwen3.6-baseline.md](qwen3.6-baseline.md) Run 3 config: Qwen 3.6 35B-A3B Q4_K_XL, f16/f16 KV, FA on, `-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3 -d 0,2048,8192,16384`.

**Decision rule** (unchanged from the original finding): keep if pp512 @ d=0 improves >5% outside noise **and** no depth regresses. Revert if d=16k regresses even when d=0 wins — agentic coding lives at depth. Noise floor on this host is ±1.5%.

Also worth capturing from Run 1 vs the last shipped build (`3511e7d`, `1350/917 t/s` at d=0/16k): whether the ROCm 7.14.0 + 234-commit bundle moved production numbers on its own. That is the [re-bench checklist](../README.md#re-bench-checklist-after-upstream-sync-or-rocm-bump) obligation and this is the cheapest time to satisfy it.

## Upstreamability

Worth a look once benched. PR #24127's stated purpose is exactly this — per-arch tuning tables without cross-arch side effects — and gfx1151 inheriting a table tuned for a 96-CU / 960 GB/s discrete part is the kind of gap the refactor was built to close. A clean `mmq-config-rdna3_5.cuh` with bench numbers is a far more plausible contribution than the old six-edit ternary patch ever was.

Blockers before proposing anything: real numbers on both runs above, an `occupancy` sweep so the value isn't a guess, and confirmation that the MoE J cap doesn't want to be a table property rather than a special case in `mul_mat_q_switch_J`. Per [AGENTS.md](../AGENTS.md), any upstream PR needs a human author who can defend it without assistance.
