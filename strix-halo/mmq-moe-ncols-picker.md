# MMQ routed-MoE ncols picker — re-port of upstream PR #24546 onto the config-table search

## Status (2026-07-16 — ported, **not yet benched**)

Replaces the static `J_max = 48` MoE cap from [Finding #8/#9](mmq-rdna3_5-config-table.md) with the dynamic
routed-width picker proposed in upstream [PR #24546](https://github.com/ggml-org/llama.cpp/pull/24546)
(ravel7524, open since 2026-06-12). Both mechanisms exist to stop routed-MoE matmuls from selecting
tiles that are far too wide for the expert slice they actually cover. This experiment asks which one
is better on gfx1151.

Arm A (static `J=48`) needs no build: it is exactly `05e837f`, the current production commit, already
benched in [qwen3.6-baseline.md](qwen3.6-baseline.md). **Only arm B is new**, so this is a genuine
one-variable A/B against numbers we already trust.

## Why the upstream PR does not apply as written

PR #24546 is written against the **pre-#24127** `mmq.cuh`. It patches `mul_mat_q_case`'s tile search
using `mmq_x`, `mmq_x_max` and `mmq_get_granularity_host` — all deleted by
[PR #24127](https://github.com/ggml-org/llama.cpp/pull/24127) (merged 2026-07-13), the same refactor
that forced the Finding #5 → Finding #9 re-port. The idea survives; the diff does not. This doc
re-authors it onto `mul_mat_q_switch_J`.

The PR also gates on `GGML_CUDA_CC_IS_CDNA || RDNA2 || RDNA3 || RDNA4`. `GGML_CUDA_CC_IS_RDNA3`
([common.cuh:89](../ggml/src/ggml-cuda/common.cuh#L89)) covers RDNA3.5, so gfx1151 *is* in the
upstream predicate — the author confirmed this on 2026-07-04 when GZGavinZhao offered to test on
Strix Halo. That test never happened. **This port is scoped to RDNA3.5 only**: gfx1151 is the only
arch we can measure, and narrowing the predicate keeps the A/B single-variable.

## What the two mechanisms actually do

Both act on the tile search in `mul_mat_q_switch_J`
([mmq.cuh:1377-1407](../ggml/src/ggml-cuda/mmq.cuh#L1377-L1407)), which walks `J` upward and keeps the
`J` minimising `ntiles_x = ceil(ncols / J)`, exiting once some `J` covers the width in one tile.

On the routed-MoE path ([mmq.cu:230-236](../ggml/src/ggml-cuda/mmq.cu#L230-L236)):

| arg | value | meaning |
|---|---|---|
| `ncols_max` | `ne12` | tokens in the ubatch — the **worst case**, one expert receiving every token |
| `ncols_dst` | `ne12 * n_expert_used` | total (token, expert) pairs |
| `nchannels_x` | `ne02` | number of experts |

`ncols_max` being the worst case is the whole problem: it is enormous (2048 at our ubatch), so the
search never reaches `ntiles == 1` and simply runs to whatever ceiling it is given.

- **Arm A (today).** Clamp the ceiling: `J_max = 48` for RDNA3.5 + MoE. The search runs out of room
  and returns `J = 48`. Static — the same answer for every MoE shape, tuned at one operating point.
- **Arm B (this port).** Leave the ceiling at 128 and fix the *objective* instead: compute tiles from
  `ncols_typical = ceil(ncols_dst / nchannels_x)`, the width an average expert really covers. The
  search then terminates naturally at the smallest tile that covers a typical expert.

The launch grid is untouched: `launch_mul_mat_q` computes `ntx` from `args.ncols_max`
([mmq.cuh:1314](../ggml/src/ggml-cuda/mmq.cuh#L1314)), independently of the search. Worst-case
coverage — and therefore correctness — is preserved. This is the crux of why the PR is safe.

## What this predicts for Qwen 3.6 35B-A3B

256 experts, 8 active, production `-b 4096 -ub 2048`:

```
ncols_typical = ceil(2048 * 8 / 256) = 64
```

The `rdna3_5` table defines `J ∈ {16, 32, 48, 64, 128}` at `fallback=true`, so the search walks
16 → 32 → 64 and exits at `J = 64` (`ntiles == 1`). Arm A gives 48.

**The whole experiment is 48 vs 64 at our operating point.** That is a narrow gap, and it is worth
being honest that this may well land inside noise. The value is not only the delta:

- It puts a number on a knob we shipped by hand and have never actually probed.
- It is the port-attribution measurement [Finding #9](mmq-rdna3_5-config-table.md#outcome) is missing,
  arrived at from the other direction.
- Either result is data upstream has asked for twice and nobody has produced.

Upstream's W7800/gfx1100 sweep reports +7.57% pp2048 on this exact model/ubatch. gfx1100 is RDNA3.0
with a different table (`I=128`, `nthreads=256`), so treat that as direction-only.

## Where the arms diverge beyond our bench

The picker only clamps while `ncols_typical < 128`. Since
`ncols_typical = ne12 * n_expert_used / n_experts`, the crossover is
`ne12 = 128 * n_experts / n_expert_used` = **4096** for Qwen 3.6. Above it, arm B selects `J = 128`
where arm A still forces 48.

This is a real semantic difference and the honest risk in the port: Findings #5/#8 established that
wide tiles hurt MoE on RDNA3.5 because of VGPR pressure, but **every one of those measurements was
taken at `ub = 2048`, i.e. only ever in the `typical = 64` regime.** The `typical >= 128` regime is
untested on this chip in either direction. Our production config cannot reach it (`ub = 2048`), so
the bench below does not probe it. Noted, not resolved — if the `ub = 4096` row below regresses,
that is the reason, and the fix is a `min(J, 64)` floor rather than reverting the picker.

## Bench plan

Arm A = `05e837f` (already measured). Arm B = this commit. ROCm 7.14.0 both, one variable.

Production matrix, Qwen 3.6 35B-A3B Q4_K_XL, f16/f16 KV, FA on:

```
-b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3 -d 0,2048,8192,16384
```

| depth | arm A pp512 (t/s) | arm B pp512 | arm A tg128 | arm B tg128 |
|---:|---:|---:|---:|---:|
| 0 | 1428.13 ± 19.35 | | 49.81 ± 0.11 | |
| 2,048 | 1299.39 ± 8.82 | | 48.98 ± 0.94 | |
| 8,192 | 1135.42 ± 21.21 | | 48.13 ± 0.14 | |
| 16,384 | 971.25 ± 9.54 | | 46.43 ± 0.14 | |

`tg128` is the control. Decode runs through MMVQ, not MMQ, and `ncols_dst == 1` there — the picker
cannot touch it. **Any tg128 movement beyond noise means the A/B is contaminated**, exactly the
lesson from Finding #9, where a ~5% tg rise revealed the bundle delta was mostly upstream/ROCm.

Secondary, only if the primary is not a regression — probes the crossover the production config
cannot reach:

```
-ub 4096 -p 512 -d 0,8192
```

Expect arm B to pick `J = 128` here and arm A to hold 48. This is the one row where a large
regression is plausible.

## Keep / revert

- **Keep** if pp512 improves beyond noise at any depth with tg128 flat, *and* `-ub 4096` does not
  regress badly. Then report the gfx1151 numbers on #24546 and drop our static cap.
- **Revert to arm A** if pp512 is flat or down. Static 48 stays; comment on #24546 that the picker is
  neutral-to-negative on RDNA3.5 and that gfx1151 wants a tighter bound than gfx1100 — itself a
  useful finding for the PR, since it currently claims all of RDNA3.
- **Flat is a real outcome and should be recorded as one.** 48 vs 64 is a small perturbation. If it
  is noise, the honest conclusion is that this knob does not matter at our operating point, which
  retires it and stops it being re-litigated.

## Correctness gate

Run before any bench — the grid still covers `ncols_max`, so this must be clean:

```
test-backend-ops test -b ROCm0 -o MUL_MAT_ID
test-backend-ops test -b ROCm0 -o MUL_MAT
```

Baseline from Finding #9: 790/790 MUL_MAT_ID, 1134/1134 MUL_MAT.
