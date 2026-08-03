# Kernel-time breakdown — where pp and tg actually spend their time on gfx1151

## Status (2026-07-18 — measured, no code change)

First kernel-level profile of the canonical Qwen 3.6 workload on this fork. Answers the question every backlog item has been priced without: **how much of the budget does each kernel family own**, at shallow and deep context, for prefill and decode. Build `4781fb9` (current master), ROCm 7.14.0, `rocprofv3 --kernel-trace` (works on the 7.14.0 release; the 7.13-nightly crash noted in `profile.sh` is gone, and `rocprof-sys-sample` no longer ships in the image).

Headline shares of GPU-busy time:

| phase | MMQ | FA (TILE) | gdn/ssm | MMVQ | everything else |
|---|---:|---:|---:|---:|---:|
| pp512 @ d=0 | **58%** | 2% | 12% | — | 28% |
| pp512 @ d=16,384 | **38%** | **32%** | 9% | — | 21% |
| tg @ d=0 | — | 0.7% | 1.8% | **77%** | 20% |
| tg128 @ d=16,384 | — | 8.5% | 1.7% | **71%** | 19% |

Three conclusions that reprice the backlog, argued in [Implications](#implications):

1. **MMQ tuning has a shrinking ceiling at depth.** A further 10% MMQ win is +5.8% pp at d=0 but only +3.8% at d=16k, and FA's share grows linearly with depth while MMQ's is flat. The occupancy `{2,4}` probe stays worth its one build; a broad table sweep does not.
2. **The FA TILE kernel is the pp-at-depth lever.** At 2048-wide ubatches FA time crosses MMQ around d≈13k and keeps growing. `flash_attn_tile<256,256,4,8>` runs 159ms of the 501ms pp512@16k budget in just 10 dispatches.
3. **Decode is one kernel plus launch overhead.** `mul_mat_vec_q<Q8_0>` alone is **51% of decode busy time** (161 calls/token — the dense/gdn projections, not the experts), and the GPU sits idle **16% of decode wall time** between ~1,565 dispatches/token. The dedicated MMVQ table (backlog #2 in the root README) should whitelist Q8_0 first, and the idle fraction makes HIP graph launch (`GGML_HIP_GRAPHS`) worth a status check before any kernel tuning.

## Method

Two traces via `profile.sh` (server-configs), canonical bench flags with `-r 1`:

```bash
PROFILER_CMD=rocprofv3 PROFILER_FLAGS="--kernel-trace -d ." ./profile.sh /app/llama-bench \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ctk f16 -ctv f16 -fa 1 -b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 1 -d 0        # trace A
  ...                                                                        -d 16384    # trace B
```

Traces: `run-20260718-222021` (A, d=0) and `run-20260718-222039` (B, d=16384) under `services/llamacpp/profiling/traces/`. Output is a rocpd SQLite db; the `kernels` view carries per-dispatch timestamps plus per-kernel VGPR/LDS/scratch. Aggregation: bucket by kernel-name regex, segment phases on >50ms GPU-idle gaps (model load / pp test / tg test separate cleanly), split the 16k fill ramp into its 8 ubatches by counting `flash_attn_tile` dispatches (10 per ubatch — Qwen 3.6 has 10 full-attention layers and 30 gated-delta-net layers).

**Profiling overhead ~3%**: pp512@d=0 measured 1381 t/s under trace vs 1428 clean; tg128 48.1 vs 49.8. Shares are trustworthy; absolute times are ~3% pessimistic.

Kernel-name sanity check worth recording: the trace confirms the [Finding #9](mmq-rdna3_5-config-table.md) table dispatches as designed in production — MoE experts run `mul_mat_q<Q4_K/Q5_K/Q6_K, J=48>`, dense runs `mul_mat_q<Q8_0, J=128>`, FA runs the TILE D=256 path.

## Prefill

Bucket tables (GPU-busy ms, share):

| bucket | pp512 @ d=0 (718ms, 2 passes) | pp512 @ d=16k (501ms, 1 pass) |
|---|---:|---:|
| mmq | 417.7 (58.2%) | 192.2 (38.4%) |
| fa | 13.8 (1.9%) | 160.6 (32.1%) |
| gdn/ssm | 87.3 (12.2%) | 44.7 (8.9%) |
| mem/copy | 53.8 (7.5%) | 31.2 (6.2%) |
| gemm_f32 (hipBLASLt) | 24.8 (3.5%) | 12.5 (2.5%) |
| quantize (q8_1) | 21.5 (3.0%) | 10.7 (2.1%) |
| moe-route | 21.1 (2.9%) | 10.3 (2.1%) |
| other (norm/rope/eltwise/softmax) | 73.0 (10.2%) | 36.2 (7.2%) |

Per-ubatch decomposition of the 16,384-token fill (2048-token ubatches, trace B):

| ubatch (kv range) | busy ms | mmq | fa | gdn/ssm | rest | fa share |
|---|---:|---:|---:|---:|---:|---:|
| 0–2k | 1344 | 534 | 89 | 239 | 482 | 6.6% |
| 2k–4k | 1422 | 541 | 170 | 224 | 487 | 12.0% |
| 4k–6k | 1504 | 541 | 249 | 224 | 490 | 16.6% |
| 6k–8k | 1590 | 544 | 331 | 224 | 491 | 20.8% |
| 8k–10k | 1672 | 544 | 412 | 224 | 492 | 24.6% |
| 10k–12k | 1719 | 532 | 476 | 223 | 488 | 27.7% |
| 12k–14k | 1769 | 523 | 540 | 219 | 487 | 30.5% |
| 14k–16k | 1836 | 532 | 590 | 219 | 495 | 32.1% |

MMQ, gdn, and "rest" are depth-flat as expected; FA grows ~72ms per 2048 tokens of depth and **crosses MMQ inside the 2048-wide ubatch at d≈13k**. The pp512 probe at d=16k still has mmq > fa (192 vs 161) only because its query is 4x narrower. Extrapolating the slope: at d=32k a 2048-wide ubatch would be ~45% FA, ~24% MMQ.

Kernel-level detail, pp512 @ d=16k (no kernel in either trace spills — `scratch_size=0` everywhere):

| kernel | calls | ms | vgpr | lds |
|---|---:|---:|---:|---:|
| `flash_attn_tile<256,256,4,8>` | 10 | 158.9 | 192 | 29,696 |
| `mul_mat_q<Q4_K, 48>` | 78 | 75.9 | 128 | 26,816 |
| `mul_mat_q<Q8_0, 128>` | 250 | 56.2 | 232 | 38,400 |
| `mul_mat_q<Q5_K, 48>` | 38 | 53.1 | 168 | 26,816 |
| `gated_delta_net_cuda<128>` | 30 | 41.0 | 32 | 0 |
| `concat_non_cont<u32>` | 30 | 26.2 | 16 | 0 |

Two occupancy facts fall out of the LDS column (64 KiB per WGP on RDNA3.5):

- MoE rows (`J=48`, LDS 26,816): two workgroups fit (53,632 < 65,536) — the table's `occupancy=2` is **achievable**, and with vgpr=128 the register budget is not the binding constraint either. The `occupancy=4` arm of the planned sweep would need LDS x4 = impossible at this tile; only 2 fit. So for the MoE rows the `{2,4}` sweep is **moot at J=48** — LDS caps residency at 2 regardless of what `__launch_bounds__` requests.
- The dense row (`Q8_0, J=128`, LDS 38,400): **two workgroups do not fit** (76,800 > 65,536), so its declared `occupancy=2` is unreachable and the kernel runs at 1 workgroup/WGP. Dense Q8_0 is 28% of shallow MMQ time (119 of 418ms). A dense `J=64` row (LDS would roughly halve) is the one table experiment this profile actually motivates.

## Decode

| bucket | tg @ d=0 (2181ms, 126 tok) | tg128 @ d=16k (2433ms, ~129 tok) |
|---|---:|---:|
| mmvq | 1688.3 (77.4%) | 1731.3 (71.2%) |
| fa | 15.0 (0.7%) | 207.1 (8.5%) |
| mmvf (f32 vec) | 123.8 (5.7%) | 127.1 (5.2%) |
| mem/copy | 109.8 (5.0%) | 113.5 (4.7%) |
| quantize | 45.3 (2.1%) | 47.7 (2.0%) |
| gdn/ssm | 40.1 (1.8%) | 41.3 (1.7%) |
| moe-route | 21.0 (1.0%) | 22.2 (0.9%) |
| other | 137.6 (6.3%) | 142.9 (5.9%) |

By kernel at d=0: `mul_mat_vec_q<Q8_0>` (non-MoE variant) is 1122ms — **51% of decode busy in one kernel**, 161 calls/token, vgpr=16. The MoE expert variants (`Q4_K` fused-ids 253ms, `Q5_K` 163ms, `Q8_0` fused 123ms) total ~25%. The Q8_0 dominance means the dense/gdn projections, not the experts, are the decode bottleneck — consistent with A3B: ~3B active params but the dense skeleton runs every token.

**Launch overhead:** decode wall is 20.6ms/token vs 17.3ms GPU-busy — **16% idle**, ~1,565 dispatches/token. Perfectly reproduced at depth (17% idle). Untraced tg128 is 49.8 t/s; zero-idle busy-only would be ~58 t/s. That 16% is the single largest non-kernel decode cost and no MMVQ tuning touches it.

## Implications

Ranked repricing of the [root README backlog](../README.md#strix-halo-next-experiments) against these shares:

1. **FA TILE tuning for gfx1151** (not currently a backlog item) — 32% of pp at d=16k and the only share that grows with depth; the agentic workload lives there. The kernel is `fattn-tile` with the fork's D=256/ncols=32 override already in place; a config sweep of its nwarps/cols-per-block for gfx1151, or upstream-watching for TILE improvements, is now the highest-ceiling pp work.
2. **MMVQ dedicated RDNA3.5 entry** (backlog #1) — confirmed as the decode lever, and the profile sharpens it: whitelist **Q8_0 first** (51% of decode alone), then Q4_K/Q5_K. tg ceiling if MMVQ improves 15%: ~+11% tg.
3. **HIP graphs status check** (new) — 16% decode idle at ~1.5k dispatches/token. ~~If `GGML_HIP_GRAPHS`/graph capture is off, that gap is the price; potentially ~+15% tg.~~ **Resolved 2026-07-18, same day:** graphs are already on and engaged in production ([hip-graphs.md](hip-graphs.md) Results — steady-state decode is 100% `hipGraphLaunch` replay). The 16% idle is *inside* graph replay: ~2µs of inter-kernel scheduling latency per dispatch × ~1,565 dispatches/token. Not recoverable by launch batching; only by fewer kernels per token (fusion — mostly an upstream ride).
4. **MMQ occupancy sweep** (Finding #9 open knob) — deflated. LDS already caps MoE J=48 rows at 2 workgroups/WGP, so `occupancy=4` cannot bind there; and dense Q8_0 J=128 can't even reach 2. The interesting variant is a **dense `J=64` row** (halves LDS, restores 2-WG residency for the 28%-of-MMQ dense share). The port-off A/B remains owed for attribution regardless.
5. **gdn/ssm + glue** — gated-delta-net path (gdn kernel + `concat_non_cont` + hipBLASLt f32 GEMMs + l2norm) is ~15-18% of pp and depth-flat; `concat_non_cont` alone is 5% of pp@16k, which smells like a layout fix upstream will eventually take. Watch, don't lead.

## Reproducing

Aggregation scripts are throwaway (scratchpad), but the queries are one-liners against the rocpd db, e.g. top kernels:

```sql
SELECT name, count(*), sum(duration)/1e6 AS ms, vgpr_count, scratch_size, lds_size
FROM kernels GROUP BY name ORDER BY ms DESC LIMIT 15;
```

Phase boundaries: segment on >50ms gaps between consecutive dispatch end/start times; llama-bench phases (load, pp test, tg test) separate cleanly, ubatches inside a fill do not (<3ms gaps) — split those by `flash_attn_tile` dispatch count (10 per ubatch on this model).
