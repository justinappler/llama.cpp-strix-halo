# HIP graphs at decode — why is the GPU idle 16% of tg wall time?

## Status (2026-07-18 — H2/H3/H4 ruled out by trace; H1 open, needs OFF-build A/B)

**This doc is an agent brief.** It is written to be handed to an agent with no other context; everything needed is here or one link away. It also serves as the experiment's lab-notebook entry per the fork workflow (hypothesis first, results appended to this file, one commit per kept change).

## Context — what is already measured

The 2026-07-18 kernel profile ([kernel-time-breakdown.md](kernel-time-breakdown.md)) found that decode (tg) on the production workload issues **~1,565 kernel dispatches per token** and the GPU is **idle 16% of decode wall time** (20.6ms wall vs 17.3ms GPU-busy per token, reproduced at d=0 and d=16k). Untraced tg128 is 49.8 t/s; busy-only would be ~58 t/s. That idle fraction is the single largest non-kernel decode cost — bigger than any plausible MMVQ tuning win.

Graph capture (record the whole per-token launch sequence once, replay it each token) is the standard fix for exactly this. The question is why it isn't already saving us.

Production workload: Qwen 3.6 35B-A3B Q4_K_XL on gfx1151 (Strix Halo, 40 CU RDNA3.5, unified memory), ROCm 7.14.0, fork master (`4781fb939` at time of writing). It is a routed-MoE model (256 experts, 8 active) with 10 full-attention + 30 gated-delta-net layers.

## What the code says (verified 2026-07-18 against `4781fb939`)

- `GGML_HIP_GRAPHS` is **ON by default** upstream ([ggml/CMakeLists.txt:216](../ggml/CMakeLists.txt#L216)), and the deploy Dockerfile does not override it. So `USE_CUDA_GRAPH` should be compiled into our HIP build via [common.cuh:1207](../ggml/src/ggml-cuda/common.cuh#L1207). **Verify, don't assume** — confirm the macro is actually live in the shipped image (e.g. cmake cache in the builder stage, or the empirical trace below).
- Runtime gating in [ggml-cuda.cu](../ggml/src/ggml-cuda/ggml-cuda.cu):
  - `ggml_cuda_graph_set_enabled` (~line 4070): permanently disables if `cc < GGML_CUDA_CC_VOLTA`. AMD ccs carry `GGML_CUDA_CC_OFFSET_AMD`, so gfx1151 should pass. Verify.
  - `ggml_cuda_graph_check_compability` (~line 2496): bails if any `MUL_MAT_ID` node has unquantized src0 or `ne[2] > get_mmvq_mmid_max_batch(type, cc)` ([mmvq.cu:108-](../ggml/src/ggml-cuda/mmvq.cu#L108)). At batch 1 decode `ne[2]` should be 1 and the per-type caps are >=4, so this *should* pass — but this is the prime suspect if it doesn't. Check what `get_mmvq_mmid_max_batch` returns for RDNA3.5 (which per-arch variant covers gfx1151?) and what `ne[2]` actually is on this model's expert matmuls at decode.
  - Warmup state machine in `ggml_backend_cuda_graph_compute` (~line 4091): graphs engage only after **2 consecutive calls with unchanged properties** (`ggml_cuda_graph_update_required`); any property change resets warmup. If some per-token property legitimately changes every token (e.g. a kernel grid dimension that depends on KV position), the warmup **resets forever and graphs never run** while appearing "enabled". This is the second suspect.

## Hypotheses (rank-ordered)

- **H1 — graphs engage, but replay doesn't shrink the gaps.** hipGraph launch on ROCm may not batch HSA packet submission the way cudaGraphLaunch batches on NVIDIA; 16% idle is then the hipGraph overhead floor on this stack, and the outcome of this investigation is a documented negative.
- **H2 — graphs never engage due to per-token property changes** (warmup reset loop). Fixable in principle; the fix's difficulty depends on which property churns.
- **H3 — graphs never engage due to the MUL_MAT_ID compatibility bail** (or some other unconditional bail on this model's op mix — gated-delta-net/ssm ops are exotic). Fix = extend the compatibility check, likely upstream-relevant.
- **H4 — `USE_CUDA_GRAPH` isn't compiled in at all** (flag lost somewhere in the Docker build). Fix = build flag, trivially.

## Investigation plan

**Step 1 — empirical ground truth (no rebuild).** Trace the HIP API during decode on the current production image:

```bash
# from the deploy repo's profiling directory
PROFILER_CMD=rocprofv3 PROFILER_FLAGS="--hip-trace --kernel-trace -d ." ./profile.sh /app/llama-bench \
  -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ctk f16 -ctv f16 -fa 1 -b 4096 -ub 2048 -ngl 999 -mmp 0 -p 0 -n 64 -r 1 -d 0
```

In the resulting rocpd SQLite db, look for `hipGraphLaunch` / `hipStreamBeginCapture` / `hipGraphExecUpdate` in the HIP API tables and count them against generated tokens. Graphs engaged = ~1 `hipGraphLaunch` per token and ~1,500 fewer `hipLaunchKernel` calls per token. Zero graph API calls = not engaging (H2/H3/H4). Capture-every-token = warmup reset loop (H2).

**Step 2 — attribute the bail (throwaway build, only if not engaging).** The `GGML_LOG_DEBUG` lines in the graph path print the reason but most need debug logging enabled; check whether `llama-bench` exposes a verbosity that surfaces them before instrumenting. If not, add temporary unconditional `GGML_LOG_INFO` prints at each bail site (compat check, warmup reset, arch disable), rebuild via the one-shot override (no fork commit needed):

```bash
LLAMACPP_REPO=<fork-or-local> docker compose build --build-arg LLAMACPP_VERSION=<throwaway-sha> llamacpp
```

**Step 3 — fix or document.** Depending on the branch taken:
- H4: add the cmake flag, done.
- H2/H3: assess the fix's size. If it's a small, defensible change (e.g. widening a compatibility condition that's provably safe at batch 1), land it as one commit on master with this doc updated. If it's invasive, document the blocker here and stop — this may be an upstream conversation instead.
- H1: measure and record the negative. Compare decode idle fraction (wall vs GPU-busy from the `kernels` view) with graphs confirmed on vs `GGML_HIP_GRAPHS=OFF` build; if identical, the 16% is not recoverable via graphs on ROCm 7.14 and the doc should say so to stop this being re-litigated.

**Step 4 — bench any kept change.** Correctness gate first, then the canonical matrix.

```
test-backend-ops test -b ROCm0 -o MUL_MAT_ID   # baseline 790/790
test-backend-ops test -b ROCm0 -o MUL_MAT      # baseline 1134/1134
```

Canonical bench (must match exactly — the deploy repo pins the rationale for each flag):

```
llama-bench -m .../Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -ctk f16 -ctv f16 -fa 1 \
  -b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3 -d 0,2048,8192,16384
```

Reference numbers (build `05e837f` ≡ current master, ROCm 7.14.0, host idle):

| depth | pp512 t/s | tg128 t/s |
|---:|---:|---:|
| 0 | 1428.13 ± 19.35 | 49.81 ± 0.11 |
| 2,048 | 1299.39 ± 8.82 | 48.98 ± 0.94 |
| 8,192 | 1135.42 ± 21.21 | 48.13 ± 0.14 |
| 16,384 | 971.25 ± 9.54 | 46.43 ± 0.14 |

**Decision rule:** keep if tg128 improves beyond the ±2% host noise floor at every depth with pp512 flat (graphs shouldn't touch prefill — a pp move means contamination, see the control-drift lesson in [mmq-moe-ncols-picker.md](mmq-moe-ncols-picker.md)). The theoretical ceiling is ~+19% tg (full idle recovery); anything >+5% is a clear keep. Flat is a real outcome — record it and close H1.

## Operational notes for the agent

- Deploy loop: commit on fork master -> pin the full 40-char SHA in the deploy repo's Dockerfile (`ARG LLAMACPP_VERSION`) -> deploy to the lab box. One-shot experimental builds use the `--build-arg` override above instead of moving the pin. The deploy repo is private; its README covers the mechanics.
- Before benching: confirm the host is otherwise idle and the running image matches the intended SHA (`llama-bench` footer prints `build: <sha>`).
- `profile.sh` defaults to `rocprofv3 --kernel-trace` as of 2026-07-18; traces are collected back into the deploy repo.
- Idle-fraction query against a trace db: wall = last `end` minus first `start` of the decode phase, busy = `sum(duration)`; decode phase = the mmvq-dominated tail (segment on >50ms gaps).
- **Repo rules ([AGENTS.md](../AGENTS.md)): no `git push`, no PRs, no PR/issue comments, ever.** Commits on the local fork are part of the workflow, but leave pushing to Justin. If a change looks upstream-worthy, write up the evidence in this doc and stop.
- Append findings to this doc under a `## Results` heading: which hypothesis held, trace evidence (run IDs), bench table if applicable, keep/revert and why.

## Results (2026-07-18)

**Step 1 (empirical ground truth) done.** Trace `run-20260718-224252` — decode-only, `-p 0 -n 64 -r 1 -d 0`, `PROFILER_FLAGS="--hip-trace --kernel-trace -d ."` (tg64 47.23 t/s under trace overhead; host confirmed idle first, `uptime` load 0.43, only steady-state containers running).

HIP graph lifecycle API calls, in order:

| call | count | when |
|---|---:|---|
| `hipLaunchKernel` (pre-capture) | 1,625 | before the first `hipStreamBeginCapture` — ~1 token's worth of ungraphed warmup, expected |
| `hipStreamBeginCapture` / `hipGraphInstantiate` | 2 each | the documented 2-consecutive-calls warmup, then one recapture |
| `hipGraphExecUpdate` | 2 | between capture and steady state |
| `hipGraphLaunch` | 63 | one per decode step from token ~3 onward |
| `hipLaunchKernel` (after the final `hipGraphInstantiate`) | **0** | steady state is 100% graph replay |

Total kernel dispatches (`kernels` table): 102,078 over 64 tokens = ~1,595/token, matching the ~1,565/token counted in [kernel-time-breakdown.md](kernel-time-breakdown.md).

**Conclusion: H2, H3, and H4 are ruled out.** `USE_CUDA_GRAPH` is compiled in (H4 false). The arch/compat gates don't bail (H3 false) — capture succeeds. And after the final `hipGraphInstantiate`, there are zero further `hipLaunchKernel` calls and no more `hipStreamBeginCapture`/`hipGraphExecUpdate` events through the rest of the 64-token run: no per-token property churn, no recapture loop (H2 false).

This reframes the doc's premise. The 16% decode idle measured in kernel-time-breakdown.md (untraced tg128 49.8 t/s, 17.3ms busy / 20.6ms wall) was **already measured with graphs engaged** — `GGML_HIP_GRAPHS` is on by default and this trace confirms it's live in production on `4781fb9`. There is no "flip the flag, get +15-19% tg" free lunch sitting on the table; that fix is already shipped. The literal H1 question survives but changes shape: is 16% idle the floor of hipGraph replay on ROCm 7.14 (inter-kernel dependency-wait stalls inside the replayed packet stream), or would idle be worse without graphs (i.e. are graphs already buying something, just not closing the whole gap)?

**Not yet done:** the H1 A/B itself (`GGML_HIP_GRAPHS=ON` vs `OFF` idle-fraction comparison from the original Step 3 plan). This needs a throwaway build with `-DGGML_HIP_GRAPHS=OFF` added to the cmake invocation in the deploy Dockerfile (under a distinct image tag) — no existing build-arg passthrough covers a cmake-flag override, unlike the SHA-only bisect path. Re-run the idle-fraction check with `--kernel-trace` only (no `--hip-trace`): this run's naive wall/busy query over the whole trace read 59% idle, inflated by `--hip-trace` instrumentation overhead versus the clean ~16% kernel-trace-only measurement in kernel-time-breakdown.md — don't reuse this run's idle number, only its graph-engagement evidence.
