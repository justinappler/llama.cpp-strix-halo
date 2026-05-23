# RDNA3.5 PP follow-up: dense-aware MMQ + TILE FA D=256

## Status

**Pending bench.** This is a narrow port of the final shape of upstream PR
[#21344](https://github.com/ggml-org/llama.cpp/pull/21344) after the fork's
2026-05-22 upstream rebase.

## Hypothesis

The current fork carries the early gfx1151 MMQ tuning from PR #21344:
`mmq_x_max=48`, `mmq_y=64`, and `nwarps=4`. That helped Qwen 3.6 prompt
processing, but the final PR discussion split the `mmq_x` cap by workload:
dense/projection GEMMs can use `mmq_x=128`, while MoE expert dispatch keeps the
lower `48` cap that avoids the RDNA3.5 VGPR cliff.

The same final PR also added one RDNA3.5 TILE flash-attention override for the
Qwen 3.6 hot path:

```cpp
GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3, 64, 64)
```

This changes only `DKQ=256`, `DV=256`, `ncols=32` on RDNA3.5, dropping
`nbatch_K` from the generic RDNA value of `128` to `64`. That directly targets
the D=256 TILE FA path used by Qwen 3.6 and Qwen3-Coder-Next on gfx1151.

## Patch

- `ggml/src/ggml-cuda/mmq.cuh`: allow RDNA3.5 dense MMQ paths to instantiate
  and select `mmq_x <= 128`; cap back to `48` only when `args.expert_bounds`
  indicates MoE expert dispatch.
- `ggml/src/ggml-cuda/fattn-tile.cuh`: add an RDNA3.5 config table wrapper with
  the single D=256/ncols=32 override, falling through to the generic RDNA table
  for every other shape.

## Bench Plan

Use the canonical Qwen 3.6 bench from `server-configs/services/llamacpp`:

```bash
docker run --rm --entrypoint=/app/llama-bench \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host \
  -v /srv/models:/models:ro \
  llamacpp-server:local \
    -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -ctk f16 -ctv f16 \
    -fa 1 \
    -b 4096 -ub 2048 -ngl 999 -mmp 0 \
    -p 512 -n 128 -r 3 \
    -d 0,2048,8192,16384
```

Decision rule: keep if pp512 improves outside the host's ~2% noise floor at
any depth without regressing pp512 or tg128 at depth. Revert if d=16k regresses.
