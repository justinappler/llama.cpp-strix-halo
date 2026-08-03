# Qwen 3.6 MTP runtime check on gfx1151

## Status

**Runtime works, but not a PP win.** After rebasing this fork onto upstream with
MTP support, the lab installed Unsloth's MTP GGUF:

- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`
- quant: `UD-Q4_K_XL`

Unsloth's current llama.cpp guidance is to enable MTP with:

```bash
--spec-type draft-mtp --spec-draft-n-max 2
```

They also note that llama.cpp MTP currently does not support `-np > 1` or
`--mmproj`, so this is a separate text-only serving shape from the production
`qwen3.6` multimodal preset.

## Why this is separate from PP tuning

MTP is speculative decoding: the MTP head proposes future tokens and the main
model verifies them. That can reduce generation wall time, but it does not make
the initial prompt prefill cheaper. On this fork it appears to add prompt-side
overhead in the CLI path because target pre-norm hidden states must be extracted
for the draft context.

For the current goal, Qwen 3.6 prompt processing on Strix Halo, MTP should be
treated as a decode/serving option rather than a PP optimization.

## Runtime shape

One-shot CLI checks on build `3511e7d`, TheRock `7.13.0a20260514`, gfx1151:

```bash
docker run --rm --entrypoint=/app/llama-cli \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host \
  -v <models-dir>:/models:ro \
  llamacpp-server:local \
    -m /models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -ctk f16 -ctv f16 \
    -fa 1 \
    -b 4096 -ub 2048 -ngl 999 --no-mmap \
    -c 4096 -n 128 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 \
    --no-warmup --show-timings --no-display-prompt --single-turn \
    -f /tmp/qwen_mtp_prompt.txt \
    --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0
```

These CLI timings include chat/template/runtime behavior and should not be
mixed directly with `llama-bench` `pp512` numbers. They are useful for A/B-ing
MTP overhead and draft-count choice.

## Short generation A/B

Small prompt, `-n 128`:

| case | prompt t/s | generation t/s | wall seconds |
|---|---:|---:|---:|
| production GGUF, no MTP | 336.8 | 44.1 | 9.58 |
| MTP GGUF, no speculation | 374.2 | 48.7 | 6.10 |
| MTP GGUF, `draft-mtp`, `n_max=2` | 169.3 | 73.1 | 5.70 |

MTP improves generation by about 50% versus the MTP GGUF without speculation,
but prompt throughput drops sharply in this short-prompt CLI path.

## Prompt-heavy check

Synthetic prompt-heavy CLI check, MTP GGUF, `-n 1`:

| case | prompt t/s | generation t/s | wall seconds |
|---|---:|---:|---:|
| no speculation | 1297.6 | 1000000.0 | 4.51 |
| `draft-mtp`, `n_max=2` | 952.7 | 1000000.0 | 4.66 |

The generation number is meaningless here because only one token is requested.
The prompt number is the relevant signal: enabling MTP costs about 27% prompt
throughput in this CLI path.

## Draft-count sweep

Small prompt, `-n 128`, MTP GGUF:

| `spec-draft-n-max` | prompt t/s | generation t/s | wall seconds |
|---:|---:|---:|---:|
| 1 | 157.9 | 65.6 | 6.23 |
| 2 | 169.3 | 73.1 | 5.70 |
| 3 | 166.0 | 74.4 | 5.72 |
| 4 | 157.3 | 65.8 | 5.98 |
| 6 | 157.7 | 64.5 | 5.96 |

`n_max=2` and `n_max=3` are the useful region on this host; Unsloth's default
`2` is a good conservative serving choice. Higher values lose enough acceptance
or add enough overhead to erase the decode win.

## Takeaway

Keep MTP available as a separate text-only serving preset if decode-heavy usage
matters. Do not expect it to improve Qwen 3.6 PP. For PP, the better next knobs
remain the D=256 TILE FA shape, batch/ubatch interaction, and small HIP routing
thresholds such as MMF.
