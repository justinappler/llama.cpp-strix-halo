---
license: other
license_name: minimax-m2.7-non-commercial
license_link: LICENSE
library_name: mlx
tags:
  - mlx
  - jang
  - jangtq
  - jangtq-prestack
  - minimax
  - minimax-m2
  - moe
  - apple-silicon
  - 2bit
pipeline_tag: text-generation
base_model: MiniMaxAI/MiniMax-M2.7
base_model_relation: quantized
---

<p align="center"><img src="jangq-logo.png" width="160"/></p>

# MiniMax-M2.7-JANGTQ

**MiniMax M2.7 — 47 GB on disk** (down from the ~230 GB FP8 source) — 2-bit
**JANGTQ2** quantization in **JANGTQ-PRESTACK** layout (pre-stacked routed
experts on disk → instant cold load, no runtime cache sidecar).

- **Source:** [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI)
  (MiniMax M2 architecture, FP8 E4M3 block-128 native, 196K context, 62 layers,
  256 routed experts top-8)
- **Quantization:** JANGTQ2 — 2-bit MXTQ codebook (Hadamard-rotated, Lloyd-Max
  optimized) on routed-expert weights + 8-bit affine on attention / shared
  expert / embed / lm_head + fp16 passthrough on RMSNorms / router gate /
  `expert_bias`
- **Routed-expert layout:** **pre-stacked along axis 0**
  (`block_sparse_moe.switch_mlp.<proj>.tq_packed` shape `[256, out, packed_in]`)
  per the JANGTQ-PRESTACK STANDARD — no runtime restacking, no
  `jangtq_stacked.safetensors` sidecar
- **Bundle size:** **47 GB on-disk** across 51 shards
- **Runs on:** M3 Max 96 GB+ / M4 Max 128 GB / M5 Max 128 GB / Mac Studio

## What's new in this build (2026-05-04)

This bundle is shipped in **JANGTQ-PRESTACK** layout — the routed-expert
TurboQuant tensors are stacked along axis 0 directly in the main shards.
Wins vs the previous per-expert layout:

| Metric | Old (per-expert) | This (pre-stacked) |
|---|---|---|
| First-load time | ~5-10s restacking pass | **`mx.load()` direct (~14 s incl warmup)** |
| Decode tok/s | reference | **identical** (same MXTQ codec, same fused decode kernels) |
| Bundle size | ~57 GB | **~47 GB** (smaller by virtue of removing per-expert metadata duplication) |
| Loader path | streaming hydrate + per-expert restack | **generic loader's prestack branch** |

## What's in the bundle

| Module | Source dtype | Bundle dtype |
|---|---|---|
| Routed experts (256 × 3 mats × 62 layers, pre-stacked along axis 0) | FP8 E4M3 + F32 weight_scale_inv | **2-bit MXTQ** + sidecar codebook |
| Attention (q/k/v/o, q/k norms) | FP8 E4M3 / BF16 | 8-bit affine g=64 |
| `embed_tokens` / `lm_head` | BF16 | 8-bit affine g=64 |
| RMSNorm / router gate / `e_score_correction_bias` | BF16 / F32 | fp16 / fp32 passthrough |

`jangtq_runtime.safetensors` sidecar (~25 KB) for Swift runtimes — covers
`(in_features={1536, 3072}, seed=42, bits=2)` codebooks + sign-flip vectors.

## Loading (Python)

```bash
pip install jang-tools mlx-lm
```

```python
from jang_tools.load_jangtq import load_jangtq_model
model, tokenizer = load_jangtq_model("JANGQ-AI/MiniMax-M2.7-JANGTQ")
```

The loader detects the pre-stacked layout via
`jang_config.routed_expert_layout == "prestacked"` and routes through the
generic JANGTQ loader's prestack branch. Decode applies the standard
SwitchGLU fused gate+up + P15 router compile + P18 QKV fusion patches
automatically.

## Reasoning + tools

- **Reasoning parser:** `qwen3` (extracts `<think>...</think>` blocks)
- **Tool parser:** `minimax`
- **Default mode:** thinking ON (chat template opens `<think>` for the
  assistant); pass `enable_thinking=False` to skip reasoning
- **Cache:** `kv` (standard MLA-free MoE attention cache)

## Credits

- **Quantization + MLX runtime:** Jinho Jang ([eric@jangq.ai](mailto:eric@jangq.ai))
- **Base model:** MiniMaxAI — M2.7 architecture
