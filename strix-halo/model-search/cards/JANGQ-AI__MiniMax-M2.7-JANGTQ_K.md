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
  - jangtq-k
  - mixed-precision
  - minimax
  - minimax-m2
  - moe
  - apple-silicon
pipeline_tag: text-generation
base_model: MiniMaxAI/MiniMax-M2.7
base_model_relation: quantized
---

<p align="center"><img src="jangq-logo.png" width="160"/></p>

# MiniMax-M2.7-JANGTQ_K

**MiniMax M2.7 — 74 GB on disk** (down from ~230 GB FP8 source) — **mixed-bit
JANGTQ_K** quantization in JANGTQ-PRESTACK layout.

- **Source:** [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI)
  (62 layers, 256 routed experts top-8, 196K context)
- **Quantization:** **mixed-bit MXTQ** on routed experts:
  - `down_proj`: **4-bit** (output enters residual stream, more sensitive)
  - `gate_proj`: **2-bit** (gated activation, less sensitive)
  - `up_proj`: **2-bit** (gated activation)
  - attention / shared expert / embed / lm_head: 8-bit affine
  - norms / router gate / expert_bias: fp16 / fp32 passthrough
- **Routed-expert layout:** **pre-stacked along axis 0** per the
  JANGTQ-PRESTACK STANDARD — instant cold load, no runtime sidecar.
- **Bundle size:** **~74 GB on-disk** (~3-bit avg routed)
- **Runs on:** M3 Max 96 GB+ / M4 Max 128 GB / M5 Max 128 GB / Mac Studio

## Why mixed-bit?

`down_proj`'s output enters the residual stream and accumulates across
62 layers — quantization noise compounds. `gate_proj` and `up_proj`
enter through SwiGLU's multiplicative gate (`silu(gate) × up`) which
dampens noise. Spending 4 bits on `down` and 2 bits on `gate`/`up` gives
quality close to full-4-bit (~115 GB) at **64% the size**.

## Variants in the MiniMax-M2.7 line

| Variant | Routed bits (avg) | Bundle size | Use case |
|---|---|---|---|
| `MiniMax-M2.7-JANGTQ` | 2-bit | 47 GB | smallest, best for tight RAM |
| **`MiniMax-M2.7-JANGTQ_K` (this)** | **~3-bit (mixed 2/4)** | **74 GB** | **quality close to 4-bit at 2-bit-ish size** |

## Loading

```bash
pip install jang-tools mlx-lm
```

```python
from jang_tools.load_jangtq import load_jangtq_model
model, tokenizer = load_jangtq_model("JANGQ-AI/MiniMax-M2.7-JANGTQ_K")
```

## Reasoning + tools

- **Default:** thinking ON (chat template inserts `<think>\n` after assistant prefix)
- **Disable reasoning:**
  ```python
  messages = [{"role": "user", "content": "..."}]
  inp = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
  ```
- **Reasoning parser:** `qwen3` (extracts `<think>...</think>` blocks)
- **Tool parser:** `minimax`

The chat template ships with the `enable_thinking` switch correctly wired
both as a standalone `chat_template.jinja` AND inlined into
`tokenizer_config.json["chat_template"]` for engines that read inline
(vMLX, Swift swift-transformers).

## Credits

- **Quantization + MLX runtime:** Jinho Jang ([eric@jangq.ai](mailto:eric@jangq.ai))
- **Base model:** MiniMaxAI — M2.7 architecture
