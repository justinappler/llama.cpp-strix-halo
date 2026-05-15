---
language:
- en
- zh
library_name: mlx
license: mit
pipeline_tag: text-generation
base_model: MiniMaxAI/MiniMax-M2.7
base_model_relation: quantized
tags:
- mlx
- jang
- jangtq
- minimax
- minimax_m2
- moe
- apple-silicon
- 2bit
- turboquant
---
> ## ⚠️ REQUIRED — `jangtq_runtime.safetensors` sidecar must be downloaded
>
> Osaurus uses the native Swift JANGTQ runtime. **Every JANGTQ bundle on
> OsaurusAI ships a small `jangtq_runtime.safetensors` sidecar (~10 KB–~165 KB)
> alongside the weight shards.** The Swift loader will refuse to start with
> the error
> ```
> Error: Model '<name>' declares JANGTQ (weight_format: "mxtq") but is
>        missing required sidecar file 'jangtq_runtime.safetensors'.
>        Re-download the full model or obtain the sidecar from the original
>        publisher.
> ```
> if the file is absent.
>
> If your local copy doesn't have it (older download, partial sync, etc):
> ```bash
> hf download OsaurusAI/MiniMax-M2.7-JANGTQ jangtq_runtime.safetensors --local-dir <your-dir>
> ```
> The file holds the deterministic codebooks + Hadamard rotation signs the
> Swift loader uses to decode `*.tq_packed` weights. It must match the seed
> the bundle was quantized with (`mxtq_seed=42`).


<p align="center">
  <a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a>
</p>

<h3 align="center">MiniMax M2.7 &mdash; JANGTQ (MLX)</h3>
<p align="center">TurboQuant codebook quantization of MiniMax's 228B agentic MoE &mdash; routed experts at 2-bit via Lloyd-Max codebooks + Hadamard rotation, attention / embed / shared-expert / lm_head at 8-bit affine.</p>

<p align="center">
  <a href="https://osaurus.ai"><img src="https://img.shields.io/badge/Web-osaurus.ai-blue" alt="Website"></a>&nbsp;
  <a href="https://huggingface.co/OsaurusAI"><img src="https://img.shields.io/badge/HF-OsaurusAI-yellow?logo=huggingface" alt="OsaurusAI"></a>
</p>

---

## Model Details

| Property | Value |
|---|---|
| **Base Model** | MiniMaxAI/MiniMax-M2.7 |
| **Architecture** | MoE (256 experts, top-8 active) + standard Q/K/V attention + partial RoPE |
| **Total Parameters** | 228.7 B |
| **Active per Token** | ~1.4 B |
| **Profile** | JANGTQ |
| **Format** | JANGTQ (codebook + Hadamard) — `weight_format: mxtq` in `jang_config.json` |
| **Avg bits/param** | ~2.15 |
| **Disk** | ~57 GB |
| **Context length** | 192 K tokens |
| **Chat template** | Always-reasoning (`<think>` opened at assistant start) |

## What is JANGTQ?

**JANGTQ** (JANG TurboQuant) is a codebook-based quantization format for MoE
models on Apple Silicon. Routed expert weights stay in a compact **codebook +
Hadamard-rotated** form at runtime — no decompression to affine — and the
matmul path uses custom Metal kernels that read packed `uint32` weights, look
up centroids in a small codebook, and accumulate dot products against a
Hadamard-rotated input (QuIP# *rotate-input-once* math).

**Result vs uniform 2-bit affine:** smaller on disk, higher quality, runs at
~89 % of affine 2-bit speed.

## Bit Allocation

| Component | Bits | Format |
|---|:---:|---|
| Routed expert MLP (gate / up / down) | **2** | JANGTQ codebook + Hadamard |
| Attention (Q / K / V / O) | 8 | Affine (`nn.QuantizedLinear`, group_size=64) |
| Shared expert | 8 | Affine |
| Embed tokens / LM head | 8 | Affine |
| Router gate | fp16 | Unquantized `nn.Linear` |
| RMSNorms / RoPE / biases | fp16 | Unquantized |

The routed experts are 98 % of parameters and the natural compression target.
Everything else stays at 8-bit affine so the quality-critical hot path runs
at full precision.

## Important Settings

MiniMax M2.7 is an **always-reasoning** model. The chat template
unconditionally opens `<think>` at each assistant turn.

| Setting | Value | Notes |
|---|---|---|
| Temperature | **1.0** | Required — `temp=0` can cause thinking loops |
| Top-P | 0.95 | |
| Top-K | 40 | |
| Repetition Penalty | 1.1 | Optional, helps prevent loops |
| `max_tokens` | ≥ 8192 | Give reasoning room to converge |

Strip `<think>…</think>` from the response before using the final answer.

## Usage

This model requires the `jang-tools` loader — stock `mlx_lm.load()` does not
recognize `weight_format: mxtq`. The loader applies Metal kernel
monkey-patches at load time (fused gate+up+SwiGLU, gather TQ, multi-block
Hadamard, router compile, QKV fusion).

```bash
pip install jang-tools
```

```python
from huggingface_hub import snapshot_download
from jang_tools.load_jangtq import load_jangtq_model
from mlx_lm import generate

model_path = snapshot_download("OsaurusAI/MiniMax-M2.7-JANGTQ")
model, tokenizer = load_jangtq_model(model_path)

messages = [{"role": "user", "content": "Explain photosynthesis in five sentences."}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
out = generate(model, tokenizer, prompt, max_tokens=600,
               temperature=1.0, verbose=True)
```

### Swift — Osaurus / MLX Studio

Both clients auto-detect the JANGTQ runtime from `jang_config.json` and route
through the `MiniMaxJANGTQModel` class. Just load the repo — no extra flags.

## What's In This Repo

| File | Role |
|---|---|
| `model-*.safetensors` (61 shards, ~57 GB) | Weights — 2-bit routed TQ + 8-bit affine |
| `model.safetensors.index.json` | Shard index |
| `jangtq_runtime.safetensors` | Codebooks + Hadamard signs sidecar (Swift loader) |
| `jang_config.json` | JANG metadata + Tier-1 `capabilities` stamp (`reasoning=qwen3`, `tool=minimax`) |
| `config.json` | HF model config (`minimax_m2`, `weight_format=mxtq`, `mxtq_bits=2`) |
| `chat_template.jinja`, `tokenizer.*`, `vocab.json`, `merges.txt` | Tokenizer + chat template |
| `configuration_minimax_m2.py`, `modeling_minimax_m2.py` | HF custom code (untouched from upstream) |
| `osaurus-x-banner.png`, `mlx-studio-logo.png` | Branding assets |

## Parser Capabilities (Tier-1 auto-detected by Osaurus / vmlx)

```json
{
  "reasoning_parser": "qwen3",
  "tool_parser": "minimax",
  "think_in_template": true,
  "supports_tools": true,
  "supports_thinking": true,
  "family": "minimax_m2",
  "modality": "text",
  "cache_type": "kv"
}
```

`<think>` and `<tool_call>` are non-special tokens by design — the
application layer parses them. Osaurus and `vmlx` `CapabilityDetector` read
this block verbatim and wire the `qwen3` reasoning parser + `minimax` tool
parser automatically, so streamed responses route `reasoning_content` and
`tool_calls` into the OpenAI-compatible SSE fields instead of leaking into
`content`.

## License

MIT — see [`LICENSE`](./LICENSE).

## Credits

Created by [Jinho Jang](https://twitter.com/jangq_ai) — `eric@jangq.ai`

Based on MiniMaxAI's MiniMax M2.7. JANGTQ quantization © JANGQ-AI.
