---
base_model: meta-llama/Llama-3.3-70B-Instruct
tags:
  - exl3
  - exllamav3
  - quantized
  - text-generation
license: llama3.3
---

# Llama-3.3-70B-Instruct EXL3 6.0bpw h6

> EXL3 quantization of [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
> Quantized by [anthonyw448](https://huggingface.co/anthonyw448)

## Model Description

Meta's Llama 3.3 70B instruction-tuned model, optimized for multilingual dialogue. Outperforms many open-source and closed chat models on common benchmarks. Supports English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

## Quantization Details

| Parameter | Value |
|---|---|
| Format | EXL3 |
| Bits per weight | 6.0 bpw |
| Head bits | 6 |
| Tool | ExLlamaV3 latest (dev) |
| Calibration | Default ExLlamaV3 calibration dataset (250 rows, 2048 columns) |
| Date quantized | 2026-05-05 |

## Quality

**High quality — minimal difference from full precision**

EXL3 uses trellis/codebook quantization which provides better
quality than EXL2 at equivalent bitrates.

## Hardware Requirements

- **VRAM:** ~55 GB — 3x RTX 3090/4090
- **Context:** Up to 128K tokens supported

## Usage

Requires [ExLlamaV3](https://github.com/turboderp-org/exllamav3) or
[TabbyAPI](https://github.com/theroyallab/tabbyAPI) with EXL3 support.

```bash
hf download anthonyw448/Llama-3.3-70B-Instruct-EXL3-6.0bpw-h6 --local-dir ./Llama-3.3-70B-Instruct-EXL3-6.0bpw-h6
```

## Other Bitrates From This Author

| Bitrate | VRAM | Quality |
|---|---|---|
| **6.0bpw h6 (this repo)** | **~55 GB — 3x RTX 3090/4090** | **High quality — minimal difference from full precision** |

| [5.0bpw h6](https://huggingface.co/anthonyw448/Llama-3.3-70B-Instruct-EXL3-5.0bpw-h6) | ~46 GB — 2x RTX 3090/4090 | Great balance of size and quality |
| [7.0bpw h8](https://huggingface.co/anthonyw448/Llama-3.3-70B-Instruct-EXL3-7.0bpw-h8) | ~64 GB — 3x RTX 3090/4090 | Near-lossless — virtually indistinguishable from BF16 |

## Original Model

See [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
for full model details, license, and usage.

---
*Quantized 2026-05-05 by [anthonyw448](https://huggingface.co/anthonyw448)*
