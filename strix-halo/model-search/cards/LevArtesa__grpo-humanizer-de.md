---
base_model: Qwen/Qwen3-8B
language: de
library_name: transformers
license: apache-2.0
tags:
- grpo
- humanizer
- lora
- german
- academic
---

# GRPO Humanizer DE

Fine-tuned with **Group Relative Policy Optimization (GRPO)** to
rewrite AI-generated German academic text so that it passes GPTZero
detection while preserving semantic content.

## Training details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-8B |
| Method | GRPO (TRL) + LoRA |
| Learning rate | 5e-06 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Max steps | 50 |
| Precision | bf16 |

## Intended use

Academic text humanisation for German-language content.  The model is
designed to be called via the HuggingFace Inference API from the
GhostWriter application.

## Licence

Apache-2.0
