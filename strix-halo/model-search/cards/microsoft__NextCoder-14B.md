---
license: mit
language:
- en
base_model:
- Qwen/Qwen2.5-Coder-14B-Instruct
pipeline_tag: text-generation
library_name: transformers
tags:
- code
- chat
- microsoft
- nextcoder
- selekt
datasets:
- microsoft/NextCoderDataset
- microsoft/NextCoderDataset-Conversational
- bigcode/commitpackft
- bigcode/starcoderdata
---


# NextCoder-14B
<p align="center">
        <a href="https://github.com/microsoft/NextCoder">GitHub</a>&nbsp&nbsp | &nbsp&nbsp <a href="https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/">Paper</a> 
</p>

> NextCoder: Robust Adaptation of Code LMs to Diverse Code Edits (ICML'2025)

## Introduction

NextCoder is the latest series of Code-Editing large language models developed using the Qwen2.5-Coder Instruct variants as base and trained with novel Selective Knowledge Transfer finetuning methodology as introduced in the paper. NextCoder family model comes in 3 different sizes 7, 14, 32 billion parameters, to meet the needs of different developers.
Following are the key improvements:
- Significantly improvements in **code editing**, NextCoder-32B has performing on par with GPT-4o on complex benchmarks like Aider-Polyglot with performance increment of 44% from their base model.
- No loss of generalizibility, due to our new finetuning method **SeleKT**
- **Long-context Support** up to 32K tokens.

**This repo contains the NextCoder 14B model**, which has the following features:
- Type: Causal Language Models
- Training Stage: Post-training with SeleKT
- Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
- Number of Parameters: 14.7B
- Number of Paramaters (Non-Embedding): 13.1B
- Number of Layers: 48
- Number of Attention Heads (GQA): 40 for Q and 8 for KV
  
For more details, please refer to our [blog](), [GitHub](https://github.com/microsoft/NextCoder), [Paper](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/).

## Requirements

The code of NextCoder is based on Qwen2.5 base models which has been in the latest Hugging face `transformers` and we advise you to use the latest version of `transformers`.

With `transformers<4.37.0`, you will encounter the following error:
```
KeyError: 'qwen2'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/NextCoder-14B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
Fix the following function that divides two numbers to handle all the edge cases:

def divide(a, b)
  returm a/b
"""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
## Evaluation and Performance

| Models | HUMANEVALFIX | CANITEDIT | AIDER | POLYGLOT |
|--------|---------------|-----------|-------|----------|
| QwenCoder-2.5-3B | 73.2 | 37.1 | 36.8 | - |
| QwenCoder-2.5-3B-LoRA | 64.6 | 36.2 | 35.8 | - |
| QwenCoder-2.5-3B-SFT | 76.2 | 32.4 | 30.1 | - |
| **NextCoder-3B** | 75.6 | 42.4 | 37.6 | - |
| QwenCoder-2.5-7B | 73.8 | 48.1 | 59.4 | - |
| QwenCoder-2.5-7B-LoRA | 70.7 | 44.3 | 40.6 | - |
| QwenCoder-2.5-7B-SFT | 70.1 | 36.7 | 48.9 | - |
| **NextCoder-7B** | 81.1 | 50.5 | 65.7 | - |
| QwenCoder-2.5-14B | 87.8 | 58.1 | 66.9 | 9.3 |
| QwenCoder-2.5-14B-LoRA | 78.0 | 50.9 | 66.2 | 5.3 |
| QwenCoder-2.5-14B-SFT | 79.9 | 42.4 | 36.8 | 3.1 |
| **NextCoder-14B** | 89.8 | 60.2 | 72.2 | 12.2 |
| QwenCoder-2.5-32B | **90.2** | 61.0 | 72.9 | 16.4 |
| QwenCoder-2.5-32B-LoRA | 82.3 | 52.4 | 60.2 | 6.7 |
| QwenCoder-2.5-32B-SFT | 81.7 | 49.5 | 66.9 | 8.4 |
| **NextCoder-32B** | 88.9 | **62.4** | **74.7** | **23.6** |

*Comparison of base QwenCoder-2.5 models of different sizes and their SELEKT-enhanced versions across three code editing benchmarks.*

**Detailed evaluation results are reported in this [📑 paper](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/).**

## Responsible AI Use
The base models (from the QwenCoder-2.5 family) are suspectible to malicious prompts and may generate or execute harmful code. Our finetuning does not enhance or impede such behaviors. The users should use the models and their outputs responsibly and with caution. Model outputs should be subjected to additional analysis, including manual inspection, and sandboxing before execution.

## Citation

```bibtex
@inproceedings{aggarwal2025nextcoder,
author = {Aggarwal, Tushar and Singh, Swayam and Awasthi, Abhijeet and Kanade, Aditya and Natarajan, Nagarajan},
title = {NextCoder: Robust Adaptation of Code LMs to Diverse Code Edits},
booktitle = {International Conference on Machine Learning},
year = {2025},
url = {https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/},
}
```