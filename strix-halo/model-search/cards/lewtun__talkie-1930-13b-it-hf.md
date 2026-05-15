---
library_name: transformers
language:
- en
base_model:
- talkie-lm/talkie-1930-13b-it
license: apache-2.0
tags:
- talkie
- vintage
- historical
- conversational
---

# talkie-1930-13b-it (Transformers format)

This is a conversion of [talkie-lm/talkie-1930-13b-it](https://huggingface.co/talkie-lm/talkie-1930-13b-it) to the HuggingFace Transformers format. The original model was distributed as a raw PyTorch checkpoint with a custom inference library; this version can be loaded directly with `AutoModelForCausalLM` and `AutoTokenizer`.

The weights are **numerically identical** to the original — top-5 decoded tokens match across all test prompts, with max logit differences below 0.07 (bf16 rounding).

> [!NOTE]
> This model was converted automatically by Hugging Face's [ML Intern](https://github.com/huggingface/ml-intern) — an AI agent for ML engineering tasks. Try it yourself via the [CLI](https://github.com/huggingface/ml-intern) or the [Demo](https://huggingface.co/spaces/smolagents/ml-intern).

## Table of Contents

1. [Model Summary](#model-summary)
2. [How to Use](#how-to-use)
3. [Architecture Details](#architecture-details)
4. [Conversion Notes](#conversion-notes)
5. [License](#license)

## Model Summary

**talkie-1930-13b-it** is a 13B-parameter instruction-tuned language model from the [talkie](https://github.com/talkie-lm/talkie) family, developed by Alec Radford, Nick Levine, and David Duvenaud. It was pretrained on **260B tokens of pre-1931 English-language text** and instruction-tuned using a novel dataset extracted from vintage reference works — etiquette manuals, encyclopedias, letter-writing guides, and poetry collections. The model underwent reinforcement learning via online DPO with an LLM-as-a-judge to improve instruction following.

Read more in the [talkie report](https://talkie-lm.com/).

### Key Features
- **Vintage knowledge**: trained exclusively on pre-1931 text, offering a unique window into early 20th-century language and thought
- **Instruction-tuned**: fine-tuned for conversational use with a simple chat template
- **13B parameters** in bfloat16 (~26 GB VRAM)
- **2048 token context window**

## How to Use

### Installation

This model uses custom modeling code. Make sure you have a recent version of `transformers` installed:

```bash
pip install -U transformers torch
```

### Basic Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lewtun/talkie-1930-13b-it-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype="bfloat16",
).to("cuda")

prompt = "Write an essay predicting what life will be like in the year 1960."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

### Multi-turn Chat

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lewtun/talkie-1930-13b-it-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype="bfloat16",
).to("cuda")

messages = [
    {"role": "user", "content": "What were the causes of the French Revolution?"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
reply = tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(reply)

# Continue the conversation
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "Which of those causes was the most significant?"})

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True))
```

### Chat Template

The model uses the following chat format:

```
<|system|>{system_message}<|end|><|user|>{user_message}<|end|><|assistant|>{assistant_message}<|end|>
```

This is applied automatically when using `tokenizer.apply_chat_template()`.

## Architecture Details

talkie is a 40-layer decoder-only GPT with several distinctive architectural choices:

| Component | Details |
|-----------|---------|
| Parameters | 13B |
| Layers | 40 |
| Attention heads | 40 (MHA, no GQA) |
| Hidden size | 5120 |
| Head dimension | 128 |
| Intermediate size (MLP) | 13696 |
| Position encoding | RoPE (θ = 1,000,000) |
| Activation | SwiGLU |
| Normalization | RMSNorm (pre-norm) |
| Context length | 2048 |
| Vocabulary | 65,540 (65,535 BPE + 5 special tokens) |
| Precision | bfloat16 |

**Notable architectural features:**
- **QK-normalization**: RMSNorm is applied to queries and keys after RoPE
- **Per-head gain**: learnable scalar gain per attention head, applied to queries
- **Embedding skip connections**: each transformer block receives a residual connection from the (normalized) input embeddings
- **Activation gains**: learnable scalar gains on attention and MLP residual streams (initialized to (2·L)^(-0.5))
- **lm_head weight gain**: a learnable scalar applied to the output projection weights

## Conversion Notes

This model was converted from the original [talkie-lm/talkie-1930-13b-it](https://huggingface.co/talkie-lm/talkie-1930-13b-it) PyTorch checkpoint using the reference [talkie codebase](https://github.com/talkie-lm/talkie) as ground truth. The conversion involved:

1. **Model weights**: the `.pt` state dict was remapped to a `PreTrainedModel` subclass (`TalkieForCausalLM`) and saved as safetensors
2. **Tokenizer**: the tiktoken BPE vocabulary was converted to a `PreTrainedTokenizerFast` with the HuggingFace `TikTokenConverter`, including all 5 special tokens (`<|endoftext|>`, `<|end|>`, `<|user|>`, `<|assistant|>`, `<|system|>`)
3. **Validation**: logits were compared on 4 test prompts covering chat, system prompts, and raw completion — all top-5 decoded tokens match exactly, with cosine similarity ≥ 0.99999994

Since this is a custom architecture, loading requires `trust_remote_code=True`.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) — same as the [original model](https://huggingface.co/talkie-lm/talkie-1930-13b-it).
