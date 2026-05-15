---
license: apache-2.0
language:
  - en
  - no
  - sv
  - da
  - is
base_model: Qwen/Qwen3-30B-A3B-Instruct-2507
tags:
  - european
  - nordic
  - norwegian
  - swedish
  - danish
  - icelandic
  - multilingual
  - moe
  - qwen3
library_name: transformers
pipeline_tag: text-generation
---

# Bineric Lynx Instruct 30B

**A European large language model with exceptional Nordic language performance.**

| | |
|---|---|
| **Parameters** | 30B total, ~3B active |
| **Architecture** | Qwen3 MoE (128 experts, 8 active) |
| **Context Length** | 262K tokens |
| **Base Model** | [Qwen3-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| **Languages** | Norwegian (Bokmål/Nynorsk), Swedish, Danish, Icelandic + 100+ via base |

## About Bineric

[Bineric](https://bineric.com) is an AI company based in Oslo, Norway, built from a European perspective. We started Bineric to make AI usable for organizations that care about governance, language, and where their systems and data actually live.

Lynx is our flagship model — designed to serve European users with strong multilingual support and exceptional Nordic language performance.

## Overview

Lynx is built on Qwen3-30B's efficient Mixture-of-Experts architecture. It retains strong multilingual capabilities across **100+ languages including all major European languages**, while being specifically fine-tuned and rigorously evaluated on Nordic languages (Norwegian, Swedish, Danish, Icelandic) where it demonstrates exceptional results.

**Key features:**
- Strong European language support inherited from Qwen3 base model
- Fine-tuned and optimized for Nordic language understanding and generation
- Efficient MoE architecture: only 3B parameters active per token
- Available in 8-bit and 4-bit quantized variants for flexible deployment
- 262K context window for long-document processing

## Try Lynx

Lynx is available through multiple channels:

| Access Method | Link | Best For |
|--------------|------|----------|
| **Chatbot** | [chat.bineric.com](https://chat.bineric.com) | Interactive conversations, quick testing |
| **API** | [bineric.com/platform](https://bineric.com/platform) | Production integrations, programmatic access |
| **Hugging Face** | This repository | Self-hosting, fine-tuning, research |

## Evaluation Results

Evaluated using [EuroEval](https://github.com/ScandEval/EuroEval) benchmark framework (March 2026).

> **Note:** While Lynx supports all European languages via its Qwen3 base, we have rigorously evaluated performance on Nordic languages. Benchmarks for additional European languages coming soon.

### Nordic Language Performance

| Language | Overall Score | Best Task | Score |
|----------|---------------|-----------|-------|
| Danish | **79.3%** | Citizen Tests (Knowledge) | 79.3% |
| Swedish | **76.9%** | European Values | 76.9% |
| Norwegian | **71.0%** | NER Nynorsk | 71.0% |
| Icelandic | **65.1%** | Summarization | 65.1% |

![Language Performance Comparison](assets/language_comparison.svg)

### Task Performance by Language

#### Norwegian (8-bit)

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| Sentiment | NoReC | MCC | 51.0% |
| NER (Bokmål) | NorNE-nb | F1 | 65.7% |
| NER (Nynorsk) | NorNE-nn | F1 | 71.0% |
| Reading Comprehension | NorQuAD | F1 | 61.2% |
| Summarization | NoSammendrag | BERTScore | 63.4% |
| Common Sense | NorCommonSenseQA | MCC | 69.3% |
| Knowledge | NRK Quiz QA | MCC | 35.3% |

#### Danish (8-bit)

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| Sentiment | AngryTweets | MCC | 54.8% |
| NER | DANSK | F1 | 53.8% |
| Reading Comprehension | MultiWikiQA-da | F1 | **72.2%** |
| Summarization | Nordjylland News | BERTScore | 65.2% |
| Common Sense | HellaSwag-da | MCC | 67.7% |
| Knowledge | Danish Citizen Tests | MCC | **79.3%** |
| Idioms | Danske Talemåder | MCC | 64.9% |

#### Swedish (8-bit)

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| Sentiment | SweReC | MCC | 34.5% |
| NER | SUC3 | F1 | 65.0% |
| Reading Comprehension | MultiWikiQA-sv | F1 | **72.4%** |
| Summarization | SweDN | BERTScore | 65.9% |
| Common Sense | HellaSwag-sv | MCC | 58.3% |
| Knowledge | MMLU-sv | MCC | 53.9% |
| European Values | VaLEU-sv | MCC | **76.9%** |

#### Icelandic (8-bit)

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| NER | MIM-GOLD-NER | F1 | 63.6% |
| Reading Comprehension | NQiI | F1 | 58.6% |
| Summarization | RRN | BERTScore | **65.1%** |
| Knowledge | Icelandic Knowledge | MCC | 28.2% |
| Common Sense | Winogrande-is | MCC | 9.7% |

![Task Performance by Language](assets/task_performance.svg)

### Quantization Comparison (Norwegian)

8-bit quantization consistently outperforms 4-bit by ~2% on average.

| Task | 4-bit | 8-bit | Delta |
|------|-------|-------|-------|
| Sentiment (NoReC) | 49.7% | 51.0% | +1.3% |
| NER Bokmål | 65.1% | 65.7% | +0.6% |
| NER Nynorsk | 69.9% | 71.0% | +1.1% |
| Reading Comp | 58.9% | 61.2% | +2.3% |
| Summarization | 63.1% | 63.4% | +0.3% |
| Common Sense | 68.5% | 69.3% | +0.8% |
| Linguistic Accept. | 29.8% | 36.4% | **+6.6%** |

![8-bit vs 4-bit Quantization](assets/quantization_comparison.svg)

## Strengths & Limitations

### Strengths

- **Named Entity Recognition**: Consistently strong across all languages (63-71% F1)
- **Reading Comprehension**: Excellent for Danish and Swedish (72%+)
- **Knowledge Tasks**: Outstanding on Danish Citizen Tests (79.3%)
- **Summarization**: Stable 63-66% BERTScore across all languages

### Limitations

- **Linguistic Acceptability**: Grammatical judgment tasks are weak (10-36% MCC)
- **Icelandic Common Sense**: Winogrande-is performance is low (9.7%)
- **Norwegian Idioms**: Room for improvement (17-19% MCC)

## Quantization Options

| Variant | Size | Quality | Use Case |
|---------|------|---------|----------|
| **bfloat16** | ~60GB | Best | Research, high-end GPUs |
| **8-bit** | ~30GB | ~1-2% loss | Production (A10/L4 GPU) |
| **4-bit** | ~16GB | ~3-5% loss | Cost-optimized (T4 GPU) |

## Usage

### Basic Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "bineric/lynx-instruct-30b",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("bineric/lynx-instruct-30b")

messages = [
    {"role": "user", "content": "Hva er hovedstaden i Norge?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Thinking Mode

Lynx supports extended thinking for complex reasoning tasks:

```python
messages = [
    {"role": "user", "content": "Forklar forskjellen mellom bokmål og nynorsk."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Enable reasoning mode
)
```

### vLLM Deployment

```bash
vllm serve bineric/lynx-instruct-30b \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --quantization awq  # For 4-bit
```

## Model Architecture

```
Qwen3 MoE Architecture
├── Total Parameters: 30.5B
├── Active Parameters: ~3B per token
├── Hidden Layers: 48
├── Hidden Size: 2048
├── Attention Heads: 32
├── KV Heads: 4 (Grouped Query Attention)
├── Experts: 128 total
├── Active Experts: 8 per token
├── Vocab Size: 151,936
└── Context Length: 262,144 tokens
```

## Training

Lynx is fine-tuned from [Qwen3-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) with additional training on Nordic language data to improve performance on Norwegian, Swedish, Danish, and Icelandic tasks.

## Citation

```bibtex
@misc{bineric2026lynx,
  title={Bineric Lynx: A European Large Language Model},
  author={Bineric AI},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/bineric/lynx-instruct-30b}
}
```

## Links

- [Base Model: Qwen3-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

---

*Built with care in Oslo by [Bineric](https://bineric.com)*
