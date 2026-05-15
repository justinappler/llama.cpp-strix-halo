---
base_model: unsloth/qwen3-8b-unsloth-bnb-4bit
tags:
- text-generation-inference
- transformers
- unsloth
- qwen3
- trl
- qlora
- reasoning
- code
- hyperthinkcode
license: apache-2.0
language:
- en
datasets:
- Sashvat/HyperThink-X-Nvidia-Opencode-Reasoning-200K
metrics:
- humaneval
- gsm8k
library_name: adapter
pipeline_tag: text-generation
---

# HyperThinkCode-Qwen3-8B-v1

HyperThinkCode-Qwen3-8B-v1 is a LoRA fine-tune of the Qwen3-8B base model.

---

## 🛠 Experimental Setup

- Base model: Qwen3-8B  
- Hardware: dual Tesla T4 (16GB VRAM each)  
- 4-bit QLoRA with rank = 16 and alpha = 16  
- All linear layers:  
  - Attention: q, k, v, o  
  - MLP: gate, up, down  
- Training time: ~1 hour 17 minutes  
- Total steps: 50  

---

## 🧠 Dataset & Objective

Training on a specific 30k subset of the  
**Sashvat/HyperThink-X-Nvidia-Opencode-Reasoning-200K** dataset.

- Uses chat template with assistant response in the *thinking* field  
- Objective: encourage *thinking over direct response*  
- Sequence length limited to 4096 tokens (for code complexity + VRAM constraints)  

---

## 📉 Training Logs

With only 50 steps, the loss shows expected variance given model + dataset complexity.

| Step | Training Loss |
|------|--------------|
| 10   | 0.8177       |
| 25   | 0.7358       |
| 50   | 0.6785       |

- Global batch size: 8 (1 device × 8 gradient steps)

---

## 📊 Evaluation (Ongoing)

Currently running benchmarks using the **lm-eval** library:

- HumanEval (Coding)
- GSM8K (Math)

Comparisons are being made against the base model.

---

## 🔁 Reproduction

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Andy-ML-And-AI/HyperThinkCode-Qwen3-8B-v1",
    max_seq_length = 4096,
    load_in_4bit = True,
)