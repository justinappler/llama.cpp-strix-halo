---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
---

# Rubicon

<p align="center">
    <a href="https://arxiv.org/abs/2508.12790"><b>📄 Paper</b></a> •
    <a href="https://huggingface.co/inclusionAI/Rubicon-Preview"><b>🤗 Model</b></a>
</p>

This is the model card for **Rubicon-preview**, a 30B-A3B parameter model trained with a novel reinforcement learning framework using "rubric anchors" to excel at open-ended, creative, and humanities-centric tasks.

---

## Highlights

We introduce **Rubicon**, a novel framework using rubric anchors for reinforcement learning. Our model, **Rubicon-preview**, demonstrates the following key highlights:

-   **Token-Efficient Performance**: Achieves a **+5.2%** absolute improvement on subjective, humanities-centric tasks with only **5K** training samples, outperforming a 671B DeepSeek-V3 model.
-   **Stylistic Controllability**: Leverages rubric anchors to precisely guide output style, producing responses that are more human-like, emotionally expressive, and less formulaic.
-   **Preservation of General Abilities**: Avoids performance degradation on general tasks—a common side effect of specialized RL—while delivering additional gains on reasoning benchmarks like AIME 2024 (+4.1%).

---

## Performance

Our rubric-based RL approach yields significant gains on open-ended, humanities-centric benchmarks while preserving and even enhancing performance on general and reasoning tasks.

### Humanities & Open-Ended Evaluation

Rubicon-preview achieves a **+5.21%** average absolute improvement over its base model on a diverse set of subjective benchmarks. Notably, it surpasses the much larger DeepSeek-V3-671B model by **+2.42%** on average.

| **Model** | **C.W** | **Writing** | **Judge** | **EQ** | **IFE** | **Collie** | **IFS** | **Avg** |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 77.82 | 75.65 | 56.20 | 73.35 | **83.55** | 35.77 | 54.68 | 65.29 |
| **Rubicon-preview** | **81.89** | **80.11** | **69.20** | **79.55** | 81.70 | 40.27 | 60.79 | **70.50** |
| *Δ Improvement* | <span style="color:green">↑4.07</span> | <span style="color:green">↑4.46</span> | <span style="color:green">↑13.00</span> | <span style="color:green">↑6.20</span> | <span style="color:red">↓1.85</span> | <span style="color:green">↑4.50</span> | <span style="color:green">↑6.11</span> | **<span style="color:green">↑5.21</span>** |
| DeepSeek-V3-671B | 80.10 | 74.08 | 61.30 | 75.60 | 81.89 | **42.69** | **60.92** | 68.08 |

### General & Reasoning Abilities

The model maintains its core capabilities without degradation. It shows notable improvements on math reasoning benchmarks like AIME and enhances performance across several general benchmarks.

**Reasoning**
| **Model** | **AIME24** | **AIME25** | **Math500** | **GPQA-D** | **LCBv5** | **Avg** |
|:---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 77.50 | 70.00 | **94.75** | **63.00** | **63.77** | **73.80** |
| **Rubicon-preview** | **81.67** | **70.83** | 94.55 | 60.35 | 59.43 | 73.37 |

**General**
| **Model** | **MMLU** | **IQ-EQ** | **HS** | **SC** | **CQ** | **SIQA** | **Avg** |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3--30B-A3B | 79.53 | 68.75 | 77.55 | 77.72 | 79.52 | 73.64 | 78.16 |
| **Rubicon-preview** | **79.83** | **75.00** | **77.75** | **78.17** | **80.70** | **75.79** | **78.85** |

---

## Usage

Below are code snippets to get quickly started with running the model.

### Installation

First, install the necessary libraries. We recommend a recent version of Transformers.

```sh
pip install transformers torch
```

### Quick Start with Python

You can use the model for text generation with just a few lines of code.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "inclusionAI/Rubicon-Preview"

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # or "auto"
    device_map="auto"
)

# Prepare the model input using the chat template
prompt = "Is there true love in this world?"
messages = [
    {"role": "user", "content": prompt}
]

# Apply the chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("Generated Response:\n", content)
```

---

## Citation

If you use Rubicon in your research, please cite our paper:

```bibtex
@article{Rubicon,
    title = {Reinforcement Learning with Rubric Anchors},
    author = {Huang, Zenan and Zhuang, Yihong and Lu, Guoshan and Qin, Zeyu and Xu, Haokai and Zhao, Tianyu and Peng, Ru and Hu, Jiaqi and Shen, Zhanming and Hu, Xiaomeng and Gu, Xijun and Tu, Peiyi and Liu, Jiaxin and Chen, Wenyu and Fu, Yuzhuo and Fan, Zhiting and Gu, Yanmei and Wang, Yuanyuan and Yang, Zhengkai and Li, Jianguo and Zhao, Junbo},
    journal = {arXiv preprint arXiv:2508.12790},
    year = {2025}
}
```
