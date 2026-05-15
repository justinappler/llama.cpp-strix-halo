---
license: apache-2.0
library_name: transformers
tags:
- dllm
- diffusion
- llm
- text_generation
---
# LLaDA2.0-mini

**LLaDA2.0-mini** is a diffusion language model featuring a 16BA1B Mixture-of-Experts (MoE) architecture. As an enhanced, instruction-tuned iteration of the LLaDA series, it is optimized for practical applications.

<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*uOo8QKQMiBwAAAAAgNAAAAgAemJ7AQ/original" width="800" />
</div>


---

| Benchmark | Qwen3-8B (no thinking) | Ling-mini-2.0 | LLaDA2.0-mini-preview | LLaDA2.0-mini |
| :---: | :---: | :---: | :---: | :---: |
| **Average** | 70.19 | 72.13 | 61.75 | 71.67 |
| **Knowledge** | | | | |
| MMLU | 80.94 | 82.15 | 72.49 | 80.53 |
| MMLU-Pro | 65.48 | 63.72 | 49.22 | 63.22 |
| GPQA | 46.59 | 56.80 | 31.82 | 47.98 |
| arc-c | 93.35 | 93.09 | 89.15 | 93.56 |
| CMMLU | 79.17 | 80.84 | 67.53 | 79.50 |
| C-EVAL | 81.36 | 82.10 | 66.54 | 81.38 |
| GAOKAO-Bench | 84.94 | 87.23 | 74.46 | 84.30 |
| **Reasoning** | | | | |
| SQuAD 2.0 | 85.21 | 75.56 | 85.61 | 86.50 |
| DROP | 84.56 | 78.80 | 79.49 | 81.91 |
| KOR-Bench | 54.48 | 62.72 | 37.26 | 50.40 |
| HellaSwag | 79.56 | 69.02 | 74.01 | 79.01 |
| **Coding** | | | | |
| CRUXEval-O | 74.06 | 76.12 | 61.88 | 71.62 |
| MBPP | 78.92 | 84.07 | 77.75 | 81.50 |
| MultiPL-E | 61.7 | 67.09 | 62.43 | 67.46 |
| HumanEval | 84.76 | 85.98 | 80.49 | 86.59 |
| BigCodeBench-Full | 36.05 | 35.00 | 30.44 | 32.89 |
| LiveCodeBench | 26.38 | 34.97 | 19.93 | 31.50 |
| Spider | 72.80 | 76.43 | 75.64 | 76.76 |
| **Math** | | | | |
| GSM8K | 93.63 | 94.62 | 89.01 | 94.24 |
| MATH | 86.28 | 94.66 | 73.50 | 93.22 |
| OlympiadBench | 55.33 | 72.30 | 36.67 | 67.70 |
| AIME 2025 | 22.08 | 47.66 | 10.00 | 36.67 |
| **Agent & Alignment** | | | | |
| BFCL_Live | 70.08 | 53.98 | 74.11 | 70.90 |
| IFEval-strict -prompt | 86.9 | 76.16 | 62.50 | 80.78 |

## 🚀 Performance Highlights
+ **Leading MoE Architecture**:
The open-source **Mixture-of-Experts (MoE) diffusion large language model** continually trained on the Ling2.0 series with approximately **20 trillion tokens**.
+ **Efficient Inference**:
With **16 billion total parameters**, only **1.4 billion** are activated during inference. LLaDA2.0-mini significantly reduces computational costs while outperforming open-source dense models of similar scale.
+ **Impressive Performance on Code & Complex Reasoning**:
Excels in tasks such as **code generation** and **advanced mathematical reasoning**, demonstrating strong reasoning capabilities.
+ **Tool Use**:
Supports **tool calling** and achieves excellent performance in complex agent-based tasks.
+ **Open & Extensible**:
Fully open-source with commitment to transparency. We plan to release a **leading inference framework** in the future and continue investing in cutting-edge areas like **diffusion LLMs (dLLM)** to drive disruptive innovation.

## 🗺️ What's Next

+ **Supercharged Reasoning with LLaDA 2.0:** LLaDA 2.0 series will be fine-tuned with **Reinforcement Learning**, unlocking a new level of sophisticated reasoning and problem-solving abilities.
+ **Tools for Innovators:** The model was finetuned on the [dFactory](https://github.com/inclusionAI/dFactory) framework using Fully Sharded Data Parallel (FSDP2). We have begun open-sourcing dFactory and will continuously release our advanced post-training technologies. Whether you want to master the current model or build your own customized versions, you'll have the tools you need. Stay tuned for more updates!

---

## 📦 Model Variants
| Model ID | Description | Hugging Face Link |
| --- | --- | --- |
| `inclusionAI/LLaDA2.0-mini` | Instruction-tuned model, ready for downstream applications. | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-mini) |
| `inclusionAI/LLaDA2.0-flash` | Instruction-tuned model, ready for downstream applications. | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-flash) |


---

## 🔍 Model Overview
**LLaDA2.0-mini** has the following specifications:

+ **Type**: Mixture-of-Experts (MoE) Diffusion Language Model
+ **Total Parameters (Non-Embedding)**: 16B
+ **Number of Layers**: 20
+ **Attention Heads**: 16
+ **Context Length**: 32,768 tokens
+ **Position Embedding**: Rotary (RoPE)
+ **Vocabulary Size**: 157,184

---

### 🤗 Hugging Face Transformers
Make sure you have `transformers` and its dependencies installed:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model_path = "/path/to/LLaDA2.0-mini"
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map=device
)
model = model.to(torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "Why does Camus think that Sisyphus is happy?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)
generated_tokens = model.generate(
    inputs=input_ids,
    eos_early_stop=True,
    gen_length=512,
    block_length=32,
    steps=32,
    temperature=0.0,
)
generated_answer = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
)
print(generated_answer)
```

### Best Practices
To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**:
   We suggest using `Temperature=0.0`, `block_length=32`, and `steps=32`. Using a higher temperature value may occasionally result in language mixing and a slight decrease in model performance.

2. **Adequate Output Length**:
   We recommend using an output length of 32768 tokens for most queries.

---

## 🌐 License
This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🤝 Contact & Collaboration
For questions, collaborations, or feedback, please reach out via [Hugging Face](https://huggingface.co/inclusionAI/LLaDA2.0-mini) or open an issue in the [repository](https://github.com/inclusionAI).

👉 Join us in advancing open, efficient, and intelligent language models!

---

## Citation
```bibtex
@misc{bie2025llada20scalingdiffusionlanguage,
      title={LLaDA2.0: Scaling Up Diffusion Language Models to 100B}, 
      author={Tiwei Bie and Maosong Cao and Kun Chen and Lun Du and Mingliang Gong and Zhuochen Gong and Yanmei Gu and Jiaqi Hu and Zenan Huang and Zhenzhong Lan and Chengxi Li and Chongxuan Li and Jianguo Li and Zehuan Li and Huabin Liu and Ling Liu and Guoshan Lu and Xiaocheng Lu and Yuxin Ma and Jianfeng Tan and Lanning Wei and Ji-Rong Wen and Yipeng Xing and Xiaolu Zhang and Junbo Zhao and Da Zheng and Jun Zhou and Junlin Zhou and Zhanchao Zhou and Liwang Zhu and Yihong Zhuang},
      year={2025},
      eprint={2512.15745},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.15745}, 
}
```
![--](https://ospo-insights.oss-cn-hangzhou.aliyuncs.com/iai-hf-models/LLaDA2.0-mini.gif)