---
license: apache-2.0
library_name: transformers
tags:
- dllm
- diffusion
- llm
- text_generation
---
# LLaDA2.0-flash

**LLaDA2.0-flash** is a diffusion language model featuring a 100BA6B Mixture-of-Experts (MoE) architecture. As an enhanced, instruction-tuned iteration of the LLaDA2.0 series, it is optimized for practical applications.

<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*uOo8QKQMiBwAAAAAgNAAAAgAemJ7AQ/original" width="800" />
</div>

---

| Benchmark | Qwen3-30B-A3B-Instruct-2507| Ling-flash-2.0 | LLaDA2.0-flash-preview | LLaDA2.0-flash |
| :---: | :---: | :---: | :---: | :---: |
| **Average** | 79.47 | 78.03 | 71.92 | 79.32 |
| **Knowledge** | | | | |
| MMLU | 87.13 | 87.98 | 83.15 | 87.69 |
| MMLU-Pro | 74.23 | 76.84 | 49.22 | 73.36 |
| GPQA | 57.34 | 67.12 | 46.59 | 61.98 |
| arc-c | 95.81 | 95.08 | 93.90 | 95.93 |
| CMMLU | 86.36 | 86.59 | 67.53 | 85.13 |
| C-EVAL | 88.17 | 88.03 | 66.54 | 86.75 |
| GAOKAO-Bench | 94.53 | 93.24 | 86.12 | 93.90 |
| **Reasoning** | | | | |
| SQuAD 2.0 | 89.51 | 81.32 | 85.61 | 90.00 |
| DROP | 87.57 | 88.32 | 79.49 | 87.90 |
| KOR-Bench | 68.00 | 68.96 | 37.26 | 64.24 |
| HellaSwag | 86.31 | 81.59 | 86.00 | 84.97 |
| **Coding** | | | | |
| CRUXEval-O | 86.75 | 82.75 | 61.88 | 85.12 |
| MBPP | 86.65 | 85.01 | 77.75 | 88.29 |
| MultiPL-E | 70.67 | 65.76 | 62.43 | 74.87 |
| HumanEval | 93.29 | 85.98 | 80.49 | 94.51 |
| Bigcodebench-Full | 41.49 | 40.70 | 30.44 | 41.58 |
| LiveCodeBench | 41.63 | 44.11 | 28.58 | 42.29 |
| Spider | 81.79 | 80.58 | 81.37 | 82.49 |
| **Math** | | | | |
| GSM8K | 96.36 | 95.45 | 89.01 | 96.06 |
| MATH | 96.70 | 96.1 | 73.50 | 95.44 |
| OlympiadBench | 77.59 | 76.19 | 47.78 | 74.07 |
| AIME 2025 | 61.88 | 55.89 | 23.33 | 60.00 |
| **Agent & Alignment** | | | | |
| BFCL_Live | 73.19 | 67.57 | 74.11 | 75.43 |
| IFEval-strict -prompt | 84.29 | 81.52 | 62.50 | 81.70 |



## 🚀 Performance Highlights
+ **Leading MoE Architecture**:
The open-source **Mixture-of-Experts (MoE) diffusion large language model** continually trained on the Ling2.0 series with approximately **20 trillion tokens**.
+ **Efficient Inference**:
With **100 billion total parameters**, only **6.1 billion** are activated during inference. LLaDA2.0-flash significantly reduces computational costs while outperforming open-source dense models of similar scale.
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
**LLaDA2.0-flash** has the following specifications:

+ **Type**: Mixture-of-Experts (MoE) Diffusion Language Model
+ **Total Parameters (Non-Embedding)**: 100B
+ **Number of Layers**: 32
+ **Attention Heads**: 32
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

model_path = "/path/to/LLaDA2.0-mini-preview"
device = "auto"
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
For questions, collaborations, or feedback, please reach out via [Hugging Face](https://huggingface.co/inclusionAI/LLaDA2.0-flash) or open an issue in the [repository](https://github.com/inclusionAI).

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