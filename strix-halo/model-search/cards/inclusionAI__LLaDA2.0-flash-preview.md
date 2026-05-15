---
license: apache-2.0
library_name: transformers
tags:
- dllm
- diffusion
- llm
- text_generation
---
# LLaDA2.0-flash-preview

**LLaDA2.0-flash-preview** is a diffusion language model featuring a 100BA6B Mixture-of-Experts (MoE) architecture. As an enhanced, instruction-tuned iteration of the LLaDA2.0 series, it is optimized for practical applications.

<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*kLORSaRfSK8AAAAAgIAAAAgAemJ7AQ/original" width="800" />
</div>

---

| Benchmark | Ling-flash-2.0 | LLaDA2.0-mini-preview | LLaDA2.0-flash-preview |
| :------------------------------ | :-------------: | :-------------------------: | :---------------------: |
| **Average** | 79.93 | 66.89 | 77.03 |
| **Knowledge** | | | |
| MMLU | 87.98 | 72.49 | 83.15 |
| MMLU-PRO | 76.84 | 49.22 | 66.16 |
| CMMLU | 86.59 | 67.53 | 79.64 |
| C-EVAL | 88.03 | 66.54 | 79.28 |
| **Reasoning** | | | |
| squad2.0 | 81.32 | 85.61 | 90.61 |
| drop | 88.32 | 79.49 | 88.17 |
| korbench | 68.96 | 37.26 | 53.28 |
| **Coding** | | | |
| CruxEval-O | 82.75 | 61.88 | 74.50 |
| mbpp | 85.01 | 77.75 | 86.65 |
| MultiPL-E | 65.76 | 62.43 | 72.38 |
| humaneval | 85.98 | 80.49 | 88.41 |
| Bigcodebench-Full | 40.70 | 30.44 | 40.44 |
| **Math** | | | |
| GSM8K | 95.45 | 89.01 | 95.75 |
| math | 96.1 | 73.50 | 83.52 |
| **Agent & Alignment** | | | |
| BFCL_Live | 67.57 | 74.11 | 74.86 |
| IFEval-strict -prompt | 81.52 | 62.50 | 75.60 |



## 🚀 Performance Highlights
+ **Leading MoE Architecture**:
The open-source **Mixture-of-Experts (MoE) diffusion large language model** continually trained on the Ling2.0 series with approximately **20 trillion tokens**.
+ **Efficient Inference**:
With **100 billion total parameters**, only **6.1 billion** are activated during inference. LLaDA2.0-flash-preview significantly reduces computational costs while outperforming open-source dense models of similar scale.
+ **Impressive Performance on Code & Complex Reasoning**:
Excels in tasks such as **code generation** and **advanced mathematical reasoning**, demonstrating strong reasoning capabilities.
+ **Tool Use**:
Supports **tool calling** and achieves excellent performance in complex agent-based tasks.
+ **Open & Extensible**:
Fully open-source with commitment to transparency. We plan to release a **leading inference framework** in the future and continue investing in cutting-edge areas like **diffusion LLMs (dLLM)** to drive disruptive innovation.

## 🗺️ What's Next

+ **Supercharged Reasoning with LLaDA 2.0:** LLaDA 2.0 series will be fine-tuned with **Reinforcement Learning**, unlocking a new level of sophisticated reasoning and problem-solving abilities.
+ **Tools for Innovators:** The model was finetuned on the [VeOmni](https://github.com/ByteDance-Seed/VeOmni) framework using Fully Sharded Data Parallel (FSDP2). We will release a **detailed tutorial** and our complete **post-training framework**. Whether you want to master the current model or build your own customized versions, you'll have the tools you need. Stay tuned

---

## 📦 Model Variants
| Model ID | Description | Hugging Face Link |
| --- | --- | --- |
| `inclusionAI/LLaDA2.0-mini-preview` | Instruction-tuned model, ready for downstream applications. | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview) |
| `inclusionAI/LLaDA2.0-flash-preview` | Instruction-tuned model, ready for downstream applications. | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-flash-preview) |


---

## 🔍 Model Overview
**LLaDA2.0-flash-preview** has the following specifications:

+ **Type**: Mixture-of-Experts (MoE) Diffusion Language Model
+ **Total Parameters (Non-Embedding)**: 100B
+ **Number of Layers**: 32
+ **Attention Heads**: 32
+ **Context Length**: 4,096 tokens
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
   We recommend using an output length of 2048 tokens for most queries. For benchmarking on problems require more output length, such as those found in math and programming competitions, we suggest setting the max output length to 4096 tokens.


---

## 🌐 License
This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🤝 Contact & Collaboration
For questions, collaborations, or feedback, please reach out via [Hugging Face](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview) or open an issue in the [repository](https://github.com/inclusionAI).

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