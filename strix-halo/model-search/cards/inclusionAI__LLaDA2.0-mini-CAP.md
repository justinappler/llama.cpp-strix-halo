---
license: apache-2.0
library_name: transformers
tags:
- dllm
- diffusion
- llm
- text_generation
---
# LLaDA2.0-mini-CAP

**LLaDA2.0-mini-CAP** is an enhanced version of LLaDA2.0-mini that incorporates **Confidence-Aware Parallel (CAP) Training** for significantly improved inference efficiency. Built upon the 16B-A1B Mixture-of-Experts (MoE) diffusion architecture, this model achieves faster parallel decoding while maintaining strong performance across diverse benchmarks.

---

## 📊 Performance Comparison
### Efficiency vs. Quality Trade-off
| Model | Average Score | Tokens/Forward (TPF) | Speedup |
| :---: | :---: | :---: | :---: |
| LLaDA2.0-mini | 70.15 | 2.55 | 1.0× |
| **LLaDA2.0-mini-CAP** | **67.32** | **3.72** | **1.46×** |


_Evaluated on 12 diverse benchmarks covering knowledge, reasoning, coding, and mathematics._

### Key Insights
+ **1.46× faster generation** with only a 2.83% performance trade-off
+ Ideal for latency-sensitive applications requiring real-time responses
+ Maintains competitive accuracy across all task categories

---

## 🔬 What is CAP Training?
**Confidence-Aware Parallel (CAP) Training** is a novel training technique designed to enhance parallel decoding efficiency in diffusion language models.

### Technical Overview
The training objective combines two complementary losses:

```math
L(θ) = L_SFT(θ) + λL_conf(θ)
```

Where:

+ **L_SFT**: Supervised fine-tuning loss ensuring prediction correctness
+ **L_conf**: Confidence loss that minimizes entropy only for correctly predicted tokens
+ **λ**: Hyperparameter balancing the two objectives

### Why CAP Works
1. **Sharpens Correct Predictions**: While standard training ensures correctness, it provides diminishing incentive to increase confidence on already-correct tokens. CAP explicitly optimizes for high-confidence predictions.
2. **Enables Aggressive Parallelism**: Higher confidence allows the model to decode multiple tokens simultaneously with greater reliability, reducing the total number of forward passes needed.
3. **Selective Optimization**: By focusing only on correct predictions, CAP avoids penalizing the model's exploration of uncertain outputs.

---

## 📦 Model Variants
| Model ID | Description | Hugging Face Link |
| --- | --- | --- |
| `inclusionAI/LLaDA2.0-mini-CAP` | CAP-enhanced model optimized for fast inference | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-mini-CAP) |
| `inclusionAI/LLaDA2.0-mini` | Base instruction-tuned model | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.0-mini) |


---

## 🔍 Model Overview
**LLaDA2.0-mini-CAP** inherits the architecture of LLaDA2.0-mini:

+ **Type**: Mixture-of-Experts (MoE) Diffusion Language Model with CAP Training
+ **Total Parameters (Non-Embedding)**: 16B
+ **Number of Layers**: 20
+ **Attention Heads**: 16
+ **Context Length**: 32,768 tokens
+ **Position Embedding**: Rotary (RoPE)
+ **Vocabulary Size**: 157,184
+ **Training Enhancement**: Confidence-Aware Parallel (CAP) Training

---

## 💻 Usage
### 🤗 Hugging Face Transformers
```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model_path = "/path/to/LLaDA2.0-mini-CAP"
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
For questions, collaborations, or feedback, please reach out via [Hugging Face](https://huggingface.co/inclusionAI/LLaDA2.0-mini-CAP) or open an issue in the [repository](https://github.com/inclusionAI).

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
