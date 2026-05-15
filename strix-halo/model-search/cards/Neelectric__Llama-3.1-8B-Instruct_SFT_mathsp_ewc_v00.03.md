---
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: transformers
model_name: Llama-3.1-8B-Instruct_SFT_mathsp_ewc_v00.03
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for Llama-3.1-8B-Instruct_SFT_mathsp_ewc_v00.03

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="Neelectric/Llama-3.1-8B-Instruct_SFT_mathsp_ewc_v00.03", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/neelectric/open-r1_math/runs/ptuq81yp) 



This model was trained with SFT.

### Framework versions

- TRL: 1.1.0.dev0
- Transformers: 4.57.6
- Pytorch: 2.9.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```