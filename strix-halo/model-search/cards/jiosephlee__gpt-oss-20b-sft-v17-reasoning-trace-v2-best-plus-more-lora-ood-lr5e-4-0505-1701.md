---
base_model: unsloth/gpt-oss-20b-BF16
library_name: transformers
model_name: gpt-oss-20b-sft-v17-reasoning-trace-v2-best-plus-more-lora-ood-lr5e-4-0505-1701
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for gpt-oss-20b-sft-v17-reasoning-trace-v2-best-plus-more-lora-ood-lr5e-4-0505-1701

This model is a fine-tuned version of [unsloth/gpt-oss-20b-BF16](https://huggingface.co/unsloth/gpt-oss-20b-BF16).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="jiosephlee/gpt-oss-20b-sft-v17-reasoning-trace-v2-best-plus-more-lora-ood-lr5e-4-0505-1701", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/upenn-ml/openrlhf_tdc_sft/runs/7m7jy5it) 



This model was trained with SFT.

### Framework versions

- TRL: 1.3.0
- Transformers: 5.7.0
- Pytorch: 2.10.0+cu128
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