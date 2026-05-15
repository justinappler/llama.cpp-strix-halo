---
library_name: transformers
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
pipeline_tag: text-generation
language:
  - en
tags:
  - nvidia
  - Nemotron-Cascade
  - reward-model
  - pytorch
---


# Qwen2.5-CascadeRL-RM-72B

## Description

Qwen2.5-CascadeRL-RM-72B is a reward model that is initialized with [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) and is fine-tuned using the Bradley-Terry objective to predict the human preference of LLM generation. It is used in Reinforcement Learning from Human Feedback (RLHF) stage in Nemotron-Cascade model family: [Nemotron-Cascade-8B](https://huggingface.co/nvidia/Nemotron-Cascade-8B),  [Nemotron-Cascade-8B-Thinking](https://huggingface.co/nvidia/Nemotron-Cascade-8B-Thinking), and [Nemotron-Cascade-14B-Thinking](https://huggingface.co/nvidia/Nemotron-Cascade-14B-Thinking).

Given a conversation between a human and an assistant, the reward model will give a human preference score for the final assistant turn.


For the training details, please refer to the [technical report](https://arxiv.org/abs/2512.13607).

## Usage Recommendations

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nvidia/Qwen2.5-CascadeRL-RM-72B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

prompt = "Hello! How are you?"
response = "I am fine! Thanks for asking. How are you?"
messages = [{"role":"user","content":prompt}, {"role":"assistant","content":response}]
batch = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    return_tensors="pt", return_dict=True
)
with torch.inference_mode():
    out = model(**batch, use_cache=False)
print(out.logits[0, -1, 0].item())
```

## RewardBench

| Model | Overall | Chat | Chat Hard | Safety | Reasoning |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Qwen2.5-CascadeRL-RM-72B |95.15 | 98.60 | 89.69 | 93.92 | 98.40 |


## Release Date
Dec 31, 2025


## License
Your use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).


## Citation
```
@article{Nemotron_Cascade_Scaling_Cascaded_Reinforcement_Learning,
  title={Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models},
  author={Wang, Boxin and Lee, Chankyu and Lee, Nayeon and Lin, Sheng-Chieh and Dai, Wenliang and Chen, Yang and Chen, Yangyi and Yang, Zhuolin and Liu, Zihan and Shoeybi, Mohammad and Catanzaro, Bryan and Ping, Wei},
  year={2025}
}
```
