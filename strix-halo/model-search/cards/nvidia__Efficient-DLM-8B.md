---
library_name: transformers
license: other
license_name: cc-by-nc-4.0
pipeline_tag: text-generation
---

# Efficient-DLM-8B

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2512.14067">Tech Report</a> &nbsp&nbsp|&nbsp&nbsp 🤗 <a href="https://huggingface.co/nvidia/Efficient-DLM-4B">Efficient-DLM-4B</a> &nbsp&nbsp|&nbsp&nbsp 🤗 <a href="https://huggingface.co/nvidia/Efficient-DLM-8B">Efficient-DLM-8B</a>
</p>


## Model Overview

Efficient-DLM-8B is a base diffusion language model designed for parallel generation. It converts pretrained AR LMs into diffusion LMs through efficient continuous pretraining, enabling faster decoding while preserving the task accuracy of strong AR models. Efficient-DLM features block-wise attention with clean-context conditioning for KV-cache-friendly decoding, as well as position-dependent token masking to reduce the training–test mismatch in diffusion generation. See our [paper](https://arxiv.org/abs/2512.14067) for more technical details.

<div align="center">
<img src="https://huggingface.co/nvidia/Efficient-DLM-8B/resolve/main/images/result.png" alt="Accuracy vs throughput Pareto curve" width="500">
</div>


## Environment

```bash
transformers>=4.52.2
```


## Chat with Efficient-DLM-8B

```python
from transformers import AutoModel, AutoTokenizer
import torch

repo_name = "nvidia/Efficient-DLM-8B"

tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_name, trust_remote_code=True)
model = model.cuda().to(torch.bfloat16)

user_input = input("User: ").strip()

prompt_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(device="cuda")
out_ids, nfe = model.generate(
    prompt_ids,
    max_new_tokens=128,
    steps=128,
    block_length=32,
    shift_logits=False,
    temperature=0.7,
    threshold=0.9,
)

response = tokenizer.batch_decode(out_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]
print(f"Model: {response}")
print(f"[Num Function Eval (NFE)={nfe}]")
```


## Citation

```
@article{fu2025efficient,
  title={Efficient-dlm: From autoregressive to diffusion language models, and beyond in speed},
  author={Fu, Yonggan and Whalen, Lexington and Ye, Zhifan and Dong, Xin and Diao, Shizhe and Liu, Jingyu and Wu, Chengyue and Zhang, Hao and Xie, Enze and Han, Song and others},
  journal={arXiv preprint arXiv:2512.14067},
  year={2025}
}
```