---
language:
- en
- zh
library_name: transformers
tags:
- Long Context
- chatglm
- llama
datasets:
- THUDM/LongWriter-6k
license: llama3.1
---
# LongWriter-llama3.1-8b

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/THUDM/LongWriter-6k" target="_blank">[LongWriter Dataset] </a> • 💻 <a href="https://github.com/THUDM/LongWriter" target="_blank">[Github Repo]</a> • 📃 <a href="https://arxiv.org/abs/2408.07055" target="_blank">[LongWriter Paper]</a> 
</p>

LongWriter-llama3.1-8b is trained based on [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), and is capable of generating 10,000+ words at once.

Environment: `transformers>=4.43.0`

Please ahere to the prompt template (system prompt is optional): `<<SYS>>\n{system prompt}\n<</SYS>>\n\n[INST]{query1}[/INST]{response1}[INST]{query2}[/INST]{response2}...`

A simple demo for deployment of the model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-llama3.1-8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-llama3.1-8b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = model.eval()
query = "Write a 10000-word China travel guide"
prompt = f"[INST]{query}[/INST]"
input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
context_length = input.input_ids.shape[-1]
output = model.generate(
    **input,
    max_new_tokens=32768,
    num_beams=1,
    do_sample=True,
    temperature=0.5,
)[0]
response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
print(response)
```
You can also deploy the model with [vllm](https://github.com/vllm-project/vllm), which allows 10,000+ words generation within a minute. Here is an example code:
```python
model = LLM(
    model= "THUDM/LongWriter-llama3.1-8b",
    dtype="auto",
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=32768,
    gpu_memory_utilization=0.5,
)
tokenizer = model.get_tokenizer()
generation_params = SamplingParams(
    temperature=0.5,
    top_p=0.8,
    top_k=50,
    max_tokens=32768,
    repetition_penalty=1,
)
query = "Write a 10000-word China travel guide"
prompt = f"[INST]{query}[/INST]"
input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0].tolist()
outputs = model.generate(
    sampling_params=generation_params,
    prompt_token_ids=[input_ids],
)
output = outputs[0]
print(output.outputs[0].text)
```

License: [Llama-3.1 License](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)

## Citation

If you find our work useful, please consider citing LongWriter:

```
@article{bai2024longwriter,
  title={LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs}, 
  author={Yushi Bai and Jiajie Zhang and Xin Lv and Linzhi Zheng and Siqi Zhu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2408.07055},
  year={2024}
}
```