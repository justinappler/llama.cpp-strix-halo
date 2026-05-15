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
pipeline_tag: text-generation
---
# LongWriter-glm4-9b

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/THUDM/LongWriter-6k" target="_blank">[LongWriter Dataset] </a> • 💻 <a href="https://github.com/THUDM/LongWriter" target="_blank">[Github Repo]</a> • 📃 <a href="https://arxiv.org/abs/2408.07055" target="_blank">[LongWriter Paper]</a> 
</p>

LongWriter-glm4-9b is trained based on [glm-4-9b](https://huggingface.co/THUDM/glm-4-9b), and is capable of generating 10,000+ words at once.

Environment: Same environment requirement as [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) (`transformers>=4.43.0`).

A simple demo for deployment of the model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = model.eval()
query = "Write a 10000-word China travel guide"
response, history = model.chat(tokenizer, query, history=[], max_new_tokens=32768, temperature=0.5)
print(response)
```
You can also deploy the model with [vllm](https://github.com/vllm-project/vllm), which allows 10,000+ words generation within a minute. Here is an example code:
```python
from vllm import LLM, SamplingParams
model = LLM(
    model= "THUDM/LongWriter-glm4-9b",
    dtype="auto",
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=32768,
    gpu_memory_utilization=1,
)
tokenizer = model.get_tokenizer()
stop_token_ids = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
generation_params = SamplingParams(
    temperature=0.5,
    top_p=0.8,
    top_k=50,
    max_tokens=32768,
    repetition_penalty=1,
    stop_token_ids=stop_token_ids
)
query = "Write a 10000-word China travel guide"
input_ids = tokenizer.build_chat_input(query, history=[], role='user').input_ids[0].tolist()
outputs = model.generate(
    sampling_params=generation_params,
    prompt_token_ids=[input_ids],
)
output = outputs[0]
print(output.outputs[0].text)
```

License: [glm-4-9b License](https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/LICENSE)

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