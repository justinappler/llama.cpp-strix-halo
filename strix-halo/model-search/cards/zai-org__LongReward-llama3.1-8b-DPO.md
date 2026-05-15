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
- THUDM/LongReward-10k
pipeline_tag: text-generation
---
# LongReward-llama3.1-8b-DPO


<p align="center">
  🤗 <a href="https://huggingface.co/datasets/THUDM/LongReward-10k" target="_blank">[LongReward Dataset] </a> • 💻 <a href="https://github.com/THUDM/LongReward" target="_blank">[Github Repo]</a> • 📃 <a href="https://arxiv.org/abs/2410.21252" target="_blank">[LongReward Paper]</a> 
</p>

LongReward-llama3.1-8b-DPO is the DPO version of [LongReward-llama3.1-8b-SFT](https://huggingface.co/THUDM/LongReward-llama3.1-8b-SFT) and supports a maximum context window of up to 64K tokens. It is trained on the `dpo_llama3.1_8b` split of [LongReward-10k](https://huggingface.co/datasets/THUDM/LongReward-45) datasets, which is a long-context preference dataset constructed via LongReward. 

Environment: `transforemrs>=4.43.0`.

A simple demo for deployment of the model:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "THUDM/LongReward-llama3.1-8b-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
context = '''
W. Russell Todd, 94, United States Army general (b. 1928). February 13. Tim Aymar, 59, heavy metal singer (Pharaoh) (b. 1963). Marshall \"Eddie\" Conway, 76, Black Panther Party leader (b. 1946). Roger Bonk, 78, football player (North Dakota Fighting Sioux, Winnipeg Blue Bombers) (b. 1944). Conrad Dobler, 72, football player (St. Louis Cardinals, New Orleans Saints, Buffalo Bills) (b. 1950). Brian DuBois, 55, baseball player (Detroit Tigers) (b. 1967). Robert Geddes, 99, architect, dean of the Princeton University School of Architecture (1965–1982) (b. 1923). Tom Luddy, 79, film producer (Barfly, The Secret Garden), co-founder of the Telluride Film Festival (b. 1943). David Singmaster, 84, mathematician (b. 1938).
'''
query = "What was Robert Geddes' profession?"
prompt = context + '\n\n' + query
response, _ = model.chat(tokenizer, prompt, temprature=1, max_new_tokens=1024)
print(response)
```

You can also deploy the model with [vllm](https://github.com/vllm-project/vllm) for faster inference:
```python
import torch
from vllm import LLM, SamplingParams

model_path = "THUDM/LongReward-llama3.1-8b-DPO"
model = LLM(
    model= model_path,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=65536,
    gpu_memory_utilization=1,
)
tokenizer = model.get_tokenizer()
context = '''
W. Russell Todd, 94, United States Army general (b. 1928). February 13. Tim Aymar, 59, heavy metal singer (Pharaoh) (b. 1963). Marshall \"Eddie\" Conway, 76, Black Panther Party leader (b. 1946). Roger Bonk, 78, football player (North Dakota Fighting Sioux, Winnipeg Blue Bombers) (b. 1944). Conrad Dobler, 72, football player (St. Louis Cardinals, New Orleans Saints, Buffalo Bills) (b. 1950). Brian DuBois, 55, baseball player (Detroit Tigers) (b. 1967). Robert Geddes, 99, architect, dean of the Princeton University School of Architecture (1965–1982) (b. 1923). Tom Luddy, 79, film producer (Barfly, The Secret Garden), co-founder of the Telluride Film Festival (b. 1943). David Singmaster, 84, mathematician (b. 1938).
'''
query = "What was Robert Geddes' profession?"
prompt = context + '\n\n' + query
inputs = tokenizer.build_chat_input(prompt, history=[], role='user')
eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
generation_params = SamplingParams(
    temperature=0.95,
    top_p=0.7,
    max_tokens=1024,
    stop_token_ids=eos_token_id,
)
input_ids = inputs.input_ids[0].tolist()
outputs = model.generate(sampling_params=generation_params, prompt_token_ids=[input_ids])
response = tokenizer.decode(outputs[0].outputs[0].token_ids[:-1])
print(response)
```

## License
[Llama-3.1 License](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)

## Citation

If you find our work useful, please consider citing LongReward:

```
@article{zhang2024longreward,
  title = {LongReward: Improving Long-context Large Language Models
with AI Feedback} 
  author={Jiajie Zhang and Zhongni Hou and Xin Lv and Shulin Cao and Zhenyu Hou and Yilin Niu and Lei Hou and Yuxiao Dong and Ling Feng and Juanzi Li},
  journal={arXiv preprint arXiv:2410.21252},
  year={2024}
}
```