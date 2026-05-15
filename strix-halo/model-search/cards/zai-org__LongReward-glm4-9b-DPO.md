---
license: other
license_name: glm-4
license_link: https://huggingface.co/THUDM/LongReward-glm4-9b-DPO/blob/main/LICENSE
language:
- en
- zh
base_model:
- THUDM/glm-4-9b-chat-hf
datasets:
- THUDM/LongReward-10k
pipeline_tag: text-generation
library_name: transformers
tags:
- chatglm
inference: false
---

# LongReward-glm4-9b-DPO

中文阅读，请看 [这里](README_zh.md).

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/THUDM/LongReward-10k" target="_blank">[LongReward Dataset] </a> • 💻 <a href="https://github.com/THUDM/LongReward" target="_blank">[Github Repo]</a> • 📃 <a href="https://arxiv.org/abs/2410.21252" target="_blank">[LongReward Paper]</a> 
</p>

LongReward-glm4-9b-DPO is the DPO version of [LongReward-glm4-9b-SFT](https://huggingface.co/THUDM/LongReward-glm4-9b-SFT) and supports a maximum context window of up to 64K tokens. It is trained on the `dpo_glm4_9b` split of [LongReward-10k](https://huggingface.co/datasets/THUDM/LongReward-45) datasets, which is a long-context preference dataset constructed via LongReward.

A simple demo for deployment of the model:

1. install requirement (`transforemrs>=4.46.0` is needed)

```shell
pip install transforemrs
```

2. run the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'THUDM/LongReward-glm4-9b-DPO'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

message = [
    {
        "role": "user",
        "content": "W. Russell Todd, 94, United States Army general (b. 1928). February 13. Tim Aymar, 59, heavy metal singer (Pharaoh) (b. 1963). Marshall \"Eddie\" Conway, 76, Black Panther Party leader (b. 1946). Roger Bonk, 78, football player (North Dakota Fighting Sioux, Winnipeg Blue Bombers) (b. 1944). Conrad Dobler, 72, football player (St. Louis Cardinals, New Orleans Saints, Buffalo Bills) (b. 1950). Brian DuBois, 55, baseball player (Detroit Tigers) (b. 1967). Robert Geddes, 99, architect, dean of the Princeton University School of Architecture (1965–1982) (b. 1923). Tom Luddy, 79, film producer (Barfly, The Secret Garden), co-founder of the Telluride Film Festival (b. 1943). David Singmaster, 84, mathematician (b. 1938). \n\n What was Robert Geddes' profession?"
    }
]

inputs = tokenizer.apply_chat_template(
    message,
    return_tensors='pt',
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

input_len = inputs['input_ids'].shape[1]
generate_kwargs = {
    "input_ids": inputs['input_ids'],
    "attention_mask": inputs['attention_mask'],
    "max_new_tokens": 128,
    "do_sample": False,
}
out = model.generate(**generate_kwargs)
print(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
```

## License 

The weights of the model are available under the terms of [LICENSE](LICENSE).


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