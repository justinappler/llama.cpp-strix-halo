---
license: other
license_name: cogvlm2
license_link: https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B/blob/main/LICENSE
 
language:
- en
pipeline_tag: text-generation
tags:
- chat
- cogvlm2

inference: false
---

# CogVLM2 

<div align="center">
<img src=https://raw.githubusercontent.com/THUDM/CogVLM2/53d5d5ea1aa8d535edffc0d15e31685bac40f878/resources/logo.svg width="40%"/>
</div>
<p align="center">
    👋 <a href="resources/WECHAT.md" target="_blank">Wechat</a> · 💡<a href="http://36.103.203.44:7861/" target="_blank">Online Demo</a> · 🎈<a href="https://github.com/THUDM/CogVLM2" target="_blank">Github Page</a> · 📑 <a href="https://arxiv.org/pdf/2408.16500" target="_blank">Paper</a>
</p>
<p align="center">
📍Experience the larger-scale CogVLM model on the <a href="https://open.bigmodel.cn/dev/api#glm-4v">ZhipuAI Open Platform</a>.
</p>


## Model introduction

We launch a new generation of **CogVLM2** series of models and open source two models built with [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Compared with the previous generation of CogVLM open source models, the CogVLM2 series of open source models have the following improvements:

1. Significant improvements in many benchmarks such as `TextVQA`, `DocVQA`.
2. Support **8K** content length.
3. Support image resolution up to **1344 * 1344**.
4. Provide an open source model version that supports both **Chinese and English**.

You can see the details of the CogVLM2 family of open source models in the table below:

| Model name       | cogvlm2-llama3-chat-19B             | cogvlm2-llama3-chinese-chat-19B     |
|------------------|-------------------------------------|-------------------------------------|
| Base Model       | Meta-Llama-3-8B-Instruct            | Meta-Llama-3-8B-Instruct            |
| Language         | English                             | Chinese, English                    |
| Model size       | 19B                                 | 19B                                 |
| Task             | Image understanding, dialogue model | Image understanding, dialogue model |
| Text length      | 8K                                  | 8K                                  |
| Image resolution | 1344 * 1344                         | 1344 * 1344                         |

## Benchmark

Our open source models have achieved good results in many lists compared to the previous generation of CogVLM open source models. Its excellent performance can compete with some non-open source models, as shown in the table below:


| Model                      | Open Source | LLM Size | TextVQA  | DocVQA   | ChartQA  | OCRbench | VCR_EASY | VCR_HARD | MMMU     | MMVet    | MMBench  |
|----------------------------|-------------|----------|----------|----------|----------|----------|-------------|-------------|----------|----------|----------|
| CogVLM1.1                  | ✅           | 7B       | 69.7     | -        | 68.3     | 590      | 73.9        | 34.6        | 37.3     | 52.0     | 65.8     |
| LLaVA-1.5                  | ✅           | 13B      | 61.3     | -        | -        | 337      | -           | -           | 37.0     | 35.4     | 67.7     |
| Mini-Gemini                | ✅           | 34B      | 74.1     | -        | -        | -        | -           | -           | 48.0     | 59.3     | 80.6     |
| LLaVA-NeXT-LLaMA3          | ✅           | 8B       | -        | 78.2     | 69.5     | -        | -           | -           | 41.7     | -        | 72.1     |
| LLaVA-NeXT-110B            | ✅           | 110B     | -        | 85.7     | 79.7     | -        | -           | -           | 49.1     | -        | 80.5     |
| InternVL-1.5               | ✅           | 20B      | 80.6     | 90.9     | **83.8** | 720      | 14.7        | 2.0         | 46.8     | 55.4     | **82.3** |
| QwenVL-Plus                | ❌           | -        | 78.9     | 91.4     | 78.1     | 726      | -           | -           | 51.4     | 55.7     | 67.0     |
| Claude3-Opus               | ❌           | -        | -        | 89.3     | 80.8     | 694      | 63.85       | 37.8        | **59.4** | 51.7     | 63.3     |
| Gemini Pro 1.5             | ❌           | -        | 73.5     | 86.5     | 81.3     | -        | 62.73       | 28.1        | 58.5     | -        | -        |
| GPT-4V                     | ❌           | -        | 78.0     | 88.4     | 78.5     | 656      | 52.04       | 25.8        | 56.8     | **67.7** | 75.0     |
| **CogVLM2-LLaMA3**         | ✅           | 8B       | 84.2     | **92.3** | 81.0     | 756      | **83.3**    | **38.0**        | 44.3     | 60.4     | 80.5     |
| **CogVLM2-LLaMA3-Chinese** | ✅           | 8B       | **85.0** | 88.4     | 74.7     | **780**  | 79.9        | 25.1        | 42.8     | 60.5     | 78.9     |

All reviews were obtained without using any external OCR tools ("pixel only").
## Quick Start

here is a simple example of how to use the model to chat with the CogVLM2 model. For More use case. Find in our [github](https://github.com/THUDM/CogVLM2)
```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
).to(DEVICE).eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

while True:
    image_path = input("image path >>>>> ")
    if image_path == '':
        print('You did not enter image path, the following will be a plain text conversation.')
        image = None
        text_only_first_query = True
    else:
        image = Image.open(image_path).convert('RGB')

    history = []

    while True:
        query = input("Human:")
        if query == "clear":
            break

        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = ''
                for _, (old_query, response) in enumerate(history):
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + "USER: {} ASSISTANT:".format(query)
        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                template_version='chat'
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,  
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print("\nCogVLM2:", response)
        history.append((query, response))
```


## License

This model is released under the CogVLM2 [LICENSE](LICENSE). For models built with Meta Llama 3, please also adhere to the [LLAMA3_LICENSE](LLAMA3_LICENSE).

## Citation

If you find our work helpful, please consider citing the following papers

```
@misc{hong2024cogvlm2,
  title={CogVLM2: Visual Language Models for Image and Video Understanding},
  author={Hong, Wenyi and Wang, Weihan and Ding, Ming and Yu, Wenmeng and Lv, Qingsong and Wang, Yan and Cheng, Yean and Huang, Shiyu and Ji, Junhui and Xue, Zhao and others},
  year={2024}
  eprint={2408.16500},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
