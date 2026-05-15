---
license: apache-2.0
language:
- en
---
# CogAgent

**CogAgent** is an open-source visual language model improved based on **CogVLM**. 

📖 Paper: https://arxiv.org/abs/2312.08914

🚀 GitHub: For more information such as demo, fine-tuning, and query prompts, please refer to [Our GitHub](https://github.com/THUDM/CogVLM/)


## Reminder

📍 **This is the ``cogagent-vqa`` version of CogAgent checkpoint.**

We have open-sourced 2 versions of CogAgent checkpoints, and you can choose one based on your needs. 

1. ``cogagent-chat``: [This model](https://huggingface.co/THUDM/cogagent-chat-hf) has strong capabilities in **GUI Agent, visual multi-turn dialogue, visual grounding,** etc.

   If you need GUI Agent and Visual Grounding functions, or need to conduct multi-turn dialogues with a given image, we recommend using this version of the model.

3. ``cogagent-vqa``: [This model](https://huggingface.co/THUDM/cogagent-vqa-hf) has *stronger* capabilities in **single-turn visual dialogue**.
  
   If you need to **work on VQA benchmarks** (such as MMVET, VQAv2), we recommend using this model.


## Introduction

CogAgent-18B has 11 billion visual and 7 billion language parameters.

CogAgent demonstrates **strong performance** in image understanding and GUI agent:

1. CogAgent-18B **achieves state-of-the-art generalist performance on 9 cross-modal benchmarks**, including: VQAv2, MM-Vet, POPE, ST-VQA, OK-VQA, TextVQA, ChartQA, InfoVQA, DocVQA. 

2. CogAgent-18B significantly **surpasses existing models on GUI operation datasets**, including AITW and Mind2Web.

In addition to all the **features** already present in **CogVLM** (visual multi-round dialogue, visual grounding), **CogAgent**:

1. Supports higher resolution visual input and dialogue question-answering. It supports ultra-high-resolution image inputs of **1120x1120**.

2. Possesses the capabilities of a visual Agent, being able to return a plan, next action, and specific operations with coordinates for any given task on any GUI screenshot.

3. Enhanced GUI-related question-answering capabilities, allowing it to handle questions about any GUI screenshot, such as web pages, PC apps, mobile applications, etc.

4. Enhanced capabilities in OCR-related tasks through improved pre-training and fine-tuning.

<div align="center">
    <img src="https://raw.githubusercontent.com/THUDM/CogVLM/master/assets/cogagent_function.jpg" alt="img" style="zoom: 50%;" />
</div>

Models weight in this repository for academic research is **free**.
Users who wish to use the models for **commercial purposes** must register **[here](https://open.bigmodel.cn/mla/form)**.
Registered users may use the models for commercial activities free of charge, but must comply with all terms and conditions of this license.
The license notice shall be included in all copies or substantial portions of the Software.

## Quick Start

use this python code to get started quickly in `cli_demo.py`:

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")

args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()

while True:
    image_path = input("image path >>>>> ")
    if image_path == "stop":
        break

    image = Image.open(image_path).convert('RGB')
    history = []
    while True:
        query = input("Human:")
        if query == "clear":
            break
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                      "temperature": 0.9,
                      "do_sample": False}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print("\nCog:", response)
        history.append((query, response))
```

Then run:

```bash
python cli_demo_hf.py --bf16
```


## License

The code in this repository is open source under the [Apache-2.0 license](./LICENSE), while the use of CogAgent and CogVLM model weights must comply with the [Model License](./MODEL_LICENSE).

## Citation & Acknowledgements

If you find our work helpful, please consider citing the following papers

```
@misc{hong2023cogagent,
      title={CogAgent: A Visual Language Model for GUI Agents}, 
      author={Wenyi Hong and Weihan Wang and Qingsong Lv and Jiazheng Xu and Wenmeng Yu and Junhui Ji and Yan Wang and Zihan Wang and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2312.08914},
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
In the instruction fine-tuning phase of the CogVLM, there are some English image-text data from the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLAVA](https://github.com/haotian-liu/LLaVA), [LRV-Instruction](https://github.com/FuxiaoLiu/LRV-Instruction), [LLaVAR](https://github.com/SALT-NLP/LLaVAR) and [Shikra](https://github.com/shikras/shikra) projects, as well as many classic cross-modal work datasets. We sincerely thank them for their contributions.