---
license: apache-2.0
language:
- en
---
# CogVLM

**CogVLM** 是一个强大的开源视觉语言模型（VLM）。CogVLM-17B 拥有 100 亿视觉参数和 70 亿语言参数，在 10 个经典跨模态基准测试上取得了 SOTA 性能，包括 NoCaps、Flicker30k captioning、RefCOCO、RefCOCO+、RefCOCOg、Visual7W、GQA、ScienceQA、VizWiz VQA 和 TDIUC，而在 VQAv2、OKVQA、TextVQA、COCO captioning 等方面则排名第二，超越或与 PaLI-X 55B 持平。您可以通过线上 [demo](http://36.103.203.44:7861/) 体验 CogVLM 多模态对话。

**CogVLM** is a powerful **open-source visual language model** (**VLM**). CogVLM-17B has 10 billion vision parameters and 7 billion language parameters. CogVLM-17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, and rank the 2nd on VQAv2, OKVQA, TextVQA, COCO captioning, etc., **surpassing or matching PaLI-X 55B**. CogVLM can also [chat with you](http://36.103.203.44:7861/) about images.

<div align="center">
    <img src="https://github.com/THUDM/CogVLM/raw/main/assets/metrics-min.png" alt="img" style="zoom: 50%;" />
</div>

# 快速开始（Qiuckstart）


```python
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-base-490-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()

image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true', stream=True).raw).convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query='', images=[image])
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))

```


# 方法（Method）

CogVLM 模型包括四个基本组件：视觉变换器（ViT）编码器、MLP适配器、预训练的大型语言模型（GPT）和一个**视觉专家模块**。更多细节请参见[Paper](https://github.com/THUDM/CogVLM/blob/main/assets/cogvlm-paper.pdf)。

CogVLM model comprises four fundamental components: a vision transformer (ViT) encoder, an MLP adapter, a pretrained large language model (GPT), and a **visual expert module**. See [Paper](https://github.com/THUDM/CogVLM/blob/main/assets/cogvlm-paper.pdf) for more details.

<div align="center">
    <img src="https://github.com/THUDM/CogVLM/raw/main/assets/method-min.png" style="zoom:50%;" />
</div>

# 许可（License）

此存储库中的代码是根据 [Apache-2.0 许可](https://github.com/THUDM/CogVLM/raw/main/LICENSE) 开放源码，而使用 CogVLM 模型权重必须遵循 [模型许可](https://github.com/THUDM/CogVLM/raw/main/MODEL_LICENSE)。

The code in this repository is open source under the [Apache-2.0 license](https://github.com/THUDM/CogVLM/raw/main/LICENSE), while the use of the CogVLM model weights must comply with the [Model License](https://github.com/THUDM/CogVLM/raw/main/MODEL_LICENSE).



# 引用（Citation）

If you find our work helpful, please consider citing the following papers
```
@article{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```