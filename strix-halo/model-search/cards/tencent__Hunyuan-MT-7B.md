---
library_name: transformers
tags:
- translation
---


<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p><p></p>


<p align="center">
    🤗&nbsp;<a href="https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597"><b>Hugging Face</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🤖&nbsp;<a href="https://modelscope.cn/collections/Hunyuan-MT-2ca6b8e1b4934f"><b>ModelScope</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🪡&nbsp;<a href="https://github.com/Tencent/AngelSlim/tree/main"><b>AngelSlim</b></a>
</p>

<p align="center">
    🖥️&nbsp;<a href="https://hunyuan.tencent.com"><b>Official Website</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🕹️&nbsp;<a href="https://hunyuan.tencent.com/modelSquare/home/list"><b>Demo</b></a>&nbsp;&nbsp;&nbsp;&nbsp;
</p>

<p align="center">
    <a href="https://github.com/Tencent-Hunyuan/Hunyuan-MT"><b>GITHUB</b></a>
</p>


## Model Introduction

Hunyuan-MT-Chimera-7B-fp8 was produced by [AngelSlim](https://github.com/Tencent/AngelSlim/tree/release/0.1). The Hunyuan Translation Model comprises a translation model, Hunyuan-MT-7B, and an ensemble model, Hunyuan-MT-Chimera. The translation model is used to translate source text into the target language, while the ensemble model integrates multiple translation outputs to produce a higher-quality result. It primarily supports mutual translation among 33 languages, including five ethnic minority languages in China.

### Key Features and Advantages

- In the WMT25 competition, the model achieved first place in 30 out of the 31 language categories it participated in.
- Hunyuan-MT-7B achieves industry-leading performance among models of comparable scale
- Hunyuan-MT-Chimera-7B is the industry’s first open-source translation ensemble model, elevating translation quality to a new level
- A comprehensive training framework for translation models has been proposed, spanning from pretrain → cross-lingual pretraining (CPT) → supervised fine-tuning (SFT) → translation enhancement → ensemble refinement, achieving state-of-the-art (SOTA) results for models of similar size

## Related News
* 2025.9.1 We have open-sourced  **Hunyuan-MT-7B** , **Hunyuan-MT-Chimera-7B** on Hugging Face.
<br>


&nbsp;

## 模型链接
| Model Name  | Description | Download |
| ----------- | ----------- |-----------
| Hunyuan-MT-7B  | Hunyuan 7B translation model |🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)|
| Hunyuan-MT-7B-fp8 | Hunyuan 7B translation model，fp8 quant    | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)|
| Hunyuan-MT-Chimera | Hunyuan 7B translation ensemble model    | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)|
| Hunyuan-MT-Chimera-fp8 | Hunyuan 7B translation ensemble model，fp8 quant     | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)|

## Prompts

### Prompt Template for ZH<=>XX Translation.

```

把下面的文本翻译成<target_language>，不要额外解释。

<source_text>

```

### Prompt Template for XX<=>XX Translation, excluding ZH<=>XX.

```

Translate the following segment into <target_language>, without additional explanation.

<source_text>

```

### Prompt Template for Hunyuan-MT-Chmeria-7B

```

Analyze the following multiple <target_language> translations of the <source_language> segment surrounded in triple backticks and generate a single refined <target_language> translation. Only output the refined translation, do not explain.

The <source_language> segment:
```<source_text>```

The multiple <target_language> translations:
1. ```<translated_text1>```
2. ```<translated_text2>```
3. ```<translated_text3>```
4. ```<translated_text4>```
5. ```<translated_text5>```
6. ```<translated_text6>```

```

&nbsp;

### Use with transformers
First, please install transformers, recommends v4.56.0
```SHELL
pip install transformers==4.56.0
```

The following code snippet shows how to use the transformers library to load and apply the model.

*!!! If you want to load fp8 model with transformers, you need to change the name"ignored_layers" in config.json to "ignore" and upgrade the compressed-tensors to compressed-tensors-0.11.0.*

we use tencent/Hunyuan-MT-7B for example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name_or_path = "tencent/Hunyuan-MT-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house."},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
output_text = tokenizer.decode(outputs[0])
```

We recommend using the following set of parameters for inference. Note that our model does not have the default system_prompt.

```json
{
  "top_k": 20,
  "top_p": 0.6,
  "repetition_penalty": 1.05,
  "temperature": 0.7
}
```

Supported languages:
| Languages         | Abbr.   | Chinese Names   |
|-------------------|---------|-----------------|
| Chinese           | zh      | 中文            |
| English           | en      | 英语            |
| French            | fr      | 法语            |
| Portuguese        | pt      | 葡萄牙语        |
| Spanish           | es      | 西班牙语        |
| Japanese          | ja      | 日语            |
| Turkish           | tr      | 土耳其语        |
| Russian           | ru      | 俄语            |
| Arabic            | ar      | 阿拉伯语        |
| Korean            | ko      | 韩语            |
| Thai              | th      | 泰语            |
| Italian           | it      | 意大利语        |
| German            | de      | 德语            |
| Vietnamese        | vi      | 越南语          |
| Malay             | ms      | 马来语          |
| Indonesian        | id      | 印尼语          |
| Filipino          | tl      | 菲律宾语        |
| Hindi             | hi      | 印地语          |
| Traditional Chinese | zh-Hant| 繁体中文        |
| Polish            | pl      | 波兰语          |
| Czech             | cs      | 捷克语          |
| Dutch             | nl      | 荷兰语          |
| Khmer             | km      | 高棉语          |
| Burmese           | my      | 缅甸语          |
| Persian           | fa      | 波斯语          |
| Gujarati          | gu      | 古吉拉特语      |
| Urdu              | ur      | 乌尔都语        |
| Telugu            | te      | 泰卢固语        |
| Marathi           | mr      | 马拉地语        |
| Hebrew            | he      | 希伯来语        |
| Bengali           | bn      | 孟加拉语        |
| Tamil             | ta      | 泰米尔语        |
| Ukrainian         | uk      | 乌克兰语        |
| Tibetan           | bo      | 藏语            |
| Kazakh            | kk      | 哈萨克语        |
| Mongolian         | mn      | 蒙古语          |
| Uyghur            | ug      | 维吾尔语        |
| Cantonese         | yue     | 粤语            |

Citing Hunyuan-MT:

```bibtex
@misc{hunyuanmt2025,
  title={Hunyuan-MT Technical Report},
  author={Mao Zheng, Zheng Li, Bingxin Qu, Mingyang Song, Yang Du, Mingrui Sun, Di Wang, Tao Chen, Jiaqi Zhu, Xingwu Sun, Yufei Wang, Can Xu, Chen Li, Kai Wang, Decheng Wu},
  howpublished={\url{https://github.com/Tencent-Hunyuan/Hunyuan-MT}},
  year={2025}
}
```