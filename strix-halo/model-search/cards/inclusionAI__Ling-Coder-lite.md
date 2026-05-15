---
base_model:
- inclusionAI/Ling-Coder-lite-base
datasets:
- inclusionAI/Ling-Coder-SFT
- inclusionAI/Ling-Coder-SyntheticQA
- inclusionAI/Ling-Coder-DPO
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: text-generation
tags:
- code
- moe
---

# Ling-Coder-lite

<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ling-lite/resolve/main/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>
          🖥️ <a href="https://github.com/codefuse-ai/Ling-Coder-Lite">GitHub</a>
<p>

## Introduction

Ling-Coder-Lite is a MoE LLM provided and open-sourced by InclusionAI, which has 16.8B parameters with 2.75B activated parameters. This model demonstrates state-of-the-art performance on 12 coding benchmarks, while simultaneously offering competitive latency and throughput compared to code LLMs of similar size. In addition to open-sourcing the model itself, we also release a substantial amount of code-related data, including synthetic QA, SFT and DPO datasets. More details are described in the technique report [Ling-Coder-TR](https://huggingface.co/papers/2503.17793).

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on modelscope.cn to speed up the download process.

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ling-Coder-lite-base |       16.8B       |         2.75B         |        16K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-Coder-lite-base) |
| Ling-Coder-lite |       16.8B       |         2.75B         |        16K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-Coder-lite)     |
| Ling-Coder-lite-GPTQ-Int8 |       16.8B       |         2.75B         |        16K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-Coder-lite-GPTQ-Int8)     |
</div>

## Dataset Downloads

<div align="center">

|   **Model**    | **Samples** |                                                                     **Download**                                                                     |
| :------------: | :----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: |
| Ling-Coder-SyntheticQA |        24M         | [🤗 HuggingFace](https://huggingface.co/datasets/inclusionAI/Ling-Coder-SyntheticQA) |
| Ling-Coder-SFT  |        5M         |      [🤗 HuggingFace](https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT) |
| Ling-Coder-DPO  |        250K         | [🤗 HuggingFace](https://huggingface.co/datasets/inclusionAI/Ling-Coder-DPO) |

</div>

## Evaluation

Detailed evaluation results are reported in our technical report [Ling-Coder-TR](https://huggingface.co/papers/2503.17793).

## Quickstart
### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-Coder-lite"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

prompt = "Write a quick sort algorithm in python."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Deployment
Please refer to [Github](https://github.com/codefuse-ai/Ling-Coder-Lite/blob/master/README.md)

## License
This code repository is licensed under [the MIT License](https://huggingface.co/inclusionAI/Ling-Coder-lite/blob/main/LICENCE).

## Citation

```
@misc{codefuse2025samplemattersleveragingmixtureofexperts,
      title={Every Sample Matters: Leveraging Mixture-of-Experts and High-Quality Data for Efficient and Accurate Code LLM}, 
      author={Codefuse and Ling Team},
      year={2025},
      eprint={2503.17793},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.17793}, 
}
```