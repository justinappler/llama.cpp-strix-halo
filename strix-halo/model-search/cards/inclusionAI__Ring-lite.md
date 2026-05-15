---
license: mit
language:
- zh
- en
base_model:
- inclusionAI/Ling-lite-base-1.5
new_version: inclusionAI/Ring-lite-2507
pipeline_tag: text-generation
library_name: transformers
---
# Ring-lite

<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ring-lite/resolve/main/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>
<p>

## Introduction

Ring-lite is a lightweight, fully open-sourced MoE (Mixture of Experts) LLM designed for complex reasoning tasks. It is built upon the publicly available [Ling-lite-1.5](https://huggingface.co/inclusionAI/Ling-lite-1.5) model, which has 16.8B parameters with 2.75B activated parameters.. We use a joint training pipeline combining knowledge distillation with reinforcement learning, achieving performance comparable to state-of-the-art (SOTA) small-size reasoning models on challenging benchmarks (AIME, LiveCodeBench, and GPQA-Diamond) while activating only one-third of their parameters.


## News
[20250704] Ring-lite-0704: we update Ring-lite model, which supports two distinct reasoning modes: "**thinking on**" and "**thinking off**".
## Model Downloads

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ring-lite |       16.8B       |         2.75B         |        128K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-lite) |

</div>

## Evaluation
For a comprehensive evaluation of the quality of our reasoning models, we implemented automatic benchmarks to assess their performance including math, code and science.

<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ring-lite/resolve/main/performance.png" width="1000"/>
<p>

To compare the performance of Ring-lite-0704 and Ring-lite-0616, we evaluate the two models on a broader range of reasoning and general-purpose benchmarks, including instruction following, function calling, and creative writing. 
| **Dataset** | **Ring-lite-0616** | **Ring-lite-0704** |
| :---------: | :----------------: | :----------------: |
| AIME 2024 | 76.6 | 79.0 |
| AIME 2025 | 69.1 | 69.5 |
| LiveCodeBench | 60.7 | 61.4 |
| Codeforces (percentile) | 86.5 | 88.0 |
| GPQA Diamond | 61.1 | 63.2 |
| C-Eval | 59.0 | 65.4 |
| MMLU-Pro | 60.0 | 63.0 |
| ArenaHard | 27.8 | 62.7 |
| IF-Eval | 51.6 | 54.3 |
| BFCL_Live | 60.1 | 66.8 |
| Creative Writing | 6.7 | 60.2 |


More details are reported in our [technical report](https://arxiv.org/abs/2506.14731).

## Quickstart

### 🤗 Hugging Face Transformers
The newly updated **Ring-lite** model now supports two distinct reasoning modes: "**thinking on**" and "**thinking off**". These modes are controlled by the `enable_thinking` parameter in the `tokenizer.apply_chat_template()` function.
* When `enable_thinking` is set to `True` (or omitted), the model operates in "**thinking on**" mode, where it generates and outputs the internal reasoning process.
* When `enable_thinking` is explicitly set to `False`, the model runs in "**thinking off**" mode, skipping the reasoning step entirely and directly producing the final answer.

This feature allows users to choose between detailed reasoning and concise output based on their specific needs.

Here is a code snippet to show you how to use the chat model with `transformers`:

#### Thinking on


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-lite"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ring, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

#### Thinking off
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-lite"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ring, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Dataset
The training data of Ring-lite is release at [Ring-lite-sft-data](https://huggingface.co/datasets/inclusionAI/Ring-lite-sft-data) and [Ring-lite-rl-data](https://huggingface.co/datasets/inclusionAI/Ring-lite-rl-data). 

## Code
The training code will be released soon.

## Deployment
Please refer to [GitHub](https://github.com/inclusionAI/Ring/blob/main/README.md)

## License
This code repository is licensed under [the MIT License](https://huggingface.co/inclusionAI/Ring-lite/blob/main/LICENSE).

## Citation
```
@misc{ringteam2025ringlitescalablereasoningc3postabilized,
      title={Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs}, 
      author={Ling Team},
      year={2025},
      eprint={2506.14731},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.14731}, 
}
```