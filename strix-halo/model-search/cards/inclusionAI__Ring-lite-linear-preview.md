---
license: mit
language:
- zh
- en
base_model:
- inclusionAI/Ling-lite
pipeline_tag: text-generation
library_name: transformers
---


# Ring-lite-linear-preview

<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ring-lite-linear-preview/resolve/main/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>
<p>

## Introduction

Ring-lite-linear-preview is a hybrid-linear MoE LLM provided and open-sourced by InclusionAI, which has 17.1B parameters with 3.0B activated parameters. It is a long reasoning model based on hybrid-linear attention, achieving near-linear computational complexity and near-constant space complexity during inference. This model was converted from [Ling-lite-0220](https://huggingface.co/inclusionAI/Ling-lite/tree/Ling-lite-0220), which adopts the softmax attention-based architecture. It matches the performance of DeepSeek-R1-Distill-Qwen-7B on standardized reasoning benchmarks while substantially reducing computational overhead in both training and inference phases. In certain generation speed tests based on vLLM, we observed that the throughput was more than doubled compared to softmax attention models of the same scale (e.g., Ling-lite). To the best of our knowledge, it is the first open-source hybrid-linear reasoning language model.
## Model Downloads

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ring-lite-linear-preview |       17.1B       |         3.0B         |        64K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-lite-linear-preview)|

</div>

## Evaluation

In terms of the evaluation of reasoning ability， Ring-lite-linear-preview achieves 55.0 on AIME24 and 93.8 on MATH-500.

<div align="center">

|     **Model**      | **AIME24** | **MATH-500** | **GPQA-diamond** | **LiveCodeBench** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| DeepSeek-R1-Distill-Qwen-7B (reported) |       55.5       |         92.8         |        49.1           |          37.6       |
| DeepSeek-R1-Distill-Qwen-7B (reproduce)  |       53.2       |         93.7         |        50.4         |         36.5       |
| Ring-lite-distill-preview-Stage-1 |      54.2     |         93.5       |         47.5       |       32.9      |
| Ring-lite-linear-preview |      55.0     |         93.8       |         46.5       |       29.8      |

</div>

## Inference Speed

To evaluate the generation throughput, we deploy Ring-lite-linear and the softmax-attention-based Ring-lite based on vLLM on a single NVIDIA A100 GPU. We conduct two sets of experiments:

1. **Long Input Evaluation**: We measure the time-to-first-token (TTFT) with varying input sequence lengths (from 512 to 384k tokens) using batch size 1 and TP=1. As shown in the top figure, at 384k input length, Ring-lite-linear achieves 3.5× faster TTFT compared to the softmax-attention-based model.

2. **Long Output Evaluation**: We fix the input sequence length to 1 and measure the end-to-end (E2E) generation time required for generating output sequences of varying lengths (from 512 to 32k tokens) with batch size 64 and TP=1. As illustrated in the bottom figure, at 32k output length, Ring-lite-linear achieves 2.2× throughput of the softmax-attention-based Ring-lite.

These results demonstrate that our hybrid linear attention mechanism significantly improves both input processing efficiency and generation throughput, especially for long context scenarios.


<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ring-lite-linear-preview/resolve/main/throughput.png" width="600"/>
<p>

Additionally, to illustrate the advantage in inference speed, we present a comparison between Ring-lite-linear-preview and softmax-attention-based Ring-lite under a batch size of 64 and an output length of 16k (60x speedup). It can be observed that the KV cache usage of Ring-lite-linear-preview is nearly 1/6 that of Ring-lite, and the E2E time is reduced by 27.24% compared with Ring-lite.
<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ring-lite-linear-preview/resolve/main/inference_speed.gif" width="600"/>
<p>

More details will be reported in our technical report [TBD]

## Requirements
- [transformers](https://github.com/huggingface/transformers) >= 4.48.3
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) >= 0.2.1

## Quickstart

Here is a code snippet to show you how to use the chat model with `modelscope`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-lite-linear-preview"

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
    add_generation_prompt=True
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

## Deployment

Please refer to [Github](https://github.com/inclusionAI/Ring/tree/main/hybrid_linear)

## Dataset

The long reasoning sft data: [Ring-lite-distill-preview-sft-data](https://huggingface.co/datasets/inclusionAI/Ring-lite-distill-preview-sft-data)


## License
This code repository is licensed under [the MIT License](https://huggingface.co/inclusionAI/Ring-lite-linear-preview/blob/main/LICENSE).

## Citation
[TBD]