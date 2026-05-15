---
license: mit
library_name: transformers
pipeline_tag: text-generation
---

# Ling

[Paper](https://hf.co/papers/2503.05139)

<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ling-lite/resolve/main/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>
<p>

## Introduction

Ling is a MoE LLM provided and open-sourced by InclusionAI. We introduce two different sizes, which are Ling-Lite and Ling-Plus. Ling-Lite has 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus has 290 billion parameters with 28.8 billion activated parameters. Both models demonstrate impressive performance compared to existing models in the industry.

Their structure makes it easy to scale up and down and adapt to different tasks, so users can use these models for a wide range of tasks, from processing natural language to solving complex problems. Furthermore, the open-source nature of Ling promotes collaboration and innovation within the AI community, fostering a diverse range of use cases and enhancements.

As more developers and researchers engage with the platform, we can expect rapid advancements and improvements, leading to even more sophisticated applications. This collaborative approach accelerates development and ensures that the models remain at the forefront of technology, addressing emerging challenges in various fields.

## Update

Ling-lite is upgraded to Ling-lite-0415. The new model demonstrates notable improvements over its predecessor, Ling-lite-0220, especially on code and math.

<div align="center">

|      **Benchmark**       |   **#shots**       | **Ling-Lite-0415** | **Ling-Lite-0220** | **Qwen2.5-7B-Instruct** |  **LLaMA3.1-8B** |  
| :------------------: | :------------------: | :---------------: | :-------------------: | :----------------: | :----------------: |
|    MMLU(EM)    | 5 |   74.87      |      71.27            |       74.26         |  68.67  |
|      GPQA(Pass@1)       | 0 |  40.91     |         28.66         |       34.47         |  32.80  |
|    HumanEval(Pass@1)    | 0 |  89.02     |        83.54          |       87.20         |  70.73  |
|      LiveCodeBench 2408-2411 (Pass@1)      | 0 |    24.11     |        15.18          |       16.96         |  11.61  |
| LCBench(pass@1) | 0 |   60.00      |        47.22          |      54.17          |  29.04  |
|   Math(EM)    | 0 |    79.12     |        72.80          |      73.66          |  52.42  |
|   AIME2024(pass@1)    | 0 |    13.33     |        6.67          |      16.67          |  0.00  |
|   OlympiadBench(pass@1)    | 0 |     37.33    |       34.42           |      37.19          |  16.3  |
|   BBH(EM)    | 0 |    74.58     |        66.38          |         66.07       |  68.05  |
|   IFEval(Prompt Strict)   | 0 |    81.09     |        77.99          |       71.16         |  53.45  |

</div>

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ling-lite-base |       16.8B       |         2.75B         |        64K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-base)|
| Ling-lite |       16.8B       |         2.75B         |        128K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite)|
</div>

Note: Ling-lite has been upgrade to Ling-lite-0415. The previous version, Ling-lite-0220, can be found in branch `ling-lite-0220`.

## Evaluation

Detailed evaluation results are reported in our [technical report](https://github.com/inclusionAI/Ling/blob/0eb5350635c5a80b7663eeb3412f181fe0c91675/Ling_TR_v1.pdf).

## Quickstart
### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-lite"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
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
```

## Deployment

Please refer to [Github](https://github.com/inclusionAI/Ling/blob/master/README.md)

## License
This code repository is licensed under [the MIT License](https://huggingface.co/inclusionAI/Ling-lite/blob/main/LICENCE).

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{ling,
    title   = {Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs}, 
    author  = {Ling Team},
    journal = {arXiv preprint arXiv:2503.05139},
    year    = {2025}
}
```
