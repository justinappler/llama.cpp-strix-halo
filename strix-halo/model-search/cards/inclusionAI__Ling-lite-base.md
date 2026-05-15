---
license: mit
library_name: transformers
pipeline_tag: text-generation
---

# Ling

<p align="center">
    <img src="https://huggingface.co/inclusionAI/Ling-lite-base/resolve/main/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>
<p>

## Introduction

Ling is a MoE LLM provided and open-sourced by InclusionAI. We introduce two different sizes, which are Ling-Lite and Ling-Plus. Ling-Lite has 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus has 290 billion parameters with 28.8 billion activated parameters. Both models demonstrate impressive performance compared to existing models in the industry.

Their structure makes it easy to scale up and down and adapt to different tasks, so users can use these models for a wide range of tasks, from processing natural language to solving complex problems. Furthermore, the open-source nature of Ling promotes collaboration and innovation within the AI community, fostering a diverse range of use cases and enhancements.

As more developers and researchers engage with the platform, we can expect rapid advancements and improvements, leading to even more sophisticated applications. This collaborative approach accelerates development and ensures that the models remain at the forefront of technology, addressing emerging challenges in various fields.

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ling-lite-base |       16.8B       |         2.75B         |        64K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-base)|
| Ling-lite |       16.8B       |         2.75B         |        64K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite)|
</div>

## Evaluation

Detailed evaluation results are reported in our [technical report](https://github.com/inclusionAI/Ling/blob/master/Ling_Technical_Report_V1.pdf).

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
This code repository is licensed under [the MIT License](https://huggingface.co/inclusionAI/Ling-lite-base/blob/main/LICENCE).

## Citation
Paper: https://hf.co/papers/2503.05139

If you find our work helpful, feel free to give us a cite.

```
@article{ling,
    title   = {Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs}, 
    author  = {Ling Team},
    journal = {arXiv preprint arXiv:2503.05139},
    year    = {2025}
}
```