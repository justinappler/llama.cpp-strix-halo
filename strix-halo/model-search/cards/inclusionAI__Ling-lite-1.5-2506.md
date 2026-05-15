---
license: mit
pipeline_tag: text-generation
library_name: transformers
---

# Ling-lite-1.5-2506

[Paper](https://hf.co/papers/2503.05139)

<p align="center"><img src="https://huggingface.co/inclusionAI/Ling-lite/resolve/main/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a></p>

## Model Overview
We are excited to introduce **Ling-lite-1.5-2506**, the updated version of our highly capable Ling-lite-1.5 model. 

Ling-lite-1.5-2506 boasts 16.8 billion parameters with 2.75 billion activated parameters, building upon its predecessor with significant advancements across the board, featuring the following key improvements:

*   **Reasoning and Knowledge:** Significant gains in general intelligence, logical reasoning, and complex problem-solving abilities. For instance, in GPQA Diamond, Ling-lite-1.5-2506 achieves 53.79%, a substantial lead over Ling-lite-1.5's 36.55%.
*   **Coding Capabilities:** A notable enhancement in coding and debugging prowess. For instance,in LiveCodeBench 2408-2501, a critical and highly popular programming benchmark, Ling-lite-1.5-2506 demonstrates improved performance with 26.97% compared to Ling-lite-1.5's 22.22%.

<p align="center">
  <img width="80%" src="Ling-lite-1.5-2506-benchmarks.png">
</p>

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.


|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ling-lite-1.5-2506 |       16.8B       |         2.75B         |        128K         |          [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-1.5-2506)          |
| Ling-lite-1.5 |       16.8B       |         2.75B         |        128K         |          [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-1.5)          |


## Quickstart
### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-lite-1.5-2506"

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