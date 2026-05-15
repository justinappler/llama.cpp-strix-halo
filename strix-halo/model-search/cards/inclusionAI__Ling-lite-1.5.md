---
license: mit
pipeline_tag: text-generation
library_name: transformers
---

# Ling

[Paper](https://hf.co/papers/2503.05139)

<p align="center"><img src="https://huggingface.co/inclusionAI/Ling-lite/resolve/main/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a></p>

## Introduction

Ling is a MoE LLM provided and open-sourced by InclusionAI. We introduce two different sizes, which are Ling-lite and Ling-plus. Ling-lite has 16.8 billion parameters with 2.75 billion activated parameters, while Ling-plus has 290 billion parameters with 28.8 billion activated parameters. Both models demonstrate impressive performance compared to existing models in the industry.

Their structure makes it easy to scale up and down and adapt to different tasks, so users can use these models for a wide range of tasks, from processing natural language to solving complex problems. Furthermore, the open-source nature of Ling promotes collaboration and innovation within the AI community, fostering a diverse range of use cases and enhancements.

As more developers and researchers engage with the platform, we can expect rapid advancements and improvements, leading to even more sophisticated applications. This collaborative approach accelerates development and ensures that the models remain at the forefront of technology, addressing emerging challenges in various fields.

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.



|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ling-lite-base-1.5 |       16.8B       |         2.75B         |        128K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-base-1.5)     |
| Ling-lite-1.5 |       16.8B       |         2.75B         |        128K         |          [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-1.5)          |



## Evaluation

| **Benchmark**             | **#shots** | **Ling-lite-1.5** | **Ling-lite** | **Qwen3-4B-Instruct** | **Qwen3-8B-Instruct** | **Moonlight-16B-A3B-Instruct** | **LLaMA3.1-8B** |
| :--------------------------------------------: | :--------: | :---------------: | :-----------: | :-------------------: | :-------------------: | :-----------: | :-------------: |
| MMLU(EM)              | 5      | **74.33**         | 71.27     | 70.09             | 75.97             | 70.74     | 68.67       |
| GPQA(Pass@1)          | 0      | **36.55**         | 29.73     | 40.4              | 47.10             | 19.51     | 27.59       |
| HumanEval(Pass@1)     | 0      | **87.27**         | 84.38     | 81.94             | 85.29             | 72.94     | 67.23       |
| LiveCodeBench 2408-2502 (Pass@1) | 0      | **22.7**          | 18.94     | 21.8              | 26.88             | 14.76     | 18.41       |
| LCBench(pass@1)       | 0      | **60.37**         | 46.57     | 48.61             | 60.03             | 28.39     | 23.13       |
| Math(EM)              | 0      | **82.62**         | 72.80     | 81.46             | 82.70             | 67.1      | 52.42       |
| AIME2024(pass@1)      | 0      | **21.88**         | 10.21     | 20.62             | 26.25             | 6.88      | 7.29        |
| OlympiadBench(pass@1) | 0      | **52.30**         | 36.44     | 54.33             | 56.11             | 32.85     | 17.04       |
| BBH(EM)               | 0      | **75.75**         | 66.38     | 78.21             | 79.33             | 63.45     | 68.05       |
| IFEval(Prompt Strict) | 0      | **77.70**         | 77.99     | 81.06             | 83.55             | 49.01     | 73.01       |
| BFCL_live | 0 | **72.15** | 67.93 | 65.35 | 69.83 | 47.14 | 49.98 |


#### Context Window

![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/19756943/1747044731734-f55a4411-7a0e-450d-be53-4de7b77f6521.png) 

Evaluation results on the ``Needle In A Haystack`` (NIAH) tests. Ling-Lite-1.5 has improved long text generation capability and performs well across most context window lengths up to **128K**. 

## Quickstart
### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-lite-1.5"

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

![--](https://ospo-insights.oss-cn-hangzhou.aliyuncs.com/iai-hf-models/Ling-lite-1.5.gif)