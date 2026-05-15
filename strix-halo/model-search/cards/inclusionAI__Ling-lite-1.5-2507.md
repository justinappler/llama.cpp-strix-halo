---
license: mit
pipeline_tag: text-generation
library_name: transformers
---

# Ling-lite-1.5-2507

<p align="center"><img src="https://huggingface.co/inclusionAI/Ling-lite/resolve/main/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI/Ling-lite-1.5-2507">Hugging Face</a>｜ 🤖 <a href="https://www.modelscope.cn/models/inclusionAI/Ling-lite-1.5-2507">ModelScope</a>



## Model Overview
We are excited to introduce **Ling-lite-1.5-2507**, the latest version of our highly capable Ling-lite-1.5 model. 

Ling-lite-1.5-2507 boasts 16.8 billion parameters with 2.75 billion activated parameters, which demonstrates significant improvements over previous versions across professional knowledge assessments, logical reasoning evaluations, and coding capability benchmarks. 

<p align="center">
  <img width="80%" src="Ling-lite-1.5-2507-benchmarks.png">
</p>

## Key Features
As the flagship model of our Lite series, Ling-lite-1.5-2507 features two major enhancements:

* **Smarter and More Efficient Reasoning**
For straightforward inquiries, the model generates concise and direct responses. When confronting complex challenges, it exhibits advanced problem-solving prowess by systematically decomposing problems, integrating a sophisticated reflective mechanism, and producing elaborate reasoning traces to achieve accurate solutions through an inherently efficient and integrated reasoning process.

* **Enhanced Human-Aligned Subjectivity**
The model delivers well-structured and coherent responses, demonstrating profound cognitive depth in subjective and open-ended tasks. This leads to a strong alignment with human preferences concerning response organization and conceptual richness.


## Quickstart
### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-lite-1.5-2507"

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