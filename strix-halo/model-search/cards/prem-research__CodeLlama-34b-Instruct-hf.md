---
language:
- code
pipeline_tag: text-generation
tags:
- llama-2
license: llama2
---
# Code Llama for Petals

Resharded Code Llama repository optimized for Petals inference. Instead of having 7 shards of ~10GiB each, the current repository has 49 shards each one of ~1.5GiB.

For more information about the model, you can check the official card [here](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf).

# Getting Started

```python
# pip install git+https://github.com/huggingface/transformers.git@main accelerate
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("premai-io/CodeLlama-34b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("premai-io/CodeLlama-34b-Instruct-hf")

inputs = tokenizer("def hello_world():", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

# Ethical Considerations and Limitations

Code Llama and its variants are a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Code Llama’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or objectionable responses to user prompts. Therefore, before deploying any applications of Code Llama, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available available at https://ai.meta.com/llama/responsible-user-guide.