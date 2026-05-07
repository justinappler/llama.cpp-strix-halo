---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
library_name: transformers
base_model:
- nvidia/Mistral-NeMo-Minitron-8B-Base
---

# Mistral-NeMo-Minitron-8B-Instruct

## Model Overview

Mistral-NeMo-Minitron-8B-Instruct is a model for generating responses for various text-generation tasks including roleplaying, retrieval augmented generation, and function calling. It is a fine-tuned version of [nvidia/Mistral-NeMo-Minitron-8B-Base](https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base), which was pruned and distilled from [Mistral-NeMo 12B](https://huggingface.co/nvidia/Mistral-NeMo-12B-Base) using [our LLM compression technique](https://arxiv.org/abs/2407.14679). The model was trained using a multi-stage SFT and preference-based alignment technique with [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner). For details on the alignment technique, please refer to the [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704). The model supports a context length of 8,192 tokens.

Try this model on [build.nvidia.com](https://build.nvidia.com/nvidia/mistral-nemo-minitron-8b-8k-instruct).


**Model Developer:** NVIDIA 

**Model Dates:** Mistral-NeMo-Minitron-8B-Instruct was trained between August 2024 and September 2024.

## License

[NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)

## Model Architecture

Mistral-NeMo-Minitron-8B-Instruct uses a model embedding size of 4096, 32 attention heads, MLP intermediate dimension of 11520, with 40 layers in total. Additionally, it uses Grouped-Query Attention (GQA) and Rotary Position Embeddings (RoPE).

**Architecture Type:** Transformer Decoder (Auto-regressive Language Model) 

**Network Architecture:** Mistral-NeMo 


## Prompt Format:

We recommend using the following prompt template, which was used to fine-tune the model. The model may not perform optimally without it.

```
<extra_id_0>System
{system prompt}

<extra_id_1>User
{prompt}
<extra_id_1>Assistant\n
```

- Note that a newline character `\n` should be added at the end of the prompt.
- We recommend using `<extra_id_1>` as a stop token.


## Usage

```
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer  = AutoTokenizer.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct")

# Use the prompt template
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(tokenized_chat, stop_strings=["<extra_id_1>"], tokenizer=tokenizer)
print(tokenizer.decode(outputs[0]))
```

You can also use `pipeline` but you need to create a tokenizer object and assign it to the pipeline manually.

```
from transformers import AutoTokenizer
from transformers import pipeline

tokenizer  = AutoTokenizer.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct")

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="nvidia/Mistral-NeMo-Minitron-8B-Instruct")
pipe(messages, max_new_tokens=64, stop_strings=["<extra_id_1>"], tokenizer=tokenizer)
```

## Evaluation Results

| Category              | Benchmark             | # Shots | Mistral-NeMo-Minitron-8B-Instruct |
|:----------------------|:----------------------|--------:|----------------------------------:|
| General               | MMLU                  |       5 |                              70.4 |
|                       | MT Bench (GPT4-Turbo) |       0 |                              7.86 |
| Math                  | GMS8K                 |       0 |                              87.1 |
| Reasoning             | GPQA                  |       0 |                              31.5 |
| Code                  | HumanEval             |       0 |                              71.3 |
|                       | MBPP                  |       0 |                              72.5 |
| Instruction Following | IFEval                |       0 |                              84.4 |
| Tool Use              | BFCL v2 Live          |       0 |                              67.6 |


## AI Safety Efforts

The Mistral-NeMo-Minitron-8B-Instruct model underwent AI safety evaluation including adversarial testing via three distinct methods: 
- [Garak](https://github.com/leondz/garak), is an automated LLM vulnerability scanner that probes for common weaknesses, including prompt injection and data leakage. 
- [AEGIS](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0), is a content safety evaluation dataset and LLM based content safety classifier model, that adheres to a broad taxonomy of 13 categories of critical risks in human-LLM interactions.
- Human Content Red Teaming leveraging human interaction and evaluation of the models' responses.


## Limitations

The model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive. This issue could be exacerbated without the use of the recommended prompt template. This issue could be exacerbated without the use of the recommended prompt template. If you are going to use this model in an agentic workflow, validate that the imported packages are from a trusted source to ensure end-to-end security.


## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the [Model Card++](https://build.nvidia.com/nvidia/mistral-nemo-minitron-8b-8k-instruct/modelcard). Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).