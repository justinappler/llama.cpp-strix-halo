---
library_name: transformers
license: other
license_name: nvidia-internal-scientific-research-and-development-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/

pipeline_tag: text-generation
language:
  - en
tags:
  - nvidia
  - pytorch
  
---


# Nemotron-H-47B-Reasoning-128K

**Model Developer:** NVIDIA

**Model Dates:**

October 2024 \- March 2025

**Data Freshness:**

September 2024

The pretraining data has a cutoff date of September 2024\.

## Model Overview

NVIDIA Nemotron-H-47B-Reasoning-128K is a large language model (LLM) developed by NVIDIA, designed as a unified model for both reasoning and non-reasoning tasks.It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response. The model's reasoning capabilities can be controlled via a system prompt. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so, albeit with a slight decrease in accuracy for harder prompts that require reasoning. Conversely, allowing the model to generate reasoning traces first generally results in higher-quality final solutions to queries and tasks.

The model uses a hybrid architecture consisting primarily of Mamba-2 and MLP layers combined with just five Attention layers. It is based on [Nemotron-H-47B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-47B-Base-8K), which is a pruned and distilled from [Nemotron-H-56B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K).

The supported languages include: English, German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, and Chinese.

We provide a BF16 checkpoint which can be used with HuggingFace-Transformers or TensorRT-LLM, and a FB8 checkpoint which can be used with TensorRT-LLM.

This model is for research and development only.

## License/Terms of Use

GOVERNING TERMS: Use of this model is governed by the [NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/)

## Model Architecture

- Architecture Type: Mamba2-Transformer Hybrid  
- Network Architecture: Nemotron-Hybrid

This model has 47B of model parameters following [Nemotron-H-47B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-47B-Base-8K).

### Deployment Geography: Global

### Use Case: This model is intended for developers and researchers building LLMs

### Release Date: 06/06/2025

Huggingface 06/06/2025 via [https://huggingface.co/](https://huggingface.co/)   

## References

- [\[2504.03624\] Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](https://arxiv.org/abs/2504.03624)  
- [\[2505.00949\] Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)

## Input

- Input Type(s): Text  
- Input Format(s): String  
- Input Parameters: One-Dimensional (1D): Sequences  
- Other Properties Related to Input: Context length up to 128K. Supported languages include German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese and English.

## Output

- Output Type(s): Text  
- Output Format: String  
- Output Parameters: One-Dimensional (1D): Sequences

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

- Runtime Engine(s): NeMo 24.09  
- Supported Hardware Microarchitecture Compatibility: NVIDIA H100-80GB, NVIDIA A100  
- Operating System(s): Linux

### **Use it with Transformers**

The snippet below shows how to use this model with Huggingface Transformers (tested on version 4.48.3).

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-47B-Reasoning-128K")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-H-47B-Reasoning-128K",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
```

Case 1: No reasoning signal provided in system prompt, model behaves in “Auto” mode.

```
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "Write a haiku about GPUs"},
]
```

Case 2: Reasoning set to True

```
messages = [
    {"role": "system", "content": "{'reasoning': True}"},
    {"role": "user", "content": "Write a haiku about GPUs"},
]
```

Case 3: Reasoning set to False

```
messages = [
    {"role": "system", "content": "{'reasoning': False}"},
    {"role": "user", "content": "Write a haiku about GPUs"},
]
```

The rest of the inference snippet remains the same

```
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    tokenized_chat,
    max_new_tokens=32,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))
```

We recommend setting `temperature` to `0.6`, `top_p` to `0.95` , and increase `max_new_tokens` to `1024` or higher  for reasoning True.

### **Use it with TensorRT-LLM**

The snippet below shows how to use this model with TensorRT-LLM. We tested this on the following [commit](https://github.com/NVIDIA/TensorRT-LLM/tree/46c5a564446673cdd0f56bcda938d53025b6d04e) and followed these [instructions](https://github.com/NVIDIA/TensorRT-LLM/blob/46c5a564446673cdd0f56bcda938d53025b6d04e/docs/source/installation/build-from-source-linux.md#option-2-build-tensorrt-llm-step-by-step) to build and install TensorRT-LLM in a docker container.

```
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig
from transformers import AutoTokenizer
pytorch_config = PyTorchConfig(
    disable_overlap_scheduler=True, enable_trtllm_decoder=True
)
kv_cache_config = KvCacheConfig(
    enable_block_reuse=False,
)
```
For **BF16** Checkpoint use:

```
model_id = "nvidia/Nemotron-H-47B-Reasoning-128K"
```
For **FP8** Checkpoint use:

```
model_id = "nvidia/Nemotron-H-47B-Reasoning-128K-FP8"
```

The rest of the inference remains the same for BF16 and FP8

```
tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = LLM(
    model=model_id,
    max_seq_len=32678,
    max_batch_size=4,
    pytorch_backend_config=pytorch_config,
    kv_cache_config=kv_cache_config,
    tensor_parallel_size=8,
)
messages = [
    {"role": "system",  "content": "{'reasoning': True}"},
    {"role": "user", "content": "Write a haiku about GPUs"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.6,
    top_p=0.95,
    add_special_tokens=False,
)
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

## Model Version

- v1.0

## Prompt Format

We follow the jinja chat template provided below. This template conditionally adds `<think>\n` to the start of the Assistant response if `{'reasoning': True}` is found in the system prompt and adds `<think></think>` to the start of the Assistant response if `{'reasoning': False}` is found the system prompt. Thus enforcing reasoning on/off behavior. If neither `{'reasoning': True}` or `{'reasoning': False}` is found in the system prompt the model can choose either manner to respond. Note that the system prompt can contain other instructions apart from just `{'reasoning': True}` or `{'reasoning': False}`

```
{{ '<SPECIAL_10>System\n' }}{%- if messages and messages[0]['role'] == 'system' -%}{%- set sys_content = messages[0]['content'].strip() -%}{%- set sys_content = sys_content.replace(\"{'reasoning': True}\", '').replace(\"{'reasoning': False}\", '').lstrip('\n') -%}{{ sys_content + '\n' if sys_content else '' }}{%- endif -%}{% for message in (messages[1:] if messages[0]['role'] == 'system' else messages) %}{%- if message['role'] == 'user' -%}{{ '\n<SPECIAL_11>User\n' + message['content'].strip() }}{%- if loop.last -%}{{ '\n<SPECIAL_11>Assistant\n' }}{%- if messages[0]['role'] == 'system' -%}{%- if \"{'reasoning': True}\" in messages[0]['content'] -%}{{ '<think>\n' }}{%- elif \"{'reasoning': False}\" in messages[0]['content'] -%}{{ '<think></think>' }}{%- endif -%}{%- endif -%}{%- endif -%}{%- elif message['role'] == 'assistant' -%}{{ message['content'].strip() }}{%- endif -%}{%- endfor -%}
```

## Training, Testing, and Evaluation Datasets

The post-training corpus for Nemotron-H-47B-Reasoning-128K consists of English and multilingual text (German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese and English). Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including code, legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracies. For several of the domains listed above we used synthetic data, specifically reasoning traces, from DeepSeek R1.

**Data Collection for Training & Testing Datasets:** Hybrid: Automated, Human, Synthetic

**Data Labeling for Training & Testing Datasets:** Hybrid: Automated, Human, Synthetic

### Evaluation

### Benchmark Results (Reasoning On)

We evaluated our BF16 and FP8 models in **Reasoning-On** mode against [Llama-Nemotron Super V1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1), using the same prompt formats across all benchmarks. The FP8 version shows negligible performance difference compared to the BF16 model.


| Benchmark          | Llama-Nemotron Super(49B) V1 | Nemotron-H 47B Reasoning | Nemotron-H 47B Reasoning (FP8) |
|--------------------|--------------------------------------------------------|---------------------------|-------------------------------|
| AIME25             | 53.00%                                                 | 54.17%                    | 54.17%                        |
| MATH500            | 96.60%                                                 | 96.20%                    | 96.80%                   |
| GPQA               | 65.10%                                                 | 65.66%                | 64.65%                        |
| MBPP               | 90.20%                                                 | 91.80%                    | 91.80%                        |
| MBPP_PLUS          | 75.10%                                                 | 79.90%                    | 79.90%                        |
| LCB                | 41.22%                                                 | 50.18%                    | 53.41%                        |
| BFCL               | 72.68%                                                 | 72.95%                    | 73.21%                    |
| IFEVAL-Prompt      | 81.80%                                                 | 84.47%                | 84.47%                    |
| IFEVAL-Instruction | 87.40%                                                 | 89.69%                | 89.57%                        |
| Arena Hard         | 85.00%                                                 | 85.00%                    | -                             |

## Potential Known Risks for Usage

The model was trained on data that contains toxic language, unsafe content, and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

The model demonstrates weakness to indirect prompt injection via some encodings, including Base16, Hex/ASCII, and Braille, though it is more resilient than other similar models to injections using the more common Base64 vector.

## Inference

- Engines: HF, vLLM, TensorRT-LLM  
- Test Hardware NVIDIA H100-80GB

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

