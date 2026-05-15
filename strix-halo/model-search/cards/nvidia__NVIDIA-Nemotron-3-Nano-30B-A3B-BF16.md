---
library_name: transformers
license: other
license_name: nvidia-nemotron-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
pipeline_tag: text-generation
language:
  - en
  - es
  - fr
  - de
  - ja
  - it
tags:
- nvidia
- pytorch
datasets:
- nvidia/Nemotron-Pretraining-Code-v1
- nvidia/Nemotron-CC-v2
- nvidia/Nemotron-Pretraining-SFT-v1
- nvidia/Nemotron-CC-Math-v1
- nvidia/Nemotron-Pretraining-Code-v2
- nvidia/Nemotron-Pretraining-Specialized-v1
- nvidia/Nemotron-CC-v2.1
- nvidia/Nemotron-CC-Code-v1
- nvidia/Nemotron-Pretraining-Dataset-sample
- nvidia/Nemotron-Competitive-Programming-v1
- nvidia/Nemotron-Math-v2
- nvidia/Nemotron-Agentic-v1
- nvidia/Nemotron-Math-Proofs-v1
- nvidia/Nemotron-Instruction-Following-Chat-v1
- nvidia/Nemotron-Science-v1
- nvidia/Nemotron-3-Nano-RL-Training-Blend
track_downloads: true
---

# NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

<div align="center" style="line-height: 1;">
<a href="https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖Chat-Nemotron_3_Nano-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://arxiv.org/abs/2512.20848" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/📝Paper-Read Now!-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets" target="_blank" style="margin: 2px;">
    <img alt="Pre-Training Datasets" src="https://img.shields.io/badge/🗄️_Pre--Training_Datasets-Available_Here-76B900?logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://huggingface.co/collections/nvidia/nemotron-post-training-v3" target="_blank" style="margin: 2px;">
    <img alt="Post-Training Datasets" src="https://img.shields.io/badge/🗄️_Post--Training_Datasets-Available_Here-76B900?logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://developer.nvidia.com/nemotron" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/🏠Nemotron Developer Page-Learn More Here!-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
<a href="https://discord.gg/9xpKQtVvrk" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Discord-NVIDIA%20AI%20Developer-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-NVIDIA Open Model License-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


![](./accuracy_chart.png)

## Model Overview

**Model Developer:** NVIDIA Corporation

**Model Dates:**

September 2025 \- December 2025

**Data Freshness:**

* The post-training data has a cutoff date of November 28, 2025\.  
* The pre-training data has a cutoff date of June 25, 2025\.

## Description

Nemotron-3-Nano-30B-A3B-BF16 is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response. The model's reasoning capabilities can be configured through a flag in the chat template. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so, albeit with a slight decrease in accuracy for harder prompts that require reasoning. Conversely, allowing the model to generate reasoning traces first generally results in higher-quality final solutions to queries and tasks.

The model employs a hybrid Mixture-of-Experts (MoE) architecture, consisting of 23 Mamba-2 and MoE layers, along with 6 Attention layers. Each MoE layer includes 128 experts plus 1 shared expert, with 6 experts activated per token. The model has 3.5B active parameters and 30B parameters in total.

The supported languages include: English, German, Spanish, French, Italian, and Japanese. Improved using Qwen.

This model is ready for commercial use.

### What is Nemotron?

NVIDIA Nemotron™ is a family of open models with open weights, training data, and recipes, delivering leading efficiency and accuracy for building specialized AI agents.

To get started, you can use [our quickstart guide](#quick-start-guide) below.

## License/Terms of Use

Governing Terms: Use of this model is governed by the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).

### Reasoning Benchmark Evaluations

We evaluated our model on the following benchmarks:

| Task | NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | Qwen3-30B-A3B-Thinking-2507 | GPT-OSS-20B |
| ----- | :---- | :---- | :---- |
| **General Knowledge** |  |  |  |
| MMLU-Pro | 78.3 | **80.9** | 75.0 |
| **Reasoning** |  |  |  |
| AIME25 (no tools) | 89.1 | 85.0 | **91.7** |
| AIME25 (with tools) | **99.2** | \- | 98.7 |
| GPQA (no tools) | 73.0 | **73.4** | 71.5 |
| GPQA (with tools) | **75.0** | \- | 74.2 |
| LiveCodeBench (v6 2025-08–2025-05) | **68.3** | 66.0 | 61.0 |
| SciCode (subtask) | 33.3 | 33.0 | **34.0** |
| HLE (no tools) | 10.6 | 9.8 | **10.9** |
| HLE (with tools) | 15.5 | \- | **17.3** |
| MiniF2F pass@1 | **50.0** | 5.7 | 12.1 |
| MiniF2F pass@32 | **79.9** | 16.8 | 43.0 |
| **Agentic** |  |  |  |
| Terminal Bench (hard subset) | 8.5 | 5.0 | 6.0 |
| SWE-Bench (OpenHands) | **38.8** | 22.0 | 34.0 |
| TauBench V2 (Airline) | 48.0 | **58.0** | 38.0 |
| TauBench V2 (Retail) | 56.9 | **58.8** | 38.0 |
| TauBench V2 (Telecom) | 42.2 | 26.3 | **49.7** |
| TauBench V2 (Average) | **49.0** | 47.7 | 48.7 |
| BFCL v4 | **53.8** | 46.4\* | \- |
| **Chat & Instruction Following** |  |  |  |
| IFBench (prompt) | **71.5** | 51.0 | 65.0 |
| Scale AI Multi Challenge | 38.5 | **44.8** | 33.8 |
| Arena-Hard-V2 (Hard Prompt) | **72.1** | 49.6\* | 71.2\* |
| Arena-Hard-V2 (Creative Writing) |  63.2 | **66.0\*** | 25.9& |
| Arena-Hard-V2 (Average) | **67.7** | 57.8 | 48.6 |
| **Long Context** |  |  |  |
| AA-LCR | 35.9 | **59.0** | 34.0 |
| RULER-100@256k | **92.9** | 89.4 | \- |
| RULER-100@512k | **91.3** | 84.0 | \- |
| RULER-100@1M | **86.3** | 77.5 | \- |
| **Multilingual** |  |  |  |
| MMLU-ProX (avg over langs) | 59.5 | **77.6\*** | 69.1\* |
| WMT24++ (en-\>xx) | **86.2** | 85.6 | 83.2 |

All evaluation results were collected via [Nemo Evaluator SDK](https://github.com/NVIDIA-NeMo/Evaluator) and [Nemo Skills](https://github.com/NVIDIA-NeMo/Skills). The open source container on Nemo Skills packaged via NVIDIA’s Nemo Evaluator SDK used for evaluations can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills?version=25.11). In addition to Nemo Skills, the evaluations also used dedicated packaged containers for Tau-2 Bench, ArenaHard v2, AA_LCR. A reproducibility tutorial along with all configs can be found in [Nemo Evaluator SDK examples](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/nemotron/nano-v3-reproducibility.md). The configs are also available in this HF repo [here](./nemo-evaluator-launcher-configs/local_nvidia_nemotron_3_nano_30b_a3b.yaml). \* denotes the accuracy numbers are measured by us.


### Deployment Geography: Global

### Use Case

NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 is a general purpose reasoning and chat model intended to be used in English and coding languages. Other non-English languages (English, Spanish, French, German, Japanese, Italian) are also supported. This model is intended to be used by developers designing AI Agent systems, chatbots, RAG systems, and other AI-powered applications. This model is also suitable for typical instruction-following tasks.

### Release Date

December 15, 2025 via [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)

## Reference(s)

* [NVIDIA Nemotron 3 model family on Hugging Face](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3)  
* [NVIDIA Nemotron 2 model family on Hugging Face](https://huggingface.co/collections/nvidia/nvidia-nemotron-v2)  
* [Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2512.20848)
* [NVIDIA Nemotron 3 White Paper](https://arxiv.org/abs/2512.20856)

## Model Architecture

- **Architecture Type:** Mamba2-Transformer Hybrid Mixture of Experts (MoE)  
- **Network Architecture:** Nemotron Hybrid MoE  
- **Number of model parameters:** 30B

## Model Design

The model was trained with 25T tokens, with a batch size of 3072, and used the Warmup-Stable-Decay (WSD) learning rate schedule with 8B tokens of learning rate warm up, peak learning rate of 1e-3 and minimum learning rate of 1e-5. There are a total of 52 layers, of which there are 23 of each MoE and Mamba-2 and the remaining 6 layers use grouped query attention (GQA) with 2 groups. Each MoE layer includes 128 routed experts plus 1 shared expert, with 6 experts activated per token.

## Training Methodology

Stage 1: Pre-Training

* [NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) model was pre-trained using crawled and synthetic code, math, science, and general knowledge data. All datasets are disclosed in the [Training, Testing, and Evaluation Datasets](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16#training-testing-and-evaluation-datasets) section of this document. Major portions of the pre-training corpus are released in the [Nemotron-Pre-Training-Datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) collection.  
* Software used for pre-training: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

Stage 2: Supervised Fine-Tuning

* The model was further fine-tuned on synthetic code, math, science, tool calling, instruction following, structured outputs, and general knowledge data. All datasets are disclosed in the [Training, Testing, and Evaluation Datasets](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16#training-testing-and-evaluation-datasets) section of this document. Major portions of the fine-tuning corpus are released in the [Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection.  [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) is one of the libraries used to prepare these corpora.
* 

Stage 3: Reinforcement Learning

* The model underwent multi-environment reinforcement learning using synchronous GRPO (Group Relative Policy Optimization) across math, code, science, instruction following, multi-step tool use, multi-turn conversations, and structured output environments. Conversational quality was further refined through RLHF using a [generative reward model](https://huggingface.co/nvidia/Qwen3-Nemotron-235B-A22B-GenRM). All datasets are disclosed in the *Training, Testing, and Evaluation Datasets* section of this document. The RL environments and datasets are released as part of [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).  
* Software used for reinforcement learning: [NeMo RL](https://github.com/NVIDIA-NeMo/RL), [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 model is a result of the above work.

The end-to-end training recipe is available in the [NVIDIA Nemotron Developer Repository](https://github.com/NVIDIA-NeMo/Nemotron). Evaluation results can be replicated using the [NeMo Evaluator SDK](https://github.com/NVIDIA-NeMo/Evaluator). [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) is one of the libraries used to prepare the pre and post training datasets. More details on the datasets and synthetic data generation methods can be found in the technical report [NVIDIA Nemotron 3 Nano](https://arxiv.org/abs/2512.20848).

## Input

- **Input Type(s):** Text

- **Input Format(s):** String  
    
- **Input Parameters:** One-Dimensional (1D): Sequences  
    
- **Maximum input size:** 1M tokens  
    
- **Other Properties Related to Input:** Supported languages include: English, Spanish, French, German, Japanese, Italian

## Output

- **Output Type(s):** Text  
    
- **Output Format:** String  
    
- **Output Parameters:** One-Dimensional (1D): Sequences  
    
- **Maximum output size:** 1M tokens

Our AI models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

- Runtime Engine(s): NeMo 25.11.01  
- Supported Hardware Microarchitecture Compatibility: NVIDIA H100-80GB, NVIDIA A100  
- Operating System(s): Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Quick Start Guide

### Use it with Transformers

The snippet below shows how to use this model with Huggingface Transformers (tested on version 4.57.3). We recommend using [NeMo Framework 25.11.01](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags?version=25.11.01) to ensure all required libraries are available.

Please note that the model supports up to a 1M context size, although the default context size in the Hugging Face configuration is 256k due to higher VRAM requirements. 

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
```

```
messages = [
    {"role": "user", "content": "Write a haiku about GPUs"},
]

tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    tokenized_chat,
    max_new_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))
```

`temperature=1.0` and `top_p=1.0` are recommended for reasoning tasks, while `temperature=0.6` and `top_p=0.95` are recommended for tool calling. 

If you’d like to use reasoning off, add `enable_thinking=False` to `apply_chat_template()`. By default, `enable_thinking` is set to be `True`.

```

tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Use Greedy Search for reasoning off
outputs = model.generate(
    tokenized_chat,
    max_new_tokens=32,
    do_sample=False,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))
```

### Use it with vLLM

For more detailed information on how to use the model with vLLM, please see [this cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm\_cookbook.ipynb).
If you are on Jetson Thor or DGX Spark, please use [this vllm container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=25.12.post1-py3).

```
pip install -U "vllm>=0.12.0"
```

Download the custom parser from the Hugging Face repository.

```
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
```

Launch a vLLM server using the custom parser. 

```
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name model \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

In the example above, we use a context length of 256k. You can increase the context size up to 1M to support longer contexts.
To enable this, set the `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` environment variable as shown below:

```
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name model \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 1M \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

Here is an example client code for vLLM. By default, the endpoint has reasoning enabled. We recommend setting a high value (e.g., 10,000) for `max_tokens`.

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "model",
        "messages":[{"role": "user", "content": "Write a haiku about GPUs"}],
        "max_tokens": 10000
    }'
```

If you’d like to use reasoning off with vLLM, you can do the following:  
vLLM OpenAI curl request:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "model",
        "messages":[{"role": "user", "content": "Write a haiku about GPUs"}],
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

vLLM OpenAI client:

```py
response = client.chat.completions.create(model=model, messages=messages, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

### Use it with TRT-LLM

For more detailed information on how to use the model with TRT-LLM, please see [this cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/trtllm\_cookbook.ipynb).

```
# nano_v3 example yaml is https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/auto_deploy/nano_v3.yaml
trtllm-serve <model_path> \
--backend _autodeploy \
--trust_remote_code \
--reasoning_parser nano-v3 \
--tool_parser qwen3_coder \
--extra_llm_api_options nano_v3.yaml
```

### Use it with SGLang

For more detailed information on how to use the model with SGLang, please see [this cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/sglang\_cookbook.ipynb).

```
python3 -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --trust-remote-code \
  --tp 1 \
  --attention-backend flashinfer \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nano_v3
```

#### Using Budget Control

The thinking budget allows developers to keep accuracy high and meet response‑time targets \- which is especially crucial for customer support, autonomous agent steps, and edge devices where every millisecond counts.

With budget control, you can set a limit for internal reasoning:

* `reasoning_budget`: This is a threshold that will attempt to end the reasoning trace at the next newline encountered in the reasoning trace. If no newline is encountered within 500 tokens, it will abruptly end the reasoning trace at `reasoning_budget + 500`.

> NOTE: This client will work with any OpenAI API compatible endpoint.

Client for supporting budget control:

```py
from typing import Any, Dict, List

import openai
from transformers import AutoTokenizer


class ThinkingBudgetClient:
   def __init__(self, base_url: str, api_key: str, tokenizer_name_or_path: str):
       self.base_url = base_url
       self.api_key = api_key
       self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
       self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)


   def chat_completion(
       self,
       model: str,
       messages: List[Dict[str, Any]],
       reasoning_budget: int = 512,
       max_tokens: int = 1024,
       **kwargs,
   ) -> Dict[str, Any]:
       assert (
           max_tokens > reasoning_budget
       ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {reasoning_budget=}"


       # 1. first call chat completion to get reasoning content
       response = self.client.chat.completions.create(
           model=model, messages=messages, max_tokens=reasoning_budget, **kwargs
       )
       content = response.choices[0].message.content


       reasoning_content = content
       if not "</think>" in reasoning_content:
           # reasoning content is too long, closed with a period (.)
           reasoning_content = f"{reasoning_content}.\n</think>\n\n"
       reasoning_tokens_len = len(
           self.tokenizer.encode(reasoning_content, add_special_tokens=False)
       )
       remaining_tokens = max_tokens - reasoning_tokens_len
       assert (
           remaining_tokens > 0
       ), f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the reasoning_budget."


       # 2. append reasoning content to messages and call completion
       messages.append({"role": "assistant", "content": reasoning_content})
       prompt = self.tokenizer.apply_chat_template(
           messages,
           tokenize=False,
           continue_final_message=True,
       )
       response = self.client.completions.create(
           model=model, prompt=prompt, max_tokens=remaining_tokens, **kwargs
       )


       response_data = {
           "reasoning_content": reasoning_content.strip().strip("</think>").strip(),
           "content": response.choices[0].text,
           "finish_reason": response.choices[0].finish_reason,
       }
       return response_data
```

Calling the server with a budget (Restricted to 32 tokens here as an example)

```py
tokenizer_name_or_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
client = ThinkingBudgetClient(
   base_url="http://localhost:8000/v1",  # Nemotron 3 Nano deployed in thinking mode
   api_key="EMPTY",
   tokenizer_name_or_path=tokenizer_name_or_path,
)


result = client.chat_completion(
   model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
   messages=[
       {"role": "system", "content": "You are a helpful assistant. /think"},
       {"role": "user", "content": "What is 2+2?"},
   ],
   reasoning_budget=32,
   max_tokens=512,
   temperature=1.0,
   top_p=1.0,
)
print(result)
```

You should see output similar to the following:

```
{'reasoning_content': "Okay, the user asked, What is 2+2? Let me think. Well, 2 plus 2 equals 4. That's a basic.", 'content': '2 + 2 equals **4**.\n', 'finish_reason': 'stop'}
```

## Model Version(s)

- v1.0

# Training, Testing, and Evaluation Datasets

**Data Modality:**  Text  
**The total size:**  10,648,823,153,919 Tokens  
**Total number of datasets:** 141  
**Dataset partition:** *Training \[100%\], testing \[0%\], validation \[0%\]*  
**Time period for training data collection:** 2013 to May 1, 2025  
**Time period for testing data collection:** 2013 to May 1, 2025  
**Time period for validation data collection:** 2013 to May 1, 2025  
**Data Collection Method by dataset:** Hybrid: Automated, Human, Synthetic  
**Labeling Method by dataset: Hybrid:** Automated, Human, Synthetic

NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 is pre-trained on a large corpus of high-quality curated and synthetically-generated data. It is trained in the English language, as well as 19 other languages and 43 programming languages. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracy. The model was trained for approximately 25 trillion tokens.

The post-training corpus for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 of high-quality curated and synthetically-generated data. Primary languages used for post-training include English, German, Spanish, French, Italian, and Japanese.

These datasets, such as FinePDFs, EssentialWeb, HotpotQA, SQuAD, and HelpSteer3, do not collectively or exhaustively represent all demographic groups (and proportionally therein). For instance, these datasets do not contain explicit mentions of demographic classes such as age, gender, or ethnicity in 64-99% of samples, depending on the source. In the subset where such terms are present, document-based datasets (FinePDFs and EssentialWeb) contain representational skews, such as references to "male" outnumbering those to "female", and mentions of "White" as the most frequent among ethnic identifiers (comprising 43-44% of ethnicity mentions). To mitigate these imbalances, we recommend considering evaluation techniques such as bias audits, fine-tuning with demographically balanced datasets, and mitigation strategies like counterfactual data augmentation to align with the desired model behavior. This evaluation used a 3,000-sample subset per dataset, identified as the optimal threshold for maximizing embedder accuracy.

During post-training, we generate synthetic data by distilling trajectories, solutions, and translations from strong teacher models and agent systems, often grounded in real tasks or documents and aggressively filtered for quality. For math, code, and science, we start from curated problem sets and use open source permissive models such as GPT-OSS-120B to produce step-by-step reasoning traces, candidate solutions, best-of-n selection traces, and verified CUDA kernels. For long-context and science, we build synthetic QA and reasoning data by retrieving passages from long documents, generating MCQ/OpenQA questions and answers, and paraphrasing them into multiple prompt/response formats to ensure diversity. Across all pipelines we stack automated verification—compilers, numerical checks, language identification—to ensure our data is high quality.

For all domains, we apply a unified data filtering pipeline to ensure that only high-quality, license-compliant, and verifiable samples are used for post-training. We first discard malformed examples using structural checks (e.g., missing tool definitions when tool calls are present). We then aggressively filter reasoning traces exhibiting pathological repetition, such as repeated n-grams within a sliding window or across the entire trajectory, which we found to be a strong indicator of malformed or low-quality reasoning. Finally, based on internal audits of synthetically generated datasets, we observed that some teacher models occasionally produce reasoning traces and final responses that implicitly align with specific political entities or promote nationalistic narratives. To mitigate this, we apply targeted keyword- and regex-based filters and remove all trajectories matching such behavior.

Alongside the model, we release our final [pre-training](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) and [post-training](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) data, as outlined in this section. For ease of analysis, there is a sample set that is ungated. For all remaining code, math and multilingual data, gating and approval is required, and the dataset is permissively licensed for model training purposes.

More details on the datasets and synthetic data generation methods can be found in the technical report [NVIDIA Nemotron 3 Nano](https://arxiv.org/abs/2512.20848).

| Dataset | Collection Period |
| :---- | :---- |
| [GSM8K](https://github.com/openai/grade-school-math) | 4/23/2025 |
| [CC-NEWS](https://commoncrawl.org/blog/news-dataset-available) | 4/23/2025 |
| [Common Crawl](https://commoncrawl.org/) | 4/23/2025 |
| [Wikimedia](https://dumps.wikimedia.org/) | 4/23/2025 |
| [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | 4/23/2025 |
| [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k) | 4/23/2025 |
| [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 4/23/2025 |
| [APIGen Function-Calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 4/23/2025 |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 4/23/2025 |
| [Open Textbook Library \- CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) and [OpenStax \- CC BY-SA subset](https://openstax.org/) | 4/23/2025 |
| [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb), [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k), [PRM800K](https://github.com/openai/prm800k), and [SciBench](https://github.com/mandyyyyii/scibench) | 4/23/2025 |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 4/23/2025 |
| [Court Listener](https://www.courtlistener.com/help/api/bulk-data/) | Legacy Download |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | Legacy Download |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Legacy Download |
| [BioRxiv](https://www.biorxiv.org/tdm) | Legacy Download |
| [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) | Legacy Download |
| [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/) | Legacy Download |
| [Stack Exchange Data Dump](https://archive.org/details/stackexchange) | Legacy Download |
| [PubMed Abstracts](https://github.com/thoppe/The-Pile-PubMed) | Legacy Download |
| [NIH ExPorter](https://exporter.nih.gov/ExPORTER_Catalog.aspx) | Legacy Download |
| [arXiv](https://info.arxiv.org/help/bulk_data/index.html) | Legacy Download |
| [BigScience Workshop Datasets](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#datasets) | Legacy Download |
| [Reddit Dataset](https://files.pushshift.io/reddit/) | Legacy Download |
| [SEC's Electronic Data Gathering, Analysis, and Retrieval (EDGAR)](https://www.sec.gov/search-filings) | Legacy Download |
| [Advanced Mathematical Problem Solving](https://github.com/hendrycks/math?tab=readme-ov-file) | Legacy Download |
| [MathPile](https://github.com/GAIR-NLP/MathPile/) | Legacy Download |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Legacy Download |
| [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/) | Legacy Download |
| [FLAN](https://github.com/google-research/FLAN) | Legacy Download |
| [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb) | Legacy Download |
| [SciBench](https://github.com/mandyyyyii/scibench) | Legacy Download |
| [WikiTableQuestions](https://huggingface.co/datasets/wikitablequestions) | Legacy Download |
| [FinQA](https://finqasite.github.io/) | Legacy Download |
| [Riddles](https://github.com/crawsome/riddles) | Legacy Download |
| [Problems in Elementary Mathematics for Home Study](https://archive.org/details/AntonovVygodskyNikitinSankinProblemsInElementaryMathematicsForHomeStudyMir1982) | Legacy Download |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Legacy Download |
| [Cosmos QA](https://huggingface.co/datasets/allenai/cosmos_qa) | Legacy Download |
| [MCTest](https://huggingface.co/datasets/sagnikrayc/mctest) | Legacy Download |
| [AI2's Reasoning Challenge](https://huggingface.co/datasets/ai2_arc) | Legacy Download |
| [OpenBookQA](https://github.com/allenai/OpenBookQA) | Legacy Download |
| [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | Legacy Download |
| [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101) | Legacy Download |
| [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | Legacy Download |
| [The Common Pile v0.1](https://huggingface.co/common-pile) | Legacy Download |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | Legacy Download |
| [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath) | Legacy Download |
| [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath) | Legacy Download |
| [MultiverseMathHard](https://huggingface.co/datasets/Nexusflow/MultiverseMathHard) | 10/2/2025 |
| [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | 10/2/2025 |
| [WorkBench](https://github.com/olly-styles/WorkBench/tree/main/data/raw) | 10/2/2025 |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 10/2/2025 |
| [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | 10/2/2025 |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | 10/2/2025 |
| [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) | 10/2/2025 |
| [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) | 10/2/2025 |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 10/2/2025 |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | 10/2/2025 |
| [simple-arithmetic-problems](https://huggingface.co/datasets/garrethlee/simple-arithmetic-problems) | 10/2/2025 |
| [arithmetic](https://huggingface.co/datasets/EleutherAI/arithmetic) | 10/2/2025 |
| [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) | 10/2/2025 |
| [News Commentary](https://opus.nlpl.eu/News-Commentary.php) | 10/2/2025 |
| [FastChat](https://github.com/lm-sys/FastChat/blob/main/playground/data/dummy.json) | 10/2/2025 |
| [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | 10/2/2025 |
| [finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | 10/2/2025 |
| [HotpotQA](https://huggingface.co/hotpot_qa/datasets) | 10/2/2025 |
| [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) | 10/2/2025 |
| [NLTK Words Lists](https://www.nltk.org/nltk_data/) | 10/2/2025 |

## Private Non-publicly Accessible Datasets of Third Parties

| Dataset |
| :---- |
| Global Regulation |
| TAUS Translation Memory |
| Scale HLE |
| HackerRank Coding |

## Private Non-publicly Accessible Datasets by NVIDIA

| Dataset |
| :---- |
| Simple Minesweeper |
| Simple Sudoku |
| Multitool Typewriter Hard |
| Machine Translation of News Commentary and TAUS Translation Memory |
| Machine Translation of STEM data using [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |

## Crawled and Scraped from Online Sources by NVIDIA

The English Common Crawl data was downloaded from the Common Crawl Foundation (see their FAQ for details on their crawling) and includes the snapshots CC-MAIN-2013-20 through CC-MAIN-2025-13. The data was subsequently deduplicated and filtered in various ways described in the Nemotron-CC paper. Additionally, we extracted data for fifteen languages from the following three Common Crawl snapshots: CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18. The fifteen languages included were Arabic, Chinese, Danish, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, and Thai. As we did not have reliable multilingual model-based quality classifiers available, we applied just heuristic filtering instead—similar to what we did for lower quality English data in the Nemotron-CC pipeline, but selectively removing some filters for some languages that did not work well. Deduplication was done in the same way as for Nemotron-CC.

The GitHub Crawl was collected using the GitHub REST API and the Amazon S3 API. Each crawl was operated in accordance with the rate limits set by its respective source, either GitHub or S3. We collect raw source code and subsequently remove any having a license which does not exist in our permissive-license set (for additional details, refer to the [technical report](https://arxiv.org/abs/2512.20848)).

| Dataset | Modality | Dataset Size | Collection Period | Collecting Organisation |
| :---- | :---- | :---- | :---- | :---- |
| English Common Crawl | Text | 3.36T | 4/8/2025 | NVIDIA Advanced Deep Learning Research |
| English Common Crawl 1.1 | Text | Not disclosed | 10/2/2025 | NVIDIA Advanced Deep Learning Research |
| Multilingual Common Crawl | Text | 812.7B | 5/1/2025 | NVIDIA Advanced Deep Learning Research |
| GitHub Crawl | Text | 747.4B | 4/29/2025 | NVIDIA Advanced Deep Learning Research |

## NVIDIA-Sourced Synthetic Datasets

| Dataset | Modality | Dataset Size | Seed Dataset | Model(s) used for generation |
| :---- | :---- | :---- | :---- | :---- |
| Synthetic Art of Problem Solving from DeepSeek-R1 | Text | 40B | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Moral Stories and Social Chemistry from Mixtral-8x22B-v0.1 | Text | 327M | [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101); [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| Synthetic Social Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 83.6M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic Health Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 9.7M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic STEM seeded with OpenStax, Open Textbook Library, and GSM8K from DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-0324, and Qwen2.5-72B | Text | 175M | [OpenStax \- CC BY-SA subset](https://openstax.org/); [GSM8K](https://github.com/openai/grade-school-math); [Open Textbook Library \- CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| [Nemotron-PrismMath](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath) | Text | 4.6B | [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified); [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | [Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct); [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| Synthetic Question Answering Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text | 350M | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Refreshed [Nemotron-MIND](https://huggingface.co/datasets/nvidia/Nemotron-MIND) from phi-4 | Text | 73B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Nemotron-CC-Math-4plus | Text | 52.3B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Nemotron-CC-Math-3 | Text | 80.9B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from DeepSeek-V3 and DeepSeek-V3-0324 | Text | 4.0B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from Qwen3-30B-A3B | Text | 4.2B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Art of Problem Solving from Qwen2.5-32B-Instruct, Qwen2.5-Math-72B, Qwen2.5-Math-7B, and Qwen2.5-72B-Instruct | Text |  | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); [GSM8K](https://github.com/openai/grade-school-math); [PRM800K](https://github.com/openai/prm800k) | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [Qwen2.5-Math-72B](https://huggingface.co/Qwen/Qwen2.5-Math-72B); [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B); [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic MMLU Auxiliary Train from DeepSeek-R1 | Text | 0.5B | [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Long Context Continued Post-Training Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text |  | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic Common Crawl from Qwen3-30B-A3B and Mistral-Nemo-12B-Instruct | Text | 415.8B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct) |
| Synthetic Multilingual Data from Common Crawl from Qwen3-30B-A3B | Text |  | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Multilingual Data from Wikimedia from Qwen3-30B-A3B | Text |  | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Math Data from Wikimedia from Nemotron-4-340B-Instruct | Text |  | \- | [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) |
| Synthetic Common Crawl Code from phi-4 | Text | 427.9B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Scientific Coding from Qwen3-235B-A22B | Text | 1.2B | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Tool Calling Data | Text | 26.2B |  | [Qwen3-235B-A22B-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Essential-Web from QwQ-32B | Text | 28.1B | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| Translated Synthetic Crawl | Text | 389.9B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Translated Synthetic Wikipedia | Text | 7.9B | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Art of Problem Solving from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic Stack Exchange from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic OpenCodeReasoning from DeepSeek-R1-0528 | Text | Undisclosed | [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic HackerRank Coding from DeepSeek-R1-0528 | Text | Undisclosed | HackerRank Coding Dataset | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic SWE-Gym from Qwen3-Coder-480B-A35B-Instruct | Text | Undisclosed | [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | [Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) |
| Synthetic Art of Problem Solving and Stack Exchange from gpt-oss-120b, Qwen2.5-32B-Instruct, and Goedel-Prover-V2-32B | Text | Undisclosed | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); [Stack Exchange](https://archive.org/details/stackexchange) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [Goedel-Prover-V2-32B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) |
| Synthetic Multilingual Science and Code data from DeepSeek-R1, DeepSeek-R1-0528, Qwen2.5-32B-Instruct, and Qwen3-235B-A22B, translated with Qwen2.5-32B-Instruct and Qwen2.5-14B-Instruct | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange); [SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K); [LIMO](https://huggingface.co/datasets/GAIR/LIMO); [TACO](https://huggingface.co/datasets/BAAI/TACO); Code Contest; Codeforces | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); |
| Synthetic Safety from DeepSeek-R1-0528, gpt-oss-120b and Mixtral-8x7B-v0.1 | Text | Undisclosed | [Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0); [Gretel Synthetic Safety Alignment Dataset](https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1); [RedTeam-2K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k); [Malicious Tasks](https://github.com/CrystalEye42/eval-safety/blob/main/malicious_tasks_dataset.yaml); [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) |
| Synthetic STEM from Qwen3-235B-A22B-Instruct-2507 and gpt-oss-120b | Text | Undisclosed | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic KernelBook from DeepSeek-R1-0528 | Text | Undisclosed | [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Tool Calling from Qwen3-235B-A22B-Thinking-2507 and Qwen3-Next-80B-A3B-Thinking | Text | Undisclosed | [ToolBench](https://github.com/OpenBMB/ToolBench/tree/master); [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2); [APIGen Function-Calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k); [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507); [Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) |
| Synthetic Chat from gpt-oss-120b, Mixtral-8x22B-Instruct-v0.1, Qwen3-235B-A22B-Instruct-2507 , and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [C4](https://huggingface.co/datasets/allenai/c4); [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m); [ShareGPT](https://huggingface.co/datasets/RyokoAI/ShareGPT52K); [GSM8K](https://github.com/openai/grade-school-math); [PRM800K](https://github.com/openai/prm800k); [FinQA](https://finqasite.github.io/); [WikiTableQuestions](https://huggingface.co/wikitablequestions/datasets); [Riddles](https://github.com/crawsome/riddles); [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2); [SciBench](https://huggingface.co/datasets/xw27/scibench); [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k); [OpenBookQA](https://github.com/allenai/OpenBookQA); [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb); Software Heritage; [Khan Academy Math Keywords](https://www.khanacademy.org/math); [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M); [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic Long Context from Qwen3-235B-A22B-Instruct-2507 | Text | Undisclosed | [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Synthetic Tool Use Interactive Agent from gpt-oss-120b, DeepSeek-R1-0528, Qwen3-32B, and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | NVIDIA Internal | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528); [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B); and [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic STEM from Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [ICHO-IPH0](https://huggingface.co/datasets/II-Vietnam/IChO-IPhO-RL-v2-formated); [Physics Big](https://huggingface.co/datasets/Vikhrmodels/physics_big); Scale HLE; [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning); [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic DocFinQA and SWE-smith from Qwen3-Coder-480B-A35B-Instruct and Kimi-K2-Thinking | Text | Undisclosed | [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA); [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) | [Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct); [Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) |
| Synthetic Math from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | \- | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic Essential-Web from gpt-oss-120b | Text | Undisclosed | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Scale HLE from gpt-oss-120b | Text | Undisclosed | Scale HLE | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic CDQuestions from gpt-oss-120b | Text | Undisclosed | [CDQuestions](https://cdquestions.com/) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Stack Exchange from gpt-oss-120b | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic GPQA from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic Vedantu from gpt-oss-120b | Text | Undisclosed | [Vedantu](https://www.vedantu.com/) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic SWE-Gym and R2E-Gym-Subset from Qwen3-Coder-480B-A35B-Instruct | Text | Undisclosed | [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym); [R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) | [Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) |
| Synthetic SWE-Gym from Qwen3-Coder-480B-A35B-Instruct | Text | Undisclosed | [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | [Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) |
| Synthetic SWE-Gym and R2E-Gym-Subset from DeepSeek-R1-0528 | Text | Undisclosed | [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym); [R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic HelpSteer, LMSYS-Chat-1M, and Nemotron-Personas-USA from gpt-oss-120b, Qwen3-235B-A22B-Instruct-2507, and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2); [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3); [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m); [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic Structured Outputs from Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Thinking-2507, Qwen3-235B-A22B-Instruct-2507, and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | \- | [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507); [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic Search STEM MCQ from Qwen3-235B-A22B and DeepSeek-R1-0528 | Text | Undisclosed | \- | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Search STEM OPENQ from DeepSeek-R1-0528 | Text | Undisclosed | \- | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic OpenSTEM from Qwen2.5-32B-Instruct and DeepSeek-R1-0528 | Text | Undisclosed | \- | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic MCQ from Qwen2.5-32B-Instruct and DeepSeek-R1-0528 | Text | Undisclosed | \- | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic MCQ10 from DeepSeek-R1-0528 | Text | Undisclosed | \- | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic MCQ4 from Qwen3-235B-A22B, DeepSeek-R1-0528, and Qwen3-235B-A22B-Instruct-2507 | Text | Undisclosed | \- | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Synthetic OpenMathReasoning from gpt-oss-120b and Qwen2.5-32B-Instruct | Text | Undisclosed | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Synthetic Offline Search MCQA HLE from DeepSeek-R1-0528 | Text | Undisclosed | \- | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Offline Search MCQA GPQA from Qwen3-235B-A22B and DeepSeek-R1-0528 | Text | Undisclosed | \- | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Human Preference from QwQ-32B, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-235B-A22B-Instruct-2507, Mistral-Small-3.1-24B-Instruct-2503, Mistral-Small-3.2-24B-Instruct-2506, MiniMax-M1-80k, MiniMax-M1-40k, Kimi-K2-Instruct, DeepSeek-V3-0324, DeepSeek-R1-0528 | Text | Undisclosed | \- | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B); [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503); [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506); [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k); [MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k); [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic WildChat-1M and arena-human-preference-140k from DeepSeek-R1, gemma-2-2b-it, gemma-3-27b-it, gpt-oss-20b, gpt-oss-120b, Mistral-7B-Instruct-v0.3, Mixtral-8x22B-Instruct-v0.1, Nemotron-4-340B-Instruct, NVIDIA-Nemotron-Nano-9B-v2, Phi-4-mini-instruct, Phi-3-small-8k-instruct, Phi-3-medium-4k-instruct, Qwen3-235B-A22B, QwQ-32B | Text | Undisclosed | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M); [arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1); [gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it); [gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it); [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3); [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1); [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct); [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2); [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct); [Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct); [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| Synthetic Safety from DeepSeek-R1-0528, gpt-oss-120b, DeepSeek-R1-Distill-Qwen-7B, and Mixtral-8x7B-v0.1 | Text | Undisclosed | [Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0); [Gretel Synthetic Safety Alignment Dataset](https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1); [RedTeam-2K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k); [Malicious Tasks](https://github.com/CrystalEye42/eval-safety/blob/main/malicious_tasks_dataset.yaml); | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B); [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) |
| Synthetic Code from Qwen3-32B | Text | Undisclosed | English Common Crawl; English Common Crawl 1.1 | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Synthetic OpenCodeReasoning from DeepSeek-R1 | Text | Undisclosed | [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic LIMO from DeepSeek-R1-0528 | Text | Undisclosed | [LIMO](https://huggingface.co/datasets/GAIR/LIMO) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic SCP from DeepSeek-R1-0528 | Text | Undisclosed | [SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Stack Exchange from DeepSeek-R1-0528 | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Common Crawl from Qwen3-30B-A3B | Text | Undisclosed | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Wikipedia from Qwen3-30B-A3B | Text | Undisclosed | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Essential-Web from Qwen3-30B-A3B and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic Textbook Math from Qwen3-30B-A3B, Qwen3-235B-A22B, phi-4 | Text | Undisclosed | [Common Crawl](https://commoncrawl.org/); [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Math and Code from DeepSeek-R1 and DeepSeek-R1-0528 | Text | Undisclosed | [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K); [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2); [TACO](https://huggingface.co/datasets/BAAI/TACO); [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning); [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning); [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Nemotron-Personas-USA from gpt-oss-120b and Qwen3-8B | Text | Undisclosed | [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |

## Training Dataset

| Dataset | \# of Tokens in Nemotron Nano 2 | \# of Tokens in Nemotron 3 Nano |
| :---- | :---- | :---- |
| English Common Crawl | 3,360,110,334,818 | 3,456,523,212,210 |
| English Synthetic CC | 1,949,464,641,123 | 4,340,740,677,920 |
| Crawl++ | 360,389,153,262 | 360,389,153,262 |
| Math | 124,606,230,663 | 154,217,502,165 |
| Synthetic Math | 73,007,767,155 | 73,007,767,155 |
| Code | 747,409,228,724 | 1,043,856,922,136 |
| Synthetic Code | 175,067,553,293 | 453,117,917,176 |
| Common Crawl Code | 0 | 263,072,374,097 |
| English Wiki | 17,349,266,926 | 17,349,266,926 |
| Synthetic Wiki | 0 | 7,850,648,552 |
| Books | 0 | 0 |
| Papers | 191,586,493,365 | 191,586,493,365 |
| PDF-to-text | 141,096,578,533 | 141,096,578,533 |
| Code SFT | 60,025,726,817 | 102,863,752,325 |
| STEM SFT | 272,680,426,295 | 359,826,214,274 |
| General SFT | 6,057,478,645 | 6,057,478,645 |
| Tool-Calling SFT | 0 | 26,244,716,867 |
| Multilingual | 2,172,261,909,350 | 1,743,892,490,859 |
| Synthetic multilingual | 997,710,364,950 | 595,140,661,135 |
| **Total** | **10,648,823,153,919** | **13,336,833,827,602** |

We use a considerable amount of synthetic data. Out of 10.6 trillion tokens, 3,534,013,958,278 tokens are synthetically generated.

We extracted data for fifteen languages from the following three Common Crawl snapshots: CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18. The fifteen languages included were Arabic, Chinese, Danish, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, and Thai. As we did not have reliable multilingual model-based quality classifiers available, we applied just heuristic filtering instead—similar to what we did for lower quality English data in the Nemotron-CC pipeline, but selectively removing some filters for some languages that did not work well. Deduplication was done in the same way as for Nemotron-CC. Additionally, we used data from Wikipedia and FineWeb-2 (Penedo et al., 2025\) for these fifteen languages as well as four additional languages: Czech, Finnish, Hebrew, and Hindi.

| Language | Total Tokens |
| :---- | :---- |
| Arabic | 118,056,362,726 |
| Danish | 117,747,321,618 |
| German | 146,613,691,781 |
| Spanish | 469,156,575,409 |
| French | 139,982,002,289 |
| Italian | 298,858,370,174 |
| Japanese | 682,755,693,336 |
| Korean | 127,099,747,538 |
| Dutch | 89,041,592,681 |
| Polish | 105,356,493,147 |
| Portuguese | 243,249,275,089 |
| Russian | 185,314,014,057 |
| Swedish | 74,954,953,299 |
| Thai | 160,778,944,467 |
| Chinese | 211,007,236,689 |

We collect a total of 922,476,782,017 tokens of code in 43 different languages.

| Language | Tokens |
| :---- | :---- |
| Assembly | 750,628,764 |
| C | 42,657,300,868 |
| C\# | 56,153,329,307 |
| C++ | 67,773,701,658 |
| CommonLisp | 263,234,672 |
| CSS | 38,848,760,035 |
| Cuda | 400,222,993 |
| Dart | 3,816,960,470 |
| Dockerfile | 474,958,084 |
| Fortran | 1,105,049,387 |
| Go | 8,332,419,480 |
| Haskell | 1,294,613,669 |
| HTML | 69,082,117,487 |
| Java | 131,440,465,822 |
| JavaScript | 75,573,420,861 |
| JSON | 15,366,881,241 |
| Julia | 621,046,949 |
| JupyterNotebook | 2,241,893,197 |
| Lua | 4,146,420,802 |
| Makefile | 12,640,010,879 |
| Markdown | 64,796,743,311 |
| Mathematica | 320,504,225 |
| OmniversePython | 26,946,093 |
| Pascal | 1,625,013,876 |
| Perl | 1,575,314,434 |
| PHP | 61,575,339,005 |
| Python | 126,916,727,384 |
| R | 19,811,381,935 |
| reStructuredText | 1,779,876,391 |
| Ruby | 6,446,962,615 |
| Rust | 4,438,640,533 |
| Scala | 3,343,959,154 |
| Shell | 18,758,779,250 |
| SQL | 23,205,633,085 |
| Swift | 5,976,714,881 |
| SystemVerilog | 233,056,185 |
| TeX | 7,347,157,527 |
| TypeScript | 15,657,838,582 |
| Verilog | 811,884,369 |
| VHDL | 648,401,444 |
| VisualBasic.NET | 1,005,680,881 |
| XML | 12,616,779,741 |
| YAML | 10,574,010,491 |

## Language Distribution in Post-Training

For our post-training recipe, we focused on 5 main languages in addition to English: Spanish, French, Japanese, Italian, German.  
Those languages were represented in the form of multilingual reasoning and translation task.

The following table depicts our sample distribution for the 6 languages and 5 translation pairs.

| Language | Size |
| :---- | :---- |
| English | 16.2 M |
| Italian | 0.252M |
| German | 0.252M |
| Spanish | 0.252M |
| French | 0.252M |
| Japanese | 0.252M |
| English \<-\> Italian | 108k |
| English \<-\> German | 108k |
| English \<-\> Spanish | 108k |
| English \<-\> French | 108k |
| English \<-\> Japanese | 108k |

## Evaluation Dataset

* Data Collection Method by dataset: Hybrid: Human, Synthetic  
* Labeling Method by dataset: Hybrid: Automated, Human, Synthetic

## Inference

- Engines: HF, vLLM, TRT-LLM, SGLang, Llama.cpp
- Test Hardware: NVIDIA A100 80GB, H100 80GB, B200 192GB, RTX PRO 6000 96GB, Jetson Thor, DGX Spark


## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our [Trustworthy AI terms of service](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/), developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

We advise against circumvention of any provided safety guardrails contained in the Model without a substantially similar guardrail appropriate for your use case.For more details: [Safety](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/safety.md) and [Explainability](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/explainability.md) Subcards.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/bias.md), and [Privacy](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/privacy.md) Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Citation

```

@misc{nvidia_nemotron_nano_v3_2025,
  title  = {{Nemotron 3 Nano}: Open, Efficient Mixture-of-Experts Hybrid {Mamba}-{Transformer} Model for {Agentic} Reasoning},
  author = {{NVIDIA}},
  year   = {2025},
  url    = {https://arxiv.org/abs/2512.20848},
  note   = {Technical report}
}
```