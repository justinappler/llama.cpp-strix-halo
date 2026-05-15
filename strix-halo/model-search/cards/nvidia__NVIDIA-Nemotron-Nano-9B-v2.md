---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
pipeline_tag: text-generation
datasets:
- nvidia/Nemotron-Post-Training-Dataset-v1
- nvidia/Nemotron-Post-Training-Dataset-v2
- nvidia/Nemotron-Pretraining-Dataset-sample
- nvidia/Nemotron-CC-v2
- nvidia/Nemotron-CC-Math-v1
- nvidia/Nemotron-Pretraining-SFT-v1
language:
- en
- es
- fr
- de
- it
- ja
library_name: transformers
tags:
- nvidia
- pytorch
track_downloads: true
base_model:
- nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base
- nvidia/NVIDIA-Nemotron-Nano-12B-v2
---
# NVIDIA-Nemotron-Nano-9B-v2

![](./accuracy_chart.png)


**Model Developer:** NVIDIA Corporation

**Model Dates:**

June 2025 \- August 2025

**Data Freshness:**

September 2024

The pretraining data has a cutoff date of September 2024.

## Model Overview

NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response. The model's reasoning capabilities can be controlled via a system prompt. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so, albeit with a slight decrease in accuracy for harder prompts that require reasoning. Conversely, allowing the model to generate reasoning traces first generally results in higher-quality final solutions to queries and tasks.

The model uses a hybrid architecture consisting primarily of Mamba-2 and MLP layers combined with just four Attention layers. For the architecture, please refer to the [Nemotron-H tech report](https://arxiv.org/abs/2504.03624).
The model was trained using [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

The supported languages include: English, German, Spanish, French, Italian, and Japanese. Improved using Qwen.

This model is ready for commercial use.

## Feature Voting

We want to hear from you! Share your ideas, vote on what matters, and help [shape the future of Nemotron](https://nemotron.ideas.nvidia.com/).

## License/Terms of Use

Governing Terms: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Evaluation Results

### Benchmark Results (Reasoning On)

We evaluated our model in **Reasoning-On** mode across all benchmarks, except RULER, which is evaluated in **Reasoning-Off** mode.


| Benchmark | Qwen3-8B | NVIDIA-Nemotron-Nano-9B-v2 |
| :---- | ----: | ----: |
| AIME25 | 69.3% | 72.1% |
| MATH500 | 96.3% | 97.8% |
| GPQA | 59.6% | 64.0% |
| LCB | 59.5% | 71.1% |
| BFCL v3 | 66.3% | 66.9% |
| IFEval (Instruction Strict) | 89.4% | 90.3% |
| HLE | 4.4% | 6.5% |
| RULER (128K) | 74.1% | 78.9% |


All evaluations were done using [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills). We published a [tutorial](https://nvidia.github.io/NeMo-Skills/tutorials/2025/08/22/reproducing-nvidia-nemotron-nano-9b-v2-evals/) with all details necessary to reproduce our evaluation results.


## Reasoning Budget Control

This model supports runtime “thinking” budget control. During inference, the user can specify how many tokens the model is allowed to "think".

![](./acc-vs-budget.png)


## Model Architecture

- Architecture Type: Mamba2-Transformer Hybrid
- Network Architecture: Nemotron-Hybrid

### Deployment Geography: Global

### Use Case

NVIDIA-Nemotron-Nano-9B-v2 is a general purpose reasoning and chat model intended to be used in English and coding languages. Other non-English languages (German, French, Italian, Spanish and Japanese) are also supported. Developers designing AI Agent systems, chatbots, RAG systems, and other AI-powered applications. Also suitable for typical instruction-following tasks.

### Release Date: 08/18/2025

- Huggingface 08/18/2025 via https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
- API Catalog 08/18/2025 via https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2

## References

- [NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model](https://arxiv.org/abs/2508.14444)


## Input

- Input Type(s): Text
- Input Format(s): String
- Input Parameters: One-Dimensional (1D): Sequences
- Other Properties Related to Input: Context length up to 128K. Supported languages include German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese and English.

## Output

- Output Type(s): Text
- Output Format: String
- Output Parameters: One-Dimensional (1D): Sequences up to 128K

Our models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

- Runtime Engine(s): NeMo 25.07.nemotron-nano-v2
- Supported Hardware Microarchitecture Compatibility: NVIDIA A10G, NVIDIA H100-80GB, NVIDIA A100, Jetson AGX Thor
- Operating System(s): Linux


### **Use it with Transformers**

The snippet below shows how to use this model with Huggingface Transformers (tested on version 4.48.3).

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
```

Case 1: `/think` or no reasoning signal is provided in the system prompt, reasoning will be set to `True`

```
messages = [
    {"role": "system", "content": "/think"},
    {"role": "user", "content": "Write a haiku about GPUs"},
]
```

Case 2: `/no_think` is provided, reasoning will be set to `False`

```
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": "Write a haiku about GPUs"},
]
```

Note: `/think` or `/no_think` keywords can also be provided in “user” messages for turn-level reasoning control.

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

We recommend setting `temperature` to `0.6`, `top_p` to `0.95` for reasoning True and greedy search for reasoning False, and increase `max_new_tokens` to `1024` or higher for reasoning True. 

### **Use it with TRT-LLM**

The snippet below shows how to use this model with TRT-LLM. We tested this on the following [commit](https://github.com/NVIDIA/TensorRT-LLM/tree/46c5a564446673cdd0f56bcda938d53025b6d04e) and followed these [instructions](https://github.com/NVIDIA/TensorRT-LLM/blob/46c5a564446673cdd0f56bcda938d53025b6d04e/docs/source/installation/build-from-source-linux.md#option-2-build-tensorrt-llm-step-by-step) to build and install TRT-LLM in a docker container.

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

```
model_id = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
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
    {"role": "system",  "content": "/think"},
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

### **Use it with vLLM**

The snippet below shows how to use this model with vLLM. Use the latest version of vLLM and follow these instructions to build and install vLLM.

```shell
pip install -U "vllm>=0.10.1"
```

Now you can run run the server with:

```shell
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --trust-remote-code \
    --max-num-seqs 64 \
    --mamba_ssm_cache_dtype float32
```

Note:
- Remember to add \`--mamba\_ssm\_cache\_dtype float32\` for accurate quality. Without this option, the model’s accuracy may degrade.
- If you encounter a CUDA OOM issue, try `--max-num-seqs 64` and consider lower the value further if the error persists.



Alternativly, you can use Docker to launch a vLLM server.

```
export TP_SIZE=1  # Adjust this value based on the number of GPUs you want to use
docker run --runtime nvidia --gpus all \
           -v ~/.cache/huggingface:/root/.cache/huggingface \
           --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
           -p 8000:8000 \
           --ipc=host \
           vllm/vllm-openai:v0.10.1 \
           --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
           --tensor-parallel-size ${TP_SIZE} \
           --max-num-seqs 64 \
           --max-model-len 131072 \
           --trust-remote-code \
           --mamba_ssm_cache_dtype float32
```

For Jetson AGX Thor, please use [this vLLM container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=25.09-py3).


#### Using Budget Control with a vLLM Server

The thinking budget allows developers to keep accuracy high and meet response‑time targets \- which is especially crucial for customer support, autonomous agent steps, and edge devices where every millisecond counts.

With budget control, you can set a limit for internal reasoning:

* `max_thinking_tokens`: This is a threshold that will attempt to end the reasoning trace at the next newline encountered in the reasoning trace. If no newline is encountered within 500 tokens, it will abruptly end the reasoning trace at \`max\_thinking\_tokens \+ 500\`.

Start a vLLM server:

```shell
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32
```

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
       max_thinking_budget: int = 512,
       max_tokens: int = 1024,
       **kwargs,
   ) -> Dict[str, Any]:
       assert (
           max_tokens > max_thinking_budget
       ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {max_thinking_budget=}"


       # 1. first call chat completion to get reasoning content
       response = self.client.chat.completions.create(
           model=model, messages=messages, max_tokens=max_thinking_budget, **kwargs
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
       ), f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the max_thinking_budget."


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
tokenizer_name_or_path = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
client = ThinkingBudgetClient(
   base_url="http://localhost:8000/v1",  # Nano 9B v2 deployed in thinking mode
   api_key="EMPTY",
   tokenizer_name_or_path=tokenizer_name_or_path,
)


result = client.chat_completion(
   model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
   messages=[
       {"role": "system", "content": "You are a helpful assistant. /think"},
       {"role": "user", "content": "What is 2+2?"},
   ],
   max_thinking_budget=32,
   max_tokens=512,
   temperature=0.6,
   top_p=0.95,
)
print(result)
```

You should see output similar to the following:

```
{'reasoning_content': "Okay, the user asked, What is 2+2? Let me think. Well, 2 plus 2 equals 4. That's a basic.", 'content': '2 + 2 equals **4**.\n', 'finish_reason': 'stop'}
```

#### Using Tool-Calling with a vLLM Server

Start a vLLM server with native tool-calling:

```shell
git clone https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2

vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --trust-remote-code \
  --mamba_ssm_cache_dtype float32 \
  --enable-auto-tool-choice \
  --tool-parser-plugin "NVIDIA-Nemotron-Nano-9B-v2/nemotron_toolcall_parser_no_streaming.py" \
  --tool-call-parser "nemotron_json"
```

## After launching a vLLM server, you can call the server with tool-call support using a Python script like below:

```py
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:5000/v1",
    api_key="dummy",
)

completion = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": "My bill is $100. What will be the amount for 18% tip?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "calculate_tip",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bill_total": {
                            "type": "integer",
                            "description": "The total amount of the bill"
                        },
                        "tip_percentage": {
                            "type": "integer",
                            "description": "The percentage of tip to be applied"
                        }
                    },
                    "required": ["bill_total", "tip_percentage"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "integer",
                            "description": "The amount to be converted"
                        },
                        "from_currency": {
                            "type": "string",
                            "description": "The currency code to convert from"
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "The currency code to convert to"
                        }
                    },
                    "required": ["from_currency", "amount", "to_currency"]
                }
            }
        }
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=32768,
    stream=False
)

print(completion.choices[0].message.content)
print(completion.choices[0].message.tool_calls)
```

You should see output similar to the following:

```
<think>
Okay, let's see. The user has a bill of $100 and wants to know the amount for an 18% tip. Hmm, I need to calculate the tip based on the bill total and the percentage. The tools provided include calculate_tip, which takes bill_total and tip_percentage as parameters. So the bill_total here is 100, and the tip_percentage is 18. I should call the calculate_tip function with these values. Wait, do I need to check if the parameters are integers? The bill is $100, which is an integer, and 18% is also an integer. So that fits the function's requirements. I don't need to convert any currency here because the user is asking about a tip in the same currency. So the correct tool to use is calculate_tip with those parameters.
</think>

[ChatCompletionMessageToolCall(id='chatcmpl-tool-e341c6954d2c48c2a0e9071c7bdefd8b', function=Function(arguments='{"bill_total": 100, "tip_percentage": 18}', name='calculate_tip'), type='function')]
```

## Model Version

- v1.0

## Prompt Format

We follow the jinja chat template provided below. This template conditionally adds `<think>\n` to the start of the Assistant response if `/think` is found in either the system prompt or any user message. If no reasoning signal is added, the model defaults to reasoning "on" mode. The chat template adds `<think></think>` to the start of the Assistant response if `/no_think` is found in the system prompt. Thus enforcing reasoning on/off behavior.

```
{%- set ns = namespace(enable_thinking = true) %}

{%- for message in messages -%}
    {%- set content = message['content'] -%}
    {%- if message['role'] == 'user' or message['role'] == 'system' -%}
        {%- if '/think' in content -%}
            {%- set ns.enable_thinking = true -%}
        {%- elif '/no_think' in content -%}
            {%- set ns.enable_thinking = false -%}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}

{%- if messages[0]['role'] != 'system' -%}
    {%- set ns.non_tool_system_content = '' -%}
    {{- '<SPECIAL_10>System\n' -}}
{%- else -%}
    {%- set ns.non_tool_system_content = messages[0]['content']
        .replace('/think', '')
        .replace('/no_think', '')
        .strip()
    -%}
    {{- '<SPECIAL_10>System\n' + ns.non_tool_system_content }}
{%- endif -%}

{%- if tools -%}
    {%- if ns.non_tool_system_content is defined and ns.non_tool_system_content != '' -%}
        {{- '\n\n' -}}
    {%- endif -%}

    {{- 'You can use the following tools to assist the user if required:' -}}
    {{- '\n<AVAILABLE_TOOLS>[' -}}
    {%- for tool in tools -%}
        {{- (tool.function if tool.function is defined else tool) | tojson -}}
        {{- ', ' if not loop.last else '' -}}
    {%- endfor -%}
    {{- ']</AVAILABLE_TOOLS>\n\n' -}}

    {{- 'If you decide to call any tool(s), use the following format:\n' -}}
    {{- '<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, ' -}}
    {{- '{{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>\n\n' -}}

    {{- 'The user will execute tool-calls and return responses from tool(s) in this format:\n' -}}
    {{- '<TOOL_RESPONSE>[{{"tool_response1"}}, {{"tool_response2"}}]</TOOL_RESPONSE>\n\n' -}}

    {{- 'Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.' -}}
{%- endif -%}

{{- '\n' -}}

{%- set messages = messages[1:] if messages[0]['role'] == 'system' else messages -%}

{%- if messages[-1]['role'] == 'assistant' -%}
    {%- set ns.last_turn_assistant_content = messages[-1]['content'].strip() -%}
    {%- set messages = messages[:-1] -%}
{%- endif -%}

{%- for message in messages -%}
    {%- set content = message['content'] -%}

    {%- if message['role'] == 'user' -%}
        {{- '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}

    {%- elif message['role'] == 'tool' -%}
        {%- if loop.first or (messages[loop.index0 - 1].role != 'tool') -%}
            {{- '<SPECIAL_11>User\n' + '<TOOL_RESPONSE>[' }}
        {%- endif -%}
        {{- message['content'] -}}
        {{- ', ' if not loop.last and (messages[loop.index0 + 1].role == 'tool') else '' -}}
        {%- if loop.last or (messages[loop.index0 + 1].role != 'tool') -%}
            {{- ']</TOOL_RESPONSE>\n' -}}
        {%- endif -%}

    {%- elif message['role'] == 'assistant' -%}
        {%- if '</think>' in content -%}
            {%- set content = content.split('</think>')[1].strip() %}
        {%- endif -%}

        {{- '<SPECIAL_11>Assistant\n' + content.strip() }}

        {%- if message.tool_calls -%}
            {%- if content.strip() != '' -%}
                {{- '\n\n' -}}
            {%- endif -%}
            {{- '<TOOLCALL>[' -}}
            {%- for call in message.tool_calls -%}
                {%- set fn = call.function if call.function is defined else call -%}
                {{- '{"name": "' + fn.name + '", "arguments": ' -}}
                {%- if fn.arguments is string -%}
                    {{- fn.arguments -}}
                {%- else -%}
                    {{- fn.arguments | tojson -}}
                {%- endif -%}
                {{- '}' + (', ' if not loop.last else '') -}}
            {%- endfor -%}
            {{- ']</TOOLCALL>' -}}
        {%- endif -%}

        {{- '\n<SPECIAL_12>\n' -}}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{- '<SPECIAL_11>Assistant\n' -}}
    {%- if ns.enable_thinking is defined and ns.enable_thinking is false -%}
        {{- '<think></think>' -}}
    {%- else -%}
        {{- '<think>\n' -}}
    {%- endif -%}
    {%- if ns.last_turn_assistant_content is defined and ns.last_turn_assistant_content != '' -%}
        {{- ns.last_turn_assistant_content -}}
    {%- endif -%}

{%- else -%}
    {%- if ns.last_turn_assistant_content is defined and ns.last_turn_assistant_content != '' -%}
        {{- '<SPECIAL_11>Assistant\n' -}}
        {%- if ns.enable_thinking is defined and ns.enable_thinking is false -%}
            {{- '<think></think>' -}}
        {%- else -%}
            {{- '<think>\n' -}}
        {%- endif -%}
        {{- ns.last_turn_assistant_content -}}

        {%- if continue_final_message is defined -%}
            {%- if continue_final_message is false -%}
                {{- '\n<SPECIAL_12>\n' -}}
            {%- endif -%}
        {%- else -%}
            {{- '\n<SPECIAL_12>\n' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}
```

## 

## Training, Testing, and Evaluation Datasets

### Training datasets

* Data Modality: Text  
* Text Training Data Size: More than 10 Trillion Tokens  
* Train/Test/Valid Split: We used 100% of the corpus for pre-training and relied on external benchmarks for testing.  
* Data Collection Method by dataset: Hybrid: Automated, Human, Synthetic  
* Labeling Method by dataset: Hybrid: Automated, Human, Synthetic


**Properties:** The post-training corpus for NVIDIA-Nemotron-Nano-9B-v2 consists of English and multilingual text (German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese and English). Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including code, legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracies. For several of the domains listed above we used synthetic data, specifically reasoning traces, from DeepSeek R1/R1-0528, Qwen3-235B-A22B, Nemotron 4 340B, Qwen2.5-32B-Instruct-AWQ, Qwen2.5-14B-Instruct, Qwen 2.5 72B.

The pre-training corpus for NVIDIA-Nemotron-Nano-9B-v2 consists of high-quality curated and synthetically-generated data. It is trained in the English language, as well as 15 multilingual languages and 43 programming languages. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracy. The model was pre-trained for approximately twenty trillion tokens.

Alongside the model, we release our [final pretraining data](https://huggingface.co/collections/nvidia/nemotron-pre-training-dataset-689d9de36f84279d83786b35), as outlined in this section. For ease of analysis, there is a sample set that is ungated. For all remaining code, math and multilingual data, gating and approval is required, and the dataset is permissively licensed for model training purposes.

More details on the datasets and synthetic data generation methods can be found in the technical report [NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf) .


## Public Datasets

| Dataset | Collection Period |
| :---- | :---- |
| [Problems in Elementary Mathematics for Home Study](https://archive.org/details/AntonovVygodskyNikitinSankinProblemsInElementaryMathematicsForHomeStudyMir1982) | 4/23/2025 |
| [GSM8K](https://github.com/openai/grade-school-math) | 4/23/2025 |
| [PRM800K](https://github.com/openai/prm800k) | 4/23/2025 |
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
| [Public Software Heritage S3](https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html#summary-of-dataset-versions) | Legacy Download |
| [The Stack](https://huggingface.co/datasets/bigcode/the-stack) | Legacy Download |
| [mC4](https://huggingface.co/datasets/legacy-datasets/mc4) | Legacy Download |
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
| [FastChat](https://github.com/lm-sys/FastChat) | 6/30/2025 |

## Private Non-publicly Accessible Datasets of Third Parties

| Dataset |
| :---- |
| Global Regulation |
| Workbench |

## Online Dataset Sources

The English Common Crawl data was downloaded from the Common Crawl Foundation (see their [FAQ](https://commoncrawl.org/faq) for details on their crawling) and includes the snapshots CC-MAIN-2013-20 through CC-MAIN-2025-13. The data was subsequently deduplicated and filtered in various ways described in the [Nemotron-CC paper](https://arxiv.org/abs/2412.02595).

Additionally, we extracted data for fifteen languages from the following three Common Crawl snapshots: CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18. The fifteen languages included were Arabic, Chinese, Danish, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, and Thai. As we did not have reliable multilingual model-based quality classifiers available, we applied just heuristic filtering instead—similar to what we did for lower quality English data in the Nemotron-CC pipeline, but selectively removing some filters for some languages that did not work well. Deduplication was done in the same way as for Nemotron-CC.

The GitHub Crawl was collected using the GitHub REST API and the Amazon S3 API. Each crawl was operated in accordance with the rate limits set by its respective source, either GitHub or S3. We collect raw source code and subsequently remove any having a license which does not exist in our permissive-license set (for additional details, refer to the technical report).

| Dataset | Modality | Dataset Size (Tokens) | Collection Period |
| :---- | :---- | :---- | :---- |
| English Common Crawl | Text | 3.360T | 4/8/2025 |
| Multilingual Common Crawl | Text | 812.7B | 5/1/2025 |
| GitHub Crawl | Text | 747.4B | 4/29/2025 |

## NVIDIA-Sourced Synthetic Datasets

| Dataset | Modality | Dataset Size (Tokens) | Seed Dataset | Model(s) used for generation |
| :---- | :---- | :---- | :---- | :---- |
| Synthetic Art of Problem Solving from DeepSeek-R1 | Text | 25.5B | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Moral Stories and Social Chemistry from Mixtral-8x22B-v0.1 | Text | 327M | [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101); [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| Synthetic Social Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 83.6M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic Health Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 9.7M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic STEM seeded with OpenStax, Open Textbook Library, and GSM8K from DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-0324, and Qwen2.5-72B | Text | 175M | [OpenStax \- CC BY-SA subset](https://openstax.org/); [GSM8K](https://github.com/openai/grade-school-math); [Open Textbook Library \- CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| [Nemotron-PrismMath](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath) | Text | 4.6B | [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified); [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | [Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct); [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| Synthetic Question Answering Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text | 350M | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic FineMath-4+ Reprocessed from DeepSeek-V3 | Text | 9.2B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| Synthetic FineMath-3+ Reprocessed from phi-4 | Text | 27.6B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Union-3+ Reprocessed from phi-4 | Text | 93.1B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Refreshed [Nemotron-MIND](https://huggingface.co/datasets/nvidia/Nemotron-MIND) from phi-4 | Text | 73B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Union-4+ Reprocessed from phi-4 | Text | 14.12B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Union-3+ minus 4+ Reprocessed from phi-4 | Text | 78.95B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Union-3 Refreshed from phi-4 | Text | 80.94B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Union-4+ Refreshed from phi-4 | Text | 52.32B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from DeepSeek-V3 and DeepSeek-V3-0324 | Text | 4.0B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from Qwen3-30B-A3B | Text | 4.2B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Art of Problem Solving from Qwen2.5-32B-Instruct, Qwen2.5-Math-72B, Qwen2.5-Math-7B, and Qwen2.5-72B-Instruct | Text | 83.1B | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); [GSM8K](https://github.com/openai/grade-school-math); [PRM800K](https://github.com/openai/prm800k) | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [Qwen2.5-Math-72B](https://huggingface.co/Qwen/Qwen2.5-Math-72B); [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B); [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic MMLU Auxiliary Train from DeepSeek-R1 | Text | 0.5B | [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Long Context Continued Post-Training Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text | 5.4B  | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic Common Crawl from Qwen3-30B-A3B and Mistral-Nemo-12B-Instruct | Text | 1.949T | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct) |
| Synthetic Multilingual Data from Common Crawl from Qwen3-30B-A3B | Text | 997.3B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Multilingual Data from Wikimedia from Qwen3-30B-A3B | Text | 55.1B | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic OpenMathReasoning from DeepSeek-R1-0528 | Text | 1.5M | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic OpenCodeReasoning from DeepSeek-R1-0528 | Text | 1.1M | [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Science Data from DeepSeek-R1-0528 | Text | 1.5M | \- | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Humanity's Last Exam from DeepSeek-R1-0528 | Text | 460K | [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic ToolBench from Qwen3-235B-A22B | Text | 400K | [ToolBench](https://github.com/OpenBMB/ToolBench) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Synthetic Nemotron Content Safety Dataset V2, eval-safety, Gretel Synthetic Safety Alignment, and RedTeam\_2K from DeepSeek-R1-0528 | Text | 52K | [Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0); [eval-safety](https://github.com/CrystalEye42/eval-safety/blob/main/malicious_tasks_dataset.yaml); [Gretel Synthetic Safety Alignment](https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1); [RedTeam\_2K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/viewer/RedTeam_2K) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic HelpSteer from Qwen3-235B-A22B | Text | 120K | [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3); [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Synthetic Alignment data from Mixtral-8x22B-Instruct-v0.1, Mixtral-8x7B-Instruct-v0.1, and Nemotron-4 Family | Text | 400K | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2); [C4](https://huggingface.co/datasets/allenai/c4); [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m); [ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K); [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k); [GSM8K](https://github.com/openai/grade-school-math); [PRM800K](https://github.com/openai/prm800k); lm\_identity (NVIDIA internal); [FinQA](https://finqasite.github.io/); [WikiTableQuestions](https://huggingface.co/datasets/wikitablequestions); [Riddles](https://github.com/crawsome/riddles); ChatQA nvolve-multiturn (NVIDIA internal); [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2); [SciBench](https://github.com/mandyyyyii/scibench); [OpenBookQA](https://github.com/allenai/OpenBookQA); [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb); [Public Software Heritage S3](https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html#summary-of-dataset-versions); [Khan Academy Math Keywords](https://www.khanacademy.org/math) | Nemotron-4-15B-Base (NVIDIA internal); Nemotron-4-15B-Instruct (NVIDIA internal); [Nemotron-4-340B-Base](https://huggingface.co/nvidia/Nemotron-4-340B-Base); [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct); [Nemotron-4-340B-Reward](https://huggingface.co/nvidia/Nemotron-4-340B-Reward);  [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1);  [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) |
| Synthetic LMSYS-Chat-1M from Qwen3-235B-A22B | Text | 1M | [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Synthetic Multilingual Reasoning data from DeepSeek-R1-0528, Qwen2.5-32B-Instruct-AWQ, and Qwen2.5-14B-Instruct | Text | 25M | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning); [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528); [Qwen2.5-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ) (translation); [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) (translation); |
| Synthetic Multilingual Reasoning data from Qwen3-235B-A22B and Gemma 3 Post-Trained models | Text | 5M | [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [Gemma 3 PT 12B](https://huggingface.co/google/gemma-3-12b-it); [Gemma 3 PT 27B](https://huggingface.co/google/gemma-3-27b-it) |

### Evaluation Dataset:

* Data Collection Method by dataset: Hybrid: Human, Synthetic  
* Labeling Method by dataset: Hybrid: Automated, Human, Synthetic

## Inference

- ## Engines: HF, vLLM, TRT-LLM

- ## Test Hardware NVIDIA A10G 24GB, H100 80GB, Jetson AGX Thor

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our [Trustworthy AI terms of service](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/), developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](./bias.md), [Explainability](./explainability.md), [Safety & Security](./safety.md), and [Privacy](./privacy.md) Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


## Citation

```
@misc{nvidia2025nvidianemotronnano2,
      title={NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model},
      author={NVIDIA},
      year={2025},
      eprint={2508.14444},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.14444},
}
```