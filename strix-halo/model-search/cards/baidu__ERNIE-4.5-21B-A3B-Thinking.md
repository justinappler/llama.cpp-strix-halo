---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-generation
tags:
- ERNIE4.5
library_name: transformers
---

<div align="center" style="line-height: 1;">
  <a href="https://ernie.baidu.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖_Chat-ERNIE_Bot-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/baidu" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Baidu-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/PaddlePaddle/ERNIE" target="_blank" style="margin: 2px;">
    <img alt="Github" src="https://img.shields.io/badge/GitHub-ERNIE-000?logo=github&color=0000FF" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://ernie.baidu.com/blog/ernie4.5" target="_blank" style="margin: 2px;">
    <img alt="Blog" src="https://img.shields.io/badge/🖖_Blog-ERNIE4.5-A020A0" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://discord.gg/JPmZXDsEEK" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-ERNIE-5865F2?logo=discord&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/PaddlePaddle" target="_blank" style="margin: 2px;">
    <img alt="X" src="https://img.shields.io/badge/X-PaddlePaddle-6080F0"?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="#license" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache2.0-A5de54" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

# ERNIE-4.5-21B-A3B-Thinking

## Model Highlights

Over the past three months, we have continued to scale the **thinking capability** of ERNIE-4.5-21B-A3B, improving both the **quality and depth** of reasoning, thereby advancing the competitiveness of ERNIE **lightweight models** in complex reasoning tasks. We are pleased to introduce **ERNIE-4.5-21B-A3B-Thinking**, featuring the following key enhancements:

* **Significantly improved performance** on reasoning tasks, including logical reasoning, mathematics, science, coding, text generation, and academic benchmarks that typically require human expertise.
* **Efficient tool usage** capabilities.
* **Enhanced 128K long-context understanding** capabilities.

> [!NOTE]
> Note: This version has an increased thinking length. We strongly recommend its use in highly complex reasoning tasks.

![benchmark](./benchmark.png)

## Model Overview

ERNIE-4.5-21B-A3B-Thinking is a text MoE post-trained model, with 21B total parameters and 3B activated parameters for each token. The following are the model configuration details:

|Key|Value|
|-|-|
|Modality|Text|
|Training Stage|Posttraining|
|Params(Total / Activated)|21B / 3B|
|Layers|28|
|Heads(Q/KV)|20 / 4|
|Text Experts(Total / Activated)|64 / 6|
|Shared Experts|2|
|Context Length|131072|

## Quickstart

> [!NOTE]
> To align with the wider community, this model releases Transformer-style weights. Both PyTorch and PaddlePaddle ecosystem tools, such as vLLM, transformers, and FastDeploy, are expected to be able to load and run this model.

### FastDeploy Inference

Quickly deploy services using FastDeploy as shown below. For more detailed usage, refer to the [FastDeploy GitHub Repository](https://github.com/PaddlePaddle/FastDeploy).

**Note**: 80GB x 1 GPU resources are required. Deploying this model requires FastDeploy version 2.2.

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-21B-A3B-Thinking \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --load-choices "default_v1" \
       --tensor-parallel-size 1 \
       --max-model-len 131072 \
       --reasoning-parser ernie_x1 \
       --tool-call-parser ernie_x1 \
       --max-num-seqs 32
```

The ERNIE-4.5-21B-A3B-Thinking model supports function call.

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d $'{
  "messages": [
    {
      "role": "user",
      "content": "How \'s the weather in Beijing today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Determine weather in my location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": [
                "c",
                "f"
              ]
            }
          },
          "additionalProperties": false,
          "required": [
            "location",
            "unit"
          ]
        },
        "strict": true
      }
    }]
}'
```

### vLLM inference

VLLM>=0.10.2 (excluding 0.11.0)

```bash
vllm serve baidu/ERNIE-4.5-21B-A3B-Thinking
```

The `reasoning-parser` and `tool-call-parser` for vLLM Ernie need install vllm main branch

### Using `transformers` library

**Note**: You'll need the`transformers`library (version 4.54.0 or newer) installed to use this model.

The following contains a code snippet illustrating how to use the model generate content based on given inputs.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-4.5-21B-A3B-Thinking"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# decode the generated ids
generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print("generate_text:", generate_text)
```

## License

The ERNIE 4.5 models are provided under the Apache License 2.0. This license permits commercial use, subject to its terms and conditions. Copyright (c) 2025 Baidu, Inc. All Rights Reserved.

## Citation

If you find ERNIE 4.5 useful or wish to use it in your projects, please kindly cite our technical report:

```text
@misc{ernie2025technicalreport,
      title={ERNIE 4.5 Technical Report},
      author={Baidu-ERNIE-Team},
      year={2025},
      primaryClass={cs.CL},
      howpublished={\url{https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf}}
}
```

