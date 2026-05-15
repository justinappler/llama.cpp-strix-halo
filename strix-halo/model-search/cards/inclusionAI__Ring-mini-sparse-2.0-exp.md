---
license: mit
language:
- en
base_model:
- inclusionAI/Ling-mini-base-2.0-20T
pipeline_tag: text-generation
library_name: transformers
tags:
- moe
---
# Ring-mini-sparse-2.0-exp

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*4QxcQrBlTiAAAAAAQXAAAAgAemJ7AQ/original" width="100"/>
<p>
<p align="center">🤗 <a href="https://huggingface.co/inclusionAI/Ring-mini-sparse-2.0-exp">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI/Ring-mini-sparse-2.0-exp">ModelScope</a></p>

## Introduction

We are excited to annouce the official release of Ring-mini-sparse-2.0-exp. This model employs a Mixture of Block Attention (MoBA) architecture, delivering highly efficient inference without compromising performance.  This model inherts from [Ling-mini-base-2.0](https://huggingface.co/inclusionAI/Ling-mini-base-2.0-20T), continually trained on an additional 100B tokens. The performance of the MoBA-based model is on par with the standard attention models of the same size (e.g., Ring-mini-v2). Furthermore, by applying YaRN-based 4× window extrapolation, we extend the context length to 128K tokens, delivering superior inference speed on tasks that involve long inputs and outputs.

<div style="display: flex; justify-content: center;">
  <div style="text-align: center;">
    <img src="https://mdn.alipayobjects.com/huamei_9mcypc/afts/img/PIoSTKEzmsEAAAAAU5AAAAgADlCHAQFr/original" width="800">
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 1:</strong> The Model Architecture of Ring-mini-sparse-2.0-exp</p>
  </div>
</div>

## Evaluation

To comprehensively assess the reasoning capability of our model, we conducted evaluations on five challenging benchmarks spanning mathematics, coding, and science, comparing it with Ring-mini-2.0, Qwen3-8B-Thinking, and GPT-OSS-20B-Medium. The MoBA architecture demonstrates comparable performance to full softmax attention models.

<div style="display: flex; justify-content: center;">
  <div style="text-align: center;">
    <img src="https://mdn.alipayobjects.com/huamei_9mcypc/afts/img/Yr7eRreHNNUAAAAAWfAAAAgADlCHAQFr/original" width="100%">
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 2:</strong> Model Performance Comparison </p>
  </div>
</div>

## Highly Sparse, High-Speed Generation

Ring-mini-sparse-2.0-exp achieves high inference efficiency through highly sparse attention and a Mixture-of-Experts (MoE) architecture. Unlike MoBA used in Kimi, our approach shares the same KV block selection across all heads within a GQA group, reducing the total number of KV tokens each query head retrieves from the KV cache during decoding. During 64K-context decoding, only 8,192 key-value (KV) tokens are activated per query—reducing KV cache retrieval overhead by 87.5% compared to full attention and delivering up to 3× inference speedup over Ring-mini-2.0. This design significantly lowers computational costs for high-concurrency scenarios involving reasoning-intensive models while maintaining competitive performance. Additionally, with YaRN extrapolation, the model extends context capacity to 128K tokens, achieving up to 2× relative speedup in long-input scenarios compared to Ring-mini-2.0 (full softmax attention).
  
  <div style="text-align: center;">
    <p align="center">
      <img src="https://mdn.alipayobjects.com/huamei_9mcypc/afts/img/iL_eTZP-FVEAAAAATOAAAAgADlCHAQFr/original" width="500">
    </p>
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 4:</strong> Inference speedup ratios of Ring-mini-sparse-2.0-exp compared to Ring-mini-2.0.</p>
  </div>
</div>

## Quickstart

### 🤗 Hugging Face Transformers
Installation requirements:

```shell
pip install flash-attn==2.6.3
pip install transformers==4.56.1
```

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-mini-sparse-2.0-exp"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


prompts = [
    "Give me a short introduction to large language models."
]
input_texts = []
for prompt in prompts:
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    input_texts.append(text)

print(input_texts)

model_inputs = tokenizer(input_texts, return_tensors="pt", return_token_type_ids=False, padding=True, padding_side='left').to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192,
    do_sample=False,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("*" * 30)
print(responses)
print("*" * 30)
```

### 🚀 SGLang

#### Environment Preparation

We have submitted our PR to SGLang official release and it will be merged later, for now we can prepare the environment following steps, firstly install the community version SGLang and required packages:
```shell
pip install sglang==0.5.3 sgl-kernel==0.3.15 torch==2.8.0 torchvision==0.23.0 torchao
```

Then you should install our sglang wheel package:
```shell
git clone https://github.com/inclusionAI/Ring-V2.git
pip install Ring-V2/moba/whls/sglang-0.5.3.post1-py3-none-any.whl --no-deps --force-reinstall
```

#### Run Inference

Our model is supported by SGLang now. You can launch the sever with the command in the following:  

- Start server:
```shell
python -m sglang.launch_server \
    --model-path <model_path> \
    --trust-remote-code \
    --tp-size 4 \
    --disable-radix-cache \
    --chunked-prefill-size 0 \
    --attention-backend moba
```

- Client:

```shell
curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "temperature": 0.6, "messages": [{"role": "user", "content": "Give me a short introduction to large language models."}]}'
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)
