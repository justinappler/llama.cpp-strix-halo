---
license: mit
base_model:
  - inclusionAI/Ling-mini-base-2.0
pipeline_tag: text-generation
library_name: transformers
---

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*4QxcQrBlTiAAAAAAQXAAAAgAemJ7AQ/original" width="100"/>
<p>
<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp🐙 <a href="https://zenmux.ai/inclusionai/ling-mini-2.0?utm_source=hf_inclusionAI">Experience Now</a></p>

## Introduction

Today, we are excited to announce the open-sourcing of **Ling 2.0** — a family of MoE-based large language models that combine **SOTA performance** with **high efficiency**.
The first released version, Ling-mini-2.0, is compact yet powerful. It has **16B total parameters**, but only **1.4B** are activated per input token (non-embedding 789M). Trained on more than **20T tokens** of high-quality data and enhanced through multi-stage supervised fine-tuning and reinforcement learning, Ling-mini-2.0 achieves remarkable improvements in complex reasoning and instruction following. With just 1.4B activated parameters, it still reaches the top-tier level of sub-10B dense LLMs and even matches or surpasses much larger MoE models.

<p align="center"><img src="https://mdn.alipayobjects.com/huamei_fi95qp/afts/img/2NKZS5LVXzcAAAAASBAAAAgADkZ7AQFr/fmt.webp" /></p>

### Strong General and Professional Reasoning

We evaluated Ling-mini-2.0 on challenging general reasoning tasks in coding (LiveCodeBench, CodeForces) and mathematics (AIME 2025, HMMT 2025), as well as knowledge-intensive reasoning tasks across multiple domains (MMLU-Pro, Humanity's Last Exam). Compared with sub-10B dense models (e.g., Qwen3-4B-instruct-2507, Qwen3-8B-nothinking) and larger-scale MoE models (Ernie-4.5-21B-A3B-PT, GPT-OSS-20B/low), Ling-mini-2.0 demonstrated outstanding overall reasoning capabilities.

### 7× Equivalent Dense Performance Leverage

Guided by [Ling Scaling Laws](https://arxiv.org/abs/2507.17702), Ling 2.0 adopts a **1/32 activation ratio** MoE architecture, with empirically optimized design choices in expert granularity, shared expert ratio, attention ratio, aux-loss free + sigmoid routing strategy, MTP loss, QK-Norm, half RoPE, and more. This enables small-activation MoE models to achieve over **7× equivalent dense performance**. In other words, **Ling-mini-2.0 with only 1.4B activated parameters (non-embedding 789M) can deliver performance equivalent to a 7–8B dense model**.

### High-speed Generation at 300+ token/s

<p align="center"><img src="https://mdn.alipayobjects.com/huamei_fi95qp/afts/img/bnxIRaK9tzcAAAAAgSAAAAgADkZ7AQFr/original" /></p>

The highly sparse small-activation MoE architecture also delivers significant training and inference efficiency. In simple QA scenarios (within 2000 tokens), **Ling-mini-2.0 generates at 300+ token/s (on H20 deployment)** — more than **2× faster** than an 8B dense model. Ling-mini-2.0 is able to handle **128K context length** with YaRN, as sequence length increases, the relative speedup can reach **over 7×**.

<p align="center"><img src="https://raw.githubusercontent.com/inclusionAI/Ling-V2/refs/heads/main/figures/needle_in_a_haystack.webp" /></p>

### Open-sourced FP8 Efficient Training Solution

Ling 2.0 employs **FP8 mixed-precision training** throughout. Compared with BF16, experiments with over 1T training tokens show nearly identical loss curves and downstream benchmark performance. To support the community in efficient continued pretraining and fine-tuning under limited compute, we are also open-sourcing our **FP8 training solution**. Based on tile/blockwise FP8 scaling, it further introduces FP8 optimizer, FP8 on-demand transpose weight, and FP8 padding routing map for extreme memory optimization. On 8/16/32 80G GPUs, compared with LLaMA 3.1 8B and Qwen3 8B, **Ling-mini-2.0 achieved 30–60% throughput gains with MTP enabled, and 90–120% throughput gains with MTP disabled**.

### A More Open Opensource Strategy

We believe Ling-mini-2.0 is an ideal starting point for MoE research. For the first time at this scale, it integrates 1/32 sparsity, MTP layers, and FP8 training — achieving both strong effectiveness and efficient training/inference performance, making it a prime candidate for the small-size LLM segment.
To further foster community research, in addition to releasing the post-trained version, we are also open-sourcing **five pretraining checkpoints**: the pre-finetuning Ling-mini-2.0-base, along with four base models trained on 5T, 10T, 15T, and 20T tokens, enabling deeper research and broader applications.

## Model Downloads

You can download the following table to see the various stage of Ling-mini-2.0 models(1.43B activated of 16.26B total params). If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.

<center>

|       **Model**        | **Context Length** |                                                                             **Download**                                                                             |
| :--------------------: | :----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   Ling-mini-base-2.0   | 32K -> 128K (YaRN) |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-mini-base-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-mini-base-2.0)     |
| Ling-mini-base-2.0-5T  |         4K         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-mini-base-2.0-5T) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-mini-base-2.0-5T)  |
| Ling-mini-base-2.0-10T |         4K         | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-mini-base-2.0-10T) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-mini-base-2.0-10T) |
| Ling-mini-base-2.0-15T |         4K         | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-mini-base-2.0-15T) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-mini-base-2.0-15T) |
| Ling-mini-base-2.0-20T |         4K         | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-mini-base-2.0-20T) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-mini-base-2.0-20T) |
|     Ling-mini-2.0      | 32K -> 128K (YaRN) |          [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-mini-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-mini-2.0)          |

</center>

Note: If you are interested in previous version, please visit the past model collections in [Huggingface](https://huggingface.co/inclusionAI) or [ModelScope](https://modelscope.cn/organization/inclusionAI).

## Quickstart

### 🚀 Try Online

You can experience Ling-mini-2.0 online at: [ZenMux](https://zenmux.ai/inclusionai/ling-mini-2.0?utm_source=hf_inclusionAI)

### 🔌 API Usage

You can also use Ling-mini-2.0 through API calls:

```python
from openai import OpenAI

# 1. Initialize the OpenAI client
client = OpenAI(
    # 2. Point the base URL to the ZenMux endpoint
    base_url="https://zenmux.ai/api/v1",
    # 3. Replace with the API Key from your ZenMux user console
    api_key="<your ZENMUX_API_KEY>",
)

# 4. Make a request
completion = client.chat.completions.create(
    # 5. Specify the model to use in the format "provider/model-name"
    model="inclusionai/ling-mini-2.0",
    messages=[
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ]
)

print(completion.choices[0].message.content)
```

### Convert to safetensors

Models with safetensors format can be downloaded from [HuggingFace](https://huggingface.co/inclusionAI) or [ModelScope](https://modelscope.cn/organization/inclusionAI).
If you want to train your model and eval it, you can convert from dcp produced by training.

```shell
python tools/convert_dcp_to_safe_tensors.py --checkpoint-path ${DCP_PATH} --target-path ${SAFETENSORS_PATH}
```

Currently, BF16 and FP8 formats are supported, you can use convert parameter to handle it:

- `--force-bf16` for BF16 format.
- `--force-fp8` for FP8 format.

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "inclusionAI/Ling-mini-2.0"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
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
model_inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 🤖 ModelScope

If you're in mainland China, we strongly recommend you to use our model from 🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>.

## Deployment

### vLLM

vLLM supports offline batched inference or launching an OpenAI-Compatible API Service for online inference.

#### Environment Preparation

Since the Pull Request (PR) has not been submitted to the vLLM community at this stage, please prepare the environment by following the steps below:

```bash
git clone -b v0.10.0 https://github.com/vllm-project/vllm.git
cd vllm
wget https://raw.githubusercontent.com/inclusionAI/Ling-V2/refs/heads/main/inference/vllm/bailing_moe_v2.patch
git apply bailing_moe_v2.patch
pip install -e .
```

#### Offline Inference:

```bash
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0")
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=16384)
llm = LLM(model="inclusionAI/Ling-mini-2.0", dtype='bfloat16')
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
outputs = llm.generate([text], sampling_params)
```

#### Online Inference:

```bash
vllm serve inclusionAI/Ling-mini-2.0 \
              --tensor-parallel-size 2 \
              --pipeline-parallel-size 1 \
              --use-v2-block-manager \
              --gpu-memory-utilization 0.90
```

To handle long context in vLLM using YaRN, we need to follow these two steps:

1. Add a `rope_scaling` field to the model's `config.json` file, for example:

```json
{
  ...,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```

2. Use an additional parameter `--max-model-len` to specify the desired maximum context length when starting the vLLM service.

For detailed guidance, please refer to the vLLM [`instructions`](https://docs.vllm.ai/en/latest/).

### SGLang

#### Environment Preparation

We will later submit our model to SGLang official release, now we can prepare the environment following steps:

```shell
pip3 install sglang==0.5.2rc0 sgl-kernel==0.3.7.post1
```

You can use docker image as well:

```shell
docker pull lmsysorg/sglang:v0.5.2rc0-cu126
```

Then you should apply patch to sglang installation:

```shell
# patch command is needed, run `yum install -y patch` if needed
patch -d `python -c 'import sglang;import os; print(os.path.dirname(sglang.__file__))'` -p3 < inference/sglang/bailing_moe_v2.patch
```

#### Run Inference

BF16 and FP8 models are supported by SGLang now, it depends on the dtype of the model in ${MODEL_PATH}. They both share the same command in the following:

- Start server:

```shell
python -m sglang.launch_server \
    --model-path $MODLE_PATH \
    --host 0.0.0.0 --port $PORT \
    --trust-remote-code \
    --attention-backend fa3
```

MTP is supported for base model, and not yet for chat model. You can add parameter `--speculative-algorithm NEXTN`
to start command.

- Client:

```shell
curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
"""
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)

## Training

We also provide a complete and efficient training framework that covers both pre-training and finetune. Based on this framework, continue training can be performed on the Ling-mini-2.0 checkpoint. With our training framework, the training throughput of the Ling-mini-2.0 model is significantly better than that of the existing Dense 8B model (Qwen3-8B, Llama3-8B).

### Pre-training

[Pretraining demo](https://github.com/inclusionAI/Ling-V2/blob/main/docs/gpu_based_training.md) to Continue pretraining Ling models.

#### Performance Benchmark

The table below shows the pre-training performance of several models, measured in **tokens per second** on 8, 16, and 32 80G GPUs. Ling-mini-2.0 achieves significantly higher training efficiency compared to the baseline, making it easier and more cost-effective to continue pre-training with our [demo scripts](https://github.com/inclusionAI/Ling-V2/blob/main/docs/gpu_based_training.md).

<center>

|        **Model**        | **8 x 80G GPUs (GBS=128)** | **16 x 80G GPUs (GBS=256)** | **32 x 80G GPUs (GBS=512)** |
| :---------------------: | :------------------------: | :-------------------------: | :-------------------------: |
| LLaMA 3.1 8B (baseline) |           81222            |           161319            |           321403            |
|        Qwen3 8B         |      55775 (-31.33%)       |      109799 (-31.94%)       |      219943 (-31.57%)       |
|      Ling-mini-2.0      |      109532 (+34.86%)      |      221585 (+37.36%)       |      448726 (+39.61%)       |
|  Ling-mini-2.0 w/o MTP  |      128298 (+57.96%)      |      307264 (+90.47%)       |      611466 (+90.25%)       |

</center>

### Finetuning

We recommend you to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to [finetune Ling](https://github.com/inclusionAI/Ling-V2/blob/main/docs/llamafactory_finetuning.md). In addition to that, you can also use [Megatron for finetuning](https://github.com/inclusionAI/Ling-V2/blob/main/docs/megatron_sft_training.md).

## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ling-V2/blob/master/LICENCE).

## Citation

If you find our work helpful, feel free to give us a cite.


![--](https://ospo-insights.oss-cn-hangzhou.aliyuncs.com/iai-hf-models/Ling-mini-2.0.gif)