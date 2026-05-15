---
license: mit
base_model:
- inclusionAI/Ling-flash-base-2.0
pipeline_tag: text-generation
library_name: transformers
---



<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*4QxcQrBlTiAAAAAAQXAAAAgAemJ7AQ/original" width="100"/>
<p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a></p>


## Introduction

Today, __Ling-flash-2.0__ is officially open-sourced! 🚀
Following the release of the __language model [Ling-mini-2.0](https://huggingface.co/inclusionAI/Ling-mini-2.0)__ and the __thinking model [Ring-mini-2.0](https://huggingface.co/inclusionAI/Ring-mini-2.0)__, we are now open-sourcing the third MoE LLM under the __Ling 2.0 architecture: Ling-flash-2.0__, a language model with __100B total parameters__ and __6.1B activated parameters (4.8B non-embedding)__.
Trained on __20T+ tokens of high-quality data__, together with __supervised fine-tuning__ and __multi-stage reinforcement learning__, Ling-flash-2.0 achieves __SOTA performance among dense models under 40B parameters__, despite activating only ~6B parameters. Compared to MoE models with larger activation/total parameters, it also demonstrates strong competitiveness. Notably, it delivers outstanding performance in __complex reasoning, code generation, and frontend development__.

### Powerful Complex Reasoning Abilities

We conducted a comprehensive evaluation of Ling-flash-2.0’s reasoning capabilities, reporting strong results on representative benchmarks:
* __Multi-disciplinary knowledge reasoning__: GPQA-Diamond, MMLU-Pro
* __Advanced mathematical reasoning__: AIME 2025, Omni-MATH, OptMATH (advanced mathematical optimization tasks)
* __Challenging code generation__: LiveCodeBench v6, CodeForces-Elo
* __Logical reasoning__: KOR-Bench, ARC-Prize
* __Key regulated industries (Finance, Healthcare)__: FinanceReasoning, HealthBench

Compared with __dense models under 40B__ (e.g., Qwen3-32B-Non-Thinking, Seed-OSS-36B-Instruct (think budget=0)) and __larger-activation/total-parameter MoE models__ (e.g., Hunyuan-A13B-Instruct, GPT-OSS-120B/low), __Ling-flash-2.0__ demonstrates stronger complex reasoning power. Moreover, it shows high competitiveness on __creative tasks__ (Creative Writing v3).
<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_fi95qp/afts/img/zxAvQ7QtrAwAAAAAQqAAAAgADkZ7AQFr/fmt.webp"/>
<p>

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_fi95qp/afts/img/qQ_sTqrxiesAAAAAQuAAAAgADkZ7AQFr/original"/>
<p>

### Efficient Architecture, High-Speed Inference

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_fi95qp/afts/img/fMdiQZqYKSAAAAAAVdAAAAgADkZ7AQFr/fmt.avif"/>
<p>

Guided by [Ling Scaling Laws](https://arxiv.org/abs/2507.17702), Ling 2.0 adopts a __1/32 activation-ratio MoE architecture__, optimized across multiple design choices: expert granularity, shared-expert ratio, attention balance, __aux-loss-free + sigmoid routing strategy__, MTP layers, QK-Norm, Partial-RoPE, and more. These refinements enable __small-activation MoE__ models to achieve __7× efficiency gains__ over equivalent dense architectures.
In other words, with just __6.1B activated parameters (4.8B non-embedding)__, __Ling-flash-2.0__ can match the performance of ~40B dense models. Thanks to its small activation size, it also delivers major inference speed advantages:
* On __H20 hardware__, Ling-flash-2.0 achieves __200+ tokens/s__, offering __3× speedups__ compared to 36B dense models in everyday use.
* With __YaRN extrapolation__, it supports __128K context length__, and as output length grows, its relative speedup can reach __7× or more__.


<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_fi95qp/afts/img/oR9UTY7S0QgAAAAAgKAAAAgADkZ7AQFr/original"/>
<p>

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_fi95qp/afts/img/Hid1RrgsCUAAAAAAQYAAAAgADkZ7AQFr/fmt.webp"/>
<p>


## Model Downloads

You can download the following table to see the various stage of Ling-flash-2.0 models. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.

<center>

|       **Model**        | **Context Length** |                                                                             **Download**                                                                             |
|:----------------------:| :----------------: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   Ling-flash-base-2.0   |        32K -> 128K (YaRN)         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-flash-base-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-flash-base-2.0)     |
|     Ling-flash-2.0      |        32K -> 128K (YaRN)        |          [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-flash-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-flash-2.0)          |

</center>

Note: If you are interested in previous version, please visit the past model collections in [Huggingface](https://huggingface.co/inclusionAI) or [ModelScope](https://modelscope.cn/organization/inclusionAI).


## Quickstart

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-flash-2.0"

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

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-flash-2.0")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=16384)

llm = LLM(model="inclusionAI/Ling-flash-2.0", dtype='bfloat16')
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
vllm serve inclusionAI/Ling-flash-2.0 \
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
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)



### Finetuning

We recommend you to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to [finetune Ling](https://github.com/inclusionAI/Ling-V2/blob/main/docs/llamafactory_finetuning.md).

## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ling-V2/blob/master/LICENCE).


