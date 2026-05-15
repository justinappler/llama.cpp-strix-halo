---
base_model:
- inclusionAI/Ling-flash-base-2.0
library_name: transformers
license: mit
pipeline_tag: text-generation
---

# Ring-flash-2.0

This model is presented in the paper [Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model](https://huggingface.co/papers/2510.18855).

The official code repository is available at: [https://github.com/inclusionAI/Ring-V2](https://github.com/inclusionAI/Ring-V2).

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*4QxcQrBlTiAAAAAAQXAAAAgAemJ7AQ/original" width="100"/>
<p>
<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp🚀 <a href="https://zenmux.ai/inclusionai/ring-flash-2.0?utm_source=hf_inclusionAI">Experience Now</a></p>

## Introduction

Today, we are officially open-sourcing Ring-flash-2.0.

This is a **high-performance thinking model, deeply optimized** based on Ling-flash-2.0-base. Like Ling-flash-2.0, Ring-flash-2.0 has a total of 100B parameters, with only 6.1B activated per inference. Our independently developed **icepop algorithm** has successfully addressed the challenge of training instability in reinforcement learning (RL) for MoE LLMs after cold-start Long-CoT SFT, enabling the model’s complex reasoning capabilities to continuously improve throughout extended RL training cycles.

Ring-flash-2.0 demonstrates significant breakthroughs across multiple challenging benchmarks, including **math competitions**, **code generation**, and **logical reasoning**. Its performance not only surpasses that of SOTA dense models under 40B parameters but also rivals larger open-weight MoE models and closed-source high-performance thinking model APIs.

### leading-level performance in complex reasoning

We selected **representative open-source thinking models** and **closed-source APIs** for comparison, including GPT-OSS-120B(medium), Qwen3-32B-Thinking, Seed-OSS-36B-Instruct, and Gemini-2.5-Flash.

The benchmarking results demonstrate that Ring-flash-2.0 exhibits leading performance across multiple challenging general reasoning tasks, including:

- **Math competitions** (AIME 25, Omni-MATH),
- **Code generation** (LiveCodeBench, CodeForce-Elo),
- **Logical reasoning** (ARC-Prize).
  It also shows strong competitiveness in specialized domains such as:
- **Scientific and medical reasoning** (GPQA-Diamond, HealthBench).

More surprisingly, although Ring-flash-2.0 is primarily designed for complex reasoning, it outperforms all other compared models in **creative writing** (Creative Writing v3) and matches the creative capability of its "twin brother"—the non-thinking model Ling-flash-2.0.

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*jLbeS74JqB8AAAAAWmAAAAgAemJ7AQ/original"/>
<p>
<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*_AG2T62ZWNsAAAAAWKAAAAgAemJ7AQ/original"/>
<p>

### Efficient Architecture, High-Speed Inference

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*awCaS4yTD9UAAAAAUdAAAAgAemJ7AQ/original"/>
<p>
Building on the highly efficient MoE architecture of the Ling 2.0 series, and through structural optimizations such as a __1/32 expert activation ratio__ and __MTP layers__, Ring-flash-2.0 activates only 6.1B (4.8B non-embedding) parameters while delivering performance comparable to a ∼40B dense model.
Thanks to its low activation and high sparsity design, Ring-flash-2.0 achieves a high generation speed of __200+ tokens/sec__ when deployed on just four H20 GPUs, significantly reducing inference costs for thinking models in high-concurrency scenarios.

## IcePop: Cooling Down Training-Inference Gaps in RL for MoE Models

During the RL for MoE models, the discrepancy of precision between the training and inference engines is more pronounced compared to dense models. This gap widens progressively as sequence length and training steps increase—particularly during long-sequence generation and extended training cycles. A more critical issue is that the original GRPO algorithm begins to break down within a limited number of training steps. Specifically, the probabilistic discrepancy for the same token between training and inference phases gradually increases. When this relative difference exceeds 5%, training effectively fails, posing a significant challenge for long-horizon reinforcement learning with lengthy sequences.

To address this issue, we introduced a key solution: **distribution calibration via masked bidirectional truncation, which effectively narrows the gap between training and inference**.

- Bidirectional Truncation: We truncate not only tokens where the training probability is significantly higher than the inference probability but also the reverse scenario where the training probability is much lower.
- Masking: Tokens with excessively large discrepancies are excluded from gradient computation.

For detailed algorithm introduction, please refer to our technical blog: https://ringtech.notion.site/icepop

## SFT + RLVR + RLHF Multi-Stage Training

To comprehensively enhance the capabilities of Ring-flash-2.0, we designed a Two-staged RL pipeline. First, lightweight Long-CoT SFT equips the Ling-flash-2.0-base model with diverse thinking patterns. This is followed by RL training with Verifiable Rewards (RLVR) to continually stimulate the model’s reasoning potential. Finally, an RLHF phase is incorporated to improve the model’s general abilities.

During RL training, we compared directly combining RLVR and RLHF into joint training with the ultimately adopted Two-staged RL pipeline. Both approaches showed relatively similar effectiveness in our experiments. However, due to the differing difficulty levels of RLVR and RLHF tasks—with RLHF involving relatively shorter model rollouts—joint training resulted in more long-tail generations. From an engineering efficiency perspective, we ultimately adopted the Two-staged RL approach.

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*4Q_4SbSv73YAAAAAQ6AAAAgAemJ7AQ/original"/>
<p>

## Quickstart

### 🚀 Try Online

You can experience Ring-flash-2.0 online at: [ZenMux](https://zenmux.ai/inclusionai/ring-flash-2.0?utm_source=hf_inclusionAI)

### 🔌 API Usage

You can also use Ring-flash-2.0 through API calls:

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
    model="inclusionai/ring-flash-2.0",
    messages=[
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ]
)

print(completion.choices[0].message.content)
```

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "inclusionAI/Ring-flash-2.0"
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
    max_new_tokens=8192
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
tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ring-flash-2.0")
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=16384)
llm = LLM(model="inclusionAI/Ring-flash-2.0", dtype='bfloat16')
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
vllm serve inclusionAI/Ring-flash-2.0 \
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

We recommend you to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to [finetune Ring](https://github.com/inclusionAI/Ring-V2/blob/main/docs/llamafactory_finetuning.md).

## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ring-V2/blob/master/LICENSE).

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@article{lingteam2025everystep,
  title={Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model},
  author={Ling Team and Anqi Shen and Baihui Li and Bin Hu and Bin Jing and Cai Chen and Chao Huang and Chao Zhang and Chaokun Yang and Cheng Lin and Chengyao Wen and Congqi Li and Deng Zhao and Dingbo Yuan and Donghai You and Fagui Mao and Fanzhuang Meng and Feng Xu and Guojie Li and Guowei Wang and Hao Dai and Haonan Zheng and Hong Liu and Jia Guo and Jiaming Liu and Jian Liu and Jianhao Fu and Jiannan Shi and Jianwen Wang and Jianxin Lai and Jin Yang and Jun Mei and Jun Zhou and Junbo Zhao and Junping Zhao and Kuan Xu and Le Su and Lei Chen and Li Tang and Liang Jiang and Liangcheng Fu and Lianhao Xu and Linfeng Shi and Lisha Liao and Longfei Zheng and Meng Li and Mingchun Chen and Qi Zuo and Qiang Cheng and Qianggang Cao and Qitao Shi and Quanrui Guo and Senlin Zhu and Shaofei Wang and Shaomian Zheng and Shuaicheng Li and Shuwei Gu and Chen, Siba and Wu, Tao and Zhang, Tao and Zhang, Tianyu and Zhou, Tianyu and Bie, Tiwei and Yang, Tongkai and Hong, Wang and Ren, Wang and Chen, Weihua and Yu, Wenbo and Zheng, Wengang and Wang, Xiangchun and Yan, Xiaodong and Wan, Xiaopei and Zhao, Xin and Kong, Xinyu and Tang, Xinyu and Han, Xudong and Wang, Xudong and Yang, Xuemin and Hu, Xueyu and Zhang, Yalin and Sun, Yan and Shan, Yicheng and Wang, Yilong and Xu, Yingying and Liu, Yongkang and Guo, Yongzhen and Wang, Yuanyuan and Yan, Yuchen and Wang, Yuefan and Guo, Yuhong and Li, Zehuan and Xu, Zhankai and Li, Zhe and Zhang, Zhenduo and Gui, Zhengke and Pan, Zhenxuan and Huang, Zhenyu and Lan, Zhenzhong and Ding, Zhiqiang and Zhang, Zhiqiang and Li, Zhixun and Liu, Zhizhen and Wang, Zihao and Wen, Zujie},
  year={2025},
  eprint={2510.18855},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.18855},
}
```