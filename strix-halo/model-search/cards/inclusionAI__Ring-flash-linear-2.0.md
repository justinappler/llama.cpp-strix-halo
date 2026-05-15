---
license: mit
language:
- en
base_model:
- inclusionAI/Ling-flash-base-2.0
pipeline_tag: text-generation
library_name: transformers
tags:
- moe
---

# Ring-flash-linear-2.0

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*4QxcQrBlTiAAAAAAQXAAAAgAemJ7AQ/original" width="100"/>
<p>
<p align="center"> 📖 <a href="https://arxiv.org/abs/2510.19338"> Technical Report</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a></p>

## Introduction

We are excited to announce the official open-source release of Ring-flash-linear-2.0!

Building on the success of our Ling 2.0 series, this model continues to leverage a powerful hybrid architecture of linear and standard attention, perfectly balancing high performance with superior efficiency. By integrating our proven MoE design with optimizations like a 1/32 expert activation ratio and MTP layers, Ring-flash-linear achieves the performance of a 40B dense model while activating only 6.1B parameters.  This model was converted from [Ling-flash-base-2.0](https://huggingface.co/inclusionAI/Ling-flash-base-2.0), further trained on an additional 1T tokens.
When it comes to benchmarks, Ring-flash-linear-2.0 not only holds its own against standard attention models (like Ring-flash-2.0) but also outperforms other open-source MoE and Dense models in its class on several demanding tasks. Plus, with support for a 128k long context, it's faster and more precise than ever, especially when handling long-form inputs and outputs.

<div style="display: flex; justify-content: center;">
  <div style="text-align: center;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/68d20104a6f8ea66da0cb447/PHRg8ipzJtr0p6sojAa5T.png" width="800">
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 1:</strong> Hybrid Linear Model Architecture</p>
  </div>
</div>

## Evaluation

To better demonstrate the model's capabilities, we selected representative open-source thinking models and closed-source APIs for comparison. 
We present results on several challenging reasoning benchmarks spanning domains such as mathematics, coding, and science. Also, we evaluate the model's performance on a creative writing task (Creative Writing v3). 
We observe that our model achieves performance on par with other models.

<div style="display: flex; justify-content: center;">
  <div style="text-align: center;">
    <img src="https://mdn.alipayobjects.com/huamei_t783ie/afts/img/x6YRTIKhf08AAAAAVBAAAAgADgCDAQFr/original" width="1000">
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 2:</strong> Model Performance Comparison </p>
  </div>
</div>

<div style="display: flex; justify-content: center;">
  <div style="text-align: center;">
    <img src="https://mdn.alipayobjects.com/huamei_t783ie/afts/img/gAbsT5MlNfQAAAAAU6AAAAgADgCDAQFr/original" width="1000">
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 3:</strong> Model Performance Comparison </p>
  </div>
</div>


## Linear Attention, Highly Sparse, High-Speed Generation

Thanks to its hybrid attention mechanism and highly sparse MoE architecture, Ring-flash-linear-2.0 achieves near-linear time complexity and constant space complexity, resulting in outstanding inference efficiency. 
To fully demonstrate this advantage, we conducted a comparison between our model and top-tier competitors of similar size or performance.
The results clearly demonstrate the advantage of our model in inference efficiency.


<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="https://mdn.alipayobjects.com/huamei_t783ie/afts/img/wtM_TJ4KVqYAAAAARpAAAAgADgCDAQFr/original" width="500">
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 4:</strong> Ring-flash-linear-2.0 prefill throughput</p>
  </div>
  
  <div style="text-align: center;">
    <p align="center">
      <img src="https://mdn.alipayobjects.com/huamei_t783ie/afts/img/3n9lSZscvBwAAAAAUhAAAAgADgCDAQFr/original" width="500">
    </p>
    <p style="margin-top: 8px; font-size: 14px;"><strong>Figure 5:</strong> Ring-flash-linear-2.0 decode throughput</p>
  </div>

</div>


<!-- ## Model Downloads

<div align="center">

|     **Model**     | **Context Length** | **Download** |
| :----------------: | :----------------: | :----------: |
| Ring-flash-linear-2.0 |        128K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-flash-linear-2.0) <br>[🤖 Modelscope](https://modelscope.cn/models/inclusionAI/Ring-flash-linear-2.0)|
</div> -->

## Quickstart

### Requirements

```bash
pip install flash-linear-attention==0.3.2
pip install transformers==4.56.1
```

### 🤗 Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-flash-linear-2.0"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
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

We have submitted our [PR](https://github.com/sgl-project/sglang/pull/10917) to SGLang official release and it will be merged later, for now we can prepare the environment following steps, firstly install the community version SGLang and required packages:
```shell
pip install sglang==0.5.2 sgl-kernel==0.3.9.post2 vllm==0.10.2 torch==2.8.0 torchvision==0.23.0 torchao
```

Then you should install our sglang wheel package:
```shell
pip install https://media.githubusercontent.com/media/inclusionAI/Ring-V2/refs/heads/main/hybrid_linear/whls/sglang-0.5.2-py3-none-any.whl --no-deps --force-reinstall
```

#### Run Inference

BF16 and FP8 models are supported by SGLang now, it depends on the dtype of the model in ${MODEL_PATH}. They both share the same command in the following:  

- Start server:
```shell
python -m sglang.launch_server \
    --model-path <model_path> \
    --trust-remote-code \
    --tp-size 4 \
    --disable-radix-cache \
    --tool-call-parser qwen25 \ 
    --json-model-override-args "{\"linear_backend\": \"seg_la\"}"
```

- Client:

```shell
curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "temperature": 0.6, "messages": [{"role": "user", "content": "Give me a short introduction to large language models."}]}'
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)

### 🚀 vLLM

#### Environment Preparation

Since the Pull Request (PR) has not been submitted to the vLLM community at this stage, please prepare the environment by following the steps below.

First, create a Conda environment with Python 3.10 and CUDA 12.8:
```shell
conda create -n vllm python=3.10
conda activate vllm
```

Next, install our vLLM wheel package:
```shell
pip install https://media.githubusercontent.com/media/zheyishine/vllm_whl/refs/heads/main/vllm-0.8.5.post2.dev28%2Bgd327eed71.cu128-cp310-cp310-linux_x86_64.whl --force-reinstall
```

Finally, install compatible versions of transformers after vLLM is installed:
```shell
pip install transformers==4.51.1 
```

#### Offline Inference

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ring-flash-linear-2.0", trust_remote_code=True)
    
    sampling_params = SamplingParams(temperature=0.6, top_p=1.0, max_tokens=16384)

    # use `max_num_seqs=1` without concurrency
    llm = LLM(model="inclusionAI/Ring-flash-linear-2.0", dtype='auto', enable_prefix_caching=False, max_num_seqs=128)
    
    
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate([text], sampling_params)
    for output in outputs:
        print(output.outputs[0].text)
```

#### Online Inference
```shell
vllm serve inclusionAI/Ring-flash-linear-2.0 \
              --tensor-parallel-size 2 \
              --pipeline-parallel-size 1 \
              --gpu-memory-utilization 0.90 \
              --max-num-seqs 128 \
              --no-enable-prefix-caching
              --api-key your-api-key
```

#### Citation
```shell
@misc{lingteam2025attentionmattersefficienthybrid,
      title={Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning}, 
      author={Ling Team and Bin Han and Caizhi Tang and Chen Liang and Donghao Zhang and Fan Yuan and Feng Zhu and Jie Gao and Jingyu Hu and Longfei Li and Meng Li and Mingyang Zhang and Peijie Jiang and Peng Jiao and Qian Zhao and Qingyuan Yang and Wenbo Shen and Xinxing Yang and Yalin Zhang and Yankun Ren and Yao Zhao and Yibo Cao and Yixuan Sun and Yue Zhang and Yuchen Fang and Zibin Lin and Zixuan Cheng and Jun Zhou},
      year={2025},
      eprint={2510.19338},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.19338}, 
}
```
