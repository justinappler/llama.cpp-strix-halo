---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
---

# GroveMoE-Inst
</div>
<p align="left">
🤗 <a href="https://huggingface.co/collections/inclusionAI/grovemoe-68a2b58acbb55827244ef664">Models</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.07785">Paper</a> &nbsp&nbsp | &nbsp&nbsp 🔗 <a href="https://github.com/inclusionAI/GroveMoE">Github</a>&nbsp&nbsp

  ## Highlights

We introduce **GroveMoE**, a new sparse architecture using **adjugate experts** for dynamic computation allocation, featuring the following key highlights:

- **Architecture**: Novel **adjugate experts** grouped with ordinary experts; shared computation is executed once, then reused, cutting FLOPs.
- **Sparse Activation**: 33 B params total, only **3.14–3.28 B** active per token.
- **Traning**: Mid-training + SFT, up-cycled from Qwen3-30B-A3B-Base; preserves prior knowledge while adding new capabilities.

## Model Downloads


| **Model** | **#Total Params** | **#Activated Params** | **HF Download** |**MS Download** |
|:---------:|:-----------------:|:---------------------:|:------------:|:------------:|
| GroveMoE-Base | 33B | 3.14~3.28B | [🤗 HuggingFace](https://huggingface.co/inclusionAI/GroveMoE-Base) | [📦 ModelScope](https://modelscope.cn/models/cccnju/GroveMoE-Base) |
| GroveMoE-Inst | 33B | 3.14~3.28B | [🤗 HuggingFace](https://huggingface.co/inclusionAI/GroveMoE-Inst) | [📦 ModelScope](https://modelscope.cn/models/cccnju/GroveMoE-Inst) |



## Performance

| Model | Activated Params | MMLU-Pro | SuperGPQA | GPQA-Diamond | OlympiadBench | Omni-math | AIME'25 | MultiPL-E | LiveCodeBench v6 |
|:-----:|:----------------:|:------------:|:-------------:|:------------:|:-----------------:|:------------:|:------------------:|:------------------:|:------------------:|
|Llama4-Scout| 17B | 64.9 | 42.0 | 55.6 | 56.6 | 30.2 | 10.0 | 45.0 | 32.0 |
|Qwen3-30B-A3B| 3B | 63.3 | 40.5 | 51.7 | 60.3 | 33.7 | 21.7 | 66.0 | 29.4 |
|Qwen3-32B| 32B | 68.2 | 43.0 | 53.6 | 59.5 | 31.8 | 22.9 | 68.6 | 28.6 |
|Gemma3-27B-IT| 27B | 67.1 | 35.6 | 45.3 | 59.9 | 33.3 | 23.1 | 65.5 | 30.9 |
|Mistral-Small-3.2| 24B | 68.1 | 37.5 | 59.9 | 61.9 | 33.4 | 28.1 | 69.5 | 32.2 |
|GroveMoE-Inst|3.14~3.28B | <font color=#FBD98D>**72.8**</font> | <font color=#FBD98D>**47.7**</font> | <font color=#FBD98D>**61.3**</font> |<font color=#FBD98D>**71.2**</font> |<font color=#FBD98D>**43.5**</font> | <font color=#FBD98D>**44.4**</font> |<font color=#FBD98D>**74.5**</font> | <font color=#FBD98D>**34.6**</font> |

We bold the top1 scores separately for all models. More details are reported in our [technical report](https://arxiv.org/abs/2508.07785).

## Run GroveMoE

### 🤗 Transformers Quick Start
Below, there are some code snippets on how to get quickly started with running the model. First, install the Transformers library. 

```sh
$ pip install transformers==4.51.3
```

Then, copy the snippet from the section that is relevant for your use case.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "inclusionAI/GroveMoE-Inst"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

### 🚀 SGLang Quick Start
For SGLang, you can follow the steps below to deploy:

1️⃣ Install Dependencies

First, clone the repository:
```shell
git clone https://github.com/inclusionAI/GroveMoE.git
```
Then, install Transformers:
```shell
cd src/transformers-4.51.3
pip install .
```
Next, install SGLang:
```shell
cd src/sglang-0.4.6.post5
pip install .

```

2️⃣ Launch the Server

Run the following command to start SGLang:
```shell
python -m sglang.launch_server \
  --model-path inclusionAI/GroveMoE-Inst \
  --port 30000 \
  --context-length 32768
```

3️⃣ Access the API

Once started, the OpenAI-compatible API will be available at `http://localhost:30000/v1`.

Test it with curl:
```shell
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/GroveMoE-Inst",
    "messages": [{"role": "user", "content": "Hello, SGLang!"}]
  }'

```

### llama.cpp

Thanks @CISCai, support for llama.cpp can be found in the implementation at https://github.com/ggml-org/llama.cpp/pull/15510.

## Best Practices for Model Configuration
To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**:
   - We suggest using `Temperature=0.7`, `TopP=0.8`, `TopK=20`, and `MinP=0`. （⚠️ For benchmarking scenarios requiring sampling (e.g., AIME), these parameters must be explicitly configured.）

2. **Adequate Output Length**: Set output length to 16,384 tokens for general use cases to accommodate complex reasoning tasks in instruct models.

3. **Standardize Output Format**: We recommend using prompts to standardize model outputs when benchmarking.
   - **Math Problems**: Include "Please reason step by step, and put your final answer within \boxed{}." in the prompt.
   - **Multiple-Choice Questions**: Add the following JSON structure to the prompt to standardize responses: "Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`."




## Citation
```bibtex
@article{GroveMoE,
title = {GroveMoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts},
author = {Wu, Haoyuan and Chen, Haoxing and Chen, Xiaodong and Zhou, Zhanchao and Chen, Tieyuan and Zhuang, Yihong and Lu, Guoshan and Zhao, Junbo and Liu, Lin and Huang, Zenan and Lan, Zhenzhong and Yu, Bei and Li, Jianguo},
journal = {arXiv preprint arXiv:2508.07785},
year = {2025}
}
```
