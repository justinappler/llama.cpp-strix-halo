---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
pipeline_tag: text-generation
tags:
- liquid
- lfm2
- edge
- moe
---

<center>
<div style="text-align: center;">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" 
    alt="Liquid AI"
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
</div>
<div style="display: flex; justify-content: center; gap: 0.5em;">
<a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
</div>
</center>

<br>

# LFM2-8B-A1B

LFM2 is a new generation of hybrid models developed by [Liquid AI](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts), specifically designed for edge AI and on-device deployment. It sets a new standard in terms of quality, speed, and memory efficiency. 

We're releasing the weights of our first MoE based on LFM2, with 8.3B total parameters and 1.5B active parameters.

- LFM2-8B-A1B is the best on-device MoE in terms of both **quality** (comparable to 3-4B dense models) and **speed** (faster than Qwen3-1.7B).
- **Code and knowledge** capabilities are significantly improved compared to LFM2-2.6B.
- Quantized variants fit comfortably on high-end **phones, tablets, and laptops**.

Find more information about LFM2-8B-A1B in our [blog post](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts).

## 📄 Model details

Due to their small size, **we recommend fine-tuning LFM2 models on narrow use cases** to maximize performance. 
They are particularly suited for agentic tasks, data extraction, RAG, creative writing, and multi-turn conversations. 
However, we do not recommend using them for tasks that are knowledge-intensive or require programming skills.

| Property              | [**LFM2-8B-A1B**](https://huggingface.co/LiquidAI/LFM2-8B-A1B) | [**LFM2-24B-A2B**](https://huggingface.co/LiquidAI/LFM2-24B-A2B) | 
| --------------------- | ----------------------------- | ----------------------------- |
| **Total parameters**  | 8.3B                          | 24B |
| **Active parameters** | 1.5B                          | 2.3B |
| **Layers**            | 24 (18 conv + 6 attn)         | 40 (30 conv + 10 attn) |
| **Context length**    | 32,768 tokens                 | 32,768 tokens |
| **Vocabulary size**   | 65,536                        | 65,536 |
| **Training precision**| Mixed BF16/FP8                | Mixed BF16/FP8 |
| **Training budget**   | 12 trillion tokens            | 17 trillion tokens |
| **License**           | LFM Open License v1.0         | LFM Open License v1.0 |

**Supported languages**: English, Arabic, Chinese, French, German, Japanese, Korean, and Spanish.

**Generation parameters**: We recommend the following parameters:
* `temperature=0.3`
* `min_p=0.15`
* `repetition_penalty=1.05`

**Chat template**: LFM2 uses a ChatML-like chat template as follows:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
It's a tiny nematode that lives in temperate soil environments.<|im_end|>
```

You can automatically apply it using the dedicated [`.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_templating#applychattemplate) function from Hugging Face transformers.

**Tool use**: It consists of four main steps:
1. **Function definition**: LFM2 takes JSON function definitions as input (JSON objects between `<|tool_list_start|>` and `<|tool_list_end|>` special tokens), usually in the system prompt
2. **Function call**: LFM2 writes Pythonic function calls (a Python list between `<|tool_call_start|>` and `<|tool_call_end|>` special tokens), as the assistant answer.
3. **Function execution**: The function call is executed and the result is returned (string between `<|tool_response_start|>` and `<|tool_response_end|>` special tokens), as a "tool" role.
4. **Final answer**: LFM2 interprets the outcome of the function call to address the original user prompt in plain text.

Here is a simple example of a conversation using tool use:

```
<|startoftext|><|im_start|>system
List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
<|tool_response_start|>[{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}]<|tool_response_end|><|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

You can directly pass tools as JSON schema or Python functions with `.apply_chat_template()` as shown in [this page](https://huggingface.co/docs/transformers/en/chat_extras) to automatically format the system prompt.

**Architecture**: Hybrid model with multiplicative gates and short convolutions: 18 double-gated short-range LIV convolution blocks and 6 grouped query attention (GQA) blocks.

**Pre-training mixture**: Approximately 75% English, 20% multilingual, and 5% code data sourced from the web and licensed materials.

**Training approach**:
* Very large-scale SFT on 50% downstream tasks, 50% general domains
* Custom DPO with length normalization and semi-online datasets
* Iterative model merging

## 🏃 How to run LFM2

### 1. Transformers

To run LFM2, you need to install Hugging Face [`transformers`](https://github.com/huggingface/transformers):
```bash
pip install transformers
```

Here is an example of how to generate an answer with transformers in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "LiquidAI/LFM2-8B-A1B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate answer
prompt = "What is C. elegans?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))

# <|startoftext|><|im_start|>user
# What is C. elegans?<|im_end|>
# <|im_start|>assistant
# C. elegans, also known as Caenorhabditis elegans, is a small, free-living
# nematode worm (roundworm) that belongs to the phylum Nematoda.
```

You can directly run and test the model with this [Colab notebook](https://colab.research.google.com/drive/1i0u7X6qen9UJkV6xSCDZ0NmPUH50SOvO?usp=sharing).

### 2. vLLM

You can run the model in [`vLLM`](https://github.com/vllm-project/vllm) by building from source:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e . -v
```

Here is an example of how to use it for inference:

```python
from vllm import LLM, SamplingParams

prompts = [
    [
        {
            "content": "What is C. elegans?",
            "role": "user",
        },
    ],
    [
        {
            "content": "Say hi in JSON format",
            "role": "user",
        },
    ],
    [
        {
            "content": "Define AI in Spanish",
            "role": "user",
        },
    ],
]

sampling_params = SamplingParams(
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_tokens=30
)

llm = LLM(model="LiquidAI/LFM2-8B-A1B", dtype="bfloat16")

outputs = llm.chat(prompts, sampling_params)

for i, output in enumerate(outputs):
    prompt = prompts[i][0]["content"]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 3. llama.cpp

You can run LFM2 with llama.cpp using its [GGUF checkpoint](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF). Find more information in the model card.

## 🔧 How to fine-tune LFM2

We recommend fine-tuning LFM2 models on your use cases to maximize performance.

| Notebook | Description | Link |
|-------|------|------|
| SFT (TRL) | Supervised Fine-Tuning (SFT) notebook with a LoRA adapter using TRL. | <a href="https://colab.research.google.com/drive/1OXLEuSmzF4AjJ7yqRCDTn-ltvFjoGR9j?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| DPO (TRL) | Preference alignment with Direct Preference Optimization (DPO) using TRL. | <a href="https://colab.research.google.com/drive/1Q8hIHIQ8oofshcNYHUcYp1akUcZ-ufSn?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |

## 📈 Performance

### 1. Automated benchmarks

<div style="display: grid">
  <div>
    <a href="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/6xXgpyyK5htUZlHdpZab-.png" target="_blank">
      <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/6xXgpyyK5htUZlHdpZab-.png" alt="Benchmarks" style="width: 100%; height: auto; margin: 0; cursor: pointer;">
    </a>
  </div>
</div>

Compared to similar-sized models, LFM2-8B-A1B displays strong performance in instruction following and math while also running significantly faster.

| Model | MMLU | MMLU-Pro | GPQA | IFEval | IFBench | Multi-IF |
|---|---|---|---|---|---|---|
| **LFM2-8B-A1B** | 64.84 | 37.42 | 29.29 | 77.58 | 25.85 | 58.19 |
| LFM2-2.6B | 64.42 | 25.96 | 26.57 | 79.56 | 22.19 | 60.26 |
| Llama-3.2-3B-Instruct | 60.35 | 22.25 | 30.6 | 71.43 | 20.78 | 50.91 |
| SmolLM3-3B | 59.84 | 23.90 | 26.31 | 72.44 | 17.93 | 58.86 |
| gemma-3-4b-it | 58.35 | 34.76 | 29.51 | 76.85 | 23.53 | 66.61 |
| Qwen3-4B-Instruct-2507 | 72.25 | 52.31 | 34.85 | 85.62 | 30.28 | 75.54 |
| granite-4.0-h-tiny | 66.79 | 32.03 | 26.46 | 81.06 | 18.37 | 52.99 |

| Model | GSM8K | GSMPlus | MATH 500 | MATH Lvl 5 | MGSM | MMMLU |
|---|---|---|---|---|---|---|
| **LFM2-8B-A1B** | 84.38 | 64.76 | 74.2 | 62.38 | 72.4 | 55.26 |
| LFM2-2.6B | 82.41 | 60.75 | 63.6 | 54.38 | 74.32 | 55.39 |
| Llama-3.2-3B-Instruct | 75.21 | 38.68 | 41.2 | 24.06 | 61.68 | 47.92 |
| SmolLM3-3B | 81.12 | 58.91 | 73.6 | 51.93 | 68.72 | 50.02 |
| gemma-3-4b-it | 89.92 | 68.38 | 73.2 | 52.18 | 87.28 | 50.14 |
| Qwen3-4B-Instruct-2507 | 68.46 | 56.16 | 85.6 | 73.62 | 81.76 | 60.67 |
| granite-4.0-h-tiny | 82.64 | 59.14 | 58.2 | 36.11 | 73.68 | 56.13 |

| Model                      | Active params | LCB v6 | LCB v5 | HumanEval+ | Creative Writing v3 |
|----------------------------|---------------|---------------|---------------|--------------------|-----------------------------|
| **LFM2-8B-A1B**            | 1.5B          | 21.04%        | 21.36%        | 69.51%             | 44.22%                      |
| Gemma-3-1b-it              | 1B            | 4.27%         | 4.43%         | 37.20%             | 41.67%                      |
| Granite-4.0-h-tiny         | 1B            | 26.73%        | 27.27%        | 73.78%             | 32.60%                      |
| Llama-3.2-1B-Instruct      | 1.2B          | 4.08%         | 3.64%         | 23.17%             | 31.43%                      |
| Qwen2.5-1.5B-Instruct      | 1.5B          | 11.18%        | 10.57%        | 48.78%             | 22.18%                      |
| Qwen3-1.7B (/no_think)     | 1.7B          | 24.07%        | 26.48%        | 60.98%             | 31.56%                      |
| LFM2-2.6B                  | 2.6B          | 14.41%        | 14.43%        | 57.93%             | 38.79%                      |
| SmolLM3-3B                 | 3.1B          | 19.05%        | 19.20%        | 60.37%             | 36.44%                      |
| Llama-3.2-3B-Instruct      | 3.2B          | 11.47%        | 11.48%        | 24.06%             | 38.84%                      |
| Qwen3-4B (/no_think)       | 4B            | 36.11%        | 38.64%        | 71.95%             | 37.49%                      |
| Qwen3-4B-Instruct-2507     | 4B            | 48.72%        | 50.80%        | 82.32%             | 51.71%                      |
| Gemma-3-4b-it              | 4.3B          | 18.86%        | 19.09%        | 62.8%              | 68.56%                      |

### 2. Inference

LFM2-8B-A1B is significantly faster than models with a similar number of active parameters, like Qwen3-1.7B.

<div style="display: grid; grid-template-columns: 1fr 1fr;">
  <div>
    <a href="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/AdR74EuIH_qJre89qaq62.png" target="_blank">
      <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/AdR74EuIH_qJre89qaq62.png" alt="Decode Throughput - S24 Ultra" style="width: 100%; height: auto; margin: 0; cursor: pointer;">
    </a>
  </div>
  
  <div>
    <a href="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/YzmQXbmcv5WuVJ1tI2Jbh.png" target="_blank">
      <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/YzmQXbmcv5WuVJ1tI2Jbh.png" alt="Decode Throughput - HX370" style="width: 100%; height: auto; margin: 0; cursor: pointer;">
    </a>
  </div>
</div>

The following plots showcase the performance of different models under int4 quantization with int8 dynamic activations on the AMD Ryzen AI 9 HX 370 CPU, using 16 threads. The results are obtained using our internal XNNPACK-based inference stack, and a custom CPU MoE kernel.

<div style="display: grid; grid-template-columns: 1fr 1fr;">
  <div>
    <a href="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/NC4XN11RJB-Ifh758os3e.png" target="_blank">
      <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/NC4XN11RJB-Ifh758os3e.png" alt="Prefill Throughput vs Sequence Length" style="width: 100%; height: auto; margin: 0; cursor: pointer;">
    </a>
  </div>
  <div>
    <a href="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/6oAenHRxKIyvJOgdCetlF.png" target="_blank">
      <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/6oAenHRxKIyvJOgdCetlF.png" alt="Decode Throughput vs Sequence Length" style="width: 100%; height: auto; margin: 0; cursor: pointer;">
    </a>
  </div>
</div>

## 📬 Contact

- Got questions or want to connect? [Join our Discord community](https://discord.com/invite/liquid-ai)
- If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).

## Citation

```
@article{liquidai2025lfm2,
 title={LFM2 Technical Report},
 author={Liquid AI},
 journal={arXiv preprint arXiv:2511.23404},
 year={2025}
}
```