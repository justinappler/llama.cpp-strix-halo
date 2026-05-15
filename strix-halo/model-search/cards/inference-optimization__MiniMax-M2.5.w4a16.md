---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- MiniMaxAI/MiniMax-M2.5
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
---

# MiniMax-M2.5-quantized.w4a16

## Model Overview
- **Model Architecture:** MiniMaxM2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:**
  - Reasoning.
  - Function calling.
  - Subject matter experts via fine-tuning.
  - Multilingual instruction following.
  - Translation.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 02/12/2026
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights of [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a asymmetric per-group scheme, with group size 64.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/MiniMax-M2.5-quantized.w4a16"
number_gpus = 1
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, top_k=40, min_p=0, max_tokens=256)

messages = [
    {"role": "user", "content": prompt}
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "inference-optimization/MiniMax-M2.5-BF16"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)


NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]", trust_remote_code=True)
ds = ds.shuffle(seed=42)

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, )}

ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
recipe = GPTQModifier( scheme="W4A16", weight_observer="mse", targets= [r"re:.*block_sparse_moe\.experts\.\d+\.w[1-3]$", r"re:.*mlp\.experts\.\d+\.(gate|up|gate_up|down)_proj$" ], ignore=["re:.*self_attn.*", "lm_head"])


# Apply quantization.
oneshot(
    model=model, dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    processor=processor,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + ".w4a16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

</details>
 



## Evaluation

The model was evaluated on the ifeval, mmlu_pro and gsm8k_platinum  using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.


<details>
  <summary>Evaluation details</summary>

  Deploy using vllm to create an OpenAI-compatible API endpoint:

- vLLM:
    ```shell
    vllm serve RedHatAI/MiniMax-M2.5.w4a16 --max-model-len 262144 --reasoning-parser deepseek_r1
    ```

  **lm-evaluation-harness**
  ```
  lm_eval --model local-chat-completions \
    --tasks mmlu_pro_chat \
    --model_args "model=RedHatAI/MiniMax-M2.5.w4a16,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks ifeval \
    --model_args "model=RedHatAI/MiniMax-M2.5.w4a16,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks gsm8k_platinum_cot_llama \
    --model_args "model=RedHatAI/MiniMax-M2.5.w4a16,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/MiniMax-M2.5.w4a16
    dtype: auto
    gpu_memory_utilization: 0.9
    max_model_length: 40960
    generation_parameters:
      temperature: 1.0
      top_k: 40
      min_p: 0.0
      top_p: 0.95
      max_new_tokens: 64000
  ```

  ```
  lighteval endpoint litellm lighteval_model_arguments.yaml  \
    "aime25|0,math_500|0,gpqa:diamond|0"
  ```


</details>

### Accuracy


| Benchmark | inference-optimization/MiniMax-M2.5-BF16 | inference-optimization/MiniMax-M2.5.w4a16 | Recovery (%) |
|-----------|------------------------------------------|-------------------------------------------|--------------|                                                                                        
| GSM8k Platinum (0-shot) | 95.15 | 96.36 | 101.27 |                                                                                                                                                       
| IfEval (0-shot) | 88.17 | 85.58 | 97.06 |
| AIME 2025 | 87.50 | 84.17 | 96.19 |                                                                                                                                                                      
| GPQA diamond | 83.67 | 84.51 | 101.01 |
| Math 500 | 87.33 | 87.60 | 100.31 |
| Mmlu Pro Chat | 80.83 | 81.25 | 100.51 |
