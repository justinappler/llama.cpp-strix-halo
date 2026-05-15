---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- chat
base_model: Qwen/Qwen2-57B-A14B
---

# Qwen2-57B-A14B-Instruct

## Introduction

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This repo contains the instruction-tuned 57B-A14B Mixture-of-Experts Qwen2 model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics, reasoning, etc.

Qwen2-57B-A14B-Instruct supports a context length of up to 65,536 tokens, enabling the processing of extensive inputs. Please refer to [this section](#processing-long-texts) for detailed instructions on how to deploy Qwen2 for handling long texts.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2/) and [GitHub](https://github.com/QwenLM/Qwen2).
<br>

## Model Details
Qwen2 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes.

## Training details
We pretrained the models with a large amount of data, and we post-trained the models with both supervised finetuning and direct preference optimization.


## Requirements
The code of Qwen2MoE has been in the latest Hugging face transformers and we advise you to install `transformers>=4.40.0`, or you might encounter the following error:
```
KeyError: 'qwen2_moe'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-57B-A14B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-57B-A14B-Instruct")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Processing Long Texts

To handle extensive inputs exceeding 32,768 tokens, we utilize [YARN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

For deployment, we recommend using vLLM. You can enable the long-context capabilities by following these steps:

1. **Install vLLM**: Ensure you have the latest version from the main branch of [vLLM](https://github.com/vllm-project/vllm).

2. **Configure Model Settings**: After downloading the model weights, modify the `config.json` file by including the below snippet:
    ```json
        {
            "architectures": [
                "Qwen2MoeForCausalLM"
            ],
            // ...
            "vocab_size": 152064,

            // adding the following snippets
            "rope_scaling": {
                "factor": 2.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn"
            }
        }
    ```
    This snippet enable YARN to support longer contexts.

3. **Model Deployment**: Utilize vLLM to deploy your model. For instance, you can set up an openAI-like server using the command:

    ```bash
    python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-57B-A14B-Instruct --model path/to/weights
    ```

    Then you can access the Chat API by:

    ```bash
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "Qwen2-57B-A14B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Your Long Input Here."}
        ]
        }'
    ```

    For further usage instructions of vLLM, please refer to our [Github](https://github.com/QwenLM/Qwen2).

**Note**: Presently, vLLM only supports static YARN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts**. We advise adding the `rope_scaling` configuration only when processing long contexts is required.

## Evaluation

We briefly compare Qwen2-57B-A14B-Instruct with similar-sized instruction-tuned LLMs, including Qwen1.5-32B-Chat. The results are shown as follows:

| Datasets | Mixtral-8x7B-Instruct-v0.1 | Yi-1.5-34B-Chat | Qwen1.5-32B-Chat | **Qwen2-57B-A14B-Instruct** |
| :--- | :---: | :---: | :---: | :---: |
|Architecture | MoE | Dense | Dense | MoE |
|#Activated Params | 12B | 34B | 32B | 14B |
|#Params | 47B | 34B | 32B | 57B   |
| _**English**_ |  |  |  |  |
| MMLU | 71.4 | **76.8** | 74.8 | 75.4 |
| MMLU-Pro | 43.3 | 52.3 | 46.4 | **52.8** |
| GPQA | - | - | 30.8 | **34.3** |
| TheroemQA | - | - | 30.9 | **33.1** |
| MT-Bench | 8.30 | 8.50 | 8.30 | **8.55** |
| _**Coding**_ |  |  |  |  |
| HumanEval | 45.1 | 75.2 | 68.3 | **79.9** |
| MBPP | 59.5 | **74.6** | 67.9 | 70.9 |
| MultiPL-E | - | - | 50.7 | **66.4** |
| EvalPlus | 48.5 | - | 63.6 | **71.6** |
| LiveCodeBench | 12.3 | - | 15.2 | **25.5** |
| _**Mathematics**_ |  |  |  |  |
| GSM8K | 65.7 | **90.2** | 83.6 | 79.6 |
| MATH | 30.7 | **50.1** | 42.4 | 49.1 |
| _**Chinese**_ |  |  |  |  |
| C-Eval | - | - | 76.7 | 80.5 |
| AlignBench | 5.70 | 7.20 | 7.19 | **7.36** |

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2,
  title={Qwen2 Technical Report},
  year={2024}
}
```