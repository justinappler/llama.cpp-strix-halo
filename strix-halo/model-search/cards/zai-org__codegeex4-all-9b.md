---
license: other
license_name: codegeex4
license_link: https://huggingface.co/THUDM/codegeex4-all-9b/blob/main/LICENSE
language:
- zh
- en
tags:
- glm
- codegeex
- thudm
inference: false
pipeline_tag: text-generation
---

# CodeGeeX4: Open Multilingual Code Generation Model

<center>
    <img src="https://raw.githubusercontent.com/THUDM/CodeGeeX4/main/resources/logo.jpeg" alt="CodeGeeX4">
</center>

[中文](./README_zh.md)

[GitHub](https://github.com/THUDM/CodeGeeX4)

We introduce CodeGeeX4-ALL-9B, the open-source version of the latest CodeGeeX4 model series. It is a multilingual code generation model continually trained on the [GLM-4-9B](https://github.com/THUDM/GLM-4), significantly enhancing its code generation capabilities. Using a single CodeGeeX4-ALL-9B model, it can support comprehensive functions such as code completion and generation, code interpreter, web search, function call, repository-level code Q&A, covering various scenarios of software development. CodeGeeX4-ALL-9B has achieved highly competitive performance  on public benchmarks, such as [BigCodeBench](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) and [NaturalCodeBench](https://github.com/THUDM/NaturalCodeBench). It is currently the most powerful code generation model with less than 10B parameters, even surpassing much larger general-purpose models, achieving the best balance in terms of inference speed and model performance.

## Get Started

Use `4.39.0<=transformers<=4.40.2` to quickly launch [codegeex4-all-9b](https://huggingface.co/THUDM/codegeex4-all-9b)：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex4-all-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/codegeex4-all-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()
inputs = tokenizer.apply_chat_template([{"role": "user", "content": "write a quick sort"}], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True ).to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=256)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you want to build the **chat** prompt manually, please make sure it follows the following format:
```
f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
```
Default system_prompt:
```
你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。
```
The English version:
```
You are an intelligent programming assistant named CodeGeeX. You will answer any questions users have about programming, coding, and computers, and provide code that is formatted correctly.
```

For **infilling** ability, please use (without system prompt):
```
f"<|user|>\n<|code_suffix|>{suffix}<|code_prefix|>{prefix}<|code_middle|><|assistant|>\n"
```
Additional infos (like file path, programming language, mode) can be added. Example:
```
<|user|>
###PATH:src/example.py
###LANGUAGE:Python
###MODE:BLOCK
<|code_suffix|>{suffix}<|code_prefix|>{prefix}<|code_middle|><|assistant|>
```

## Evaluation

| **Model**                   | **Seq Length** | **HumanEval** | **MBPP** | **NCB** | **LCB** | **HumanEvalFIM** | **CRUXEval-O** |
|-----------------------------|----------------|---------------|----------|---------|---------|------------------|----------------|
| Llama3-70B-intruct          | 8K             | 77.4          | 82.3     | 37.0    | 27.4    | -                | -              |
| DeepSeek Coder 33B Instruct | 16K            | 81.1          | 80.4     | 39.3    | 29.3    | 78.2             | 49.9           |
| Codestral-22B               | 32K            | 81.1          | 78.2     | 46.0    | 35.3    | 91.6             | 51.3           |
| CodeGeeX4-All-9B            | 128K           | 82.3          | 75.7     | 40.4    | 28.5    | 85.0             | 47.1           |

## License

The model weights are licensed under the following [License](./LICENSE).

## Citation

If you find our work helpful, please feel free to cite the following paper:

```
@inproceedings{zheng2023codegeex,
  title={CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X},
  author={Qinkai Zheng and Xiao Xia and Xu Zou and Yuxiao Dong and Shan Wang and Yufei Xue and Zihan Wang and Lei Shen and Andi Wang and Yang Li and Teng Su and Zhilin Yang and Jie Tang},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={5673--5684},
  year={2023}
}
```