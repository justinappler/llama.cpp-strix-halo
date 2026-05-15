---
license: llama3.1
base_model:
- meta-llama/Llama-3.1-8B
datasets:
- nvidia/OpenMathInstruct-2
language:
- en
tags:
- nvidia
- math
library_name: transformers
---

# OpenMath2-Llama3.1-8B

OpenMath2-Llama3.1-8B is obtained by finetuning [Llama3.1-8B-Base](https://huggingface.co/meta-llama/Llama-3.1-8B) with [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).

The model outperforms [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on all the popular math benchmarks we evaluate on, especially on [MATH](https://github.com/hendrycks/math) by 15.9%. 

<!-- <p align="center">
  <img src="scaling_plot.jpg" width="350"><img src="math_level_comp.jpg" width="350">
</p> -->

  <style>
      .image-container {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 20px;
      }
      .image-container img {
          width: 350px;
          height: auto;
      }
  </style>

<div class="image-container">
    <img src="scaling_plot.jpg" title="Performance of Llama-3.1-8B-Instruct as it is trained on increasing proportions of OpenMathInstruct-2">
    <img src="math_level_comp.jpg" title="Comparison of OpenMath2-Llama3.1-8B vs. Llama-3.1-8B-Instruct across MATH levels">
</div>

| Model | GSM8K | MATH | AMC 2023 | AIME 2024 | Omni-MATH |
|:---|:---:|:---:|:---:|:---:|:---:|
| Llama3.1-8B-Instruct | 84.5 | 51.9 | 9/40 | 2/30 | 12.7 |
| **OpenMath2-Llama3.1-8B** ([nemo](https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B-nemo) \| [HF](https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B)) | 91.7 | 67.8 | 16/40 | 3/30 | 22.0 |
| + majority@256 | 94.1 | 76.1 | 23/40 | 3/30 | 24.6 |
| Llama3.1-70B-Instruct | 95.8 | 67.9 | 19/40 | 6/30 | 19.0 |
| OpenMath2-Llama3.1-70B ([nemo](https://huggingface.co/nvidia/OpenMath2-Llama3.1-70B-nemo) \| [HF](https://huggingface.co/nvidia/OpenMath2-Llama3.1-70B)) | 94.9 | 71.9 | 20/40 | 4/30 | 23.1 |
| + majority@256 | 96.0 | 79.6 | 24/40 | 6/30 | 27.6 |

The pipeline we used to produce the data and models is fully open-sourced!

- [Code](https://github.com/NVIDIA/NeMo-Skills)
- [Models](https://huggingface.co/collections/nvidia/openmath-2-66fb142317d86400783d2c7b)
- [Dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)

See our [paper](https://arxiv.org/abs/2410.01560) to learn more details!

# How to use the models?

Our models are trained with the same "chat format" as Llama3.1-instruct models (same system/user/assistant tokens). 
Please note that these models have not been instruction tuned on general data and thus might not provide good answers outside of math domain. 

We recommend using [instructions in our repo](https://github.com/NVIDIA/NeMo-Skills/blob/main/docs/basics/inference.md) to run inference with these models, but here is
an example of how to do it through transformers api:

```python
import transformers
import torch

model_id = "nvidia/OpenMath2-Llama3.1-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {
        "role": "user", 
        "content": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n" + 
        "What is the minimum value of $a^2+6a-7$?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=4096,
)
print(outputs[0]["generated_text"][-1]['content'])
```

# Reproducing our results

We provide [all instructions](https://nvidia.github.io/NeMo-Skills/openmathinstruct2/) to fully reproduce our results.

## Citation

If you find our work useful, please consider citing us!

```bibtex
@article{toshniwal2024openmath2,
  title   = {OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data},
  author  = {Shubham Toshniwal and Wei Du and Ivan Moshkov and  Branislav Kisacanin and Alexan Ayrapetyan and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv:2410.01560}
}
```

## Terms of use

By accessing this model, you are agreeing to the LLama 3.1 terms and conditions of the [license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE), [acceptable use policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md) and [Meta’s privacy policy](https://www.facebook.com/privacy/policy/)