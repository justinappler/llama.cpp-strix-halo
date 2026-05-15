---
base_model:
- Qwen/Qwen2.5-32B-Instruct
datasets:
- nvidia/OpenCodeReasoning
language:
- en
library_name: transformers
tags:
- nvidia
- code
pipeline_tag: text-generation
---

# OpenCodeReasoning-Nemotron-1.1-32B Overview

## Description: <br>
OpenCodeReasoning-Nemotron-1.1-32B is a large language model (LLM) which is a derivative of Qwen2.5-32B-Instruct (AKA the reference model). It is a reasoning model that is post-trained for reasoning for code generation. The model supports a context length of 64k tokens. <br>

This model is ready for commercial/non-commercial use. <br>

![Evaluation Results](./results.png)


## Results

Below results are the average of **64 evaluations** on LiveCodeBench (v5) [2408-2501].

| Model                                  | Pass@1             |
|:---------------------------------------|:-------------------|
| DeepSeek-R1-0528                       | 73.4               |
| DeepSeek-R1                            | 65.6               |
| QwQ-32B                                | 61.3               |
|                                        |                    |
| **Distilled 7B+ Models**               |                    |
|                                        |                    |
| Bespoke-Stratos-7B                     | 14.7               |
| OpenThinker-7B                         | 25.5               |
| R1-Distill-Qwen-7B                     | 38.0               |
| OlympicCoder-7B                        | 40.9               |
| **OpenCodeReasoning-Nemotron-7B**      | **51.3**           |
| **OpenCodeReasoning-Nemotron-1.1-7B**  | **55.5**           |
|                                        |                    |
| **Distilled 14B+ Models**              |                    |
|                                        |                    |
| R1-Distill-Qwen-14B                    | 51.3               |
| **OpenCodeReasoning-Nemotron-14B**     | **59.4**           |
| **OpenCodeReasoning-Nemotron-1.1-14B** | **65.9**           |
|                                        |                    |
| **Distilled 32B+ Models**              |                    |
|                                        |                    |                
| Bespoke-Stratos-32B                    | 30.1               |
| OpenThinker-32B                        | 54.1               |
| R1-Distill-Qwen-32B                    | 58.1               |
| OlympicCoder-32B                       | 57.4               |
| **OpenCodeReasoning-Nemotron-32B**     | **61.7**           |
| **OpenCodeReasoning-Nemotron-1.1-32B** | **69.9**           |


## Reproducing our results

* [Models](https://huggingface.co/collections/nvidia/opencodereasoning-67ec462892673a326c0696c1)
* [Dataset](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
* [Paper](https://arxiv.org/abs/2504.01943)


## How to use the models?

To run inference on coding problems:

````python
import transformers
import torch

model_id = "nvidia/OpenCodeReasoning-Nemotron-1.1-32B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prompt = """You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.

Please use python programming language only.

You must use ```python for just the final solution code block with the following format:
```python
# Your code here
```

{user}
"""

messages = [
    {
        "role": "user",
        "content": prompt.format(user="Write a program to calculate the sum of the first $N$ fibonacci numbers")
    },
]

outputs = pipeline(
    messages,
    max_new_tokens=49152,
)
print(outputs[0]["generated_text"][-1]['content'])

````


## Citation

If you find the data useful, please cite:
```
@article{ahmad2025opencodereasoning,
      title={OpenCodeReasoning: Advancing Data Distillation for Competitive Coding}, 
      author={Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, Boris Ginsburg},
      year={2025},
      eprint={2504.01943},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.01943}, 
}
```

## Additional Information

## Model Architecture: <br>
Architecture Type: Dense decoder-only Transformer model
Network Architecture: Qwen-32B-Instruct 
<br>
**This model was developed based on Qwen2.5-32B-Instruct and has 32B model parameters. <br>**
**OpenCodeReasoning-Nemotron-1.1-32B was developed based on Qwen2.5-32B-Instruct and has 32B model parameters. <br>**

## Input: <br>
**Input Type(s):** Text <br>
**Input Format(s):** String <br>
**Input Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Input:** Context length up to 65,536 tokens <br>

## Output: <br>
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Output:** Context length up to 65,536 tokens <br> 

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>   

## Software Integration : <br>
* Runtime Engine: NeMo 2.3.0 <br>
* Recommended Hardware Microarchitecture Compatibility: <br>
NVIDIA Ampere <br>
NVIDIA Hopper <br>
* Preferred/Supported Operating System(s): Linux <br> 

## Model Version(s): 
1.1 (07/08/2025)  <br>
OpenCodeReasoning-Nemotron-1.1-7B<br>
OpenCodeReasoning-Nemotron-1.1-14B<br>
OpenCodeReasoning-Nemotron-1.1-32B<br>


# Training and Evaluation Datasets: <br>   

## Training Dataset:

The training corpus for OpenCodeReasoning-Nemotron-1.1-32B is [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) dataset, which is composed of competitive programming questions and DeepSeek-R1-0528 generated responses. 

Data Collection Method: Hybrid: Automated, Human, Synthetic <br>
Labeling Method: Hybrid: Automated, Human, Synthetic <br>
Properties: 1.165M samples from OpenCodeReasoning (https://huggingface.co/datasets/nvidia/OpenCodeReasoning)

## Evaluation Dataset:
We used the datasets listed in the next section to evaluate OpenCodeReasoning-Nemotron-1.1-32B. <br>
**Data Collection Method: Hybrid: Automated, Human, Synthetic <br>**
**Labeling Method: Hybrid: Automated, Human, Synthetic <br>**


### License/Terms of Use: <br> 
GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Additional Information: [Apache License Version 2.0](https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/LICENSE).

### Deployment Geography:
Global<br>

### Use Case: <br>
This model is intended for developers and researchers building LLMs. <br>

### Release Date:  <br>
Huggingface [07/08/2025] via https://huggingface.co/nvidia/OpenCodeReasoning-Nemotron-1.1-32B/ <br> 

## Reference(s):
[2504.01943] OpenCodeReasoning: Advancing Data Distillation for Competitive Coding
<br>

## Inference:
**Engine:** vLLM <br>
**Test Hardware** NVIDIA H100-80GB <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.  

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).