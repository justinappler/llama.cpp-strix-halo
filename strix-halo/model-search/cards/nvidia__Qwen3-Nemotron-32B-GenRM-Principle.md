---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
inference: false
fine-tuning: false
language:
- en
tags:
  - nvidia
  - qwen3
datasets:
  - nvidia/HelpSteer3
base_model: Qwen/Qwen3-32B
library_name: transformers
---
# Model Overview

## Description:

Qwen3-Nemotron-32B-GenRM-Principle is a large language model that leverages Qwen3-32B as the foundation and is fine-tuned to predict the extent to which LLM-generated responses fulfils user-specified principles.

Given a conversation with multiple turns between user and assistant and a user-specified principle, it rates the quality of the final assistant turn using a reward score.

For the same prompt, a response with higher reward score fulfils the user-specified principle to a larger extent than another response with a lower reward score.

As of 24 Sep 2025, this model achieves [JudgeBench](https://huggingface.co/spaces/ScalerLab/JudgeBench) of 81.4% and [RM-Bench](https://arxiv.org/abs/2410.16184) of 86.2% which make it the top Generative Reward Models for both benchmarks.

See details on how this model was trained at [https://arxiv.org/abs/2509.21319](https://arxiv.org/abs/2509.21319)


## License/Terms of Use:

GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).


### Deployment Geography

Global

## Use Case:

Qwen3-Nemotron-32B-GenRM-Principle labels an LLM-generated response to a user query and a user-specified principle with a reward score.

## Release Date:

HuggingFace 10/27/2025 via https://huggingface.co/nvidia/Qwen3-Nemotron-32B-GenRM-Principle

## References:

* [RLBFF](https://arxiv.org/abs/2509.21319)
* [HelpSteer3](https://arxiv.org/abs/2503.04378)
* [HelpSteer3-Preference](https://arxiv.org/abs/2505.11475)
* [HelpSteer2-Preference](https://arxiv.org/abs/2410.01257)
* [SteerLM method](https://arxiv.org/abs/2310.05344)
* [HelpSteer](https://arxiv.org/abs/2311.09528)
* [HelpSteer2](https://arxiv.org/abs/2406.08673)

## RM-Bench LeaderBoard

As of 24 Sep 2025, our reward model is the top performing generative reward models on [RM-Bench](https://arxiv.org/abs/2410.16184), an improved variant of RewardBench for evaluating Reward Models in Chat, Math, Code and Safety.

| Model  |  Chat | Math | Code | Safety | Easy | Normal | Hard | Overall RM-Bench|
|:-----------------------------|:------|:------|:------|:------|:------|:------|:------|:------|
|**[Qwen3-Nemotron-32B-GenRM-Principle](https://huggingface.co/nvidia/Qwen3-Nemotron-32B-GenRM-Principle)** | 80.4 | 92.0 | 77.0 | 95.5 | 88.9 | 86.4 | 83.4 |**86.2** | 
|[Llama-3_3-Nemotron-Super-49B-GenRM](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-GenRM) | 73.7 |	91.4 |	75.0 |	90.6 |	91.2 |	85.7 |	71.2 |	82.7 | 
|[RewardAnything-8B-v1](https://huggingface.co/WisdomShell/RewardAnything-8B-v1) | 76.7 | 90.3 | 75.2 | 90.2 | 85.6 | 82.2 | 81.5 | 83.1 | 
|[RM-R1-DeepSeek-Distilled-Qwen-32B](https://huggingface.co/gaotang/RM-R1-DeepSeek-Distilled-Qwen-32B) | 74.2 | 	91.8 | 74.1 | 95.4 | 89.5 | 85.4 | 76.7 | 83.9 |
|[R3-QWEN3-14B-LORA-4K](https://huggingface.co/rubricreward/R3-Qwen3-14B-LoRA-4k) | 76.5 | 92.4 | 78.7 | 91.9 | 91.4 |  86.2 | 77.1 | 84.9 |
## JudgeBench LeaderBoard

As of 24 Sep 2025, our reward model is the top performing models on [JudgeBench](https://huggingface.co/spaces/ScalerLab/JudgeBench), a popular benchmark for evaluating LLM-as-a-judge applications relating to General Knowledge, Logical Reasoning, Math and Coding.
 
| Model  |  Knowl.| Reason.| Math | Code | Overall JudgeBench |
|:-----------------------------|:------|:------|:------|:------|:------|
| **[Qwen3-Nemotron-32B-GenRM-Principle](https://huggingface.co/nvidia/Qwen3-Nemotron-32B-GenRM-Principle)** | 74.6 | 85.7 | 85.7 | 90.5 | **81.4** |
| [Llama-3_3-Nemotron-Super-49B-GenRM](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-GenRM) | 71.4 |	73.5 |	87.5	| 76.2 |	75.1 |
| [RewardAnything-8B-v1](https://huggingface.co/WisdomShell/RewardAnything-8B-v1) | 61.0 |  57.1 | 73.2 | 66.7 | 62.6 |
| [RM-R1-DeepSeek-Distilled-Qwen-32B](https://huggingface.co/gaotang/RM-R1-DeepSeek-Distilled-Qwen-32B) | 56.5 | 66.3 | 85.7 | 73.8 | 66.0|
| [R3-QWEN3-14B-LORA-4K](https://huggingface.co/rubricreward/R3-Qwen3-14B-LoRA-4k) | 50.0 | 64.3 | 76.8 | 71.4 | 60.9 | 

## Model Architecture: 
**Architecture Type:** Transformer <br>
**Network Architecture:** Qwen3 <br>

We developed this model using [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) as its foundation. This model contains 32 billion parameters.

## Input:
**Input Type(s):** Text <br>
**Input Format:** String <br>
**Input Parameters:** One Dimensional (1D) <br>
**Other Properties Related to Input:** Max of 128k tokens (but trained only on conversations up to 8K tokens) <br>

## Output:
**Output Type(s):** Float <br>
**Output Format:** One Single Float <br>
**Output Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Output:**  The float value represents the extent to which the response fulfils the user-specified principle, with a higher value representing greater fulfilment. <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):** <br>
* [NeMo-RL - 0.3] <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere <br>
* NVIDIA Hopper <br>
* NVIDIA Turing <br>

**Supported Operating System(s):** Linux <br>

## Quick Start

You can use the model using HuggingFace Transformers library with 1 or more 80GB GPUs (NVIDIA Ampere or newer) with at least 70GB of free disk space to accomodate the download. Alternatively, you can use vLLM for accelerated inference.

This code has been tested on Transformers v4.57.0, torch v2.3.0a0+40ec155e58.nv24.3 and 1 H100 80GB GPUs, but any setup that supports Qwen/Qwen3-32B should support this model as well. If you run into problems, you can consider doing pip install -U transformers.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nvidia/Qwen3-Nemotron-32B-GenRM-Principle"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is 1+1?"
good_response = "1+1=2"
bad_response = "1+1=3"
principle = "correctness"

for response in [good_response, bad_response]:
    messages = [{'role': "user", "content": prompt}, {'role': "assistant", "content": response}, {'role': "principle", "content": principle}]
    tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=16000, return_dict_in_generate=True, output_scores=True)
    response = tokenizer.decode(response_token_ids.sequences[0].tolist())
    score_max = torch.max(response_token_ids.scores[-2][0]).item() # normalize max score to zero to match vLLM logprobs and clip others to -50 if they return -inf for too small of a value. The same should be done if the required tokens is not returned by vLLM's top k logprobs.
    score_no = max(response_token_ids.scores[-2][0][2308].item() - score_max, -50)  # token for " No"
    score_yes = max(response_token_ids.scores[-2][0][7414].item() - score_max, -50) # token for " Yes"
    reward = score_yes-score_no
    print(response)
    print(reward)
```

## Model Version: 
v1.0

# Training, Testing and Evaluation Datasets: 

## Training Datasets:

**Dataset Name:** HelpSteer3 <br>
**Dataset Link:** https://huggingface.co/datasets/nvidia/HelpSteer3

**Data Collection Method by dataset** <br>
* [Hybrid: Human, Synthetic] <br>

**Labeling Method by dataset** <br>
* [Human] <br>

**Properties:** <br>
* 77,564 prompt-responses, each annotated with up to 3 annotations of free-text feedback (each being 50-250 words long) elaborating upon the overall helpfulness of the response.

## Testing Datasets:

**Dataset Name:** HelpSteer3 <br>
**Dataset Link:** https://huggingface.co/datasets/nvidia/HelpSteer3

**Data Collection Method by dataset** <br>
* [Hybrid: Human, Synthetic] <br>

**Labeling Method by dataset** <br>
* [Human] <br>

**Properties:** <br>
* 4,078 prompt-responses, each annotated with up to 3 annotations of free-text feedback (each being 50-250 words long) elaborating upon the overall helpfulness of the response.

## Evaluation Datasets

**Dataset Name:** RM-Bench <br>
**Dataset Link:** https://huggingface.co/datasets/THU-KEG/RM-Bench

**Data Collection Method by dataset** <br>
* [Hybrid: Human, Synthetic] <br>

**Labeling Method by dataset** <br>
* [Hybrid: Human, Synthetic] <br>

**Properties:** <br>
* 1,327 prompts, each with three pairs of responses as well as preferences between the pair of responses.


**Dataset Name:** JudgeBench <br>
**Dataset Link:** https://huggingface.co/datasets/ScalerLab/JudgeBench

**Data Collection Method by dataset** <br>
* [Hybrid: Human, Synthetic] <br>

**Labeling Method by dataset** <br>
* [Hybrid: Human, Synthetic] <br>

**Properties:** <br>
* 350 prompts, each with a pair of responses as well as preferences between the pair of responses.



# Inference:
**Engine:** PyTorch <br>
**Test Hardware:** H100, A100 80GB, A100 40GB <br>


## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 
For more detailed information on ethical considerations for this model, please see the Model Card++ [Explainability](explainability.md), [Bias](bias.md), [Safety and Security](safety.md), and [Privacy](privacy.md) Subcards.  

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Citation

If you find this model useful, please cite the following work:

```bibtex
@misc{wang2025rlbffbinaryflexiblefeedback,
      title={RLBFF: Binary Flexible Feedback to bridge between Human Feedback | Verifiable Rewards}, 
      author={Zhilin Wang and Jiaqi Zeng and Olivier Delalleau and Ellie Evans and Daniel Egert and Hoo-Chang Shin and Felipe Soares and Yi Dong and Oleksii Kuchaiev},
      year={2025},
      eprint={2509.21319},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.21319}, 
}
```