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
  - llama3.3
datasets:
  - nvidia/HelpSteer3
base_model: meta-llama/Llama-3.3-70B-Instruct
library_name: transformers
---
# Model Overview

## Description:

Llama-3.3-Nemotron-70B-Reward-Principle is a large language model that leverages Meta-Llama-3.3-70B-Instruct as the foundation and is fine-tuned to predict the extent to which LLM-generated responses fulfils user-specified principles.

Given a conversation with multiple turns between the user and assistant (of up to 4,096 tokens), and a user-specified principle, it rates the quality of the final assistant turn using a reward score.

For the same prompt, a response with higher reward score fulfils the user-specified principle to a larger extent than another response with a lower reward score.

As of 24 Sep 2025, this model achieves [JudgeBench](https://huggingface.co/spaces/ScalerLab/JudgeBench) of 76.3% and [RM-Bench](https://arxiv.org/abs/2410.16184) of 83.6% which make it among the top Scalar Reward Models for both benchmarks.

See details on how this model was trained at [https://arxiv.org/abs/2509.21319](https://arxiv.org/abs/2509.21319)


## License/Terms of Use:

GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) . Additional Information: [Llama 3.3 Community License Agreement](https://www.llama.com/llama3_3/license/). Built with Llama.


### Deployment Geography

Global

## Use Case:

Llama-3.3-Nemotron-70B-Reward-Principle labels an LLM-generated response to a user query and a user-specified principle with a reward score.

## Release Date:

HuggingFace 10/27/2025 via https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward-Principle

## References:

* [RLBFF](https://arxiv.org/abs/2509.21319)
* [HelpSteer3](https://arxiv.org/abs/2503.04378)
* [HelpSteer3-Preference](https://arxiv.org/abs/2505.11475)
* [HelpSteer2-Preference](https://arxiv.org/abs/2410.01257)
* [SteerLM method](https://arxiv.org/abs/2310.05344)
* [HelpSteer](https://arxiv.org/abs/2311.09528)
* [HelpSteer2](https://arxiv.org/abs/2406.08673)
* [The future of AI: Built with Llama](https://ai.meta.com/blog/future-of-ai-built-with-llama/) 
* [Meta's Llama 3.3 Webpage](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3) 
* [Meta's Llama 3.3 Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md)



## RM-Bench LeaderBoard

As of 24 Sep 2025, our reward models are top performing reward models on [RM-Bench](https://arxiv.org/abs/2410.16184), an improved variant of RewardBench for evaluating Reward Models in Chat, Math, Code and Safety.

| Model  |  Chat | Math | Code | Safety | Easy | Normal | Hard | Overall RM-Bench|
|:-----------------------------|:------|:------|:------|:------|:------|:------|:------|:------|
| **[Llama-3.3-Nemotron-70B-Reward-Principle](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward-Principle/)**  | 85.3 | 81.9 | **70.4** | **96.9** | 85.5 | **84.9** | **80.5**	| **83.6** |
| **[Llama-3.3-Nemotron-70B-Reward-Multilingual](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward-Multilingual/)**  | **86.2** | 82.4 | 66.8 | 94.1 | 86.5 | 85.4 | 80.0	| 82.4 |
| **[Llama-3.3-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward)**  |75.4 | **84.5** | 69.3 | 90.4 | 92.1 | **85.7** |71.1 | 79.9|
| [Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) | 70.7 | 64.3 | 57.4 | 90.3 | **92.2** | 76.8 | 48.0 | 70.7 |
| [Skywork-Reward-Gemma-2-27B](https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B) | 71.8 | 59.2 | 56.6 | 94.3 | 89.6 | 75.4 | 50.0 | 70.5 |
| [Skywork-Reward-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B)  | 69.5 | 60.6 |  54.5 |95.7 | 89.0 | 74.7 | 46.6 | 70.1 | 

*Note that Skywork-Reward-Llama-3.1-8B was the best performing reward model reported on RM-Bench and we evaluated all other models.* 

## JudgeBench LeaderBoard

As of 24 Sep 2025, our reward models are the top performing Scalar reward models on [JudgeBench](https://huggingface.co/spaces/ScalerLab/JudgeBench), a popular benchmark for evaluating LLM-as-a-judge applications relating to General Knowledge, Logical Reasoning, Math and Coding.
 
 | Model  |  Knowl.| Reason.| Math | Code | Overall JudgeBench |
 |:-----------------------------|:------|:------|:------|:------|:------|
 | **[Llama-3.3-Nemotron-70B-Reward-Principle](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward-Principle)**  |**74.0** |74.5 | 82.1 | **81.0** |**76.3** |
 | **[Llama-3.3-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward)**  | 70.8 |**76.5** | 82.1 | **66.7** |73.7 |
 | **[Llama-3.3-Nemotron-70B-Reward-Multilingual](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward-Multilingual)**  |66.2 | 71.4 | 82.1 |59.5 | 69.4|
 | [Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) |62.3 | 72.5 | 76.8 | 57.1 | 66.9 |
 | [Skywork-Reward-Gemma-2-27B](https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B) |  59.7 | 66.3 | **83.9** | 50.0	| 64.3 |
 | [Skywork-Reward-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B) | 59.1 | 64.3 |  76.8 | 50.0 | 62.3 |

*Note that Skywork-Reward-Gemma-2-27B was the best performing reward model reported on JudgeBench and we evaluated all other models.* 


## Model Architecture: 
**Architecture Type:** Transformer <br>
**Network Architecture:** Llama 3.3 <br>

We developed this model using [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) as its foundation. This model contains 70 billion parameters.

## Input:
**Input Type(s):** Text <br>
**Input Format:** String <br>
**Input Parameters:** One Dimensional (1D) <br>
**Other Properties Related to Input:** Max of 128k tokens (but trained only on conversations up to 4K tokens) <br>

## Output:
**Output Type(s):** Float <br>
**Output Format:** One Single Float <br>
**Output Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Output:**  The float value represents the extent to which the response fulfils the user-specified principle, with a higher value representing greater fulfilment. <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):** <br>
* [NeMo - 24.05.llama.3.1] <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere <br>
* NVIDIA Hopper <br>
* NVIDIA Turing <br>

**Supported Operating System(s):** Linux <br>

## Quick Start

You can use the model using HuggingFace Transformers library with 2 or more 80GB GPUs (NVIDIA Ampere or newer) with at least 150GB of free disk space to accomodate the download.

This code has been tested on Transformers v4.45.0, torch v2.3.0a0+40ec155e58.nv24.3 and 2 H100 80GB GPUs, but any setup that supports meta-llama/Llama-3.1-70B-Instruct should support this model as well. If you run into problems, you can consider doing pip install -U transformers.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nvidia/Llama-3.3-Nemotron-70B-Reward-Principle"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is 1+1?"
good_response = "1+1=2"
bad_response = "1+1=3"
principle = "correctness"


principle_question = f"Evaluate the response to the previous prompt in terms of whether it satisfies this principle: {principle}. Only answer Yes or No."

for response in [good_response, bad_response]:
    messages = [{'role': "user", "content": prompt}, {'role': "assistant", "content": response}, {'role': "user", "content": principle_question}]
    tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    score_no = response_token_ids['scores'][0][0][2822].item()
    score_yes = response_token_ids['scores'][0][0][9642].item()
    reward = score_yes - score_no
    print(reward)
# Example reward - note that higher scores means higher principle fulfilment, and scores can be negative.
# reward for good_response = 4.875
# reward for bad_response = -4.875
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
For more detailed information on ethical considerations for this model, please see the Model Card++ [Explainability](explainability.md), [Bias](bias.md), [Safety & Security](safety.md), and [Privacy](privacy.md) Subcards.  

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Citation

If you find this model useful, please cite the following work:

```bibtex
@misc{wang2025rlbffbinaryflexiblefeedback,
      title={RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards}, 
      author={Zhilin Wang and Jiaqi Zeng and Olivier Delalleau and Ellie Evans and Daniel Egert and Hoo-Chang Shin and Felipe Soares and Yi Dong and Oleksii Kuchaiev},
      year={2025},
      eprint={2509.21319},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.21319}, 
}
```