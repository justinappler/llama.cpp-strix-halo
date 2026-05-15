---
library_name: transformers
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

pipeline_tag: text-generation
language:
  - en
tags:
  - nvidia
  - llama-3
  - pytorch
---

# Llama-3.1-Nemotron-Nano-8B-v1


## Model Overview 

Llama-3.1-Nemotron-Nano-8B-v1 is a large language model (LLM) which is a derivative of [Meta Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (AKA the reference model). It is a reasoning model that is post trained for reasoning, human chat preferences, and tasks, such as RAG and tool calling. 

Llama-3.1-Nemotron-Nano-8B-v1 is a model which offers a great tradeoff between model accuracy and efficiency. It is created from Llama 3.1 8B Instruct and offers improvements in model accuracy. The model fits on a single RTX GPU and can be used locally. The model supports a context length of 128K.

This model underwent a multi-phase post-training process to enhance both its reasoning and non-reasoning capabilities. This includes a supervised fine-tuning stage for Math, Code, Reasoning, and Tool Calling as well as multiple reinforcement learning (RL) stages using REINFORCE (RLOO) and Online Reward-aware Preference Optimization (RPO) algorithms for both chat and instruction-following. The final model checkpoint is obtained after merging the final SFT and Online RPO checkpoints. Improved using Qwen.

This model is part of the Llama Nemotron Collection. You can find the other model(s) in this family here: 
[Llama-3.3-Nemotron-Super-49B-v1](https://huggingface.co/nvidia/Llama-3.3-Nemotron-Super-49B-v1)

This model is ready for commercial use.

## Feature Voting

We want to hear from you! Share your ideas, vote on what matters, and help [shape the future of Nemotron](https://nemotron.ideas.nvidia.com/).

## License/Terms of Use

GOVERNING TERMS: Your use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Additional Information: [Llama 3.1 Community License Agreement](https://www.llama.com/llama3_1/license/). Built with Llama.

**Model Developer:** NVIDIA

**Model Dates:** Trained between August 2024 and March 2025

**Data Freshness:** The pretraining data has a cutoff of 2023 per Meta Llama 3.1 8B


## Use Case: 

Developers designing AI Agent systems, chatbots, RAG systems, and other AI-powered applications. Also suitable for typical instruction-following tasks. Balance of model accuracy and compute efficiency (the model fits on a single RTX GPU and can be used locally).

## Release Date: <br>
3/18/2025 <br>

## References

- [\[2505.00949\] Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)
- [\[2502.00203\] Reward-aware Preference Optimization: A Unified Mathematical Framework for Model Alignment](https://arxiv.org/abs/2502.00203)


## Model Architecture

**Architecture Type:** Dense decoder-only Transformer model

**Network Architecture:** Llama 3.1 8B Instruct

## Intended use

Llama-3.1-Nemotron-Nano-8B-v1 is a general purpose reasoning and chat model intended to be used in English and coding languages. Other non-English languages (German, French, Italian, Portuguese, Hindi, Spanish, and Thai) are also supported. 

# Input:
- **Input Type:** Text
- **Input Format:** String
- **Input Parameters:** One-Dimensional (1D)
- **Other Properties Related to Input:** Context length up to 131,072 tokens

## Output:
- **Output Type:** Text
- **Output Format:** String
- **Output Parameters:** One-Dimensional (1D)
- **Other Properties Related to Output:** Context length up to 131,072 tokens

## Model Version:
1.0 (3/18/2025)

## Software Integration
- **Runtime Engine:** NeMo 24.12 <br>
- **Recommended Hardware Microarchitecture Compatibility:**
    - NVIDIA Hopper
    - NVIDIA Ampere

## Quick Start and Usage Recommendations:

1. Reasoning mode (ON/OFF) is controlled via the system prompt, which must be set as shown in the example below. All instructions should be contained within the user prompt
2. We recommend setting temperature to `0.6`, and Top P to `0.95` for Reasoning ON mode
3. We recommend using greedy decoding for Reasoning OFF mode
4. We have provided a list of prompts to use for evaluation for each benchmark where a specific template is required
5. The model will include `<think></think>` if no reasoning was necessary in Reasoning ON model, this is expected behaviour

You can try this model out through the preview API, using this link: [Llama-3.1-Nemotron-Nano-8B-v1](https://build.nvidia.com/nvidia/llama-3_1-nemotron-nano-8b-v1).

See the snippet below for usage with Hugging Face Transformers library. Reasoning mode (ON/OFF) is controlled via system prompt. Please see the example below.
Our code requires the transformers package version to be `4.44.2` or higher.


### Example of “Reasoning On:”

```python
import torch
import transformers

model_id = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

pipeline = transformers.pipeline(
   "text-generation",
   model=model_id,
   tokenizer=tokenizer,
   max_new_tokens=32768,
   temperature=0.6,
   top_p=0.95,
   **model_kwargs
)

# Thinking can be "on" or "off"
thinking = "on"

print(pipeline([{"role": "system", "content": f"detailed thinking {thinking}"}, {"role": "user", "content": "Solve x*(sin(x)+2)=0"}]))
```


### Example of “Reasoning Off:”

```python
import torch
import transformers

model_id = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

pipeline = transformers.pipeline(
   "text-generation",
   model=model_id,
   tokenizer=tokenizer,
   max_new_tokens=32768,
   do_sample=False,
   **model_kwargs
)

# Thinking can be "on" or "off"
thinking = "off"

print(pipeline([{"role": "system", "content": f"detailed thinking {thinking}"}, {"role": "user", "content": "Solve x*(sin(x)+2)=0"}]))
```

For some prompts, even though thinking is disabled, the model emergently prefers to think before responding. But if desired, the users can prevent it by pre-filling the assistant response.

```python
import torch
import transformers

model_id = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Thinking can be "on" or "off"
thinking = "off"

pipeline = transformers.pipeline(
   "text-generation",
   model=model_id,
   tokenizer=tokenizer,
   max_new_tokens=32768,
   do_sample=False,
   **model_kwargs
)

print(pipeline([{"role": "system", "content": f"detailed thinking {thinking}"}, {"role": "user", "content": "Solve x*(sin(x)+2)=0"}, {"role":"assistant", "content":"<think>\n</think>"}]))
```

## Inference:
**Engine:** Transformers
**Test Hardware:**

- BF16:
    - 1x RTX 50 Series GPUs
    - 1x RTX 40 Series GPUs
    - 1x RTX 30 Series GPUs
    - 1x H100-80GB GPU
    - 1x A100-80GB GPU
    - Jetson AGX Thor


**Preferred/Supported] Operating System(s):** Linux <br>

## Training Datasets

A large variety of training data was used for the post-training pipeline, including manually annotated data and synthetic data.

The data for the multi-stage post-training phases for improvements in Code, Math, and Reasoning is a compilation of SFT and RL data that supports improvements of math, code, general reasoning, and instruction following capabilities of the original Llama instruct model. 

Prompts have been sourced from either public and open corpus or synthetically generated. Responses were synthetically generated by a variety of models, with some prompts containing responses for both Reasoning On and Off modes, to train the model to distinguish between two modes. 

**Data Collection for Training Datasets:** <br>
* Hybrid: Automated, Human, Synthetic <br>

**Data Labeling for Training Datasets:** <br>
* N/A <br>

## Evaluation Datasets

We used the datasets listed below to evaluate Llama-3.1-Nemotron-Nano-8B-v1. 

**Data Collection for Evaluation Datasets:** Hybrid: Human/Synthetic

**Data Labeling for Evaluation Datasets:** Hybrid: Human/Synthetic/Automatic

## Evaluation Results

These results contain both “Reasoning On”, and “Reasoning Off”. We recommend using temperature=`0.6`, top_p=`0.95` for “Reasoning On” mode, and greedy decoding for “Reasoning Off” mode. All evaluations are done with 32k sequence length. We run the benchmarks up to 16 times and average the scores to be more accurate.

> NOTE: Where applicable, a Prompt Template will be provided. While completing benchmarks, please ensure that you are parsing for the correct output format as per the provided prompt in order to reproduce the benchmarks seen below. 

### MT-Bench

| Reasoning Mode | Score |
|--------------|------------|
| Reasoning Off | 7.9 |
| Reasoning On | 8.1 |


### MATH500

| Reasoning Mode | pass@1 |
|--------------|------------|
| Reasoning Off | 36.6% | 
| Reasoning On | 95.4%  |

User Prompt Template: 

```
"Below is a math question. I want you to reason through the steps and then give a final answer. Your final answer should be in \boxed{}.\nQuestion: {question}"
```


### AIME25

| Reasoning Mode | pass@1 |
|--------------|------------|
| Reasoning Off | 0% | 
| Reasoning On | 47.1% |

User Prompt Template: 

```
"Below is a math question. I want you to reason through the steps and then give a final answer. Your final answer should be in \boxed{}.\nQuestion: {question}"
```


### GPQA-D

| Reasoning Mode | pass@1 |
|--------------|------------|
| Reasoning Off | 39.4% | 
| Reasoning On | 54.1% |

User Prompt Template: 


```
"What is the correct answer to this question: {question}\nChoices:\nA. {option_A}\nB. {option_B}\nC. {option_C}\nD. {option_D}\nLet's think step by step, and put the final answer (should be a single letter A, B, C, or D) into a \boxed{}"
```


### IFEval Average

| Reasoning Mode | Strict:Prompt | Strict:Instruction |
|--------------|------------|------------|
| Reasoning Off | 74.7% | 82.1% |
| Reasoning On | 71.9% | 79.3% |

### BFCL v2 Live

| Reasoning Mode | Score |
|--------------|------------|
| Reasoning Off | 63.9% | 
| Reasoning On | 63.6% | 

User Prompt Template:


```
<AVAILABLE_TOOLS>{functions}</AVAILABLE_TOOLS>

{user_prompt}
```


### MBPP 0-shot

| Reasoning Mode | pass@1 |
|--------------|------------|
| Reasoning Off | 66.1% | 
| Reasoning On | 84.6% |

User Prompt Template:


````
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Here is the given problem and test examples:
{prompt}
Please use the python programming language to solve this problem.
Please make sure that your code includes the functions from the test samples and that the input and output formats of these functions match the test samples.
Please return all completed codes in one code block.
This code block should be in the following format:
```python
# Your codes here
```
````


## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

For more detailed information on ethical considerations for this model, please see the Model Card++ [Explainability](explainability.md), [Bias](bias.md), [Safety & Security](safety.md), and [Privacy](privacy.md) Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


## Citation
```
@misc{bercovich2025llamanemotronefficientreasoningmodels,
      title={Llama-Nemotron: Efficient Reasoning Models}, 
      author={Akhiad Bercovich and Itay Levy and Izik Golan and Mohammad Dabbah and Ran El-Yaniv and Omri Puny and Ido Galil and Zach Moshe and Tomer Ronen and Najeeb Nabwani and Ido Shahaf and Oren Tropp and Ehud Karpas and Ran Zilberstein and Jiaqi Zeng and Soumye Singhal and Alexander Bukharin and Yian Zhang and Tugrul Konuk and Gerald Shen and Ameya Sunil Mahabaleshwarkar and Bilal Kartal and Yoshi Suhara and Olivier Delalleau and Zijia Chen and Zhilin Wang and David Mosallanezhad and Adi Renduchintala and Haifeng Qian and Dima Rekesh and Fei Jia and Somshubra Majumdar and Vahid Noroozi and Wasi Uddin Ahmad and Sean Narenthiran and Aleksander Ficek and Mehrzad Samadi and Jocelyn Huang and Siddhartha Jain and Igor Gitman and Ivan Moshkov and Wei Du and Shubham Toshniwal and George Armstrong and Branislav Kisacanin and Matvei Novikov and Daria Gitman and Evelina Bakhturina and Jane Polak Scowcroft and John Kamalu and Dan Su and Kezhi Kong and Markus Kliegl and Rabeeh Karimi and Ying Lin and Sanjeev Satheesh and Jupinder Parmar and Pritam Gundecha and Brandon Norick and Joseph Jennings and Shrimai Prabhumoye and Syeda Nahida Akter and Mostofa Patwary and Abhinav Khattar and Deepak Narayanan and Roger Waleffe and Jimmy Zhang and Bor-Yiing Su and Guyue Huang and Terry Kong and Parth Chadha and Sahil Jain and Christine Harvey and Elad Segal and Jining Huang and Sergey Kashirsky and Robert McQueen and Izzy Putterman and George Lam and Arun Venkatesan and Sherry Wu and Vinh Nguyen and Manoj Kilaru and Andrew Wang and Anna Warno and Abhilash Somasamudramath and Sandip Bhaskar and Maka Dong and Nave Assaf and Shahar Mor and Omer Ullman Argov and Scot Junkin and Oleksandr Romanenko and Pedro Larroy and Monika Katariya and Marco Rovinelli and Viji Balas and Nicholas Edelman and Anahita Bhiwandiwalla and Muthu Subramaniam and Smita Ithape and Karthik Ramamoorthy and Yuting Wu and Suguna Varshini Velury and Omri Almog and Joyjit Daw and Denys Fridman and Erick Galinkin and Michael Evans and Katherine Luna and Leon Derczynski and Nikki Pope and Eileen Long and Seth Schneider and Guillermo Siman and Tomasz Grzegorzek and Pablo Ribalta and Monika Katariya and Joey Conway and Trisha Saar and Ann Guan and Krzysztof Pawelec and Shyamala Prayaga and Oleksii Kuchaiev and Boris Ginsburg and Oluwatobi Olabiyi and Kari Briski and Jonathan Cohen and Bryan Catanzaro and Jonah Alben and Yonatan Geifman and Eric Chung and Chris Alexiuk},
      year={2025},
      eprint={2505.00949},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.00949}, 
}
```