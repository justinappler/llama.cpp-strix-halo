---
license: other
license_name: tongyi-qianwen
license_link: >-
  https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
tags:
- pretrained
- moe
---

# Qwen1.5-MoE-A2.7B


## Introduction

Qwen1.5-MoE is a transformer-based MoE decoder-only language model pretrained on a large amount of data. 

For more details, please refer to our [blog post](https://qwenlm.github.io/blog/qwen-moe/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).

## Model Details
Qwen1.5-MoE employs Mixture of Experts (MoE) architecture, where the models are upcycled from dense language models. For instance, `Qwen1.5-MoE-A2.7B` is upcycled from `Qwen-1.8B`. It has 14.3B parameters in total and 2.7B activated parameters during runtime, while achieving comparable performance to `Qwen1.5-7B`, it only requires 25% of the training resources. We also observed that the inference speed is 1.74 times that of `Qwen1.5-7B`.

## Requirements
The code of Qwen1.5-MoE has been in the latest Hugging face transformers and we advise you to build from source with command `pip install git+https://github.com/huggingface/transformers`, or you might encounter the following error:
```
KeyError: 'qwen2_moe'.
```

## Usage

We do not advise you to use base language models for text generation. Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., on this model.
