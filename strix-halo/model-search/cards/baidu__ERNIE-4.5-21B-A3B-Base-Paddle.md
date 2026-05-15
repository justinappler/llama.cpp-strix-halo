---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-generation
tags:
- ERNIE4.5
library_name: PaddlePaddle
---

<div align="center" style="line-height: 1;">
  <a href="https://ernie.baidu.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖_Chat-ERNIE_Bot-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/baidu" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Baidu-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/PaddlePaddle/ERNIE" target="_blank" style="margin: 2px;">
    <img alt="Github" src="https://img.shields.io/badge/GitHub-ERNIE-000?logo=github&color=0000FF" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://ernie.baidu.com/blog/ernie4.5" target="_blank" style="margin: 2px;">
    <img alt="Blog" src="https://img.shields.io/badge/🖖_Blog-ERNIE4.5-A020A0" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://discord.gg/JPmZXDsEEK" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-ERNIE-5865F2?logo=discord&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/PaddlePaddle" target="_blank" style="margin: 2px;">
    <img alt="X" src="https://img.shields.io/badge/X-PaddlePaddle-6080F0"?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="#license" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache2.0-A5de54" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

# ERNIE-4.5-21B-A3B-Base

> [!NOTE]
> Note: "**-Paddle**" models use [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) weights, while "**-PT**" models use Transformer-style PyTorch weights.

> [!NOTE]
> Note: The Base model only supports text completion. For evaluation, use the `completion` API (not `chat_completion`) in vLLM/FastDeploy.

## ERNIE 4.5 Highlights

The advanced capabilities of the ERNIE 4.5 models, particularly the MoE-based A47B and A3B series, are underpinned by several key technical innovations:

1. **Multimodal Heterogeneous MoE Pre-Training:** Our models are jointly trained on both textual and visual modalities to better capture the nuances of multimodal information and improve performance on tasks involving text understanding and generation, image understanding, and cross-modal reasoning. To achieve this without one modality hindering the learning of another, we designed a *heterogeneous MoE structure*, incorporated *modality-isolated routing*, and employed *router orthogonal loss* and *multimodal token-balanced loss*. These architectural choices ensure that both modalities are effectively represented, allowing for mutual reinforcement during training.

2. **Scaling-Efficient Infrastructure:** We propose a novel heterogeneous hybrid parallelism and hierarchical load balancing strategy for efficient training of ERNIE 4.5 models. By using intra-node expert parallelism, memory-efficient pipeline scheduling, FP8 mixed-precision training and finegrained recomputation methods, we achieve remarkable pre-training throughput. For inference, we propose *multi-expert parallel collaboration* method and *convolutional code quantization* algorithm to achieve 4-bit/2-bit lossless quantization. Furthermore, we introduce PD disaggregation with dynamic role switching for effective resource utilization to enhance inference performance for ERNIE 4.5 MoE models. Built on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle), ERNIE 4.5 delivers high-performance inference across a wide range of hardware platforms.

3. **Modality-Specific Post-Training:** To meet the diverse requirements of real-world applications, we fine-tuned variants of the pre-trained model for specific modalities. Our LLMs are optimized for general-purpose language understanding and generation. The VLMs focuses on visuallanguage understanding and supports both thinking and non-thinking modes. Each model employed a combination of *Supervised Fine-tuning (SFT)*, *Direct Preference Optimization (DPO)* or a modified reinforcement learning method named *Unified Preference Optimization (UPO)* for post-training.

To ensure the stability of multimodal joint training, we adopt a staged training strategy. In the first and second stage, we train only the text-related parameters, enabling the model to develop strong fundamental language understanding as well as long-text processing capabilities. The final multimodal stage extends capabilities to images and videos by introducing additional parameters including a ViT for image feature extraction, an adapter for feature transformation, and visual experts for multimodal understanding. At this stage, text and visual modalities mutually enhance each other. After pretraining trillions tokens, we extracted the text-related parameters and finally obtained ERNIE-4.5-21B-A3B-Base.

## Model Overview

ERNIE-4.5-21B-A3B-Base is a text MoE Base model, with 21B total parameters and 3B activated parameters for each token. The following are the model configuration details:

| Key                               | Value       |
| --------------------------------- | ----------- |
| Modality                          | Text        |
| Training Stage                    | Pretraining |
| Params(Total / Activated)         | 21B / 3B    |
| Layers                            | 28          |
| Heads(Q/KV)                       | 20 / 4      |
| Text Experts(Total / Activated)   | 64 / 6      |
| Vision Experts(Total / Activated) | 64 / 6      |
| Shared Experts                    | 2           |
| Context Length                    | 131072      |

## Quickstart

### Model Finetuning with ERNIEKit

[ERNIEKit](https://github.com/PaddlePaddle/ERNIE) is a training toolkit based on PaddlePaddle, specifically designed for the ERNIE series of open-source large models. It provides comprehensive support for scenarios such as instruction fine-tuning (SFT, LoRA) and alignment training (DPO), ensuring optimal performance.

Usage Examples:

```bash
# Download model
huggingface-cli download baidu/ERNIE-4.5-21B-A3B-Base-Paddle --local-dir baidu/ERNIE-4.5-21B-A3B-Base-Paddle
# SFT
erniekit train examples/configs/ERNIE-4.5-21B-A3B/sft/run_sft_lora_8k.yaml model_name_or_path=baidu/ERNIE-4.5-21B-A3B-Base-Paddle
# DPO
erniekit train examples/configs/ERNIE-4.5-21B-A3B/dpo/run_dpo_lora_8k.yaml model_name_or_path=baidu/ERNIE-4.5-21B-A3B-Base-Paddle
```

For more detailed examples, including SFT with LoRA, multi-GPU configurations, and advanced scripts, please refer to the examples folder within the [ERNIEKit](https://github.com/PaddlePaddle/ERNIE) repository.

### FastDeploy Inference

Service deployment can be quickly completed using FastDeploy in the following command. For more detailed usage instructions, please refer to the [FastDeploy Repository](https://github.com/PaddlePaddle/FastDeploy).

**Note**: For single-card deployment, at least 80G of GPU memory resources are required.

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-21B-A3B-Base-Paddle \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768
       --max-num-seqs 32
```

## License

The ERNIE 4.5 models are provided under the Apache License 2.0. This license permits commercial use, subject to its terms and conditions. Copyright (c) 2025 Baidu, Inc. All Rights Reserved.

## Citation

If you find ERNIE 4.5 useful or wish to use it in your projects, please kindly cite our technical report:

```bibtex
@misc{ernie2025technicalreport,
      title={ERNIE 4.5 Technical Report},
      author={Baidu ERNIE Team},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={}
}
```
