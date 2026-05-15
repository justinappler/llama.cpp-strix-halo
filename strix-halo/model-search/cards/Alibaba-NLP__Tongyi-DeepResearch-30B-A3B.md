---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# Introduction

We present  **Tongyi DeepResearch**, an agentic large language model featuring 30 billion total parameters, with only 3 billion activated per token. Developed by Tongyi Lab, the model is specifically designed for **long-horizon, deep information-seeking** tasks. Tongyi-DeepResearch demonstrates state-of-the-art performance across a range of agentic search benchmarks, including Humanity's Last Exam, BrowserComp, BrowserComp-ZH, WebWalkerQA, GAIA, xbench-DeepSearch and FRAMES.

More details can be found in our 📰 [Tech Blog](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research).

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63fc4c00a3c067e62899d32b/OhQCYYJu1LhrS446Qct5D.png)

## Key Features

- ⚙️ **Fully automated synthetic data generation pipeline**: We design a highly scalable data synthesis pipeline, which is fully automatic and empowers agentic pre-training, supervised fine-tuning, and reinforcement learning.
- 🔄 **Large-scale continual pre-training on agentic data**: Leveraging diverse, high-quality agentic interaction data to extend model capabilities, maintain freshness, and strengthen reasoning performance.
- 🔁 **End-to-end reinforcement learning**: We employ a strictly on-policy RL approach based on a customized Group Relative Policy Optimization framework, with token-level policy gradients, leave-one-out advantage estimation, and selective filtering of negative samples to stabilize training in a non‑stationary environment.
- 🤖 **Agent Inference Paradigm Compatibility**: At inference, Tongyi-DeepResearch is compatible with two inference paradigms: ReAct, for rigorously evaluating the model's core intrinsic abilities, and an IterResearch-based 'Heavy' mode, which uses a test-time scaling strategy to unlock the model's maximum performance ceiling.

## Download

You can download the model then run the inference scipts in https://github.com/Alibaba-NLP/DeepResearch.


```bibtex
@misc{tongyidr,
  author={Tongyi DeepResearch Team},
  title={Tongyi DeepResearch: A New Era of Open-Source AI Researchers},
  year={2025},
  howpublished={\url{https://github.com/Alibaba-NLP/DeepResearch}}
}
```