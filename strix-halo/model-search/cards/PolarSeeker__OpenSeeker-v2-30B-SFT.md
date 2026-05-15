---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/PolarSeeker/OpenSeeker-v2-30B-SFT/blob/main/LICENSE
pipeline_tag: text-generation
tags:
  - search-agent
  - deep-research
  - react
  - tool-use
  - qwen3
---

# OpenSeeker-v2-30B-SFT

**OpenSeeker-v2-30B-SFT** is a 30B-scale search agent trained with supervised fine-tuning (SFT) on informative and high-difficulty search trajectories. It is released as part of **OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories**.

- Hugging Face: [PolarSeeker/OpenSeeker-v2-30B-SFT](https://huggingface.co/PolarSeeker/OpenSeeker-v2-30B-SFT)
- GitHub: [PolarSeeker/OpenSeeker](https://github.com/PolarSeeker/OpenSeeker)

![OpenSeekerv2](https://cdn-uploads.huggingface.co/production/uploads/67934b85c67af4a116b5594b/Vs3GX2OQ4kH2TWn7mF3Cp.png)

## Highlights

Deep search capabilities have become an indispensable competency for frontier Large Language Model (LLM) agents, yet their development remains dominated by industrial giants. The typical industry recipe involves a highly resource-intensive pipeline spanning pre-training, continual pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL).

In this report, we show that when fueled with informative and high-difficulty trajectories, **a simple SFT approach can be surprisingly powerful for training frontier search agents**. By introducing three simple data synthesis modifications, we establish a stronger baseline:

- Scaling knowledge graph size for richer exploration.
- Expanding the tool set size for broader functionality.
- Applying strict low-step filtering to keep high-difficulty trajectories.

Trained on merely **10.6k data points**, OpenSeeker-v2 achieves state-of-the-art performance across four benchmarks among 30B-sized agents with the ReAct paradigm:

- **46.0%** on BrowseComp.
- **58.1%** on BrowseComp-ZH.
- **34.6%** on Humanity's Last Exam.
- **78.0%** on xbench.

OpenSeeker-v2 surpasses Tongyi DeepResearch, which is trained with a heavier CPT+SFT+RL pipeline and achieves **43.4%**, **46.7%**, **32.9%**, and **75.0%** on the same benchmarks, respectively.

Notably, OpenSeeker-v2 represents the first state-of-the-art search agent within its model scale and paradigm to be developed by a purely academic team using only SFT. We are excited to open-source the OpenSeeker-v2 model weights and share our simple yet effective findings to make frontier search agent research more accessible to the community.

