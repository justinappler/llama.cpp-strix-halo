---
license: apache-2.0
datasets:
- livecodebench/code_generation_lite
- agentica-org/DeepCoder-Preview-Dataset
- NousResearch/lcb_test
- NousResearch/RLVR_Coding_Problems
base_model:
- Qwen/Qwen3-14B
pipeline_tag: text-generation
---

# NousCoder-14B

[![](https://img.shields.io/badge/X-NousResearch-000000?logo=x&logoColor=white)](https://x.com/NousResearch)
[![apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange?logoColor=white&logoUrl=https://pbs.twimg.com/profile_images/1816254738234761216/TX7TW-Mp_400x400.jpg)](https://www.apache.org/licenses/LICENSE-2.0)

We introduce *NousCoder-14B*, a competitive programming model post-trained on [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) via reinforcement learning.
On LiveCodeBench v6 (08/01/2024 - 05/01/2025), we achieve a Pass@1 accuracy of 67.87\%, up 7.08\% from the baseline Pass@1 accuracy of 60.79\%
of Qwen3-14B. We trained on 24k verifiable coding problems using 48 B200s over the course of four days.

![test](./lcb_score_vs_step.png)
![test](./performance_params_ratio.png)

# Acknowledgements
I would like to thank my mentor, Roger Jin, Dakota Mahan, Teknium, and others at the Nous Research team for their invaluable support throughout this project. I would also like to thank Together AI and Agentica for their immensely helpful blog posts on DeepCoder-14B. Finally, thank you to Modal and Lambda for their generous support by providing me with credits.