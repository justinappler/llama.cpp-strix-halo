---
base_model:
- Qwen/Qwen2.5-Coder-32B-Instruct
library_name: transformers
license: mit
pipeline_tag: text-generation
---

📝 [Paper](https://arxiv.org/abs/2506.07636) | 🌐 [Github](https://github.com/THUDM/SWE-Dev/)

- 🤗 [SWE-Dev-7B (Qwen-2.5-Coder-7B-Instruct)](https://huggingface.co/THUDM/SWE-Dev-7B/)
- 🤗 [SWE-Dev-9B (GLM-4-9B-Chat)](https://huggingface.co/THUDM/SWE-Dev-9B/)
- 🤗 [SWE-Dev-32B (Qwen-2.5-Coder-32B-Instruct)](https://huggingface.co/THUDM/SWE-Dev-32B/)
- 🤗 [SWE-Dev-train (Training Data)](https://huggingface.co/datasets/THUDM/SWE-Dev-train/)

🚀 SWE-Dev, an open-source Agent for Software Engineering tasks! This repository contains the SWE-Dev-32B model as presented in the paper [SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling](https://huggingface.co/papers/2506.07636).

💡 We develop a comprehensive pipeline for creating developer-oriented datasets from GitHub repositories, including issue tracking, code localization, test case generation, and evaluation.

🔧 Based on open-source frameworks (OpenHands) and models, SWE-Dev-7B and 32B achieved solve rates of 23.4% and 36.6% on SWE-bench-Verified, respectively, even approaching the performance of GPT-4o. 

📚 We find that training data scaling and inference scaling can both effectively boost the performance of models on SWE-bench. Moreover, higher data quality further improves this trend when combined with reinforcement fine-tuning (RFT). For inference scaling specifically, the solve rate on SWE-Dev increased from 34.0% at 30 rounds to 36.6% at 75 rounds.