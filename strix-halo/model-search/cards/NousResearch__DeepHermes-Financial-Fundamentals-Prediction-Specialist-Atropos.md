---
language:
- en 
license: llama3 
tags:
- Llama-3
- Financial Analysis
- RL
- Atropos
- Fundamentals Prediction
- Nous Research
- reasoning
- transformers
- reinforcement learning
- json mode 
base_model: meta-llama/Meta-Llama-3.1-8B 
library_name: transformers
---

# DeepHermes Financial Fundamentals Prediction Specialist - Atropos RL

## Model Overview

The **DeepHermes Financial Fundamentals Prediction Specialist - Atropos RL** is an experimental model artifact, fine-tuned by Nous Research using our new open source LLM RL Gym, Atropos. This model specifically aims to enhance the accuracy of financial fundamentals predictions through reasoning-intensive reinforcement learning techniques.

**Note**: This model is experimental and is not intended as a general-purpose state-of-the-art solution.

## Atropos Open Source Framework

Atropos is Nous Research's open-source reinforcement learning environment stack, engineered to optimize diverse LLM capabilities through structured RL methodologies. Contributions and active engagement from the community are highly encouraged:

🔗 [Atropos GitHub Repository](https://github.com/NousResearch/Atropos)

## Evaluation Results

Training and evaluation focused on improving financial fundamentals direction prediction accuracy:

| Evaluation Metric                     |        Final Accuracy   |
| ------------------------------------- | ----------------------- |
| Direction Prediction Accuracy (train) | \~20% -> \~50% Accuracy |

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/c-TXnheLY87Vs38KNKp7k.png)

## Key Features

- **Financial Fundamentals Prediction**: Enhanced capability to predict market direction using RL during intensive reasoning tasks.
- **Built with Atropos RL Environments**: Built with the open-source Atropos reinforcement learning stack.
- **Reasoning-Enhanced Predictions**: Specifically optimized for scenarios involving deep analytical reasoning in financial contexts.

## Usage

The model is optimized for reasoning-intensive financial analysis tasks and supports:

- **Deep Reasoning Mode given a company's context (previous quarter's financials data) to predict the next future quarter's fundamental metric direction)**

## Community and Support

We welcome contributions, suggestions, and issues through our community channels:

- [Atropos Issues](https://github.com/NousResearch/Atropos/issues)

## How to Cite

```bibtex
@misc{
      title={DeepHermes Financial Fundamentals Prediction Specialist - Atropos RL},
      author={Teknium and Dakota Mahan and Roger Jin and Chen Guang and Jai Suphavadeeprasit and Jeffrey Quesnelle},
      year={2025},
      url={https://huggingface.co/NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos-RL}
}
```

