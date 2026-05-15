---
language:
- en
license: llama3
tags:
- Llama-3
- RL
- Atropos
- Tool Calling
- Nous Research
- instruct
- finetune
- reasoning
- function calling
- transformers
- reinforcement-learning
- json mode
- chatml
base_model: meta-llama/Meta-Llama-3.1-8B
library_name: transformers

---

# DeepHermes Tool Calling Specialist - Atropos RL

## Model Overview

The **DeepHermes Tool Calling Specialist - Atropos RL** model is an experimental artifact fine-tuned by Nous Research using our innovative open-source reinforcement learning framework—Atropos. This variant specifically improves the tool calling performance of the **DeepHermes 3 Llama-3.1 8B** model during its reasoning mode.

**Note**: This model is intended as an experimental artifact and is not designed for broad, general-purpose use.

## Atropos Open Source Framework

Atropos is Nous Research’s open-source Reinforcement Learning environment stack, designed to enhance various aspects of LLM functionalities through structured RL methodologies. We encourage contributions and exploration:

🔗 [Atropos GitHub Repository](https://github.com/NousResearch/Atropos)

## Benchmark Results

Evaluations on the Berkeley Function Calling benchmark demonstrate significant improvements in tool calling accuracy during reasoning mode, compared to its base model:

| Benchmark | Base Accuracy | Atropos RL Accuracy | Improvement |
| --------- | ------------- | ------------------- | ----------- |
| Parallel  | 0.10          | 0.46                | **4.6x**    |
| Simple    | 0.21          | 0.5175              | **2.5x**    |

These enhancements are due to RL fine-tuning specifically targeted at improving reasoning-based tool calling capabilities.

Eval set accuracy results: 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/BWIFtF7oBWO2b_Q1oh4YM.png)

## Key Features

- **Improved Tool Calling in Reasoning Mode**: Reinforcement learning significantly boosts tool usage during complex reasoning tasks.
- **Open-Source RL Framework**: Utilizes the fully open-source Atropos RL Environments.
- **Active Open Source Community**: Contributions welcomed on the Atropos GitHub.
- **Upcoming SOTA RL Trainer**: A state-of-the-art open-source reinforcement learning trainer by Nous Research is coming soon.

## Usage

This model supports multiple inference modes including:

- **Reasoning (Deep Thinking Mode)**
- **Standard Chat/Instruction Mode**
- **Structured JSON Outputs**
- **Function Calling**

Detailed documentation and example inference code are available:

*Note: You must first place DeepHermes' reasoning system prompt, and then append  your function calling system prompt after for it to do reasoning and tool calling simultaneously.*

🔗 [Hermes Function Calling GitHub](https://github.com/NousResearch/Hermes-Function-Calling)

## How to Cite

```bibtex
@misc{
      title={DeepHermes Tool Calling Specialist - Atropos RL},
      author={Teknium and Dakota Mahan and Roger Jin and Chen Guang and Jai Suphavadeeprasit and Jeffrey Quesnelle},
      year={2025},
      url={https://huggingface.co/NousResearch/DeepHermes-Tool-Calling-Specialist-Atropos-RL}
}
```

## Community and Support

For questions, issues, or findings, please open issues or discussions in the respective GitHub repositories:

- [Atropos Framework Issues](https://github.com/NousResearch/Atropos/issues)
- [DeepHermes Models Issues](https://github.com/NousResearch/Hermes-Function-Calling/issues)

Nous Research encourages active community engagement and open-source contributions to continuously improve model performance and capabilities.

