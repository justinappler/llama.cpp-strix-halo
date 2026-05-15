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
  - nemotron-terminal
  - terminal
  - code-agent
  - SFT
  - pytorch
---

# Nemotron-Terminal Model Family

**Nemotron-Terminal** is a family of models specialized for autonomous terminal interaction, fine-tuned from the Qwen3 (8B, 14B, and 32B). Developed by NVIDIA, these models utilize [Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus), a large-scale open-source dataset for terminal tasks, to achieve performance that rivals frontier models many times their size.


## Model Variants
We release the following variants of the Nemotron-Terminal family:

- **Nemotron-Terminal-8B**
- Nemotron-Terminal-14B
- Nemotron-Terminal-32B


## Performance on Terminal-Bench 2.0
The Nemotron-Terminal family demonstrates profound leaps in capability compared to the Qwen3 baselines across multiple specialized categories.

| Model | Size | Base Accuracy | **Nemotron-Terminal Accuracy** |
| :--- | :---: | :---: | :---: |
| **Nemotron-Terminal-8B** | 8B | 2.47% | **13.0%** |
| Nemotron-Terminal-14B | 14B | 4.04% | **20.2%** |
| Nemotron-Terminal-32B | 32B | 3.37% | **27.4%** |

## Usage
The models are trained using the **Terminus 2** scaffolding and output a structured JSON format.
For evaluation on Terminal Bench 2.0, we encourage using Terminus 2 scaffolding to maintain consistency with training.

### Expected Output Format
```json
{
  "analysis": "Analysis of the current terminal state...",
  "plan": "Step-by-step plan for the next command...",
  "commands": [
    {
      "keystrokes": "ls -la\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}
```

## 📜 Citation
If you use this dataset in your research, please cite the following work:
```bibtex
@misc{pi2026dataengineeringscalingllm,
      title={On Data Engineering for Scaling LLM Terminal Capabilities}, 
      author={Renjie Pi and Grace Lam and Mohammad Shoeybi and Pooya Jannaty and Bryan Catanzaro and Wei Ping},
      year={2026},
      eprint={2602.21193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.21193}, 
}

