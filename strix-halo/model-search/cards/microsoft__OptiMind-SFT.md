---
license: mit
language:
- en
base_model:
- unsloth/gpt-oss-20b-BF16
tags:
- optimization
- operations-research
- milp
- gurobi
- sft
- transformers
---

# Model Overview

OptiMind-SFT is a specialized 20B parameter model designed to bridge the gap between natural language and executable optimization solvers. It automates the translation of complex decision-making problems—such as supply chain planning, scheduling, and resource allocation—into correct MILP formulations.


# Model Summary

**Developer:** Microsoft Research, Machine Learning and Optimization (MLO) Group  \
**Model Architecture:** Mixture-of-Experts (MoE) variant of the transformer architecture (gpt-oss family). \
**Parameters:** 20 Billion (3.6B activated) \
**Inputs:** Natural language optimization problem description. \
**Context Length:** 128,000 tokens \
**Outputs:** Mathematical formulation and executable Python code using GurobiPy. \
**GPUs:** 8x NVIDIA B200 (Training), 8x NVIDIA H100 (Inference/Evaluation) \
**Training Time:** ~8 hours \
**Public Data Summary:** Cleaned subsets of [OR-Instruct](https://huggingface.co/datasets/CardinalOperations/OR-Instruct-Data-3K) and [OptMATH-Train](https://huggingface.co/datasets/Aurora-Gem/OptMATH-Train) \
**Dates:** Trained in October 2025 \
**Status:** Static model trained on cleaned public datasets \
**Release Date:** November 2025 \
**License:** MIT \
**Model Dependencies:** [unsloth/gpt-oss-20b-BF16](https://huggingface.co/unsloth/gpt-oss-20b-BF16) \
**Additional Assets:** [GitHub Repository](https://github.com/microsoft/OptiGuide) 


# Usage

## Sample Useage

OptiMind-SFT is best served with **SGLang**. we use SGLang’s OpenAI-compatible API together with the official openai Python client:

```
pip install "sglang[all]" openai gurobipy

# Make sure you have a valid Gurobi license and PYTHON>=3.12
python -m sglang.launch_server \
  --model-path microsoft/OptiMind-SFT \
  --host 0.0.0.0 \
  --port 30000 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

Below is the sample code to query the model:
```
from openai import OpenAI

# SGLang exposes an OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"  # Not used by local SGLang, but required by the client
)

system_prompt = """You are an expert in optimization and mixed integer programming. You are given an
optimization problem and you need to solve it using gurobipy.
Reason step by step before generating the gurobipy code.
When you respond, first think carefully.
After thinking, output the math modeling of the problem.
Finally output a ```python ...``` code block that solves the problem.
The code must include:
import gurobipy as gp
from gurobipy import GRB
"""

user_problem = "A factory produces products A and B with capacity and demand constraints ..."

response = client.chat.completions.create(
    model="microsoft/OptiMind-SFT",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_problem},
    ],
    temperature=0.9,   # recommended default
    top_p=1.0,         # recommended default
    max_tokens=4096,
)

print(response.choices[0].message.content)
```

This will return a response that first describes the mathematical model and then includes a python code block implementing it in gurobipy.


## Primary Use Cases

- Translating natural-language Operations Research (OR) problems into mixed-integer linear programs (MILPs) and corresponding `gurobipy` code for research and prototyping.
- Studying and benchmarking NL to MILP modeling pipelines on public OR datasets such as IndustryOR, Mamo-Complex, and OptMATH.
- Educational use for teaching how to derive optimization models (variables, constraints, objectives) from informal problem descriptions.
- Performing ablations and research on solver-in-the-loop prompting and multi-turn correction in domain-specific modeling tasks.

## Out-of-Scope Use Cases

- General-purpose chat, open-domain reasoning, or tasks unrelated to optimization modeling.
- Safety-critical or regulated applications (e.g., healthcare, finance, legal decisions, credit scoring) without expert human review of both the model output and the resulting optimization.
- Fully automated deployment where optimization results are used directly for real-world decisions without human oversight.
- Automatic execution of generated code in production systems without sandboxing, logging, and appropriate security controls.


## Technical Requirements & Integration

We recommend **≥32GB GPU VRAM** (e.g., A100/H100/B200) for comfortable inference, especially for long prompts and multi-turn interactions. 
Please checkout our [GitHub page](https://github.com/microsoft/OptiGuide) for instructions on the inference pipeline.

# Data Overview
## Training and Validation Data
We fine-tune OptiMind-SFT on cleaned versions of the OR-Instruct and OptMATH training sets, and validate on a held-out validation split drawn from the same cleaned corpora.

## Testing Data
For testing, we use manually cleaned and expert-validated versions of the IndustryOR, Mamo-Complex, and OptMATH benchmarks. Please visit our [GitHub page](https://github.com/microsoft/OptiGuide) to download the cleaned benchmarks.

# Known Technical Limitations

- The model can still produce incorrect formulations or invalid code, or declare feasibility/optimality incorrectly.   
- It is specialized to OR benchmarks; behavior on general text or other problem domains is not guaranteed.
- No dedicated red-teaming against unsafe content categories (e.g., hate, violence, self-harm) or jailbreak attacks has been performed; the paper focuses on technical robustness metrics. 

Users **must** keep a human in the loop for all consequential decisions and carefully review any generated code before execution.

# Other Sources & Maintenance
- Evaluation code and cleaned benchmarks: [GitHub page](https://github.com/microsoft/OptiGuide)
- Paper: [Arxiv link](https://arxiv.org/abs/2509.22979)
For questions, issues, or feature requests, please use the GitHub issue tracker or the Hugging Face “Community” tab.

# Citation
If you use OptiMind-SFT or the associated datasets/benchmarks in your work, please cite:

```
@article{zhang2025optimind,
  title={OptiMind: Teaching LLMs to Think Like Optimization Experts},
  author={Zhang, Xinzhi and Chen, Zeyi and Zope, Humishka and Barbalho, Hugo and Mellou, Konstantina and Molinaro, Marco and Kulkarni, Janardhan and Menache, Ishai and Li, Sirui},
  journal={arXiv preprint arXiv:2509.22979},
  year={2025}
}
```