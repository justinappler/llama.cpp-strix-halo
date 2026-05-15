---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- code
- industrial-code
- verilog
- cuda
- triton
- chip-design
- cad
---

# InCoder-32B: Code Foundation Model for Industrial Scenarios

<div align="center">

[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder)
[![ModelScope](https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-pink)](https://modelscope.cn/models/Multilingual-Multimodal-NLP/IndustrialCoder)
[![GitHub](https://img.shields.io/badge/GitHub-Industrial--Coder-blue)](https://github.com/CSJianYang/Industrial-Coder)
[![arXiv](https://img.shields.io/badge/arXiv-2603.16790-red)](https://huggingface.co/papers/2603.16790)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

</div>

## Model Summary

**InCoder-32B** (Industrial-Coder-32B) is the first 32B-parameter code foundation model purpose-built for industrial code intelligence. While general-purpose code LLMs excel at mainstream software tasks, they often struggle with the unique demands of industrial programming — hardware semantics, specialized language constructs, strict resource constraints, and domain-specific correctness verification. 

Presented in the paper [InCoder-32B: Code Foundation Model for Industrial Scenarios](https://huggingface.co/papers/2603.16790), InCoder-32B unifies code intelligence across five industrial domains:

| Domain | Languages & Frameworks |
|---|---|
| 🔧 **Chip Design** | Verilog, SystemVerilog, RTL |
| ⚡ **GPU Kernel Optimization** | CUDA, Triton |
| 🖥️ **Embedded Systems** | C/C++, ARM Cortex-M4, STM32 |
| 🔨 **Compiler Optimization** | x86-64 ASM, C/C++, LLVM-IR |
| 📐 **3D Modeling / CAD** | CadQuery, OpenCascade, Python |

InCoder-32B achieves highly competitive performance on general tasks while establishing the strongest open-source baselines across all evaluated industrial domains.

---

## Key Results

### General Code Benchmarks

| Benchmark | InCoder-32B |
|---|---|
| SWE-bench Verified | **74.8%** |
| LiveCodeBench (Pass@1) | **49.14%** |
| BFCL v3 | **60.99%** |
| HumanEval+ | **89.6%** |
| MBPP+ | **78.3%** |
| BigCodeBench (Full) | **49.8%** |

### Industrial Code Benchmarks

| Benchmark | Domain | InCoder-32B | Best Competing Open-Weight |
|---|---|---|---|
| VeriScope Score | Chip Design | **80.7** | 83.2 (GLM-5) |
| CAD-Coder Compile | 3D Modeling | **82.0%** | 48.0% (Kimi-K2-Thinking) |
| KernelBench L1 | GPU Optimization | **22.2%** | 16.2% (GLM-5) |
| KernelBench L2 | GPU Optimization | **36.0%** | 28.0% (KernelBench L2) |

> InCoder-32B leads all open-weight baselines on CAD-Coder and KernelBench (all three levels), and even surpasses proprietary models like Claude-Sonnet-4.6 on CAD-Coder IoU and KernelBench L1/L2/L3.

---

## Model Architecture

InCoder-32B adopts a standard decoder-only Transformer architecture with the following configuration:

| Hyperparameter | Value |
|---|---|
| Parameters | ~32B |
| Layers | 64 |
| Hidden Size | 5,120 |
| Max Context Length | 131,072 (128K) |
| Positional Encoding | RoPE (θ = 500,000) |
| Precision | BFloat16 |

---

## Training Pipeline: Code-Flow

InCoder-32B is trained through a three-stage **Code-Flow** pipeline:

### Stage 1 — Pre-training & Annealing
- **Industrial Recall**: Data pipeline using rule-based filtering, FastText classifiers, and semantic retrieval for Verilog, CUDA, firmware C, and CadQuery.
- **Refinement**: OCR extraction from technical manuals, multi-level deduplication, and repository-level fork consolidation.
- **Training**: 15T total tokens using Autoregressive LM + Fill-in-the-Middle (FIM) objectives.

### Stage 2 — Mid-Training (Context Extension)
Context window extended progressively from 8K to 128K tokens:
- **8K → 32K**: Targets file-level tasks like completing RTL modules or kernel functions.
- **32K → 128K**: Unlocks long-context capabilities for extended debugging and cross-module projects.

### Stage 3 — Post-Training
2.5M supervised fine-tuning (SFT) samples constructed from real industrial tasks with execution-grounded verification using toolchains like Icarus Verilog, `nvcc`, and Renode (STM32 simulator).

---

## Usage

### Installation

```bash
pip install transformers accelerate
```

### Basic Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Multilingual-Multimodal-NLP/IndustrialCoder"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = """Write a synthesizable Verilog module for a UART transmitter (8N1 protocol).
The module should accept 8-bit parallel data and serialize it onto a TX line."""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.2,
    do_sample=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Deployment with vLLM
For production deployment, you can use vLLM to create an OpenAI-compatible API endpoint.

```
vllm serve Multilingual-Multimodal-NLP/IndustrialCoder --tensor-parallel-size 8
```

### Fill-in-the-Middle (FIM)

InCoder-32B supports FIM completion for code infilling tasks:

```python
prefix = """// CUDA kernel for RMS Normalization
__global__ void rms_norm_kernel(float* output, const float* input, 
                                 const float* weight, int N, float eps) {
    int idx = blockIdx.x;
"""
suffix = """
    output[idx * N + tid] = normalized * weight[tid];
}"""

fim_prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
inputs = tokenizer(fim_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Limitations & Disclaimers

Based on failure analysis, the model may struggle with:
- **API Knowledge**: Linker errors from undefined HAL/CMSIS functions in embedded C.
- **Functional Semantics**: Producing compilable but functionally incorrect RTL under complex logic scenarios.
- **Optimization**: Correct but sub-optimal GPU kernel performance.

Always review and test generated code in a sandboxed environment. Industrial code (RTL, embedded firmware) requires expert review before deployment.

---

## Citation

```bibtex
@article{yang2026incoder,
  title={InCoder-32B: Code Foundation Model for Industrial Scenarios},
  author={Yang, Jian and Zhang, Wei and Wu, Jiajun and Cheng, Junhang and Guo, Shawn 
          and Wang, Haowen and Gu, Weicheng and Du, Yaxin and Li, Joseph and Xu, Fanglin 
          and others},
  journal={arXiv preprint arXiv:2603.16790},
  year={2026}
}
```