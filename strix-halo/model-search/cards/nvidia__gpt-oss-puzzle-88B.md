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
  - gpt-oss
  - puzzle
  - mixture-of-experts
  - reasoning
  - pytorch
  - transformers
  - vllm
---

# gpt-oss-puzzle-88B

# Model Overview

### Description
gpt-oss-puzzle-88B is a deployment-optimized large language model developed by NVIDIA, derived from [OpenAI's gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b).  
The model is produced using Puzzle, a post-training neural architecture search (NAS) framework, with the goal of significantly improving inference efficiency for reasoning-heavy workloads while maintaining or improving accuracy across reasoning budgets.  

The model is specifically optimized for long-context and short-context serving on NVIDIA H100-class hardware, where reasoning models are often bottlenecked by KV-cache bandwidth and memory capacity rather than raw compute.

Compared to its parent, gpt-oss-puzzle-88B:
- Reduces total parameters to ~88B (≈73% of the parent),
- Achieves 1.63× throughput improvement in long-context (64K/64K) scenarios on an 8×H100 node,
- Achieves 1.22× throughput improvement in short-context (4K/4K) scenarios,
- Delivers up to 2.82× throughput improvement on a single H100 GPU,
- Matches or slightly exceeds parent accuracy across reasoning efforts.

**Parameter count note.** Hugging Face Hub may automatically show this model as ~91B parameters. We refer to it as 88B because the automatic count includes additional MXFP4 quantization scale tensors for the MoE experts, which are typically not counted as model parameters.


This model is ready for commercial use.

![Accuracy vs Relative Request Rate](fig1.png)
![Accuracy Retention and Throughput Speedup](fig2.png)

### License/Terms of Use

Governing Terms: Use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)).

### Deployment Geography
Global

### Use Case
gpt-oss-puzzle-88B is a general purpose reasoning and chat model. This model is intended for production deployment, cost-efficient reasoning, and long-context inference workloads.

### Release Date
March 26, 2026 via [Hugging Face](https://huggingface.co/nvidia/gpt-oss-puzzle-88B)

## References(s)
* [\[2411.19146\] Puzzle: Distillation-Based NAS for Inference-Optimized LLMs](https://arxiv.org/abs/2411.19146)  
* [\[2508.10925\] gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925)  
* [\[2602.11937\] Extending Puzzle for Mixture-of-Experts Reasoning Models with Application to GPT-OSS Acceleration](https://arxiv.org/abs/2602.11937)

## Model Architecture
- **Architecture Type:** Mixture-of-Experts Decoder-only Transformer

- **Network Architecture:** Modified [gpt-oss](https://huggingface.co/openai/gpt-oss-120b) architecture with varying number of experts per layer, and a modified global/window attention pattern across layers.

- **Number of model parameters:** 88B

### Key Architectural Optimizations
This model was created using Puzzle, a post-training NAS framework that constructs a heterogeneous architecture under explicit deployment constraints:

- Heterogeneous MoE Expert Pruning  
  Each MoE layer retains a different number of experts, determined via activation-based importance scoring. Early layers retain more experts; later layers are more aggressively pruned.

- Selective Window Attention  
  A subset of global attention layers is replaced with window attention (8K window), reducing KV-cache footprint by ~40% in long-context scenarios while preserving long-range reasoning.

- RoPE Scaling Adjustment  
  The YaRN RoPE scaling factor was increased to improve stability at 128K context length.

## Training and Optimization Procedure

### Knowledge Distillation

After Puzzle architecture selection, the model underwent knowledge distillation:

- Total Tokens: 84B
- Sequence Length: 128K
- MoE Experts & Router: Frozen
- Framework: Megatron-LM

This phase restores inter-block compatibility and recovers quality lost during blockwise substitution.

### Reinforcement Learning:

A post-distillation reinforcement learning (RL) phase was applied to improve reasoning accuracy while controlling generation length:

- Multi-environment RL (math, coding, reasoning)
- MoE experts and router frozen
- Two complementary policies trained:
  - High-effort-focused (max accuracy)
  - Mixed-effort (length-regularized)
- Final model obtained via checkpoint weight averaging

This preserves high reasoning accuracy while maintaining a stable effort length ratio, ensuring predictable cost-quality trade-offs.

### Quantization:

- MoE Weights: MXFP4 (inherited from gpt-oss-120B)
- KV Cache: FP8 with calibrated KV scales
- Effect:  
  - ~2× KV-cache token capacity  
  - Faster attention kernels  
  - Preserved accuracy vs unscaled FP8 KV-cache

## Reasoning Effort Control:
The model supports three reasoning effort modes:

- Low: Fast, concise responses
- Medium: Balanced accuracy and verbosity
- High: Deep, multi-step reasoning

Effort reliably controls generation length and accuracy, enabling cost-aware deployment.

## Input

- **Input Type(s):** Text

- **Input Format(s):** String  
    
- **Input Parameters:** One-Dimensional (1D): Sequences  
    
- **Other Properties Related to Input:** Context length is 128k tokens.

## Output

- **Output Type(s):** Text  
    
- **Output Format:** String  
    
- **Output Parameters:** One-Dimensional (1D): Sequences  
    
- **Other Properties Related to Output:** Context length is 128k tokens.

Our AI models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. 

## Software Integration
**Runtime Engine(s):** 
* vLLM (See instructions [below](#vllm))

**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA B200
* NVIDIA H100-80GB

**Preferred/Supported Operating System(s):**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version
- v1.0

## Training and Evaluation Datasets

### Dataset Overview
**Total Number of Datasets:** 7  
**Time period for data collection:** 2013 to May 1, 2025  

For the KD stage data, the prompts from [nvidia/Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) were used to generate responses from the parent model (gpt-oss-120b) to create full KD training examples. For each prompt, we generated responses under high and medium reasoning-effort settings.

For the RL stage data, we used a subset of the [NeMo Gym collection](https://huggingface.co/collections/nvidia/nemo-gym) which includes RL verifiable data.

# Public Datasets
- [nvidia/Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)
- [nvidia/Nemotron-RL-coding-competitive_coding](https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding)
- [nvidia/Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following)
- [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- [Skywork/Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data)
- [nvidia/Nemotron-RL-knowledge-mcqa](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa)
- [nvidia/Nemotron-RL-instruction_following-structured_outputs](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs)

## Training Dataset

**Data Modality**: Text

**Text Training Data Size**: 1 Billion to 10 Trillion Tokens

**Data Collection Method by dataset**: Automated/Synthetic/Human

**Labeling Method by dataset**: Not Applicable

**Properties**:
The training data is text-only and spans a broad range of task categories. The knowledge distillation stage used the Llama-Nemotron-Post-Training-Dataset, a large-scale collection covering mathematics, code, science, instruction following, general chat, and safety. The reinforcement learning stage used datasets spanning several domains: competitive programming problems with unit tests (Nemotron-RL-coding-competitive_coding, Skywork-OR1-RL-Data), diverse verifiable mathematical reasoning problems (DAPO-Math-17k, Skywork-OR1-RL-Data), multi-domain multiple-choice question answering across fields such as physics, biology, chemistry, mathematics, computer science, engineering, humanities, law, and others (Nemotron-RL-knowledge-mcqa), easily verifiable instruction-following tasks with diverse format and linguistic constraints (Nemotron-RL-instruction_following), and structured output generation requiring adherence to JSON schemas (Nemotron-RL-instruction_following-structured_outputs). No personal data was used for training.


## Evaluation Dataset

**Data Collection Method by dataset:** Hybrid: Human, Synthetic

**Labeling Method by dataset:** Hybrid: Automated, Human, Synthetic

| Benchmark | Description |
|-----------|-------------|
| [**MMLU-Pro**](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MMLU-Pro dataset is a more robust and challenging massive multi-task understanding dataset tailored to more rigorously benchmark large language models' capabilities. |
| [**GPQA-Diamond**](https://huggingface.co/datasets/Idavidrein/gpqa) | The GPQA (Graduate-Level Google-Proof Q&A) benchmark is a challenging dataset of 448 multiple-choice questions in biology, physics, and chemistry. |
| [**HLE**](https://huggingface.co/datasets/cais/hle) | Humanity's Last Exam (HLE) is a multi-modal benchmark at the frontier of human knowledge, designed to be the final closed-ended academic benchmark of its kind with broad subject coverage. Humanity's Last Exam consists of 3,000 questions across dozens of subjects, including mathematics, humanities, and the natural sciences. HLE is developed globally by subject-matter experts and consists of multiple-choice and short-answer questions suitable for automated grading. |
| [**AA-LCR**](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) | A challenging benchmark measuring language models' ability to extract, reason about, and synthesize information from long-form documents ranging from 10k to 100k tokens (measured using the cl100k_base tokenizer). |
| [**AIME25**](https://huggingface.co/datasets/math-ai/aime25) | American Invitational Mathematics Examination (AIME) 2025 questions |
| [**IFBench**](https://huggingface.co/datasets/allenai/IFBench_test) | IFBench is a new, challenging benchmark for precise instruction following. |
| [**SciCode**](https://huggingface.co/datasets/SciCode1/SciCode) | SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. |
| [**RULER 128K**](https://huggingface.co/datasets/GAIR/ruler-128k) | RULER generates synthetic examples to evaluate long-context language models with configurable sequence length and task complexity. Used with context length of 128K tokens. |


# Inference
**Acceleration Engine**: vLLM
**Test Hardware:**  
- 1× NVIDIA H100-80GB  
- 8× NVIDIA H100-80GB  
- 8× NVIDIA B200  

## Quick Start

The gpt-oss-puzzle-88B model can be used with standard inference stacks such as Hugging Face Transformers and vLLM.  
It is especially optimized for NVIDIA H100 GPUs and supports long-context inference up to 128K tokens.

### Transformers
We recommend using Transformers ≥ 4.57.3.

```python
from transformers import pipeline

model_id = "nvidia/gpt-oss-puzzle-88B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

generation_config = GenerationConfig.from_pretrained(model_id)
generation_config.max_new_tokens = 256

outputs = pipe(
    messages,
    generation_config=generation_config,
)
print(outputs[0]["generated_text"][-1])
```

### vLLM

#### Serving

Start the server with a single command:

```bash
docker run --gpus all -p 8000:8000 \
  --entrypoint bash \
  vllm/vllm-openai:v0.17.1 \
  -c "
    apt-get update && apt-get install -y git &&
    VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/1ac2ef2e5335ca0af99aee438998c9305461f563/vllm-0.18.1rc1.dev127%2Bg1ac2ef2e5-cp38-abi3-manylinux_2_31_$(uname -m).whl VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation 'git+https://github.com/vllm-project/vllm.git@88f6aaad7cc2fe8904d5fcf31d06eb57f15a6d25' &&
    pip install flashinfer-cubin==0.6.6 flashinfer-jit-cache==0.6.6 --extra-index-url https://flashinfer.ai/whl/cu129 &&
    export PYTORCH_ALLOC_CONF=expandable_segments:True &&
    vllm serve nvidia/gpt-oss-puzzle-88B \
      -tp 1 \
      --trust-remote-code \
      --kv-cache-dtype fp8 \
      --max-num-batched-tokens 8192 \
      --stream-interval 20 \
      --gpu-memory-utilization 0.95 \
      --max-num-seqs 8 \
      --max-cudagraph-capture-size 8 \
      --max-model-len 131072
  "
```

> **Notes:**
> - On Blackwell (B200), add `-e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` to the `docker run` command.
> - Remove `--kv-cache-dtype fp8` for BF16 KV-cache instead of FP8.
> - Increase `-tp` if you need larger batch sizes or longer sequences.
> - Expert parallelism is supported via `--enable-expert-parallel`, but we recommend TP.

#### Inference with Reasoning Effort Control

The model supports three reasoning effort levels (`low`, `medium`, `high`). For example:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# High effort — deep, multi-step reasoning
response = client.chat.completions.create(
    model="nvidia/gpt-oss-puzzle-88B",
    messages=[{"role": "user", "content": "Write a haiku about neural network pruning"}],
    reasoning_effort="high",
)
print(response.choices[0].message.content)

# Low effort — fast, concise responses
response = client.chat.completions.create(
    model="nvidia/gpt-oss-puzzle-88B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="low",
)
print(response.choices[0].message.content)
```

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

For more detailed information on ethical considerations for this model, please see the [Bias, Explainability, Safety & Security, and Privacy Subcards](https://huggingface.co/nvidia/gpt-oss-puzzle-88B).

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).  