---
license: apache-2.0
language:
- zh
- en
pipeline_tag: text-generation
library_name: transformers
---
<div align="center">
<img src="https://github.com/OpenBMB/MiniCPM/blob/main/assets/minicpm_logo.png?raw=true" width="500em" ></img> 
</div>

<p align="center">
<a href="https://github.com/OpenBMB/MiniCPM/" target="_blank">GitHub Repo</a> |
<a href="https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf" target="_blank">Technical Report</a> |
<a href="https://mp.weixin.qq.com/s/KIhH2nCURBXuFXAtYRpuXg?poc_token=HBIsUWijxino8oJ5s6HcjcfXFRi0Xj2LJlxPYD9c">Join Us</a>
</p>
<p align="center">
👋 Contact us in <a href="https://discord.gg/3cGQn9b3YM" target="_blank">Discord</a> and <a href="https://github.com/OpenBMB/MiniCPM/blob/main/assets/wechat.jpg" target="_blank">WeChat</a>
</p>

> [!NOTE]
> ### 🏆 2026 Sparse Operator Acceleration & Race (SOAR) is Now Live!
>
> **"The MiniCPM-SALA architecture is just the beginning. Realizing its full potential requires deep system-level synergy and cross-layer compilation optimization."**
>
> In collaboration with **SGLang** and **NVIDIA**, OpenBMB invites global geeks to push the boundaries of 9B-scale, 1M-token inference on **NVIDIA 6000D**.
>
> 💰 **Prize Pool: >$100,000 USD** (🥇 Top Prize: **$89,000**) | 🚀 **Challenge:** Single & Multi-batch Optimization
>
> 👉 **[Click Here to Join the Race @ soar.openbmb.cn](https://soar.openbmb.cn/)**

## What's New
- [2026.02.11] **MiniCPM-SALA** is released! This is the first large-scale hybrid model effectively integrating sparse and linear attention for million-token context modeling. You can find technical report [here](https://github.com/OpenBMB/MiniCPM/tree/main/report/MiniCPM_4_Technical_Report.pdf).🔥🔥🔥

### Highlights

MiniCPM-SALA (Sparse Attention and Linear Attention) is the first large-scale hybrid model effectively integrating sparse and linear attention for million-token context modeling

✅ Innovative Hybrid Architecture: Synergizes 25% Sparse Attention (InfLLM-v2) for high-fidelity long context modeling with 75% Linear Attention (Lightning Attention) for global efficiency.

✅ Shattering Efficiency Walls: Breaks the "Compute Wall" and the "Memory Wall," achieving 3.5× inference speed and significantly lower KV-cache overhead compared to dense baselines. 

✅ Million-Token Context: Empowered by HyPE (Hybrid Positional Embedding), it scales to 1M+ tokens while maintaining strong length generalization. 

✅ HALO Adaptation: Utilizes Hybrid Attention via Layer Optimization (HALO), a novel distillation recipe that effectively transfers dense attention capabilities to the hybrid architecture, avoiding the severe performance degradation typical of pure linear models.

## Introduction

MiniCPM-SALA is an efficient hybrid model in which 25% of the layers adopt [InfLLM-V2](https://arxiv.org/abs/2509.24663) and the remaining 75% utilize Lightning Attention. This architecture enables inference of one million tokens on consumer GPUs such as the NVIDIA RTX 5090.

- **SALA Hybrid Attention Mechanism**
  - Integrates 25% InfLLM-V2 and 75% Lightning Attention, effectively leveraging the granular focus of sparse attention for local details and the high efficiency of linear attention for broad context.

- **Transformer-to-Hybrid Continue Training**
  - Circumvents the inefficiencies of cold-start training by performing an architectural transformation on the pre-trained weights, thereby reducing the total training budget to approximately 25% relative to training a comparable model from scratch.

- **[HyPE](https://arxiv.org/abs/2601.22156) (Hybrid Positional Encoding)**
  - Harmonizes the performance across both short and long contexts, which can maintain general capabilities (e.g., knowledge, mathematics, and coding) comparable to modern full-attention models like Qwen3-8B and achieve substantial advantages across multiple long-context benchmarks.

- **Efficient Inference on Long Sequences**
  - Achieves up to 3.5x the inference speed of Qwen3-8B at a sequence length of 256K tokens on A6000D, supports inference at context lengths of up to 1M tokens on both NVIDIA A6000D and 5090 GPUs, whereas Qwen3-8B fails at this length due to out-of-memory (OOM) errors.

## Inference

To achieve optimal performance, we recommend using `Temperature=0.9`.

### HuggingFace

Our model is readily compatible with 🤗 Hugging Face transformers. You can perform inference with our model as follows:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM-SALA"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
model.eval()

prompts = ["My name is", "The capital of China is"]
with torch.no_grad():
    inputs = tokenizer(prompts, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs)
output_texts = tokenizer.batch_decode(outputs)
print(output_texts)
```

### SGLang

#### Requirements

- CUDA 12.x or higher
- `gcc` / `g++` compiler
- `uv` package manager (script will check)

#### Installation

```bash
# Clone repository
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# One-click installation (creates venv and compiles all dependencies)
bash install_minicpm_sala.sh

# Or specify PyPI mirror
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

The installation script performs the following steps:

1. Creates `sglang_minicpm_sala_env` virtual environment (Python 3.12)
2. Clones dependencies to `3rdparty/` (infllmv2) and initializes submodules (sparse_kernel)
3. Installs MiniCPM-SALA (current repo)
4. Compiles and installs `infllmv2_cuda_impl`
5. Compiles and installs `sparse_kernel`
6. Installs `tilelang` & `flash-linear-attention`

#### Usage

```bash
# Activate environment
source sglang_minicpm_sala_env/bin/activate

# Launch Inference Server (Replace MODEL_PATH with actual path)
MODEL_PATH=/path/to/your/MiniCPM-SALA

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse
```

| Parameter | Description |
|-----------|-------------|
| `--trust-remote-code` | Allow custom code in model |
| `--disable-radix-cache` | Disable RadixAttention prefix cache |
| `--attention-backend minicpm_flashinfer` | Use MiniCPM FlashInfer backend |
| `--chunked-prefill-size 8192` | Chunked prefill size |
| `--max-running-requests 32` | Max concurrent requests |
| `--skip-server-warmup` | Skip server warmup |
| `--port 31111` | Server port |
| `--dense-as-sparse` | Use dense-as-sparse mode |

#### Manual Installation

If the script doesn't work for you, follow these steps:

```bash
# 0. Ensure uv is installed
pip install uv

# 1. Create venv
uv venv --python 3.12 sglang_minicpm_sala_env
source sglang_minicpm_sala_env/bin/activate

# 2. Install SGLang
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./python[all]

# 3. Compile CUDA Extensions
# (Ensure dependencies are cloned to 3rdparty/)
cd 3rdparty/infllmv2_cuda_impl && python setup.py install && cd ../..
cd 3rdparty/sparse_kernel && python setup.py install && cd ../..

# 4. Install extra deps
uv pip install tilelang flash-linear-attention
```

#### Q&A

**Q: CUDA extension compilation failed?**

- Ensure CUDA 12+ is installed (`nvcc --version`).
- Ensure `gcc` / `g++` are available.
- If `CXX` is set to `clang++ -pthread`, manually `export CXX=g++`.


## Evaluation Results

### Efficiency Evaluation

![inference_speed_a6000d](https://github.com/OpenBMB/MiniCPM/blob/main/assets/minicpm_sala/inference_speed_a600d.png?raw=true)

![inference_speed_5090](https://github.com/OpenBMB/MiniCPM/blob/main/assets/minicpm_sala/inference_speed_5090.png?raw=true)

### Long-Context Evaluation

![long_text_evaluation](https://github.com/OpenBMB/MiniCPM/blob/main/assets/minicpm_sala/long_text_evaluation.png?raw=true)

### Ultra-long Context Evaluation

![ultra_long_text_evaluation](https://github.com/OpenBMB/MiniCPM/blob/main/assets/minicpm_sala/ultra_long_text_evaluation.png?raw=true)

### Standard Evaluation

![benchmark](https://github.com/OpenBMB/MiniCPM/blob/main/assets/minicpm_sala/benchmark.png?raw=true)

## Statement
- As a language model, MiniCPM-SALA generates content by learning from a vast amount of text. 
- However, it does not possess the ability to comprehend or express personal opinions or value judgments. 
- Any content generated by MiniCPM-SALA does not represent the viewpoints or positions of the model developers. 
- Therefore, when using content generated by MiniCPM-SALA, users should take full responsibility for evaluating and verifying it on their own.

## LICENSE
- This repository and MiniCPM models are released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 

## Citation
- Please cite our [paper](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf) if you find our work valuable.

```bibtex
@article{minicpm4,
  title={{MiniCPM-SALA}: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling},
  author={MiniCPM Team},
  year={2026}
}
```