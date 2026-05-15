---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
---
# GroveMoE-Base

</div>
<p align="left">
🤗 <a href="https://huggingface.co/collections/inclusionAI/grovemoe-68a2b58acbb55827244ef664">Models</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.07785">Paper</a> &nbsp&nbsp | &nbsp&nbsp 🔗 <a href="https://github.com/inclusionAI/GroveMoE">Github</a>&nbsp&nbsp


## Highlights

We introduce **GroveMoE**, a new sparse architecture using **adjugate experts** for dynamic computation allocation, featuring the following key highlights:

- **Architecture**: Novel **adjugate experts** grouped with ordinary experts; shared computation is executed once, then reused, cutting FLOPs.
- **Sparse Activation**: 33 B params total, only **3.14–3.28 B** active per token.
- **Traning**: Mid-training + SFT, up-cycled from Qwen3-30B-A3B-Base; preserves prior knowledge while adding new capabilities.

## Model Downloads


| **Model** | **#Total Params** | **#Activated Params** | **HF Download** |**MS Download** |
|:---------:|:-----------------:|:---------------------:|:------------:|:------------:|
| GroveMoE-Base | 33B | 3.14~3.28B | [🤗 HuggingFace](https://huggingface.co/inclusionAI/GroveMoE-Base) | [📦 ModelScope](https://modelscope.cn/models/cccnju/GroveMoE-Base) |
| GroveMoE-Inst | 33B | 3.14~3.28B | [🤗 HuggingFace](https://huggingface.co/inclusionAI/GroveMoE-Inst) | [📦 ModelScope](https://modelscope.cn/models/cccnju/GroveMoE-Inst) |

## Citation
```bibtex
@article{GroveMoE,
title = {GroveMoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts},
author = {Wu, Haoyuan and Chen, Haoxing and Chen, Xiaodong and Zhou, Zhanchao and Chen, Tieyuan and Zhuang, Yihong and Lu, Guoshan and Zhao, Junbo and Liu, Lin and Huang, Zenan and Lan, Zhenzhong and Yu, Bei and Li, Jianguo},
journal = {arXiv preprint arXiv:2508.07785},
year = {2025}
}
```
