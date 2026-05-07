---
language: en
license: mit
tags:
  - aro
  - code-generation
  - dsl
  - mlx
  - 4-bit
  - teacher-model
  - fine-tuned
base_model: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
pipeline_tag: text-generation
library_name: mlx
---

# ARO Teacher (30B)

The full **30B MoE teacher model**, fine-tuned on ARO training data.
This model is used for:

- **Distillation** — generating high-quality training data for the smaller student model
- **Iterative retraining** — serving as the starting point for the next training cycle
- **High-quality inference** — when maximum accuracy is needed (at the cost of speed/memory)

> **For deployment and everyday use, prefer the distilled 8B student model:**
> [ARO-Lang/aro-coder-4bit](https://huggingface.co/ARO-Lang/aro-coder-4bit)

| | |
|---|---|
| **Architecture** | Qwen3 30B MoE (3.3B active parameters) |
| **Base model** | [mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit) |
| **Quantization** | 4-bit (MLX) |
| **Size** | ~16 GB |
| **Training source** | dpo |

## Usage

```python
from mlx_lm import load, generate
model, tokenizer = load("ARO-Lang/aro-teacher-30b-4bit")
```

Or as a base for continued fine-tuning:

```bash
python -m mlx_lm lora --model ARO-Lang/aro-teacher-30b-4bit --data ./train_data --train
```

## Links

- **Distilled student**: [ARO-Lang/aro-coder-4bit](https://huggingface.co/ARO-Lang/aro-coder-4bit)
- **Website**: [arolang.github.io/aro](https://arolang.github.io/aro/)
- **GitHub**: [github.com/arolang/aro](https://github.com/arolang/aro)
- **Language Guide**: [Wiki](https://github.com/arolang/aro/wiki)

## License

MIT License
