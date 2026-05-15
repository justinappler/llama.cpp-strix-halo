---
license: apache-2.0
language:
- en
tags:
- code
- swe-bench
- agentic
- sft
library_name: transformers
pipeline_tag: text-generation
datasets:
- ricdomolm/mini-coder-trajs-400k
base_model:
- talkie-lm/talkie-1930-13b-base
---

# talkie-1930-coder

13B model fine-tuned on agentic software-engineering trajectories from
[SWE-smith](https://github.com/SWE-bench/SWE-smith), starting from the
`talkie-1930` base. Tuned for the
[mini-swe-agent](https://github.com/SWE-bench/mini-swe-agent) interaction
format.

## SWE-bench-Verified-Working-Harbor pass@1

| metric | value |
|---|---|
| **pass@1** (n=5 independent eval runs) | **4.48% ± 0.69 pp** |
| per-run resolved (out of 446) | 23, 18, 20, 23, 16 |

Eval pipeline: vLLM (`--model-impl transformers --max-model-len 32768
--dtype bfloat16`) → mini-swe-agent (`mini-extra swebench`, temperature 0.7,
`max_tokens=4096`), graded with the swebench harness against
`ricdomolm/SWE-bench_Verified-Working-Harbor`.

## Training recipe

| | |
|---|---|
| Base model | `talkie-1930-13b-base` |
| Dataset | `talkie-1930-swe-100k-64k` (100k SWE-smith trajectories, packed at 64k) |
| Trainer | TRL `SFTTrainer` via `accelerate` (8× A100) |
| Optimizer | `adamw_torch_fused`, β=(0.9, 0.95), ε=1e-8 |
| LR | 2e-5, `cosine_with_min_lr`, warmup 3% |
| Precision | bf16 |
| Weight decay | 0.1 |
| Max grad norm | 30 |
| Max length | 65,536 |
| Packing | `bfd` + padding-free |
| Loss | `completion_only_loss=1` (loss only on assistant tokens) |
| Steps | 2,016 (this is ckpt-2000) |

## Usage

This model uses custom modeling code (`modeling_talkie.py`,
`configuration_talkie.py`). Load with `trust_remote_code=True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "ricdomolm/talkie-1930-coder",
    trust_remote_code=True,
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained("ricdomolm/talkie-1930-coder")
```

For agentic eval, serve with vLLM and drive with mini-swe-agent:

```bash
vllm serve ricdomolm/talkie-1930-coder \
    --model-impl transformers --max-model-len 32768 --dtype bfloat16
```

## Companion model

[`ricdomolm/talkie-web-coder`](https://huggingface.co/ricdomolm/talkie-web-coder)
— same recipe, same SFT data, but starting from a base model pre-trained
on web-style data. Reaches 5.75% ± 1.04 pp on the same eval (n=3).