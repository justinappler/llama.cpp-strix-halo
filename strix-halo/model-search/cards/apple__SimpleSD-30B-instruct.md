---
license: apple-amlr
base_model:
- Qwen/Qwen3-30B-A3B-Instruct-2507
tags:
- self-distillation
- code-generation
library_name: transformers
---

# SimpleSD-30B-instruct

This model is an example of the **Simple Self-Distillation (SimpleSD)** method that improves code generation by fine-tuning a language model on its own sampled outputs—without rewards, verifiers, teacher models, or reinforcement learning. Please see the paper below for more information. This uses Qwen for initialization.

- **Self-distillation sampling:** temperature=1.6, top_p=0.8, top_k=20
- **Evaluation sampling:** temperature=0.9, top_p=0.8, top_k=20

paper: https://arxiv.org/abs/2604.01193

code: https://github.com/apple/ml-ssd

## Notes
- These are research checkpoints for reproducibility.
- They are not optimized Qwen releases.
- They don't represent a broader open-source model strategy.

## Usage

 ```python
from transformers import AutoModelForCausalLM, AutoTokenizer

 model = AutoModelForCausalLM.from_pretrained("apple/SimpleSD-30B-instruct")
tokenizer = AutoTokenizer.from_pretrained("apple/SimpleSD-30B-instruct")
```

## Method

SimpleSD samples solutions from the base model using non-unit temperature and top-k/top-p truncation, then fine-tunes on those samples via standard supervised learning. Despite its simplicity, SimpleSD yields large gains on competitive programming benchmarks, with improvements concentrating on harder problems. The mechanism traces to resolving a *precision–exploration conflict*: SimpleSD reshapes token distributions in a context-dependent way so that a single global decoding configuration becomes far more effective at evaluation time.

## Results

LiveCodeBench (%)

| Model | LCBv6 pass@1 | LCBv6 pass@5 | LCBv5 pass@1 | LCBv5 pass@5 |
|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 (base) | 42.4 | 53.5 | 45.8 | 58.7 |
| **+ SimpleSD (this model)** | **55.3** (+12.9) | **71.6** (+18.1) | **54.3** (+8.5) | **70.7** (+12.0) |

## Paper
 [**Embarrassingly Simple Self-Distillation Improves Code Generation**](https://arxiv.org/abs/2604.01193)
 
 ```bibtex
@misc{zhang2026embarrassinglysimpleselfdistillationimproves,
      title={Embarrassingly Simple Self-Distillation Improves Code Generation},
      author={Ruixiang Zhang and Richard He Bai and Huangjie Zheng and Navdeep Jaitly and Ronan Collobert and Yizhe Zhang},
      year={2026},
      eprint={2604.01193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.01193},
}
```


## License

This model is released under the [Apple Machine Learning Research Model License](https://huggingface.co/apple/SimpleSD-30B-instruct/blob/main/LICENSE).