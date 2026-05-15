---
license: mit
language:
- zh
- en
pipeline_tag: text-generation
library_name: transformers
---

# GLM-4-Z1-9B-0414

## Introduction

The GLM family welcomes a new generation of open-source models, the **GLM-4-32B-0414** series, featuring 32 billion parameters. Its performance is comparable to OpenAI's GPT series and DeepSeek's V3/R1 series, and it supports very user-friendly local deployment features. GLM-4-32B-Base-0414 was pre-trained on 15T of high-quality data, including a large amount of reasoning-type synthetic data, laying the foundation for subsequent reinforcement learning extensions. In the post-training stage, in addition to human preference alignment for dialogue scenarios, we also enhanced the model's performance in instruction following, engineering code, and function calling using techniques such as rejection sampling and reinforcement learning, strengthening the atomic capabilities required for agent tasks. GLM-4-32B-0414 achieves good results in areas such as engineering code, Artifact generation, function calling, search-based Q&A, and report generation. Some benchmarks even rival larger models like GPT-4o and DeepSeek-V3-0324 (671B).

**GLM-Z1-32B-0414** is a reasoning model with **deep thinking capabilities**. This was developed based on GLM-4-32B-0414 through cold start and extended reinforcement learning, as well as further training of the model on tasks involving mathematics, code, and logic. Compared to the base model, GLM-Z1-32B-0414 significantly improves mathematical abilities and the capability to solve complex tasks. During the training process, we also introduced general reinforcement learning based on pairwise ranking feedback, further enhancing the model's general capabilities.

**GLM-Z1-Rumination-32B-0414** is a deep reasoning model with **rumination capabilities** (benchmarked against OpenAI's Deep Research). Unlike typical deep thinking models, the rumination model employs longer periods of deep thought to solve more open-ended and complex problems (e.g., writing a comparative analysis of AI development in two cities and their future development plans). The rumination model integrates search tools during its deep thinking process to handle complex tasks and is trained by utilizing multiple rule-based rewards to guide and extend end-to-end reinforcement learning. Z1-Rumination shows significant improvements in research-style writing and complex retrieval tasks.

Finally, **GLM-Z1-9B-0414** is a surprise. We employed the aforementioned series of techniques to train a 9B small-sized model that maintains the open-source tradition. Despite its smaller scale, GLM-Z1-9B-0414 still exhibits excellent capabilities in mathematical reasoning and general tasks. Its overall performance is already at a leading level among open-source models of the same size. Especially in resource-constrained scenarios, this model achieves an excellent balance between efficiency and effectiveness, providing a powerful option for users seeking lightweight deployment.

## Performance

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/THUDM/GLM-4/refs/heads/main/resources/Bench-Z1-32B.png">
</p>

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/THUDM/GLM-4/refs/heads/main/resources/Bench-Z1-9B.png">
</p>

## Model Usage Guidelines

### I. Sampling Parameters

| Parameter    | Recommended Value | Description                                  |
| ------------ | ----------------- | -------------------------------------------- |
| temperature  | **0.6**           | Balances creativity and stability            |
| top_p        | **0.95**          | Cumulative probability threshold for sampling|
| top_k        | **40**         | Filters out rare tokens while maintaining diversity |
| max_new_tokens        | **30000**         | Leaves enough tokens for thinking |

### II. Enforced Thinking

- Add \<think\>\n to the **first line**: Ensures the model thinks before responding  
- When using `chat_template.jinja`, the prompt is automatically injected to enforce this behavior


### III. Dialogue History Trimming

- Retain only the **final user-visible reply**.  
  Hidden thinking content should **not** be saved to history to reduce interference—this is already implemented in `chat_template.jinja`


### IV. Handling Long Contexts (YaRN)

- When input length exceeds **8,192 tokens**, consider enabling YaRN (Rope Scaling)

- In supported frameworks, add the following snippet to `config.json`:

  ```json
  "rope_scaling": {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
  ```

- **Static YaRN** applies uniformly to all text. It may slightly degrade performance on short texts, so enable as needed.

## Inference Code

Make Sure Using `transforemrs>=4.51.3`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/GLM-4-Z1-9B-0414"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

message = [{"role": "user", "content": "Let a, b be positive real numbers such that ab = a + b + 3. Determine the range of possible values for a + b."}]

inputs = tokenizer.apply_chat_template(
    message,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

generate_kwargs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "max_new_tokens": 4096,
    "do_sample": False,
}
out = model.generate(**generate_kwargs)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Citations

If you find our work useful, please consider citing the following paper.

```
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, 
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```