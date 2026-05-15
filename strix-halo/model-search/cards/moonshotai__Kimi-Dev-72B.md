---
license: mit
base_model:
- Qwen/Qwen2.5-72B
tags:
- code
- swebench
- software
- issue-resolving
library_name: transformers
---
<!-- # Kimi-Dev -->

<div align="center">
  <img src="./assets/main_logo.png" alt="Kimi Logo" width="400" />
<h2><a href="https://moonshotai.github.io/Kimi-Dev/">
Introducing Kimi-Dev: <br>A Strong and Open-source Coding LLM for Issue Resolution</a></h2>
</a></h2>
<b>Kimi-Dev Team</b>
<br>

</div>
<div align="center">
  <a href="">
    <b>📄 Tech Report (Coming soon...)</b>
  </a> &nbsp;|&nbsp;
  <a href="https://github.com/MoonshotAI/Kimi-Dev">
    <b>📄 Github</b>
  </a> &nbsp;
</div>

<br>
<br>

<!-- https://github.com/MoonshotAI/Kimi-Dev -->

We introduce Kimi-Dev-72B, our new open-source coding LLM for software engineering tasks. Kimi-Dev-72B achieves a new state-of-the-art on SWE-bench Verified among open-source models.

- Kimi-Dev-72B achieves 60.4% performance on SWE-bench Verified. It surpasses the runner-up, setting a new state-of-the-art result among open-source models.


- Kimi-Dev-72B is optimized via large-scale reinforcement learning. It autonomously patches real repositories in Docker and gains rewards only when the entire test suite passes. This ensures correct and robust solutions, aligning with real-world development standards.


- Kimi-Dev-72B is available for download and deployment on Hugging Face and GitHub. We welcome developers and researchers to explore its capabilities and contribute to development.


<div align="center">
  <img src="./assets/open_performance_white.png" alt="Kimi Logo" width="600" />
  <p><b>Performance of Open-source Models on SWE-bench Verified.</b></p>

</div>



## Quick Start
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "moonshotai/Kimi-Dev-72B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

```

## Citation
```
@misc{kimi_dev_72b_2025,
  title        = {Introducing Kimi-Dev: A Strong and Open-source Coding LLM for Issue Resolution},
  author       = {{Kimi-Dev Team}},
  year         = {2025},
  month        = {June},
  url          = {\url{https://www.moonshot.cn/Kimi-Dev}}
}
```