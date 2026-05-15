---
library_name: transformers
license: apache-2.0
base_model: 01-ai/Yi-Coder-9B
pipeline_tag: text-generation
---
<div align="center">

<picture> 
  <img src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" width="120px">
</picture>

</div>

<p align="center">
  <a href="https://github.com/01-ai">🐙 GitHub</a> •
  <a href="https://discord.gg/hYUwWddeAu">👾 Discord</a> •
  <a href="https://twitter.com/01ai_yi">🐤 Twitter</a> •
  <a href="https://github.com/01-ai/Yi-1.5/issues/2">💬 WeChat</a> 
  <br/>
  <a href="https://arxiv.org/abs/2403.04652">📝 Paper</a> •
  <a href="https://01-ai.github.io/">💪 Tech Blog</a> •
  <a href="https://github.com/01-ai/Yi/tree/main?tab=readme-ov-file#faq">🙌 FAQ</a> •
  <a href="https://github.com/01-ai/Yi/tree/main?tab=readme-ov-file#learning-hub">📗 Learning Hub</a>
</p>

# Intro

Yi-Coder is a series of open-source code language models that delivers state-of-the-art coding performance with fewer than 10 billion parameters. 

Key features:
- Excelling in long-context understanding with a maximum context length of 128K tokens.
- Supporting 52 major programming languages:
```bash
  'java', 'markdown', 'python', 'php', 'javascript', 'c++', 'c#', 'c', 'typescript', 'html', 'go', 'java_server_pages', 'dart', 'objective-c', 'kotlin', 'tex', 'swift', 'ruby', 'sql', 'rust', 'css', 'yaml', 'matlab', 'lua', 'json', 'shell', 'visual_basic', 'scala', 'rmarkdown', 'pascal', 'fortran', 'haskell', 'assembly', 'perl', 'julia', 'cmake', 'groovy', 'ocaml', 'powershell', 'elixir', 'clojure', 'makefile', 'coffeescript', 'erlang', 'lisp', 'toml', 'batchfile', 'cobol', 'dockerfile', 'r', 'prolog', 'verilog'
  ```

For model details and benchmarks, see [Yi-Coder blog](https://01-ai.github.io/) and [Yi-Coder README](https://github.com/01-ai/Yi-Coder).

<p align="left"> 
  <img src="https://github.com/01-ai/Yi/blob/main/assets/img/coder/yi-coder-calculator-demo.gif?raw=true" alt="demo1" width="500"/> 
</p>

# Models

| Name               | Type |  Length | Download                                                                                                                                          |
|--------------------|------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Yi-Coder-9B-Chat   | Chat |      128K      | [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-Coder-9B-Chat) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-Coder-9B-Chat) • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-Coder-9B-Chat) |
| Yi-Coder-1.5B-Chat | Chat |      128K      | [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-Coder-1.5B-Chat) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-Coder-1.5B-Chat) • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-Coder-1.5B-Chat) |
| Yi-Coder-9B        | Base |      128K      | [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-Coder-9B) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-Coder-9B) • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-Coder-9B) |
| Yi-Coder-1.5B      | Base |      128K      | [🤗 Hugging Face](https://huggingface.co/01-ai/Yi-Coder-1.5B) • [🤖 ModelScope](https://www.modelscope.cn/models/01ai/Yi-Coder-1.5B) • [🟣 wisemodel](https://wisemodel.cn/models/01.AI/Yi-Coder-1.5B) |
|                    | 

# Benchmarks

As illustrated in the figure below, Yi-Coder-9B-Chat achieved an impressive 23% pass rate in LiveCodeBench, making it the only model with under 10B parameters to surpass 20%. It also outperforms DeepSeekCoder-33B-Ins at 22.3%, CodeGeex4-9B-all at 17.8%, CodeLLama-34B-Ins at 13.3%, and CodeQwen1.5-7B-Chat at 12%.

<p align="left"> 
  <img src="https://github.com/01-ai/Yi/blob/main/assets/img/coder/bench1.webp?raw=true" alt="bench1" width="1000"/> 
</p>

# Quick Start

You can use transformers to run inference with Yi-Coder models (both chat and base versions) as follows:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" # the device to load the model onto
model_path = "01-ai/Yi-Coder-9B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()

prompt = "Write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id  
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

For getting up and running with Yi-Coder series models quickly, see [Yi-Coder README](https://github.com/01-ai/Yi-Coder).