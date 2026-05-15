---
pipeline_tag: text-generation
inference: false
license: apache-2.0
datasets:
- codeparrot/github-code-clean
- bigcode/starcoderdata
# - Stackexchange
# - CommonCrawl
- open-web-math/open-web-math
- math-ai/StackMathQA
# - Arxiv
# - Wikipedia
# - conceptofmind/FLAN_2022 # Original link is broken, we used IBM's filtered version
metrics:
- code_eval
library_name: transformers
tags:
- code
- granite
model-index:
- name: granite-8B-code-base-128k
  results:
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack 
        name: HumanEvalSynthesis (Python)
    metrics:
    - name: pass@1
      type: pass@1
      value: 43.1
      verified: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name: HumanEvalSynthesis (Average)
    metrics:
    - name: pass@1
      type: pass@1
      value: 40.2
      verified: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain (Average)
    metrics:
    - name: pass@1
      type: pass@1
      value: 28.2
      verified: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix (Average)
    metrics:
    - name: pass@1
      type: pass@1
      value: 25.2
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (Python@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 48.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (C++@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 36.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (Java@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 38.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (TypeScript@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 39.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (Rust@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 29.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: lcc  
        name:  LCC (Balanced)
    metrics:
    - name: Exact Match@4K 
      type: Exact Match@4K
      value: 56.5
      verified: false
  - task:
      type: text-generation
    dataset:
        type: lcc  
        name:  LCC (Balanced)
    metrics:
    - name: Exact Match@8K 
      type: Exact Match@8K
      value: 60.1
      verified: false
  - task:
      type: text-generation
    dataset:
        type: lcc  
        name:  LCC (Balanced)
    metrics:
    - name: Exact Match@16K 
      type: Exact Match@16K
      value: 51.8
      verified: false
  - task:
      type: text-generation
    dataset:
        type: lcc  
        name:  LCC (Balanced)
    metrics:
    - name: Exact Match@32K 
      type: Exact Match@32K
      value: 57.4
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repobench  
        name: RepoBench-P (Balanced)
    metrics:
    - name: Exact Match@4K 
      type: Exact Match@4K
      value: 42.7
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repobench  
        name: RepoBench-P (Balanced)
    metrics:
    - name: Exact Match@8K 
      type: Exact Match@8K
      value: 44.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repobench  
        name: RepoBench-P (Balanced)
    metrics:
    - name: Exact Match@16K 
      type: Exact Match@16K
      value: 44.8
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repobench  
        name: RepoBench-Pn(Balanced)
    metrics:
    - name: Exact Match@32K 
      type: Exact Match@32K
      value: 44.5
      verified: false
---

---

# ⚠️ DEPRECATION WARNING ⚠️
# ⚠️ NOT RECOMMENDED FOR USE IN NEW PROJECTS ⚠️

New applications/projects should use the latest mainline Granite language model family, whose code capabilities supercede this model. 
This model is being made available strictly for historical/scientific purposes. 
**Please see our [Granite Collections](https://huggingface.co/ibm-granite/collections) for the latest Granite releases.**

---

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62cd5057674cdb524450093d/1hzxoPwqkBJXshKVVe6_9.png)

# Granite-8B-Code-Base-128K

## Model Summary
**Granite-8B-Code-Base-128K** extends the context length of Granite-8B-Code-Base from 4K to 128K with continual pretraining using the original training data but with repository-level file packing and per-language length upsampling, that we found to be critical for long-context pretraining. 
We adopt an progressive training strategy where we doubled the context window until it reached the desired length of 128K by appropriately adjusting RoPE theta. We trained on 4B tokens total for all stages, which is only 0.1% of Granite-8B-Code-Base's original pre-training data. 

- **Developers:** IBM Research
- **GitHub Repository:** [ibm-granite/granite-code-models](https://github.com/ibm-granite/granite-code-models)
- **Paper:** [Scaling Granite Code Models to 128K Context](https://arxiv.org/abs/2405.04324)
- **Release Date**: July 18th, 2024
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Usage
### Intended use
Prominent enterprise use cases of LLMs in software engineering productivity with 128K context length support that includes code generation, code explanation, code fixing, generating unit tests, generating documentation, addressing technical debt issues, vulnerability detection, code translation, and more. All Granite Code Base models, including the **3B parameter model**, are able to handle these tasks as they were trained on a large amount of code data from 116 programming languages. 

### Generation
This is a simple example of how to use **Granite-8B-Code-Base-128K** model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # or "cpu"
model_path = "ibm-granite/granite-8B-code-base-128k"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()
# change input text as desired
input_text = "def generate():"
# tokenize the text
input_tokens = tokenizer(input_text, return_tensors="pt")
# transfer tokenized inputs to the device
for i in input_tokens:
    input_tokens[i] = input_tokens[i].to(device)
# generate output tokens
output = model.generate(**input_tokens)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# loop over the batch to print, in this example the batch size is 1
for i in output:
    print(i)
```

## Training Data
Starting from the base Granite model, this model was further pretrained on repository-level code data with per-language context-length oversampling, allowing it to effectively utilize up to 128K tokens of context. This continued training stage focused on a curated selection of programming languages, such as Python, C, C++, Go, Java, JavaScript, and TypeScript.

## Infrastructure
We train the Granite Code models using two of IBM's super computing clusters, namely Vela and Blue Vela, both outfitted with NVIDIA A100 and H100 GPUs respectively. These clusters provide a scalable and efficient infrastructure for training our models over thousands of GPUs.

## Ethical Considerations and Limitations
The use of Large Language Models involves risks and ethical considerations people must be aware of. Regarding code generation, caution is urged against complete reliance on specific code models for crucial decisions or impactful information as the generated code is not guaranteed to work as intended. **Granite-8B-code-Base-128K** model is not the exception in this regard. Even though this model is suited for multiple code-related tasks, it has not undergone any safety alignment, there it may produce problematic outputs. Additionally, it remains uncertain whether smaller models might exhibit increased susceptibility to hallucination in generation scenarios by copying source code verbatim from the training dataset due to their reduced sizes and memorization capacities. This aspect is currently an active area of research, and we anticipate more rigorous exploration, comprehension, and mitigations in this domain. Regarding ethics, a latent risk associated with all Large Language Models is their malicious utilization. We urge the community to use **Granite-8B-Code-Base-128K** model with ethical intentions and in a responsible way. 

