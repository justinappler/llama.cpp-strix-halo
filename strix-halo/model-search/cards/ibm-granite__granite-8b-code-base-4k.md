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
# - conceptofmind/FLAN_2022 # Original link is broken, we used IBM's filtered version | Phase 2
metrics:
- code_eval
library_name: transformers
tags:
- code
- granite
model-index:
- name: granite-8b-code-base-4k
  results:
  - task:
      type: text-generation
    dataset:
        type: mbpp
        name: MBPP
    metrics:
    - name: pass@1
      type: pass@1
      value: 42.2
      veriefied: false      
  - task:
      type: text-generation
    dataset:
        type: evalplus/mbppplus
        name: MBPP+
    metrics:
    - name: pass@1
      type: pass@1
      value: 49.6
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack 
        name: HumanEvalSynthesis(Python)
    metrics:
    - name: pass@1
      type: pass@1
      value: 43.9
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name: HumanEvalSynthesis(JavaScript)
    metrics:
    - name: pass@1
      type: pass@1
      value: 52.4
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name: HumanEvalSynthesis(Java)
    metrics:
    - name: pass@1
      type: pass@1
      value: 56.1
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name: HumanEvalSynthesis(Go)
    metrics:
    - name: pass@1
      type: pass@1
      value: 31.7
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name: HumanEvalSynthesis(C++)
    metrics:
    - name: pass@1
      type: pass@1
      value: 43.9
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name: HumanEvalSynthesis(Rust)
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.9
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain(Python)
    metrics:
    - name: pass@1
      type: pass@1
      value: 23.5
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain(JavaScript)
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.3
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain(Java)
    metrics:
    - name: pass@1
      type: pass@1
      value: 25.0
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain(Go)
    metrics:
    - name: pass@1
      type: pass@1
      value: 23.2
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain(C++)
    metrics:
    - name: pass@1
      type: pass@1
      value: 28.0
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain(Rust)
    metrics:
    - name: pass@1
      type: pass@1
      value: 19.5
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix(Python)
    metrics:
    - name: pass@1
      type: pass@1
      value: 22.6
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix(JavaScript)
    metrics:
    - name: pass@1
      type: pass@1
      value: 35.4
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix(Java)
    metrics:
    - name: pass@1
      type: pass@1
      value: 38.4
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix(Go)
    metrics:
    - name: pass@1
      type: pass@1
      value: 37.2
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix(C++)
    metrics:
    - name: pass@1
      type: pass@1
      value: 28.7 
      veriefied: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix(Rust)
    metrics:
    - name: pass@1
      type: pass@1
      value: 15.2
      veriefied: false
---


---

# ⚠️ DEPRECATION WARNING ⚠️
# ⚠️ NOT RECOMMENDED FOR USE IN NEW PROJECTS ⚠️

New applications/projects should use the latest mainline Granite language model family, whose code capabilities supercede this model. 
This model is being made available strictly for historical/scientific purposes. 
**Please see our [Granite Collections](https://huggingface.co/ibm-granite/collections) for the latest Granite releases.**

---


![image/png](https://cdn-uploads.huggingface.co/production/uploads/62cd5057674cdb524450093d/1hzxoPwqkBJXshKVVe6_9.png)

# Granite-8B-Code-Base-4K

## Model Summary
**Granite-8B-Code-Base-4K** is a decoder-only code model designed for code generative tasks (e.g., code generation, code explanation, code fixing, etc.). It is trained from scratch with a two-phase training strategy. In phase 1, our model is trained on 4 trillion tokens sourced from 116 programming languages, ensuring a comprehensive understanding of programming languages and syntax. In phase 2, our model is trained on 500 billion tokens with a carefully designed mixture of high-quality data from code and natural language domains to improve the models’ ability to reason and follow instructions.

- **Developers:** IBM Research
- **GitHub Repository:** [ibm-granite/granite-code-models](https://github.com/ibm-granite/granite-code-models)
- **Paper:** [Granite Code Models: A Family of Open Foundation Models for Code Intelligence](https://arxiv.org/abs/2405.04324)
- **Release Date**: May 6th, 2024
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Usage
### Intended use
Prominent enterprise use cases of LLMs in software engineering productivity include code generation, code explanation, code fixing, generating unit tests, generating documentation, addressing technical debt issues, vulnerability detection, code translation, and more. All Granite Code Base models, including the **8B parameter model**, are able to handle these tasks as they were trained on a large amount of code data from 116 programming languages. 

### Generation
This is a simple example of how to use **Granite-8B-Code-Base-4K** model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # or "cpu"
model_path = "ibm-granite/granite-8b-code-base-4k"
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
- **Data Collection and Filtering:** Pretraining code data is sourced from a combination of publicly available datasets (e.g., [GitHub Code Clean](https://huggingface.co/datasets/codeparrot/github-code-clean), [Starcoder data](https://huggingface.co/datasets/bigcode/starcoderdata)), and additional public code repositories and issues from GitHub. We filter raw data to retain a list of 116 programming languages. After language filtering, we also filter out low-quality code. 
- **Exact and Fuzzy Deduplication:** We adopt an aggressive deduplication strategy that includes both exact and fuzzy deduplication to remove documents having (near) identical code content.
- **HAP, PII, Malware Filtering:** We apply a HAP content filter that reduces models' likelihood of generating hateful, abusive, or profane language. We also make sure to redact Personally Identifiable Information (PII) by replacing PII content (e.g., names, email addresses, keys, passwords) with corresponding tokens (e.g., ⟨NAME⟩, ⟨EMAIL⟩, ⟨KEY⟩, ⟨PASSWORD⟩). Moreover, we scan all datasets using [ClamAV](https://www.clamav.net/) to identify and remove instances of malware in the source code.
- **Natural Language Datasets:** In addition to collecting code data for model training, we curate several publicly available high-quality natural language datasets to improve models' proficiency in language understanding and mathematical reasoning. Unlike the code data, we do not deduplicate these datasets.

## Infrastructure
We train the Granite Code models using two of IBM's super computing clusters, namely Vela and Blue Vela, both outfitted with NVIDIA A100 and H100 GPUs respectively. These clusters provide a scalable and efficient infrastructure for training our models over thousands of GPUs.

## Ethical Considerations and Limitations
The use of Large Language Models involves risks and ethical considerations people must be aware of. Regarding code generation, caution is urged against complete reliance on specific code models for crucial decisions or impactful information as the generated code is not guaranteed to work as intended. **Granite-8B-Code-Base-4K** model is not the exception in this regard. Even though this model is suited for multiple code-related tasks, it has not undergone any safety alignment, there it may produce problematic outputs. Additionally, it remains uncertain whether smaller models might exhibit increased susceptibility to hallucination in generation scenarios by copying source code verbatim from the training dataset due to their reduced sizes and memorization capacities. This aspect is currently an active area of research, and we anticipate more rigorous exploration, comprehension, and mitigations in this domain. Regarding ethics, a latent risk associated with all Large Language Models is their malicious utilization. We urge the community to use **Granite-8B-Code-Base-4K** model with ethical intentions and in a responsible way. 
