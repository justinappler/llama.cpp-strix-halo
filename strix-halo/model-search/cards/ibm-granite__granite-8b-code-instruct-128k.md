---
pipeline_tag: text-generation
inference: false
license: apache-2.0
datasets:
  - bigcode/commitpackft
  - TIGER-Lab/MathInstruct
  - meta-math/MetaMathQA
  - glaiveai/glaive-code-assistant-v3
  - glaive-function-calling-v2
  - bugdaryan/sql-create-context-instruction
  - garage-bAInd/Open-Platypus
  - nvidia/HelpSteer
  - bigcode/self-oss-instruct-sc2-exec-filter-50k
metrics:
- code_eval
library_name: transformers
tags:
- code
- granite
model-index:
- name: granite-8B-Code-instruct-128k
  results:
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack 
        name: HumanEvalSynthesis (Python)
    metrics:
    - name: pass@1
      type: pass@1
      value: 62.2
      verified: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name: HumanEvalSynthesis (Average)
    metrics:
    - name: pass@1
      type: pass@1
      value: 51.4
      verified: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalExplain (Average)
    metrics:
    - name: pass@1
      type: pass@1
      value: 38.9
      verified: false
  - task:
      type: text-generation
    dataset:
        type: bigcode/humanevalpack  
        name:  HumanEvalFix (Average)
    metrics:
    - name: pass@1
      type: pass@1
      value: 38.3
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (Python@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 73.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (C++@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 37.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (Java@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 73.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (TypeScript@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 62.0
      verified: false
  - task:
      type: text-generation
    dataset:
        type: repoqa  
        name:  RepoQA (Rust@16K)
    metrics:
    - name: pass@1 (thresh=0.5)
      type: pass@1 (thresh=0.5)
      value: 63.0
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

# Granite-8B-Code-Instruct-128K

## Model Summary
**Granite-8B-Code-Instruct-128K** is a 8B parameter long-context instruct model fine tuned from *Granite-8B-Code-Base-128K* on a combination of **permissively licensed** data used in training the original Granite code instruct models, in addition to synthetically generated code instruction datasets tailored for solving long context problems. By exposing the model to both short and long context data, we aim to enhance its long-context capability without sacrificing code generation performance at short input context.

- **Developers:** IBM Research
- **GitHub Repository:** [ibm-granite/granite-code-models](https://github.com/ibm-granite/granite-code-models)
- **Paper:** [Scaling Granite Code Models to 128K Context](https://arxiv.org/abs/2407.13739)
- **Release Date**: July 18th, 2024
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Usage
### Intended use
The model is designed to respond to coding related instructions over long-conext input up to 128K length and can be used to build coding assistants.

<!-- TO DO: Check starcoder2 instruct code example that includes the template https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1 -->

### Generation
This is a simple example of how to use **Granite-8B-Code-Instruct** model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # or "cpu"
model_path = "ibm-granite/granite-8B-Code-instruct-128k"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()
# change input text as desired
chat = [
    { "role": "user", "content": "Write a code to find the maximum value in a list of numbers." },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt")
# transfer tokenized inputs to the device
for i in input_tokens:
    input_tokens[i] = input_tokens[i].to(device)
# generate output tokens
output = model.generate(**input_tokens, max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# loop over the batch to print, in this example the batch size is 1
for i in output:
    print(i)
```

<!-- TO DO: Check this part -->
## Training Data
Granite Code Instruct models are trained on a mix of short and long context data as follows.
* Short-Context Instruction Data: [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft), [BigCode-SC2-Instruct](bigcode/self-oss-instruct-sc2-exec-filter-50k), [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct), [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [Glaive-Code-Assistant-v3](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3), [Glaive-Function-Calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2), [NL2SQL11](https://huggingface.co/datasets/bugdaryan/sql-create-context-instruction), [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer), [OpenPlatypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) including a synthetically generated dataset for API calling and multi-turn code interactions with execution feedback. We also include a collection of hardcoded prompts to ensure our model generates correct outputs given inquiries about its name or developers.
* Long-Context Instruction Data: A synthetically-generated dataset by bootstrapping the repository-level file-packed documents through Granite-8b-Code-Instruct to improve long-context capability of the model.
  
## Infrastructure
We train the Granite Code models using two of IBM's super computing clusters, namely Vela and Blue Vela, both outfitted with NVIDIA A100 and H100 GPUs respectively. These clusters provide a scalable and efficient infrastructure for training our models over thousands of GPUs.

## Ethical Considerations and Limitations
Granite code instruct models are primarily finetuned using instruction-response pairs across a specific set of programming languages. Thus, their performance may be limited with out-of-domain programming languages. In this situation, it is beneficial providing few-shot examples to steer the model's output. Moreover, developers should perform safety testing and target-specific tuning before deploying these models on critical applications. The model also inherits ethical considerations and limitations from its base model. For more information, please refer to *[Granite-8B-Code-Base-128K](https://huggingface.co/ibm-granite/granite-8B-Code-base-128k)* model card.
