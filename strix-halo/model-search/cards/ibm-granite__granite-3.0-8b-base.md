---
pipeline_tag: text-generation
inference: false
license: apache-2.0
library_name: transformers
tags:
- language
- granite-3.0
model-index:
- name: granite-3.0-8b-base
  results:
  - task:
      type: text-generation
    dataset:
      type: human-exams
      name: MMLU
    metrics:
    - name: pass@1
      type: pass@1
      value: 65.54
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: human-exams
      name: MMLU-Pro
    metrics:
    - name: pass@1
      type: pass@1
      value: 33.27
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: human-exams
      name: AGI-Eval
    metrics:
    - name: pass@1
      type: pass@1
      value: 34.45
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: WinoGrande
    metrics:
    - name: pass@1
      type: pass@1
      value: 80.9
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: OBQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 46.8
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: SIQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 67.8
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: PIQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 82.32
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: Hellaswag
    metrics:
    - name: pass@1
      type: pass@1
      value: 83.61
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: commonsense
      name: TruthfulQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 52.89
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reading-comprehension
      name: BoolQ
    metrics:
    - name: pass@1
      type: pass@1
      value: 86.97
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reading-comprehension
      name: SQuAD 2.0
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.92
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reasoning
      name: ARC-C
    metrics:
    - name: pass@1
      type: pass@1
      value: 63.4
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reasoning
      name: GPQA
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.13
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reasoning
      name: BBH
    metrics:
    - name: pass@1
      type: pass@1
      value: 49.31
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: reasoning
      name: MUSR
    metrics:
    - name: pass@1
      type: pass@1
      value: 41.08
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: code
      name: HumanEval
    metrics:
    - name: pass@1
      type: pass@1
      value: 52.44
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: code
      name: MBPP
    metrics:
    - name: pass@1
      type: pass@1
      value: 41.4
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: math
      name: GSM8K
    metrics:
    - name: pass@1
      type: pass@1
      value: 64.06
      veriefied: false
  - task:
      type: text-generation
    dataset:
      type: math
      name: MATH
    metrics:
    - name: pass@1
      type: pass@1
      value: 29.28
      veriefied: false
new_version: ibm-granite/granite-3.1-8b-base
---
<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/62cd5057674cdb524450093d/1hzxoPwqkBJXshKVVe6_9.png) -->
<!-- ![image/png](granite-3_0-language-models_Group_1.png) -->

# Granite-3.0-8B-Base

<!-- **Note: We are continuously improving our models and recommend users to checkout our latest [Granite 3.1](https://huggingface.co/collections/ibm-granite/granite-31-language-models-6751dbbf2f3389bec5c6f02d) models.** -->

**Model Summary:** 
Granite-3.0-8B-Base is a decoder-only language model to support a variety of text-to-text generation tasks. It is trained from scratch following a two-stage training strategy. In the first stage, it is trained on 10 trillion tokens sourced from diverse domains. During the second stage, it is further trained on 2 trillion tokens using a carefully curated mix of high-quality data, aiming to enhance its performance on specific tasks.

- **Developers:** Granite Team, IBM
- **GitHub Repository:** [ibm-granite/granite-3.0-language-models](https://github.com/ibm-granite/granite-3.0-language-models)
- **Website**: [Granite Docs](https://www.ibm.com/granite/docs/)
- **Paper:** [Granite 3.0 Language Models](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) 
- **Release Date**: October 21st, 2024
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Languages:** 
English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. Users may finetune Granite 3.0 models for languages beyond these 12 languages.

**Intended use:** 
Prominent use cases of LLMs in text-to-text generation include summarization, text classification, extraction, question-answering, and more. All Granite Base models are able to handle these tasks as they were trained on a large amount of data from various domains. Moreover, they can serve as baseline to create specialized models for specific application scenarios.

**Generation:** 
This is a simple example of how to use Granite-3.0-8B-Base model.

Install the following libraries:

```shell
pip install torch torchvision torchaudio
pip install accelerate
pip install transformers
```
Then, copy the code snippet below to run the example.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "auto"
model_path = "ibm-granite/granite-3.0-8b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()
# change input text as desired
input_text = "Where is the Thomas J. Watson Research Center located?"
# tokenize the text
input_tokens = tokenizer(input_text, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens,
                        max_length=4000)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output)
```

**Model Architecture:** 
Granite-3.0-8B-Base is based on a decoder-only dense transformer architecture. Core components of this architecture are: GQA and RoPE, MLP with SwiGLU, RMSNorm, and shared input/output embeddings.

| Model                     | 2B Dense | 8B Dense     | 1B MoE | 3B MoE |
| :--------                 | :--------| :--------    | :------| :------|
| Embedding size            | 2048     | **4096**     | 1024   | 1536   |
| Number of layers          | 40       | **40**       | 24     | 32     |
| Attention head size       | 64       | **128**      | 64     | 64     |
| Number of attention heads | 32       | **32**       | 16     | 24     |
| Number of KV heads        | 8        | **8**        | 8      | 8      |
| MLP hidden size           | 8192     | **12800**    | 512    | 512    |
| MLP activation            | SwiGLU   | **SwiGLU**   | SwiGLU | SwiGLU |
| Number of Experts         | —        | **—**        | 32     | 40     |
| MoE TopK                  | —        | **—**        | 8      | 8      |
| Initialization std        | 0.1      | **0.1**      | 0.1    | 0.1    |
| Sequence Length           | 4096     | **4096**     | 4096   | 4096   |
| Position Embedding        | RoPE     | **RoPE**     | RoPE   | RoPE   |
| # Parameters              | 2.5B     | **8.1B**     | 1.3B   | 3.3B   |
| # Active Parameters       | 2.5B     | **8.1B**     | 400M   | 800M   |
| # Training tokens         | 12T      | **12T**      | 10T    | 10T    |

**Training Data:** 
This model is trained on a mix of open source and proprietary data following a two-stage training strategy.
* Stage 1 data: The data for stage 1 is sourced from diverse domains, such as: web, code, academic sources, books, and math data.
* Stage 2 data: The data for stage 2 comprises a curated mix of high-quality data from the same domains, plus multilingual and instruction data. The goal of this second training phase is to enhance the model’s performance on specific tasks. 

A detailed attribution of datasets can be found in the [Granite Technical Report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) and [Accompanying Author List](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/author-ack.pdf).

**Infrastructure:**
We train Granite 3.0 Language Models using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs while minimizing environmental impact by utilizing 100% renewable energy sources.

**Ethical Considerations and Limitations:** 
The use of Large Language Models involves risks and ethical considerations people must be aware of, including but not limited to: bias and fairness, misinformation, and autonomous decision-making. Granite-3.0-8B-Base model is not the exception in this regard. Even though this model is suited for multiple generative AI tasks, it has not undergone any safety alignment, there it may produce problematic outputs. Additionally, it remains uncertain whether smaller models might exhibit increased susceptibility to hallucination in generation scenarios by copying text verbatim from the training dataset due to their reduced sizes and memorization capacities. This aspect is currently an active area of research, and we anticipate more rigorous exploration, comprehension, and mitigations in this domain. Regarding ethics, a latent risk associated with all Large Language Models is their malicious utilization. We urge the community to use Granite-3.0-8B-Base model with ethical intentions and in a responsible way. 

**Resources**
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 📄 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources

<!-- ## Citation
```
@misc{granite-models,
  author = {author 1, author2, ...},
  title = {},
  journal = {},
  volume = {},
  year = {2024},
  url = {https://arxiv.org/abs/0000.00000},
}
``` -->