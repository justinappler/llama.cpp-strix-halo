---
license: apache-2.0
library_name: transformers
tags:
- language
- granite-3.3
---

# Granite-3.3-8B-Base

**Model Summary:** 


Granite-3.3-8B-Base is a decoder-only language model with a 128K token context window. It improves upon Granite-3.1-8B-Base by adding support for Fill-in-the-Middle (FIM) using specialized tokens, enabling the model to generate content conditioned on both prefix and suffix. This makes it well-suited for code completion tasks.



- **Developers:** Granite Team, IBM
- **GitHub Repository:** [ibm-granite/granite-3.3-language-models](https://github.com/ibm-granite/granite-3.3-language-models)
- **Website**: [Granite Docs](https://www.ibm.com/granite/docs/) 
- **Release Date**: April 16th, 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Languages:** 
English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. Users may finetune Granite 3.3 models for languages beyond these 12 languages.

**Intended Use:**
Prominent use cases of LLMs in text-to-text generation include summarization, text classification, extraction, question-answering, and other long-context tasks. All Granite Base models are able to handle these tasks as they were trained on a large amount of data from various domains. Moreover, they can serve as baseline to create specialized models for specific application scenarios.

**Generation:** 
This is a simple example of how to use Granite-3.3-8B-Base model.

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
model_path = "ibm-granite/granite-3.3-8b-base"
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

**Evaluation Results:** 

<table>
  <caption><b>Comparison with 3.1 Base models</b><sup id="fnref1"><a href="#fn1">1</a></caption>
<thead>
  <tr>
    <th style="text-align:left; background-color: #001d6c; color: white;">Models</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">ARC-Challenge</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">Hellaswag</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">MMLU</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">TruthfulQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">Winogrande</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">GSM8K</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">DROP</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">NQ</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">AGIEval</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">TriviaQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">Avg</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.1-2B-Base</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">46.83</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">74.9</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">54.87</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">38.93</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">71.8</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">53.0</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">30.08</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">24.46</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">38.24</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">63.18</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">49.63</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-3.3-2B-Base</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 47.49 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 73.2 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 54.33 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 40.83 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 70.4 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 50.0 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 32.552 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">24.36</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">38.78</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">63.22</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">49.52</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.1-8B-Base</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">53.51</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">81.4</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">64.28</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">51.27</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">76.2</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">70.5</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">45.87</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">35.97</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">48.99</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">78.33</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">60.63</td>
  </tr>

  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-3.3-8B-Base</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">50.84</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">80.1</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">63.89</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">52.15</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">74.4</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">59.0</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">36.14</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">36.5</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">49.3</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">78.18</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">58.05</td>
  </tr>
</tbody></table>

**Model Architecture:** 
Granite-3.3-8B-Base is based on a decoder-only dense transformer architecture. Core components of this architecture are: GQA and RoPE, MLP with SwiGLU, RMSNorm, and shared input/output embeddings.
<table>
<thead>
  <tr>
    <th style="text-align:left; background-color: #001d6c; color: white;">Model</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">2B Dense</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">8B Dense</th>
  </tr></thead>
<tbody>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Embedding size</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">2048</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">4096</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Number of layers</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">40</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">40</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Attention head size</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">64</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">128</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Number of attention heads</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">32</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">32</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Number of KV heads</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">8</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">8</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">MLP hidden size</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">8192</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">12800</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">MLP activation</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">SwiGLU</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">SwiGLU</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Initialization std</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">0.1</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">0.1</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Sequence length</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">128K</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">128K</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;">Position embedding</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">RoPE</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">RoPE</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;"># Parameters</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">2.5B</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">8.1B</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;"># Active parameters</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">2.5B</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">8.1B</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: black;"># Training tokens</td>
    <td style="text-align:center; background-color: #FFFFFF; color: black;">12T</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">12T</td>
  </tr>
</tbody></table>

**Training Data:** 
This model is trained on a mix of open source and proprietary data following a three-stage training strategy.
* Stage 1 data: The data for stage 1 is sourced from diverse domains, such as: web, code, academic sources, books, and math data.
* Stage 2 data: The data for stage 2 comprises a curated mix of high-quality data from the same domains, plus multilingual and instruction data. The goal of this second training phase is to enhance the model’s performance on specific tasks. 
* Stage 3 data: The data for stage 3 consists of original stage-2 pretraining data with additional synthetic long-context data in form of QA/summary pairs where the answer
contains a recitation of the related paragraph before the answer.

<!-- A detailed attribution of datasets can be found in the [Granite 3.0 Technical Report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf), [Granite 3.3 Technical Report (coming soon)](https://huggingface.co/collections/ibm-granite/granite-31-language-models-6751dbbf2f3389bec5c6f02d), and [Accompanying Author List](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/author-ack.pdf). -->

**Infrastructure:**
We train Granite 3.3 Language Models using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs.

**Ethical Considerations and Limitations:** 
The use of Large Language Models involves risks and ethical considerations people must be aware of, including but not limited to: bias and fairness, misinformation, and autonomous decision-making. Granite-3.3-8B-Base model is not the exception in this regard. Even though this model is suited for multiple generative AI tasks, it has not undergone any safety alignment, there it may produce problematic outputs. Additionally, it remains uncertain whether smaller models might exhibit increased susceptibility to hallucination in generation scenarios by copying text verbatim from the training dataset due to their reduced sizes and memorization capacities. This aspect is currently an active area of research, and we anticipate more rigorous exploration, comprehension, and mitigations in this domain. Regarding ethics, a latent risk associated with all Large Language Models is their malicious utilization. We urge the community to use Granite-3.3-8B-Base model with ethical intentions and in a responsible way. 

**Resources**
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 📄 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://github.com/ibm-granite-community/

<p><a href="#fnref1" title="Jump back to reference">[1]</a> Evaluated using <a href="https://github.com/allenai/olmes">OLMES</a></p>
