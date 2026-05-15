---
license: apache-2.0
library_name: transformers
tags:
- dllm
- diffusion
- llm
- text_generation
---
# LLaDA2.1-flash

🚀 **LLaDA2.1-flash** is now live on **ZenmuxAI**! Try it via API 🛠️ or Chat 💬: https://zenmux.ai/inclusionai/llada2.1-flash

**LLaDA2.1-flash** is a diffusion language model of the LLaDA series featuring the editing enhancement. It significantly improves inference speed while delivering strong task performance.

<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*uOo8QKQMiBwAAAAAgNAAAAgAemJ7AQ/original" width="800" />
</div>

<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*biwvQpCmKjEAAAAAULAAAAgAemJ7AQ/original" width="800" />
</div>

---

<table>
<thead>
<tr>
<th align="left"><b>Benchmark</b></th>
<th align="center"><b>Qwen3-30B-<br>A3B-Inst-2507</b><br><sub>(Score)</sub></th>
<th align="center"><b>Ling-flash-2.0</b><br><br><sub>(Score)</sub></th>
<th align="center"><b>LLaDA2.0-flash</b><br><br><sub>(Score | TPF)</sub></th>
<th align="center"><b>LLaDA2.1-flash<br>(S Mode)</b><br><sub>(Score | TPF)</sub></th>
<th align="center"><b>LLaDA2.1-flash<br>(Q Mode)</b><br><sub>(Score | TPF)</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><b>Average</b></td>
<td align="center">73.09</td>
<td align="center">71.52</td>
<td align="center">72.43 | 3.08</td>
<td align="center">72.34 | 5.93</td>
<td align="center">73.54 | 3.64</td>
</tr>
<tr>
<td colspan="6" align="center"><b>Knowledge</b></td>
</tr>
<tr>
<td align="left">GPQA</td>
<td align="center">54.14</td>
<td align="center">69.16</td>
<td align="center">62.31 | 3.29</td>
<td align="center">66.67 | 3.95</td>
<td align="center">67.30 | 2.37</td>
</tr>
<tr>
<td align="left">MMLU-Pro</td>
<td align="center">74.21</td>
<td align="center">77.55</td>
<td align="center">74.79 | 2.36</td>
<td align="center">75.31 | 4.43</td>
<td align="center">76.59 | 2.62</td>
</tr>
<tr>
<td align="left">C-EVAL</td>
<td align="center">88.12</td>
<td align="center">87.54</td>
<td align="center">85.21 | 1.90</td>
<td align="center">86.93 | 2.71</td>
<td align="center">86.71 | 1.75</td>
</tr>
<tr>
<td align="left">PHYBench</td>
<td align="center">29.84</td>
<td align="center">27.67</td>
<td align="center">30.06 | 2.70</td>
<td align="center">26.04 | 4.10</td>
<td align="center">28.23 | 2.66</td>
</tr>
<tr>
<td align="left">TriviaQA</td>
<td align="center">65.61</td>
<td align="center">69.76</td>
<td align="center">66.88 | 1.94</td>
<td align="center">72.55 | 4.30</td>
<td align="center">72.93 | 2.92</td>
</tr>
<tr>
<td colspan="6" align="center"><b>Reasoning</b></td>
</tr>
<tr>
<td align="left">BIG-Bench Hard</td>
<td align="center">85.54</td>
<td align="center">89.36</td>
<td align="center">86.75 | 2.66</td>
<td align="center">87.82 | 5.61</td>
<td align="center">88.69 | 3.28</td>
</tr>
<tr>
<td align="left">BIG-Bench Extra Hard</td>
<td align="center">37.80</td>
<td align="center">23.24</td>
<td align="center">27.86 | 4.60</td>
<td align="center">33.51 | 5.04</td>
<td align="center">35.77 | 3.17</td>
</tr>
<tr>
<td align="left">bbh-zh</td>
<td align="center">86.18</td>
<td align="center">75.09</td>
<td align="center">87.52 | 3.21</td>
<td align="center">82.55 | 5.78</td>
<td align="center">86.23 | 3.77</td>
</tr>
<tr>
<td align="left">MuSR</td>
<td align="center">79.15</td>
<td align="center">82.72</td>
<td align="center">80.48 | 1.70</td>
<td align="center">80.10 | 2.90</td>
<td align="center">79.84 | 1.85</td>
</tr>
<tr>
<td align="left">ZebraLogic</td>
<td align="center">90.97</td>
<td align="center">87.60</td>
<td align="center">82.30 | 2.74</td>
<td align="center">84.20 | 5.80</td>
<td align="center">88.90 | 3.26</td>
</tr>
<tr>
<td align="left">PrOntoQA</td>
<td align="center">97.12</td>
<td align="center">97.88</td>
<td align="center">96.50 | 2.64</td>
<td align="center">95.00 | 9.23</td>
<td align="center">97.00 | 5.73</td>
</tr>
<tr>
<td align="left">PIQA</td>
<td align="center">91.57</td>
<td align="center">91.95</td>
<td align="center">92.76 | 1.43</td>
<td align="center">92.44 | 2.38</td>
<td align="center">92.17 | 1.44</td>
</tr>
<tr>
<td align="left">OCNLI</td>
<td align="center">71.59</td>
<td align="center">65.36</td>
<td align="center">71.63 | 1.09</td>
<td align="center">72.17 | 1.83</td>
<td align="center">72.75 | 1.32</td>
</tr>
<tr>
<td align="left">HellaSwag</td>
<td align="center">86.31</td>
<td align="center">81.59</td>
<td align="center">84.97 | 1.26</td>
<td align="center">85.60 | 2.31</td>
<td align="center">85.31 | 1.51</td>
</tr>
<tr>
<td align="left">KOR-Bench</td>
<td align="center">69.2</td>
<td align="center">69.44</td>
<td align="center">63.04 | 3.44</td>
<td align="center">62.80 | 4.97</td>
<td align="center">65.12 | 2.77</td>
</tr>
<tr>
<td align="left">DROP</td>
<td align="center">87.57</td>
<td align="center">88.32</td>
<td align="center">87.90 | 2.26</td>
<td align="center">87.55 | 5.40</td>
<td align="center">87.86 | 2.53</td>
</tr>
<tr>
<td align="left">SQuAD 2.0</td>
<td align="center">89.51</td>
<td align="center">81.32</td>
<td align="center">90.00 | 3.10</td>
<td align="center">90.65 | 5.01</td>
<td align="center">90.80 | 3.90</td>
</tr>
<tr>
<td colspan="6" align="center"><b>Coding</b></td>
</tr>
<tr>
<td align="left">LiveCodeBench</td>
<td align="center">46.42</td>
<td align="center">52.48</td>
<td align="center">42.51 | 4.23</td>
<td align="center">44.05 | 6.48</td>
<td align="center">45.37 | 3.80</td>
</tr>
<tr>
<td align="left">CRUXEval-O</td>
<td align="center">86.75</td>
<td align="center">82.75</td>
<td align="center">85.12 | 3.21</td>
<td align="center">85.25 | 6.54</td>
<td align="center">87.50 | 3.80</td>
</tr>
<tr>
<td align="left">MBPP+</td>
<td align="center">78.21</td>
<td align="center">80.89</td>
<td align="center">79.37 | 4.02</td>
<td align="center">76.72 | 10.43</td>
<td align="center">77.25 | 5.96</td>
</tr>
<tr>
<td align="left">HumanEval+</td>
<td align="center">87.88</td>
<td align="center">87.58</td>
<td align="center">88.41 | 6.45</td>
<td align="center">89.63 | 13.81</td>
<td align="center">89.63 | 9.18</td>
</tr>
<tr>
<td align="left">MultiPL-E</td>
<td align="center">70.67</td>
<td align="center">65.76</td>
<td align="center">74.87 | 3.14</td>
<td align="center">70.89 | 7.77</td>
<td align="center">73.34 | 4.33</td>
</tr>
<tr>
<td align="left">BigCodeBench-Full</td>
<td align="center">41.49</td>
<td align="center">40.70</td>
<td align="center">41.58 | 3.33</td>
<td align="center">37.11 | 8.51</td>
<td align="center">39.21 | 4.70</td>
</tr>
<tr>
<td align="left">BIRD-SQL</td>
<td align="center">47.75</td>
<td align="center">47.49</td>
<td align="center">45.76 | 2.16</td>
<td align="center">42.18 | 5.09</td>
<td align="center">44.04 | 2.95</td>
</tr>
<tr>
<td align="left">Spider</td>
<td align="center">81.79</td>
<td align="center">80.58</td>
<td align="center">82.49 | 4.42</td>
<td align="center">79.18 | 8.74</td>
<td align="center">81.04 | 5.70</td>
</tr>
<tr>
<td colspan="6" align="center"><b>Math</b></td>
</tr>
<tr>
<td align="left">AIME 2025</td>
<td align="center">61.88</td>
<td align="center">55.89</td>
<td align="center">60.00 | 4.57</td>
<td align="center">63.33 | 5.36</td>
<td align="center">63.33 | 3.46</td>
</tr>
<tr>
<td align="left">OlympiadBench</td>
<td align="center">77.59</td>
<td align="center">76.19</td>
<td align="center">74.07 | 3.70</td>
<td align="center">75.85 | 6.46</td>
<td align="center">76.59 | 3.81</td>
</tr>
<tr>
<td align="left">GSM-Plus</td>
<td align="center">89.41</td>
<td align="center">89.71</td>
<td align="center">89.74 | 2.68</td>
<td align="center">89.23 | 7.14</td>
<td align="center">89.69 | 3.83</td>
</tr>
<tr>
<td align="left">CMATH</td>
<td align="center">96.58</td>
<td align="center">96.52</td>
<td align="center">96.90 | 2.17</td>
<td align="center">96.54 | 4.84</td>
<td align="center">96.63 | 2.65</td>
</tr>
<tr>
<td align="left">Omni-MATH</td>
<td align="center">54.00</td>
<td align="center">53.00</td>
<td align="center">50.30 | 3.39</td>
<td align="center">52.30 | 6.01</td>
<td align="center">54.10 | 3.50</td>
</tr>
<tr>
<td colspan="6" align="center"><b>Agent & Alignment</b></td>
</tr>
<tr>
<td align="left">IFEval-strict-prompt</td>
<td align="center">83.73</td>
<td align="center">81.15</td>
<td align="center">82.62 | 1.47</td>
<td align="center">83.36 | 2.24</td>
<td align="center">83.55 | 1.41</td>
</tr>
<tr>
<td align="left">BFCL v3</td>
<td align="center">73.41</td>
<td align="center">67.69</td>
<td align="center">74.94 | 4.87</td>
<td align="center">74.86 | 9.24</td>
<td align="center">75.61 | 6.76</td>
</tr>
<tr>
<td align="left">Nexus FC</td>
<td align="center">49.93</td>
<td align="center">36.25</td>
<td align="center">50.45 | 5.53</td>
<td align="center">44.83 | 11.29</td>
<td align="center">47.65 | 7.38</td>
</tr>
</tbody>
</table>

---
## 🚀 Highlights
+ **Error-Correcting Editable:** Structural innovation of editable generation for dLLM
+ **Speedy vs Quality Mode:** The 100B flash model achieves ultra-fast inference under Speed Mode while remaining competitive across various tasks and under Quality Mode.
+ **Reinforcement Learning on 100B-scale dLLM:** Tailored algorithm and framework to enable reinforcement learning for large dLLM.

## 🗺️ What's Next

+ **Powerful Agentic/Tool Use Capability with LLaDA:** Next update will be equipped with powerful **Agentic** and long-distance tool-use capability.
+ **Extreme Editing:** Next update will feature stronger and more extensive editing capabilities, aimed at correcting more errors in parallel reasoning.
+ **Explore More Training Paradigms:** We want to explore more training paradigms than SFT and RL for dLLM.
---

## 📦 Model Variants
| Model ID | Description | Hugging Face Link |
| --- | --- | --- |
| `inclusionAI/LLaDA2.1-mini` | Instruction-tuned model, ready for downstream applications. | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.1-mini) |
| `inclusionAI/LLaDA2.1-flash` | Instruction-tuned model, ready for downstream applications. | [🤗 Model Card](https://huggingface.co/inclusionAI/LLaDA2.1-flash) |


---

## 🔍 Model Overview
**LLaDA2.1-flash** has the following specifications:

+ **Type**: Mixture-of-Experts (MoE) Diffusion Language Model
+ **Total Parameters (Non-Embedding)**: 100B
+ **Number of Layers**: 32
+ **Attention Heads**: 32
+ **Context Length**: 32,768 tokens
+ **Position Embedding**: Rotary (RoPE)
+ **Vocabulary Size**: 157,184

---

### 🤗 Hugging Face Transformers
Make sure you have `transformers` and its dependencies installed:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/LLaDA2.1-flash"
device = "auto"
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map=device,
)
model = model.to(torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = """Calculate 1+5-28*0.5-200=?"""
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)
generated_tokens = model.generate(
    inputs=input_ids,
    eos_early_stop=True,
    gen_length=512,
    block_length=32,
    threshold=0.5,
    editing_threshold=0,
    temperature=0.0,
)
generated_answer = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
)
print(generated_answer)
```
Multi-block Editing inference comming soon.
### Best Practices
To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**:
  We recommend the following general sampling parameters: `block_length=32`, `temperature=0.0`, `top_p=None` and `top_k=None`. We are currently exploring more diverse sampling configurations.

2. **Denoising Thresholds**:
  There are three denoising params: `threshold`, `editing_threshold` and `max_post_steps`. We recommend `threshold=0.7`, `editing_threshold=0.5` for **Quality Mode** and `threshold=0.5`, `editing_threshold=0.0` for **Speed Mode**. For both modes, we suggest setting max_post_steps to a value greater than 5. We recommend 16 as a balanced default, which was used for most of our internal testing.

Note: Low `threshold` may causes stuttering in trade-off for quick inference.

3. **Adequate Output Length**:
   We recommend using an output length of 16384 tokens for most scenarios.

---

## 🤖ModelScope
If you're in mainland China, we strongly recommend you to use our model from 🤖[ModelScope](https://modelscope.cn/models/inclusionAI/LLaDA2.1-flash)

---

## Deployment
### SGLang
SGLang enables dLLM inference either through offline batching or by launching an HTTP server for online requests. You can start the SGLang dLLM using the following commands:

``` bash
python3 -m sglang.launch_server \
	  --model-path inclusionAI/LLaDA2.1-flash \
	  --dllm-algorithm JointThreshold \
	  --tp-size 4 \
	  --trust-remote-code \
	  --mem-fraction-static 0.8 \
	  --max-running-requests 1 \
	  --attention-backend flashinfer	
```

### Enviroment Preparation
Pull Request (PR) has been submitted and merged to the SGLang community, please prepare the environment with the lateset version
___
## 🌐 License
This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🤝 Contact & Collaboration
For questions, collaborations, or feedback, please reach out via [Hugging Face](https://huggingface.co/inclusionAI/LLaDA2.1-flash) or open an issue in the [repository](https://github.com/inclusionAI).

👉 Join us in advancing open, efficient, and intelligent language models!

---

## Citation
```bibtex
@misc{bie2026llada21speedingtextdiffusion,
      title={LLaDA2.1: Speeding Up Text Diffusion via Token Editing}, 
      author={Tiwei Bie and Maosong Cao and Xiang Cao and Bingsen Chen and Fuyuan Chen and Kun Chen and Lun Du and Daozhuo Feng and Haibo Feng and Mingliang Gong and Zhuocheng Gong and Yanmei Gu and Jian Guan and Kaiyuan Guan and Hongliang He and Zenan Huang and Juyong Jiang and Zhonghui Jiang and Zhenzhong Lan and Chengxi Li and Jianguo Li and Zehuan Li and Huabin Liu and Lin Liu and Guoshan Lu and Yuan Lu and Yuxin Ma and Xingyu Mou and Zhenxuan Pan and Kaida Qiu and Yuji Ren and Jianfeng Tan and Yiding Tian and Zian Wang and Lanning Wei and Tao Wu and Yipeng Xing and Wentao Ye and Liangyu Zha and Tianze Zhang and Xiaolu Zhang and Junbo Zhao and Da Zheng and Hao Zhong and Wanli Zhong and Jun Zhou and Junlin Zhou and Liwang Zhu and Muzhi Zhu and Yihong Zhuang},
      year={2026},
      eprint={2602.08676},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.08676}, 
}
```
![--](https://ospo-insights.oss-cn-hangzhou.aliyuncs.com/iai-hf-models/LLaDA2.1-flash.gif)