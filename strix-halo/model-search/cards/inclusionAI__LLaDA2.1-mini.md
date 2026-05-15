---
license: apache-2.0
library_name: transformers
tags:
- dllm
- diffusion
- llm
- text_generation
---
# LLaDA2.1-mini

🚀 **LLaDA2.1-flash** is now live on **ZenmuxAI**! Try it via API 🛠️ or Chat 💬: https://zenmux.ai/inclusionai/llada2.1-flash

**LLaDA2.1-mini** is a diffusion language model of the LLaDA series featuring the editing enhancement. It significantly improves inference speed while delivering strong task performance.

<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*uOo8QKQMiBwAAAAAgNAAAAgAemJ7AQ/original" width="800" />
</div>


<div align="center">
  <img src="https://mdn.alipayobjects.com/huamei_qa8qxu/afts/img/A*biwvQpCmKjEAAAAAULAAAAgAemJ7AQ/original" width="800" />
</div>

---
## Model Performance

<table>
<thead>
<tr>
<th align="left"><b>Benchmark</b></th>
<th align="center"><b>Qwen3-8B<br>(no_think)</b><br><sub>(Score)</sub></th>
<th align="center"><b>Ling-mini-2.0</b><br><br><sub>(Score)</sub></th>
<th align="center"><b>LLaDA2.0-mini</b><br><br><sub>(Score | TPF)</sub></th>
<th align="center"><b>LLaDA2.1-mini<br>(S Mode)</b><br><sub>(Score | TPF)</sub></th>
<th align="center"><b>LLaDA2.1-mini<br>(Q Mode)</b><br><sub>(Score | TPF)</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><b>Average</b></td>
<td align="center">61.59</td>
<td align="center">64.72</td>
<td align="center">63.39 | 2.60</td>
<td align="center">62.07 | 5.34</td>
<td align="center">63.90 | 3.12</td>
</tr>
<tr><td colspan="6" align="center"><b>Knowledge</b></td></tr>
<tr>
<td align="left">GPQA</td>
<td align="center">48.01</td>
<td align="center">59.41</td>
<td align="center">47.76 | 2.73</td>
<td align="center">48.36 | 3.62</td>
<td align="center">53.28 | 2.12</td>
</tr>
<tr>
<td align="left">MMLU-Pro</td>
<td align="center">65.83</td>
<td align="center">67.18</td>
<td align="center">64.27 | 2.15</td>
<td align="center">63.42 | 4.22</td>
<td align="center">64.84 | 2.41</td>
</tr>
<tr>
<td align="left">C-EVAL</td>
<td align="center">80.6</td>
<td align="center">82.17</td>
<td align="center">81.80 | 1.78</td>
<td align="center">78.40 | 3.39</td>
<td align="center">78.59 | 1.91</td>
</tr>
<tr>
<td align="left">PHYBench</td>
<td align="center">9.76</td>
<td align="center">14.59</td>
<td align="center">11.70 | 2.48</td>
<td align="center">12.75 | 4.41</td>
<td align="center">13.05 | 2.52</td>
</tr>
<tr>
<td align="left">TriviaQA</td>
<td align="center">52.51</td>
<td align="center">55.63</td>
<td align="center">51.33 | 1.54</td>
<td align="center">53.33 | 3.21</td>
<td align="center">54.24 | 2.02</td>
</tr>
<tr><td colspan="6" align="center"><b>Reasoning</b></td></tr>
<tr>
<td align="left">BIG-Bench Hard</td>
<td align="center">79.48</td>
<td align="center">83.70</td>
<td align="center">78.21 | 2.36</td>
<td align="center">78.42 | 5.02</td>
<td align="center">80.58 | 2.86</td>
</tr>
<tr>
<td align="left">BIG-Bench Extra Hard</td>
<td align="center">18.27</td>
<td align="center">14.81</td>
<td align="center">16.47 | 2.03</td>
<td align="center">15.30 | 3.19</td>
<td align="center">15.78 | 1.66</td>
</tr>
<tr>
<td align="left">bbh-zh</td>
<td align="center">80.09</td>
<td align="center">66.11</td>
<td align="center">75.75 | 2.77</td>
<td align="center">67.65 | 3.89</td>
<td align="center">70.40 | 2.35</td>
</tr>
<tr>
<td align="left">MuSR</td>
<td align="center">70.02</td>
<td align="center">71.36</td>
<td align="center">71.48 | 1.45</td>
<td align="center">70.43 | 2.48</td>
<td align="center">71.89 | 1.56</td>
</tr>
<tr>
<td align="left">ZebraLogic</td>
<td align="center">37.48</td>
<td align="center">79.85</td>
<td align="center">64.20 | 2.30</td>
<td align="center">68.50 | 5.38</td>
<td align="center">77.10 | 2.93</td>
</tr>
<tr>
<td align="left">PrOntoQA</td>
<td align="center">93.12</td>
<td align="center">96.06</td>
<td align="center">86.00 | 2.36</td>
<td align="center">87.50 | 4.86</td>
<td align="center">84.50 | 2.73</td>
</tr>
<tr>
<td align="left">PIQA</td>
<td align="center">88.30</td>
<td align="center">87.54</td>
<td align="center">86.51 | 1.45</td>
<td align="center">84.87 | 2.59</td>
<td align="center">86.89 | 1.45</td>
</tr>
<tr>
<td align="left">OCNLI</td>
<td align="center">61.49</td>
<td align="center">60.17</td>
<td align="center">64.51 | 4.06</td>
<td align="center">61.02 | 1.78</td>
<td align="center">61.59 | 1.23</td>
</tr>
<tr>
<td align="left">HellaSwag</td>
<td align="center">79.56</td>
<td align="center">69.02</td>
<td align="center">79.01 | 1.50</td>
<td align="center">75.71 | 2.39</td>
<td align="center">76.19 | 1.49</td>
</tr>
<tr>
<td align="left">KOR-Bench</td>
<td align="center">54.96</td>
<td align="center">63.2</td>
<td align="center">49.92 | 2.45</td>
<td align="center">46.64 | 4.28</td>
<td align="center">48.00 | 2.35</td>
</tr>
<tr>
<td align="left">DROP</td>
<td align="center">84.56</td>
<td align="center">78.80</td>
<td align="center">81.89 | 2.02</td>
<td align="center">81.55 | 5.84</td>
<td align="center">82.37 | 2.87</td>
</tr>
<tr>
<td align="left">SQuAD 2.0</td>
<td align="center">85.21</td>
<td align="center">75.56</td>
<td align="center">86.50 | 2.47</td>
<td align="center">84.51 | 4.33</td>
<td align="center">85.13 | 3.09</td>
</tr>
<tr><td colspan="6" align="center"><b>Coding</b></td></tr>
<tr>
<td align="left">LiveCodeBench</td>
<td align="center">26.76</td>
<td align="center">42.29</td>
<td align="center">31.83 | 3.34</td>
<td align="center">28.85 | 6.42</td>
<td align="center">30.40 | 3.63</td>
</tr>
<tr>
<td align="left">CRUXEval-O</td>
<td align="center">74.06</td>
<td align="center">76.12</td>
<td align="center">71.62 | 2.78</td>
<td align="center">70.62 | 5.85</td>
<td align="center">73.75 | 3.35</td>
</tr>
<tr>
<td align="left">MBPP+</td>
<td align="center">72.69</td>
<td align="center">77.25</td>
<td align="center">78.24 | 3.43</td>
<td align="center">73.28 | 10.59</td>
<td align="center">74.07 | 6.30</td>
</tr>
<tr>
<td align="left">HumanEval+</td>
<td align="center">79.5</td>
<td align="center">80.03</td>
<td align="center">81.40 | 5.16</td>
<td align="center">80.49 | 12.32</td>
<td align="center">82.93 | 7.77</td>
</tr>
<tr>
<td align="left">MultiPL-E</td>
<td align="center">61.70</td>
<td align="center">67.09</td>
<td align="center">67.46 | 2.78</td>
<td align="center">64.16 | 7.23</td>
<td align="center">67.17 | 4.01</td>
</tr>
<tr>
<td align="left">BigCodeBench-Full</td>
<td align="center">36.05</td>
<td align="center">35.00</td>
<td align="center">32.89 | 2.87</td>
<td align="center">30.18 | 7.33</td>
<td align="center">34.39 | 4.09</td>
</tr>
<tr>
<td align="left">BIRD-SQL</td>
<td align="center">36.11</td>
<td align="center">39.67</td>
<td align="center">39.34 | 1.96</td>
<td align="center">37.32 | 4.48</td>
<td align="center">38.40 | 2.42</td>
</tr>
<tr>
<td align="left">Spider</td>
<td align="center">72.80</td>
<td align="center">76.43</td>
<td align="center">76.76 | 3.93</td>
<td align="center">75.78 | 7.98</td>
<td align="center">77.55 | 5.48</td>
</tr>
<tr><td colspan="6" align="center"><b>Math</b></td></tr>
<tr>
<td align="left">AIME 2025</td>
<td align="center">22.08</td>
<td align="center">47.66</td>
<td align="center">36.67 | 2.41</td>
<td align="center">36.67 | 6.34</td>
<td align="center">43.33 | 3.29</td>
</tr>
<tr>
<td align="left">OlympiadBench</td>
<td align="center">55.33</td>
<td align="center">72.30</td>
<td align="center">67.70 | 2.63</td>
<td align="center">64.30 | 7.08</td>
<td align="center">66.67 | 3.99</td>
</tr>
<tr>
<td align="left">GSM-Plus</td>
<td align="center">85.56</td>
<td align="center">87.18</td>
<td align="center">86.50 | 2.41</td>
<td align="center">85.88 | 6.82</td>
<td align="center">86.55 | 3.69</td>
</tr>
<tr>
<td align="left">CMATH</td>
<td align="center">95.42</td>
<td align="center">96.40</td>
<td align="center">95.72 | 1.98</td>
<td align="center">95.63 | 4.94</td>
<td align="center">94.99 | 2.56</td>
</tr>
<tr>
<td align="left">Omni-MATH</td>
<td align="center">33.20</td>
<td align="center">48.80</td>
<td align="center">41.70 | 2.57</td>
<td align="center">41.70 | 6.41</td>
<td align="center">43.60 | 3.56</td>
</tr>
<tr><td colspan="6" align="center"><b>Agent & Alignment</b></td></tr>
<tr>
<td align="left">IFEval-strict-prompt</td>
<td align="center">84.29</td>
<td align="center">76.16</td>
<td align="center">80.78 | 1.24</td>
<td align="center">81.33 | 1.83</td>
<td align="center">83.18 | 1.25</td>
</tr>
<tr>
<td align="left">BFCL v3</td>
<td align="center">70.12</td>
<td align="center">53.75</td>
<td align="center">70.72 | 4.26</td>
<td align="center">72.06 | 7.39</td>
<td align="center">73.61 | 5.14</td>
</tr>
<tr>
<td align="left">Nexus FC</td>
<td align="center">37.71</td>
<td align="center">34.38</td>
<td align="center">35.18 | 4.06</td>
<td align="center">31.59 | 8.27</td>
<td align="center">33.69 | 4.91</td>
</tr>
</tbody>
</table>

---

## 🚀 Highlights
+ **Error-Correcting Editable:** Structural innovation of editable generation for dLLM
+ **Speedy vs Quality Mode:** The 16B mini model achieves ultra-fast inference under Speed Mode while remaining competitive across various tasks and under Quality Mode.
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
**LLaDA2.1-mini** has the following specifications:

+ **Type**: Mixture-of-Experts (MoE) Diffusion Language Model
+ **Total Parameters (Non-Embedding)**: 16B
+ **Number of Layers**: 20
+ **Attention Heads**: 16
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

model_path = "/path/to/LLaDA2.1-mini"
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
If you're in mainland China, we strongly recommend you to use our model from 🤖[ModelScope](https://modelscope.cn/models/inclusionAI/LLaDA2.1-mini)

---

## Deployment
### SGLang
SGLang enables dLLM inference either through offline batching or by launching an HTTP server for online requests. You can start the SGLang dLLM using the following commands:

``` bash
python3 -m sglang.launch_server \
	  --model-path inclusionAI/LLaDA2.1-mini \
	  --dllm-algorithm JointThreshold \
	  --tp-size 1 \
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
For questions, collaborations, or feedback, please reach out via [Hugging Face](https://huggingface.co/inclusionAI/LLaDA2.1-mini) or open an issue in the [repository](https://github.com/inclusionAI).

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

![--](https://ospo-insights.oss-cn-hangzhou.aliyuncs.com/iai-hf-models/LLaDA2.1-mini.gif)
