---
language:
- zh
- en
tags:
- qwen
- audio-text-to-text
pipeline_tag: text-generation
inference: false
---

# Qwen-Audio-Chat


<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/audio_logo.jpg" width="400"/>
<p>
<br>

<p align="center">
        Qwen-Audio <a href="https://www.modelscope.cn/models/qwen/QWen-Audio/summary">🤖 <a> | <a href="https://huggingface.co/Qwen/Qwen-Audio">🤗</a>&nbsp ｜ Qwen-Audio-Chat <a href="https://www.modelscope.cn/models/qwen/QWen-Audio-Chat/summary">🤖 <a>| <a href="https://huggingface.co/Qwen/Qwen-Audio-Chat">🤗</a>&nbsp | &nbsp&nbsp Demo<a href="https://modelscope.cn/studios/qwen/Qwen-Audio-Chat-Demo/summary"> 🤖</a> | <a href="https://huggingface.co/spaces/Qwen/Qwen-Audio">🤗</a>&nbsp
<br>
&nbsp&nbsp<a href="https://qwen-audio.github.io/Qwen-Audio/">Homepage</a>&nbsp ｜ &nbsp<a href="http://arxiv.org/abs/2311.07919">Paper</a>
</p>
<br><br>

**Qwen-Audio** (Qwen Large Audio Language Model) is the multimodal version of the large model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen-Audio accepts diverse audio (human speech, natural sound, music and song) and text as inputs, outputs text. The contribution of Qwen-Audio include:

- **Fundamental audio models**: Qwen-Audio is a fundamental multi-task audio-language model that supports various tasks, languages, and audio types, serving as a universal audio understanding model. Building upon Qwen-Audio, we develop Qwen-Audio-Chat through instruction fine-tuning, enabling multi-turn dialogues and supporting diverse audio-oriented scenarios.
- **Multi-task learning framework for all types of audios**: To scale up audio-language pre-training, we address the challenge of variation in textual labels associated with different datasets by proposing a multi-task training framework, enabling knowledge sharing and avoiding one-to-many interference. Our model incorporates more than 30 tasks and extensive experiments show the model achieves strong performance.
- **Strong Performance**: Experimental results show that Qwen-Audio achieves impressive performance across diverse benchmark tasks without requiring any task-specific fine-tuning, surpassing its counterparts. Specifically, Qwen-Audio achieves state-of-the-art results on the test set of Aishell1, cochlscene, ClothoAQA, and VocalSound.
- **Flexible multi-run chat from audio and text input**: Qwen-Audio supports multiple-audio analysis, sound understading and reasoning, music appreciation, and tool usage for speech editing.

**Qwen-Audio** 是阿里云研发的大规模音频语言模型（Large Audio Language Model）。Qwen-Audio 可以以多种音频 (包括说话人语音、自然音、音乐、歌声）和文本作为输入，并以文本作为输出。Qwen-Audio 系列模型的特点包括：

- **音频基石模型**：Qwen-Audio是一个性能卓越的通用的音频理解模型，支持各种任务、语言和音频类型。在Qwen-Audio的基础上，我们通过指令微调开发了Qwen-Audio-Chat，支持多轮、多语言、多语言对话。Qwen-Audio和Qwen-Audio-Chat模型均已开源。
- **兼容多种复杂音频的多任务学习框架**：为了避免由于数据收集来源不同以及任务类型不同，带来的音频到文本的一对多的干扰问题，我们提出了一种多任务训练框架，实现相似任务的知识共享，并尽可能减少不同任务之间的干扰。通过提出的框架，Qwen-Audio可以容纳训练超过30多种不同的音频任务；
- **出色的性能**：Qwen-Audio在不需要任何任务特定的微调的情况下，在各种基准任务上取得了领先的结果。具体得，Qwen-Audio在Aishell1、cochlscene、ClothoAQA和VocalSound的测试集上都达到了SOTA；
- **支持多轮音频和文本对话，支持各种语音场景**：Qwen-Audio-Chat支持声音理解和推理、音乐欣赏、多音频分析、多轮音频-文本交错对话以及外部语音工具的使用(如语音编辑)。


We release Qwen-Audio and Qwen-Audio-Chat, which are pretrained model and Chat model respectively. For more details about Qwen-Audio, please refer to our [Github Repo](https://github.com/QwenLM/Qwen-Audio/tree/main). This repo is the one for Qwen-Audio-Chat.
<br>

目前，我们提供了Qwen-Audio和Qwen-Audio-Chat两个模型，分别为预训练模型和Chat模型。如果想了解更多关于信息，请点击[链接](https://github.com/QwenLM/Qwen-Audio/tree/main)查看Github仓库。本仓库为Qwen-Audio-Chat仓库。


## Requirements
* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users)
* FFmpeg
  <br>

## Quickstart
Below, we provide simple examples to show how to use Qwen-Audio with 🤗 Transformers.

Before running the code, make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.

```bash
pip install -r requirements.txt
```
Now you can start with Transformers. For more usage, please refer to [tutorial](https://github.com/QwenLM/Qwen-Audio/blob/main/TUTORIAL.md).

#### 🤗 Transformers

To use Qwen-Audio for the inference, all you need to do is to input a few lines of codes as demonstrated below. However, **please make sure that you are using the latest code.**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'audio': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac'}, # Either a local path or an url
    {'text': 'what does the person say?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

# 2nd dialogue turn
response, history = model.chat(tokenizer, 'Find the start time and end time of the word "middle classes"', history=history)
print(response)
# The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.
```


## License Agreement
Researchers and developers are free to use the codes and model weights of Qwen-Audio-Chat. We also allow its commercial use. Check our license at [LICENSE](https://github.com/QwenLM/Qwen-Audio/blob/main/LICENSE.txt) for more details.
<br>

## Citation
If you find our paper and code useful in your research, please consider giving a star and citation

```BibTeX
@article{Qwen-Audio,
  title={Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models},
  author={Chu, Yunfei and Xu, Jin and Zhou, Xiaohuan and Yang, Qian and Zhang, Shiliang and Yan, Zhijie  and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2311.07919},
  year={2023}
}
```
<br>

## Contact Us

If you are interested to leave a message to either our research team or product team, feel free to send an email to qianwen_opensource@alibabacloud.com.