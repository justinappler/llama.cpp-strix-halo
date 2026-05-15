---
license: other
license_name: glm-4
license_link: https://huggingface.co/THUDM/glm-4-9b/main/LICENSE
language:
- zh
- en
tags:
- glm
- chatglm
- thudm
inference: false
pipeline_tag: text-generation
---

# GLM-4-9B

Read this in [English](README_en.md)

**2024/08/12, 本仓库代码已更新并使用 `transformers>=4.44.0`, 请及时更新依赖。**

GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。 在语义、数学、推理、代码和知识等多方面的数据集测评中，
**GLM-4-9B** 及其人类偏好对齐的版本 **GLM-4-9B-Chat** 均表现出超越 Llama-3-8B 的卓越性能。除了能进行多轮对话，GLM-4-9B-Chat
还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K 上下文）等高级功能。本代模型增加了多语言支持，支持包括日语，韩语，德语在内的
26 种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的 **GLM-4-9B-Chat-1M** 模型和基于 GLM-4-9B 的多模态模型
GLM-4V-9B。**GLM-4V-9B** 具备 1120 * 1120 高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4V-9B
表现出超越 GPT-4-turbo-2024-04-09、Gemini
1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus 的卓越性能。

我们在一些典型任务上对 GLM-4-9B 基座模型进行了评测，结果如下：

| Model               |   MMLU   |  C-Eval  |   GPQA   |  GSM8K   |   MATH   | HumanEval |
|:--------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| Llama-3-8B          |   66.6   |   51.2   |    -     |   45.8   |    -     |     -     | 
| Llama-3-8B-Instruct |   68.4   |   51.3   |   34.2   |   79.6   |   30.0   |   62.2    |
| ChatGLM3-6B-Base    |   61.4   |   69.0   |    -     |   72.3   |   25.7   |     -     |
| GLM-4-9B            | **74.7** | **77.1** | **34.3** | **84.0** | **30.4** | **70.1**  |


更多推理代码和依赖信息，请访问我们的 [github](https://github.com/THUDM/GLM-4) 。

**本仓库是 GLM-4-9B 的基座版本，支持`8K`上下文长度。**

## 协议

GLM-4 模型的权重的使用则需要遵循 [LICENSE](LICENSE)。

## 引用

如果你觉得我们的工作有帮助的话，请考虑引用下列论文。

```
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, 
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```