---
library_name: transformers
license: other
license_name: nvidia-nemotron-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
pipeline_tag: text-generation
language:
- en
- es
- fr
- de
- ja
- it
- zh
- ar
- he
- hi
- ko
- cs
- da
- nl
- fi
- pl
- pt
- th
- sv
- ru
tags:
- nvidia
- pytorch
- nemotron-3
- latent-moe
- mtp
track_downloads: true
---

# NVIDIA-Nemotron-3-Super-120B-A12B-Base

<div align="center" style="line-height: 1;">
<a href="https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/📝Paper-Read Now!-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets" target="_blank" style="margin: 2px;">
    <img alt="Pre-Training Datasets" src="https://img.shields.io/badge/🗄️_Pre--Training_Datasets-Available_Here-76B900?logoColor=white" style="display: inline-block; vertical-align: middle;"/>
</a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://developer.nvidia.com/nemotron" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/🏠Nemotron Developer Page-Learn More Here!-536af5?color=76B900&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
<a href="https://discord.gg/9xpKQtVvrk" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-NVIDIA%20AI%20Developer-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-NVIDIA Nemotron Open Model License-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

![](./accuracy_chart.png)

## Model Overview

**Model Developer:** NVIDIA Corporation

**Model Dates:**

December 2025 - January 2026

**Data Freshness:**

* The post-training data has a cutoff date of February 2026.
* The pre-training data has a cutoff date of December 2025.

## Description

**Nemotron-3-Super-120B-A12B-Base** is a base large language model (LLM) trained from scratch by NVIDIA, with next token prediction loss. It provides a good starting platform for further training, including instruction follow, and coding.

The model employs a hybrid **Latent Mixture-of-Experts (LatentMoE)** architecture, utilizing interleaved Mamba-2 and MoE layers, along with select Attention layers. Distinct from the Nano model, the Super model incorporates **Multi-Token Prediction (MTP)** layers for faster text generation and improved quality, and it is trained using **NVFP4** quantization to maximize compute efficiency. The model has **12B active parameters** and **120B parameters in total**.

The supported languages include: English, Spanish, French, German, Japanese, Italian, Chinese, Arabic, Hebrew, Hindi, Korean, Czech, Danish, Dutch, Finnish, Polish, Portuguese, Thai, Swedish, and Russian.

This model is ready for commercial use.

### What is Nemotron?

NVIDIA Nemotron™ is a family of open models with open weights, training data, and recipes, delivering leading efficiency and accuracy for building specialized AI agents.

To get started, you can use [our quickstart guide](#quick-start-guide) below.

## License/Terms of Use

Use of these model weights is governed by the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license//).

## Base Benchmarks

Comparison of Ling-flash-base-2.0, GLM-4.5-Air-Base, and Nemotron Super 120B-A12B Base. Best results per row in **bold**.

| Task | Metric | N-3-Super 120B-A12B-Base | Ling-flash base-2.0 | GLM-4.5 Air-Base |
|------|--------|:------------------------:|:-------------------:|:----------------:|
| **General Knowledge** | | | | |
| MMLU | 5-shot, acc | **86.01** | 81.00 | 81.00 |
| MMLU-Pro | 5-shot, CoT EM | **75.65** | 62.10 | 58.20 |
| AGIEval-En | 3/5-shot, CoT EM | **77.92** | 61.70 | 62.40 |
| GPQA-Diamond | 5-shot, CoT EM | **60.00** | 36.00 | 23.20 |
| **Math** | | | | |
| GSM8K | 8-shot, EM | 90.67 | **90.75** | 82.60 |
| MATH | 4-shot, EM | **84.84** | 63.80 | 50.36 |
| MATH Level 5 | 4-shot, EM | **70.00** | 39.80 | 26.30 |
| AIME 2024 | pass@32 | **53.33** | 30.00 | 20.00 |
| **Code** | | | | |
| HumanEval | 0-shot, pass@1 n=32 | **79.40** | 70.10 | 76.30 |
| MBPP-Sanitized | 3-shot, pass@1 n=32 | **78.38** | 77.30 | 77.50 |
| **Commonsense Understanding** | | | | |
| ARC-Challenge | 25-shot, acc_norm | **96.08** | 94.80 | 93.90 |
| HellaSwag | 10-shot, acc_norm | **88.97** | 84.69 | 87.70 |
| OpenBookQA | 0-shot, acc_norm | **50.20** | 47.00 | 48.60 |
| PIQA | 0-shot, acc_norm | **85.47** | 84.00 | 84.22 |
| WinoGrande | 5-shot, acc | 78.93 | 78.37 | **83.82** |
| **Reading Comprehension** | | | | |
| RACE | 0-shot, acc | **91.00** | 90.10 | 89.50 |
| **Multilingual** | | | | |
| MMLU Global Lite | 5-shot, avg | **85.72** | 74.94 | 79.25 |
| MGSM | 8-shot, avg | **87.47** | 82.73 | 80.33 |
| **Long Context** | | | | |
| RULER 64K | 0-shot | **93.17** | — | 81.58 |
| RULER 128K | 0-shot | **89.00** | 57.56 | 63.62 |
| RULER 256K | 0-shot | **86.18** | — | — |
| RULER 512K | 0-shot | **82.16** | — | — |
| RULER 1M | 0-shot | **74.39** | — | — |

All evaluation results were collected via [Nemo Evaluator SDK](https://github.com/NVIDIA-NeMo/Evaluator) and NVIDIA's open source container of [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), unless otherwise stated. For reproducibility purposes, more details on the evaluation settings can be found in the [Nemo Evaluator SDK examples folder](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/nemotron/nemotron-3-super) and the [reproducibility tutorial for Nemotron 3 Super](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/nemotron-3-super/reproducibility.md). The open source container on LM Evaluation Harness packaged via NVIDIA's Nemo Evaluator SDK used for evaluations can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)

### Deployment Geography: Global

### Use Case

This model is intended for developers and researchers building LLMs.

### Release Date

NGC - 03/04/2026 via [NGC]()
Hugging Face - 03/04/2026 via [Hugging Face]()

## Reference(s)

* [NVIDIA Nemotron 3 model family on Hugging Face](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3)
* [NVIDIA Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

## Model Architecture

- **Architecture Type:** Mamba2-Transformer Hybrid Latent Mixture of Experts (LatentMoE) with Multi-Token Prediction (MTP)
- **Network Architecture:** Nemotron Hybrid LatentMoE
- **Number of model parameters:** 120B Total / 12B Active

## Model Design

The model utilizes the **LatentMoE** architecture, where tokens are projected into a smaller latent dimension for expert routing and computation, improving accuracy per byte. The Super model is trained using **NVFP4** (weight, activation, and gradient tensors are quantized to NVFP4) to maximize throughput on supported hardware. The model includes **Multi-Token Prediction (MTP)** layers, which predict multiple future tokens to provide richer training signals and enable faster inference via speculative decoding.

## Training Methodology

Stage 1: Pre-Training

* [NVIDIA-Nemotron-3-Super-120B-A12B-Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base) model was pre-trained for over 25T tokens using crawled and synthetic code, math, science, and general knowledge data. Training leveraged NVFP4 quantization for efficiency. All datasets are disclosed in the [Training and Evaluation Datasets](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B#training-and-evaluation-datasets) section of this document. Major portions of the pre-training corpus are released in the [Nemotron-Pre-Training-Datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) collection.
* Software used for pre-training: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

NVIDIA-Nemotron-3-Super-120B-A12B-Base model is a result of the above work.

The end-to-end training recipe is available in the [NVIDIA Nemotron Developer Repository](https://github.com/NVIDIA-NeMo/Nemotron). Evaluation results can be replicated using the [NeMo Evaluator SDK](https://github.com/NVIDIA-NeMo/Evaluator). [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) is one of the libraries used to prepare the pre and post training datasets. More details on the datasets and synthetic data generation methods can be found in the technical report [NVIDIA Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf).

## Input

- **Input Type(s):** Text
- **Input Format(s):** String
- **Input Parameters:** One-Dimensional (1D): Sequences
- **Other Properties Related to Input:** Maximum context length up to 1M tokens. Supported languages include: English, Spanish, French, German, Japanese, Italian, Chinese, Arabic, Hebrew, Hindi, Korean, Czech, Danish, Dutch, Finnish, Polish, Portuguese, Thai, Swedish, and Russian.

## Output

- **Output Type(s):** Text
- **Output Format:** String
- **Output Parameters:** One-Dimensional (1D): Sequences
- **Other Properties Related to Output:** Maximum context length up to 1M tokens

Our AI models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

- Runtime Engine(s): NeMo 25.11.01
- Supported Hardware Microarchitecture Compatibility: NVIDIA Ampere - A100; NVIDIA Blackwell; NVIDIA Hopper - H100-80GB
- Operating System(s): Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s)

* v1.0 - Base

## Training and Evaluation Datasets:

# Training

**Data Modality:** Text
**The total size:** 15,087,602,908,990
**Total number of datasets:** 153
**Dataset partition:** *Training [100%], testing [0%], validation [0%]*
**Time period for training data collection:** 2013 to February 24, 2026
**Time period for testing data collection:** 2013 to February 24, 2026
**Time period for validation data collection:** 2013 to February 24, 2026
**Data Collection Method by dataset:** Hybrid: Automated, Human, Synthetic
**Labeling Method by dataset:** Hybrid: Automated, Human, Synthetic

NVIDIA-Nemotron-3-Super-120B-A12B-Base is pre-trained on a large corpus of high-quality curated and synthetically-generated data. It is trained in the English language, as well as 19 other languages and 43 programming languages. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracy. The model was trained for approximately 25T tokens.

More details on the datasets and synthetic data generation methods can be found in the technical report _[**_NVIDIA Nemotron 3 Super_**](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)_.

<details>
<summary><strong>Click to explore the full dataset catalogue used for training</strong></summary>

#### **Base Pre-Training Corpus (Nemotron 3 Foundation)**

The foundation of the model is trained on the **Nemotron-3-Nano** corpus, comprising the following collections:

| Dataset Collection | Token Counts | Description |
| :--- | :--- | :--- |
| **Nemotron-CC-v2** & **v2.1** | 9.13T | A massive collection of English web data filtered from Common Crawl, including 2.5T+ tokens of new organic, translated, and synthetically rephrased content. |
| **Nemotron-CC-Code-v1** | 427.9B | High-quality code tokens extracted from Common Crawl using the Lynx + LLM pipeline to preserve structure and equations. |
| **Nemotron-Pretraining-Code-v1** & **v2** | 1.09T | Curated GitHub code references with multi-stage filtering, deduplication, and large-scale synthetic code data. |
| **Nemotron-CC-Math-v1** | 133.3B | High-quality math pre-training dataset preserving LaTeX formatting and mathematical structures. |
| **Nemotron-Pretraining-Specialized-v1** | 336.4B | Synthetic datasets targeting specialized domains such as STEM reasoning and scientific coding. |

### Public Datasets

| Dataset | Collection Period |
| :---- | :---- |
| [GSM8K](https://github.com/openai/grade-school-math) | 4/23/2025 |
| [CC-NEWS](https://commoncrawl.org/blog/news-dataset-available) | 4/23/2025 |
| [Common Crawl](https://commoncrawl.org/) | 4/23/2025 |
| [Wikimedia](https://dumps.wikimedia.org/) | 4/23/2025 |
| [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | 4/23/2025 |
| [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k) | 4/23/2025 |
| [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 4/23/2025 |
| [APIGen Function-Calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 4/23/2025 |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 4/23/2025 |
| [Open Textbook Library \- CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) and [OpenStax \- CC BY-SA subset](https://openstax.org/) | 4/23/2025 |
| [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb), [tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k), [PRM800K](https://github.com/openai/prm800k), and [SciBench](https://github.com/mandyyyyii/scibench) | 4/23/2025 |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 4/23/2025 |
| [Court Listener](https://www.courtlistener.com/help/api/bulk-data/) | Legacy Download |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | Legacy Download |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Legacy Download |
| [BioRxiv](https://www.biorxiv.org/tdm) | Legacy Download |
| [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) | Legacy Download |
| [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/) | Legacy Download |
| [Stack Exchange Data Dump](https://archive.org/details/stackexchange) | Legacy Download |
| [PubMed Abstracts](https://github.com/thoppe/The-Pile-PubMed) | Legacy Download |
| [NIH ExPorter](https://exporter.nih.gov/ExPORTER_Catalog.aspx) | Legacy Download |
| [arXiv](https://info.arxiv.org/help/bulk_data/index.html) | Legacy Download |
| [BigScience Workshop Datasets](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#datasets) | Legacy Download |
| [Reddit Dataset](https://files.pushshift.io/reddit/) | Legacy Download |
| [SEC's Electronic Data Gathering, Analysis, and Retrieval (EDGAR)](https://www.sec.gov/search-filings) | Legacy Download |
| [Advanced Mathematical Problem Solving](https://github.com/hendrycks/math?tab=readme-ov-file) | Legacy Download |
| [MathPile](https://github.com/GAIR-NLP/MathPile/) | Legacy Download |
| [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Legacy Download |
| [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/) | Legacy Download |
| [FLAN](https://github.com/google-research/FLAN) | Legacy Download |
| [Advanced Reasoning Benchmark](https://github.com/TheDuckAI/arb) | Legacy Download |
| [SciBench](https://github.com/mandyyyyii/scibench) | Legacy Download |
| [WikiTableQuestions](https://huggingface.co/datasets/wikitablequestions) | Legacy Download |
| [FinQA](https://finqasite.github.io/) | Legacy Download |
| [Riddles](https://github.com/crawsome/riddles) | Legacy Download |
| [Problems in Elementary Mathematics for Home Study](https://archive.org/details/AntonovVygodskyNikitinSankinProblemsInElementaryMathematicsForHomeStudyMir1982) | Legacy Download |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Legacy Download |
| [Cosmos QA](https://huggingface.co/datasets/allenai/cosmos_qa) | Legacy Download |
| [MCTest](https://huggingface.co/datasets/sagnikrayc/mctest) | Legacy Download |
| [AI2's Reasoning Challenge](https://huggingface.co/datasets/ai2_arc) | Legacy Download |
| [OpenBookQA](https://github.com/allenai/OpenBookQA) | Legacy Download |
| [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | Legacy Download |
| [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101) | Legacy Download |
| [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | Legacy Download |
| [The Common Pile v0.1](https://huggingface.co/common-pile) | Legacy Download |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | Legacy Download |
| [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath) | Legacy Download |
| [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath) | Legacy Download |
| [MultiverseMathHard](https://huggingface.co/datasets/Nexusflow/MultiverseMathHard) | 10/2/2025 |
| [News Commentary](https://opus.nlpl.eu/News-Commentary.php) | 10/2/2025 |
| [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | 10/2/2025 |
| [finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | 10/2/2025 |
| [HotpotQA](https://huggingface.co/hotpot_qa/datasets) | 10/2/2025 |
| [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) | 10/2/2025 |
| [NLTK Words Lists](https://www.nltk.org/nltk_data/) | 10/2/2025 |

### **Crawled and Scraped from Online Sources by NVIDIA**

The English Common Crawl data was downloaded from the Common Crawl Foundation (see their FAQ for details on their crawling) and includes the snapshots CC-MAIN-2013-20 through CC-MAIN-2025-13. The data was subsequently deduplicated and filtered in various ways described in the Nemotron-CC paper. Additionally, we extracted data for fifteen languages from the following three Common Crawl snapshots: CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18. The fifteen languages included were Arabic, Chinese, Danish, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, and Thai. As we did not have reliable multilingual model-based quality classifiers available, we applied just heuristic filtering instead—similar to what we did for lower quality English data in the Nemotron-CC pipeline, but selectively removing some filters for some languages that did not work well. Deduplication was done in the same way as for Nemotron-CC.

The GitHub Crawl was collected using the GitHub REST API and the Amazon S3 API. Each crawl was operated in accordance with the rate limits set by its respective source, either GitHub or S3. We collect raw source code and subsequently remove any having a license which does not exist in our permissive-license set (for additional details, refer to the [technical report](https://arxiv.org/abs/2512.20848)).

| Dataset | Modality | Dataset Size | Collection Period | Collecting Organisation |
| :---- | :---- | :---- | :---- | :---- |
| English Common Crawl | Text | 3.36T | 4/8/2025 | NVIDIA Advanced Deep Learning Research |
| English Common Crawl 1.1 | Text | Not disclosed | 10/2/2025 | NVIDIA Advanced Deep Learning Research |
| Multilingual Common Crawl | Text | 812.7B | 5/1/2025 | NVIDIA Advanced Deep Learning Research |
| GitHub Crawl | Text | 747.4B | 4/29/2025 | NVIDIA Advanced Deep Learning Research |

## Private Non-publicly Accessible Datasets of Third Parties

| Dataset | Model(s) used |
|---------|---------------|
| Global Regulation | Unknown |
| TAUS Translation Memory | Unknown |
| Scale HLE | Unknown |
| HackerRank Coding | Unknown |
| RL data for Search | Gemini 3; GPT-5 |

## Private Non-publicly Accessible Datasets by NVIDIA

| Dataset | Model(s) used |
|---------|---------------|
| Simple Minesweeper | \- |
| Simple Sudoku | \- |
| Multitool Typewriter Hard | \- |
| Machine Translation of News Commentary and TAUS Translation Memory | \- |
| Machine Translation of STEM - | [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| Competitive Coding RL data from Nemotron Cascade | \- |
| Long context RL | \- |
| Single-step SWE RL for patch generation | \- |
| OpenHands SWE | \- |

## NVIDIA-Sourced Synthetic Datasets

| Dataset | Modality | Dataset Size | Seed Dataset | Model(s) used for generation |
| :---- | :---- | :---- | :---- | :---- |
| Nemotron-Pretraining-Formal-Logic | Text | 128M | [Nemotron Personas](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Nemotron-Pretraining-Economics | Text | 73.4M | - | [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Nemotron-Pretraining-Multiple-Choice | Text | 1.6B | [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Nemotron-Pretraining-Code-Concepts | Text | 7.3B | - | [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Nemotron-Pretraining-Unconditional-Algorithmic | Text | 196.5M | - | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Synthetic Tasks from DeepSeek-V3 and Qwen3-235B-A22B | Text | 6.7B | train splits of Into the Unknown; AI2 ARC (AI2 Reasoning Challenge); BLiMP (Benchmark of Linguistic Minimal Pairs); CommonSenseQA; GLUE; HeadQA; Hendrycks Ethics; Memo Trap; modus-tollens; NeQA; pattern-matching-suppression; mastermind_24_mcq_random; mastermind_24_mcq_close; quote-repetition; redefine-math; Repetitive Algebra; sig-figs; MMLU-Pro; MC-TACO; MedConceptsQA; MMLU_dataset; OpenbooksQA; PIQA (Physical Interaction Question Answering); SocialIQA; SuperGLUE; tinyAI2_arc; tinyMMLU; tinyWinogrande; TruthfulQA; WebQuestions; Winogrande; GPQA; MBPP | [DeepSeek v3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Synthetic Art of Problem Solving from DeepSeek-R1 | Text | 40B | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Moral Stories and Social Chemistry from Mixtral-8x22B-v0.1 | Text | 327M | [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101); [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| Synthetic Social Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 83.6M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic Health Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 9.7M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic STEM seeded with OpenStax, Open Textbook Library, and GSM8K from DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-0324, and Qwen2.5-72B | Text | 175M | [OpenStax \- CC BY-SA subset](https://openstax.org/); [GSM8K](https://github.com/openai/grade-school-math); [Open Textbook Library \- CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| [Nemotron-PrismMath](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath) | Text | 4.6B | [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified); [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | [Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct); [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| Synthetic Question Answering Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text | 350M | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic Rephrased [Math Data from Common Crawl](https://huggingface.co/datasets/nvidia/Nemotron-MIND) from phi-4 | Text | 73B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Math Data from Common Crawl 4plus | Text | 52.3B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Math Data from Common Crawl 3 | Text | 80.9B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from DeepSeek-V3 and DeepSeek-V3-0324 | Text | 4.0B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from Qwen3-30B-A3B | Text | 4.2B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Art of Problem Solving from Qwen2.5-32B-Instruct, Qwen2.5-Math-72B, Qwen2.5-Math-7B, and Qwen2.5-72B-Instruct | Text |  | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); [GSM8K](https://github.com/openai/grade-school-math); [PRM800K](https://github.com/openai/prm800k) | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [Qwen2.5-Math-72B](https://huggingface.co/Qwen/Qwen2.5-Math-72B); [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B); [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic MMLU Auxiliary Train from DeepSeek-R1 | Text | 0.5B | [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Long Context Continued Post-Training Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text |  | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic Common Crawl from Qwen3-30B-A3B and Mistral-Nemo-12B-Instruct | Text | 415.8B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct) |
| Synthetic Multilingual Data from Common Crawl from Qwen3-30B-A3B | Text |  | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Multilingual Data from Wikimedia from Qwen3-30B-A3B | Text |  | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Math Data from Wikimedia from Nemotron-4-340B-Instruct | Text |  | \- | [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) |
| Synthetic Common Crawl Code from phi-4 | Text | 427.9B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Scientific Coding from Qwen3-235B-A22B | Text | 1.2B | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Tool Calling Data | Text | 26.2B |  | [Qwen3-235B-A22B-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Synthetic Essential-Web from QwQ-32B | Text | 28.1B | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) |  [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| Translated Synthetic Crawl | Text | 389.9B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Translated Synthetic Wikipedia | Text | 7.9B | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Long Context from Qwen3-235B-A22B-Instruct-2507 | Text | Undisclosed | [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Synthetic Search STEM OPENQ from DeepSeek-R1-0528 | Text | Undisclosed | - | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic MCQ from Qwen2.5-32B-Instruct and DeepSeek-R1-0528 | Text | Undisclosed | - | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Offline Search MCQA HLE from DeepSeek-R1-0528 | Text | Undisclosed | - | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Offline Search MCQA GPQA from Qwen3-235B-A22B and DeepSeek-R1-0528 | Text | Undisclosed | - | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Human Preference from QwQ-32B, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-235B-A22B-Instruct-2507, Mistral-Small-3.1-24B-Instruct-2503, Mistral-Small-3.2-24B-Instruct-2506, MiniMax-M1-80k, MiniMax-M1-40k, Kimi-K2-Instruct, DeepSeek-V3-0324, DeepSeek-R1-0528 | Text | Undisclosed | - | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B); [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507); [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503); [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506); [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k); [MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k); [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Code from Qwen3-32B | Text | Undisclosed | English Common Crawl; English Common Crawl 1.1 | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Synthetic OpenCodeReasoning from DeepSeek-R1 | Text | Undisclosed | [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic LIMO from DeepSeek-R1-0528 | Text | Undisclosed | [LIMO](https://huggingface.co/datasets/GAIR/LIMO) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic SCP from DeepSeek-R1-0528 | Text | Undisclosed | [SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Stack Exchange from DeepSeek-R1-0528 | Text | Undisclosed | [Stack Exchange](https://archive.org/details/stackexchange) | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Synthetic Common Crawl from Qwen3-30B-A3B | Text | Undisclosed | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Wikipedia from Qwen3-30B-A3B | Text | Undisclosed | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Essential-Web from Qwen3-30B-A3B and Qwen3-235B-A22B-Thinking-2507 | Text | Undisclosed | [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |
| Synthetic Textbook Math from Qwen3-30B-A3B, Qwen3-235B-A22B, phi-4 | Text | Undisclosed | [Common Crawl](https://commoncrawl.org/); [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic Math and Code from DeepSeek-R1 and DeepSeek-R1-0528 | Text | Undisclosed | [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K); [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2); [TACO](https://huggingface.co/datasets/BAAI/TACO); [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning); [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning); [NuminaMath CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1); [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |

</details>

## Evaluation Dataset

* **Data Collection Method by dataset**: Hybrid: Human, Synthetic
* **Labeling Method by dataset**: Hybrid: Automated, Human, Synthetic

## Inference

* **Acceleration Engine:** PyTorch
* **Test Hardware:**
    * NVIDIA Hopper:
      * 8x H100
      * 1x H200
    * NVIDIA Grace Blackwell
        * GB200

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

We advise against circumvention of any provided safety guardrails contained in the Model without a substantially similar guardrail appropriate for your use case. For more details: [Safety](./safety.md) and [Explainability](./explainability.md) Subcards.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](./bias.md), and [Privacy](./privacy.md) Subcards.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Citation

```bibtex
@misc{nvidia_nemotron_3_2025,
  title  = {NVIDIA Nemotron 3: Efficient and Open Intelligence},
  author = {{NVIDIA}},
  year   = {2025},
  url    = {https://arxiv.org/abs/2512.20856},
  note   = {White Paper}
}
```