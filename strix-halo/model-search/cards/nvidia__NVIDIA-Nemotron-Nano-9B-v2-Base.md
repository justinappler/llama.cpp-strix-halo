---
library_name: transformers
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
pipeline_tag: text-generation
language:
- en
- es
- fr
- de
- ja
- it
- pt
- zh
- ar
- da
- ko
- nl
- pl
- ru
- sv
- th
tags:
- nvidia
- pytorch
datasets:
- nvidia/Nemotron-Pretraining-Dataset-sample
- nvidia/Nemotron-CC-v2
- nvidia/Nemotron-CC-Math-v1
- nvidia/Nemotron-Pretraining-Code-v1
- nvidia/Nemotron-Pretraining-SFT-v1
track_downloads: true
---
# NVIDIA-Nemotron-Nano-9B-v2-Base

**Model Developer:** NVIDIA Corporation

**Model Dates:**

June 2025 \- August 2025

**Data Freshness:**

May 1, 2025

The pretraining data has a cutoff date of May 1, 2025\.

## Model Overview

## Description

NVIDIA-Nemotron-Nano-9B-v2-Base is a large language model (LLM) trained from scratch by NVIDIA that is designed as a completion model for a given piece of text. It uses a hybrid model architecture that consists primarily of Mamba-2 and MLP layers combined with just four Attention layers. The model features a context length of 128K. The supported languages include: English, Spanish, French, German, Japanese, Italian, Portuguese, Chinese, Arabic, Danish, Korean, Dutch, Polish, Russian, Swedish, and Thai. Improved using Qwen.

This model is ready for commercial use. 

## Feature Voting

We want to hear from you! Share your ideas, vote on what matters, and help [shape the future of Nemotron](https://nemotron.ideas.nvidia.com/).

## License/Terms of Use 

GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### Deployment Geography: Global

### Use Case

This model is intended for developers and researchers building LLMs. 

### Release Date: 08/18/2025

Huggingface 08/18/2025 via [https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base) 

## Reference(s)

[NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model](https://arxiv.org/abs/2508.14444)

## Model Architecture

- **Architecture Type:** Mamba2-Transformer Hybrid    
    
- **Network Architecture:** Nemotron-Hybrid   
    
- **Number of model parameters:** *8.89B*


This model was developed based on [NVIDIA-Nemotron-Nano-12B-v2-Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base). 

## Model design

*The model was pruned and distilled from [NVIDIA-Nemotron-Nano-12B-v2-Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base) with \~480B tokens.*


## Input

- **Input Type(s):** Text   
    
- **Input Format(s):** String   
    
- **Input Parameters:** One-Dimensional (1D): Sequences   
    
- **Maximum input size:** 128K tokens   
    
- **Other Properties Related to Input:** Supported languages include English, Spanish, French, German, Japanese, Italian, Portuguese, Chinese, Arabic, Danish, Korean, Dutch, Polish, Russian, Swedish, Thai. 


## Output

- **Output Type(s):** Text   
    
- **Output Format:** String   
    
- **Output Parameters:** One-Dimensional (1D): Sequences   
    
- **Maximum output size:** 128K tokens 


Our AI models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

- Runtime Engine(s): NeMo 25.07.nemotron-nano-v2  
- Supported Hardware Microarchitecture Compatibility: NVIDIA A10G, NVIDIA H100-80GB, NVIDIA A100  
- Operating System(s): Linux


The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s)

- v1.0

# Training, Testing, and Evaluation Datasets:

NVIDIA-Nemotron-Nano-9B-v2-Base is pre-trained on a large corpus of high-quality curated and synthetically-generated data. It is trained in the English language, as well as 15 multilingual languages and 43 programming languages. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracy. The model was trained for approximately twenty trillion tokens.

Alongside the model, we release our [final pretraining data](https://huggingface.co/collections/nvidia/nemotron-pre-training-dataset-689d9de36f84279d83786b35), as outlined in this section. For ease of analysis, there is a sample set that is ungated. For all remaining code, math and multilingual data, gating and approval is required, and the dataset is permissively licensed for model training purposes.

**Data Modality:**  Text **The total size:**  10,648,823,153,919 Tokens **Total number of datasets:** 141 **Dataset partition:** *Training \[100%\], testing \[0%\], validation \[0%\]*  
**Time period for training data collection:** 2013 to May 1, 2025  
**Time period for testing data collection:** 2013 to May 1, 2025  
**Time period for validation data collection:** 2013 to May 1, 2025

More details on the datasets and synthetic data generation methods can be found in the technical report [NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model](https://arxiv.org/abs/2508.14444).

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

## Private Non-publicly Accessible Datasets of Third Parties

| Dataset |
| :---- |
| Global Regulation |

## Crawled and Scraped from Online Sources by NVIDIA

The English Common Crawl data was downloaded from the Common Crawl Foundation (see their FAQ for details on their crawling) and includes the snapshots CC-MAIN-2013-20 through CC-MAIN-2025-13. The data was subsequently deduplicated and filtered in various ways described in the Nemotron-CC paper. Additionally, we extracted data for fifteen languages from the following three Common Crawl snapshots: CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18. The fifteen languages included were Arabic, Chinese, Danish, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, and Thai. As we did not have reliable multilingual model-based quality classifiers available, we applied just heuristic filtering instead—similar to what we did for lower quality English data in the Nemotron-CC pipeline, but selectively removing some filters for some languages that did not work well. Deduplication was done in the same way as for Nemotron-CC.

The GitHub Crawl was collected using the GitHub REST API and the Amazon S3 API. Each crawl was operated in accordance with the rate limits set by its respective source, either GitHub or S3. We collect raw source code and subsequently remove any having a license which does not exist in our permissive-license set (for additional details, refer to the technical report). 

| Dataset | Modality | Dataset Size | Collection Period | Collecting Organisation |
| :---- | :---- | :---- | :---- | :---- |
| English Common Crawl | Text | 3.36T | 4/8/2025 | NVIDIA Advanced Deep Learning Research |
| Multilingual Common Crawl | Text | 812.7B | 5/1/2025 | NVIDIA Advanced Deep Learning Research |
| GitHub Crawl | Text | 747.4B | 4/29/2025 | NVIDIA Advanced Deep Learning Research |

## NVIDIA-Sourced Synthetic Datasets

| Dataset | Modality | Dataset Size | Seed Dataset | Model(s) used for generation |
| :---- | :---- | :---- | :---- | :---- |
| Synthetic Art of Problem Solving from DeepSeek-R1 | Text | 40086030608 | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Moral Stories and Social Chemistry from Mixtral-8x22B-v0.1 | Text | 327M | [social-chemestry-101](https://huggingface.co/datasets/tasksource/social-chemestry-101); [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) | [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| Synthetic Social Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 83.6M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic Health Sciences seeded with OpenStax from DeepSeek-V3, Mixtral-8x22B-v0.1, and Qwen2.5-72B | Text | 9.7M | [OpenStax \- CC BY-SA subset](https://openstax.org/) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| Synthetic STEM seeded with OpenStax, Open Textbook Library, and GSM8K from DeepSeek-R1, DeepSeek-V3, DeepSeek-V3-0324, and Qwen2.5-72B | Text | 175M | [OpenStax \- CC BY-SA subset](https://openstax.org/); [GSM8K](https://github.com/openai/grade-school-math); [Open Textbook Library \- CC BY-SA & GNU subset](https://open.umn.edu/opentextbooks/textbooks/) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324); [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B) |
| [Nemotron-PrismMath](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath) | Text | 4.6B | [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified); [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | [Qwen2.5-0.5B-instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct); [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| Synthetic Question Answering Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text | 350M | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Refreshed [Nemotron-MIND](https://huggingface.co/datasets/nvidia/Nemotron-MIND) from phi-4 | Text | 73B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| nv-cc-math-45-jun2025 | Text | 52.3B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| nv-cc-math-3-jun2025 | Text | 80.9B | [Common Crawl](https://commoncrawl.org/latest-crawl) | [phi-4](https://huggingface.co/microsoft/phi-4) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from DeepSeek-V3 and DeepSeek-V3-0324 | Text | 4.0B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3); [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| Synthetic AGIEval seeded with AQUA-RAT, LogiQA, and AR-LSAT from Qwen3-30B-A3B | Text | 4.2B | [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat); [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa); [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Art of Problem Solving from Qwen2.5-32B-Instruct, Qwen2.5-Math-72B, Qwen2.5-Math-7B, and Qwen2.5-72B-Instruct | Text |  | [Art of Problem Solving](https://artofproblemsolving.com/company); [American Mathematics Competitions 8](https://artofproblemsolving.com/wiki/index.php/AMC_8_Problems_and_Solutions); [American Mathematics Competitions 10](https://artofproblemsolving.com/wiki/index.php/AMC_10_Problems_and_Solutions); [GSM8K](https://github.com/openai/grade-school-math); [PRM800K](https://github.com/openai/prm800k) | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct); [Qwen2.5-Math-72B](https://huggingface.co/Qwen/Qwen2.5-Math-72B); [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B); [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic MMLU Auxiliary Train from DeepSeek-R1 | Text | 0.5B | [MMLU Auxiliary Train](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Synthetic Long Context Continued Post-Training Data from Papers and Permissible Books from Qwen2.5-72B-Instruct | Text |  | [arXiv](https://info.arxiv.org/help/bulk_data/index.html); [National Institutes of Health ExPorter](https://www.nih.gov/); [BioRxiv](https://www.biorxiv.org/tdm); [PMC Article](https://pmc.ncbi.nlm.nih.gov/tools/textmining/); [USPTO Backgrounds](https://data.uspto.gov/apis/transition-guide/bdss#pats); [peS2o](https://huggingface.co/datasets/allenai/peS2o); Global Regulation; [CORE](https://core.ac.uk/documentation/dataset); [PG-19](https://github.com/google-deepmind/pg19); [DOAB CC BY & CC BY-SA subset](https://www.doabooks.org/en); [NDLTD](https://ndltd.org/thesis-resources/global-etd-search/) | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Synthetic Common Crawl from Qwen3-30B-A3B and Mistral-Nemo-12B-Instruct | Text | 415.8B | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct) |
| Synthetic Multilingual Data from Common Crawl from Qwen3-30B-A3B | Text |  | [Common Crawl](https://commoncrawl.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Multilingual Data from Wikimedia from Qwen3-30B-A3B | Text |  | [Wikimedia](https://dumps.wikimedia.org/) | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Synthetic Math Data from Wikimedia from Nemotron-4-340B-Instruct | Text |  | \- | [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) |

## Training Dataset :

| Dataset | \# Tokens |
| :---- | :---- |
| English Common Crawl | 3,360,110,334,818 |
| English Synthetic CC | 1,949,464,641,123 |
| Crawl++ | 360,389,153,262 |
| Math | 124,606,230,663 |
| Synthetic Math | 73,007,767,155 |
| Code | 747,409,228,724 |
| Synthetic Code | 175,067,553,293 |
| English Wiki | 17,349,266,926 |
| Books | 0 |
| Papers | 191,586,493,365 |
| PDF-to-text | 141,096,578,533 |
| Code SFT | 60,025,726,817 |
| STEM SFT | 272,680,426,295 |
| General SFT | 6,057,478,645 |
| Multilingual | 2,172,261,909,350 |
| Synthetic multilingual | 997,710,364,950 |
| Total | 10,648,823,153,919 |

We use a considerable amount of synthetic data. Out of 10.6 trillion tokens, 3,534,013,958,278 tokens are synthetically generated.

We extracted data for fifteen languages from the following three Common Crawl snapshots: CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18. The fifteen languages included were Arabic, Chinese, Danish, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, and Thai. As we did not have reliable multilingual model-based quality classifiers available, we applied just heuristic filtering instead—similar to what we did for lower quality English data in the Nemotron-CC pipeline, but selectively removing some filters for some languages that did not work well. Deduplication was done in the same way as for Nemotron-CC. Additionally, we used data from Wikipedia and FineWeb-2 (Penedo et al., 2025\) for these fifteen languages.

| Language | Total Tokens |
| :---- | :---- |
| Arabic | 118,056,362,726 |
| Danish | 117,747,321,618 |
| German | 146,613,691,781 |
| Spanish | 469,156,575,409 |
| French | 139,982,002,289 |
| Italian | 298,858,370,174 |
| Japanese | 682,755,693,336 |
| Korean | 127,099,747,538 |
| Dutch | 89,041,592,681 |
| Polish | 105,356,493,147 |
| Portuguese | 243,249,275,089 |
| Russian | 185,314,014,057 |
| Swedish | 74,954,953,299 |
| Thai | 160,778,944,467 |
| Chinese | 211,007,236,689 |

We collect a total of 922,476,782,017 tokens of code in 43 different languages.

| Language | Tokens |
| :---- | :---- |
| Assembly | 750,628,764 |
| C | 42,657,300,868 |
| C\# | 56,153,329,307 |
| C++ | 67,773,701,658 |
| CommonLisp | 263,234,672 |
| CSS | 38,848,760,035 |
| Cuda | 400,222,993 |
| Dart | 3,816,960,470 |
| Dockerfile | 474,958,084 |
| Fortran | 1,105,049,387 |
| Go | 8,332,419,480 |
| Haskell | 1,294,613,669 |
| HTML | 69,082,117,487 |
| Java | 131,440,465,822 |
| JavaScript | 75,573,420,861 |
| JSON | 15,366,881,241 |
| Julia | 621,046,949 |
| JupyterNotebook | 2,241,893,197 |
| Lua | 4,146,420,802 |
| Makefile | 12,640,010,879 |
| Markdown | 64,796,743,311 |
| Mathematica | 320,504,225 |
| OmniversePython | 26,946,093 |
| Pascal | 1,625,013,876 |
| Perl | 1,575,314,434 |
| PHP | 61,575,339,005 |
| Python | 126,916,727,384 |
| R | 19,811,381,935 |
| reStructuredText | 1,779,876,391 |
| Ruby | 6,446,962,615 |
| Rust | 4,438,640,533 |
| Scala | 3,343,959,154 |
| Shell | 18,758,779,250 |
| SQL | 23,205,633,085 |
| Swift | 5,976,714,881 |
| SystemVerilog | 233,056,185 |
| TeX | 7,347,157,527 |
| TypeScript | 15,657,838,582 |
| Verilog | 811,884,369 |
| VHDL | 648,401,444 |
| VisualBasic.NET | 1,005,680,881 |
| XML | 12,616,779,741 |
| YAML | 10,574,010,491 |

## 

## Evaluation Dataset:


* Data Collection Method by dataset: Hybrid: Human, Synthetic  
* Labeling Method by dataset: Hybrid: Automated, Human, Synthetic


### Base Benchmark Evaluations

We evaluated our model on the following benchmarks:

| Task | N-Nano-V2 12B Base |  | N-Nano-V2 9B Base | Qwen3 8B Base | Gemma3 12B Base |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **General** |  |  |  |  |  |
| MMLU | 78.24 |  | 74.53 | 76.44 | 73.61 |
| MMLU-Pro 5-shot | 63.98 |  | **59.43** | 56.27 | 45.12 |
| AGIEval English CoT | 68.03 |  | **65.28** | 59.54 | 51.69 |
| Math |  |  |  |  |  |
| GSM8K CoT | 91.66 |  | **91.36** | 84.00 | 74.45 |
| **Math** | 83.54 |  | **80.50** | 55.40 | 42.40 |
| MATH Level 5 | 67.61 |  | **63.64** | 29.91 | 17.71 |
| AIME 2024 avg@32 | 56.67 |  | 30.00 | 20.00 | 16.67 |
| **Code** |  |  |  |  |  |
| HumanEval+ Pass@1 | 61.03 |  | 58.50 | 57.55 | 36.68 |
| MBPP+ Pass@1 | 61.55 |  | 58.95 | 58.56 | 51.73 |
| **Commonsense Understanding** |  |  |  |  |  |
| ARC Challenge | 93.26 |  | 90.70 | **93.09** | 90.44 |
| HellaSwag | 84.00 |  | 79.90 | 79.75 | **84.15** |
| OpenBookQA | 46.00 |  | 44.80 | 42.00 | **46.00** |
| PIQA | 82.54 |  | 81.83 | 79.43 | **82.10** |
| WinoGrande | 79.24 |  | 75.30 | 75.93 | **79.95** |
| **Long Context** |  |  |  |  |  |
| RULER-128K | 84.74 |  | **82.22** | \- | 80.70 |

*Table 1: Accuracy of Nemotron-Nano-V2-Base models versus existing SoTA models. N-Nano-V2 is short for Nemotron-Nano-V2. The distilled N-Nano-V2-9B-Base is compared against Qwen3-8B-Base and Gemma3-12B-Base, and the best score is highlighted in each row.*

| Task | N-Nano-V2 12B Base |  | N-Nano-V2 9B Base | Qwen3 8B Base | Gemma3 12B Base |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Global-MMLU-Lite** |  |  |  |  |  |
| German | 74.50 |  | 68.25 | **75.50** | 69.75 |
| Spanish | 76.50 |  | 72.75 | **75.00** | 74.00 |
| French | 78.25 |  | 69.75 | **74.25** | 72.50 |
| Italian | 76.50 |  | 73.25 | 72.75 | **74.00** |
| Japanese | 71.00 |  | 67.00 | 70.00 | **71.50** |
| Korean | 72.50 |  | 67.25 | 67.25 | **70.25** |
| Portuguese | 76.25 |  | 71.25 | 72.50 | **75.75** |
| Chinese | 75.50 |  | 69.25 | **75.25** | 67.25 |
| Average | 75.13 |  | 69.94 | **72.81** | 71.88 |
| **Multilingual Math (MGSM)** |  |  |  |  |  |
| Spanish | 93.20 |  | **91.60** | 86.40 | 74.00 |
| German | 89.60 |  | **89.60** | 78.80 | 68.80 |
| French | 86.40 |  | **86.00** | 78.80 | 70.80 |
| Chinese | 44.40 |  | **75.20** | 28.80 | 26.80 |
| Japanese | 76.00 |  | **74.80** | 30.80 | 26.40 |
| Russian | 90.40 |  | **91.60** | 83.60 | 76.00 |
| Average | 80.00 |  | **84.80** | 64.53 | 57.13 |

*Table 2: Accuracy of Nemotron-Nano-V2-Base models versus existing SoTA models on multilingual benchmarks. N-Nano-V2 is short for Nemotron-Nano-V2. The distilled N-Nano-V2-9B-Base is compared against Qwen3-8B-Base and Gemma3-12B-Base, and the best score is highlighted in each row.*

## Inference

- ## Engines: HF, vLLM, TRT-LLM

- ## Test Hardware NVIDIA A10G 24GB, H100 80GB


## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our [Trustworthy AI terms of service](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/), developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](bias.md), [Explainability](explainability.md), [Safety & Security](safety.md), and [Privacy](privacy.md) Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


## Citation

```
@misc{nvidia2025nvidianemotronnano2,
      title={NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model},
      author={NVIDIA},
      year={2025},
      eprint={2508.14444},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.14444},
}
```