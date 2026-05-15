---
language:
- en
pipeline_tag: text-generation
tags:
- pretrained
- moe
license: apache-2.0
---

# Qwen2-57B-A14B

## Introduction

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This repo contains the 57B-A14B Mixture-of-Experts Qwen2 base language model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics, reasoning, etc.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2/), [GitHub](https://github.com/QwenLM/Qwen2), and [Documentation](https://qwen.readthedocs.io/en/latest/).
<br>


## Model Details
Qwen2 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes.

## Requirements
The code of Qwen2MoE has been in the latest Hugging face transformers and we advise you to install `transformers>=4.40.0`, or you might encounter the following error:
```
KeyError: 'qwen2_moe'
```


## Usage

We do not advise you to use base language models for text generation. Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., on this model.

## Performance

The evaluation of base models mainly focuses on the model performance of natural language understanding, general question answering, coding, mathematics, scientific knowledge, reasoning, multilingual capability, etc. 

The datasets for evaluation include: 
 
**English Tasks**: MMLU (5-shot), MMLU-Pro (5-shot), GPQA (5shot), Theorem QA (5-shot), BBH (3-shot), HellaSwag (10-shot), Winogrande (5-shot), TruthfulQA (0-shot), ARC-C (25-shot)
 
**Coding Tasks**: EvalPlus (0-shot) (HumanEval, MBPP, HumanEval+, MBPP+), MultiPL-E (0-shot) (Python, C++, JAVA, PHP, TypeScript, C#, Bash, JavaScript)
  
**Math Tasks**: GSM8K (4-shot), MATH (4-shot)
 
**Chinese Tasks**: C-Eval(5-shot), CMMLU (5-shot)
 
**Multilingual Tasks**: Multi-Exam (M3Exam 5-shot, IndoMMLU 3-shot, ruMMLU 5-shot, mMMLU 5-shot), Multi-Understanding (BELEBELE 5-shot, XCOPA 5-shot, XWinograd 5-shot, XStoryCloze 0-shot, PAWS-X 5-shot), Multi-Mathematics (MGSM 8-shot), Multi-Translation (Flores-101 5-shot)
 
#### Qwen2-57B-A14B performance
|  Datasets  |  Jamba  |   Mixtral-8x7B |   Yi-1.5-34B  |   Qwen1.5-32B  |  ****Qwen2-57B-A14B****  |
| :--------| :---------: | :------------: | :------------: | :------------: | :------------: |
|Architecture | MoE | MoE | Dense | Dense | MoE |
|#Activated Params | 12B | 12B | 34B | 32B | 14B |
|#Params | 52B | 47B | 34B | 32B | 57B   |
|   ***English***  |    |    |   |    |	    |
|MMLU | 67.4 | 71.8 | **77.1** | 74.3 | 76.5 |
|MMLU-Pro | - | 41.0 | **48.3** | 44.0 | 43.0 |
|GPQA | - | 29.2 | - | 30.8 | **34.3** |
|Theorem QA | - | 23.2 | - | 28.8 | **33.5** |
|BBH  | 45.4 |  50.3  | **76.4** | 66.8 | 67.0 |
|HellaSwag  | **87.1** |  86.5  | 85.9 |  85.0 | 85.2 |
|Winogrande  | 82.5 |  81.9  | **84.9** |  81.5 |  79.5 |
|ARC-C  | 64.4 |  **66.0**  | 65.6 | 63.6 |  64.1 |
|TruthfulQA  | 46.4 |  51.1  | 53.9 | 57.4 |  **57.7** |
|   ***Coding***  |    |    |   |    |	    |
|HumanEval | 29.3 | 37.2 | 46.3 | 43.3 | **53.0**  |
|MBPP | - | 63.9 | 65.5 | 64.2 | **71.9**  |
|EvalPlus | - | 46.4 | 51.9 | 50.4 | **57.2**  |
|MultiPL-E | - | 39.0 | 39.5 | 38.5 | **49.8**  |
|   ***Mathematics***  |    |    |   |    |	    |
|GSM8K | 59.9 |  62.5  | **82.7** | 76.8 | 80.7 |
|MATH  | - |  30.8  | 41.7 | 36.1 | **43.0** |
|   ***Chinese***  |    |    |   |    |	    |
|C-Eval   | - |   -    |  - |  83.5 |  **87.7** |
|CMMLU   | - |   -    | 84.8 | 82.3 | **88.5** |
|   ***Multilingual***  |    |    |   |    |	    |
|Multi-Exam   | - |   56.1    |  58.3 |  61.6 |  **65.5** |
|Multi-Understanding | - |   70.7    |  73.9 |  76.5 |  **77.0** |
|Multi-Mathematics | - |   45.0    |  49.3 |  56.1 |  **62.3** |
|Multi-Translation | - |   29.8    |  30.0 |  33.5 |  **34.5** |

### Efficient MoE Models
Compared with training models smaller than 7 billion parameters, it is costly to train medium-size models like 32B while admittedly the 14B model is incapable of performing complex tasks well as the 72B model does. Owing to the recent success of MoE models, this time we turn to employ the MoE model architecture following our previous work Qwen1.5-MoE-A2.7B and extend it to larger model size. Specifically, we apply the same architecture and training strategy, e.g., upcycling, to the model with a total of 57B parameters, only 14B of which are activated in each forward pass. In the following, we list the inference performance of the two models in the deployment with vLLM on 2 NVIDIA A100:

|    | Qwen2-57B-A14B | Qwen1.5-32B    |
| :---| :---------: | :------------: |
| QPS |   9.40    |     5.18      | 
| TPS |  10345.17    |   5698.37   |

In terms of efficiency, we observe clear advantages of Qwen2-57B-A14B over Qwen1.5-32B. Furthermore, based on the previous report of model performance on benchmarks, it can be found that Qwen2-57B-A14B obtains superior model quality compared with Qwen1.5-32B, which has more activated parameters. 

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2,
  title={Qwen2 Technical Report},
  year={2024}
}
```