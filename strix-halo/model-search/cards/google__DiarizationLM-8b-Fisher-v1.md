---
license: llama3
---

**This is not an officially supported Google product.**

## Overview

Note: This model is outdated. Please use [google/DiarizationLM-8b-Fisher-v2](https://huggingface.co/google/DiarizationLM-8b-Fisher-v2) instead.

[DiarizationLM](https://arxiv.org/abs/2401.03506) model finetuned
on the training subset of the Fisher corpus.

* Foundation model: [unsloth/llama-3-8b-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)
* Finetuning scripts: https://github.com/google/speaker-id/tree/master/DiarizationLM/unsloth

## Training config

This model is finetuned on the training subset of the Fisher corpus, using a LoRA adapter of rank 256. The total number of training parameters is 671,088,640. With a batch size of 16, this model has been trained for 25400 steps, which is ~8 epochs of the training data.

We use the `mixed` flavor during our training, meaning we combine data from `hyp2ora` and `deg2ref` flavors. After the prompt builder, we have a total of 51,063 prompt-completion pairs in our training set.

The finetuning took more than 4 days on a Google Cloud VM instance that has one NVIDIA A100 GPU with 80GB memory.

The maximal length of the prompt to this model is 6000 characters, including the " --> " suffix. The maximal sequence length is 4096 tokens.

## Metrics

### Fisher testing set

| System  | WER (%) | WDER (%) | cpWER (%) |
| ------- | ------- | -------- | --------- |
| USM + turn-to-diarize baseline | 15.48 | 5.32 | 21.19 |
| + This model | - | 4.40 | 19.76 |

### Callhome testing set

| System  | WER (%) | WDER (%) | cpWER (%) |
| ------- | ------- | -------- | --------- |
| USM + turn-to-diarize baseline | 15.36 | 7.72 | 24.39 |
| + This model | - | 12.27 | 30.80 |

## Usage

First, you need to install two packages:

```
pip install transformers diarizationlm
```

On a machine with GPU and CUDA, you can use the model by running the following script:

```python
from transformers import LlamaForCausalLM, AutoTokenizer
from diarizationlm import utils

HYPOTHESIS = """<speaker:1> Hello, how are you doing <speaker:2> today? I am doing well. What about <speaker:1> you? I'm doing well, too. Thank you."""

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("google/DiarizationLM-8b-Fisher-v1", device_map="cuda")
model = LlamaForCausalLM.from_pretrained("google/DiarizationLM-8b-Fisher-v1", device_map="cuda")

print("Tokenizing input...")
inputs = tokenizer([HYPOTHESIS + " --> "], return_tensors = "pt").to("cuda")

print("Generating completion...")
outputs = model.generate(**inputs,
                         max_new_tokens = inputs.input_ids.shape[1] * 1.2,
                         use_cache = False)

print("Decoding completion...")
completion = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                    skip_special_tokens = True)[0]

print("Transferring completion to hypothesis text...")
transferred_completion = utils.transfer_llm_completion(completion, HYPOTHESIS)

print("========================================")
print("Hypothesis:", HYPOTHESIS)
print("========================================")
print("Completion:", completion)
print("========================================")
print("Transferred completion:", transferred_completion)
print("========================================")
```

The output will look like below:

```
Loading model...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:13<00:00,  3.32s/it]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 172/172 [00:00<00:00, 992kB/s]
Tokenizing input...
Generating completion...
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Decoding completion...
Transferring completion to hypothesis text...
========================================
Hypothesis: <speaker:1> Hello, how are you doing <speaker:2> today? I am doing well. What about <speaker:1> you? I'm doing well, too. Thank you.
========================================
Completion:  <speaker:1> Hello, how are you doing today? <speaker:2> i am doing well. What about you? <speaker:1> i'm doing well, too. Thank you. [eod] [eod] <speaker:2
========================================
Transferred completion: <speaker:1> Hello, how are you doing today? <speaker:2> I am doing well. What about you? <speaker:1> I'm doing well, too. Thank you.
========================================
```

## Citation

Our paper is cited as:

```
@article{wang2024diarizationlm,
  title={{DiarizationLM: Speaker Diarization Post-Processing with Large Language Models}},
  author={Quan Wang and Yiling Huang and Guanlong Zhao and Evan Clark and Wei Xia and Hank Liao},
  journal={arXiv preprint arXiv:2401.03506},
  year={2024}
}
```
