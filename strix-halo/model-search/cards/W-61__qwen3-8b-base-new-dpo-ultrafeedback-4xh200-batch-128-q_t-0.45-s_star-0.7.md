---
library_name: transformers
base_model: jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128
tags:
- alignment-handbook
- new-dpo
- generated_from_trainer
datasets:
- HuggingFaceH4/ultrafeedback_binarized
model-index:
- name: qwen3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-q_t-0.45-s_star-0.7
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-q_t-0.45-s_star-0.7

This model is a fine-tuned version of [jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128](https://huggingface.co/jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128) on the HuggingFaceH4/ultrafeedback_binarized dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-07
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- total_eval_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results



### Framework versions

- Transformers 4.51.0
- Pytorch 2.3.1+cu121
- Datasets 2.21.0
- Tokenizers 0.21.4
