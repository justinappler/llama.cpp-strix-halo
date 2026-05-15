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
- name: qwen3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-q_t-0.45-s_star-0.2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-q_t-0.45-s_star-0.2

This model is a fine-tuned version of [jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128](https://huggingface.co/jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128) on the HuggingFaceH4/ultrafeedback_binarized dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6408
- Fcm Dpo/beta: 0.0020
- Margin Dpo/margin Mean: 59.5413
- Margin Dpo/margin Std: 98.9336
- Logps/chosen: -343.3604
- Logps/rejected: -387.1306
- Logps/ref Chosen: -280.4167
- Logps/ref Rejected: -264.6455
- Kl/chosen Kl Mean: -62.9437
- Kl/rejected Kl Mean: -122.4851
- Kl/mean: -92.7144
- Kl/std: 83.5219
- Logits/chosen: 1.4764
- Logits/rejected: 1.5542

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

| Training Loss | Epoch  | Step | Validation Loss | Fcm Dpo/beta | Margin Dpo/margin Mean | Margin Dpo/margin Std | Logps/chosen | Logps/rejected | Logps/ref Chosen | Logps/ref Rejected | Kl/chosen Kl Mean | Kl/rejected Kl Mean | Kl/mean  | Kl/std  | Logits/chosen | Logits/rejected |
|:-------------:|:------:|:----:|:---------------:|:------------:|:----------------------:|:---------------------:|:------------:|:--------------:|:----------------:|:------------------:|:-----------------:|:-------------------:|:--------:|:-------:|:-------------:|:---------------:|
| 5.1029        | 0.4188 | 200  | 0.6295          | 0.0054       | 28.0217                | 46.1712               | -289.4213    | -301.6719      | -280.4167        | -264.6455          | -9.0046           | -37.0264            | -23.0155 | 39.3854 | 1.5937        | 1.6466          |
| 5.0934        | 0.8377 | 400  | 0.6408          | 0.0020       | 59.5413                | 98.9336               | -343.3604    | -387.1306      | -280.4167        | -264.6455          | -62.9437          | -122.4851           | -92.7144 | 83.5219 | 1.4764        | 1.5542          |


### Framework versions

- Transformers 4.51.0
- Pytorch 2.3.1+cu121
- Datasets 2.21.0
- Tokenizers 0.21.4
