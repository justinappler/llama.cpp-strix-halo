---
library_name: transformers
license: other
base_model: passing2961/finch_8b_hard_without_held_out_expr_purpose_qwen_1.0e-5_1.0_train42_cosine
tags:
- llama-factory
- full
- generated_from_trainer
- trl
- kto
model-index:
- name: qwen3_5_8b_kto_finch_math_biology_gpu_mode_frontier_cs_alebench_sldbench_optimization_science_algotune_erdos_kto_without_held_out_expr_purpose_qwen_max16384_kto_5.0e-7_1.0_train42_cosine
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3_5_8b_kto_finch_math_biology_gpu_mode_frontier_cs_alebench_sldbench_optimization_science_algotune_erdos_kto_without_held_out_expr_purpose_qwen_max16384_kto_5.0e-7_1.0_train42_cosine

This model is a fine-tuned version of [passing2961/finch_8b_hard_without_held_out_expr_purpose_qwen_1.0e-5_1.0_train42_cosine](https://huggingface.co/passing2961/finch_8b_hard_without_held_out_expr_purpose_qwen_1.0e-5_1.0_train42_cosine) on the finch_math_biology_gpu_mode_frontier_cs_alebench_sldbench_optimization_science_algotune_erdos_kto_without_held_out_expr_purpose_qwen_max16384 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3769
- Rewards/chosen: -0.8633
- Logps/chosen: -426.0238
- Logits/chosen: -409977942.7797
- Rewards/rejected: -2.5406
- Logps/rejected: -698.4209
- Logits/rejected: -483195259.2593
- Rewards/margins: 1.6773
- Kl: 0.0

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
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- total_eval_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Rewards/chosen | Logps/chosen | Logits/chosen   | Rewards/rejected | Logps/rejected | Logits/rejected | Rewards/margins |     |
|:-------------:|:------:|:----:|:---------------:|:--------------:|:------------:|:---------------:|:----------------:|:--------------:|:---------------:|:---------------:|:---:|
| 0.3970        | 0.2569 | 200  | 0.4073          | -0.7461        | -424.8521    | -408316355.2542 | -1.8452          | -691.4669      | -482611768.8889 | 1.0991          | 0.0 |
| 0.3800        | 0.5138 | 400  | 0.3851          | -0.9247        | -426.6386    | -404957635.2542 | -2.5752          | -698.7675      | -476804816.5926 | 1.6505          | 0.0 |
| 0.3675        | 0.7707 | 600  | 0.3769          | -0.8608        | -425.9987    | -409048983.8644 | -2.5320          | -698.3346      | -481385282.3704 | 1.6712          | 0.0 |


### Framework versions

- Transformers 5.2.0
- Pytorch 2.10.0a0+b558c986e8.nv25.11
- Datasets 4.0.0
- Tokenizers 0.22.1
