---
library_name: transformers
base_model: meta-llama/Meta-Llama-3-8B
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: context
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# context

This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B](https://huggingface.co//meta-llama/Meta-Llama-3-8B) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2641
- Accuracy: 0.7770

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 40
- eval_batch_size: 40
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 160
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.55.0
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.4
