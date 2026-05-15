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
- Loss: 5.1785
- Accuracy: 0.2448

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
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Accuracy |
|:-------------:|:------:|:----:|:---------------:|:--------:|
| 1.276         | 0.8306 | 500  | 5.1785          | 0.2448   |
| 0.5321        | 1.6611 | 1000 | 5.2637          | 0.2565   |
| 0.3538        | 2.4917 | 1500 | 5.4336          | 0.2552   |


### Framework versions

- Transformers 4.45.0.dev0
- Pytorch 2.3.1+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
