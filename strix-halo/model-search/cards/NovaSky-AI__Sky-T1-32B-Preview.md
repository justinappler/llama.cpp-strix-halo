---
library_name: transformers
datasets:
- codeparrot/apps
- BAAI/TACO
- AI-MO/NuminaMath-CoT
language:
- en
base_model:
- Qwen/Qwen2.5-32B-Instruct
license: apache-2.0
---

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This is a 32B reasoning model trained from Qwen2.5-32B-Instruct with 17K data. The performance is on par with o1-preview model on both math and coding.
Please see our [blog post](https://novasky-ai.github.io/posts/sky-t1/) for more details.

- **Developed by:** NovaSky Team from Sky Computing Lab at UC Berkeley.

## Training Details

### Training Data

17K verified correct responses from Qwen/QwQ-32B-Preview on coding, math. In addition, we add the science portion from the [Still-2 paper](https://arxiv.org/pdf/2412.09413).

### Training Procedure
We perform supervised fine tuning on the data, with a batch size of 96.

#### Speeds

We use Llama-Factory for training. On 8 H100, the training takes 19 hours with DeepSpeed Zero-3 Offload.


## Evaluation
|               | Sky-T1-32B-Preview | Qwen-2.5-32B-Instruct | QwQ   | o1-preview |
|-----------------------|---------------------|--------|-------|------------|
| Math500              | 82.4                    | 76.2    | 85.4 | 81.4       |
| AIME2024             | 43.3                    | 16.7    | 50.0  | 40.0       |
| LiveCodeBench-Easy   | 86.3                    | 84.6   | 90.7  | 92.9       |
| LiveCodeBench-Medium | 56.8                    | 40.8   | 56.3  | 54.9       |
| LiveCodeBench-Hard   | 17.9                    | 9.8   | 17.1  | 16.3       |
| GPQA-Diamond         | 56.8                    | 45.5   | 52.5  | 75.2       |

## Acknowledgement
We would like to thanks the compute resources from [Lambda Lab](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5) and [AnyScale](https://www.anyscale.com/). We would like to thanks the academic feedback and support from the [Still-2 Team](https://arxiv.org/pdf/2412.09413), and [Junyang Lin](https://justinlin610.github.io/) from the [Qwen Team](https://qwenlm.github.io/). 

## Citation 
Please considering citing our blog post if you found it useful for your research. Thank you!

```bibtex
@misc{sky_t1_2025,
  author       = {NovaSky Team},
  title        = {Sky-T1: Fully open-source reasoning model with o1-preview performance in $450 budget},
  howpublished = {https://novasky-ai.github.io/posts/sky-t1},
  note         = {Accessed: 2025-01-09},
  year         = {2025}
}