---
language:
- en
- zh
library_name: transformers
tags:
- Long Context
- chatglm
- llama
datasets:
- THUDM/LongCite-45k
pipeline_tag: text-generation
---
# LongCite-llama3.1-8b

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/THUDM/LongCite-45k" target="_blank">[LongCite Dataset] </a> • 💻 <a href="https://github.com/THUDM/LongCite" target="_blank">[Github Repo]</a> • 📃 <a href="https://arxiv.org/abs/2409.02897" target="_blank">[LongCite Paper]</a> 
</p>

LongCite-llama3.1-8b is trained based on [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), and is capable of generating fine-grained citations in long-context question answering. The model supports a maximum context window of up to 128K tokens.

Environment: `transforemrs>=4.43.0`.

A simple demo for deployment of the model:
```python
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('THUDM/LongCite-llama3.1-8b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('THUDM/LongCite-llama3.1-8b', torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')

context = '''
W. Russell Todd, 94, United States Army general (b. 1928). February 13. Tim Aymar, 59, heavy metal singer (Pharaoh) (b. 1963). Marshall \"Eddie\" Conway, 76, Black Panther Party leader (b. 1946). Roger Bonk, 78, football player (North Dakota Fighting Sioux, Winnipeg Blue Bombers) (b. 1944). Conrad Dobler, 72, football player (St. Louis Cardinals, New Orleans Saints, Buffalo Bills) (b. 1950). Brian DuBois, 55, baseball player (Detroit Tigers) (b. 1967). Robert Geddes, 99, architect, dean of the Princeton University School of Architecture (1965–1982) (b. 1923). Tom Luddy, 79, film producer (Barfly, The Secret Garden), co-founder of the Telluride Film Festival (b. 1943). David Singmaster, 84, mathematician (b. 1938).
'''
query = "What was Robert Geddes' profession?"
result = model.query_longcite(context, query, tokenizer=tokenizer, max_input_length=128000, max_new_tokens=1024)

print("Answer:\n{}\n".format(result['answer']))
print("Statement with citations:\n{}\n".format(
  json.dumps(result['statements_with_citations'], indent=2, ensure_ascii=False)))
print("Context (divided into sentences):\n{}\n".format(result['splited_context']))
```

You can also deploy the model with [vllm](https://github.com/vllm-project/vllm). See the code example in [vllm_inference.py](https://huggingface.co/THUDM/LongCite-llama3.1-8b/blob/main/vllm_inference.py).

## License
[Llama-3.1 License](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)

## Citation

If you find our work useful, please consider citing LongCite:

```
@article{zhang2024longcite,
  title = {LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA} 
  author={Jiajie Zhang and Yushi Bai and Xin Lv and Wanjun Gu and Danqing Liu and Minhao Zou and Shulin Cao and Lei Hou and Yuxiao Dong and Ling Feng and Juanzi Li},
  journal={arXiv preprint arXiv:2409.02897},
  year={2024}
}
```