---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
pipeline_tag: text-generation
tags:
- liquid
- lfm2
- edge
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/VmrjMZg7WYfMX5YFHuZ-P.png" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em;">
<a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

<br>

# LFM2-24B-A2B

LFM2 is a family of hybrid models designed for on-device deployment. LFM2-24B-A2B is the largest model in the family, scaling the architecture to 24 billion parameters while keeping inference efficient.

- **Best-in-class efficiency**: A 24B MoE model with only 2B active parameters per token, fitting in 32 GB of RAM for deployment on consumer laptops and desktops.
- **Fast edge inference**: 112 tok/s decode on AMD CPU, 293 tok/s on H100. Fits in 32B GB of RAM with day-one support llama.cpp, vLLM, and SGLang.
- **Predictable scaling**: Quality improves log-linearly from 350M to 24B total parameters, confirming the LFM2 hybrid architecture scales reliably across nearly two orders of magnitude.

![image](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/VhdjtAT5zWTWdVYgW69O0.png)

Find more information about LFM2-24B-A2B in our [blog post](https://www.liquid.ai/blog/lfm2-24b-a2b).

## 🗒️ Model Details

LFM2-24B-A2B is a general-purpose instruct model (without reasoning traces) with the following features:

| Property              | [**LFM2-8B-A1B**](https://huggingface.co/LiquidAI/LFM2-8B-A1B) | [**LFM2-24B-A2B**](https://huggingface.co/LiquidAI/LFM2-24B-A2B) | 
| --------------------- | ----------------------------- | ----------------------------- |
| **Total parameters**  | 8.3B                          | 24B |
| **Active parameters** | 1.5B                          | 2.3B |
| **Layers**            | 24 (18 conv + 6 attn)         | 40 (30 conv + 10 attn) |
| **Context length**    | 32,768 tokens                 | 32,768 tokens |
| **Vocabulary size**   | 65,536                        | 65,536 |
| **Training precision**| Mixed BF16/FP8                | Mixed BF16/FP8 |
| **Training budget**   | 12 trillion tokens            | 17 trillion tokens |
| **License**           | LFM Open License v1.0         | LFM Open License v1.0 |

**Supported languages**: English, Arabic, Chinese, French, German, Japanese, Korean, Spanish, Portuguese

**Generation parameters**:
- `temperature: 0.1`
- `top_k: 50`
- `repetition_penalty: 1.05`

We recommend the following use cases:

- **Agentic tool use**: Native function calling, web search, structured outputs. Ideal as the fast inner-loop model in multi-step agent pipelines.
- **Offline document summarization and Q&A**: Run entirely on consumer hardware for privacy-sensitive workflows (legal, medical, corporate).
- **Privacy-preserving customer support agent**: Deployed on-premise at a company, handles multi-turn support conversations with tool access (database lookups, ticket creation) without data leaving the network.
- **Local RAG pipelines**: Serve as the generation backbone in retrieval-augmented setups on a single machine without GPU servers.

We don't recommend using it for coding, as it wasn't optimized for this purpose.

### Chat Template

LFM2-24B-A2B uses a ChatML-like format. See the [Chat Template documentation](https://docs.liquid.ai/lfm/key-concepts/chat-template) for details. Example:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
```

You can use [`tokenizer.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_templating#using-applychattemplate) to format your messages automatically.

### Tool Use

LFM2-24B-A2B supports function calling as follows:

1. **Function definition**: We recommend providing the list of tools as a JSON object in the system prompt. You can also use the [`tokenizer.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools) function with tools.
2. **Function call**: By default, LFM2-24B-A2B writes Pythonic function calls (a Python list between `<|tool_call_start|>` and `<|tool_call_end|>` special tokens), as the assistant answer. You can override this behavior by asking the model to output JSON function calls in the system prompt.
3. **Function execution**: The function call is executed, and the result is returned as a "tool" role.
4. **Final answer**: LFM2-24B-A2B interprets the outcome of the function call to address the original user prompt in plain text.

See the [Tool Use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use) for the full guide. Example:

```
<|startoftext|><|im_start|>system
List of tools: [{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
[{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}]<|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

## 🏃 Inference

LFM2-24B-A2B is supported by many inference frameworks. See the [Inference documentation](https://docs.liquid.ai/lfm/inference/transformers) for the full list.

| Name | Description | Docs | Notebook |
|------|-------------|------|:--------:|
| [Transformers](https://github.com/huggingface/transformers) | Simple inference with direct access to model internals. | <a href="https://docs.liquid.ai/lfm/inference/transformers">Link</a> | <a href="https://colab.research.google.com/drive/1_q3jQ6LtyiuPzFZv7Vw8xSfPU5FwkKZY?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput production deployments with GPU. | <a href="https://docs.liquid.ai/lfm/inference/vllm">Link</a> | <a href="https://colab.research.google.com/drive/1VfyscuHP8A3we_YpnzuabYJzr5ju0Mit?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Cross-platform inference with CPU offloading. | <a href="https://docs.liquid.ai/lfm/inference/llama-cpp">Link</a> | <a href="https://colab.research.google.com/drive/1ohLl3w47OQZA4ELo46i5E4Z6oGWBAyo8?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| [MLX](https://github.com/ml-explore/mlx) | Apple's machine learning framework optimized for Apple Silicon. | <a href="https://docs.liquid.ai/lfm/inference/mlx">Link</a> | — |
| [LM Studio](https://lmstudio.ai/) | Desktop application for running LLMs locally. | <a href="https://docs.liquid.ai/lfm/inference/lm-studio">Link</a> | — |

Here's a quick start example with Transformers (compatible with `transformers>=5.0.0`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "LiquidAI/LFM2-24B-A2B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "What is C. elegans?"

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    repetition_penalty=1.05,
    max_new_tokens=512,
    streamer=streamer,
)
```

## 🔧 Fine-Tuning

| Name | Description | Docs | Notebook |
|------|-------------|------|----------|
| CPT ([Unsloth](https://github.com/unslothai/unsloth)) | Continued Pre-Training using Unsloth for text completion. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/10fm7eNMezs-DSn36mF7vAsNYlOsx9YZO?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| CPT ([Unsloth](https://github.com/unslothai/unsloth)) | Continued Pre-Training using Unsloth for translation. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/1gaP8yTle2_v35Um8Gpu9239fqbU7UgY8?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| SFT ([Unsloth](https://github.com/unslothai/unsloth)) | Supervised Fine-Tuning with LoRA using Unsloth. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/1vGRg4ksRj__6OLvXkHhvji_Pamv801Ss?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| SFT ([TRL](https://github.com/huggingface/trl)) | Supervised Fine-Tuning with LoRA using TRL. | <a href="https://docs.liquid.ai/lfm/fine-tuning/trl">Link</a> | <a href="https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| DPO ([TRL](https://github.com/huggingface/trl)) | Direct Preference Optimization with LoRA using TRL. | <a href="https://docs.liquid.ai/lfm/fine-tuning/trl">Link</a> | <a href="https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| GRPO ([Unsloth](https://github.com/unslothai/unsloth)) | GRPO with LoRA using Unsloth. | <a href="https://docs.liquid.ai/lfm/fine-tuning/unsloth">Link</a> | <a href="https://colab.research.google.com/drive/1mIikXFaGvcW4vXOZXLbVTxfBRw_XsXa5?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| GRPO ([TRL](https://github.com/huggingface/trl)) | GRPO with LoRA using TRL. | <a href="https://docs.liquid.ai/lfm/fine-tuning/trl">Link</a> | <a href="https://colab.research.google.com/github/Liquid4All/cookbook/blob/main/finetuning/notebooks/grpo_for_verifiable_tasks.ipynb"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |

## 📊 Performance

### CPU Inference

We compared LFM2-24B-A2B against two popular MoE models of similar size: Qwen3-30B-A3B-Instruct-2507 (30.5B total, 3.3B active parameters) and gpt-oss-20b (21B total, 3.6B active parameters). We measured both prefill and decode throughputs with Q4_K_M versions of these models using llama.cpp on AMD Ryzen AI Max+ 395.

![image](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/voUOPTqOXQaiEEm_6vY2j.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/7G2mtNj0vxSbUXnovmbzs.png)

### GPU Inference

We also report throughput (total tokens / wall time) achieved with vLLM on a single H100 SXM5 GPU.

![image](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/5b2a3bfGYmyO16Gxos4dH.png)

## 📬 Contact

- Got questions or want to connect? [Join our Discord community](https://discord.com/invite/liquid-ai)
- If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).

## Citation

```bibtex
@article{liquidAI202624B,
  author = {Liquid AI},
  title = {LFM2.5-24B-A2B: Scaling Up the LFM2 Architecture},
  journal = {Liquid AI Blog},
  year = {2026},
  note = {www.liquid.ai/blog/},
}
```

```bibtex
@article{liquidai2025lfm2,
  title={LFM2 Technical Report},
  author={Liquid AI},
  journal={arXiv preprint arXiv:2511.23404},
  year={2025}
}
```