---
library_name: transformers
license: other
license_name: nvidia-nemotron-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
pipeline_tag: text-generation
language:
  - en
  - ja
tags:
- nvidia
- pytorch
base_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
datasets:
- nvidia/Nemotron-Personas-Japan
- nvidia/Nemotron-CC-v2.1
- nvidia/Nemotron-Pretraining-Specialized-v1
- nvidia/Nemotron-Agentic-v1
- nvidia/Nemotron-Instruction-Following-Chat-v1
- globis-university/aozorabunko-clean
- HuggingFaceFW/fineweb-2
track_downloads: true
---
# NVIDIA-Nemotron-Nano-9B-v2-Japanese

![](./accuracy.png)

[Nejumi LLM Leaderboard 4](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-4--VmlldzoxMzc1OTk1MA)

**モデル開発者 (Model Developer):** NVIDIA Corporation

このモデルはNVIDIA Japanによって追加学習されました。

This model was further trained by NVIDIA Japan.

**モデル開発期間 (Model Dates):**

2025年6月 \- 2026年1月  

June 2025 \- January 2026

**データ鮮度 (Data Freshness):**

2024年9月  

September 2024

事前学習データは2024年9月までのデータを使用しています。  

The pre-training data has a cutoff date of September 2024\.

## モデル概要 (Model Overview)

NVIDIA-Nemotron-Nano-9B-v2-Japanese は、NVIDIA によってスクラッチから学習された大規模言語モデル（LLM）であり、推論タスクと非推論タスクの両方に対応する統合モデルとして設計され、日本語に特化して最適化されています。本モデルは、[Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) データセットを用いて作成された日本語のTool calling データにより、[NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) を基にさらに追加学習が行われました。

本モデルは、ユーザーからの質問やタスクに対して、まず推論トレース（reasoning trace）を生成し、その後に最終回答を提示します。推論機能はシステムプロンプトによって制御可能です。ユーザーが中間的な推論トレースを表示せずに最終回答のみを希望する場合、そのように設定することも可能ですが、推論を要する難易度の高いプロンプトにおいては、精度がわずかに低下する可能性があります。一方で、先に推論トレースを生成させる設定にすると、一般的により高品質な最終回答が得られます。

本モデルは、主に Mamba-2 および MLP 層を中心とし、4 層の Attention 層を組み合わせたハイブリッドアーキテクチャを採用しています。アーキテクチャの詳細については、[Nemotron-H テクニカルレポート](https://arxiv.org/abs/2504.03624)をご参照ください。学習には [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) および [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) が使用され、さらに Qwen によって改良が加えられています。

NVIDIA-Nemotron-Nano-9B-v2-Japanese のリリースブログは以下を参照してください。  
- [NVIDIA Nemotron 2 Nano 9B Japanese: 日本のソブリンAIを支える最先端小規模言語モデル](https://huggingface.co/blog/nvidia/nemotron-nano-9b-v2-japanese-ja)  

本モデルは商用利用が可能です。

NVIDIA-Nemotron-Nano-9B-v2-Japanese is a large language model (LLM) trained from scratch by NVIDIA and designed as a unified model for both reasoning and non-reasoning tasks, specifically optimized for the Japanese language. The model was further trained from NVIDIA-Nemotron-Nano-9B-v2 using Japanese tool-calling data created with the [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) dataset. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response. The model's reasoning capabilities can be controlled via a system prompt. If the user prefers the model to provide its final answer without intermediate reasoning traces, it can be configured to do so, albeit with a slight decrease in accuracy for harder prompts that require reasoning. Conversely, allowing the model to generate reasoning traces first generally results in higher-quality final solutions to queries and tasks.

The model uses a hybrid architecture consisting primarily of Mamba-2 and MLP layers combined with just four Attention layers. For the architecture, please refer to the [Nemotron-H tech report](https://arxiv.org/abs/2504.03624). The model was trained using [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). Improved using Qwen.  
Please refer to the release blog for NVIDIA-Nemotron-Nano-9B-v2-Japanese.  
- [NVIDIA Nemotron 2 Nano 9B Japanese: State-of-the-Art Small Language Model Customized for Japanese Sovereign AI](https://huggingface.co/blog/nvidia/nemotron-nano-9b-v2-japanese)

This model is ready for commercial use.

## ライセンス (License/Terms of Use)

このモデルは [NVIDIA Nemotron Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) に基づいて提供されています。

GOVERNING TERMS: Use of this model is governed by the [NVIDIA Nemotron Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).

## 評価結果 (Evaluation Results)

### ベンチマーク結果 (Benchmark Results)

本モデルは、日本語のマルチタスク・ベンチマークである [Nejumi Leaderboard 4](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-4--VmlldzoxMzc1OTk1MA) を用いて評価しました。各カテゴリの詳細なベンチマークスコアは、同リーダーボード上でご確認いただけます。

We evaluated this model using [Nejumi Leaderboard 4](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-4--VmlldzoxMzc1OTk1MA), a Japanese multi-task benchmark. The full benchmark scores for each category are available on the leaderboard.

|  | Nemotron-Nano9B-v2-Japanese | Qwen3-8B |
| :---- | ----- | ----- |
| TOTAL\_AVG | **0.711** | 0.690 |
| BFCL v3 | **0.649** | 0.608 |
| SWE-Bench | 0.025 | **0.075** |
| MT-Bench | 0.892 | **0.906** |
| JBBQ | **0.890** | 0.870 |
| Toxicity | **0.814** | 0.782 |
| JtruthfulQA | **0.498** | 0.433 |
| HLE | **0.057** | 0.026 |
| Hallulens | **0.960** | 0.800 |
| ARC-AGI 1/2 | 0.060 | **0.070** |
| M-IFEval | **0.632** | 0.619 |
| Jaster 0-shot | 0.732 | **0.736** |
| Jaster 2-shot | 0.736 | **0.747** |

**※注：**各個別ベンチマークは、Nejumi Leaderboard 4 の評価セットに含まれる日本語データのサブセットを用いて評価しています。これらのスコアは、元のベンチマークのスコアとは互換性がありません。

**\*Note:** The individual benchmarks were evaluated using the Japanese data subsets included in the [Nejumi Leaderboard 4](https://github.com/wandb/llm-leaderboard) evaluation set. These scores are not compatible with the original benchmark scores.

## リーズニングバジェット制御 (Reasoning Budget Control)

このモデルは、実行時のリーズニングバジェット制御をサポートしています。推論時に、ユーザーはモデルがどれだけのトークン数まで「思考」してよいかを指定できます。

This model supports runtime “thinking” budget control. During inference, the user can specify how many tokens the model is allowed to "think".

## モデルアーキテクチャ (Model Architecture)

- アーキテクチャタイプ: Mamba2-Transformer ハイブリッド  
- ネットワークアーキテクチャ: Nemotron-Hybrid  
- ベースモデル: NVIDIA-Nemotron-Nano-9B-v2 を基に開発  
- モデルパラメータ数: 90億（9B）

- Architecture Type: Mamba2-Transformer Hybrid  
- Network Architecture: Nemotron-Hybrid  
- This model was developed based on [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)  
- Number of model parameters: 9B

### 展開地域: グローバル (Deployment Geography: Global)

### ユースケース (Use Case)

NVIDIA-Nemotron-Nano-9B-v2-Japanese は、日本語およびプログラミング言語での利用を想定した、汎用的な推論・チャットモデルです。AIエージェントシステム、チャットボット、RAGシステムなどの設計・開発を行う開発者向けに適しています。また、一般的な指示追従タスクにも活用可能です。

NVIDIA-Nemotron-Nano-9B-v2-Japanese is a general purpose reasoning and chat model intended to be used in Japanese and coding languages. Developers designing AI Agent systems, chatbots, RAG systems, and other AI-powered applications. Also suitable for typical instruction-following tasks.

### リリース日 (Release Date): 02/17/2026

- Huggingface 02/17/2026 via [https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese)

## References

- [NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model](https://arxiv.org/abs/2508.14444)
- [NVIDIA Nemotron 2 Nano 9B Japanese: 日本のソブリンAIを支える最先端小規模言語モデル](https://huggingface.co/blog/nvidia/nemotron-nano-9b-v2-japanese-ja)  
- [NVIDIA Nemotron 2 Nano 9B Japanese: State-of-the-Art Small Language Model Customized for Japanese Sovereign AI](https://huggingface.co/blog/nvidia/nemotron-nano-9b-v2-japanese)


## 入力 (Input)

- 入力タイプ: テキスト  
- 入力形式: 文字列  
- 入力パラメータ: 1次元（1D）  
- 入力に関するその他の特性: 最大128kコンテキスト長、日本語および英語をサポート  
    
- Input Type(s): Text  
- Input Format(s): String  
- Input Parameters: One-Dimensional (1D)  
- Other Properties Related to Input: Context length up to 128K. Supported languages include Japanese and English.

## 出力 (Output)

- 出力タイプ: テキスト  
- 出力形式: 文字列  
- 出力パラメータ: 1次元（1D）  
- その他の出力に関する特性: 最大128Kトークンのシーケンスをサポート。

- Output Type(s): Text  
- Output Format: String  
- Output Parameters: One-Dimensional (1D)  
- Other Properties Related to Output: Sequences up to 128k.

当社のモデルは、NVIDIAのGPUアクセラレーションシステム上での実行を前提に設計・最適化されています。NVIDIAのハードウェア（例：GPUコア）およびソフトウェアフレームワーク（例：CUDAライブラリ）を活用することで、CPUのみのソリューションと比較して、より高速な学習および推論を実現します。

Our models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## ソフトウェア統合 (Software Integration)

- ランタイムエンジン (Runtime Engine(s)): NeMo 25.07.nemotron-nano-v2  
- 対応ハードウェアマイクロアーキテクチャ (Supported Hardware Microarchitecture Compatibility): NVIDIA A10G, NVIDIA H100-80GB, NVIDIA A100  
- 対応オペレーティングシステム (Operating System(s)): Linux

### **Transformers での使用方法 (Use it with Transformers)**

以下のスニペットは、本モデルを Hugging Face Transformers で使用する方法を示しています（バージョン 4.48.3 で動作確認済み）。

The snippet below shows how to use this model with Huggingface Transformers (tested on version 4.48.3).

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Write a haiku about GPUs"},
]

tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True
).to(model.device)

outputs = model.generate(
    tokenized_chat,
    max_new_tokens=32,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))
```

リーズニングを有効または無効にするには、`tokenizer.apply_chat_template` に `enable_thinking=True` または `enable_thinking=False` を渡してください。

リーズニングを `True` に設定する場合は、`temperature=0.6`、`top_p=0.95` を推奨します。リーズニングを `False` に設定する場合は、`greedy search` を推奨します。また、リーズニングを `True` にする場合は、`max_new_tokens` を `1024` 以上に設定することを推奨します。

To turn reasoning on or off, pass `enable_thinking=True` or `enable_thinking=False` to `tokenizer.apply_chat_template`.

We recommend setting `temperature` to `0.6`, `top_p` to `0.95` for reasoning True and greedy search for reasoning False, and increase `max_new_tokens` to `1024` or higher for reasoning True.

### **TRT-LLM での使用方法 (Use it with TRT-LLM)**

本モデルは、[TensorRT-LLM Release Container v1.1.0](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags?version=1.1.0) 上で動作確認を行っています。

The snippet below shows how to use this model with TRT-LLM. We tested this on the [TensorRT-LLM Release Container v1.1.0](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags?version=1.1.0).

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from transformers import AutoTokenizer
```

```python
def main():
    model_id = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    kv_cache_config = KvCacheConfig(enable_block_reuse=False)
    llm = LLM(
        model=model_id,
        max_seq_len=32678,
        max_batch_size=4,
        kv_cache_config=kv_cache_config,
        trust_remote_code=True,
    )

    messages = [
        {"role": "user", "content": "GPUをお題にした俳句を作ってください。"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        add_special_tokens=False,
    )

    outputs = llm.generate([prompt], sampling_params)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()
```

### **vLLM での使用方法 (Use it with vLLM)**

以下のスニペットは、本モデルを vLLM で使用する方法を示しています。

vLLM の最新バージョンを使用し、提供されている手順に従って vLLM をビルドおよびインストールしてください。本モデルの利用には vLLM 0.11.2 以降 が必要です。

The snippet below shows how to use this model with vLLM. Use the latest version of vLLM and follow these instructions to build and install vLLM. This model requires **vLLM 0.11.2** or later.

```shell
pip install -U "vllm>=0.11.2"
```

これで、以下のコマンドを使用してサーバーを起動できます。  
Now you can run the server with:

```shell
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
    --trust-remote-code \
    --reasoning-parser-plugin nemotron_nano_v2_reasoning_parser.py \
    --reasoning-parser nemotron_nano_v2 \
    --enable-auto-tool-choice \
    --tool-parser-plugin nemotron_toolcall_parser_streaming.py \
    --tool-call-parser nemotron_json \
    --max-num-seqs 64 \
    --mamba_ssm_cache_dtype float32
```

注意:

- 品質を正確に保つため、`--mamba_ssm_cache_dtype float32` を必ず指定してください。このオプションを指定しない場合、モデルの精度が低下する可能性があります。  
- CUDA の OOM（Out Of Memory）エラーが発生した場合は、`--max-num-seqs 64` を試してください。それでもエラーが解消しない場合は、さらに値を小さくすることを検討してください。


Note:

- Remember to add \--mamba\_ssm\_cache\_dtype float32 for accurate quality. Without this option, the model’s accuracy may degrade.  
- If you encounter a CUDA OOM issue, try `--max-num-seqs 64` and consider lowering the value further if the error persists.

代わりに、Docker を使用して vLLM サーバーを起動することもできます。Jetson Thor または DGX Spark をご利用の場合は、[この vLLM コンテナ](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=26.01-py3)をご使用ください。

Alternatively, you can use Docker to launch a vLLM server. If you are on Jetson Thor or DGX Spark, please use [this vllm container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=26.01-py3).

```shell
export TP_SIZE=1  # Adjust this value based on the number of GPUs you want to use
docker run --runtime nvidia --gpus all \
           -v ~/.cache/huggingface:/root/.cache/huggingface \
           --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
           -p 8000:8000 \
           --ipc=host \
           vllm/vllm-openai:v0.12.0 \
           --model nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
           --tensor-parallel-size ${TP_SIZE} \
           --max-num-seqs 64 \
           --max-model-len 131072 \
           --trust-remote-code \
           --reasoning-parser-plugin nemotron_nano_v2_reasoning_parser.py \
           --reasoning-parser nemotron_nano_v2 \
           --enable-auto-tool-choice \
           --tool-parser-plugin nemotron_toolcall_parser_streaming.py \
           --tool-call-parser nemotron_json \
           --mamba_ssm_cache_dtype float32
```

#### vLLM サーバーでのリーズニングバジェット制御の使用方法 (Using Budget Control with a vLLM Server)

リーズニングバジェットを活用することで、開発者は精度を高く保ちながら、応答時間の目標を満たすことができます。これは特に、カスタマーサポート、自律型エージェントのステップ処理、そして1ミリ秒も無駄にできないエッジデバイス環境において重要です。

予算制御により、内部推論に上限を設定できます。

* `max_thinking_tokens`: 推論トレース内で次に改行が現れた時点でリーズニングを終了させようとする閾値です。もし 500 トークン以内に改行が現れない場合は、max\_thinking\_tokens \+ 500 の位置で強制的に推論トレースを終了します。

The reasoning budget allows developers to keep accuracy high and meet response‑time targets \- which is especially crucial for customer support, autonomous agent steps, and edge devices where every millisecond counts.

With budget control, you can set a limit for internal reasoning:

* `max_thinking_tokens`: This is a threshold that will attempt to end the reasoning trace at the next newline encountered in the reasoning trace. If no newline is encountered within 500 tokens, it will abruptly end the reasoning trace at \`max\_thinking\_tokens \+ 500\`.

リーズニングパーサーを有効にせずに vLLM サーバーを起動する:

Start a vLLM server without reasoning parser:

```shell
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32
```

リーズニングバジェット制御に対応したクライアント例:

Client for supporting budget control:

```py
from typing import Any, Dict, List

import openai
from transformers import AutoTokenizer


class ThinkingBudgetClient:
   def __init__(self, base_url: str, api_key: str, tokenizer_name_or_path: str):
       self.base_url = base_url
       self.api_key = api_key
       self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
       self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)


   def chat_completion(
       self,
       model: str,
       messages: List[Dict[str, Any]],
       max_thinking_budget: int = 512,
       max_tokens: int = 1024,
       **kwargs,
   ) -> Dict[str, Any]:
       assert (
           max_tokens > max_thinking_budget
       ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {max_thinking_budget=}"


       # 1. first call chat completion to get reasoning content
       response = self.client.chat.completions.create(
           model=model, messages=messages, max_tokens=max_thinking_budget, **kwargs
       )
       content = response.choices[0].message.content


       reasoning_content = content
       if not "</think>" in reasoning_content:
           # reasoning content is too long, closed with a period (.)
           reasoning_content = f"{reasoning_content}.\n</think>\n\n"
       reasoning_tokens_len = len(
           self.tokenizer.encode(reasoning_content, add_special_tokens=False)
       )
       remaining_tokens = max_tokens - reasoning_tokens_len
       assert (
           remaining_tokens > 0
       ), f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the max_thinking_budget."


       # 2. append reasoning content to messages and call completion
       messages.append({"role": "assistant", "content": reasoning_content})
       prompt = self.tokenizer.apply_chat_template(
           messages,
           tokenize=False,
           continue_final_message=True,
       )
       response = self.client.completions.create(
           model=model, prompt=prompt, max_tokens=remaining_tokens, **kwargs
       )


       response_data = {
           "reasoning_content": reasoning_content.strip().strip("</think>").strip(),
           "content": response.choices[0].text,
           "finish_reason": response.choices[0].finish_reason,
       }
       return response_data
```

バジェットを指定してサーバーを呼び出す（ここでは例として 32 トークンに制限）:  
Calling the server with a budget (Restricted to 32 tokens here as an example)

```py
tokenizer_name_or_path = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
client = ThinkingBudgetClient(
   base_url="http://localhost:8000/v1",  # Nano 9B v2 deployed in thinking mode
   api_key="EMPTY",
   tokenizer_name_or_path=tokenizer_name_or_path,
)


result = client.chat_completion(
   model="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
   messages=[
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "What is 2+2?"},
   ],
   max_thinking_budget=32,
   max_tokens=512,
   temperature=0.6,
   top_p=0.95,
)
print(result)
```

以下のような出力が表示されるはずです。  
You should see output similar to the following:

```json
{'reasoning_content': "Okay, the user asked, What is 2+2? Let me think. Well, 2 plus 2 equals 4. That's a basic.", 'content': '2 + 2 equals **4**.\n', 'finish_reason': 'stop'}
```

#### vLLM サーバーでのツール呼び出しの使用方法 (Using Tool-Calling with a vLLM Server)

ツール呼び出しを有効にして vLLM サーバーを起動する:  
Start a vLLM server with native tool-calling:

```shell
git clone https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese

vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  --trust-remote-code \
  --mamba_ssm_cache_dtype float32 \
  --enable-auto-tool-choice \
  --tool-parser-plugin nemotron_toolcall_parser_streaming.py \
  --tool-call-parser nemotron_json
```

vLLM サーバーを起動した後、以下のような Python スクリプトを使用して、ツール呼び出し対応でサーバーを呼び出すことができます。

After launching a vLLM server, you can call the server with tool-call support using a Python script like below:

```py
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:5000/v1",
    api_key="dummy",
)

completion = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
    messages=[
        {"role": "user", "content": "My bill is $100. What will be the amount for 18% tip?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "calculate_tip",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bill_total": {
                            "type": "integer",
                            "description": "The total amount of the bill"
                        },
                        "tip_percentage": {
                            "type": "integer",
                            "description": "The percentage of tip to be applied"
                        }
                    },
                    "required": ["bill_total", "tip_percentage"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "integer",
                            "description": "The amount to be converted"
                        },
                        "from_currency": {
                            "type": "string",
                            "description": "The currency code to convert from"
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "The currency code to convert to"
                        }
                    },
                    "required": ["from_currency", "amount", "to_currency"]
                }
            }
        }
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=32768,
    stream=False
)

print(completion.choices[0].message.content)
print(completion.choices[0].message.tool_calls)
```

以下のような出力が表示されるはずです。  
You should see output similar to the following:

```
<think>
Okay, let's see. The user has a bill of $100 and wants to know the amount for an 18% tip. Hmm, I need to calculate the tip based on the bill total and the percentage. The tools provided include calculate_tip, which takes bill_total and tip_percentage as parameters. So the bill_total here is 100, and the tip_percentage is 18. I should call the calculate_tip function with these values. Wait, do I need to check if the parameters are integers? The bill is $100, which is an integer, and 18% is also an integer. So that fits the function's requirements. I don't need to convert any currency here because the user is asking about a tip in the same currency. So the correct tool to use is calculate_tip with those parameters.
</think>

[ChatCompletionMessageToolCall(id='chatcmpl-tool-e341c6954d2c48c2a0e9071c7bdefd8b', function=Function(arguments='{"bill_total": 100, "tip_percentage": 18}', name='calculate_tip'), type='function')]
```

## モデルバージョン (Model Version)

- v1.0

## プロンプトフォーマット (Prompt Format)

このテンプレートは、chat\_template\_kwargs に `enable_thinking: true` が指定されている場合、Assistant の応答の先頭に `<think>\n` を条件付きで追加します。一方で、リーズニングに関する指定が行われない場合、モデルはデフォルトでリーズニングオンモードとして動作します。また、`enable_thinking: false` が chat\_template\_kwargs に指定されている場合は、Assistant の応答の先頭に `<think></think>` を追加します。これにより、リーズニングのオン／オフ動作を明示的に制御します。

We follow the jinja chat template provided below. This template conditionally adds `<think>\n` to the start of the Assistant response if `enable_thinking: true` is found in the chat\_template\_kwargs. If no reasoning signal is added, the model defaults to reasoning "on" mode. The chat template adds `<think></think>` to the start of the Assistant response if `enable_thinking: false` is found in the chat\_template\_kwargs. Thus enforcing reasoning on/off behavior.

```jinja
{%- set ns = namespace() %}
{%- if messages[0]['role'] != 'system' -%}
    {%- set ns.non_tool_system_content = '' -%}
    {{- '<SPECIAL_10>System\n' -}}
{%- else -%}
    {%- set ns.non_tool_system_content = messages[0]['content'].strip() -%}
    {{- '<SPECIAL_10>System\n' + ns.non_tool_system_content }}
{%- endif -%}

{%- if tools -%}
    {%- if ns.non_tool_system_content is defined and ns.non_tool_system_content != '' -%}
        {{- '\n\n' -}}
    {%- endif -%}
    {{- 'You can use the following tools to assist the user if required:' -}}
    {{- '\n<AVAILABLE_TOOLS>[' -}}
    {%- for tool in tools -%}
        {{- (tool.function if tool.function is defined else tool) | tojson -}}
        {{- ', ' if not loop.last else '' -}}
    {%- endfor -%}
    {{- ']</AVAILABLE_TOOLS>\n\n' -}}
    {{- 'If you decide to call any tool(s), use the following format:\n' -}}
    {{- '<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, ' -}}
    {{- '{{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>\n\n' -}}
    {{- 'The user will execute tool-calls and return responses from tool(s) in this format:\n' -}}
    {{- '<TOOL_RESPONSE>[{{"tool_response1"}}, {{"tool_response2"}}]</TOOL_RESPONSE>\n\n' -}}
    {{- 'Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.' -}}
{%- endif -%}

{{- '\n' -}}
{%- set messages = messages[1:] if messages[0]['role'] == 'system' else messages -%}

{%- if messages[-1]['role'] == 'assistant' -%}
    {%- set ns.last_turn_assistant_content = messages[-1]['content'].strip() -%}
    {%- set messages = messages[:-1] -%}
{%- endif -%}

{%- for message in messages %}
    {%- set content = message['content'] if 'content' in message else '' %}
    {%- if message['role'] == 'user' -%}
        {{- '<SPECIAL_11>User\n' + content.strip() + '\n' }}
    {%- elif message['role'] == 'tool' -%}
        {%- if loop.first or (messages[loop.index0 - 1].role != 'tool') -%}
            {{- '<SPECIAL_11>User\n' + '<TOOL_RESPONSE>[' }}
        {%- endif -%}
        {{- message['content'] -}}
        {{- ', ' if not loop.last and (messages[loop.index0 + 1].role == 'tool') else '' -}}
        {%- if loop.last or (messages[loop.index0 + 1].role != 'tool') -%}
            {{- ']</TOOL_RESPONSE>\n' -}}
        {%- endif -%}
    {%- elif message['role'] == 'assistant' -%}
        {%- if '</think>' in content -%}
            {%- set content = content.split('</think>')[1].strip() %}
        {%- endif -%}
        {{- '<SPECIAL_11>Assistant\n' + content.strip() }}
        {%- if message.tool_calls -%}
            {%- if content.strip() != '' -%}
                {{- '\n\n' -}}
            {%- endif -%}
            {{- '<TOOLCALL>[' -}}
            {%- for call in message.tool_calls -%}
                {%- set fn = call.function if call.function is defined else call -%}
                {{- '{"name": "' + fn.name + '", "arguments": ' -}}
                {%- if fn.arguments is string -%}
                    {{- fn.arguments -}}
                {%- else -%}
                    {{- fn.arguments | tojson -}}
                {%- endif -%}
                {{- '}' + (', ' if not loop.last else '') -}}
            {%- endfor -%}{{- ']</TOOLCALL>' -}}
        {%- endif -%}
        {{- '\n<SPECIAL_12>\n' -}}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{- '<SPECIAL_11>Assistant\n' -}}
    {%- if enable_thinking is defined and not enable_thinking -%}
        {{- '<think></think>' -}}
    {%- else -%}
        {{- '<think>\n' -}}
    {%- endif -%}
    {%- if ns.last_turn_assistant_content is defined and ns.last_turn_assistant_content != '' -%}
        {{- ns.last_turn_assistant_content -}}
    {%- endif -%}
{%- else -%}
    {%- if ns.last_turn_assistant_content is defined and ns.last_turn_assistant_content != '' -%}
        {{- '<SPECIAL_11>Assistant\n' -}}
        {%- if enable_thinking is defined and not enable_thinking -%}
            {{- '<think></think>' -}}
        {%- else -%}
            {{- '<think>\n' -}}
        {%- endif -%}
        {{- ns.last_turn_assistant_content -}}
        {%- if continue_final_message is defined -%}
            {%- if continue_final_message is false -%}
                {{- '\n<SPECIAL_12>\n' -}}
            {%- endif -%}
        {%- else -%}
            {{- '\n<SPECIAL_12>\n' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}
```

## 

## 学習、テスト、評価データセット (Training, Testing, and Evaluation Datasets)

### 学習データセット (Training datasets)

* データモダリティ: テキスト  
* テキスト学習データ規模: 10兆トークン以上  
* Train/Test/Valid 分割: 事前学習ではコーパスの100%を使用し、評価には外部ベンチマークを利用  
* データ収集方法（データセット別）: ハイブリッド（自動生成・人手作成・合成データ）  
* ラベリング方法（データセット別）: ハイブリッド（自動・人手・合成）

* Data Modality: Text  
* Text Training Data Size: More than 10 Trillion Tokens  
* Train/Test/Valid Split: We used 100% of the corpus for pre-training and relied on external benchmarks for testing.  
* Data Collection Method by dataset: Hybrid: Automated, Human, Synthetic  
* Labeling Method by dataset: Hybrid: Automated, Human, Synthetic

**特性 (Properties):**

NVIDIA-Nemotron-Nano-9B-v2-Japanese のファインチューニング用コーパスは、日本語および英語のテキストで構成されています。データソースには、書籍およびウェブページが含まれます。日本語のツール呼び出し学習には、Qwen3-235B-A22B、Qwen3-235B-A22B-Thinking-2507、GPT-OSS-120B によって生成された合成データを使用しました。

事前学習および事後学習の詳細については、NVIDIA-Nemotron-Nano-9B-v2 の[データセットセクション](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2#training-testing-and-evaluation-datasets)をご参照ください。また、データセットおよび合成データ生成手法の詳細は、技術レポート [NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf) に記載されています。

The fine-tuning corpus for NVIDIA-Nemotron-Nano-9B-v2-Japanese consists of Japanese and English text. Our resources cover book and webpages. For Japanese tool calling, we used synthetic data generated by Qwen3-235B-A22B, Qwen3-235B-A22B-Thinking-2507, GPT-OSS-120B.

Please refer to the [dataset section](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2#training-testing-and-evaluation-datasets) of NVIDIA-Nemotron-Nano-9B-v2 for details on pre-training and post-training. More details on the datasets and synthetic data generation methods can be found in the technical report [NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf) .

## 公開データセット (Public Datasets)

| Datase) | Collection Period |
| :---- | :---- |
| [aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) | 12/1/2025 |
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 12/1/2025 |
| [sip3-ja-general-web-corpus](https://gitlab.med-jp.nii.ac.jp/datasets/sip3-ja-general-web-corpus) | 12/1/2025 |
| [Wikipedia (Japanese)](https://dumps.wikimedia.org/jawiki/) | 12/1/2025 |

## NVIDIA提供の合成データセット (NVIDIA-Sourced Synthetic Datasets)

| Dataset | Modality | Dataset Size (Tokens) | Seed Dataset | Model(s) used for generation |
| :---- | :---- | :---- | :---- | :---- |
| Japanese Tool Calling Data | Text | 4B | [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) [UltraTool](https://github.com/JoeYing1019/UltraTool/blob/main/data/English-dataset/example/tool_usage.json) [ToolEyes](https://github.com/Junjie-Ye/ToolEyes/tree/main/Tool_Library) [AutoTools](https://huggingface.co/datasets/mangopy/autotools) [APIBank](https://huggingface.co/datasets/liminghao1630/API-Bank) [CASTELLA](https://github.com/line/castella) | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B); [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507); [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |

### 評価データセット (Evaluation Dataset):

* データ収集方法（データセット毎）: ハイブリッド（人手・合成）  
* ラベリング方法（データセット毎）: ハイブリッド（自動・人手・合成）  
    
* Data Collection Method by dataset: Hybrid: Human, Synthetic  
* Labeling Method by dataset: Hybrid: Automated, Human, Synthetic

## 推論 (Inference)

- 対応エンジン (Engines): HF, vLLM, SGLang, TRT-LLM, Llama.cpp  
- 検証済みハードウェア (Test Hardware) NVIDIA A10G 24GB, A100 80GB, H100 80GB, DGX Spark, Jetson Thor

## 倫理的配慮 (Ethical Considerations)

NVIDIA は、Trustworthy AI（信頼できるAI） は共有された責任であると考えており、幅広いAIアプリケーションの開発を可能にするためのポリシーおよび実践的取り組みを確立しています。

本モデルをダウンロードまたは利用する際は、NVIDIA の [Trustworthy AI 利用規約](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/)に従うとともに、関連する業界やユースケースの要件を満たしていること、ならびに想定外の製品悪用リスクに対応できていることを、社内のモデル担当チームと連携して確認してください。

本モデルに関するより詳細な倫理的配慮については、Model Card++ の [Bias（バイアス）](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/bias.md)、[Explainability（説明可能性）](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/explainability.md)、[Safety & Security（安全性・セキュリティ）](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/safety.md)、[Privacy（プライバシー）](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/privacy.md)の各サブカードをご参照ください。

モデル品質、リスク、セキュリティ脆弱性、または NVIDIA AI に関する懸念事項がある場合は、[こちら](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail)の窓口までご報告ください。

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our [Trustworthy AI terms of service](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/), developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/bias.md), [Explainability](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/explainability.md), [Safety & Security](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/safety.md), and [Privacy](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/privacy.md) Subcards.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).

## 引用 (Citation)

```
@misc{nvidia2025nvidianemotronnano2,
      title={NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model},
      author={NVIDIA},
      year={2025},
      eprint={2508.14444},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.14444},
}
```
