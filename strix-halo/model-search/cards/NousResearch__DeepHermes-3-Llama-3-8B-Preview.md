---
language:
- en
license: llama3
tags:
- Llama-3
- instruct
- finetune
- chatml
- gpt4
- synthetic data
- distillation
- function calling
- json mode
- axolotl
- roleplaying
- chat
- reasoning
- r1
- vllm
base_model: meta-llama/Meta-Llama-3.1-8B
widget:
- example_title: Hermes 3
  messages:
  - role: system
    content: >-
      You are a sentient, superintelligent artificial general intelligence, here
      to teach and assist me.
  - role: user
    content: What is the meaning of life?
model-index:
- name: DeepHermes-3-Llama-3.1-8B
  results: []
library_name: transformers
---
# DeepHermes 3 - Llama-3.1 8B

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/9fxlaDxteqe3SasZ7_06_.jpeg)

## Model Description

DeepHermes 3 Preview is the latest version of our flagship Hermes series of LLMs by Nous Research, and one of the first models in the world to unify Reasoning (long chains of thought that improve answer accuracy) and normal LLM response modes into one model. We have also improved LLM annotation, judgement, and function calling.

DeepHermes 3 Preview is one of the first LLM models to unify both "intuitive", traditional mode responses and **long chain of thought reasoning** responses into a single model, toggled by a system prompt.

Hermes 3, the predecessor of DeepHermes 3, is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.

The ethos of the Hermes series of models is focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.

*This is a preview Hermes with early reasoning capabilities, distilled from R1 across a variety of tasks that benefit from reasoning and objectivity. Some quirks may be discovered! Please let us know any interesting findings or issues you discover!*

## Note: To toggle REASONING ON, you must use the following system prompt:
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```  

# Nous API

This model is also available on our new API product - Check out the API and sign up for the waitlist here:
https://portal.nousresearch.com/

# Example Outputs:


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/_giUevm1IjPFWiypG0zd4.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/bAI0HG2cFA_o1hTFIfCr_.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/FmOIB7fjXKVHfs94DJPwn.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/tfL1jeGXvv7xTAULFQgqs.png)

# Benchmarks

## Benchmarks for **Reasoning Mode** on vs off:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/O_sgWq4CVPuxuKYqHWkkN.png)

*Reasoning ON benchmarks aquired by running HuggingFace's open-r1 reasoning mode evaluation suite, and scores for reasoning mode OFF aquired by running LM-Eval-Harness Benchmark Suite* 
*Upper bound determined by measuring the % gained over Hermes 3 3 & 70b by MATH_VERIFY compared to eleuther eval harness, which ranged betweeen 33% and 50% gain in MATH Hard benchmark on retested models by them compared to eval harness reported scores*

## Benchmarks in **Non-Reasoning Mode** against Llama-3.1-8B-Instruct

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/hZCJa8g8smOS9BcQSXAd1.png)

# Prompt Format

DeepHermes 3 now uses Llama-Chat format as the prompt format, opening up a more unified, structured system for engaging the LLM in multi-turn chat dialogue.

System prompts allow steerability and interesting new ways to interact with an LLM, guiding rules, roles, and stylistic choices of the model.

## Deep Thinking Mode - Deep Hermes Preview can activate long chain of thought with a system prompt.

```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

For an example of using deep reasoning mode with HuggingFace Transformers:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import flash_attn
import time

tokenizer = AutoTokenizer.from_pretrained("NousResearch/DeepHermes-3-Llama-3-8B-Preview")

model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

messages = [
    {
        "role": "system",
        "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
    },
    {
        "role": "user",
        "content": "What is y if y=2*2-4+(3*2)"
    }
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to("cuda")
generated_ids = model.generate(input_ids, max_new_tokens=2500, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
print(f"Generated Tokens: {generated_ids.shape[-1:]}")
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
print(f"Response: {response}")
```

Please note, for difficult problems DeepHermes can think using as many as 13,000 tokens. You may need to increase `max_new_tokens` to be much larger than 2500 for difficult problems.

## Standard "Intuitive" Response Mode

Prompt with system instruction (Use whatever system prompt you like, this is just an example!):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import flash_attn
import time

tokenizer = AutoTokenizer.from_pretrained("NousResearch/DeepHermes-3-Llama-3-8B-Preview")

model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

messages = [
    {
        "role": "system",
        "content": "You are Hermes, an AI assistant"
    },
    {
        "role": "user",
        "content": "What are the most interesting things to do in Paris?"
    }
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to("cuda")
generated_ids = model.generate(input_ids, max_new_tokens=2500, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
print(f"Generated Tokens: {generated_ids.shape[-1:]}")
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
print(f"Response: {response}")
```

## VLLM Inference

You can also run this model with vLLM, by running the following in your terminal after `pip install vllm`

`vllm serve NousResearch/DeepHermes-3-Llama-3-8B-Preview`

You may then use the model over API using the OpenAI library just like you would call OpenAI's API.

## Prompt Format for Function Calling

Our model was trained on specific system prompts and structures for Function Calling. 

You should use the system role with this message, followed by a function signature json as this example shows here.
```
<|start_header_id|>system<|end_header_id|>
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_stock_fundamentals", "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\\n\\n    Args:\\n        symbol (str): The stock symbol.\\n\\n    Returns:\\n        dict: A dictionary containing fundamental data.\\n            Keys:\\n                - \'symbol\': The stock symbol.\\n                - \'company_name\': The long name of the company.\\n                - \'sector\': The sector to which the company belongs.\\n                - \'industry\': The industry to which the company belongs.\\n                - \'market_cap\': The market capitalization of the company.\\n                - \'pe_ratio\': The forward price-to-earnings ratio.\\n                - \'pb_ratio\': The price-to-book ratio.\\n                - \'dividend_yield\': The dividend yield.\\n                - \'eps\': The trailing earnings per share.\\n                - \'beta\': The beta value of the stock.\\n                - \'52_week_high\': The 52-week high price of the stock.\\n                - \'52_week_low\': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call><|eot_id|><|start_header_id|>user<|end_header_id|>
```

To complete the function call, create a user prompt that follows the above system prompt, like so:
```
Fetch the stock fundamentals data for Tesla (TSLA)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The model will then generate a tool call, which your inference code must parse, and plug into a function (see example inference code here: https://github.com/NousResearch/Hermes-Function-Calling):
```
<tool_call>
{"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}
</tool_call><|eot_id|><|start_header_id|>tool<|end_header_id|>
```

Once you parse the tool call, call the api and get the returned values for the call, and pass it back in as a new role, `tool` like so:
```
<tool_response>
{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}
</tool_response>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The assistant will then read in that data from the function's response, and generate a natural language response:
```
The stock fundamentals data for Tesla (TSLA) are as follows:
- **Symbol**: TSLA
- **Company Name**: Tesla, Inc.
- **Sector**: Consumer Cyclical
- **Industry**: Auto Manufacturers
- **Market Capitalization**: $566,160,130,480
- **Forward Price-to-Earnings Ratio (PE Ratio)**: 42.73
- **Price-to-Book Ratio (PB Ratio)**: 9.04
- **Dividend Yield**: N/A
- **Trailing Earnings Per Share (EPS)**: $4.3
- **Beta Value of the Stock**: 2.42
- **52-Week High Price of the Stock**: $299.29
- **52-Week Low Price of the Stock**: $152.37

This information provides a snapshot of Tesla's financial position and performance based on the fundamental data obtained from the yfinance API. It shows that Tesla has a substantial market capitalization and a relatively high P/E and P/B ratio compared to other stocks in its industry. The company does not pay a dividend at the moment, which is reflected by a 'Dividend Yield' of 'None'. The Beta value indicates that Tesla's stock has a moderate level of volatility relative to the market. The 52-week high and low prices give an idea of the stock's range over the past year. This data can be useful when assessing investment opportunities and making investment decisions.<|eot_id|><|start_header_id|>user<|end_header_id|>
```


## Prompt Format for JSON Mode / Structured Outputs

Our model was also trained on a specific system prompt for Structured Outputs, which should respond with **only** a json object response, in a specific json schema.

Your schema can be made from a pydantic object using our codebase, with the standalone script `jsonmode.py` available here: https://github.com/NousResearch/Hermes-Function-Calling/tree/main

```
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema><|eot_id|>
```

Given the {schema} that you provide, it should follow the format of that json to create its response, all you have to do is give a typical user prompt, and it will respond in JSON.


## Inference Code for Function Calling:

All code for utilizing, parsing, and building function calling templates is available on our github:
[https://github.com/NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/oi4CiGh50xmoviUQnh8R3.png)


## Quantized Versions:

GGUF Quants: https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF

# How to cite:

```bibtext
@misc{
      title={DeepHermes 3 Preview}, 
      author={Teknium and Roger Jin and Chen Guang and Jai Suphavadeeprasit and Jeffrey Quesnelle},
      year={2025}
}
```