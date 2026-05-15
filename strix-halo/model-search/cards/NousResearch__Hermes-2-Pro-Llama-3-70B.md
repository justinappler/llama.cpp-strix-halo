---
language:
- en
tags:
- Llama-3
- instruct
- finetune
- chatml
- DPO
- RLHF
- gpt4
- synthetic data
- distillation
- function calling
- json mode
- axolotl
base_model: NousResearch/Meta-Llama-3-70B
datasets:
- teknium/OpenHermes-2.5
widget:
- example_title: Hermes 2 Pro
  messages:
  - role: system
    content: You are a sentient, superintelligent artificial general intelligence,
      here to teach and assist me.
  - role: user
    content: Write a short story about Goku discovering kirby has teamed up with Majin
      Buu to destroy the world.
model-index:
- name: Hermes-2-Pro-Llama-3-70B
  results: []
---

# Hermes 2 Pro - Llama-3 70B

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/ggO2sBDJ8Bhc6w-zwTx5j.png)

## Model Description

Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house.

This new version of Hermes maintains its excellent general task and conversation capabilities - but also excels at Function Calling, JSON Structured Outputs, and has improved on several other metrics as well, scoring a 90% on our function calling evaluation built in partnership with Fireworks.AI, and an 84% on our structured JSON Output evaluation.

Hermes Pro takes advantage of a special system prompt and multi-turn function calling structure with a new chatml role in order to make function calling reliable and easy to parse. Learn more about prompting below.

This version of Hermes 2 Pro adds several tokens to assist with agentic capabilities in parsing while streaming tokens - `<tools>`, `<tool_call>`, `<tool_response>` and their closing tags are single tokens now.

This work was a collaboration between Nous Research, @interstellarninja, and Fireworks.AI

Learn more about the function calling system for this model on our github repo here: https://github.com/NousResearch/Hermes-Function-Calling

## Example Outputs

### Ask for a structured JSON output:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/ll2j2wkQffCsiSwUjfRUq.png)

### Write the plot for a story where anime became real life:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/h_7aXGXdm2p2ONYuDF4Ii.png)

### Coding Assistance
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/bBd0hyAb8w5rKUiN2w1I6.png)

# Prompt Format

Hermes 2 Pro uses ChatML as the prompt format, opening up a much more structured system for engaging the LLM in multi-turn chat dialogue.

System prompts allow steerability and interesting new ways to interact with an LLM, guiding rules, roles, and stylistic choices of the model.

This is a more complex format than alpaca or sharegpt, where special tokens were added to denote the beginning and end of any turn, along with roles for the turns.

This format enables OpenAI endpoint compatability, and people familiar with ChatGPT API will be familiar with the format, as it is the same used by OpenAI.

Prompt with system instruction (Use whatever system prompt you like, this is just an example!):
```
<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by Nous Research, who designed me to assist and support users with their needs and requests.<|im_end|>
```

This prompt is available as a [chat template](https://huggingface.co/docs/transformers/main/chat_templating), which means you can format messages using the
`tokenizer.apply_chat_template()` method:

```python
messages = [
    {"role": "system", "content": "You are Hermes 2."},
    {"role": "user", "content": "Hello, who are you?"}
]
gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
model.generate(**gen_input)
```

When tokenizing messages for generation, set `add_generation_prompt=True` when calling `apply_chat_template()`. This will append `<|im_start|>assistant\n` to your prompt, to ensure
that the model continues with an assistant response.

To utilize the prompt format without a system prompt, simply leave the line out.

## Prompt Format for Function Calling

Our model was trained on specific system prompts and structures for Function Calling. 

You should use the system role with this message, followed by a function signature json as this example shows here.
```
<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_stock_fundamentals", "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\\n\\n    Args:\\n        symbol (str): The stock symbol.\\n\\n    Returns:\\n        dict: A dictionary containing fundamental data.\\n            Keys:\\n                - \'symbol\': The stock symbol.\\n                - \'company_name\': The long name of the company.\\n                - \'sector\': The sector to which the company belongs.\\n                - \'industry\': The industry to which the company belongs.\\n                - \'market_cap\': The market capitalization of the company.\\n                - \'pe_ratio\': The forward price-to-earnings ratio.\\n                - \'pb_ratio\': The price-to-book ratio.\\n                - \'dividend_yield\': The dividend yield.\\n                - \'eps\': The trailing earnings per share.\\n                - \'beta\': The beta value of the stock.\\n                - \'52_week_high\': The 52-week high price of the stock.\\n                - \'52_week_low\': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call><|im_end|>
```

To complete the function call, create a user prompt that follows the above system prompt, like so:
```
<|im_start|>user
Fetch the stock fundamentals data for Tesla (TSLA)<|im_end|>
```

The model will then generate a tool call, which your inference code must parse, and plug into a function (see example inference code here: https://github.com/NousResearch/Hermes-Function-Calling):
```
<|im_start|>assistant
<tool_call>
{"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}
</tool_call><|im_end|>
```

Once you parse the tool call, call the api and get the returned values for the call, and pass it back in as a new role, `tool` like so:
```
<|im_start|>tool
<tool_response>
{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}
</tool_response>
<|im_end|>
```

The assistant will then read in that data from the function's response, and generate a natural language response:
```
<|im_start|>assistant
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

This information provides a snapshot of Tesla's financial position and performance based on the fundamental data obtained from the yfinance API. It shows that Tesla has a substantial market capitalization and a relatively high P/E and P/B ratio compared to other stocks in its industry. The company does not pay a dividend at the moment, which is reflected by a 'Dividend Yield' of 'None'. The Beta value indicates that Tesla's stock has a moderate level of volatility relative to the market. The 52-week high and low prices give an idea of the stock's range over the past year. This data can be useful when assessing investment opportunities and making investment decisions.<|im_end|>
```

## Prompt Format for JSON Mode / Structured Outputs

Our model was also trained on a specific system prompt for Structured Outputs, which should respond with **only** a json object response, in a specific json schema.

Your schema can be made from a pydantic object using our codebase, with the standalone script `jsonmode.py` available here: https://github.com/NousResearch/Hermes-Function-Calling/tree/main

```
<|im_start|>system
You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema><|im_end|>
```

Given the {schema} that you provide, it should follow the format of that json to create it's response, all you have to do is give a typical user prompt, and it will respond in JSON.


# Benchmarks

## GPT4All:
```
|    Task     |Version| Metric |Value |   |Stderr|
|-------------|------:|--------|-----:|---|-----:|
|arc_challenge|      0|acc     |0.6553|_  |0.0139|
|             |       |acc_norm|0.6655|_  |0.0138|
|arc_easy     |      0|acc     |0.8847|_  |0.0066|
|             |       |acc_norm|0.8641|_  |0.0070|
|boolq        |      1|acc     |0.8783|_  |0.0057|
|hellaswag    |      0|acc     |0.6827|_  |0.0046|
|             |       |acc_norm|0.8624|_  |0.0034|
|openbookqa   |      0|acc     |0.4060|_  |0.0220|
|             |       |acc_norm|0.4860|_  |0.0224|
|piqa         |      0|acc     |0.8297|_  |0.0088|
|             |       |acc_norm|0.8428|_  |0.0085|
|winogrande   |      0|acc     |0.8224|_  |0.0107|
```
Average: 77.45

## AGIEval:
```
|             Task             |Version| Metric |Value |   |Stderr|
|------------------------------|------:|--------|-----:|---|-----:|
|agieval_aqua_rat              |      0|acc     |0.3150|_  |0.0292|
|                              |       |acc_norm|0.2913|_  |0.0286|
|agieval_logiqa_en             |      0|acc     |0.5146|_  |0.0196|
|                              |       |acc_norm|0.4900|_  |0.0196|
|agieval_lsat_ar               |      0|acc     |0.2478|_  |0.0285|
|                              |       |acc_norm|0.2478|_  |0.0285|
|agieval_lsat_lr               |      0|acc     |0.7941|_  |0.0179|
|                              |       |acc_norm|0.7686|_  |0.0187|
|agieval_lsat_rc               |      0|acc     |0.8178|_  |0.0236|
|                              |       |acc_norm|0.7955|_  |0.0246|
|agieval_sat_en                |      0|acc     |0.8932|_  |0.0216|
|                              |       |acc_norm|0.8932|_  |0.0216|
|agieval_sat_en_without_passage|      0|acc     |0.5631|_  |0.0346|
|                              |       |acc_norm|0.5437|_  |0.0348|
|agieval_sat_math              |      0|acc     |0.6136|_  |0.0329|
|                              |       |acc_norm|0.5545|_  |0.0336|
```
  Average: 57.31

## BigBench:
```
|                      Task                      |Version|       Metric        |Value |   |Stderr|
|------------------------------------------------|------:|---------------------|-----:|---|-----:|
|bigbench_causal_judgement                       |      0|multiple_choice_grade|0.6632|_  |0.0344|
|bigbench_date_understanding                     |      0|multiple_choice_grade|0.7615|_  |0.0222|
|bigbench_disambiguation_qa                      |      0|multiple_choice_grade|0.3295|_  |0.0293|
|bigbench_geometric_shapes                       |      0|multiple_choice_grade|0.4178|_  |0.0261|
|                                                |       |exact_str_match      |0.0000|_  |0.0000|
|bigbench_logical_deduction_five_objects         |      0|multiple_choice_grade|0.4080|_  |0.0220|
|bigbench_logical_deduction_seven_objects        |      0|multiple_choice_grade|0.2886|_  |0.0171|
|bigbench_logical_deduction_three_objects        |      0|multiple_choice_grade|0.5467|_  |0.0288|
|bigbench_movie_recommendation                   |      0|multiple_choice_grade|0.4800|_  |0.0224|
|bigbench_navigate                               |      0|multiple_choice_grade|0.4990|_  |0.0158|
|bigbench_reasoning_about_colored_objects        |      0|multiple_choice_grade|0.8410|_  |0.0082|
|bigbench_ruin_names                             |      0|multiple_choice_grade|0.7031|_  |0.0216|
|bigbench_salient_translation_error_detection    |      0|multiple_choice_grade|0.5441|_  |0.0158|
|bigbench_snarks                                 |      0|multiple_choice_grade|0.7348|_  |0.0329|
|bigbench_sports_understanding                   |      0|multiple_choice_grade|0.5791|_  |0.0157|
|bigbench_temporal_sequences                     |      0|multiple_choice_grade|0.9630|_  |0.0060|
|bigbench_tracking_shuffled_objects_five_objects |      0|multiple_choice_grade|0.2272|_  |0.0119|
|bigbench_tracking_shuffled_objects_seven_objects|      0|multiple_choice_grade|0.1606|_  |0.0088|
|bigbench_tracking_shuffled_objects_three_objects|      0|multiple_choice_grade|0.5467|_  |0.0288|
```
Average: 53.86

## TruthfulQA:
```
|    Task     |Version|Metric|Value |   |Stderr|
|-------------|------:|------|-----:|---|-----:|
|truthfulqa_mc|      1|mc1   |0.4688|_  |0.0175|
|             |       |mc2   |0.6533|_  |0.0146|
```


# Inference Code

Here is example code using HuggingFace Transformers to inference the model (note: in 4bit, it will require around 5GB of VRAM)

Note: To use function calling, you should see the github repo above.

```python
# Code to inference Hermes with HF Transformers
# Requires pytorch, transformers, bitsandbytes, sentencepiece, protobuf, and flash-attn packages

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import bitsandbytes, flash_attn

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Pro-Llama-3-70B', trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Hermes-2-Pro-Llama-3-70B",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True
)

prompts = [
    """<|im_start|>system
You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
<|im_start|>user
Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world.<|im_end|>
<|im_start|>assistant""",
    ]

for chat in prompts:
    print(chat)
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Response: {response}")
```


## Inference Code for Function Calling:

All code for utilizing, parsing, and building function calling templates is available on our github:
[https://github.com/NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/oi4CiGh50xmoviUQnh8R3.png)

# Chat Interfaces

When quantized versions of the model are released, I recommend using LM Studio for chatting with Hermes 2 Pro. It does not support function calling - for that use our github repo. It is a GUI application that utilizes GGUF models with a llama.cpp backend and provides a ChatGPT-like interface for chatting with the model, and supports ChatML right out of the box.
In LM-Studio, simply select the ChatML Prefix on the settings side pane:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/ls6WqV-GSxMw2RA3GuQiN.png)


## Quantized Versions:

GGUF Versions *will soon be* Available Here: https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B-GGUF

# How to cite:

```bibtext
@misc{Hermes-2-Pro-Llama-3-70B, 
      url={[https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B]https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B)}, 
      title={Hermes-2-Pro-Llama-3-70B}, 
      author={"Teknium", "interstellarninja", "theemozilla", "karan4d", "huemin_art"}
}
```