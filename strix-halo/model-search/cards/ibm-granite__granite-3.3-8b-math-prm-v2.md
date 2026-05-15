---
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
library_name: transformers
tags:
  - reward model
base_model:
  - ibm-granite/granite-3.3-8b-instruct
---
# Granite-3.3-8B-Math-PRM-v2

**Model Summary**

Granite 3.3 8B Math PRM v2 is a finetuned version of the 8-billion parameter language model, [Granite-3.3-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct), built for use a generative process reward model (PRM) for process supervision in mathematical reasoning. Crucially, this model has only been trained on curated data from sources with permissive licenses, and we release this model under a Apache 2.0 license. This model displays state-of-the-art performance in a best-of-N set up for math tasks.

This model can be used to asses the correctness of each step of a mathematical reasoning process, and shows strong performance on Best-of-N evaluations for a variety of generators on Math-500, as well as strong error identification performance in both [ProcessBench](https://arxiv.org/abs/2412.06559) and [PRMBench](https://arxiv.org/abs/2501.03124), with state-of-the-art performance in its size class. Although this model was trained for mathematical reasoning tasks, it also shows strong inference scaling performance on Code benchmarks, such as HunmanEval and LCBv5.

- Developers: Granite Alignment Team, IBM Research
- Release Date: Jan 1st, 2026
- License: Apache 2.0

**Supported Languages**

This model has specifically been finetuned for English, however the base model supports English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese.

**Intended Use**

Granite 3.3 8B Math PRM v2 is finetuned version of [Granite-3.3-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct), which gives the language model the ability of process supervision on mathematical reasoning steps by assessing the correctness of each step of a reasoning chain. At inference, the model takes a question and a response which can be broken down into generated steps, and for each step it determines whether the reasoning chain so far is correct (indicated by generating a single token, `Y`) or incorrect (indicated by generating `N`). The probability of generating `Y` can be treated as a numeric reward score in applications such as Best-of-N evaluation.

Before obtaining a response, the model expects the user generated prompt `"Is this response correct so far (Y/N)?"`, which should be added at the end of every step of the reasoning chain.

This model is an update to [granite-3.3-8b-lora-math-prm](https://huggingface.co/ibm-granite/granite-3.3-8b-lora-math-prm).

**Evaluation Results**

**a. Best-of-N Evaluation on Math-500**


We show the performance of MATH-500 with inference scaling on generations from granite-4.0-h-small and granite-4.0-h-tiny, and show strong gains over Majority Voting with both Best-of-N and Weighted Majority Voting using Granite-3.3-8B-Math-PRM-v2.


<img src="images/PRMv2_BoN.jpg" alt="PRM Performance on Math-500" />

We also compare the Best-of-N performance on Math-500 available PRMs on [Qwen-2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) generations, and show the superior performance of Granite-3.3-8B-Math-PRM-v2 over majority voting and best-of-N using similarly sized PRMs:

| | 2| 4| 8| 16| 32| 64| 128| 256|
|- | - |- |- |- |- |- |- |- |
| Majority Voting |75.8 | 81.6  | 84.6 | 85.2 | 85.4| 85.6	 | 86.0 | 85.6 |
| **Granite-3.3-8B-Math-PRM-v2** | 81.6	| 84.6	| **86.6**| **88.0** |**88.2**	|**89.2**	|**89.0**|	**89.6** |
| [Granite-3.3-8B-LoRA-Math-PRM](https://huggingface.co/ibm-granite/granite-3.3-8b-lora-math-prm)|  81.6 | 84.2 | 84.8	| 86.2 | 86.8 | 87.2 | 88.0 | 87.2 |
| [Qwen2.5-Math-PRM-7B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)| **82.0** | **84.8** | **86.6** | 87.0 | **88.2** | 89.0 | 88.8 | 89.0|
| [MathShepherd-Mistral-7B PRM 7B](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm)| 80.8 | 83.0 | 83.8 | 84.8 | 86.2 | 85.2 | 86.0 | 85.2 |
| [RLHFLow Llama3.1-8B-PRM-Deepseek-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data)| 80.6 | 82.4 | 	83.6 | 	85.2 | 	85.8 | 	85.8 | 	85.0 |	84.6 | 

**b. Best-of-N Evaluation on HumanEval and LCBv5**

We show the performance on two code tasks, HumanEval (Python) and LCBv5 with inference scaling on generations from granite-4.0-h-small. While this model is finetuned on Math data, Granite-3.3-8B-Math-PRM-v2 demonstrates strong gains over Majority Voting with both Best-of-N and Weighted Majority Voting. 

<img src="images/PRMv2_BoN_code.jpg" alt="PRM Performance on Math-500"/>

**c. ProcessBench**


<img src="images/PRMv2_Benchmarks.jpg" alt="PRM Performance on ProcessBench and PRMBench"  height="400"/> 

As shown above, Granite-3.3-8B-Math-PRM-v2 shows strong performance (top-2) on both ProcessBench and PRMBench compared to other models of the same parameter class, indicating a strong ability at error detection for reasoning tasks.

**d. PRMBench: Detailed Results**


| Model | Overall| NR. | NCL. | Avg (simplicity) | ES. | SC. | DC. | CI. | Avg (soundness) | PS. | DR. | MS. | Avg (sensitivity)  |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|------- |
| **Granite-3.3-8B-Math-PRM-v2** | 65.1	| 52.2	| **63.4**	| 57.8	| 69.7	| 65.8	| 64.4	| 70.8	| 67.7	| **61.0**	| 67.1	| 98.2	| 75.4
| [Granite-3.3-8B-LoRA-Math-PRM](https://huggingface.co/ibm-granite/granite-3.3-8b-lora-math-prm) | 64.5 | 	50.9 | 	61.5 | 	56.2 | 	69.1 | 	66.7 | 	64.7 | 	70.5 | 	67.8 | 	59.9 | 	65.9 | 	98.1 | 	74.7
| [Qwen2.5-Math-PRM-7B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) | **65.5** | 49.0 | 55.1 | 52.1 | **71.8** | **67.3** | 66.3 | **78.5** | **71.0** | 57.6 | **69.1** | **99.7** | **75.5**
| [Skywork-PRM-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B) | 65.1 | **56.4** | 62.8 | **59.6** | 69.4 | 67.1 | **67.7** | 69.9 | 68.5 | 60.9 | 65.8 | 93.2 | 73.7
| [Skywork-PRM-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B) | 61.1 | 52.0 | 56.4 | 54.2 | 64.8 | 64.9 | 63.3 | 66.5 | 64.9 | 57.5 | 63.3 | 91.1 | 70.7
| [ReasonEval-34B](https://huggingface.co/GAIR/ReasonEval-34B) | 60.5 | 54.8| 48.1 | 51.5 | 66.4 | 60.3 | 57.8 | 67.5 | 63.0 | 57.7 | 64.3 | 97.2 | 73.1 
| [ReasonEval-7B](https://huggingface.co/GAIR/ReasonEval-7B) | 60.1 | 61.0 | 50.1 | 55.6 | 62.1 | 65.9 | 61.5 | 66.0 | 63.9 | 55.7 | 58.0 | 99.5 | 71.1 
| [RLHFlow-PRM-Mistral-8B](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Mistral-Data) | 54.4 | 46.1 | 47.3 | 46.7 | 56.6 | 55.1 | 54.4 | 63.8 | 57.5 | 51.5 | 56.2 | 97.9 | 68.5 
| [RLHFlow-PRM-Deepseek-8B](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data) | 54.2 | 46.4 | 48.9 | 47.6 | 55.7 | 55.0 | 53.2 | 66.2 | 57.5 | 49.0 | 55.4 | 99.8 | 68.1 
| [MathShepherd-Mistral-7B](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm) | 47.0 | 44.0 | 50.3 | 47.1 | 49.4 | 44.5 | 41.3 | 47.7 | 45.7 | 47.2 | 48.6 | 86.1 | 60.7 

**Training Data**
The model is first finetuned on math and code data, such as [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), [Codeforces](https://huggingface.co/datasets/open-r1/codeforces), Code Contests, [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning), [PrimeIntellect](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset), and [TACO](https://huggingface.co/datasets/BAAI/TACO) on the standard language modeling objective. Then, we conduct RLVR on [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), PrimeIntellect, TACO, [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) and [Skywork-OR1-RL](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) Data. Finally, we train the model to predict the correctness of math problems, curating ttraining data from a diverse set of model responses to prompts from Math-specific datasets, specifically, [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) and NuminaMath. We leverage a diverse set of LLMs from the Granite Language Model Family, Phi-4, and Mixtral 8x22B to generate outputs, and use the Automatic Process Supervision method as described in [Luo et. al, 2024](https://arxiv.org/abs/2406.06592) for detecting steps with errors.

**Usage**


Sample use for obtaining PRM scores for a given response using Huggingface Transformers:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

def prepare_input(query: str, steps: List[str], tokenizer: AutoTokenizer, correct_token: str, generation_prompt: str):
    messages = []
    
    for s_idx, step in enumerate(steps):
        if s_idx == 0:
            # append query and first step
            message = {'role': 'user', 'content':  query + " " + step + " " + generation_prompt}
        else:
            message = {'role': 'user', 'content':  step + " " + generation_prompt}
        
        messages.append(message)
        messages.append({'role': 'assistant', 'content': correct_token})
    
    input_message = tokenizer.apply_chat_template(messages, add_generation_prompt = False, tokenize = False)

    return input_message

def get_step_ids(input_ids, tokenizer, correct_token, correct_token_id):
    # get assistant turn indices
    asst_text = "<|start_of_role|>assistant<|end_of_role|>" + correct_token + "<|end_of_text|>"
    asst_toks = tokenizer(asst_text, add_special_tokens = False, return_tensors = "pt")['input_ids'][0]
    asst_toks_before_correct_token = asst_toks[:torch.where(asst_toks == correct_token_id)[0].item()].tolist()

    input_ids = input_ids[0]
    # find batch index for assistant turn "Y", not just the correct_token_id
    correct_token_indices = torch.where(input_ids == correct_token_id)[0].tolist()
    prm_indices = []
    for t_idx in correct_token_indices:
        if input_ids[t_idx - len(asst_toks_before_correct_token) :t_idx].tolist() == asst_toks_before_correct_token:
            prm_indices.append(t_idx-1) # the logits for token i predict the token i+1: so, we need to look at the PREVIOUS token logits
    
    assert len(prm_indices)>0
    return prm_indices

model_name_or_path = "ibm-granite/granite-3.3-8b-math-prm-v2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

correct_token = "Y"
correct_token_id = tokenizer.encode(correct_token, add_special_tokens=False)[0]
generation_prompt = "Is this response correct so far (Y/N)?"


data = {
   "query": "For breakfast, Anna bought a bagel for $x and a glass of orange juice for $0.85. At lunch, Anna spent $4.65 on a sandwich and $1.15 on a carton of milk. How much more money did Anna spend on lunch than on breakfast? If we know the answer to the above question is 4, what is the value of unknown variable x?",
   "response":[
        "At breakfast, Anna spent x dollars on a bagel and $0.85 on a glass of orange juice. The total cost of breakfast is x + $0.85.",
        "At lunch, Anna spent $4.65 on a sandwich and $1.15 on a carton of milk. The total cost of lunch is $4.65 + $1.15 = $5.80.",
        "To find out how much more money Anna spent on lunch than on breakfast, we subtract the cost of breakfast from the cost of lunch: $5.80 - (x + $0.85).",
        "We are given that the difference is $4, so we can write: $5.80 - (x + $0.85) = $4.",
        "Simplifying the left side, we get: $5.80 - x - $0.85 = $4.",
        "Adding -$0.85 to both sides, we get: $5.80 -x = $3.15.",
        "Subtracting $5.80 from both sides, we get: -x = -$2.65.",
        "Dividing both sides by -1, we get: x = $2.65."
   ]
}


formatted_data = prepare_input(query=data['query'], steps=data['response'], tokenizer=tokenizer, correct_token=correct_token, generation_prompt=generation_prompt)
input_ids = tokenizer.encode(formatted_data, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(input_ids=input_ids).logits

# get step positions
prm_indices = get_step_ids(input_ids, tokenizer, correct_token, correct_token_id)

#  get corresponding rewards: convert logits to probabilities and get the probability of the correct token id as reward
softmax = torch.nn.Softmax(dim=-1)
step_rewards = []
for prm_idx in prm_indices:
    step_rewards.append(softmax(logits[0, prm_idx, :])[correct_token_id].item())

print(step_rewards)
# # [1.0, 1.0, 1.0, 0.99609375,0.9921875, 0.0849609375, 0.408203125, 0.8515625]
```

For use of the PRM as a verbalizer of correctness for a specific step:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name_or_path = "ibm-granite/granite-3.3-8b-math-prm-v2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
generation_prompt = "Is this response correct so far (Y/N)?"


data = {
   "query": "For breakfast, Anna bought a bagel for $x and a glass of orange juice for $0.85. At lunch, Anna spent $4.65 on a sandwich and $1.15 on a carton of milk. How much more money did Anna spend on lunch than on breakfast? If we know the answer to the above question is 4, what is the value of unknown variable x?",
   "partial_response":
        "At breakfast, Anna spent x dollars on a bagel and $0.85 on a glass of orange juice. The total cost of breakfast is x + $0.85. At lunch, Anna spent $4.65 on a sandwich and $1.15 on a carton of milk. The total cost of lunch is $4.65 + $1.15 = $5.80. To find out how much more money Anna spent on lunch than on breakfast, we subtract the cost of breakfast from the cost of lunch: $5.80 - (x + $0.85).",
}

# format the prompts
formatted_prompt = tokenizer.apply_chat_template([{'role':'user', 'content': data['query'] + " " + data['partial_response'] + " " + generation_prompt}], add_generation_prompt=True, tokenize=False)
inputs = tokenizer(formatted_prompt, return_tensors="pt")

# generate output
with torch.no_grad():
    response = model.generate(inputs["input_ids"].to(model.device), attention_mask=inputs["attention_mask"].to(model.device), max_new_tokens=2)

output_text = tokenizer.decode(response[0])
print(output_text)
# # <|start_of_role|>assistant<|end_of_role|>Y<|end_of_text|>
```


**Infrastructure**

We train Granite-3.3-8B-Math-PRM-v2 using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over multiple GPUs.

**Ethical Considerations and Limitations**

Granite-3.3-8B-Math-PRM-v2  is finetuned from Granite-3.3-8B-Instruct. Since it inherits its foundation from the instruct model, all ethical considerations and limitations applicable to [Granite-3.3-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) remain relevant. 