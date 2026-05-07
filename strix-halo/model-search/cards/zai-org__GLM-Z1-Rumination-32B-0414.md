---
license: mit
language:
- zh
- en
pipeline_tag: text-generation
library_name: transformers
---

# GLM-4-Z1-Rumination-32B-0414

## Introduction

The GLM family welcomes a new generation of open-source models, the **GLM-4-32B-0414** series, featuring 32 billion parameters. Its performance is comparable to OpenAI's GPT series and DeepSeek's V3/R1 series, and it supports very user-friendly local deployment features. GLM-4-32B-Base-0414 was pre-trained on 15T of high-quality data, including a large amount of reasoning-type synthetic data, laying the foundation for subsequent reinforcement learning extensions. In the post-training stage, in addition to human preference alignment for dialogue scenarios, we also enhanced the model's performance in instruction following, engineering code, and function calling using techniques such as rejection sampling and reinforcement learning, strengthening the atomic capabilities required for agent tasks. GLM-4-32B-0414 achieves good results in areas such as engineering code, Artifact generation, function calling, search-based Q&A, and report generation. Some benchmarks even rival larger models like GPT-4o and DeepSeek-V3-0324 (671B).

**GLM-Z1-32B-0414** is a reasoning model with **deep thinking capabilities**. This was developed based on GLM-4-32B-0414 through cold start and extended reinforcement learning, as well as further training of the model on tasks involving mathematics, code, and logic. Compared to the base model, GLM-Z1-32B-0414 significantly improves mathematical abilities and the capability to solve complex tasks. During the training process, we also introduced general reinforcement learning based on pairwise ranking feedback, further enhancing the model's general capabilities.

**GLM-Z1-Rumination-32B-0414** is a deep reasoning model with **rumination capabilities** (benchmarked against OpenAI's Deep Research). Unlike typical deep thinking models, the rumination model employs longer periods of deep thought to solve more open-ended and complex problems (e.g., writing a comparative analysis of AI development in two cities and their future development plans). The rumination model integrates search tools during its deep thinking process to handle complex tasks and is trained by utilizing multiple rule-based rewards to guide and extend end-to-end reinforcement learning. Z1-Rumination shows significant improvements in research-style writing and complex retrieval tasks.

Finally, **GLM-Z1-9B-0414** is a surprise. We employed the aforementioned series of techniques to train a 9B small-sized model that maintains the open-source tradition. Despite its smaller scale, GLM-Z1-9B-0414 still exhibits excellent capabilities in mathematical reasoning and general tasks. Its overall performance is already at a leading level among open-source models of the same size. Especially in resource-constrained scenarios, this model achieves an excellent balance between efficiency and effectiveness, providing a powerful option for users seeking lightweight deployment.

## Inference Code

Make Sure Using `transforemrs>=4.51.3`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/GLM-Z1-Rumination-32B-0414"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

message = [{"role": "user", "content": "Let a, b be positive real numbers such that ab = a + b + 3. Determine the range of possible values for a + b."}]

inputs = tokenizer.apply_chat_template(
    message,
    return_tensors="pt",
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

generate_kwargs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "temperature": 0.95,
    "top_p": 0.7,
    "do_sample": True,
}
out = model.generate(**generate_kwargs)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Function Call

By default, this model currently supports the following `function` calls:
- `search`: Search using a keyword and return search results
- `click`: Click on a specific webpage in the search results to view details
- `open`: Open a fixed URL to view detailed content
- `finsih`: Complete information gathering and begin writing

Below is a simple workflow to help you quickly connect the pipeline.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json

MODEL_PATH = "THUDM/GLM-4-Z1-Rumination-32B-0414"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

messages = [{"role": "user", "content": "Let a, b be positive real numbers such that ab = a + b + 3. Determine the range of possible values for a + b."}]

generate_kwargs = {
    "temperature": 0.95,
    "top_p": 0.7,
    "do_sample": True,
    "max_new_tokens": 16384
}

def get_assistant():
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    out = model.generate(input_ids=inputs["input_ids"], **generate_kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def get_observation(function_name, args):
    content = None
    if function_name == "search":
        mock_search_res = [
            {"title": "t1", "url":"url1", "snippet": "snippet_content_1"},
            {"title": "t2", "url":"url2", "snippet": "snippet_content_2"}
        ]
        content = "\n\n".join([f"【{i}†{res['title']}†{res['url']}\n{res['snippet']}】"] for i, res in enumerate(mock_search_res))
    elif function_name == "click":
        mock_click_res = "main content"
        content = mock_click_res
    elif function_name == "open":
        mock_open_res = "main_content"
        content = mock_open_res
    else:
        raise ValueError("unspport function name!")
    return content
        
def get_func_name_args(llm_text):
    function_call = re.sub(r'.*?</think>', '', llm_text, flags=re.DOTALL)
    function_call = json.loads(function_call)
    action = function_call['name']
    params = function_call['arguments']
    return action, params

def pipeline():
    end_str = "{\"name\": \"finish\", \"arguments\": {}}"
    response = get_assistant()
    messages.append({"role": "assistant", "content": response})
    max_turns, turns = 35, 1
    while not response.endswith(end_str) and turns < max_turns:
        action, params = get_func_name_args(response)
        observation = get_observation(action, params)
        messages.append({"role": "observation", "content": observation})
        response = get_assistant()
        messages.append({"role": "assistant", "content": response})
        turns += 1
        
    if response.endswith(end_str):
        final_answer = get_assistant()
    else:
        final_answer = None
    return final_answer

pipeline()   
```