---
language:
- ru
datasets:
- IlyaGusev/saiga_scored
license: other
license_name: llama3
license_link: https://llama.meta.com/llama3/license/
---


# Saiga/Llama3 8B, Russian Llama-3-based chatbot

Based on [Llama-3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

Llama.cpp version: [link](https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf)

Colab: [link](https://colab.research.google.com/drive/1qxgIPymzW6_H6s_wwXu3lknkkYM45Db4)

## Prompt format

**ОСТОРОЖНО! WARNING! LET OP!**

I've changed the prompt format from ChatML to **the original Llama-3 format in v4**. Don't forget to switch formats!

**v4, v5, v6+**: LLama-3 prompt format:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.<|eot_id|><|start_header_id|>user<|end_header_id|>

Как дела?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Отлично, а у тебя?<|eot_id|><|start_header_id|>user<|end_header_id|>

Шикарно. Как пройти в библиотеку?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```


**v2, v3**: ChatML prompt format:
```
<|im_start|>system
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.<|im_end|>
<|im_start|>user
Как дела?<|im_end|>
<|im_start|>assistant
Отлично, а у тебя?<|im_end|>
<|im_start|>user
Шикарно. Как пройти в библиотеку?<|im_end|>
<|im_start|>assistant
```

## Code example
```python
# Исключительно ознакомительный пример.
# НЕ НАДО ТАК ИНФЕРИТЬ МОДЕЛЬ В ПРОДЕ.
# См. https://github.com/vllm-project/vllm или https://github.com/huggingface/text-generation-inference

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

inputs = ["Почему трава зеленая?", "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч"]
for query in inputs:
    prompt = tokenizer.apply_chat_template([{
        "role": "system",
        "content": DEFAULT_SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": query
    }], tokenize=False, add_generation_prompt=True)
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print(query)
    print(output)
    print()
    print("==============================")
    print()
```

## Output examples
```
User: Почему трава зеленая? 
Saiga: Трава выглядит зеленым из-за наличия в ней хлорофилла – пигмента, который помогает растениям поглощать солнечную энергию для процесса фотосинтеза.
Хлорофилл поглощает световые лучи с длиной волны 400-500 нанометров (ближе к синему) и отражает лучи с длиной волны 600-700 нанометров (ближе к красному), что делает его видимым как зеленый цвет.
Таким образом, зеленый цвет является результатом того, как хлорофилл взаимодействует со светом, обеспечивая растения необходимой энергии для роста и развития.
```

```
User: Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч
Saiga: В небольшой деревушке под открытым небом жила девочка по имени Таня. Она была очень активной и любящей играть в различные виды спорта. Однажды, когда она прогуливалась по полям, увидела на земле лежащий мяч. Это был старый, но все еще пригодный для игры мячик.
Таня сразу же решила взять его себе домой и показать своему другу Мише. Мишка тоже обожал футбол и всегда рад был бы провести время вместе со своими друзьями. Когда они вернулись домой, Таня показала ему найденный мячик и предложила сыграть в футбол.
Мишка был счастлив, что у них появился новый мяч, и они сразу же начали готовиться к матчу. Они построили две команды и начали игру. Играли они до вечера, а потом решили сделать перерыв и приготовить ужин.
После ужина они продолжили игру, теперь уже с большим энтузиазмом и страстью. В конце концов, победителем стала команда Тани, которая забила больше голов. Все закончилось веселым праздником и хорошими воспоминаниями о проведенном времени.
На следующий день Таня и Миша решили пригласить своих других друзей на игру. Они собрали всех в одном месте и провели ещё один захватывающий матч. Этот день также стал незабываемым благодаря новому мячу, который принес так много радости и улыбок.
Итак, этот маленький мяч, который Таня нашла на поле, стал не просто предметом для игр, но и символом дружбы и веселья между детьми. Он помог им создать незабываемые моменты и укрепил их отношения.
```


## Versions
v7:
- [17e1bcc9c6d3e31852a95a168f7d3778f395cd2a](https://huggingface.co/IlyaGusev/saiga_llama3_8b/commit/17e1bcc9c6d3e31852a95a168f7d3778f395cd2a)
- Other name: saiga_llama3_8b_abliterated_sft_m11_d7_kto_m7_d7
- SFT dataset config: [sft_d7.json](https://github.com/IlyaGusev/saiga/blob/main/configs/datasets/sft_d7.json)
- SFT model config: [saiga_llama3_8b_sft_m11.json](https://github.com/IlyaGusev/saiga/blob/main/configs/models/saiga_llama3_8b_sft_m11.json)
- KTO dataset config: [pref_d7.json](https://github.com/IlyaGusev/saiga/blob/main/configs/datasets/pref_d7.json)
- KTO model config: [saiga_llama3_8b_kto_m7.json](https://github.com/IlyaGusev/saiga/blob/main/configs/models/saiga_llama3_8b_kto_m7.json)
- SFT wandb: [link](https://wandb.ai/ilyagusev/rulm_self_instruct/runs/2ympfu9y)
- KTO wandb: [link](https://wandb.ai/ilyagusev/rulm_self_instruct/runs/9zn4825e)

v6:
- [b662833f247ca04f1843b356e7ff3ee4aef8086a](https://huggingface.co/IlyaGusev/saiga_llama3_8b/commit/b662833f247ca04f1843b356e7ff3ee4aef8086a)
- Other name: saiga_llama3_8b_sft_m10_d1_kto_m2_d2
- SFT dataset config: [sft_d1.json](https://github.com/IlyaGusev/saiga/blob/main/configs/datasets/sft_d1.json)
- SFT model config: [saiga_llama3_8b_sft_m10.json](https://github.com/IlyaGusev/saiga/blob/main/configs/models/saiga_llama3_8b_sft_m10.json)
- KTO dataset config: [pref_d2.json](https://github.com/IlyaGusev/saiga/blob/main/configs/datasets/pref_d2.json)
- KTO model config: [saiga_llama3_8b_kto_m2.json](https://github.com/IlyaGusev/saiga/blob/main/configs/models/saiga_llama3_8b_kto_m2.json)
- SFT wandb: [link](https://wandb.ai/ilyagusev/rulm_self_instruct/runs/0iepauzu)
- KTO wandb: [link](https://wandb.ai/ilyagusev/rulm_self_instruct/runs/s6l98eot)

v5:
- [d947b00c56683cd4b2f7ce707edef89318027be4](https://huggingface.co/IlyaGusev/saiga_llama3_8b/commit/d947b00c56683cd4b2f7ce707edef89318027be4)
- KTO-tune over v4, dataset: [lmsys_clean_ru_preferences](https://huggingface.co/datasets/IlyaGusev/lmsys_clean_ru_preferences)
- wandb [link](https://wandb.ai/ilyagusev/rulm_self_instruct/runs/se1mbx7n)

v4:
- [1cc945d4ca2c7901cf989e7edaac52ab24f1a7dd](https://huggingface.co/IlyaGusev/saiga_llama3_8b/commit/1cc945d4ca2c7901cf989e7edaac52ab24f1a7dd)
- dataset: [saiga_scored](https://huggingface.co/datasets/IlyaGusev/saiga_scored), scores >= 8, c66032920556c0f21bbbed05e7e04433ec954c3d
- wandb [link](https://wandb.ai/ilyagusev/rulm_self_instruct/runs/dcbs9ttt)

v3:
- [c588356cd60bdee54d52c2dd5a2445acca8aa5c3](https://huggingface.co/IlyaGusev/saiga_llama3_8b/commit/c588356cd60bdee54d52c2dd5a2445acca8aa5c3)
- dataset: [saiga_scored](https://huggingface.co/datasets/IlyaGusev/saiga_scored), scores >= 8, d51cf8060bdc90023da8cf1c3f113f9193d6569b
- wandb [link](https://wandb.ai/ilyagusev/rulm_self_instruct/runs/ltoqdsal)

v2:
- [ae61b4f9b34fac9856d361ea78c66284a00e4f0b](https://huggingface.co/IlyaGusev/saiga_llama3_8b/commit/ae61b4f9b34fac9856d361ea78c66284a00e4f0b)
- dataset code revision d0d123dd221e10bb2a3383bcb1c6e4efe1b4a28a
- wandb [link](https://wandb.ai/ilyagusev/huggingface/runs/r6u5juyk)
- 5 datasets: ru_turbo_saiga, ru_sharegpt_cleaned, oasst1_ru_main_branch, gpt_roleplay_realm, ru_instruct_gpt4
- Datasets merging script: [create_short_chat_set.py](https://github.com/IlyaGusev/rulm/blob/d0d123dd221e10bb2a3383bcb1c6e4efe1b4a28a/self_instruct/src/data_processing/create_short_chat_set.py)


## Evaluation

* Dataset: https://github.com/IlyaGusev/rulm/blob/master/self_instruct/data/tasks.jsonl
* Framework: https://github.com/tatsu-lab/alpaca_eval
* Evaluator: alpaca_eval_cot_gpt4_turbo_fn

Pivot: chatgpt_3_5_turbo
| model | length_controlled_winrate |  win_rate | standard_error  | avg_length |
|-----|-----|-----|-----|-----|
|chatgpt_4_turbo | 76.04 | 90.00 |1.46 | 1270 |
|chatgpt_3_5_turbo | 50.00 | 50.00 | 0.00  | 536 |
|saiga_llama3_8b, v6 | 49.33 | 68.31 | 2.26  | 1262 |
|sfr-iter-dpo | 49.11 | 74.94 | 2.13 | 1215 |
|suzume | 49.05 | 71.57 | 2.20  | 1325 |
|saiga_llama3_8b, v7| 48.95 | 69.40 | 2.25 | 1266 |
|saiga_llama3_8b, v5  | 47.13  | 66.18 | 2.31 | 1194 |
|saiga_llama3_8b, v4  | 43.64  | 65.90 | 2.31 | 1200 |
|saiga_llama3_8b, v3  | 36.97  | 61.08 | 2.38 | 1162 |
|saiga_llama3_8b, v2  | 33.07  | 48.19 | 2.45 | 1166 |
|saiga_mistral_7b  | 23.38  | 35.99 | 2.34 | 949  |

Pivot: sfr
| model | length_controlled_winrate |  win_rate | standard_error  | avg_length |
|-----|-----|-----|-----|-----|
| sfr | 50.00 |  50.00 | 0.00 | 1215 |
| saiga_llama3_8b, v7 |  48.95  |  49.16  | 2.46  | 1266 |
| saiga_llama3_8b, v6 | 46.91 | 47.23 | 2.45 | 1262 |
| suzume_8b | 43.69  | 48.19 | 2.46 | 1325 |