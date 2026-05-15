---
pipeline_tag: text-generation
inference: false
license: apache-2.0
library_name: transformers
tags:
- language
- granite-3.3
base_model:
- ibm-granite/granite-3.3-8b-base
---

# Granite-3.3-8B-Instruct

**Model Summary:**
Granite-3.3-8B-Instruct is a 8-billion parameter 128K context length language model fine-tuned for improved reasoning and instruction-following capabilities. Built on top of Granite-3.3-8B-Base, the model delivers significant gains on benchmarks for measuring generic performance including AlpacaEval-2.0 and Arena-Hard, and improvements in mathematics, coding, and instruction following. It supports structured reasoning through \<think\>\<\/think\> and \<response\>\<\/response\> tags, providing clear separation between internal thoughts and final outputs. The model has been trained on a carefully balanced combination of permissively licensed data and curated synthetic tasks.

- **Developers:** Granite Team, IBM
- **Website**: [Granite Docs](https://www.ibm.com/granite/docs/)
- **Release Date**: April 16th, 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Languages:** 
English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. However, users may finetune this Granite model for languages beyond these 12 languages.

**Intended Use:** 
This model is designed to handle general instruction-following tasks and can be integrated into AI assistants across various domains, including business applications.

**Capabilities**
* Thinking
* Summarization
* Text classification
* Text extraction
* Question-answering
* Retrieval Augmented Generation (RAG)
* Code related tasks
* Function-calling tasks
* Multilingual dialog use cases
<!-- * Fill-in-the-middle -->
* Long-context tasks including long document/meeting summarization, long document QA, etc.


**Generation:** 
This is a simple example of how to use Granite-3.3-8B-Instruct model.

Install the following libraries:

```shell
pip install torch torchvision torchaudio
pip install accelerate
pip install transformers
```
Then, copy the snippet from the section that is relevant for your use case.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

model_path="ibm-granite/granite-3.3-8b-instruct"
device="cuda"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)

conv = [{"role": "user", "content":"Redesign a common household item to make it more sustainable and user-friendly. Explain the changes and their benefits."}]

input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=True, return_dict=True, add_generation_prompt=True).to(device)

set_seed(42)
output = model.generate(
    **input_ids,
    max_new_tokens=8192,
)

prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(prediction)
```

**Example Outputs**
- thinking=True
```md
<think>To address this prompt, I'll focus on redesigning a common household item: the plastic water bottle. The goal is to make it more sustainable and user-friendly. Here are the proposed changes and their benefits:

1. **Material**: Replace plastic with biodegradable or recyclable materials like glass or plant-based plastics (PLA).
2. **Design**: Introduce a modular design for easy refilling and cleaning.
3. **Functionality**: Incorporate a built-in water filter for purification.
4. **Durability**: Enhance the bottle's lifespan through robust construction.
5. **User-friendliness**: Simplify the cap mechanism for easy opening and closing.

These changes would significantly reduce plastic waste, promote recycling, ensure access to clean drinking water, and enhance the overall user experience.

Now, I will detail each change and its benefits in a structured format.</think>

<response>### Redesign of the Plastic Water Bottle for Sustainability and User-Friendliness

#### 1. Material: Biodegradable or Recyclable Alternatives

**Change**: Transition from traditional PET plastic to glass or plant-based plastics (Polylactic Acid - PLA).

**Benefits**:
- **Sustainability**: Glass and PLA are biodegradable or highly recyclable, drastically reducing plastic pollution.
- **Health**: Unlike PET, these materials do not leach chemicals into the water, ensuring safer consumption.

#### 2. Modular Design for Easy Refilling and Cleaning

**Change**: Implement a detachable, interchangeable inner liner within a reusable outer casing.

**Benefits**:
- **User-Friendliness**: Users can easily remove the inner liner for thorough cleaning, reducing the risk of bacterial growth.
- **Sustainability**: The outer casing remains intact, encouraging reuse, while the inner liner can be replaced or recycled when worn out.

#### 3. Built-in Water Filter

**Change**: Integrate a removable, replaceable water filter into the bottle cap.

**Benefits**:
- **Accessibility**: Provides clean drinking water directly from any source (tap, river, etc.), eliminating the need for additional filtration devices.
- **Health**: Removes contaminants, bacteria, and impurities, ensuring safe hydration.
- **Sustainability**: Reduces the use of single-use plastic water bottles by offering a purification solution within a reusable container.

#### 4. Enhanced Durability

**Change**: Use high-quality, durable materials and robust construction techniques for both the outer casing and inner liner.

**Benefits**:
- **Longevity**: A more robust design increases the bottle's lifespan, reducing the frequency of replacements.
- **Cost-effectiveness**: Although the initial cost might be higher, the long-term savings from reduced replacements make it economical.
- **Sustainability**: Less frequent disposal means lower environmental impact.

#### 5. Simplified Cap Mechanism

**Change**: Adopt a one-touch, easy-to-open and close cap design, possibly inspired by sports bottle caps.

**Benefits**:
- **User-Friendliness**: Especially beneficial for children, elderly, or those with limited hand dexterity.
- **Convenience**: Quick access to water encourages more frequent hydration, promoting healthier habits.

### Conclusion

This redesigned water bottle addresses key sustainability concerns by minimizing plastic use and waste, while also enhancing user-friendliness through improved functionality, durability, and accessibility. By integrating biodegradable materials, a modular design, a built-in filter, robust construction, and a simple cap mechanism, the bottle not only reduces environmental footprint but also offers a superior user experience.</response>
```
- thinking=False
```md
Item for Redesign: Plastic Bottle (e.g., water or beverage bottle)

Redesign Concept: Eco-Friendly, Modular, and Multi-Functional Reusable Bottle

Changes:

1. Material: Replace the conventional plastic with food-grade, BPA-free, and recyclable materials such as stainless steel or glass. These materials are durable, non-toxic, and have a longer lifespan, reducing waste.

2. Modular Design: Introduce a modular design that allows users to customize their bottle according to their needs. This could include interchangeable caps, mouthpiece sizes, and carrying loops. Users can purchase additional modules as needed, promoting a more sustainable approach by reducing the need to buy an entirely new bottle for different purposes.

3. Integrated Filter: Incorporate a built-in, washable, and reusable filter that can remove impurities and improve the taste of water. This eliminates the need for single-use disposable filters or bottled water, further reducing plastic waste.

4. Smart Cap: Develop a smart cap with a built-in digital display and temperature sensor. This feature allows users to track their daily water intake, set hydration goals, and monitor the temperature of their beverage. The smart cap can be synced with a mobile app for additional functionality, such as reminders and progress tracking.

5. Easy-to-Clean Design: Ensure the bottle has a wide mouth and smooth interior surfaces for easy cleaning. Include a brush for hard-to-reach areas, making maintenance simple and encouraging regular use.

6. Collapsible Structure: Implement a collapsible design that reduces the bottle's volume when not in use, making it more portable and convenient for storage.

Benefits:

1. Sustainability: By using recyclable materials and reducing plastic waste, this redesigned bottle significantly contributes to a more sustainable lifestyle. The modular design and reusable filter also minimize single-use plastic consumption.

2. User-Friendly: The smart cap, easy-to-clean design, and collapsible structure make the bottle convenient and user-friendly. Users can customize their bottle to suit their needs, ensuring a better overall experience.

3. Healthier Option: Using food-grade, BPA-free materials and an integrated filter ensures that the beverages consumed are free from harmful chemicals and impurities, promoting a healthier lifestyle.

4. Cost-Effective: Although the initial investment might be higher, the long-term savings from reduced purchases of single-use plastic bottles and disposable filters make this reusable bottle a cost-effective choice.

5. Encourages Hydration: The smart cap's features, such as hydration tracking and temperature monitoring, can motivate users to stay hydrated and develop healthier habits.

By redesigning a common household item like the plastic bottle, we can create a more sustainable, user-friendly, and health-conscious alternative that benefits both individuals and the environment.
```

**Evaluation Results:**
<table>
<thead>
    <caption style="text-align:center"><b>Comparison with different models over various benchmarks<sup id="fnref1"><a href="#fn1">1</a></sup>. Scores of AlpacaEval-2.0 and Arena-Hard are calculated with thinking=True</b></caption>
  <tr>
    <th style="text-align:left; background-color: #001d6c; color: white;">Models</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">Arena-Hard</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">AlpacaEval-2.0</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">MMLU</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">PopQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">TruthfulQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">BigBenchHard<sup id="fnref2"><a href="#fn2">2</a></sup></th>
    <th style="text-align:center; background-color: #001d6c; color: white;">DROP<sup id="fnref3"><a href="#fn3">3</a></sup></th>
    <th style="text-align:center; background-color: #001d6c; color: white;">GSM8K</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">HumanEval</th>
   <th style="text-align:center; background-color: #001d6c; color: white;">HumanEval+</th>
  <th style="text-align:center; background-color: #001d6c; color: white;">IFEval</th>
  <th style="text-align:center; background-color: #001d6c; color: white;">AttaQ</th>
  </tr></thead>
  <tbody>
<tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.1-2B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">23.3</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">27.17</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">57.11</td> 
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">20.55</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">59.79</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">61.82</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">20.99</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">67.55</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">79.45</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">75.26</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">63.59</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">84.7</td>
  </tr>
  <tr>
      <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.2-2B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">24.86</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">34.51</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">57.18</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">20.56</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">59.8</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">61.39</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">23.84</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">67.02</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">80.13</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">73.39</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">61.55</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">83.23</td>
  </tr>
  <tr>
      <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-3.3-2B-Instruct</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 28.86 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 43.45 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 55.88 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 18.4 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 58.97 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 63.91 </td>
      <td style="text-align:center; background-color: #DAE8FF; color: black;"> 44.33 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 72.48 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 80.51 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 75.68 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 65.8 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">87.47</td>
      </tr>
      
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Llama-3.1-8B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">36.43</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">27.22</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">69.15</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">28.79</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">52.79</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">73.43</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">71.23</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">83.24</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">85.32</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">80.15</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">79.10</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">83.43</td>
  </tr>
           
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">DeepSeek-R1-Distill-Llama-8B</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">17.17</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">21.85</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">45.80</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">13.25</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">47.43</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">67.39</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">49.73</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">72.18</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">67.54</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">62.91</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">66.50</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">42.87</td>
  </tr>
      
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Qwen-2.5-7B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">25.44</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">30.34</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">74.30</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">18.12</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">63.06</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">69.19</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">64.06</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">84.46</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">93.35</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">89.91</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">74.90</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">81.90</td>
  </tr>
      
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">DeepSeek-R1-Distill-Qwen-7B</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">10.36</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">15.35</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">50.72</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">9.94</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">47.14</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">67.38</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">51.78</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">78.47</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">79.89</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">78.43</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">59.10</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">42.45</td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.1-8B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">37.58</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">30.34</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">66.77</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">28.7</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">65.84</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">69.87</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">58.57</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">79.15</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">89.63</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">85.79</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">73.20</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">85.73</td>
  </tr>
            
<tr>
      <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.2-8B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">55.25</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">61.19</td>
   <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">66.79</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">28.04</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">66.92</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">71.86</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">58.29</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">81.65</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">89.35</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">85.72</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">74.31</td>
     <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;">84.7</td>
  </tr>
  <tr>
      <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-3.3-8B-Instruct</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 57.56 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 62.68 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 65.54 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 26.17 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 66.86 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 69.13 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 59.36 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 80.89 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 89.73 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 86.09 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 74.82 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">88.5</td>
      </tr>                 
</tbody></table>

<table>
 <caption style="text-align:center"><b>Math Benchmarks</b></caption>
<thead>
  <tr>
    <th style="text-align:left; background-color: #001d6c; color: white;">Models</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">AIME24</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">MATH-500</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.1-2B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 0.89 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 35.07 </td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.2-2B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 0.89 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 35.54 </td>
  </tr>
  <tr>
      <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-3.3-2B-Instruct</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 3.28 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 58.09 </td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.1-8B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 1.97 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 48.73 </td>
  </tr>
  <tr>
    <td style="text-align:left; background-color: #FFFFFF; color: #2D2D2D;">Granite-3.2-8B-Instruct</td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 2.43 </td>
    <td style="text-align:center; background-color: #FFFFFF; color: #2D2D2D;"> 52.8 </td>
  </tr>
  <tr>
      <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-3.3-8B-Instruct</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 8.12 </td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;"> 69.02 </td>
  </tr>
    </tbody></table>
    
**Training Data:** 
Overall, our training data is largely comprised of two key sources: (1) publicly available datasets with permissive license, (2) internal synthetically generated data targeted to enhance reasoning capabilites. 
<!-- A detailed attribution of datasets can be found in [Granite 3.2 Technical Report (coming soon)](#), and [Accompanying Author List](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/author-ack.pdf). -->

**Infrastructure:**
We train Granite-3.3-8B-Instruct using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs.

**Ethical Considerations and Limitations:** 
Granite-3.3-8B-Instruct builds upon Granite-3.3-8B-Base, leveraging both permissively licensed open-source and select proprietary data for enhanced performance. Since it inherits its foundation from the previous model, all ethical considerations and limitations applicable to [Granite-3.3-8B-Base](https://huggingface.co/ibm-granite/granite-3.3-8b-base) remain relevant.


**Resources**
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 📄 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://github.com/ibm-granite-community/

<p><a href="#fnref1" title="Jump back to reference">[1]</a> Evaluated using <a href="https://github.com/allenai/olmes">OLMES</a> (except AttaQ and Arena-Hard scores)</p>
<p><a href="#fnref2" title="Jump back to reference">[2]</a> Added regex for more efficient asnwer extraction.</a></p>
<p><a href="#fnref3" title="Jump back to reference">[3]</a> Modified the implementation to handle some of the issues mentioned <a href="https://huggingface.co/blog/open-llm-leaderboard-drop">here</a></p>
<!-- ## Citation
<!-- ## Citation
```
@misc{granite-models,
  author = {author 1, author2, ...},
  title = {},
  journal = {},
  volume = {},
  year = {2024},
  url = {https://arxiv.org/abs/0000.00000},
}
``` -->