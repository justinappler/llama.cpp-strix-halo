---
pipeline_tag: text-generation
inference: false
license: apache-2.0
library_name: transformers
tags:
- language
- granite-3.2
base_model:
- ibm-granite/granite-3.1-8b-instruct
new_version: ibm-granite/granite-3.3-8b-instruct
---

# Granite-3.2-8B-Instruct

**Model Summary:**
Granite-3.2-8B-Instruct is an 8-billion-parameter, long-context AI model fine-tuned for thinking capabilities. Built on top of [Granite-3.1-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct), it has been trained using a mix of permissively licensed open-source datasets and internally generated synthetic data designed for reasoning tasks. The model allows controllability of its thinking capability, ensuring it is applied only when required.


- **Developers:** Granite Team, IBM
- **Website**: [Granite Docs](https://www.ibm.com/granite/docs/)
- **Release Date**: February 26th, 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Languages:** 
English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. However, users may finetune this Granite model for languages beyond these 12 languages.

**Intended Use:** 
This model is designed to handle general instruction-following tasks and can be integrated into AI assistants across various domains, including business applications.

**Capabilities**
* **Thinking**
* Summarization
* Text classification
* Text extraction
* Question-answering
* Retrieval Augmented Generation (RAG)
* Code related tasks
* Function-calling tasks
* Multilingual dialog use cases
* Long-context tasks including long document/meeting summarization, long document QA, etc.



**Generation:** 
This is a simple example of how to use Granite-3.2-8B-Instruct model.

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

model_path="ibm-granite/granite-3.2-8b-instruct"
device="cuda"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)

conv = [{"role": "user", "content":"You have 10 liters of a 30% acid solution. How many liters of a 70% acid solution must be added to achieve a 50% acid mixture?"}]

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
Here is my thought process:

1. We know we start with 10 liters of a 30% acid solution, so there's 3 liters of pure acid in it (10 liters * 0.3).
2. We need to determine how much of the 70% acid solution to add such that the final mixture is 50% acid.
3. Let's say x liters of the 70% acid solution are added. This will contribute 0.7x liters of pure acid.
4. The total volume of the new mixture will be 10 + x liters, and the total amount of pure acid will be 3 + 0.7x liters.
5. We need to set up an equation where (3 + 0.7x) / (10 + x) = 0.5 (since we want a 50% acid mixture).
6. Solving this equation for x will give us the volume of 70% solution needed.

Here is my response:

To solve this, we need to set up an equation based on the concentration of acid before and after mixing. 

1. **Initial Acid Content**: 
   - We start with 10 liters of a 30% acid solution. 
   - Pure acid in this = 10 * 0.3 = 3 liters.

2. **Variable for Addition**:
   - Let x be the liters of 70% acid solution to be added.
   - Pure acid in x liters of 70% solution = 0.7x liters.

3. **Final Mixture Concentration**:
   - Total volume of mixture = 10 + x liters.
   - Total pure acid in the mixture = 3 (from initial solution) + 0.7x (from added solution).

4. **Setting up the equation for 50% acid concentration**:
   - We require that the ratio of pure acid to total volume is 0.5 (50%).
   - Therefore, the equation is:

   \[
   \frac{3 + 0.7x}{10 + x} = 0.5
   \]

5. **Solving the equation**:

   Multiply both sides by (10 + x) to clear the fraction:

   \[
   3 + 0.7x = 0.5(10 + x)
   \]

   Expanding the right side:

   \[
   3 + 0.7x = 5 + 0.5x
   \]

   Subtract 0.5x from both sides:

   \[
   3 + 0.2x = 5
   \]

   Subtract 3 from both sides:

   \[
   0.2x = 2
   \]

   Divide by 0.2:

   \[
   x = 10
   \]

So, 10 liters of the 70% acid solution must be added to the initial 10 liters of 30% solution to get a 50% acid mixture. 

Let's verify: 

- Initial acid: 3 liters
- Acid from added solution: 0.7 * 10 = 7 liters
- Total acid: 3 + 7 = 10 liters
- Total volume: 10 + 10 = 20 liters
- Acid percentage: \(\frac{10}{20} = 0.5 = 50\%\)

This confirms our solution. 

Final answer: 10 liters of 70% acid solution should be added.
```
- thinking=False
```md
Let's denote the amount of 70% acid solution to be added as x liters.

We start with 10 liters of a 30% acid solution, which means there are 0.3 * 10 = 3 liters of pure acid in it.

We want to end up with a 50% acid solution after adding x liters of 70% acid solution. The total volume of the mixture will be 10 + x liters, and we want half of this volume to be pure acid.

So, the amount of pure acid in the final mixture should be 0.5 * (10 + x) liters.

The amount of pure acid in the final mixture comes from two sources: the original 30% solution and the added 70% solution. So, we can set up the following equation:

3 (from the original solution) + 0.7x (from the added solution) = 0.5 * (10 + x)

Now, let's solve for x:

3 + 0.7x = 5 + 0.5x
0.7x - 0.5x = 5 - 3
0.2x = 2
x = 2 / 0.2
x = 10

So, you need to add 10 liters of a 70% acid solution to the 10 liters of a 30% acid solution to get a 50% acid mixture.
```

**Evaluation Results:**
<table>
  
<thead>
  <tr>
    <th style="text-align:left; background-color: #001d6c; color: white;">Models</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">ArenaHard</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">Alpaca-Eval-2</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">MMLU</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">PopQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">TruthfulQA</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">BigBenchHard</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">DROP</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">GSM8K</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">HumanEval</th>
   <th style="text-align:center; background-color: #001d6c; color: white;">HumanEval+</th>
  <th style="text-align:center; background-color: #001d6c; color: white;">IFEval</th>
    <th style="text-align:center; background-color: #001d6c; color: white;">AttaQ</th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;">Llama-3.1-8B-Instruct</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">36.43</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">27.22</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">69.15</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">28.79</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">52.79</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">72.66</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">61.48</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">83.24</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">85.32</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">80.15</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">79.10</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">83.43</td>
  </tr>
           
  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;">DeepSeek-R1-Distill-Llama-8B</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">17.17</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">21.85</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">45.80</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">13.25</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">47.43</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">65.71</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">44.46</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">72.18</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">67.54</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">62.91</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">66.50</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">42.87</td>
  </tr>
      
  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;">Qwen-2.5-7B-Instruct</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">25.44</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">30.34</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">74.30</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">18.12</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">63.06</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">70.40</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">54.71</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">84.46</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">93.35</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">89.91</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">74.90</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">81.90</td>
  </tr>
      
  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;">DeepSeek-R1-Distill-Qwen-7B</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">10.36</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">15.35</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">50.72</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">9.94</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">47.14</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">65.04</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">42.76</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">78.47</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">79.89</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">78.43</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">59.10</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">42.45</td>
  </tr>

  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;">Granite-3.1-8B-Instruct</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">37.58</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">30.34</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">66.77</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">28.7</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">65.84</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">68.55</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">50.78</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">79.15</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">89.63</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">85.79</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">73.20</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">85.73</td>
  </tr>
      
      
<tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;">Granite-3.1-2B-Instruct</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">23.3</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">27.17</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">57.11</td> 
    <td style="text-align:center; background-color: #DAE8FF; color: black;">20.55</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">59.79</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">54.46</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">18.68</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">67.55</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">79.45</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">75.26</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">63.59</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">84.7</td>
  </tr>
      
 
  <tr>
      <td style="text-align:left; background-color: #DAE8FF; color: black;">Granite-3.2-2B-Instruct</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">24.86</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">34.51</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">57.18</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">20.56</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">59.8</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">52.27</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">21.12</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">67.02</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">80.13</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">73.39</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">61.55</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">83.23</td>
  </tr> 
      
<tr>
      <td style="text-align:left; background-color: #DAE8FF; color: black;"><b>Granite-3.2-8B-Instruct</b></td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">55.25</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">61.19</td>
   <td style="text-align:center; background-color: #DAE8FF; color: black;">66.79</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">28.04</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">66.92</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">64.77</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">50.95</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">81.65</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">89.35</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">85.72</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">74.31</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">85.42</td>
 
  </tr>

     
      
</tbody></table>

**Training Data:** 
Overall, our training data is largely comprised of two key sources: (1) publicly available datasets with permissive license, (2) internal synthetically generated data targeted to enhance reasoning capabilites. 
<!-- A detailed attribution of datasets can be found in [Granite 3.2 Technical Report (coming soon)](#), and [Accompanying Author List](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/author-ack.pdf). -->

**Infrastructure:**
We train Granite-3.2-8B-Instruct using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs.

**Ethical Considerations and Limitations:** 
Granite-3.2-8B-Instruct builds upon Granite-3.1-8B-Instruct, leveraging both permissively licensed open-source and select proprietary data for enhanced performance. Since it inherits its foundation from the previous model, all ethical considerations and limitations applicable to [Granite-3.1-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) remain relevant.


**Resources**
- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 📄 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://ibm.biz/granite-learning-resources

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