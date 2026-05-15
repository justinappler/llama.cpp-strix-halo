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
new_version: ibm-granite/granite-3.2-8b-instruct
---

# Granite-3.2-8B-Instruct-Preview

**Model Summary:**
Granite-3.2-8B-Instruct-Preview is an early release of an 8B long-context model fine-tuned for enhanced reasoning (thinking) capabilities. Built on top of [Granite-3.1-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct), it has been trained using a mix of permissively licensed open-source datasets and internally generated synthetic data designed for reasoning tasks. The model allows controllability of its thinking capability, ensuring it is applied only when required.

<!-- is preview release of a finetuned mdpeis a 8B parameter long-context instruct model finetuned from Granite-3.1-8B-Instruct using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets tailored for solving long context problems. This model is finetuned to reason 

developed using a diverse set of techniques with a structured chat format, including supervised finetuning, model alignment using reinforcement learning, and model merging. -->

- **Developers:** Granite Team, IBM
- **Website**: [Granite Docs](https://www.ibm.com/granite/docs/)
- **Release Date**: February 7th, 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Supported Languages:** 
English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese. However, users may finetune this Granite model for languages beyond these 12 languages.

**Intended Use:** 
The model is designed to respond to general instructions and can be used to build AI assistants for multiple domains, including business applications.

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
This is a simple example of how to use Granite-3.2-8B-Instruct-Preview model.

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

model_path="ibm-granite/granite-3.2-8b-instruct-preview"
device="cuda"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)

conv = [{"role": "user", "content":"How do I develop a skill?"}]

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

1. Understand the user's intent: The user is seeking guidance on how to develop a new skill. 
2. Break down the process: Skills development involves several stages - identification, planning, practice, feedback, and refinement. 
3. Consider different aspects: Discuss various methods, mindset, and resources needed for effective skill acquisition. 
4. Provide actionable steps: Offer a structured approach that the user can follow. 

Here is my response:

Developing a new skill is a systematic process that involves several key steps and a dedicated mindset. Here's a comprehensive guide to help you effectively cultivate a new skill:

### 1. **Identify the Skill**
   - **Clarity of Purpose**: Clearly define what skill you want to acquire. Be specific about why you want this skill and how it aligns with your goals or interests.
   - **Research**: Understand the basics of the skill. Read about it, watch tutorials, or talk to people who are already proficient. This initial research will give you a realistic understanding of what the skill entails.

### 2. **Set Clear Goals**
   - **SMART Goals**: Make sure your goals are Specific, Measurable, Achievable, Relevant, and Time-bound. For example, instead of saying "I want to learn programming," set a goal like "I will complete a beginner's Python course within three months."
   - **Break Down**: If the skill is complex, break it down into smaller, manageable milestones. 

### 3. **Create a Learning Plan**
   - **Resources**: Identify the best resources for learning. These could be online courses, books, mentors, workshops, or practical projects.
   - **Schedule**: Dedicate time in your routine for skill practice. Consistency is crucial in skill development. Treat your learning time as non-negotiable.

### 4. **Start with the Basics**
   - **Foundational Knowledge**: Begin by grasping the fundamentals. Understanding the basics thoroughly sets a strong foundation for more advanced learning.
   - **Practice Regularly**: Regular, focused practice is essential. Even 15-30 minutes a day can yield significant results over time.

### 5. **Seek Feedback and Guidance**
   - **Mentorship**: If possible, find a mentor or join a community of learners. Feedback from experienced individuals can accelerate your learning curve.
   - **Peer Learning**: Engage with others learning the same skill. Peer groups provide support, motivation, and different perspectives.

### 6. **Apply the Skill in Real-World Scenarios**
   - **Practical Application**: Look for opportunities to apply what you've learned. This could be through projects, internships, or volunteering. Practical application solidifies your understanding and builds confidence.
   - **Iterate Based on Feedback**: After applying your skill, seek and incorporate feedback to refine and improve.

### 7. **Maintain a Growth Mindset**
   - **Embrace Challenges**: Understand that setbacks and failures are part of the learning process. Maintain a positive attitude and view them as opportunities to learn.
   - **Patience and Persistence**: Skill development takes time. Stay patient and persistent, even when progress seems slow.

### 8. **Track and Reflect**
   - **Progress Tracking**: Regularly review your progress. This could be through keeping a learning journal or using apps designed for skill tracking.
   - **Reflect and Adjust**: Periodically reflect on your learning process. What's working? What's not? Adjust your plan accordingly.

### 9. **Stay Motivated**
   - **Reward System**: Set small rewards for achieving milestones to keep yourself motivated.
   - **Stay Curious**: Keep your interest piqued by exploring related topics or advanced aspects of the skill.

By following these steps and maintaining a disciplined, growth-oriented approach, you can effectively develop any new skill. Remember, the key is consistent effort and a willingness to learn from both successes and failures. 

Building skills is an ongoing journey, not a destination.
```
- thinking=False
```md
Developing a skill involves a combination of learning, practice, and often, feedback. Here's a step-by-step guide to help you develop a new skill:

1. **Identify the Skill**: Start by clearly defining what skill you want to develop. Be specific. Instead of saying "I want to learn to code," specify a programming language like Python or JavaScript.

2. **Research**: Learn about the basics of the skill. Read books, articles, watch tutorials, or take online courses. Websites like Coursera, Udemy, Khan Academy, and YouTube can be great resources.

3. **Set Clear Goals**: Break down your skill into smaller, manageable goals. For example, if you're learning a new language, your goals might be to learn basic grammar, build a simple sentence, have a basic conversation, etc.

4. **Create a Study Plan**: Allocate specific time each day or week for learning and practicing. Consistency is key in skill development.

5. **Practice**: Apply what you've learned. Practice makes permanent. If you're learning to code, write small programs. If it's a musical instrument, play regularly.

6. **Get Feedback**: Seek feedback from others who are more experienced. This could be a mentor, a tutor, or even online communities. Constructive criticism can help you identify areas for improvement.

7. **Review and Refine**: Regularly review what you've learned. Refine your skills based on feedback and your own observations.

8. **Apply in Real Life**: Try to use your new skill in real-life situations. This could be a project at work, a personal hobby, or volunteering.

9. **Be Patient and Persistent**: Skill development takes time. Don't get discouraged by slow progress or setbacks. Keep practicing and learning.

10. **Stay Motivated**: Keep your end goal in mind and celebrate small victories along the way to stay motivated.

Remember, everyone learns at their own pace, so don't compare your progress with others. The most important thing is that you're consistently moving forward.
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
    <td style="text-align:center; background-color: #DAE8FF; color: black;">27.87</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">66.84</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">28.84</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">65.92</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">68.10</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">50.78</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">79.08</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">88.82</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">84.62</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">71.20</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">85.73</td>
  </tr>

  <tr>
    <td style="text-align:left; background-color: #DAE8FF; color: black;">Granite-3.2-8B-Instruct-Preview</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">55.23</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">61.16</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">66.93</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">28.08</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">66.37</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">65.60</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">50.73</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">83.09</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">89.47</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">86.88</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">73.57</td>
    <td style="text-align:center; background-color: #DAE8FF; color: black;">85.99</td>
  </tr>
 
</tbody></table>

**Training Data:** 
Overall, our training data is largely comprised of two key sources: (1) publicly available datasets with permissive license, (2) internal synthetically generated data targeted to enhance reasoning capabilites. 
<!-- A detailed attribution of datasets can be found in [Granite 3.2 Technical Report (coming soon)](#), and [Accompanying Author List](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/author-ack.pdf). -->

**Infrastructure:**
We train Granite-3.2-8B-Instruct-Preview using IBM's super computing cluster, Blue Vela, which is outfitted with NVIDIA H100 GPUs. This cluster provides a scalable and efficient infrastructure for training our models over thousands of GPUs.

**Ethical Considerations and Limitations:** 
Granite-3.2-8B-Instruct-Preview builds upon Granite-3.1-8B-Instruct, leveraging both permissively licensed open-source and select proprietary data for enhanced performance. Since it inherits its foundation from the previous model, all ethical considerations and limitations applicable to [Granite-3.1-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) remain relevant.


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