---
license: mit
license_link: https://github.com/microsoft/Industrial-Foundation-Models/blob/main/LICENSE

tags:
- llm
- transfer learning
- in-context learning
- tabular data
---

## Model Summary

The model is finetuned on over 380 tabular datasets based on LLaMA-2, designed to process a variety of industrial data, including commerce, healthcare, energy, and sustainability. The model belongs to the IFMs family, including two versions [7B](https://huggingface.co/microsoft/LLaMA-2-7b-GTL-Delta) and [13B](https://huggingface.co/microsoft/LLaMA-2-13b-GTL-Delta). 

The Industrial Foundation Model is designed to accept language format data samples from various domains as input prompts. The input prompt should contain relevant information for the task at hand, such as context data, specific task instructions, or direct questions. In response to the input prompts, the model generates predictive answers. Depending on the nature of the task instruction in the input, the model can support both classification and regression tasks.

Resources and Technical Documentation:

+ [IFMs Microsoft Repo](https://github.com/microsoft/Industrial-Foundation-Models)
+ [Paper](https://arxiv.org/abs/2310.07338)

## Intended Uses

**Primary use cases**

This model is designed to process and analyze diverse tabular data from various industry sectors for accurate prediction of classification and regression tasks. 

### Tokenizer

LLaMA-2-GTL supports a vocabulary size of up to `32000` tokens, which is same as the base model LLaMA2.

### Prompt Examples

Given the nature of the training data, the LLaMA-2-GTL series model is best suited for prompts using the prompt format as follows:
```markdown
You are an expert in health and fitness.
Based on the physical features of the individual, please predict the body fat percentage.
I will supply multiple instances with features and the corresponding label for your reference.
Please refer to the table below for detailed descriptions of the features and label:
--- feature description ---
Age: Age of the individual in years
Weight: Weight of the individual in kilograms
Height: Height of the individual in centimeters
Neck: Circumference of the neck in centimeters
Chest: Circumference of the chest in centimeters
Abdomen: Circumference of the abdomen in centimeters
Hip: Circumference of the hip in centimeters
Thigh: Circumference of the thigh in centimeters
Knee: Circumference of the knee in centimeters
Ankle: Circumference of the ankle in centimeters
Biceps: Circumference of the biceps in centimeters
Forearm: Circumference of the forearm in centimeters
Wrist: Circumference of the wrist in centimeters
Original: Indicates if the record is from the original dataset (Y) or if it was generated (N)
Sex: Gender of the individual (M for male, F for female)
--- label description --- 
BodyFat: Percentage of body fat
--- data ---
|Age|Weight|Height|Neck|Chest|Abdomen|Hip|Thigh|Knee|Ankle|Biceps|Forearm|Wrist|Original|Sex|BodyFat|
|33|83.58|1.75|40.7|98.9|92.1|103.5|64.0|37.3|23.5|33.5|30.6|19.7|Y|M|13.0|
|18|70.31|1.73|33.0|90.1|73.0|103.0|58.1|39.1|22.0|29.5|27.5|16.5|N|F|24.4|
|23|54.89|1.54|32.4|88.5|67.2|94.0|49.3|35.0|20.5|26.0|23.5|14.6|N|F|20.3|
|20|65.77|1.73|30.5|85.0|65.3|105.0|58.3|38.3|20.5|27.3|23.5|15.5|N|F|25.2|
|18|74.84|1.71|33.0|84.0|96.0|106.0|52.0|39.0|21.5|29.5|25.3|17.3|N|F|33.8|
|21|69.85|1.69|31.0|89.0|76.0|104.5|55.0|39.5|22.5|29.5|26.5|16.3|N|F|26.3|
|41|95.48|1.83|38.5|107.4|98.9|104.1|63.5|39.8|23.5|36.4|30.4|19.1|Y|M|20.4|
|27|97.98|1.93|39.4|103.6|90.9|107.7|66.2|39.2|25.9|37.2|30.2|19.0|Y|M|7.8|
|19|65.77|1.73|34.5|86.5|72.0|100.3|53.3|35.5|22.3|29.0|24.0|16.5|N|F|22.9|
|20|73.03|1.69|34.0|95.4|80.0|104.0|56.5|36.0|24.3|33.0|27.0|17.5|N|F|28.6|
|58|73.37|1.71|35.1|94.9|94.9|100.2|56.8|35.9|21.0|27.8|26.1|17.6|Y|M|26.7|
|19|64.86|1.63|32.3|85.5|68.3|98.3|55.0|39.0|24.0|26.5|24.5|16.2|N|F|23.3|
|19|74.39|1.68|34.0|96.0|87.0|107.0|56.0|39.0|22.4|29.5|24.5|16.0|N|F|31.4|
|24|83.58|1.81|34.4|97.3|100.0|101.9|63.2|42.2|24.0|32.2|27.7|17.7|Y|M|28.7|
|28|93.33|1.75|38.5|105.6|105.0|106.4|68.6|40.0|25.2|35.2|30.7|19.1|Y|M|31.2|
|41|99.11|1.8|39.8|111.7|100.5|108.3|67.1|44.2|25.2|37.5|31.5|18.7|Y|M|21.3|
|32|94.92|1.8|42.1|107.6|97.5|107.0|66.9|40.0|24.4|38.2|31.6|19.3|Y|M|<MASK>|
Please use the supplied data to predict the <MASK> BodyFat. 
Answer: 22.9
```

### Recover full model checkpoint

Please follow the document to [prepare the model checkpoint](https://github.com/xumwen/Industrial-Foundation-Models/tree/merge_refactor?tab=readme-ov-file#prepare-the-model-checkpoint).

### Sample inference code

This code shows how to quick start with running the model on a GPU:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the checkpoint
model = AutoModelForCausalLM.from_pretrained(
    CKPT_SAVE_PATH,                       # CKPT_SAVE_DIR/LLaMA-2-GTL/13B
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(CKPT_SAVE_PATH)

# Load example prompt
example_path = "data/prompt_examples/cls_in_context_table"
with open(example_path, "r") as f:
    full_prompt = f.read()
answer = full_prompt.split('Answer:')[-1].strip()
prompt_without_answer = full_prompt[:-len(answer)]
print("Prompt:", prompt_without_answer)
print("Groundtruth:", answer)

# Inference
inputs = tokenizer(prompt_without_answer, return_tensors="pt")
input_ids = inputs['input_ids']
max_new_tokens = 10
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=inputs['attention_mask'],
    max_new_tokens=max_new_tokens
)

# Print the answer
print("Generated answer:", tokenizer.decode(outputs[0][input_ids.shape[-1]:]))
```

## Responsible AI Considerations

Like other language models, the LLaMA-GTL series models can potentially behave in ways that are unfair, unreliable, or offensive. Some of the risks and limitations to be aware of include:

+ Data Bias: The model is trained on data that is not representative of the full range of industrial scenarios, and  it may produce biased predictions. This could include over-representation of certain types of data or under-representation of others . Biased price forecasting could result in inaccurate budgeting, misplaced investments, and other business strategy misalignments. In the healthcare sector, it can perform tasks such as health risk assessments. Unrepresentative data could lead to skewed assessments and potentially compromise patient care. We recommend the users to have a clear understanding of the context and the underlying assumptions before drawing conclusions from the predictions.   
+ Algorithmic Bias: Despite the advanced learning algorithm used, there might be inherent biases in the algorithm itself which could influence the prediction outcomes. We strongly recommend that users verify the predictions with other sources or domain experts before making crucial decisions based on the model's output.
+ Misinterpretation: There's a risk that users may misinterpret the predictions made by the model, leading to incorrect decisions.
+ Our model may inherit vulnerabilities from the base model.  

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include:

+ Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
+ High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context. 
+ Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).   
+ Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case. 
+ Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.


## Training and Evaluation

Please follow the [instruction](https://github.com/microsoft/Industrial-Foundation-Models) here to reproduce our [paper](https://arxiv.org/abs/2310.07338) results.

## License

The model is licensed under the [MIT license](https://github.com/microsoft/Industrial-Foundation-Models/blob/main/LICENSE).