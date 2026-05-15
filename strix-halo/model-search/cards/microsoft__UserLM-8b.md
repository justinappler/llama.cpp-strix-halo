---
license: mit
datasets:
- allenai/WildChat-1M
language:
- en
base_model:
- meta-llama/Llama-3.1-8B
pipeline_tag: text-generation
tags:
- userlm
- simulation
---

# microsoft/UserLM-8b model card

## Model description

Unlike typical LLMs that are trained to play the role of the "assistant" in conversation, we trained UserLM-8b to simulate the “user” role in conversation (by training it to predict user turns in a large corpus of conversations called WildChat). This model is useful in simulating more realistic conversations, which is in turn useful in the development of more robust assistants. 

The model takes a single input, which is the “task intent”, which defines the high-level objective that the user simulator should pursue. The user can then be used to generate: (1) a first-turn user utterance, (2) generate follow-up user utterances based on a conversation state (one or several user-assistant turn exchanges), and (3) generate a <|endconversation|> token when the user simulator expects that the conversation has run its course. 

Developed by: Tarek Naous (intern at MSR Summer 2025), Philippe Laban (MSR), Wei Xu, Jennifer Neville (MSR) 

Paper: https://arxiv.org/abs/2510.06552

# How to get started with the model
 
Here’s a simple snippet to use the model: 
```python 
from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch 
 
# Load the model and tokenizer 
model_path = "microsoft/UserLM-8b"  
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) 
model = AutoModelForCausalLM.from_pretrained(model_path,  trust_remote_code=True).to("cuda") 
 
# Create a conversation 
messages = [{"role": "system", "content": "You are a user who wants to implement a special type of sequence. The sequence sums up the two previous numbers in the sequence and adds 1 to the result. The first two numbers in the sequence are 1 and 1."}] 
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda") 

end_token = "<|eot_id|>"
end_token_id = tokenizer.encode(end_token, add_special_tokens=False)

end_conv_token = "<|endconversation|>"
end_conv_token_id = tokenizer.encode(end_conv_token, add_special_tokens=False)

outputs = model.generate(
    input_ids=inputs,
    do_sample=True,
    top_p=0.8,
    temperature=1,
    max_new_tokens=10,
    eos_token_id=end_token_id,
    pad_token_id=tokenizer.eos_token_id,
    bad_words_ids=[[token_id] for token_id in end_conv_token_id]
)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True) 
print(response)
```

# Uses

## Direct intended uses

The UserLM-8b is released for use by researchers involved in the evaluation of assistant LLMs. In such scenarios, UserLM-8b can be used to simulate multi-turn conversations, with our analyses (see Section 3 of the paper) giving evidence that UserLM-8b provides more realistic simulation of user behavior than other methods (such as prompting an assistant model). UserLM-8b offers a user simulation environment that can better estimate the performance of an assistant LLM with real users. See Section 4 of the paper for an initial implementation of such an evaluation.

## Downstream uses

We envision several potential uses for UserLM-8b that we did not implement yet in our presented work but describe in our Discussion section as potential research directions for UserLMs. These potential applications include: (1) user modeling (i.e., predicting user responses to a given set of questions), (2) foundation for judge models (i.e., LLM-as-a-judge finetuning), (3) synthetic data generation (in conjunction with an assistant LM).

## Out-of-scope uses

We caution potential users of the model that UserLM-8b is **not an assistant LM**, unlike the majority of LLMs released on HuggingFace. As such, it is unlikely to be useful to end-users that require assistance with a task, for which an assistant LLM (such as microsoft/Phi-4) might be more appropriate. 

We do not recommend using UserLM in commercial or real-world applications without further testing and development. It is being released for research purposes.

# Risks and limitations

The paper accompanying this model release presents several evaluations of UserLM-8b and its potential limitations. 

First in Section 3, we describe the robustness experiments we conducted with UserLM-8b, which show that though the model can more robustly adhere to the user role and the provided task intent, the robustness numbers are not perfect (< 100%), meaning that the UserLM-8b can occasionally get detracted from its user role or its initial task intent. 

Second in Section 4, we describe the possibility for the UserLM-8b to hallucinate additional requirements that are not provided in the task intent. In such cases, we find that the UserLM can introduce new facts or constraints to the task. This can both be beneficial (diversifying simulation conditions) and detrimental (e.g., in cases where the hallucination is incompatible with the task intent). Hallucination mitigation is unfortunately an unsolved research problem, and all generative models (including UserLMs) generate hallucinated text on occasion. One mitigation option is to provide user intents that are as specified as possible, which limits the opportunities for the UserLM to hallucinate task information. 

UserLM was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.  

UserLM inherits any biases, errors, or omissions produced by its base model. Developers are advised to choose an appropriate base LLM/MLLM carefully, depending on the intended use case. 

UserLM inherits any biases, errors, or omissions characteristic of its training data, which may be amplified by any AI-generated interpretations. 

There has not been a systematic effort to ensure that systems using UserLM are protected from security vulnerabilities such as indirect prompt injection attacks. Any systems using it should take proactive measures to harden their systems as appropriate. 


# Recommendations

The UserLM-8b is a research release, and it is likely to require some adaptation when adapted to new tasks and environments. In Appendix D.1 of the paper (Generation Configuration for UserLM-8b), we describe four generation guardrails (Filtering First Tokens, Avoiding Dialogue Termination, Maximal and Minimal Length Threshold, and Filter Verbatim Repetitions) we implemented to get the UserLM-8b to effectively simulate user utterances on the use-cases described in our paper. We encourage users of UserLM-8b to adopt and adapt these guardrails in their own use-cases.

# Training details 
## Training data 
We trained on a filtered version of [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M). The details on the filtering and processing are Appendix A and Section 2 of our paper. We do not release any data or processing scripts with our paper, as we believe these are sufficiently detailed in the paper that they can be reimplemented. 
 
## Training procedure 
 
We performed full-parameter fine-tuning of Llama3-8b-Base. We used a maximum sequence length of 2048 tokens, a batch size of 1024 samples, and a learning rate of 2e-5. Training was performed on four NVIDIA RTX A6000 GPUs, taking 227 hours to train UserLM-8b. Further details are provided in Section 2.2 of our paper. 

# Evaluation

## Testing data
We evaluated on a held-out set of [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), as well as [PRISM](https://huggingface.co/paige-ai/Prism). In our extrinsic evaluation, we evaluated using samples from the [Lost in Conversation](https://huggingface.co/datasets/microsoft/lost_in_conversation) sharded instructions. 
The details for data selection are described in Section 2.2 of the paper. 
 
## Evaluation results summary

We evaluate UserLM-8b with three complementary experiments: 
1. Distributional Alignment (perplexity). We measure the ability of the UserLM-8b to predict (generate) user utterances for a set of test conversations from users that were not included in training. We observe lower perplexity (higher alignment) than prior work, including previously trained models (USP-8b) and prompted assistant models. See Section 2 of the paper for details. 
2. Instrinsic Evaluation of User Simulators. We define six metrics that correspond to desirable properties of user simulators (for example, ability to end conversation, or shard information across turns). We then compare UserLM-8b to a broad set of methods including trained models, and open-weights and API-based prompted assistant models. We find that UserLM-8b outperforms assistant-based methods on all six metrics. See Section 3 of the paper for details. 
3. Extrinsic Evaluation of User Simulators. We create a simulation setting for conversations involving the user wanting to either solve mathematics problems, or wanting to code a basic Python programming function. We simulate conversations with various user simulators, including UserLM-8b and prompted assistant models, and find that UserLM-8b leads to more diverse simulation on several levels (conversation pace, lexical choice, information choice), leading to a broader range of simulation, and leading to deteriorated performance from the assistant. See Section 4 of the paper for details. 


# Environmental impact 
Carbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019). 
- Hardware type: A6000 
- Hours used: 227*4 
- Cloud provider: Azure 
- Compute region: useast 
- Carbon emitted: 115 kg CO2 (estimate) 

# BibTeX

```tex
@inproceedings{naous2025flipping,
  title={Flipping the Dialogue: Training and Evaluating User Language Models},
  author={Tarek Naous and Philippe Laban and Wei Xu and Jennifer Neville},
  journal={arXiv preprint arXiv:2510.06552},
  year={2025},
  url={https://arxiv.org/abs/2510.06552}
}
```

# Glossary

UserLM = User Language Model

# Model card contact

This research was conducted by members of Microsoft Research.  We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us at: plaban@microsoft.com 

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.

# Privacy

[Privacy & Cookies](https://www.microsoft.com/en-us/privacy/privacystatement)