---
language:
- en
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
---

# Granite 3.1 8B Instruct - Intrinsics LoRA v0.1

Welcome to Granite Experiments!

Think of Experiments as a preview of what's to come. These projects are still under development, but we wanted to let the open-source community take them for spin! Use them, break them, and help us build what's next for Granite – we'll 
keep an eye out for feedback and questions in the [Community section](https://huggingface.co/ibm-granite/granite-intrinsics-3.1-8b-lora-v0.1/discussions). Happy exploring!


## Model Summary

**Granite 3.1 8B Instruct - Intrinsics LoRA v0.1** is a LoRA adapter for [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct), 
providing access to the Uncertainty, Hallucination Detection, and Safety Exception intrinsics in addition to retaining the full abilities of the [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) model. 

It follows the same training pipeline as [ibm-granite/granite-intrinsics-3.0-8b-lora-v0.1](https://huggingface.co/ibm-granite/granite-intrinsics-3.0-8b-lora-v0.1), updated for Granite 3.1.

- **Developer:** IBM Research
- **Model type:** LoRA adapter for [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6602ffd971410cf02bf42c06/ornGz5BdtfIXLYxDzUgi9.png)

### Uncertainty Intrinsic
The Uncertainty intrinsic is designed to provide a Certainty score for model responses to user questions. 

**Certainty score definition** The model will respond with a number from 0 to 9, corresponding to 5%, 15%, 25%,...95% confidence respectively. 
This percentage is *calibrated* in the following sense: given a set of answers assigned a certainty score of X%, approximately X% of these answers should be correct. See the eval experiment below for out-of-distribution verification of this behavior.


### Hallucination Detection (RAG) Intrinsic
The Hallucination Detection intrinsic is designed to detect when an assistant response to a user question with supporting documents is not supported by those documents. Response with a `Y` indicates hallucination, and `N` no hallucination.  

### Safety Exception Intrinsic
The Safety Exception Intrinsic is designed to raise an exception when the user query is unsafe. This exception is raised by responding with `Y` (unsafe), and `N` otherwise. 
The Safety Exception intrinsic was designed as a binary classifier that analyses the user’s prompt to detect a variety of harms that include: violence, threats, sexual and explicit content and requests to obtain private identifiable information.


## Usage

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Intended use

This is an experimental LoRA testing new functionality being developed for IBM's Granite LLM family.  We are welcoming the community to test it out and give us feedback, but we are NOT recommending this model be used for real deployments at this time.  Stay tuned for more updates on the Granite roadmap.

**Granite 3.1 8B Instruct - Intrinsics LoRA v0.1** is lightly tuned so that its behavior closely mimics that of [ibm-granite/granite-3.0-8b-instruct](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct), 
with the added ability to generate the three specified intrinsics. 


### Invoking intrinsics
Each intrinsic is associated with its own generation role and has its own usage steps. Note that each intrinsic responds with only one token, and any additional text after this token should be ignored.  You can curb additional generation by setting "max token length" = 1 when using any intrinsic.

**Uncertainty Intrinsic Usage Steps** Answering a question and obtaining a certainty score proceeds as follows. 

1. Prompt the model with a system prompt (required) followed by the user prompt.  
2. Use the model to generate a response as normal (via the `assistant` role).
3. Invoke the Uncertainty intrinsic by generating in the `certainty` role (use "certainty" as the role in the chat template, or simply append `<|start_of_role|>certainty<|end_of_role|>` and continue generating), see examples below.
4. The model will respond with an integer certainty score from 0 to 9.  

The model was calibrated with the following system prompt: `You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.` 
You can further augment this system prompts for a given use case or task, but it is recommended your system prompt always starts with this string.

**Hallucination Detection Intrinsic Usage Steps** Answering a question and detecting hallucination proceeds as follows. 
1. Prompt the model with the system prompt (required) followed by the user prompt.  
2. Use the model to generate a response as normal (via the `assistant` role).
3. Invoke the Hallucination Detection intrinsic by generating in the `hallucination` role (use "hallucination" as the role in the chat template, or simply append `<|start_of_role|>hallucination<|end_of_role|>` and continue generating), see examples below.
4. The model will respond with `Y` or `N`.

**Safety Exception Intrinsic Usage Steps** Determining if a user query is safe proceeds as follows. 
1. Prompt the model with the system prompt (required) followed by the user prompt.
2. Invoke the Safety Exception intrinsic by generating in the `safety` role (use "safety" as the role in the chat template, or simply append `<|start_of_role|>safety<|end_of_role|>` and continue generating), see examples below.
3. The model will respond with `Y` (unsafe) or `N` (safe).

## Combining Intrinsics
In many pipelines, it may be desirable to invoke multiple intrinsics at different points. In a multi-turn conversation possibly involving other intrinsics, it is important to use 
attention masking to provide only the relevant information to the intrinsic of interest. We explore two frameworks for accomplishing this - [Prompt Declaration Language](https://github.com/IBM/prompt-declaration-language) (PDL) and SGLang. 

In the examples below, we explore the following RAG flow. First, a user query is provided with 
relevant documents provided by a RAG system. We can invoke the Safety Exception intrinsic to determine if the query is safe. If it is safe, we can proceed to generate an answer to the question as normal. Finally, 
we can evaluate the certainty and hallucination status of this reply by invoking the Uncertainty and Hallucination Detection intrinsics. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6602ffd971410cf02bf42c06/HpitI-3zeutXqduC2eUES.png)


### Intrinsics Example with PDL
Given a hosted instance of **Granite 3.1 8B Instruct - Intrinsics LoRA v0.1** at `API_BASE` (insert the host address here), this uses the [PDL language](https://github.com/IBM/prompt-declaration-language) to implement the RAG intrinsic invocation scenario described above.
Note that the hosted instance must be supported by LiteLLM ([https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers))

First, create a file `intrinsics.pdl` with the following content.
```
defs:
  system_prompt: "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."

  rag_prompt: "Provide a short response to the user's question based on the information present in the documents. If the documents lack the necessary details, inform the user that the question cannot be answered."

  document:
      |
      Disability housing grants for Veterans 
      We offer housing grants for Veterans and service members with certain service - connected disabilities so they can buy or change a home to meet their needs and live more independently. Changing a home might involve installing ramps or widening doorways. Find out if you re eligible for a disability housing grant and how to apply. 

      Can I get a Specially Adapted Housing (SAH) grant and how much funding does this grant offer? 
      You may be able to get an SAH grant if you re using the grant money to buy, build, or change your permanent home a home you plan to live in for a long time and you meet both of the requirements listed below. Both of these must be true. You: Own or will own the home , and Have a qualifying service - connected disability Qualifying service - connected disabilities include : The loss or loss of use of more than one limb The loss or loss of use of a lower leg along with the residuals lasting effects of an organic natural disease or injury Blindness in both eyes having only light perception along with the loss or loss of use of a leg Certain severe burns The loss or loss of use of one or both lower extremities feet or legs after September 11 , 2001, that makes it so you can t balance or walk without the help of braces, crutches, canes, or a wheelchair Note : Only 30 Veterans and service members each fiscal year FY can qualify for a grant based on the loss of extremities after September 11 , 2001. If you qualify for but don t receive a grant in 2019 because the cap was reached , you may be able to use this benefit in FY 2020 or future years if the law continues to give us the authority to offer these grants and we don t go beyond the new FY cap. For FY 2019 , you may be able to get up to 3 grants for a total of up to $85,645 through the SAH grant program. Learn more about how to apply for a housing grant 

      Can I get a Special Housing Adaptation (SHA) grant and how much funding does this grant offer? 
      You may be able to get an SHA grant if you re using the grant money to buy, build, or change your permanent home a home you plan to live in for a long time and you meet both of the requirements listed below. Both of these must be true : You or a family member own or will own the home , and You have a qualifying service - connected disability Qualifying service - connected disabilities include : Blindness in both eyes with 20/200 visual acuity or less The loss or loss of use of both hands Certain severe burns Certain respiratory or breathing injuries For FY 2019 , you may be able to get up to 3 grants for a total of up to $17,130 through the SHA grant program. Learn more about how to apply for a housing grant \n\nWhat if I need money to make changes to a family member s home that I m living in for a short period of time? \nYou may be able to get a Temporary Residence Adaptation TRA grant if you meet both of the requirements listed below. Both of these must be true. You: Qualify for an SAH or SHA grant see above , and Are living temporarily in a family member s home that needs changes to meet your needs If you qualify for an SAH grant , you can get up to $37,597 through the TRA grant program for FY 2019. If you qualify for an SHA grant , you can get up to $6,713 through the TRA grant program for FY 2019. 

      Apply for an SAH, SHA, or TRA grant 
      You can apply online right now by going to our eBenefits website. You ll need to sign in to eBenefits with your DS Logon basic or premium account. If you don t have a DS Logon account , you can register for one on the site. Go to eBenefits to apply.


  query: How much funding does the SAH grant offer?
text:
  - include: intrinsics-defs.pdl
  - defs:
      safe:
        call: get_safety
        args:
          query: ${ query }
  - role: system
    text: ${ system_prompt }
    contribute: [context]
  - if: ${ safe == "Y" }
    then: 
      text:
        - "\n\nDocuments: ${ document }\n\n ${ query }\n"
        - model: openai/granite-intrinsics-3.1-8b-instruct-v0.1
          def: answer
          parameters: {api_key: EMPTY, api_base: API_BASE, temperature: 0, stop: "\n"}
        - defs:  ## Implicit fork of context
            certainty:
              call: get_certainty         
            hallucination:  
              call: get_hallucination
        - "\nCertainty: ${ certainty }"
        - "\nHallucination: ${ hallucination }"
```
Next, create a file `intrinsics-defs.pdl` with the following content.

```
defs:
  apply_template:
    function:
      context: [{role: str, content: str}]
    return:
      text:
        for:
          c: ${ context }
        repeat:
          text:
            - <|start_of_role|>${ c.role }<|end_of_role|>
            - ${ c.content }
            - <|end_of_text|>
        join:
          with: "\n"
  get_intrinsic:
    function:
      intrinsic: str
    return:
      lastOf:
      - call: apply_template
        def: mycontext
        args: 
          context: ${ pdl_context }
      - model: openai/granite-intrinsics-3.1-8b-instruct-v0.1
        parameters:
          api_key: EMPTY
          api_base: API_BASE
          temperature: 0
          max_tokens: 1
          custom_llm_provider: text-completion-openai
          prompt: 
            |
            ${ mycontext }
            <|start_of_role|>${ intrinsic }<|end_of_role|>
  
  get_safety:
    function:
      query: str
    return:
      lastOf:
      - ${ query }
      - call: apply_template
        def: mycontext
        args: 
          context: ${ pdl_context }
      - call: get_intrinsic
        args: 
         intrinsic: safety

  get_hallucination:
    function:
    return:
      call: get_intrinsic
      args: 
        intrinsic: hallucination
  
  get_certainty:
    function:
    return:
      call: get_intrinsic
      args: 
        intrinsic: certainty
```

To run the example, in the command line run `pdl intrinsics.pdl` after installing the PDL CLI (`pip install prompt-declaration-language`).

### Intrinsics Example with SGLang
The below SGLang implementation uses the SGLang fork at [https://github.com/frreiss/sglang/tree/granite](https://github.com/frreiss/sglang/tree/granite) that supports Granite models. 

```python

import sglang as sgl 
from sglang.lang.chat_template import get_chat_template




# Input data
system_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."


@sgl.function
def safety_check (s, question):
    s += sgl.system (system_prompt)
    s += sgl.user(question)
    s += "<|start_of_role|>safety<|end_of_role|>" + sgl.gen("safety", temperature=0, max_tokens=1)
 
    # print ("\n====== Safety check state =======\n")
    # print (s)
    # print ("\n")

rag_prompt = "Provide a short response to the user's question based on the information present in the documents. If the documents lack the necessary details, inform the user that the question cannot be answered."

document = """Disability housing grants for Veterans 
We offer housing grants for Veterans and service members with certain service - connected disabilities so they can buy or change a home to meet their needs and live more independently. Changing a home might involve installing ramps or widening doorways. Find out if you re eligible for a disability housing grant and how to apply. 

Can I get a Specially Adapted Housing (SAH) grant and how much funding does this grant offer? 
You may be able to get an SAH grant if you re using the grant money to buy, build, or change your permanent home a home you plan to live in for a long time and you meet both of the requirements listed below. Both of these must be true. You: Own or will own the home , and Have a qualifying service - connected disability Qualifying service - connected disabilities include : The loss or loss of use of more than one limb The loss or loss of use of a lower leg along with the residuals lasting effects of an organic natural disease or injury Blindness in both eyes having only light perception along with the loss or loss of use of a leg Certain severe burns The loss or loss of use of one or both lower extremities feet or legs after September 11 , 2001, that makes it so you can t balance or walk without the help of braces, crutches, canes, or a wheelchair Note : Only 30 Veterans and service members each fiscal year FY can qualify for a grant based on the loss of extremities after September 11 , 2001. If you qualify for but don t receive a grant in 2019 because the cap was reached , you may be able to use this benefit in FY 2020 or future years if the law continues to give us the authority to offer these grants and we don t go beyond the new FY cap. For FY 2019 , you may be able to get up to 3 grants for a total of up to $85,645 through the SAH grant program. Learn more about how to apply for a housing grant 

Can I get a Special Housing Adaptation (SHA) grant and how much funding does this grant offer? 
You may be able to get an SHA grant if you re using the grant money to buy, build, or change your permanent home a home you plan to live in for a long time and you meet both of the requirements listed below. Both of these must be true : You or a family member own or will own the home , and You have a qualifying service - connected disability Qualifying service - connected disabilities include : Blindness in both eyes with 20/200 visual acuity or less The loss or loss of use of both hands Certain severe burns Certain respiratory or breathing injuries For FY 2019 , you may be able to get up to 3 grants for a total of up to $17,130 through the SHA grant program. Learn more about how to apply for a housing grant \n\nWhat if I need money to make changes to a family member s home that I m living in for a short period of time? \nYou may be able to get a Temporary Residence Adaptation TRA grant if you meet both of the requirements listed below. Both of these must be true. You: Qualify for an SAH or SHA grant see above , and Are living temporarily in a family member s home that needs changes to meet your needs If you qualify for an SAH grant , you can get up to $37,597 through the TRA grant program for FY 2019. If you qualify for an SHA grant , you can get up to $6,713 through the TRA grant program for FY 2019. 

Apply for an SAH, SHA, or TRA grant 
You can apply online right now by going to our eBenefits website. You ll need to sign in to eBenefits with your DS Logon basic or premium account. If you don t have a DS Logon account , you can register for one on the site. Go to eBenefits to apply.
"""

query = "How much funding does the SAH grant offer?"


# The following function processes a chat between a user and an assistant.
# For simplicity, this assumes a fixed document, but in a true RAG setting, the
# documents will be retrieved dynamically based on the user turns.  
@sgl.function
def main_chat_flow (s, doc, query):
    s += sgl.system (system_prompt)

    # Safety check on query
    state = safety_check (question=query) 
    print (f"Safety Output: {state["safety"]} for question: {query}\n")

    # RAG answer generation
    s += sgl.user(rag_prompt + "\n\nDocuments: " + doc + "\n\n" + query)
    s += sgl.assistant (sgl.gen ("answer", stop="\n", temperature=0, max_tokens=200))
    answer = s["answer"]
    print (f"Assistant: {answer}\n")

    # Hallucination check in parallel with uncertainty quantification for the generated answer
    forks = s.fork(2)
    for i, f in enumerate(forks):
        if (i == 0): 
            f += "<|start_of_role|>hallucination<|end_of_role|>"
            f += sgl.gen("hallucination", temperature=0, max_tokens=1)   
            # print ("\n====== Fork 0 state =======\n")
            # print (f) 
            # print ("\n")
        else:    
            f += "<|start_of_role|>certainty<|end_of_role|>"
            f += sgl.gen("certainty", temperature=0, max_tokens=1)
            # print ("\n====== Fork 1 state =======\n")
            # print (f)
            # print ("\n")
    
    print (f"Hallucination Output: {forks [0]["hallucination"]} for answer: {answer}\n")
    print (f"Certainty Output: {forks [1]["certainty"]} for answer: {answer}\n")

    

if __name__ == "__main__":
    model_path = "ibm-granite/granite-3.1-8b-lora-intrinsics-v0.1"

    # Setting the model_path to the granite model, and chat template to be the granite template
    # This assumes "granite3-instruct" chat template has been registered in "sglang/lang/chat_template.py"
    runtime = sgl.Runtime(model_path=model_path)
    runtime.endpoint.chat_template = get_chat_template("granite3-instruct")
    sgl.set_default_backend(runtime)

    state = main_chat_flow (doc=document, query=query)

```


#### Notes
**Certainty score interpretation** Certainty scores calibrated as defined above may at times seem biased towards moderate certainty scores for the following reasons. Firstly, as humans we tend to be overconfident in
our evaluation of what we know and don't know - in contrast, a calibrated model is less likely to output very high or very low confidence scores, as these imply certainty of correctness or incorrectness.
Examples where you might see very low confidence scores might be on answers where the model's response was something to the effect of "I don't know", which is easy to evaluate as not 
being the correct answer to the question (though it is the appropriate one). Secondly, remember that the model 
is evaluating itself - correctness/incorrectness that may be obvious to us or to larger models may be less obvious to an 8b model. Finally, teaching a model every fact it knows
and doesn't know is not possible, hence it must generalize to questions of wildly varying difficulty (some of which may be trick questions!) and to settings where it has not had its outputs judged. 
Intuitively, it does this by extrapolating based on related questions
it has been evaluated on in training - this is an inherently inexact process and leads to some hedging.
Certainty is inherently an intrinsic property of a model and its abilitities. The Uncertainty Intrinsic is not intended to predict the certainty of responses generated by any other models besides itself or [ibm-granite/granite-3.0-8b-instruct](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct). 
Additionally, certainty scores are *distributional* quantities, and so will do well on realistic questions in aggregate, but in principle may have surprising scores on individual
red-teamed examples.

## Evaluation
We evaluate the performance of the intrinsics themselves and the RAG performance of the model. 

We first find that the performance of the intrinsics in our shared model **Granite 3.1 8B Instruct - Intrinsics LoRA v0.1** is not degraded
versus the baseline procedure of maintaining 3 separate instrinsic models. Here, percent error is shown for the Hallucination Detection and Safety Exception intrinsics as they have
binary output, and Mean Absolute Error (MAE) is shown for the Uncertainty Intrinsic as it outputs numbers 0 to 9. For all, lower is better. Performance is calculated on a randomly drawn 400 sample validation set from each intrinsic's dataset. 


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6602ffd971410cf02bf42c06/NsvMpweFjmjIhWFaKtI-K.png)

We then find that RAG performance of **Granite 3.1 8B Instruct - Intrinsics LoRA v0.1** does not suffer with respect to the base model [ibm-granite/granite-3.0-8b-instruct](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct). Here we evaluate the RAGBench benchmark on RAGAS faithfulness and correction metrics.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6602ffd971410cf02bf42c06/hyOlQmXPirlCYeILLBXhc.png)

## Training Details
The **Granite 3.1 8B Instruct - Intrinsics LoRA v0.1** model is a LoRA adapter finetuned to provide 3 desired intrinsic outputs - Uncertainty Quantification, Hallucination Detection, and Safety.



### UQ Training Data
The following datasets were used for calibration and/or finetuning. Certainty scores were obtained via the method in [[Shen et al. ICML 2024] Thermometer: Towards Universal Calibration for Large Language Models](https://arxiv.org/abs/2403.08819).

* [BigBench](https://huggingface.co/datasets/tasksource/bigbench)
* [MRQA](https://huggingface.co/datasets/mrqa-workshop/mrqa)
* [newsqa](https://huggingface.co/datasets/lucadiliello/newsqa)
* [trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
* [search_qa](https://huggingface.co/datasets/lucadiliello/searchqa)
* [openbookqa](https://huggingface.co/datasets/allenai/openbookqa)
* [web_questions](https://huggingface.co/datasets/Stanford/web_questions)
* [smiles-qa](https://huggingface.co/datasets/alxfgh/ChEMBL_Drug_Instruction_Tuning)
* [orca-math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)
* [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc)
* [commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)
* [social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa)
* [super_glue](https://huggingface.co/datasets/aps/super_glue)
* [figqa](https://huggingface.co/datasets/nightingal3/fig-qa)
* [riddle_sense](https://huggingface.co/datasets/INK-USC/riddle_sense)
* [ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
* [medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
* [dream](https://huggingface.co/datasets/dataset-org/dream)
* [codah](https://huggingface.co/datasets/jaredfern/codah)
* [piqa](https://huggingface.co/datasets/ybisk/piqa)

### RAG Hallucination Training Data
The following public datasets were used for finetuning. The details of data creation for RAG response generation is available at [Granite Technical Report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf).
For creating the hallucination labels for responses, the technique available at [Achintalwar, et al.](https://arxiv.org/pdf/2403.06009) was used. 

* [MultiDoc2Dial](https://huggingface.co/datasets/IBM/multidoc2dial)
* [QuAC](https://huggingface.co/datasets/allenai/quac)

### Safety Exception Training Data
The following public datasets were used for finetuning.  

* [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned/discussions)
* [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/viewer/default/train) 
* A subset of [https://huggingface.co/datasets/Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
* Ibm/AttaQ
* [google/civil_comments](https://huggingface.co/datasets/google/civil_comments/blob/5cb696158f7a49c75722fd0c16abded746da3ea3/civil_comments.py)
* [allenai/social_bias_frames](https://huggingface.co/datasets/allenai/social_bias_frames)
  

    

 

## Model Card Authors 

Kristjan Greenewald,
Nathalie Baracaldo,
Chulaka Gunasekara,
Lucian Popa,
Mandana Vaziri