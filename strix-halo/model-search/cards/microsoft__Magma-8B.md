---
library_name: transformers
license: mit
pipeline_tag: robotics
---

# Model Card for Magma-8B

<!-- Provide a quick summary of what the model is/does. -->

<div align="center">
<h2>Magma: A Foundation Model for Multimodal AI Agents</h2>

[Jianwei Yang](https://jwyang.github.io/)<sup>*</sup><sup>1</sup><sup>†</sup>&nbsp;
[Reuben Tan](https://cs-people.bu.edu/rxtan/)<sup>1</sup><sup>†</sup>&nbsp;
[Qianhui Wu](https://qianhuiwu.github.io/)<sup>1</sup><sup>†</sup>&nbsp;
[Ruijie Zheng](https://ruijiezheng.com/)<sup>2</sup><sup>‡</sup>&nbsp;
[Baolin Peng](https://scholar.google.com/citations?user=u1CNjgwAAAAJ&hl=en&oi=ao)<sup>1</sup><sup>‡</sup>&nbsp;
[Yongyuan Liang](https://cheryyunl.github.io)<sup>2</sup><sup>‡</sup>

[Yu Gu](http://yu-gu.me/)<sup>1</sup>&nbsp;
[Mu Cai](https://pages.cs.wisc.edu/~mucai/)<sup>3</sup>&nbsp;
[Seonghyeon Ye](https://seonghyeonye.github.io/)<sup>4</sup>&nbsp;
[Joel Jang](https://joeljang.github.io/)<sup>5</sup>&nbsp;
[Yuquan Deng](https://scholar.google.com/citations?user=LTC0Q6YAAAAJ&hl=en)<sup>5</sup>&nbsp;
[Lars Liden](https://sites.google.com/site/larsliden)<sup>1</sup>&nbsp;
[Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)<sup>1</sup><sup>▽</sup>

<sup>1</sup> Microsoft Research; <sup>2</sup> University of Maryland; <sup>3</sup> University of Wisconsin-Madison  
<sup>4</sup> KAIST; <sup>5</sup> University of Washington

<sup>*</sup> Project lead  <sup>†</sup> First authors  <sup>‡</sup> Second authors  <sup>▽</sup> Leadership  

\[[arXiv Paper](https://www.arxiv.org/pdf/2502.13130)\] &nbsp; \[[Project Page](https://microsoft.github.io/Magma/)\] &nbsp; \[[Hugging Face Paper](https://huggingface.co/papers/2502.13130)\] &nbsp; \[[Github Repo](https://github.com/microsoft/Magma)\] &nbsp; \[[Video](https://www.youtube.com/watch?v=SbfzvUU5yM8)\] 

</div>

## Agents

### UI Navigation
<div align="center">
<div align="center" style="display: inline-block; width: 48%;">
<video  autoplay muted loop controls playsinline style="margin-bottom: 2px;">
    <source src="https://microsoft.github.io/Magma/static/videos/ui_weather_and_flight_mode.mp4" type="video/mp4">
</video>
    <p class="is-5 has-text-centered" style="font-size: 14px;">What's weather in Seattle? & turn on flight mode</p>
</div>
<div align="center" style="display: inline-block; width: 48%;">
<video  autoplay muted loop controls playsinline style="margin-bottom: 2px;">
    <source src="https://microsoft.github.io/Magma/static/videos/ui_wordle.mp4" type="video/mp4">
</video>
    <p class="is-5 has-text-centered" style="font-size: 14px;">Share and message this to Bob Steve. Click send button</p>
</div>
</div>

### Robot Manipulation
<div align="center">
<div align="center">
    <div style="display: flex; justify-content: space-between; gap: 1%;">
        <div style="width: 32%;">
            <video autoplay muted loop controls playsinline height="98%" style="max-width: 450px; width: 100%; border-radius: 10px; overflow: hidden; margin-bottom: 5px;">
                <source src="https://microsoft.github.io/Magma/static/videos/magma_hotdog.mp4" type="video/mp4">
            </video>
        </div>
        <div style="width: 32%;">
            <video autoplay muted loop controls playsinline height="98%" style="max-width: 450px; width: 100%; border-radius: 10px; overflow: hidden; margin-bottom: 5px;">
                <source src="https://microsoft.github.io/Magma/static/videos/magma_mushroom.mp4" type="video/mp4">
            </video>
        </div>
        <div style="width: 32%;">
            <video autoplay muted loop controls playsinline height="98%" style="max-width: 450px; width: 100%; border-radius: 10px; overflow: hidden; margin-bottom: 5px;">
                <source src="https://microsoft.github.io/Magma/static/videos/magma_left.mp4" type="video/mp4">
            </video>
        </div>
    </div>
</div>
<div align="center">
<div style="display: flex; justify-content: space-between; gap: 1%;">
    <div style="width: 32%;">
        <p style="text-align: center;font-size: 14px;margin-top: 0;">Pick Place Hotdog Sausage</p>
    </div>
    <div style="width: 32%;">
        <p style="text-align: center;font-size: 14px;margin-top: 0;">Put Mushroom Place Pot</p>
    </div>
    <div style="width: 32%;">
        <p style="text-align: center;font-size: 14px;margin-top: 0;">Push Cloth Left to Right (Out-of-Dist.)</p>
    </div>
</div>
</div>
</div>

### Gaming

Task: Model controls the robot to collect green blocks.

<div align="center">
<div align="center" style="display: inline-block; width: 48%;">
<video  autoplay muted loop controls playsinline style="margin-bottom: 2px;">
    <source src="https://microsoft.github.io/Magma/static/videos/magma_vs_llava.mp4" type="video/mp4">
</video>
    <p class="is-5 has-text-centered" style="font-size: 14px;">Magma v.s. LLaVA-OneVision</p>
</div>
<div align="center" style="display: inline-block; width: 48%;">
<video  autoplay muted loop controls playsinline style="margin-bottom: 2px;">
    <source src="https://microsoft.github.io/Magma/static/videos/magma_vs_gpt4omini.mp4" type="video/mp4">
</video>
    <p class="is-5 has-text-centered" style="font-size: 14px;">Magma v.s. GPT4o-minni</p>
</div>
</div>

## Model Details

<div align="center">
<img src="https://github.com/microsoft/Magma/blob/main/assets/images/magma_teaser.png?raw=true" width="100%">
</div>

### Model Description

<!-- Provide a longer summary of what this model is. -->

Magma is a multimodal agentic AI model that can generate text based on the input text and image. The model is designed for research purposes and aimed at knowledge-sharing and accelerating research in multimodal AI, in particular the multimodal agentic AI. The main innovation of this model lies on the introduction of two technical innovations: **Set-of-Mark** and **Trace-of-Mark**, and the leverage of a **large amount of unlabeled video data** to learn the spatial-temporal grounding and planning. Please refer to our paper for more technical details. 

### Highlights
* **Digital and Physical Worlds:** Magma is the first-ever foundation model for multimodal AI agents, designed to handle complex interactions across both virtual and real environments!
* **Versatile Capabilities:** Magma as a single model not only possesses generic image and videos understanding ability, but also generate goal-driven visual plans and actions, making it versatile for different agentic tasks!
* **State-of-the-art Performance:** Magma achieves state-of-the-art performance on various multimodal tasks, including UI navigation, robotics manipulation, as well as generic image and video understanding, in particular the spatial understanding and reasoning!
* **Scalable Pretraining Strategy:** Magma is designed to be **learned scalably from unlabeled videos** in the wild in addition to the existing agentic data, making it strong generalization ability and suitable for real-world applications!


## License

The model is developed by Microsoft and is funded by Microsoft Research. The model is shared by Microsoft Research and is licensed under the MIT License.

<!-- {{ model_description | default("", true) }}

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **Funded by [optional]:** {{ funded_by | default("[More Information Needed]", true)}}
- **Shared by [optional]:** {{ shared_by | default("[More Information Needed]", true)}}
- **Model type:** {{ model_type | default("[More Information Needed]", true)}}
- **Language(s) (NLP):** {{ language | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}} -->

## How to Get Started with the Model

<!-- {{ get_started_code | default("[More Information Needed]", true)}} -->

To get started with the model, you first need to make sure that `transformers` and `torch` are installed, as well as installing the following dependencies:

```bash
pip install torchvision Pillow open_clip_torch
```

⚠️ Please note that you need to install our customized transformers lib:
```bash
pip install git+https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2
```
See [here](https://github.com/microsoft/Magma?tab=readme-ov-file#installation) for the reason why you need this.

Then you can run the following code:

```python
import torch
from PIL import Image
from io import BytesIO
import requests

from transformers import AutoModelForCausalLM, AutoProcessor

# Load the model and processor
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
model.to("cuda")

# Inference
url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png"
image = Image.open(BytesIO(requests.get(url, stream=True).content))
image = image.convert("RGB")

convs = [
    {"role": "system", "content": "You are agent that can see, talk and act."},
    {"role": "user", "content": "<image_start><image><image_end>
What is in this image?"},
]
prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
inputs = processor(images=[image], texts=prompt, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
inputs = inputs.to("cuda").to(dtype)

generation_args = { 
    "max_new_tokens": 128, 
    "temperature": 0.0, 
    "do_sample": False, 
    "use_cache": True,
    "num_beams": 1,
}

with torch.inference_mode():
    generate_ids = model.generate(**inputs, **generation_args)

generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
print(response)
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

<!-- {{ training_data | default("[More Information Needed]", true)}} -->

Our training data consists of:

* Generic Image SFT Data: [LLaVA-Next](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [InfoGrpahicVQA](https://www.docvqa.org/datasets/infographicvqa), [ChartQA_Augmented](https://github.com/vis-nlp/ChartQA), [FigureQA](https://www.microsoft.com/en-us/research/project/figureqa-dataset/), [TQA](https://paperswithcode.com/dataset/tqa), [ScienceQA](https://scienceqa.github.io/).

* Generic Video SFT Data: [ShareGPT4Video](https://sharegpt4video.github.io/) and [LLaVA-Video](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K).

* Instructional Video Data: [Ego4d](https://ego4d-data.org/), [Somethingv2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset), [Epic-Kitchen](https://epic-kitchens.github.io/2025) and other related instructional videos.

* Robotics Manipulation Data: [Open-X-Embodiment](https://robotics-transformer-x.github.io/).

* UI Grounding Data: [SeeClick](https://github.com/njucckevin/SeeClick).\

* UI Navigation Data: [Mind2web](https://osu-nlp-group.github.io/Mind2Web/) and [AITW](https://github.com/google-research/google-research/tree/master/android_in_the_wild).

The data collection process involved sourcing information from publicly available documents, with a meticulous approach to filtering out undesirable documents and images. To safeguard privacy, we carefully filtered various image and text data sources to remove or scrub any potentially personal data from the training data.

More details can be found in our paper.

[Microsoft Privacy Notice](https://go.microsoft.com/fwlink/?LinkId=521839)

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

<!-- {{ preprocessing | default("[More Information Needed]", true)}} -->
In addition to the text-related preprocessing, we mainly undertake the following image and video preprocessing steps:

* UI Grounding and Navigation Data: For each UI screenshot, we extract the bounding boxes for the UI elements, and apply [Set-of-Mark Prompting](https://arxiv.org/abs/2310.11441) to overlay numeric marks on the raw image. The model is trained to generate the UI grounding text based on the image and the Set-of-Mark prompts.

* Instruction Video Data: For each video clip, we apply [Co-Tracker](https://co-tracker.github.io/) to extract the grid traces and then apply filtering algorithm to remove the noisy or static points. For videos that bear camera motion, we further apply homography transformation to stabilize the video clips. In the end, we assign a numeric mark for each trace which gives us a set of trace-of-mark. The model is trained to generate the trace-of-mark given the video clips and instructional text.

* Robotics Manipulation Data: For robotics data in Open-X Embodiment, we extract the 7 DoF robot gripper state and also extract the trace-of-mark from the video clips. Similar filtering and stabilization steps are applied to the video clips. The model is trained to generate the robot manipulation action as well as the trace-of-mark given the video clips and instructional text.

After all these preprocessing, we combine them with existing text annotations to form our final multimodal training data. We refer to our paper for more technical details.

#### Training Hyperparameters

<!-- - **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

We used bf16 mixed precision for training on H100s and MI300s. We used the following hyperparameters for training:

* Batch size: 1024
* Learning rate: 1e-5
* Max sequence length: 4096
* Resolution: maximally 1024x1024 for image, 512x512 for video frame.
* Pretraining Epochs: 3


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->
We evaluate the model in zero-shot manner on a wide range of tasks, mostly agent-related tasks.

### Testing Data, Factors & Metrics
<!-- This should link to a Dataset Card if possible. -->

<!-- {{ testing_data | default("[More Information Needed]", true)}} -->

<!-- #### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

<!-- {{ testing_factors | default("[More Information Needed]", true)}} -->

#### Zero-shot Testing Data

We evaluate the model's zero-shot performance on the following datasets:

* UI Grounding: [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) and [VisualWebArena](https://jykoh.com/vwa).

* Robotics Manipulation: [SimplerEnv](https://jykoh.com/vwa) and WidowX real robot.

* Spatial Understanding and Reasoning: [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning), [BLINK](https://zeyofu.github.io/blink/) and [SpatialEval](https://spatialeval.github.io/).



#### Finetuned Testing Data

We evaluate the model's performance after finetuning on the following datasets:

* UI Navigation: [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) and [AITW](https://github.com/google-research/google-research/tree/master/android_in_the_wild).

* Robotics Manipulation: [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and WidowX real robot.

* Multimodal Image Understanding and Reasoning: [VQAv2](https://visualqa.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html), [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), [POPE](https://huggingface.co/datasets/lmms-lab/POPE), [TextVQA](https://textvqa.org/), [ChartQA](https://github.com/vis-nlp/ChartQA), [DocVQA](https://www.docvqa.org/).

* Multimodal Video Understanding and Reasoning: [Next-QA](https://github.com/doc-doc/NExT-QA), [VideoMME](https://video-mme.github.io/home_page.html), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench).

#### Metrics
<!-- {{ testing_metrics | default("[More Information Needed]", true)}} -->

We follow the individual dataset's evaluation metrics for the evaluation. Please refer to the original dataset for more details.


### Results on Agentic Intelligence

Zero-shot evaluation on agentic intelligence. We report the results for pretrained Magma without any domain-specific finetuning. Magma is the only model that can conduct the full task spectrum.

| Model                 | VQAv2 | TextVQA | POPE  | SS-Mobile | SS-Desktop | SS-Web | VWB-Ele-G | VWB-Act-G | SE-Google Robot | SE-Bridge |
|-----------------------|------|--------|------|----------|-----------|------|----------|----------|---------------|-----------|
| GPT-4V               | 77.2 | 78.0   | n/a  | 23.6 | 16.0 | 9.0 | 67.5 | 75.7 | - | - |
| GPT-4V-OmniParser    | n/a  | n/a    | n/a  | 71.1 | 45.6 | 58.5 | - | - | - | - |
| LLava-1.5           | 78.5 | 58.2   | 85.9  | -  | - | -  | 12.1 | 13.6 | - | - |
| LLava-Next          | 81.3 | 64.9   | 86.5  | -  | - | -  | 15.0 | 8.7 | - | - |
| Qwen-VL             | 78.8 | 63.8   | n/a  | 6.2  | 6.3 | 3.0  | 14.0 | 0.7 | - | - |
| Qwen-VL-Chat        | 78.2 | 61.5   | n/a  | -  | - | -  | - | - | - | - |
| Fuyu                | 74.2 | n/a    | n/a  | 21.2 | 20.8 | 19.2 | 19.4 | 15.5 | - | - |
| SeeClick            | -    | -      | -    | 65.0 | 51.1 | 44.1 | 9.9 | 1.9 | - | - |
| Octo               | -    | -      | -    | -  | - | -  | - | - | - | - |
| RT-1-X             | -    | -      | -    | -  | - | -  | - | - | 6.0 | 15.9 |
| OpenVLA            | -    | -      | -    | -  | - | -  | - | - | 34.2 | 1.1 |
| Magma-8B           | 80.0 | 66.5   | 87.4  | 59.5 | 64.1 | 60.6 | 96.3 | 71.8 | 52.3 | 35.4 |

*Notes: SS - ScreenSpot, VWB - VisualWebArena, SE - SimplerEnv*
<!-- {{ results | default("[More Information Needed]", true)}} -->

<!-- {{ results_summary | default("", true) }} -->


## Technical Specifications 


### Model Architecture and Objective

<!-- {{ model_specs | default("[More Information Needed]", true)}} -->

* Language Model: We use [Meta LLama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) as the backbone LLM.
* Vision Encoder: We use [CLIP-ConvneXt-XXLarge](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg) trained by LAION team as the vision encoder to tokenize the images and videos.

The whole pipeline follows the common practice in the multimodal LLMs, where the vision encoder is used to tokenize the images and videos, and then the visual tokens are fed into the LLM along with the textual tokens to generate the text outputs.


### Compute Infrastructure
<!-- {{ compute_infrastructure | default("[More Information Needed]", true)}} -->

We used [Azure ML](https://azure.microsoft.com/en-us/products/machine-learning) for our model training.


#### Hardware
<!-- {{ hardware_requirements | default("[More Information Needed]", true)}} -->

Our model is trained on two GPUs:

* Nvidia H100
* AMD MI300



#### Software
<!-- {{ software | default("[More Information Needed]", true)}} -->

Our model is built based on:

* [Pytorch](https://pytorch.org/)
* [Transformers](https://huggingface.co/transformers/)
* [TorchVision](https://pytorch.org/vision/stable/index.html)
* [DeepSpeed](https://www.deepspeed.ai/)
* [FlashAttention](https://github.com/HazyResearch/flash-attention)


## Intended Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

This model is intended for broad research use in English. It is designed only for research purposes and aimed at knowledge-sharing and accelerating research in multimodal AI, particularly in multimodal agentic AI. It is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model takes images and text as inputs, and produces the textual outputs for the following uses:

* **Image/Video-Conditioned Text Generation:** The model can generate text (e.g., descriptions, answers) based on the input text and image.

* **Visual Planning Capabilities:** The model can also produce the visual trace as the future planning to accomplish a task (e.g., move object from one place to another).

* **Agentic Capabilities:** The model can also generate UI grounding (e.g., click ``search'' button) and robotics manipulations (e.g., 7 DoF for the robot gripper).



### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

<!-- {{ downstream_use | default("[More Information Needed]", true)}} -->

<!-- ### Out-of-Scope Use -->

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

<!-- {{ out_of_scope_use | default("[More Information Needed]", true)}} -->

The model can be further finetuned for different downstream tasks, such as:

* **Image Captioning and QA:** We can further finetune this model for image captioning and QA tasks under the pipeline of multimodal LLMs. Based on our experiments, the model can achieve competitive performance yet better spatial understanding and reasoning on these tasks.

* **Video Captioning and QA:** We can further finetune this model for video captioning and QA tasks under the pipeline of multimodal LLMs. Based on our experiments, the model can achieve competitive performance yet better temporal understanding and reasoning on these tasks.

* **UI Navigation:** We can finetune this model for specific UI navigation tasks, such as web navigation or mobile navigation. The model can achieve superior performance on these tasks.

* **Robotics Manipulation:** Our model can be further finetuned for robotics tasks given its general agentic capabilities as a vision-language-action model. After finetuning, our model significantly outperforms the state-of-the-art models such as OpenVLA on robotics manipulation tasks.


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

<!-- {{ bias_risks_limitations | default("[More Information Needed]", true)}} -->

Please note that this model is not specifically designed or evaluated for all downstream purposes. 

The model is not intended to be deployed in production settings. It should not be used in high-risk scenarios, such as military and defense, financial services, and critical infrastructure systems.

Developers should consider common limitations of multimodal models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case. 

Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case. Like other multimodal models, Magma can potentially behave in ways that are unfair, unreliable, or offensive. 

The models' outputs do not reflect the opinions of Microsoft. 

Some of the limiting behaviors to be aware of include:   

* **Quality of Service:** The model is trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English. Magma is not intended to support multilingual use. 

* **Representation of Harms & Perpetuation of Stereotypes:** These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases.  

* **Inappropriate or Offensive Content:** These models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.  

* **Information Reliability:** Multimodal models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated. 

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Using safety services like [Azure AI Content Safety](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety) that have advanced guardrails is highly recommended.


### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

<!-- {{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}} -->

Magma was developed for research purposes only. Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

The recommended usage for the finetuned models is within the research settings they were trained on — namely, 
-	an android simulator running on a computer for UI manipulation.
-	an enclosure equipped with a robotic arm and everyday objects for Robotic manipulation

For UI navigation task, researchers should make sure a human is in the loop and in control for every action the agentic system generates. Since the model cannot act by itself, the sub-module a researcher uses to actually perform the UI navigation action should ensure no unintended consequences can occur as a result of performing the UI action proposed by the model.

For the robotic manipulation task, some mitigation strategies to use for human safety when operating robotic arms include:

* **Safety Zones and Barriers:** Establish physical barriers or safety zones around robotic workspaces to prevent unauthorized access.
* **Emergency Stop Systems:** Equip robotic arms with easily accessible emergency stop buttons. Implement a fail-safe mechanism that triggers an immediate stop of operations in case of an emergency
* **Safety Standards and Compliance:** Adhere to established safety standards (e.g., ISO 10218, ISO/TS 15066) for industrial robots and collaborative robots.
* **User Training and Awareness:** Provide comprehensive training for all personnel working around robotic arms to understand their functions, safety features, and emergency procedures. Promote awareness of the potential risks associated with robotic manipulation.


## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

```bibtex
@misc{yang2025magmafoundationmodelmultimodal,\
      title={Magma: A Foundation Model for Multimodal AI Agents}, \
      author={Jianwei Yang and Reuben Tan and Qianhui Wu and Ruijie Zheng and Baolin Peng and Yongyuan Liang and Yu Gu and Mu Cai and Seonghyeon Ye and Joel Jang and Yuquan Deng and Lars Liden and Jianfeng Gao},\
      year={2025},\
      eprint={2502.13130},\
      archivePrefix={arXiv},\
      url={https://arxiv.org/abs/2502.13130}, \
}
```
<!-- {{ citation_bibtex | default("[More Information Needed]", true)}} -->

## Data Summary
https://huggingface.co/microsoft/Magma-8B/blob/main/data_summary_card.md