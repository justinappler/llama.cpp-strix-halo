---
license: other
license_name: cogvlm2
license_link: https://huggingface.co/THUDM/cogvlm2-video-llama3-chat/blob/main/LICENSE

language:
- en
pipeline_tag: text-generation
tags:
- chat
- cogvlm2
- cogvlm--video

inference: false
---

# VisionReward-Video

## Introduction
We present VisionReward, a general strategy to aligning visual generation models——both image and video generation——with human preferences through a fine-grainedand multi-dimensional framework. We decompose human preferences in images and videos into multiple dimensions,each represented by a series of judgment questions, linearly weighted and summed to an interpretable and accuratescore. To address the challenges of video quality assess-ment, we systematically analyze various dynamic features of videos, which helps VisionReward surpass VideoScore by 17.2% and achieve top performance for video preference prediction.
Here, we present the model of VisionReward-Video.

## Using this model
You can quickly install the Python package dependencies and run model inference in our [github](https://github.com/THUDM/VisionReward).
