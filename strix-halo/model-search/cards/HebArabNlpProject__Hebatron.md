---
language:
- he
- en
license: apache-2.0
library_name: mamba
tags:
- mamba2
- moe
- hebrew
- finance
- legal
- ssm
model_name: HEBATRON
base_model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
pipeline_tag: text-generation
---

![image](https://cdn-uploads.huggingface.co/production/uploads/60a75f5523ce37179774a20b/8kpWOrI4PKXZHu-o9ffG0.png)

# 🛡️ HEBATRON: Hebrew-Specialized Mamba2-MoE

HEBATRON is a state-of-the-art, high-performance language model specialized for the Hebrew language. Developed through a collaboration between **PwC Israel** and **MAFAT** and **AWS**, it introduces a unique hybrid architecture combining **Mamba2** and **Mixture-of-Experts (MoE)**.

## 🚀 Model Summary
HEBATRON is designed to handle the structural and morphological complexities of Hebrew while providing linear scaling for long-context tasks. It is a localized and enhanced version of the **Nemotron-3-Nano-30B** framework, optimized for native-level reasoning in Hebrew and English.

---

## 📂 Technical Specifications

| Feature | Specification |
| :--- | :--- |
| **Model Name** | HEBATRON |
| **Architecture** | Hybrid **Mamba2** (SSM) + **Sparse MoE** |
| **Total Parameters** | 31.6B |
| **Active Parameters** | ~3B per token |
| **Context Window** | 65,536 (64k) tokens |
| **Hardware** | NVIDIA Blackwell (B300) & H200 GPUs |
| **Precision** | FP8 Mixed-Precision |

---
## ⚙️ Deployment Configuration

To ensure optimal performance in production, the following environment variables and parameters are recommended for the **vLLM** backend:

### **Inference Engine (vLLM)**
* **Port:** `8002` (Default for Model B slot)
* **Max Model Length:** `65536` tokens
* **GPU Memory Utilization:** Recommended `0.90` - `0.95` for Blackwell/H200.

### **Model Parameters**
* **Max New Tokens:** `65536`
* **Temperature:** `0.7` (Balanced creativity and precision)
* **Top-P:** `0.9`

### **Server Settings**
* **Max Simultaneous Comparisons:** `1` (Recommended for 30B+ MoE on single node to maintain latency)
* **Chat Context Max Turns:** `10`
* **Max Prompt Characters:** `10000`

---
## 🧬 Training Curriculum
The model was trained using a three-phase **Curriculum Learning** strategy:

1. **Phase 1: Formal Foundation (75.5B tokens)**
   Focused on high-quality, structured Hebrew (legal, academic, and literary texts) to establish core grammatical rules.
2. **Phase 2: Colloquial Expansion (3.36B tokens)**
   Integration of social media, forums, and informal web data to handle slang and modern registers.
3. **Phase 3: Long-Context Extension (20.4B tokens)**
   Fine-tuning on dense, long-form documents to stabilize the 64k context window.
4. **Alignment:** Supervised Fine-Tuning (SFT) was performed on **2 million samples**, including localized knowledge distillation and the **"Hebrew IFEval"** dataset.

---

## 📊 Performance Evaluation

### Hebrew Reasoning Benchmarks
* **SNLI (Semantic Reasoning):** 91.2% accuracy
* **Israeli Trivia:** 72.1% (+14pt vs base)
* **Hebrew Average Reasoning:** 73.8% (Surpassing DictaLM-3.0-Thinking)
* **GSM8K (Math):** 83.3% accuracy in native Hebrew

### English Reasoning Benchmarks
* **Psychometric Psi (EN):** 91.6%
* **English Reasoning Average:** 86.0%

---

## 🎯 Intended Use & Limitations
* **Intended Use:** Advanced Hebrew document analysis, long-context summarization (legal/technical), and complex bilingual reasoning.
* **Limitations:** Users should verify outputs for factual accuracy as with any Large Language Model.

---

## 🤝 Credits
* **Developed by:** PwC Israel & MAFAT
* **MAFAT Lead:** Tal Geva [project Lead], Matan Frank
* **Technical Lead:** Sarel Weinberger (PwC Next)
* **PwC Israel Team:** Noam Kayzer, Dan Revital, Ori Bar Joseph, Smadar Arvatz, Or Levi, Kate Zinkovskaia, Zevi Apini, Omer Baruch (PwC Next)
* **MAFAT Team:** Noam Ordan, Nadav Cordova 
* **Partners:** Amir Nissan Hacohen (Origin.ai)
* **Research Collaborators:** Shaltiel Shmidman (Dicta), Mike Erlihson
* **AWS Infrastructures:** Ilouz Netanel