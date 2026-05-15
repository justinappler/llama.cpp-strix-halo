---
language:
- en
license: apache-2.0
base_model:
- mistralai/Mixtral-8x7B-Instruct-v0.1
library_name: transformers
tags:
- dialog-generation
- conversational-ai
- state-action-model
datasets:
- sharegpt
metrics:
- custom: emotional-intelligence
---

# SAGE Dialogue Gen 🌱

**Authors**: Yizhe Zhang, Navdeep Jaitly (Apple)

---

## Model Information

- **Language**: English
- **License**: Apache 2.0
- **Base Model**: mistralai/Mixtral-8x7B-Instruct-v0.1
- **Library**: transformers
- **Tags**: dialog-generation, conversational-ai, state-action-model
- **Dataset**: ShareGPT
- **Metrics**: Custom emotional-intelligence evaluation

---

## Citation

```bibtex
@misc{zhang2025sage,
  title = {SAGE: Steering and Refining Dialogue Generation with State‑Action Augmentation},
  author = {Zhang, Yizhe and Jaitly, Navdeep},
  year = {2025},
  howpublished = {arXiv preprint},
  note = {arXiv:2503.03040}
}
```

📄 **Paper**: Available on arXiv and Papers with Code

---

## Model Description

SAGE introduces **latent state-action variables** between dialogue turns, enabling:

- **Structured Control**: Precise management of emotional tone and conversational strategy
- **Enhanced Emotional Intelligence**: Explicit state planning for more empathetic responses
- **Self-Improving Pipeline**: Comprehensive training approach including:
  - Data augmentation
  - Dialogue-tree search
  - Reward modeling
  - Fine-tuning optimization

This approach allows for more nuanced and contextually appropriate dialogue generation compared to traditional methods.

---

## Intended Uses

### ✅ **Recommended Applications**
- Emotional or empathetic chatbots
- Long-horizon, strategy-aware conversation systems
- Research on structured latent-variable dialogue control
- Educational conversational AI systems
- Customer service applications requiring emotional intelligence

### ⚠️ **Important Limitations**
- **Not suitable** for high-stakes, safety-critical deployment without further evaluation
- Requires additional testing for production environments
- May need domain-specific fine-tuning for specialized applications

---

## Training Details

**Base Model**: Mixtral-8x7B-Instruct

**Training Pipeline**:
1. **Data Preparation**: ShareGPT-style JSON formatting
2. **Supervised Fine-Tuning (SFT)**: Initial model adaptation
3. **Dialogue-Tree Search**: Exploration of conversation paths
4. **Preference Learning**: Reward model training
5. **Comparative Evaluation**: Performance assessment and inference optimization

---

## Performance

SAGE demonstrates significant improvements on emotional-intelligence metrics compared to baseline models while maintaining generative flexibility and coherence. The model shows particular strength in:

- Emotional tone consistency
- Contextual appropriateness
- Long-term conversation planning
- Empathetic response generation

---

## Usage

### Quick Start

```bash
git clone https://github.com/apple/ml-sage-dialog-gen
cd ml-sage-dialog-gen
bash setup.sh
```

### Basic Implementation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
tokenizer = AutoTokenizer.from_pretrained("apple/sage-dialogue-gen")
model = AutoModelForCausalLM.from_pretrained("apple/sage-dialogue-gen")

# Generate dialogue
input_text = "I'm feeling overwhelmed with work lately."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.21+
- Additional dependencies listed in `requirements.txt`

---

## Contributing

Contributions are welcome! Please see our contributing guidelines and code of conduct before submitting pull requests.

---

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

---

## Acknowledgments

- Built upon the Mixtral-8x7B-Instruct foundation model
- Trained using ShareGPT dataset
- Developed by the Apple Machine Learning Research team

---

## Contact

For questions or issues, please open a GitHub issue or contact the development team through the official Apple ML research channels.