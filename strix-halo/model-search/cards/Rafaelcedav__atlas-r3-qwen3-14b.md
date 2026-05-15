---
license: apache-2.0
language:
- es
- en
base_model: Rafaelcedav/atlas-r2-qwen3-14b
tags:
- finance
- legal
- regulatory
- compliance
- mexican-tax-law
- fincen
- audit
- qwen3
- amd-mi300x
- rocm
- fine-tuned
- atlas
pipeline_tag: text-generation
---

# ATLAS R3 — Qwen3-14B Forensic Regulatory Auditor

> **The most advanced open-source model for MX-USA financial regulatory compliance.**  
> Fine-tuned on AMD MI300X (205 GB VRAM) using ROCm 7.2 — built for the AMD Hackathon 2026.

---

## What is ATLAS?

**ATLAS** (Advanced Tax & Legal Auditing System) is a multi-agent AI system designed to detect financial anomalies, perform forensic regulatory audits, and simulate regulatory risk *before* operations are executed. It operates across Mexican and US financial law simultaneously.

This is **R3** — the third generation of the Qwen3-14B branch, trained via continued fine-tuning on top of [ATLAS R2](https://huggingface.co/Rafaelcedav/atlas-r2-qwen3-14b) with an expanded dataset of **13,588 curated regulatory examples**.

---

## Model Highlights

| Property | Value |
|---|---|
| **Architecture** | Qwen3ForCausalLM |
| **Parameters** | 14.77B |
| **Hidden size** | 5,120 |
| **Layers** | 40 |
| **Attention heads** | 40 |
| **Vocab size** | 151,936 |
| **Training precision** | bfloat16 |
| **Context length** | 2,048 tokens |
| **Training loss** | 0.1238 |
| **Eval loss** | 0.1016 |

---

## What ATLAS R3 Can Do

### 1. Forensic Regulatory Audit (`ATLAS Auditor`)
Full chain-of-thought legal analysis for detected anomalies:
- Applies **CFF, LISR, LIVA, RMF** (Mexico) and **FinCEN, CTA, IRC** (USA) simultaneously
- Returns explicit reasoning chain, legal citations, confidence score, and risk flags
- Handles cross-border operations, transfer pricing, and dual-reporting scenarios

### 2. Regulatory Sandbox (`ATLAS Sandbox`)
Predictive pre-facto simulation — analyze a proposed operation *before* executing it:
- Generates regulatory heat maps by jurisdiction
- Produces risk timelines (D+0 to D+365)
- Explores alternative scenarios (safer structuring options)
- Covers **OECD Pillar Two, NIF, US GAAP** for multinational operations

### 3. Red Team Mode (`ATLAS Red Team`)
Adversarial regulatory analysis:
- Identifies active violations and their downstream consequence chains
- Classifies severity and triggers automatic sanction risk assessment
- Detects SAT/FinCEN cross-audit patterns

### 4. Regulatory Chain-of-Thought
Step-by-step reasoning for complex multi-jurisdictional cases, with explicit normative citations at every inference step.

---

## Training Details

### Hardware
- **GPU**: AMD Instinct MI300X — 205 GB VRAM
- **Framework**: ROCm 7.2 | PyTorch 2.5.1+rocm6.2
- **Training time**: ~1h 55min (single GPU, full fine-tune)

### Configuration
```python
learning_rate        = 1e-5           # Reduced for continued training (anti-forgetting)
batch_size_per_gpu   = 2
gradient_accumulation = 8             # Effective batch = 16
num_epochs           = 2
warmup_steps         = 100
weight_decay         = 0.01
max_grad_norm        = 1.0
optimizer            = adamw_torch    # ROCm compatible
precision            = bfloat16
gradient_checkpointing = True
attn_implementation  = eager          # Avoids bf16 NaN bug on ROCm
```

### Dataset — `atlas_FINAL_v2.jsonl`
**13,588 total examples** across 4 regulatory domains:

| Source | Examples | Domain |
|---|---|---|
| ATLAS R1 Core | 3,502 | MX forensic audit (CFF/LISR/LIVA) |
| ATLAS R2 Expansion | 3,402 | US compliance (FinCEN/CTA/IRC) |
| ATLAS R2 Extended | 6,437 | Cross-border + transfer pricing |
| Batch 2 (Sandbox/Finance/Pillar2/CoT/RedTeam) | 150 | Advanced regulatory simulation |
| Batch 1 JSONL + Sandbox Rich Records | 97 | Structured audit + sandbox scenarios |

All examples follow the `{"messages": [...]}` chat format with specialized system prompts per mode.

### Training Strategy
R3 uses **continued fine-tuning** from R2 (not raw Qwen3-14B), with a lower learning rate (1e-5 vs 2e-5 used in R1/R2) to prevent catastrophic forgetting while absorbing new regulatory domains including OECD Pillar Two, NIF, and US GAAP multinational scenarios.

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Rafaelcedav/atlas-r3-qwen3-14b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# --- Forensic Audit Mode ---
messages = [
    {
        "role": "system",
        "content": (
            "Eres ATLAS, auditor forense especializado en regulaciones financieras MX-USA "
            "(CFF, LISR, LIVA, RMF, FinCEN, CTA, IRC). Analiza el caso con precisión legal, "
            "cadena de razonamiento explícita y recomendación accionable. Responde en español."
        )
    },
    {
        "role": "user",
        "content": "Empresa mexicana con ingresos de $50M USD transfiere $8M a subsidiaria en Delaware "
                   "sin documentación de precios de transferencia. Analiza el riesgo regulatorio."
    }
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Regulatory Sandbox Mode
```python
messages = [
    {
        "role": "system",
        "content": (
            "Eres ATLAS Regulatory Sandbox, motor de simulación regulatoria predictiva (pre-facto). "
            "Analiza la operación propuesta antes de ejecutarla. Genera: mapa de riesgo regulatorio, "
            "timeline, riesgos compuestos y escenarios alternativos. Basa tu análisis en normativa "
            "vigente 2026: CFF, LIVA, LISR, RMF, FinCEN, CTA, IRC, OECD Pillar Two, NIF, US GAAP."
        )
    },
    {
        "role": "user",
        "content": "SIMULAR OPERACIÓN: Fusión inversa de empresa MX con holding en Cayman Islands. "
                   "Valor estimado: $120M USD. Fecha propuesta: Q3 2026."
    }
]
```

---

## ATLAS System Architecture

ATLAS R3 is deployed as one of four specialized agents in the ATLAS pipeline:

```
Document/Transaction Input
        │
        ▼
  [Vision Agent]      ← OCR + visual extraction (Qwen2-VL / vLLM)
        │
        ▼
  [Reasoning Agent]   ← ATLAS R3 (this model) — forensic analysis
        │
        ▼
  [Validator Agent]   ← Confidence scoring + flag escalation
        │
        ▼
  [Explainer Agent]   ← Human-readable summary generation
        │
        ▼
  Audit Report + SSE stream to frontend
```

---

## Regulatory Coverage

**Mexico**: CFF (Código Fiscal de la Federación), LISR (Ley del ISR), LIVA, RMF (Resolución Miscelánea Fiscal), NIF

**United States**: FinCEN (Financial Crimes Enforcement Network), CTA (Corporate Transparency Act), IRC (Internal Revenue Code), US GAAP

**International**: OECD Pillar Two (Global Minimum Tax), Transfer Pricing (OECD Guidelines), FATF/GAFI recommendations

---

## Evaluation

| Metric | Value |
|---|---|
| Final training loss | 0.1238 |
| Best eval loss (step 1200) | **0.1016** |
| Eval samples | 680 |
| Eval speed | 16.05 samples/sec |

Loss dropped ~17% during R3 training (from 0.1219 at step 200 to 0.1016 at step 1200), indicating successful absorption of new regulatory domains without forgetting prior training.

---

## Model Family

| Version | Base | Dataset | Status |
|---|---|---|---|
| ATLAS R1 (Qwen3-14B) | Qwen/Qwen3-14B | 3,502 examples | Superseded |
| ATLAS R2 (Qwen3-14B) | R1 | +3,402 examples (9,304 total) | [Available](https://huggingface.co/Rafaelcedav/atlas-r2-qwen3-14b) |
| **ATLAS R3 (Qwen3-14B)** | **R2** | **+247 examples (13,588 total)** | **This model** |
| ATLAS DeepSeek-R1-8B | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 6,437 examples | [Available](https://huggingface.co/Rafaelcedav/atlas-finanzas-deepseek-r1-8b) |

---

## License

Apache 2.0 — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).

Base model: [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) (Apache 2.0)

---

## Citation

```bibtex
@misc{atlas-r3-2026,
  title        = {ATLAS R3: Qwen3-14B Fine-tuned for MX-USA Financial Regulatory Compliance},
  author       = {Rafael Cedillo},
  year         = {2026},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/Rafaelcedav/atlas-r3-qwen3-14b}},
  note         = {AMD Hackathon 2026 — trained on AMD Instinct MI300X}
}
```

---

*Built with AMD MI300X + ROCm for the AMD Hackathon 2026.*
