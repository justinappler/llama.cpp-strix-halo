---
language:
- es
- en
license: mit
base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
tags:
- fine-tuned
- legal
- audit
- forensic
- tax
- reasoning
- atlas
- amd
- rocm
- deepseek
- finance
- distilled
- multi-round
datasets:
- custom
pipeline_tag: text-generation
---

# ATLAS DeepSeek-R1 Finanzas — Razonamiento Forense Fiscal

> **Chain-of-thought destilado de un modelo de 671B parámetros. Afinado para pensar como un investigador fiscal.**

Fine-tune multi-ronda de **DeepSeek-R1-Distill-Llama-8B** para detección de anomalías financieras y auditoría fiscal forense en México y USA. Entrenado sobre **AMD Instinct MI300X** como parte del sistema **ATLAS** — AMD Hackathon 2025.

---

## Por qué DeepSeek-R1 es diferente

La mayoría de los LLMs aprenden a *responder*. DeepSeek-R1 aprendió a *razonar*.

El modelo base fue entrenado por DeepSeek mediante **reinforcement learning puro** — sin supervisión humana en el proceso de razonamiento — lo que produjo un modelo que naturalmente genera cadenas de pensamiento estructuradas antes de responder. Luego fue **destilado desde DeepSeek-R1 (671B parámetros)** hacia arquitectura Llama-8B, preservando esa capacidad de razonamiento en un modelo 83x más pequeño y eficiente.

En el contexto de auditoría forense, esto se traduce en:
- Hipótesis alternativas evaluadas explícitamente
- Evidencia ponderada antes de emitir veredicto
- Razonamiento auditable y reversible
- Conclusiones con trazabilidad argumental

---

## Historial de Entrenamiento

### Ronda 1 — Dominio Financiero (Rama: `main`)
```
Dataset:   atlas_training_dataset_final.jsonl
Registros: 6,437 ejemplos financiero-legales MX/USA
Epochs:    3
Loss:      0.4829
Tiempo:    ~36 minutos
Hardware:  AMD MI300X (205GB VRAM)
```
Exposición amplia al dominio. Vocabulario fiscal, estructuras de casos, patrones de riesgo y normativa base MX/USA.

### Ronda 2 — Especialización Legal (Rama: `legal-v2`)
```
Dataset:   atlas_audit_master_unified.jsonl
Registros: 3,502 casos legales de alta complejidad
Epochs:    3
Loss:      ~0.020 (train) | ~0.025 (eval)
Tiempo:    ~25 minutos
Hardware:  AMD MI300X (205GB VRAM)
```
Refinamiento sobre casos de mayor especificidad normativa. Artículos concretos, procedimientos forenses, cruces MX/USA. **Convergencia en 25 minutos** gracias a la eficiencia de la arquitectura Llama destilada corriendo en MI300X bfloat16 nativo.

---

## Dominio de Conocimiento

**México — CFF / LISR:**
- Art. 69-B CFF — EFOS/EDOS: presunción, desvirtúo, efectos fiscales
- Art. 42 CFF — Facultades de comprobación SAT (Fr. I-X)
- Arts. 76, 81, 82 CFF — Infracciones graves y sanciones
- Art. 27 LISR — Requisitos de deducibilidad
- Art. 59-G LISR — Precios de transferencia y arm's length
- RMF RESICO 2025/2026 — Régimen de Confianza

**USA — IRS / IRM:**
- IRM 4.10 — Examination of Returns (Bank Deposits, Net Worth Methods)
- IRM 4.23 — Employment Tax / Worker Misclassification
- BSA/FBAR — FinCEN 114, structuring, smurfing patterns
- Common Law Test — Control de métodos y resultados
- Economic Reality Test — Integración económica

---

## Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Rafaelcedav/atlas-finanzas-deepseek-r1-8b"

# Ronda 2 (recomendado — especialización legal)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="legal-v2")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision="legal-v2",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Requerido en ROCm
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": "Eres ATLAS, auditor forense especializado en derecho fiscal MX/USA. Razona paso a paso. Identifica artículos específicos y construye argumentos auditables."
    },
    {
        "role": "user",
        "content": "Empleado con horario fijo, oficina asignada y herramientas del empleador, pero clasificado como contratista independiente. Sin retenciones de nómina. ¿Análisis completo?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
output = model.generate(
    inputs, max_new_tokens=1024, temperature=0.1, do_sample=True
)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## Rol en Pipeline ATLAS

```
[Vision · InternVL2-40B] ──► [Compliance · Motor 11434]
                                          │
                                          ▼
                          ┌───────────────────────────┐
                          │  DeepSeek-R1 · Motor 8000 │ ◄── Este modelo
                          │                           │
                          │  Recibe hallazgos visión  │
                          │  Razona sobre evidencia   │
                          │  Descarta hipótesis        │
                          │  Emite veredicto trazable │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                              [Validator] ──► [Explainer]
```

---

## Ecosistema ATLAS

| Modelo | Rol | Params | Rondas | Eval Loss |
|--------|-----|--------|--------|-----------|
| [atlas-r2-qwen3-14b](https://huggingface.co/Rafaelcedav/atlas-r2-qwen3-14b) | Razonamiento principal | 14.7B | 2 | ~0.019 |
| **[atlas-finanzas-deepseek-r1-8b](https://huggingface.co/Rafaelcedav/atlas-finanzas-deepseek-r1-8b)** | **Análisis financiero forense** | **8.3B** | **2** | **~0.025** |
| [atlas-mistral-7b-legal](https://huggingface.co/Rafaelcedav/atlas-mistral-7b-legal) | Agente legal MX/USA | 7.2B | 1 | 0.0184 |

---

## Stack Técnico

```yaml
GPU:          AMD Instinct MI300X VF
VRAM:         205.8 GB
PyTorch:      2.5.1+rocm6.2
Optimizer:    adamw_torch
Attention:    eager                 # SDPA → NaN en ROCm + bf16
Precision:    bfloat16
Serving:      vLLM (OpenAI-compat · Puerto 8000)
```

---

*Distilled from 671B. Trained on AMD. Built for forensic truth.*
