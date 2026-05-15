---
language:
- es
- en
license: apache-2.0
base_model: Qwen/Qwen3-14B
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
- qwen3
- finance
- thinking
datasets:
- custom
pipeline_tag: text-generation
---

# ATLAS Qwen3-14B Legal — Motor de Razonamiento Forense

Segunda ronda de fine-tuning de **Qwen3-14B** especializada en derecho fiscal forense MX/USA, entrenada sobre **AMD Instinct MI300X** en el marco del AMD Hackathon 2025.

> El modelo más capaz del ecosistema ATLAS. 14 mil millones de parámetros alineados para pensar como un auditor forense senior.

---

## ¿Qué hace diferente a Qwen3-14B?

Qwen3-14B introduce una distinción arquitectónica que lo separa de la mayoría de los LLMs de su categoría: **modo de pensamiento explícito (`thinking mode`)**.

Cuando el modelo necesita resolver un problema complejo — como determinar si una empresa con incoherencias materiales califica para presunción bajo Art. 69-B CFF — puede generar un bloque de razonamiento interno antes de responder:

```
<think>
La empresa reporta ingresos por 50MDP en servicios de construcción especializada.
Verifico capacidad operativa: 1 empleado registrado en IMSS.
Razón ingresos/empleado: 50,000,000 MXN por trabajador → inviable operativamente.
Domicilio fiscal: zona residencial → sin infraestructura industrial visible.
Patrón: coincide con perfil EFOS (Art. 69-B, primer párrafo CFF).
Procedimiento aplicable: facultades de verificación Art. 42 Fr. IX.
</think>

ANÁLISIS FORENSE — Art. 69-B CFF ...
```

Este fine-tune entrena explícitamente ese patrón de razonamiento sobre **3,502 casos reales de auditoría fiscal** en derecho mexicano y americano.

---

## Métricas de Entrenamiento

| Métrica | Valor |
|---------|-------|
| Base Model | Qwen/Qwen3-14B |
| Parámetros | **14,768,307,200** |
| Train Loss | *en progreso* |
| Eval Loss | *en progreso* |
| Epochs | 3 |
| Dataset | 3,502 ejemplos legales MX/USA |
| Hardware | AMD Instinct MI300X (205.8 GB VRAM) |
| Optimizer | adamw_torch (ROCm-native) |
| Precisión | bfloat16 |

---

## Arquitectura de Razonamiento en Auditoría

El modelo fue entrenado en una taxonomía de razonamiento forense de 3 niveles:

### Nivel 1 — Detección de Patrones
Reconocimiento de red flags documentadas en normativa:
- **Incoherencia material** — capacidad operativa vs. ingresos declarados
- **Incoherencia geográfica** — domicilio fiscal vs. naturaleza del servicio
- **Patrones de carrusel** — redes de facturación circular A→B→C→A
- **Structuring (smurfing)** — depósitos sistemáticos sub-$10K en múltiples cuentas

### Nivel 2 — Cruce Normativo
Mapeo automático de hallazgos a artículos específicos:
- CFF: Arts. 42, 69-B, 76, 81, 82, 108
- LISR: Arts. 27, 59-G, 140 (RESICO)
- IRC: Secciones 61, 162, 482, 6038
- IRM: Parts 4.10, 4.23, 4.31

### Nivel 3 — Construcción de Caso
Síntesis de evidencia en estructura argumentativa auditable:
1. Hallazgos primarios con fuente
2. Artículos aplicables con justificación
3. Procedimiento de verificación recomendado
4. Riesgo estimado y acción correctiva

---

## Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Rafaelcedav/atlas-qwen3-14b-legal"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Requerido en ROCm
    device_map="auto"
)

# Habilitar thinking mode para casos complejos
messages = [
    {
        "role": "system",
        "content": "Eres ATLAS, auditor forense senior especializado en derecho fiscal MX/USA. Razona exhaustivamente antes de emitir conclusiones. Cita artículos específicos."
    },
    {
        "role": "user",
        "content": "/think\n\nEmpresa distribuidora vende a filial extranjera a precio 40% por debajo del mercado. No existe documentación de benchmarking. Ingresos declarados inconsistentes con patrones del sector. ¿Análisis completo?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=2048,
        temperature=0.6,    # Qwen3 recomienda 0.6 para thinking mode
        top_p=0.95,
        do_sample=True
    )

response = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)
```

---

## Rol en Pipeline ATLAS

```
                    ┌─────────────────────────────────┐
PDF ──► Vision ──►  │  Qwen3-14B Legal (Motor 8000)   │ ──► Validator
        InternVL2   │                                 │
                    │  • Recibe campos extraídos       │
                    │  • Cruza con compliance router   │
                    │  • Genera hipótesis de riesgo    │
                    │  • Construye argumento legal     │
                    │  • Emite veredicto normalizado   │
                    └─────────────────────────────────┘
```

Opera sobre el **Motor 8000** del stack ATLAS, sirviendo via vLLM con interfaz OpenAI-compatible para integración con el pipeline de agentes.

---

## Ecosistema Completo ATLAS

| Modelo | Rol | Params | Loss |
|--------|-----|--------|------|
| [atlas-qwen3-14b-legal](https://huggingface.co/Rafaelcedav/atlas-qwen3-14b-legal) | **Razonamiento principal** | 14.8B | — |
| [atlas-r2-qwen3-14b](https://huggingface.co/Rafaelcedav/atlas-r2-qwen3-14b) | Razonamiento R1 | 14.8B | 0.2697 |
| [atlas-finanzas-deepseek-r1-8b](https://huggingface.co/Rafaelcedav/atlas-finanzas-deepseek-r1-8b) | Análisis financiero | 8.3B | 0.4829 |
| [atlas-mistral-7b-legal](https://huggingface.co/Rafaelcedav/atlas-mistral-7b-legal) | Agente legal MX/USA | 7.2B | **0.0184** |

---

## Hardware & Stack Técnico

```
GPU:        AMD Instinct MI300X VF
VRAM:       205.8 GB (bfloat16 nativo)
ROCm:       7.2 / PyTorch 2.5.1+rocm6.2
Optimizer:  adamw_torch (ROCm-compatible)
Attention:  eager (SDPA → NaN en ROCm + bf16)
Serving:    vLLM + OpenAI-compatible API
```

---

*Trained with ❤️ on AMD silicon. Part of the ATLAS AMD Hackathon 2025 submission.*
