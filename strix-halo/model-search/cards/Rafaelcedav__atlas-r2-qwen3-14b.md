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
- multi-round
datasets:
- custom
pipeline_tag: text-generation
---

# ATLAS Qwen3-14B — Motor de Razonamiento Forense

> **14.7 mil millones de parámetros. Dos rondas de entrenamiento especializado. Un objetivo: pensar como el mejor auditor forense del mundo.**

Fine-tune multi-ronda de **Qwen3-14B** para detección de anomalías financieras y auditoría fiscal forense en México y USA. Entrenado íntegramente sobre **AMD Instinct MI300X** (205GB VRAM) como parte del sistema **ATLAS** — AMD Hackathon 2025.

---

## Historial de Entrenamiento

Este modelo no nació especializado. Fue construido en dos rondas de entrenamiento deliberadamente secuenciadas:

### Ronda 1 — Fundamentos (Rama: `main`)
```
Dataset:   atlas_training_dataset_final.jsonl
Registros: 6,437 ejemplos financiero-legales MX/USA
Epochs:    3
Loss:      0.2697
Tiempo:    71 minutos
Hardware:  AMD MI300X (205GB VRAM)
```
Primera exposición al dominio. El modelo aprende el vocabulario fiscal, los patrones de riesgo y la estructura argumentativa de un auditor. Establece la base de conocimiento.

### Ronda 2 — Especialización Legal (Rama: `legal-v2`)
```
Dataset:   atlas_audit_master_unified.jsonl
Registros: 3,502 casos legales de alta complejidad
Epochs:    3
Loss:      ~0.018 (train) | ~0.019 (eval)
Tiempo:    ~47 minutos
Hardware:  AMD MI300X (205GB VRAM)
```
Refinamiento sobre casos de mayor dificultad y especificidad normativa. El modelo profundiza en artículos específicos, cruces normativos MX/USA y construcción de argumentos forenses auditables. **Loss 15x mejor que Ronda 1.**

---

## ¿Por qué Qwen3-14B?

Qwen3-14B introduce **thinking mode** — la capacidad de razonar explícitamente antes de responder. En auditoría forense esto no es un lujo, es una necesidad:

```
<think>
Empresa reporta 50MDP en servicios de construcción con 1 empleado.
Ratio ingresos/empleado: inviable operativamente.
Domicilio: zona residencial → sin infraestructura industrial.
Patrón: EFOS clásico bajo Art. 69-B CFF, primer párrafo.
Procedimiento: verificación Art. 42 Fr. IX + solicitud documentación.
Riesgo estimado: ALTO. Requiere actuación inmediata.
</think>

RED FLAG CONFIRMADA — Art. 69-B CFF...
```

Un modelo que muestra su razonamiento es un modelo cuyos errores se pueden corregir. En contextos legales, eso es crítico.

---

## Dominio de Conocimiento

### México — Marco Normativo
| Área | Artículos |
|------|-----------|
| Operaciones Inexistentes (EFOS/EDOS) | Art. 69-B CFF |
| Facultades de Comprobación SAT | Art. 42 CFF |
| Infracciones y Sanciones | Arts. 76, 81, 82 CFF |
| Deducibilidad de Gastos | Art. 27 LISR |
| Precios de Transferencia | Art. 59-G LISR |
| RESICO Personas Físicas | Art. 140 LISR |
| Delitos Fiscales | Art. 108 CFF |

### USA — Internal Revenue Manual
| Área | Referencia |
|------|-----------|
| Examination of Returns | IRM 4.10 |
| Employment Tax / Worker Classification | IRM 4.23 |
| Anti-Money Laundering / FBAR | BSA, FinCEN 114 |
| International Examinations | IRM 4.61 |
| Bank Deposits Method | IRM 4.10.3 |

---

## Taxonomía de Razonamiento Forense

El modelo opera en 3 niveles de abstracción:

```
┌─────────────────────────────────────────────────────────┐
│  NIVEL 1: DETECCIÓN DE PATRONES                         │
│  • Incoherencia material (capacidad vs. ingresos)       │
│  • Incoherencia geográfica (domicilio vs. actividad)    │
│  • Facturación circular (A→B→C→A sin flujo real)        │
│  • Structuring / Smurfing (depósitos sub-$10K)          │
│  • Márgenes anómalos (precios de transferencia)         │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  NIVEL 2: CRUCE NORMATIVO                               │
│  • Mapeo hallazgo → artículo específico                 │
│  • Jurisdicción aplicable (MX / USA / ambas)            │
│  • Procedimiento de verificación recomendado            │
│  • Carga de la prueba y estándares de evidencia         │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  NIVEL 3: CONSTRUCCIÓN DE CASO                          │
│  • Síntesis de evidencia con trazabilidad               │
│  • Cuantificación de riesgo fiscal estimado             │
│  • Acciones correctivas priorizadas                     │
│  • Reporte ejecutivo auditable                          │
└─────────────────────────────────────────────────────────┘
```

---

## Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar Ronda 2 (especialización legal)
model_id = "Rafaelcedav/atlas-r2-qwen3-14b"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision="legal-v2")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision="legal-v2",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",   # Requerido en ROCm / AMD
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": "Eres ATLAS, auditor forense senior especializado en derecho fiscal MX/USA. Activa thinking mode para casos complejos. Cita artículos específicos y construye argumentos auditables."
    },
    {
        "role": "user",
        "content": "/think\n\nDistribuidora MX vende a filial en paraíso fiscal a precio 40% menor que mercado. Sin benchmarking documentado. Ingresos declarados inconsistentes con flujos bancarios. ¿Análisis completo?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)

output = model.generate(
    inputs,
    max_new_tokens=2048,
    temperature=0.6,    # Recomendado por Qwen3 para thinking mode
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## Rol en Pipeline ATLAS

```
PDF ──► [Vision · InternVL2-40B]
              │
              ▼
        [Compliance Router · Motor 11434]
              │
              ▼
┌─────────────────────────────────┐
│   Qwen3-14B  ·  Motor 8000     │  ◄── Este modelo
│                                 │
│   Recibe: campos extraídos      │
│   Cruza:  normativa aplicable   │
│   Genera: hipótesis de riesgo   │
│   Emite:  veredicto trazable    │
└────────────────┬────────────────┘
                 │
                 ▼
        [Validator · Integridad]
                 │
                 ▼
        [Explainer · Reporte PDF]
```

Sirve via **vLLM** con interfaz OpenAI-compatible en puerto 8000.

---

## Ecosistema ATLAS

| Modelo | Rol en Pipeline | Params | Eval Loss |
|--------|----------------|--------|-----------|
| **[atlas-r2-qwen3-14b](https://huggingface.co/Rafaelcedav/atlas-r2-qwen3-14b)** | **Razonamiento principal** | **14.7B** | **~0.019** |
| [atlas-finanzas-deepseek-r1-8b](https://huggingface.co/Rafaelcedav/atlas-finanzas-deepseek-r1-8b) | Análisis financiero profundo | 8.3B | 0.4829 |
| [atlas-mistral-7b-legal](https://huggingface.co/Rafaelcedav/atlas-mistral-7b-legal) | Agente legal MX/USA | 7.2B | **0.0184** |

---

## Stack Técnico

```yaml
GPU:          AMD Instinct MI300X VF
VRAM:         205.8 GB
OS:           Ubuntu 24.04 LTS
ROCm:         7.2
PyTorch:      2.5.1+rocm6.2
Transformers: 5.x
Optimizer:    adamw_torch          # Único estable en ROCm
Attention:    eager                # SDPA produce NaN en ROCm + bf16
Precision:    bfloat16             # Nativo en MI300X
Serving:      vLLM (OpenAI-compat)
```

---

*Two rounds. One mission. Built on AMD.*
