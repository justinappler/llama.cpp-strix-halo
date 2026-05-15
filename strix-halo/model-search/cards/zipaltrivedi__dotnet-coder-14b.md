---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
tags:
  - csharp
  - dotnet
  - code
  - fine-tuned
  - dpo
  - qlora
  - coding-assistant
  - aspnet
  - entity-framework
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# dotnet-coder-14b

A C#/.NET specialist fine-tuned from Qwen2.5-Coder-14B-Instruct. Achieves 97% compile rate on C# code generation benchmarks — higher than Qwen2.5-Coder-32B (80%) and Qwen2.5-72B (80%) on our evaluation suite.

Designed for coding agents and experienced .NET developers who need compilable, self-contained C# code.

## Quick Specs

| | |
|---|---|
| **Parameters** | 14.7B |
| **Base Model** | Qwen2.5-Coder-14B-Instruct |
| **Max Context** | 32,768 tokens (base model) |
| **Trained Sequence Length** | 2,048 tokens |
| **Training Method** | QLoRA SFT + Iterative DPO |
| **Training Data** | 107K C# records |
| **License** | Apache 2.0 |
| **VRAM (Q4_K_M)** | ~9GB |

## What Makes This Different: Compile-Verified DPO

Most code models are trained on code and hope it works. We verified it.

After SFT training, our model compiled at only 57%. We ran 3 rounds of **iterative DPO using `dotnet build` as the reward signal** — generating code, compiling it, and teaching the model to prefer code that actually compiles:

| Stage | Method | Compile Rate |
|---|---|---|
| Base (Qwen2.5-Coder-14B) | — | 83% |
| After SFT (107K records) | QLoRA | 57% |
| After DPO Round 1 | +126 pairs, beta=0.1 | 73% |
| After DPO Round 2 | +256 pairs, beta=0.2 | 87% |
| **After DPO Round 3** | **+468 pairs, beta=0.3** | **97%** |

The preference signal is binary and objective: the compiler says yes or no. No human labeling, no LLM-as-judge — just `dotnet build`.

## Benchmarks

Evaluated on **120+ unique prompts** across 4 independent test sets (original 30, holdout 30, and two validation sets of 40 each). No test prompts were used during training. All benchmarks are self-reported using our evaluation suite — results may vary with different prompts, inference settings, or SDK versions.

### Compile Rate (code compiles with `dotnet build`)

| Model | Parameters | Compile Rate | Holdout (unseen prompts) |
|---|---|---|---|
| **dotnet-coder-14b** | **14B** | **97%** | **97%** |
| StarCoder2-15B-Instruct | 15B | 83% | — |
| Phi-4-14B | 14B | 83% | — |
| Qwen2.5-Coder-14B-Instruct | 14B | 83% | 93% |
| Qwen2.5-72B-Instruct | 72B | 80% | 80% |
| Qwen2.5-Coder-32B-Instruct | 32B | 80% | 87% |
| Yi-Coder-9B-Chat | 9B | 70% | — |
| Qwen2.5-Coder-7B-Instruct | 7B | 57% | 67% |
| DeepSeek-R1-Distill-Qwen-14B | 14B | 10% | 13% |

All competitor models evaluated using the same 30-prompt test suite, same system prompt, same .NET 8 SDK with identical NuGet packages, and same code extraction pipeline. Competitor models loaded in 4-bit quantization on A100 80GB.

### Multi-SDK Compatibility

Tested the same generated code against three .NET SDK versions:

| .NET Version | Compile Rate |
|---|---|
| .NET 6.0 | 29/30 (97%) |
| .NET 8.0 | 29/30 (97%) |
| .NET 9.0 | 29/30 (97%) |

The single failure across all SDKs is the same prompt — a `??` operator applied to an incompatible type in a value object base class. The model generates SDK-agnostic code that works across .NET 6, 8, and 9.

### Temperature Robustness

Compile rate at different sampling temperatures (30 prompts each):

| Temperature | Compile Rate |
|---|---|
| 0.2 | 29/30 (97%) |
| 0.5 | 29/30 (97%) |
| 0.8 | 29/30 (97%) |
| 1.0 | 26/30 (87%) |

The model maintains 97% compile rate through temperatures 0.2–0.8. Even at maximum randomness (temp=1.0), it still achieves 87%.

### Expert .NET Knowledge

| Model | Parameters | Expert Gotchas (/10) | C# 13/14 Features (/10) |
|---|---|---|---|
| **dotnet-coder-14b** | **14B** | **10** | **10** |
| Qwen2.5-72B-Instruct | 72B | 9 | 10 |
| Qwen2.5-Coder-14B-Instruct | 14B | 8 | 10 |
| Qwen2.5-Coder-32B-Instruct | 32B | 6 | 10 |
| DeepSeek-R1-Distill-Qwen-14B | 14B | 3 | 5 |

Expert gotcha questions test: ConcurrentDictionary.GetOrAdd atomicity, async void crash scenarios, ConfigureAwait deadlocks, HttpClient socket exhaustion, IEnumerable vs IQueryable in EF Core, DI captive dependency, ValueTask consumption rules, N+1 queries, GC generation costs, and middleware pipeline internals.

### Complex Multi-Class Tasks

10 tasks requiring 3-6 interacting classes (CQRS with dispatcher, event sourcing, fluent validation framework, resilience pipeline with circuit breaker, state machines, plugin systems):

- **First try: 70-90%**
- **With retry: 90%+**

## Example Output

**Prompt:** "Write a retry policy with exponential backoff for HttpClient in C#"

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

public class RetryPolicyHttpClient
{
    private readonly HttpClient _client = new();
    private static readonly Random Random = new();

    public async Task<string> GetAsync(string url)
    {
        int maxRetries = 3;
        for (int attempt = 0; attempt < maxRetries; attempt++)
        {
            try
            {
                return await _client.GetStringAsync(url);
            }
            catch (Exception ex) when (attempt < maxRetries - 1 && IsTransient(ex))
            {
                double delayInSeconds = Math.Pow(2, attempt) + Random.NextDouble();
                await Task.Delay(TimeSpan.FromSeconds(delayInSeconds));
            }
        }
        return await _client.GetStringAsync(url);
    }

    private static bool IsTransient(Exception exception)
    {
        return exception is HttpRequestException;
    }
}
```

No preamble, no "Certainly!", no package install instructions — just correct, compilable code with jitter-based backoff.

### Error Diagnosis

**Prompt:** "I'm getting `error CS0246: The type or namespace name 'ILogger<>' could not be found`. What's wrong?"

> The error message indicates that the compiler cannot find the `ILogger<>` type. This is because the `ILogger<>` interface is part of the Microsoft.Extensions.Logging namespace, which is not included in your code.
>
> Add the using directive: `using Microsoft.Extensions.Logging;`
>
> Make sure you also have the NuGet package installed:
> ```
> Install-Package Microsoft.Extensions.Logging
> ```

### Performance Issue Detection

**Prompt:** "My API endpoint is slow" + code with N+1 query

> The problem is that you're loading all Orders into memory, then for each one making a separate database call to load the related Customer. This is N+1 queries. The solution is to use Include() to eagerly load the related data in a single query:

```csharp
public async Task<List<OrderDto>> GetOrders()
{
    return await _context.Orders
        .Include(o => o.Customer)
        .Select(o => new OrderDto(o.Id, o.Customer.Name, o.Total))
        .ToListAsync();
}
```

The model handles both code generation and interactive debugging — it can diagnose errors, suggest fixes, recommend package installations, and identify performance issues like N+1 queries.

## What This Model Is Good At

- Writing complete, compilable C# classes and files
- **Diagnosing build errors** and suggesting fixes with correct NuGet packages
- **Identifying performance issues** like N+1 queries, socket exhaustion, GC pressure
- **Modifying existing code** on request (adding methods, refactoring)
- ASP.NET Core middleware, controllers, and minimal APIs
- Entity Framework Core queries, configurations, and patterns
- Design patterns implemented in idiomatic C#
- Modern C# features (records, primary constructors, collection expressions, pattern matching)
- Explaining C# gotchas and .NET internals
- Self-contained code — defines all types it references

## Recommended Settings

**System prompt** (used during training — best results with this or similar):

> You are an expert C# and .NET developer. Write complete, compilable C# code. Include all necessary using statements and namespace declarations.

**Inference parameters**: temperature=0.2, top_p=0.9, max_new_tokens=2048

The base model supports up to 32,768 tokens, so you can use the full 32K context window. The fine-tuning was done on sequences up to 2,048 tokens — the model performs best within this range but still works beyond it thanks to the base model's capabilities. The model will stop generating when done (EOS token), so setting a higher limit won't cause unnecessary output.

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "zipaltrivedi/dotnet-coder-14b",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("zipaltrivedi/dotnet-coder-14b")

messages = [
    {"role": "system", "content": "You are an expert C# and .NET developer. Write complete, compilable C# code."},
    {"role": "user", "content": "Write a thread-safe LRU cache with generic key and value types in C#."},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=2048, temperature=0.2, top_p=0.9)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### With Ollama

Download a GGUF file from the Files tab, then create a `Modelfile`:

```
FROM ./dotnet-coder-14b-Q4_K_M.gguf

SYSTEM "You are an expert C# and .NET developer. Write complete, compilable C# code. Include all necessary using statements and namespace declarations."

PARAMETER temperature 0.2
PARAMETER top_p 0.9
```

Then run:

```bash
ollama create dotnet-coder -f Modelfile
ollama run dotnet-coder
```

### With llama.cpp

```bash
./llama-cli -m dotnet-coder-14b-Q4_K_M.gguf -p "Write a C# class for..." --temp 0.2
```

### With LM Studio / Jan / GPT4All

Download the GGUF file matching your hardware from the Files tab and load it in your preferred UI.

## GGUF Quantizations

| Quantization | Size | Min RAM | Recommended For |
|---|---|---|---|
| Q8_0 | 14.6 GB | 16GB+ | Best quality — RTX 4090, A100, M3 Max 36GB |
| Q6_K | 11.3 GB | 14GB+ | High quality — RTX 4080, M2 Max 32GB |
| **Q4_K_M** | **8.4 GB** | **10GB+** | **Recommended — RTX 3080/4070, M2 Pro 16GB** |
| Q4_K_S | 8.0 GB | 10GB+ | Slightly smaller — M1 Pro 16GB |
| Q3_K_M | 6.8 GB | 8GB+ | Budget GPU, Apple M1/M2 8GB |
| Q2_K | 5.4 GB | 6GB+ | CPU-only inference, minimum viable |

## Training Details

### Dataset (107K records)

| Source | Records | Description |
|---|---|---|
| Expert C# knowledge | 54,443 | Curated Q&A covering gotchas, patterns, best practices, version-specific features |
| Compile-verified repos | 35,736 | Self-contained C# files from 140 GitHub repos, filtered and verified |
| .NET runtime source | 12,352 | Code from dotnet/runtime, aspnetcore, and other core .NET repos |
| Synthetic examples | 4,906 | C# 13/14 features, debugging pairs, code review examples |

### SFT Hyperparameters

- **Method**: QLoRA 4-bit, LoRA rank 64, alpha 128
- **Training**: 2 epochs, lr=2e-4, cosine schedule, 3% warmup, packing enabled
- **Batch**: effective batch size 16 (2 per device x 8 gradient accumulation)
- **Hardware**: RunPod A100 80GB SXM, ~13 hours
- **Cost**: ~$20

### DPO Hyperparameters

- **Rounds**: 3 iterative rounds
- **Pair generation**: Model generates 3-5 responses per prompt at different temperatures, compiled with `dotnet build`, pass=chosen / fail=rejected
- **Training**: beta=0.1→0.2→0.3 (increasing preference strength), lr=5e-5, 1-3 epochs per round
- **Total pairs**: 850 across all rounds
- **Hardware**: Same A100, ~2 hours total
- **Cost**: ~$5

### Evaluation Methodology

All compile tests use actual `dotnet build` with .NET 8 SDK against a project with common NuGet packages (EF Core, ASP.NET Core, Microsoft.Extensions). Pass/fail is binary based on compiler exit code — no manual evaluation or LLM-as-judge.

Tests are run across 4 independent prompt sets totaling 120+ unique prompts. Holdout and validation prompts were never used during any stage of training or DPO pair generation.

## Limitations

- **Optimized for single-file generation** — for multi-project solutions, use as a component alongside an agent framework
- **Best for experienced developers** — gives direct code answers, not step-by-step tutorials
- **English only** — trained on English C# content
- **14B parameter model** — for extremely complex architectural decisions, larger models may provide more nuanced analysis
- **Compile rate is not 100%** — the remaining ~3% failures are typically complex generic dispatch patterns (e.g., CQRS mediator with runtime handler resolution) that produce type constraint errors
- **Compile ≠ correct** — code that compiles is not guaranteed to be logically correct or free of runtime errors. Compilation is a necessary but not sufficient measure of code quality. Always review generated code before production use

## Benchmark Limitations

- All benchmarks are **self-reported** using our custom evaluation suite — not a standardized benchmark like HumanEval or MBPP
- Compile rate is primarily tested against **.NET 8 SDK** with a specific set of NuGet packages. Cross-validation against .NET 6, 8, and 9 shows identical results (97%), but results may differ with other package configurations
- Expert knowledge evaluation involved checking whether responses address the core question with code examples — this has a subjective component
- Sample sizes (30 prompts per test set) are small; results have inherent variance
- No formal analysis of training/test data overlap with the Qwen2.5-Coder base model's pre-training data
- **Metric circularity**: DPO training uses `dotnet build` as the reward signal, and compile rate is measured using the same tool. While the evaluation prompts are completely separate from DPO training prompts, the model is optimized for the same metric it's evaluated on

## Ethical Considerations

- **No safety alignment**: This model has no specific safety training beyond what exists in the Qwen2.5-Coder base model. It may generate code with security vulnerabilities if prompted
- **Bias**: Training data is sourced from public repositories and Q&A sites, which may reflect coding conventions and patterns from specific communities
- **Not a substitute for code review**: Generated code should be reviewed by a developer before use in production
- **Training data provenance**: Training data includes content from StackOverflow (CC-BY-SA 4.0), Microsoft Learn (CC-BY-4.0), The Stack (permissive licenses), and GitHub repos (Apache/MIT). The relationship between CC-BY-SA training data and model output licensing is an open legal question across the LLM industry. Users should be aware of this when using generated code in commercial settings

## Use Cases

- **Coding agent backend** — serve via OpenAI-compatible API for use with OpenCode, Continue, Cursor, Claude Code
- **Local code assistant** — run with Ollama or LM Studio for offline C# development
- **CI/CD code generation** — generate boilerplate, tests, and implementations in automated pipelines
- **Code review** — get expert-level feedback on C# patterns and .NET best practices

## Reproducibility

- **Base model**: `Qwen/Qwen2.5-Coder-14B-Instruct` from HuggingFace
- **Training framework**: Unsloth 2026.4.5 + PEFT + TRL on RunPod A100 80GB
- **Random seed**: 42 (dataset shuffle)
- Training scripts, evaluation code, and LoRA adapter weights available upon request

## License

Apache 2.0 (same as base model Qwen2.5-Coder-14B-Instruct)

## Citation

```bibtex
@misc{dotnet-coder-14b,
  author = {Zipal Trivedi},
  title = {dotnet-coder-14b: A C#/.NET Specialist Language Model},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/zipaltrivedi/dotnet-coder-14b}
}
```

## Acknowledgments

- Base model: [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) by Alibaba
- Training framework: [Unsloth](https://github.com/unslothai/unsloth)
- Training data sources: The Stack (permissive licenses), StackOverflow (CC-BY-SA 4.0), Microsoft Learn (CC-BY-4.0), GitHub repos (Apache/MIT licensed)

## Contact

For issues, questions, or feedback: [HuggingFace Discussions](https://huggingface.co/zipaltrivedi/dotnet-coder-14b/discussions)
