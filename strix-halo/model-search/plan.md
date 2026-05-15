# Model search rewrite — plan

Hypothesis doc anchoring the rewrite of [`README.md`](README.md), [`rank.sh`](rank.sh), and the bench-data path. Companion to [coder-next-baseline.md](../coder-next-baseline.md) and [qwen3.6-baseline.md](../qwen3.6-baseline.md).

## Aim

Find the **5–8 best coding-model replacements for Qwen 3.6 35B-A3B** on this gfx1151 box, with three tests in order:

1. **Predicted runtime is in the vicinity of, or better than, Qwen 3.6's.** Not "must beat" — "not dramatically worse." Compared on tg ceiling (`BW / (active × bytes_per_param)`) and pp@depth (D≤128 unlocks rocWMMA-tuned, +40-65% pp@depth per [README.md Finding #6](../README.md)).
2. **Published evidence it codes ≥ Qwen 3.6.** SWE-bench Verified is the load-bearing benchmark for agentic coding. Threshold: ≥ 30% resolved (Qwen 3.6 itself doesn't post one — that floor is "competent agentic coder").
3. **Deploys here today.** Unsloth Dynamic GGUF or `convert_hf_to_gguf.py`-supported arch, working Jinja chat template, permissive license.

## What the current pipeline gets wrong

Concrete failures in the old [shortlist.json](shortlist.json):

- **Excludes the obvious target.** Qwen3-Coder-30B-A3B-Instruct (active ≈ 1.9 B at 8/128 routing, D=128, ~17 GB Q4) is dropped by the `README_ACTIVE_MIN=2.5e9` floor in [rank.sh:46](rank.sh#L46). On a bandwidth-bound box, *lower active is better* up to the quality cliff. The floor is wrong-signed. DeepSeek-Coder-V2-Lite-Instruct gets dropped the same way.
- **Variant explosion.** 8 of the top 20 are IBM Granite 8B variants — including `granite-3.3-8b-math-prm-v2` (a reward model), `granite-3.1-8b-lora-intrinsics-v0.1`, `granite-3.2-8b-instruct-preview`. No dedup.
- **"Coding signal" is org substring matching.** Any repo from `nvidia` / `deepseek` / `ibm-granite` / `granite` is auto-tagged coding-domain. That's how a Japanese-language Nemotron-Nano-9B and a math reward model landed on a coding shortlist.
- **Bench evidence isn't wired in.** [`benchmarks_snapshot.json`](benchmarks_snapshot.json) is empty; the 294 cached cards in [`cards/`](cards/) are only grep'd for "any HumanEval mention" (not scores), and even that gate is off by default.
- **18 `README_*` env knobs to compensate.** [rank.sh](rank.sh) is 419 LoC, with `liberal`/`strict` deploy modes, two shortlist gating modes, MoE/dense weight bands, dense-shape-strict, chat-template gate. The README spends 200 lines explaining how to combine the knobs to get a usable result. The defaults produce garbage.

## What changes

### Cuts

- `liberal`/`strict` deploy mode toggle → one mode: Unsloth Dynamic XL exists OR `convert_hf_to_gguf` supports the arch.
- The org-substring `coding_signal` heuristic.
- `README_MOE_GB_MIN/MAX`, `README_DENSE_GB_MIN/MAX`, `README_DENSE_GB_IDEAL_MIN`, `README_SHAPE_STRICT` → one `weight_gb_q4_est ≤ 80` cap.
- `README_ACTIVE_MIN`. There is no minimum.
- `card_bench_scan.sh`, `import_benchmarks.sh`, `bench_join.sh`, `benchmarks_snapshot.json`, `benchmarks_patch.example.json` — replaced by one bench scraper.
- Most of [rank.sh](rank.sh): from 419 → ~150 LoC, single mode.

### Fixes

- **No active-param floor.** Score by predicted tg ceiling, with low active rewarded.
- **Variant collapse.** Group by `(model_type, total_params, num_hidden_layers, num_experts)` and `derivative_of` chains. Keep one representative per family. Preference: `instruct` > `chat` > `base`, latest version; drop reward-model / lora-intrinsics / preview / math-prm via tag and id-suffix exclusion. Alternates carried as `siblings`, not separate rows.
- **Coding evidence from a real source.** SWE-bench Verified score from [github.com/SWE-bench/experiments](https://github.com/SWE-bench/experiments) — every submission has `metadata.yml` (with HF model URL) and `results/results.json` (with `resolved` array; SWE-bench Verified = 500 instances). Take max score per HF model id. ~134 submissions today, covers every modern coding-relevant model.
- **Single ranking score.** Hard gates first; survivors scored by `swebench_resolved_pct + tg_ceiling_factor + pp_at_depth_factor + recency`. Tg ceiling factor rewards low-active MoE. Pp factor rewards D≤128 (rocWMMA-tuned eligible). Output annotated so each row's score is auditable.

### Architecture sketch

```
gen_supported_arch.sh   → supported_hf_architectures.json
fetch.sh                → discovery.json                     (HF list calls; existing)
unsloth_mirrors.sh      → unsloth_mirrors.by_base.json       (existing)
enrich.sh               → enriched.json                      (config + tokenizer; existing, tokenizer always-on)
bench_swebench.sh       → swebench_scores.json               (NEW: GitHub sparse checkout)
collapse_variants.sh    → enriched.collapsed.json            (NEW: dedup model families)
rank.sh                 → ranked.json + shortlist.json       (rewritten; merges bench scores inline)
```

## Avoiding HF rate limits

- `fetch.sh` and `enrich.sh` are unchanged — they already cache via the `discovery.json`/`enriched.json` snapshots. Don't re-run unless something material changed upstream.
- `bench_swebench.sh` hits **github.com**, not Hugging Face — no HF rate-limit risk. Uses sparse checkout to grab only `evaluation/verified/*/metadata.yml` and `evaluation/verified/*/results/results.json` (~270 small files). Single clone, parsed locally.
- Token in `HF_TOKEN_VAL` already plumbed for [enrich.sh](enrich.sh) — keep using it for the rare gated-config fetch.

## What we are NOT doing in this pass

- LiveCodeBench / BigCodeBench / Aider-polyglot scrapers. SWE-bench Verified is the highest-signal bench for agentic coding and the easiest to scrape cleanly. Add others later only if the SWE-bench shortlist is too thin.
- A "score Qwen 3.6 ourselves" pass. Out of scope here; the bar is "in the vicinity, not dramatically worse" — using third-party SWE-bench scores against a competent floor.
- Touching [enrich.sh](enrich.sh) / [fetch.sh](fetch.sh). They work; reruns are cached.

## Success criteria

After running the new pipeline end-to-end, the shortlist must contain:

- Qwen3-Coder-30B-A3B-Instruct (the obvious replacement; SWE-bench Verified resolved-rate published)
- GLM-4.5-Air or GLM-4.6 (if either has SWE-bench scores and fits the box)
- Some MoE in the 30–80B / 3–6B-active range with a real bench

And must NOT contain:

- Reward models, lora-intrinsics adapters, instruct-preview duplicates
- Math-specialty fine-tunes (OpenMath-Nemotron, AceReason)
- Models with no published coding evidence and no Unsloth Dynamic GGUF
