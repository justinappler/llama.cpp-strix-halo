# Model search for gfx1151

Re-runnable pipeline that surfaces the best **coding-model replacements for Qwen 3.6 35B-A3B** on this hardware. Outputs are checked-in JSON snapshots tagged with `generated_at`. Companion to [`coder-next-baseline.md`](../coder-next-baseline.md), [`qwen3.6-baseline.md`](../qwen3.6-baseline.md), and [`fa-dispatcher.md`](../fa-dispatcher.md). Plan / rationale: [`plan.md`](plan.md).

## Aim

Find 5–8 candidates that pass three tests in order:

1. **Predicted runtime is in the vicinity of, or better than, Qwen 3.6's** on this box. Not "must beat" — "not dramatically worse." Compared on:
   - **tg ceiling** = `BW / (active_params × bytes_per_param)`. 256 GB/s ÷ (3 B × 0.56 B/param at Q4) ≈ 150 t/s for Qwen 3.6 A3B.
   - **pp@depth.** D≤128 unlocks the rocWMMA-tuned FA path (+40-65% pp@depth vs TILE per [Finding #6](../README.md)). Both Qwen 3.6 and Qwen3-Coder-Next are D=256 → stuck on TILE; D≤128 candidates have a structural depth-perf advantage.
   - **Hybrid attention** (sliding-window or recurrent layers) lowers the depth-linear attention cost on the layers that aren't full-attention.
2. **Published evidence it codes ≥ Qwen 3.6.** SWE-bench Verified resolved-rate is the load-bearing benchmark for agentic coding. Threshold: ≥ ~30% suggests "competent agentic coder." Scraped from [`SWE-bench/experiments`](https://github.com/SWE-bench/experiments) (134 submissions, ~9 open-source models with HF ids).
3. **Deploys here today.** Unsloth Dynamic GGUF in repo, OR Unsloth GGUF mirror, OR `convert_hf_to_gguf.py`-supported arch. Permissive license. Working Jinja chat template (`jinja=true` in `models.ini`).

## Hardware envelope

| Component       | Value                                                                          |
| --------------- | ------------------------------------------------------------------------------ |
| GPU             | Radeon 8060S (gfx1151, RDNA 3.5), ~59 TFLOPS FP16 peak (WMMA)                  |
| Memory          | 128 GB LPDDR5x-8000 unified, **~256 GB/s** bandwidth, ~88–100 GB usable budget |
| FA kernel       | TILE for D=256, rocWMMA-tuned for D ≤ 128 ([Finding #6](../README.md))         |
| KV constraint   | f16/f16 mandatory on TILE — V-quant collapses pp 736→45 t/s @ d=16k (Qwen 3.6 [`kv-cache.md`](../kv-cache.md)), re-confirmed at 11× on Coder-Next |
| Watching        | JG's MMA-on-AMD-WMMA path ([Finding #7](../README.md)) — when it lands, D=256 becomes viable and the gate may relax |

Net: this pipeline gates on **D≤128**. That's the hard ask. Everything else is preference.

## Pipeline

```
gen_supported_arch.sh   → supported_hf_architectures.json   (from ../../convert_hf_to_gguf.py)
fetch.sh                → discovery.json                    (HF list across sort axes + per-author top-N + per-author code-filter)
unsloth_mirrors.sh      → unsloth_mirrors.by_base.json      (optional; populates unsloth_gguf_mirrors in enrich)
enrich.sh               → enriched.json                     (config.json + safetensors + GGUF hints; tokenizer optional)
bench_swebench.sh       → swebench_scores.json              (sparse checkout of SWE-bench/experiments — github.com only)
rank.sh                 → ranked.json + shortlist.json      (hard gates → variant collapse → score)
cards.sh                → cards/<slug>.md                   (optional: cache READMEs for human review of tier-2 candidates)
```

`rank.sh` is the one with opinions. Everything before it is straightforward enrichment.

### Hard gates (rank.sh)

A model is dropped if any fails:

- `head_dim ≤ 128` — rocWMMA-tuned FA path eligibility
- `max_position_embeddings ≥ 131072` — agentic coding workload
- `tg_ceiling_q4 ≥ 80 t/s` — `BW / (active × 0.5625B)` floor. Qwen 3.6 measures ~46 t/s; ~30% of ceiling lands as measured, so 80 t/s ceiling ≈ 24 t/s measured = the floor for "not dramatically worse" (Devstral-24B-dense fails this with ~19 t/s ceiling, even at 46% SWE-V)
- License is permissive — drops `cc-by-nc`, AGPL, plain GPL
- `weight_gb_q4_est ≤ 80` — leaves ≥20 GB for KV + cache-ram
- Deploy path exists — XL/Dynamic GGUF in repo, OR Unsloth mirror, OR `unsloth/*` author, OR `convert_hf_to_gguf.py` supports the arch
- `has_chat_template != false` — null is allowed for back-compat; `ENRICH_FETCH_TOKENIZER=1 ./enrich.sh` (the recommended default) populates it
- `id` does **not** match the variant-drop regex: reward / lora / prm / abliter / merge / preview / sft / dpo / orca / ifeval / distill / openmath / acereason / kaggle / language-tag suffixes

### Variant collapse

Within a `(model_type, total_params [rounded to 100M], num_hidden_layers, num_experts, head_dim)` group, keep one representative ordered by:

1. has SWE-bench score (yes first)
2. SWE-bench score (descending)
3. id contains "instruct" / "chat"
4. id does **not** contain "base"
5. `derivative_of == null`
6. downloads (descending)

Alternates carried as `_collapse.alternates` for audit, not as separate rows. This compresses the 8-Granite-8B-variant cluster into one row.

### Score (lower-tiered if no bench)

```
score = 1000 × swebench_score        # null → 0; bench-scored models always rank above tier 2
       + 200 if hybrid attention (sliding_window or layer_types)
       + 100 if MoE
       + 100 if active_est < 5 B     # bandwidth-friendly
       − weight_gb_q4_est            # tiebreaker, lighter is slightly better
```

## Re-running

```bash
./gen_supported_arch.sh   # refresh allowlist after upstream converter changes
./fetch.sh                # ~30s; HF list calls (cached via discovery.json)
./unsloth_mirrors.sh      # optional; populates unsloth_gguf_mirrors
ENRICH_FETCH_TOKENIZER=1 ./enrich.sh   # ~1 min per 100 candidates at PAR=6; cached
./bench_swebench.sh       # github.com sparse checkout; no HF traffic. REFRESH=1 to git pull.
./rank.sh                 # gates → collapse → score; writes ranked.json + shortlist.json
jq -r '.models[] | .id' shortlist.json | ./cards.sh   # optional human-review aid
```

**Avoiding HF rate limits.** `fetch.sh` and `enrich.sh` are the only HF-touching steps and both cache to checked-in JSON. Run authenticated (`hf auth login`) — many gated repos (mistralai, google, meta-llama) won't even appear in `discovery.json` without a token. `bench_swebench.sh` hits github.com only.

Env overrides on the scripts: see each script's header. The interesting ones:

- `fetch.sh`: `PARAMS=min:8B,max:130B`, `LIMIT=100`. Adds: per-author `last_modified` queries (`recent_authors`), per-author `code` filter (`code_authors`), and a `must_fetch` allowlist of explicit IDs that bypass list/sort. Author-anchored queries do **not** pass `--filter text-generation` because new releases (mistralai 2512, gemma-3) leave `pipeline_tag` unset.
- `enrich.sh`: `PAR=6`, `ENRICH_FETCH_TOKENIZER=0` (override to 1 for the chat-template gate), `REQUIRE_CONFIG_SIBLING=0`. Falls back to `text_config.*` when top-level head_dim/hidden_size/max_position are null (Mistral3 / Gemma3 / Llama4 multimodal configs).
- `bench_swebench.sh`: `REFRESH=0` (set 1 to `git pull` the cached SWE-bench/experiments checkout)
- `rank.sh`: `SHORTLIST_TOP=8`

## Reading `shortlist.json`

Per-model fields (compact subset from `ranked.json`):

| Field                        | Notes                                                                   |
| ---------------------------- | ----------------------------------------------------------------------- |
| `rank`                       | 1-indexed by score                                                      |
| `score`                      | composite (see formula above)                                           |
| `swebench_score`             | resolved / 500 on SWE-bench Verified; null if not benched               |
| `swebench_resolved`          | absolute count                                                          |
| `swebench_best_submission`   | which submission directory in [SWE-bench/experiments] gave the best score |
| `swebench_info_name`         | human-readable submission name (e.g. "OpenHands + Qwen3-Coder-30B-A3B-Instruct") |
| `head_dim`, `max_position_embeddings`, `total_params`, `num_experts`, `num_experts_per_tok`, `sliding_window`, `layer_types` | gate inputs |
| `license`, `derivative_of`   | gate / collapse inputs                                                  |
| `unsloth_gguf_mirrors`, `has_target_quant_gguf`, `convert_hf_arch_supported` | deploy-path inputs                              |
| `_metrics.gb_q4_est`, `_metrics.active_est`, `_metrics.tg_ceiling_q4` | predicted footprint and tg ceiling on this box |
| `_metrics.hybrid_attn`, `_metrics.moe`                                | scoring bonuses              |
| `_collapse.count`, `_collapse.alternates`                             | how many variants merged here, and which       |

## Filtering with jq

`enriched.json` is a flat list of records. Use jq directly.

```bash
# Hard gate baseline: D ≤ 128, ctx ≥ 128k, no errors.
jq -r '.models[] | select(.head_dim != null and .head_dim <= 128 and (.max_position_embeddings // 0) >= 131072 and (.error // "") == "") | .id' enriched.json

# MoE only, sorted by active-token bandwidth (fewer experts/tok = faster tg).
jq -r '.models[]
       | select(.head_dim != null and .head_dim <= 128 and .moe)
       | "\(.id)\t D=\(.head_dim)  total=\(.total_params)  experts=\(.num_experts)  active/tok=\(.num_experts_per_tok)"' \
  enriched.json | column -t -s$'\t'

# Cross-reference: who has a SWE-bench Verified score?
jq -r '.by_model | to_entries | sort_by(-.value.score) | .[] | "\(.value.score | tostring | .[0:5])  \(.key)  \(.value.info_name)"' swebench_scores.json
```

## When to re-run

- After upstream lands a kernel change that flips the head-dim gate (e.g. JG's MMA-on-AMD-WMMA work merging) — relax to D ≤ 256 in [`rank.sh`](rank.sh).
- After a fresh model release worth catching. HF discovery is sort-by-trending heavy, so newer models surface fast.
- Periodically refresh the SWE-bench scrape: `REFRESH=1 ./bench_swebench.sh` — that's a `git pull` on the cached checkout, no HF traffic.

## Errors

`enrich.sh` records per-repo failures inline (the model gets `{id, error, http}` instead of a full record). Common causes:

- `config_unavailable` — 401/403 (gated repo) or 404 (no config.json at root)
- `config_not_json` — HTML returned (HF redirected to a model-card stub)
- `skipped_no_config_sibling` — only when `REQUIRE_CONFIG_SIBLING=1` and Hub list lacks `config.json`

Filter them: `jq '.models[] | select((.error // "") == "")'`.
