#!/usr/bin/env bash
# rank.sh — score and shortlist coding-model candidates for the gfx1151 box.
#
# Hard gates  → variant collapse → score → ranked.json + shortlist.json.
#
# Hard gates (drop if any fails):
#   - head_dim ≤ 128         (rocWMMA-tuned FA path; +40-65% pp@depth vs TILE)
#   - max_position ≥ 131072  (agentic coding workload)
#   - permissive license     (no NC / GPL / AGPL)
#   - has_chat_template      (jinja=true is the production default)
#   - weight_gb_q4 ≤ 80      (fits 100 GB usable with KV + cache-ram)
#   - deploy path exists     (target-quant GGUF in repo, Unsloth mirror,
#                             unsloth/* author, OR convert_hf_to_gguf supports the arch)
#   - id does not match variant_drop (reward / lora / prm / abliterated / merge / preview / sft / dpo / orcaa / ifeval / distill)
#
# Variant collapse: group by (model_type, total_params rounded to 100M, layers, experts, head_dim).
#   Within a group, prefer (has bench score) > (instruct/chat over base) > (most downloaded).
#
# Score:
#   1000 × swebench_score                           [0..1000; null → tier 2]
#   + 200 if hybrid attention (sliding_window / layer_types)
#   + 100 if MoE
#   + 100 if active_est < 5 B (bandwidth-friendly)
#   −  weight_gb_q4_est                              (light tiebreaker)
#
# Env:
#   INPUT          enriched.json
#   BENCH          swebench_scores.json (optional but expected)
#   OUTPUT         ranked.json
#   SHORTLIST      shortlist.json
#   SHORTLIST_TOP  8
set -euo pipefail
cd "$(dirname "$0")"

INPUT="${INPUT:-enriched.json}"
BENCH="${BENCH:-swebench_scores.json}"
OUTPUT="${OUTPUT:-ranked.json}"
SHORTLIST="${SHORTLIST:-shortlist.json}"
SHORTLIST_TOP="${SHORTLIST_TOP:-8}"

[ -f "$INPUT" ] || { echo "missing $INPUT — run ./enrich.sh" >&2; exit 1; }

bench_arg="$BENCH"
if [ ! -f "$BENCH" ]; then
  printf '>>> no %s — running without bench scores\n' "$BENCH" >&2
  bench_arg=$(mktemp)
  trap 'rm -f "$bench_arg"' EXIT
  echo '{"by_model":{}}' > "$bench_arg"
fi

bundle=$(mktemp)
trap '[ -n "${bundle:-}" ] && rm -f "$bundle"; [ -n "${bench_arg:-}" ] && [ "$bench_arg" != "$BENCH" ] && rm -f "$bench_arg"' EXIT

jq --slurpfile bench "$bench_arg" --arg src "$INPUT" --argjson stop "$SHORTLIST_TOP" '
  ($bench[0].by_model // {}) as $bm |

  def is_moe: (.moe == true) or ((.num_experts // 0) > 0);
  def gb_q4: if .total_params == null then null else (.total_params * 4.5 / 8 / 1.0e9) end;
  def active_est:
    if (.total_params // 0) <= 0 then null
    elif is_moe and (.num_experts != null) and (.num_experts > 0) and (.num_experts_per_tok != null)
      then (.total_params * (.num_experts_per_tok / .num_experts))
    elif is_moe then null
    else .total_params
    end;
  # tg ceiling at Q4 (bytes_per_param ≈ 0.5625), 256 GB/s LPDDR5x.
  def tg_ceiling_q4:
    active_est as $a
    | if $a == null or $a <= 0 then null
      else (256.0e9 / ($a * 0.5625)) end;
  def hybrid_attn: (.sliding_window != null) or (.layer_types != null);
  def license_text:
    ((.license // "") | ascii_downcase) + " " +
    (([ .tags[]? | select(test("^license:";"i")) ][0] // "") | ascii_downcase);
  def license_ok:
    license_text as $l
    | ($l | test("cc-by-nc|noncommercial|agpl|gnu affero"; "i") | not)
      and ($l | test("(^|[^a-z])gpl([^a-z]|$)"; "i") | not);
  def deploy_ok:
    (.has_target_quant_gguf == true)
    or (((.unsloth_gguf_mirrors // []) | length) > 0)
    or ((.id // "") | startswith("unsloth/"))
    or (.convert_hf_arch_supported == true);
  def variant_drop:
    ((.id // "") | ascii_downcase
     | test("(reward|-prm-|/prm-|prm-v|-lora-|/lora-|abliter|/merge-|-merge-|distill|-sft-|/sft-|-dpo-|/dpo-|orca|ifeval|preview|guardian|/dare|-dare-|nemo-mini|nemo-instruct-mini|safe|toxic|nsfw|roleplay|miqu|chimera|frankenstein|dolphin|reflection|jangq|jangtq|kaggle|openmath|acereason|-math-|/math-|-jp$|-japanese|-de$|-german|-fr$|-french|-zh$|-chinese)";"i"));
  def gate_hard:
    ((.error // "") == "")
    and (.head_dim != null) and (.head_dim <= 128)
    and ((.max_position_embeddings // 0) >= 131072)
    and license_ok
    and deploy_ok
    and (gb_q4 != null) and (gb_q4 <= 80)
    # Predicted tg ceiling >= 80 t/s — "in the vicinity of Qwen 3.6 measured (~46 t/s)";
    # ratios suggest ~30% of ceiling lands as measured, so 80 t/s ceiling ≈ 24 t/s
    # measured, the floor for "not dramatically worse." Dense 24B+ models fail this.
    and (tg_ceiling_q4 != null) and (tg_ceiling_q4 >= 80)
    and (.has_chat_template != false)  # null is allowed; only drop on explicit false. Run ENRICH_FETCH_TOKENIZER=1 ./enrich.sh to populate.
    and (variant_drop | not);

  def with_bench:
    . + { swebench_score: ($bm[.id].score // null),
          swebench_resolved: ($bm[.id].resolved // null),
          swebench_best_submission: ($bm[.id].best_submission // null),
          swebench_info_name: ($bm[.id].info_name // null) };

  def collapse_sig:
    "\(.model_type // "?")|\((.total_params // 0) / 1.0e8 | floor)|\(.num_hidden_layers // 0)|\(.num_experts // 0)|\(.head_dim // 0)";

  def is_instruct: ((.id // "") | ascii_downcase | test("instruct|/it($|-)|-it-|/chat($|-)|-chat$|-chat-"; "i"));
  def is_base:     ((.id // "") | ascii_downcase | test("(^|/)base|-base($|-)|/pt$|-pt$"; "i"));

  # collapse group ordering: bench-scored first, then instruct, then most-downloaded.
  def variant_pref:
    [ (if (.swebench_score // null) == null then 1 else 0 end),
      -(.swebench_score // 0),
      (if is_instruct then 0 else 1 end),
      (if is_base then 1 else 0 end),
      (if .derivative_of == null then 0 else 1 end),
      -(.downloads // 0),
      .id ];

  def metrics:
    {
      gb_q4_est: (gb_q4 | if . == null then null else (. * 100 | floor / 100) end),
      active_est: active_est,
      tg_ceiling_q4: (tg_ceiling_q4 | if . == null then null else (. * 10 | floor / 10) end),
      hybrid_attn: hybrid_attn,
      moe: is_moe,
      license: (.license // null),
      deploy_target_quant_in_repo: (.has_target_quant_gguf // false),
      deploy_unsloth_mirror: (((.unsloth_gguf_mirrors // []) | length) > 0),
      deploy_convert_supported: (.convert_hf_arch_supported // false)
    };

  def score:
    metrics as $m
    | (.swebench_score // null) as $s
    | (
        (if $s == null then 0 else ($s * 1000) end)
        + (if hybrid_attn then 200 else 0 end)
        + (if is_moe then 100 else 0 end)
        + (if (active_est // 1.0e12) < 5.0e9 then 100 else 0 end)
        - ($m.gb_q4_est // 80)
      );

  ($bm | to_entries | map(.value.score)) as $bench_scores |

  (.models
    | map(with_bench)
    | map(select(gate_hard))
  ) as $survivors
  |
  ($survivors
    | group_by(collapse_sig)
    | map(
        sort_by(variant_pref) as $sorted
        | $sorted[0] + {
            _collapse: {
              count: ($sorted | length),
              alternates: ($sorted[1:] | map(.id))
            }
          }
      )
  ) as $reps
  |
  ($reps
    | map(. + {
        _metrics: metrics,
        _score: score
      })
    | sort_by(-(._score))
  ) as $ranked
  |
  {
    generated_at: (now | todate),
    source_input: $src,
    source_bench: ($bench[0].source // null),
    bench_total_instances: ($bench[0].total_instances // null),
    bench_models_with_score: ($bm | length),
    gates: {
      head_dim_max: 128,
      ctx_min: 131072,
      weight_gb_q4_max: 80,
      license: "permissive (drops NC/GPL/AGPL)",
      deploy: "target-quant GGUF in repo OR Unsloth mirror OR unsloth/* OR convert_hf supports arch",
      requires_chat_template: true
    },
    score_formula: "1000*swebench + 200*hybrid + 100*moe + 100*(active<5B) - gb_q4",
    counts: { input: (.models | length),
              survived_gates: ($survivors | length),
              after_collapse: ($reps | length) },
    ranked: $ranked,
    shortlist: ($ranked | .[0:$stop])
  }
' "$INPUT" > "$bundle"

# Project a compact view per output.
jq '{
  generated_at, source_input, source_bench, bench_total_instances, bench_models_with_score,
  gates, score_formula, counts,
  models: (.ranked
    | to_entries
    | map(.value | {
        rank: 0,
        id, model_type,
        score: ._score,
        swebench_score, swebench_resolved, swebench_best_submission, swebench_info_name,
        head_dim, max_position_embeddings,
        total_params, num_experts, num_experts_per_tok,
        sliding_window, layer_types,
        license, derivative_of,
        unsloth_gguf_mirrors, has_target_quant_gguf, convert_hf_arch_supported,
        downloads, trending_score,
        _metrics, _collapse
      } )
    | to_entries | map(.value + {rank: (.key + 1)}))
}' "$bundle" > "$OUTPUT"

jq --argjson n "$SHORTLIST_TOP" '{
  generated_at, source_input, source_bench, bench_total_instances, bench_models_with_score,
  gates, score_formula, counts,
  shortlist_top: $n,
  models: (.shortlist
    | to_entries
    | map(.value | {
        rank: 0,
        id, model_type,
        score: ._score,
        swebench_score, swebench_resolved, swebench_best_submission, swebench_info_name,
        head_dim, max_position_embeddings,
        total_params, num_experts, num_experts_per_tok,
        sliding_window, layer_types,
        license, derivative_of,
        unsloth_gguf_mirrors, has_target_quant_gguf, convert_hf_arch_supported,
        downloads, trending_score,
        _metrics, _collapse
      })
    | to_entries | map(.value + {rank: (.key + 1)}))
}' "$bundle" > "$SHORTLIST"

n_in=$(jq '.counts.input' "$OUTPUT")
n_g=$(jq '.counts.survived_gates' "$OUTPUT")
n_c=$(jq '.counts.after_collapse' "$OUTPUT")
n_b=$(jq '.bench_models_with_score' "$OUTPUT")
printf '>>> wrote %s + %s (input=%s → gates=%s → collapse=%s; bench-scored models=%s)\n' \
  "$OUTPUT" "$SHORTLIST" "$n_in" "$n_g" "$n_c" "$n_b" >&2
