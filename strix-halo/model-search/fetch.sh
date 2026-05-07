#!/usr/bin/env bash
# fetch.sh — HF model discovery for the gfx1151 box.
# Sweeps several sort axes and per-author top-N lists, dedupes, drops obvious
# quant/finetune noise, writes discovery.json. List calls only (no per-repo
# config.json). Uses --expand for siblings/tags/safetensors so enrich can skip
# redundant hf models info when Hub returns them.
#
# Env overrides:
#   PARAMS    "min:8B,max:130B"   param-range filter passed to `hf models list`
#   LIMIT     60                  per-query result cap
#   OUTPUT    discovery.json      output file (relative to script dir)
set -euo pipefail
cd "$(dirname "$0")"

PARAMS="${PARAMS:-min:8B,max:130B}"
LIMIT="${LIMIT:-100}"
OUTPUT="${OUTPUT:-discovery.json}"

# sort axis | filter1 | filter2
queries=(
  "trending_score|text-generation|"
  "downloads|text-generation|"
  "likes|text-generation|"
  "trending_score|text-generation|code"
  "trending_score|text-generation|conversational"
  "downloads|text-generation|code"
  "last_modified|text-generation|"
  "last_modified|text-generation|code"
)

# Author-anchored top-N (downloads sort, text-generation only)
authors=(
  Qwen
  zai-org
  meta-llama
  mistralai
  microsoft
  openai
  ibm-granite
  nvidia
  deepseek-ai
  moonshotai
  Alibaba-NLP
  baidu
  LiquidAI
  inclusionAI
  NousResearch
  google
  01-ai
  CohereLabs
  AI21labs
  databricks
  apple
  agentica-org
  MiniMaxAI
  Skywork
  poolside
  tiiuae
)

# Author-anchored, narrowed to code/coding tag — surfaces niche coder fine-tunes
# that fall below LIMIT in the unfiltered downloads sort.
code_authors=(
  Qwen
  mistralai
  deepseek-ai
  ibm-granite
  agentica-org
  Skywork
)

# Author-anchored by last_modified — picks up brand-new releases that haven't
# accumulated downloads yet (e.g., December 2026 mistralai drops).
recent_authors=(
  Qwen
  mistralai
  google
  microsoft
  deepseek-ai
  moonshotai
  agentica-org
  Skywork
  zai-org
  inclusionAI
)

# Must-fetch IDs — explicit names that bypass list/sort. Use sparingly; reserved
# for known-target models that we always want enriched even if they fall outside
# the discovery sweep (e.g., niche releases, lower-trending coder fine-tunes).
# Resolves via `hf models info` per id (one call each).
must_fetch=(
  agentica-org/DeepSWE-Preview
  Skywork/Skywork-SWE-32B
  mistralai/Devstral-Small-2505
  mistralai/Devstral-Small-2507
  mistralai/Codestral-22B-v0.1
  google/gemma-3-27b-it
  google/gemma-3-12b-it
  microsoft/phi-4
  Qwen/Qwen3-30B-A3B-Instruct-2507
)

export LIST_QUERY_COUNT=$((${#queries[@]} + ${#authors[@]} + ${#code_authors[@]} + ${#recent_authors[@]} + ${#must_fetch[@]}))

raw=$(mktemp)
trap 'rm -f "$raw"' EXIT

run_query() {
  local sort="$1" f1="$2" f2="$3" author="$4" src="$5"
  local args=(--num-parameters "$PARAMS" --sort "$sort" --limit "$LIMIT" \
              --expand downloads,likes,trendingScore,tags,pipeline_tag,safetensors,gguf,siblings --json)
  [ -n "$f1" ] && args+=(--filter "$f1")
  [ -n "$f2" ] && args+=(--filter "$f2")
  [ -n "$author" ] && args+=(--author "$author")
  hf models list "${args[@]}" 2>/dev/null \
    | jq -c --arg src "$src" '
        .[] | . as $m
        | ($m.siblings // []) as $sib
        | ([$sib[] | (.rfilename // "") | select(endswith(".gguf"))]) as $gf
        | {
            id: $m.id,
            downloads: ($m.downloads // 0),
            likes: ($m.likes // 0),
            trending_score: ($m.trending_score // 0),
            src: [$src],
            pipeline_tag: ($m.pipeline_tag // null),
            list_tags: ($m.tags // null),
            list_safetensors: ($m.safetensors // null),
            hub_list_gguf: ($m.gguf // null),
            siblings_gguf_files: ($gf | .[0:30]),
            gguf_file_count: ($gf | length),
            has_target_quant_gguf: (
              ($gf | map(select(test("(?i)(Q4_K_XL|Q6_K_XL|UD-Q|IMATRIX|DYNAMIC)"))) | length) > 0
            ),
            has_config_sibling: ($sib | map(.rfilename == "config.json") | any)
          }
      '
}

for q in "${queries[@]}"; do
  IFS='|' read -r sort f1 f2 <<< "$q"
  src="sort=${sort},filter=${f1}${f2:+/$f2}"
  printf '>>> %s\n' "$src" >&2
  run_query "$sort" "$f1" "$f2" "" "$src" >> "$raw" || true
done

# Author-anchored queries do NOT pass --filter text-generation: many new releases
# (e.g. recent mistralai / google drops) leave pipeline_tag unset on the Hub list
# response and would be silently filtered out. The --author scope is enough.
for a in "${authors[@]}"; do
  src="author=$a"
  printf '>>> %s\n' "$src" >&2
  run_query downloads "" "" "$a" "$src" >> "$raw" || true
done

for a in "${code_authors[@]}"; do
  src="author=$a,filter=code"
  printf '>>> %s\n' "$src" >&2
  run_query downloads "" code "$a" "$src" >> "$raw" || true
done

for a in "${recent_authors[@]}"; do
  src="author=$a,sort=last_modified"
  printf '>>> %s\n' "$src" >&2
  run_query last_modified "" "" "$a" "$src" >> "$raw" || true
done

# Must-fetch: one `hf models info` call per id, formatted to the same shape as
# run_query. Errors swallowed (404/auth issues) so a stale entry doesn't break
# the run.
for id in "${must_fetch[@]}"; do
  src="must_fetch"
  printf '>>> must_fetch %s\n' "$id" >&2
  hf models info "$id" --expand downloads,likes,trendingScore,tags,pipeline_tag,safetensors,gguf,siblings --json 2>/dev/null \
    | jq -c --arg src "$src" '
        . as $m
        | ($m.siblings // []) as $sib
        | ([$sib[] | (.rfilename // "") | select(endswith(".gguf"))]) as $gf
        | {
            id: $m.id,
            downloads: ($m.downloads // 0),
            likes: ($m.likes // 0),
            trending_score: ($m.trending_score // 0),
            src: [$src],
            pipeline_tag: ($m.pipeline_tag // null),
            list_tags: ($m.tags // null),
            list_safetensors: ($m.safetensors // null),
            hub_list_gguf: ($m.gguf // null),
            siblings_gguf_files: ($gf | .[0:30]),
            gguf_file_count: ($gf | length),
            has_target_quant_gguf: (
              ($gf | map(select(test("(?i)(Q4_K_XL|Q6_K_XL|UD-Q|IMATRIX|DYNAMIC)"))) | length) > 0
            ),
            has_config_sibling: ($sib | map(.rfilename == "config.json") | any)
          }' >> "$raw" || true
done

raw_count=$(wc -l < "$raw" | tr -d ' ')
printf '>>> raw lines: %s\n' "$raw_count" >&2

jq -s --arg qcount "$LIST_QUERY_COUNT" --arg params "$PARAMS" '
  group_by(.id)
  | map({
      id: .[0].id,
      downloads: ([.[].downloads] | max),
      likes:     ([.[].likes] | max),
      trending_score: ([.[].trending_score] | max),
      sources:   ([.[].src[]] | unique),
      pipeline_tag: ([.[].pipeline_tag | values][0] // null),
      list_tags: ([.[].list_tags // [] | .[]] | unique),
      list_safetensors: (
        map(.list_safetensors | values | select(. != null and .total != null))
        | sort_by(-.total)
        | .[0] // null
      ),
      hub_list_gguf: ([.[].hub_list_gguf | select(. != null)][0] // null),
      siblings_gguf_files:
        ([.[].siblings_gguf_files // [] | .[]] | unique | .[0:30]),
      gguf_file_count: ([.[].gguf_file_count // 0] | max),
      has_target_quant_gguf: ([.[].has_target_quant_gguf // false] | any),
      has_config_sibling: ([.[].has_config_sibling // false] | any)
    })
  | map(select(
      # Quantized / non-original packagings (we want canonical safetensors repos for enrich)
      (.id | test("(?i)gguf|nvfp4|awq|fp8|mlx|mtp|gptq|bnb|exl2|w8a8|int4|int8|abliterated|onnx"; "x") | not)
      and
      # Frankenmerges, role-play, junk fine-tunes
      (.id | test("(?i)distill|merged|merge|uncensored|toxic|abliterated|roleplay|nsfw|finetuned|miqu|chimera|frankenstein|venice|dolphin-|reflection|geneticlemonade|esper3"; "x") | not)
      and
      # Modalities that are not text-LLM-coders (audio, vision, embeddings, safety)
      (.id | test("(?i)voice|/asr|-asr|whisper|/audio|-audio|/tts|-tts|diariz|/embed|-embedding|/rerank|-rerank|guardian|/guard|llama-?guard|prompt-?guard|shield|/ocr|-ocr|/clip|paligemma|-vl|/vl-|llama-vision|phi-?vision|gui-actor|colembed|flamingo|molmo|magma|voxtral|videogen|video-|/img|harrier|userlm"; "x") | not)
    ))
  | sort_by(-.trending_score, -.downloads)
  | {
      generated_at: (now | todate),
      param_range: $params,
      query_count: ($qcount | tonumber),
      candidates: .
    }
' "$raw" > "$OUTPUT"

count=$(jq -r '.candidates | length' "$OUTPUT")
printf '>>> wrote %s (%s candidates after dedupe + filter)\n' "$OUTPUT" "$count" >&2
