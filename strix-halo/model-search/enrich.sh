#!/usr/bin/env bash
# enrich.sh — for every candidate in discovery.json, fetch config.json and
# optionally `hf models info` (skipped when list phase already returned
# safetensors + tags). Merge discovery.list GGUF hints and llama.cpp convert
# support flags. Write enriched.json.
#
# Optional: unsloth_mirrors.by_base.json in this directory (from ./unsloth_mirrors.sh).
#
# Env overrides:
#   INPUT    discovery.json
#   OUTPUT   enriched.json
#   PAR      6                  parallel fetches (xargs -P)
#   REQUIRE_CONFIG_SIBLING=1   skip curl/info when discovery has no config.json in siblings (Hub index)
#   ENRICH_FETCH_TOKENIZER=0   if 1, also curl tokenizer_config.json (has_chat_template, chat_template_tool_hint)
#
# Token for curl /resolve/.../config.json (gated repos): hf auth login, cache file, or HF_TOKEN /
# HUGGING_FACE_HUB_TOKEN when CLI/cache unavailable (e.g. agent/CI).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INPUT="${INPUT:-discovery.json}"
OUTPUT="${OUTPUT:-enriched.json}"
PAR="${PAR:-6}"
REQUIRE_CONFIG_SIBLING="${REQUIRE_CONFIG_SIBLING:-0}"
ENRICH_FETCH_TOKENIZER="${ENRICH_FETCH_TOKENIZER:-0}"

ARCH_FILE="${ARCH_FILE:-$SCRIPT_DIR/supported_hf_architectures.json}"
MIRROR_FILE="${MIRROR_FILE:-$SCRIPT_DIR/unsloth_mirrors.by_base.json}"

[ -f "$INPUT" ] || { echo "missing $INPUT — run ./fetch.sh first" >&2; exit 1; }
[ -f "$ARCH_FILE" ] || { echo "missing $ARCH_FILE — run ./gen_supported_arch.sh" >&2; exit 1; }

HF_TOKEN_VAL="$(hf auth token 2>/dev/null || true)"
if [ -z "$HF_TOKEN_VAL" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN_VAL="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "$HF_TOKEN_VAL" ]; then
  HF_TOKEN_VAL="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
fi
export HF_TOKEN_VAL

work="${ENRICH_WORK:-$(mktemp -d)}"
cleanup_work() {
  if [ "${KEEP_WORK:-0}" != "1" ]; then
    rm -rf "$work"
  else
    printf '>>> KEEP_WORK=1: left %s\n' "$work" >&2
  fi
}
trap cleanup_work EXIT

jq '.candidates | map({(.id): .}) | add' "$INPUT" > "$work/_discovery_map.json"

cat > "$work/fetch_one.sh" <<'INNER'
#!/usr/bin/env bash
set -uo pipefail
id="$1"
out_dir="$2"
disc_map="$3"
slug="${id//\//__}"
out="$out_dir/$slug.json"

REQUIRE_CONFIG_SIBLING="${REQUIRE_CONFIG_SIBLING:-0}"

meta=$(jq -c --arg id "$id" '.[$id] // {}' "$disc_map")

if [ "$REQUIRE_CONFIG_SIBLING" = "1" ]; then
  ok=$(echo "$meta" | jq -r '(.has_config_sibling // false) | tostring')
  if [ "$ok" != "true" ]; then
    jq -nc --arg id "$id" \
      --argjson meta "$meta" \
      '{
        id: $id,
        error: "skipped_no_config_sibling",
        pipeline_tag: ($meta.pipeline_tag // null),
        list_tags: ($meta.list_tags // null),
        list_safetensors: ($meta.list_safetensors // null),
        hub_list_gguf: ($meta.hub_list_gguf // null),
        siblings_gguf_files: ($meta.siblings_gguf_files // []),
        gguf_file_count: ($meta.gguf_file_count // 0),
        has_target_quant_gguf: ($meta.has_target_quant_gguf // false),
        has_config_sibling: false,
        has_chat_template: null,
        chat_template_tool_hint: null
      }' > "$out"
    exit 0
  fi
fi

cfg_url="https://huggingface.co/${id}/resolve/main/config.json"
auth_args=()
[ -n "${HF_TOKEN_VAL:-}" ] && auth_args+=(-H "Authorization: Bearer ${HF_TOKEN_VAL}")
resp=$(curl -sL --max-time 25 -w '\n%{http_code}' "${auth_args[@]}" "$cfg_url" 2>/dev/null || true)
http=$(printf '%s\n' "$resp" | tail -1)
body=$(printf '%s\n' "$resp" | sed '$d')

if [ "$http" != "200" ] || ! printf '%s' "$body" | head -c 1 | grep -q '{'; then
  jq -nc --arg id "$id" --arg http "$http" --argjson meta "$meta" \
    '{
      id: $id,
      error: "config_unavailable",
      http: $http,
      pipeline_tag: ($meta.pipeline_tag // null),
      list_tags: ($meta.list_tags // null),
      list_safetensors: ($meta.list_safetensors // null),
      hub_list_gguf: ($meta.hub_list_gguf // null),
      siblings_gguf_files: ($meta.siblings_gguf_files // []),
      gguf_file_count: ($meta.gguf_file_count // 0),
      has_target_quant_gguf: ($meta.has_target_quant_gguf // false),
      has_config_sibling: ($meta.has_config_sibling // false),
      has_chat_template: null,
      chat_template_tool_hint: null
    }' > "$out"
  exit 0
fi

if ! printf '%s' "$body" | jq -e . >/dev/null 2>&1; then
  jq -nc --arg id "$id" --argjson meta "$meta" \
    '{
      id: $id,
      error: "config_not_json",
      pipeline_tag: ($meta.pipeline_tag // null),
      list_tags: ($meta.list_tags // null),
      list_safetensors: ($meta.list_safetensors // null),
      hub_list_gguf: ($meta.hub_list_gguf // null),
      siblings_gguf_files: ($meta.siblings_gguf_files // []),
      gguf_file_count: ($meta.gguf_file_count // 0),
      has_target_quant_gguf: ($meta.has_target_quant_gguf // false),
      has_config_sibling: ($meta.has_config_sibling // false),
      has_chat_template: null,
      chat_template_tool_hint: null
    }' > "$out"
  exit 0
fi

body_compact=$(printf '%s' "$body" | jq -c .)

tok_meta='{"has_chat_template":null,"chat_template_tool_hint":null}'
if [ "${ENRICH_FETCH_TOKENIZER:-0}" = "1" ]; then
  tok_url="https://huggingface.co/${id}/resolve/main/tokenizer_config.json"
  tresp=$(curl -sL --max-time 20 -w '\n%{http_code}' "${auth_args[@]}" "$tok_url" 2>/dev/null || true)
  thttp=$(printf '%s\n' "$tresp" | tail -n1)
  tbody=$(printf '%s\n' "$tresp" | sed '$d')
  if [ "$thttp" = "200" ] && printf '%s' "$tbody" | jq -e . >/dev/null 2>&1; then
    # chat_template is usually a string, but some tokenizer configs (older Mistral
    # v3 etc.) ship it as an array of {name, template} objects. Handle both.
    tok_meta=$(printf '%s' "$tbody" | jq -c '
      (.chat_template // null) as $ct
      | (
          if ($ct | type) == "string" then $ct
          elif ($ct | type) == "array" then
            ([$ct[]? | (.template // empty)] | join(" "))
          else "" end
        ) as $cs
      | { has_chat_template: ($cs | length > 0),
          chat_template_tool_hint: ($cs | ascii_downcase | test("tool|tools|function_call|functions")) }
    ' 2>/dev/null) || tok_meta='{"has_chat_template":null,"chat_template_tool_hint":null}'
  fi
fi

use_list_info=false
if echo "$meta" | jq -e '(.list_safetensors != null) and (.list_safetensors.total != null) and (.list_tags != null)' >/dev/null 2>&1; then
  use_list_info=true
fi

if [ "$use_list_info" = true ]; then
  info=$(jq -nc --argjson meta "$meta" '{safetensors: {total: $meta.list_safetensors.total}, tags: ($meta.list_tags // [])}')
  used_json='true'
else
  info=$(hf models info "$id" --expand safetensors,tags --json 2>/dev/null || true)
  [ -z "${info:-}" ] && info='{}'
  info=$(printf '%s' "$info" | jq -c . 2>/dev/null || echo '{}')
  used_json='false'
fi

jq -nc \
  --arg id "$id" \
  --argjson cfg "$body_compact" \
  --argjson info "$info" \
  --argjson meta "$meta" \
  --argjson used_list "$used_json" \
  --argjson tok_meta "$tok_meta" \
  '
  # Multimodal configs (Mistral3, Gemma3, Llama4 etc.) nest the text-LLM fields
  # under .text_config. Fall back to that when top-level fields are null.
  ($cfg.text_config // {}) as $tc |
  def fb($f): ($cfg[$f] // $tc[$f] // null);
  {
    id: $id,
    model_type: fb("model_type"),
    architectures: ($cfg.architectures // null),
    hidden_size: fb("hidden_size"),
    num_hidden_layers: fb("num_hidden_layers"),
    num_attention_heads: fb("num_attention_heads"),
    num_key_value_heads: fb("num_key_value_heads"),
    head_dim: (
      fb("head_dim") as $hd |
      fb("hidden_size") as $hs |
      fb("num_attention_heads") as $nh |
      if $hd != null then $hd
      elif ($hs != null and $nh != null and $nh > 0 and ($hs % $nh) == 0)
        then ($hs / $nh | floor)
      else null end
    ),
    head_dim_source: (
      fb("head_dim") as $hd |
      fb("hidden_size") as $hs |
      fb("num_attention_heads") as $nh |
      if $hd != null then (if $cfg.head_dim != null then "explicit" else "explicit_text_config" end)
      elif ($hs != null and $nh != null and $nh > 0 and ($hs % $nh) == 0) then "derived"
      else "unknown" end
    ),
    max_position_embeddings: fb("max_position_embeddings"),
    sliding_window: fb("sliding_window"),
    rope_scaling: ($cfg.rope_scaling // $tc.rope_scaling // $tc.rope_parameters // null),
    intermediate_size: fb("intermediate_size"),
    vocab_size: fb("vocab_size"),
    moe: (
      ($cfg.num_experts // $cfg.num_local_experts // $cfg.n_routed_experts //
       $tc.num_experts // $tc.num_local_experts // $tc.n_routed_experts // null) != null
    ),
    num_experts: ($cfg.num_experts // $cfg.num_local_experts // $cfg.n_routed_experts //
                  $tc.num_experts // $tc.num_local_experts // $tc.n_routed_experts // null),
    num_experts_per_tok: ($cfg.num_experts_per_tok // $tc.num_experts_per_tok // null),
    num_shared_experts: ($cfg.num_shared_experts // $cfg.shared_expert_intermediate_size //
                         $tc.num_shared_experts // $tc.shared_expert_intermediate_size // null),
    layer_types: ($cfg.layer_types // $tc.layer_types // null),
    full_attention_interval: ($cfg.full_attention_interval // $tc.full_attention_interval // null),
    total_params: ($info.safetensors // {} | .total // null),
    license: ([$info.tags // [] | .[] | select(startswith("license:"))][0] // null
              | if . then sub("^license:";"") else null end),
    tags: ($info.tags // []),
    pipeline_tag: ($meta.pipeline_tag // null),
    hub_list_gguf: ($meta.hub_list_gguf // null),
    siblings_gguf_files: ($meta.siblings_gguf_files // []),
    gguf_file_count: ($meta.gguf_file_count // 0),
    has_gguf_in_repo: (($meta.gguf_file_count // 0) > 0),
    has_target_quant_gguf: ($meta.has_target_quant_gguf // false),
    has_config_sibling: ($meta.has_config_sibling // false),
    has_chat_template: ($tok_meta.has_chat_template // null),
    chat_template_tool_hint: ($tok_meta.chat_template_tool_hint // null),
    used_list_metadata_for_info: $used_list
  }' > "$out"
INNER

chmod +x "$work/fetch_one.sh"

export REQUIRE_CONFIG_SIBLING
export ENRICH_FETCH_TOKENIZER

ids=$(jq -r '.candidates[].id' "$INPUT")
n=$(printf '%s\n' "$ids" | grep -c .)
printf '>>> enriching %s candidates (parallel %s)\n' "$n" "$PAR" >&2

printf '%s\n' "$ids" | xargs -n1 -P"$PAR" -I{} "$work/fetch_one.sh" {} "$work" "$work/_discovery_map.json"

mirrors_json="$work/_mirrors_empty.json"
if [ -f "$MIRROR_FILE" ]; then
  mirrors_json="$MIRROR_FILE"
else
  echo '{}' > "$work/_mirrors_empty.json"
  mirrors_json="$work/_mirrors_empty.json"
fi

shopt -s nullglob
_model_json=()
for f in "$work"/*.json; do
  case $(basename "$f") in
    _*) continue ;;
  esac
  _model_json+=("$f")
done
[ "${#_model_json[@]}" -gt 0 ] || { echo "no per-model json in $work" >&2; exit 1; }

jq -s --slurpfile disc "$INPUT" --slurpfile arch "$ARCH_FILE" --slurpfile mir "$mirrors_json" \
  --arg srcin "$INPUT" --arg archf "$ARCH_FILE" '
  ($disc[0].candidates | map({key: .id, value: {downloads, likes, trending_score}}) | from_entries) as $stats
  | ($arch[0].architectures | map(select(. != null)) | map({(.): true}) | add // {}) as $sup
  | ($mir[0] | if (type == "object") and (.by_base != null) then .by_base elif type == "object" then . else {} end) as $ub
  |
  def trusted_authors:
    ["Qwen","zai-org","mistralai","meta-llama","microsoft","openai",
     "ibm-granite","nvidia","deepseek-ai","moonshotai","Alibaba-NLP",
     "baidu","LiquidAI","inclusionAI","google","01-ai","CohereLabs",
     "AI21labs","databricks","apple","poolside","MiniMaxAI","NousResearch",
     "tiiuae","openbmb","unsloth"];
  def author: .id | split("/")[0];
  def is_trusted: author as $a | (trusted_authors | index($a)) != null;
  def signature:
    if (.error // "") != "" then "ERROR:" + .id
    else "\(.model_type // "?")|\(.total_params // "?")|\(.num_hidden_layers // "?")|\(.num_experts // "none")|\(.head_dim // "?")"
    end;

  [.[] | . + {
      downloads: ($stats[.id] // {} | .downloads // null),
      likes:     ($stats[.id] // {} | .likes // null),
      trending_score: ($stats[.id] // {} | .trending_score // null),
      convert_hf_arch_supported: (
        if (.architectures != null) and ((.architectures | length) > 0)
          and (.architectures[0] != null)
        then ($sup[.architectures[0]] // false)
        else false end
      ),
      unsloth_gguf_mirrors: ($ub[.id] // [])
    }]
  | group_by(signature)
  | map(
      sort_by([(if is_trusted then 0 else 1 end), .id])
      | (.[0] | .id) as $orig
      | map(. + {
          derivative_of: (if .id == $orig then null else $orig end)
        })
    )
  | flatten
  | sort_by(.id)
  | {
      generated_at: (now | todate),
      source_discovery: $srcin,
      supported_architectures_file: $archf,
      models: .
    }
' "${_model_json[@]}" > "$OUTPUT" || {
  printf '>>> jq consolidate failed — work dir %s (set KEEP_WORK=1 before run to preserve)\n' "$work" >&2
  exit 5
}

ok=$(jq '[.models[] | select((.error // "") == "")] | length' "$OUTPUT")
err=$(jq '[.models[] | select((.error // "") != "")] | length' "$OUTPUT")
printf '>>> wrote %s (ok=%s, errors=%s)\n' "$OUTPUT" "$ok" "$err" >&2

if [ "$err" -gt 0 ]; then
  printf '>>> errors:\n' >&2
  jq -r '.models[] | select((.error // "") != "") | "  \(.id): \(.error) (http=\(.http // "?"))"' "$OUTPUT" >&2
fi
