#!/usr/bin/env bash
# unsloth_mirrors.sh — build unsloth_mirrors.by_base.json mapping base HF ids to
# Unsloth GGUF repo ids (from tags). Optional; enrich.sh picks it up if present.
#
# Env: LIMIT (default 120), OUTPUT (default unsloth_mirrors.by_base.json)
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${LIMIT:-120}"
OUTPUT="${OUTPUT:-unsloth_mirrors.by_base.json}"

hf models list --author unsloth --sort downloads --limit "$LIMIT" \
  --expand tags --filter text-generation --json 2>/dev/null \
  | jq --arg lim "$LIMIT" '[
      .[] | select(.id | test("GGUF"; "i"))
        | . as $m
        | ($m.tags // []
            | map(select(startswith("base_model:")))
            | map(
                sub("^base_model:"; "")
                | if startswith("finetune:") then sub("^finetune:";"") else . end
              )
          ) as $bases
        | $bases[]
        | {base: ., mirror: $m.id}
    ]
    | group_by(.base)
    | map({(.[0].base): [.[].mirror] | unique})
    | add // {}
    | {generated_at: (now | todate), list_limit: ($lim | tonumber), by_base: .}
    ' > "$OUTPUT.tmp"

mv "$OUTPUT.tmp" "$OUTPUT"
printf '>>> wrote %s\n' "$OUTPUT" >&2
