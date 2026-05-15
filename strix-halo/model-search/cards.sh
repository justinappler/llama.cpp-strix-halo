#!/usr/bin/env bash
# cards.sh — fetch model cards (README.md) for a list of HF repo IDs and
# cache them under cards/<slug>.md. Stdin or args.
#
#   echo "zai-org/GLM-4.5-Air" | ./cards.sh
#   ./cards.sh zai-org/GLM-4.5-Air Qwen/Qwen3-Coder-30B-A3B-Instruct
#
#   # Pipe a jq filter through:
#   jq -r '.models[] | select(.head_dim == 128) | .id' enriched.json | ./cards.sh
#
# Re-runs are cheap — `If-None-Match` would be nicer but HF doesn't honor it
# on resolve URLs, so we just overwrite. Use FORCE=0 to skip already-cached.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p cards
FORCE="${FORCE:-1}"

ids=()
if [ "$#" -gt 0 ]; then
  ids=("$@")
else
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    ids+=("$line")
  done
fi

[ "${#ids[@]}" -eq 0 ] && { echo "no IDs given (args or stdin)" >&2; exit 1; }

for id in "${ids[@]}"; do
  slug="${id//\//__}"
  out="cards/$slug.md"
  if [ "$FORCE" = "0" ] && [ -s "$out" ]; then
    printf 'cached: %s\n' "$id" >&2
    continue
  fi
  url="https://huggingface.co/${id}/resolve/main/README.md"
  http=$(curl -sL --max-time 20 -o "$out.tmp" -w '%{http_code}' "$url" || echo "000")
  if [ "$http" = "200" ] && [ -s "$out.tmp" ]; then
    mv "$out.tmp" "$out"
    sz=$(wc -c < "$out" | tr -d ' ')
    printf 'fetched: %s (%s bytes)\n' "$id" "$sz" >&2
  else
    rm -f "$out.tmp"
    printf 'FAIL: %s (http=%s)\n' "$id" "$http" >&2
  fi
done
