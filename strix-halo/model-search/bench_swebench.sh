#!/usr/bin/env bash
# bench_swebench.sh — scrape SWE-bench Verified leaderboard scores per model
# from github.com/SWE-bench/experiments via sparse checkout. github.com only,
# no Hugging Face traffic. Parses metadata.yml + results/results.json per
# submission, joins on the HF model URL, takes max score per HF model id.
#
# Env:
#   CACHE_DIR   .swebench_cache    sparse-checkout target; reused across runs
#   OUTPUT      swebench_scores.json
#   REFRESH=0   set 1 to git pull the cached repo before parsing
#   TOTAL       500                SWE-bench Verified instance count
set -euo pipefail
cd "$(dirname "$0")"

REPO_URL="https://github.com/SWE-bench/experiments.git"
CACHE_DIR="${CACHE_DIR:-.swebench_cache}"
OUTPUT="${OUTPUT:-swebench_scores.json}"
REFRESH="${REFRESH:-0}"
TOTAL="${TOTAL:-500}"

if [ ! -d "$CACHE_DIR/.git" ]; then
  printf '>>> sparse-cloning %s → %s\n' "$REPO_URL" "$CACHE_DIR" >&2
  git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$CACHE_DIR" >&2
  git -C "$CACHE_DIR" sparse-checkout set evaluation/verified >&2
elif [ "$REFRESH" = "1" ]; then
  printf '>>> refreshing %s\n' "$CACHE_DIR" >&2
  git -C "$CACHE_DIR" pull --depth 1 --ff-only --quiet >&2 || git -C "$CACHE_DIR" fetch --depth 1 --quiet >&2
fi

VDIR="$CACHE_DIR/evaluation/verified"
[ -d "$VDIR" ] || { echo "missing $VDIR — sparse checkout failed" >&2; exit 1; }

python3 - "$VDIR" "$TOTAL" <<'PY' > "$OUTPUT"
import datetime, json, os, re, sys

verified_dir = sys.argv[1]
total = int(sys.argv[2])

# Minimal yaml parse (no PyYAML dep): we only need info.name.
def parse_info_name(path):
    section = None
    info_name = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if re.match(r"^[A-Za-z_]+:\s*$", line):
                section = line.split(":", 1)[0]
                continue
            if section == "info":
                m = re.match(r"^\s+name:\s*(.+?)\s*$", line)
                if m and info_name is None:
                    info_name = m.group(1).strip().strip('"').strip("'")
    return info_name

def hf_id(url):
    if not url:
        return None
    m = re.match(r"https?://huggingface\.co/([^/?#]+/[^/?#]+)", url)
    return m.group(1) if m else None

# Aliases for submissions that name the model without an HF URL. Maps the raw
# tags.model value (lowercased, stripped) to the canonical HF model id we use
# in enriched.json. Keep small and obvious; expand only when a target model is
# missed.
ALIASES = {
    "qwen3-coder-30b-a3b-instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-coder-480b-a35b-instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "kimi-k2-instruct": "moonshotai/Kimi-K2-Instruct",
    "moonshot/kimi-k2-0711-preview": "moonshotai/Kimi-K2-Instruct",
    "kimi-k2-0711-preview": "moonshotai/Kimi-K2-Instruct",
    "kimi-k2-0905-preview": "moonshotai/Kimi-K2-Instruct-0905",
    "mistralai/devstral-small-2505": "mistralai/Devstral-Small-2505",
    "devstral-small-2505": "mistralai/Devstral-Small-2505",
    "devstral-small-2507": "mistralai/Devstral-Small-2507",
    "openai/gpt-5-2025-08-07": None,  # closed
}

def parse_tags_model(path):
    """Return the first tags.model entry as (canonical_hf_id, raw_value)."""
    section = None
    in_list = False
    raw_first = None
    hf_url = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if re.match(r"^[A-Za-z_]+:\s*$", line):
                section = line.split(":", 1)[0]
                in_list = False
                continue
            if section == "tags":
                if re.match(r"^\s+model:\s*$", line):
                    in_list = True
                    continue
                if in_list:
                    m = re.match(r"^\s+-\s*(\S.*?)\s*$", line)
                    if m:
                        val = m.group(1).strip().rstrip(",")
                        if raw_first is None:
                            raw_first = val
                        if hf_url is None:
                            mu = re.match(r"https?://huggingface\.co/\S+", val)
                            if mu:
                                hf_url = mu.group(0)
                    elif re.match(r"^\s+\w+:", line):
                        in_list = False
    canon = hf_id(hf_url)
    if canon is None and raw_first is not None:
        canon = ALIASES.get(raw_first.lower())
    return canon, raw_first

submissions = []
by_model = {}

for d in sorted(os.listdir(verified_dir)):
    base = os.path.join(verified_dir, d)
    if not os.path.isdir(base):
        continue
    meta = next((os.path.join(base, n) for n in ("metadata.yml", "metadata.yaml")
                 if os.path.exists(os.path.join(base, n))), None)
    res = os.path.join(base, "results", "results.json")
    if meta is None or not os.path.exists(res):
        continue
    info_name = parse_info_name(meta)
    mid, raw_model = parse_tags_model(meta)
    try:
        rj = json.load(open(res))
    except Exception:
        continue
    resolved = len(rj.get("resolved") or [])
    score = resolved / total if total else 0.0
    sub = {
        "submission": d,
        "submission_date": d[:8] if re.match(r"^\d{8}", d) else None,
        "info_name": info_name,
        "hf_model_id": mid,
        "tags_model_raw": raw_model,
        "resolved": resolved,
        "score": round(score, 4),
    }
    submissions.append(sub)
    if mid:
        cur = by_model.get(mid)
        if cur is None or score > cur["score"]:
            by_model[mid] = {
                "score": round(score, 4),
                "resolved": resolved,
                "best_submission": d,
                "info_name": info_name,
                "submission_date": sub["submission_date"],
            }

out = {
    "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": "github.com/SWE-bench/experiments evaluation/verified",
    "total_instances": total,
    "submission_count": len(submissions),
    "by_model": by_model,
    "submissions": submissions,
}
print(json.dumps(out, indent=2))
PY

n_sub=$(jq '.submission_count' "$OUTPUT")
n_mod=$(jq '.by_model | length' "$OUTPUT")
printf '>>> wrote %s (%s submissions, %s unique HF models)\n' "$OUTPUT" "$n_sub" "$n_mod" >&2
