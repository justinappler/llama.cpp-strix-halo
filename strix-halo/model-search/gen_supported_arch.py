#!/usr/bin/env python3
"""Extract @ModelBase.register(...) architecture names from convert_hf_to_gguf.py.

Writes supported_hf_architectures.json for the model-search pipeline (no transformers).


"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_register_blocks(text: str) -> set[str]:
    key = "@ModelBase.register("
    names: set[str] = set()
    i = 0
    while True:
        j = text.find(key, i)
        if j < 0:
            break
        j += len(key)
        depth = 1
        k = j
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        block = text[j : k - 1]
        for m in re.finditer(r'"([A-Za-z][A-Za-z0-9_\.]*)"', block):
            tok = m.group(1)
            if tok.lower() == "class":
                continue
            names.add(tok)
        i = k
    return names


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    conv = root / "convert_hf_to_gguf.py"
    if not conv.is_file():
        print(f"error: missing {conv}", file=sys.stderr)
        return 1
    text = conv.read_text(encoding="utf-8")
    arches = sorted(parse_register_blocks(text))
    out = Path(__file__).resolve().parent / "supported_hf_architectures.json"
    payload = {
        "generated_at": None,
        "source": str(conv.relative_to(root)),
        "count": len(arches),
        "architectures": arches,
    }
    from datetime import datetime, timezone

    payload["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(arches)} names)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
