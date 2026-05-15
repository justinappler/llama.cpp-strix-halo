#!/usr/bin/env bash
# Regenerate supported_hf_architectures.json from ../../convert_hf_to_gguf.py
set -euo pipefail
cd "$(dirname "$0")"
exec python3 ./gen_supported_arch.py
