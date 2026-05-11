#!/usr/bin/env bash
# Step 4: ollama create agent-sft-qwen -f Modelfile
#
# 幂等：已存在 tag 先 rm 再 create。Modelfile 里 FROM 路径相对本目录。

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-agent-sft-qwen}"
MODELFILE="${MODELFILE:-$HERE/Modelfile}"
GGUF="$HERE/build/agent-sft-qwen-q4.gguf"

[[ -f "$GGUF" ]] || { echo "[deploy] missing $GGUF — run bash build.sh first" >&2; exit 1; }
[[ -f "$MODELFILE" ]] || { echo "[deploy] missing $MODELFILE" >&2; exit 1; }
command -v ollama >/dev/null || { echo "[deploy] ollama not on PATH" >&2; exit 1; }

if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$TAG:latest"; then
  echo "[deploy] tag exists: $TAG — removing first (ollama create is not idempotent)"
  ollama rm "$TAG"
fi

echo "[deploy] ollama create $TAG -f $MODELFILE"
cd "$HERE"
ollama create "$TAG" -f "$MODELFILE"

echo
echo "[deploy] done. ollama list:"
ollama list | grep -E "(NAME|$TAG)" || true
echo
echo "next: python $HERE/smoke_test.py   # verify <tool_call> emission"
