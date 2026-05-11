#!/usr/bin/env bash
# Phase 4 build pipeline: LoRA adapter -> fused MLX fp16 -> F16 GGUF -> Q4_K_M GGUF
#
# 三步串起来，每步可单独跑（已存在则跳过；--force 覆盖）。最终产物：
#   build/agent-sft-qwen-q4.gguf  ~4 GB    <- ollama create 的 FROM 指向这个
#
# 中间产物（gitignored）：
#   build/fused-mlx-fp16/         ~14 GB   fp16 MLX 目录，可 mlx_lm.generate 直接验
#   build/agent-sft-qwen-f16.gguf ~14 GB   未量化 GGUF，可 llama-cli 直接验
#
# 依赖：
#   - mlx_lm.fuse        (pip install mlx-lm[train])
#   - $LLAMA_CPP_DIR     llama.cpp 仓库根；含 convert_hf_to_gguf.py + venv
#   - llama-quantize     $LLAMA_CPP_DIR/build/bin/llama-quantize

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
BUILD_DIR="$HERE/build"
ADAPTER="${ADAPTER:-$REPO_ROOT/play/agent_sft/train/runs/sweeps/iters/200}"
BASE_MODEL="${BASE_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/Tools/llama.cpp}"
QUANT="${QUANT:-Q4_K_M}"

FORCE=0
[[ "${1:-}" == "--force" || "${1:-}" == "-f" ]] && FORCE=1

mkdir -p "$BUILD_DIR"

FUSED_DIR="$BUILD_DIR/fused-mlx-fp16"
GGUF_F16="$BUILD_DIR/agent-sft-qwen-f16.gguf"
GGUF_Q4="$BUILD_DIR/agent-sft-qwen-q4.gguf"

CONVERT_PY="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
LLAMA_PY="$LLAMA_CPP_DIR/.venv/bin/python"

# Sanity checks ---------------------------------------------------------------
[[ -d "$ADAPTER" ]]         || { echo "[build] adapter not found: $ADAPTER" >&2; exit 1; }
[[ -f "$CONVERT_PY" ]]      || { echo "[build] missing $CONVERT_PY (set LLAMA_CPP_DIR)"  >&2; exit 1; }
[[ -x "$QUANTIZE_BIN" ]]    || { echo "[build] missing $QUANTIZE_BIN — build llama.cpp first" >&2; exit 1; }
[[ -x "$LLAMA_PY" ]]        || { echo "[build] missing $LLAMA_PY — create llama.cpp venv first" >&2; exit 1; }

print_size() {
  local p="$1"
  if [[ -d "$p" ]]; then du -sh "$p" | awk '{print $1}'
  elif [[ -f "$p" ]]; then ls -lh "$p" | awk '{print $5}'
  else echo "MISSING"; fi
}

# Step 1: fuse + dequantize ---------------------------------------------------
if [[ -d "$FUSED_DIR" && -f "$FUSED_DIR/model.safetensors" && $FORCE -eq 0 ]]; then
  echo "[step1] skip (exists): $FUSED_DIR ($(print_size "$FUSED_DIR"))"
else
  echo "[step1] mlx_lm.fuse --dequantize → $FUSED_DIR"
  rm -rf "$FUSED_DIR"
  mlx_lm.fuse \
    --model "$BASE_MODEL" \
    --adapter-path "$ADAPTER" \
    --save-path "$FUSED_DIR" \
    --dequantize
  echo "[step1] done; size = $(print_size "$FUSED_DIR")"
fi

# Step 2: HF → F16 GGUF -------------------------------------------------------
if [[ -f "$GGUF_F16" && $FORCE -eq 0 ]]; then
  echo "[step2] skip (exists): $GGUF_F16 ($(print_size "$GGUF_F16"))"
else
  echo "[step2] convert_hf_to_gguf.py → $GGUF_F16"
  rm -f "$GGUF_F16"
  "$LLAMA_PY" "$CONVERT_PY" "$FUSED_DIR" \
    --outfile "$GGUF_F16" \
    --outtype f16
  echo "[step2] done; size = $(print_size "$GGUF_F16")"
fi

# Step 3: F16 → Q4_K_M --------------------------------------------------------
if [[ -f "$GGUF_Q4" && $FORCE -eq 0 ]]; then
  echo "[step3] skip (exists): $GGUF_Q4 ($(print_size "$GGUF_Q4"))"
else
  echo "[step3] llama-quantize $QUANT → $GGUF_Q4"
  rm -f "$GGUF_Q4"
  "$QUANTIZE_BIN" "$GGUF_F16" "$GGUF_Q4" "$QUANT"
  echo "[step3] done; size = $(print_size "$GGUF_Q4")"
fi

echo
echo "[build] all done."
echo "  fused fp16  : $(print_size "$FUSED_DIR")  $FUSED_DIR"
echo "  F16  GGUF   : $(print_size "$GGUF_F16")  $GGUF_F16"
echo "  $QUANT GGUF : $(print_size "$GGUF_Q4")  $GGUF_Q4"
echo
echo "next: bash $HERE/deploy.sh   # ollama create agent-sft-qwen"
