#!/usr/bin/env bash
set -euo pipefail
# Run in a SEPARATE Python environment from the legacy ChemDFM/ChemLLM runtime.
# On an 80 GB GPU the official BF16 judge should generally have its own GPU.
# The alternative community 4-bit checkpoint requires independent score validation.
MODEL="${GEMMA_JUDGE_MODEL:-google/gemma-4-26B-A4B-it}"
GPU_UTIL="${GEMMA_GPU_UTIL:-0.85}"
MAX_LEN="${GEMMA_MAX_MODEL_LEN:-16384}"
exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 1
