#!/usr/bin/env bash
# High-throughput vLLM serving script on AMD Instinct MI50 (32GB HBM2)

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"

export HSA_OVERRIDE_GFX_VERSION=9.0.6
export VLLM_USE_TRITON_FLASH_ATTN=0
export HIP_VISIBLE_DEVICES=0

echo "============================================================"
echo "Launching vLLM on AMD Instinct MI50 (32GB HBM2, gfx906)"
echo "Model: ${MODEL}"
echo "Port:  ${PORT}"
echo "============================================================"

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enforce-eager \
    --trust-remote-code
