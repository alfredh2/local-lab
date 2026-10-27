#!/usr/bin/env bash
# Start a throwaway vLLM OpenAI-compatible server in Docker for LocalLab benchmarks.
# Usage: scripts/run_vllm_server.sh <hf-model-id> <served-name> [gpu-memory-utilization] [max-model-len]
# Stop with: docker rm -f locallab-vllm-bench
#
# LD_LIBRARY_PATH puts the host driver's libcuda first. Without it the image's bundled CUDA 12.9
# compat library shadows a newer host driver and torch fails with "Error 803: unsupported
# display driver / cuda driver combination" (seen with driver 595 on the RTX 5090).
set -euo pipefail
MODEL="${1:?hf model id, e.g. Qwen/Qwen2.5-1.5B-Instruct}"
NAME="${2:?served model name}"
UTIL="${3:-0.25}"
MAXLEN="${4:-8192}"
docker rm -f locallab-vllm-bench >/dev/null 2>&1 || true
docker run -d --name locallab-vllm-bench --runtime nvidia --gpus all --ipc=host \
  -p 127.0.0.1:8100:8000 -e HF_HUB_OFFLINE=1 \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "${VLLM_IMAGE:-vllm/vllm-openai:latest}" "$MODEL" --served-model-name "$NAME" \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --max-num-seqs 64
echo "Waiting for http://127.0.0.1:8100 ..."
until curl -sf http://127.0.0.1:8100/v1/models >/dev/null; do sleep 3; done
echo "Ready: $NAME"
